//! Fold orchestration for Akita proofs.
//!
//! `verify` selects the root and suffix paths, `root` and `suffix` prepare
//! each fold's replay inputs, `challenges` draws the fold challenges,
//! `relation_instance` builds the fold's public ring relation, and `terminal`
//! dispatches the terminal fold. [`verify_fold_native`] runs the per-fold
//! stage checks from [`crate::stages`] in transcript order. FoldSchedule and
//! config dispatch stay with the scheme crate until the verifier-facing
//! config boundary is extracted.

mod challenges;
mod relation_instance;
mod root;
mod suffix;
mod terminal;
mod verify;

use crate::relation::RingSwitchReplay;
use crate::stages::opening_claims::FoldPrefix;
use crate::stages::opening_semantics::{prepare_opening_semantics, OpeningSemanticsInput};
use crate::stages::relation_claim::prepare_relation_claim;
use crate::stages::ring_switch::ring_switch_verifier_native;
use crate::stages::stage1::verify_stage1_native;
use crate::stages::stage2::{replay_stage2_native, validate_stage2_replay};
use crate::stages::stage3::verify_stage3;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::NativeGrinding;
use akita_types::{
    AkitaVerifierSetup, BasisMode, CommittedGroupParams, FpExtEncoding, OpeningClaimsLayout,
    RingVec,
};
use challenges::derive_multi_group_stage1_challenges_native;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use relation_instance::{assemble_relation_instance, validate_fold_payloads};

pub(crate) type SetupPrefixOpening<E> = (Vec<E>, E);

pub(crate) struct NativePreparedFoldReplay<'a, F: Field, E: Field> {
    pub(crate) lp: &'a CommittedGroupParams,
    pub(crate) level: u32,
    pub(crate) opening_payload: RingVec<F>,
    pub(crate) opening_shape: OpeningClaimsLayout,
    pub(crate) commitment_payloads: Vec<RingVec<F>>,
    pub(crate) prefix: FoldPrefix<F, E>,
    pub(crate) w_len: usize,
    pub(crate) level_layout: akita_types::NativeNonterminalLevelLayout,
    pub(crate) next_witness: NativeNextWitnessPlan,
    pub(crate) next_witness_ring_dim: usize,
    pub(crate) next_opening_source_len: usize,
    pub(crate) stage3: Option<&'a CommittedGroupParams>,
    pub(crate) evaluation_trace_basis: BasisMode,
}

#[derive(Clone, Copy)]
pub(crate) enum NativeNextWitnessPlan {
    OuterPayload { coefficient_count: usize },
    TerminalT { coefficient_count: usize },
}

pub(crate) struct NativeFoldVerifyOutput<F: Field, E: Field> {
    pub(crate) challenges: Vec<E>,
    pub(crate) setup_prefix_opening: Option<SetupPrefixOpening<E>>,
    pub(crate) next_witness: RingVec<F>,
    pub(crate) opening: E,
}

/// Replay one complete fold directly from the native Spongefish argument.
///
/// Stage order is the transcript order: fold response, fold challenges,
/// successor witness, ring switch, Stage 1, Stage 2 rounds, Stage 3, and the
/// Stage 2 output check, which needs the Stage 3 setup claim.
#[inline(never)]
pub(crate) fn verify_fold_native<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    prepared: NativePreparedFoldReplay<'_, F, E>,
) -> Result<NativeFoldVerifyOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    let level = prepared.level;
    let relation_geometry = validate_fold_payloads(&prepared)?;
    grinding.read_fold_response(akita_types::GrindingSite::FoldResponse { level })?;
    let group_challenges = derive_multi_group_stage1_challenges_native::<F, E>(
        grinding,
        level,
        &prepared.opening_shape,
        prepared.lp,
    )?;
    let relation_instance =
        assemble_relation_instance(&prepared, &relation_geometry, group_challenges)?;
    let next_witness = receive_next_witness(grinding, &prepared)?;
    let rs = ring_switch_verifier_native::<F, E>(
        &RingSwitchReplay {
            setup: setup.expanded(),
            relation: &relation_instance,
            row_coefficients: &prepared.prefix.row_coefficients,
            lp: prepared.lp,
            opening_source_len: prepared.next_opening_source_len,
            opening_ring_dim: prepared.next_witness_ring_dim,
        },
        prepared.w_len,
        grinding,
        level,
    )?;
    let relation = prepare_relation_claim::<F, E>(
        prepared.lp,
        relation_geometry,
        &relation_instance,
        &prepared.prefix,
        &rs,
    )?;
    let stage1 = verify_stage1_native::<F, E>(
        &rs,
        prepared.lp,
        &relation.range_image_plan,
        grinding,
        level,
        &prepared.level_layout,
    )?;
    let stage2_span = tracing::info_span!(
        "stage2_verifier",
        level,
        relation_mode = ?prepared.lp.ring_relation_mode,
        reduced = prepared.lp.ring_relation_mode.is_reduced_evaluation(),
    )
    .entered();
    let opening_semantics = prepare_opening_semantics::<F, E>(
        &OpeningSemanticsInput {
            lp: prepared.lp,
            opening_batch: relation_instance.opening_batch(),
            w_len: prepared.w_len,
            evaluation_trace_basis: prepared.evaluation_trace_basis,
            prefix: &prepared.prefix,
        },
        &rs,
        &relation,
    )?;
    let stage2_rounds = replay_stage2_native::<F, E>(
        grinding,
        level,
        &stage1,
        relation.claim,
        &opening_semantics,
        prepared.level_layout.stage2_sumcheck(),
    )?;
    let stage3 = match prepared.stage3 {
        Some(next_params) => Some(verify_stage3::<F, E>(
            setup,
            &rs,
            &stage2_rounds.challenges,
            next_params,
            grinding,
            level,
        )?),
        None => None,
    };
    let stage2 = validate_stage2_replay(
        setup,
        stage1,
        &rs,
        stage3.as_ref().map(|replay| replay.claim),
        opening_semantics,
        stage2_rounds,
    )?;
    drop(stage2_span);
    Ok(NativeFoldVerifyOutput {
        challenges: stage2.point,
        setup_prefix_opening: stage3.map(|replay| (replay.challenges, replay.setup_prefix_eval)),
        next_witness,
        opening: stage2.witness_eval,
    })
}

/// Receive the successor witness payload for the next fold or the terminal.
fn receive_next_witness<F, E>(
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    prepared: &NativePreparedFoldReplay<'_, F, E>,
) -> Result<RingVec<F>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    let next_witness = match prepared.next_witness {
        NativeNextWitnessPlan::OuterPayload { coefficient_count } => {
            if coefficient_count != prepared.level_layout.next_outer_payload_coeffs() {
                return Err(AkitaError::InvalidSetup(
                    "native successor payload disagrees with the level grammar".into(),
                ));
            }
            akita_transcript::receive_native_field_group::<F>(
                grinding.state_mut(),
                akita_types::NativeFoldSite::NextWitnessPayload {
                    level: prepared.level,
                }
                .id()?,
                coefficient_count,
            )
        }
        NativeNextWitnessPlan::TerminalT { coefficient_count } => {
            akita_transcript::receive_native_field_group::<F>(
                grinding.state_mut(),
                akita_types::NativeFoldSite::NextWitnessInnerState {
                    level: prepared.level,
                }
                .id()?,
                coefficient_count,
            )
        }
    }
    .map(RingVec::from_coeffs)?;
    if prepared.next_witness_ring_dim == 0
        || matches!(
            prepared.next_witness,
            NativeNextWitnessPlan::TerminalT { .. }
        ) && !next_witness.can_decode_vec(prepared.next_witness_ring_dim)
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(next_witness)
}
