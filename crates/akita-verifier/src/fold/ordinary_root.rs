use super::{
    receive_opening_payload, verify_fold, FoldVerifyOutput, NextWitnessPlan, PreparedFoldReplay,
};
use crate::claims::ClaimReplay;
use crate::stages::opening_claims::{finalize_claims, verify_coefficient_packing_root_prefix};
use akita_error::AkitaError;
use akita_params::{
    BasisMode, CommittedGroupParams, FoldParams, OpeningClaimsLayout, RelationWitnessGeometry,
    SetupContributionMode, TerminalFoldParams,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::ChallengeCodec;
use akita_types::CommittedGroup;
use akita_types::{AkitaVerifierSetup, FpExtEncoding, OpeningClaims};
use jolt_algebra::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Replays the root fold. `claims` supplies the statement's commitments and
/// points, and `openings` its claims in group order; the evaluations stored in
/// `claims` are not read.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_root<F, E, G: ClaimReplay<F, E>>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut G,
    claims: &OpeningClaims<'_, E, &CommittedGroup<F>>,
    openings: Vec<G::Claim>,
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
    next_fold_params: Option<&FoldParams>,
    terminal: &TerminalFoldParams,
    setup_policy: &mut crate::SetupPolicy<'_, '_, E>,
) -> Result<FoldVerifyOutput<F, E, G::Claim>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>
        + ChallengeCodec,
{
    root_lp.validate_opening_batch(opening_batch)?;
    for group_index in 0..opening_batch.num_groups() {
        if !matches!(
            root_lp
                .group_params_geometry(opening_batch, group_index)?
                .opening_method(),
            akita_params::OpeningMethod::SubringCoefficientPacking { .. }
        ) {
            return Err(AkitaError::InvalidProof);
        }
    }
    let relation_geometry = RelationWitnessGeometry::for_level(root_lp, opening_batch, E::DEGREE)?;
    let relation_layout = relation_geometry.rhs_layout();
    for group_index in 0..opening_batch.num_groups() {
        let commitment = claims.group_commitment(group_index)?;
        let plan = relation_layout.compression_plan_for_group(group_index)?;
        if commitment.rows().coeff_len() != plan.terminal_coefficients() {
            return Err(AkitaError::InvalidProof);
        }
        let ring_dim = plan
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        // The digest and the rows used below come from this same group.
        akita_transcript::public_bytes(
            grinding.state_mut(),
            akita_types::FoldSite::RootCommitment {
                group: group_index,
                ring_dimension: ring_dim,
            }
            .id()?,
            commitment.digest().as_bytes(),
        )?;
    }
    for (group_index, group) in claims.groups().iter().enumerate() {
        akita_transcript::public_extensions::<F, E, _>(
            grinding.state_mut(),
            akita_types::FoldSite::RootPoint { group: group_index }.id()?,
            group.point(),
        )?;
    }
    // Schedule validation admits no setup prefix at the root.
    grinding.bind_openings(0, None, &openings)?;
    let material = verify_coefficient_packing_root_prefix::<F, E, _>(
        claims,
        openings,
        opening_batch,
        basis,
        root_lp,
    )?;
    let payload_geometry = relation_layout.opening_payload_geometry()?;
    let opening_payload = receive_opening_payload::<F, E, _>(grinding, 0, &payload_geometry)?;
    let prefix = finalize_claims::<F, E, _>(opening_batch, material, grinding, 0)?;
    let order = opening_batch.root_group_order()?;
    let commitment_payloads = order
        .into_iter()
        .map(|group_index| Ok(claims.group_commitment(group_index)?.rows().clone()))
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let witness_len = root_lp.output_witness_len::<F>(opening_batch, E::DEGREE)?;
    let (next_witness, next_witness_ring_dim, next_opening_source_len, stage3) =
        if let Some(next) = next_fold_params {
            let ring_dim = next.params.d_a();
            let coefficient_count = next
                .params
                .outer_payload_geometry()?
                .transmitted_coefficients();
            let committed_len = akita_params::witness_commitment_domain_len(witness_len, ring_dim)?;
            (
                NextWitnessPlan::OuterPayload { coefficient_count },
                ring_dim,
                committed_len / ring_dim,
                matches!(
                    next.predecessor_setup_contribution_mode(),
                    SetupContributionMode::Recursive
                )
                .then_some(&next.params),
            )
        } else {
            let ring_dim = terminal.d_a();
            let coefficient_count = terminal
                .response_shape
                .layout
                .groups
                .first()
                .ok_or(AkitaError::InvalidProof)?
                .t_field_elems;
            let committed_len = akita_params::witness_commitment_domain_len(witness_len, ring_dim)?;
            (
                NextWitnessPlan::TerminalT { coefficient_count },
                ring_dim,
                committed_len / ring_dim,
                None,
            )
        };
    let challenge_field_bits = F::MODULUS_BITS
        .checked_mul(
            u32::try_from(E::DEGREE)
                .map_err(|_| AkitaError::InvalidSetup("extension degree overflow".into()))?,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("challenge field width overflow".into()))?;
    let level_layout = akita_params::nonterminal_level_layout(
        F::MODULUS_BITS,
        challenge_field_bits,
        root_lp,
        root_lp.relation_address_geometry(
            opening_batch,
            E::DEGREE,
            next_witness_ring_dim,
            witness_len,
        )?,
        next_fold_params.map(|next| &next.params),
    )?;
    verify_fold(
        setup,
        grinding,
        PreparedFoldReplay {
            lp: root_lp,
            level: 0,
            opening_payload,
            opening_shape: opening_batch.clone(),
            commitment_payloads,
            prefix,
            w_len: witness_len,
            level_layout,
            next_witness,
            next_witness_ring_dim,
            next_opening_source_len,
            stage3,
            evaluation_trace_basis: basis,
        },
        setup_policy,
    )
}
