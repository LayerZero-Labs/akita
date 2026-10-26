use super::root::verify_root;
use super::suffix::{verify_suffix, SuffixVerifierState};
// Top-level batched verifier orchestration once a schedule is selected.

use akita_config::{transcript_instance_descriptor, CommitmentConfig};
use akita_error::AkitaError;
use akita_serialization::{AkitaSerialize, Valid};
use jolt_field::{CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};

use akita_params::{validate_schedule_ring_dims, BasisMode, CommittedGroupBatchProfile};
use akita_types::{FpExtEncoding, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};

use crate::AkitaVerifier;

struct PreparedOpening<'verifier, 'claims, Cfg: CommitmentConfig> {
    schedule: &'verifier akita_params::FoldSchedule,
    opening_batch: akita_params::OpeningClaimsLayout,
    claims: OpeningClaims<'claims, Cfg::ExtField, &'claims akita_types::Commitment<Cfg::Field>>,
    grinding_plan: akita_params::GrindingPlan,
    descriptor_bytes: Vec<u8>,
}

impl<Cfg> AkitaVerifier<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field:
        Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + ExtField<Cfg::Field> + Ring + AkitaSerialize + Valid,
{
    /// Verify one authoritative Spongefish argument.
    #[inline(never)]
    #[tracing::instrument(skip_all, name = "AkitaVerifier::batched_verify")]
    pub fn batched_verify(
        &self,
        proof: &[u8],
        session: &[u8],
        statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        let prepared = self.prepare_opening(statement, basis)?;
        let state =
            akita_transcript::new_verifier_channel(session, &prepared.descriptor_bytes, proof)?;
        let mut grinding = akita_types::VerifierGrinding::new(state, &prepared.grinding_plan);
        self.verify_prepared(&prepared, basis, &mut grinding)?;
        grinding.finish().map(|_accepted| ())
    }

    /// Verify exactly one multi-group opening on the caller's session.
    ///
    /// Absorbs the native protocol/version and setup/catalog/profile context.
    /// This checks the opening's grinding plan without requiring stream EOF;
    /// the caller retains responsibility for final session acceptance.
    /// Any failure permanently invalidates the caller's verifier session.
    pub fn batched_verify_in_session(
        &self,
        session: &mut akita_transcript::VerifierSession<'_>,
        statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        let result = (|| {
            let prepared = self.prepare_opening(statement, basis)?;
            akita_transcript::absorb_subprotocol_context(
                session,
                b"akita-pcs",
                akita_transcript::PROTOCOL_VERSION,
                &prepared.descriptor_bytes,
            )?;
            let mut grinding =
                akita_types::VerifierGrinding::new_borrowed(session, &prepared.grinding_plan);
            self.verify_prepared(&prepared, basis, &mut grinding)?;
            grinding.finish_opening()
        })();
        if result.is_err() {
            session.invalidate();
        }
        result
    }

    fn prepare_opening<'verifier, 'claims>(
        &'verifier self,
        statement: GroupBatchStatement<'claims, Cfg::ExtField, Cfg::Field>,
        basis: BasisMode,
    ) -> Result<PreparedOpening<'verifier, 'claims, Cfg>, AkitaError> {
        let setup = &self.setup;
        let schedules = &self.schedules;
        let selection = statement.selection();
        let claims = statement.into_claims();
        claims.validate(setup.expanded().descriptor())?;
        let opening_batch = claims.committed_layout()?;
        let (final_group, precommitteds) = claims
            .groups()
            .split_last()
            .ok_or(AkitaError::InvalidProof)?;
        let final_descriptor = *final_group.commitment().profile();
        for group in claims.groups() {
            let committed = group.commitment();
            let descriptor = committed.profile();
            descriptor.validate_frozen_precommit(Cfg::decomposition().field_bits())?;
            let plan = descriptor.outer_compression_plan()?;
            if committed.commitment().rows().coeff_len() != plan.terminal_coefficients() {
                return Err(AkitaError::InvalidInput(
                    "commitment length does not match its compression chain".to_string(),
                ));
            }
        }
        let batch_profile = CommittedGroupBatchProfile {
            final_group: final_descriptor,
            precommitteds: precommitteds
                .iter()
                .map(|group| *group.commitment().profile())
                .collect(),
        };
        batch_profile.validate(Cfg::decomposition().field_bits())?;
        let resolved = schedules.resolve_selection(selection)?;
        resolved.validate_opening_layout(&opening_batch)?;
        if resolved.profiles() != &batch_profile {
            return Err(AkitaError::InvalidInput(
                "commitment profiles do not match the selected schedule row".to_string(),
            ));
        }
        let schedule = resolved.schedule();
        let root_params = &schedule.root_fold().params;
        let expected_final_descriptor = akita_params::GroupCommitPhaseParams::try_from_params(
            final_descriptor.group,
            root_params,
        )?;
        if final_descriptor != expected_final_descriptor
            || root_params.precommitted_groups().len() != precommitteds.len()
            || root_params
                .precommitted_groups()
                .iter()
                .zip(precommitteds)
                .any(|(params, claims_group)| {
                    params.profile != *claims_group.commitment().profile()
                })
        {
            return Err(AkitaError::InvalidInput(
                "commitment profiles do not match the selected schedule's root fold".to_string(),
            ));
        }
        validate_schedule_ring_dims(schedule)?;
        if !self.admits(selection.row_digest) {
            return Err(AkitaError::InvalidSetup(
                "selected schedule row does not fit this verifier's setup".to_string(),
            ));
        }
        let (grinding_plan, descriptor_bytes) = transcript_instance_descriptor::<Cfg::Field, Cfg>(
            &setup.expanded().descriptor,
            &opening_batch,
            selection,
            schedule,
            basis,
        )?;
        let raw_groups = claims
            .groups()
            .iter()
            .map(|group| {
                PolynomialGroupClaims::new(
                    group.point().to_vec(),
                    group.evaluations().to_vec(),
                    group.commitment().commitment(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let raw_claims = OpeningClaims::from_groups(raw_groups)?;
        Ok(PreparedOpening {
            schedule,
            opening_batch,
            claims: raw_claims,
            grinding_plan,
            descriptor_bytes,
        })
    }

    fn verify_prepared(
        &self,
        prepared: &PreparedOpening<'_, '_, Cfg>,
        basis: BasisMode,
        grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    ) -> Result<(), AkitaError> {
        let setup = &self.setup;
        let schedule = prepared.schedule;
        let root = verify_root::<Cfg::Field, Cfg::ExtField>(
            setup,
            grinding,
            &prepared.claims,
            &prepared.opening_batch,
            basis,
            &schedule.root_fold().params,
            schedule.recursive_folds.first(),
            &schedule.terminal,
        )?;
        verify_suffix::<Cfg::Field, Cfg::ExtField>(
            setup,
            &self.terminal_ntt,
            grinding,
            schedule,
            SuffixVerifierState {
                opening_point: root.challenges,
                opening: root.opening,
                witness: root.next_witness,
                witness_last_block_digest: root.next_witness_last_block_digest,
                basis: BasisMode::Lagrange,
                witness_len: schedule.root_fold().output_witness_len,
                setup_prefix_opening: root.setup_prefix_opening,
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{RingVec, RingView};
    use jolt_field::{Fp32, Zero};

    type F = Fp32<251>;
    const D: usize = 32;

    /// The D-free commitment read path validates the flat coefficient length
    /// against the schedule-derived ring dimension via `RingView::new` and
    /// returns an error (never panics) when the length is not a multiple of the
    /// ring dimension. This is the no-panic gate the verifier relies on before
    /// interpreting any ring-shaped commitment.
    #[test]
    fn flat_commitment_length_not_multiple_of_ring_dim_rejects() {
        // 33 coefficients is not a multiple of D = 32.
        let commitment = RingVec::from_coeffs(vec![F::zero(); D + 1]);
        let err = RingView::new(commitment.coeffs(), D)
            .expect_err("commitment length must be a multiple of the ring dimension");
        assert!(matches!(err, AkitaError::InvalidProof));

        // A well-formed buffer (2 * D) is accepted and yields the expected ring count.
        let well_formed = vec![F::zero(); 2 * D];
        let ok = RingView::new(&well_formed, D).expect("valid flat commitment");
        assert_eq!(ok.num_rings(), 2);
    }
}
