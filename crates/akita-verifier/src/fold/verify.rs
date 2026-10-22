use super::root::verify_root;
use super::suffix::{verify_suffix, SuffixVerifierState};
// Top-level batched verifier orchestration once a schedule is selected.

use akita_config::{transcript_instance_descriptor, CommitmentConfig};
use akita_error::AkitaError;
use akita_serialization::{AkitaSerialize, Valid};
use jolt_algebra::{CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};

use akita_params::{
    validate_schedule_ring_dims, BasisMode, CommittedGroupBatchProfile, ProtocolSiteId,
};
use akita_types::{
    AkitaSponge, FpExtEncoding, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
    PROOF_STREAM_PROTOCOL,
};
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::AkitaVerifier;

impl<Cfg> AkitaVerifier<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field:
        Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + ExtField<Cfg::Field> + Ring + AkitaSerialize + Valid,
{
    /// Verify a standalone Akita proof: one argument string under
    /// [`PROOF_STREAM_PROTOCOL`] bound to `session`, which must be consumed
    /// exactly.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the proof is rejected or has
    /// trailing bytes, and the [`batched_verify`](Self::batched_verify)
    /// errors otherwise.
    pub fn verify_standalone(
        &self,
        proof: &[u8],
        session: &[u8],
        statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        let mut transcript =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, session, proof);
        self.batched_verify(&mut transcript, statement, basis)?;
        Ok(transcript.finish()?)
    }

    /// Verify one Akita opening by reading its messages from the caller's
    /// transcript.
    ///
    /// The first transcript operation absorbs the instance descriptor as a
    /// public message. The caller owns the transcript's protocol id, session
    /// binding, and the final [`VerifierTranscript::finish`] check.
    #[inline(never)]
    #[tracing::instrument(skip_all, name = "AkitaVerifier::batched_verify")]
    pub fn batched_verify<H: Sponge>(
        &self,
        transcript: &mut VerifierTranscript<'_, H>,
        statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        let setup = &self.setup;
        let selection = statement.selection();
        let claims = statement.into_claims();
        let opening = self.plan_opening(selection, &claims, basis)?;
        opening.absorb_descriptor(transcript);
        let schedule = opening.schedule;
        let mut grinding = akita_types::VerifierGrinding::new(transcript, &opening.grinding_plan);
        let raw_claims = opening.raw_claims(&claims)?;
        let root = verify_root::<Cfg::Field, Cfg::ExtField, _>(
            setup,
            &mut grinding,
            &raw_claims,
            raw_claims.flat_evaluations(),
            &opening.opening_batch,
            basis,
            &schedule.root_fold().params,
            schedule.recursive_folds.first(),
            &schedule.terminal,
        )?;
        verify_suffix::<Cfg::Field, Cfg::ExtField, _>(
            setup,
            &self.terminal_ntt,
            &mut grinding,
            schedule,
            SuffixVerifierState::from_root(root, schedule.root_fold().output_witness_len),
        )?;
        grinding.finish()
    }

    /// The checks and the transcript instance of one opening, before its
    /// first transcript operation: the statement against the catalog row it
    /// selects and this verifier's setup, and the descriptor and grinding
    /// plan of that row.
    pub(crate) fn plan_opening(
        &self,
        selection: akita_params::OpeningScheduleSelection,
        claims: &OpeningClaims<'_, Cfg::ExtField, &akita_types::CommittedGroup<Cfg::Field>>,
        basis: BasisMode,
    ) -> Result<OpeningPlan<'_>, AkitaError> {
        let setup = &self.setup;
        let schedules = &self.schedules;
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
            let source_coefficients = descriptor.outer_slice_count.complete_source_coefficients(
                descriptor.outer.matrix.output_rank(),
                descriptor.outer.matrix.ring_dimension(),
            )?;
            let plan = akita_params::CompressionChainPlan::for_complete_source(
                descriptor.outer.matrix.sis_table_key().modulus_profile,
                source_coefficients,
            )?;
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
        Ok(OpeningPlan {
            schedule,
            grinding_plan,
            descriptor_bytes,
            opening_batch,
        })
    }
}

/// One opening's checked instance, from [`AkitaVerifier::plan_opening`].
pub(crate) struct OpeningPlan<'s> {
    pub(crate) schedule: &'s akita_params::FoldSchedule,
    pub(crate) grinding_plan: akita_params::GrindingPlan,
    descriptor_bytes: Vec<u8>,
    pub(crate) opening_batch: akita_params::OpeningClaimsLayout,
}

impl OpeningPlan<'_> {
    /// The opening's first transcript operation: its instance descriptor,
    /// public.
    pub(crate) fn absorb_descriptor<C: Channel>(&self, channel: &mut C) {
        channel.site(
            ProtocolSiteId {
                family: akita_params::transcript_site::SITE_FAMILY_ROOT_STATEMENT,
                ..ProtocolSiteId::default()
            }
            .into(),
        );
        channel.public_bytes(&self.descriptor_bytes);
    }

    /// The statement's claims over raw commitments, as the root fold reads
    /// them.
    pub(crate) fn raw_claims<'c, E: Clone, F: Field>(
        &self,
        claims: &'c OpeningClaims<'_, E, &akita_types::CommittedGroup<F>>,
    ) -> Result<OpeningClaims<'c, E, &'c akita_types::Commitment<F>>, AkitaError> {
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
        OpeningClaims::from_groups(raw_groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{RingVec, RingView};
    use jolt_algebra::{Fp32, Zero};

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
