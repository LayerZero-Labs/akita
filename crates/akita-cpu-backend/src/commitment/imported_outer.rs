//! Completion of root commitments whose outer image was computed elsewhere.

use super::{
    api::validate_commitment_geometry, CommitmentExecutionPlan, CommitmentStateBinding,
    CompressionOperation, PortableCompressionState,
};
use crate::opaque::{CpuBackend, CpuCompressionOperation};
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_types::{field_modulus, Commitment, CommittedGroup, GroupCommitPhaseParams, RingVec};
use jolt_field::{CanonicalEncoding, Field};

impl<Cfg: CommitmentConfig> CpuBackend<Cfg> {
    /// Finish a root commitment whose canonical outer image `u` was computed
    /// by another arithmetic backend.
    ///
    /// The input is the complete, globally aggregated B image in
    /// `[slice][outer row][ring coefficient]` order. This executes only the
    /// scheduled commitment compression. It never reads source polynomials or
    /// executes the A or B commitment stages. The external backend must retain
    /// the returned [`PortableCompressionState`] for later commitment-relation
    /// proof construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the field, frozen profile, setup capacity, or outer
    /// image shape disagrees with the root commitment plan, or if compression
    /// fails to produce valid retained state.
    pub fn compress_root_outer_image<F>(
        &self,
        profile: GroupCommitPhaseParams,
        outer_image: RingVec<F>,
    ) -> Result<(CommittedGroup<F>, PortableCompressionState<F>), AkitaError>
    where
        Cfg: CommitmentConfig<Field = F>,
        F: Field + CanonicalEncoding + 'static,
    {
        let prepared = self.prepared()?;
        let expanded = prepared.expanded();

        profile.validate_frozen_precommit(F::MODULUS_BITS)?;
        if profile.inner.matrix.sis_modulus_profile().modulus() != field_modulus::<F>()? {
            return Err(AkitaError::InvalidSetup(
                "root commitment modulus profile does not match the backend field".into(),
            ));
        }
        if profile.group.num_vars() > expanded.descriptor.max_num_vars
            || profile.group.num_polynomials() > expanded.descriptor.max_num_batched_polys
        {
            return Err(AkitaError::InvalidSetup(
                "root commitment group exceeds the backend setup limits".into(),
            ));
        }
        validate_commitment_geometry::<F>(&profile, expanded)?;

        let execution = CommitmentExecutionPlan::for_root(&profile)?;
        let required_setup = execution.max_setup_field_elements()?;
        if required_setup > expanded.descriptor.num_field_elements {
            return Err(AkitaError::InvalidSetup(format!(
                "root commitment requires {required_setup} setup field elements, but setup has {}",
                expanded.descriptor.num_field_elements
            )));
        }
        let compression_plan = execution.compression().cloned().ok_or_else(|| {
            AkitaError::InvalidSetup("root commitment has no compression plan".into())
        })?;
        let relation_mode = execution.relation_mode().ok_or_else(|| {
            AkitaError::InvalidSetup("root commitment has no compression relation mode".into())
        })?;

        if outer_image.ring_dim() != profile.outer.matrix.ring_dimension()
            || outer_image.coeff_len() != compression_plan.source_coefficients()
        {
            return Err(AkitaError::InvalidInput(
                "external root outer image does not match the frozen commitment profile".into(),
            ));
        }

        let binding = CommitmentStateBinding::new(
            expanded.descriptor.clone(),
            *execution.inner(),
            profile.group.num_polynomials(),
            Some(relation_mode),
            Some(compression_plan.clone()),
        )?;
        let operation = CpuCompressionOperation::new(self, prepared, expanded)?;
        let exporter = operation.portable_exporter();
        let output = operation.compress(&binding, &compression_plan, relation_mode, outer_image)?;
        let (terminal_payload, state) = output.into_parts();
        let compression_state = exporter.consume_compression_state(state)?;
        compression_state.validate(&compression_plan, relation_mode)?;

        Ok((
            CommittedGroup::new(profile, Commitment::new(terminal_payload)),
            compression_state,
        ))
    }
}
