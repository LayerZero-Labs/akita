use super::{
    BackendStateRef, CommitmentStateBinding, CompressionState, InnerImage, ResolvedCommitSource,
    UncompressedCommitPlan,
};
use crate::compute::CommitInnerPlan;
use akita_error::AkitaError;
use akita_types::{CompressionChainPlan, RingRelationMode, RingVec};
use jolt_field::Field;

/// State output of one inner commitment stage.
pub struct InnerCommitOutput {
    image: BackendStateRef<InnerImage>,
}

impl InnerCommitOutput {
    /// Bind a checked resident inner image to the stage output.
    pub fn new(image: BackendStateRef<InnerImage>) -> Self {
        Self { image }
    }

    /// Borrow the resident inner image.
    pub const fn image(&self) -> &BackendStateRef<InnerImage> {
        &self.image
    }

    /// Consume the output into its resident inner image.
    pub fn into_image(self) -> BackendStateRef<InnerImage> {
        self.image
    }
}

/// Input accepted by an outer commitment stage.
pub enum InnerImageInput<'a, F: Field> {
    /// Same-owner resident image.
    Owned(&'a BackendStateRef<InnerImage>),
    /// Explicitly exported canonical host rows, one `RingVec` per source.
    HostRows(&'a [RingVec<F>]),
}

/// Output of an inner-plus-outer operation.
pub struct UncompressedCommitmentOutput<F: Field> {
    image: BackendStateRef<InnerImage>,
    u: RingVec<F>,
}

impl<F: Field> UncompressedCommitmentOutput<F> {
    /// Construct after validating the B image against the checked plan.
    pub fn new(
        image: BackendStateRef<InnerImage>,
        u: RingVec<F>,
        plan: &UncompressedCommitPlan,
    ) -> Result<Self, AkitaError> {
        if image.binding().inner_plan() != plan.inner()
            || u.ring_dim() != plan.outer().ring_dimension()
            || u.coeff_len() != plan.outer().output_coefficient_len()?
        {
            return Err(AkitaError::InvalidInput(
                "uncompressed commitment output disagrees with its checked plan".into(),
            ));
        }
        Ok(Self { image, u })
    }

    /// Borrow the resident inner image.
    pub const fn image(&self) -> &BackendStateRef<InnerImage> {
        &self.image
    }

    /// Borrow the canonical stacked B image.
    pub const fn u(&self) -> &RingVec<F> {
        &self.u
    }

    /// Consume into the resident image and canonical B image.
    pub fn into_parts(self) -> (BackendStateRef<InnerImage>, RingVec<F>) {
        (self.image, self.u)
    }
}

/// Output of commitment compression with mode-bound retained state.
pub struct CompressionStageOutput<F: Field> {
    terminal_payload: RingVec<F>,
    state: BackendStateRef<CompressionState>,
}

/// Output of a complete inner, outer, and compression execution.
pub struct FullCommitmentOutput<F: Field> {
    image: BackendStateRef<InnerImage>,
    compression: CompressionStageOutput<F>,
}

impl<F: Field> FullCommitmentOutput<F> {
    /// Join stage outputs that belong to the same checked request.
    pub fn new(
        image: BackendStateRef<InnerImage>,
        compression: CompressionStageOutput<F>,
    ) -> Result<Self, AkitaError> {
        if image.binding() != compression.state().binding() {
            return Err(AkitaError::InvalidInput(
                "complete commitment stage outputs belong to different requests".into(),
            ));
        }
        Ok(Self { image, compression })
    }

    /// Resident inner image retained for later opening.
    pub const fn image(&self) -> &BackendStateRef<InnerImage> {
        &self.image
    }

    /// Public terminal compressed payload.
    pub const fn terminal_payload(&self) -> &RingVec<F> {
        self.compression.terminal_payload()
    }

    /// Resident mode-specific compression state.
    pub const fn compression_state(&self) -> &BackendStateRef<CompressionState> {
        self.compression.state()
    }

    /// Consume into the resident inner image and compression-stage output.
    pub fn into_parts(self) -> (BackendStateRef<InnerImage>, CompressionStageOutput<F>) {
        (self.image, self.compression)
    }
}

impl<F: Field> CompressionStageOutput<F> {
    /// Construct after checking the retained state and terminal payload against
    /// the exact compression plan.
    pub fn new(
        terminal_payload: RingVec<F>,
        state: BackendStateRef<CompressionState>,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<Self, AkitaError> {
        let terminal_map = plan.maps().last().ok_or_else(|| {
            AkitaError::InvalidSetup("compression chain has no terminal map".into())
        })?;
        if state.binding().relation_mode() != Some(relation_mode)
            || terminal_payload.ring_dim() != terminal_map.ring_dimension()
            || terminal_payload.coeff_len() != plan.terminal_coefficients()
        {
            return Err(AkitaError::InvalidInput(
                "compression stage output disagrees with its checked plan or state binding".into(),
            ));
        }
        Ok(Self {
            terminal_payload,
            state,
        })
    }

    /// Terminal public payload.
    pub const fn terminal_payload(&self) -> &RingVec<F> {
        &self.terminal_payload
    }

    /// Mode-bound retained compression state.
    pub const fn state(&self) -> &BackendStateRef<CompressionState> {
        &self.state
    }

    /// Consume into the public terminal payload and retained state.
    pub fn into_parts(self) -> (RingVec<F>, BackendStateRef<CompressionState>) {
        (self.terminal_payload, self.state)
    }
}

/// Object-safe inner commitment operation.
pub trait InnerCommitOperation<F: Field>: Send + Sync {
    /// Commit one request-compiled same-shape source group.
    fn commit_inner(
        &self,
        binding: &CommitmentStateBinding,
        plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError>;
}

/// Object-safe outer commitment operation.
pub trait OuterCommitOperation<F: Field>: Send + Sync {
    /// Decompose the inner image and apply canonical B slicing.
    fn commit_outer(
        &self,
        plan: &UncompressedCommitPlan,
        inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError>;
}

/// Object-safe commitment-compression operation.
pub trait CompressionOperation<F: Field>: Send + Sync {
    /// Compress one canonical B image in the selected relation mode.
    fn compress(
        &self,
        binding: &CommitmentStateBinding,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
        u: RingVec<F>,
    ) -> Result<CompressionStageOutput<F>, AkitaError>;
}

/// Explicitly fused inner-plus-outer operation.
pub trait FusedInnerOuterOperation<F: Field>: Send + Sync {
    /// Execute both stages from the same resolved source representations.
    fn commit_inner_outer(
        &self,
        binding: &CommitmentStateBinding,
        plan: &UncompressedCommitPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<UncompressedCommitmentOutput<F>, AkitaError>;
}

/// Explicit owner-specific host export edge for an inner image.
pub trait InnerImageExportOperation<F: Field>: Send + Sync {
    /// Export canonical source-ordered rows from a resident image.
    fn export_inner_rows(
        &self,
        plan: &CommitInnerPlan,
        image: &BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{CommitmentStateBinding, StateOwnerCapability};
    use akita_types::{AkitaSetupDescriptor, AkitaSetupSeed, SisModulusProfileId};
    use jolt_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn compression_state(relation_mode: RingRelationMode) -> BackendStateRef<CompressionState> {
        let owner = StateOwnerCapability::<CompressionState>::new();
        let binding = CommitmentStateBinding::new(
            AkitaSetupDescriptor {
                max_num_vars: 8,
                max_num_batched_polys: 1,
                num_field_elements: 64,
                setup_seed: AkitaSetupSeed::shake256_paged_v1([9; 32]),
            },
            CommitInnerPlan {
                ring_dimension: 64,
                num_live_blocks: 1,
                n_a: 1,
                num_positions_per_block: 1,
                num_digits_inner: 1,
                log_basis_inner: 1,
            },
            1,
            Some(relation_mode),
        )
        .unwrap();
        owner.bind(binding, 0, ())
    }

    #[test]
    fn compression_output_rejects_wrong_terminal_shape_and_mode() {
        let mode = RingRelationMode::QuotientLift;
        let plan =
            CompressionChainPlan::for_complete_source(SisModulusProfileId::Q128OffsetA7F7, 64)
                .unwrap();
        let terminal_dimension = plan.maps().last().unwrap().ring_dimension();
        let correct = RingVec::from_coeffs_with_ring_dim(
            vec![F::default(); plan.terminal_coefficients()],
            terminal_dimension,
        )
        .unwrap();
        let state = compression_state(mode);

        assert!(CompressionStageOutput::new(correct.clone(), state.clone(), &plan, mode).is_ok());
        assert!(CompressionStageOutput::new(
            RingVec::from_coeffs(correct.clone().into_coeffs()),
            state.clone(),
            &plan,
            mode,
        )
        .is_err());
        assert!(CompressionStageOutput::new(
            RingVec::from_coeffs_with_ring_dim(
                vec![F::default(); plan.terminal_coefficients() * 2],
                terminal_dimension,
            )
            .unwrap(),
            state.clone(),
            &plan,
            mode,
        )
        .is_err());
        assert!(CompressionStageOutput::new(
            correct,
            state,
            &plan,
            RingRelationMode::ReducedEvaluation,
        )
        .is_err());
    }
}
