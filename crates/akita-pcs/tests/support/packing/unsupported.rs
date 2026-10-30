//! Deliberately unsupported evaluation-trace rows for admission regressions.
use super::{RootCoefficientPackingConfig, UnsupportedEvaluationTraceConfig};
use akita_challenges::FoldChallengeFamily;
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::{CommittedGroupBatchProfile, ScheduleLookupKey};

impl<Base, const LEVEL: usize> UnsupportedEvaluationTraceConfig<Base, LEVEL>
where
    Base: CommitmentConfig + 'static,
{
    /// Rewrite a packing row to open by evaluation trace at `LEVEL`.
    ///
    /// Returns the raw row rather than a resolved one, so that tests hand it to
    /// catalog admission themselves and observe admission reject it.
    pub(crate) fn derive_row(
        catalog: &akita_config::ValidatedScheduleCatalog,
        key: &ScheduleLookupKey,
    ) -> Result<(CommittedGroupBatchProfile, akita_params::FoldSchedule), AkitaError> {
        let base = RootCoefficientPackingConfig::<Base>::derive_catalog_row(catalog, key, 64, 1)?;
        let profiles = base.profiles().clone();
        let mut schedule = base.schedule().clone();
        let params = if LEVEL == 0 {
            &mut schedule.root.params
        } else if LEVEL == 1 {
            let step = schedule.recursive_folds.first_mut().ok_or_else(|| {
                AkitaError::InvalidSetup("early-ET test row needs a recursive fold".into())
            })?;
            if let Some(mut prefix) = step.params.setup_prefix().copied() {
                prefix.opening.opening_method = akita_params::OpeningMethod::EvaluationTrace;
                let d_a = prefix.profile.inner.matrix.ring_dimension();
                prefix.opening.fold_challenge_config =
                    FoldChallengeFamily::production_for_ring_dim(d_a).ok_or_else(|| {
                        AkitaError::InvalidSetup("missing early-ET prefix challenge family".into())
                    })?;
                step.params.set_setup_prefix(Some(prefix))?;
            }
            &mut step.params
        } else {
            return Err(AkitaError::InvalidSetup(
                "early-ET test level must be zero or one".into(),
            ));
        };
        params.own_group_mut().opening.opening_method =
            akita_params::OpeningMethod::EvaluationTrace;
        params.own_group_mut().opening.fold_challenge_config =
            FoldChallengeFamily::production_for_ring_dim(params.d_a()).ok_or_else(|| {
                AkitaError::InvalidSetup("missing early-ET witness challenge family".into())
            })?;
        params.source_encoding = akita_params::CommittedSourceEncoding::for_producer(
            params.opening_method(),
            Self::EXT_DEGREE,
            params.d_a(),
            0,
            LEVEL == 0,
        );
        if LEVEL == 1 {
            params.witness_chunk = akita_params::ChunkedWitnessCfg {
                num_chunks: 2,
                num_activated_levels: 2,
            };
            schedule.recursive_folds[0]
                .params
                .own_group_mut()
                .opening
                .fold_challenge_config = params.own_group_mut().opening.fold_challenge_config;
        } else {
            // EvaluationTrace expands the root witness. Retarget the successor's
            // frozen commitment geometry so structure accepts this test row.
            let opening_batch = key.opening_layout()?;
            let output_witness_len = schedule.root.params.output_witness_len_for_field_bits(
                catalog.policy().decomposition.field_bits(),
                Self::EXT_DEGREE,
                &opening_batch,
            )?;
            schedule.root.output_witness_len = output_witness_len;
            let successor = &mut schedule.recursive_folds[0];
            successor.input_witness_len = output_witness_len;
            let successor_d_a = successor.params.d_a();
            let blocks = &mut successor.params.own_group_mut().profile.blocks;
            blocks.live_ring_elements_per_claim = output_witness_len.div_ceil(successor_d_a);
            blocks.live_blocks = blocks
                .live_ring_elements_per_claim
                .div_ceil(blocks.positions_per_block);
            let successor_params = &mut successor.params;
            let dims = successor_params.role_dims();
            let group = successor_params.own_group_mut();
            let group_num_vars = group
                .profile
                .blocks
                .live_ring_elements_per_claim
                .checked_mul(successor_d_a)
                .and_then(usize::checked_next_power_of_two)
                .ok_or_else(|| AkitaError::InvalidSetup("early-ET root input overflow".into()))?
                .trailing_zeros() as usize;
            group.profile.group = akita_params::PolynomialGroupLayout::singleton(group_num_vars);
            let outer_width = akita_params::CommitmentSliceGeometry::try_new(
                group.profile.outer_slice_count,
                group.profile.blocks.live_blocks,
                1,
                group.profile.inner.matrix.output_rank(),
                group.profile.outer.digits.num_digits,
                dims.d_a(),
                dims.d_b(),
            )?
            .physical_input_width();
            group.profile.outer.matrix =
                akita_params::OuterCommitMatrixParams::try_new_with_min_rank(
                    group.profile.outer.matrix.sis_table_key(),
                    outer_width,
                )?;
        }
        Ok((profiles, schedule))
    }
}
