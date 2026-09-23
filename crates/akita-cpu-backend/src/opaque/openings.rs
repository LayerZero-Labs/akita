//! Openings retain their source and computation identity through fold acceptance.
use super::owned::CommittedSource;
use crate::opaque::CpuWitnessHandle;
use crate::opaque::*;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, Ring, Unreduced};
use std::sync::Arc;

pub(super) enum RetainedOpeningSource<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F>,
> {
    Commitment(Arc<CommittedSource<F, E, Cfg>>),
    Witness(Box<CpuWitnessHandle>),
}

pub(super) enum PreparedOpeningSource<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F>,
> {
    Retained(RetainedOpeningSource<F, E, Cfg>),
    TerminalNative,
}
impl<F, E, Cfg> OpaqueOpeningKernel<F, E> for CpuBackend<Cfg>
where
    Cfg: akita_config::CommitmentConfig<Field = F>,
    F: Field
        + CanonicalEncoding
        + AkitaSerialize
        + jolt_field::PseudoMersenne
        + Ring
        + Unreduced
        + 'static,
    F::Wide: From<F> + jolt_field::AdditiveGroup,
    E: ExtField<F>
        + FpExtEncoding<F>
        + MulBaseUnreduced<F>
        + Unreduced
        + Fold
        + Ring
        + AkitaSerialize
        + 'static,
{
    fn prepare_opening(
        &self,
        context: &ProofContext,
        source: OpeningSource<'_, Self::CommitmentHandle, Self::WitnessHandle>,
        plan: &ValidatedRecursiveGroupOpeningPlan<'_, E>,
    ) -> Result<PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError> {
        self.validate_extension::<E>()?;
        self.validate_context(context)?;
        let (parameters, _) = self.admitted_group(context)?;
        if plan.ring_dimension() != parameters.inner_commit_matrix_params().ring_dimension()
            || plan.positions_per_block() != parameters.num_positions_per_block()
            || plan.live_blocks() != parameters.num_live_blocks()
            || plan.opening_method() != parameters.opening_method()
        {
            return Err(AkitaError::InvalidInput(
                "opening does not match the admitted group".into(),
            ));
        }
        let binding = self.binding(context)?;
        let opening = match source {
            OpeningSource::Commitment(handle) => {
                if handle.owner != self.owner().backend_id() {
                    return Err(AkitaError::InvalidInput(
                        "commitment belongs to another backend".into(),
                    ));
                }
                self.owner()
                    .validate_commitment(context, handle.committed.commitment_id)?;
                if handle.committed.parameters != parameters.profile {
                    return Err(AkitaError::InvalidProof);
                }
                if handle.committed.metadata.num_vars() < plan.point().len()
                    || handle.committed.parameters.inner.matrix.ring_dimension()
                        != plan.ring_dimension()
                {
                    return Err(AkitaError::InvalidInput(
                        "commitment and opening geometry disagree".into(),
                    ));
                }
                handle.committed.source.opening(
                    self,
                    context,
                    plan,
                    PreparedOpeningSource::Retained(RetainedOpeningSource::Commitment(
                        handle.committed.clone(),
                    )),
                )?
            }
            OpeningSource::Witness(witness) => {
                self.validate_binding(&witness.operation_binding())?;
                binding.validate_lineage(&witness.operation_binding())?;
                let (schedule, _) = self.owner().proof_plan(context.scope_id())?;
                let parameters = if context.fold_level() == 0 {
                    &schedule.root.params
                } else {
                    &schedule
                        .recursive_folds
                        .get(context.fold_level() as usize - 1)
                        .ok_or(AkitaError::InvalidProof)?
                        .params
                };
                witness.operation_binding().validate_group(
                    context.group_index().ok_or(AkitaError::InvalidProof)?,
                    parameters.groups().len(),
                )?;
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Inner),
                    F,
                    plan.ring_dimension(),
                    |D| {
                        crate::opaque::prepare_recursive_witness_opening::<F, E, Cfg, D>(
                            self,
                            Some(self.prepared()?),
                            binding,
                            PreparedOpeningSource::Retained(RetainedOpeningSource::Witness(
                                Box::new(witness.snapshot()),
                            )),
                            witness,
                            plan,
                        )
                    }
                )?
            }
        };
        let (messages, opening) = opening.into_parts();
        #[cfg(feature = "response-model-diagnostics")]
        let opening = {
            let mut opening = opening;
            if crate::opaque::fold::response_model_diagnostics_enabled() {
                let source_l2_sq = match opening.source()? {
                    RetainedOpeningSource::Commitment(source) => source.source.source_l2_sq(),
                    RetainedOpeningSource::Witness(witness) => witness.source_l2_sq::<F>(),
                };
                opening.set_source_l2_sq(source_l2_sq);
            }
            opening
        };
        Ok(PreparedGroupOpening::new(messages, opening))
    }
    fn probe_opening_fold(
        &self,
        context: &ProofContext,
        opening: &Self::PreparedOpeningHandle,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedFoldHandle>, AkitaError> {
        self.validate_context(context)?;
        let (parameters, chunks) = self.admitted_group(context)?;
        let (negative, positive) = akita_types::sis::balanced_digit_representable_bounds(
            parameters.log_basis_open(),
            parameters.num_digits_fold(),
        );
        let l2 = match parameters.inner_commit_matrix_params().security_route() {
            InnerCommitSecurityRoute::Linf(_) => None,
            InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } => Some(response_l2_sq_cap),
        };
        if plan.ring_dimension() != parameters.inner_commit_matrix_params().ring_dimension()
            || plan.num_positions_per_block() != parameters.num_positions_per_block()
            || plan.num_digits() != parameters.num_digits_inner()
            || plan.log_basis() != parameters.log_basis_inner()
            || plan.opening_method() != parameters.opening_method()
            || plan.challenges().num_claims() != parameters.profile.group.num_polynomials()
            || plan.challenges().num_live_blocks_per_claim() != parameters.num_live_blocks()
            || plan.acceptance().digit_negative_abs_bound() != negative
            || plan.acceptance().digit_positive_bound() != positive
            || plan.acceptance().response_l2_sq_cap() != l2
            || plan
                .geometry()
                .chunk_ranges()
                .map_or(1, |ranges| ranges.len())
                != chunks
        {
            return Err(AkitaError::InvalidInput(
                "fold probe does not match the admitted group policy".into(),
            ));
        }
        let binding = opening.operation_binding();
        self.validate_binding(&binding)?;
        let expected = self.binding(context)?;
        binding.validate_lineage(&expected)?;
        if context.group_index().is_some() {
            binding.validate_group(
                context.group_index().ok_or(AkitaError::InvalidProof)?,
                usize::MAX,
            )?;
        }
        let outcome = match opening.source()? {
            RetainedOpeningSource::Commitment(source) => source.source.probe(self, plan)?,
            RetainedOpeningSource::Witness(witness) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                F,
                plan.ring_dimension(),
                |D| {
                    crate::opaque::consumer_kernels::RecursiveWitnessFoldKernel::<_,F,D>::probe_recursive_witness(self, Some(self.prepared()?), witness, plan)
                }
            )?,
        };
        Ok(match outcome {
            FoldProbeOutcome::Rejected => FoldProbeOutcome::Rejected,
            FoldProbeOutcome::Accepted {
                mut fold_handle,
                diagnostics,
            } => {
                // Retain the opening's operation ID: dimension and group agreement
                // alone would permit combining different openings of one source.
                fold_handle.bind(binding);
                #[cfg(feature = "response-model-diagnostics")]
                let diagnostics = diagnostics.with_source_l2_sq(opening.source_l2_sq());
                FoldProbeOutcome::Accepted {
                    fold_handle,
                    diagnostics,
                }
            }
        })
    }
}
