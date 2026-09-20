mod recursive_kernels;
pub(crate) use recursive_kernels::prepare_recursive_witness_opening;
mod tensor;

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

use crate::sources::packed_digits::{PackedSignedDigitView, PackedSignedDigits};
use crate::sources::poly_helpers::packed_tight_digit_fold_partitioned;
use crate::opaque::CpuBackend;
use akita_types::{RingVec, WitnessLayout};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::opaque::DecomposeFoldWitness;

#[derive(Clone)]
enum OpaquePreparedGroupOpeningKind<F: Field, E: Field> {
    EvaluationTrace {
        point: akita_types::PreparedOpeningPoint<F, E>,
        folded_by_claim: Vec<akita_types::RingVec<F>>,
    },
    CoefficientPacking {
        point: akita_types::PreparedSubringCoefficientPackingPoint<E>,
        partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
    },
}

/// Consumer-private prepared opening state. Its witness-derived rows never
/// appear in a protocol-facing carrier.
#[doc(hidden)]
#[derive(Clone)]
pub struct CpuPreparedOpeningHandle<F: Field, E: Field> {
    binding: crate::opaque::OperationBinding,
    kind: OpaquePreparedGroupOpeningKind<F, E>,
    scalar_openings: Vec<E>,
    terminal_native: bool,
    retained_source: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

impl<F: Field, E: Field> CpuPreparedOpeningHandle<F, E> {
    pub(crate) fn mark_terminal_native(&mut self) {
        self.terminal_native = true;
    }

    pub(crate) const fn is_terminal_native(&self) -> bool {
        self.terminal_native
    }

    pub(crate) fn retain_source<S: Send + Sync + 'static>(&mut self, source: S) {
        self.retained_source = Some(std::sync::Arc::new(source));
    }
    pub(crate) fn source<S: Send + Sync + 'static>(&self) -> Result<&S, AkitaError> {
        self.retained_source.as_ref().and_then(|source| source.downcast_ref()).ok_or_else(|| AkitaError::InvalidInput("opening has no matching retained source".into()))
    }

    pub(crate) fn scalar_openings(&self) -> &[E] {
        &self.scalar_openings
    }

    pub(crate) const fn operation_binding(
        &self,
    ) -> crate::opaque::OperationBinding {
        self.binding
    }

    pub(crate) fn set_operation_binding(
        &mut self,
        binding: crate::opaque::OperationBinding,
    ) {
        self.binding = binding;
    }

    pub(crate) fn relation_opening<const D: usize>(
        &self,
        level: &akita_types::CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        geometry: &akita_types::RelationWitnessGeometry,
        group_index: usize,
        group_dims: akita_types::CommitmentRingDims,
    ) -> Result<
        (
            crate::opaque::PreparedOpeningWitness<F>,
            crate::opaque::PublicPreparedRelationOpening<F, E>,
        ),
        AkitaError,
    >
    where
        F: CanonicalEncoding,
    {
        let group = level.group_params_geometry(opening_batch, group_index)?;
        match &self.kind {
            OpaquePreparedGroupOpeningKind::EvaluationTrace {
                point,
                folded_by_claim,
            } => {
                if group.opening_method() != akita_types::OpeningMethod::EvaluationTrace
                    || point.ring_multiplier_point.position_len() != group.num_positions_per_block()
                    || point.ring_multiplier_point.fold_len() != group.num_live_blocks()
                    || folded_by_claim.len()
                        != opening_batch.group_layout(group_index)?.num_polynomials()
                {
                    return Err(AkitaError::InvalidInput(
                        "batched prover EvaluationTrace point layout mismatch".into(),
                    ));
                }
                let opening =
                    crate::opaque::PreparedOpeningWitness::evaluation_trace::<D, E>(
                        point,
                        folded_by_claim,
                        group_dims.d_a() / group_dims.d_d(),
                        group.num_digits_open(),
                        group.log_basis_open(),
                    )?;
                Ok((
                    opening,
                    akita_types::OpeningFamily::EvaluationTrace(point.clone()),
                ))
            }
            OpaquePreparedGroupOpeningKind::CoefficientPacking {
                point,
                partials_by_claim,
            } => {
                if point.num_positions_per_block() != group.num_positions_per_block()
                    || point.num_live_blocks() != group.num_live_blocks()
                    || geometry.group_opening_method(group_index)? != group.opening_method()
                {
                    return Err(AkitaError::InvalidInput(
                        "batched prover coefficient-packing point layout mismatch".into(),
                    ));
                }
                let opening =
                    crate::opaque::PreparedOpeningWitness::coefficient_packing::<D>(
                        level,
                        opening_batch,
                        geometry,
                        group_index,
                        partials_by_claim.clone(),
                    )?;
                Ok((
                    opening,
                    akita_types::OpeningFamily::SubringCoefficientPacking(point.clone()),
                ))
            }
        }
    }
}

macro_rules! impl_prepared_group_opening_kernel {
    ($backend:ty) => {
        impl<F, E> crate::opaque::PreparedGroupOpeningKernel<F, E> for $backend
        where
            F: Field + CanonicalEncoding,
            E: Field + 'static,
        {
            fn retain_evaluation_trace_opening(
                &self,
                proof_context: Option<&crate::opaque::ProofContext>,
                _prepared: Option<&Self::PreparedSetup>,
                point: akita_types::PreparedOpeningPoint<F, E>,
                folded_by_claim: Vec<akita_types::RingVec<F>>,
                scalar_openings: Vec<E>,
            ) -> Result<
                crate::opaque::PreparedGroupOpening<E, Self::PreparedOpeningHandle>,
                AkitaError,
            > {
                Ok(crate::opaque::PreparedGroupOpening::new(
                    scalar_openings.clone(),
                    CpuPreparedOpeningHandle {
                        binding: proof_context
                            .map(|context| self.binding(context))
                            .transpose()?
                            .unwrap_or_else(
                                crate::opaque::OperationBinding::legacy_unscoped,
                            ),
                        scalar_openings,
                        terminal_native: false,
                        retained_source: None,
                        kind: OpaquePreparedGroupOpeningKind::EvaluationTrace {
                            point,
                            folded_by_claim,
                        },
                    },
                ))
            }

            fn retain_coefficient_packing_opening(
                &self,
                proof_context: Option<&crate::opaque::ProofContext>,
                _prepared: Option<&Self::PreparedSetup>,
                point: akita_types::PreparedSubringCoefficientPackingPoint<E>,
                partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
                scalar_openings: Vec<E>,
            ) -> Result<
                crate::opaque::PreparedGroupOpening<E, Self::PreparedOpeningHandle>,
                AkitaError,
            > {
                Ok(crate::opaque::PreparedGroupOpening::new(
                    scalar_openings.clone(),
                    CpuPreparedOpeningHandle {
                        binding: proof_context
                            .map(|context| self.binding(context))
                            .transpose()?
                            .unwrap_or_else(
                                crate::opaque::OperationBinding::legacy_unscoped,
                            ),
                        scalar_openings,
                        terminal_native: false,
                        retained_source: None,
                        kind: OpaquePreparedGroupOpeningKind::CoefficientPacking {
                            point,
                            partials_by_claim,
                        },
                    },
                ))
            }

            fn terminal_evaluation_trace_opening(
                &self,
                _prepared: Option<&Self::PreparedSetup>,
                opening: Self::PreparedOpeningHandle,
            ) -> Result<Vec<akita_types::RingVec<F>>, AkitaError> {
                match opening.kind {
                    OpaquePreparedGroupOpeningKind::EvaluationTrace {
                        folded_by_claim, ..
                    } => Ok(folded_by_claim),
                    OpaquePreparedGroupOpeningKind::CoefficientPacking { .. } => {
                        Err(AkitaError::InvalidProof)
                    }
                }
            }
        }
    };
}

impl_prepared_group_opening_kernel!(crate::opaque::CpuBackend);

/// D-agnostic owner for the recursive witness vector `w`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(unreachable_pub)]
pub(crate) struct RecursiveWitnessFlat {
    digits: PackedSignedDigits,
    live_coeff_len: usize,
    committed_coeff_len: Option<usize>,
    commitment_ring_dim: Option<usize>,
}

/// Opaque CPU-consumer handle for a complete recursive witness.
///
/// Protocol orchestration can transport this handle and query public shape
/// metadata, but the packed coefficient representation remains private to the
/// recursive-witness adapter.
pub struct CpuWitnessHandle {
    pub(crate) pending_successor: Option<u32>,
    pub(crate) relation_plan: Option<Arc<akita_types::RelationRangeImagePlan>>,
    pub(super) manifest: crate::opaque::RecursiveWitnessManifest,
    pub(super) binding: crate::opaque::OperationBinding,
    pub(super) logical: RecursiveWitnessFlat,
    pub(super) committed: Option<RecursiveWitnessFlat>,
}

pub(crate) type OpaqueRecursiveWitness = CpuWitnessHandle;

impl CpuWitnessHandle {
    pub(crate) fn snapshot(&self) -> Self {
        Self { pending_successor: self.pending_successor, relation_plan:self.relation_plan.clone(),manifest: self.manifest, binding: self.binding, logical: self.logical.clone(), committed: self.committed.clone() }
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) fn source_l2_sq<F: Field>(&self) -> Option<u128> {
        crate::opaque::RootPolyMeta::<F>::exact_integer_coeff_l2_sq(
            self.committed.as_ref().unwrap_or(&self.logical),
        )
    }

    pub(crate) const fn operation_binding(
        &self,
    ) -> crate::opaque::OperationBinding {
        self.binding
    }

    pub(crate) fn set_operation_binding(
        &mut self,
        binding: crate::opaque::OperationBinding,
    ) {
        self.binding = binding;
    }
}

/// Consumer-owned state retained between recursive opening preparation and EOR.
pub(crate) struct CpuWitnessOpeningHandle<E: Field> {
    source_operation: u128,
    point: Vec<E>,
    tensor_evals: Vec<E>,
    witness_len: usize,
    ring_dimension: usize,
}

pub struct CpuRelationHandle {
    pub(crate) relation_plan: Option<Arc<akita_types::RelationRangeImagePlan>>,
    binding: crate::opaque::OperationBinding,
    packed: PackedSignedDigits,
}

pub struct CpuStage2SessionHandle<E: Field> {
    binding: crate::opaque::OperationBinding,
    prover: super::relation_range_image::RelationRangeImageProver<E>,
    claim: E,
    next_round: usize,
    pending: Option<akita_algebra::uni_poly::UniPoly<E>>,
}

pub struct CpuStage1SessionHandle<E: Field> {
    pub(crate) binding: crate::opaque::OperationBinding,
    pub(crate) session_state: super::digit_range::DigitRangeSession<E>,
}

pub(crate) struct CpuExtensionOpeningSession<E: Field> {
    prover: crate::opaque::recursive::opening::ExtensionOpeningReductionProver<E>,
    claim: E,
    next_round: usize,
    pending: Option<akita_algebra::uni_poly::UniPoly<E>>,
    num_terms: usize,
}

pub(crate) type OpaqueWitnessOpeningState<E> = CpuWitnessOpeningHandle<E>;
pub(crate) type ConsumerRelationWitness = CpuRelationHandle;
pub(crate) type ConsumerStage2Session<E> = CpuStage2SessionHandle<E>;

macro_rules! impl_bound_handle {
    ($handle:ident $(<$field:ident>)?) => {
        impl$(<$field: Field>)? $handle$(<$field>)? {
            pub(crate) const fn operation_binding(
                &self,
            ) -> crate::opaque::OperationBinding {
                self.binding
            }

            pub(crate) fn set_operation_binding(
                &mut self,
                binding: crate::opaque::OperationBinding,
            ) {
                self.binding = binding;
            }
        }
    };
}

impl_bound_handle!(CpuRelationHandle);
impl<E: Field> CpuStage1SessionHandle<E> {
    pub(crate) const fn operation_binding(
        &self,
    ) -> crate::opaque::OperationBinding {
        self.binding
    }
}
impl_bound_handle!(CpuStage2SessionHandle<E>);

#[cfg(test)]
pub(crate) fn cpu_extension_opening_session<E>(
    group: crate::opaque::recursive::opening::ExtensionOpeningReductionGroup<E>,
    input_claim: E,
) -> Result<Box<dyn crate::opaque::eor::ExtensionOpeningSession<E>>, AkitaError>
where
    E: Field + jolt_field::Unreduced + jolt_field::Fold + 'static,
{
    Ok(Box::new(
        cpu_witness_eor_session(
            group,
            input_claim,
        )?
        ,
    ))
}

fn cpu_witness_eor_session<E>(
    group: crate::opaque::recursive::opening::ExtensionOpeningReductionGroup<E>,
    input_claim: E,
 ) -> Result<CpuExtensionOpeningSession<E>, AkitaError>
where
    E: Field + jolt_field::Unreduced + jolt_field::Fold + 'static,
{
    let num_terms = group.num_terms();
    Ok(CpuExtensionOpeningSession {
            prover: crate::opaque::recursive::opening::ExtensionOpeningReductionProver::new(
                vec![group],
                input_claim,
            )?,
            claim: input_claim,
            next_round: 0,
            pending: None,
            num_terms,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_extension_opening_session_from_witnesses<F, E>(
    witnesses: Vec<Vec<E>>,
    claim_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
    extra_point: Vec<E>,
    input_claim: E,
) -> Result<Box<dyn crate::opaque::eor::ExtensionOpeningSession<E>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + 'static,
{
    Ok(Box::new(
        cpu_witness_eor_session_from_witnesses::<F, E>(
            witnesses,
            claim_coefficients,
            tail_point,
            eta,
            extra_point,
            input_claim,
        )?
        ,
    ))
}

#[allow(clippy::too_many_arguments)]
fn cpu_witness_eor_session_from_witnesses<F, E>(
    witnesses: Vec<Vec<E>>,
    claim_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
    extra_point: Vec<E>,
    input_claim: E,
 ) -> Result<CpuExtensionOpeningSession<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + 'static,
{
    if witnesses.len() != claim_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: witnesses.len(),
            actual: claim_coefficients.len(),
        });
    }
    let terms = witnesses
        .into_iter()
        .zip(claim_coefficients)
        .map(|(witness, &coefficient)| {
            crate::opaque::recursive::opening::ExtensionOpeningReductionTerm::new(
                witness,
                coefficient,
            )
        })
        .collect();
    let factor = akita_types::tensor_equality_factor_evals::<F, E>(tail_point, eta)?;
    let group = crate::opaque::recursive::opening::ExtensionOpeningReductionGroup::new(
        terms, factor,
    )?
    .extend_cylindrically(extra_point)?;
    cpu_witness_eor_session(group, input_claim)
}

impl<E> crate::opaque::eor::ExtensionOpeningSession<E> for CpuExtensionOpeningSession<E>
where
    E: Field + jolt_field::Unreduced + jolt_field::Fold,
{
    fn num_rounds(&self) -> usize {
        akita_sumcheck::SumcheckInstanceProver::num_rounds(&self.prover)
    }

    fn num_terms(&self) -> usize {
        self.num_terms
    }

    fn round_polynomial(
        &mut self,
        round: usize,
        previous_claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        if round != self.next_round || self.pending.is_some() || previous_claim != self.claim {
            return Err(AkitaError::InvalidInput(
                "extension-opening session round order or claim mismatch".into(),
            ));
        }
        let polynomial = akita_sumcheck::SumcheckInstanceProver::compute_round_univariate(
            &mut self.prover,
            round,
            previous_claim,
        );
        if polynomial.degree() > akita_types::EXTENSION_OPENING_REDUCTION_DEGREE
            || polynomial.evaluate(&E::zero()) + polynomial.evaluate(&E::one()) != previous_claim
        {
            return Err(AkitaError::InvalidInput(
                "extension-opening session returned an invalid round polynomial".into(),
            ));
        }
        self.pending = Some(polynomial.clone());
        Ok(polynomial)
    }

    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::InvalidInput(
                "extension-opening session challenge order mismatch".into(),
            ));
        }
        let polynomial = self.pending.take().ok_or_else(|| {
            AkitaError::InvalidInput("extension-opening session challenge arrived early".into())
        })?;
        self.claim = polynomial.evaluate(&challenge);
        akita_sumcheck::SumcheckInstanceProver::ingest_challenge(
            &mut self.prover,
            round,
            challenge,
        );
        self.next_round += 1;
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<Vec<(E, E, E)>, AkitaError> {
        if self.pending.is_some() || self.next_round != self.num_rounds() {
            return Err(AkitaError::InvalidInput(
                "extension-opening session finished before all rounds".into(),
            ));
        }
        akita_sumcheck::SumcheckInstanceProver::finalize(&mut self.prover);
        self.prover.final_terms().ok_or_else(|| {
            AkitaError::InvalidInput("extension-opening session has no final claims".into())
        })
    }
}

impl<E> crate::opaque::consumer_kernels::RelationWitnessSession<E> for ConsumerStage2Session<E>
where
    E: Field + Ring + jolt_field::Unreduced + jolt_field::Fold + 'static,
{
    fn num_rounds(&self) -> usize {
        akita_sumcheck::SumcheckInstanceProver::num_rounds(&self.prover)
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn round_polynomial(
        &mut self,
        round: usize,
        previous_claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        if round != self.next_round || self.pending.is_some() || previous_claim != self.claim {
            return Err(AkitaError::InvalidInput(
                "relation session round order or claim mismatch".into(),
            ));
        }
        let polynomial = akita_sumcheck::SumcheckInstanceProver::compute_round_univariate(
            &mut self.prover,
            round,
            previous_claim,
        );
        if polynomial.degree() > 3
            || polynomial.evaluate(&E::zero()) + polynomial.evaluate(&E::one()) != previous_claim
        {
            return Err(AkitaError::InvalidInput(
                "relation session returned an invalid round polynomial".into(),
            ));
        }
        self.pending = Some(polynomial.clone());
        Ok(polynomial)
    }

    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::InvalidInput(
                "relation session challenge order mismatch".into(),
            ));
        }
        let polynomial = self.pending.take().ok_or_else(|| {
            AkitaError::InvalidInput("relation session challenge arrived early".into())
        })?;
        self.claim = polynomial.evaluate(&challenge);
        akita_sumcheck::SumcheckInstanceProver::ingest_challenge(
            &mut self.prover,
            round,
            challenge,
        );
        self.next_round += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<crate::opaque::RelationWitnessFinalClaims<E>, AkitaError> {
        if self.pending.is_some() || self.next_round != self.num_rounds() {
            return Err(AkitaError::InvalidInput(
                "relation session finished before all rounds".into(),
            ));
        }
        akita_sumcheck::SumcheckInstanceProver::finalize(&mut self.prover);
        if self.claim != self.prover.expected_final_claim()? {
            return Err(AkitaError::InvalidInput(
                "relation session final claim disagrees with its folded oracle".into(),
            ));
        }
        Ok(crate::opaque::RelationWitnessFinalClaims::new(
            self.prover.final_w_eval(),
            self.claim,
        ))
    }
}

impl CpuRelationHandle {
    pub(crate) fn len(&self) -> usize {
        self.packed.len()
    }
}

impl<F, E, B>
    crate::opaque::consumer_kernels::RecursiveWitnessRelationKernel<ConsumerRelationWitness, F, E>
    for B
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: Field + Ring + jolt_field::Unreduced + jolt_field::Fold + akita_types::FpExtEncoding<F> + jolt_field::MulBaseUnreduced<F> + 'static,
    B: crate::opaque::ComputeBackendSetup<F>,
{
    type Session = ConsumerStage2Session<E>;

    fn begin_relation_session(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: ConsumerRelationWitness,
        plan: crate::opaque::ValidatedRelationSessionPlan<'_, F, E>,
    ) -> Result<Self::Session, AkitaError> {
        let prepared = prepared.ok_or_else(|| {
            AkitaError::InvalidInput("Stage 2 requires prepared backend state".into())
        })?;
        if witness.packed.len() != plan.witness_len() {
            return Err(AkitaError::InvalidInput(
                "Stage 2 plan disagrees with its witness manifest".into(),
            ));
        }
        let weights = crate::opaque::relation_weights::compile_stage2_weights(
            self.prepared_expanded_setup(prepared), &plan,
        )?;
        let additional = (!weights.linear.is_empty() || !weights.binary_intervals.is_empty())
            .then(|| {
                super::relation_range_image::AdditionalRelationTerms::new(
                    &witness.packed,
                    plan.domain_len(),
                    weights.linear,
                    &weights.binary_intervals,
                    plan.stage1_point(),
                    plan.binary_batching(),
                )
            })
            .transpose()?;
        let ordinary_claim = plan.relation_claim() + plan.physical_l2_claim()
            - additional.as_ref().map_or_else(
                E::zero,
                super::relation_range_image::AdditionalRelationTerms::input_claim,
            );
        let relation_weights = match weights.ordinary {
            crate::opaque::RelationWeightDescription::QuotientFactored(weights) => {
                super::relation_range_image::RelationWeightOracle::QuotientFactored(weights)
            }
            crate::opaque::RelationWeightDescription::ReducedEvaluations {
                evaluations,
                live_len,
            } => super::relation_range_image::RelationWeightOracle::ReducedDense(
                super::relation_range_image::DenseRelationWeights::new(evaluations, live_len)?,
            ),
        };
        let batching_coefficient = plan.batching_coefficient();
        let stage1_point = plan.stage1_point().to_vec();
        let range_image_evaluation = plan.range_image_evaluation();
        let basis = plan.basis();
        let live_lane_count = plan.live_lane_count();
        let lane_bits = plan.lane_bits();
        let coefficient_bits = plan.coefficient_bits();
        let linear_opening_claim = plan.linear_opening_claim();
        let linear_terms = match plan.into_linear_terms() {
            crate::opaque::Stage2OpeningDescription::EvaluationTrace {
                trace,
                output_scale,
            } => {
                let coefficient_count = trace.relation_coefficient_block_len;
                let weights = super::relation_range_image::build_evaluation_trace_weights(trace)?;
                super::relation_range_image::PreparedProverLinearTerms::from_evaluation_trace(
                    &weights,
                    coefficient_count,
                    output_scale,
                )?
            }
            crate::opaque::Stage2OpeningDescription::CoefficientPacking(terms) => {
                let mut terms = terms.into_iter();
                let mut prepared = super::relation_range_image::PreparedProverLinearTerms::from_coefficient_packing(terms.next().ok_or(AkitaError::InvalidProof)?)?;
                for term in terms {
                    prepared.merge(super::relation_range_image::PreparedProverLinearTerms::from_coefficient_packing(term)?)?;
                }
                prepared
            }
        };
        let prover = super::relation_range_image::RelationRangeImageProver::new(
            batching_coefficient,
            witness.packed,
            &stage1_point,
            range_image_evaluation,
            basis,
            relation_weights,
            live_lane_count,
            lane_bits,
            coefficient_bits,
            ordinary_claim,
            linear_terms,
            linear_opening_claim,
            additional,
        )?;
        let claim = akita_sumcheck::SumcheckInstanceProver::input_claim(&prover);
        Ok(ConsumerStage2Session {
            binding: witness.binding.for_operation(0),
            prover,
            claim,
            next_round: 0,
            pending: None,
        })
    }
}

impl<F, E, B>
    crate::opaque::consumer_kernels::RecursiveWitnessStage1Kernel<ConsumerRelationWitness, F, E>
    for B
where
    F: Field + CanonicalEncoding,
    E: Field + Ring + jolt_field::Unreduced + jolt_field::Fold + 'static,
    B: crate::opaque::ComputeBackendSetup<F>,
{
    type Session = super::digit_range::DigitRangeSession<E>;

    fn begin_stage1(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &ConsumerRelationWitness,
        plan: &crate::opaque::ValidatedStage1Plan<E>,
    ) -> Result<Self::Session, AkitaError> {
        prepared.ok_or_else(|| {
            AkitaError::InvalidInput("Stage 1 requires prepared backend state".into())
        })?;
        if witness.len() != plan.witness_len()
            || plan.domain().live_len() != witness.len()
        {
            return Err(AkitaError::InvalidInput(
                "Stage 1 plan disagrees with its witness or operation context".into(),
            ));
        }
        super::digit_range::DigitRangeSession::new(
            super::DigitRangeProver::from_packed_digits(
                witness.packed.clone(),
                plan.digit_range(),
                plan.domain(),
                plan.equality(),
            )?,
            plan.physical(),
        )
    }

    fn stage1_round_polynomial(
        &self,
        session: &mut Self::Session,
        step: crate::opaque::Stage1Step,
        round: usize,
        previous_local_claim: E,
    ) -> Result<crate::opaque::Stage1RoundPolynomial<E>, AkitaError> {
        session.round_polynomial(step, round, previous_local_claim)
    }

    fn bind_stage1_challenge(
        &self,
        session: &mut Self::Session,
        step: crate::opaque::Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        session.bind_challenge(step, round, challenge)
    }

    fn stage1_public_transition(
        &self,
        session: &mut Self::Session,
        step: crate::opaque::Stage1Step,
    ) -> Result<crate::opaque::Stage1PublicTransition<E>, AkitaError> {
        session.public_transition(step)
    }

    fn bind_stage1_batch_challenge(
        &self,
        session: &mut Self::Session,
        transition: crate::opaque::Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError> {
        session.bind_batch_challenge(transition, challenge)
    }

    fn finish_stage1(
        &self,
        session: Self::Session,
    ) -> Result<crate::opaque::Stage1FinalClaims<E>, AkitaError> {
        session.finish()
    }
}

impl<F, B>
    crate::opaque::consumer_kernels::RecursiveRelationWitnessKernel<OpaqueRecursiveWitness, F>
    for B
where
    F: Field + CanonicalEncoding,
    B: crate::opaque::ComputeBackendSetup<F>,
{
    type RelationWitness = ConsumerRelationWitness;

    fn prepare_relation_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &OpaqueRecursiveWitness,
        plan: &crate::opaque::ValidatedRelationWitnessPlan,
    ) -> Result<crate::opaque::PreparedRelationWitness<Self::RelationWitness>, AkitaError> {
        prepared.ok_or_else(|| {
            AkitaError::InvalidInput("relation witness preparation requires prepared setup".into())
        })?;
        if witness.logical.live_coeff_len() != plan.witness_len()
        {
            return Err(AkitaError::InvalidInput(
                "relation witness plan has a different operation context".into(),
            ));
        }
        let (packed, column_bits, coefficient_bits) = crate::opaque::build_w_evals_compact(
            witness.logical.packed_digits().clone(),
            plan.coefficient_count(),
            plan.extension_degree(),
            plan.opening_source_len(),
        )?;
        Ok(crate::opaque::PreparedRelationWitness::new(
            ConsumerRelationWitness {
                relation_plan:witness.relation_plan.clone(),
                binding: witness.binding.for_operation(0),
                packed,
            },
            column_bits,
            coefficient_bits,
        ))
    }
}
