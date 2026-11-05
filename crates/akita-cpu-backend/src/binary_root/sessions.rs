//! CPU sessions for a native root on the ordinary successor digit table.

use crate::opaque::sumcheck::relation_range_image::StructuredRelationWeights;
use crate::opaque::sumcheck::stage3::structured::StructuredSetupWeights;
use crate::opaque::{
    CpuBackend, CpuProofSessionHandle, CpuStage1SessionHandle, CpuStage2SessionHandle,
    CpuWitnessHandle, OperationBinding, ProofContext, RecursiveWitnessFlat, WitnessPhase,
};
use crate::sources::packed_digits::PackedSignedDigits;
use akita_error::AkitaError;
use akita_params::{DigitRangePlan, FlatBooleanDomain};
use akita_types::binary_source::TypedPrefixWeights;
use akita_types::native_root_relation::{NativeRootWeights, RootWeightProfiles};
use akita_types::DigitRangeEqualityPoint;
use jolt_algebra::{Field, Fold, Ring, Unreduced};

pub(super) fn import_root_witness<F: Field, E>(
    backend: &CpuBackend<F, E>,
    session: &CpuProofSessionHandle,
    context: &ProofContext,
    digits: Vec<i8>,
    destination_ring_dimension: usize,
) -> Result<CpuWitnessHandle, AkitaError> {
    let binding = backend.binding(session, context)?;
    if context.fold_level() != 0 || context.group_index().is_some() {
        return Err(AkitaError::InvalidInput(
            "native root witness requires the ungrouped root context".into(),
        ));
    }
    let (schedule, _) = binding.scope_lease().proof_plan()?;
    let expected_ring_dimension = schedule
        .recursive_folds
        .first()
        .map_or(schedule.terminal.d_a(), |fold| fold.params.d_a());
    if digits.len() != schedule.root.output_witness_len
        || destination_ring_dimension != expected_ring_dimension
        || digits.iter().any(|digit| !(-4..=3).contains(digit))
    {
        return Err(AkitaError::InvalidInput(
            "native root witness differs from its admitted successor geometry or digit range"
                .into(),
        ));
    }
    CpuWitnessHandle::from_cpu(
        RecursiveWitnessFlat::from_i8_digits(digits),
        destination_ring_dimension,
        binding,
    )
}

fn validate_root_witness<F: Field, E>(
    backend: &CpuBackend<F, E>,
    witness: &CpuWitnessHandle,
) -> Result<OperationBinding, AkitaError> {
    let binding = witness.operation_binding();
    backend.validate_binding(&binding)?;
    let (schedule, _) = binding.scope_lease().proof_plan()?;
    if binding.fold_level() != 0
        || binding.group_index().is_some()
        || witness.phase
            != (WitnessPhase::CommittedForSuccessor {
                producer: 0,
                successor: 1,
            })
        || witness.manifest.logical_len() != schedule.root.output_witness_len
    {
        return Err(AkitaError::InvalidInput(
            "native root session requires its committed root witness".into(),
        ));
    }
    Ok(binding)
}

pub(super) fn begin_root_digit_range<F, E>(
    backend: &CpuBackend<F, E>,
    witness: &CpuWitnessHandle,
    domain: FlatBooleanDomain,
    equality_point: &[E],
    range: DigitRangePlan,
) -> Result<CpuStage1SessionHandle<E>, AkitaError>
where
    F: Field,
    E: Field + Ring + Fold + Unreduced + 'static,
{
    let parent = validate_root_witness(backend, witness)?;
    if range.basis() != 8
        || domain.live_len() != witness.manifest.logical_len()
        || domain.domain_len() != witness.manifest.commitment_domain_len()
    {
        return Err(AkitaError::InvalidInput(
            "native root digit range differs from its committed witness domain".into(),
        ));
    }
    // The native root supplies challenges directly in physical-address order.
    let equality = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        equality_point,
        domain.num_vars(),
        0,
    )?;
    let prover = crate::opaque::sumcheck::digit_range::DigitRangeProver::from_packed_digits(
        witness.logical.packed_digits().clone(),
        range,
        domain,
        equality,
    )?;
    let session_state = crate::opaque::sumcheck::digit_range::DigitRangeSession::new(prover, None)?;
    let binding = backend.next_binding(parent)?;
    let lease = binding.scope_lease().clone();
    Ok(CpuStage1SessionHandle {
        binding,
        lease,
        session_state,
    })
}

/// Every lane of a recomposed binary family (direct A, image, prime) spans at
/// least 256 addresses and starts on a multiple of its stride, so the first
/// eight Stage 2 rounds bind only in-lane bits for those families. Narrower
/// lanes, such as mixed ordinary segments, carry their weights in per-block
/// rows, which may span lane boundaries.
const ROOT_PROFILE_BLOCK_BITS: usize = 8;

impl<E: Field + Fold> StructuredRelationWeights<E> for RootWeightProfiles<E> {
    fn len(&self) -> usize {
        RootWeightProfiles::len(self)
    }

    fn pair(&self, left: usize) -> (E, E) {
        RootWeightProfiles::pair(self, left)
    }

    fn bind(&mut self, challenge: E) {
        RootWeightProfiles::bind(self, challenge);
    }

    fn take_dense(&mut self) -> Vec<E> {
        RootWeightProfiles::take_dense(self)
    }
}

impl<E: Field> StructuredSetupWeights<E> for TypedPrefixWeights<E> {
    fn len(&self) -> usize {
        self.domain_len()
    }

    fn width(&self) -> usize {
        TypedPrefixWeights::width(self)
    }

    fn blocks(&self) -> &[usize] {
        TypedPrefixWeights::blocks(self)
    }

    fn block_terms(&self, live: usize) -> Result<[(E, &[E]); 2], AkitaError> {
        TypedPrefixWeights::block_terms(self, live)
    }

    fn bind(&mut self, challenge: E) -> Result<(), AkitaError> {
        TypedPrefixWeights::bind(self, challenge)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn begin_root_relation<F, E>(
    backend: &CpuBackend<F, E>,
    witness: &CpuWitnessHandle,
    weights: NativeRootWeights<E>,
    negative_binary_weights: Vec<(usize, E)>,
    claim: E,
    stage1_point: &[E],
    range_image_evaluation: E,
    batching: E,
) -> Result<CpuStage2SessionHandle<E>, AkitaError>
where
    F: Field,
    E: Field + Ring + Fold + Unreduced + 'static,
{
    let parent = validate_root_witness(backend, witness)?;
    let domain_len = witness.manifest.commitment_domain_len();
    if weights.table_len() != domain_len || !domain_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "native root weights differ from the committed witness domain".into(),
        ));
    }
    let block_bits = (domain_len.trailing_zeros() as usize).min(ROOT_PROFILE_BLOCK_BITS);
    let span = tracing::info_span!("binary_root_weight_profiles").entered();
    let profiles = weights.profiles(block_bits)?;
    drop(weights);
    drop(span);
    // Stage 2 reads the live prefix; the rest of the domain is zero. The
    // prefix covers whole profile blocks.
    let digits = witness.logical.packed_digits();
    let live_len = digits.len().next_multiple_of(1 << block_bits);
    let digits = if live_len == digits.len() {
        digits.clone()
    } else {
        let mut padded = Vec::new();
        padded.try_reserve_exact(live_len).map_err(|_| {
            AkitaError::InvalidInput("native root witness allocation failed".into())
        })?;
        padded.extend(digits.iter());
        padded.resize(live_len, 0);
        PackedSignedDigits::from_i8_digits_auto(padded)
    };
    let mut session = CpuStage2SessionHandle::from_structured_relation(
        digits,
        Box::new(profiles),
        block_bits,
        claim,
        stage1_point,
        range_image_evaluation,
        batching,
    )?
    .with_negative_binary_weights(negative_binary_weights)?;
    let binding = backend.next_binding(parent)?;
    let lease = binding.scope_lease().clone();
    session.set_operation_binding(binding, lease);
    Ok(session)
}

#[cfg(test)]
#[path = "stage2_oracle_tests.rs"]
mod stage2_oracle_tests;
