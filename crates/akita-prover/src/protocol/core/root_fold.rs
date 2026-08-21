use super::*;
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks, ProverComputeStack,
    RuntimeCommitBackendFor, RuntimeRingSwitchProveBackend,
};
use crate::RecursiveWitnessFlat;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;

fn validate_packing_root_opening_shape<F, E>(
    ring_d: usize,
    alpha_bits: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FpExtEncoding<F>,
{
    let ext_degree = <E as ExtField<F>>::EXT_DEGREE;
    if ext_degree == 0
        || !ring_d.is_multiple_of(ext_degree)
        || !(ring_d / ext_degree).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "extension-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let packed_slots = ring_d / ext_degree;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: alpha_bits,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_root<F, E, T, P, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    transcript: &mut T,
    claims: ProverOpeningData<'_, E, P, F>,
    root_params: &CommittedGroupParams,
    basis: BasisMode,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: RootProverGroupOpening<F, E, O> + Clone,
    TS: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F>,
    C: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
{
    let opening_batch = claims.opening_layout()?;
    let opening_method = root_params.uniform_opening_method(&opening_batch)?;
    if !matches!(
        opening_method,
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ) || root_params.source_encoding
        != akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
    {
        return Err(AkitaError::InvalidSetup(
            "root folds require canonical coefficient packing".into(),
        ));
    }
    // A-role root fold ring dimension (schedule-derived).
    let root_ring_d = root_params.role_dims().d_a();
    let alpha_bits = root_ring_d.trailing_zeros() as usize;
    prepare_single_field_fold::<F, E, T, P, _, C, O, TS, R>(
        stack,
        claims,
        false,
        transcript,
        || validate_packing_root_opening_shape::<F, E>(root_ring_d, alpha_bits),
        root_params,
        basis,
    )
}

/// Prove the folded-root proof payload for an intermediate root.
///
/// The caller owns schedule/config selection and passes the validated schedule
/// execution for level 0. This function owns root polynomial folding, public
/// root transcript setup, root ring-relation construction, and the folded-root
/// prover mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// ring-relation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn prove_root<'stack, F, E, T, P, C, O, TS, R, Cfg>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    stacks: &'stack impl LevelProveStacks<
        'stack,
        F,
        Commit = C,
        Opening = O,
        Tensor = TS,
        RingSwitch = R,
    >,
    transcript: &mut T,
    claims: ProverOpeningData<'_, E, P, F>,
    scheduled: &akita_types::FoldParams,
    next_params: super::fold::FoldSuccessorParams<'_>,
    next_witness_binding: akita_types::NextWitnessBindingPolicy,
    basis: BasisMode,
) -> Result<ProveLevelOutput<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + PseudoMersenneField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: RootProverGroupOpening<F, E, O> + Clone,
    C: RuntimeCommitBackendFor<F, RecursiveWitnessFlat> + ComputeBackendSetup<F> + 'stack,
    O: DigitRowsComputeBackend<F> + ComputeBackendSetup<F> + 'stack,
    TS: ComputeBackendSetup<F> + 'stack,
    R: RuntimeRingSwitchProveBackend<F>
        + DigitRowsComputeBackend<F>
        + ComputeBackendSetup<F>
        + 'stack,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    <C as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <O as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <TS as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <R as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
{
    let stack = stacks.prove_stack_at_level(0);
    let root_params = &scheduled.params;
    let opening_layout = claims.opening_layout()?;
    let opening_method = root_params.uniform_opening_method(&opening_layout)?;
    if !matches!(
        opening_method,
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ) || root_params.source_encoding
        != akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
    {
        return Err(AkitaError::InvalidSetup(
            "root folds require canonical coefficient packing".into(),
        ));
    }

    // Absorb root claims through the D-free flat commitment encoder keyed on the
    // root level's B-role dimension (byte-identical to the verifier's
    // `claims.append_to_transcript` and to the former typed path; S2/S7 parity).
    claims.append_to_transcript::<T>(root_params, transcript)?;

    let prepared_fold =
        prepare_root::<F, E, T, P, C, O, TS, R>(stack, transcript, claims, root_params, basis)
            .map_err(|err| AkitaError::InvalidInput(format!("prepare root failed: {err:?}")))?;

    prove_fold::<F, E, T, C, O, TS, R, Cfg>(
        expanded,
        prefix_slots,
        stack,
        transcript,
        0,
        root_params,
        Some(next_params),
        Some(scheduled.output_witness_len),
        Some(next_witness_binding),
        prepared_fold,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("prove root fold failed: {err:?}")))
}
