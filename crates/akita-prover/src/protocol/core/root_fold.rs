use super::*;
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks, RuntimeCommitBackendFor,
    RuntimeRingSwitchProveBackend,
};
use crate::RecursiveWitnessFlat;
use jolt_field::AdditiveGroup;

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
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + Unreduced
        + Field
        + PseudoMersenne
        + Ring
        + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
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
    // Absorb root claims through the D-free flat commitment encoder keyed on the
    // root level's B-role dimension (byte-identical to the verifier's
    // `claims.append_to_transcript` and to the former typed path; S2/S7 parity).
    claims.append_to_transcript::<T>(root_params, transcript)?;

    let prepared_fold = prepare_single_field_fold::<F, E, T, P, C, O, TS, R>(
        stack,
        claims,
        false,
        transcript,
        0,
        root_params,
        basis,
    )
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
