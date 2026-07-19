use super::*;
use crate::compute::{
    CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks,
    ProverComputeStack, RuntimeOpeningProveBackendFor, RuntimeRingSwitchProveBackend,
    RuntimeRootProvePoly, RuntimeTensorBackendFor,
};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;

fn validate_non_eor_root_opening_shape<F, E>(
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
    root_params: &LevelParams,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    terminal_tail_t_vectors: Option<usize>,
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
    P: RuntimeRootProvePoly<F>,
    TS: RuntimeTensorBackendFor<F, P, E>,
    O: DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, P>
        + RuntimeOpeningProveBackendFor<F, RootTensorProjectionPoly<F>>,
    C: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F>,
{
    let opening_batch = claims.opening_layout::<F>()?;
    let num_claims = opening_batch.num_total_polynomials();
    let opening_num_vars = opening_batch.max_num_vars();
    // A-role root fold ring dimension (schedule-derived).
    let root_ring_d = root_params.role_dims().d_a();
    let alpha_bits = root_ring_d.trailing_zeros() as usize;
    let needs_extension_reduction =
        root_tensor_projection_enabled::<F, E>(root_ring_d, opening_num_vars);

    if claims.point().len() > opening_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: opening_num_vars,
            actual: claims.point().len(),
        });
    }
    let flat_polys = claims.flat_polys();
    if flat_polys.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }

    let eor_opening_batch =
        OpeningClaims::with_padded_point(claims.point(), opening_num_vars, num_claims)?;
    let non_eor_protocol_point = claims.point().to_vec();
    prepare_fold_inner::<F, E, T, P, _, C, O, TS, R>(
        stack,
        needs_extension_reduction,
        claims,
        &flat_polys,
        &eor_opening_batch,
        false,
        transcript,
        non_eor_protocol_point,
        || validate_non_eor_root_opening_shape::<F, E>(root_ring_d, alpha_bits),
        root_params,
        alpha_bits,
        basis,
        relation_matrix_row_layout,
        terminal_tail_t_vectors,
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
pub fn prove_root<'stack, F, E, T, P, C, O, TS, R, Cfg>(
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
    scheduled: &ExecutionSchedule,
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
    P: RuntimeRootProvePoly<F>,
    C: CommitmentComputeBackend<F> + ComputeBackendSetup<F> + 'stack,
    O: RuntimeOpeningProveBackendFor<F, P>
        + RuntimeOpeningProveBackendFor<F, RootTensorProjectionPoly<F>>
        + DigitRowsComputeBackend<F>
        + ComputeBackendSetup<F>
        + 'stack,
    TS: RuntimeTensorBackendFor<F, P, E>
        + RuntimeTensorBackendFor<F, RootTensorProjectionPoly<F>, E>
        + ComputeBackendSetup<F>
        + 'stack,
    R: RuntimeRingSwitchProveBackend<F> + ComputeBackendSetup<F> + 'stack,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    <C as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <O as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <TS as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <R as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
{
    let stack = stacks.prove_stack_at_level(0);
    let opening_batch = claims.opening_layout::<F>()?;
    let num_claims = opening_batch.num_total_polynomials();
    let root_params = &scheduled.params;

    if claims.flat_polys().len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }

    // Absorb root claims through the D-free flat commitment encoder keyed on the
    // root level's B-role dimension (byte-identical to the verifier's
    // `claims.append_to_transcript` and to the former typed path; S2/S7 parity).
    claims.append_to_transcript::<T>(root_params.role_dims().d_b(), transcript)?;

    let prepared_fold = prepare_root::<F, E, T, P, C, O, TS, R>(
        stack,
        transcript,
        claims,
        root_params,
        RelationMatrixRowLayout::WithDBlock,
        None,
        basis,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("prepare root failed: {err:?}")))?;

    prove_fold::<F, E, T, C, O, TS, R, Cfg>(
        expanded,
        prefix_slots,
        stack,
        transcript,
        0,
        scheduled,
        prepared_fold,
        false,
        None,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("prove root fold failed: {err:?}")))?
    .get_intermediate()
}
