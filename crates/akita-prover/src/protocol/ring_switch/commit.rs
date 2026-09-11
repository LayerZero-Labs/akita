use super::*;
use crate::api::commitment::commit_outer_slices;
use crate::compute::compression::{
    execute_compression_chains, CompressionExecutionInput, CompressionRelationOutput,
};
use crate::compute::{CommitInnerPlan, OperationCtx, RuntimeCommitBackendFor};
use crate::kernels::linear::decompose_commit_blocks_into;
use akita_types::{
    dispatch_for_field, CommittedSourceEncoding, CompressionChainPlan, TerminalFoldParams,
};

/// Public state bound for the witness produced by one intermediate fold.
pub enum NextWitnessState<F: Field> {
    /// Ordinary recursive edge, bound by the terminal compressed payload.
    OuterPayload(RingVec<F>),
    /// Last recursive edge, bound directly by the canonical inner `t` state.
    TerminalInnerState,
}

/// Result of preparing the next logical recursive witness and its public state.
pub struct NextWitnessStateOutput<F: Field> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Transcript-bound public state for the next level.
    pub binding: NextWitnessState<F>,
    /// Prover hint for opening the physical next-level witness.
    pub hint: AkitaCommitmentHint<F>,
}

/// Commit the next recursive witness under config `Cfg`.
///
/// The commitment ring dimension is schedule-owned (`commit_params.ring_dimension`).
/// This function warms the target NTT slot on the caller's D-free prepared setup,
/// dispatches locally to the typed commit kernel, and returns D-free protocol
/// storage.
///
/// # Errors
///
/// Returns an error if layout selection, commitment, cache preparation, or
/// D-erased hint construction fails.
#[inline(never)]
pub fn commit_w<Cfg, B>(
    commit_params: &CommittedGroupParams,
    fold_level: usize,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    commit_ctx: &OperationCtx<'_, Cfg::Field, B>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessStateOutput<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    B: RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>,
{
    let dims = commit_params.role_dims();
    let backend = commit_ctx.backend();
    let prepared = commit_ctx.prepared();
    backend.validate_prepared_setup(prepared, expanded.as_ref())?;
    validate_commit_level_params::<Cfg::Field>(commit_params, expanded.as_ref(), fold_level, 1)?;
    let slice_geometry = commit_params.own_group().profile.derive_slice_geometry()?;

    let (packed_witness, inner_rows, commitment, compression_witness) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        dims.d_a(),
        |D_A| {
            let packed_witness = match commit_params.source_encoding {
                CommittedSourceEncoding::CanonicalCoefficientTable => None,
                CommittedSourceEncoding::TensorSubfieldProjection { extension_degree } => {
                    if extension_degree != <Cfg::ExtField as ExtField<Cfg::Field>>::DEGREE {
                        return Err(AkitaError::InvalidSetup(
                            "recursive tensor source encoding does not match the protocol extension degree"
                                .into(),
                        ));
                    }
                    Some(tensor_pack_recursive_witness::<
                        Cfg::Field,
                        Cfg::ExtField,
                        D_A,
                    >(logical_w)?)
                }
            };
            let w = packed_witness.as_ref().unwrap_or(logical_w);
            let committed_coeff_len = w.committed_coeff_len()?;
            if !committed_coeff_len.is_multiple_of(D_A) {
                return Err(AkitaError::InvalidSize {
                    expected: D_A,
                    actual: committed_coeff_len,
                });
            }

            let num_ring_elems = committed_coeff_len / D_A;
            tracing::debug!(
                num_ring_elems,
                num_live_blocks = commit_params.blocks().live_blocks,
                num_positions_per_block = commit_params.blocks().positions_per_block,
                depth_commit = commit_params.inner().digits.num_digits,
                depth_open = commit_params.open().digits.num_digits,
                position_index_bits = commit_params.position_index_bits(),
                block_index_bits = commit_params.block_index_bits(),
                inner_width = commit_params.inner_width(),
                pow2_block = 1usize << commit_params.position_index_bits(),
                "commit_w layout"
            );

            let w_view = w.view::<Cfg::Field, D_A>()?;
            let plan = CommitInnerPlan::from_level(commit_params);
            let inner_group = backend.commit_inner_group(prepared, vec![w_view], plan)?;
            let [inner] = inner_group
                .try_into()
                .map_err(|_: Vec<_>| AkitaError::InvalidProof)?;
            validate_commit_inner_shape::<Cfg::Field, D_A>(
                &inner,
                commit_params.blocks().live_blocks,
                commit_params.inner().matrix.output_rank(),
            )?;
            let n_a = commit_params.inner().matrix.output_rank();
            let blocks = (0..commit_params.blocks().live_blocks)
                .map(|block| inner.block_rows::<D_A>(block, n_a))
                .collect::<Result<Vec<_>, _>>()?;
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                Cfg::Field,
                dims.d_b(),
                |D_B| {
                    let decomposed_inner_rows = decompose_commit_blocks_into::<Cfg::Field, D_A, D_B>(
                        &blocks,
                        commit_params.outer().digits.num_digits,
                        commit_params.outer().digits.log_basis,
                    )?;
                    let u: Vec<CyclotomicRing<Cfg::Field, D_B>> = commit_outer_slices(
                        backend,
                        prepared,
                        commit_params.outer().matrix.output_rank(),
                        std::iter::once(&decomposed_inner_rows),
                        &slice_geometry,
                        commit_params.outer().digits.log_basis,
                    )?;
                    let source = RingVec::from_ring_elems(&u);
                    if !commit_params.payload_mode.is_compressed() {
                        Ok::<_, AkitaError>((packed_witness, inner.into_inner_rows(), source, None))
                    } else {
                        let plan = CompressionChainPlan::for_complete_source(
                            commit_params.outer().matrix.sis_table_key().modulus_profile,
                            source.coeff_len(),
                        )?;
                        let (mut outputs, _) = execute_compression_chains(
                            commit_ctx,
                            vec![CompressionExecutionInput {
                                id: (),
                                plan,
                                coefficients: source.into_coeffs(),
                                relation_mode: commit_params.ring_relation_mode,
                            }],
                        )?;
                        let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
                        let terminal_ring_dim = output
                            .witness
                            .plan()
                            .maps()
                            .last()
                            .ok_or(AkitaError::InvalidProof)?
                            .ring_dimension();
                        let payload = RingVec::from_coeffs_with_ring_dim(
                            output.terminal.into_coefficients(),
                            terminal_ring_dim,
                        )?;
                        Ok::<_, AkitaError>((
                            packed_witness,
                            inner.into_inner_rows(),
                            payload,
                            Some((output.witness, output.relation)),
                        ))
                    }
                }
            )
        }
    )?;
    let hint = match compression_witness {
        Some((compression_witness, CompressionRelationOutput::QuotientLift { quotients })) => {
            AkitaCommitmentHint::singleton_with_outer_compression(
                inner_rows,
                &compression_witness,
                &quotients,
            )?
        }
        Some((compression_witness, CompressionRelationOutput::ReducedEvaluation)) => {
            AkitaCommitmentHint::singleton_with_reduced_outer_compression(
                inner_rows,
                &compression_witness,
            )?
        }
        None => AkitaCommitmentHint::singleton(inner_rows)?,
    };
    Ok(NextWitnessStateOutput {
        witness: packed_witness,
        binding: NextWitnessState::OuterPayload(commitment),
        hint,
    })
}

/// Bind the witness entering the terminal fold with its canonical inner
/// commitment state. No outer digits or outer commitment are computed.
#[inline(never)]
pub fn commit_terminal_w<Cfg, B>(
    commit_params: &TerminalFoldParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    commit_ctx: &OperationCtx<'_, Cfg::Field, B>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessStateOutput<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    B: RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>,
{
    let ring_dim = commit_params.d_a();
    let backend = commit_ctx.backend();
    let prepared = commit_ctx.prepared();
    backend.validate_prepared_setup(prepared, expanded.as_ref())?;

    let (packed_witness, t_state) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        ring_dim,
        |D_A| {
            let packed_witness = if <Cfg::ExtField as ExtField<Cfg::Field>>::DEGREE == 1 {
                None
            } else {
                Some(tensor_pack_recursive_witness::<
                    Cfg::Field,
                    Cfg::ExtField,
                    D_A,
                >(logical_w)?)
            };
            let witness = packed_witness.as_ref().unwrap_or(logical_w);
            let view = witness.view::<Cfg::Field, D_A>()?;
            let plan = CommitInnerPlan {
                n_a: commit_params.inner.matrix.output_rank(),
                num_positions_per_block: commit_params.blocks.positions_per_block,
                num_digits_inner: commit_params.inner.digits.num_digits,
                log_basis_inner: commit_params.inner.digits.log_basis,
            };
            let inner_group = backend.commit_inner_group(prepared, vec![view], plan)?;
            let [inner] = inner_group
                .try_into()
                .map_err(|_: Vec<_>| AkitaError::InvalidProof)?;
            Ok::<_, AkitaError>((packed_witness, inner.into_inner_rows()))
        }
    )?;
    Ok(NextWitnessStateOutput {
        witness: packed_witness,
        binding: NextWitnessState::TerminalInnerState,
        hint: AkitaCommitmentHint::singleton(t_state)?,
    })
}
