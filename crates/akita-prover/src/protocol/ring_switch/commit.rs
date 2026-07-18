use super::*;
use crate::compute::{CommitInnerPlan, OperationCtx};
use akita_types::{dispatch_for_field, NextWitnessBindingPolicy};

/// Public state bound for the witness produced by one intermediate fold.
pub enum NextWitnessState<F: FieldCore> {
    /// Ordinary recursive edge, bound by the outer image `u = B * decompose(t)`.
    OuterCommitment(RingVec<F>),
    /// Last recursive edge, bound directly by the canonical inner `t` state.
    TerminalInnerState {
        /// Flat canonical `t` state absorbed by the transcript.
        t_state: RingVec<F>,
        /// Original per-block inner rows retained for the terminal witness.
        recomposed_inner_rows: Vec<RingVec<F>>,
    },
}

/// Result of preparing the next logical recursive witness and its public state.
pub struct NextWitnessStateOutput<F: FieldCore> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Transcript-bound public state for the next level.
    pub binding: NextWitnessState<F>,
    /// Prover hint for opening the physical next-level witness.
    pub hint: RecursiveCommitmentHintCache<F>,
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
    commit_params: &LevelParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    commit_ctx: &OperationCtx<'_, Cfg::Field, B>,
    logical_w: &RecursiveWitnessFlat,
    binding_policy: NextWitnessBindingPolicy,
) -> Result<NextWitnessStateOutput<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<Cfg::Field>,
{
    let terminal_inner = match binding_policy {
        NextWitnessBindingPolicy::OuterCommitment => false,
        NextWitnessBindingPolicy::TerminalInnerState => true,
        NextWitnessBindingPolicy::TerminalCleartextWitness => {
            return Err(AkitaError::InvalidInput(
                "commit_w cannot bind a current-level terminal cleartext witness".into(),
            ));
        }
    };
    let dims = commit_params.role_dims();
    commit_ctx.ensure_envelope_ntt(expanded.as_ref(), dims.d_a())?;
    if !terminal_inner {
        commit_ctx.ensure_envelope_ntt(expanded.as_ref(), dims.d_b())?;
    }
    let backend = commit_ctx.backend();
    let prepared = commit_ctx.prepared();
    backend.validate_prepared_setup(prepared, expanded.as_ref())?;
    validate_commit_level_params::<Cfg::Field>(commit_params, expanded.as_ref())?;

    let (packed_witness, decomposed_inner_rows, terminal_t_state) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        dims.d_a(),
        |D_A| {
            let packed_witness = if <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE == 1 {
                None
            } else {
                Some(tensor_pack_recursive_witness::<
                    Cfg::Field,
                    Cfg::ExtField,
                    D_A,
                >(logical_w)?)
            };
            let w = packed_witness.as_ref().unwrap_or(logical_w);
            if !w.len().is_multiple_of(D_A) {
                return Err(AkitaError::InvalidSize {
                    expected: D_A,
                    actual: w.len(),
                });
            }

            let num_ring_elems = w.len() / D_A;
            tracing::debug!(
                num_ring_elems,
                num_live_blocks = commit_params.num_live_blocks,
                num_positions_per_block = commit_params.num_positions_per_block,
                depth_commit = commit_params.num_digits_commit,
                depth_open = commit_params.num_digits_open,
                position_index_bits = commit_params.position_index_bits(),
                block_index_bits = commit_params.block_index_bits(),
                inner_width = commit_params.inner_width(),
                pow2_block = 1usize << commit_params.position_index_bits(),
                "commit_w layout"
            );

            let w_view = w.view::<Cfg::Field, D_A>()?;
            let plan = CommitInnerPlan::from_level(commit_params);
            let inner = w_view.commit_inner(backend, prepared, plan)?;
            validate_commit_inner_shape::<Cfg::Field, D_A>(
                &inner,
                commit_params.num_live_blocks,
                commit_params.a_key.row_len(),
                commit_params.num_digits_open,
                commit_params.log_basis,
            )?;

            let terminal_t_state = if terminal_inner {
                let coeff_len =
                    inner
                        .recomposed_inner_rows
                        .iter()
                        .try_fold(0usize, |len, block| {
                            len.checked_add(block.coeff_len())
                                .ok_or(AkitaError::InvalidProof)
                        })?;
                let mut coeffs = Vec::with_capacity(coeff_len);
                for block in &inner.recomposed_inner_rows {
                    coeffs.extend_from_slice(block.coeffs());
                }
                Some((RingVec::from_coeffs(coeffs), inner.recomposed_inner_rows))
            } else {
                None
            };

            Ok::<_, AkitaError>((
                packed_witness,
                inner.decomposed_inner_rows,
                terminal_t_state,
            ))
        }
    )?;

    let binding = if terminal_inner {
        let (t_state, recomposed_inner_rows) = terminal_t_state.ok_or(AkitaError::InvalidProof)?;
        NextWitnessState::TerminalInnerState {
            t_state,
            recomposed_inner_rows,
        }
    } else {
        validate_commit_outer_input_nonempty(decomposed_inner_rows.total_planes())?;
        let commitment = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            Cfg::Field,
            dims.d_b(),
            |D_B| {
                let (outer_input, remainder) = decomposed_inner_rows.digits().as_chunks::<D_B>();
                if !remainder.is_empty() {
                    return Err(AkitaError::InvalidSetup(
                        "recursive commit digit carrier is not aligned to the B-role dimension"
                            .into(),
                    ));
                }
                let u: Vec<CyclotomicRing<Cfg::Field, D_B>> = backend.digit_rows::<D_B>(
                    prepared,
                    commit_params.b_key.row_len(),
                    outer_input,
                    commit_params.log_basis,
                )?;
                if u.len() != commit_params.b_key.row_len() {
                    return Err(AkitaError::InvalidProof);
                }
                Ok::<_, AkitaError>(RingVec::from_ring_elems(&u))
            }
        )?;
        NextWitnessState::OuterCommitment(commitment)
    };
    let hint = AkitaCommitmentHint::singleton(decomposed_inner_rows);
    Ok(NextWitnessStateOutput {
        witness: packed_witness,
        binding,
        hint: RecursiveCommitmentHintCache::from_hint(hint),
    })
}
