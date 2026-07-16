use super::*;
use crate::compute::{CommitInnerPlan, OperationCtx};
use akita_types::dispatch_for_field;

/// Result of committing the next logical recursive witness.
pub struct NextWitnessCommitment<F: FieldCore> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Commitment to the physical next-level witness.
    pub commitment: RingVec<F>,
    /// Prover hint for `commitment`.
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
) -> Result<NextWitnessCommitment<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<Cfg::Field>,
{
    let dims = commit_params.role_dims();
    commit_ctx.ensure_envelope_ntt(expanded.as_ref(), dims.d_a())?;
    commit_ctx.ensure_envelope_ntt(expanded.as_ref(), dims.d_b())?;
    let backend = commit_ctx.backend();
    let prepared = commit_ctx.prepared();
    backend.validate_prepared_setup(prepared, expanded.as_ref())?;
    validate_commit_level_params::<Cfg::Field>(commit_params, expanded.as_ref())?;

    let (packed_witness, decomposed_inner_rows) = dispatch_for_field!(
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
                live_block_count = commit_params.live_block_count,
                positions_per_block = commit_params.positions_per_block,
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
                commit_params.live_block_count,
                commit_params.a_key.row_len(),
                commit_params.num_digits_open,
                commit_params.log_basis,
            )?;

            Ok::<_, AkitaError>((packed_witness, inner.decomposed_inner_rows))
        }
    )?;

    validate_commit_outer_input_nonempty(decomposed_inner_rows.total_planes())?;
    let commitment = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        Cfg::Field,
        dims.d_b(),
        |D_B| {
            let (outer_input, remainder) = decomposed_inner_rows.digits().as_chunks::<D_B>();
            if !remainder.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "recursive commit digit carrier is not aligned to the B-role dimension".into(),
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
    let hint = AkitaCommitmentHint::singleton(decomposed_inner_rows);
    Ok(NextWitnessCommitment {
        witness: packed_witness,
        commitment,
        hint: RecursiveCommitmentHintCache::from_hint(hint),
    })
}
