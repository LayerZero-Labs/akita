use super::*;
use crate::compute::{CommitInnerPlan, OperationCtx};

/// Result of committing the next logical recursive witness.
pub struct NextWitnessCommitment<F: FieldCore> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Commitment to the physical next-level witness.
    pub commitment: RingVec<F>,
    /// Prover hint for `commitment`.
    pub hint: RecursiveCommitmentHintCache<F>,
}

/// Commit the D-agnostic ring-switch witness `w` at the caller-selected ring
/// dimension.
///
/// This is the D-boundary in the protocol: ring switching produces a flat
/// witness using the current level's ring dimension, then this function
/// re-chunks that witness into `D`-sized ring elements and commits it with the
/// recursive commitment layout supplied by the root scheduler.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by `D` or if the
/// recursive inner commitment fails.
#[tracing::instrument(skip_all, name = "commit_w")]
#[inline(never)]
pub fn commit_w<F, B, const D: usize>(
    w: &RecursiveWitnessFlat,
    expanded: &AkitaExpandedSetup<F>,
    commit_ctx: &OperationCtx<'_, F, B>,
    commit_layout: &LevelParams,
) -> Result<(RingVec<F>, AkitaCommitmentHint<F>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    let backend = commit_ctx.backend();
    let prepared = commit_ctx.prepared();
    if commit_layout.ring_dimension != D {
        return Err(AkitaError::InvalidInput(format!(
            "commit_w layout D={} does not match target D={D}",
            commit_layout.ring_dimension
        )));
    }
    if !w.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: w.len(),
        });
    }
    backend.validate_prepared_setup(prepared, expanded)?;
    validate_commit_level_params::<F, D>(commit_layout, expanded)?;

    let num_ring_elems = w.len() / D;
    tracing::debug!(
        num_ring_elems,
        num_blocks = commit_layout.num_blocks,
        block_len = commit_layout.block_len,
        depth_commit = commit_layout.num_digits_commit,
        depth_open = commit_layout.num_digits_open,
        m_vars = commit_layout.m_vars,
        r_vars = commit_layout.r_vars,
        inner_width = commit_layout.inner_width(),
        pow2_block = 1usize << commit_layout.m_vars,
        "commit_w layout"
    );

    let w_view = w.view::<F, D>()?;
    let plan = CommitInnerPlan::from_level(commit_layout);
    let inner = w_view.commit_inner(backend, prepared, plan)?;
    validate_commit_inner_shape::<F, D>(
        &inner,
        commit_layout.num_blocks,
        commit_layout.a_key.row_len(),
        commit_layout.num_digits_open,
        commit_layout.log_basis,
    )?;

    let typed_digits = inner.decomposed_inner_rows_trusted::<D>()?;
    let outer_input = typed_digits.flat_digits().to_vec();
    validate_commit_outer_input_nonempty(outer_input.len())?;
    let u: Vec<CyclotomicRing<F, D>> = if commit_layout.f_key.is_some() {
        // Tiered: u_final = F·decompose(blockdiag(B')·t̂). ZK blinding of the F
        // tier is a non-goal; tiered proofs are exercised non-zk.
        crate::api::commitment::tiered_commit_u_final::<F, D, B>(
            backend,
            prepared,
            commit_layout,
            &outer_input,
        )?
    } else {
        let u: Vec<CyclotomicRing<F, D>> = backend.digit_rows::<D>(
            prepared,
            commit_layout.b_key.row_len(),
            &outer_input,
            commit_layout.log_basis,
        )?;
        if u.len() != commit_layout.b_key.row_len() {
            return Err(AkitaError::InvalidProof);
        }
        u
    };
    // Protocol storage is D-free: the typed digit planes flatten into a
    // `DigitBlocks` hint; recomposed rows are recomputed on demand downstream.
    let hint = AkitaCommitmentHint::singleton(inner.decomposed_inner_rows.clone());
    Ok((RingVec::from_ring_elems(&u), hint))
}

/// Commit the next recursive witness under config `Cfg`.
///
/// The commitment ring dimension is schedule-owned (`commit_params.ring_dimension`).
/// This adapter warms the target NTT slot on the caller's D-free prepared setup,
/// dispatches locally to the typed `commit_w` kernel, and returns D-free protocol
/// storage.
///
/// # Errors
///
/// Returns an error if layout selection, commitment, cache preparation, or
/// D-erased hint construction fails.
#[inline(never)]
pub fn commit_next_witness<Cfg, B>(
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
    let commit_d = commit_params.ring_dimension;
    commit_ctx.ensure_envelope_ntt(expanded.as_ref(), commit_d)?;
    dispatch_ring_dim_result!(commit_d, |D_COMMIT| {
        if <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE == 1 {
            let (wc, wh) = commit_w::<Cfg::Field, B, { D_COMMIT }>(
                logical_w,
                expanded.as_ref(),
                commit_ctx,
                commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: None,
                commitment: wc,
                hint: RecursiveCommitmentHintCache::from_hint(wh),
            })
        } else {
            // The tensor pack is length-preserving, so the committed witness
            // fits the schedule's recursive commit params directly.
            let committed_w =
                tensor_pack_recursive_witness::<Cfg::Field, Cfg::ExtField, { D_COMMIT }>(
                    logical_w,
                )?;
            let (wc, wh) = commit_w::<Cfg::Field, B, { D_COMMIT }>(
                &committed_w,
                expanded.as_ref(),
                commit_ctx,
                commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: Some(committed_w),
                commitment: wc,
                hint: RecursiveCommitmentHintCache::from_hint(wh),
            })
        }
    })
}
