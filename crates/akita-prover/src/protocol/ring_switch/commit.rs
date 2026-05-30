use super::*;

/// Result of committing the next logical recursive witness.
pub struct NextWitnessCommitment<F: FieldCore> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Commitment to the physical next-level witness.
    pub commitment: FlatRingVec<F>,
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
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    commit_layout: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
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
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
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
    let inner = w_view.commit_inner_witness(
        backend,
        prepared,
        commit_layout.a_key.row_len(),
        commit_layout.block_len,
        commit_layout.num_blocks,
        commit_layout.num_digits_commit,
        commit_layout.num_digits_open,
        commit_layout.log_basis,
    )?;
    validate_commit_inner_witness_shape(
        &inner,
        commit_layout.num_blocks,
        commit_layout.a_key.row_len(),
        commit_layout.num_digits_open,
        commit_layout.log_basis,
    )?;

    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(commit_layout.b_key.row_len(), commit_layout.log_basis)?;
    let outer_input = inner.decomposed_inner_rows.flat_digits().to_vec();
    validate_commit_outer_input_nonempty(outer_input.len())?;
    #[cfg(feature = "zk")]
    let mut u: Vec<CyclotomicRing<F, D>> = backend.digit_rows::<D>(
        prepared,
        commit_layout.b_key.row_len(),
        &outer_input,
        commit_layout.log_basis,
    )?;
    #[cfg(not(feature = "zk"))]
    let u: Vec<CyclotomicRing<F, D>> = backend.digit_rows::<D>(
        prepared,
        commit_layout.b_key.row_len(),
        &outer_input,
        commit_layout.log_basis,
    )?;
    #[cfg(feature = "zk")]
    {
        let blinding_rows = backend.zk_b_digit_rows::<D>(
            prepared,
            commit_layout.b_key.row_len(),
            b_blinding_digits.flat_digits().len(),
            b_blinding_digits.flat_digits(),
        )?;
        for (row, blinding) in u.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
    }
    if u.len() != commit_layout.b_key.row_len() {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(feature = "zk")]
    let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
        inner.decomposed_inner_rows,
        inner.recomposed_inner_rows,
        b_blinding_digits,
    );
    #[cfg(not(feature = "zk"))]
    let hint = {
        AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
            inner.decomposed_inner_rows,
            inner.recomposed_inner_rows,
        )
    };
    Ok((RingCommitment { u }, hint))
}

/// Dispatch a recursive `w` commitment to the selected ring dimension.
///
/// The prover crate owns typed backend preparation and `commit_w` execution.
/// Callers supply the config-specific layout policy for the selected
/// commitment dimension.
///
/// # Errors
///
/// Returns an error if layout selection, backend preparation, commitment, or
/// D-erased hint conversion fails.
#[allow(clippy::type_complexity)]
#[inline(never)]
fn dispatch_commit_w_with_layout_policy<F, L, B, Layout>(
    backend: &B,
    commit_params: LevelParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<F>>,
    logical_w: &RecursiveWitnessFlat,
    layout_for_d: Layout,
) -> Result<NextWitnessCommitment<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    L: ExtField<F>,
    B: CommitmentComputeBackend<F>,
    Layout: Fn(usize, &LevelParams, usize) -> Result<LevelParams, AkitaError>,
{
    let commit_d = commit_params.ring_dimension;
    dispatch_ring_dim_result!(commit_d, |D_COMMIT| {
        let prepared_commit = backend.prepare_expanded::<D_COMMIT>(expanded.clone())?;
        if L::EXT_DEGREE == 1 {
            let commit_layout = layout_for_d(D_COMMIT, &commit_params, logical_w.len())?;
            let (wc, wh) = commit_w::<F, B, { D_COMMIT }>(
                logical_w,
                expanded.as_ref(),
                backend,
                &prepared_commit,
                &commit_layout,
            )?;
            Ok(NextWitnessCommitment {
                witness: None,
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        } else {
            let committed_w = tensor_pack_recursive_witness::<F, L, { D_COMMIT }>(logical_w)?;
            let commit_layout = layout_for_d(D_COMMIT, &commit_params, committed_w.len())?;
            let (wc, wh) = commit_w::<F, B, { D_COMMIT }>(
                &committed_w,
                expanded.as_ref(),
                backend,
                &prepared_commit,
                &commit_layout,
            )?;
            Ok(NextWitnessCommitment {
                witness: Some(committed_w),
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        }
    })
}

/// Commit the next recursive witness using caller-supplied layout policy.
///
/// The same-D fast path reuses the caller's prepared backend context. Cross-D
/// commitments prepare a typed backend context for the target ring dimension.
///
/// # Errors
///
/// Returns an error if layout selection, commitment, backend preparation, or
/// D-erased hint conversion fails.
#[allow(clippy::type_complexity)]
#[inline(never)]
pub fn commit_next_w_with_policy<F, L, B, SameLayout, DispatchLayout, const D: usize>(
    commit_params: &LevelParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<F>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    logical_w: &RecursiveWitnessFlat,
    same_d_layout: SameLayout,
    dispatch_layout: DispatchLayout,
) -> Result<NextWitnessCommitment<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    L: ExtField<F>,
    B: CommitmentComputeBackend<F>,
    SameLayout: FnOnce(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
    DispatchLayout: Fn(usize, &LevelParams, usize) -> Result<LevelParams, AkitaError>,
{
    if commit_params.ring_dimension == D {
        if L::EXT_DEGREE == 1 {
            let commit_layout = same_d_layout(commit_params, logical_w.len())?;
            let (wc, wh) = commit_w::<F, B, D>(
                logical_w,
                expanded.as_ref(),
                backend,
                prepared,
                &commit_layout,
            )?;
            Ok(NextWitnessCommitment {
                witness: None,
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        } else {
            let committed_w = tensor_pack_recursive_witness::<F, L, D>(logical_w)?;
            let commit_layout = same_d_layout(commit_params, committed_w.len())?;
            let (wc, wh) = commit_w::<F, B, D>(
                &committed_w,
                expanded.as_ref(),
                backend,
                prepared,
                &commit_layout,
            )?;
            Ok(NextWitnessCommitment {
                witness: Some(committed_w),
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        }
    } else {
        dispatch_commit_w_with_layout_policy::<F, L, B, DispatchLayout>(
            backend,
            commit_params.clone(),
            expanded,
            logical_w,
            dispatch_layout,
        )
    }
}
