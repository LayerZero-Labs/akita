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
    let commit_d = commit_params.ring_dimension;
    commit_ctx.ensure_envelope_ntt(expanded.as_ref(), commit_d)?;
    dispatch_ring_dim_result!(commit_d, |D| {
        let packed_witness = if <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE == 1 {
            None
        } else {
            Some(tensor_pack_recursive_witness::<Cfg::Field, Cfg::ExtField, D>(logical_w)?)
        };
        let w = packed_witness.as_ref().unwrap_or(logical_w);
        let backend = commit_ctx.backend();
        let prepared = commit_ctx.prepared();
        if commit_params.ring_dimension != D {
            return Err(AkitaError::InvalidInput(format!(
                "commit_w layout D={} does not match target D={D}",
                commit_params.ring_dimension
            )));
        }
        if !w.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: w.len(),
            });
        }
        backend.validate_prepared_setup(prepared, expanded.as_ref())?;
        validate_commit_level_params::<Cfg::Field, D>(commit_params, expanded.as_ref())?;

        let num_ring_elems = w.len() / D;
        tracing::debug!(
            num_ring_elems,
            num_blocks = commit_params.num_blocks,
            block_len = commit_params.block_len,
            depth_commit = commit_params.num_digits_commit,
            depth_open = commit_params.num_digits_open,
            m_vars = commit_params.m_vars,
            r_vars = commit_params.r_vars,
            inner_width = commit_params.inner_width(),
            pow2_block = 1usize << commit_params.m_vars,
            "commit_w layout"
        );

        let w_view = w.view::<Cfg::Field, D>()?;
        let plan = CommitInnerPlan::from_level(commit_params);
        let inner = w_view.commit_inner(backend, prepared, plan)?;
        validate_commit_inner_shape::<Cfg::Field, D>(
            &inner,
            commit_params.num_blocks,
            commit_params.a_key.row_len(),
            commit_params.num_digits_open,
            commit_params.log_basis,
        )?;

        let typed_digits = inner.decomposed_inner_rows_trusted::<D>()?;
        let outer_input = typed_digits.flat_digits().to_vec();
        validate_commit_outer_input_nonempty(outer_input.len())?;
        let u: Vec<CyclotomicRing<Cfg::Field, D>> = if commit_params.f_key.is_some() {
            crate::api::commitment::tiered_commit_u_final::<Cfg::Field, D, B>(
                backend,
                prepared,
                commit_params,
                &outer_input,
            )?
        } else {
            let u: Vec<CyclotomicRing<Cfg::Field, D>> = backend.digit_rows::<D>(
                prepared,
                commit_params.b_key.row_len(),
                &outer_input,
                commit_params.log_basis,
            )?;
            if u.len() != commit_params.b_key.row_len() {
                return Err(AkitaError::InvalidProof);
            }
            u
        };
        let hint = AkitaCommitmentHint::singleton(inner.decomposed_inner_rows.clone());
        Ok(NextWitnessCommitment {
            witness: packed_witness,
            commitment: RingVec::from_ring_elems(&u),
            hint: RecursiveCommitmentHintCache::from_hint(hint),
        })
    })
}
