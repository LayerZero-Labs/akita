//! Prover-owned helpers for the Akita ring-switch handoff.

use crate::dispatch_with_ntt;
use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::protocol::quadratic_equation::{compute_r_split_eq, QuadraticEquation};
use crate::{
    tensor_pack_recursive_witness, MultiDNttCaches, RecursiveCommitmentHintCache,
    RecursiveWitnessFlat,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::ring::eval_ring_at_pows;
use akita_algebra::ring::scalar_powers;
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_config::CommitmentConfig as _;
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField, LiftBase,
    MulBase, RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    embed_ring_subfield_scalar, gadget_row_scalars, r_decomp_levels,
    validate_opening_points_for_claims, AkitaCommitmentHint, AkitaExpandedSetup, FlatDigitBlocks,
    FlatRingVec, LevelParams, MRowLayout, RingCommitment, RingMultiplierOpeningPoint,
    RingOpeningPoint, RingSubfieldEncoding,
};

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub struct RingSwitchOutput<E: FieldCore> {
    /// Compact evaluation table of w, stored as x-outer/y-inner slices.
    pub w_evals_compact: Vec<i8>,
    /// Physical x width before zero-extension to the next power of two.
    pub live_x_cols: usize,
    /// Evaluation table of M_alpha(x) (tau1-weighted).
    pub m_evals_x: Vec<E>,
    /// Evaluation table of alpha powers (y dimension).
    pub alpha_evals_y: Vec<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<E>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}

/// Result of committing the next logical recursive witness.
pub struct NextWitnessCommitment<F: FieldCore> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Commitment to the physical next-level witness.
    pub commitment: FlatRingVec<F>,
    /// Prover hint for `commitment`.
    pub hint: RecursiveCommitmentHintCache<F>,
}

/// Build the witness vector `w` from the quadratic equation state.
///
/// This is the first half of the ring switch: it computes `r` and assembles
/// `w` as a flat recursive witness. The resulting `w` is D-agnostic and can be
/// committed at any supported ring dimension by the recursive commitment path.
///
/// # Errors
///
/// Returns an error if the quadratic equation is missing prover-side data.
///
/// # Panics
///
/// Panics with `feature = "zk"` enabled if the zero-length `FlatDigitBlocks`
/// constructor rejects an empty vector (an invariant of the type).
#[tracing::instrument(skip_all, name = "ring_switch_build_w")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_build_w<F, const D: usize>(
    quad_eq: &mut QuadraticEquation<F, D>,
    setup: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    lp: &LevelParams,
) -> Result<RecursiveWitnessFlat, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HalvingField,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            "ring_switch_build_w"
        );
    }
    let w_hat = quad_eq
        .take_w_hat()
        .ok_or_else(|| AkitaError::InvalidInput("missing w_hat in prover".to_string()))?;
    #[cfg(feature = "zk")]
    let d_blinding_digits = quad_eq.take_d_blinding_digits().ok_or_else(|| {
        AkitaError::InvalidInput("missing D-blinding digits in prover".to_string())
    })?;
    let z_pre = quad_eq
        .take_z_pre()
        .ok_or_else(|| AkitaError::InvalidInput("missing centered z_pre in prover".to_string()))?;
    let mut hint = quad_eq
        .take_hint()
        .ok_or_else(|| AkitaError::InvalidInput("missing hint in prover".to_string()))?;
    hint.ensure_recomposed_inner_rows(lp.num_digits_open, lp.log_basis)?;
    #[cfg(feature = "zk")]
    let (decomposed_inner_rows, recomposed_inner_rows, b_blinding_digits) = hint.into_flat_parts();
    #[cfg(not(feature = "zk"))]
    let (decomposed_inner_rows, recomposed_inner_rows) = hint.into_flat_parts();
    let recomposed_inner_rows = recomposed_inner_rows.ok_or_else(|| {
        AkitaError::InvalidInput("missing recomposed inner rows in prover hint".to_string())
    })?;
    let w_folded = quad_eq
        .take_w_folded()
        .ok_or_else(|| AkitaError::InvalidInput("missing w_folded in prover".to_string()))?;

    let r = compute_r_split_eq::<F, D>(
        lp,
        setup,
        &quad_eq.challenges,
        w_hat.flat_digits(),
        #[cfg(feature = "zk")]
        &d_blinding_digits,
        &decomposed_inner_rows,
        #[cfg(feature = "zk")]
        &b_blinding_digits,
        &recomposed_inner_rows,
        &w_folded,
        quad_eq.ring_multiplier_points(),
        quad_eq.claim_to_point(),
        quad_eq.claim_to_point_poly(),
        quad_eq.claim_poly_indices(),
        quad_eq.row_coefficient_rings(),
        &z_pre.centered_coeffs,
        z_pre.centered_inf_norm,
        quad_eq.y(),
        quad_eq.num_polys_per_point(),
        quad_eq.num_public_rows(),
        lp.num_blocks,
        lp.inner_width(),
        setup.seed.max_stride,
        ntt_shared,
        quad_eq.m_row_layout(),
    )?;
    // Terminal layout drops the D-block from M and from the witness; the
    // d-blinding column segment must also disappear so the prover witness
    // matches the verifier's column offsets.
    #[cfg(feature = "zk")]
    let d_blinding_for_w: FlatDigitBlocks<D> = match quad_eq.m_row_layout() {
        MRowLayout::Intermediate => d_blinding_digits,
        MRowLayout::Terminal => {
            FlatDigitBlocks::zeroed(Vec::new()).expect("empty FlatDigitBlocks always valid")
        }
    };
    let w = {
        let _span = tracing::info_span!("build_w_coeffs").entered();
        build_w_coeffs::<F, D>(
            &w_hat,
            #[cfg(feature = "zk")]
            &d_blinding_for_w,
            &decomposed_inner_rows,
            #[cfg(feature = "zk")]
            &b_blinding_digits,
            &z_pre.centered_coeffs,
            &r,
            lp,
        )
    };
    Ok(w)
}

/// Complete the ring switch after `w` has been committed.
///
/// Takes the already-committed `w` and finishes the protocol: absorbs the
/// commitment into the transcript, samples challenges, and builds the
/// evaluation tables for the fused sumcheck.
///
/// Only the current level's `D` is needed for M-alpha expansion and
/// `alpha_evals_y`. The commitment's ring dimension is encoded in the
/// [`FlatRingVec`] and does not require a separate const generic.
///
/// # Errors
///
/// Returns an error if matrix expansion or evaluation-table construction fails.
#[tracing::instrument(skip_all, name = "ring_switch_finalize")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize<F, E, T, const D: usize>(
    quad_eq: &QuadraticEquation<F, D>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    w_commitment_proof: &FlatRingVec<F>,
    lp: &LevelParams,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    ring_switch_finalize_with_claim_groups::<F, E, T, D>(
        quad_eq,
        setup,
        transcript,
        w,
        w_commitment_proof,
        lp,
        m_row_layout,
    )
}

/// Complete the ring switch for a batched root with explicit claim groups.
///
/// # Errors
///
/// Returns an error if matrix expansion or evaluation-table construction fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize_with_claim_groups<F, E, T, const D: usize>(
    quad_eq: &QuadraticEquation<F, D>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    w_commitment_proof: &FlatRingVec<F>,
    lp: &LevelParams,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let gamma = quad_eq
        .gamma()
        .iter()
        .copied()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    ring_switch_finalize_with_gamma::<F, E, T, D>(
        quad_eq,
        setup,
        transcript,
        w,
        w_commitment_proof,
        lp,
        &gamma,
        m_row_layout,
    )
}

/// Variant of [`ring_switch_finalize`] that assumes the caller has already
/// absorbed the `ABSORB_SUMCHECK_W` bytes into `transcript`.
///
/// Used by terminal fold levels, which absorb the cleartext `final_witness`
/// in place of the recursive `next_w_commitment`.
///
/// # Errors
///
/// Returns an error if matrix expansion or evaluation-table construction fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize_after_absorb<F, E, T, const D: usize>(
    quad_eq: &QuadraticEquation<F, D>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    lp: &LevelParams,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let gamma = quad_eq
        .gamma()
        .iter()
        .copied()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    ring_switch_finalize_with_gamma_after_absorb::<F, E, T, D>(
        quad_eq,
        setup,
        transcript,
        w,
        lp,
        &gamma,
        m_row_layout,
    )
}

/// Complete ring switching with caller-supplied proof-scalar batching
/// coefficients.
///
/// The folded-root path uses this to keep same-point batching challenges in
/// the proof scalar field instead of first projecting them through the base
/// field. Recursive degree-one paths continue to call
/// [`ring_switch_finalize_with_claim_groups`].
///
/// # Errors
///
/// Returns an error if the supplied gamma vector does not match the claim
/// count or if matrix expansion or evaluation-table construction fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize_with_gamma<F, E, T, const D: usize>(
    quad_eq: &QuadraticEquation<F, D>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    w_commitment_proof: &FlatRingVec<F>,
    lp: &LevelParams,
    gamma: &[E],
    m_row_layout: MRowLayout,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment_proof);
    ring_switch_finalize_with_gamma_after_absorb::<F, E, T, D>(
        quad_eq,
        setup,
        transcript,
        w,
        lp,
        gamma,
        m_row_layout,
    )
}

/// Variant of [`ring_switch_finalize_with_gamma`] that assumes the caller has
/// already absorbed the `ABSORB_SUMCHECK_W` bytes into `transcript`.
///
/// Intermediate fold levels absorb `next_w_commitment` before calling this;
/// terminal fold levels absorb the cleartext `final_witness` instead. Keeping
/// the absorb in the caller lets the protocol bind whichever payload is
/// shipped at this step without needing a duality flag here.
///
/// # Errors
///
/// Returns an error if the supplied gamma vector does not match the claim
/// count or if matrix expansion or evaluation-table construction fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize_with_gamma_after_absorb<F, E, T, const D: usize>(
    quad_eq: &QuadraticEquation<F, D>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    lp: &LevelParams,
    gamma: &[E],
    m_row_layout: MRowLayout,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_polys_per_point = quad_eq.num_polys_per_point();
    let num_points = num_polys_per_point.len();
    let num_public_rows = quad_eq.num_public_rows();

    let num_ring_elems = w.len() / D;
    let live_x_cols = num_ring_elems;
    let col_bits = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch column count overflow".to_string()))?
        .trailing_zeros() as usize;
    let ring_bits = D.trailing_zeros() as usize;
    let m_rows = lp.m_row_count_for(num_points, num_public_rows, m_row_layout)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let tau0: Vec<E> = (0..num_sc_vars)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
        .collect();
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let ring_alpha_evals_y = scalar_powers(alpha, D);
    let alpha_evals_y = scalar_powers(alpha, D);

    let opening_points = quad_eq.opening_points();
    let ring_multiplier_points = quad_eq.ring_multiplier_points();
    let claim_to_point = quad_eq.claim_to_point();
    let claim_to_point_poly = quad_eq.claim_to_point_poly();
    let claim_poly_indices = quad_eq.claim_poly_indices();
    let challenges = &quad_eq.challenges;
    if gamma.len() != claim_to_point.len() {
        return Err(AkitaError::InvalidInput(
            "ring-switch gamma length does not match claim count".to_string(),
        ));
    }

    #[cfg(feature = "parallel")]
    let (m_evals_x_result, w_result) = rayon::join(
        || {
            compute_m_evals_x::<F, E, D>(
                setup,
                opening_points,
                ring_multiplier_points,
                claim_to_point,
                challenges,
                alpha,
                &ring_alpha_evals_y,
                lp,
                &tau1,
                num_polys_per_point,
                claim_to_point_poly,
                claim_poly_indices,
                gamma,
                num_public_rows,
                m_row_layout,
            )
        },
        || build_w_evals_compact(w.as_i8_digits(), D, 1),
    );
    #[cfg(not(feature = "parallel"))]
    let (m_evals_x_result, w_result) = {
        let m_evals_x = compute_m_evals_x::<F, E, D>(
            setup,
            opening_points,
            ring_multiplier_points,
            claim_to_point,
            challenges,
            alpha,
            &ring_alpha_evals_y,
            lp,
            &tau1,
            num_polys_per_point,
            claim_to_point_poly,
            claim_poly_indices,
            gamma,
            num_public_rows,
            m_row_layout,
        )?;
        let w_compact = build_w_evals_compact(w.as_i8_digits(), D, 1);
        (Ok(m_evals_x), w_compact)
    };

    let m_evals_x = m_evals_x_result?;
    let (w_evals_compact, _, _) = w_result?;

    Ok(RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize << lp.log_basis,
        alpha,
    })
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
pub fn commit_w<F, const D: usize>(
    w: &RecursiveWitnessFlat,
    ntt_shared: &NttSlotCache<D>,
    commit_layout: &LevelParams,
    stride: usize,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
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
        ntt_shared,
        commit_layout.a_key.row_len(),
        commit_layout.block_len,
        commit_layout.num_blocks,
        commit_layout.num_digits_commit,
        commit_layout.num_digits_open,
        commit_layout.log_basis,
        stride,
    )?;

    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(commit_layout.b_key.row_len(), commit_layout.log_basis)?;
    #[cfg(feature = "zk")]
    let mut outer_input = inner.decomposed_inner_rows.flat_digits().to_vec();
    #[cfg(not(feature = "zk"))]
    let outer_input = inner.decomposed_inner_rows.flat_digits().to_vec();
    #[cfg(feature = "zk")]
    outer_input.extend_from_slice(b_blinding_digits.flat_digits());
    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        ntt_shared,
        commit_layout.b_key.row_len(),
        stride,
        &outer_input,
    );
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
/// The prover crate owns runtime-D NTT cache construction and `commit_w`
/// execution. Callers supply the config-specific layout policy for the selected
/// commitment dimension.
///
/// # Errors
///
/// Returns an error if layout selection, NTT cache construction, commitment, or
/// D-erased hint conversion fails.
#[allow(clippy::type_complexity)]
#[inline(never)]
fn dispatch_commit_w_with_layout_policy<F, L, Layout>(
    commit_params: LevelParams,
    commit_ntt_cache: &mut MultiDNttCaches,
    expanded: &AkitaExpandedSetup<F>,
    logical_w: &RecursiveWitnessFlat,
    layout_for_d: Layout,
) -> Result<NextWitnessCommitment<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    L: ExtField<F>,
    Layout: Fn(usize, &LevelParams, usize) -> Result<LevelParams, AkitaError>,
{
    let commit_d = commit_params.ring_dimension;
    let stride = expanded.seed.max_stride;
    dispatch_with_ntt!(
        commit_d,
        commit_ntt_cache,
        expanded,
        |D_COMMIT, ntt_shared| {
            if L::EXT_DEGREE == 1 {
                let commit_layout = layout_for_d(D_COMMIT, &commit_params, logical_w.len())?;
                let (wc, wh) =
                    commit_w::<F, { D_COMMIT }>(logical_w, ntt_shared, &commit_layout, stride)?;
                Ok(NextWitnessCommitment {
                    witness: None,
                    commitment: FlatRingVec::from_commitment(&wc),
                    hint: RecursiveCommitmentHintCache::from_typed(wh)?,
                })
            } else {
                let committed_w = tensor_pack_recursive_witness::<F, L, { D_COMMIT }>(logical_w)?;
                let commit_layout = layout_for_d(D_COMMIT, &commit_params, committed_w.len())?;
                let (wc, wh) =
                    commit_w::<F, { D_COMMIT }>(&committed_w, ntt_shared, &commit_layout, stride)?;
                Ok(NextWitnessCommitment {
                    witness: Some(committed_w),
                    commitment: FlatRingVec::from_commitment(&wc),
                    hint: RecursiveCommitmentHintCache::from_typed(wh)?,
                })
            }
        }
    )
}

/// Commit the next recursive witness.
///
/// The same-`D` fast path reuses the current level's NTT slot and applies
/// `same_d_decomposition` as the `recursive_level_layout_from_params`
/// decomposition. Cross-`D` commitments dispatch through
/// [`MultiDNttCaches`] and always use the `WCommitmentConfig::<D_COMMIT,
/// Cfg>::decomposition()` recursive-w decomposition for the dispatched ring
/// dimension.
///
/// Callers pin `same_d_decomposition` per call site:
/// - root → first recursive commit: `Cfg::decomposition()` (the root
///   decomposition).
/// - recursive → recursive commit: `WCommitmentConfig::<D, Cfg>::decomposition()`.
///
/// # Errors
///
/// Returns an error if layout selection, commitment, cache construction, or
/// `D`-erased hint conversion fails.
#[inline(never)]
pub fn commit_next_w<F, Cfg, const D: usize>(
    commit_params: &LevelParams,
    ntt_shared: &NttSlotCache<D>,
    commit_ntt_cache: &mut MultiDNttCaches,
    expanded: &AkitaExpandedSetup<F>,
    logical_w: &RecursiveWitnessFlat,
    same_d_decomposition: akita_types::DecompositionParams,
) -> Result<NextWitnessCommitment<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    Cfg: akita_config::CommitmentConfig<Field = F>,
{
    if commit_params.ring_dimension == D {
        if <Cfg::ChallengeField as ExtField<F>>::EXT_DEGREE == 1 {
            let commit_layout = akita_types::recursive_level_layout_from_params(
                commit_params,
                logical_w.len(),
                same_d_decomposition,
            )?;
            let (wc, wh) = commit_w::<F, D>(
                logical_w,
                ntt_shared,
                &commit_layout,
                expanded.seed.max_stride,
            )?;
            Ok(NextWitnessCommitment {
                witness: None,
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        } else {
            let committed_w =
                tensor_pack_recursive_witness::<F, Cfg::ChallengeField, D>(logical_w)?;
            let commit_layout = akita_types::recursive_level_layout_from_params(
                commit_params,
                committed_w.len(),
                same_d_decomposition,
            )?;
            let (wc, wh) = commit_w::<F, D>(
                &committed_w,
                ntt_shared,
                &commit_layout,
                expanded.seed.max_stride,
            )?;
            Ok(NextWitnessCommitment {
                witness: Some(committed_w),
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        }
    } else {
        dispatch_commit_w_with_layout_policy::<F, Cfg::ChallengeField, _>(
            commit_params.clone(),
            commit_ntt_cache,
            expanded,
            logical_w,
            |commit_d, commit_params, current_w_len| {
                recursive_w_commit_layout_for_d::<Cfg>(commit_d, commit_params, current_w_len)
            },
        )
    }
}

/// Recursive `w` commitment layout for a runtime-selected ring dimension.
///
/// Public so scheme-side integration tests that drive `commit_next_w` at a
/// non-`D` recursive level can pick the matching layout without
/// re-implementing the multi-`D` dispatch.
pub fn recursive_w_commit_layout_for_d<Cfg>(
    commit_d: usize,
    commit_params: &LevelParams,
    current_w_len: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: akita_config::CommitmentConfig,
{
    crate::dispatch_ring_dim!(commit_d, |D_COMMIT| {
        akita_types::recursive_level_layout_from_params(
            commit_params,
            current_w_len,
            akita_config::WCommitmentConfig::<{ D_COMMIT }, Cfg>::decomposition(),
        )
    })
}

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw [`build_w_coeffs`] order:
/// `w[x * y_len + y]`, with x outer and y inner.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by the ring
/// dimension.
pub fn build_w_evals_compact(
    w: &[i8],
    d: usize,
    extension_degree: usize,
) -> Result<(Vec<i8>, usize, usize), AkitaError> {
    if !w.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let live_x_cols = w.len() / d;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    if extension_degree == 1 {
        let ring_bits = d.trailing_zeros() as usize;
        return Ok((w.to_vec(), col_bits, ring_bits));
    }
    let packed_len = d / extension_degree;
    if packed_len == 0 || !packed_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "packed recursive witness has invalid slot count".to_string(),
        ));
    }
    let half = d / (2 * extension_degree);
    let mut compact = Vec::with_capacity(live_x_cols * packed_len);
    for ring in w.chunks_exact(d) {
        compact.extend_from_slice(&ring[..half]);
        compact.extend((half..packed_len).map(|low| ring[d / 2 + low - half]));
    }
    Ok((compact, col_bits, packed_len.trailing_zeros() as usize))
}

/// Unified M-table evaluation for the batched CWSS protocol.
///
/// `opening_points` holds the distinct ring-level opening points used by the
/// batch, `claim_to_point` maps each flattened claim index to its opening-point
/// index, and `gamma` provides the per-claim random linear-combination
/// coefficients. The matrix carries one public y-row per distinct opening
/// point (`num_public_rows = opening_points.len()`).
///
/// # Errors
///
/// Returns an error if the batch shape, opening-point layout, challenge count,
/// or expanded matrix dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_m_evals_x_batched")]
pub fn compute_m_evals_x<F, E, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &Challenges,
    alpha: E,
    alpha_pows: &[E],
    lp: &LevelParams,
    tau1: &[E],
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    num_public_rows: usize,
    m_row_layout: MRowLayout,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
{
    if alpha_pows.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    let num_claims = claim_to_point.len();
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    if ring_multiplier_points.len() != opening_points.len()
        || ring_multiplier_points
            .iter()
            .any(|point| point.a_len() < lp.block_len || point.b_len() != lp.num_blocks)
    {
        return Err(AkitaError::InvalidInput(
            "batched prover ring-multiplier opening-point layout mismatch".to_string(),
        ));
    }
    if claim_to_point_poly.len() != num_claims || claim_poly_indices.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "batched prover claim incidence lengths do not match".to_string(),
        ));
    }
    let num_points = num_polys_per_point.len();
    for claim_idx in 0..num_claims {
        let point_idx = claim_to_point_poly[claim_idx];
        if point_idx >= num_points
            || claim_poly_indices[claim_idx] >= num_polys_per_point[point_idx]
        {
            return Err(AkitaError::InvalidInput(
                "batched prover claim incidence index out of range".to_string(),
            ));
        }
    }

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold;
    let log_basis = lp.log_basis;
    let num_blocks = lp.num_blocks;
    let num_t_vectors = num_polys_per_point
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))
        .ok_or_else(|| AkitaError::InvalidSetup("batched t-vector count overflow".to_string()))?;
    let t_vector_to_group: Vec<(usize, usize)> = num_polys_per_point
        .iter()
        .enumerate()
        .flat_map(|(point_idx, &group_poly_count)| {
            (0..group_poly_count).map(move |poly_idx| (point_idx, poly_idx))
        })
        .collect();
    // Per-point t-vector starting indices; precomputed so the per-claim
    // mapping below stays O(num_points + num_claims) instead of recomputing
    // the prefix sum on every claim.
    let t_vector_offsets: Vec<usize> = num_polys_per_point
        .iter()
        .scan(0usize, |acc, &count| {
            let offset = *acc;
            *acc += count;
            Some(offset)
        })
        .collect();
    let claim_to_t_vector: Vec<usize> = claim_to_point_poly
        .iter()
        .zip(claim_poly_indices.iter())
        .map(|(&point_idx, &poly_idx)| t_vector_offsets[point_idx] + poly_idx)
        .collect();

    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    let t_total_blocks = num_blocks
        .checked_mul(num_t_vectors)
        .ok_or_else(|| AkitaError::InvalidSetup("batched t block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let block_len = lp.block_len;
    let w_len = depth_open * total_blocks;
    let n_a = lp.a_key.row_len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    // Terminal layout drops the D-block from the M-matrix entirely; offsets
    // and per-row gates must use 0 for the n_d position.
    let n_d_active = match m_row_layout {
        MRowLayout::Intermediate => n_d,
        MRowLayout::Terminal => 0,
    };
    let t_len = depth_open * n_a * t_total_blocks;
    #[cfg(feature = "zk")]
    let d_blinding_segment_len = match m_row_layout {
        MRowLayout::Intermediate => {
            akita_types::zk::blinding_digit_plane_count::<F>(n_d, D, log_basis)
        }
        // Terminal omits the D-block, so its blinding columns vanish too.
        MRowLayout::Terminal => 0,
    };
    #[cfg(feature = "zk")]
    let b_blinding_digit_planes_per_point =
        akita_types::zk::blinding_digit_plane_count::<F>(n_b, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_segment_len = num_points
        .checked_mul(b_blinding_digit_planes_per_point)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK blinding width overflow".to_string()))?;
    let inner_width = block_len * depth_commit;
    let z_base_len = opening_points
        .len()
        .checked_mul(inner_width)
        .ok_or_else(|| AkitaError::InvalidSetup("batched z width overflow".to_string()))?;
    let z_len = depth_fold
        .checked_mul(z_base_len)
        .ok_or_else(|| AkitaError::InvalidSetup("batched z width overflow".to_string()))?;
    let rows = lp.m_row_count_for(num_points, num_public_rows, m_row_layout)?;
    let levels = r_decomp_levels::<F>(log_basis);
    #[cfg(feature = "zk")]
    let total_cols = w_len
        .checked_add(d_blinding_segment_len)
        .and_then(|cols| cols.checked_add(t_len))
        .and_then(|cols| cols.checked_add(b_blinding_segment_len))
        .and_then(|cols| cols.checked_add(z_len))
        .and_then(|cols| cols.checked_add(rows.checked_mul(levels)?))
        .ok_or_else(|| AkitaError::InvalidSetup("expanded M width overflow".to_string()))?;
    #[cfg(not(feature = "zk"))]
    let total_cols = w_len
        .checked_add(t_len)
        .and_then(|cols| cols.checked_add(z_len))
        .and_then(|cols| cols.checked_add(rows.checked_mul(levels)?))
        .ok_or_else(|| AkitaError::InvalidSetup("expanded M width overflow".to_string()))?;

    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let g1_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let g1_commit: Vec<E> = gadget_row_scalars::<F>(depth_commit, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let x_len = total_cols.next_power_of_two();
    let mut out = Vec::with_capacity(x_len);

    let c_alphas: Vec<E> = match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => sparse
            .iter()
            .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
            .collect::<Result<_, _>>()?,
        Challenges::Tensor { factored: _ } => challenges.evals_at_pows::<F, E, D>(alpha_pows)?,
    };

    let stride = setup.seed.max_stride;
    let d_view = setup.shared_matrix.ring_view::<D>(n_d, stride)?;
    let b_view = setup.shared_matrix.ring_view::<D>(n_b, stride)?;
    let a_view = setup.shared_matrix.ring_view::<D>(n_a, stride)?;
    let d_rows: Vec<_> = d_view.rows().collect();
    let b_rows: Vec<_> = b_view.rows().collect();
    let a_rows: Vec<_> = a_view.rows().collect();

    // Row layout: consistency (1) | public (num_public_rows) | D (n_d_active)
    //             | B (n_b * num_points) | A (n_a). At terminal layout
    // `n_d_active == 0`, collapsing the D-block out of the offset chain.
    let commitment_row_count = n_b * num_points;
    let consistency_weight = eq_tau1[0];
    let public_weights = &eq_tau1[1..(1 + num_public_rows)];
    let d_start = 1 + num_public_rows;
    let b_start = d_start + n_d_active;
    let a_start = b_start + commitment_row_count;
    let a_weights = &eq_tau1[a_start..rows];
    let t_compound_per_block = n_a * depth_open;

    let uses_ring_multipliers = ring_multiplier_points
        .iter()
        .any(|point| point.as_base().is_none());
    let row_coefficient_rings = if uses_ring_multipliers {
        Some(
            gamma
                .iter()
                .copied()
                .map(|coefficient| {
                    embed_ring_subfield_scalar::<F, E, D>(
                        coefficient,
                        AkitaError::InvalidInput(
                            "public-row coefficient does not encode in the ring-subfield basis"
                                .to_string(),
                        ),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let public_b_evals = (0..num_claims)
        .map(|claim_idx| {
            let point_idx = claim_to_point[claim_idx];
            let opening_point = &ring_multiplier_points[point_idx];
            let coefficient_ring = row_coefficient_rings
                .as_ref()
                .map(|rings| &rings[claim_idx]);
            (0..num_blocks)
                .map(|block_idx| {
                    opening_point.eval_b_with_coefficient(
                        block_idx,
                        gamma[claim_idx],
                        coefficient_ring,
                        alpha_pows,
                    )
                })
                .collect::<Result<Vec<_>, AkitaError>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let w_segment: Vec<E> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let d_phys_col = blk * depth_open + dig;
            let point_idx = claim_to_point[claim_idx];
            let b_eval = public_b_evals[claim_idx][block_idx];
            let mut acc = (public_weights[point_idx] * b_eval + consistency_weight * c_alphas[blk])
                * g1_open[dig];
            // Terminal layout: `n_d_active == 0`, so this loop is empty and
            // the D-block contribution is omitted.
            for (di, eq_i) in eq_tau1[d_start..(d_start + n_d_active)].iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&d_rows[di][d_phys_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    #[cfg(feature = "zk")]
    let d_blinding_segment: Vec<E> = if d_blinding_segment_len == 0 {
        Vec::new()
    } else {
        let d_weights = &eq_tau1[d_start..(d_start + n_d_active)];
        cfg_into_iter!(0..d_blinding_segment_len)
            .map(|local| {
                let local_col = w_len + local;
                let mut acc = E::zero();
                for (row_idx, eq_i) in d_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i * eval_ring_at_pows(&d_rows[row_idx][local_col], alpha_pows);
                    }
                }
                acc
            })
            .collect()
    };

    let t_cols_per_vector = t_compound_per_block * num_blocks;
    let mut challenge_sums_by_t_block = vec![E::zero(); t_total_blocks];
    for (claim_idx, &t_vector_idx) in claim_to_t_vector.iter().enumerate() {
        let dst_offset = t_vector_idx * num_blocks;
        let src_offset = claim_idx * num_blocks;
        for block_idx in 0..num_blocks {
            challenge_sums_by_t_block[dst_offset + block_idx] += c_alphas[src_offset + block_idx];
        }
    }
    let t_segment: Vec<E> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let compound_dig = x / t_total_blocks;
            let blk = x % t_total_blocks;
            let a_idx = compound_dig / depth_open;
            let digit_idx = compound_dig % depth_open;
            let t_vector_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let (point_idx, poly_idx_within_group) = t_vector_to_group[t_vector_idx];
            let phys_claim_offset =
                block_idx * t_compound_per_block + a_idx * depth_open + digit_idx;
            let local_col = poly_idx_within_group * t_cols_per_vector + phys_claim_offset;
            let commitment_weights =
                &eq_tau1[(b_start + point_idx * n_b)..(b_start + (point_idx + 1) * n_b)];
            let mut acc = a_weights[a_idx] * challenge_sums_by_t_block[blk] * g1_open[digit_idx];
            for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&b_rows[row_idx][local_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    #[cfg(feature = "zk")]
    let b_blinding_segment: Vec<E> = if b_blinding_digit_planes_per_point == 0 {
        Vec::new()
    } else {
        // Each commitment group is committed independently with a group-local B
        // input `[group t_hat || group blinding]`, even though the ring-switch
        // witness stores all groups in one concatenated segment.
        cfg_into_iter!(0..b_blinding_segment_len)
            .map(|idx| {
                let group_stride = b_blinding_digit_planes_per_point;
                let point_idx = idx / group_stride;
                let local = idx % group_stride;
                let group_message_planes = num_polys_per_point[point_idx] * t_cols_per_vector;
                let local_col = group_message_planes + local;
                let commitment_weights =
                    &eq_tau1[(b_start + point_idx * n_b)..(b_start + (point_idx + 1) * n_b)];
                let mut acc = E::zero();
                for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i * eval_ring_at_pows(&b_rows[row_idx][local_col], alpha_pows);
                    }
                }
                acc
            })
            .collect()
    };

    let z_base: Vec<E> = cfg_into_iter!(0..z_base_len)
        .map(|k| {
            let point_idx = k / inner_width;
            let local_k = k % inner_width;
            let block_idx = local_k / depth_commit;
            let digit_idx = local_k % depth_commit;
            let opening_point = &ring_multiplier_points[point_idx];
            let a_eval = opening_point.eval_a_at::<E>(block_idx, alpha_pows)?;
            let mut acc = consistency_weight * a_eval * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&a_rows[a_idx][local_k], alpha_pows);
                }
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    let num_points = opening_points.len();
    let z_total_blocks = num_points * block_len;
    let z_segment: Vec<E> = cfg_into_iter!(0..z_len)
        .map(|x| {
            let compound_dig = x / z_total_blocks;
            let global_blk = x % z_total_blocks;
            let dc = compound_dig / depth_fold;
            let df = compound_dig % depth_fold;
            let point_idx = global_blk / block_len;
            let blk = global_blk % block_len;
            let phys_k = point_idx * inner_width + blk * depth_commit + dc;
            -(z_base[phys_k] * fold_gadget[df])
        })
        .collect();

    let alpha_pow_d = alpha_pows[D - 1] * alpha;
    let denom = alpha_pow_d + E::one();
    let r_tail_len = rows * levels;
    let r_tail: Vec<E> = cfg_into_iter!(0..r_tail_len)
        .map(|idx| {
            let row_idx = idx / levels;
            let level_idx = idx % levels;
            -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
        })
        .collect();

    let z_first = lp.m_vars >= lp.r_vars;
    if z_first {
        out.extend(z_segment);
        out.extend(w_segment);
        out.extend(t_segment);
        #[cfg(feature = "zk")]
        out.extend(b_blinding_segment);
        #[cfg(feature = "zk")]
        out.extend(d_blinding_segment);
    } else {
        out.extend(w_segment);
        out.extend(t_segment);
        #[cfg(feature = "zk")]
        out.extend(b_blinding_segment);
        #[cfg(feature = "zk")]
        out.extend(d_blinding_segment);
        out.extend(z_segment);
    }
    out.extend(r_tail);
    out.resize(x_len, E::zero());
    Ok(out)
}

fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 6,
        "log_basis must be in 1..=6 for i8 output"
    );
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
        "levels * log_basis must be <= 128 + log_basis"
    );

    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;

    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx] as i128;
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

/// Transpose block-major digit planes to digit-major order (block index
/// innermost): for each compound digit index, emit all blocks in order.
fn emit_planes_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    flat: &[[i8; D]],
    total_blocks: usize,
    planes_per_block: usize,
) {
    debug_assert_eq!(
        flat.len(),
        total_blocks * planes_per_block,
        "emit_planes_block_inner: flat.len()={} != total_blocks({}) * planes_per_block({})",
        flat.len(),
        total_blocks,
        planes_per_block
    );
    for compound_dig in 0..planes_per_block {
        for blk in 0..total_blocks {
            out.extend_from_slice(&flat[blk * planes_per_block + compound_dig]);
        }
    }
}

#[cfg(feature = "zk")]
fn emit_blinding_planes<const D: usize>(
    out: &mut Vec<i8>,
    blinding_by_group: &[FlatDigitBlocks<D>],
) {
    for blinding in blinding_by_group {
        for plane in blinding.flat_digits() {
            out.extend_from_slice(plane);
        }
    }
}

/// Decompose z_pre elements and emit in digit-major order.
///
/// z_pre has `num_points * block_len * depth_commit` elements indexed as
/// `z[point * inner_width + blk * depth_commit + dc]`. Each decomposes into
/// `num_digits_fold` planes.
///
/// Output order: for each `(dc, df)`, emit all `(point, blk)` pairs with
/// the global block index `point * block_len + blk` innermost.
fn emit_z_pre_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    z_pre_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    log_basis: u32,
) {
    let total_elems = z_pre_centered.len();
    let inner_width = block_len * depth_commit;
    debug_assert_eq!(
        total_elems % inner_width,
        0,
        "z_pre length {total_elems} not divisible by inner_width {inner_width}",
    );
    let num_points = total_elems / inner_width;

    let mut all_planes = vec![[0i8; D]; total_elems * num_digits_fold];
    for (k, z_j) in z_pre_centered.iter().enumerate() {
        balanced_decompose_centered_i32_i8_into(
            z_j,
            &mut all_planes[k * num_digits_fold..(k + 1) * num_digits_fold],
            log_basis,
        );
    }

    for dc in 0..depth_commit {
        for df in 0..num_digits_fold {
            for pt in 0..num_points {
                for blk in 0..block_len {
                    let k = pt * inner_width + blk * depth_commit + dc;
                    out.extend_from_slice(&all_planes[k * num_digits_fold + df]);
                }
            }
        }
    }
}

/// Build the committed witness polynomial from ring-domain digit planes.
///
/// Emits field-domain coefficients in digit-major order (block index innermost)
/// with adaptive segment ordering: the segment whose block dimension is the
/// larger power of two comes first.
///
/// Segment ordering:
/// - If `m_vars >= r_vars`: z-hat (`2^m` blocks), e-hat + t-hat (`2^r` blocks), r-hat
/// - If `m_vars < r_vars`: e-hat + t-hat (`2^r` blocks), z-hat (`2^m` blocks), r-hat
///
/// Within each segment, the power-of-2 block index is the fastest-varying
/// (innermost) dimension.
///
/// `FlatDigitBlocks` stores ring-domain data in block-major order (all digit
/// planes for one block contiguously), which is natural for ring-domain matvec
/// and recomposition. This function transposes opening digits to digit-major at
/// the ring-to-field boundary; ZK blinding streams are already direct
/// digit-plane sources and are emitted in matrix-column order.
///
/// # Panics
///
/// Panics if the caller supplies digit blocks whose plane counts do not match
/// the fold layout in `lp`, or if ZK blinding digit counts do not match the
/// configured blinding columns.
pub fn build_w_coeffs<F: CanonicalField, const D: usize>(
    w_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] b_blinding_digits: &[FlatDigitBlocks<D>],
    z_pre_centered: &[[i32; D]],
    r: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
) -> RecursiveWitnessFlat {
    let log_basis = lp.log_basis;
    let num_digits_fold = lp.num_digits_fold;
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let block_len = lp.block_len;
    let levels = r_decomp_levels::<F>(log_basis);

    let w_hat_planes = w_hat.flat_digits().len();
    let t_hat_planes = t_hat.flat_digits().len();
    #[cfg(feature = "zk")]
    let d_blinding_planes = d_blinding_digits.flat_digits().len();
    #[cfg(not(feature = "zk"))]
    let d_blinding_planes = 0usize;
    #[cfg(feature = "zk")]
    let b_blinding_planes: usize = b_blinding_digits
        .iter()
        .map(|digits| digits.flat_digits().len())
        .sum();
    #[cfg(not(feature = "zk"))]
    let b_blinding_planes = 0usize;
    let z_count = w_hat_planes
        + d_blinding_planes
        + t_hat_planes
        + b_blinding_planes
        + z_pre_centered.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    let z_first = lp.m_vars >= lp.r_vars;
    tracing::debug!(
        w_hat_planes,
        d_blinding_planes,
        t_hat_planes,
        b_blinding_planes,
        z_pre_elems = z_pre_centered.len(),
        z_pre_planes = z_pre_centered.len() * num_digits_fold,
        r_elems = r.len(),
        r_planes = r_hat_count,
        total_ring = z_count + r_hat_count,
        total_field = (z_count + r_hat_count) * D,
        z_first,
        "build_w_coeffs"
    );
    let total_planes = z_count + r_hat_count;
    let total_elems = total_planes * D;

    let mut out = Vec::with_capacity(total_elems);

    let w_block_count = w_hat.block_count();
    assert_eq!(
        w_hat_planes,
        w_block_count * depth_open,
        "build_w_coeffs: w_hat block layout does not match open digit depth"
    );
    let t_block_count = t_hat.block_count();
    let t_planes_per_block = if t_block_count == 0 {
        0
    } else {
        assert_eq!(
            t_hat_planes % t_block_count,
            0,
            "build_w_coeffs: t_hat block layout must be uniform"
        );
        t_hat_planes / t_block_count
    };

    if z_first {
        emit_z_pre_block_inner(
            &mut out,
            z_pre_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
        emit_planes_block_inner(&mut out, w_hat.flat_digits(), w_block_count, depth_open);
        emit_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            t_block_count,
            t_planes_per_block,
        );
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, b_blinding_digits);
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, std::slice::from_ref(d_blinding_digits));
    } else {
        emit_planes_block_inner(&mut out, w_hat.flat_digits(), w_block_count, depth_open);
        emit_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            t_block_count,
            t_planes_per_block,
        );
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, b_blinding_digits);
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, std::slice::from_ref(d_blinding_digits));
        emit_z_pre_block_inner(
            &mut out,
            z_pre_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
    }

    let mut r_planes = vec![[0i8; D]; levels];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    for ri in r {
        r_planes.fill([0i8; D]);
        ri.balanced_decompose_pow2_i8_into_with_params(&mut r_planes, &decompose_params);
        for plane in &r_planes {
            out.extend_from_slice(plane);
        }
    }
    RecursiveWitnessFlat::from_i8_digits(out)
}

#[cfg(test)]
mod tests {
    use super::balanced_decompose_centered_i32_i8_into;
    use akita_algebra::CyclotomicRing;
    use akita_field::Prime128OffsetA7F7;
    use std::array::from_fn;

    #[test]
    fn centered_i32_decompose_matches_ring_decompose() {
        type F = Prime128OffsetA7F7;
        const D: usize = 128;

        let centered = from_fn(|i| ((37 * i as i32 + 11) % 95) - 47);
        let ring =
            CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| F::from_i64(centered[i] as i64)));

        for (num_digits, log_basis) in [
            (7usize, 3u32),
            (10usize, 2u32),
            (5usize, 5u32),
            (4usize, 6u32),
        ] {
            let mut got = vec![[0i8; D]; num_digits];
            balanced_decompose_centered_i32_i8_into(&centered, &mut got, log_basis);

            let mut expected = vec![[0i8; D]; num_digits];
            ring.balanced_decompose_pow2_i8_into(&mut expected, log_basis);
            assert_eq!(
                got, expected,
                "centered i32 decomposition mismatch for num_digits={num_digits} log_basis={log_basis}"
            );
        }
    }
}
