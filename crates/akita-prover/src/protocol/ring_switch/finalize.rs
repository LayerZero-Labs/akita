use super::*;

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
    instance: &RingRelationInstance<F, D>,
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
        instance,
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
    instance: &RingRelationInstance<F, D>,
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
    let gamma = instance
        .gamma()
        .iter()
        .copied()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    ring_switch_finalize_with_gamma::<F, E, T, D>(
        instance,
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
    instance: &RingRelationInstance<F, D>,
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
    let gamma = instance
        .gamma()
        .iter()
        .copied()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    ring_switch_finalize_with_gamma_after_absorb::<F, E, T, D>(
        instance,
        setup,
        transcript,
        w,
        lp,
        &gamma,
        m_row_layout,
    )
}

/// Terminal variant of [`ring_switch_finalize`].
///
/// The terminal fold binds logical `w_hat` before fold challenge sampling.
/// This function binds the remaining final-witness bytes before ring-switch
/// challenge sampling.
///
/// # Errors
///
/// Returns an error if terminal witness slicing fails, the final witness does
/// not match the ring-switch witness, or ring-switch finalization fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize_terminal<F, E, T, const D: usize>(
    instance: &RingRelationInstance<F, D>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    final_witness: &CleartextWitnessProof<F>,
    terminal_layout: TerminalWitnessSegmentLayout,
    lp: &LevelParams,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let gamma = instance
        .gamma()
        .iter()
        .copied()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    ring_switch_finalize_terminal_with_gamma::<F, E, T, D>(
        instance,
        setup,
        transcript,
        w,
        final_witness,
        terminal_layout,
        lp,
        &gamma,
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
    instance: &RingRelationInstance<F, D>,
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
        instance,
        setup,
        transcript,
        w,
        lp,
        gamma,
        m_row_layout,
    )
}

/// Terminal variant of [`ring_switch_finalize_with_gamma`].
///
/// This owns the terminal final-witness remainder absorb before sampling
/// ring-switch challenges.
///
/// # Errors
///
/// Returns an error if terminal witness slicing fails, the supplied gamma
/// vector has the wrong shape, or ring-switch finalization fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize_terminal_with_gamma<F, E, T, const D: usize>(
    instance: &RingRelationInstance<F, D>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    final_witness: &CleartextWitnessProof<F>,
    terminal_layout: TerminalWitnessSegmentLayout,
    lp: &LevelParams,
    gamma: &[E],
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let parts = final_witness.terminal_transcript_parts(terminal_layout)?;
    if final_witness.packed_i8_digits()?.as_slice() != w.as_i8_digits() {
        return Err(AkitaError::InvalidInput(
            "terminal final witness does not match ring-switch witness".to_string(),
        ));
    }
    transcript.append_bytes(ABSORB_TERMINAL_W_REMAINDER, &parts.remainder);
    ring_switch_finalize_with_gamma_after_absorb::<F, E, T, D>(
        instance,
        setup,
        transcript,
        w,
        lp,
        gamma,
        MRowLayout::WithoutDBlock,
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
    instance: &RingRelationInstance<F, D>,
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

    let routing = instance.commitment_routing();
    let num_polys_per_commitment_group = routing.num_polys_per_commitment_group();
    let num_points = num_polys_per_commitment_group.len();
    let num_public_rows = instance.num_public_rows();

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

    let tau0: Vec<E> = match m_row_layout {
        MRowLayout::WithDBlock => (0..num_sc_vars)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
            .collect(),
        MRowLayout::WithoutDBlock => Vec::new(),
    };
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let ring_alpha_evals_y = scalar_powers(alpha, D);
    let alpha_evals_y = scalar_powers(alpha, D);

    let opening_points = instance.opening_points();
    let ring_multiplier_points = instance.ring_multiplier_points();
    let claim_to_point = instance.claim_to_point();
    let claim_to_commitment_group = routing.claim_to_commitment_group();
    let claim_poly_in_commitment_group = routing.claim_poly_in_commitment_group();
    let challenges = &instance.challenges;
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
                num_polys_per_commitment_group,
                claim_to_commitment_group,
                claim_poly_in_commitment_group,
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
            num_polys_per_commitment_group,
            claim_to_commitment_group,
            claim_poly_in_commitment_group,
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
