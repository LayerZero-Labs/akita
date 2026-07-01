//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_challenges::Challenges;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_NEXT_LEVEL_WITNESS_BINDING, ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_RING_SWITCH,
    CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, FpExtEncoding, LevelParams,
    MRowLayout, RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance,
    RingRelationSegmentLayout, RingVec, SetupContributionPlanInputs,
    TerminalWitnessTranscriptParts,
};

use super::slice_mle::{
    compute_r_contribution, high_eq_window, EStructuredSlicesEvaluator, SetupEvaluation,
    SetupEvaluator, SetupEvaluatorMode, StructuredSliceMleEvaluator, TStructuredSlicesEvaluator,
    ZDenseSlicesEvaluator, ZStructuredPow2SlicesEvaluator,
};
use super::{validate_level_dispatch, validate_log_basis, validate_ring_dispatch};
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod tensor_challenges;
#[cfg(test)]
mod tests;

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Prepared data for deferred ring-switch row MLE evaluation.
    pub prepared_row_eval: RingSwitchDeferredRowEval<E>,
    /// Evaluation table of alpha powers over the ring-coordinate dimension.
    pub alpha_evals_y: Vec<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for the stage-1 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for the stage-2 M-row combination.
    pub tau1: Vec<E>,
    /// Basis size `b = 2^log_basis`.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}

struct RingSwitchVerifyCoreOutput<E: FieldCore> {
    prepared_row_eval: RingSwitchDeferredRowEval<E>,
    alpha_evals_y: Vec<E>,
    col_bits: usize,
    ring_bits: usize,
    tau0: Option<Vec<E>>,
    tau1: Vec<E>,
    b: usize,
    alpha: E,
}

impl<E: FieldCore> RingSwitchVerifyCoreOutput<E> {
    fn into_intermediate(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        let tau0 = self.tau0.ok_or(AkitaError::InvalidProof)?;
        Ok(RingSwitchVerifyOutput {
            prepared_row_eval: self.prepared_row_eval,
            alpha_evals_y: self.alpha_evals_y,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0,
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }

    fn into_terminal_as_output(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        if self.tau0.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(RingSwitchVerifyOutput {
            prepared_row_eval: self.prepared_row_eval,
            alpha_evals_y: self.alpha_evals_y,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0: Vec::new(),
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }
}

/// Precomputed challenge-derived data for deferred ring-switch row MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
#[derive(Clone)]
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) eq_tau1: Vec<F>,
    pub(crate) num_t_vectors: usize,
    pub(crate) num_blocks: usize,
    pub(crate) num_claims: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_fold: usize,
    pub(crate) block_len: usize,
    pub(crate) inner_width: usize,
    pub(crate) log_basis: u32,
    pub(crate) n_a: usize,
    pub(crate) n_d: usize,
    pub(crate) m_row_layout: MRowLayout,
    pub(crate) n_b: usize,
    /// Tiered split factor `f` (`1` = single-tier).
    pub(crate) tier_split: usize,
    /// Second-tier `F` rank (`0` = single-tier); the sent-commitment length.
    pub(crate) n_f: usize,
    pub(crate) rows: usize,
    pub(crate) num_polys: usize,
    pub(crate) witness_segment_layout: RingRelationSegmentLayout,
}

pub(crate) type RingSwitchSegmentLayout = RingRelationSegmentLayout;

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: FieldCore, E, const D: usize> {
    pub relation: &'a RingRelationInstance<F>,
    pub row_coefficients: &'a [E],
    pub lp: &'a LevelParams,
}

/// Replay the verifier half of ring switching.
///
/// This handles the single-point relation replay for one committed polynomial
/// bundle.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or ring-switch row-eval
/// preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    w_commitment: &RingVec<F>,
    transcript: &mut T,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    // `validate_ring_dispatch` is called inside `ring_switch_verifier_core`;
    // the outer wrapper just performs the witness absorb before delegating.
    if !w_commitment.can_decode_vec(D) {
        return Err(AkitaError::InvalidProof);
    }
    transcript.absorb_and_record_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, w_commitment);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithDBlock)?
        .into_intermediate()
}

/// Terminal variant of [`ring_switch_verifier`].
///
/// This owns the required terminal final-witness remainder absorb before
/// sampling ring-switch challenges.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or
/// ring-switch row-eval preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier_terminal")]
#[inline(never)]
pub(crate) fn ring_switch_verifier_terminal<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    transcript: &mut T,
    terminal_parts: &TerminalWitnessTranscriptParts,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithoutDBlock)?
        .into_terminal_as_output()
}

#[tracing::instrument(skip_all, name = "ring_switch_verifier_core")]
#[inline(never)]
fn ring_switch_verifier_core<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    transcript: &mut T,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchVerifyCoreOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    let num_polys = opening_batch.num_polynomials();
    let gamma = replay.row_coefficients;

    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_claims = relation.opening_batch().num_polynomials();
    if relation.opening_point().a.len() < lp.block_len
        || relation.opening_point().b.len() != lp.num_blocks
    {
        return Err(AkitaError::InvalidProof);
    }
    if relation.ring_multiplier_point().a_len() < lp.block_len
        || relation.ring_multiplier_point().b_len() != lp.num_blocks
    {
        return Err(AkitaError::InvalidProof);
    }
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    if w_len == 0 || !w_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch column count overflow".to_string()))?
        .trailing_zeros() as usize;
    let ring_bits = validate_ring_dispatch::<D>()?;
    let m_rows = lp.m_row_count_for(1, 0, m_row_layout)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let tau0 = match m_row_layout {
        MRowLayout::WithDBlock => Some(
            (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
        ),
        MRowLayout::WithoutDBlock => None,
    };
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_row_eval = prepare_ring_switch_row_eval::<F, E, D>(replay, alpha, &tau1)?;

    Ok(RingSwitchVerifyCoreOutput {
        prepared_row_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("basis size overflow".to_string()))?,
        alpha,
    })
}

/// Prepare deferred verifier ring-switch row evaluation data from a fixed
/// [`RingRelationInstance`] and transcript-sampled row coefficients.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[tracing::instrument(skip_all, name = "prepare_ring_switch_row_eval")]
pub fn prepare_ring_switch_row_eval<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    alpha: E,
    tau1: &[E],
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let witness_segment_layout = relation.segment_layout(lp)?;
    let opening_batch = relation.opening_batch();
    prepare_ring_switch_row_eval_inner::<F, E, D>(
        &relation.challenges,
        alpha,
        lp,
        tau1,
        opening_batch.num_polynomials(),
        replay.row_coefficients,
        relation.m_row_layout(),
        witness_segment_layout,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_ring_switch_row_eval_inner<F, E, const D: usize>(
    challenges: &Challenges,
    alpha: E,
    lp: &LevelParams,
    tau1: &[E],
    num_polys: usize,
    gamma: &[E],
    m_row_layout: MRowLayout,
    witness_segment_layout: RingRelationSegmentLayout,
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    validate_level_dispatch::<D>(lp)?;
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = gamma.len();
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let log_basis = lp.log_basis;
    validate_log_basis(log_basis)?;
    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;
    let num_blocks = lp.num_blocks;
    if num_blocks == 0 || !num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "num_blocks must be a non-zero power of two".to_string(),
        ));
    }
    if lp.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "block_len must be non-zero".to_string(),
        ));
    }
    if depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "digit depths must be non-zero".to_string(),
        ));
    }
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let num_t_vectors = num_polys;
    // Must match [`RingSwitchDeferredRowEval::total_blocks`] on the prepared value.
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let block_len = lp.block_len;
    let inner_width = block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
    if lp.a_key.col_len() < inner_width {
        return Err(AkitaError::InvalidSetup(
            "A-key column width is too small for verifier layout".to_string(),
        ));
    }
    let _expected_d_width = depth_open
        .checked_mul(num_blocks)
        .and_then(|width| width.checked_mul(num_claims))
        .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;
    // TODO: re-enable (or gate on schedule shape) once root-direct
    // commit params no longer carry zero-width D-key placeholders.
    // The planner emits `d_key.col_len = 0` for root-direct schedules
    // since the relation fold (which is what consumes D) doesn't run.
    // if lp.d_key.col_len() < expected_d_width {
    //     return Err(AkitaError::InvalidSetup(
    //         "D-key column width is too small for verifier layout".to_string(),
    //     ));
    // }
    let expected_b_width = num_polys
        .checked_mul(lp.a_key.row_len())
        .and_then(|width| width.checked_mul(depth_open))
        .and_then(|width| width.checked_mul(num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".to_string()))?;
    // Tiered: the stored first-tier `B'` is the full B width divided by the
    // reuse factor `tier_split`.
    let expected_stored_b_width = if lp.f_key.is_some() {
        expected_b_width.div_ceil(lp.tier_split.max(1))
    } else {
        expected_b_width
    };
    if lp.b_key.col_len() < expected_stored_b_width {
        return Err(AkitaError::InvalidSetup(
            "B-key column width is too small for verifier layout".to_string(),
        ));
    }
    let rows = lp.m_row_count_for(1, 0, m_row_layout)?;

    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let c_alphas: PreparedChallengeEvals<E> = match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => PreparedChallengeEvals::Flat(
            sparse
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(&alpha_pows))
                .collect::<Result<_, _>>()?,
        ),
        Challenges::Tensor { factored } => {
            if D < 2 {
                return Err(AkitaError::InvalidInput(
                    "tensor challenge factored evaluation requires D >= 2".to_string(),
                ));
            }
            factored.validate::<D>()?;
            if factored.num_claims != num_claims {
                return Err(AkitaError::InvalidSize {
                    expected: num_claims,
                    actual: factored.num_claims,
                });
            }
            let blocks_per_claim = factored.blocks_per_claim()?;
            if blocks_per_claim != lp.num_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: lp.num_blocks,
                    actual: blocks_per_claim,
                });
            }
            PreparedChallengeEvals::Tensor {
                challenges: factored.clone(),
                alpha_pows: alpha_pows.clone(),
            }
        }
    };

    Ok(RingSwitchDeferredRowEval {
        c_alphas,
        eq_tau1,
        num_t_vectors,
        num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        block_len,
        inner_width,
        log_basis,
        n_a: lp.a_key.row_len(),
        n_d,
        m_row_layout,
        n_b,
        tier_split: lp.tier_split,
        n_f: lp.f_key.as_ref().map_or(0, |fk| fk.row_len()),
        rows,
        num_polys,
        witness_segment_layout,
    })
}

impl<E: FieldCore> RingSwitchDeferredRowEval<E> {
    /// `num_blocks * num_claims` (W/D challenge logical length).
    ///
    /// Prepare validates the product with checked arithmetic before building
    /// this struct; replay uses the unchecked product on those same fields.
    #[inline(always)]
    pub(crate) fn total_blocks(&self) -> usize {
        self.num_blocks * self.num_claims
    }

    /// Number of active D rows in the selected M-row layout.
    pub(crate) fn n_d_active(&self) -> usize {
        match self.m_row_layout {
            MRowLayout::WithDBlock => self.n_d,
            MRowLayout::WithoutDBlock => 0,
        }
    }

    pub(crate) fn segment_layout(&self) -> Result<RingSwitchSegmentLayout, AkitaError> {
        Ok(self.witness_segment_layout)
    }

    pub(crate) fn create_setup_contribution_inputs(&self) -> SetupContributionPlanInputs<E> {
        SetupContributionPlanInputs {
            eq_tau1: self.eq_tau1.clone(),
            num_t_vectors: self.num_t_vectors,
            num_blocks: self.num_blocks,
            num_claims: self.num_claims,
            depth_open: self.depth_open,
            depth_commit: self.depth_commit,
            depth_fold: self.depth_fold,
            block_len: self.block_len,
            inner_width: self.inner_width,
            n_a: self.n_a,
            n_d: self.n_d,
            m_row_layout: self.m_row_layout,
            n_b: self.n_b,
            num_segments: 1,
            rows: self.rows,
            num_polys_per_segment: vec![self.num_polys],
            num_public_rows: 0,
            tier_split: self.tier_split,
            n_f: self.n_f,
        }
    }

    /// Evaluate the prepared ring-switch row table at the supplied point.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup matrix cannot be viewed at `D` or an
    /// internal offset-eq evaluation receives inconsistent dimensions.
    #[inline]
    pub fn eval_at_point<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_point: &RingOpeningPoint<F>,
        ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt,
    {
        let _ring_bits = validate_ring_dispatch::<D>()?;
        // ----- Witness-layout offsets ----------------------------------------
        let layout = self.segment_layout()?;
        validate_log_basis(self.log_basis)?;
        if opening_point.b.len() != self.num_blocks || opening_point.a.len() < self.block_len {
            return Err(AkitaError::InvalidProof);
        }
        if ring_multiplier_point.b_len() != self.num_blocks
            || ring_multiplier_point.a_len() < self.block_len
        {
            return Err(AkitaError::InvalidProof);
        }

        // ----- Shared precomputes --------------------------------------------
        let alpha_pows = scalar_powers(alpha, D);
        let g1_open = gadget_row_scalars::<F>(self.depth_open, self.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);

        // Eq table over the low `log₂(num_blocks)` bits, shared by e-hat/T
        // peeled summaries and by `SetupEvaluator` direct mode.
        let offset_low_bits = self.num_blocks.trailing_zeros() as usize;
        if offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&x_challenges[..offset_low_bits])?;
        let block_offset_low = layout.offset_e & (self.num_blocks - 1);
        debug_assert_eq!(block_offset_low, layout.offset_t & (self.num_blocks - 1));

        // `z` peels `block_len` (not `num_blocks`) and uses its own
        // low-bit eq table.
        let z_offset_low_bits = self.block_len.trailing_zeros() as usize;
        if z_offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let z_block_low_eq = EqPolynomial::evals(&x_challenges[..z_offset_low_bits])?;

        let high_challenges = &x_challenges[offset_low_bits..];

        let x_low_challenges = &x_challenges[..offset_low_bits];
        let total_blocks = self.total_blocks();
        if let Some(c_alphas) = self.c_alphas.as_flat() {
            if c_alphas.len() != total_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: total_blocks,
                    actual: c_alphas.len(),
                });
            }
        }
        let challenge_block_summaries: Vec<[E; 2]> =
            self.c_alphas.summarize_all_block_carries::<F, D>(
                self.num_claims,
                x_low_challenges,
                &eq_low,
                block_offset_low,
                self.num_blocks,
            )?;
        if self.num_t_vectors != self.num_claims {
            return Err(AkitaError::InvalidProof);
        }

        // ----- E-hat ---------------------------------------------------------
        let e_offset_high = layout.offset_e >> offset_low_bits;
        let eq_hi_e_table = high_eq_window(
            high_challenges,
            e_offset_high,
            self.num_claims * self.depth_open,
        );
        let e_structured_contribution = {
            let _span = tracing::info_span!("e_structured").entered();
            EStructuredSlicesEvaluator {
                gadget_vector: &g1_open,
                challenge_block_summaries: &challenge_block_summaries,
                challenge_weight: self.eq_tau1[0],
                high_eq_table: &eq_hi_e_table,
            }
            .evaluate()
        };

        // Canonical A-block start (tiered-aware): consistency | public | D |
        // COMMIT (F when tiered, else B) | B_inner (tiered) | A.
        let commit_rows_pg = if self.tier_split > 1 {
            self.n_f
        } else {
            self.n_b
        };
        let b_inner_rows_pg = if self.tier_split > 1 {
            self.tier_split * self.n_b
        } else {
            0
        };
        let a_start = 1 + self.n_d_active() + (commit_rows_pg + b_inner_rows_pg);

        // ----- T -------------------------------------------------------------
        let t_offset_high = layout.offset_t >> offset_low_bits;
        let a_row_count = self.rows.saturating_sub(a_start);
        let eq_hi_t_table = high_eq_window(
            high_challenges,
            t_offset_high,
            self.num_claims * self.depth_open * a_row_count,
        );
        let t_structured_contribution = {
            let _span = tracing::info_span!("t_structured").entered();
            TStructuredSlicesEvaluator {
                gadget_vector: &g1_open,
                challenge_block_summaries: &challenge_block_summaries,
                a_row_weights: &self.eq_tau1[a_start..self.rows],
                high_eq_table: &eq_hi_t_table,
            }
            .evaluate()
        };

        // ----- Fused D·ŵ + B·t̂ + A·ẑ ---------------------------------------
        let setup_contribution = {
            let _span = tracing::info_span!("setup_contribution").entered();
            let result = if let Some(claim) = setup_claim {
                Ok(claim)
            } else {
                let setup_contribution_inputs = self.create_setup_contribution_inputs();
                let evaluator = SetupEvaluator::new(
                    &setup_contribution_inputs,
                    x_challenges,
                    Some(&eq_low),
                    Some(&z_block_low_eq),
                    &alpha_pows,
                    &fold_gadget,
                    layout.offset_e,
                    layout.offset_t,
                    layout.offset_z,
                    layout.offset_u,
                    Some(&eq_hi_e_table),
                    Some(&eq_hi_t_table),
                );
                match evaluator.evaluate::<D>(SetupEvaluatorMode::Direct { setup })? {
                    SetupEvaluation::Direct(value) => Ok(value),
                    #[cfg(test)]
                    SetupEvaluation::Recursive(_) => Err(AkitaError::InvalidSetup(
                        "setup evaluator returned recursive output for direct mode".into(),
                    )),
                }
            };
            result?
        };

        // ----- Z (consistency-row) ------------------------------------------
        let z_structured_contribution = {
            let _span = tracing::info_span!("z_structured").entered();
            let g1_commit = gadget_row_scalars::<F>(self.depth_commit, self.log_basis);
            if self.block_len.is_power_of_two() {
                let z_offset_low = layout.offset_z & (self.block_len - 1);
                let a_block_summary = vec![summarize_pow2_multiplier_block_carries(
                    &z_block_low_eq,
                    z_offset_low,
                    self.block_len,
                    |idx| ring_multiplier_point.eval_a_at::<D, E>(idx, &alpha_pows),
                )?];
                let z_offset_high = layout.offset_z >> z_offset_low_bits;
                let z_hi_len = a_block_summary.len() * fold_gadget.len() * g1_commit.len();
                let eq_hi_z_table =
                    high_eq_window(&x_challenges[z_offset_low_bits..], z_offset_high, z_hi_len);
                ZStructuredPow2SlicesEvaluator {
                    g1_commit: &g1_commit,
                    fold_gadget: &fold_gadget,
                    a_block_summary: &a_block_summary,
                    consistency_weight: self.eq_tau1[0],
                    high_eq_table: &eq_hi_z_table,
                }
                .evaluate()
            } else {
                let a_evals_by_point = vec![(0..self.block_len)
                    .map(|idx| ring_multiplier_point.eval_a_at::<D, E>(idx, &alpha_pows))
                    .collect::<Result<Vec<_>, _>>()?];
                ZDenseSlicesEvaluator {
                    g1_commit: &g1_commit,
                    fold_gadget: &fold_gadget,
                    consistency_weight: self.eq_tau1[0],
                    a_evals_by_point: &a_evals_by_point,
                    full_vec_randomness: x_challenges,
                    offset_z: layout.offset_z,
                    block_len: self.block_len,
                }
                .evaluate()?
            }
        };

        // ----- r-tail --------------------------------------------------------
        let r_contribution = {
            let r_gadget =
                gadget_row_scalars::<F>(r_decomp_levels::<F>(self.log_basis), self.log_basis);
            let denom = alpha_pows[D - 1] * alpha + E::one();
            compute_r_contribution(self, x_challenges, layout.offset_r, denom, &r_gadget)?
        };

        // ----- Tiered B_inner RHS: -recompose(û_concat) ----------------------
        // The B_inner block enforces `B'·t̂_slice - recompose(û) = 0`. The B'
        // matrix part is in `setup_contribution`; this is the witness-side
        // `-recompose(û)` term (a constant gadget map on the `û_concat`
        // columns), weighted by the B_inner row eq. Zero for single-tier.
        let u_recompose_contribution = if self.tier_split > 1 {
            let n_d_active = self.n_d_active();
            let f_start = 1 + n_d_active;
            let b_inner_start = f_start + commit_rows_pg;
            let n_b_small = self.n_b;
            let inner_rows_pg = self.tier_split * n_b_small;
            let offset_u = layout.offset_u;
            let mut acc = E::zero();
            for slice_row in 0..inner_rows_pg {
                let row = b_inner_start + slice_row;
                let row_w = self.eq_tau1[row];
                if row_w.is_zero() {
                    continue;
                }
                let base_col = offset_u + slice_row * self.depth_open;
                let mut recomp = E::zero();
                for (digit, &gd) in g1_open.iter().enumerate().take(self.depth_open) {
                    let eq_col =
                        akita_algebra::offset_eq::eq_eval_at_index(x_challenges, base_col + digit);
                    recomp += eq_col.mul_base(gd);
                }
                acc -= row_w * recomp;
            }
            acc
        } else {
            E::zero()
        };

        let total = e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution
            + u_recompose_contribution;

        Ok(total)
    }
}

#[inline]
fn summarize_pow2_multiplier_block_carries<E, EvalAt>(
    eq_low: &[E],
    offset_low: usize,
    values_len: usize,
    mut eval_at: EvalAt,
) -> Result<[E; 2], AkitaError>
where
    E: FieldCore,
    EvalAt: FnMut(usize) -> Result<E, AkitaError>,
{
    if !values_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "peeled inner block length must be a power of two".to_string(),
        ));
    }
    if eq_low.len() != values_len {
        return Err(AkitaError::InvalidSize {
            expected: values_len,
            actual: eq_low.len(),
        });
    }
    if offset_low >= values_len {
        return Err(AkitaError::InvalidInput(
            "low offset must lie inside the peeled block".to_string(),
        ));
    }

    let inner_bits = values_len.trailing_zeros() as usize;
    let inner_mask = values_len - 1;
    let mut out = [E::zero(), E::zero()];

    for u in 0..values_len {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += eq_low[low_idx] * eval_at(u)?;
    }

    Ok(out)
}

#[cfg(test)]
#[inline]
pub(crate) fn summarize_pow2_block_carries_base<F, E>(
    eq_low: &[E],
    offset_low: usize,
    values: &[F],
) -> Result<[E; 2], AkitaError>
where
    F: FieldCore,
    E: akita_field::ExtField<F>,
{
    if !values.len().is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "peeled inner block length must be a power of two".to_string(),
        ));
    }
    if eq_low.len() != values.len() {
        return Err(AkitaError::InvalidSize {
            expected: values.len(),
            actual: eq_low.len(),
        });
    }
    if offset_low >= values.len() {
        return Err(AkitaError::InvalidInput(
            "low offset must lie inside the peeled block".to_string(),
        ));
    }

    let inner_bits = values.len().trailing_zeros() as usize;
    let inner_mask = values.len() - 1;
    let mut out = [E::zero(), E::zero()];

    for (u, &value) in values.iter().enumerate() {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += eq_low[low_idx].mul_base(value);
    }

    Ok(out)
}
