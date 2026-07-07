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
    gadget_row_scalars, validate_role_dispatch, AkitaExpandedSetup, FpExtEncoding, LevelParams,
    MRowLayout, RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance, RingRole,
    RingVec, SetupContributionPlanInputs, TerminalWitnessTranscriptParts, WitnessLayout,
    FOLD_CONSISTENCY_ROW, FOLD_EVALUATION_ROW,
};

use super::slice_mle::{
    compute_r_contribution, high_eq_window, EStructuredSlicesEvaluator, SetupEvaluation,
    SetupEvaluator, SetupEvaluatorMode, StructuredSliceMleEvaluator, TStructuredSlicesEvaluator,
    ZDenseSlicesEvaluator, ZStructuredPow2SlicesEvaluator,
};
use super::validate_log_basis;
use akita_types::validate_ring_dispatch;
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod tensor_challenges;
#[cfg(test)]
mod tests;

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Deferred row evaluation state for relation-weight and stage-3 setup.
    pub deferred: RingSwitchDeferredRowEval<E>,
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
    deferred: RingSwitchDeferredRowEval<E>,
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
            deferred: self.deferred,
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
            deferred: self.deferred,
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
    pub(crate) log_basis: u32,
    pub(crate) n_a: usize,
    /// Resolved witness column layout (one chunk for the single-chunk case,
    /// `W` chunks for the distributed-prover layout).
    pub(crate) chunk_layout: WitnessLayout,
    pub(crate) setup_contribution_inputs: SetupContributionPlanInputs<F>,
}

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: FieldCore, E> {
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
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    w_commitment: &RingVec<F>,
    next_ring_dim: usize,
    transcript: &mut T,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    // `validate_ring_dispatch` is called inside `ring_switch_verifier_core`;
    // the outer wrapper just performs the witness absorb before delegating.
    // The next-witness commitment is shaped at the *next* level's schedule
    // ring dimension, which may differ from this level's dispatch `D` in
    // mixed-D schedules.
    if next_ring_dim == 0 || !w_commitment.can_decode_vec(next_ring_dim) {
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
    replay: &RingSwitchReplay<'_, F, E>,
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
    replay: &RingSwitchReplay<'_, F, E>,
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
    let num_polys = opening_batch.num_total_polynomials();
    let gamma = replay.row_coefficients;

    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_claims = relation.opening_batch().num_total_polynomials();
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
    let row_layout = relation.relation_row_layout(lp)?;
    let m_rows = row_layout.total_row_count();
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
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let deferred =
        prepare_ring_switch_row_eval::<F, E, D>(replay, alpha, &tau1, Some(num_ring_elems))?;

    Ok(RingSwitchVerifyCoreOutput {
        deferred,
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
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    witness_ring_len: Option<usize>,
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let chunk_layout = relation.segment_layout(lp, witness_ring_len)?;
    let opening_batch = relation.opening_batch();
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold(num_polys, F::modulus_bits())?;
    let row_layout = relation.relation_row_layout(lp)?;
    let rows = row_layout.total_row_count();
    prepare_ring_switch_row_eval_inner::<F, E, D>(
        &relation.challenges,
        alpha,
        lp,
        tau1,
        num_polys,
        replay.row_coefficients,
        relation.m_row_layout(),
        chunk_layout,
        depth_fold,
        rows,
        opening_batch.num_groups(),
        opening_batch.group_sizes(),
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
    chunk_layout: WitnessLayout,
    depth_fold: usize,
    rows: usize,
    num_commitment_groups: usize,
    num_polys_per_group: Vec<usize>,
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    validate_role_dispatch::<D>(lp.role_dims, RingRole::Inner)?;
    let setup_contribution_inputs = SetupContributionPlanInputs::from_level_params(
        lp,
        num_polys,
        num_commitment_groups,
        num_polys_per_group,
        m_row_layout,
        depth_fold,
    )?
    .with_eq_tau1_from_tau(tau1, rows)?;
    let eq_tau1 = setup_contribution_inputs.eq_tau1.clone();
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = gamma.len();
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let log_basis = lp.log_basis;
    validate_log_basis(log_basis)?;
    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let num_blocks = lp.num_blocks;
    let num_t_vectors = num_polys;
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
    let n_a = lp.a_key.row_len();

    let c_alphas: PreparedChallengeEvals<E> = match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => PreparedChallengeEvals::Flat(
            sparse
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(&alpha_pows))
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
        log_basis,
        n_a,
        chunk_layout,
        setup_contribution_inputs,
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

    pub(crate) fn chunk_layout(&self) -> &WitnessLayout {
        &self.chunk_layout
    }

    pub(crate) fn create_setup_contribution_inputs(&self) -> SetupContributionPlanInputs<E> {
        self.setup_contribution_inputs.clone()
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
        // ----- Witness layout (chunk list) -----------------------------------
        let layout = self.chunk_layout();
        let blocks_per_chunk = layout.blocks_per_chunk;
        if blocks_per_chunk == 0 || !blocks_per_chunk.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "witness chunk block window must be a power of two".to_string(),
            ));
        }
        validate_log_basis(self.log_basis)?;
        if opening_point.b.len() != self.num_blocks || opening_point.a.len() < self.block_len {
            return Err(AkitaError::InvalidProof);
        }
        if ring_multiplier_point.b_len() != self.num_blocks
            || ring_multiplier_point.a_len() < self.block_len
        {
            return Err(AkitaError::InvalidProof);
        }
        if self.num_t_vectors != self.num_claims {
            return Err(AkitaError::InvalidProof);
        }

        // ----- Shared precomputes --------------------------------------------
        let alpha_pows = scalar_powers(alpha, D);
        let g1_open = gadget_row_scalars::<F>(self.depth_open, self.log_basis);
        let g1_commit = gadget_row_scalars::<F>(self.depth_commit, self.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);

        // e/t block peel is over `blocks_per_chunk` (`== num_blocks` single-chunk);
        // the `eq_low` table is shared across chunks. z peels `block_len`.
        let block_bits = blocks_per_chunk.trailing_zeros() as usize;
        if block_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: block_bits,
                actual: x_challenges.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&x_challenges[..block_bits])?;
        let high_challenges = &x_challenges[block_bits..];
        let x_low_challenges = &x_challenges[..block_bits];

        let z_offset_low_bits = self.block_len.trailing_zeros() as usize;
        if z_offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let z_block_low_eq = EqPolynomial::evals(&x_challenges[..z_offset_low_bits])?;

        let total_blocks = self.total_blocks();
        if let Some(c_alphas) = self.c_alphas.as_flat() {
            if c_alphas.len() != total_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: total_blocks,
                    actual: c_alphas.len(),
                });
            }
        }

        // EvaluationTrace | FoldEvaluation | FoldConsistency | B | D.
        let fold_evaluation_row = FOLD_EVALUATION_ROW;
        let a_start = FOLD_CONSISTENCY_ROW;
        let a_row_count = self.n_a;

        // ----- E-hat / T-hat / Z structured: fold over chunks ----------------
        // `e`/`t` are partitioned (each chunk covers a disjoint global block
        // window, so the contributions sum to the whole component); `z` is
        // replicated (each chunk carries a full `block_len` fold). The cost
        // asymmetry falls out of the chunk geometry, not control flow.
        let mut e_structured_contribution = E::zero();
        let mut t_structured_contribution = E::zero();
        let mut z_structured_contribution = E::zero();
        {
            let _span = tracing::info_span!("structured_chunks").entered();
            for chunk in &layout.chunks {
                // e and t share the in-window block residue: `|e^j|` is a
                // multiple of `blocks_per_chunk`, so `offset_t ≡ offset_e`.
                let block_offset_low = chunk.offset_e & (blocks_per_chunk - 1);
                let summaries = self.c_alphas.summarize_chunk_block_carries::<F, D>(
                    self.num_claims,
                    x_low_challenges,
                    &eq_low,
                    block_offset_low,
                    chunk.global_block_base,
                    blocks_per_chunk,
                    self.num_blocks,
                )?;

                let e_offset_high = chunk.offset_e >> block_bits;
                let eq_hi_e_table = high_eq_window(
                    high_challenges,
                    e_offset_high,
                    self.num_claims * self.depth_open,
                );
                e_structured_contribution += EStructuredSlicesEvaluator {
                    gadget_vector: &g1_open,
                    challenge_block_summaries: &summaries,
                    challenge_weight: self.eq_tau1[fold_evaluation_row],
                    high_eq_table: &eq_hi_e_table,
                }
                .evaluate();

                let t_offset_high = chunk.offset_t >> block_bits;
                let eq_hi_t_table = high_eq_window(
                    high_challenges,
                    t_offset_high,
                    self.num_claims * self.depth_open * a_row_count,
                );
                t_structured_contribution += TStructuredSlicesEvaluator {
                    gadget_vector: &g1_open,
                    challenge_block_summaries: &summaries,
                    a_row_weights: &self.eq_tau1[a_start..(a_start + a_row_count)],
                    high_eq_table: &eq_hi_t_table,
                }
                .evaluate();
            }

            // z dispatches once on `block_len` (chunk-independent); the chunk
            // loop sits outside the case split. Chunk `j>0` exercises a nonzero
            // in-block shift `z_lo = offset_z mod block_len`.
            if self.block_len.is_power_of_two() {
                for chunk in &layout.chunks {
                    let z_offset_low = chunk.offset_z & (self.block_len - 1);
                    let a_block_summary = vec![summarize_pow2_multiplier_block_carries(
                        &z_block_low_eq,
                        z_offset_low,
                        self.block_len,
                        |idx| ring_multiplier_point.eval_a_at::<D, E>(idx, &alpha_pows),
                    )?];
                    let z_offset_high = chunk.offset_z >> z_offset_low_bits;
                    let z_hi_len = a_block_summary.len() * fold_gadget.len() * g1_commit.len();
                    let eq_hi_z_table =
                        high_eq_window(&x_challenges[z_offset_low_bits..], z_offset_high, z_hi_len);
                    z_structured_contribution += ZStructuredPow2SlicesEvaluator {
                        g1_commit: &g1_commit,
                        fold_gadget: &fold_gadget,
                        a_block_summary: &a_block_summary,
                        consistency_weight: self.eq_tau1[fold_evaluation_row],
                        high_eq_table: &eq_hi_z_table,
                    }
                    .evaluate();
                }
            } else {
                // `a_evals_by_point` is chunk-independent (a[blk] is global), so
                // the dense `z` segment is identical in every chunk; only the
                // offset shifts.
                let a_evals_by_point = vec![(0..self.block_len)
                    .map(|idx| ring_multiplier_point.eval_a_at::<D, E>(idx, &alpha_pows))
                    .collect::<Result<Vec<_>, _>>()?];
                for chunk in &layout.chunks {
                    z_structured_contribution += ZDenseSlicesEvaluator {
                        g1_commit: &g1_commit,
                        fold_gadget: &fold_gadget,
                        consistency_weight: self.eq_tau1[fold_evaluation_row],
                        a_evals_by_point: &a_evals_by_point,
                        full_vec_randomness: x_challenges,
                        offset_z: chunk.offset_z,
                        block_len: self.block_len,
                    }
                    .evaluate()?;
                }
            }
        }

        // ----- Fused D·ŵ + B·t̂ + A·ẑ (one shared setup scan) ---------------
        let setup_contribution = {
            let _span = tracing::info_span!("setup_contribution").entered();
            if let Some(claim) = setup_claim {
                claim
            } else {
                let setup_contribution_inputs = self.create_setup_contribution_inputs();
                let evaluator = SetupEvaluator::new(
                    &setup_contribution_inputs,
                    x_challenges,
                    Some(&eq_low),
                    Some(&z_block_low_eq),
                    &alpha_pows,
                    &fold_gadget,
                    layout,
                );
                match evaluator.evaluate::<D>(SetupEvaluatorMode::Direct { setup })? {
                    SetupEvaluation::Direct(value) => value,
                    #[cfg(test)]
                    SetupEvaluation::Recursive(_) => {
                        return Err(AkitaError::InvalidSetup(
                            "setup evaluator returned recursive output for direct mode".into(),
                        ))
                    }
                }
            }
        };

        // ----- r-tail (single shared quotient on the last chunk) -------------
        let r_contribution = {
            let offset_r = layout.r_offset()?;
            compute_r_contribution(self, x_challenges, offset_r, &layout.quotient_layout, alpha)?
        };

        let total = e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution;

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
