//! Setup-product sumcheck for a dense table against two disjoint factors.
//!
//! The table is laid out as `left * right_len + right`. The right factor is
//! bound first, then the left factor. This matches setup products of the form
//! `S(lambda, y) * omega(lambda) * alpha(y)` without materializing the full
//! `omega(lambda) * alpha(y)` table.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_algebra::uni_poly::UniPoly;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::{labels::ABSORB_SETUP_PREFIX_SLOT, Transcript};
use akita_types::{
    gadget_row_scalars, select_setup_prefix_slot, AkitaExpandedSetup, FpExtEncoding, LevelParams,
    RingRelationInstance, SetupContributionPlan, SetupContributionPlanInputs,
    SetupPrefixProverRegistry, SETUP_OFFLOAD_D_SETUP, SETUP_SUMCHECK_DEGREE,
};

/// Proves `sum_{l,r} table[l,r] * left_factor[l] * right_factor[r]`.
pub struct SetupSumcheckProver<E: FieldCore> {
    table: Vec<E>,
    left_factor: Vec<E>,
    right_factor: Vec<E>,
    input_claim: E,
    right_rounds: usize,
    total_rounds: usize,
}

/// Output of the batched stage-3 prover.
pub struct BatchedStage3SumcheckProverOutput<E: FieldCore> {
    /// Claimed setup contribution fed into the stage-2 final row evaluation.
    pub setup_claim: E,
    /// Re-randomized next-witness opening after the batched stage-3 point projection.
    pub next_w_eval: E,
    /// Batched next-witness opening point.
    pub next_w_point: Vec<E>,
    /// Degree-two batched setup-product + carried-witness sumcheck.
    pub sumcheck: SumcheckProof<E>,
}

impl<E: FieldCore> SetupSumcheckProver<E> {
    /// Construct a factored product-sumcheck prover.
    ///
    /// # Errors
    ///
    /// Returns an error if factor lengths are not powers of two, are empty, or
    /// if `table.len() != left_factor.len() * right_factor.len()`.
    fn new(table: Vec<E>, left_factor: Vec<E>, right_factor: Vec<E>) -> Result<Self, AkitaError> {
        if left_factor.is_empty()
            || right_factor.is_empty()
            || !left_factor.len().is_power_of_two()
            || !right_factor.len().is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "factored product dimensions must be non-empty powers of two".to_string(),
            ));
        }
        let expected_len = left_factor
            .len()
            .checked_mul(right_factor.len())
            .ok_or_else(|| AkitaError::InvalidInput("factored product size overflow".into()))?;
        if table.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: table.len(),
            });
        }

        let input_claim = product_claim(&table, &left_factor, &right_factor);
        let right_rounds = right_factor.len().trailing_zeros() as usize;
        let total_rounds = right_rounds + left_factor.len().trailing_zeros() as usize;
        Ok(Self {
            table,
            left_factor,
            right_factor,
            input_claim,
            right_rounds,
            total_rounds,
        })
    }

    /// Prove the batched recursive stage-3 sumcheck.
    ///
    /// This carries the stage-2 next-witness opening `W(stage2_point)` to a new
    /// point that is a prefix/projection of the same batched challenge vector used
    /// by the setup-product opening.
    #[allow(clippy::too_many_arguments)]
    pub fn prove<F, T, SampleRound, const D: usize>(
        expanded: &AkitaExpandedSetup<F>,
        prefix_slots: &SetupPrefixProverRegistry<F, D>,
        lp: &LevelParams,
        next_fold_level_params: &LevelParams,
        relation: &RingRelationInstance<F, D>,
        tau1: &[E],
        alpha: E,
        stage2_challenges: &[E],
        stage2_next_w_eval: E,
        logical_w: &[i8],
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
        eta: E,
        transcript: &mut T,
        sample_round: SampleRound,
    ) -> Result<BatchedStage3SumcheckProverOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + AkitaSerialize,
        T: Transcript<F>,
        SampleRound: FnMut(&mut T) -> E,
    {
        let setup_prover = build_setup_product_prover::<F, E, T, D>(
            expanded,
            prefix_slots,
            lp,
            next_fold_level_params,
            relation,
            tau1,
            alpha,
            &stage2_challenges[ring_bits..],
            transcript,
        )?;
        let setup_claim = setup_prover.input_claim();
        let witness_prover = build_witness_carry_prover::<E>(
            logical_w,
            live_x_cols,
            col_bits,
            ring_bits,
            stage2_challenges,
            stage2_next_w_eval,
        )?;
        let mut batched = BatchedStage3Prover::new(setup_prover, witness_prover, eta)?;
        let (sumcheck, batched_point, _final_claim) =
            <BatchedStage3Prover<E> as SumcheckInstanceProverExt<E>>::prove::<F, T, _>(
                &mut batched,
                transcript,
                sample_round,
            )?;
        let next_w_point = batched_point[..batched.witness.native_rounds].to_vec();
        let next_w_eval =
            evaluate_witness_at_point(logical_w, live_x_cols, col_bits, ring_bits, &next_w_point)?;
        Ok(BatchedStage3SumcheckProverOutput {
            setup_claim,
            next_w_eval,
            next_w_point,
            sumcheck,
        })
    }
}

fn evaluate_witness_at_point<E>(
    logical_w: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
    point: &[E],
) -> Result<E, AkitaError>
where
    E: FieldCore + FromPrimitiveInt,
{
    let num_vars = col_bits
        .checked_add(ring_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("witness eval variable count overflow".into()))?;
    if point.len() != num_vars {
        return Err(AkitaError::InvalidSize {
            expected: num_vars,
            actual: point.len(),
        });
    }
    let y_len = 1usize
        .checked_shl(u32::try_from(ring_bits).map_err(|_| AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    let x_len = 1usize
        .checked_shl(u32::try_from(col_bits).map_err(|_| AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    if live_x_cols > x_len {
        return Err(AkitaError::InvalidSize {
            expected: x_len,
            actual: live_x_cols,
        });
    }
    let live_len = live_x_cols
        .checked_mul(y_len)
        .ok_or_else(|| AkitaError::InvalidSetup("witness eval live length overflow".into()))?;
    if logical_w.len() != live_len {
        return Err(AkitaError::InvalidSize {
            expected: live_len,
            actual: logical_w.len(),
        });
    }
    let eq_y = EqPolynomial::evals(&point[..ring_bits])?;
    let eq_x = EqPolynomial::evals(&point[ring_bits..])?;
    let mut acc = E::zero();
    for (x, &x_weight) in eq_x.iter().take(live_x_cols).enumerate() {
        let base = x * y_len;
        let mut y_eval = E::zero();
        for (y, &y_weight) in eq_y.iter().enumerate() {
            y_eval += y_weight * E::from_i64(i64::from(logical_w[base + y]));
        }
        acc += x_weight * y_eval;
    }
    Ok(acc)
}

struct BatchedStage3Term<E: FieldCore> {
    prover: SetupSumcheckProver<E>,
    current_claim: E,
    native_rounds: usize,
}

struct PendingRound<E: FieldCore> {
    setup_poly: UniPoly<E>,
    witness_poly: UniPoly<E>,
}

struct BatchedStage3Prover<E: FieldCore> {
    setup: BatchedStage3Term<E>,
    witness: BatchedStage3Term<E>,
    eta: E,
    total_rounds: usize,
    pending_round: Option<PendingRound<E>>,
}

impl<E: FieldCore + FromPrimitiveInt> BatchedStage3Prover<E> {
    fn new(
        setup_prover: SetupSumcheckProver<E>,
        witness_prover: SetupSumcheckProver<E>,
        eta: E,
    ) -> Result<Self, AkitaError> {
        let setup_rounds = setup_prover.num_rounds();
        let witness_rounds = witness_prover.num_rounds();
        let total_rounds = setup_rounds.max(witness_rounds);
        Ok(Self {
            setup: BatchedStage3Term {
                current_claim: setup_prover.input_claim(),
                native_rounds: setup_rounds,
                prover: setup_prover,
            },
            witness: BatchedStage3Term {
                current_claim: witness_prover.input_claim(),
                native_rounds: witness_rounds,
                prover: witness_prover,
            },
            eta,
            total_rounds,
            pending_round: None,
        })
    }

    #[inline]
    fn term_round_poly(term: &mut BatchedStage3Term<E>, round: usize) -> UniPoly<E> {
        if round < term.native_rounds {
            term.prover
                .compute_round_univariate(round, term.current_claim)
        } else {
            // The term is independent of this padded variable. The normalized
            // common-cube lift contributes a constant half-claim polynomial.
            UniPoly::from_coeffs(vec![half(term.current_claim), E::zero(), E::zero()])
        }
    }

    #[inline]
    fn combine_polys(&self, setup_poly: &UniPoly<E>, witness_poly: &UniPoly<E>) -> UniPoly<E> {
        let len = setup_poly
            .coeffs
            .len()
            .max(witness_poly.coeffs.len())
            .max(3);
        let mut coeffs = vec![E::zero(); len];
        for (idx, coeff) in setup_poly.coeffs.iter().enumerate() {
            coeffs[idx] += *coeff;
        }
        for (idx, coeff) in witness_poly.coeffs.iter().enumerate() {
            coeffs[idx] += self.eta * *coeff;
        }
        UniPoly::from_coeffs(coeffs)
    }
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceProver<E> for BatchedStage3Prover<E> {
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    fn degree_bound(&self) -> usize {
        SETUP_SUMCHECK_DEGREE
    }

    fn input_claim(&self) -> E {
        self.setup.current_claim + self.eta * self.witness.current_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let setup_poly = Self::term_round_poly(&mut self.setup, round);
        let witness_poly = Self::term_round_poly(&mut self.witness, round);
        let combined = self.combine_polys(&setup_poly, &witness_poly);
        self.pending_round = Some(PendingRound {
            setup_poly,
            witness_poly,
        });
        combined
    }

    fn ingest_challenge(&mut self, round: usize, r_round: E) {
        let pending = self
            .pending_round
            .take()
            .expect("batched stage-3 challenge ingested before round polynomial");
        self.setup.current_claim = pending.setup_poly.evaluate(&r_round);
        self.witness.current_claim = pending.witness_poly.evaluate(&r_round);
        if round < self.setup.native_rounds {
            self.setup.prover.ingest_challenge(round, r_round);
        }
        if round < self.witness.native_rounds {
            self.witness.prover.ingest_challenge(round, r_round);
        }
    }
}

#[inline]
fn half<E: FieldCore + FromPrimitiveInt>(value: E) -> E {
    let inv_two = E::from_u64(2)
        .inverse()
        .expect("two must be invertible in Akita fields");
    value * inv_two
}

#[allow(clippy::too_many_arguments)]
fn build_setup_product_prover<F, E, T, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    lp: &LevelParams,
    next_fold_level_params: &LevelParams,
    relation: &RingRelationInstance<F, D>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
    transcript: &mut T,
) -> Result<SetupSumcheckProver<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let (required, mut bar_omega, alpha_pows) =
        prepare_setup_sumcheck_terms::<F, E, D>(lp, relation, tau1, alpha, x_challenges)?;

    let natural_field_len = required.checked_mul(D).ok_or_else(|| {
        AkitaError::InvalidSetup("setup product natural field length overflow".to_string())
    })?;
    let setup_len = expanded.shared_matrix().total_ring_elements_at::<D>()?;
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected setup product".to_string(),
        ));
    }
    let setup_eval_len = if D == SETUP_OFFLOAD_D_SETUP {
        let setup_prefix_selection = select_setup_prefix_slot(
            expanded.seed(),
            setup_len,
            |slot_id| {
                prefix_slots
                    .get(slot_id)
                    .map(|slot| (slot, slot.natural_len, slot.padded_len))
            },
            next_fold_level_params,
            natural_field_len,
            D,
            "selected setup-prefix slot does not cover setup product",
        )?;
        if let Some((slot, setup_eval_len)) = setup_prefix_selection {
            transcript.append_serde(ABSORB_SETUP_PREFIX_SLOT, &slot.id);
            setup_eval_len
        } else {
            setup_len
        }
    } else {
        setup_len
    };
    let setup_view = expanded.shared_matrix().ring_view::<D>(1, setup_eval_len)?;
    let setup_entries = setup_view.as_slice();

    let lambda_len = required
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("setup product lambda length overflow".into()))?;
    bar_omega.resize(lambda_len, E::zero());

    let table_len = lambda_len
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("setup product table length overflow".into()))?;
    let mut setup_table = vec![E::zero(); table_len];
    cfg_chunks_mut!(&mut setup_table, D)
        .enumerate()
        .for_each(|(lambda, row)| {
            if lambda < required {
                for (slot, &coeff) in row.iter_mut().zip(setup_entries[lambda].coefficients()) {
                    *slot = E::lift_base(coeff);
                }
            }
        });

    SetupSumcheckProver::new(setup_table, bar_omega, alpha_pows.to_vec())
}

fn build_witness_carry_prover<E>(
    logical_w: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
    stage2_challenges: &[E],
    stage2_next_w_eval: E,
) -> Result<SetupSumcheckProver<E>, AkitaError>
where
    E: FieldCore + FromPrimitiveInt,
{
    let num_vars = col_bits
        .checked_add(ring_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("witness carry variable count overflow".into()))?;
    if stage2_challenges.len() != num_vars {
        return Err(AkitaError::InvalidSize {
            expected: num_vars,
            actual: stage2_challenges.len(),
        });
    }
    let y_len = 1usize
        .checked_shl(u32::try_from(ring_bits).map_err(|_| AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    let x_len = 1usize
        .checked_shl(u32::try_from(col_bits).map_err(|_| AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    if live_x_cols > x_len {
        return Err(AkitaError::InvalidSize {
            expected: x_len,
            actual: live_x_cols,
        });
    }
    let live_len = live_x_cols
        .checked_mul(y_len)
        .ok_or_else(|| AkitaError::InvalidSetup("witness carry live length overflow".into()))?;
    if logical_w.len() != live_len {
        return Err(AkitaError::InvalidSize {
            expected: live_len,
            actual: logical_w.len(),
        });
    }
    let table_len = x_len
        .checked_mul(y_len)
        .ok_or_else(|| AkitaError::InvalidSetup("witness carry table length overflow".into()))?;
    let mut table = vec![E::zero(); table_len];
    for (dst, &digit) in table.iter_mut().zip(logical_w) {
        *dst = E::from_i64(i64::from(digit));
    }
    let right_factor = EqPolynomial::evals(&stage2_challenges[..ring_bits])?;
    let left_factor = EqPolynomial::evals(&stage2_challenges[ring_bits..])?;
    let prover = SetupSumcheckProver::new(table, left_factor, right_factor)?;
    if prover.input_claim() != stage2_next_w_eval {
        return Err(AkitaError::InvalidProof);
    }
    Ok(prover)
}

impl<E: FieldCore> SumcheckInstanceProver<E> for SetupSumcheckProver<E> {
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    fn degree_bound(&self) -> usize {
        SETUP_SUMCHECK_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let (constant, linear, quadratic) = if round < self.right_rounds {
            accumulate_right_round(&self.table, &self.left_factor, &self.right_factor)
        } else {
            accumulate_left_round(&self.table, &self.left_factor, self.right_factor[0])
        };
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, round: usize, r_round: E) {
        if round < self.right_rounds {
            fold_right_round(&mut self.table, &mut self.right_factor, r_round);
        } else {
            fold_left_round(&mut self.table, &mut self.left_factor, r_round);
        }
    }
}

/// Derive the factored product-sumcheck terms `(required, bar_omega, alpha_pows)`
/// from the level parameters and ring relation via the ring-switch row
/// evaluation.
fn prepare_setup_sumcheck_terms<F, E, const D: usize>(
    lp: &LevelParams,
    relation: &RingRelationInstance<F, D>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
) -> Result<(usize, Vec<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F>,
{
    let alpha_pows = scalar_powers(alpha, D);
    let inputs = create_setup_contribution_inputs::<F, E, D>(relation, lp, tau1)?;
    let num_t_vectors = relation.opening_batch().num_polynomials();
    let fold_gadget = gadget_row_scalars::<F>(
        lp.num_digits_fold(num_t_vectors, F::modulus_bits())?,
        lp.log_basis,
    );
    let layout = relation.segment_layout(lp)?;
    let plan = SetupContributionPlan::prepare(
        &inputs,
        x_challenges,
        None,
        None,
        &fold_gadget,
        layout.offset_e,
        layout.offset_t,
        layout.offset_z,
        layout.offset_u,
    )?;
    let required = plan.required();
    let bar_omega = plan.materialize_bar_omega();
    Ok((required, bar_omega, alpha_pows.to_vec()))
}

/// Build the setup-contribution artifact from prover-owned relation data.
fn create_setup_contribution_inputs<F, E, const D: usize>(
    relation: &RingRelationInstance<F, D>,
    lp: &LevelParams,
    tau1: &[E],
) -> Result<SetupContributionPlanInputs<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
{
    let opening_batch = relation.opening_batch();
    let num_claims = relation.opening_batch().num_claims();
    let num_polys = opening_batch.num_polynomials();

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;
    if lp.num_blocks == 0 || !lp.num_blocks.is_power_of_two() {
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

    let num_t_vectors = num_polys;
    let inner_width = lp
        .block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
    if lp.a_key.col_len() < inner_width {
        return Err(AkitaError::InvalidSetup(
            "A-key column width is too small for setup contribution layout".to_string(),
        ));
    }
    let expected_b_width = num_polys
        .checked_mul(lp.a_key.row_len())
        .and_then(|width| width.checked_mul(depth_open))
        .and_then(|width| width.checked_mul(lp.num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".to_string()))?;
    // Tiered: the stored first-tier `B'` is the full B width divided by the
    // reuse factor `tier_split` (mirrors the verifier-side check in
    // `akita-verifier`'s `prepare_ring_switch_row_eval_inner`).
    let expected_stored_b_width = if lp.f_key.is_some() {
        expected_b_width.div_ceil(lp.tier_split.max(1))
    } else {
        expected_b_width
    };
    if lp.b_key.col_len() < expected_stored_b_width {
        return Err(AkitaError::InvalidSetup(
            "B-key column width is too small for setup contribution layout".to_string(),
        ));
    }

    let m_row_layout = relation.m_row_layout();
    // Public-output M rows are enforced by the fused trace term, not M itself.
    let num_public_m_rows = 0usize;
    let rows = lp.m_row_count_for(1, num_public_m_rows, m_row_layout)?;
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    Ok(SetupContributionPlanInputs {
        eq_tau1,
        num_t_vectors,
        num_blocks: lp.num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        block_len: lp.block_len,
        inner_width,
        n_a: lp.a_key.row_len(),
        n_d: lp.d_key.row_len(),
        m_row_layout,
        n_b: lp.b_key.row_len(),
        num_segments: 1,
        rows,
        num_polys_per_segment: vec![num_polys],
        num_public_rows: num_public_m_rows,
        // Stage-3 (recursive setup-contribution mode) tiered support is a
        // follow-up; the default Direct verifier path uses `eval_at_point`.
        tier_split: lp.tier_split,
        n_f: lp.f_key.as_ref().map_or(0, |fk| fk.row_len()),
    })
}

fn product_claim<E: FieldCore>(table: &[E], left_factor: &[E], right_factor: &[E]) -> E {
    let right_len = right_factor.len();
    cfg_fold_reduce!(
        0..left_factor.len(),
        E::zero,
        |mut acc, left_idx| {
            let left_weight = left_factor[left_idx];
            let row = &table[left_idx * right_len..(left_idx + 1) * right_len];
            for (&value, &right_weight) in row.iter().zip(right_factor.iter()) {
                acc += value * left_weight * right_weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

fn accumulate_right_round<E: FieldCore>(
    table: &[E],
    left_factor: &[E],
    right_factor: &[E],
) -> (E, E, E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    cfg_fold_reduce!(
        0..left_factor.len(),
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), left_idx| {
            let left_weight = left_factor[left_idx];
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                let s0 = table[row_base + 2 * pair_idx];
                let s1 = table[row_base + 2 * pair_idx + 1];
                let f0 = left_weight * right_factor[2 * pair_idx];
                let f1 = left_weight * right_factor[2 * pair_idx + 1];
                let ds = s1 - s0;
                let df = f1 - f0;
                constant += s0 * f0;
                linear += s0 * df + ds * f0;
                quadratic += ds * df;
            }
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

fn accumulate_left_round<E: FieldCore>(
    table: &[E],
    left_factor: &[E],
    right_weight: E,
) -> (E, E, E) {
    let half = left_factor.len() / 2;
    cfg_fold_reduce!(
        0..half,
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), pair_idx| {
            let s0 = table[2 * pair_idx];
            let s1 = table[2 * pair_idx + 1];
            let f0 = left_factor[2 * pair_idx] * right_weight;
            let f1 = left_factor[2 * pair_idx + 1] * right_weight;
            let ds = s1 - s0;
            let df = f1 - f0;
            constant += s0 * f0;
            linear += s0 * df + ds * f0;
            quadratic += ds * df;
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

fn fold_pair<E: FieldCore>(left: E, right: E, r: E) -> E {
    left + r * (right - left)
}

fn fold_right_round<E: FieldCore>(table: &mut Vec<E>, right_factor: &mut Vec<E>, r: E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    let left_len = table.len() / right_len;
    let mut folded = vec![E::zero(); left_len * half];
    cfg_chunks_mut!(&mut folded, half)
        .enumerate()
        .for_each(|(left_idx, row)| {
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                row[pair_idx] = fold_pair(
                    table[row_base + 2 * pair_idx],
                    table[row_base + 2 * pair_idx + 1],
                    r,
                );
            }
        });
    let folded_right = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(right_factor[2 * idx], right_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *right_factor = folded_right;
    *table = folded;
}

fn fold_left_round<E: FieldCore>(table: &mut Vec<E>, left_factor: &mut Vec<E>, r: E) {
    let half = left_factor.len() / 2;
    let folded_table = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(table[2 * idx], table[2 * idx + 1], r))
        .collect::<Vec<_>>();
    let folded_left = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(left_factor[2 * idx], left_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *table = folded_table;
    *left_factor = folded_left;
}
