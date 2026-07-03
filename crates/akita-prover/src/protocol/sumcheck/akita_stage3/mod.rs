//! Setup-product sumcheck for a dense table against two disjoint factors.
//!
//! The table is laid out as `left * right_len + right`. The right factor is
//! bound first, then the left factor. This matches setup products of the form
//! `S(lambda, y) * omega(lambda) * alpha(y)` without materializing the full
//! `omega(lambda) * alpha(y)` table.

mod product_table;
mod utils;

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
use product_table::FactoredProductTerm;
use std::sync::Arc;

/// Output of the batched stage-3 prover.
pub struct AkitaStage3ProverOutput<E: FieldCore> {
    /// Unbatched setup-product claim carried in the serialized stage-3 proof.
    pub setup_product_claim: E,
    /// Re-randomized next-witness opening after the batched stage-3 point projection.
    pub next_w_eval: E,
    /// Batched next-witness opening point.
    pub next_w_point: Vec<E>,
    /// Degree-two batched setup-product + carried-witness sumcheck.
    pub sumcheck: SumcheckProof<E>,
}

struct BatchedStage3Term<E: FieldCore> {
    term: FactoredProductTerm<E>,
    current_claim: E,
    native_rounds: usize,
}

struct PendingRound<E: FieldCore> {
    setup_poly: UniPoly<E>,
    witness_poly: UniPoly<E>,
}

/// Batched Stage-3 setup-product + carried-witness sumcheck prover.
pub struct AkitaStage3Prover<E: FieldCore> {
    setup: BatchedStage3Term<E>,
    witness: BatchedStage3Term<E>,
    eta: E,
    total_rounds: usize,
    setup_product_claim: E,
    pending_round: Option<PendingRound<E>>,
}

impl<E: FieldCore + FromPrimitiveInt> AkitaStage3Prover<E> {
    /// Construct a batched recursive stage-3 sumcheck prover.
    ///
    /// This carries the stage-2 next-witness opening `W(stage2_point)` to a new
    /// point that is a prefix/projection of the same batched challenge vector used
    /// by the setup-product opening.
    #[allow(clippy::too_many_arguments)]
    pub fn new<F, T>(
        expanded: &AkitaExpandedSetup<F>,
        prefix_slots: &SetupPrefixProverRegistry<F>,
        lp: &LevelParams,
        next_fold_level_params: &LevelParams,
        relation: &RingRelationInstance<F>,
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
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + LiftBase<F> + AkitaSerialize,
        T: Transcript<F>,
    {
        let ring_d = relation.role_dims().d_a();
        let setup_term = build_setup_product_term::<F, E, T>(
            ring_d,
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
        let setup_product_claim = setup_term.input_claim();
        let witness_digits = Arc::<[i8]>::from(logical_w);
        let witness_term = build_witness_carry_term::<E>(
            Arc::clone(&witness_digits),
            live_x_cols,
            col_bits,
            ring_bits,
            stage2_challenges,
            stage2_next_w_eval,
        )?;
        let setup_rounds = setup_term.num_rounds();
        let witness_rounds = witness_term.num_rounds();
        let total_rounds = setup_rounds.max(witness_rounds);
        Ok(Self {
            setup: BatchedStage3Term {
                current_claim: setup_term.input_claim(),
                native_rounds: setup_rounds,
                term: setup_term,
            },
            witness: BatchedStage3Term {
                current_claim: witness_term.input_claim(),
                native_rounds: witness_rounds,
                term: witness_term,
            },
            eta,
            total_rounds,
            setup_product_claim,
            pending_round: None,
        })
    }

    pub fn prove<F, T, SampleRound>(
        &mut self,
        transcript: &mut T,
        sample_round: SampleRound,
    ) -> Result<AkitaStage3ProverOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: AkitaSerialize,
        T: Transcript<F>,
        SampleRound: FnMut(&mut T) -> E,
    {
        let (sumcheck, batched_point, _final_claim) =
            <Self as SumcheckInstanceProverExt<E>>::prove::<F, T, _>(
                self,
                transcript,
                sample_round,
            )?;
        let next_w_point = batched_point[..self.witness.native_rounds].to_vec();
        let next_w_eval = self.witness.term.folded_table_value()?;
        Ok(AkitaStage3ProverOutput {
            setup_product_claim: self.setup_product_claim,
            next_w_eval,
            next_w_point,
            sumcheck,
        })
    }

    #[inline]
    fn term_round_poly(term: &mut BatchedStage3Term<E>, round: usize) -> UniPoly<E> {
        if round < term.native_rounds {
            term.term
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

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceProver<E> for AkitaStage3Prover<E> {
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
        let pending: PendingRound<E> = self
            .pending_round
            .take()
            .expect("batched stage-3 challenge ingested before round polynomial");
        self.setup.current_claim = pending.setup_poly.evaluate(&r_round);
        self.witness.current_claim = pending.witness_poly.evaluate(&r_round);
        if round < self.setup.native_rounds {
            self.setup.term.ingest_challenge(round, r_round);
        }
        if round < self.witness.native_rounds {
            self.witness.term.ingest_challenge(round, r_round);
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
fn build_setup_product_term<F, E, T>(
    ring_d: usize,
    expanded: &AkitaExpandedSetup<F>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    lp: &LevelParams,
    next_fold_level_params: &LevelParams,
    relation: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
    transcript: &mut T,
) -> Result<FactoredProductTerm<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let (required, mut bar_omega, alpha_pows) =
        prepare_setup_sumcheck_terms::<F, E>(ring_d, lp, relation, tau1, alpha, x_challenges)?;

    let natural_field_len = required.checked_mul(ring_d).ok_or_else(|| {
        AkitaError::InvalidSetup("setup product natural field length overflow".to_string())
    })?;
    let setup_len = expanded
        .shared_matrix()
        .total_ring_elements_at_dyn(ring_d)?;
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected setup product".to_string(),
        ));
    }
    let setup_eval_len = if ring_d == SETUP_OFFLOAD_D_SETUP {
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
            ring_d,
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
    // Ring elements at `ring_d` are `ring_d` consecutive field coefficients of
    // the flat shared matrix; read them directly instead of building a typed
    // ring view that would immediately be flattened back into the table. The
    // former view carried the bounds the table fill relies on; assert them
    // explicitly here.
    let setup_field = expanded.shared_matrix().as_field_slice();
    let setup_eval_field_len = setup_eval_len.checked_mul(ring_d).ok_or_else(|| {
        AkitaError::InvalidSetup("setup product view field length overflow".to_string())
    })?;
    if setup_eval_field_len > setup_field.len() || required > setup_eval_len {
        return Err(AkitaError::InvalidSetup(
            "setup product exceeds selected setup view".to_string(),
        ));
    }

    let lambda_len = required
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("setup product lambda length overflow".into()))?;
    bar_omega.resize(lambda_len, E::zero());

    let table_len = lambda_len
        .checked_mul(ring_d)
        .ok_or_else(|| AkitaError::InvalidSetup("setup product table length overflow".into()))?;
    let mut setup_table = vec![E::zero(); table_len];
    cfg_chunks_mut!(&mut setup_table, ring_d)
        .enumerate()
        .for_each(|(lambda, row)| {
            if lambda < required {
                let src = &setup_field[lambda * ring_d..(lambda + 1) * ring_d];
                for (slot, &coeff) in row.iter_mut().zip(src) {
                    *slot = E::lift_base(coeff);
                }
            }
        });

    FactoredProductTerm::new_dense(setup_table, bar_omega, alpha_pows.to_vec())
}

fn build_witness_carry_term<E>(
    logical_w: Arc<[i8]>,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
    stage2_challenges: &[E],
    stage2_next_w_eval: E,
) -> Result<FactoredProductTerm<E>, AkitaError>
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
    let right_factor = EqPolynomial::evals(&stage2_challenges[..ring_bits])?;
    let left_factor = EqPolynomial::evals(&stage2_challenges[ring_bits..])?;
    let term = FactoredProductTerm::new_compact(logical_w, table_len, left_factor, right_factor)?;
    if term.input_claim() != stage2_next_w_eval {
        return Err(AkitaError::InvalidProof);
    }
    Ok(term)
}

/// Derive the factored product-sumcheck terms `(required, bar_omega, alpha_pows)`
/// from the level parameters and ring relation via the ring-switch row
/// evaluation.
fn prepare_setup_sumcheck_terms<F, E>(
    ring_d: usize,
    lp: &LevelParams,
    relation: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
) -> Result<(usize, Vec<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F>,
{
    let alpha_pows = scalar_powers(alpha, ring_d);
    let inputs = create_setup_contribution_inputs::<F, E>(relation, lp, tau1)?;
    let num_t_vectors = relation.opening_batch().num_total_polynomials();
    let fold_gadget = gadget_row_scalars::<F>(
        lp.num_digits_fold(num_t_vectors, F::modulus_bits())?,
        lp.log_basis,
    );
    let layout = relation.segment_layout(lp, None)?;
    let plan =
        SetupContributionPlan::prepare(&inputs, x_challenges, None, None, &fold_gadget, &layout)?;
    let required = plan.required();
    let bar_omega = plan.materialize_bar_omega();
    Ok((required, bar_omega, alpha_pows.to_vec()))
}

/// Build the setup-contribution artifact from prover-owned relation data.
fn create_setup_contribution_inputs<F, E>(
    relation: &RingRelationInstance<F>,
    lp: &LevelParams,
    tau1: &[E],
) -> Result<SetupContributionPlanInputs<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
{
    let opening_batch = relation.opening_batch();
    let num_polynomials = opening_batch.num_total_polynomials();

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold(num_polynomials, F::modulus_bits())?;
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

    let inner_width = lp
        .block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
    if lp.a_key.col_len() < inner_width {
        return Err(AkitaError::InvalidSetup(
            "A-key column width is too small for setup contribution layout".to_string(),
        ));
    }
    let expected_b_width = num_polynomials
        .checked_mul(lp.a_key.row_len())
        .and_then(|width| width.checked_mul(depth_open))
        .and_then(|width| width.checked_mul(lp.num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".to_string()))?;
    if lp.b_key.col_len() < expected_b_width {
        return Err(AkitaError::InvalidSetup(
            "B-key column width is too small for setup contribution layout".to_string(),
        ));
    }

    let m_row_layout = relation.m_row_layout();
    let rows = lp.m_row_count_for(1, m_row_layout)?;
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    Ok(SetupContributionPlanInputs {
        eq_tau1,
        num_t_vectors: num_polynomials,
        num_blocks: lp.num_blocks,
        num_claims: num_polynomials,
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
        num_polys_per_segment: vec![num_polynomials],
    })
}
