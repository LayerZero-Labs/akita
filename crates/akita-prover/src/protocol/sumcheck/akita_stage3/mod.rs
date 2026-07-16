//! Setup-product sumcheck for a dense table against two disjoint factors.
//!
//! The table is laid out as `left * fold_low_len + right`. The right factor is
//! bound first, then the left factor. This matches setup products of the form
//! `S(i, y) * setup_index_weight(i) * alpha(y)` without materializing the full
//! `setup_index_weight(i) * alpha(y)` table.

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
    ensure_setup_envelope, prepare_setup_contribution_artifact, select_setup_prefix_slot,
    shared_setup_fold_gadget, AkitaExpandedSetup, BatchedStage3Geometry, FpExtEncoding,
    LevelParams, RingRelationInstance, SetupContributionPlan, SetupPrefixProverRegistry,
    SetupProjectionGeometry, SETUP_OFFLOAD_D_SETUP, SETUP_SUMCHECK_DEGREE,
};
use product_table::FactoredProductTerm;
use std::sync::Arc;

/// Output of the batched stage-3 prover.
pub struct AkitaStage3ProverOutput<E: FieldCore> {
    /// Unbatched setup-product claim carried in the serialized stage-3 proof.
    pub setup_product_claim: E,
    /// Setup-prefix MLE value at the stage-3 setup suffix challenge.
    pub setup_prefix_eval: E,
    /// Setup-prefix opening point at the setup suffix of the stage-3 challenge.
    pub setup_prefix_point: Vec<E>,
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
    geometry: BatchedStage3Geometry,
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
        level: usize,
        eta: E,
        transcript: &mut T,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + LiftBase<F> + AkitaSerialize,
        T: Transcript<F>,
    {
        let setup_coefficient_bits = lp.d_a().trailing_zeros() as usize;
        let setup_x_challenges = stage2_challenges
            .get(setup_coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let setup_term = build_setup_product_term::<F, E, T>(
            expanded,
            prefix_slots,
            lp,
            next_fold_level_params,
            relation,
            tau1,
            alpha,
            setup_x_challenges,
            transcript,
        )?;
        let setup_product_claim = setup_term.input_claim();
        let witness_digits = Arc::<[i8]>::from(logical_w);
        if !witness_digits
            .len()
            .is_multiple_of(next_fold_level_params.d_a())
        {
            return Err(AkitaError::InvalidProof);
        }
        let opening_source_len = witness_digits.len() / next_fold_level_params.d_a();
        let witness_term = build_witness_carry_term::<E>(
            Arc::clone(&witness_digits),
            opening_source_len,
            next_fold_level_params.d_a(),
            live_x_cols,
            col_bits,
            ring_bits,
            level,
            stage2_challenges,
            stage2_next_w_eval,
        )?;
        let setup_rounds = setup_term.num_rounds();
        let witness_rounds = witness_term.num_rounds();
        let geometry = BatchedStage3Geometry::new(witness_rounds, setup_rounds)?;
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
            geometry,
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
        let next_w_point = self.geometry.witness_point(&batched_point)?;
        let setup_prefix_point = self.geometry.setup_point(&batched_point)?;
        let setup_prefix_eval = self.setup.term.folded_table_value()?;
        let next_w_eval = self.witness.term.folded_table_value()?;
        Ok(AkitaStage3ProverOutput {
            setup_product_claim: self.setup_product_claim,
            setup_prefix_eval,
            setup_prefix_point,
            next_w_eval,
            next_w_point,
            sumcheck,
        })
    }

    #[inline]
    fn term_round_poly(
        term: &mut BatchedStage3Term<E>,
        total_rounds: usize,
        round: usize,
    ) -> UniPoly<E> {
        let inactive_rounds = total_rounds - term.native_rounds;
        if round < inactive_rounds {
            // The term is independent of this leading padded variable. Active
            // low-order coordinates are the suffix of the batched challenge.
            UniPoly::from_coeffs(vec![half(term.current_claim), E::zero(), E::zero()])
        } else {
            let mut poly = term
                .term
                .compute_round_univariate(round - inactive_rounds, term.current_claim);
            let scale = (0..inactive_rounds).fold(E::one(), |acc, _| acc * half(E::one()));
            for coeff in &mut poly.coeffs {
                *coeff *= scale;
            }
            poly
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
        self.geometry.batched_rounds()
    }

    fn degree_bound(&self) -> usize {
        SETUP_SUMCHECK_DEGREE
    }

    fn input_claim(&self) -> E {
        self.setup.current_claim + self.eta * self.witness.current_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let total_rounds = self.geometry.batched_rounds();
        let setup_poly = Self::term_round_poly(&mut self.setup, total_rounds, round);
        let witness_poly = Self::term_round_poly(&mut self.witness, total_rounds, round);
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
        let total_rounds = self.geometry.batched_rounds();
        let setup_inactive_rounds = total_rounds - self.setup.native_rounds;
        if round >= setup_inactive_rounds {
            self.setup
                .term
                .ingest_challenge(round - setup_inactive_rounds, r_round);
        }
        let witness_inactive_rounds = total_rounds - self.witness.native_rounds;
        if round >= witness_inactive_rounds {
            self.witness
                .term
                .ingest_challenge(round - witness_inactive_rounds, r_round);
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
    let (geometry, mut setup_index_weight, alpha_pows) =
        prepare_setup_sumcheck_terms::<F, E>(lp, relation, tau1, alpha, x_challenges)?;

    let required = geometry.required();
    let ring_d = geometry.base_ring_dim();
    ensure_setup_envelope(expanded, required, ring_d)?;
    let natural_field_len = geometry.natural_field_len();
    let setup_len = expanded
        .shared_matrix()
        .total_ring_elements_at_dyn(ring_d)?;
    let setup_eval_len = if ring_d == SETUP_OFFLOAD_D_SETUP {
        let setup_prefix_selection = select_setup_prefix_slot(
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
        } else if next_fold_level_params.setup_prefix.is_some() {
            return Err(AkitaError::InvalidSetup(
                "planned setup-prefix slot is missing from prover setup".to_string(),
            ));
        } else {
            setup_len
        }
    } else {
        setup_len
    };
    // Ring elements at `ring_d` are `ring_d` consecutive field coefficients of
    // the flat shared matrix; read them directly instead of building a typed
    // ring view that would immediately be flattened back into the table. A
    // setup-prefix slot may have a padded evaluation domain larger than this
    // source: only `required` natural rows are read, and the table remainder is
    // explicit zero padding.
    let setup_field = expanded.shared_matrix().as_field_slice();
    if required > setup_eval_len {
        return Err(AkitaError::InvalidSetup(
            "setup product exceeds selected setup view".to_string(),
        ));
    }

    let setup_idx_len = required
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("setup product index length overflow".into()))?;
    setup_index_weight.resize(setup_idx_len, E::zero());

    let table_len = setup_idx_len
        .checked_mul(ring_d)
        .ok_or_else(|| AkitaError::InvalidSetup("setup product table length overflow".into()))?;
    let mut setup_table = vec![E::zero(); table_len];
    cfg_chunks_mut!(&mut setup_table, ring_d)
        .enumerate()
        .for_each(|(setup_idx, row)| {
            if setup_idx < required {
                let src = &setup_field[setup_idx * ring_d..(setup_idx + 1) * ring_d];
                for (slot, &coeff) in row.iter_mut().zip(src) {
                    *slot = E::lift_base(coeff);
                }
            }
        });

    FactoredProductTerm::new_dense(setup_table, setup_index_weight, alpha_pows.to_vec())
}

#[allow(clippy::too_many_arguments)]
fn build_witness_carry_term<E>(
    logical_w: Arc<[i8]>,
    opening_source_len: usize,
    opening_ring_dim: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
    level: usize,
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
    let physical_capacity = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or(AkitaError::InvalidProof)?;
    if opening_source_len == 0
        || opening_ring_dim == 0
        || !logical_w.len().is_multiple_of(opening_ring_dim)
        || logical_w.len() > physical_capacity
    {
        return Err(AkitaError::InvalidProof);
    }
    let expected_live_x_cols = if ring_bits == 0 {
        logical_w.len()
    } else {
        logical_w.len() / opening_ring_dim
    };
    if live_x_cols != expected_live_x_cols || live_x_cols > x_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_live_x_cols,
            actual: live_x_cols,
        });
    }
    let table_len = x_len
        .checked_mul(y_len)
        .ok_or_else(|| AkitaError::InvalidSetup("witness carry table length overflow".into()))?;
    // The (x, y) split must tile the full Boolean field domain
    // `opening_domain_len(source) * opening_ring_dim`. The flattened fallback
    // (`ring_bits == 0`) keeps the whole domain in `x`; the uniform layout puts
    // the `opening_ring_dim` inner coefficients in `y`.
    let expected_field_len = akita_types::opening_domain_len(opening_source_len)?
        .checked_mul(opening_ring_dim)
        .ok_or(AkitaError::InvalidProof)?;
    if table_len != expected_field_len || (ring_bits != 0 && y_len != opening_ring_dim) {
        return Err(AkitaError::InvalidSize {
            expected: expected_field_len,
            actual: table_len,
        });
    }
    let right_factor = EqPolynomial::evals(&stage2_challenges[..ring_bits]).map_err(|err| {
        AkitaError::InvalidInput(format!(
            "stage-3 witness carry right equality factor failed at fold level {level}: \
             ring_bits={ring_bits}, col_bits={col_bits}, live_x_cols={live_x_cols}: {err}"
        ))
    })?;
    let mut opening_table = vec![0i8; table_len];
    let live_physical_cols = logical_w.len() / opening_ring_dim;
    for physical_index in 0..live_physical_cols {
        let opening_index =
            akita_types::checked_opening_source_index(opening_source_len, physical_index)?;
        let src_start = physical_index * opening_ring_dim;
        let dst_start = opening_index * opening_ring_dim;
        opening_table[dst_start..dst_start + opening_ring_dim]
            .copy_from_slice(&logical_w[src_start..src_start + opening_ring_dim]);
    }
    let term = if ring_bits == 0 {
        FactoredProductTerm::new_compact_equality(
            Arc::from(opening_table),
            table_len,
            Arc::from(stage2_challenges[ring_bits..].to_vec()),
            right_factor,
        )?
    } else {
        let left_factor = EqPolynomial::evals(&stage2_challenges[ring_bits..]).map_err(|err| {
            AkitaError::InvalidInput(format!(
                "stage-3 witness carry left equality factor failed at fold level {level}: \
                 col_bits={col_bits}, ring_bits={ring_bits}, live_x_cols={live_x_cols}: {err}"
            ))
        })?;
        FactoredProductTerm::new_compact(
            Arc::from(opening_table),
            table_len,
            left_factor,
            right_factor,
        )?
    };
    if term.input_claim() != stage2_next_w_eval {
        return Err(AkitaError::InvalidProof);
    }
    Ok(term)
}

/// Derive the factored product-sumcheck terms `(required, setup_index_weight, alpha_pows)`
/// from the level parameters and ring relation via the ring-switch row
/// evaluation.
fn prepare_setup_sumcheck_terms<F, E>(
    lp: &LevelParams,
    relation: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
) -> Result<(SetupProjectionGeometry, Vec<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F>,
{
    let setup_artifact =
        prepare_setup_contribution_artifact::<F, E>(relation, lp, tau1, None, None)?;
    let fold_gadget = shared_setup_fold_gadget::<F>(setup_artifact.layout.groups());
    let plan = SetupContributionPlan::finish_plan::<F>(
        &setup_artifact.static_plan,
        x_challenges,
        None,
        None,
        fold_gadget.as_deref(),
        &setup_artifact.layout,
        relation.role_dims(),
    )?;
    let geometry = plan.projection_geometry();
    let alpha_pows = scalar_powers(alpha, geometry.alpha_power_len());
    let setup_index_weight = plan.materialize_setup_index_weights(alpha)?;
    Ok((geometry, setup_index_weight, alpha_pows.to_vec()))
}
