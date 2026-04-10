//! Stage-1 range-check tree prover/verifier for the Hachi PCS.
//!
//! For `b <= 8`, stage 1 is still a single eq-factored sumcheck over
//! `Q(S(z))`, where `S(z) = w(z)(w(z)+1)` and `Q` is the full range polynomial.
//! For larger supported bases, stage 1 is written as a short root-to-leaf tree:
//!
//! - a root stage proves the product of `2` or `4` quartic leaf factors,
//! - the prover sends those child-node claims at the sampled root point,
//! - a leaf stage proves a random linear combination of the quartic factors
//!   directly from `S`.
//!
//! This matches the proof-size study's current tree cutover for `log_basis <= 6`
//! without widening the recursive witness encoding beyond the existing runtime
//! bound.

use super::hachi_stage1 as single_stage_backend;
use super::{
    fold_evals_in_place, prove_eq_factored_sumcheck, verify_eq_factored_sumcheck,
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier, EqFactoredUniPoly,
};
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::split_eq::GruenSplitEq;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::proof::{HachiStage1Proof, HachiStage1StageProof, HachiStage1StageShape};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FromSmallInt};

fn compact_s_from_w(w: i8) -> i64 {
    let w = i64::from(w);
    w * (w + 1)
}

fn validate_stage1_tree_basis(b: usize) -> Result<(), HachiError> {
    if b < 4 || !b.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "stage1 tree requires a power-of-two basis >= 4, got {b}"
        )));
    }
    Ok(())
}

const MAX_TREE_STAGE_Q_DEGREE: usize = 4;

fn padded_s_table<E: FieldCore + FromSmallInt>(
    w_evals_compact: &[i8],
    live_x_cols: usize,
    num_u: usize,
    num_l: usize,
) -> Result<Vec<E>, HachiError> {
    let x_len = 1usize << num_u;
    let y_len = 1usize << num_l;
    let expected = live_x_cols * y_len;
    if w_evals_compact.len() != expected {
        return Err(HachiError::InvalidSize {
            expected,
            actual: w_evals_compact.len(),
        });
    }

    let mut out = vec![E::zero(); x_len * y_len];
    for x in 0..live_x_cols {
        let src_start = x * y_len;
        for y in 0..y_len {
            out[x * y_len + y] = E::from_i64(compact_s_from_w(w_evals_compact[src_start + y]));
        }
    }
    Ok(out)
}

fn stage1_root_values<E: FieldCore + FromSmallInt>(b: usize) -> Vec<E> {
    let half = b / 2;
    (0..half)
        .map(|k| {
            let k = k as i64;
            E::from_i64(k * (k + 1))
        })
        .collect()
}

fn poly_coeffs_from_roots<E: FieldCore>(roots: &[E]) -> Vec<E> {
    let mut coeffs = vec![E::one()];
    for &root in roots {
        let mut next = vec![E::zero(); coeffs.len() + 1];
        for (idx, &coeff) in coeffs.iter().enumerate() {
            next[idx] -= coeff * root;
            next[idx + 1] += coeff;
        }
        coeffs = next;
    }
    coeffs
}

fn eval_poly<E: FieldCore>(coeffs: &[E], x: E) -> E {
    coeffs
        .iter()
        .rev()
        .copied()
        .fold(E::zero(), |acc, coeff| acc * x + coeff)
}

fn compose_small_poly_with_affine<E: FieldCore>(coeffs: &[E], offset: E, slope: E) -> [E; 5] {
    debug_assert!(coeffs.len() <= MAX_TREE_STAGE_Q_DEGREE + 1);

    let mut out = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
    let mut power = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
    power[0] = E::one();

    for (idx, &coeff) in coeffs.iter().enumerate() {
        if idx > 0 {
            for k in (0..idx).rev() {
                power[k + 1] += power[k] * slope;
                power[k] = power[k] * offset;
            }
        }
        for k in 0..=idx {
            out[k] += coeff * power[k];
        }
    }

    out
}

fn stage1_leaf_groups<E: FieldCore + FromSmallInt>(b: usize) -> Vec<Vec<E>> {
    stage1_root_values::<E>(b)
        .chunks(4)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn stage1_leaf_coeffs<E: FieldCore + FromSmallInt>(b: usize) -> Vec<Vec<E>> {
    stage1_leaf_groups::<E>(b)
        .into_iter()
        .map(|roots| poly_coeffs_from_roots(&roots))
        .collect()
}

fn stage1_tree_binary_levels(b: usize) -> usize {
    debug_assert!(b >= 4 && b.is_power_of_two());
    b.trailing_zeros() as usize - 1
}

fn stage1_tree_stage_arities(b: usize) -> Vec<usize> {
    debug_assert!(b > 8 && b.is_power_of_two());
    let binary_levels = stage1_tree_binary_levels(b);
    let mut out = Vec::with_capacity(binary_levels.div_ceil(2));
    if binary_levels % 2 == 1 {
        out.push(2);
    }
    out.extend(std::iter::repeat_n(4, binary_levels / 2));
    out
}

fn stage1_tree_product_stage_arities(b: usize) -> Vec<usize> {
    let mut out = stage1_tree_stage_arities(b);
    out.pop();
    out
}

fn stage1_leaf_factor_count(b: usize) -> usize {
    debug_assert!(b >= 8 && b.is_power_of_two());
    b / 8
}

fn build_leaf_tables<E: FieldCore>(leaf_coeffs: &[Vec<E>], s_table: &[E]) -> Vec<Vec<E>> {
    cfg_iter!(leaf_coeffs)
        .map(|coeffs| {
            s_table
                .iter()
                .copied()
                .map(|s| eval_poly(coeffs, s))
                .collect()
        })
        .collect()
}

fn pointwise_product<E: FieldCore>(tables: &[Vec<E>]) -> Vec<E> {
    debug_assert!(!tables.is_empty());
    let len = tables[0].len();
    let mut out = vec![E::one(); len];
    for table in tables {
        debug_assert_eq!(table.len(), len);
        for (acc, value) in out.iter_mut().zip(table.iter()) {
            *acc = *acc * *value;
        }
    }
    out
}

struct ProductStageLayer<E: FieldCore> {
    child_tables_by_parent: Vec<Vec<Vec<E>>>,
}

fn build_product_stage_layers<E: FieldCore>(
    leaf_tables: Vec<Vec<E>>,
    product_stage_arities: &[usize],
) -> Vec<ProductStageLayer<E>> {
    let mut current_nodes = leaf_tables;
    let mut bottom_up_layers = Vec::with_capacity(product_stage_arities.len());

    for (rev_idx, &arity) in product_stage_arities.iter().rev().enumerate() {
        debug_assert!(matches!(arity, 2 | 4));
        debug_assert_eq!(current_nodes.len() % arity, 0);
        let needs_parent_nodes = rev_idx + 1 != product_stage_arities.len();

        let mut next_nodes =
            needs_parent_nodes.then(|| Vec::with_capacity(current_nodes.len() / arity));
        let mut child_tables_by_parent = Vec::with_capacity(current_nodes.len() / arity);
        let mut current_iter = current_nodes.into_iter();

        loop {
            let Some(first_child) = current_iter.next() else {
                break;
            };
            let mut child_tables = Vec::with_capacity(arity);
            child_tables.push(first_child);
            for _ in 1..arity {
                child_tables.push(
                    current_iter
                        .next()
                        .expect("product stage nodes should group evenly"),
                );
            }
            if let Some(next_nodes) = &mut next_nodes {
                next_nodes.push(pointwise_product(&child_tables));
            }
            child_tables_by_parent.push(child_tables);
        }

        current_nodes = next_nodes.unwrap_or_default();
        bottom_up_layers.push(ProductStageLayer {
            child_tables_by_parent,
        });
    }

    bottom_up_layers.reverse();
    bottom_up_layers
}

fn stage1_tree_stage_shape_from_b(rounds: usize, b: usize) -> Vec<HachiStage1StageShape> {
    debug_assert!(b >= 4 && b.is_power_of_two());
    if b <= 8 {
        return vec![HachiStage1StageShape {
            sumcheck: (rounds, b / 2),
            child_claims: 0,
        }];
    }

    let mut parent_count = 1usize;
    let mut out = Vec::new();
    for arity in stage1_tree_product_stage_arities(b) {
        let child_claims = parent_count * arity;
        out.push(HachiStage1StageShape {
            sumcheck: (rounds, arity),
            child_claims,
        });
        parent_count = child_claims;
    }
    debug_assert_eq!(parent_count, stage1_leaf_factor_count(b));
    out.push(HachiStage1StageShape {
        sumcheck: (rounds, 4),
        child_claims: 0,
    });
    out
}

pub(crate) fn stage1_tree_stage_shapes(rounds: usize, b: usize) -> Vec<HachiStage1StageShape> {
    stage1_tree_stage_shape_from_b(rounds, b)
}

fn stage1_stage_count(b: usize) -> usize {
    stage1_tree_stage_shape_from_b(0, b).len()
}

fn stage1_interstage_batch_weights<E: FieldCore>(gamma: E, count: usize) -> Vec<E> {
    let mut out = Vec::with_capacity(count);
    let mut weight = E::one();
    for _ in 0..count {
        out.push(weight);
        weight = weight * gamma;
    }
    out
}

fn combine_polys<E: FieldCore>(weights: &[E], polys: &[Vec<E>]) -> Vec<E> {
    debug_assert_eq!(weights.len(), polys.len());
    let max_len = polys.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = vec![E::zero(); max_len];
    for (weight, poly) in weights.iter().zip(polys.iter()) {
        for (idx, &coeff) in poly.iter().enumerate() {
            out[idx] += *weight * coeff;
        }
    }
    out
}

fn linear_combination<E: FieldCore>(weights: &[E], values: &[E]) -> E {
    debug_assert_eq!(weights.len(), values.len());
    weights
        .iter()
        .zip(values.iter())
        .fold(E::zero(), |acc, (&weight, &value)| acc + weight * value)
}

fn absorb_interstage_claims<F: FieldCore + CanonicalField, T: Transcript<F>>(
    claims: &[F],
    transcript: &mut T,
) {
    for claim in claims {
        transcript.append_field(labels::ABSORB_SUMCHECK_INTERSTAGE_CLAIM, claim);
    }
}

struct ProductStageProver<E: FieldCore> {
    child_tables_by_parent: Vec<Vec<Vec<E>>>,
    batch_weights: Vec<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ProductStageProver<E> {
    fn new(
        child_tables_by_parent: Vec<Vec<Vec<E>>>,
        batch_weights: Vec<E>,
        tau: &[E],
        input_claim: E,
    ) -> Self {
        debug_assert!(!child_tables_by_parent.is_empty());
        debug_assert_eq!(child_tables_by_parent.len(), batch_weights.len());
        let arity = child_tables_by_parent[0].len();
        debug_assert!(matches!(arity, 2 | 4));
        for child_tables in &child_tables_by_parent {
            debug_assert_eq!(child_tables.len(), arity);
        }
        Self {
            child_tables_by_parent,
            batch_weights,
            split_eq: GruenSplitEq::new(tau),
            input_claim,
            num_rounds: tau.len(),
        }
    }

    fn final_child_claims(&self) -> Vec<E> {
        self.child_tables_by_parent
            .iter()
            .flat_map(|child_tables| child_tables.iter())
            .map(|table| {
                debug_assert_eq!(table.len(), 1);
                table[0]
            })
            .collect()
    }
}

impl<E: FieldCore> EqFactoredSumcheckInstanceProver<E> for ProductStageProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.child_tables_by_parent[0].len()
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn current_linear_factor_evals(&self) -> (E, E) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<E> {
        debug_assert!(self.degree_bound() <= MAX_TREE_STAGE_Q_DEGREE);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let degree = self.degree_bound();
        let expected_pairs = num_first * e_second.len();
        debug_assert_eq!(
            self.child_tables_by_parent[0][0].len(),
            expected_pairs * 2,
            "product stage table length should match split-eq shape",
        );

        let q_coeffs = cfg_fold_reduce!(
            0..e_second.len(),
            || [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
            |mut outer, j_high| {
                let e_out = e_second[j_high];
                let base = j_high * num_first;
                let mut inner = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = base + j_low;
                    let mut batched_poly = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
                    for (parent_idx, child_tables) in self.child_tables_by_parent.iter().enumerate()
                    {
                        let mut poly = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
                        poly[0] = E::one();
                        for (current_degree, table) in child_tables.iter().enumerate() {
                            let left = table[2 * j];
                            let slope = table[2 * j + 1] - left;
                            for k in (0..=current_degree).rev() {
                                poly[k + 1] += poly[k] * slope;
                                poly[k] = poly[k] * left;
                            }
                        }
                        let weight = self.batch_weights[parent_idx];
                        for k in 0..=degree {
                            batched_poly[k] += weight * poly[k];
                        }
                    }
                    for k in 0..=degree {
                        inner[k] += e_in * batched_poly[k];
                    }
                }
                for k in 0..=degree {
                    outer[k] += e_out * inner[k];
                }
                outer
            },
            |mut a, b| {
                for k in 0..=degree {
                    a[k] += b[k];
                }
                a
            }
        );

        EqFactoredUniPoly::from_q_coeffs(q_coeffs[..=degree].to_vec())
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        self.split_eq.bind(r_round);
        for child_tables in &mut self.child_tables_by_parent {
            for table in child_tables {
                fold_evals_in_place(table, r_round);
            }
        }
    }
}

struct ProductStageVerifier<E: FieldCore> {
    tau: Vec<E>,
    input_claim: E,
    child_claims: Vec<E>,
    batch_weights: Vec<E>,
    arity: usize,
}

impl<E: FieldCore> ProductStageVerifier<E> {
    fn new(
        tau: Vec<E>,
        input_claim: E,
        child_claims: Vec<E>,
        batch_weights: Vec<E>,
        arity: usize,
    ) -> Self {
        debug_assert!(matches!(arity, 2 | 4));
        debug_assert_eq!(child_claims.len(), batch_weights.len() * arity);
        Self {
            tau,
            input_claim,
            child_claims,
            batch_weights,
            arity,
        }
    }
}

impl<E: FieldCore> EqFactoredSumcheckInstanceVerifier<E> for ProductStageVerifier<E> {
    type RoundState = GruenSplitEq<E>;

    fn num_rounds(&self) -> usize {
        self.tau.len()
    }

    fn degree_bound(&self) -> usize {
        self.arity
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn start_round_state(&self) -> Self::RoundState {
        GruenSplitEq::new(&self.tau)
    }

    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[E],
    ) -> Result<E, HachiError> {
        let batched_output = self
            .batch_weights
            .iter()
            .zip(self.child_claims.chunks_exact(self.arity))
            .fold(E::zero(), |acc, (&weight, child_claims)| {
                let product = child_claims
                    .iter()
                    .copied()
                    .fold(E::one(), |prod, claim| prod * claim);
                acc + weight * product
            });
        Ok(round_state.current_scalar() * batched_output)
    }
}

struct PolynomialStageProver<E: FieldCore> {
    s_table: Vec<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    poly_coeffs: Vec<E>,
    num_rounds: usize,
}

impl<E: FieldCore> PolynomialStageProver<E> {
    fn new(s_table: Vec<E>, tau: &[E], input_claim: E, poly_coeffs: Vec<E>) -> Self {
        Self {
            s_table,
            split_eq: GruenSplitEq::new(tau),
            input_claim,
            poly_coeffs,
            num_rounds: tau.len(),
        }
    }

    fn final_s_claim(&self) -> E {
        debug_assert_eq!(self.s_table.len(), 1);
        self.s_table[0]
    }
}

impl<E: FieldCore> EqFactoredSumcheckInstanceProver<E> for PolynomialStageProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.poly_coeffs.len().saturating_sub(1)
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn current_linear_factor_evals(&self) -> (E, E) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<E> {
        debug_assert!(self.degree_bound() <= MAX_TREE_STAGE_Q_DEGREE);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let degree = self.degree_bound();
        let expected_pairs = num_first * e_second.len();
        debug_assert_eq!(
            self.s_table.len(),
            expected_pairs * 2,
            "polynomial stage table length should match split-eq shape",
        );

        let q_coeffs = cfg_fold_reduce!(
            0..e_second.len(),
            || [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
            |mut outer, j_high| {
                let e_out = e_second[j_high];
                let base = j_high * num_first;
                let mut inner = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = base + j_low;
                    let coeffs = compose_small_poly_with_affine(
                        &self.poly_coeffs,
                        self.s_table[2 * j],
                        self.s_table[2 * j + 1] - self.s_table[2 * j],
                    );
                    for k in 0..=degree {
                        inner[k] += e_in * coeffs[k];
                    }
                }
                for k in 0..=degree {
                    outer[k] += e_out * inner[k];
                }
                outer
            },
            |mut a, b| {
                for k in 0..=degree {
                    a[k] += b[k];
                }
                a
            }
        );

        EqFactoredUniPoly::from_q_coeffs(q_coeffs[..=degree].to_vec())
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        self.split_eq.bind(r_round);
        fold_evals_in_place(&mut self.s_table, r_round);
    }
}

struct PolynomialStageVerifier<E: FieldCore> {
    tau: Vec<E>,
    input_claim: E,
    poly_coeffs: Vec<E>,
    s_claim: E,
}

impl<E: FieldCore> PolynomialStageVerifier<E> {
    fn new(tau: Vec<E>, input_claim: E, poly_coeffs: Vec<E>, s_claim: E) -> Self {
        Self {
            tau,
            input_claim,
            poly_coeffs,
            s_claim,
        }
    }
}

impl<E: FieldCore> EqFactoredSumcheckInstanceVerifier<E> for PolynomialStageVerifier<E> {
    type RoundState = GruenSplitEq<E>;

    fn num_rounds(&self) -> usize {
        self.tau.len()
    }

    fn degree_bound(&self) -> usize {
        self.poly_coeffs.len().saturating_sub(1)
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn start_round_state(&self) -> Self::RoundState {
        GruenSplitEq::new(&self.tau)
    }

    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[E],
    ) -> Result<E, HachiError> {
        Ok(round_state.current_scalar() * eval_poly(&self.poly_coeffs, self.s_claim))
    }
}

/// Backend-specific Stage 1 witness representation.
enum Stage1Witness<E: FieldCore> {
    Compact(Vec<i8>),
    PaddedS(Vec<E>),
}

/// Stage-1 range-check prover, including the root/leaf tree choreography.
pub struct HachiStage1Prover<E: FieldCore> {
    witness: Stage1Witness<E>,
    tau0: Vec<E>,
    b: usize,
    live_x_cols: usize,
    num_u: usize,
    num_l: usize,
}

impl<E: FieldCore + FromSmallInt> HachiStage1Prover<E> {
    /// Build the stage-1 prover from the compact witness table.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidSize`] if the compact witness rows do not
    /// match `live_x_cols * 2^num_l`.
    pub fn new(
        w_evals_compact: &[i8],
        tau0: &[E],
        b: usize,
        live_x_cols: usize,
        num_u: usize,
        num_l: usize,
    ) -> Result<Self, HachiError> {
        validate_stage1_tree_basis(b)?;
        Ok(Self {
            witness: if b <= 8 {
                Stage1Witness::Compact(w_evals_compact.to_vec())
            } else {
                Stage1Witness::PaddedS(padded_s_table(w_evals_compact, live_x_cols, num_u, num_l)?)
            },
            tau0: tau0.to_vec(),
            b,
            live_x_cols,
            num_u,
            num_l,
        })
    }
}

impl<E: FieldCore + CanonicalField + FromSmallInt + HasUnreducedOps> HachiStage1Prover<E> {
    /// Produce the full stage-1 tree proof and return the final `r_stage1`.
    ///
    /// # Errors
    ///
    /// Propagates any transcript or sumcheck failure from the internal root
    /// and leaf-stage proofs.
    pub fn prove<T: Transcript<E>>(
        self,
        transcript: &mut T,
    ) -> Result<(HachiStage1Proof<E>, Vec<E>), HachiError> {
        let Self {
            witness,
            tau0,
            b,
            live_x_cols,
            num_u,
            num_l,
        } = self;
        validate_stage1_tree_basis(b)?;
        let s_table = match witness {
            Stage1Witness::Compact(w_evals_compact) => {
                // Keep the tree wire shape, but reuse the old compact/prefix-aware
                // stage-1 backend for the single-stage `b <= 8` path.
                let mut leaf_stage = single_stage_backend::HachiStage1Prover::new(
                    &w_evals_compact,
                    &tau0,
                    b,
                    live_x_cols,
                    num_u,
                    num_l,
                );
                let (sumcheck, r_stage1, _final_claim) = prove_eq_factored_sumcheck::<E, _, E, _, _>(
                    &mut leaf_stage,
                    transcript,
                    |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
                )?;
                let proof = HachiStage1Proof {
                    stages: vec![HachiStage1StageProof {
                        sumcheck,
                        child_claims: Vec::new(),
                    }],
                    s_claim: leaf_stage.final_s_claim(),
                };
                return Ok((proof, r_stage1));
            }
            Stage1Witness::PaddedS(s_table) => s_table,
        };

        let leaf_coeffs = stage1_leaf_coeffs::<E>(b);
        let product_layers = build_product_stage_layers(
            build_leaf_tables(&leaf_coeffs, &s_table),
            &stage1_tree_product_stage_arities(b),
        );
        let mut stage_proofs = Vec::with_capacity(product_layers.len() + 1);
        let mut current_tau = tau0;
        let mut current_claim = E::zero();
        let mut current_weights = vec![E::one()];

        for layer in product_layers {
            let mut product_stage = ProductStageProver::new(
                layer.child_tables_by_parent,
                current_weights,
                &current_tau,
                current_claim,
            );
            let (sumcheck, next_tau, _final_claim) = prove_eq_factored_sumcheck::<E, _, E, _, _>(
                &mut product_stage,
                transcript,
                |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
            )?;
            let child_claims = product_stage.final_child_claims();
            stage_proofs.push(HachiStage1StageProof {
                sumcheck,
                child_claims: child_claims.clone(),
            });

            absorb_interstage_claims(&child_claims, transcript);
            let gamma = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH);
            current_weights = stage1_interstage_batch_weights(gamma, child_claims.len());
            current_claim = linear_combination(&current_weights, &child_claims);
            current_tau = next_tau;
        }

        let batched_leaf_coeffs = combine_polys(&current_weights, &leaf_coeffs);
        let mut leaf_stage =
            PolynomialStageProver::new(s_table, &current_tau, current_claim, batched_leaf_coeffs);
        let (leaf_sumcheck, r_stage1, _leaf_final_claim) =
            prove_eq_factored_sumcheck::<E, _, E, _, _>(&mut leaf_stage, transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })?;
        stage_proofs.push(HachiStage1StageProof {
            sumcheck: leaf_sumcheck,
            child_claims: Vec::new(),
        });

        Ok((
            HachiStage1Proof {
                stages: stage_proofs,
                s_claim: leaf_stage.final_s_claim(),
            },
            r_stage1,
        ))
    }
}

/// Stage-1 range-check verifier, including the root/leaf tree choreography.
pub struct HachiStage1Verifier<E: FieldCore> {
    tau0: Vec<E>,
    b: usize,
}

impl<E: FieldCore> HachiStage1Verifier<E> {
    /// Construct the stage-1 verifier from `tau0` and `b`.
    pub fn new(tau0: Vec<E>, b: usize) -> Self {
        Self { tau0, b }
    }
}

impl<E: FieldCore + CanonicalField + FromSmallInt> HachiStage1Verifier<E> {
    /// Verify the full stage-1 tree proof and return the final `r_stage1`.
    ///
    /// # Errors
    ///
    /// Returns an error if the staged proof shape is inconsistent with `b`, if
    /// any internal stage sumcheck fails, or if the final oracle check fails.
    pub fn verify<T: Transcript<E>>(
        &self,
        proof: &HachiStage1Proof<E>,
        transcript: &mut T,
    ) -> Result<Vec<E>, HachiError> {
        validate_stage1_tree_basis(self.b)?;
        let expected_stage_count = stage1_stage_count(self.b);
        if proof.stages.len() != expected_stage_count {
            return Err(HachiError::InvalidSize {
                expected: expected_stage_count,
                actual: proof.stages.len(),
            });
        }

        let leaf_coeffs = stage1_leaf_coeffs::<E>(self.b);
        if leaf_coeffs.len() == 1 {
            if !proof.stages[0].child_claims.is_empty() {
                return Err(HachiError::InvalidProof);
            }
            let leaf_verifier = single_stage_backend::HachiStage1Verifier::new(
                self.tau0.clone(),
                proof.s_claim,
                self.b,
            );
            return verify_eq_factored_sumcheck::<E, _, E, _, _>(
                &proof.stages[0].sumcheck,
                &leaf_verifier,
                transcript,
                |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
            );
        }

        let product_stage_arities = stage1_tree_product_stage_arities(self.b);
        let Some((leaf_stage_proof, product_stage_proofs)) = proof.stages.split_last() else {
            return Err(HachiError::InvalidProof);
        };
        if !leaf_stage_proof.child_claims.is_empty() {
            return Err(HachiError::InvalidProof);
        }

        let mut current_tau = self.tau0.clone();
        let mut current_claim = E::zero();
        let mut current_weights = vec![E::one()];

        for (&arity, stage_proof) in product_stage_arities
            .iter()
            .zip(product_stage_proofs.iter())
        {
            let expected_child_claims = current_weights.len() * arity;
            if stage_proof.child_claims.len() != expected_child_claims {
                return Err(HachiError::InvalidSize {
                    expected: expected_child_claims,
                    actual: stage_proof.child_claims.len(),
                });
            }

            let product_verifier = ProductStageVerifier::new(
                current_tau,
                current_claim,
                stage_proof.child_claims.clone(),
                current_weights,
                arity,
            );
            current_tau = verify_eq_factored_sumcheck::<E, _, E, _, _>(
                &stage_proof.sumcheck,
                &product_verifier,
                transcript,
                |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
            )?;

            absorb_interstage_claims(&stage_proof.child_claims, transcript);
            let gamma = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH);
            current_weights =
                stage1_interstage_batch_weights(gamma, stage_proof.child_claims.len());
            current_claim = linear_combination(&current_weights, &stage_proof.child_claims);
        }

        let batched_leaf_coeffs = combine_polys(&current_weights, &leaf_coeffs);
        let leaf_verifier = PolynomialStageVerifier::new(
            current_tau,
            current_claim,
            batched_leaf_coeffs,
            proof.s_claim,
        );
        verify_eq_factored_sumcheck::<E, _, E, _, _>(
            &leaf_stage_proof.sumcheck,
            &leaf_verifier,
            transcript,
            |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128Offset275;
    use crate::protocol::transcript::Blake2bTranscript;

    type F = Prime128Offset275;

    fn sample_stage1_witness(b: usize, live_x_cols: usize, num_l: usize) -> Vec<i8> {
        let half = (b / 2) as i16;
        let y_len = 1usize << num_l;
        (0..live_x_cols * y_len)
            .map(|idx| {
                (idx as i16 % half)
                    .try_into()
                    .expect("test digit should fit in i8")
            })
            .collect()
    }

    fn reorder_tau0_y_first(tau0: &[F], num_u: usize, num_l: usize) -> Vec<F> {
        let mut reordered = Vec::with_capacity(tau0.len());
        reordered.extend_from_slice(&tau0[num_u..num_u + num_l]);
        reordered.extend_from_slice(&tau0[..num_u]);
        reordered
    }

    fn assert_stage1_roundtrip(
        b: usize,
        live_x_cols: usize,
        tau0: Vec<F>,
        expected_child_claim_counts: &[usize],
    ) {
        let num_u = 3;
        let num_l = 1;
        let witness = sample_stage1_witness(b, live_x_cols, num_l);
        let tau0 = reorder_tau0_y_first(&tau0, num_u, num_l);

        let prover = HachiStage1Prover::new(&witness, &tau0, b, live_x_cols, num_u, num_l)
            .expect("stage1 prover should build");
        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof, r_stage1) = prover
            .prove(&mut prover_transcript)
            .expect("stage1 proof should succeed");

        let verifier = HachiStage1Verifier::new(tau0, b);
        let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let verified_r = verifier
            .verify(&proof, &mut verifier_transcript)
            .expect("stage1 verification should succeed");

        assert_eq!(r_stage1, verified_r);
        assert_eq!(proof.stages.len(), expected_child_claim_counts.len());
        for (stage, &expected_child_claims) in proof.stages.iter().zip(expected_child_claim_counts)
        {
            assert_eq!(stage.child_claims.len(), expected_child_claims);
        }
    }

    #[test]
    fn stage1_tree_shapes_match_generic_quartic_chain() {
        assert_eq!(
            stage1_tree_stage_shapes(7, 4)
                .into_iter()
                .map(|shape| (shape.sumcheck.1, shape.child_claims))
                .collect::<Vec<_>>(),
            vec![(2, 0)]
        );
        assert_eq!(
            stage1_tree_stage_shapes(7, 8)
                .into_iter()
                .map(|shape| (shape.sumcheck.1, shape.child_claims))
                .collect::<Vec<_>>(),
            vec![(4, 0)]
        );
        assert_eq!(
            stage1_tree_stage_shapes(7, 16)
                .into_iter()
                .map(|shape| (shape.sumcheck.1, shape.child_claims))
                .collect::<Vec<_>>(),
            vec![(2, 2), (4, 0)]
        );
        assert_eq!(
            stage1_tree_stage_shapes(7, 32)
                .into_iter()
                .map(|shape| (shape.sumcheck.1, shape.child_claims))
                .collect::<Vec<_>>(),
            vec![(4, 4), (4, 0)]
        );
        assert_eq!(
            stage1_tree_stage_shapes(7, 64)
                .into_iter()
                .map(|shape| (shape.sumcheck.1, shape.child_claims))
                .collect::<Vec<_>>(),
            vec![(2, 2), (4, 8), (4, 0)]
        );
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b16() {
        assert_stage1_roundtrip(
            16,
            6,
            vec![
                F::from_u64(3),
                F::from_u64(5),
                F::from_u64(7),
                F::from_u64(9),
            ],
            &[2, 0],
        );
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b32() {
        assert_stage1_roundtrip(
            32,
            5,
            vec![
                F::from_u64(11),
                F::from_u64(13),
                F::from_u64(17),
                F::from_u64(19),
            ],
            &[4, 0],
        );
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b64() {
        assert_stage1_roundtrip(
            64,
            5,
            vec![
                F::from_u64(23),
                F::from_u64(29),
                F::from_u64(31),
                F::from_u64(37),
            ],
            &[2, 8, 0],
        );
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b4() {
        assert_stage1_roundtrip(
            4,
            5,
            vec![
                F::from_u64(3),
                F::from_u64(5),
                F::from_u64(7),
                F::from_u64(9),
            ],
            &[0],
        );
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b8() {
        assert_stage1_roundtrip(
            8,
            5,
            vec![
                F::from_u64(11),
                F::from_u64(13),
                F::from_u64(17),
                F::from_u64(19),
            ],
            &[0],
        );
    }
}
