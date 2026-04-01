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
//! This matches the proof-size study's current tree cutover for `log_basis <= 5`
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
use crate::{cfg_fold_reduce, cfg_iter, CanonicalField, FieldCore, FromSmallInt};

fn compact_s_from_w(w: i8) -> i64 {
    let w = i64::from(w);
    w * (w + 1)
}

fn validate_stage1_tree_basis(b: usize) -> Result<(), HachiError> {
    if !matches!(b, 4 | 8 | 16 | 32) {
        return Err(HachiError::InvalidInput(format!(
            "stage1 tree currently supports b in {{4, 8, 16, 32}}, got {b}"
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
    for y in 0..y_len {
        let src_start = y * live_x_cols;
        let dst_start = y * x_len;
        for x in 0..live_x_cols {
            out[dst_start + x] = E::from_i64(compact_s_from_w(w_evals_compact[src_start + x]));
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

fn build_leaf_tables<E: FieldCore>(leaf_coeffs: &[Vec<E>], s_table: &[E]) -> Vec<Vec<E>> {
    cfg_iter!(leaf_coeffs)
        .map(|coeffs| s_table.iter().copied().map(|s| eval_poly(coeffs, s)).collect())
        .collect()
}

fn stage1_tree_stage_shape_from_b(rounds: usize, b: usize) -> Vec<HachiStage1StageShape> {
    debug_assert!(matches!(b, 4 | 8 | 16 | 32));
    let leaf_groups = b.div_ceil(8);
    if leaf_groups <= 1 {
        return vec![HachiStage1StageShape {
            sumcheck: (rounds, b / 2),
            child_claims: 0,
        }];
    }

    vec![
        HachiStage1StageShape {
            sumcheck: (rounds, leaf_groups),
            child_claims: leaf_groups,
        },
        HachiStage1StageShape {
            sumcheck: (rounds, 4),
            child_claims: 0,
        },
    ]
}

pub(crate) fn stage1_tree_stage_shapes(rounds: usize, b: usize) -> Vec<HachiStage1StageShape> {
    stage1_tree_stage_shape_from_b(rounds, b)
}

fn stage1_stage_count(b: usize) -> usize {
    if b <= 8 {
        1
    } else {
        2
    }
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
    tables: Vec<Vec<E>>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ProductStageProver<E> {
    fn new(tables: Vec<Vec<E>>, tau: &[E], input_claim: E) -> Self {
        debug_assert!(!tables.is_empty());
        Self {
            tables,
            split_eq: GruenSplitEq::new(tau),
            input_claim,
            num_rounds: tau.len(),
        }
    }

    fn final_table_claims(&self) -> Vec<E> {
        self.tables
            .iter()
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
        self.tables.len()
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
            self.tables[0].len(),
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
                    let mut poly = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
                    poly[0] = E::one();
                    for (current_degree, table) in self.tables.iter().enumerate() {
                        let left = table[2 * j];
                        let slope = table[2 * j + 1] - left;
                        for k in (0..=current_degree).rev() {
                            poly[k + 1] += poly[k] * slope;
                            poly[k] = poly[k] * left;
                        }
                    }
                    for k in 0..=degree {
                        inner[k] += e_in * poly[k];
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
        for table in &mut self.tables {
            fold_evals_in_place(table, r_round);
        }
    }
}

struct ProductStageVerifier<E: FieldCore> {
    tau: Vec<E>,
    input_claim: E,
    child_claims: Vec<E>,
}

impl<E: FieldCore> ProductStageVerifier<E> {
    fn new(tau: Vec<E>, input_claim: E, child_claims: Vec<E>) -> Self {
        Self {
            tau,
            input_claim,
            child_claims,
        }
    }
}

impl<E: FieldCore> EqFactoredSumcheckInstanceVerifier<E> for ProductStageVerifier<E> {
    type RoundState = GruenSplitEq<E>;

    fn num_rounds(&self) -> usize {
        self.tau.len()
    }

    fn degree_bound(&self) -> usize {
        self.child_claims.len()
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
        Ok(round_state.current_scalar()
            * self
                .child_claims
                .iter()
                .copied()
                .fold(E::one(), |acc, claim| acc * claim))
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
                Stage1Witness::PaddedS(padded_s_table(
                    w_evals_compact,
                    live_x_cols,
                    num_u,
                    num_l,
                )?)
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
                let (sumcheck, r_stage1, _final_claim) =
                    prove_eq_factored_sumcheck::<E, _, E, _, _>(&mut leaf_stage, transcript, |tr| {
                        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
                    })?;
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
        let leaf_tables = build_leaf_tables(&leaf_coeffs, &s_table);
        let mut root_stage = ProductStageProver::new(leaf_tables, &tau0, E::zero());
        let (root_sumcheck, r_root, _root_final_claim) =
            prove_eq_factored_sumcheck::<E, _, E, _, _>(&mut root_stage, transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })?;
        let child_claims = root_stage.final_table_claims();

        absorb_interstage_claims(&child_claims, transcript);
        let gamma = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH);
        let weights = stage1_interstage_batch_weights(gamma, child_claims.len());
        let batched_claim = linear_combination(&weights, &child_claims);
        let batched_leaf_coeffs = combine_polys(&weights, &leaf_coeffs);

        let mut leaf_stage =
            PolynomialStageProver::new(s_table, &r_root, batched_claim, batched_leaf_coeffs);
        let (leaf_sumcheck, r_stage1, _leaf_final_claim) =
            prove_eq_factored_sumcheck::<E, _, E, _, _>(&mut leaf_stage, transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })?;

        Ok((
            HachiStage1Proof {
                stages: vec![
                    HachiStage1StageProof {
                        sumcheck: root_sumcheck,
                        child_claims,
                    },
                    HachiStage1StageProof {
                        sumcheck: leaf_sumcheck,
                        child_claims: Vec::new(),
                    },
                ],
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

        let root_stage = &proof.stages[0];
        if root_stage.child_claims.len() != leaf_coeffs.len() {
            return Err(HachiError::InvalidSize {
                expected: leaf_coeffs.len(),
                actual: root_stage.child_claims.len(),
            });
        }
        if !proof.stages[1].child_claims.is_empty() {
            return Err(HachiError::InvalidProof);
        }

        let root_verifier = ProductStageVerifier::new(
            self.tau0.clone(),
            E::zero(),
            root_stage.child_claims.clone(),
        );
        let r_root = verify_eq_factored_sumcheck::<E, _, E, _, _>(
            &root_stage.sumcheck,
            &root_verifier,
            transcript,
            |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
        )?;

        absorb_interstage_claims(&root_stage.child_claims, transcript);
        let gamma = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH);
        let weights = stage1_interstage_batch_weights(gamma, root_stage.child_claims.len());
        let batched_claim = linear_combination(&weights, &root_stage.child_claims);
        let batched_leaf_coeffs = combine_polys(&weights, &leaf_coeffs);

        let leaf_verifier =
            PolynomialStageVerifier::new(r_root, batched_claim, batched_leaf_coeffs, proof.s_claim);
        verify_eq_factored_sumcheck::<E, _, E, _, _>(
            &proof.stages[1].sumcheck,
            &leaf_verifier,
            transcript,
            |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128Offset5823;
    use crate::protocol::transcript::Blake2bTranscript;

    type F = Prime128Offset5823;

    fn sample_stage1_witness(b: usize, live_x_cols: usize, num_l: usize) -> Vec<i8> {
        let half = (b / 2) as i8;
        let y_len = 1usize << num_l;
        (0..live_x_cols * y_len)
            .map(|idx| (idx as i8 % half).max(0))
            .collect()
    }

    #[test]
    fn stage1_tree_shapes_match_supported_bases() {
        assert_eq!(stage1_tree_stage_shapes(7, 4).len(), 1);
        assert_eq!(stage1_tree_stage_shapes(7, 8).len(), 1);
        assert_eq!(stage1_tree_stage_shapes(7, 16).len(), 2);
        assert_eq!(stage1_tree_stage_shapes(7, 32).len(), 2);
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b16() {
        let b = 16;
        let num_u = 3;
        let num_l = 1;
        let live_x_cols = 6;
        let tau0 = vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(9),
        ];
        let witness = sample_stage1_witness(b, live_x_cols, num_l);

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
        assert_eq!(proof.stages.len(), 2);
        assert_eq!(proof.stages[0].child_claims.len(), 2);
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b32() {
        let b = 32;
        let num_u = 3;
        let num_l = 1;
        let live_x_cols = 5;
        let tau0 = vec![
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
        ];
        let witness = sample_stage1_witness(b, live_x_cols, num_l);

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
        assert_eq!(proof.stages.len(), 2);
        assert_eq!(proof.stages[0].child_claims.len(), 4);
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b4() {
        let b = 4;
        let num_u = 3;
        let num_l = 1;
        let live_x_cols = 5;
        let tau0 = vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(9),
        ];
        let witness = sample_stage1_witness(b, live_x_cols, num_l);

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
        assert_eq!(proof.stages.len(), 1);
        assert!(proof.stages[0].child_claims.is_empty());
    }

    #[test]
    fn stage1_tree_prove_verify_roundtrip_b8() {
        let b = 8;
        let num_u = 3;
        let num_l = 1;
        let live_x_cols = 5;
        let tau0 = vec![
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
        ];
        let witness = sample_stage1_witness(b, live_x_cols, num_l);

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
        assert_eq!(proof.stages.len(), 1);
        assert!(proof.stages[0].child_claims.is_empty());
    }
}
