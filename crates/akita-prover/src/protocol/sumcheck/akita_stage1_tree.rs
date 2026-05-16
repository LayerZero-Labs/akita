//! Stage-1 range-check tree prover for the Akita PCS.
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

use super::akita_stage1 as single_stage_backend;
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::fields::HasUnreducedOps;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_sumcheck::{
    fold_evals_in_place, prove_eq_factored_sumcheck, EqFactoredSumcheckInstanceProver,
    EqFactoredUniPoly,
};
use akita_transcript::labels;
use akita_transcript::Transcript;
use akita_types::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    stage1_interstage_batch_weights, stage1_leaf_coeffs, stage1_tree_product_stage_arities,
    validate_stage1_tree_basis, AkitaStage1Proof, AkitaStage1StageProof,
};

fn compact_s_from_w(w: i8) -> i64 {
    let w = i64::from(w);
    w * (w + 1)
}

const MAX_TREE_STAGE_Q_DEGREE: usize = 4;

fn padded_s_table<E: FieldCore + FromPrimitiveInt>(
    w_evals_compact: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Result<Vec<E>, AkitaError> {
    let x_len = 1usize << col_bits;
    let y_len = 1usize << ring_bits;
    let expected = live_x_cols * y_len;
    if w_evals_compact.len() != expected {
        return Err(AkitaError::InvalidSize {
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

fn compose_small_poly_with_affine<E: FieldCore>(coeffs: &[E], offset: E, slope: E) -> [E; 5] {
    debug_assert!(coeffs.len() <= MAX_TREE_STAGE_Q_DEGREE + 1);

    let mut out = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
    let mut power = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
    power[0] = E::one();

    for (idx, &coeff) in coeffs.iter().enumerate() {
        if idx > 0 {
            for k in (0..idx).rev() {
                power[k + 1] += power[k] * slope;
                power[k] *= offset;
            }
        }
        for k in 0..=idx {
            out[k] += coeff * power[k];
        }
    }

    out
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
            *acc *= *value;
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
            split_eq: GruenSplitEq::new(tau).expect("valid prover product-stage challenge shape"),
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
                                poly[k] *= left;
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
            split_eq: GruenSplitEq::new(tau)
                .expect("valid prover polynomial-stage challenge shape"),
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

/// Backend-specific Stage 1 witness representation.
enum Stage1Witness<E: FieldCore> {
    Compact(Vec<i8>),
    PaddedS(Vec<E>),
}

/// Stage-1 range-check prover, including the root/leaf tree choreography.
pub struct AkitaStage1Prover<E: FieldCore> {
    witness: Stage1Witness<E>,
    tau0: Vec<E>,
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
}

impl<E: FieldCore + FromPrimitiveInt> AkitaStage1Prover<E> {
    /// Build the stage-1 prover from the compact witness table.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSize`] if the compact witness rows do not
    /// match `live_x_cols * 2^ring_bits`.
    pub fn new(
        w_evals_compact: &[i8],
        tau0: &[E],
        b: usize,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Result<Self, AkitaError> {
        validate_stage1_tree_basis(b)?;
        Ok(Self {
            witness: if b <= 8 {
                Stage1Witness::Compact(w_evals_compact.to_vec())
            } else {
                Stage1Witness::PaddedS(padded_s_table(
                    w_evals_compact,
                    live_x_cols,
                    col_bits,
                    ring_bits,
                )?)
            },
            tau0: tau0.to_vec(),
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        })
    }
}

impl<E: FieldCore + CanonicalField + FromPrimitiveInt + HasUnreducedOps> AkitaStage1Prover<E> {
    /// Produce the full stage-1 tree proof and return the final `r_stage1`.
    ///
    /// # Errors
    ///
    /// Propagates any transcript or sumcheck failure from the internal root
    /// and leaf-stage proofs.
    pub fn prove<T: Transcript<E>>(
        self,
        transcript: &mut T,
    ) -> Result<(AkitaStage1Proof<E>, Vec<E>), AkitaError> {
        let Self {
            witness,
            tau0,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        } = self;
        validate_stage1_tree_basis(b)?;
        let s_table = match witness {
            Stage1Witness::Compact(w_evals_compact) => {
                // Keep the tree wire shape, but reuse the old compact/prefix-aware
                // stage-1 backend for the single-stage `b <= 8` path.
                let mut leaf_stage = single_stage_backend::AkitaStage1Prover::new(
                    &w_evals_compact,
                    &tau0,
                    b,
                    live_x_cols,
                    col_bits,
                    ring_bits,
                );
                let (sumcheck, r_stage1, _final_claim) = prove_eq_factored_sumcheck::<E, _, E, _, _>(
                    &mut leaf_stage,
                    transcript,
                    |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
                )?;
                let proof = AkitaStage1Proof {
                    stages: vec![AkitaStage1StageProof {
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
            stage_proofs.push(AkitaStage1StageProof {
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
        stage_proofs.push(AkitaStage1StageProof {
            sumcheck: leaf_sumcheck,
            child_claims: Vec::new(),
        });

        Ok((
            AkitaStage1Proof {
                stages: stage_proofs,
                s_claim: leaf_stage.final_s_claim(),
            },
            r_stage1,
        ))
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use akita_types::stage1_tree_stage_shapes;

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
}
