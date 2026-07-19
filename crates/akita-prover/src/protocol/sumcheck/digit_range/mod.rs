//! Stage-1 range-check tree prover for the Akita PCS.
//!
//! For `b <= 8`, stage 1 is still a single eq-factored sumcheck over
//! `Q(range_image(z))`, where `range_image(z) = w(z)(w(z)+1)` and `Q` is the
//! full range polynomial.
//! For larger supported bases, stage 1 is written as a short root-to-leaf tree:
//!
//! - a root stage proves the product of `2` or `4` quartic leaf factors,
//! - the prover sends those child-node claims at the sampled root point,
//! - a leaf stage proves a random linear combination of the quartic factors
//!   directly from `range_image`.
//!
//! This matches the proof-size study's current tree cutover for `log_basis <= 6`
//! without widening the recursive witness encoding beyond the existing runtime
//! bound.

pub(crate) mod direct_range_leaf;
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::EqFactoredSumcheckInstanceProverExt;
use akita_sumcheck::{fold_evals_in_place, EqFactoredSumcheckInstanceProver, EqFactoredUniPoly};
use akita_transcript::labels;
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::{
    AkitaStage1Proof, AkitaStage1StageProof, DigitRangeEqualityPoint, DigitRangePlan,
    FlatBooleanDomain,
};

type Stage1ProveOutput<E> = (AkitaStage1Proof<E>, Vec<E>);

fn range_image_from_digit(w: i8) -> i64 {
    let w = i64::from(w);
    w * (w + 1)
}

const MAX_TREE_STAGE_Q_DEGREE: usize = 4;

fn padded_range_image_table<E: FieldCore + FromPrimitiveInt>(
    digit_witness: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Result<Vec<E>, AkitaError> {
    let col_bits = u32::try_from(col_bits)
        .map_err(|_| AkitaError::InvalidInput("stage-1 column width overflow".to_string()))?;
    let ring_bits = u32::try_from(ring_bits)
        .map_err(|_| AkitaError::InvalidInput("stage-1 ring width overflow".to_string()))?;
    let x_len = 1usize
        .checked_shl(col_bits)
        .ok_or_else(|| AkitaError::InvalidInput("stage-1 column width overflow".to_string()))?;
    let y_len = 1usize
        .checked_shl(ring_bits)
        .ok_or_else(|| AkitaError::InvalidInput("stage-1 ring width overflow".to_string()))?;
    let expected = live_x_cols
        .checked_mul(y_len)
        .ok_or_else(|| AkitaError::InvalidInput("stage-1 witness size overflow".to_string()))?;
    if digit_witness.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: digit_witness.len(),
        });
    }

    let padded_len = x_len
        .checked_mul(y_len)
        .ok_or_else(|| AkitaError::InvalidInput("stage-1 padded table overflow".to_string()))?;
    let mut out = vec![E::zero(); padded_len];
    for x in 0..live_x_cols {
        let src_start = x * y_len;
        for y in 0..y_len {
            out[x * y_len + y] = E::from_i64(range_image_from_digit(digit_witness[src_start + y]));
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

fn build_leaf_tables<E: FieldCore>(
    plan: DigitRangePlan,
    leaf_coeffs: &[Vec<E>],
    range_image: &[E],
) -> Vec<Vec<E>> {
    cfg_iter!(leaf_coeffs)
        .map(|coeffs| {
            range_image
                .iter()
                .copied()
                .map(|range_image_eval| plan.evaluate_leaf_polynomial(coeffs, range_image_eval))
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

        while let Some(first_child) = current_iter.next() {
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

struct ProductStageState<E: FieldCore> {
    child_tables_by_parent: Vec<Vec<Vec<E>>>,
    batch_weights: Vec<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ProductStageState<E> {
    fn new(
        child_tables_by_parent: Vec<Vec<Vec<E>>>,
        batch_weights: Vec<E>,
        tau: &[E],
        input_claim: E,
    ) -> Result<Self, AkitaError> {
        debug_assert!(!child_tables_by_parent.is_empty());
        debug_assert_eq!(child_tables_by_parent.len(), batch_weights.len());
        let arity = child_tables_by_parent[0].len();
        debug_assert!(matches!(arity, 2 | 4));
        for child_tables in &child_tables_by_parent {
            debug_assert_eq!(child_tables.len(), arity);
        }
        Ok(Self {
            child_tables_by_parent,
            batch_weights,
            split_eq: GruenSplitEq::new(tau)?,
            input_claim,
            num_rounds: tau.len(),
        })
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

impl<E: FieldCore + HasOptimizedFold> EqFactoredSumcheckInstanceProver<E> for ProductStageState<E> {
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

struct PolynomialLeafState<E: FieldCore> {
    range_image: Vec<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    poly_coeffs: Vec<E>,
    num_rounds: usize,
}

impl<E: FieldCore> PolynomialLeafState<E> {
    fn new(
        range_image: Vec<E>,
        tau: &[E],
        input_claim: E,
        poly_coeffs: Vec<E>,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            range_image,
            split_eq: GruenSplitEq::new(tau)?,
            input_claim,
            poly_coeffs,
            num_rounds: tau.len(),
        })
    }

    fn final_range_image_eval(&self) -> E {
        debug_assert_eq!(self.range_image.len(), 1);
        self.range_image[0]
    }
}

impl<E: FieldCore + HasOptimizedFold> EqFactoredSumcheckInstanceProver<E>
    for PolynomialLeafState<E>
{
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
            self.range_image.len(),
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
                        self.range_image[2 * j],
                        self.range_image[2 * j + 1] - self.range_image[2 * j],
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
        fold_evals_in_place(&mut self.range_image, r_round);
    }
}

/// Backend-specific Stage 1 witness representation.
enum DigitRangeState<E: FieldCore> {
    Compact(std::sync::Arc<[i8]>),
    PaddedRangeImage(Vec<E>),
}

/// Stage-1 range-check prover, including the root/leaf tree choreography.
pub struct DigitRangeProver<E: FieldCore> {
    witness: DigitRangeState<E>,
    equality_point: Vec<E>,
    plan: DigitRangePlan,
    live_block_count: usize,
    high_variable_count: usize,
    low_variable_count: usize,
}

impl<E: FieldCore + FromPrimitiveInt> DigitRangeProver<E> {
    /// Build the prover from the shared compact digit witness and checked layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness length, domain, or equality point are
    /// inconsistent.
    pub fn new(
        digit_witness: std::sync::Arc<[i8]>,
        plan: DigitRangePlan,
        domain: FlatBooleanDomain,
        equality_point: DigitRangeEqualityPoint<E>,
    ) -> Result<Self, AkitaError> {
        equality_point.validate_domain(domain)?;
        if digit_witness.len() != domain.live_len() {
            return Err(AkitaError::InvalidSize {
                expected: domain.live_len(),
                actual: digit_witness.len(),
            });
        }
        let low_variable_count = equality_point.low_variable_count();
        let high_variable_count = domain.num_vars() - low_variable_count;
        let live_block_count = domain.live_block_count(low_variable_count)?;
        let coordinates = equality_point.into_coordinates();
        let witness = if plan.basis() <= 8 {
            DigitRangeState::Compact(digit_witness)
        } else {
            let _span = tracing::info_span!(
                "digit_range_materialize_range_image",
                basis = plan.basis(),
                live_len = domain.live_len(),
                domain_len = domain.domain_len(),
            )
            .entered();
            DigitRangeState::PaddedRangeImage(padded_range_image_table(
                &digit_witness,
                live_block_count,
                high_variable_count,
                low_variable_count,
            )?)
        };
        Ok(Self {
            witness,
            equality_point: coordinates,
            plan,
            live_block_count,
            high_variable_count,
            low_variable_count,
        })
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold + AkitaSerialize>
    DigitRangeProver<E>
{
    /// Produce the full stage-1 tree proof and return the final `stage1_point`.
    ///
    /// # Errors
    ///
    /// Propagates any transcript or sumcheck failure from the internal root
    /// and leaf-stage proofs.
    pub fn prove<F, T>(self, transcript: &mut T) -> Result<Stage1ProveOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F>,
        T: Transcript<F>,
    {
        fn absorb_child_claims<F, E, T>(claims: &[E], transcript: &mut T)
        where
            F: FieldCore + CanonicalField,
            E: ExtField<F>,
            T: Transcript<F>,
        {
            for claim in claims {
                append_ext_field::<F, E, T>(
                    transcript,
                    labels::ABSORB_SUMCHECK_INTERSTAGE_CLAIM,
                    claim,
                );
            }
        }
        let Self {
            witness,
            equality_point,
            plan,
            live_block_count,
            high_variable_count,
            low_variable_count,
        } = self;
        let _prove_span = tracing::info_span!(
            "digit_range_prove",
            basis = plan.basis(),
            rounds = equality_point.len(),
        )
        .entered();
        let range_image = match witness {
            DigitRangeState::Compact(digit_witness) => {
                let _leaf_span = tracing::info_span!("digit_range_direct_leaf").entered();
                let mut leaf_stage = direct_range_leaf::DirectRangeLeafState::new_owned(
                    digit_witness,
                    &equality_point,
                    plan.basis(),
                    live_block_count,
                    high_variable_count,
                    low_variable_count,
                )?;
                let (sumcheck, stage1_point, _final_claim) = leaf_stage
                    .prove::<F, T, _>(transcript, |tr| {
                        sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND)
                    })?;
                let range_image_eval = leaf_stage.final_range_image_eval();
                let proof = AkitaStage1Proof {
                    stages: vec![AkitaStage1StageProof {
                        sumcheck_proof: sumcheck,
                        child_claims: Vec::new(),
                    }],
                    s_claim: range_image_eval,
                };
                return Ok((proof, stage1_point));
            }
            DigitRangeState::PaddedRangeImage(range_image) => range_image,
        };

        let leaf_coeffs = plan.leaf_coeffs::<E>();
        let leaf_tables = {
            let _span = tracing::info_span!(
                "digit_range_materialize_leaf_tables",
                leaf_count = plan.leaf_factor_count(),
                domain_len = range_image.len(),
            )
            .entered();
            build_leaf_tables(plan, &leaf_coeffs, &range_image)
        };
        let product_layers = {
            let _span = tracing::info_span!("digit_range_build_product_layers").entered();
            build_product_stage_layers(leaf_tables, plan.product_stage_arities())
        };
        let mut stage_proofs = Vec::with_capacity(product_layers.len() + 1);
        let mut current_tau = equality_point;
        let mut current_claim = E::zero();
        let mut current_weights = vec![E::one()];

        for (stage_index, (&arity, layer)) in plan
            .product_stage_arities()
            .iter()
            .zip(product_layers)
            .enumerate()
        {
            let _stage_span =
                tracing::info_span!("digit_range_product_stage", stage_index, arity,).entered();
            let mut product_stage = ProductStageState::new(
                layer.child_tables_by_parent,
                current_weights,
                &current_tau,
                current_claim,
            )?;
            let (sumcheck, next_tau, _final_claim) = product_stage
                .prove::<F, T, _>(transcript, |tr| {
                    sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND)
                })?;
            let true_child_claims = product_stage.final_child_claims();
            let child_claims = true_child_claims;
            stage_proofs.push(AkitaStage1StageProof {
                sumcheck_proof: sumcheck,
                child_claims: child_claims.clone(),
            });

            absorb_child_claims::<F, E, T>(&child_claims, transcript);
            let gamma = sample_ext_challenge::<F, E, T>(
                transcript,
                labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
            );
            current_weights = plan.interstage_batch_weights(gamma, child_claims.len());
            current_claim = plan.batch_claims(&current_weights, &child_claims);
            current_tau = next_tau;
        }

        let batched_leaf_coeffs = plan.batch_leaf_polynomials(&current_weights, &leaf_coeffs);
        let _leaf_span = tracing::info_span!("digit_range_polynomial_leaf").entered();
        let mut leaf_stage = PolynomialLeafState::new(
            range_image,
            &current_tau,
            current_claim,
            batched_leaf_coeffs,
        )?;
        let (leaf_sumcheck, stage1_point, _leaf_final_claim) = leaf_stage
            .prove::<F, T, _>(transcript, |tr| {
                sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND)
            })?;
        stage_proofs.push(AkitaStage1StageProof {
            sumcheck_proof: leaf_sumcheck,
            child_claims: Vec::new(),
        });

        let range_image_eval = leaf_stage.final_range_image_eval();
        let proof = AkitaStage1Proof {
            stages: stage_proofs,
            s_claim: range_image_eval,
        };
        Ok((proof, stage1_point))
    }
}
