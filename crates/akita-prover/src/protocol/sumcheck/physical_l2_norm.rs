//! Fused final Stage-1 leaf for range checking and a physical response norm.

use akita_algebra::{eq_poly::EqPolynomial, UniPoly};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt};
use akita_transcript::labels::{
    ABSORB_L2_NORM_INTEGER, ABSORB_L2_NORM_SUBCLAIM, ABSORB_L2_VIRTUAL_EVALUATION,
    CHALLENGE_L2_NORM_BATCH, CHALLENGE_L2_NORM_MERGE, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    reconstruct_l2_sq_from_gram, PhysicalL2NormProof, PhysicalL2NormProofShape,
    PhysicalResponsePlan,
};

enum NormTerms<E: FieldCore> {
    Direct,
    LimbGram {
        selectors: Vec<Vec<E>>,
        pairs: Vec<(usize, usize)>,
    },
}

struct FusedRangeNormProver<E: FieldCore> {
    equality_table: Vec<E>,
    range_image_table: Vec<E>,
    leaf_coefficients: Vec<E>,
    virtual_tables: Vec<Vec<E>>,
    norm_terms: NormTerms<E>,
    norm_merge: E,
    input_claim: E,
    num_rounds: usize,
    rounds_completed: usize,
}

impl<E: FieldCore + FromPrimitiveInt> FusedRangeNormProver<E> {
    fn affine(left: E, right: E, point: E) -> E {
        left + point * (right - left)
    }

    fn fold_table(table: &mut Vec<E>, challenge: E) {
        let next_len = table.len() / 2;
        for index in 0..next_len {
            table[index] = Self::affine(table[2 * index], table[2 * index + 1], challenge);
        }
        table.truncate(next_len);
    }

    fn evaluate_leaf(&self, value: E) -> E {
        self.leaf_coefficients
            .iter()
            .rev()
            .fold(E::zero(), |acc, &coefficient| acc * value + coefficient)
    }

    fn norm_at_pair(&self, pair_index: usize, point: E) -> E {
        match &self.norm_terms {
            NormTerms::Direct => {
                let table = &self.virtual_tables[0];
                let value = Self::affine(table[2 * pair_index], table[2 * pair_index + 1], point);
                value * value
            }
            NormTerms::LimbGram { selectors, pairs } => {
                selectors
                    .iter()
                    .zip(pairs)
                    .fold(E::zero(), |sum, (selector, &(left, right))| {
                        let left_table = &self.virtual_tables[left];
                        let right_table = &self.virtual_tables[right];
                        let selector = Self::affine(
                            selector[2 * pair_index],
                            selector[2 * pair_index + 1],
                            point,
                        );
                        let left = Self::affine(
                            left_table[2 * pair_index],
                            left_table[2 * pair_index + 1],
                            point,
                        );
                        let right = Self::affine(
                            right_table[2 * pair_index],
                            right_table[2 * pair_index + 1],
                            point,
                        );
                        sum + selector * left * right
                    })
            }
        }
    }

    fn final_values(&self) -> Result<(E, Vec<E>), AkitaError> {
        let range_image = self
            .range_image_table
            .first()
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let virtual_evaluations = self
            .virtual_tables
            .iter()
            .map(|table| table.first().copied().ok_or(AkitaError::InvalidProof))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((range_image, virtual_evaluations))
    }

    fn final_norm(&self) -> Result<E, AkitaError> {
        match &self.norm_terms {
            NormTerms::Direct => {
                let value = self
                    .virtual_tables
                    .first()
                    .and_then(|table| table.first())
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                Ok(value * value)
            }
            NormTerms::LimbGram { selectors, pairs } => selectors.iter().zip(pairs).try_fold(
                E::zero(),
                |sum, (selector, &(left, right))| {
                    let selector = selector.first().copied().ok_or(AkitaError::InvalidProof)?;
                    let left = self
                        .virtual_tables
                        .get(left)
                        .and_then(|table| table.first())
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                    let right = self
                        .virtual_tables
                        .get(right)
                        .and_then(|table| table.first())
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                    Ok(sum + selector * left * right)
                },
            ),
        }
    }

    fn expected_final_claim(&self) -> Result<E, AkitaError> {
        let equality = self
            .equality_table
            .first()
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let (range_image, _) = self.final_values()?;
        Ok(equality * self.evaluate_leaf(range_image) + self.norm_merge * self.final_norm()?)
    }
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceProver<E> for FusedRangeNormProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.leaf_coefficients.len()
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        debug_assert_eq!(round, self.rounds_completed);
        let pair_count = self.range_image_table.len() / 2;
        let evaluations = (0..=self.degree_bound())
            .map(|point| {
                let point = E::from_u64(point as u64);
                (0..pair_count).fold(E::zero(), |sum, pair_index| {
                    let equality = Self::affine(
                        self.equality_table[2 * pair_index],
                        self.equality_table[2 * pair_index + 1],
                        point,
                    );
                    let range_image = Self::affine(
                        self.range_image_table[2 * pair_index],
                        self.range_image_table[2 * pair_index + 1],
                        point,
                    );
                    sum + equality * self.evaluate_leaf(range_image)
                        + self.norm_merge * self.norm_at_pair(pair_index, point)
                })
            })
            .collect::<Vec<_>>();
        UniPoly::from_evals(&evaluations)
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        debug_assert_eq!(round, self.rounds_completed);
        Self::fold_table(&mut self.equality_table, challenge);
        Self::fold_table(&mut self.range_image_table, challenge);
        for table in &mut self.virtual_tables {
            Self::fold_table(table, challenge);
        }
        if let NormTerms::LimbGram { selectors, .. } = &mut self.norm_terms {
            for selector in selectors {
                Self::fold_table(selector, challenge);
            }
        }
        self.rounds_completed += 1;
    }
}

fn exact_claims<E: FieldCore + FromPrimitiveInt>(
    plan: &PhysicalResponsePlan,
    compact_witness: &[i8],
) -> Result<(u128, Vec<E>), AkitaError> {
    let integers = plan.materialize_virtual_integers(compact_witness)?;
    match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => {
            let response = integers.first().ok_or(AkitaError::InvalidProof)?;
            let response_l2_sq = response.iter().try_fold(0u128, |sum, &value| {
                let magnitude = value.unsigned_abs();
                sum.checked_add(magnitude.checked_mul(magnitude).ok_or_else(|| {
                    AkitaError::InvalidInput("physical response square overflow".into())
                })?)
                .ok_or_else(|| AkitaError::InvalidInput("physical response norm overflow".into()))
            })?;
            Ok((response_l2_sq, Vec::new()))
        }
        shape @ PhysicalL2NormProofShape::LimbGram { .. } => {
            let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
            let mut integer_claims = Vec::with_capacity(layout.subclaim_count());
            for block in layout.block_ranges() {
                for (left, right) in layout.limb_pairs() {
                    let left_values = integers.get(left).ok_or(AkitaError::InvalidProof)?;
                    let right_values = integers.get(right).ok_or(AkitaError::InvalidProof)?;
                    let claim = block.clone().try_fold(0i128, |sum, index| {
                        let product = left_values
                            .get(index)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?
                            .checked_mul(
                                right_values
                                    .get(index)
                                    .copied()
                                    .ok_or(AkitaError::InvalidProof)?,
                            )
                            .ok_or_else(|| {
                                AkitaError::InvalidInput("limb product overflow".into())
                            })?;
                        sum.checked_add(product).ok_or_else(|| {
                            AkitaError::InvalidInput("limb inner product overflow".into())
                        })
                    })?;
                    integer_claims.push(claim);
                }
            }
            let response_l2_sq =
                reconstruct_l2_sq_from_gram(plan.shape(), plan.fold_basis(), &integer_claims)?;
            Ok((
                response_l2_sq,
                integer_claims.into_iter().map(E::from_i128).collect(),
            ))
        }
    }
}

fn range_image_table<E: FieldCore + FromPrimitiveInt>(
    compact_witness: &[i8],
    domain_len: usize,
) -> Result<Vec<E>, AkitaError> {
    if compact_witness.len() > domain_len {
        return Err(AkitaError::InvalidSize {
            expected: domain_len,
            actual: compact_witness.len(),
        });
    }
    let mut table = vec![E::zero(); domain_len];
    for (&digit, value) in compact_witness.iter().zip(&mut table) {
        let digit = i64::from(digit);
        *value = E::from_i64(digit * (digit + 1));
    }
    Ok(table)
}

/// Prove the final Stage-1 leaf obtained by batching the range identity and
/// the schedule-selected exact physical norm identity.
// This item is re-exported only by the `test-support` feature. It remains
// public here so the feature-gated re-export can cross the crate boundary.
#[cfg_attr(not(feature = "test-support"), allow(unreachable_pub))]
#[allow(clippy::too_many_arguments)]
pub fn prove_physical_l2_norm<F, E, T>(
    plan: &PhysicalResponsePlan,
    compact_witness: &[i8],
    range_equality_point: &[E],
    range_input_claim: E,
    leaf_coefficients: Vec<E>,
    transcript: &mut T,
) -> Result<(PhysicalL2NormProof<E>, Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    if range_equality_point.len() != plan.domain().num_vars() || leaf_coefficients.len() < 3 {
        return Err(AkitaError::InvalidSetup(
            "fused Stage-1 leaf has inconsistent range geometry".into(),
        ));
    }
    let virtual_tables = plan.materialize_virtual_tables::<E>(compact_witness)?;
    let (response_l2_sq, subclaims) = exact_claims::<E>(plan, compact_witness)?;
    transcript.append_serde(ABSORB_L2_NORM_INTEGER, &response_l2_sq);
    for claim in &subclaims {
        transcript.append_serde(ABSORB_L2_NORM_SUBCLAIM, claim);
    }

    let (norm_terms, norm_input_claim) = match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => {
            (NormTerms::Direct, E::from_u128(response_l2_sq))
        }
        shape @ PhysicalL2NormProofShape::LimbGram { .. } => {
            let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
            let gamma = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_L2_NORM_BATCH);
            let mut powers = Vec::with_capacity(subclaims.len());
            let mut power = E::one();
            for _ in 0..subclaims.len() {
                powers.push(power);
                power *= gamma;
            }
            let input_claim = subclaims
                .iter()
                .zip(&powers)
                .fold(E::zero(), |sum, (&claim, &weight)| sum + weight * claim);
            let mut selectors =
                vec![vec![E::zero(); plan.domain().domain_len()]; layout.pair_count()];
            for (block_index, block) in layout.block_ranges().enumerate() {
                for physical_index in block {
                    for (pair_index, (left, right)) in layout.limb_pairs().enumerate() {
                        let power_index = layout
                            .subclaim_index(block_index, left, right)
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("L2 selector index overflow".into())
                            })?;
                        let value = powers
                            .get(power_index)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?;
                        *selectors
                            .get_mut(pair_index)
                            .and_then(|selector| selector.get_mut(physical_index))
                            .ok_or(AkitaError::InvalidProof)? = value;
                    }
                }
            }
            let pairs = layout.limb_pairs().collect();
            (NormTerms::LimbGram { selectors, pairs }, input_claim)
        }
    };
    let norm_merge = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_L2_NORM_MERGE);
    let mut prover = FusedRangeNormProver {
        equality_table: EqPolynomial::evals(range_equality_point)?,
        range_image_table: range_image_table(compact_witness, plan.domain().domain_len())?,
        leaf_coefficients,
        virtual_tables,
        norm_terms,
        norm_merge,
        input_claim: range_input_claim + norm_merge * norm_input_claim,
        num_rounds: plan.domain().num_vars(),
        rounds_completed: 0,
    };
    let (sumcheck, point, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
        sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    if final_claim != prover.expected_final_claim()? {
        return Err(AkitaError::InvalidInput(
            "fused range/norm prover final claim mismatch".into(),
        ));
    }
    let (range_image_evaluation, virtual_evaluations) = prover.final_values()?;
    for evaluation in &virtual_evaluations {
        transcript.append_serde(ABSORB_L2_VIRTUAL_EVALUATION, evaluation);
    }
    Ok((
        PhysicalL2NormProof {
            response_l2_sq,
            subclaims,
            virtual_evaluations,
            sumcheck,
        },
        point,
        range_image_evaluation,
    ))
}
