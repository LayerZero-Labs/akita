//! General Stage-1 sumcheck for the schedule-selected physical response norm.

use akita_algebra::UniPoly;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt};
use akita_transcript::labels::{
    ABSORB_L2_NORM_INTEGER, ABSORB_L2_NORM_SUBCLAIM, ABSORB_L2_VIRTUAL_EVALUATION,
    CHALLENGE_L2_NORM_BATCH, CHALLENGE_SUMCHECK_ROUND,
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

struct PhysicalL2NormProver<E: FieldCore> {
    virtual_tables: Vec<Vec<E>>,
    terms: NormTerms<E>,
    input_claim: E,
    num_rounds: usize,
    rounds_completed: usize,
}

impl<E: FieldCore + FromPrimitiveInt> PhysicalL2NormProver<E> {
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

    fn final_virtual_evaluations(&self) -> Result<Vec<E>, AkitaError> {
        self.virtual_tables
            .iter()
            .map(|table| table.first().copied().ok_or(AkitaError::InvalidProof))
            .collect()
    }

    fn expected_final_claim(&self) -> Result<E, AkitaError> {
        match &self.terms {
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
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceProver<E> for PhysicalL2NormProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        match self.terms {
            NormTerms::Direct => 2,
            NormTerms::LimbGram { .. } => 3,
        }
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        debug_assert_eq!(round, self.rounds_completed);
        let pair_count = self.virtual_tables[0].len() / 2;
        let evaluations = (0..=self.degree_bound())
            .map(|point| {
                let point = E::from_u64(point as u64);
                match &self.terms {
                    NormTerms::Direct => (0..pair_count).fold(E::zero(), |sum, index| {
                        let table = &self.virtual_tables[0];
                        let value = Self::affine(table[2 * index], table[2 * index + 1], point);
                        sum + value * value
                    }),
                    NormTerms::LimbGram { selectors, pairs } => selectors.iter().zip(pairs).fold(
                        E::zero(),
                        |sum, (selector, &(left, right))| {
                            let left_table = &self.virtual_tables[left];
                            let right_table = &self.virtual_tables[right];
                            (0..pair_count).fold(sum, |inner, index| {
                                let selector = Self::affine(
                                    selector[2 * index],
                                    selector[2 * index + 1],
                                    point,
                                );
                                let left = Self::affine(
                                    left_table[2 * index],
                                    left_table[2 * index + 1],
                                    point,
                                );
                                let right = Self::affine(
                                    right_table[2 * index],
                                    right_table[2 * index + 1],
                                    point,
                                );
                                inner + selector * left * right
                            })
                        },
                    ),
                }
            })
            .collect::<Vec<_>>();
        UniPoly::from_evals(&evaluations)
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        debug_assert_eq!(round, self.rounds_completed);
        for table in &mut self.virtual_tables {
            Self::fold_table(table, challenge);
        }
        if let NormTerms::LimbGram { selectors, .. } = &mut self.terms {
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
        PhysicalL2NormProofShape::LimbGram {
            physical_response_len,
            block_len,
            limb_count,
        } => {
            let mut integer_claims = Vec::new();
            for block_start in (0..physical_response_len).step_by(block_len) {
                let block_end = block_start
                    .checked_add(block_len)
                    .ok_or_else(|| AkitaError::InvalidSetup("L2 block end overflow".into()))?
                    .min(physical_response_len);
                for left in 0..limb_count {
                    for right in left..limb_count {
                        let left_values = integers.get(left).ok_or(AkitaError::InvalidProof)?;
                        let right_values = integers.get(right).ok_or(AkitaError::InvalidProof)?;
                        let claim = (block_start..block_end).try_fold(0i128, |sum, index| {
                            sum.checked_add(
                                left_values
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
                                    })?,
                            )
                            .ok_or_else(|| {
                                AkitaError::InvalidInput("limb inner product overflow".into())
                            })
                        })?;
                        integer_claims.push(claim);
                    }
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

/// Prove the exact physical response norm selected by `plan`.
pub(crate) fn prove_physical_l2_norm<F, E, T>(
    plan: &PhysicalResponsePlan,
    compact_witness: &[i8],
    transcript: &mut T,
) -> Result<(PhysicalL2NormProof<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let virtual_tables = plan.materialize_virtual_tables::<E>(compact_witness)?;
    let (response_l2_sq, subclaims) = exact_claims::<E>(plan, compact_witness)?;
    transcript.append_serde(ABSORB_L2_NORM_INTEGER, &response_l2_sq);
    for claim in &subclaims {
        transcript.append_serde(ABSORB_L2_NORM_SUBCLAIM, claim);
    }

    let (terms, input_claim) = match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => {
            (NormTerms::Direct, E::from_u128(response_l2_sq))
        }
        PhysicalL2NormProofShape::LimbGram {
            physical_response_len,
            block_len,
            limb_count,
        } => {
            let gamma = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_L2_NORM_BATCH);
            let pair_count = limb_count
                .checked_mul(limb_count.checked_add(1).ok_or_else(|| {
                    AkitaError::InvalidSetup("L2 limb-pair count overflow".into())
                })?)
                .and_then(|value| value.checked_div(2))
                .ok_or_else(|| AkitaError::InvalidSetup("L2 limb-pair count overflow".into()))?;
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
            let mut selectors = vec![vec![E::zero(); plan.domain().domain_len()]; pair_count];
            for physical_index in 0..physical_response_len {
                let block = physical_index / block_len;
                for pair in 0..pair_count {
                    let power_index = block
                        .checked_mul(pair_count)
                        .and_then(|base| base.checked_add(pair))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("L2 selector index overflow".into())
                        })?;
                    let value = powers
                        .get(power_index)
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                    let selector = selectors.get_mut(pair).ok_or(AkitaError::InvalidProof)?;
                    let entry = selector
                        .get_mut(physical_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    *entry = value;
                }
            }
            let pairs = (0..limb_count)
                .flat_map(|left| (left..limb_count).map(move |right| (left, right)))
                .collect();
            (NormTerms::LimbGram { selectors, pairs }, input_claim)
        }
    };
    let mut prover = PhysicalL2NormProver {
        virtual_tables,
        terms,
        input_claim,
        num_rounds: plan.domain().num_vars(),
        rounds_completed: 0,
    };
    let (sumcheck, point, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
        sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    if final_claim != prover.expected_final_claim()? {
        return Err(AkitaError::InvalidInput(
            "physical L2 norm prover final claim mismatch".into(),
        ));
    }
    let virtual_evaluations = prover.final_virtual_evaluations()?;
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
    ))
}
