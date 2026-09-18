//! Physical-norm addend for the existing final Stage-1 range leaf.

use super::digit_range::class_indexed_range_leaf::ClassIndexedRangeLeafProver;
use super::digit_range::exact_prefix::ExactPrefixTable;
use akita_algebra::UniPoly;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, SumcheckInstanceProver};
use akita_types::{reconstruct_l2_sq_from_gram, PhysicalL2NormProofShape, PhysicalResponsePlan};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};
use jolt_field::{Fold, Unreduced};

const RANGE_Q_MAX_DEGREE: usize = 4;
const FUSED_MAX_DEGREE: usize = RANGE_Q_MAX_DEGREE + 1;
const NORM_MAX_DEGREE: usize = 3;

#[allow(dead_code)] // Consumed by the native fold driver during production cutover.
pub(in crate::protocol) struct NativePhysicalL2Proof<E: Field> {
    pub(in crate::protocol) response_l2_sq: u128,
    pub(in crate::protocol) virtual_evaluations: Vec<E>,
}

trait PhysicalL2ProverStream<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    type Proof;

    fn prefix(&mut self, response_l2_sq: u128, subclaims: &[E]) -> Result<(), AkitaError>;
    fn subclaim_batch_challenge(&mut self) -> Result<E, AkitaError>;
    fn norm_merge_challenge(&mut self) -> Result<E, AkitaError>;
    fn prove_sumcheck<P>(&mut self, prover: &mut P) -> Result<(Self::Proof, Vec<E>, E), AkitaError>
    where
        P: SumcheckInstanceProver<E> + ?Sized;
    fn virtual_evaluations(&mut self, evaluations: &[E]) -> Result<(), AkitaError>;
    fn finish_proof(proof: Self::Proof, evaluations: Vec<E>) -> Self::Proof;
}

struct NativePhysicalL2ProverStream<'a, 'plan> {
    grinding: &'a mut akita_types::NativeProverGrinding<'plan>,
    level: u32,
    response_l2_sq: u128,
}

impl<F, E> PhysicalL2ProverStream<F, E> for NativePhysicalL2ProverStream<'_, '_>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    type Proof = NativePhysicalL2Proof<E>;

    fn prefix(&mut self, response_l2_sq: u128, subclaims: &[E]) -> Result<(), AkitaError> {
        self.response_l2_sq = response_l2_sq;
        akita_types::native_l2_prover_prefix::<F, E>(
            self.grinding,
            self.level,
            response_l2_sq,
            subclaims,
        )
    }

    fn subclaim_batch_challenge(&mut self) -> Result<E, AkitaError> {
        self.grinding
            .grinded_ext_challenge::<F, E>(akita_types::GrindingSite::L2SubclaimBatch {
                level: self.level,
            })
    }

    fn norm_merge_challenge(&mut self) -> Result<E, AkitaError> {
        self.grinding
            .grinded_ext_challenge::<F, E>(akita_types::GrindingSite::L2NormMerge {
                level: self.level,
            })
    }

    fn prove_sumcheck<P>(&mut self, prover: &mut P) -> Result<(Self::Proof, Vec<E>, E), AkitaError>
    where
        P: SumcheckInstanceProver<E> + ?Sized,
    {
        let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
            self.grinding,
            akita_types::SumcheckProtocol::PhysicalL2,
            self.level,
            0,
        );
        let (point, claim) =
            akita_sumcheck::prove_sumcheck_native::<F, E, _, _>(prover, &mut channel, 0)?;
        Ok((
            NativePhysicalL2Proof {
                response_l2_sq: self.response_l2_sq,
                virtual_evaluations: Vec::new(),
            },
            point,
            claim,
        ))
    }

    fn virtual_evaluations(&mut self, evaluations: &[E]) -> Result<(), AkitaError> {
        akita_types::native_l2_prover_virtual_evaluations::<F, E>(
            self.grinding,
            self.level,
            evaluations,
        )
    }

    fn finish_proof(mut proof: Self::Proof, evaluations: Vec<E>) -> Self::Proof {
        proof.virtual_evaluations = evaluations;
        proof
    }
}

enum PhysicalNormTerm<E: Field> {
    Direct {
        response: ExactPrefixTable<E>,
    },
    LimbGram {
        limbs: Vec<ExactPrefixTable<E>>,
        selectors: Vec<ExactPrefixTable<E>>,
        pairs: Vec<(usize, usize)>,
    },
}

impl<E: Field + Fold> PhysicalNormTerm<E> {
    #[inline(always)]
    fn affine_pair(table: &ExactPrefixTable<E>, pair_index: usize) -> (E, E) {
        let left = table.value_or_default(2 * pair_index);
        (left, table.value_or_default(2 * pair_index + 1) - left)
    }

    fn round_coefficients(&self) -> [E; NORM_MAX_DEGREE + 1] {
        match self {
            Self::Direct { response } => cfg_fold_reduce!(
                0..response.explicit_len().div_ceil(2),
                || [E::zero(); NORM_MAX_DEGREE + 1],
                |mut sum, pair_index| {
                    let (value, delta) = Self::affine_pair(response, pair_index);
                    sum[0] += value * value;
                    sum[1] += (value + value) * delta;
                    sum[2] += delta * delta;
                    sum
                },
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                }
            ),
            Self::LimbGram {
                limbs,
                selectors,
                pairs,
            } => {
                let pair_count = limbs
                    .first()
                    .map_or(0, |table| table.explicit_len().div_ceil(2));
                cfg_fold_reduce!(
                    0..pair_count,
                    || [E::zero(); NORM_MAX_DEGREE + 1],
                    |mut sum, pair_index| {
                        for (selector, &(left, right)) in selectors.iter().zip(pairs) {
                            let (selector, selector_delta) =
                                Self::affine_pair(selector, pair_index);
                            let (left, left_delta) = Self::affine_pair(&limbs[left], pair_index);
                            let (right, right_delta) = Self::affine_pair(&limbs[right], pair_index);
                            let product_constant = left * right;
                            let product_linear = left * right_delta + left_delta * right;
                            let product_quadratic = left_delta * right_delta;
                            sum[0] += selector * product_constant;
                            sum[1] += selector * product_linear + selector_delta * product_constant;
                            sum[2] +=
                                selector * product_quadratic + selector_delta * product_linear;
                            sum[3] += selector_delta * product_quadratic;
                        }
                        sum
                    },
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    }
                )
            }
        }
    }

    fn bind(&mut self, challenge: E) -> Result<(), AkitaError> {
        let context = E::precompute(challenge);
        let fold = |left, right| E::fold_one(&context, left, right);
        match self {
            Self::Direct { response } => response.fold_in_place(fold),
            Self::LimbGram {
                limbs, selectors, ..
            } => {
                for table in limbs.iter_mut().chain(selectors) {
                    table.fold_in_place(fold)?;
                }
                Ok(())
            }
        }
    }

    fn virtual_evaluations(&self) -> Result<Vec<E>, AkitaError> {
        match self {
            Self::Direct { response } => response
                .final_value()
                .map(|value| vec![value])
                .ok_or(AkitaError::InvalidProof),
            Self::LimbGram { limbs, .. } => limbs
                .iter()
                .map(|table| table.final_value().ok_or(AkitaError::InvalidProof))
                .collect(),
        }
    }

    fn final_claim(&self) -> Result<E, AkitaError> {
        match self {
            Self::Direct { response } => {
                let value = response.final_value().ok_or(AkitaError::InvalidProof)?;
                Ok(value * value)
            }
            Self::LimbGram {
                limbs,
                selectors,
                pairs,
            } => selectors.iter().zip(pairs).try_fold(
                E::zero(),
                |sum, (selector, &(left, right))| {
                    let selector = selector.final_value().ok_or(AkitaError::InvalidProof)?;
                    let left = limbs
                        .get(left)
                        .and_then(ExactPrefixTable::final_value)
                        .ok_or(AkitaError::InvalidProof)?;
                    let right = limbs
                        .get(right)
                        .and_then(ExactPrefixTable::final_value)
                        .ok_or(AkitaError::InvalidProof)?;
                    Ok(sum + selector * left * right)
                },
            ),
        }
    }
}

struct FusedRangeNormProver<E: Field> {
    range: ClassIndexedRangeLeafProver<E>,
    norm: PhysicalNormTerm<E>,
    norm_merge: E,
    input_claim: E,
    rounds_completed: usize,
}

impl<E: Field + Ring + Fold + Unreduced> SumcheckInstanceProver<E> for FusedRangeNormProver<E> {
    fn num_rounds(&self) -> usize {
        EqFactoredSumcheckInstanceProver::num_rounds(&self.range)
    }

    fn degree_bound(&self) -> usize {
        EqFactoredSumcheckInstanceProver::degree_bound(&self.range) + 1
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        debug_assert_eq!(round, self.rounds_completed);
        let q_coefficients = self.range.round_q_coefficients(round);
        let (factor_at_zero, factor_at_one) = self.range.current_full_eq_factor_evals();
        let factor_delta = factor_at_one - factor_at_zero;
        let mut coefficients = [E::zero(); FUSED_MAX_DEGREE + 1];
        for (degree, coefficient) in q_coefficients
            .into_iter()
            .take(EqFactoredSumcheckInstanceProver::degree_bound(&self.range) + 1)
            .enumerate()
        {
            coefficients[degree] += factor_at_zero * coefficient;
            coefficients[degree + 1] += factor_delta * coefficient;
        }
        for (destination, coefficient) in
            coefficients.iter_mut().zip(self.norm.round_coefficients())
        {
            *destination += self.norm_merge * coefficient;
        }
        UniPoly::from_coeffs(coefficients[..=self.degree_bound()].to_vec())
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        debug_assert_eq!(round, self.rounds_completed);
        self.range.ingest_challenge(round, challenge);
        self.norm
            .bind(challenge)
            .expect("validated physical norm prefix can fold");
        self.rounds_completed += 1;
    }
}

fn exact_claims<E: Field + Ring>(
    plan: &PhysicalResponsePlan,
    integers: &[Vec<i128>],
) -> Result<(u128, Vec<E>), AkitaError> {
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

fn prepare_norm_term<E: Field + Ring>(
    plan: &PhysicalResponsePlan,
    integers: Vec<Vec<i128>>,
    subclaim_weights: &[E],
) -> Result<PhysicalNormTerm<E>, AkitaError> {
    let domain_len = plan.domain().domain_len();
    let tables = integers
        .into_iter()
        .map(|values| {
            ExactPrefixTable::new(
                domain_len,
                values.into_iter().map(E::from_i128).collect(),
                E::zero(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => {
            let mut tables = tables.into_iter();
            let response = tables.next().ok_or(AkitaError::InvalidProof)?;
            if tables.next().is_some() || !subclaim_weights.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            Ok(PhysicalNormTerm::Direct { response })
        }
        shape @ PhysicalL2NormProofShape::LimbGram { .. } => {
            let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
            if subclaim_weights.len() != layout.subclaim_count() {
                return Err(AkitaError::InvalidSize {
                    expected: layout.subclaim_count(),
                    actual: subclaim_weights.len(),
                });
            }
            let mut selectors = Vec::with_capacity(layout.pair_count());
            for (left, right) in layout.limb_pairs() {
                let mut values = Vec::with_capacity(layout.physical_response_len());
                for (block_index, block) in layout.block_ranges().enumerate() {
                    let weight_index = layout
                        .subclaim_index(block_index, left, right)
                        .ok_or(AkitaError::InvalidProof)?;
                    let weight = *subclaim_weights
                        .get(weight_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    values.resize(block.end, weight);
                }
                selectors.push(ExactPrefixTable::new(domain_len, values, E::zero())?);
            }
            Ok(PhysicalNormTerm::LimbGram {
                limbs: tables,
                selectors,
                pairs: layout.limb_pairs().collect(),
            })
        }
    }
}

pub(in crate::protocol::sumcheck) fn prove_physical_l2_norm_native<F, E>(
    plan: &PhysicalResponsePlan,
    compact_witness: &crate::backend::packed_digits::PackedSignedDigits,
    range: ClassIndexedRangeLeafProver<E>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
) -> Result<(NativePhysicalL2Proof<E>, Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Ring + Fold + Unreduced + AkitaSerialize,
{
    let mut stream = NativePhysicalL2ProverStream {
        grinding,
        level,
        response_l2_sq: 0,
    };
    prove_physical_l2_norm_with_stream::<F, E, _>(plan, compact_witness, range, &mut stream)
}

fn prove_physical_l2_norm_with_stream<F, E, S>(
    plan: &PhysicalResponsePlan,
    compact_witness: &crate::backend::packed_digits::PackedSignedDigits,
    range: ClassIndexedRangeLeafProver<E>,
    stream: &mut S,
) -> Result<(S::Proof, Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Ring + Fold + Unreduced + AkitaSerialize,
    S: PhysicalL2ProverStream<F, E>,
{
    if EqFactoredSumcheckInstanceProver::num_rounds(&range) != plan.domain().num_vars() {
        return Err(AkitaError::InvalidSetup(
            "fused Stage-1 leaf has inconsistent range geometry".into(),
        ));
    }
    let integers = plan.materialize_virtual_integers(compact_witness.len(), |start, output| {
        compact_witness.view().decode_range(start, output)?;
        Ok(())
    })?;
    let (response_l2_sq, subclaims) = exact_claims::<E>(plan, &integers)?;
    stream.prefix(response_l2_sq, &subclaims)?;

    let (norm_input_claim, subclaim_weights) = match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => (E::from_u128(response_l2_sq), Vec::new()),
        PhysicalL2NormProofShape::LimbGram { .. } => {
            let gamma = stream.subclaim_batch_challenge()?;
            let mut power = E::one();
            let mut weights = Vec::with_capacity(subclaims.len());
            let mut claim = E::zero();
            for &subclaim in &subclaims {
                weights.push(power);
                claim += power * subclaim;
                power *= gamma;
            }
            (claim, weights)
        }
    };
    let norm = prepare_norm_term(plan, integers, &subclaim_weights)?;
    let norm_merge = stream.norm_merge_challenge()?;
    let range_input_claim = EqFactoredSumcheckInstanceProver::input_claim(&range);
    let mut prover = FusedRangeNormProver {
        range,
        norm,
        norm_merge,
        input_claim: range_input_claim + norm_merge * norm_input_claim,
        rounds_completed: 0,
    };
    let (proof, point, final_claim) = stream.prove_sumcheck(&mut prover)?;
    let expected_final_claim =
        prover.range.final_range_claim() + norm_merge * prover.norm.final_claim()?;
    if final_claim != expected_final_claim {
        return Err(AkitaError::InvalidInput(
            "fused range/norm prover final claim mismatch".into(),
        ));
    }
    let range_image_evaluation = prover.range.final_range_image_eval();
    let virtual_evaluations = prover.norm.virtual_evaluations()?;
    stream.virtual_evaluations(&virtual_evaluations)?;
    Ok((
        S::finish_proof(proof, virtual_evaluations),
        point,
        range_image_evaluation,
    ))
}
