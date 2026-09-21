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

mod class_indexed_product;
pub(in crate::protocol::sumcheck) mod class_indexed_range_leaf;
mod class_indexed_state;
mod compact_digit_source;
pub(crate) mod direct_range_leaf;
pub(in crate::protocol::sumcheck) mod exact_prefix;
mod range_class_tables;
mod round_accumulation;

pub use direct_range_leaf::LowBasisRangeCheckProver;

use crate::backend::packed_digits::PackedSignedDigits;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{
    DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain, PhysicalResponsePlan,
};
use class_indexed_product::ClassIndexedProductSubcheckProver;
use class_indexed_range_leaf::ClassIndexedRangeLeafProver;
use compact_digit_source::CompactDigitSource;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};
use jolt_field::{Fold, Unreduced};
pub(in crate::protocol) struct NativeDigitRangeProveOutput<E: Field> {
    pub(in crate::protocol) point: Vec<E>,
    pub(in crate::protocol) range_image_evaluation: E,
    pub(in crate::protocol) physical_l2: Option<super::physical_l2_norm::NativePhysicalL2Proof<E>>,
}

const MAX_TREE_STAGE_Q_DEGREE: usize = 4;
const MAX_QUARTET_TABLE_CLASS_COUNT: usize = 8;

struct ProductSubcheckInput<'a, E: Field> {
    source: CompactDigitSource,
    plan: DigitRangePlan,
    leaf_polynomials: &'a [Vec<E>],
    stage_index: usize,
    parent_weights: Vec<E>,
    equality_point: &'a [E],
    input_claim: E,
}

fn prove_class_indexed_product_subcheck_native<F, E, const LANES: usize>(
    input: ProductSubcheckInput<'_, E>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    stage_index: u32,
) -> Result<(Vec<E>, Vec<E>), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Ring + Fold + Unreduced + AkitaSerialize,
{
    let mut stage = ClassIndexedProductSubcheckProver::<E, LANES>::new(
        input.source,
        input.plan,
        input.leaf_polynomials,
        input.stage_index,
        input.parent_weights,
        input.equality_point,
        input.input_claim,
    )?;
    let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
        grinding,
        akita_types::SumcheckProtocol::Stage1,
        level,
        stage_index,
    );
    let (next_equality_point, _) = akita_sumcheck::prove_eq_factored_sumcheck_native::<F, E, _, _>(
        &mut stage,
        &mut channel,
        0,
    )?;
    Ok((stage.final_child_claims(), next_equality_point))
}

struct NativeProductPrefix<E: Field> {
    digit_source: CompactDigitSource,
    plan: DigitRangePlan,
    leaf_coeffs: Vec<Vec<E>>,
    equality_point: Vec<E>,
    claim: E,
    weights: Vec<E>,
    stage_count: usize,
}

fn prove_product_prefix_native<F, E>(
    digit_source: CompactDigitSource,
    plan: DigitRangePlan,
    equality_point: Vec<E>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
) -> Result<NativeProductPrefix<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Ring + Fold + Unreduced + AkitaSerialize,
{
    let leaf_coeffs = plan.leaf_coeffs::<E>();
    let mut current_equality_point = equality_point;
    let mut current_claim = E::zero();
    let mut current_weights = vec![E::one()];
    for (stage_index, &_arity) in plan.product_stage_arities().iter().enumerate() {
        let lane_count = plan
            .product_stage_lane_count(stage_index)
            .ok_or(AkitaError::InvalidProof)?;
        let product_input = ProductSubcheckInput {
            source: digit_source.clone(),
            plan,
            leaf_polynomials: &leaf_coeffs,
            stage_index,
            parent_weights: current_weights,
            equality_point: &current_equality_point,
            input_claim: current_claim,
        };
        let stage = u32::try_from(stage_index)
            .map_err(|_| AkitaError::InvalidSetup("Stage 1 index exceeds u32".into()))?;
        let (child_claims, next_equality_point) = match lane_count {
            2 => prove_class_indexed_product_subcheck_native::<F, E, 2>(
                product_input,
                grinding,
                level,
                stage,
            )?,
            4 => prove_class_indexed_product_subcheck_native::<F, E, 4>(
                product_input,
                grinding,
                level,
                stage,
            )?,
            8 => prove_class_indexed_product_subcheck_native::<F, E, 8>(
                product_input,
                grinding,
                level,
                stage,
            )?,
            _ => return Err(AkitaError::InvalidProof),
        };
        akita_types::native_stage1_prover_child_claims::<F, E>(
            grinding,
            level,
            stage,
            &child_claims,
        )?;
        let gamma = grinding.grinded_ext_challenge::<F, E>(
            akita_types::GrindingSite::Stage1InterstageBatch { level, stage },
        )?;
        current_weights = plan.interstage_batch_weights(gamma, child_claims.len());
        current_claim = plan.batch_claims(&current_weights, &child_claims)?;
        current_equality_point = next_equality_point;
    }
    Ok(NativeProductPrefix {
        digit_source,
        plan,
        leaf_coeffs,
        equality_point: current_equality_point,
        claim: current_claim,
        weights: current_weights,
        stage_count: plan.product_stage_arities().len(),
    })
}

fn compose_small_poly_with_affine<E: Field>(coeffs: &[E], offset: E, slope: E) -> [E; 5] {
    debug_assert!(coeffs.len() <= MAX_TREE_STAGE_Q_DEGREE + 1);
    let [constant, linear, quadratic, cubic, quartic] = match coeffs {
        [] => return [E::zero(); 5],
        [c0] => return [*c0, E::zero(), E::zero(), E::zero(), E::zero()],
        [c0, c1] => {
            return [
                *c0 + *c1 * offset,
                *c1 * slope,
                E::zero(),
                E::zero(),
                E::zero(),
            ]
        }
        [c0, c1, c2] => [*c0, *c1, *c2, E::zero(), E::zero()],
        [c0, c1, c2, c3] => [*c0, *c1, *c2, *c3, E::zero()],
        [c0, c1, c2, c3, c4] => [*c0, *c1, *c2, *c3, *c4],
        _ => unreachable!("range polynomial degree is at most four"),
    };

    let two_quadratic = quadratic + quadratic;
    let three_cubic = cubic + cubic + cubic;
    let four_quartic = (quartic + quartic) + (quartic + quartic);
    let six_quartic = four_quartic + quartic + quartic;

    let value =
        constant + offset * (linear + offset * (quadratic + offset * (cubic + offset * quartic)));
    let first_derivative =
        linear + offset * (two_quadratic + offset * (three_cubic + offset * four_quartic));
    let second_divided_derivative = quadratic + offset * (three_cubic + offset * six_quartic);
    let third_divided_derivative = cubic + offset * four_quartic;
    let slope_squared = slope * slope;

    [
        value,
        slope * first_derivative,
        slope_squared * second_divided_derivative,
        slope_squared * slope * third_divided_derivative,
        slope_squared * slope_squared * quartic,
    ]
}

/// Stage-1 range-check prover, including the root/leaf tree choreography.
pub struct DigitRangeProver<E: Field> {
    digit_source: CompactDigitSource,
    equality_point: Vec<E>,
    plan: DigitRangePlan,
    live_block_count: usize,
    high_variable_count: usize,
    low_variable_count: usize,
}

impl<E: Field + Ring> DigitRangeProver<E> {
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
        Self::from_packed_digits(
            PackedSignedDigits::from_i8_digits_auto(digit_witness.as_ref().to_vec()),
            plan,
            domain,
            equality_point,
        )
    }

    pub(crate) fn from_packed_digits(
        digit_witness: PackedSignedDigits,
        plan: DigitRangePlan,
        domain: FlatBooleanDomain,
        equality_point: DigitRangeEqualityPoint<E>,
    ) -> Result<Self, AkitaError> {
        equality_point.validate_domain(domain)?;
        let low_variable_count = equality_point.low_variable_count();
        let high_variable_count = domain.num_vars() - low_variable_count;
        let live_block_count = domain.live_block_count(low_variable_count)?;
        let coordinates = equality_point.into_coordinates();
        let digit_source = {
            let _span = tracing::info_span!(
                "digit_range_prepare_compact_source",
                basis = plan.basis(),
                live_len = domain.live_len(),
                domain_len = domain.domain_len(),
            )
            .entered();
            CompactDigitSource::new(digit_witness, domain, plan)?
        };
        Ok(Self {
            digit_source,
            equality_point: coordinates,
            plan,
            live_block_count,
            high_variable_count,
            low_variable_count,
        })
    }
}

impl<E: Field + Ring + Unreduced + Fold + AkitaSerialize> DigitRangeProver<E> {
    /// Stream the non-L2 stage-1 range proof directly into Spongefish.
    pub(in crate::protocol) fn prove_native<F>(
        self,
        grinding: &mut akita_types::NativeProverGrinding<'_>,
        physical_plan: Option<&PhysicalResponsePlan>,
        level: u32,
    ) -> Result<NativeDigitRangeProveOutput<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let Self {
            mut digit_source,
            equality_point,
            plan,
            live_block_count,
            high_variable_count,
            low_variable_count,
        } = self;
        if physical_plan.is_none() && plan.basis() <= 8 {
            let mut leaf_stage = direct_range_leaf::LowBasisRangeCheckProver::new(
                digit_source.digits(),
                &equality_point,
                plan,
                live_block_count,
                high_variable_count,
                low_variable_count,
            )?;
            let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
                grinding,
                akita_types::SumcheckProtocol::Stage1,
                level,
                0,
            );
            let (point, _) = akita_sumcheck::prove_eq_factored_sumcheck_native::<F, E, _, _>(
                &mut leaf_stage,
                &mut channel,
                0,
            )?;
            let range_image_evaluation = leaf_stage.final_range_image_eval();
            akita_types::native_stage1_prover_range_image::<F, E>(
                grinding,
                level,
                0,
                range_image_evaluation,
            )?;
            return Ok(NativeDigitRangeProveOutput {
                point,
                range_image_evaluation,
                physical_l2: None,
            });
        }

        if physical_plan.is_some() {
            digit_source.prepare_class_indexed_leaf();
        }

        let prefix = prove_product_prefix_native::<F, E>(
            digit_source,
            plan,
            equality_point,
            grinding,
            level,
        )?;
        let batched_leaf_coeffs = prefix
            .plan
            .batch_leaf_polynomials(&prefix.weights, &prefix.leaf_coeffs)?;
        if let Some(physical_plan) = physical_plan {
            let compact_witness = prefix.digit_source.digits();
            let range_leaf = ClassIndexedRangeLeafProver::new(
                prefix.digit_source,
                &prefix.equality_point,
                prefix.claim,
                batched_leaf_coeffs,
            )?;
            let (physical_l2, point, range_image_evaluation) =
                super::physical_l2_norm::prove_physical_l2_norm_native::<F, E>(
                    physical_plan,
                    &compact_witness,
                    range_leaf,
                    grinding,
                    level,
                )?;
            let stage = u32::try_from(prefix.stage_count)
                .map_err(|_| AkitaError::InvalidSetup("Stage 1 index exceeds u32".into()))?;
            akita_types::native_stage1_prover_range_image::<F, E>(
                grinding,
                level,
                stage,
                range_image_evaluation,
            )?;
            return Ok(NativeDigitRangeProveOutput {
                point,
                range_image_evaluation,
                physical_l2: Some(physical_l2),
            });
        }
        let mut leaf_stage = ClassIndexedRangeLeafProver::new(
            prefix.digit_source,
            &prefix.equality_point,
            prefix.claim,
            batched_leaf_coeffs,
        )?;
        let stage = u32::try_from(prefix.stage_count)
            .map_err(|_| AkitaError::InvalidSetup("Stage 1 index exceeds u32".into()))?;
        let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
            grinding,
            akita_types::SumcheckProtocol::Stage1,
            level,
            stage,
        );
        let (point, _) = akita_sumcheck::prove_eq_factored_sumcheck_native::<F, E, _, _>(
            &mut leaf_stage,
            &mut channel,
            0,
        )?;
        let range_image_evaluation = leaf_stage.final_range_image_eval();
        akita_types::native_stage1_prover_range_image::<F, E>(
            grinding,
            level,
            stage,
            range_image_evaluation,
        )?;
        Ok(NativeDigitRangeProveOutput {
            point,
            range_image_evaluation,
            physical_l2: None,
        })
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;
    use akita_transcript::new_native_prover;
    use akita_types::{GrindingPlan, GrindingRun, GrindingSite, SumcheckProtocol};
    use jolt_field::Prime128Offset275 as F;

    #[test]
    fn native_direct_leaf_streams_range_image_after_rounds() {
        let level = 0;
        let num_vars = 3usize;
        let range_plan = DigitRangePlan::new(4).unwrap();
        let domain = FlatBooleanDomain::new(8, num_vars).unwrap();
        let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
            &[F::from_u64(2), F::from_u64(3), F::from_u64(5)],
            2,
            1,
        )
        .unwrap();
        let prover = DigitRangeProver::new(
            std::sync::Arc::from([0, 1, -1, 1, 0, -1, 1, 0]),
            range_plan,
            domain,
            equality_point,
        )
        .unwrap();
        let runs = (0..num_vars)
            .map(|round| {
                GrindingRun::proof_of_work(
                    GrindingSite::SumcheckRound {
                        protocol: SumcheckProtocol::Stage1,
                        level,
                        stage: 0,
                        round: u32::try_from(round).unwrap(),
                    },
                    1,
                    128,
                )
                .unwrap()
            })
            .collect();
        let grinding_plan = GrindingPlan::new(runs, 128).unwrap();
        let state = new_native_prover(b"native-stage1-prover", b"fixture").unwrap();
        let mut grinding = akita_types::NativeProverGrinding::new(state, &grinding_plan);
        let output = prover
            .prove_native::<F>(&mut grinding, None, level)
            .unwrap();
        let proof = grinding.finish().unwrap();

        assert_eq!(output.point.len(), num_vars);
        assert!(!proof.is_empty());
    }
}
