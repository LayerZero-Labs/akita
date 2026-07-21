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
mod class_indexed_range_leaf;
mod class_indexed_state;
mod compact_digit_source;
pub(crate) mod direct_range_leaf;
mod exact_prefix;
mod range_class_tables;
mod round_accumulation;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::EqFactoredSumcheckInstanceProverExt;
use akita_transcript::labels;
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    append_digit_range_child_claims, AkitaStage1Proof, AkitaStage1StageProof,
    DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain,
};
use class_indexed_product::ClassIndexedProductSubcheckProver;
use class_indexed_range_leaf::ClassIndexedRangeLeafProver;
use compact_digit_source::CompactDigitSource;

type DigitRangeProveOutput<E> = (AkitaStage1Proof<E>, Vec<E>);

const MAX_TREE_STAGE_Q_DEGREE: usize = 4;
const MAX_QUARTET_TABLE_CLASS_COUNT: usize = 8;

struct ProductSubcheckInput<'a, E: FieldCore> {
    source: CompactDigitSource,
    plan: DigitRangePlan,
    leaf_polynomials: &'a [Vec<E>],
    stage_index: usize,
    parent_weights: Vec<E>,
    equality_point: &'a [E],
    input_claim: E,
}

fn prove_class_indexed_product_subcheck<F, E, T, const LANES: usize>(
    input: ProductSubcheckInput<'_, E>,
    transcript: &mut T,
) -> Result<(AkitaStage1StageProof<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt + HasOptimizedFold + HasUnreducedOps + AkitaSerialize,
    T: Transcript<F>,
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
    let (sumcheck_proof, next_equality_point, _final_claim) = stage
        .prove::<F, T, _>(transcript, |transcript| {
            sample_ext_challenge::<F, E, T>(transcript, labels::CHALLENGE_SUMCHECK_ROUND)
        })?;
    Ok((
        AkitaStage1StageProof {
            sumcheck_proof,
            child_claims: stage.final_child_claims(),
        },
        next_equality_point,
    ))
}

fn compose_small_poly_with_affine<E: FieldCore>(coeffs: &[E], offset: E, slope: E) -> [E; 5] {
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
pub struct DigitRangeProver<E: FieldCore> {
    digit_source: CompactDigitSource,
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

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold + AkitaSerialize>
    DigitRangeProver<E>
{
    /// Produce the full stage-1 tree proof and return the final `stage1_point`.
    ///
    /// # Errors
    ///
    /// Propagates any transcript or sumcheck failure from the internal root
    /// and leaf-stage proofs.
    pub fn prove<F, T>(self, transcript: &mut T) -> Result<DigitRangeProveOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F>,
        T: Transcript<F>,
    {
        let Self {
            digit_source,
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
        if plan.basis() <= 8 {
            let _leaf_span = tracing::info_span!("digit_range_direct_leaf").entered();
            let mut leaf_stage = direct_range_leaf::LowBasisRangeCheckProver::new(
                digit_source.digits(),
                &equality_point,
                plan,
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
                range_image_evaluation: range_image_eval,
            };
            return Ok((proof, stage1_point));
        }

        let leaf_coeffs = plan.leaf_coeffs::<E>();
        let mut stage_proofs = Vec::with_capacity(plan.stage_count());
        let mut current_equality_point = equality_point;
        let mut current_claim = E::zero();
        let mut current_weights = vec![E::one()];

        for (stage_index, &arity) in plan.product_stage_arities().iter().enumerate() {
            let lane_count = plan
                .product_stage_lane_count(stage_index)
                .ok_or(AkitaError::InvalidProof)?;
            let _stage_span = tracing::info_span!(
                "digit_range_product_substage",
                basis = plan.basis(),
                stage_index,
                arity,
                lane_count,
                live_len = digit_source.live_len(),
                domain_len = digit_source.domain_len(),
            )
            .entered();
            let product_input = ProductSubcheckInput {
                source: digit_source.clone(),
                plan,
                leaf_polynomials: &leaf_coeffs,
                stage_index,
                parent_weights: current_weights,
                equality_point: &current_equality_point,
                input_claim: current_claim,
            };
            let (stage_proof, next_equality_point) = match lane_count {
                2 => prove_class_indexed_product_subcheck::<F, E, T, 2>(product_input, transcript)?,
                4 => prove_class_indexed_product_subcheck::<F, E, T, 4>(product_input, transcript)?,
                8 => prove_class_indexed_product_subcheck::<F, E, T, 8>(product_input, transcript)?,
                _ => return Err(AkitaError::InvalidProof),
            };
            let child_claims = &stage_proof.child_claims;
            append_digit_range_child_claims::<F, E, T>(child_claims, transcript);
            let gamma = sample_ext_challenge::<F, E, T>(
                transcript,
                labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
            );
            current_weights = plan.interstage_batch_weights(gamma, child_claims.len());
            current_claim = plan.batch_claims(&current_weights, child_claims)?;
            current_equality_point = next_equality_point;
            stage_proofs.push(stage_proof);
        }

        let batched_leaf_coeffs = plan.batch_leaf_polynomials(&current_weights, &leaf_coeffs)?;
        let _leaf_span = tracing::info_span!("digit_range_polynomial_leaf").entered();
        let mut leaf_stage = ClassIndexedRangeLeafProver::new(
            digit_source,
            &current_equality_point,
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
            range_image_evaluation: range_image_eval,
        };
        Ok((proof, stage1_point))
    }
}
