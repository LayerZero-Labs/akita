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
pub(crate) mod class_indexed_range_leaf;
mod class_indexed_state;
mod compact_digit_source;
pub(crate) mod direct_range_leaf;
pub(crate) mod exact_prefix;
mod range_class_tables;
mod round_accumulation;
mod session;

pub use direct_range_leaf::LowBasisRangeCheckProver;
pub(crate) use session::DigitRangeSession;

use crate::sources::packed_digits::PackedSignedDigits;
use akita_error::AkitaError;
use akita_types::{DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain};
use class_indexed_product::ClassIndexedProductSubcheckProver;
use class_indexed_range_leaf::ClassIndexedRangeLeafProver;
use compact_digit_source::CompactDigitSource;
use jolt_field::{ExtField, Field, Fold, Ring, Unreduced};

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
    /// Build a standalone prover from a compact digit witness.
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

impl<E> DigitRangeProver<E>
where
    E: Field + Ring + Unreduced + Fold + akita_serialization::AkitaSerialize,
{
    /// Drive the standalone Stage 1 oracle through the same interactive
    /// session used by the recursive protocol.
    ///
    /// The production recursive path drives the session from Akita's fold
    /// orchestration. This convenience entry point is retained for standalone
    /// round-trip tests and benchmarks; the consumer session never receives
    /// the transcript.
    pub fn prove<F, T>(
        self,
        transcript: &mut T,
        physical_plan: Option<&akita_types::PhysicalResponsePlan>,
        level: u32,
    ) -> Result<(akita_types::AkitaStage1Proof<E>, Vec<E>), AkitaError>
    where
        F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
        E: ExtField<F>,
        T: akita_types::ProverTranscriptGrinding<F>,
    {
        if physical_plan.is_some() {
            return Err(AkitaError::InvalidInput(
                "standalone Stage 1 physical-L2 proving is not supported".into(),
            ));
        }
        let range_plan = self.plan;
        let product_count = range_plan.product_stage_arities().len();
        let rounds = self.equality_point.len();
        let mut equality_coordinates = self.equality_point.clone();
        let mut session = DigitRangeSession::new(self, None)?;
        let mut claim = E::zero();
        let mut stages = Vec::with_capacity(product_count + 1);
        for stage_index in 0..product_count {
            transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_CLAIM, &claim);
            let step = crate::opaque::Stage1Step::Product(stage_index);
            let mut round_polys = Vec::with_capacity(rounds);
            let mut challenges = Vec::with_capacity(rounds);
            for (round, &equality_coordinate) in
                equality_coordinates.iter().enumerate().take(rounds)
            {
                let crate::opaque::Stage1RoundPolynomial::EqFactored(polynomial) =
                    session.round_polynomial(step, round, claim)?
                else {
                    return Err(AkitaError::InvalidProof);
                };
                transcript
                    .append_serde(akita_transcript::labels::ABSORB_SUMCHECK_ROUND, &polynomial);
                let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                    transcript,
                    akita_types::SumcheckProtocol::Stage1,
                    level,
                    u32::try_from(stage_index)
                        .map_err(|_| AkitaError::InvalidSetup("Stage 1 index overflow".into()))?,
                    u32::try_from(round)
                        .map_err(|_| AkitaError::InvalidSetup("Stage 1 round overflow".into()))?,
                )?;
                claim = akita_sumcheck::advance_eq_factored_claim(
                    claim,
                    equality_coordinate,
                    &polynomial,
                    challenge,
                );
                session.bind_challenge(step, round, challenge)?;
                round_polys.push(polynomial);
                challenges.push(challenge);
            }
            let crate::opaque::Stage1PublicTransition::ProductChildClaims(child_claims) =
                session.public_transition(step)?
            else {
                return Err(AkitaError::InvalidProof);
            };
            akita_types::append_digit_range_child_claims::<F, E, T>(&child_claims, transcript);
            transcript.grind_query(akita_types::GrindingSite::Stage1InterstageBatch {
                level,
                stage: u32::try_from(stage_index)
                    .map_err(|_| AkitaError::InvalidSetup("Stage 1 index overflow".into()))?,
            })?;
            let gamma = akita_transcript::sample_ext_challenge::<F, E, T>(
                transcript,
                akita_transcript::labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
            );
            let weights = range_plan.interstage_batch_weights(gamma, child_claims.len());
            claim = range_plan.batch_claims(&weights, &child_claims)?;
            session.bind_batch_challenge(
                crate::opaque::Stage1Transition::ProductBatch(stage_index),
                gamma,
            )?;
            equality_coordinates = challenges;
            stages.push(akita_types::AkitaStage1StageProof {
                sumcheck_proof: akita_sumcheck::EqFactoredSumcheckProof { round_polys },
                child_claims,
            });
        }
        transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let step = crate::opaque::Stage1Step::RangeLeaf;
        let mut round_polys = Vec::with_capacity(rounds);
        for (round, &equality_coordinate) in equality_coordinates.iter().enumerate().take(rounds) {
            let crate::opaque::Stage1RoundPolynomial::EqFactored(polynomial) =
                session.round_polynomial(step, round, claim)?
            else {
                return Err(AkitaError::InvalidProof);
            };
            transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_ROUND, &polynomial);
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                transcript,
                akita_types::SumcheckProtocol::Stage1,
                level,
                u32::try_from(product_count)
                    .map_err(|_| AkitaError::InvalidSetup("Stage 1 index overflow".into()))?,
                u32::try_from(round)
                    .map_err(|_| AkitaError::InvalidSetup("Stage 1 round overflow".into()))?,
            )?;
            claim = akita_sumcheck::advance_eq_factored_claim(
                claim,
                equality_coordinate,
                &polynomial,
                challenge,
            );
            session.bind_challenge(step, round, challenge)?;
            round_polys.push(polynomial);
        }
        let crate::opaque::Stage1PublicTransition::Final {
            range_image_evaluation,
            virtual_evaluations,
        } = session.public_transition(step)?
        else {
            return Err(AkitaError::InvalidProof);
        };
        if !virtual_evaluations.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        stages.push(akita_types::AkitaStage1StageProof {
            sumcheck_proof: akita_sumcheck::EqFactoredSumcheckProof { round_polys },
            child_claims: Vec::new(),
        });
        let final_claims = session.finish()?;
        if final_claims.final_claim() != claim {
            return Err(AkitaError::InvalidProof);
        }
        Ok((
            akita_types::AkitaStage1Proof {
                stages,
                range_image_evaluation,
                norm_proof: None,
            },
            final_claims.point().to_vec(),
        ))
    }
}
