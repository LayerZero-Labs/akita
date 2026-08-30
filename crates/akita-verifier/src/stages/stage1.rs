//! Stage-1 verifier instances for Akita range-check proofs.
//!
//! This module owns verifier-side replay for both the compact single-stage
//! `b <= 8` path and the staged range-check tree used for larger bases. The
//! prover-side compact witness scans and two-round-prefix kernels stay in the
//! prover/root path.

use akita_algebra::split_eq::GruenSplitEq;
use akita_challenges::LiveFoldDraw;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{EqFactoredSumcheckInstanceVerifier, EqFactoredSumcheckInstanceVerifierExt};
use akita_transcript::labels;
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    append_digit_range_child_claims, draw_group_fold_challenges, AkitaStage1Proof,
    CommittedGroupParams, DigitRangeEqualityPoint, DigitRangePlan, GroupFoldChallenges,
    OpeningClaimsLayout,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

type DigitRangeVerifyOutput<E> = Vec<E>;

pub(crate) struct RangeLeafVerifierInput<E: Field> {
    pub(crate) equality_point: Vec<E>,
    pub(crate) input_claim: E,
    pub(crate) polynomial_coefficients: Vec<E>,
}

/// Absorb the prover's `v` rows once, then sample one
/// [`akita_challenges::Challenges`] set per
/// commitment group in `OpeningClaims` order.
///
/// This mirrors the prover's multi-group [`RingRelationProver`] live sampling: the
/// D-block `v = D · concat_g(ê_g)` is absorbed a single time (it spans every
/// group; the terminal layout drops the D-block so the absorb is skipped on
/// both sides), then each group samples with its own `num_live_blocks`/`K_g` under
/// each group's native fold-challenge config and the shared
/// accepted grind nonce. A scalar batch (`num_groups == 1`) samples a single
/// [`akita_challenges::Challenges`] set with
/// `lp.blocks.live_blocks`/`num_total_polynomials`.
///
/// # Errors
///
/// Returns an error if the group layout is malformed or challenge sampling fails.
pub(crate) fn derive_multi_group_stage1_challenges<F, E, T>(
    transcript: &mut T,
    opening_batch: &OpeningClaimsLayout,
    lp: &CommittedGroupParams,
    grind_nonce: u32,
) -> Result<Vec<GroupFoldChallenges>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: ExtField<F>,
    T: Transcript<F>,
{
    let mut group_challenges = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let k_g = opening_batch.group_layout(group_index)?.num_polynomials();
        let drawn = draw_group_fold_challenges::<F, E, _>(
            &mut LiveFoldDraw::<F, T>::new(transcript),
            &group_lp,
            group_index,
            k_g,
            grind_nonce,
        )?;
        group_challenges.push(drawn);
    }
    Ok(group_challenges)
}

struct ProductSubcheckVerifier<'a, E: Field> {
    equality_point: Vec<E>,
    input_claim: E,
    child_claims: &'a [E],
    batch_weights: Vec<E>,
    arity: usize,
}

impl<E: Field> EqFactoredSumcheckInstanceVerifier<E> for ProductSubcheckVerifier<'_, E> {
    type RoundState = GruenSplitEq<E>;

    fn num_rounds(&self) -> usize {
        self.equality_point.len()
    }

    fn degree_bound(&self) -> usize {
        self.arity
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn start_round_state(&self) -> Result<Self::RoundState, AkitaError> {
        GruenSplitEq::new(&self.equality_point)
    }

    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[E],
    ) -> Result<E, AkitaError> {
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

struct RangePolynomialLeafVerifier<E: Field> {
    plan: DigitRangePlan,
    equality_point: Vec<E>,
    input_claim: E,
    poly_coeffs: Vec<E>,
    range_image_evaluation: E,
}

impl<E: Field> EqFactoredSumcheckInstanceVerifier<E> for RangePolynomialLeafVerifier<E> {
    type RoundState = GruenSplitEq<E>;

    fn num_rounds(&self) -> usize {
        self.equality_point.len()
    }

    fn degree_bound(&self) -> usize {
        self.poly_coeffs.len().saturating_sub(1)
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn start_round_state(&self) -> Result<Self::RoundState, AkitaError> {
        GruenSplitEq::new(&self.equality_point)
    }

    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[E],
    ) -> Result<E, AkitaError> {
        Ok(round_state.current_scalar()
            * self
                .plan
                .evaluate_leaf_polynomial(&self.poly_coeffs, self.range_image_evaluation))
    }
}

/// Stage-1 range-check verifier, including the root/leaf tree choreography.
pub struct AkitaStage1Verifier<E: Field> {
    equality_point: DigitRangeEqualityPoint<E>,
    plan: DigitRangePlan,
}

impl<E: Field> AkitaStage1Verifier<E> {
    /// Construct the stage-1 verifier from a checked range topology.
    pub fn new(equality_point: DigitRangeEqualityPoint<E>, plan: DigitRangePlan) -> Self {
        Self {
            equality_point,
            plan,
        }
    }
}

impl<E: Field + Ring + AkitaSerialize> AkitaStage1Verifier<E> {
    pub(crate) fn verify_product_prefix<F, T>(
        &self,
        product_stage_proofs: &[akita_types::AkitaStage1StageProof<E>],
        transcript: &mut T,
    ) -> Result<RangeLeafVerifierInput<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
        T: Transcript<F>,
    {
        let product_stage_arities = self.plan.product_stage_arities();
        if product_stage_proofs.len() != product_stage_arities.len() {
            return Err(AkitaError::InvalidSize {
                expected: product_stage_arities.len(),
                actual: product_stage_proofs.len(),
            });
        }
        let rounds = self.equality_point.coordinates().len();
        for (stage_index, stage) in product_stage_proofs.iter().enumerate() {
            let expected = self
                .plan
                .stage_shape(rounds, stage_index)
                .ok_or(AkitaError::InvalidProof)?;
            if stage.sumcheck_proof.round_polys.len() != expected.sumcheck_proof.0
                || stage.child_claims.len() != expected.child_claims
                || stage
                    .sumcheck_proof
                    .round_polys
                    .iter()
                    .any(|round| round.coeffs_except_linear_term.len() != expected.sumcheck_proof.1)
            {
                return Err(AkitaError::InvalidProof);
            }
        }

        let leaf_coeffs = self.plan.leaf_coeffs::<E>();
        let mut current_equality_point = self.equality_point.coordinates().to_vec();
        let mut current_claim = E::zero();
        let mut current_weights = vec![E::one()];
        for (&arity, stage_proof) in product_stage_arities
            .iter()
            .zip(product_stage_proofs.iter())
        {
            let product_verifier = ProductSubcheckVerifier {
                equality_point: current_equality_point,
                input_claim: current_claim,
                child_claims: &stage_proof.child_claims,
                batch_weights: current_weights,
                arity,
            };
            current_equality_point = product_verifier.verify::<F, T, _>(
                &stage_proof.sumcheck_proof,
                transcript,
                |tr| sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND),
            )?;
            append_digit_range_child_claims::<F, E, T>(&stage_proof.child_claims, transcript);
            let gamma = sample_ext_challenge::<F, E, T>(
                transcript,
                labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
            );
            current_weights = self
                .plan
                .interstage_batch_weights(gamma, stage_proof.child_claims.len());
            current_claim = self
                .plan
                .batch_claims(&current_weights, &stage_proof.child_claims)?;
        }
        Ok(RangeLeafVerifierInput {
            equality_point: current_equality_point,
            input_claim: current_claim,
            polynomial_coefficients: self
                .plan
                .batch_leaf_polynomials(&current_weights, &leaf_coeffs)?,
        })
    }

    /// Verify the full stage-1 tree proof and return the final `stage1_point`.
    ///
    /// # Errors
    ///
    /// Returns an error if the staged proof shape is inconsistent with `b`, if
    /// any internal stage sumcheck fails, or if the final oracle check fails.
    pub fn verify<F, T>(
        &self,
        proof: &AkitaStage1Proof<E>,
        transcript: &mut T,
    ) -> Result<DigitRangeVerifyOutput<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
        T: Transcript<F>,
    {
        self.plan
            .validate_proof_shape(proof, self.equality_point.coordinates().len())?;

        let product_stage_arities = self.plan.product_stage_arities();
        let Some((leaf_stage_proof, product_stage_proofs)) = proof.stages.split_last() else {
            return Err(AkitaError::InvalidProof);
        };
        debug_assert_eq!(product_stage_proofs.len(), product_stage_arities.len());
        let leaf = self.verify_product_prefix::<F, T>(product_stage_proofs, transcript)?;
        let leaf_verifier = RangePolynomialLeafVerifier {
            plan: self.plan,
            equality_point: leaf.equality_point,
            input_claim: leaf.input_claim,
            poly_coeffs: leaf.polynomial_coefficients,
            range_image_evaluation: proof.range_image_evaluation,
        };
        leaf_verifier.verify::<F, T, _>(&leaf_stage_proof.sumcheck_proof, transcript, |tr| {
            sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND)
        })
    }
}
