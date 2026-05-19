//! Stage-1 verifier instances for Akita range-check proofs.
//!
//! This module owns verifier-side replay for both the compact single-stage
//! `b <= 8` path and the staged range-check tree used for larger bases. The
//! prover-side compact witness scans and two-round-prefix kernels stay in the
//! prover/root path.

use akita_algebra::split_eq::GruenSplitEq;
use akita_algebra::CyclotomicRing;
use akita_challenges::sample_sparse_challenges;
use akita_challenges::SparseChallenge;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
#[cfg(not(feature = "zk"))]
use akita_sumcheck::verify_eq_factored_sumcheck;
use akita_sumcheck::EqFactoredSumcheckInstanceVerifier;
#[cfg(feature = "zk")]
use akita_sumcheck::EqFactoredUniPoly;
#[cfg(feature = "zk")]
use akita_sumcheck::ZkRelationAccumulator;
#[cfg(feature = "zk")]
use akita_sumcheck::{
    ZkEqFactoredFinalRelation, ZkEqFactoredSumcheckInstanceVerifierExt, ZkR1csLinearCombination,
    ZkR1csVariable,
};
use akita_transcript::labels::{self, ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use akita_transcript::Transcript;
use akita_types::{
    absorb_interstage_claims, combine_polys, linear_combination, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    validate_stage1_tree_basis, AkitaStage1Proof, LevelParams, RingSliceSerializer,
};
#[cfg(not(feature = "zk"))]
use akita_types::{eval_poly, range_check_eval_from_s};

#[cfg(feature = "zk")]
type Stage1VerifyOutput<E> = (Vec<E>, ZkR1csLinearCombination<E>);

#[cfg(not(feature = "zk"))]
type Stage1VerifyOutput<E> = Vec<E>;

/// Absorb the prover's `v` rows and sample the sparse stage-1 fold challenges.
///
/// # Errors
///
/// Returns an error if sparse challenge sampling fails.
pub fn derive_stage1_challenges<F, T, const D: usize>(
    transcript: &mut T,
    v: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    lp: &LevelParams,
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(v));
    sample_sparse_challenges::<F, T, D>(
        transcript,
        CHALLENGE_STAGE1_FOLD,
        num_blocks,
        &lp.stage1_config,
    )
}

#[cfg(feature = "zk")]
fn zk_child_claim_product<E: FieldCore>(
    child_claims: &[E],
    child_claim_masks: &[ZkR1csLinearCombination<E>],
    relations: &mut ZkRelationAccumulator<E>,
) -> Result<ZkR1csLinearCombination<E>, AkitaError> {
    debug_assert_eq!(child_claims.len(), child_claim_masks.len());
    let mut child_claim_lcs = child_claims
        .iter()
        .zip(child_claim_masks.iter())
        .map(|(&claim, mask)| ZkRelationAccumulator::unmask_lc(claim, mask))
        .collect::<Vec<_>>()
        .into_iter();
    let Some(mut acc) = child_claim_lcs.next() else {
        return Ok(ZkR1csLinearCombination::one());
    };
    for next in child_claim_lcs {
        acc = relations.new_auxilary("stage-1 child claim product", acc, next)?;
    }
    Ok(acc)
}

#[cfg(feature = "zk")]
fn zk_record_polynomial_eval<E: FieldCore>(
    description: &'static str,
    coeffs: &[E],
    x_lc: ZkR1csLinearCombination<E>,
    relations: &mut ZkRelationAccumulator<E>,
) -> Result<ZkR1csLinearCombination<E>, AkitaError> {
    let Some((&highest_coeff, remaining_coeffs)) = coeffs.split_last() else {
        return Ok(ZkR1csLinearCombination::zero());
    };
    let mut acc = ZkR1csLinearCombination::constant(highest_coeff);
    for &coeff in remaining_coeffs.iter().rev() {
        let product = relations.new_auxilary(description, acc, x_lc.clone())?;
        let mut next = ZkR1csLinearCombination::constant(coeff);
        next.add_scaled(E::one(), &product);
        acc = next;
    }
    Ok(acc)
}

struct SingleStageVerifier<F: FieldCore> {
    tau0: Vec<F>,
    s_claim: F,
    b: usize,
}

impl<F: FieldCore> SingleStageVerifier<F> {
    fn new(tau0: Vec<F>, s_claim: F, b: usize) -> Self {
        Self { tau0, s_claim, b }
    }
}

impl<F: FieldCore + FromPrimitiveInt> EqFactoredSumcheckInstanceVerifier<F>
    for SingleStageVerifier<F>
{
    type RoundState = GruenSplitEq<F>;

    fn num_rounds(&self) -> usize {
        self.tau0.len()
    }

    fn degree_bound(&self) -> usize {
        self.b / 2
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn start_round_state(&self) -> Self::RoundState {
        GruenSplitEq::new(&self.tau0)
    }

    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[F],
    ) -> Result<F, AkitaError> {
        #[cfg(feature = "zk")]
        {
            let _ = round_state;
            Ok(F::zero())
        }
        #[cfg(not(feature = "zk"))]
        Ok(round_state.current_scalar() * range_check_eval_from_s(self.s_claim, self.b))
    }
}

#[cfg(feature = "zk")]
impl<F: FieldCore + FromPrimitiveInt> ZkEqFactoredFinalRelation<F> for SingleStageVerifier<F> {
    fn record_final_relation(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[F],
        scaled_claim: ZkR1csLinearCombination<F>,
        claim_scale: F,
        handoff_mask: ZkR1csLinearCombination<F>,
        relations: &mut ZkRelationAccumulator<F>,
    ) -> Result<(), AkitaError> {
        let coeffs = stage1_leaf_coeffs::<F>(self.b)
            .into_iter()
            .next()
            .ok_or(AkitaError::InvalidProof)?;
        let s_claim = ZkRelationAccumulator::unmask_lc(self.s_claim, &handoff_mask);
        let range_eval = zk_record_polynomial_eval(
            "stage-1 range polynomial evaluation",
            &coeffs,
            s_claim,
            relations,
        )?;
        relations.push_r1cs(
            "stage-1 final oracle",
            range_eval,
            ZkR1csLinearCombination::constant(claim_scale * round_state.current_scalar()),
            scaled_claim,
        );
        Ok(())
    }
}

struct ProductStageVerifier<E: FieldCore> {
    tau: Vec<E>,
    input_claim: E,
    #[cfg(feature = "zk")]
    input_claim_mask: ZkR1csLinearCombination<E>,
    child_claims: Vec<E>,
    #[cfg(feature = "zk")]
    child_claim_masks: Vec<ZkR1csLinearCombination<E>>,
    batch_weights: Vec<E>,
    arity: usize,
}

impl<E: FieldCore> ProductStageVerifier<E> {
    fn new(
        tau: Vec<E>,
        input_claim: E,
        #[cfg(feature = "zk")] input_claim_mask: ZkR1csLinearCombination<E>,
        child_claims: Vec<E>,
        #[cfg(feature = "zk")] child_claim_masks: Vec<ZkR1csLinearCombination<E>>,
        batch_weights: Vec<E>,
        arity: usize,
    ) -> Self {
        debug_assert!(matches!(arity, 2 | 4));
        debug_assert_eq!(child_claims.len(), batch_weights.len() * arity);
        #[cfg(feature = "zk")]
        debug_assert_eq!(child_claims.len(), child_claim_masks.len());
        Self {
            tau,
            input_claim,
            #[cfg(feature = "zk")]
            input_claim_mask,
            child_claims,
            #[cfg(feature = "zk")]
            child_claim_masks,
            batch_weights,
            arity,
        }
    }
}

impl<E: FieldCore> EqFactoredSumcheckInstanceVerifier<E> for ProductStageVerifier<E> {
    type RoundState = GruenSplitEq<E>;

    fn num_rounds(&self) -> usize {
        self.tau.len()
    }

    fn degree_bound(&self) -> usize {
        self.arity
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

#[cfg(feature = "zk")]
impl<E: FieldCore> ZkEqFactoredFinalRelation<E> for ProductStageVerifier<E> {
    fn initial_claim_mask(&self) -> ZkR1csLinearCombination<E> {
        self.input_claim_mask.clone()
    }

    fn record_final_relation(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[E],
        scaled_claim: ZkR1csLinearCombination<E>,
        claim_scale: E,
        _handoff_mask: ZkR1csLinearCombination<E>,
        relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<(), AkitaError> {
        let mut output = ZkR1csLinearCombination::zero();
        for ((&weight, child_claims), child_claim_masks) in self
            .batch_weights
            .iter()
            .zip(self.child_claims.chunks_exact(self.arity))
            .zip(self.child_claim_masks.chunks_exact(self.arity))
        {
            let product = zk_child_claim_product(child_claims, child_claim_masks, relations)?;
            output.add_scaled(weight, &product);
        }
        // The product stage oracle is a batched product of child claims:
        //
        //   O = sum_parent weight_parent * prod_child true_child_claim.
        //
        // Each public child claim is masked, so `zk_child_claim_product`
        // builds the product from symbolic unmasked values
        // (child_claim_masked - child_claim_mask). The eq-factored driver has
        // already unmasked the final scaled claim into `scaled_claim`, so this
        // row enforces:
        //
        //   O * (claim_scale * current_eq_scalar) = final_scaled_claim.
        relations.push_r1cs(
            "stage-1 product-stage final oracle",
            output,
            ZkR1csLinearCombination::constant(claim_scale * round_state.current_scalar()),
            scaled_claim,
        );
        Ok(())
    }
}

struct PolynomialStageVerifier<E: FieldCore> {
    tau: Vec<E>,
    input_claim: E,
    #[cfg(feature = "zk")]
    input_claim_mask: ZkR1csLinearCombination<E>,
    poly_coeffs: Vec<E>,
    s_claim: E,
}

impl<E: FieldCore> PolynomialStageVerifier<E> {
    fn new(
        tau: Vec<E>,
        input_claim: E,
        #[cfg(feature = "zk")] input_claim_mask: ZkR1csLinearCombination<E>,
        poly_coeffs: Vec<E>,
        s_claim: E,
    ) -> Self {
        Self {
            tau,
            input_claim,
            #[cfg(feature = "zk")]
            input_claim_mask,
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
    ) -> Result<E, AkitaError> {
        #[cfg(feature = "zk")]
        {
            let _ = round_state;
            Ok(E::zero())
        }
        #[cfg(not(feature = "zk"))]
        Ok(round_state.current_scalar() * eval_poly(&self.poly_coeffs, self.s_claim))
    }
}

#[cfg(feature = "zk")]
impl<E: FieldCore> ZkEqFactoredFinalRelation<E> for PolynomialStageVerifier<E> {
    fn initial_claim_mask(&self) -> ZkR1csLinearCombination<E> {
        self.input_claim_mask.clone()
    }

    fn record_final_relation(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[E],
        scaled_claim: ZkR1csLinearCombination<E>,
        claim_scale: E,
        handoff_mask: ZkR1csLinearCombination<E>,
        relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<(), AkitaError> {
        // The leaf stage hands off the folded S-value to Stage 2 as a public
        // masked claim:
        //
        //   s_claim_masked = S(r) + handoff_mask.
        //
        // `handoff_mask` is a symbolic LC over hidden pad coefficients.
        // Unmasking gives the true leaf input S(r), then this final relation
        // enforces:
        //
        //   P(S(r)) * (claim_scale * current_eq_scalar) = final_scaled_claim,
        //
        // where P is the batched leaf/range polynomial.
        let s_claim = ZkRelationAccumulator::unmask_lc(self.s_claim, &handoff_mask);
        let poly_eval = zk_record_polynomial_eval(
            "stage-1 leaf polynomial evaluation",
            &self.poly_coeffs,
            s_claim,
            relations,
        )?;
        relations.push_r1cs(
            "stage-1 leaf final oracle",
            poly_eval,
            ZkR1csLinearCombination::constant(claim_scale * round_state.current_scalar()),
            scaled_claim,
        );
        Ok(())
    }
}

/// Stage-1 range-check verifier, including the root/leaf tree choreography.
pub struct AkitaStage1Verifier<E: FieldCore> {
    tau0: Vec<E>,
    b: usize,
}

impl<E: FieldCore> AkitaStage1Verifier<E> {
    /// Construct the stage-1 verifier from `tau0` and `b`.
    pub fn new(tau0: Vec<E>, b: usize) -> Self {
        Self { tau0, b }
    }
}

impl<E: FieldCore + CanonicalField + FromPrimitiveInt> AkitaStage1Verifier<E> {
    /// Verify the full stage-1 tree proof and return the final `r_stage1`.
    ///
    /// # Errors
    ///
    /// Returns an error if the staged proof shape is inconsistent with `b`, if
    /// any internal stage sumcheck fails, or if the final oracle check fails.
    pub fn verify<T: Transcript<E>>(
        &self,
        proof: &AkitaStage1Proof<E>,
        transcript: &mut T,
        #[cfg(feature = "zk")] relations: &mut ZkRelationAccumulator<E>,
        #[cfg(feature = "zk")] hiding_cursor: &mut usize,
    ) -> Result<Stage1VerifyOutput<E>, AkitaError> {
        validate_stage1_tree_basis(self.b)?;
        let expected_stage_count = stage1_stage_count(self.b);
        if proof.stages.len() != expected_stage_count {
            return Err(AkitaError::InvalidSize {
                expected: expected_stage_count,
                actual: proof.stages.len(),
            });
        }

        let leaf_coeffs = stage1_leaf_coeffs::<E>(self.b);
        if leaf_coeffs.len() == 1 {
            if !proof.stages[0].child_claims.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            let leaf_verifier = SingleStageVerifier::new(self.tau0.clone(), proof.s_claim, self.b);
            #[cfg(feature = "zk")]
            return leaf_verifier.verify_zk::<E, _, _>(
                &proof.stages[0].sumcheck_proof_masked,
                transcript,
                relations,
                hiding_cursor,
                |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
            );
            #[cfg(not(feature = "zk"))]
            return verify_eq_factored_sumcheck::<E, _, E, _, _>(
                &proof.stages[0].sumcheck_proof,
                &leaf_verifier,
                transcript,
                |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
            );
        }

        let product_stage_arities = stage1_tree_product_stage_arities(self.b);
        let Some((leaf_stage_proof, product_stage_proofs)) = proof.stages.split_last() else {
            return Err(AkitaError::InvalidProof);
        };
        if !leaf_stage_proof.child_claims.is_empty() {
            return Err(AkitaError::InvalidProof);
        }

        let mut current_tau = self.tau0.clone();
        let mut current_claim = E::zero();
        #[cfg(feature = "zk")]
        let mut current_claim_mask = ZkR1csLinearCombination::zero();
        let mut current_weights = vec![E::one()];

        for (&arity, stage_proof) in product_stage_arities
            .iter()
            .zip(product_stage_proofs.iter())
        {
            let expected_child_claims = current_weights.len() * arity;
            if stage_proof.child_claims.len() != expected_child_claims {
                return Err(AkitaError::InvalidSize {
                    expected: expected_child_claims,
                    actual: stage_proof.child_claims.len(),
                });
            }

            #[cfg(feature = "zk")]
            let child_claim_masks = {
                let round_mask_count = current_tau.len()
                    * EqFactoredUniPoly::<E>::stored_coeff_count_for_degree(arity);
                let child_mask_start = (*hiding_cursor)
                    .checked_add(round_mask_count)
                    .ok_or(AkitaError::InvalidProof)?;
                (0..expected_child_claims)
                    .map(|idx| {
                        ZkR1csLinearCombination::variable(
                            ZkR1csVariable::HiddenWitness(child_mask_start + idx),
                            E::one(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let product_verifier = ProductStageVerifier::new(
                current_tau,
                current_claim,
                #[cfg(feature = "zk")]
                current_claim_mask.clone(),
                stage_proof.child_claims.clone(),
                #[cfg(feature = "zk")]
                child_claim_masks.clone(),
                current_weights,
                arity,
            );
            #[cfg(feature = "zk")]
            {
                let (next_tau, _stage_handoff_mask) = product_verifier.verify_zk::<E, _, _>(
                    &stage_proof.sumcheck_proof_masked,
                    transcript,
                    relations,
                    hiding_cursor,
                    |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
                )?;
                current_tau = next_tau;
                *hiding_cursor = (*hiding_cursor)
                    .checked_add(expected_child_claims)
                    .ok_or(AkitaError::InvalidProof)?;
            }
            #[cfg(not(feature = "zk"))]
            {
                current_tau = verify_eq_factored_sumcheck::<E, _, E, _, _>(
                    &stage_proof.sumcheck_proof,
                    &product_verifier,
                    transcript,
                    |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
                )?;
            }

            absorb_interstage_claims(&stage_proof.child_claims, transcript);
            let gamma = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH);
            current_weights =
                stage1_interstage_batch_weights(gamma, stage_proof.child_claims.len());
            current_claim = linear_combination(&current_weights, &stage_proof.child_claims);
            #[cfg(feature = "zk")]
            {
                current_claim_mask = ZkR1csLinearCombination::zero();
                for (&weight, mask) in current_weights.iter().zip(child_claim_masks.iter()) {
                    current_claim_mask.constant += weight * mask.constant;
                    current_claim_mask
                        .terms
                        .extend(mask.terms.iter().cloned().map(|term| {
                            akita_sumcheck::ZkR1csTerm {
                                variable: term.variable,
                                coeff: weight * term.coeff,
                            }
                        }));
                }
            }
        }

        let batched_leaf_coeffs = combine_polys(&current_weights, &leaf_coeffs);
        let leaf_verifier = PolynomialStageVerifier::new(
            current_tau,
            current_claim,
            #[cfg(feature = "zk")]
            current_claim_mask,
            batched_leaf_coeffs,
            proof.s_claim,
        );
        #[cfg(feature = "zk")]
        {
            leaf_verifier.verify_zk::<E, _, _>(
                &leaf_stage_proof.sumcheck_proof_masked,
                transcript,
                relations,
                hiding_cursor,
                |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
            )
        }
        #[cfg(not(feature = "zk"))]
        {
            verify_eq_factored_sumcheck::<E, _, E, _, _>(
                &leaf_stage_proof.sumcheck_proof,
                &leaf_verifier,
                transcript,
                |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
            )
        }
    }
}
