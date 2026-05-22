//! Stage-1 verifier instances for Akita range-check proofs.
//!
//! This module owns verifier-side replay for both the compact single-stage
//! `b <= 8` path and the staged range-check tree used for larger bases. The
//! prover-side compact witness scans and two-round-prefix kernels stay in the
//! prover/root path.

use akita_algebra::split_eq::GruenSplitEq;
use akita_algebra::CyclotomicRing;
use akita_challenges::{
    sample_folding_challenges, stage1_fold_challenge_labels, FoldingChallenges,
};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{verify_eq_factored_sumcheck, EqFactoredSumcheckInstanceVerifier};
use akita_transcript::labels::{self, ABSORB_PROVER_V};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::{
    combine_polys, eval_poly, linear_combination, range_check_eval_from_s,
    stage1_interstage_batch_weights, stage1_leaf_coeffs, stage1_stage_count,
    stage1_tree_product_stage_arities, validate_stage1_tree_basis, AkitaStage1Proof, LevelParams,
    MRowLayout, RingSliceSerializer,
};

/// Absorb the prover's `v` rows and sample the stage-1 fold challenges in the
/// shape declared by `lp.fold_challenge_shape` (flat per-block or tensor
/// product of two `√num_blocks`-length factors).
///
/// # Errors
///
/// Returns an error if challenge sampling fails (e.g. shape validation,
/// non-power-of-two block count in the tensor case, or count overflow).
pub(crate) fn derive_stage1_challenges<F, T, const D: usize>(
    transcript: &mut T,
    v: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    num_claims: usize,
    lp: &LevelParams,
    m_row_layout: MRowLayout,
) -> Result<FoldingChallenges, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    // Terminal layout drops the D-block (`v = D · ŵ`) from M entirely;
    // `v` never travels on the wire, so the absorb must be skipped on
    // both prover and verifier to keep the Fiat-Shamir transcript in
    // sync. Intermediate layouts still bind the prover's `v` rows.
    if matches!(m_row_layout, MRowLayout::Intermediate) {
        transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(v));
    }
    sample_folding_challenges::<F, T, D>(
        transcript,
        num_blocks,
        num_claims,
        &lp.stage1_config,
        &lp.fold_challenge_shape,
        stage1_fold_challenge_labels(),
    )
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

    fn start_round_state(&self) -> Result<Self::RoundState, AkitaError> {
        GruenSplitEq::new(&self.tau0)
    }

    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[F],
    ) -> Result<F, AkitaError> {
        Ok(round_state.current_scalar() * range_check_eval_from_s(self.s_claim, self.b))
    }
}

struct ProductStageVerifier<E: FieldCore> {
    tau: Vec<E>,
    input_claim: E,
    child_claims: Vec<E>,
    batch_weights: Vec<E>,
    arity: usize,
}

impl<E: FieldCore> ProductStageVerifier<E> {
    fn new(
        tau: Vec<E>,
        input_claim: E,
        child_claims: Vec<E>,
        batch_weights: Vec<E>,
        arity: usize,
    ) -> Self {
        debug_assert!(matches!(arity, 2 | 4));
        debug_assert_eq!(child_claims.len(), batch_weights.len() * arity);
        Self {
            tau,
            input_claim,
            child_claims,
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

    fn start_round_state(&self) -> Result<Self::RoundState, AkitaError> {
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

struct PolynomialStageVerifier<E: FieldCore> {
    tau: Vec<E>,
    input_claim: E,
    poly_coeffs: Vec<E>,
    s_claim: E,
}

impl<E: FieldCore> PolynomialStageVerifier<E> {
    fn new(tau: Vec<E>, input_claim: E, poly_coeffs: Vec<E>, s_claim: E) -> Self {
        Self {
            tau,
            input_claim,
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

    fn start_round_state(&self) -> Result<Self::RoundState, AkitaError> {
        GruenSplitEq::new(&self.tau)
    }

    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        _challenges: &[E],
    ) -> Result<E, AkitaError> {
        Ok(round_state.current_scalar() * eval_poly(&self.poly_coeffs, self.s_claim))
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

impl<E: FieldCore + FromPrimitiveInt + AkitaSerialize> AkitaStage1Verifier<E> {
    /// Verify the full stage-1 tree proof and return the final `r_stage1`.
    ///
    /// # Errors
    ///
    /// Returns an error if the staged proof shape is inconsistent with `b`, if
    /// any internal stage sumcheck fails, or if the final oracle check fails.
    pub fn verify<F, T>(
        &self,
        proof: &AkitaStage1Proof<E>,
        transcript: &mut T,
    ) -> Result<Vec<E>, AkitaError>
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
            return verify_eq_factored_sumcheck::<F, _, E, _, _>(
                &proof.stages[0].sumcheck,
                &leaf_verifier,
                transcript,
                |tr| sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND),
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

            let product_verifier = ProductStageVerifier::new(
                current_tau,
                current_claim,
                stage_proof.child_claims.clone(),
                current_weights,
                arity,
            );
            current_tau = verify_eq_factored_sumcheck::<F, _, E, _, _>(
                &stage_proof.sumcheck,
                &product_verifier,
                transcript,
                |tr| sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND),
            )?;

            absorb_child_claims::<F, E, T>(&stage_proof.child_claims, transcript);
            let gamma = sample_ext_challenge::<F, E, T>(
                transcript,
                labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
            );
            current_weights =
                stage1_interstage_batch_weights(gamma, stage_proof.child_claims.len());
            current_claim = linear_combination(&current_weights, &stage_proof.child_claims);
        }

        let batched_leaf_coeffs = combine_polys(&current_weights, &leaf_coeffs);
        let leaf_verifier = PolynomialStageVerifier::new(
            current_tau,
            current_claim,
            batched_leaf_coeffs,
            proof.s_claim,
        );
        verify_eq_factored_sumcheck::<F, _, E, _, _>(
            &leaf_stage_proof.sumcheck,
            &leaf_verifier,
            transcript,
            |tr| sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND),
        )
    }
}
