//! Stage-1 verifier instances for Akita range-check proofs.
//!
//! This module owns verifier-side replay for both the compact single-stage
//! `b <= 8` path and the staged range-check tree used for larger bases. The
//! prover-side compact witness scans and two-round-prefix kernels stay in the
//! prover/root path.

use akita_algebra::split_eq::GruenSplitEq;
use akita_challenges::{witness_fold_challenge_labels, Challenges, FoldDraw, LiveFoldDraw};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{EqFactoredSumcheckInstanceVerifier, EqFactoredSumcheckInstanceVerifierExt};
use akita_transcript::labels::{self, ABSORB_PROVER_V};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::eval_poly;
use akita_types::proof::append_flat_coefficients;
use akita_types::{
    combine_polys, linear_combination, stage1_interstage_batch_weights, stage1_leaf_coeffs,
    stage1_stage_count, stage1_tree_product_stage_arities, validate_stage1_tree_basis,
    AkitaStage1Proof, LevelParams, OpeningClaimsLayout, RelationMatrixRowLayout,
};

type Stage1VerifyOutput<E> = Vec<E>;

/// Absorb the prover's `v` rows once, then sample one [`Challenges`] set per
/// commitment group in `OpeningClaims` order.
///
/// This mirrors the prover's multi-group [`RingRelationProver`] live sampling: the
/// D-block `v = D · concat_g(ê_g)` is absorbed a single time (it spans every
/// group; the terminal layout drops the D-block so the absorb is skipped on
/// both sides), then each group samples with its own `num_live_blocks`/`K_g` under
/// the shared root `fold_challenge_config`, each group's local challenge shape,
/// and the shared
/// accepted grind nonce. A scalar batch (`num_groups == 1`) samples a single
/// `Challenges` set with `lp.num_live_blocks`/`num_total_polynomials`.
///
/// # Errors
///
/// Returns an error if the group layout is malformed or challenge sampling fails.
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_multi_group_stage1_challenges<F, T>(
    transcript: &mut T,
    v_coeffs: &[F],
    v_ring_d: usize,
    challenge_ring_d: usize,
    opening_batch: &OpeningClaimsLayout,
    lp: &LevelParams,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    grind_nonce: u32,
) -> Result<Vec<Challenges>, AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
    T: Transcript<F>,
{
    if matches!(
        relation_matrix_row_layout,
        RelationMatrixRowLayout::WithDBlock
    ) {
        append_flat_coefficients(ABSORB_PROVER_V, v_coeffs, v_ring_d, transcript)?;
    }
    let labels = witness_fold_challenge_labels();
    let mut group_challenges = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let k_g = opening_batch.group_layout(group_index)?.num_polynomials();
        group_challenges.push(
            LiveFoldDraw::<F, T>::new(transcript).draw_folding_challenges(
                challenge_ring_d,
                group_index,
                group_lp.num_live_blocks(),
                k_g,
                &lp.fold_challenge_config,
                &group_lp.fold_challenge_shape(),
                labels,
                grind_nonce,
            )?,
        );
    }
    Ok(group_challenges)
}

struct ProductStageVerifier<E: FieldCore> {
    tau: Vec<E>,
    input_claim: E,
    child_claims: Vec<E>,
    batch_weights: Vec<E>,
    arity: usize,
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
    ) -> Result<Stage1VerifyOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F>,
        T: Transcript<F>,
    {
        validate_stage1_tree_basis(self.b)?;
        let expected_stage_count = stage1_stage_count(self.b);
        if proof.stages.len() != expected_stage_count {
            return Err(AkitaError::InvalidSize {
                expected: expected_stage_count,
                actual: proof.stages.len(),
            });
        }

        let leaf_coeffs = stage1_leaf_coeffs::<E>(self.b);
        let product_stage_arities = if leaf_coeffs.len() == 1 {
            Vec::new()
        } else {
            stage1_tree_product_stage_arities(self.b)
        };
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

            let product_verifier = ProductStageVerifier {
                tau: current_tau,
                input_claim: current_claim,
                child_claims: stage_proof.child_claims.clone(),
                batch_weights: current_weights,
                arity,
            };
            current_tau = product_verifier.verify::<F, T, _>(
                &stage_proof.sumcheck_proof,
                transcript,
                |tr| sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND),
            )?;

            for claim in &stage_proof.child_claims {
                append_ext_field::<F, E, T>(
                    transcript,
                    labels::ABSORB_SUMCHECK_INTERSTAGE_CLAIM,
                    claim,
                );
            }
            let gamma = sample_ext_challenge::<F, E, T>(
                transcript,
                labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
            );
            current_weights =
                stage1_interstage_batch_weights(gamma, stage_proof.child_claims.len());
            current_claim = linear_combination(&current_weights, &stage_proof.child_claims);
        }

        let batched_leaf_coeffs = combine_polys(&current_weights, &leaf_coeffs);
        let leaf_verifier = PolynomialStageVerifier {
            tau: current_tau,
            input_claim: current_claim,
            poly_coeffs: batched_leaf_coeffs,
            s_claim: proof.s_claim,
        };
        leaf_verifier.verify::<F, T, _>(&leaf_stage_proof.sumcheck_proof, transcript, |tr| {
            sample_ext_challenge::<F, E, T>(tr, labels::CHALLENGE_SUMCHECK_ROUND)
        })
    }
}

#[cfg(test)]
mod fold_grind_nonce_tests {
    use super::*;
    use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
    use akita_types::{FoldLinfProtocolBinding, SisModulusProfileId};

    fn sample_level_params(
        fold_challenge_config: SparseChallengeConfig,
        fold_shape: TensorChallengeShape,
    ) -> LevelParams {
        LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            3,
            2,
            4,
            3,
            fold_challenge_config,
        )
        .with_decomp(16, 64, 2, 2, 2)
        .expect("level params")
        .with_fold_challenge_shape(fold_shape)
        .expect("fold challenge shape")
    }

    #[test]
    fn worst_case_beta_only_rejects_nonzero_nonce() {
        let lp = sample_level_params(
            SparseChallengeConfig::pm1_only(31),
            TensorChallengeShape::Tensor { fold_low_len: 2 },
        );
        let contract = lp.fold_witness_grind_contract(1).expect("contract");
        assert_eq!(
            contract.policy,
            akita_types::sis::FoldWitnessLinfCapPolicy::WorstCaseBetaOnly
        );
        let max_grind_attempts = FoldLinfProtocolBinding::CURRENT.max_grind_attempts;
        assert!(contract.validate_nonce(0, max_grind_attempts).is_ok());
        assert!(contract.validate_nonce(1, max_grind_attempts).is_err());
    }

    #[test]
    fn tail_bound_with_grind_accepts_nonce_below_cap() {
        let lp = sample_level_params(
            SparseChallengeConfig {
                count_pm1: 30,
                count_pm2: 12,
            },
            TensorChallengeShape::Flat,
        );
        let contract = lp.fold_witness_grind_contract(1).expect("contract");
        assert_eq!(
            contract.policy,
            akita_types::sis::FoldWitnessLinfCapPolicy::TailBoundWithGrind
        );
        let cap = FoldLinfProtocolBinding::CURRENT.max_grind_attempts;
        assert!(contract.validate_nonce(0, cap).is_ok());
        assert!(contract.validate_nonce(cap - 1, cap).is_ok());
        assert!(contract.validate_nonce(cap, cap).is_err());
    }

    #[test]
    fn tensor_tail_bound_with_grind_accepts_nonce_below_cap() {
        let lp = sample_level_params(
            SparseChallengeConfig {
                count_pm1: 30,
                count_pm2: 12,
            },
            TensorChallengeShape::Tensor { fold_low_len: 2 },
        );
        let contract = lp.fold_witness_grind_contract(1).expect("contract");
        assert_eq!(
            contract.policy,
            akita_types::sis::FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind
        );
        let cap = FoldLinfProtocolBinding::CURRENT.max_grind_attempts;
        assert!(contract.validate_nonce(0, cap).is_ok());
        assert!(contract.validate_nonce(cap - 1, cap).is_ok());
        assert!(contract.validate_nonce(cap, cap).is_err());
    }
}
