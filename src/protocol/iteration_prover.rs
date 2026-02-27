//! Prover for a single Hachi protocol iteration.
//!
//! Assumes the input multilinear polynomial is already reduced to ring form.

use crate::algebra::ring::{CyclotomicRing, SparseChallengeConfig};
use crate::error::HachiError;
use crate::protocol::challenges::sparse::sample_dense_challenges;
use crate::protocol::commitment::utils::linear::mat_vec_mul_unchecked;
use crate::protocol::commitment::utils::norm::{detect_field_modulus, vec_inf_norm};
use crate::protocol::commitment::{CommitmentConfig, RingCommitment, RingCommitmentSetup};
use crate::protocol::commitment_scheme::HachiCommitmentHint;
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::{HachiProof, SumcheckAux};
use crate::protocol::sumcheck::SumcheckProof;
use crate::protocol::transcript::labels::{ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, HachiSerialize};

/// Stateful prover accumulating witness data across protocol stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProver<F: FieldCore, const D: usize> {
    /// Commitment-phase hint carrying `s_i` and `t̂_i`.
    pub hint: HachiCommitmentHint<F, D>,
    /// Decomposed `w` vectors: `ŵ_i = G_1^{-1}(w_i)` for `i ∈ [2^r]`.
    ///
    /// Each inner `Vec` has length `δ`.
    pub w_hat: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed folded witness: `ẑ = J^{-1}(z)`.
    ///
    /// Length: `2^m · δ · τ`.
    pub z_hat: Vec<CyclotomicRing<F, D>>,
}

// ---------------------------------------------------------------------------
// HachiProver implementation
// ---------------------------------------------------------------------------

impl<F: FieldCore, const D: usize> Default for HachiProver<F, D> {
    fn default() -> Self {
        Self {
            hint: HachiCommitmentHint {
                s: Vec::new(),
                t_hat: Vec::new(),
                ring_coeffs: Vec::new(),
            },
            w_hat: Vec::new(),
            z_hat: Vec::new(),
        }
    }
}

impl<F: FieldCore + CanonicalField + HachiSerialize, const D: usize> HachiProver<F, D> {
    /// Run the Hachi prover protocol for one iteration.
    ///
    /// Currently executes stage 1 only. Future stages will be added here.
    ///
    /// # Errors
    ///
    /// Returns an error if any stage fails.
    pub fn prove<T, Cfg>(
        setup: &RingCommitmentSetup<F, D>,
        point: &RingOpeningPoint<F, D>,
        transcript: &mut T,
        hint: &HachiCommitmentHint<F, D>,
    ) -> Result<HachiProof<F, D>, HachiError>
    where
        T: Transcript<F>,
        Cfg: CommitmentConfig,
    {
        let mut prover = Self::new();
        let (v, _challenges) = prover.prove_stage1::<T, Cfg>(setup, point, transcript, hint)?;
        Ok(HachiProof {
            v,
            y_ring: CyclotomicRing::<F, D>::zero(),
            f0_proof: SumcheckProof {
                round_polys: Vec::new(),
            },
            f_alpha_proof: SumcheckProof {
                round_polys: Vec::new(),
            },
            sumcheck_aux: SumcheckAux { w: Vec::new() },
            w_commitment: RingCommitment { u: Vec::new() },
        })
    }

    /// Run §4.2 prover stage 1 (Figure 3, prover side).
    ///
    /// Populates the prover state with `(ŵ, t̂, ẑ)` and returns `(v, challenges)`
    /// where `v = D · ŵ` is the verifier-facing proof and `challenges` are the
    /// stage-1 folding challenges derived from the transcript.
    ///
    /// # Errors
    ///
    /// Returns an error if the norm check fails (`‖z‖_∞ > β`) or challenge
    /// sampling fails.
    pub fn prove_stage1<T, Cfg>(
        &mut self,
        setup: &RingCommitmentSetup<F, D>,
        point: &RingOpeningPoint<F, D>,
        transcript: &mut T,
        hint: &HachiCommitmentHint<F, D>,
    ) -> Result<(Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), HachiError>
    where
        T: Transcript<F>,
        Cfg: CommitmentConfig,
    {
        self.hint = hint.clone();

        // Steps 1–3: w_i = a^T G_{2^m} s_i, then ŵ_i = G_1^{-1}(w_i)
        self.compute_w_hat::<Cfg>(point);

        // Step 4: v = D · ŵ
        let v = self.compute_v(&setup.D);

        // Step 5: append v to transcript (first prover message)
        transcript.append_serde(ABSORB_PROVER_V, &v);

        // Step 6: sample 2^r sparse challenges from transcript
        let challenge_cfg = SparseChallengeConfig {
            weight: Cfg::CHALLENGE_WEIGHT,
            nonzero_coeffs: vec![-1, 1],
        };
        let challenges = sample_dense_challenges::<F, T, D>(
            transcript,
            CHALLENGE_STAGE1_FOLD,
            1usize << Cfg::R,
            &challenge_cfg,
        )?;

        // Steps 7–9: z = Σ c_i · s_i, check ‖z‖_∞ ≤ β, then ẑ = J^{-1}(z)
        self.compute_z_hat::<Cfg>(&challenges)?;

        Ok((v, challenges))
    }

    /// Create a new prover with empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// **Steps 1–3.** Compute `w_i = a^T G_{2^m} s_i` and decompose: `ŵ_i = G_1^{-1}(w_i)`.
    ///
    /// For each block `s_i`, recompose each `δ`-chunk back to a ring element,
    /// take the inner product with `a`, then gadget-decompose the result into
    /// `δ` digits.
    fn compute_w_hat<Cfg>(&mut self, opening_point: &RingOpeningPoint<F, D>)
    where
        Cfg: CommitmentConfig,
    {
        let a = &opening_point.a;
        let block_len = 1usize << Cfg::M;
        let delta = Cfg::DELTA;
        let log_basis = Cfg::LOG_BASIS;

        debug_assert_eq!(a.len(), block_len);

        self.w_hat = self
            .hint
            .s
            .iter()
            .map(|s_i| {
                let mut w_i = CyclotomicRing::<F, D>::zero();
                for (j, a_j) in a.iter().enumerate().take(block_len) {
                    let start = j * delta;
                    let end = start + delta;
                    let recomp_j =
                        CyclotomicRing::gadget_recompose_pow2(&s_i[start..end], log_basis);
                    w_i += *a_j * recomp_j;
                }
                w_i.balanced_decompose_pow2(delta, log_basis)
            })
            .collect();
    }

    /// **Step 4.** Compute `v = D · ŵ` (first prover message).
    #[allow(non_snake_case)]
    fn compute_v(&self, d: &[Vec<CyclotomicRing<F, D>>]) -> Vec<CyclotomicRing<F, D>> {
        let w_hat_flat: Vec<CyclotomicRing<F, D>> =
            self.w_hat.iter().flat_map(|v| v.iter().copied()).collect();
        mat_vec_mul_unchecked(d, &w_hat_flat)
    }

    /// **Steps 7–9.** Fold `z = Σ c_i · s_i`, check `‖z‖_∞ ≤ β`, and decompose `ẑ = J^{-1}(z)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the norm bound is exceeded (prover should abort / retry).
    fn compute_z_hat<Cfg>(&mut self, challenges: &[CyclotomicRing<F, D>]) -> Result<(), HachiError>
    where
        Cfg: CommitmentConfig,
    {
        debug_assert_eq!(challenges.len(), self.hint.s.len());
        let len = self.hint.s[0].len();
        let mut z = vec![CyclotomicRing::<F, D>::zero(); len];
        for (c_i, s_i) in challenges.iter().zip(self.hint.s.iter()) {
            for (z_j, s_ij) in z.iter_mut().zip(s_i.iter()) {
                *z_j += *c_i * *s_ij;
            }
        }

        let modulus = detect_field_modulus::<F>();
        let norm = vec_inf_norm(&z, modulus);
        if norm > Cfg::BETA {
            return Err(HachiError::InvalidInput(format!(
                "prover abort: ||z||_inf = {norm} > beta = {}",
                Cfg::BETA
            )));
        }

        self.z_hat = z
            .iter()
            .flat_map(|z_j| z_j.balanced_decompose_pow2(Cfg::TAU, Cfg::LOG_BASIS))
            .collect();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::CyclotomicRing;
    use crate::protocol::challenges::sparse::sample_dense_challenges;
    use crate::protocol::commitment::{HachiCommitmentCore, RingCommitmentScheme};
    use crate::protocol::commitment_scheme::HachiCommitmentHint;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::*;
    use crate::Transcript;

    const TRANSCRIPT_SEED: &[u8] = b"test/prover-relation";

    fn replay_challenges(proof: &HachiProof<F, D>) -> Vec<CyclotomicRing<F, D>> {
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_SEED);
        transcript.append_serde(ABSORB_PROVER_V, &proof.v);

        let challenge_cfg = SparseChallengeConfig {
            weight: TinyConfig::CHALLENGE_WEIGHT,
            nonzero_coeffs: vec![-1, 1],
        };
        sample_dense_challenges::<F, Blake2bTranscript<F>, D>(
            &mut transcript,
            CHALLENGE_STAGE1_FOLD,
            NUM_BLOCKS,
            &challenge_cfg,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Shared fixture — driven entirely by HachiProver::prove_stage1()
    // -----------------------------------------------------------------------

    struct Fixture {
        setup: RingCommitmentSetup<F, D>,
        commitment_u: Vec<CyclotomicRing<F, D>>,
        point: RingOpeningPoint<F, D>,
        blocks: Vec<Vec<CyclotomicRing<F, D>>>,
        prover: HachiProver<F, D>,
        proof: HachiProof<F, D>,
        challenges: Vec<CyclotomicRing<F, D>>,
    }

    fn build_fixture() -> Fixture {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();

        let blocks = sample_blocks();
        let (commitment, s, t_hat) =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
                &blocks, &setup,
            )
            .unwrap();

        let point = RingOpeningPoint {
            a: sample_a(),
            b: sample_b(),
        };

        let hint = HachiCommitmentHint {
            s,
            t_hat,
            ring_coeffs: Vec::new(),
        };
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_SEED);
        let mut prover = HachiProver::<F, D>::new();
        let (v, _challenges) = prover
            .prove_stage1::<Blake2bTranscript<F>, TinyConfig>(
                &setup,
                &point,
                &mut transcript,
                &hint,
            )
            .unwrap();
        let proof = HachiProof {
            v,
            y_ring: CyclotomicRing::<F, D>::zero(),
            f0_proof: SumcheckProof {
                round_polys: Vec::new(),
            },
            f_alpha_proof: SumcheckProof {
                round_polys: Vec::new(),
            },
            sumcheck_aux: SumcheckAux { w: Vec::new() },
            w_commitment: RingCommitment { u: Vec::new() },
        };

        let challenges = replay_challenges(&proof);

        Fixture {
            setup,
            commitment_u: commitment.u,
            point,
            blocks,
            prover,
            proof,
            challenges,
        }
    }

    // =======================================================================
    // Row 1:  D · ŵ  =  v
    // =======================================================================

    #[test]
    fn row1_d_times_w_hat_equals_v() {
        let f = build_fixture();

        let w_hat_flat: Vec<CyclotomicRing<F, D>> = f
            .prover
            .w_hat
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let lhs = mat_vec_mul(&f.setup.D, &w_hat_flat);

        assert_eq!(lhs, f.proof.v, "Row 1 failed: D · ŵ ≠ v");
    }

    // =======================================================================
    // Row 2:  B · t̂  =  u  (commitment vector)
    // =======================================================================

    #[test]
    fn row2_b_times_t_hat_equals_u_commitment() {
        let f = build_fixture();

        let t_hat_flat: Vec<CyclotomicRing<F, D>> = f
            .prover
            .hint
            .t_hat
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let lhs = mat_vec_mul(&f.setup.B, &t_hat_flat);

        assert_eq!(lhs, f.commitment_u, "Row 2 failed: B · t̂ ≠ u");
    }

    // =======================================================================
    // Row 3:  b^T · G_{2^r} · ŵ  =  u_eval
    // =======================================================================

    #[test]
    fn row3_bt_gadget_w_hat_equals_u_eval() {
        let f = build_fixture();

        let w_recomposed: Vec<CyclotomicRing<F, D>> = f
            .prover
            .w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2(w_hat_i, LOG_BASIS))
            .collect();

        let u_eval = w_recomposed
            .iter()
            .zip(f.point.b.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (w_i, b_i)| {
                acc + (*b_i * *w_i)
            });

        let u_eval_direct = f.blocks.iter().zip(f.point.b.iter()).fold(
            CyclotomicRing::<F, D>::zero(),
            |acc, (block_i, b_i)| {
                let inner: CyclotomicRing<F, D> = block_i
                    .iter()
                    .zip(f.point.a.iter())
                    .fold(CyclotomicRing::<F, D>::zero(), |acc2, (f_ij, a_j)| {
                        acc2 + (*a_j * *f_ij)
                    });
                acc + (*b_i * inner)
            },
        );

        assert_eq!(
            u_eval, u_eval_direct,
            "Row 3 failed: b^T G ŵ ≠ Σ b_i (a^T f_i)"
        );
    }

    // =======================================================================
    // Row 4:  (c^T ⊗ G_1) · ŵ  =  a^T · G_{2^m} · J · ẑ
    // =======================================================================

    #[test]
    fn row4_challenge_fold_w_equals_a_gadget_j_z_hat() {
        let f = build_fixture();

        let w: Vec<CyclotomicRing<F, D>> = f
            .prover
            .w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2(w_hat_i, LOG_BASIS))
            .collect();

        let lhs = f
            .challenges
            .iter()
            .zip(w.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (c_i, w_i)| {
                acc + (*c_i * *w_i)
            });

        let z_recovered = recompose_z_hat(&f.prover.z_hat);
        let rhs = a_transpose_gadget_times_vec(&f.point.a, &z_recovered);

        assert_eq!(lhs, rhs, "Row 4 failed: (c^T ⊗ G_1)ŵ ≠ a^T G J ẑ");
    }

    // =======================================================================
    // Row 5:  (c^T ⊗ G_{n_A}) · t̂  =  A · J · ẑ
    // =======================================================================

    #[test]
    fn row5_challenge_fold_t_equals_a_j_z_hat() {
        let f = build_fixture();

        let mut lhs = vec![CyclotomicRing::<F, D>::zero(); N_A];
        for (c_i, t_hat_i) in f.challenges.iter().zip(f.prover.hint.t_hat.iter()) {
            let t_i = gadget_recompose_vec(t_hat_i);
            assert_eq!(t_i.len(), N_A);
            for (lhs_j, t_ij) in lhs.iter_mut().zip(t_i.iter()) {
                *lhs_j += *c_i * *t_ij;
            }
        }

        let z_recovered = recompose_z_hat(&f.prover.z_hat);
        let rhs = mat_vec_mul(&f.setup.A, &z_recovered);

        assert_eq!(lhs, rhs, "Row 5 failed: (c^T ⊗ G_nA)t̂ ≠ A · J · ẑ");
    }

    // =======================================================================
    // Shape sanity
    // =======================================================================

    #[test]
    fn prove_output_shapes_are_correct() {
        let f = build_fixture();

        assert_eq!(f.proof.v.len(), TinyConfig::N_D);

        assert_eq!(f.prover.w_hat.len(), NUM_BLOCKS);
        assert!(f.prover.w_hat.iter().all(|v| v.len() == DELTA));

        assert_eq!(f.prover.hint.t_hat.len(), NUM_BLOCKS);
        assert!(f.prover.hint.t_hat.iter().all(|v| v.len() == N_A * DELTA));

        assert_eq!(f.prover.z_hat.len(), BLOCK_LEN * DELTA * TAU);
    }
}
