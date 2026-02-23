//! Prover stage 1: §4.2 polynomial evaluation as quadratic equation.
//!
//! Implements Figure 3 of the paper (prover side). Given the commitment
//! opening `(s_i, t̂_i)` from §4.1 and an evaluation vector `a`, the prover:
//!
//! 1. Computes `w_i = a^T G_{2^m} s_i` and decomposes into `ŵ_i`.
//! 2. Commits via `v = D · ŵ` (first prover message).
//! 3. Receives sparse challenges `c_i` from the transcript.
//! 4. Folds: `z = Σ c_i · s_i`, checks `‖z‖_∞ ≤ β`, decomposes `ẑ`.
//! 5. Returns the proof `(v, ŵ, t̂, ẑ)`.

use crate::algebra::ring::{CyclotomicRing, SparseChallengeConfig};
use crate::error::HachiError;
use crate::protocol::challenges::sparse::sparse_challenge_from_transcript;
use crate::protocol::commitment::utils::linear::mat_vec_mul_unchecked;
use crate::protocol::commitment::utils::norm::{detect_field_modulus, vec_inf_norm};
use crate::protocol::commitment::{CommitmentConfig, RingCommitmentSetup, RingOpening};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, HachiSerialize};

// ---------------------------------------------------------------------------
// Transcript labels
// ---------------------------------------------------------------------------

const ABSORB_PROVER_V: &[u8] = b"hachi/absorb/prover-stage1-v";
const LABEL_STAGE1_CHALLENGE: &[u8] = b"hachi/challenge/stage1-fold";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parameter bundle for the §4.2 prover (extends `CommitmentConfig`).
pub trait ProverStage1Config: CommitmentConfig {
    /// Decomposition levels for the folded witness `z` (`τ` in the paper).
    const TAU: usize;
    /// L∞ norm bound for `z` (`β` in the paper). Prover aborts if exceeded.
    const BETA: u128;
    /// Hamming weight of sparse challenges (`ω` in the paper).
    const CHALLENGE_WEIGHT: usize;
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Ring-native proof output from §4.2 prover stage 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore, const D: usize> {
    /// First prover message: `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Decomposed `w` vectors: `ŵ_i = G_1^{-1}(w_i)` for `i ∈ [2^r]`.
    ///
    /// Each inner `Vec` has length `δ`.
    pub w_hat: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed inner commitments `t̂_i` from §4.1, for `i ∈ [2^r]`.
    ///
    /// Each inner `Vec` has length `n_A · δ`.
    pub t_hat: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed folded witness: `ẑ = J^{-1}(z)`.
    ///
    /// Length: `2^m · δ · τ`.
    pub z_hat: Vec<CyclotomicRing<F, D>>,
}

// ---------------------------------------------------------------------------
// Step functions
// ---------------------------------------------------------------------------

/// **Step 1–2.** Compute `w_i = a^T G_{2^m} s_i` for each `i ∈ [2^r]`.
///
/// The gadget product is evaluated by recomposing each `δ`-chunk of `s_i`
/// back to a ring element, then taking the inner product with `a`.
pub fn compute_w<F, const D: usize, Cfg>(
    a: &[CyclotomicRing<F, D>],
    s: &[Vec<CyclotomicRing<F, D>>],
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
    Cfg: ProverStage1Config,
{
    let block_len = 1usize << Cfg::M;
    let delta = Cfg::DELTA;
    let log_basis = Cfg::LOG_BASIS;

    debug_assert_eq!(a.len(), block_len);

    s.iter()
        .map(|s_i| {
            let mut w_i = CyclotomicRing::<F, D>::zero();
            for (j, a_j) in a.iter().enumerate().take(block_len) {
                let start = j * delta;
                let end = start + delta;
                let recomp_j = CyclotomicRing::gadget_recompose_pow2(&s_i[start..end], log_basis);
                w_i += *a_j * recomp_j;
            }
            w_i
        })
        .collect()
}

/// **Step 3.** Decompose each `w_i`: `ŵ_i = G_1^{-1}(w_i)`.
///
/// Each scalar ring element is gadget-decomposed into `δ` digits.
pub fn compute_w_hat<F, const D: usize, Cfg>(
    w: &[CyclotomicRing<F, D>],
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField,
    Cfg: ProverStage1Config,
{
    w.iter()
        .map(|w_i| w_i.gadget_decompose_pow2(Cfg::DELTA, Cfg::LOG_BASIS))
        .collect()
}

/// **Step 4.** Compute `v = D · ŵ` (first prover message).
#[allow(non_snake_case)]
pub fn compute_v<F, const D: usize>(
    d: &[Vec<CyclotomicRing<F, D>>],
    w_hat: &[Vec<CyclotomicRing<F, D>>],
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
{
    let w_hat_flat: Vec<CyclotomicRing<F, D>> =
        w_hat.iter().flat_map(|v| v.iter().copied()).collect();
    mat_vec_mul_unchecked(d, &w_hat_flat)
}

/// **Step 7.** Compute `z = Σ c_i · s_i` (challenge-weighted fold).
pub fn compute_z<F, const D: usize>(
    challenges: &[CyclotomicRing<F, D>],
    s: &[Vec<CyclotomicRing<F, D>>],
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
{
    debug_assert_eq!(challenges.len(), s.len());
    let len = s[0].len();
    let mut z = vec![CyclotomicRing::<F, D>::zero(); len];
    for (c_i, s_i) in challenges.iter().zip(s.iter()) {
        for (z_j, s_ij) in z.iter_mut().zip(s_i.iter()) {
            *z_j += *c_i * *s_ij;
        }
    }
    z
}

/// **Step 8.** Check `‖z‖_∞ ≤ β`.
///
/// # Errors
///
/// Returns an error if the bound is exceeded (prover should abort / retry
/// in the interactive setting).
pub fn check_norm_bound<F, const D: usize>(
    z: &[CyclotomicRing<F, D>],
    beta: u128,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField,
{
    let modulus = detect_field_modulus::<F>();
    let norm = vec_inf_norm(z, modulus);
    if norm > beta {
        return Err(HachiError::InvalidInput(format!(
            "prover abort: ||z||_inf = {norm} > beta = {beta}"
        )));
    }
    Ok(())
}

/// **Step 9.** Decompose `z` via balanced gadget decomposition: `ẑ = J^{-1}(z)`.
///
/// Each of the `2^m · δ` ring elements in `z` is decomposed into `τ`
/// balanced base-`b` digits, yielding `2^m · δ · τ` ring elements total.
pub fn compute_z_hat<F, const D: usize, Cfg>(
    z: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
    Cfg: ProverStage1Config,
{
    let modulus = detect_field_modulus::<F>();
    z.iter()
        .flat_map(|z_j| balanced_decompose_ring(z_j, Cfg::TAU, Cfg::LOG_BASIS, modulus))
        .collect()
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Run the full §4.2 prover stage 1 (Figure 3, prover side).
///
/// # Arguments
///
/// * `setup` — unified commitment setup containing matrices `A`, `B`, and `D`.
/// * `opening` — commitment opening `(s_i, t̂_i)` from §4.1.
/// * `a` — evaluation vector of length `2^m` derived from the opening point.
/// * `transcript` — Fiat–Shamir transcript for challenge derivation.
///
/// # Errors
///
/// Returns an error if the norm check fails (`‖z‖_∞ > β`) or challenge
/// sampling fails.
pub fn prove_opening<F, T, const D: usize, Cfg>(
    setup: &RingCommitmentSetup<F, D>,
    opening: &RingOpening<F, D>,
    a: &[CyclotomicRing<F, D>],
    transcript: &mut T,
) -> Result<HachiProof<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + HachiSerialize,
    T: Transcript<F>,
    Cfg: ProverStage1Config,
{
    // Steps 1–2: w_i = a^T G_{2^m} s_i
    let w = compute_w::<F, D, Cfg>(a, &opening.s);

    // Step 3: ŵ_i = G_1^{-1}(w_i)
    let w_hat = compute_w_hat::<F, D, Cfg>(&w);

    // Step 4: v = D · ŵ
    let v = compute_v::<F, D>(&setup.D, &w_hat);

    // Step 5: append v to transcript (first prover message)
    transcript.append_serde(ABSORB_PROVER_V, &v);

    // Step 6: sample 2^r sparse challenges from transcript
    let num_blocks = 1usize << Cfg::R;
    let challenge_cfg = SparseChallengeConfig {
        weight: Cfg::CHALLENGE_WEIGHT,
        nonzero_coeffs: vec![-1, 1],
    };
    let mut challenges = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let sparse = sparse_challenge_from_transcript::<F, T, D>(
            transcript,
            LABEL_STAGE1_CHALLENGE,
            i as u64,
            &challenge_cfg,
        )?;
        let dense = sparse
            .to_dense::<F, D>()
            .map_err(|e| HachiError::InvalidInput(e.to_string()))?;
        challenges.push(dense);
    }

    // Step 7: z = Σ c_i · s_i
    let z = compute_z::<F, D>(&challenges, &opening.s);

    // Step 8: abort if ‖z‖_∞ > β
    check_norm_bound::<F, D>(&z, Cfg::BETA)?;

    // Step 9: ẑ = J^{-1}(z) (balanced gadget decomposition)
    let z_hat = compute_z_hat::<F, D, Cfg>(&z);

    Ok(HachiProof {
        v,
        w_hat,
        t_hat: opening.t_hat.clone(),
        z_hat,
    })
}

// ---------------------------------------------------------------------------
// Balanced gadget decomposition (private helper)
// ---------------------------------------------------------------------------

/// Balanced (centered) base-`2^log_basis` gadget decomposition of a ring element.
///
/// Each coefficient `c` (in centered representation `(-q/2, q/2]`) is decomposed
/// into `levels` balanced digits `d_k ∈ [-b/2, b/2)` satisfying
/// `c ≡ Σ_k d_k · b^k  (mod q)`.
///
/// Digits are stored as field elements reduced modulo `q`.
fn balanced_decompose_ring<F: CanonicalField, const D: usize>(
    elem: &CyclotomicRing<F, D>,
    levels: usize,
    log_basis: u32,
    modulus: u128,
) -> Vec<CyclotomicRing<F, D>> {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128,
        "levels * log_basis must be <= 128"
    );

    let b = 1i128 << log_basis;
    let half_b = b / 2;
    let q = modulus as i128;
    let half_q = q / 2;

    let mut digit_planes: Vec<[F; D]> = (0..levels).map(|_| [F::zero(); D]).collect();

    for i in 0..D {
        let canonical = elem.coefficients()[i].to_canonical_u128() as i128;
        let mut c = if canonical > half_q {
            canonical - q
        } else {
            canonical
        };

        for plane in digit_planes.iter_mut() {
            let d = c.rem_euclid(b);
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) / b;

            plane[i] = if balanced >= 0 {
                F::from_canonical_u128_reduced(balanced as u128)
            } else {
                F::from_canonical_u128_reduced((q + balanced) as u128)
            };
        }
    }

    digit_planes
        .into_iter()
        .map(CyclotomicRing::from_coefficients)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{CyclotomicRing, Fp64};
    use crate::protocol::commitment::{
        CommitmentConfig, HachiCommitmentCore, RingCommitmentScheme, RingCommitmentSetup,
        RingOpening,
    };
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::{CanonicalField, FieldCore, Transcript};

    type F = Fp64<4294967197>;
    const D: usize = 64;

    // -----------------------------------------------------------------------
    // Config
    // -----------------------------------------------------------------------

    #[derive(Clone)]
    struct TestConfig;

    impl CommitmentConfig for TestConfig {
        const D: usize = 64;
        const M: usize = 1;
        const R: usize = 1;
        const N_A: usize = 2;
        const N_B: usize = 2;
        const N_D: usize = 2;
        const LOG_BASIS: u32 = 4;
        const DELTA: usize = 8;
    }

    impl ProverStage1Config for TestConfig {
        const TAU: usize = 4;
        const BETA: u128 = 1_000_000;
        const CHALLENGE_WEIGHT: usize = 3;
    }

    const BLOCK_LEN: usize = 1 << TestConfig::M;
    const NUM_BLOCKS: usize = 1 << TestConfig::R;
    const DELTA: usize = TestConfig::DELTA;
    const LOG_BASIS: u32 = TestConfig::LOG_BASIS;
    const N_A: usize = TestConfig::N_A;
    const TAU: usize = TestConfig::TAU;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn mat_vec_mul(
        mat: &[Vec<CyclotomicRing<F, D>>],
        vec: &[CyclotomicRing<F, D>],
    ) -> Vec<CyclotomicRing<F, D>> {
        mat.iter()
            .map(|row| {
                assert_eq!(row.len(), vec.len());
                row.iter()
                    .zip(vec.iter())
                    .fold(CyclotomicRing::<F, D>::zero(), |acc, (a, x)| {
                        acc + (*a * *x)
                    })
            })
            .collect()
    }

    fn sample_blocks() -> Vec<Vec<CyclotomicRing<F, D>>> {
        (0..NUM_BLOCKS)
            .map(|bi| {
                (0..BLOCK_LEN)
                    .map(|bj| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((bi * 1_000 + bj * 100 + k) as u64)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect()
    }

    fn sample_a() -> Vec<CyclotomicRing<F, D>> {
        (0..BLOCK_LEN)
            .map(|j| {
                let coeffs = std::array::from_fn(|k| F::from_u64((j * 10 + k + 1) as u64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    fn sample_b() -> Vec<CyclotomicRing<F, D>> {
        (0..NUM_BLOCKS)
            .map(|i| {
                let coeffs = std::array::from_fn(|k| F::from_u64((i * 7 + k + 3) as u64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    fn make_test_challenges() -> Vec<CyclotomicRing<F, D>> {
        let mut c1_coeffs = [F::zero(); D];
        c1_coeffs[0] = F::one();
        c1_coeffs[2] = -F::one();

        let mut c2_coeffs = [F::zero(); D];
        c2_coeffs[0] = -F::one();
        c2_coeffs[1] = F::one();

        vec![
            CyclotomicRing::from_coefficients(c1_coeffs),
            CyclotomicRing::from_coefficients(c2_coeffs),
        ]
    }

    fn field_gadget_recompose(
        parts: &[CyclotomicRing<F, D>],
        log_basis: u32,
    ) -> CyclotomicRing<F, D> {
        let b = F::from_u64(1u64 << log_basis);
        let mut result = CyclotomicRing::<F, D>::zero();
        let mut b_power = F::one();
        for part in parts {
            result += part.scale(&b_power);
            b_power = b_power * b;
        }
        result
    }

    fn recompose_z_hat(z_hat: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
        z_hat
            .chunks(TAU)
            .map(|chunk| field_gadget_recompose(chunk, LOG_BASIS))
            .collect()
    }

    fn gadget_recompose_vec(x_hat: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
        x_hat
            .chunks(DELTA)
            .map(|chunk| CyclotomicRing::gadget_recompose_pow2(chunk, LOG_BASIS))
            .collect()
    }

    fn field_gadget_recompose_vec(v: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
        v.chunks(DELTA)
            .map(|chunk| field_gadget_recompose(chunk, LOG_BASIS))
            .collect()
    }

    fn a_transpose_gadget_times_vec(
        a: &[CyclotomicRing<F, D>],
        z: &[CyclotomicRing<F, D>],
    ) -> CyclotomicRing<F, D> {
        let recomposed = field_gadget_recompose_vec(z);
        assert_eq!(recomposed.len(), a.len());
        recomposed
            .iter()
            .zip(a.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (z_j, a_j)| {
                acc + (*a_j * *z_j)
            })
    }

    // -----------------------------------------------------------------------
    // Shared fixture
    // -----------------------------------------------------------------------

    struct Fixture {
        commit_setup: RingCommitmentSetup<F, D>,
        opening: RingOpening<F, D>,
        commitment_u: Vec<CyclotomicRing<F, D>>,
        a: Vec<CyclotomicRing<F, D>>,
        b: Vec<CyclotomicRing<F, D>>,
        challenges: Vec<CyclotomicRing<F, D>>,
        w: Vec<CyclotomicRing<F, D>>,
        w_hat: Vec<Vec<CyclotomicRing<F, D>>>,
        v: Vec<CyclotomicRing<F, D>>,
        z: Vec<CyclotomicRing<F, D>>,
        z_hat: Vec<CyclotomicRing<F, D>>,
        u_eval: CyclotomicRing<F, D>,
        blocks: Vec<Vec<CyclotomicRing<F, D>>>,
    }

    fn build_fixture() -> Fixture {
        let (commit_setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TestConfig>>::setup(16).unwrap();

        let blocks = sample_blocks();
        let (commitment, opening) = <HachiCommitmentCore as RingCommitmentScheme<
            F,
            D,
            TestConfig,
        >>::commit_ring_blocks(&blocks, &commit_setup)
        .unwrap();

        let a = sample_a();
        let b = sample_b();
        let challenges = make_test_challenges();

        let w = compute_w::<F, D, TestConfig>(&a, &opening.s);
        let w_hat = compute_w_hat::<F, D, TestConfig>(&w);
        let v = compute_v::<F, D>(&commit_setup.D, &w_hat);
        let z = compute_z::<F, D>(&challenges, &opening.s);
        let z_hat = compute_z_hat::<F, D, TestConfig>(&z);

        let u_eval = w
            .iter()
            .zip(b.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (w_i, b_i)| {
                acc + (*b_i * *w_i)
            });

        Fixture {
            commit_setup,
            opening,
            commitment_u: commitment.u,
            a,
            b,
            challenges,
            w,
            w_hat,
            v,
            z,
            z_hat,
            u_eval,
            blocks,
        }
    }

    // =======================================================================
    // Row 1:  D · ŵ  =  v
    // =======================================================================

    #[test]
    fn eq20_row1_d_times_w_hat_equals_v() {
        let f = build_fixture();

        let w_hat_flat: Vec<CyclotomicRing<F, D>> =
            f.w_hat.iter().flat_map(|v| v.iter().copied()).collect();
        let lhs = mat_vec_mul(&f.commit_setup.D, &w_hat_flat);

        assert_eq!(lhs, f.v, "Row 1 failed: D · ŵ ≠ v");
    }

    // =======================================================================
    // Row 2:  B · t̂  =  u  (commitment vector)
    // =======================================================================

    #[test]
    fn eq20_row2_b_times_t_hat_equals_u_commitment() {
        let f = build_fixture();

        let t_hat_flat: Vec<CyclotomicRing<F, D>> = f
            .opening
            .t_hat
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let lhs = mat_vec_mul(&f.commit_setup.B, &t_hat_flat);

        assert_eq!(lhs, f.commitment_u, "Row 2 failed: B · t̂ ≠ u");
    }

    // =======================================================================
    // Row 3:  b^T · G_{2^r} · ŵ  =  u_eval
    // =======================================================================

    #[test]
    fn eq20_row3_bt_gadget_w_hat_equals_u_eval() {
        let f = build_fixture();

        let w_recomposed: Vec<CyclotomicRing<F, D>> = f
            .w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2(w_hat_i, LOG_BASIS))
            .collect();

        let lhs = w_recomposed
            .iter()
            .zip(f.b.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (w_i, b_i)| {
                acc + (*b_i * *w_i)
            });

        assert_eq!(lhs, f.u_eval, "Row 3 failed: b^T G ŵ ≠ u_eval");

        let u_eval_direct = f.blocks.iter().zip(f.b.iter()).fold(
            CyclotomicRing::<F, D>::zero(),
            |acc, (block_i, b_i)| {
                let inner: CyclotomicRing<F, D> = block_i
                    .iter()
                    .zip(f.a.iter())
                    .fold(CyclotomicRing::<F, D>::zero(), |acc2, (f_ij, a_j)| {
                        acc2 + (*a_j * *f_ij)
                    });
                acc + (*b_i * inner)
            },
        );
        assert_eq!(
            f.u_eval, u_eval_direct,
            "Cross-check failed: u_eval from w ≠ u_eval from blocks"
        );
    }

    // =======================================================================
    // Row 4:  (c^T ⊗ G_1) · ŵ  =  a^T · G_{2^m} · J · ẑ
    // =======================================================================

    #[test]
    fn eq20_row4_challenge_fold_w_equals_a_gadget_j_z_hat() {
        let f = build_fixture();

        let lhs = f
            .challenges
            .iter()
            .zip(f.w.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (c_i, w_i)| {
                acc + (*c_i * *w_i)
            });

        let z_recovered = recompose_z_hat(&f.z_hat);
        let rhs = a_transpose_gadget_times_vec(&f.a, &z_recovered);

        assert_eq!(lhs, rhs, "Row 4 failed: (c^T ⊗ G_1)ŵ ≠ a^T G J ẑ");
        assert_eq!(z_recovered, f.z, "J · ẑ did not round-trip to z");
    }

    // =======================================================================
    // Row 5:  (c^T ⊗ G_{n_A}) · t̂  =  A · J · ẑ
    // =======================================================================

    #[test]
    fn eq20_row5_challenge_fold_t_equals_a_j_z_hat() {
        let f = build_fixture();

        let mut lhs = vec![CyclotomicRing::<F, D>::zero(); N_A];
        for (c_i, t_hat_i) in f.challenges.iter().zip(f.opening.t_hat.iter()) {
            let t_i = gadget_recompose_vec(t_hat_i);
            assert_eq!(t_i.len(), N_A);
            for (lhs_j, t_ij) in lhs.iter_mut().zip(t_i.iter()) {
                *lhs_j += *c_i * *t_ij;
            }
        }

        let z_recovered = recompose_z_hat(&f.z_hat);
        let rhs = mat_vec_mul(&f.commit_setup.A, &z_recovered);

        assert_eq!(lhs, rhs, "Row 5 failed: (c^T ⊗ G_nA)t̂ ≠ A · J · ẑ");
    }

    // =======================================================================
    // Full orchestrator
    // =======================================================================

    #[test]
    fn prove_opening_orchestrator_runs_and_shapes_correct() {
        let (commit_setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TestConfig>>::setup(16).unwrap();

        let blocks = sample_blocks();
        let (_, opening) =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TestConfig>>::commit_ring_blocks(
                &blocks,
                &commit_setup,
            )
            .unwrap();

        let a = sample_a();
        let mut transcript = Blake2bTranscript::<F>::new(b"test/prover-relation");
        let result = prove_opening::<F, Blake2bTranscript<F>, D, TestConfig>(
            &commit_setup,
            &opening,
            &a,
            &mut transcript,
        )
        .unwrap();

        assert_eq!(result.v.len(), TestConfig::N_D);

        assert_eq!(result.w_hat.len(), NUM_BLOCKS);
        assert!(result.w_hat.iter().all(|v| v.len() == DELTA));

        assert_eq!(result.t_hat.len(), NUM_BLOCKS);
        assert!(result.t_hat.iter().all(|v| v.len() == N_A * DELTA));

        assert_eq!(result.z_hat.len(), BLOCK_LEN * DELTA * TAU);
    }
}
