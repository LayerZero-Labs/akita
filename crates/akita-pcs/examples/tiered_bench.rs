//! Tiered commitment verifier benchmark.
//!
//! Compares verifier wall-clock for the legacy (`split_factor = 1`) and
//! tiered (`split_factor = 3`) configurations on a one-hot polynomial at
//! `nv = 32`, `D = 32`, single polynomial, single opening point.
//!
//! Both configurations use a single root-fold + direct-terminal schedule
//! (no recursive recursion) with the level-0 layout shape of
//! `fp128::D32OneHot` at `nv = 32`
//! (see `docs/onehot-d32-nv32-matrix-sizes.md`). The single-level
//! schedule isolates the root-level verifier work — which is where the
//! tiered optimization moves cost (per `specs/tiered_commit.md` §10).
//!
//! Memory note: a single onehot polynomial at `nv = 32, D = 32,
//! onehot_k = 256` stores `2^32 / 256 = 2^24 = ~16M` `Option<u8>`
//! indices (≈ 32 MiB), and the prover needs a few-hundred-MiB working
//! set during commit / prove.
//!
//! Usage:
//! ```sh
//! cargo run --release --example tiered_bench
//! ```
//!
//! Env knobs:
//! - `AKITA_BENCH_NV` (default 32): polynomial arity. Override only for
//!   quick smoke testing — the comparison is only meaningful at the
//!   target `nv = 32` shape that the layout constants encode.
//! - `AKITA_BENCH_TRIALS` (default 10): how many verify trials to run
//!   per configuration.

#![allow(missing_docs)]

use akita_challenges::SparseChallengeConfig;
use akita_config::CommitmentConfig;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, CommittedPolynomials, OneHotPoly};
use akita_transcript::Blake2bTranscript;
use akita_types::{
    root_current_w_len, w_ring_element_count_with_counts, AjtaiKeyParams, AjtaiRole,
    AkitaScheduleInputs, AkitaScheduleLookupKey, BasisMode, ClaimIncidenceSummary,
    CommitmentEnvelope, DecompositionParams, DirectStep, DirectWitnessShape, FoldStep, LevelParams,
    Schedule, SisModulusFamily, Step,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

type Field = akita_config::proof_optimized::fp128::Field;

const D: usize = 32;
const ONEHOT_K_PROD: usize = 256;

// fp128 D32OneHot level-0 layout (see docs/onehot-d32-nv32-matrix-sizes.md).
// Override via AKITA_BENCH_NUM_BLOCKS / AKITA_BENCH_BLOCK_LEN for smaller
// sweeps (e.g., to bisect tier-3 bugs at smaller scale).
const N_A: usize = 3;
static N_B_OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
fn n_b() -> usize {
    *N_B_OVERRIDE.get_or_init(|| {
        env::var("AKITA_BENCH_NB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2usize)
    })
}
static N_D_OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
fn n_d() -> usize {
    *N_D_OVERRIDE.get_or_init(|| {
        env::var("AKITA_BENCH_ND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2usize)
    })
}
const DEFAULT_NUM_BLOCKS: usize = 2048;
const DEFAULT_BLOCK_LEN: usize = 65536;
static NUM_BLOCKS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static BLOCK_LEN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
fn num_blocks() -> usize {
    *NUM_BLOCKS.get_or_init(|| {
        env::var("AKITA_BENCH_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_NUM_BLOCKS)
    })
}
fn block_len() -> usize {
    *BLOCK_LEN.get_or_init(|| {
        env::var("AKITA_BENCH_BLOCK_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_BLOCK_LEN)
    })
}
// Matches production fp128::D32OneHot (onehot poly evals are binary
// in {0,1}, so 1 commit digit suffices). Earlier the bench used 12
// as a workaround for what looked like a tier-3 dc=1 bug, but it
// turned out the real bug is upstream and oversizing dc=1 just hid
// a separate dimension-coupling issue. Putting it back to the
// production value.
const DEPTH_COMMIT: u32 = 1;
static DEPTH_OPEN_OVERRIDE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
fn depth_open() -> u32 {
    *DEPTH_OPEN_OVERRIDE.get_or_init(|| {
        env::var("AKITA_BENCH_DEPTH_OPEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64)
    })
}
static DEPTH_FOLD_OVERRIDE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
fn depth_fold() -> u32 {
    *DEPTH_FOLD_OVERRIDE.get_or_init(|| {
        env::var("AKITA_BENCH_DEPTH_FOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10)
    })
}
static TIER_NUM_DIGITS_OUTER_OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
fn tier_num_digits_outer() -> usize {
    *TIER_NUM_DIGITS_OUTER_OVERRIDE.get_or_init(|| {
        env::var("AKITA_BENCH_NUM_DIGITS_OUTER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(65)
    })
}
const NUM_RING_TAIL: usize = 0;
const LOG_BASIS: u32 = 2;
const LOG_COMMIT_BOUND: u32 = 1;
const LOG_OPEN_BOUND: u32 = 128;

// Tiered fixture sizing for `split_factor = 3`. `outer_log_basis` must
// equal `log_basis` so the stage-1/2 sumcheck digit lookup tables (sized
// for `b = 2^log_basis`) cover `ûhat` cells; `num_digits_outer` must
// satisfy `outer_log_basis · num_digits_outer ≥ field_bits` so the
// gadget decomposition of `u_i` is lossless (per the gadget-identity
// test in `crates/akita-prover/src/protocol/tiered_commit.rs`).
const TIER_OUTER_LOG_BASIS: u32 = LOG_BASIS; // = 2
                                             // `num_digits_outer = 65` (not 64) so the balanced gadget can
                                             // represent every `u_i` coefficient in Q128's full centered range
                                             // `[-q/2, q/2)`. For basis `b = 2^outer_log_basis = 4`, balanced
                                             // digits are in `[-b/2, b/2-1] = [-2, 1]`, so:
                                             //   max positive representable = 1·(4^δ - 1)/3
                                             //   min negative representable = -2·(4^δ - 1)/3
                                             // At `δ = 64` the positive bound is only `~2^126.4 < 2^127 ≈ q/2`,
                                             // so random `u_i` coefficients in `[2^126.4, 2^127)` would silently
                                             // overflow the decomp, breaking the gadget identity
                                             // `G·ûhat = u_i` that the tiered protocol's r_quotient relies on.
                                             // `δ = 65` gives `~2^128.4`, comfortably above `q/2`.
const TIER_N_F: usize = 2;

fn make_root_lp(split_factor: usize) -> LevelParams {
    // Level-0 LP shape pulled from the production schedule for
    // `fp128::D32OneHot` at `nv = 32` (see docs file).
    let base = LevelParams::params_only(
        SisModulusFamily::Q128,
        D,
        LOG_BASIS,
        N_A,
        n_b(),
        n_d(),
        SparseChallengeConfig::BoundedL1Norm,
    )
    .with_decomp(
        block_len().trailing_zeros() as usize,  // m_vars
        num_blocks().trailing_zeros() as usize, // r_vars
        DEPTH_COMMIT as usize,
        depth_open() as usize,
        depth_fold() as usize,
        NUM_RING_TAIL,
    )
    .expect("base level params");

    if split_factor == 1 {
        return base;
    }

    // Tiered B' has the same row count as legacy B but a `chunk_width`
    // column count (= `full_outer_width / split`). For nv=32 d=32 onehot:
    // full_outer_width = n_a · depth_open · num_blocks = 3·64·2048 =
    // 393_216, chunk_width = 131_072 (cleanly divisible by 3).
    let full_outer = base.full_outer_width();
    assert!(
        full_outer % split_factor == 0,
        "outer width {full_outer} not divisible by split factor {split_factor}",
    );
    let chunk_width = full_outer / split_factor;
    let n_b_prime = n_b();
    let f_width = n_b_prime * split_factor * tier_num_digits_outer();
    let tiered_b_key = AjtaiKeyParams::new_unchecked(
        base.b_key.sis_family(),
        base.b_key.row_len(),
        chunk_width,
        base.b_key.collision_inf(),
        base.ring_dimension,
    );
    let f_key = AjtaiKeyParams::new_unchecked(
        SisModulusFamily::Q128,
        TIER_N_F,
        f_width,
        akita_types::layout::sis_derivation::balanced_digit_delta_bound(TIER_OUTER_LOG_BASIS),
        base.ring_dimension,
    );
    LevelParams {
        split_factor,
        outer_log_basis: TIER_OUTER_LOG_BASIS,
        num_digits_outer: tier_num_digits_outer(),
        f_key,
        b_key: tiered_b_key,
        ..base
    }
}

fn setup_matrix_size_for_lp(
    lp: &LevelParams,
    max_num_claims: usize,
) -> Result<(usize, usize), AkitaError> {
    let inner = lp.inner_width();
    // For LEGACY (`split_factor == 1`), `lp.outer_width() ==
    // full_outer_width` (= legacy B's column count). For TIERED, the
    // b_key stores only the column window B' actually uses
    // (`chunk_width = full_outer / split`), so `lp.outer_width() ==
    // chunk_width`. Using `lp.outer_width()` here lets the tiered
    // setup envelope shrink to fit B' rather than carrying full B
    // around — which is exactly what gives tiering its verifier
    // speedup (`compute_setup_contribution`'s scan range collapses
    // to `chunk_width`).
    let outer = lp.outer_width();
    let d_matrix = lp
        .d_matrix_width()
        .checked_mul(max_num_claims.max(1))
        .ok_or_else(|| AkitaError::InvalidSetup("D matrix width overflow".to_string()))?;
    let max_stride = inner.max(outer).max(d_matrix);
    let max_rows = lp
        .a_key
        .row_len()
        .max(lp.b_key.row_len())
        .max(lp.d_key.row_len());
    Ok((max_rows, max_stride))
}

// One Cfg per split factor. They only differ in `make_root_lp`'s split
// argument; everything else is identical.
macro_rules! impl_bench_cfg {
    ($name:ident, $split:expr, $label:expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl $name {
            pub const SPLIT_FACTOR: usize = $split;
            pub const LABEL: &'static str = $label;

            fn root_lp() -> LevelParams {
                make_root_lp(Self::SPLIT_FACTOR)
            }
        }

        impl akita_types::ScheduleProvider for $name {
            fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
                None
            }
            fn schedule_key(key: AkitaScheduleLookupKey) -> String {
                format!("bench/{}/{key:?}", Self::LABEL)
            }
            fn schedule_plan(
                _key: AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
                Ok(None)
            }
        }

        #[cfg(feature = "planner")]
        impl akita_planner::PlannerConfig for $name {
            type PlannerField = Field;
            const PLANNER_D: usize = D;

            fn planner_field_bits() -> u32 {
                <Self as CommitmentConfig>::decomposition().field_bits()
            }

            fn planner_challenge_field_bits() -> u32 {
                Self::planner_field_bits() * (<Self as CommitmentConfig>::CHAL_EXT_DEGREE as u32)
            }

            fn planner_extension_opening_width() -> usize {
                <Self as CommitmentConfig>::CLAIM_EXT_DEGREE
            }

            fn planner_sis_modulus_family() -> SisModulusFamily {
                SisModulusFamily::Q128
            }

            fn planner_stage1_challenge_config(_d: usize) -> SparseChallengeConfig {
                SparseChallengeConfig::BoundedL1Norm
            }

            fn planner_schedule_plan(
                _key: AkitaScheduleLookupKey,
            ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
                Ok(None)
            }

            fn planner_root_level_layout_with_log_basis(
                _inputs: AkitaScheduleInputs,
                _log_basis: u32,
            ) -> Result<LevelParams, AkitaError> {
                Ok(Self::root_lp())
            }

            fn planner_current_level_layout_with_log_basis(
                _inputs: AkitaScheduleInputs,
                _log_basis: u32,
            ) -> Result<LevelParams, AkitaError> {
                Ok(Self::root_lp())
            }

            fn planner_root_level_params_for_layout_with_log_basis(
                _inputs: AkitaScheduleInputs,
                lp: &LevelParams,
            ) -> Result<LevelParams, AkitaError> {
                Ok(Self::root_lp().with_layout(lp))
            }

            fn planner_log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
                (LOG_BASIS, LOG_BASIS)
            }
        }

        impl CommitmentConfig for $name {
            type Field = Field;
            type ClaimField = Field;
            type ChallengeField = Field;
            const D: usize = D;

            fn decomposition() -> DecompositionParams {
                DecompositionParams {
                    log_basis: LOG_BASIS,
                    log_commit_bound: LOG_COMMIT_BOUND,
                    log_open_bound: Some(LOG_OPEN_BOUND),
                }
            }

            fn stage1_challenge_config(_d: usize) -> SparseChallengeConfig {
                SparseChallengeConfig::BoundedL1Norm
            }

            fn sis_modulus_family() -> SisModulusFamily {
                SisModulusFamily::Q128
            }

            fn audited_root_rank(_role: AjtaiRole, _max_num_vars: usize) -> usize {
                // Single-level schedule means recursive level-rank
                // overrides don't apply.
                1
            }

            fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
                CommitmentEnvelope {
                    max_n_a: N_A,
                    max_n_b: n_b(),
                    max_n_d: n_d(),
                }
            }

            fn max_setup_matrix_size(
                _max_num_vars: usize,
                max_num_batched_polys: usize,
                max_num_points: usize,
            ) -> Result<(usize, usize), AkitaError> {
                let lp = Self::root_lp();
                let max_num_claims = max_num_batched_polys
                    .checked_mul(max_num_points)
                    .ok_or_else(|| AkitaError::InvalidSetup("claim count overflow".to_string()))?;
                setup_matrix_size_for_lp(&lp, max_num_claims)
            }

            fn level_params_with_log_basis(
                _inputs: AkitaScheduleInputs,
                _log_basis: u32,
            ) -> LevelParams {
                Self::root_lp()
            }

            fn root_level_params_for_layout_with_log_basis(
                _inputs: AkitaScheduleInputs,
                lp: &LevelParams,
            ) -> Result<LevelParams, AkitaError> {
                Ok(Self::root_lp().with_layout(lp))
            }

            fn root_level_layout_with_log_basis(
                _inputs: AkitaScheduleInputs,
                _log_basis: u32,
            ) -> Result<LevelParams, AkitaError> {
                Ok(Self::root_lp())
            }

            fn log_basis_at_level(_inputs: AkitaScheduleInputs) -> u32 {
                LOG_BASIS
            }

            fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
                (LOG_BASIS, LOG_BASIS)
            }

            fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, AkitaError> {
                Ok(Self::root_lp())
            }

            fn get_params_for_commitment(
                _num_vars: usize,
                _num_polys_per_point: usize,
                _max_num_points: usize,
            ) -> Result<LevelParams, AkitaError> {
                Ok(Self::root_lp())
            }

            fn get_params_for_prove(
                incidence: &ClaimIncidenceSummary,
            ) -> Result<Schedule, AkitaError> {
                let lp = Self::root_lp();
                let w_ring = w_ring_element_count_with_counts::<Self::Field>(
                    &lp,
                    incidence.num_points(),
                    incidence.num_polynomials(),
                    incidence.num_claims(),
                    incidence.num_public_rows(),
                );
                let next_w_len = w_ring * Self::D;
                Ok(Schedule {
                    steps: vec![
                        Step::Fold(FoldStep {
                            params: lp.clone(),
                            current_w_len: root_current_w_len(&lp),
                            delta_fold_per_poly: lp.num_digits_fold,
                            w_ring,
                            next_w_len,
                            level_bytes: 0,
                        }),
                        Step::Direct(DirectStep {
                            current_w_len: next_w_len,
                            witness_shape: DirectWitnessShape::PackedDigits((
                                next_w_len, LOG_BASIS,
                            )),
                            direct_bytes: next_w_len,
                        }),
                    ],
                    total_bytes: 0,
                })
            }
        }
    };
}

impl_bench_cfg!(LegacyBenchCfg, 1, "legacy_f1");
impl_bench_cfg!(Tier2BenchCfg, 2, "tier2");
impl_bench_cfg!(Tier3BenchCfg, 3, "tier3");

#[derive(Default)]
struct Stats {
    samples: Vec<f64>,
}

impl Stats {
    fn push(&mut self, secs: f64) {
        self.samples.push(secs);
    }

    fn summary(&self) -> (f64, f64, f64, f64) {
        let mut s = self.samples.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len() as f64;
        let mean = s.iter().sum::<f64>() / n;
        let min = *s.first().unwrap();
        let max = *s.last().unwrap();
        let median = if s.len() % 2 == 0 {
            (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0
        } else {
            s[s.len() / 2]
        };
        (mean, median, min, max)
    }
}

/// Compute `eq(point, i) = Π_j (bit_j(i)==1 ? point[j] : 1 - point[j])`
/// for one specific `i ∈ [0, 2^point.len())`. Avoids materializing the
/// full `2^point.len()` Lagrange weights table (which is 64 GiB at
/// `nv = 32` over Fp128).
fn lagrange_weight_at<E: FieldCore>(point: &[E], idx: usize) -> E {
    let mut w = E::one();
    for (j, &p) in point.iter().enumerate() {
        if (idx >> j) & 1 == 1 {
            w *= p;
        } else {
            w *= E::one() - p;
        }
    }
    w
}

fn opening_from_onehot_indices(indices: &[Option<u8>], onehot_k: usize, point: &[Field]) -> Field {
    // For a OneHot poly, `evals[chunk * onehot_k + idx] = 1`, all else 0.
    // So `<weights, evals> = Σ_chunk eq(point, chunk*onehot_k + indices[chunk])`.
    let mut acc = Field::zero();
    for (chunk, &maybe_idx) in indices.iter().enumerate() {
        if let Some(idx) = maybe_idx {
            let flat_idx = chunk * onehot_k + idx as usize;
            acc += lagrange_weight_at(point, flat_idx);
        }
    }
    acc
}

fn run_bench<Cfg>(label: &str, nv: usize, trials: usize, rng: &mut StdRng) -> Stats
where
    Cfg: CommitmentConfig<Field = Field, ClaimField = Field, ChallengeField = Field>,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            Field,
            D,
            ClaimField = Field,
            VerifierSetup = akita_types::AkitaVerifierSetup<Field>,
            Commitment = akita_types::RingCommitment<Field, D>,
            BatchedProof = akita_types::AkitaBatchedProof<Field, Field>,
            CommitHint = akita_types::AkitaCommitmentHint<Field, D>,
        > + CommitmentVerifier<
            Field,
            D,
            ClaimField = Field,
            VerifierSetup = akita_types::AkitaVerifierSetup<Field>,
            Commitment = akita_types::RingCommitment<Field, D>,
            BatchedProof = akita_types::AkitaBatchedProof<Field, Field>,
        >,
{
    type Scheme<const DD: usize, Cfg> = AkitaCommitmentScheme<DD, Cfg>;
    let lp = Cfg::commitment_layout(nv).expect("commitment_layout");
    println!(
        "[{label}] root lp shape: n_a={}, n_b={}, n_d={}, num_blocks={}, block_len={}, depth_open={}, depth_fold={}, split_factor={}, num_digits_outer={}, full_outer_width={}",
        lp.a_key.row_len(),
        lp.b_key.row_len(),
        lp.d_key.row_len(),
        lp.num_blocks,
        lp.block_len,
        lp.num_digits_open,
        lp.num_digits_fold,
        lp.split_factor,
        lp.num_digits_outer,
        lp.full_outer_width(),
    );

    // Build the onehot poly.
    let total_field = num_blocks() * block_len() * D;
    let total_chunks = total_field / ONEHOT_K_PROD;
    let t_indices = Instant::now();
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K_PROD) as u8))
        .collect();
    println!(
        "[{label}] generated {total_chunks} onehot indices ({:.2}s)",
        t_indices.elapsed().as_secs_f64()
    );
    let poly =
        OneHotPoly::<Field, D, u8>::new(ONEHOT_K_PROD, indices.clone()).expect("onehot poly");

    // Opening point + opening (computed lazily — never materialize
    // the full `2^nv` Lagrange weights table).
    let point: Vec<Field> = (0..nv)
        .map(|_| Field::from_u128(rng.gen::<u128>()))
        .collect();
    let t_open = Instant::now();
    let opening = opening_from_onehot_indices(&indices, ONEHOT_K_PROD, &point);
    println!(
        "[{label}] poly + opening built (opening eval {:.2}s)",
        t_open.elapsed().as_secs_f64()
    );

    // Setup.
    let t_setup = Instant::now();
    let setup = <Scheme<D, Cfg> as CommitmentProver<Field, D>>::setup_prover(nv, 1, 1);
    let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<Field, D>>::setup_verifier(&setup);
    println!(
        "[{label}] setup_prover + setup_verifier: {:.2}s",
        t_setup.elapsed().as_secs_f64()
    );

    // Commit.
    let t_commit = Instant::now();
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentProver<Field, D>>::commit(std::slice::from_ref(&poly), &setup)
            .expect("commit");
    println!("[{label}] commit: {:.2}s", t_commit.elapsed().as_secs_f64());

    // Prove.
    let poly_refs = [&poly];
    let commitments = [commitment];
    let t_prove = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<Field>::new(b"tiered_bench");
    let proof = <Scheme<D, Cfg> as CommitmentProver<Field, D>>::batched_prove(
        &setup,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("prove");
    println!(
        "[{label}] prove: {:.2}s (proof bytes: {})",
        t_prove.elapsed().as_secs_f64(),
        proof.size(),
    );

    // Verify N times, recording timing.
    let openings = [opening];
    let mut stats = Stats::default();
    println!("[{label}] running {trials} verify trials...");
    for trial in 0..trials {
        let mut verifier_transcript = Blake2bTranscript::<Field>::new(b"tiered_bench");
        let t = Instant::now();
        let result = <Scheme<D, Cfg> as CommitmentVerifier<Field, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &point[..],
                CommittedOpenings {
                    openings: &openings[..],
                    commitment: &commitments[0],
                },
            )],
            BasisMode::Lagrange,
        );
        if let Err(e) = &result {
            panic!("[{label}] verify failed: {e:#?}");
        }
        let elapsed = t.elapsed().as_secs_f64();
        stats.push(elapsed);
        println!(
            "[{label}]   trial {:>2}: {:.4} ms",
            trial + 1,
            elapsed * 1000.0
        );
    }
    stats
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_target(false)
        .init();
    let nv: usize = env::var("AKITA_BENCH_NV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let trials: usize = env::var("AKITA_BENCH_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let expected_nv = (num_blocks() * block_len() * D).trailing_zeros() as usize;
    if nv != expected_nv {
        eprintln!(
            "warning: AKITA_BENCH_NV={nv} ≠ implied nv {expected_nv} from \
             NUM_BLOCKS({}) · BLOCK_LEN({}) · D({}). Override your layout to match.",
            num_blocks(),
            block_len(),
            D,
        );
    }

    println!("=========================================================");
    println!(
        " Tiered verifier benchmark: onehot, nv={}, D={}, onehot_k={}, single poly, single point",
        nv, D, ONEHOT_K_PROD
    );
    println!(" Schedule: single root fold + direct terminal (no recursion)");
    println!(" Trials per config: {trials}");
    println!("=========================================================");

    let only_legacy = env::var("AKITA_BENCH_LEGACY_ONLY")
        .ok()
        .map(|s| s != "0")
        .unwrap_or(false);
    let only_tier = env::var("AKITA_BENCH_TIER_ONLY")
        .ok()
        .map(|s| s != "0")
        .unwrap_or(false);

    let legacy_stats = if !only_tier {
        let mut rng_legacy = StdRng::seed_from_u64(0xa11ce);
        Some(run_bench::<LegacyBenchCfg>(
            "legacy_f1",
            nv,
            trials,
            &mut rng_legacy,
        ))
    } else {
        None
    };

    let tier3_stats = if !only_legacy {
        let split: usize = env::var("AKITA_BENCH_SPLIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let mut rng_tier = StdRng::seed_from_u64(0xb0b);
        Some(match split {
            2 => run_bench::<Tier2BenchCfg>("tier2_f2", nv, trials, &mut rng_tier),
            3 => run_bench::<Tier3BenchCfg>("tier3_f3", nv, trials, &mut rng_tier),
            _ => panic!("only split=2 or 3 supported"),
        })
    } else {
        None
    };

    println!();
    println!("=========================================================");
    println!(" Summary (verify wall-clock, seconds)");
    println!("=========================================================");
    if let Some(ls) = &legacy_stats {
        let (lm, lmed, lmin, lmax) = ls.summary();
        println!(
            "  legacy (f=1):  mean={:.4}s  median={:.4}s  min={:.4}s  max={:.4}s",
            lm, lmed, lmin, lmax
        );
    }
    if let Some(ts) = &tier3_stats {
        let (tm, tmed, tmin, tmax) = ts.summary();
        println!(
            "  tiered (f=3):  mean={:.4}s  median={:.4}s  min={:.4}s  max={:.4}s",
            tm, tmed, tmin, tmax
        );
    }
    if let (Some(ls), Some(ts)) = (&legacy_stats, &tier3_stats) {
        let (lm, _, _, _) = ls.summary();
        let (tm, _, _, _) = ts.summary();
        let speedup = lm / tm;
        println!("  mean speedup (legacy / tiered): {:.2}x", speedup);
    }
}
