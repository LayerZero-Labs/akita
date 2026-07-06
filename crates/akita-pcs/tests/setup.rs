//! Setup-capacity E2E tests: exercise production `fp128` D64 presets under three
//! setup-vs-use size relationships.
//!
//! For each preset we run commit/prove/verify (or batched variants) with:
//!
//! 1. **Same size.** `setup_prover` is called with the same `max_num_vars`
//!    and `max_num_batched_polys` used by commit/prove/verify. These must
//!    succeed.
//! 2. **Small setup.** `setup_prover` is called with a *smaller* parameter
//!    than commit/prove/verify. These must fail (panic). We cover the
//!    `max_num_vars` axis and the `max_num_batched_polys` axis separately.
//! 3. **Large setup.** `setup_prover` is called with a *larger* parameter
//!    than commit/prove/verify. These must succeed — the setup envelope is
//!    an upper bound, not a tight match.
//!
//! Every preset listed in `presets.rs` for the production D64 merge gate gets its
//! own module with the five tests.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::test_support::{
    dense_field_evals, init_rayon_pool, opening_from_poly, prove_input, random_point,
    run_on_large_stack, verify_input, F, STACK_SIZE,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::DensePoly;
use akita_prover::OneHotPoly;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaBatchedProof, BasisMode};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---------------------------------------------------------------------------
// Shared test sizes
// ---------------------------------------------------------------------------

/// Number of variables for the polynomial we actually commit/prove/verify.
///
/// This is chosen to ensure these tests exercise a folded schedule (not the
/// root-direct fast path) while keeping CI runtime reasonable.
const POLY_NV: usize = 16;
/// How many polynomials we actually commit in the "same size" tests.
const USE_BATCH: usize = 1;

fn assert_folded_proof(label: &str, proof: &AkitaBatchedProof<F, F>) {
    assert!(
        !proof.is_root_direct(),
        "{label} should exercise a folded proof path, not the root-direct fast path"
    );
}

/// Run `f` on a large-stack worker thread and re-raise its panic payload
/// unchanged, so that `#[should_panic(expected = "...")]` can match the
/// original inner message instead of the generic `"test thread panicked"`
/// string that bare [`run_on_large_stack`] would surface.
fn run_on_large_stack_propagate<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || catch_unwind(AssertUnwindSafe(f)))
        .expect("failed to spawn thread");
    match handle
        .join()
        .expect("worker thread finished without a result")
    {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

fn onehot_lagrange_opening(indices: &[Option<usize>], onehot_k: usize, point: &[F]) -> F {
    assert_eq!(indices.len() * onehot_k, 1usize << point.len());
    indices
        .iter()
        .enumerate()
        .filter_map(|(chunk_idx, hot_idx)| hot_idx.map(|hot_idx| chunk_idx * onehot_k + hot_idx))
        .fold(F::zero(), |acc, field_pos| {
            acc + point
                .iter()
                .enumerate()
                .fold(F::one(), |weight, (bit, &r)| {
                    if ((field_pos >> bit) & 1) == 1 {
                        weight * r
                    } else {
                        weight * (F::one() - r)
                    }
                })
        })
}

/// Run a single-polynomial commit/prove/verify round-trip with the requested
/// setup capacity (`setup_nv`, `setup_polys`) against a polynomial with
/// `poly_nv` variables.  `commit_batch` is the number of polynomial slots we
/// use at commit time (always `1` for the single-poly path, kept explicit so
/// the batch-capacity axis can reuse the same builder).
fn run_dense_e2e<Cfg, const D: usize>(setup_nv: usize, setup_polys: usize, poly_nv: usize)
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
    Cfg: 'static,
{
    assert_eq!(Cfg::D, D);
    assert!(poly_nv >= D.trailing_zeros() as usize);

    let layout = Cfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(poly_nv, 1).expect("singleton opening batch"),
    )
    .expect("layout");

    let evals = dense_field_evals(poly_nv, 0xdead_beef_0000 + poly_nv as u64);
    let poly = DensePoly::<F>::from_field_evals(poly_nv, D, &evals).expect("dense poly");

    let pt = random_point(poly_nv, 0xcafe_0000 + poly_nv as u64);
    let expected_opening = opening_from_poly::<D, _>(&poly, &pt, &layout);

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(setup_nv, setup_polys).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup);

    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .expect("commit");

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [expected_opening];
    let opening_groups = [&openings[..]];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"setup-tests/dense");
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("prove");
    assert_folded_proof("single dense setup-capacity round trip", &proof);

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"setup-tests/dense");
    AkitaCommitmentScheme::<Cfg>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("verify");
}

/// Onehot variant of [`run_dense_e2e`].  `K` is the onehot chunk size; in
/// practice we set `K = D` so `(total_ring * K) == 2^poly_nv`.
fn run_onehot_e2e<Cfg, const D: usize>(setup_nv: usize, setup_polys: usize, poly_nv: usize)
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
    Cfg: 'static,
{
    assert_eq!(Cfg::D, D);

    let layout = Cfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(poly_nv, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    // The committed poly's one-hot chunk size must match the config's required
    // `onehot_chunk_size` (e.g. 256 for D64OneHot); configs with no constraint
    // (`<= 1`) use the K = D one-chunk-per-ring-element representation.
    let k = if layout.onehot_chunk_size > 1 {
        layout.onehot_chunk_size
    } else {
        D
    };
    let total_ring = layout.num_blocks * layout.block_len;
    assert_eq!(
        total_ring * D,
        1usize << poly_nv,
        "onehot layout mismatch at nv={poly_nv}"
    );
    let total_chunks = total_ring * D / k;

    let mut rng = StdRng::seed_from_u64(0xdead_beef_0001 + poly_nv as u64);
    let indices: Vec<Option<usize>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..k)))
        .collect();
    let poly = OneHotPoly::<F, usize>::new(k, D, indices.clone()).expect("onehot poly");

    let pt = random_point(poly_nv, 0xcafe_0001 + poly_nv as u64);
    let expected_opening = onehot_lagrange_opening(&indices, k, &pt);

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(setup_nv, setup_polys).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup);

    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .expect("commit");

    let poly_refs: [&OneHotPoly<F, usize>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [expected_opening];
    let opening_groups = [&openings[..]];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"setup-tests/onehot");
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("prove");
    assert_folded_proof("single onehot setup-capacity round trip", &proof);

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"setup-tests/onehot");
    AkitaCommitmentScheme::<Cfg>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("verify");
}

/// Batched dense round-trip: commit `commit_batch` dense polynomials of
/// `poly_nv` variables into a single grouped commitment, then run
/// `batched_prove`/`batched_verify` at one shared opening point.
fn run_dense_batched_e2e<Cfg, const D: usize>(
    setup_nv: usize,
    setup_polys: usize,
    poly_nv: usize,
    commit_batch: usize,
) where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
    Cfg: 'static,
{
    assert_eq!(Cfg::D, D);
    assert!(commit_batch >= 1);

    let layout = Cfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(poly_nv, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    let polys: Vec<DensePoly<F>> = (0..commit_batch)
        .map(|idx| {
            let mut rng = StdRng::seed_from_u64(0xbeef_cafe_0000 + idx as u64);
            let evals: Vec<F> = (0..1usize << poly_nv)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect();
            DensePoly::<F>::from_field_evals(poly_nv, D, &evals).expect("dense poly")
        })
        .collect();

    let pt = random_point(poly_nv, 0xbabe_0000 + poly_nv as u64);
    let openings: Vec<F> = polys
        .iter()
        .map(|poly| opening_from_poly::<D, _>(poly, &pt, &layout))
        .collect();

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(setup_nv, setup_polys).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup);

    let poly_refs: Vec<&DensePoly<F>> = polys.iter().collect();
    let (commitment, hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack)
        .expect("batched commit");
    let commitments = [commitment];
    let hints = vec![hint];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"setup-tests/batched-dense");
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched prove");
    assert_folded_proof("batched dense setup-capacity round trip", &proof);

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"setup-tests/batched-dense");
    AkitaCommitmentScheme::<Cfg>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched verify");
}

/// Batched onehot round-trip.
///
/// Important: onehot polys bake their `(r_vars, m_vars)` block split in at
/// construction time, unlike dense polys which rebuild blocks from the
/// prover-supplied `block_len`. Under batched commits that split must match
/// the layout the prover will use, which is
/// `test_support::akita_batched_root_layout(nv, setup_polys)` — i.e., sized
/// for the setup's `max_num_batched_polys`, not for a lone poly.
fn run_onehot_batched_e2e<Cfg, const D: usize>(
    setup_nv: usize,
    setup_polys: usize,
    poly_nv: usize,
    commit_batch: usize,
) where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
    Cfg: 'static,
{
    assert_eq!(Cfg::D, D);
    assert!(commit_batch >= 1);

    let layout =
        akita_config::test_support::akita_batched_root_layout::<Cfg>(poly_nv, commit_batch)
            .expect("batched layout");
    let k = if layout.onehot_chunk_size > 1 {
        layout.onehot_chunk_size
    } else {
        D
    };
    let total_ring = layout.num_blocks * layout.block_len;
    assert_eq!(total_ring * D, 1usize << poly_nv);
    let total_chunks = total_ring * D / k;

    let (polys, onehot_indices): (Vec<_>, Vec<_>) = (0..commit_batch)
        .map(|idx| {
            let mut rng = StdRng::seed_from_u64(0xbabe_f00d_0000 + idx as u64);
            let indices: Vec<Option<usize>> = (0..total_chunks)
                .map(|_| Some(rng.gen_range(0..k)))
                .collect();
            let poly = OneHotPoly::<F, usize>::new(k, D, indices.clone()).expect("onehot poly");
            (poly, indices)
        })
        .unzip();

    let pt = random_point(poly_nv, 0xbabe_0001 + poly_nv as u64);
    let openings: Vec<F> = polys
        .iter()
        .zip(onehot_indices.iter())
        .map(|(_, indices)| onehot_lagrange_opening(indices, k, &pt))
        .collect();

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(setup_nv, setup_polys).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup);

    let poly_refs: Vec<&OneHotPoly<F, usize>> = polys.iter().collect();
    let (commitment, hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack)
        .expect("batched onehot commit");
    let commitments = [commitment];
    let hints = vec![hint];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"setup-tests/batched-onehot");
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched onehot prove");
    assert_folded_proof("batched onehot setup-capacity round trip", &proof);

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"setup-tests/batched-onehot");
    AkitaCommitmentScheme::<Cfg>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched onehot verify");
}

// ---------------------------------------------------------------------------
// Preset modules
// ---------------------------------------------------------------------------

// Each module defines the five tests for one preset. The bodies are identical
// up to `type Cfg` / `const D`, so the bulk of the work is in the shared
// helpers above.

macro_rules! preset_module {
    (
        $mod_name:ident,
        $cfg:ty,
        $d:expr,
        $runner:ident,
        $batched_runner:ident $(,)?
    ) => {
        mod $mod_name {
            use super::*;

            type Cfg = $cfg;
            const D: usize = $d;

            // --- Group 1: setup size matches use size -----------------------

            #[test]
            fn same_size_passes() {
                init_rayon_pool();
                run_on_large_stack(|| {
                    $runner::<Cfg, D>(POLY_NV, USE_BATCH, POLY_NV);
                });
            }

            // --- Group 2: smaller setup than what commit/prove/verify use ---

            /// Setup is sized for `alpha` variables (so `outer_vars = 0` and the
            /// shared matrix has its smallest possible stride), but we then try
            /// to commit a polynomial at `POLY_NV` variables. The commit path
            /// explicitly rejects this with `AkitaError::InvalidInput("commit
            /// received a polynomial with N variables but setup supports at
            /// most M")`, which our `.expect("commit")` in the runner turns
            /// into a panic.
            #[test]
            #[should_panic(
                expected = "commit received a polynomial with 16 variables but setup supports at most 15"
            )]
            fn small_setup_nv_panics() {
                init_rayon_pool();
                run_on_large_stack_propagate(|| {
                    $runner::<Cfg, D>(POLY_NV - 1, USE_BATCH, POLY_NV);
                });
            }

            /// Setup is sized for `max_num_batched_polys = 1`, but we then try
            /// to commit a two-polynomial grouped batch. The commit path
            /// explicitly rejects this with `AkitaError::InvalidInput("commit
            /// received N polynomials but setup supports at most M")`, which
            /// our `.expect("batched … commit")` turns into a panic.
            #[test]
            #[should_panic(expected = "commit received 2 polynomials but setup supports at most 1")]
            fn small_setup_batch_panics() {
                init_rayon_pool();
                run_on_large_stack_propagate(|| {
                    $batched_runner::<Cfg, D>(POLY_NV, 1, POLY_NV, 2);
                });
            }

            // --- Group 3: larger setup than what commit/prove/verify use ----

            #[test]
            fn large_setup_nv_passes() {
                init_rayon_pool();
                run_on_large_stack(|| {
                    $runner::<Cfg, D>(POLY_NV + 2, USE_BATCH, POLY_NV);
                });
            }

            #[test]
            fn large_setup_batch_passes() {
                init_rayon_pool();
                run_on_large_stack(|| {
                    // Setup for 4 polys but only commit 2.
                    $batched_runner::<Cfg, D>(POLY_NV, 4, POLY_NV, 2);
                });
            }
        }
    };
}

// Multipoint/batched setup sizing falls through to the planner DP via the
// default `runtime_schedule` fallback, so bare presets suffice — even
// tables-only configs (`D128*` has no table at all).
preset_module!(
    d128_full,
    fp128::D128Full,
    128,
    run_dense_e2e,
    run_dense_batched_e2e
);
preset_module!(
    d64_full,
    fp128::D64Full,
    64,
    run_dense_e2e,
    run_dense_batched_e2e
);
preset_module!(
    d64_onehot,
    fp128::D64OneHot,
    64,
    run_onehot_e2e,
    run_onehot_batched_e2e
);
