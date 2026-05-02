//! Setup-capacity E2E tests: exercise every `fp128` preset under three
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
//! Every preset listed in `presets.rs` (`D128Full`, `D64Full`, `D64OneHot`,
//! `D32Full`, `D32OneHot`) gets its own module with the five tests.

#![allow(missing_docs)]

mod common;

use common::{
    init_rayon_pool, opening_from_poly, prove_input, random_point, run_on_large_stack,
    verify_input, F,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::config::proof_optimized::fp128;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentProver, CommitmentVerifier, FieldCore, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---------------------------------------------------------------------------
// Shared test sizes
// ---------------------------------------------------------------------------

/// Number of variables for the polynomial we actually commit/prove/verify.
const POLY_NV: usize = 20;
/// How many polynomials we actually commit in the "same size" tests.
const USE_BATCH: usize = 1;

/// Run `f` on a large-stack worker thread and re-raise its panic payload
/// unchanged, so that `#[should_panic(expected = "...")]` can match the
/// original inner message instead of the generic `"test thread panicked"`
/// string that bare [`run_on_large_stack`] would surface.
fn run_on_large_stack_propagate<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    let handle = std::thread::Builder::new()
        .stack_size(common::STACK_SIZE)
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
    Cfg: CommitmentConfig<Field = F>,
    Cfg: 'static,
{
    assert_eq!(Cfg::D, D);
    assert!(poly_nv >= D.trailing_zeros() as usize);

    let layout = Cfg::commitment_layout(poly_nv).expect("layout");

    let mut rng = StdRng::seed_from_u64(0xdead_beef_0000 + poly_nv as u64);
    let evals: Vec<F> = (0..1usize << poly_nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let poly = DensePoly::<F, D>::from_field_evals(poly_nv, &evals).expect("dense poly");

    let pt = random_point(poly_nv, 0xcafe_0000 + poly_nv as u64);
    let expected_opening = opening_from_poly(&poly, &pt, &layout);

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        setup_nv,
        setup_polys,
        1,
    );
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .expect("commit");

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [expected_opening];
    let opening_groups = [&openings[..]];
    let hints = vec![hint];

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"setup-tests/dense");
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("prove");

    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"setup-tests/dense");
    <HachiCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
    )
    .expect("verify");
}

/// Onehot variant of [`run_dense_e2e`].  `K` is the onehot chunk size; in
/// practice we set `K = D` so `(total_ring * K) == 2^poly_nv`.
fn run_onehot_e2e<Cfg, const D: usize>(setup_nv: usize, setup_polys: usize, poly_nv: usize)
where
    Cfg: CommitmentConfig<Field = F>,
    Cfg: 'static,
{
    assert_eq!(Cfg::D, D);

    let k = D;
    let layout = Cfg::commitment_layout(poly_nv).expect("layout");
    let total_ring = layout.num_blocks * layout.block_len;
    assert_eq!(
        total_ring * k,
        1usize << poly_nv,
        "onehot layout mismatch at nv={poly_nv}"
    );

    let mut rng = StdRng::seed_from_u64(0xdead_beef_0001 + poly_nv as u64);
    let indices: Vec<Option<usize>> = (0..total_ring).map(|_| Some(rng.gen_range(0..k))).collect();
    let poly = OneHotPoly::<F, D, usize>::new(k, indices.clone()).expect("onehot poly");

    let pt = random_point(poly_nv, 0xcafe_0001 + poly_nv as u64);
    let expected_opening = onehot_lagrange_opening(&indices, k, &pt);

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        setup_nv,
        setup_polys,
        1,
    );
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .expect("commit");

    let poly_refs: [&OneHotPoly<F, D, usize>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [expected_opening];
    let opening_groups = [&openings[..]];
    let hints = vec![hint];

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"setup-tests/onehot");
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("prove");

    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"setup-tests/onehot");
    <HachiCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
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
    Cfg: CommitmentConfig<Field = F>,
    Cfg: 'static,
{
    assert_eq!(Cfg::D, D);
    assert!(commit_batch >= 1);

    let layout = Cfg::commitment_layout(poly_nv).expect("layout");
    let polys: Vec<DensePoly<F, D>> = (0..commit_batch)
        .map(|idx| {
            let mut rng = StdRng::seed_from_u64(0xbeef_cafe_0000 + idx as u64);
            let evals: Vec<F> = (0..1usize << poly_nv)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect();
            DensePoly::<F, D>::from_field_evals(poly_nv, &evals).expect("dense poly")
        })
        .collect();

    let pt = random_point(poly_nv, 0xbabe_0000 + poly_nv as u64);
    let openings: Vec<F> = polys
        .iter()
        .map(|poly| opening_from_poly(poly, &pt, &layout))
        .collect();

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        setup_nv,
        setup_polys,
        1,
    );
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);

    let poly_refs: Vec<&DensePoly<F, D>> = polys.iter().collect();
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(&polys, &setup)
            .expect("batched commit");
    let commitments = [commitment];
    let hints = vec![hint];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"setup-tests/batched-dense");
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched prove");

    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"setup-tests/batched-dense");
    <HachiCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
    )
    .expect("batched verify");
}

/// Batched onehot round-trip.
///
/// Important: onehot polys bake their `(r_vars, m_vars)` block split in at
/// construction time, unlike dense polys which rebuild blocks from the
/// prover-supplied `block_len`. Under batched commits that split must match
/// the layout the prover will use, which is
/// [`hachi_batched_root_layout(nv, setup_polys)`] — i.e., sized for the
/// setup's `max_num_batched_polys`, not for a lone poly.
fn run_onehot_batched_e2e<Cfg, const D: usize>(
    setup_nv: usize,
    setup_polys: usize,
    poly_nv: usize,
    commit_batch: usize,
) where
    Cfg: CommitmentConfig<Field = F>,
    Cfg: 'static,
{
    assert_eq!(Cfg::D, D);
    assert!(commit_batch >= 1);

    let k = D;
    let layout =
        hachi_pcs::protocol::commitment::hachi_batched_root_layout::<Cfg>(poly_nv, commit_batch)
            .expect("batched layout");
    let total_ring = layout.num_blocks * layout.block_len;
    assert_eq!(total_ring * k, 1usize << poly_nv);

    let (polys, onehot_indices): (Vec<_>, Vec<_>) = (0..commit_batch)
        .map(|idx| {
            let mut rng = StdRng::seed_from_u64(0xbabe_f00d_0000 + idx as u64);
            let indices: Vec<Option<usize>> =
                (0..total_ring).map(|_| Some(rng.gen_range(0..k))).collect();
            let poly = OneHotPoly::<F, D, usize>::new(k, indices.clone()).expect("onehot poly");
            (poly, indices)
        })
        .unzip();

    let pt = random_point(poly_nv, 0xbabe_0001 + poly_nv as u64);
    let openings: Vec<F> = polys
        .iter()
        .zip(onehot_indices.iter())
        .map(|(_, indices)| onehot_lagrange_opening(indices, k, &pt))
        .collect();

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        setup_nv,
        setup_polys,
        1,
    );
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);

    let poly_refs: Vec<&OneHotPoly<F, D, usize>> = polys.iter().collect();
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(&polys, &setup)
            .expect("batched onehot commit");
    let commitments = [commitment];
    let hints = vec![hint];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"setup-tests/batched-onehot");
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched onehot prove");

    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"setup-tests/batched-onehot");
    <HachiCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
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
            /// explicitly rejects this with `HachiError::InvalidInput("commit
            /// received a polynomial with N variables but setup supports at
            /// most M")`, which our `.expect("commit")` in the runner turns
            /// into a panic.
            #[test]
            #[should_panic(
                expected = "commit received a polynomial with 20 variables but setup supports at most 19"
            )]
            fn small_setup_nv_panics() {
                init_rayon_pool();
                run_on_large_stack_propagate(|| {
                    $runner::<Cfg, D>(POLY_NV - 1, USE_BATCH, POLY_NV);
                });
            }

            /// Setup is sized for `max_num_batched_polys = 1`, but we then try
            /// to commit a two-polynomial grouped batch. The commit path
            /// explicitly rejects this with `HachiError::InvalidInput("commit
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
preset_module!(
    d32_full,
    fp128::D32Full,
    32,
    run_dense_e2e,
    run_dense_batched_e2e
);
preset_module!(
    d32_onehot,
    fp128::D32OneHot,
    32,
    run_onehot_e2e,
    run_onehot_batched_e2e
);
