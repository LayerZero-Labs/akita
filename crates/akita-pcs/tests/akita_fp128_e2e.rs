//! Correctness matrix for fp128 Akita PCS prove→verify roundtrips.
//!
//! # Group B — fp128 full correctness matrix
//!
//! `CommitmentConfig<Field=F, ExtField=F>`, so the generic driver is used directly.
//! The table covers the full cartesian product:
//!   poly ∈ {Dense, OneHot} × chunk ∈ {sc, mc} × precommit ∈ {direct, pre} × recursion ∈ {nonrec, rec}
//!
//! In the recursive rows, `direct` and `pre` distinguish whether the *caller*
//! supplies precommitted groups, not whether the schedule is multi-group:
//! recursive mode always resolves against a multi-group setup internally.
//! `direct` is `RecursiveCommitmentConfig` with no user precommit; `pre` adds
//! user precommitted groups on top. Both columns therefore exist and are
//! declared below.
//!
//! Legend:
//!   ✓        — runs in default `cargo test` using checked-in external artifacts
//!   ign      — skipped in default `cargo test` due to production-sized nv; needs `-- --ignored`
//!   NA       — no production schedule exists for this combination; cell is intentionally absent
//!
//! Non-recursive `pre` cells use one 14-variable pre-group; recursive `pre`
//! cells use the shared recursive profile's two 16-variable pre-groups. Every ✓
//! cell below runs and is backed by a real generated catalog row — there are no
//! `#[ignore]`d placeholders for missing catalog entries in this file.
//!
//! ```text
//! ╔══════════╦══════════╦═══════════════════════════════╦═══════════════════════════════╗
//! ║          ║          ║      single-chunk (sc)        ║      multi-chunk (mc)         ║
//! ║ poly     ║ rec?     ╠═══════════════╦═══════════════╬═══════════════╦═══════════════╣
//! ║          ║          ║    direct     ║      pre      ║    direct     ║      pre      ║
//! ╠══════════╬══════════╬═══════════════╬═══════════════╬═══════════════╬═══════════════╣
//! ║ Dense    ║ nonrec   ║ ✓ [14,16,     ║ ✓ final=16    ║ ✓ [16]        ║      NA       ║
//! ║          ║          ║    24,26]     ║               ║               ║               ║
//! ║ Dense    ║ rec      ║      NA       ║      NA       ║      NA       ║      NA       ║
//! ╠══════════╬══════════╬═══════════════╬═══════════════╬═══════════════╬═══════════════╣
//! ║ OneHot   ║ nonrec   ║ ✓ [12,15,     ║ ✓ final=      ║      ign      ║      NA       ║
//! ║          ║          ║    20,28]     ║   [16,20]     ║               ║               ║
//! ║ OneHot   ║ rec      ║      ign      ║      ign      ║      ign      ║      ign      ║
//! ╚══════════╩══════════╩═══════════════╩═══════════════╩═══════════════╩═══════════════╝
//! ```
//!
//! Dense + recursive: no production schedule exists; those cells are permanently NA.
//! Dense mc pre: NA. The multi-chunk family ships only nv=16, and the DP finds
//! no multi-group multi-chunk schedule below final_nv=20, so backing this cell
//! would mean adding a production size purely for a test.
//! OneHot mc nonrec direct: nv=32 is production-sized (ign).
//!   `fp128_onehot_mc_catalog_resolves` is the cheap always-run companion that
//!   checks the same catalog row without proving at nv=32.
//! OneHot mc nonrec pre: NA. The catalog has no combined final=32, pre=14 row.
//! OneHot sc rec: nv=32 is production-sized (ign).
//!   direct = RecursiveCommitmentConfig only, no user precommit.
//!   pre    = RecursiveCommitmentConfig + two 16-variable user precommits,
//!            committed under the base config's scalar row.
//! OneHot mc rec: nv=32 is production-sized (ign).
//!   direct = RecursiveCommitmentConfig<OneHotMultiChunk>.
//!   pre    = same + two 16-variable user precommits, committed under the
//!            base config's scalar row.
//!
//! Every ✓ cell resolves against a real shipped catalog row; no cell here is
//! backed by a schedule added solely to make a test pass.
//!
//! # Group E — heterogeneous cells (see `akita_fp128_e2e/heterogeneous.rs`)
//!
//! The matrix above is indexed by polynomial type; the committed-source **bound**
//! is a second, orthogonal axis. `Dense` rows are full-width
//! (`log_commit_bound = 128`) and `OneHot` rows are the unit endpoint
//! (`log_commit_bound = 1`); a bounded source is any value in between. Group E
//! carries the mixed-bound cell:
//!
//! - `bounded_dense_precommit_with_onehot_final_group`.
//!   A `fp128::DenseBounded` precommit (bound 65 inside the 128-bit field) opened
//!   jointly with a `fp128::OneHot` final group, so the two groups in one root
//!   disagree on their committed-source bound.
//! - `bounded_dense_roundtrip_over_u64_coefficients_at_every_catalog_size` —
//!   The bounded family's own scalar rows [14, 24, 26] over the workload
//!   the preset exists for: full-width `u64` coefficients on both signs.
//! - `bounded_dense_declares_a_bound_that_contains_every_u64`. The
//!   bound is a *signed* bit width, so covering `u64::MAX` takes 65, not 64.
//! - `bounded_dense_commit_rejects_a_coefficient_above_the_declared_bound`. The
//!   producer-side guard enforces the *declared* interval, which is
//!   strictly tighter than what the digits can represent.

#![allow(missing_docs)]

mod common;
#[path = "akita_fp128_e2e/heterogeneous.rs"]
mod heterogeneous;
mod matrix_drivers;

use akita_config::{proof_optimized::fp128, CommitmentConfig};
use akita_prover::{
    batched_prove, CommitCluster, ComputeBackendSetup, CpuBackend, MultilinearPolynomial,
    OpeningCluster, ProverComputeStack, RingSwitchCluster, TensorCluster, UniformProverStack,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, BasisMode, GroupBatchStatement, OpeningClaims, OpeningClaimsLayout,
    PolynomialGroupClaims,
};
use common::*;
use matrix_drivers::*;

// ============================================================================
// matrix_test! — generic driver for fp128 (Field = ExtField = F)
// ============================================================================

macro_rules! matrix_test {
    (dense; $name:ident; $cfg:ty; nvs=[$($nv:expr),+]) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_dense_roundtrip::<$cfg>(
                    &[$($nv),+],
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    (onehot; $name:ident; $cfg:ty; nvs=[$($nv:expr),+]) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_onehot_roundtrip::<$cfg>(
                    &[$($nv),+],
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    ($(#[$attr:meta])* dense_pre; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_dense_precommitted_roundtrip::<$cfg>(
                    &[$($nv),+],
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    ($(#[$attr:meta])* onehot_pre; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_onehot_precommitted_roundtrip::<$cfg>(
                    &[$($nv),+],
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    // recursive mode, no user precommit (fp128_onehot_recursive.rs schedule)
    (recursive_direct; $name:ident; $base_cfg:ty) => {
        #[test]
        #[ignore = "production-sized; run explicitly with --release"]
        fn $name() {
            prove_verify_recursive_direct_roundtrip::<$base_cfg>(
                concat!("completeness/", stringify!($name)).as_bytes(),
            );
        }
    };
    // recursive mode + user precommitted groups (profiles from the base config's scalar row)
    (recursive_pre; $name:ident; $base_cfg:ty) => {
        #[test]
        #[ignore = "production-sized; run explicitly with --release"]
        fn $name() {
            recursive_multi_group_round_trip::<$base_cfg>(
                concat!("completeness/", stringify!($name)).as_bytes(),
                |_| {},
            );
        }
    };
}

// ============================================================================
// GROUP B — fp128  (Field = ExtField = fp128::Field)
//
// Full cartesian product: {Dense, OneHot} × {sc, mc} × {direct, pre} × {nonrec, rec}
// Generic driver (prove_verify_*) used throughout.
//
// NA cells have no exact production catalog row and are absent from the source
// rather than marked #[ignore]. These are all Dense recursive cells, Dense
// multi-chunk pre, and OneHot multi-chunk nonrecursive pre. Declared recursive
// and multi-chunk cells are feature-gated and #[ignore]d only when their exact
// production workloads are too large for the default suite.
// ============================================================================

// ----------------------------------------------------------------------------
// Dense × single-chunk × direct × non-recursive    [14, 16, 24, 26]
// ----------------------------------------------------------------------------
matrix_test!(dense; fp128_dense; fp128::Dense; nvs=[14, 16, 24, 26]);

// Dense × single-chunk × precommitted × non-recursive    [16]
// Catalog row: final=(16,1) <- pre=[(14,1)].
matrix_test!(dense_pre; fp128_dense_pre; fp128::Dense; final_nvs=[16]);

// ----------------------------------------------------------------------------
// Dense × multi-chunk × direct × non-recursive    [16]  (feature-gated)
// ----------------------------------------------------------------------------
#[test]
fn fp128_dense_mc() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let catalog =
            akita_config::test_support::workspace_schedule_catalog::<fp128::DenseMultiChunk>()
                .expect("dense multi-chunk catalog");
        let schedule = catalog
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::singleton(16),
            ))
            .expect("dense multi-chunk schedule")
            .schedule()
            .clone();
        assert_eq!(
            schedule.root.params.witness_chunk.num_chunks, 8,
            "W8R2 regression profile must retain eight root witness chunks"
        );
        let first_fold = schedule
            .recursive_folds
            .first()
            .expect("dense multi-chunk schedule must have a recursive fold");
        assert_eq!(
            first_fold.params.witness_chunk.num_chunks, 8,
            "W8R2 regression profile must retain eight witness chunks"
        );
        prove_verify_dense_roundtrip::<fp128::DenseMultiChunk>(
            &[16],
            b"completeness/fp128_dense_mc",
        );
    });
}

// ----------------------------------------------------------------------------
// Dense × multi-chunk × precommitted × non-recursive — NA
// ----------------------------------------------------------------------------
// The fp128::DenseMultiChunk catalog ships a single scalar size (nv=16), and
// nv=16 has no multi-group schedule with the required two folds — the DP needs
// final_nv >= 20. Backing this cell would mean adding a new production size to
// the multi-chunk family purely for a test, so the cell is intentionally
// absent rather than backed by a test-only schedule.

// ----------------------------------------------------------------------------
// OneHot × single-chunk × direct × non-recursive    [12, 15, 20, 28]
// ----------------------------------------------------------------------------
matrix_test!(onehot; fp128_onehot; fp128::OneHot; nvs=[12, 15, 20, 28]);

// OneHot × single-chunk × precommitted × non-recursive    [16, 20]
// Catalog rows: final=(16,1) and final=(20,1), both <- pre=[(14,1)].
matrix_test!(onehot_pre; fp128_onehot_pre; fp128::OneHot; final_nvs=[16, 20]);

// ----------------------------------------------------------------------------
// OneHot × single-chunk × direct × recursive    (production-sized, ignored)
// RecursiveCommitmentConfig, no user precommit; uses fp128_onehot_recursive.rs schedule.
// ----------------------------------------------------------------------------
matrix_test!(recursive_direct; fp128_onehot_rec; fp128::OneHot);

// ----------------------------------------------------------------------------
// OneHot × single-chunk × precommitted × recursive    (production-sized, ignored)
// RecursiveCommitmentConfig + user precommit; profiles from the base config's scalar row.
// ----------------------------------------------------------------------------
matrix_test!(recursive_pre; fp128_onehot_rec_pre; fp128::OneHot);

// ----------------------------------------------------------------------------
// OneHot × multi-chunk × direct × recursive    (production-sized, ignored)
// RecursiveCommitmentConfig<OneHotMultiChunk>; uses fp128_onehot_recursive_multi_chunk_w8r2.rs.
// ----------------------------------------------------------------------------
matrix_test!(recursive_direct; fp128_onehot_mc_rec; fp128::OneHotMultiChunk);

// ----------------------------------------------------------------------------
// OneHot × multi-chunk × precommitted × recursive    (production-sized, ignored)
// RecursiveCommitmentConfig<OneHotMultiChunk> + user precommit;
// profiles from the base config's scalar row.
// ----------------------------------------------------------------------------
matrix_test!(recursive_pre; fp128_onehot_mc_rec_pre; fp128::OneHotMultiChunk);

// ----------------------------------------------------------------------------
// OneHot × multi-chunk × direct × non-recursive    [32]
// (production-sized schedule; run explicitly with --release)
// ----------------------------------------------------------------------------
// Catalog-only companion of `fp128_onehot_mc`: the roundtrip below is
// production-sized and stays ignored, so this cheap check is what CI runs to
// keep the W8R2 feature graph wired to a real catalog row.
#[test]
fn fp128_onehot_mc_catalog_resolves() {
    let catalog =
        akita_config::test_support::workspace_schedule_catalog::<fp128::OneHotMultiChunk>()
            .expect("one-hot multi-chunk catalog");
    let opening_batch = OpeningClaimsLayout::new(32, 1).expect("opening batch");
    catalog
        .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
            opening_batch
                .root_final_group_layout()
                .expect("root group layout"),
        ))
        .expect("W8R2 multi-chunk catalog row");
}

#[test]
#[ignore = "production-sized; run explicitly with --release"]
fn fp128_onehot_mc() {
    init_rayon_pool();
    run_on_large_stack(|| {
        prove_verify_onehot_roundtrip::<fp128::OneHotMultiChunk>(
            &[32],
            b"completeness/fp128_onehot_mc",
        );
    });
}

// ----------------------------------------------------------------------------
// OneHot × multi-chunk × precommitted × non-recursive — NA
// ----------------------------------------------------------------------------
// The catalog has direct final=32 and combined final=14, pre=14 rows, but no
// combined final=32, pre=14 row. The exact matrix cell therefore cannot run.

// ============================================================================
// GROUP C — Batched commitment (multiple polynomials in a single group)
//
// Tests that the batch-commit path correctly handles >1 polynomials per group,
// Homogeneous dense and one-hot groups round-trip. A mixed-representation
// group has the same public geometry and is therefore protocol equivalent.
// ============================================================================

#[test]
fn fp128_onehot_batched() {
    fn run(nv: usize, batch_size: usize) {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let polys: Vec<_> = (0..batch_size)
            .map(|i| make_onehot_poly::<OneHotCfg>(nv, 0xa66e_0000 + nv as u64 * 100 + i as u64))
            .collect();
        let pt = random_point(nv, 0xf00d_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|p| onehot_opening_lagrange(p, &pt))
            .collect();

        let setup = scheme.setup_prover(nv, batch_size).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit::<_, _>(
                &setup,
                &polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let poly_refs: Vec<_> = polys.iter().collect();

        let mut prover_transcript = AkitaTranscript::<F>::new(b"completeness/fp128_onehot_batched");
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<OneHotCfg, _>(
                    &pt[..],
                    &poly_refs[..],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_onehot_batched");
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<OneHotCfg>(&pt[..], &openings, &commitment, scheme.schedules()),
                BasisMode::Lagrange,
            )
            .unwrap_or_else(|e| panic!("onehot nv={nv} batch={batch_size}: {e:?}"));
    }
    init_rayon_pool();
    run_on_large_stack(|| {
        run(12, 1);
        run(20, 4);
    });
}

#[test]
fn fp128_dense_batched() {
    fn run(nv: usize, batch_size: usize) {
        let scheme = load_workspace_scheme::<DenseCfg>().expect("workspace schedule catalog");
        let seeds: Vec<u64> = (0..batch_size)
            .map(|i| 0xd3e5_0000 + nv as u64 * 100 + i as u64)
            .collect();
        let evals: Vec<Vec<F>> = seeds.iter().map(|&s| dense_field_evals(nv, s)).collect();
        let polys: Vec<_> = evals
            .iter()
            .map(|e| akita_prover::DensePoly::<F>::from_field_evals(nv, e).expect("dense poly"))
            .collect();
        let pt = random_point(nv, 0xaaaa_0000 + nv as u64);
        let openings: Vec<F> = evals
            .iter()
            .map(|e| dense_opening_lagrange(e, &pt))
            .collect();

        let setup = scheme.setup_prover(nv, batch_size).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit::<_, _>(
                &setup,
                &polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let poly_refs: Vec<_> = polys.iter().collect();

        let mut prover_transcript = AkitaTranscript::<F>::new(b"completeness/fp128_dense_batched");
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<DenseCfg, _>(
                    &pt[..],
                    &poly_refs[..],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_dense_batched");
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<DenseCfg>(&pt[..], &openings, &commitment, scheme.schedules()),
                BasisMode::Lagrange,
            )
            .unwrap_or_else(|e| panic!("dense nv={nv} batch={batch_size}: {e:?}"));
    }
    init_rayon_pool();
    run_on_large_stack(|| {
        run(14, 1);
        run(17, 4);
    });
}

#[test]
fn fp128_mixed_batched_uses_source_free_group_geometry() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const NV: usize = 17;
        const BATCH: usize = 4;
        let scheme = load_workspace_scheme::<DenseCfg>().expect("workspace schedule catalog");
        let opening_batch = OpeningClaimsLayout::new(NV, BATCH).expect("opening batch");
        let layout = scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                opening_batch
                    .root_final_group_layout()
                    .expect("root group layout"),
            ))
            .expect("layout")
            .schedule()
            .clone()
            .root
            .params;

        let root_d = layout.d_a();
        let total_field =
            layout.blocks().live_blocks * layout.blocks().positions_per_block * root_d;
        let onehot_k = root_d;
        let num_chunks = total_field / onehot_k;
        let make_mixed_onehot = |seed: u64| {
            let mut r = StdRng::seed_from_u64(seed);
            let indices: Vec<Option<u8>> = (0..num_chunks)
                .map(|_| Some(r.gen_range(0..onehot_k) as u8))
                .collect();
            akita_prover::OneHotPoly::<F, u8>::new(onehot_k, indices).expect("mixed onehot poly")
        };

        let evals_a = dense_field_evals(NV, 0x4d10_0001);
        let evals_b = dense_field_evals(NV, 0x4d10_0002);
        let dense_a =
            akita_prover::DensePoly::<F>::from_field_evals(NV, &evals_a).expect("dense a");
        let dense_b =
            akita_prover::DensePoly::<F>::from_field_evals(NV, &evals_b).expect("dense b");
        let onehot_a = make_mixed_onehot(0x4d10_1001);
        let onehot_b = make_mixed_onehot(0x4d10_1002);

        let polys = [
            MultilinearPolynomial::dense(dense_a),
            MultilinearPolynomial::onehot(onehot_a),
            MultilinearPolynomial::dense(dense_b),
            MultilinearPolynomial::onehot(onehot_b),
        ];

        let setup = scheme.setup_prover(NV, BATCH).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let output = scheme
            .commit::<_, _>(
                &setup,
                &polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("mixed source representations share one public geometry");
        assert_eq!(
            output.committed_group.profile.group,
            opening_batch
                .root_final_group_layout()
                .expect("final group layout")
        );
    });
}

// ============================================================================
// GROUP D — Edge cases and special configurations
//
// Tests that are important for correctness but do not fit neatly into the
// Group A/B cartesian product (oversized setup, monomial basis mode, etc.).
// ============================================================================

// Setup allocated for a larger nv than the polynomial actually occupies.
#[test]
fn fp128_onehot_oversized_setup() {
    fn run(setup_nv: usize, poly_nv: usize) {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let opening_batch = OpeningClaimsLayout::new(poly_nv, 1).expect("singleton opening batch");
        let layout = scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                opening_batch
                    .root_final_group_layout()
                    .expect("singleton group layout"),
            ))
            .expect("layout")
            .schedule()
            .clone()
            .root
            .params;
        let d = layout.d_a();
        let total_field = layout.blocks().live_blocks * layout.blocks().positions_per_block * d;
        let onehot_k =
            akita_config::unit_onehot_source_chunk_size::<OneHotCfg>().expect("one-hot config");
        let total_chunks = total_field / onehot_k;

        let mut rng = StdRng::seed_from_u64(0xdead_beef_0000 + poly_nv as u64);
        let indices: Vec<Option<u8>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
            .collect();
        let poly = akita_prover::OneHotPoly::<F, u8>::new(onehot_k, indices).expect("onehot poly");

        let pt = random_point(poly_nv, 0xcafe_0000 + poly_nv as u64);
        let expected_opening = onehot_opening_lagrange(&poly, &pt);

        let setup = scheme.setup_prover(setup_nv, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit::<_, _>(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let poly_refs = [&poly];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_onehot_oversized_setup");
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<OneHotCfg, _>(
                    &pt[..],
                    &poly_refs[..],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let openings = [expected_opening];
        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_onehot_oversized_setup");
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<OneHotCfg>(&pt[..], &openings[..], &commitment, scheme.schedules()),
                BasisMode::Lagrange,
            )
            .unwrap_or_else(|e| {
                panic!("oversized setup (setup_nv={setup_nv}, poly_nv={poly_nv}): {e:?}")
            });
    }
    init_rayon_pool();
    run_on_large_stack(|| {
        run(15, 12);
        run(20, 15);
    });
}

// Monomial basis mode: prover and verifier both use BasisMode::Monomial.
#[test]
fn fp128_dense_monomial_basis() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const NV: usize = 14;
        let scheme = load_workspace_scheme::<DenseCfg>().expect("workspace schedule catalog");
        let evals = dense_field_evals(NV, 0xb0b0_0000);
        let poly = akita_prover::DensePoly::<F>::from_field_evals(NV, &evals).expect("dense poly");
        let pt = random_point(NV, 0xc0de_0000);
        let expected_opening = dense_opening_monomial(&evals, &pt);

        let setup = scheme.setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit::<_, _>(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let poly_refs = [&poly];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_dense_monomial_basis");
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<DenseCfg, _>(
                    &pt[..],
                    &poly_refs[..],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Monomial,
            )
            .expect("monomial prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let openings = [expected_opening];
        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_dense_monomial_basis");
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<DenseCfg>(&pt[..], &openings[..], &commitment, scheme.schedules()),
                BasisMode::Monomial,
            )
            .expect("monomial verify");
    });
}
