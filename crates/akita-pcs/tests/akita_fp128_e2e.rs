//! Correctness matrix for fp128 Akita PCS prove→verify roundtrips.
//!
//! # Group B — fp128 full correctness matrix
//!
//! `CommitmentConfig<Field=F, ExtField=F>`, so the generic driver is used directly.
//! The table covers the full cartesian product:
//!   poly ∈ {Dense, OneHot} × chunk ∈ {sc, mc} × precommit ∈ {direct, pre} × recursion ∈ {nonrec, rec}
//!
//! "Recursive" always implies precommitted groups (recursive mode operates on a
//! multi-group setup), so the `direct × rec` column is structurally absent.
//!
//! Legend:
//!   ✓        — runs in default `cargo test`  (schedules-default feature, small nv)
//!   cfg      — requires an extra feature flag to compile the schedule tables
//!   ign      — skipped in default `cargo test` due to production-sized nv; needs `-- --ignored`
//!   NA       — no production schedule exists for this combination; cell is intentionally absent
//!
//! cfg and ign are independent: a cell can be cfg-only (schedule tables must be opted in, but
//! the test is fast once compiled), ign-only (default tables, but nv is too large for CI), or
//! both (large tables AND large nv).
//!
//! ```text
//! ╔══════════╦══════════╦══════════════════════════════════╦══════════════════════════════════╗
//! ║          ║          ║      single-chunk (sc)           ║      multi-chunk (mc)            ║
//! ║ poly     ║ rec?     ╠══════════════╦═══════════════════╬══════════════╦═══════════════════╣
//! ║          ║          ║    direct    ║        pre        ║    direct    ║        pre        ║
//! ╠══════════╬══════════╬══════════════╬═══════════════════╬══════════════╬═══════════════════╣
//! ║ Dense    ║ nonrec   ║  ✓ [14,16]  ║  ✓ [16]           ║  ✓cfg [16]  ║  ✓cfg             ║
//! ║ Dense    ║ rec      ║      NA      ║  NA               ║      NA      ║  NA               ║
//! ╠══════════╬══════════╬══════════════╬═══════════════════╬══════════════╬═══════════════════╣
//! ║ OneHot   ║ nonrec   ║  ✓ [12,15]  ║  ✓ [16,20]        ║  cfg+ign    ║  cfg+ign          ║
//! ║ OneHot   ║ rec      ║  cfg+ign    ║  cfg+ign           ║  cfg+ign    ║  cfg+ign          ║
//! ╚══════════╩══════════╩══════════════╩═══════════════════╩══════════════╩═══════════════════╝
//! ```
//!
//! Dense + recursive: no production schedule exists; those cells are permanently NA.
//! OneHot mc nonrec:  cfg=schedules-fp128-onehot-multi-chunk; nv=32 is production-sized (ign).
//! OneHot sc rec:     cfg=schedules-fp128-onehot-recursive; nv=32 is production-sized (ign).
//!   direct = RecursiveCommitmentConfig only, no user precommit (fp128_onehot_recursive.rs).
//!   pre    = RecursiveCommitmentConfig + user precommit (fp128_onehot_recursive_precommitted.rs).
//! OneHot mc rec:     cfg=schedules-fp128-onehot-recursive-multi-chunk; nv=32 is production-sized (ign).
//!   direct = RecursiveCommitmentConfig<OneHotMultiChunk> (fp128_onehot_recursive_multi_chunk_w8r2.rs).
//!   pre    = same + user precommit (fp128_onehot_recursive_multi_chunk_w8r2_precommitted.rs).

#![allow(missing_docs)]
#![cfg(feature = "schedules-default")]

mod common;

use akita_config::proof_optimized::fp128;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    batched_prove, CommitCluster, ComputeBackendSetup, CpuBackend, MultilinearPolynomial,
    OpeningCluster, ProverComputeStack, RingSwitchCluster, TensorCluster, UniformProverStack,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode, GroupBatchStatement, OpeningClaims,
    OpeningClaimsLayout, PolynomialGroupClaims, PolynomialGroupLayout,
};
use common::*;

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
    (onehot; $name:ident; $cfg:ty; nvs=[$($nv:expr),+]; k=$k:expr) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_onehot_roundtrip::<$cfg>(
                    &[$($nv),+],
                    $k,
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    (dense_pre; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]) => {
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
    (onehot_pre; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]; k=$k:expr) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_onehot_precommitted_roundtrip::<$cfg>(
                    &[$($nv),+],
                    $k,
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
    // recursive mode + user precommitted groups (fp128_onehot_recursive_precommitted.rs profiles)
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
// NA cells (Dense + recursive, mc + recursive) have no production schedule and
// are intentionally absent from the source rather than marked #[ignore].
// ============================================================================

// ----------------------------------------------------------------------------
// Dense × single-chunk × direct × non-recursive    [14, 16, 24, 26]
// ----------------------------------------------------------------------------
matrix_test!(dense; fp128_dense; fp128::Dense; nvs=[14, 16, 24, 26]);

// ----------------------------------------------------------------------------
// Dense × single-chunk × precommitted × non-recursive    [16]
// ----------------------------------------------------------------------------
matrix_test!(dense_pre; fp128_dense_pre; fp128::Dense; final_nvs=[16]);

// ----------------------------------------------------------------------------
// Dense × multi-chunk × direct × non-recursive    [16]  (feature-gated)
// ----------------------------------------------------------------------------
#[cfg(feature = "schedules-fp128-dense-multi-chunk")]
matrix_test!(dense; fp128_dense_mc; fp128::DenseMultiChunk; nvs=[16]);

// ----------------------------------------------------------------------------
// Dense × multi-chunk × precommitted × non-recursive    [16]  (feature-gated)
// ----------------------------------------------------------------------------
#[cfg(feature = "schedules-fp128-dense-multi-chunk")]
matrix_test!(dense_pre; fp128_dense_mc_pre; fp128::DenseMultiChunk; final_nvs=[16]);

// ----------------------------------------------------------------------------
// OneHot × single-chunk × direct × non-recursive    [12, 15, 20, 28]
// ----------------------------------------------------------------------------
matrix_test!(onehot; fp128_onehot; fp128::OneHot; nvs=[12, 15, 20, 28]; k=256);

// ----------------------------------------------------------------------------
// OneHot × single-chunk × precommitted × non-recursive    [16, 20]
// ----------------------------------------------------------------------------
matrix_test!(onehot_pre; fp128_onehot_pre; fp128::OneHot; final_nvs=[16, 20]; k=256);

// ----------------------------------------------------------------------------
// OneHot × single-chunk × direct × recursive    (production-sized, ignored)
// RecursiveCommitmentConfig, no user precommit; uses fp128_onehot_recursive.rs schedule.
// ----------------------------------------------------------------------------
#[cfg(feature = "schedules-fp128-onehot-recursive")]
matrix_test!(recursive_direct; fp128_onehot_rec; fp128::OneHot);

// ----------------------------------------------------------------------------
// OneHot × single-chunk × precommitted × recursive    (production-sized, ignored)
// RecursiveCommitmentConfig + user precommit; profiles from fp128_onehot_recursive_precommitted.rs.
// ----------------------------------------------------------------------------
#[cfg(feature = "schedules-fp128-onehot-recursive")]
matrix_test!(recursive_pre; fp128_onehot_rec_pre; fp128::OneHot);

// ----------------------------------------------------------------------------
// OneHot × multi-chunk × direct × recursive    (production-sized, ignored)
// RecursiveCommitmentConfig<OneHotMultiChunk>; uses fp128_onehot_recursive_multi_chunk_w8r2.rs.
// ----------------------------------------------------------------------------
#[cfg(feature = "schedules-fp128-onehot-recursive-multi-chunk")]
matrix_test!(recursive_direct; fp128_onehot_mc_rec; fp128::OneHotMultiChunk);

// ----------------------------------------------------------------------------
// OneHot × multi-chunk × precommitted × recursive    (production-sized, ignored)
// RecursiveCommitmentConfig<OneHotMultiChunk> + user precommit;
// profiles from fp128_onehot_recursive_multi_chunk_w8r2_precommitted.rs.
// ----------------------------------------------------------------------------
#[cfg(feature = "schedules-fp128-onehot-recursive-multi-chunk")]
matrix_test!(recursive_pre; fp128_onehot_mc_rec_pre; fp128::OneHotMultiChunk);

// ----------------------------------------------------------------------------
// OneHot × multi-chunk × direct × non-recursive    [32]
// (production-sized schedule; run explicitly with --release)
// ----------------------------------------------------------------------------
#[cfg(feature = "schedules-fp128-onehot-multi-chunk")]
#[test]
#[ignore = "production-sized; run explicitly with --release"]
fn fp128_onehot_mc() {
    init_rayon_pool();
    run_on_large_stack(|| {
        prove_verify_onehot_roundtrip::<fp128::OneHotMultiChunkW2R2>(
            &[32],
            256,
            b"completeness/fp128_onehot_mc",
        );
    });
}

// ----------------------------------------------------------------------------
// OneHot × multi-chunk × precommitted × non-recursive    [32]
// (production-sized schedule; run explicitly with --release)
// ----------------------------------------------------------------------------
#[cfg(feature = "schedules-fp128-onehot-multi-chunk")]
#[test]
#[ignore = "production-sized; run explicitly with --release"]
fn fp128_onehot_mc_pre() {
    init_rayon_pool();
    run_on_large_stack(|| {
        prove_verify_onehot_precommitted_roundtrip::<fp128::OneHotMultiChunkW2R2>(
            &[32],
            256,
            b"completeness/fp128_onehot_mc_pre",
        );
    });
}

// ============================================================================
// GROUP C — Batched commitment (multiple polynomials in a single group)
//
// Tests that the batch-commit path correctly handles >1 polynomials per group,
// including homogeneous (all dense / all one-hot) and mixed batches.
// ============================================================================

#[test]
fn fp128_onehot_batched() {
    fn run(nv: usize, batch_size: usize) {
        let opening_batch = OpeningClaimsLayout::new(nv, batch_size).expect("opening batch");
        let layout = OneHotCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let polys: Vec<_> = (0..batch_size)
            .map(|i| make_onehot_poly(nv, 0xa66e_0000 + nv as u64 * 100 + i as u64))
            .collect();
        let pt = random_point(nv, 0xf00d_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|p| opening_from_poly_for_layout(p, &pt, &layout))
            .collect();

        let setup = AkitaCommitmentScheme::<OneHotCfg>::setup_prover(nv, batch_size).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<OneHotCfg>::setup_verifier(&setup).expect("verifier setup");

        let (commitment, hint) =
            AkitaCommitmentScheme::<OneHotCfg>::commit::<_, _>(&setup, &polys, &stack)
                .expect("commit");
        let poly_refs: Vec<_> = polys.iter().collect();

        let mut prover_transcript = AkitaTranscript::<F>::new(b"completeness/fp128_onehot_batched");
        let proof = AkitaCommitmentScheme::<OneHotCfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input::<OneHotCfg, _>(&pt[..], &poly_refs[..], &commitment, hint),
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
        AkitaCommitmentScheme::<OneHotCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<OneHotCfg>(&pt[..], &openings, &commitment),
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
        let opening_batch = OpeningClaimsLayout::new(nv, batch_size).expect("opening batch");
        let layout = DenseCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let polys: Vec<_> = (0..batch_size)
            .map(|i| make_dense_poly(nv, 0xd3e5_0000 + nv as u64 * 100 + i as u64))
            .collect();
        let pt = random_point(nv, 0xaaaa_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|p| opening_from_poly_for_layout(p, &pt, &layout))
            .collect();

        let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(nv, batch_size).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<DenseCfg>::setup_verifier(&setup).expect("verifier setup");

        let (commitment, hint) =
            AkitaCommitmentScheme::<DenseCfg>::commit::<_, _>(&setup, &polys, &stack)
                .expect("commit");
        let poly_refs: Vec<_> = polys.iter().collect();

        let mut prover_transcript = AkitaTranscript::<F>::new(b"completeness/fp128_dense_batched");
        let proof = AkitaCommitmentScheme::<DenseCfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input::<DenseCfg, _>(&pt[..], &poly_refs[..], &commitment, hint),
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
        AkitaCommitmentScheme::<DenseCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<DenseCfg>(&pt[..], &openings, &commitment),
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
fn fp128_mixed_batched() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const NV: usize = 17;
        const BATCH: usize = 4;
        let opening_batch = OpeningClaimsLayout::new(NV, BATCH).expect("opening batch");
        let layout = DenseCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");

        let root_d = layout.d_a();
        let total_field = layout.num_live_blocks * layout.num_positions_per_block * root_d;
        let onehot_k = root_d;
        let num_chunks = total_field / onehot_k;
        let make_mixed_onehot = |seed: u64| {
            let mut r = StdRng::seed_from_u64(seed);
            let indices: Vec<Option<u8>> = (0..num_chunks)
                .map(|_| Some(r.gen_range(0..onehot_k) as u8))
                .collect();
            akita_prover::OneHotPoly::<F, u8>::new(onehot_k, root_d, indices)
                .expect("mixed onehot poly")
        };

        let dense_a = make_dense_poly(NV, 0x4d10_0001);
        let dense_b = make_dense_poly(NV, 0x4d10_0002);
        let onehot_a = make_mixed_onehot(0x4d10_1001);
        let onehot_b = make_mixed_onehot(0x4d10_1002);

        let polys = [
            MultilinearPolynomial::dense(dense_a),
            MultilinearPolynomial::onehot(onehot_a),
            MultilinearPolynomial::dense(dense_b),
            MultilinearPolynomial::onehot(onehot_b),
        ];
        let pt = random_point(NV, 0x4d10_ffff);
        let openings: Vec<F> = polys
            .iter()
            .map(|p| opening_from_poly_for_layout(p, &pt, &layout))
            .collect();

        let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(NV, BATCH).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<DenseCfg>::setup_verifier(&setup).expect("verifier setup");

        let (commitment, hint) =
            AkitaCommitmentScheme::<DenseCfg>::commit::<_, _>(&setup, &polys, &stack)
                .expect("mixed commit");
        let poly_refs: Vec<_> = polys.iter().collect();

        let mut prover_transcript = AkitaTranscript::<F>::new(b"completeness/fp128_mixed_batched");
        let proof = AkitaCommitmentScheme::<DenseCfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input::<DenseCfg, _>(&pt[..], &poly_refs[..], &commitment, hint),
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
            AkitaTranscript::<F>::new(b"completeness/fp128_mixed_batched");
        AkitaCommitmentScheme::<DenseCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<DenseCfg>(&pt[..], &openings, &commitment),
            BasisMode::Lagrange,
        )
        .expect("mixed verify");
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
        let opening_batch = OpeningClaimsLayout::new(poly_nv, 1).expect("singleton opening batch");
        let layout = OneHotCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let d = layout.d_a();
        let total_field = layout.num_live_blocks * layout.num_positions_per_block * d;
        let total_chunks = total_field / ONEHOT_K;

        let mut rng = StdRng::seed_from_u64(0xdead_beef_0000 + poly_nv as u64);
        let indices: Vec<Option<u8>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
            .collect();
        let poly =
            akita_prover::OneHotPoly::<F, u8>::new(ONEHOT_K, d, indices).expect("onehot poly");

        let pt = random_point(poly_nv, 0xcafe_0000 + poly_nv as u64);
        let expected_opening = opening_from_poly_for_layout(&poly, &pt, &layout);

        let setup = AkitaCommitmentScheme::<OneHotCfg>::setup_prover(setup_nv, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<OneHotCfg>::setup_verifier(&setup).expect("verifier setup");

        let (commitment, hint) = AkitaCommitmentScheme::<OneHotCfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .expect("commit");
        let poly_refs = [&poly];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_onehot_oversized_setup");
        let proof = AkitaCommitmentScheme::<OneHotCfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input::<OneHotCfg, _>(&pt[..], &poly_refs[..], &commitment, hint),
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
        AkitaCommitmentScheme::<OneHotCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<OneHotCfg>(&pt[..], &openings[..], &commitment),
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
        let opening_batch = OpeningClaimsLayout::new(NV, 1).expect("opening batch");
        let layout = DenseCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let poly = make_dense_poly(NV, 0xb0b0_0000);
        let pt = random_point(NV, 0xc0de_0000);
        let expected_opening =
            opening_from_poly_with_basis::<64, _>(&poly, &pt, &layout, BasisMode::Monomial);

        let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<DenseCfg>::setup_verifier(&setup).expect("verifier setup");

        let (commitment, hint) = AkitaCommitmentScheme::<DenseCfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .expect("commit");
        let poly_refs = [&poly];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_dense_monomial_basis");
        let proof = AkitaCommitmentScheme::<DenseCfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input::<DenseCfg, _>(&pt[..], &poly_refs[..], &commitment, hint),
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
        AkitaCommitmentScheme::<DenseCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<DenseCfg>(&pt[..], &openings[..], &commitment),
            BasisMode::Monomial,
        )
        .expect("monomial verify");
    });
}

// ============================================================================
// GROUP E — Heterogeneous configurations (fp128)
//
// Tests that span multiple commitment groups with different polynomial types or
// compute backends.  Orthogonal to the Group B matrix.
// ============================================================================

// fp128: three commitment groups with heterogeneous polynomial types
// (one-hot precommit + dense precommit + one-hot final), proved jointly.
// This is the key test for the heterogeneous-group code path.
#[test]
fn heterogeneous_group_types() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const ONEHOT_PRE_NV: usize = 14;
        const DENSE_PRE_NV: usize = 15;
        const FINAL_NV: usize = 16;

        let setup = AkitaCommitmentScheme::<OneHotCfg>::setup_prover(FINAL_NV, 4).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");

        let onehot_pre_params: CommittedGroupParams = OneHotCfg::runtime_schedule(
            AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(ONEHOT_PRE_NV, 1)),
        )
        .expect("onehot pre schedule")
        .root
        .params
        .final_group
        .commitment;
        let pre_d = onehot_pre_params.d_a();
        let onehot_k_pre = 16usize;
        let pre_chunks = (1usize << ONEHOT_PRE_NV) / onehot_k_pre;
        let onehot_pre = akita_prover::OneHotPoly::<F, u8>::new(
            onehot_k_pre,
            pre_d,
            (0..pre_chunks)
                .map(|i| (i % 3 == 0).then_some((i % onehot_k_pre) as u8))
                .collect(),
        )
        .expect("K=16 precommitted poly");

        let dense_evals_a = (0..(1usize << DENSE_PRE_NV))
            .map(|i| F::from_u64((i % 257) as u64))
            .collect::<Vec<_>>();
        let dense_evals_b = (0..(1usize << DENSE_PRE_NV))
            .map(|i| F::from_u64((i % 509) as u64))
            .collect::<Vec<_>>();
        let dense_a =
            akita_prover::DensePoly::from_field_evals(DENSE_PRE_NV, DENSE_D, &dense_evals_a)
                .expect("dense a");
        let dense_b =
            akita_prover::DensePoly::from_field_evals(DENSE_PRE_NV, DENSE_D, &dense_evals_b)
                .expect("dense b");

        let final_onehot = make_onehot_poly(FINAL_NV, 0x1701_0000);

        let dense_polys = [dense_a.clone(), dense_b.clone()];
        let final_polys = [MultilinearPolynomial::onehot(final_onehot.clone())];

        let (onehot_pre_commitment, onehot_pre_hint) =
            AkitaCommitmentScheme::<OneHotCfg>::commit_group(
                &setup,
                std::slice::from_ref(&onehot_pre),
                &stack,
            )
            .expect("K=16 precommit");
        let (dense_commitment, dense_hint) =
            AkitaCommitmentScheme::<OneHotCfg>::commit_group(&setup, &dense_polys, &stack)
                .expect("dense precommit");
        let (final_commitment, final_hint, selection) =
            AkitaCommitmentScheme::<OneHotCfg>::commit_final_group(
                &setup,
                &final_polys,
                &stack,
                vec![onehot_pre_commitment.profile, dense_commitment.profile],
            )
            .expect("final commit");

        let schedule = OneHotCfg::resolve_schedule_selection(selection)
            .expect("heterogeneous schedule")
            .schedule()
            .clone();
        let final_params = &schedule.root.params.final_group.commitment;
        let dense_pre_params: CommittedGroupParams = OneHotCfg::runtime_schedule(
            AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(DENSE_PRE_NV, 2)),
        )
        .expect("dense pre schedule")
        .root
        .params
        .final_group
        .commitment;

        let onehot_pre_point: Vec<F> = (0..ONEHOT_PRE_NV)
            .map(|i| F::from_u64((i + 2) as u64))
            .collect();
        let dense_point: Vec<F> = (0..DENSE_PRE_NV)
            .map(|i| F::from_u64((i + 37) as u64))
            .collect();
        let final_point: Vec<F> = (0..FINAL_NV)
            .map(|i| F::from_u64((i + 71) as u64))
            .collect();

        let onehot_pre_opening =
            opening_from_poly_for_layout(&onehot_pre, &onehot_pre_point, &onehot_pre_params);
        let dense_opening_a =
            opening_from_poly_for_layout(&dense_a, &dense_point, &dense_pre_params);
        let dense_opening_b =
            opening_from_poly_for_layout(&dense_b, &dense_point, &dense_pre_params);
        let final_opening = opening_from_poly_for_layout(&final_onehot, &final_point, final_params);

        let onehot_pre_refs = [&MultilinearPolynomial::onehot(onehot_pre.clone())];
        let dense_refs = [
            &MultilinearPolynomial::dense(dense_a.clone()),
            &MultilinearPolynomial::dense(dense_b.clone()),
        ];
        let final_refs = [&final_polys[0]];

        let prover_data = (
            selection,
            akita_prover::ProverOpeningData::new(
                OpeningClaims::from_groups(vec![
                    PolynomialGroupClaims::new(
                        onehot_pre_point.clone(),
                        vec![onehot_pre_opening],
                        onehot_pre_commitment.clone(),
                    )
                    .expect("K=16 prover group"),
                    PolynomialGroupClaims::new(
                        dense_point.clone(),
                        vec![dense_opening_a, dense_opening_b],
                        dense_commitment.clone(),
                    )
                    .expect("dense prover group"),
                    PolynomialGroupClaims::new(
                        final_point.clone(),
                        vec![final_opening],
                        final_commitment.clone(),
                    )
                    .expect("final prover group"),
                ])
                .expect("prover claims"),
                vec![onehot_pre_hint, dense_hint, final_hint],
                vec![&onehot_pre_refs, &dense_refs, &final_refs],
            )
            .expect("prover opening data"),
        );

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_group_types");
        let proof = AkitaCommitmentScheme::<OneHotCfg>::batched_prove(
            &setup,
            prover_data,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("heterogeneous prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let verifier_setup =
            AkitaCommitmentScheme::<OneHotCfg>::setup_verifier(&setup).expect("verifier setup");
        let verify_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                onehot_pre_point,
                vec![onehot_pre_opening],
                &onehot_pre_commitment,
            )
            .expect("K=16 verifier group"),
            PolynomialGroupClaims::new(
                dense_point,
                vec![dense_opening_a, dense_opening_b],
                &dense_commitment,
            )
            .expect("dense verifier group"),
            PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
                .expect("final verifier group"),
        ])
        .expect("verifier claims");
        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_group_types");
        AkitaCommitmentScheme::<OneHotCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            GroupBatchStatement::new(selection, verify_claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("heterogeneous verify");
    });
}

// Compute backend heterogeneity: commit uses CpuBackend, prove uses a split
// ProverComputeStack with separate backends for each phase.
#[test]
fn heterogeneous_compute_backends() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const NV: usize = 16;
        type Cfg = fp128::Dense;
        type Scheme = AkitaCommitmentScheme<Cfg>;

        let opening_batch = OpeningClaimsLayout::new(NV, 1).expect("opening batch");
        let layout = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let evals: Vec<F> = (0..(1usize << NV)).map(|i| F::from_u64(i as u64)).collect();
        let poly = akita_prover::DensePoly::<F>::from_field_evals(NV, DENSE_D, &evals).unwrap();

        let setup = Scheme::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");

        let commit_backend = CommitCluster;
        let opening_backend = OpeningCluster;
        let tensor = TensorCluster;
        let ring = RingSwitchCluster;
        let stack = ProverComputeStack::new(
            (&commit_backend, &prepared),
            (&opening_backend, &prepared),
            (&tensor, &prepared),
            (&ring, &prepared),
            setup.expanded.as_ref(),
        )
        .expect("heterogeneous stack");

        let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
        let commit_stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("commit stack");
        let (commitment, hint) =
            akita_prover::commit::<Cfg, akita_prover::DensePoly<F>, CpuBackend>(
                std::slice::from_ref(&poly),
                setup.expanded.as_ref(),
                &commit_stack,
            )
            .expect("commit");

        let pt: Vec<F> = (0..NV).map(|i| F::from_u64((i + 2) as u64)).collect();
        let expected_opening = opening_from_poly_for_layout(&poly, &pt, &layout);

        let poly_refs = [&poly];
        let commitments = [commitment];
        let prover_data = selected_prover_data::<Cfg, _>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                pt.clone(),
                vec![expected_opening],
                commitments[0].clone(),
            )
            .expect("prover group")])
            .expect("prover claims"),
            vec![hint],
            vec![&poly_refs[..]],
        );
        let (selection, prover_claims) = prover_data;

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_compute_backends");
        let proof = batched_prove::<Cfg, _, _, _, _, _, _>(
            &setup.expanded,
            &setup.prefix_slots,
            &stack,
            selection,
            prover_claims,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("heterogeneous prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_compute_backends");
        Scheme::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            GroupBatchStatement::new(
                selection,
                OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    pt.clone(),
                    vec![expected_opening],
                    &commitments[0],
                )
                .expect("verifier group")])
                .expect("verifier claims"),
            )
            .expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("heterogeneous verify");
    });
}
