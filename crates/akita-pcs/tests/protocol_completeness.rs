//! Canonical home for every honest prove→verify correctness test in the Akita PCS.
//!
//! All tests assert that an honest prover/verifier cycle succeeds for a particular
//! (config, nv, poly-type) combination. Rejection/tamper/structural tests live in
//! their respective crates and files.

#![allow(missing_docs)]
#![cfg(feature = "schedules-default")]

mod common;

use akita_config::proof_optimized::{fp32, fp128};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    batched_prove, CommitCluster, ComputeBackendSetup, CpuBackend, MultilinearPolynomial,
    OpeningCluster, ProverComputeStack, RingSwitchCluster, TensorCluster, UniformProverStack,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, BasisMode, GroupBatchStatement, OpeningClaims, OpeningClaimsLayout,
    PolynomialGroupClaims, PolynomialGroupLayout,
};
use common::*;

// ---------------------------------------------------------------------------
// matrix_test! macro: generates a test function for each fp128 config variant
// ---------------------------------------------------------------------------

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
    (dense_precommitted; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]) => {
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
    (onehot_precommitted; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]; k=$k:expr) => {
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
    (recursive; $name:ident; $base_cfg:ty) => {
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

// ===========================================================================
// Group A – Single-chunk, no precommit
// ===========================================================================

matrix_test!(dense;  fp128_dense;  fp128::Dense;  nvs=[14, 16, 24, 26]);
matrix_test!(onehot; fp128_onehot; fp128::OneHot; nvs=[12, 15, 20, 28]; k=256);

#[test]
fn fp32_onehot() {
    type SmallCfg = fp32::OneHot;
    type SmallF = fp32::Field;
    type SmallE = fp32::ExtensionField;
    type SmallScheme = AkitaCommitmentScheme<SmallCfg>;
    const SMALL_D: usize = SmallCfg::D;
    const SMALL_NV: usize = 16;
    const SMALL_BATCH: usize = 2;
    const LABEL: &[u8] = b"completeness/fp32_onehot";

    use akita_field::ExtField;
    use akita_types::lagrange_weights;

    init_rayon_pool();
    run_on_large_stack(|| {
        let onehot_k = 256usize;
        let num_chunks = (1usize << SMALL_NV) / onehot_k;
        let make_poly = |seed: usize| {
            let indices = (0..num_chunks)
                .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
                .collect();
            akita_prover::OneHotPoly::<SmallF, u8>::new(onehot_k, SMALL_D, indices)
                .expect("fp32 onehot poly")
        };

        let point: Vec<SmallE> = (0..SMALL_NV)
            .map(|i| {
                SmallE::from_base_slice(&[
                    SmallF::from_u64((i * 5 + 1) as u64),
                    SmallF::from_u64((i * 5 + 2) as u64),
                    SmallF::from_u64((i * 5 + 3) as u64),
                    SmallF::from_u64((i * 5 + 4) as u64),
                ])
            })
            .collect();

        let weights = lagrange_weights(&point).expect("lagrange weights");
        let opening_for = |poly: &akita_prover::OneHotPoly<SmallF, u8>| -> SmallE {
            poly.indices()
                .iter()
                .enumerate()
                .filter_map(|(chunk, hot)| {
                    hot.map(|idx| weights[chunk * onehot_k + usize::from(idx)])
                })
                .fold(SmallE::zero(), |acc, w| acc + w)
        };

        let polys = [make_poly(0), make_poly(1)];
        let openings: Vec<SmallE> = polys.iter().map(opening_for).collect();
        let poly_refs: Vec<_> = polys.iter().collect();

        let setup = SmallScheme::setup_prover(SMALL_NV, SMALL_BATCH).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack = UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
            .expect("stack");
        let verifier_setup = SmallScheme::setup_verifier(&setup).expect("verifier setup");

        let (commitment, hint) =
            SmallScheme::commit(&setup, &polys, &stack).expect("fp32 commit");

        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![SmallE::zero(); SMALL_BATCH],
            commitment.clone(),
        )
        .expect("prover group")])
        .expect("prover claims");

        let prover_data = selected_prover_data::<SmallCfg, _>(
            prover_claims,
            vec![hint],
            vec![&poly_refs[..]],
        );
        let selection = prover_data.0;

        let mut prover_transcript = AkitaTranscript::<SmallF>::new(LABEL);
        let proof = SmallScheme::batched_prove::<_, _, _>(
            &setup,
            prover_data,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("fp32 prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_uncompressed(&mut bytes).expect("serialize");
        let decoded =
            AkitaBatchedProof::<SmallF, SmallE>::deserialize_uncompressed(&bytes[..], &shape)
                .expect("deserialize");

        let verify_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            openings.clone(),
            &commitment,
        )
        .expect("verifier group")])
        .expect("verifier claims");
        let mut verifier_transcript = AkitaTranscript::<SmallF>::new(LABEL);
        SmallScheme::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            GroupBatchStatement::new(selection, verify_claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("fp32 verify");
    });
}

// ===========================================================================
// Group B – Multi-chunk
// ===========================================================================

#[cfg(feature = "schedules-fp128-dense-multi-chunk")]
matrix_test!(dense; fp128_dense_multi_chunk; fp128::DenseMultiChunk; nvs=[16]);

#[cfg(feature = "schedules-fp128-onehot-multi-chunk")]
#[test]
#[ignore = "production-sized; run explicitly with --release"]
fn fp128_onehot_multi_chunk() {
    init_rayon_pool();
    run_on_large_stack(|| {
        prove_verify_onehot_roundtrip::<fp128::OneHotMultiChunkW2R2>(
            &[32],
            256,
            b"completeness/fp128_onehot_multi_chunk",
        );
    });
}

// ===========================================================================
// Group C – Precommitted (non-recursive)
// ===========================================================================

matrix_test!(dense_precommitted;  fp128_dense_precommitted;  fp128::Dense;  final_nvs=[16]);
matrix_test!(onehot_precommitted; fp128_onehot_precommitted; fp128::OneHot; final_nvs=[16, 20]; k=256);

// ===========================================================================
// Group D – Aggregated batch (multiple polys in one commitment)
// ===========================================================================

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
        let stack = UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
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
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut std::io::Cursor::new(bytes), &shape)
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
        let stack = UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
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
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut std::io::Cursor::new(bytes), &shape)
                .expect("deserialize");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"completeness/fp128_dense_batched");
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
            let indices: Vec<Option<u8>> =
                (0..num_chunks).map(|_| Some(r.gen_range(0..onehot_k) as u8)).collect();
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

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/fp128_mixed_batched");
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
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut std::io::Cursor::new(bytes), &shape)
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

// ===========================================================================
// Group E – Special / edge-case configurations
// ===========================================================================

#[test]
fn fp128_onehot_oversized_setup() {
    fn run(setup_nv: usize, poly_nv: usize) {
        let opening_batch =
            OpeningClaimsLayout::new(poly_nv, 1).expect("singleton opening batch");
        let layout = OneHotCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let d = layout.d_a();
        let total_field = layout.num_live_blocks * layout.num_positions_per_block * d;
        let total_chunks = total_field / ONEHOT_K;

        let mut rng = StdRng::seed_from_u64(0xdead_beef_0000 + poly_nv as u64);
        let indices: Vec<Option<u8>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
            .collect();
        let poly = akita_prover::OneHotPoly::<F, u8>::new(ONEHOT_K, d, indices)
            .expect("onehot poly");

        let pt = random_point(poly_nv, 0xcafe_0000 + poly_nv as u64);
        let expected_opening = opening_from_poly_for_layout(&poly, &pt, &layout);

        let setup = AkitaCommitmentScheme::<OneHotCfg>::setup_prover(setup_nv, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<OneHotCfg>::setup_verifier(&setup).expect("verifier setup");

        let (commitment, hint) =
            AkitaCommitmentScheme::<OneHotCfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
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
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut std::io::Cursor::new(bytes), &shape)
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

#[test]
fn fp128_dense_monomial_basis() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const NV: usize = 14;
        let opening_batch = OpeningClaimsLayout::new(NV, 1).expect("opening batch");
        let layout = DenseCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let poly = make_dense_poly(NV, 0xb0b0_0000);
        let pt = random_point(NV, 0xc0de_0000);
        let expected_opening = opening_from_poly_with_basis::<64, _>(
            &poly,
            &pt,
            &layout,
            BasisMode::Monomial,
        );

        let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<DenseCfg>::setup_verifier(&setup).expect("verifier setup");

        let (commitment, hint) =
            AkitaCommitmentScheme::<DenseCfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
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
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut std::io::Cursor::new(bytes), &shape)
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

#[test]
fn fp32_onehot_multi_group() {
    type SmallCfg = fp32::OneHot;
    type SmallF = fp32::Field;
    type SmallE = fp32::ExtensionField;
    type SmallScheme = AkitaCommitmentScheme<SmallCfg>;
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;

    use akita_field::ExtField;
    use akita_types::AkitaScheduleLookupKey;

    init_rayon_pool();
    run_on_large_stack(|| {
        let make_ext_point = |nv: usize| -> Vec<SmallE> {
            (0..nv)
                .map(|i| {
                    SmallE::from_base_slice(&[
                        SmallF::from_u64((i * 5 + 1) as u64),
                        SmallF::from_u64((i * 5 + 2) as u64),
                        SmallF::from_u64((i * 5 + 3) as u64),
                        SmallF::from_u64((i * 5 + 4) as u64),
                    ])
                })
                .collect()
        };

        let onehot_opening_at =
            |poly: &akita_prover::OneHotPoly<SmallF, u8>, point: &[SmallE]| -> SmallE {
                let k = poly.onehot_k();
                poly.indices()
                    .iter()
                    .enumerate()
                    .filter_map(|(chunk, hot)| {
                        hot.map(|idx| {
                            let eval_idx = chunk * k + usize::from(idx);
                            point.iter().enumerate().fold(SmallE::one(), |w, (v, &c)| {
                                if (eval_idx >> v) & 1 == 0 {
                                    w * (SmallE::one() - c)
                                } else {
                                    w * c
                                }
                            })
                        })
                    })
                    .fold(SmallE::zero(), |acc, w| acc + w)
            };

        let grouped_poly = |params: &CommittedGroupParams, seed: usize| {
            let onehot_k = 256usize;
            let total = params.num_live_blocks * params.num_positions_per_block * params.d_a();
            let indices = (0..total / onehot_k)
                .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
                .collect();
            akita_prover::OneHotPoly::<SmallF, u8>::new(onehot_k, params.d_a(), indices)
                .expect("grouped fp32 poly")
        };

        let pre_group_schedule = SmallCfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::new(PRE_NV, 1),
        ))
        .expect("pre schedule");
        let pre_params = &pre_group_schedule.root.params.final_group.commitment;
        let pre_poly = grouped_poly(pre_params, 1);

        let pre_setup = SmallScheme::setup_prover(PRE_NV, 1).expect("pre setup");
        let pre_prepared = CpuBackend::DEFAULT.prepare_setup(&pre_setup).expect("prepared");
        let pre_stack = UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &pre_prepared,
            pre_setup.expanded.as_ref(),
        )
        .expect("pre stack");
        let (pre_commitment, pre_hint) =
            SmallScheme::commit_group(&pre_setup, std::slice::from_ref(&pre_poly), &pre_stack)
                .expect("precommit");

        let multi_schedule = SmallCfg::runtime_schedule(AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(FINAL_NV, 1),
            precommitteds: vec![pre_commitment.profile],
        })
        .expect("multi-group schedule");
        let final_params = &multi_schedule.root.params.final_group.commitment;
        let final_poly = grouped_poly(final_params, 2);

        let setup = SmallScheme::setup_prover(FINAL_NV, 2).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");
        let verifier_setup = SmallScheme::setup_verifier(&setup).expect("verifier setup");

        let (final_commitment, final_hint, _sel) = SmallScheme::commit_final_group(
            &setup,
            std::slice::from_ref(&final_poly),
            &stack,
            vec![pre_commitment.profile],
        )
        .expect("final commit");

        let mut pre_point = make_ext_point(PRE_NV);
        pre_point[0] += SmallE::one();
        let final_point = make_ext_point(FINAL_NV);
        let pre_opening = onehot_opening_at(&pre_poly, &pre_point);
        let final_opening = onehot_opening_at(&final_poly, &final_point);

        let pre_refs = [&pre_poly];
        let final_refs = [&final_poly];
        let prover_data = selected_prover_data::<SmallCfg, _>(
            OpeningClaims::from_groups(vec![
                PolynomialGroupClaims::new(
                    pre_point.clone(),
                    vec![pre_opening],
                    pre_commitment.clone(),
                )
                .expect("pre prover group"),
                PolynomialGroupClaims::new(
                    final_point.clone(),
                    vec![final_opening],
                    final_commitment.clone(),
                )
                .expect("final prover group"),
            ])
            .expect("prover claims"),
            vec![pre_hint, final_hint],
            vec![&pre_refs[..], &final_refs[..]],
        );
        let selection = prover_data.0;

        let mut prover_transcript =
            AkitaTranscript::<SmallF>::new(b"completeness/fp32_onehot_multi_group");
        let proof = SmallScheme::batched_prove(
            &setup,
            prover_data,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("fp32 multi-group prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_uncompressed(&mut bytes).expect("serialize");
        let decoded =
            AkitaBatchedProof::<SmallF, SmallE>::deserialize_uncompressed(&bytes[..], &shape)
                .expect("deserialize");

        let verify_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(pre_point, vec![pre_opening], &pre_commitment)
                .expect("pre verifier group"),
            PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
                .expect("final verifier group"),
        ])
        .expect("verifier claims");
        let mut verifier_transcript =
            AkitaTranscript::<SmallF>::new(b"completeness/fp32_onehot_multi_group");
        SmallScheme::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            GroupBatchStatement::new(selection, verify_claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("fp32 multi-group verify");
    });
}

#[test]
fn heterogeneous_group_types() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const ONEHOT_PRE_NV: usize = 14;
        const DENSE_PRE_NV: usize = 15;
        const FINAL_NV: usize = 16;

        let setup = AkitaCommitmentScheme::<OneHotCfg>::setup_prover(FINAL_NV, 4).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack = UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");

        let onehot_pre_params: CommittedGroupParams =
            OneHotCfg::runtime_schedule(akita_types::AkitaScheduleLookupKey::single(
                PolynomialGroupLayout::new(ONEHOT_PRE_NV, 1),
            ))
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
        let dense_a = akita_prover::DensePoly::from_field_evals(DENSE_PRE_NV, DENSE_D, &dense_evals_a)
            .expect("dense a");
        let dense_b = akita_prover::DensePoly::from_field_evals(DENSE_PRE_NV, DENSE_D, &dense_evals_b)
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
        let dense_pre_params: CommittedGroupParams =
            OneHotCfg::runtime_schedule(akita_types::AkitaScheduleLookupKey::single(
                PolynomialGroupLayout::new(DENSE_PRE_NV, 2),
            ))
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
        let dense_opening_a = opening_from_poly_for_layout(&dense_a, &dense_point, &dense_pre_params);
        let dense_opening_b = opening_from_poly_for_layout(&dense_b, &dense_point, &dense_pre_params);
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
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut std::io::Cursor::new(bytes), &shape)
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
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut std::io::Cursor::new(bytes), &shape)
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

// ===========================================================================
// Group F – Recursive (single stage-3 sumcheck)
// ===========================================================================

#[cfg(feature = "schedules-fp128-onehot-recursive")]
matrix_test!(recursive; fp128_onehot_recursive; fp128::OneHot);
