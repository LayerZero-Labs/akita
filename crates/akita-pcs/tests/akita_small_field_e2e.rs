//! Correctness matrix for small-field Akita PCS prove→verify roundtrips.
//!
//! # Group A — Small fields
//!
//! Tests the full cartesian product for configurations where `ExtField ≠ Field`
//! (fp32, fp64).  Because the generic fp128 driver cannot be reused, each cell
//! inlines its own Lagrange-weight opening computation via the `small_field_test!`
//! macro.
//!
//! ```text
//! ╔══════════╦═══════════════════╦═══════════════════╗
//! ║ field    ║ Dense             ║ OneHot            ║
//! ╠══════════╬═════════╦═════════╬═════════╦═════════╣
//! ║          ║ direct  ║   pre   ║ direct  ║   pre   ║
//! ╠══════════╬═════════╬═════════╬═════════╬═════════╣
//! ║  fp32    ║    ✓    ║    ✓    ║    ✓    ║    ✓    ║
//! ║  fp64    ║    ✓    ║    ✓    ║    ✓    ║    ✓    ║
//! ╚══════════╩═════════╩═════════╩═════════╩═════════╝
//! ```
//!
//! # Group E (small-field) — Heterogeneous configurations
//!
//! `fp32_onehot_multi_group`: two precommit groups proved jointly, verifying the
//! multi-group code path with a small field.

#![allow(missing_docs)]
#![cfg(feature = "schedules-default")]

mod common;

use akita_config::proof_optimized::{fp32, fp64};
use akita_config::CommitmentConfig;
use akita_field::LiftBase;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend, UniformProverStack};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode,
    CommittedGroupBatchProfile, GroupBatchStatement, OpeningClaims, OpeningClaimsLayout,
    PolynomialGroupClaims, PolynomialGroupLayout,
};
use common::*;

// ============================================================================
// small_field_test! — inline driver for small fields (ExtField ≠ Field)
//
// The opening is computed directly using Lagrange weights over the extension
// field rather than the CpuBackend fold kernel, because the generic
// opening_from_poly_for_layout helper is hardcoded to fp128::Field.
//
// Arms:
//   dense          — single-group, non-precommitted, dense polynomial
//   dense_pre      — two-group (precommit + final), dense polynomial
//   onehot         — single-group, non-precommitted, one-hot polynomial
//   onehot_pre     — two-group (precommit + final), one-hot polynomial
//
// Parameters:
//   $name      — test function identifier
//   $cfg       — CommitmentConfig type (e.g. fp32::Dense)
//   $sf        — base field type  (Cfg::Field)
//   $se        — extension field type  (Cfg::ExtField)
//   nvs        — list of num_vars to test (non-precommitted arms)
//   final_nvs  — list of final-group num_vars (precommitted arms; pre nv = 14)
//   k          — one-hot group size K (onehot arms)
// ============================================================================

macro_rules! small_field_test {
    // ------------------------------------------------------------------
    // dense — single-group, non-precommitted
    // ------------------------------------------------------------------
    (dense; $name:ident; $cfg:ty; $sf:ty; $se:ty; nvs=[$($nv:expr),+]) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                for &nv in &[$($nv),+] {
                    let n = 1usize << nv;
                    let opening_batch =
                        OpeningClaimsLayout::new(nv, 1).expect("opening batch");
                    let layout = <$cfg as CommitmentConfig>::get_params_for_batched_commitment(
                        &opening_batch,
                    )
                    .expect("layout");
                    let d = layout.d_a();

                    let evals: Vec<$sf> = (0..n)
                        .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(7).wrapping_add(13)))
                        .collect();
                    let poly =
                        akita_prover::DensePoly::<$sf>::from_field_evals(nv, d, &evals)
                            .expect("dense poly");

                    let point: Vec<$se> = (0..nv)
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(3).wrapping_add(1)))
                        .collect();
                    let weights = lagrange_weights::<$se>(&point).expect("weights");
                    let expected: $se = (0..n)
                        .map(|i| weights[i] * <$se>::lift_base(evals[i]))
                        .fold(<$se>::from_u64(0), |a, b| a + b);

                    let setup =
                        AkitaCommitmentScheme::<$cfg>::setup_prover(nv, 1).expect("setup");
                    let prepared =
                        CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
                    let stack = UniformProverStack::uniform(
                        &CpuBackend::DEFAULT,
                        &prepared,
                        setup.expanded.as_ref(),
                    )
                    .expect("stack");
                    let verifier_setup =
                        AkitaCommitmentScheme::<$cfg>::setup_verifier(&setup)
                            .expect("verifier setup");

                    let (commitment, hint) = AkitaCommitmentScheme::<$cfg>::commit::<_, _>(
                        &setup,
                        std::slice::from_ref(&poly),
                        &stack,
                    )
                    .expect("commit");
                    let poly_refs = [&poly];

                    let profiles = CommittedGroupBatchProfile {
                        final_group: *commitment.profile(),
                        precommitteds: Vec::new(),
                    };
                    let selection =
                        <$cfg as CommitmentConfig>::select_schedule_for_profiles(&profiles)
                            .expect("schedule")
                            .selection();

                    let prover_claims = OpeningClaims::from_groups(vec![
                        PolynomialGroupClaims::new(
                            point.clone(),
                            vec![<$se>::from_u64(0)],
                            commitment.clone(),
                        )
                        .expect("prover group"),
                    ])
                    .expect("prover claims");
                    let prover_data = (
                        selection,
                        ProverOpeningData::new(prover_claims, vec![hint], vec![&poly_refs[..]])
                            .expect("prover data"),
                    );

                    let mut pt = AkitaTranscript::<$sf>::new(label);
                    let proof = AkitaCommitmentScheme::<$cfg>::batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut pt,
                        BasisMode::Lagrange,
                    )
                    .expect("prove");

                    let shape = proof.shape();
                    let mut bytes = Vec::new();
                    proof.serialize_uncompressed(&mut bytes).expect("serialize");
                    let decoded = AkitaBatchedProof::<$sf, $se>::deserialize_uncompressed(
                        &bytes[..],
                        &shape,
                    )
                    .expect("deserialize");

                    let verify_claims = OpeningClaims::from_groups(vec![
                        PolynomialGroupClaims::new(point, vec![expected], &commitment)
                            .expect("verifier group"),
                    ])
                    .expect("verifier claims");
                    let mut vt = AkitaTranscript::<$sf>::new(label);
                    AkitaCommitmentScheme::<$cfg>::batched_verify(
                        &decoded,
                        &verifier_setup,
                        &mut vt,
                        GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                        BasisMode::Lagrange,
                    )
                    .unwrap_or_else(|e| panic!("{} nv={nv}: {e:?}", stringify!($name)));
                }
            });
        }
    };

    // ------------------------------------------------------------------
    // dense_pre — two-group precommitted, dense polynomial
    // pre-group: nv=PRE_NV=14  |  final-group: nv from final_nvs list
    // ------------------------------------------------------------------
    (dense_pre; $name:ident; $cfg:ty; $sf:ty; $se:ty; final_nvs=[$($fnv:expr),+]) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                const PRE_NV: usize = 14;

                let pre_schedule = <$cfg as CommitmentConfig>::runtime_schedule(
                    AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(PRE_NV, 1)),
                )
                .expect("pre single-group schedule");
                let pre_d = pre_schedule.root.params.final_group.commitment.d_a();
                let pre_n = 1usize << PRE_NV;
                let pre_evals: Vec<$sf> = (0..pre_n)
                    .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(7).wrapping_add(13)))
                    .collect();

                for &final_nv in &[$($fnv),+] {
                    let setup = AkitaCommitmentScheme::<$cfg>::setup_prover(
                        final_nv.max(PRE_NV),
                        2,
                    )
                    .expect("setup");
                    let prepared =
                        CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
                    let stack = UniformProverStack::uniform(
                        &CpuBackend::DEFAULT,
                        &prepared,
                        setup.expanded.as_ref(),
                    )
                    .expect("stack");
                    let verifier_setup =
                        AkitaCommitmentScheme::<$cfg>::setup_verifier(&setup)
                            .expect("verifier setup");

                    let pre_poly = akita_prover::DensePoly::<$sf>::from_field_evals(
                        PRE_NV,
                        pre_d,
                        &pre_evals,
                    )
                    .expect("pre dense poly");
                    let (pre_commitment, pre_hint) =
                        AkitaCommitmentScheme::<$cfg>::commit_group(
                            &setup,
                            std::slice::from_ref(&pre_poly),
                            &stack,
                        )
                        .expect("precommit");

                    let multi_schedule = <$cfg as CommitmentConfig>::runtime_schedule(
                        AkitaScheduleLookupKey {
                            final_group: PolynomialGroupLayout::new(final_nv, 1),
                            precommitteds: vec![pre_commitment.profile],
                        },
                    )
                    .expect("multi-group schedule");
                    let final_d = multi_schedule.root.params.final_group.commitment.d_a();
                    let final_n = 1usize << final_nv;
                    let final_evals: Vec<$sf> = (0..final_n)
                        .map(|i| <$sf>::from_u64((i as u64).wrapping_mul(11).wrapping_add(7)))
                        .collect();
                    let final_poly = akita_prover::DensePoly::<$sf>::from_field_evals(
                        final_nv,
                        final_d,
                        &final_evals,
                    )
                    .expect("final dense poly");
                    let (final_commitment, final_hint, _sel) =
                        AkitaCommitmentScheme::<$cfg>::commit_final_group(
                            &setup,
                            std::slice::from_ref(&final_poly),
                            &stack,
                            vec![pre_commitment.profile],
                        )
                        .expect("final commit");

                    let point: Vec<$se> = (0..final_nv.max(PRE_NV))
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(3).wrapping_add(1)))
                        .collect();
                    let pre_weights =
                        lagrange_weights::<$se>(&point[..PRE_NV]).expect("pre weights");
                    let pre_opening: $se = (0..pre_n)
                        .map(|i| pre_weights[i] * <$se>::lift_base(pre_evals[i]))
                        .fold(<$se>::from_u64(0), |a, b| a + b);
                    let final_weights =
                        lagrange_weights::<$se>(&point[..final_nv]).expect("final weights");
                    let final_opening: $se = (0..final_n)
                        .map(|i| final_weights[i] * <$se>::lift_base(final_evals[i]))
                        .fold(<$se>::from_u64(0), |a, b| a + b);

                    let pre_refs = [&pre_poly];
                    let final_refs = [&final_poly];
                    let prover_data = selected_prover_data::<$cfg, _>(
                        OpeningClaims::from_groups(vec![
                            PolynomialGroupClaims::new(
                                point[..PRE_NV].to_vec(),
                                vec![pre_opening],
                                pre_commitment.clone(),
                            )
                            .expect("pre prover group"),
                            PolynomialGroupClaims::new(
                                point[..final_nv].to_vec(),
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

                    let mut pt = AkitaTranscript::<$sf>::new(label);
                    let proof = AkitaCommitmentScheme::<$cfg>::batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut pt,
                        BasisMode::Lagrange,
                    )
                    .expect("prove");

                    let shape = proof.shape();
                    let mut bytes = Vec::new();
                    proof.serialize_uncompressed(&mut bytes).expect("serialize");
                    let decoded = AkitaBatchedProof::<$sf, $se>::deserialize_uncompressed(
                        &bytes[..],
                        &shape,
                    )
                    .expect("deserialize");

                    let verify_claims = OpeningClaims::from_groups(vec![
                        PolynomialGroupClaims::new(
                            point[..PRE_NV].to_vec(),
                            vec![pre_opening],
                            &pre_commitment,
                        )
                        .expect("pre verifier group"),
                        PolynomialGroupClaims::new(
                            point[..final_nv].to_vec(),
                            vec![final_opening],
                            &final_commitment,
                        )
                        .expect("final verifier group"),
                    ])
                    .expect("verifier claims");
                    let mut vt = AkitaTranscript::<$sf>::new(label);
                    AkitaCommitmentScheme::<$cfg>::batched_verify(
                        &decoded,
                        &verifier_setup,
                        &mut vt,
                        GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                        BasisMode::Lagrange,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "{} pre_nv={PRE_NV} final_nv={final_nv}: {e:?}",
                            stringify!($name)
                        )
                    });
                }
            });
        }
    };

    // ------------------------------------------------------------------
    // onehot — single-group, non-precommitted, one-hot polynomial
    // ------------------------------------------------------------------
    (onehot; $name:ident; $cfg:ty; $sf:ty; $se:ty; nvs=[$($nv:expr),+]; k=$k:expr) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                let onehot_k: usize = $k;
                for &nv in &[$($nv),+] {
                    let opening_batch =
                        OpeningClaimsLayout::new(nv, 1).expect("opening batch");
                    let layout = <$cfg as CommitmentConfig>::get_params_for_batched_commitment(
                        &opening_batch,
                    )
                    .expect("layout");
                    let d = layout.d_a();
                    let num_chunks = (1usize << nv) / onehot_k;
                    let indices: Vec<Option<u8>> = (0..num_chunks)
                        .map(|chunk| {
                            Some(((chunk * 29 + nv * 41 + 7) % onehot_k) as u8)
                        })
                        .collect();
                    let poly =
                        akita_prover::OneHotPoly::<$sf, u8>::new(onehot_k, d, indices)
                            .expect("onehot poly");

                    let point: Vec<$se> = (0..nv)
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
                        .collect();
                    let weights = lagrange_weights::<$se>(&point).expect("weights");
                    let expected: $se = poly
                        .indices()
                        .iter()
                        .enumerate()
                        .filter_map(|(chunk, hot)| {
                            hot.map(|idx| weights[chunk * onehot_k + usize::from(idx)])
                        })
                        .fold(<$se>::from_u64(0), |a, b| a + b);

                    let setup =
                        AkitaCommitmentScheme::<$cfg>::setup_prover(nv, 1).expect("setup");
                    let prepared =
                        CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
                    let stack = UniformProverStack::uniform(
                        &CpuBackend::DEFAULT,
                        &prepared,
                        setup.expanded.as_ref(),
                    )
                    .expect("stack");
                    let verifier_setup =
                        AkitaCommitmentScheme::<$cfg>::setup_verifier(&setup)
                            .expect("verifier setup");

                    let (commitment, hint) = AkitaCommitmentScheme::<$cfg>::commit::<_, _>(
                        &setup,
                        std::slice::from_ref(&poly),
                        &stack,
                    )
                    .expect("commit");
                    let poly_refs = [&poly];

                    let profiles = CommittedGroupBatchProfile {
                        final_group: *commitment.profile(),
                        precommitteds: Vec::new(),
                    };
                    let selection =
                        <$cfg as CommitmentConfig>::select_schedule_for_profiles(&profiles)
                            .expect("schedule")
                            .selection();

                    let prover_claims = OpeningClaims::from_groups(vec![
                        PolynomialGroupClaims::new(
                            point.clone(),
                            vec![<$se>::from_u64(0)],
                            commitment.clone(),
                        )
                        .expect("prover group"),
                    ])
                    .expect("prover claims");
                    let prover_data = (
                        selection,
                        ProverOpeningData::new(prover_claims, vec![hint], vec![&poly_refs[..]])
                            .expect("prover data"),
                    );

                    let mut pt = AkitaTranscript::<$sf>::new(label);
                    let proof = AkitaCommitmentScheme::<$cfg>::batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut pt,
                        BasisMode::Lagrange,
                    )
                    .expect("prove");

                    let shape = proof.shape();
                    let mut bytes = Vec::new();
                    proof.serialize_uncompressed(&mut bytes).expect("serialize");
                    let decoded = AkitaBatchedProof::<$sf, $se>::deserialize_uncompressed(
                        &bytes[..],
                        &shape,
                    )
                    .expect("deserialize");

                    let verify_claims = OpeningClaims::from_groups(vec![
                        PolynomialGroupClaims::new(point, vec![expected], &commitment)
                            .expect("verifier group"),
                    ])
                    .expect("verifier claims");
                    let mut vt = AkitaTranscript::<$sf>::new(label);
                    AkitaCommitmentScheme::<$cfg>::batched_verify(
                        &decoded,
                        &verifier_setup,
                        &mut vt,
                        GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                        BasisMode::Lagrange,
                    )
                    .unwrap_or_else(|e| panic!("{} nv={nv}: {e:?}", stringify!($name)));
                }
            });
        }
    };

    // ------------------------------------------------------------------
    // onehot_pre — two-group precommitted, one-hot polynomial
    // pre-group: nv=PRE_NV=14  |  final-group: nv from final_nvs list
    // ------------------------------------------------------------------
    (onehot_pre; $name:ident; $cfg:ty; $sf:ty; $se:ty; final_nvs=[$($fnv:expr),+]; k=$k:expr) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                let label = concat!("completeness/", stringify!($name)).as_bytes();
                const PRE_NV: usize = 14;
                let onehot_k: usize = $k;

                let pre_schedule = <$cfg as CommitmentConfig>::runtime_schedule(
                    AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(PRE_NV, 1)),
                )
                .expect("pre single-group schedule");
                let pre_d = pre_schedule.root.params.final_group.commitment.d_a();
                let pre_chunks = (1usize << PRE_NV) / onehot_k;
                let pre_indices: Vec<Option<u8>> = (0..pre_chunks)
                    .map(|chunk| Some(((chunk * 29 + 7) % onehot_k) as u8))
                    .collect();

                for &final_nv in &[$($fnv),+] {
                    let setup = AkitaCommitmentScheme::<$cfg>::setup_prover(
                        final_nv.max(PRE_NV),
                        2,
                    )
                    .expect("setup");
                    let prepared =
                        CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
                    let stack = UniformProverStack::uniform(
                        &CpuBackend::DEFAULT,
                        &prepared,
                        setup.expanded.as_ref(),
                    )
                    .expect("stack");
                    let verifier_setup =
                        AkitaCommitmentScheme::<$cfg>::setup_verifier(&setup)
                            .expect("verifier setup");

                    let pre_poly = akita_prover::OneHotPoly::<$sf, u8>::new(
                        onehot_k,
                        pre_d,
                        pre_indices.clone(),
                    )
                    .expect("pre onehot poly");
                    let (pre_commitment, pre_hint) =
                        AkitaCommitmentScheme::<$cfg>::commit_group(
                            &setup,
                            std::slice::from_ref(&pre_poly),
                            &stack,
                        )
                        .expect("precommit");

                    let multi_schedule = <$cfg as CommitmentConfig>::runtime_schedule(
                        AkitaScheduleLookupKey {
                            final_group: PolynomialGroupLayout::new(final_nv, 1),
                            precommitteds: vec![pre_commitment.profile],
                        },
                    )
                    .expect("multi-group schedule");
                    let final_d = multi_schedule.root.params.final_group.commitment.d_a();
                    let final_chunks = (1usize << final_nv) / onehot_k;
                    let final_indices: Vec<Option<u8>> = (0..final_chunks)
                        .map(|chunk| Some(((chunk * 37 + 11) % onehot_k) as u8))
                        .collect();
                    let final_poly = akita_prover::OneHotPoly::<$sf, u8>::new(
                        onehot_k,
                        final_d,
                        final_indices,
                    )
                    .expect("final onehot poly");
                    let (final_commitment, final_hint, _sel) =
                        AkitaCommitmentScheme::<$cfg>::commit_final_group(
                            &setup,
                            std::slice::from_ref(&final_poly),
                            &stack,
                            vec![pre_commitment.profile],
                        )
                        .expect("final commit");

                    let point: Vec<$se> = (0..final_nv.max(PRE_NV))
                        .map(|i| <$se>::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
                        .collect();
                    let pre_weights =
                        lagrange_weights::<$se>(&point[..PRE_NV]).expect("pre weights");
                    let pre_opening: $se = pre_poly
                        .indices()
                        .iter()
                        .enumerate()
                        .filter_map(|(chunk, hot)| {
                            hot.map(|idx| pre_weights[chunk * onehot_k + usize::from(idx)])
                        })
                        .fold(<$se>::from_u64(0), |a, b| a + b);
                    let final_weights =
                        lagrange_weights::<$se>(&point[..final_nv]).expect("final weights");
                    let final_opening: $se = final_poly
                        .indices()
                        .iter()
                        .enumerate()
                        .filter_map(|(chunk, hot)| {
                            hot.map(|idx| final_weights[chunk * onehot_k + usize::from(idx)])
                        })
                        .fold(<$se>::from_u64(0), |a, b| a + b);

                    let pre_refs = [&pre_poly];
                    let final_refs = [&final_poly];
                    let prover_data = selected_prover_data::<$cfg, _>(
                        OpeningClaims::from_groups(vec![
                            PolynomialGroupClaims::new(
                                point[..PRE_NV].to_vec(),
                                vec![pre_opening],
                                pre_commitment.clone(),
                            )
                            .expect("pre prover group"),
                            PolynomialGroupClaims::new(
                                point[..final_nv].to_vec(),
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

                    let mut pt = AkitaTranscript::<$sf>::new(label);
                    let proof = AkitaCommitmentScheme::<$cfg>::batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut pt,
                        BasisMode::Lagrange,
                    )
                    .expect("prove");

                    let shape = proof.shape();
                    let mut bytes = Vec::new();
                    proof.serialize_uncompressed(&mut bytes).expect("serialize");
                    let decoded = AkitaBatchedProof::<$sf, $se>::deserialize_uncompressed(
                        &bytes[..],
                        &shape,
                    )
                    .expect("deserialize");

                    let verify_claims = OpeningClaims::from_groups(vec![
                        PolynomialGroupClaims::new(
                            point[..PRE_NV].to_vec(),
                            vec![pre_opening],
                            &pre_commitment,
                        )
                        .expect("pre verifier group"),
                        PolynomialGroupClaims::new(
                            point[..final_nv].to_vec(),
                            vec![final_opening],
                            &final_commitment,
                        )
                        .expect("final verifier group"),
                    ])
                    .expect("verifier claims");
                    let mut vt = AkitaTranscript::<$sf>::new(label);
                    AkitaCommitmentScheme::<$cfg>::batched_verify(
                        &decoded,
                        &verifier_setup,
                        &mut vt,
                        GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                        BasisMode::Lagrange,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "{} pre_nv={PRE_NV} final_nv={final_nv}: {e:?}",
                            stringify!($name)
                        )
                    });
                }
            });
        }
    };
}

// ============================================================================
// GROUP A — Small fields (fp32, fp64)
//
// Cartesian product: field × {Dense, OneHot} × {direct, precommitted}
// Opening computed via Lagrange weights over the extension field.
// ============================================================================

// ----------------------------------------------------------------------------
// fp32  (Field = Prime32Offset99, ExtField = FpExt4)
// ----------------------------------------------------------------------------

// fp32 × Dense × direct
small_field_test!(dense;     fp32_dense;     fp32::Dense;  fp32::Field; fp32::ExtensionField; nvs=[12, 14]);
// fp32 × Dense × precommitted   (pre nv=14, final nv from list)
small_field_test!(dense_pre; fp32_dense_pre; fp32::Dense;  fp32::Field; fp32::ExtensionField; final_nvs=[16]);
// fp32 × OneHot × direct
small_field_test!(onehot;     fp32_onehot;     fp32::OneHot; fp32::Field; fp32::ExtensionField; nvs=[12, 16]; k=256);
// fp32 × OneHot × precommitted
small_field_test!(onehot_pre; fp32_onehot_pre; fp32::OneHot; fp32::Field; fp32::ExtensionField; final_nvs=[16]; k=256);

// ----------------------------------------------------------------------------
// fp64  (Field = Prime64Offset59, ExtField = Ext2)
// Also covered by schedules-default; no extra feature gate needed.
// ----------------------------------------------------------------------------

// fp64 × Dense × direct
small_field_test!(dense;     fp64_dense;     fp64::Dense;  fp64::Field; fp64::ExtensionField; nvs=[12, 14]);
// fp64 × Dense × precommitted
small_field_test!(dense_pre; fp64_dense_pre; fp64::Dense;  fp64::Field; fp64::ExtensionField; final_nvs=[16]);
// fp64 × OneHot × direct
small_field_test!(onehot;     fp64_onehot;     fp64::OneHot; fp64::Field; fp64::ExtensionField; nvs=[12, 16]; k=256);
// fp64 × OneHot × precommitted
small_field_test!(onehot_pre; fp64_onehot_pre; fp64::OneHot; fp64::Field; fp64::ExtensionField; final_nvs=[16]; k=256);

// ============================================================================
// GROUP E (small-field) — fp32 multi-group
//
// fp32 one-hot: two separate commitment groups (precommit + final) proved jointly.
// ============================================================================

// fp32 one-hot: two separate commitment groups (precommit + final) proved jointly.
#[test]
fn fp32_onehot_multi_group() {
    type SmallCfg = fp32::OneHot;
    type SmallF = fp32::Field;
    type SmallE = fp32::ExtensionField;
    type SmallScheme = AkitaCommitmentScheme<SmallCfg>;
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;

    init_rayon_pool();
    run_on_large_stack(|| {
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
        let pre_prepared = CpuBackend::DEFAULT
            .prepare_setup(&pre_setup)
            .expect("prepared");
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

        let mut pre_point = (0..PRE_NV)
            .map(|i| SmallE::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
            .collect::<Vec<_>>();
        pre_point[0] += SmallE::one();
        let final_point = (0..FINAL_NV)
            .map(|i| SmallE::from_u64((i as u64).wrapping_mul(5).wrapping_add(2)))
            .collect::<Vec<_>>();
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
