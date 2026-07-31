//! Recursive mixed ring-dimension transition E2E.
//!
//! Synthetic profile geometry (not planner output):
//! - L0: A/B/D = `256/128/128` with a D128 setup-prefix handoff
//! - L1: A/B/D = `128/64/64`
//! - L2+: uniform D64
//!
//! The fixture is intentionally compact: two precommitted singleton groups at
//! `nv=14` and one final polynomial at `nv=24`. This still crosses the
//! `256/128/128 -> 128/64/64 -> 64` boundary and exercises a real setup-prefix
//! handoff without allocating the former `nv=32`, four-polynomial workload.
//! Coverage is intentionally layered: honest plain and W8R2 prove/verify,
//! serialization round-trip, and a wrong final-opening rejection. Registry,
//! NTT-slot, and malformed-geometry failures stay in focused unit tests.

#![allow(missing_docs)]

mod common;

use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, PrecommittedCommitmentConfig};
use akita_pcs::test_support::{
    materialize_schedule_setup_prefix_slots, RecursiveRingDimensionTransitionConfig,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{commit_setup_prefix, ComputeBackendSetup, CpuBackend, OneHotIndex, OneHotPoly};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{
    active_setup_field_len, dispatch_for_field, lagrange_weights, padded_setup_prefix_len,
    AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode, CommitmentRingDims,
    CommittedGroupProfile, FoldSchedule, GroupBatchStatement, NttCacheKey, OpeningClaims,
    OpeningClaimsLayout, PolynomialGroupClaims, PolynomialGroupLayout, WitnessPartition,
};
use common::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

type Root = fp128::D256OneHot;
type Mid = fp128::D128OneHot;
type Suffix = fp128::D64OneHot;
type PlainCfg = RecursiveRingDimensionTransitionConfig<Root, Mid, Suffix, Suffix, 128, 64>;
type W8R2Cfg =
    RecursiveRingDimensionTransitionConfig<Root, Mid, Suffix, fp128::D64OneHotMultiChunk, 128, 64>;

const PRE_NV: usize = 14;
const FINAL_NV: usize = 24;
const PRE_GROUPS: usize = 2;
const PRE_GROUP_SIZE: usize = 1;
const FINAL_GROUP_SIZE: usize = 1;
const TOTAL_GROUP_SIZE: usize = PRE_GROUPS * PRE_GROUP_SIZE + FINAL_GROUP_SIZE;
const ONEHOT_K: usize = 256;
static RECURSIVE_E2E_LOCK: Mutex<()> = Mutex::new(());

fn make_layout_onehot_poly(
    layout: &akita_types::CommittedGroupParams,
    seed: u64,
) -> OneHotPoly<F, u8> {
    let d = layout.d_a();
    let total_field = layout.num_live_blocks * layout.num_positions_per_block * d;
    assert_eq!(total_field % ONEHOT_K, 0);
    let total_chunks = total_field / ONEHOT_K;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, u8>::new(ONEHOT_K, d, indices).expect("onehot poly")
}

fn onehot_lagrange_opening<I: OneHotIndex>(poly: &OneHotPoly<F, I>, point: &[F]) -> F {
    let onehot_k = poly.onehot_k();
    assert!(onehot_k.is_power_of_two());
    assert_eq!(poly.indices().len() * onehot_k, 1usize << point.len());
    let low_vars = onehot_k.trailing_zeros() as usize;
    let low_weights = lagrange_weights(&point[..low_vars]).expect("low weights");
    let high_weights = lagrange_weights(&point[low_vars..]).expect("high weights");
    poly.indices()
        .iter()
        .enumerate()
        .filter_map(|(chunk_idx, hot_idx)| {
            hot_idx.map(|hot_idx| high_weights[chunk_idx] * low_weights[hot_idx.as_usize()])
        })
        .fold(F::zero(), |acc, weight| acc + weight)
}

fn schedule_uses_setup_prefix(schedule: &FoldSchedule) -> bool {
    schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.incoming_setup_prefix.is_some())
}

fn proof_has_recursive_setup_sumcheck(proof: &AkitaBatchedProof<F, F>) -> bool {
    proof.root.stage3_sumcheck_proof.is_some()
        || proof
            .recursive_folds
            .iter()
            .any(|step| step.stage3_sumcheck_proof.is_some())
}

fn assert_mixed_recursive_geometry(schedule: &FoldSchedule) {
    assert_eq!(
        schedule.root.params.final_group.commitment.role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 128
        },
        "L0 must be 256/128/128"
    );
    assert_eq!(
        schedule.recursive_folds[0].params.witness.role_dims(),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64
        },
        "L1 must be 128/64/64"
    );
    let prefix = schedule.recursive_folds[0]
        .params
        .incoming_setup_prefix
        .as_ref()
        .expect("L1 must carry a setup prefix");
    assert_eq!(prefix.d_setup, 128, "setup prefix must use D128");
    assert_eq!(
        prefix
            .commitment_params
            .layout
            .inner_commit_matrix
            .ring_dimension(),
        128,
        "setup-prefix A dimension must be 128"
    );
    assert_eq!(
        prefix
            .commitment_params
            .layout
            .outer_commit_matrix
            .ring_dimension(),
        64,
        "setup-prefix B dimension must be 64"
    );
}

fn recursive_mixed_d_multi_group_round_trip<ProofCfg>(
    transcript_domain: &'static [u8],
    on_schedule: fn(&FoldSchedule),
) where
    ProofCfg: CommitmentConfig<Field = F, ExtField = F>,
{
    type Scheme<ProofCfg> = AkitaCommitmentScheme<ProofCfg>;
    type Precommitted<ProofCfg> = AkitaCommitmentScheme<PrecommittedCommitmentConfig<ProofCfg>>;

    init_rayon_pool();
    run_on_large_stack(move || {
        let pre_key = PolynomialGroupLayout::new(PRE_NV, PRE_GROUP_SIZE);
        let pre_layout =
            PrecommittedCommitmentConfig::<ProofCfg>::get_params_for_batched_commitment(
                &OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE).expect("precommit batch"),
            )
            .expect("precommit params");
        let pre_frozen = CommittedGroupProfile::from_params(pre_key, &pre_layout);
        let schedule_key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(FINAL_NV, FINAL_GROUP_SIZE),
            precommitteds: vec![pre_frozen, pre_frozen],
        };
        let opening_layout = schedule_key.opening_layout().expect("opening layout");

        let schedule =
            ProofCfg::runtime_schedule(schedule_key).expect("mixed recursive schedule resolves");
        assert!(
            schedule_uses_setup_prefix(&schedule),
            "mixed recursive schedule must carry setup-prefix metadata"
        );
        assert_mixed_recursive_geometry(&schedule);
        on_schedule(&schedule);
        let root_params = &schedule.root.params.final_group.commitment;

        let mut setup =
            Scheme::<ProofCfg>::setup_prover(FINAL_NV, TOTAL_GROUP_SIZE).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        materialize_schedule_setup_prefix_slots(&mut setup, &CpuBackend, &prepared, &schedule)
            .expect("materialize mixed setup-prefix slots");
        let prefix_id = schedule.recursive_folds[0]
            .params
            .incoming_setup_prefix
            .as_ref()
            .expect("L1 setup prefix");
        let natural_len = active_setup_field_len(root_params, &opening_layout)
            .expect("canonical Stage 3 setup projection");
        assert_eq!(prefix_id.natural_len, natural_len);
        assert_eq!(
            prefix_id.n_prefix().expect("planned padded prefix"),
            padded_setup_prefix_len(natural_len)
        );
        let prefix_slot = setup
            .prefix_slots
            .get(prefix_id)
            .expect("exact planned prefix slot");
        assert_eq!(prefix_slot.natural_len, natural_len);
        assert_eq!(prefix_slot.padded_len, padded_setup_prefix_len(natural_len));
        assert!(
            setup.expanded.shared_matrix().as_field_slice().len() >= natural_len,
            "prepared setup must cover the canonical natural prefix"
        );
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");

        let mut pre_polys_by_group = Vec::new();
        let mut pre_commitments = Vec::new();
        let mut pre_hints = Vec::new();
        for group_idx in 0..PRE_GROUPS {
            let poly =
                make_layout_onehot_poly(&pre_layout, 0x0bee_fcaf_3340_0000 + group_idx as u64);
            let (commitment, hint) = Precommitted::<ProofCfg>::batched_commit(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
            )
            .expect("precommit group");
            pre_polys_by_group.push(vec![poly]);
            pre_commitments.push(commitment);
            pre_hints.push(hint);
        }

        let final_polys: Vec<OneHotPoly<F, u8>> = (0..FINAL_GROUP_SIZE)
            .map(|poly_idx| {
                make_layout_onehot_poly(root_params, 0x0bee_fcaf_3340_1000 + poly_idx as u64)
            })
            .collect();
        let (final_commitment, final_hint, _selection) = Scheme::<ProofCfg>::commit_final_group(
            &setup,
            &final_polys,
            &stack,
            pre_commitments.iter().map(|group| group.profile).collect(),
        )
        .expect("final mixed commitment");

        let point = random_point(FINAL_NV, 0xcafe_3340_0001);
        let pre_openings: Vec<Vec<F>> = pre_polys_by_group
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| onehot_lagrange_opening(poly, &point[..PRE_NV]))
                    .collect()
            })
            .collect();
        let final_openings: Vec<F> = final_polys
            .iter()
            .map(|poly| onehot_lagrange_opening(poly, &point))
            .collect();

        let pre_refs_by_group: Vec<Vec<&OneHotPoly<F, u8>>> = pre_polys_by_group
            .iter()
            .map(|polys| polys.iter().collect())
            .collect();
        let final_refs: Vec<&OneHotPoly<F, u8>> = final_polys.iter().collect();

        let mut prover_groups = Vec::new();
        for (group_idx, openings) in pre_openings.iter().enumerate() {
            prover_groups.push(
                PolynomialGroupClaims::new(
                    point[..PRE_NV].to_vec(),
                    openings.clone(),
                    pre_commitments[group_idx].clone(),
                )
                .expect("pre prover group"),
            );
        }
        prover_groups.push(
            PolynomialGroupClaims::new(
                point.clone(),
                final_openings.clone(),
                final_commitment.clone(),
            )
            .expect("final prover group"),
        );

        let mut prover_polys: Vec<&[&OneHotPoly<F, u8>]> = Vec::new();
        for refs in &pre_refs_by_group {
            prover_polys.push(&refs[..]);
        }
        prover_polys.push(&final_refs[..]);
        let mut prover_hints = pre_hints;
        prover_hints.push(final_hint);

        let prover_claims = selected_prover_data::<ProofCfg, _>(
            OpeningClaims::from_groups(prover_groups).expect("prover claims"),
            prover_hints,
            prover_polys,
        );
        let selection = prover_claims.0;

        let mut prover_transcript = AkitaTranscript::<F>::new(transcript_domain);
        let proof = Scheme::<ProofCfg>::batched_prove(
            &setup,
            prover_claims,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("mixed recursive proof");
        assert!(
            proof_has_recursive_setup_sumcheck(&proof),
            "mixed recursive proof must carry stage-3 setup sumcheck evidence"
        );

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof
            .serialize_compressed(&mut bytes)
            .expect("serialize mixed recursive proof");
        let proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize mixed recursive proof");

        let verifier_setup = setup.verifier_setup().expect("verifier setup");
        let verify_claims = |final_openings: Vec<F>| {
            let mut verifier_groups = Vec::new();
            for (group_idx, openings) in pre_openings.iter().enumerate() {
                verifier_groups.push(
                    PolynomialGroupClaims::new(
                        point[..PRE_NV].to_vec(),
                        openings.clone(),
                        &pre_commitments[group_idx],
                    )
                    .expect("pre verifier group"),
                );
            }
            verifier_groups.push(
                PolynomialGroupClaims::new(point.clone(), final_openings, &final_commitment)
                    .expect("final verifier group"),
            );
            let claims = OpeningClaims::from_groups(verifier_groups).expect("verifier claims");
            GroupBatchStatement::new(selection, claims).expect("verifier statement")
        };

        let mut verifier_transcript = AkitaTranscript::<F>::new(transcript_domain);
        Scheme::<ProofCfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_claims(final_openings.clone()),
            BasisMode::Lagrange,
        )
        .expect("mixed recursive verify");
        assert_eq!(
            prover_transcript.challenge_bytes(b"test/transcript-agreement", 32),
            verifier_transcript.challenge_bytes(b"test/transcript-agreement", 32),
            "prover and verifier transcripts must agree after the recursive mixed-D proof"
        );

        let mut tampered = final_openings.clone();
        tampered[0] += F::from_canonical_u128_reduced(1);
        let mut tampered_transcript = AkitaTranscript::<F>::new(transcript_domain);
        let tampered_result = Scheme::<ProofCfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut tampered_transcript,
            verify_claims(tampered),
            BasisMode::Lagrange,
        );
        assert!(
            tampered_result.is_err(),
            "mixed recursive verify must reject a tampered final opening"
        );
    });
}

#[test]
fn recursive_mixed_d_prefix_commit_rejects_missing_outer_ntt_slot() {
    let _serial = RECURSIVE_E2E_LOCK.lock().expect("recursive E2E lock");
    init_rayon_pool();
    run_on_large_stack(|| {
        let pre_key = PolynomialGroupLayout::new(PRE_NV, PRE_GROUP_SIZE);
        let pre_layout =
            PrecommittedCommitmentConfig::<PlainCfg>::get_params_for_batched_commitment(
                &OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE).expect("precommit batch"),
            )
            .expect("precommit params");
        let descriptor = CommittedGroupProfile::from_params(pre_key, &pre_layout);
        let schedule = PlainCfg::runtime_schedule(AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(FINAL_NV, FINAL_GROUP_SIZE),
            precommitteds: vec![descriptor, descriptor],
        })
        .expect("mixed recursive schedule");
        let prefix = schedule.recursive_folds[0]
            .params
            .incoming_setup_prefix
            .as_ref()
            .expect("dynamic setup prefix");

        let setup = AkitaCommitmentScheme::<PlainCfg>::setup_prover(FINAL_NV, TOTAL_GROUP_SIZE)
            .expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let source_ntt =
            NttCacheKey::from_envelope(&setup.expanded, prefix.d_setup).expect("source NTT key");
        CpuBackend
            .ensure_ntt_slot(&prepared, source_ntt)
            .expect("warm only source NTT slot");

        let n_prefix = prefix.n_prefix().expect("padded prefix length");
        let error = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            prefix.d_setup,
            |D_SETUP| {
                commit_setup_prefix::<F, D_SETUP, _>(
                    &setup.expanded,
                    &CpuBackend,
                    &prepared,
                    &prefix.commitment_params,
                    n_prefix,
                    prefix.natural_len,
                )
            }
        )
        .expect_err("missing D64 outer NTT slot must reject");
        assert!(
            error.to_string().contains("NTT slot not warmed"),
            "unexpected missing-slot error: {error}"
        );
    });
}

#[test]
fn recursive_mixed_d_plain_multi_group_honest_round_trip() {
    let _serial = RECURSIVE_E2E_LOCK.lock().expect("recursive E2E lock");
    recursive_mixed_d_multi_group_round_trip::<PlainCfg>(
        b"test/recursive_ring_dimension_transition_e2e/plain",
        |_schedule| {},
    );
}

#[test]
fn recursive_mixed_d_w8r2_multi_group_honest_round_trip() {
    let _serial = RECURSIVE_E2E_LOCK.lock().expect("recursive E2E lock");
    recursive_mixed_d_multi_group_round_trip::<W8R2Cfg>(
        b"test/recursive_ring_dimension_transition_e2e/w8r2",
        |schedule| {
            assert_eq!(
                schedule.recursive_folds[0].params.witness_partition,
                WitnessPartition::Distributed { num_chunks: 8 },
                "W8R2 middle partition must remain distributed with 8 chunks"
            );
        },
    );
}
