//! fp128 correctness-matrix drivers.
//!
//! Split out of `common/` so only the matrix target compiles them. Nine
//! integration targets declare `mod common`, but only `akita_fp128_e2e` calls
//! these setup/commit/prove/serialize/verify drivers, so keeping them here
//! stops the other eight from compiling ~450 lines they never use.
//!
//! The independent opening oracles stay in `common`: the shared
//! `recursive_multi_group_round_trip` driver needs them too.

use crate::common::*;
use akita_config::{
    recursive_commitment::RecursiveScheduleConfig, CommitmentConfig, RecursiveCommitmentConfig,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend, DensePoly, OneHotPoly};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims, PolynomialGroupLayout,
};

/// Single-group recursive roundtrip: one two-polynomial final group at `nv=32`, no
/// user precommitted groups. Uses `RecursiveCommitmentConfig<BaseCfg>` so the proof
/// carries a stage-3 recursive setup-sumcheck, offloading the setup contribution.
///
/// Covers schedule resolution, setup-prefix precomputation, prove, a serialization
/// round-trip, and an honest verify. Rejection coverage for the recursive path lives
/// in `recursive_multi_group_round_trip` and `protocol_soundness.rs`.
// Only called from the expensive recursive matrix cells, which are normally
// filtered out of focused test runs.
#[allow(dead_code)]
pub(super) fn prove_verify_recursive_direct_roundtrip<BaseCfg>(transcript_domain: &'static [u8])
where
    BaseCfg: CommitmentConfig<Field = F, ExtField = F> + RecursiveScheduleConfig,
{
    type Recursive<BaseCfg> = AkitaCommitmentScheme<RecursiveCommitmentConfig<BaseCfg>>;

    const FINAL_NV: usize = 32;
    const FINAL_GROUP_SIZE: usize = 2;

    init_rayon_pool();
    run_on_large_stack(move || {
        let scheme = load_workspace_scheme::<RecursiveCommitmentConfig<BaseCfg>>()
            .expect("workspace schedule artifact");
        let schedule_key =
            AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(FINAL_NV, FINAL_GROUP_SIZE));
        let opening_layout = schedule_key.opening_layout().expect("opening layout");
        let schedule = scheme
            .schedules()
            .resolve_key(&schedule_key)
            .expect("recursive direct schedule")
            .schedule()
            .clone();
        assert!(
            schedule_uses_setup_prefix(&schedule),
            "recursive schedule must carry setup-prefix metadata"
        );

        let setup = scheme
            .setup_prover(FINAL_NV, FINAL_GROUP_SIZE)
            .expect("recursive direct setup");
        assert!(
            !setup.prefix_slots.is_empty(),
            "recursive setup must precompute prefix slots"
        );
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");

        let final_polys: Vec<OneHotPoly<F, u8>> = (0..FINAL_GROUP_SIZE)
            .map(|i| make_onehot_poly::<BaseCfg>(FINAL_NV, 0x0bee_fcaf_2027_0000 + i as u64))
            .collect();
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit::<_, _>(
                &setup,
                &final_polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("recursive direct commit");

        let point = random_point(FINAL_NV, 0xcafe_2027_0001);
        // Independent oracle: sums of Lagrange weights at the hot indices.
        let openings: Vec<F> = final_polys
            .iter()
            .map(|poly| onehot_opening_lagrange(poly, &point))
            .collect();

        let poly_refs: Vec<&OneHotPoly<F, u8>> = final_polys.iter().collect();
        let prover_data = selected_prover_data::<RecursiveCommitmentConfig<BaseCfg>, _>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                openings.clone(),
                commitment.clone(),
            )
            .expect("prover group")])
            .expect("prover claims"),
            vec![hint],
            vec![&poly_refs[..]],
            scheme.schedules(),
        );
        let selection = prover_data.selection();

        let mut prover_transcript = AkitaTranscript::<F>::new(transcript_domain);
        let proof = scheme
            .batched_prove(
                &setup,
                prover_data,
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("recursive direct prove");
        assert!(
            proof_has_recursive_setup_sumcheck(&proof),
            "recursive proof must carry stage-3 setup sumcheck evidence"
        );

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let verifier_setup = scheme
            .setup_verifier_for_schedule(&setup, &schedule, &opening_layout)
            .expect("verifier setup");
        let verify_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point,
            openings,
            &commitment,
        )
        .expect("verifier group")])
        .expect("verifier claims");
        let mut verifier_transcript = AkitaTranscript::<F>::new(transcript_domain);
        scheme
            .batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                BasisMode::Lagrange,
            )
            .expect("recursive direct verify");
    });
}

pub(super) fn prove_verify_dense_roundtrip<Cfg>(nv_values: &[usize], label: &[u8])
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
{
    prove_verify_dense_roundtrip_with_evals::<Cfg>(nv_values, label, dense_field_evals);
}

/// Dense prove/verify round trip over a caller-supplied evaluation source.
pub(super) fn prove_verify_dense_roundtrip_with_evals<Cfg>(
    nv_values: &[usize],
    label: &[u8],
    evals_for: impl Fn(usize, u64) -> Vec<F>,
) where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
{
    let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    for &nv in nv_values {
        let seed = 0x7e57_0000_u64 ^ nv as u64;
        let evals = evals_for(nv, seed);
        let poly = DensePoly::<F>::from_field_evals(nv, &evals).expect("dense poly");
        let pt = random_point(nv, seed ^ 0xcafe_0000);
        // Independent oracle: raw evaluations folded against the point.
        let expected_opening = dense_opening_lagrange(&evals, &pt);

        let setup = scheme.setup_prover(nv, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
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
            .unwrap();
        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(label);
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<Cfg, _>(
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
        let mut verifier_transcript = AkitaTranscript::<F>::new(label);
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<Cfg>(&pt[..], &openings[..], &commitment, scheme.schedules()),
                BasisMode::Lagrange,
            )
            .unwrap_or_else(|e| panic!("verify dense nv={nv}: {e:?}"));
    }
}

pub(super) fn prove_verify_onehot_roundtrip<Cfg>(nv_values: &[usize], label: &[u8])
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
{
    let k = akita_config::unit_onehot_source_chunk_size::<Cfg>()
        .expect("one-hot roundtrip requires a unit-one-hot config");
    let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    for &nv in nv_values {
        let seed = 0x0bee_0000_u64 ^ nv as u64;
        let poly = make_onehot_poly_with_k(nv, k, seed);
        let pt = random_point(nv, seed ^ 0xcafe_0000);
        // Independent oracle: sum of Lagrange weights at the hot indices.
        let expected_opening = onehot_opening_lagrange(&poly, &pt);

        let setup = scheme.setup_prover(nv, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
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
            .unwrap();
        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(label);
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<Cfg, _>(
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
        let mut verifier_transcript = AkitaTranscript::<F>::new(label);
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<Cfg>(&pt[..], &openings[..], &commitment, scheme.schedules()),
                BasisMode::Lagrange,
            )
            .unwrap_or_else(|e| panic!("verify onehot nv={nv}: {e:?}"));
    }
}

// Pre-commit nv used by both precommitted drivers. Must exist in the precommit catalog.
const PRE_NV: usize = 14;

pub(super) fn prove_verify_dense_precommitted_roundtrip<Cfg>(final_nvs: &[usize], label: &[u8])
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
{
    let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    for &final_nv in final_nvs {
        let setup = scheme.setup_prover(final_nv.max(PRE_NV), 2).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

        let pre_seed = 0xd0d0_0000_u64 ^ PRE_NV as u64;
        let pre_evals = dense_field_evals(PRE_NV, pre_seed);
        let pre_poly =
            DensePoly::<F>::from_field_evals(PRE_NV, &pre_evals).expect("pre dense poly");
        let akita_prover::CommitOutput {
            committed_group: pre_commitment,
            hint: pre_hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&pre_poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("precommit");

        let final_seed = 0xd1d1_0000_u64 ^ final_nv as u64;
        let final_evals = dense_field_evals(final_nv, final_seed);
        let final_poly =
            DensePoly::<F>::from_field_evals(final_nv, &final_evals).expect("final dense poly");
        let precommitteds = PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile])
            .expect("nonempty precommitted groups");
        let akita_prover::CommitOutput {
            committed_group: final_commitment,
            hint: final_hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&final_poly),
                &stack,
                akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
            )
            .expect("final commit");

        let schedule_key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(final_nv, 1),
            precommitteds: vec![pre_commitment.profile],
        };
        // The openings come from independent oracles, so the schedule is not
        // needed to project them. Keep the lookup as a structural check that
        // the combined key resolves to a real catalog row.
        let schedule = scheme
            .schedules()
            .resolve_key(&schedule_key)
            .expect("schedule")
            .schedule()
            .clone();
        assert_eq!(
            schedule.root.params.precommitted_groups().len(),
            1,
            "dense precommitted key must resolve to a one-precommit entry"
        );
        let point = random_point(final_nv.max(PRE_NV), 0xcafe_0000_u64 ^ final_nv as u64);

        // Independent oracles: raw evaluations folded against the point.
        let pre_opening = dense_opening_lagrange(&pre_evals, &point[..PRE_NV]);
        let final_opening = dense_opening_lagrange(&final_evals, &point[..final_nv]);

        let prover_groups = vec![
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
        ];
        let pre_refs = [&pre_poly];
        let final_refs = [&final_poly];
        let prover_data = selected_prover_data::<Cfg, _>(
            OpeningClaims::from_groups(prover_groups).expect("prover claims"),
            vec![pre_hint, final_hint],
            vec![&pre_refs[..], &final_refs[..]],
            scheme.schedules(),
        );
        let selection = prover_data.selection();

        let mut prover_transcript = AkitaTranscript::<F>::new(label);
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prover_data,
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

        let verifier_groups = vec![
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
        ];
        let verify_claims = OpeningClaims::from_groups(verifier_groups).expect("verifier claims");
        let mut verifier_transcript = AkitaTranscript::<F>::new(label);
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                BasisMode::Lagrange,
            )
            .unwrap_or_else(|e| {
                panic!("dense precommitted pre_nv={PRE_NV} final_nv={final_nv}: {e:?}")
            });
    }
}

pub(super) fn prove_verify_onehot_precommitted_roundtrip<Cfg>(final_nvs: &[usize], label: &[u8])
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
{
    let k = akita_config::unit_onehot_source_chunk_size::<Cfg>()
        .expect("one-hot roundtrip requires a unit-one-hot config");
    let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    for &final_nv in final_nvs {
        let setup = scheme.setup_prover(final_nv.max(PRE_NV), 2).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

        let pre_poly = make_onehot_poly_with_k(PRE_NV, k, 0x0bee_f000_u64 ^ PRE_NV as u64);
        let akita_prover::CommitOutput {
            committed_group: pre_commitment,
            hint: pre_hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&pre_poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("precommit");

        let final_poly = make_onehot_poly_with_k(final_nv, k, 0x0bee_f001_u64 ^ final_nv as u64);
        let precommitteds = PrecommittedGroupProfiles::from_profiles(vec![pre_commitment.profile])
            .expect("nonempty precommitted groups");
        let akita_prover::CommitOutput {
            committed_group: final_commitment,
            hint: final_hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&final_poly),
                &stack,
                akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
            )
            .expect("final commit");

        let point = random_point(final_nv.max(PRE_NV), 0xcafe_babe_u64 ^ final_nv as u64);
        // Independent oracles: sums of Lagrange weights at the hot indices.
        let pre_opening = onehot_opening_lagrange(&pre_poly, &point[..PRE_NV]);
        let final_opening = onehot_opening_lagrange(&final_poly, &point[..final_nv]);

        let prover_groups = vec![
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
        ];
        let pre_refs = [&pre_poly];
        let final_refs = [&final_poly];
        let prover_data = selected_prover_data::<Cfg, _>(
            OpeningClaims::from_groups(prover_groups).expect("prover claims"),
            vec![pre_hint, final_hint],
            vec![&pre_refs[..], &final_refs[..]],
            scheme.schedules(),
        );
        let selection = prover_data.selection();

        let mut prover_transcript = AkitaTranscript::<F>::new(label);
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prover_data,
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

        let verifier_groups = vec![
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
        ];
        let verify_claims = OpeningClaims::from_groups(verifier_groups).expect("verifier claims");
        let mut verifier_transcript = AkitaTranscript::<F>::new(label);
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                GroupBatchStatement::new(selection, verify_claims).expect("statement"),
                BasisMode::Lagrange,
            )
            .unwrap_or_else(|e| {
                panic!("onehot precommitted pre_nv={PRE_NV} final_nv={final_nv}: {e:?}")
            });
    }
}
