//! Independent runtime catalogs must coexist without type-global geometry.

use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};
use akita_cpu_backend::{CpuBackend, GroupContext, OneHotPoly};
use akita_params::{
    lagrange_weights, BasisMode, CommittedGroupBatchProfile, GroupCommitPhaseParams,
    MultiChunkProfileId, PolynomialGroupLayout, ScheduleLookupKey,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_schedules::ValidatedScheduleCatalog;
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::{Ring, Zero};

#[test]
fn independent_runtime_catalogs_prove_verify_and_reject_each_other() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            type F = <OneHot as CommitmentConfig>::Field;
            const NV: usize = 16;
            const DOMAIN: &[u8] = b"independent-runtime-catalogs";
            let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(NV));
            // Both schemes use exactly OneHot, with independently selected widths and depths.
            let contexts = [MultiChunkProfileId::W2R3, MultiChunkProfileId::W4R2].map(|profile| {
                let chunking = profile.cfg();
                let policy = policy_of::<OneHot>(chunking);
                let schedule = akita_planner::find_schedule(
                    &key,
                    OneHot::committed_source_contract().unwrap(),
                    &[],
                    &policy,
                    OneHot::ring_challenge_config,
                )
                .unwrap()
                .schedule;
                let chunked_folds = std::iter::once(&schedule.root)
                    .chain(&schedule.recursive_folds)
                    .filter(|fold| fold.params.witness_chunk.num_chunks == chunking.num_chunks)
                    .count();
                assert_eq!(chunked_folds, chunking.num_activated_levels);
                let descriptor = CommittedGroupBatchProfile {
                    final_group: GroupCommitPhaseParams::try_from_params(
                        key.final_group,
                        &schedule.root.params,
                    )
                    .unwrap(),
                    precommitteds: Vec::new(),
                };
                let catalog = ValidatedScheduleCatalog::try_new(
                    OneHot::schedule_family_name(),
                    [(descriptor, schedule)],
                    &policy,
                    OneHot::ring_challenge_config,
                )
                .unwrap();
                let scheme = AkitaCommitmentScheme::<OneHot>::from_schedule_artifact(
                    &catalog.to_artifact_bytes().unwrap(),
                    chunking,
                )
                .unwrap();
                let setup = scheme.setup_prover(NV, 1).unwrap();
                let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
                (scheme, setup, backend)
            });
            assert_ne!(
                contexts[0].0.schedules().catalog_digest(),
                contexts[1].0.schedules().catalog_digest()
            );

            let chunk_size = akita_config::unit_onehot_source_chunk_size::<OneHot>().unwrap();
            let indices = (0..((1 << NV) / chunk_size))
                .map(|chunk| Some(((chunk * 29 + 7) % chunk_size) as u8))
                .collect();
            let poly = OneHotPoly::<F, u8>::new(chunk_size, indices).unwrap();
            let point: Vec<F> = (0..NV)
                .map(|index| F::from_u64((index + 2) as u64))
                .collect();
            let weights = lagrange_weights(&point).unwrap();
            let evaluation =
                poly.indices()
                    .iter()
                    .enumerate()
                    .fold(F::zero(), |sum, (chunk, hot)| {
                        sum + hot.map_or(F::zero(), |index| {
                            weights[chunk * chunk_size + usize::from(index)]
                        })
                    });
            // Keep both contexts alive while producing both proofs before any verification.
            let proofs = contexts
                .iter()
                .map(|(scheme, setup, backend)| {
                    let output = backend
                        .commit(
                            scheme.schedules(),
                            &backend.import_source(vec![poly.clone()]).unwrap(),
                            GroupContext::scheduler_without_precommitted_groups(),
                        )
                        .unwrap();
                    let commitment = output.committed_group;
                    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                        point.clone(),
                        vec![evaluation],
                        commitment.clone(),
                    )
                    .unwrap()])
                    .unwrap();
                    let opening = SelectedProverOpeningData::from_committed_claims::<OneHot>(
                        claims,
                        vec![output.private_handle],
                        scheme.schedules(),
                    )
                    .unwrap();
                    let selection = opening.selection();
                    let proof = scheme
                        .batched_prove(setup, opening, backend, DOMAIN, BasisMode::Lagrange)
                        .unwrap();
                    (commitment, selection, proof)
                })
                .collect::<Vec<_>>();
            for (verifier_index, (scheme, setup, _)) in contexts.iter().enumerate() {
                let verifier = scheme
                    .verifier(scheme.setup_verifier(setup).unwrap())
                    .unwrap();
                for (proof_index, (commitment, selection, proof)) in proofs.iter().enumerate() {
                    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                        point.clone(),
                        vec![evaluation],
                        commitment,
                    )
                    .unwrap()])
                    .unwrap();
                    let statement = GroupBatchStatement::new(*selection, claims).unwrap();
                    let result = verifier.batched_verify(
                        proof,
                        DOMAIN,
                        statement,
                        BasisMode::Lagrange,
                        #[cfg(feature = "external-setup-accumulation")]
                        None,
                        #[cfg(feature = "external-setup-accumulation")]
                        &mut akita_verifier::SetupMode::Check,
                    );
                    if verifier_index == proof_index {
                        result.expect("the owning catalog verifies its proof");
                    } else {
                        assert!(result.is_err(), "a different catalog must reject the proof");
                    }
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
