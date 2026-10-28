//! Byte contract for an ordinary grouped opening with a nonterminal root.
//!
//! The fixture is intentionally independent of extension feature selection.
//! Run it with the default-off extension features enabled as well as disabled.
#![cfg(not(feature = "dev-protocol"))]

mod common;

use akita_cpu_backend::{CpuBackend, GroupContext};
use akita_serialization::AkitaSerialize;
use akita_transcript::{new_prover_channel, new_verifier_channel};
use common::*;

#[test]
fn ordinary_grouped_root_proof_bytes_are_pinned() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const PRE_NV: usize = 14;
        const FINAL_NV: usize = 16;
        let scheme = load_workspace_scheme::<OneHotCfg>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap();
        let setup = scheme.setup_prover(FINAL_NV, 2).unwrap();
        let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
        let polys = [
            make_onehot_poly::<OneHotCfg>(PRE_NV, 17),
            make_onehot_poly::<OneHotCfg>(FINAL_NV, 23),
        ];
        let first = backend
            .commit(
                scheme.schedules(),
                &backend.import_source(vec![polys[0].clone()]).unwrap(),
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let precommitteds =
            PrecommittedGroupProfiles::from_profiles(vec![*first.committed_group.profile()])
                .unwrap();
        let second = backend
            .commit(
                scheme.schedules(),
                &backend.import_source(vec![polys[1].clone()]).unwrap(),
                GroupContext::scheduler_with_precommitted_groups(&precommitteds),
            )
            .unwrap();
        let commitments = [first.committed_group, second.committed_group];
        let handles = [first.private_handle, second.private_handle];
        let points = [random_point(PRE_NV, 31), random_point(FINAL_NV, 37)];
        let profiles = akita_params::CommittedGroupBatchProfile::from_profiles(
            commitments.iter().map(|group| *group.profile()).collect(),
        )
        .unwrap();
        let row = scheme.schedules().resolve_profiles(&profiles).unwrap();
        assert!(
            !row.schedule().recursive_folds.is_empty(),
            "the pinned grouped root must produce a successor witness"
        );
        let opening_layout = profiles.opening_layout().unwrap();
        let values = std::array::from_fn::<_, 2, _>(|index| {
            opening_from_poly_for_layout(
                &polys[index],
                &points[index],
                &row.schedule()
                    .root
                    .params
                    .group_params(&opening_layout, index)
                    .unwrap(),
                BasisMode::Lagrange,
            )
        });
        let prover_claims = OpeningClaims::from_groups(
            (0..2)
                .map(|index| {
                    PolynomialGroupClaims::new(
                        points[index].clone(),
                        vec![values[index]],
                        commitments[index].clone(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let prover_data =
            selected_prover_data::<OneHotCfg>(prover_claims, handles.to_vec(), scheme.schedules());
        let verifier_claims = OpeningClaims::from_groups(
            (0..2)
                .map(|index| {
                    PolynomialGroupClaims::new(
                        points[index].clone(),
                        vec![values[index]],
                        &commitments[index],
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let statement = selected_statement::<OneHotCfg>(verifier_claims, scheme.schedules());
        let mut prover = new_prover_channel(b"ordinary-grouped-root-v1", b"caller").unwrap();
        prover.public_message(b"before-root");
        scheme
            .batched_prove_in_session(
                &setup,
                prover_data,
                &backend,
                &mut prover,
                BasisMode::Lagrange,
            )
            .unwrap();
        let proof = prover.narg_string().to_vec();
        prover.public_message(b"after-root");
        let continuation: [u8; 32] = prover.verifier_message();
        let verifier = scheme
            .verifier(scheme.setup_verifier(&setup).unwrap())
            .unwrap();
        let mut state =
            new_verifier_channel(b"ordinary-grouped-root-v1", b"caller", &proof).unwrap();
        state.public_message(b"before-root");
        verifier
            .batched_verify_in_session(&mut state, statement, BasisMode::Lagrange)
            .unwrap();
        state.public_message(b"after-root");
        assert_eq!(state.verifier_message::<[u8; 32]>().unwrap(), continuation);
        state.check_eof().unwrap();

        let mut commitment_bytes = Vec::new();
        for commitment in &commitments {
            commitment
                .serialize_compressed(&mut commitment_bytes)
                .unwrap();
        }
        let mut setup_bytes = Vec::new();
        setup
            .expanded
            .descriptor()
            .serialize_compressed(&mut setup_bytes)
            .unwrap();
        for coefficient in setup.expanded.shared_matrix().as_field_slice() {
            setup_bytes.extend_from_slice(&coefficient.to_bytes_le_vec());
        }
        let observed = (
            proof.len(),
            protocol_epoch_digest::<F>(&proof),
            protocol_epoch_digest::<F>(&commitment_bytes),
            protocol_epoch_digest::<F>(&setup_bytes),
            continuation,
        );
        assert_eq!(
            observed,
            (
                70_541,
                "8956d163098c1c2198c971b9850ae2ce".to_owned(),
                "4362adef133f80d1bc62cf427b6725cc".to_owned(),
                "5a93d1bf3a49f1f2f46756199b674f50".to_owned(),
                [
                    255, 9, 227, 159, 50, 52, 229, 201, 73, 146, 121, 17, 78, 55, 245, 228, 118,
                    255, 4, 131, 136, 111, 227, 163, 35, 212, 173, 128, 2, 50, 252, 85,
                ],
            ),
        );
    });
}
