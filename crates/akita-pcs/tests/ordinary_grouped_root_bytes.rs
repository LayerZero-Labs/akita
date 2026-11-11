//! Byte contract for an ordinary grouped opening with a nonterminal root.
//!
//! The fixture is intentionally independent of extension feature selection.
//! Run it with the default-off extension features enabled as well as disabled.
#![cfg(not(feature = "dev-protocol"))]

mod common;

use akita_cpu_backend::{CpuBackend, GroupContext};
use akita_serialization::AkitaSerialize;
use akita_transcript::{new_prover_channel, new_verifier_channel};
use akita_verifier::SetupPolicy;
use common::*;
use jolt_transcript::BytesMsg;

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
        let row = scheme.schedules().resolve_profiles(&profiles, 0).unwrap();
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
        let mut prover = new_prover_channel(b"ordinary-grouped-root-v1");
        prover.public_message(&BytesMsg(b"before-root"));
        scheme
            .batched_prove_embedded(
                &setup,
                prover_data,
                &backend,
                &mut prover,
                BasisMode::Lagrange,
            )
            .unwrap();
        let proof = prover.narg_string().to_vec();
        prover.public_message(&BytesMsg(b"after-root"));
        let continuation: [u8; 32] = prover.verifier_message();
        let verifier = scheme
            .verifier(scheme.setup_verifier(&setup).unwrap())
            .unwrap();
        let mut state = new_verifier_channel(b"ordinary-grouped-root-v1", &proof);
        state.public_message(&BytesMsg(b"before-root"));
        let pending = verifier
            .batched_verify_embedded(
                &mut state,
                statement,
                BasisMode::Lagrange,
                SetupPolicy::Check(None),
            )
            .unwrap();
        state.public_message(&BytesMsg(b"after-root"));
        assert_eq!(state.verifier_message::<[u8; 32]>(), continuation);
        pending.check_eof(state).unwrap();

        let mut commitment_bytes = Vec::new();
        for commitment in &commitments {
            commitment
                .serialize_compressed(&mut commitment_bytes)
                .unwrap();
        }
        let mut setup_bytes = Vec::new();
        let required = akita_config::SetupRequirements::from_catalog::<OneHotCfg>(
            scheme.schedules(),
            FINAL_NV,
            2,
        )
        .unwrap()
        .matrix_capacity()
        .num_field_elements;
        // Disk persistence may retain a larger cached prefix. Pin the entire
        // catalog-required prefix, excluding that unrelated surplus capacity.
        let mut descriptor = setup.expanded.descriptor().clone();
        descriptor.num_field_elements = required;
        descriptor.serialize_compressed(&mut setup_bytes).unwrap();
        for coefficient in setup
            .expanded
            .shared_matrix()
            .as_field_slice()
            .get(..required)
            .unwrap()
        {
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
                65_526,
                "6056be85889ba2a7bcdfb04430831f8c".to_owned(),
                "bff3296bfded86b137988b7ea187897c".to_owned(),
                "e5db2d330e8986f4b9fac8b33c9c1070".to_owned(),
                [
                    250, 136, 203, 115, 56, 72, 49, 242, 106, 28, 72, 165, 20, 255, 65, 238, 24,
                    210, 154, 123, 220, 103, 84, 203, 246, 98, 178, 229, 164, 88, 75, 6,
                ],
            ),
        );
    });
}
