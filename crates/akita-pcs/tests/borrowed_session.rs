//! Multi-group PCS composition without constructing a nested transcript.
mod common;

use akita_cpu_backend::{CpuBackend, GroupContext};
use akita_transcript::{new_prover_channel, new_verifier_channel};
use common::*;

#[test]
fn borrowed_multi_group_opening_preserves_the_callers_session() {
    run_on_large_stack(|| {
        // Use the shipped fp128_onehot row final=16:1;precommitted=14:1
        // from artifacts/schedule-catalog.tsv.
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
        let prover_data = || {
            selected_prover_data::<OneHotCfg>(
                OpeningClaims::from_groups(
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
                .unwrap(),
                handles.to_vec(),
                scheme.schedules(),
            )
        };
        let statement = || {
            selected_statement::<OneHotCfg>(
                OpeningClaims::from_groups(
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
                .unwrap(),
                scheme.schedules(),
            )
        };
        let verifier = scheme
            .verifier(scheme.setup_verifier(&setup).unwrap())
            .unwrap();
        let standalone = scheme
            .batched_prove(
                &setup,
                prover_data(),
                &backend,
                b"borrowed-test",
                BasisMode::Lagrange,
            )
            .unwrap();
        verifier
            .batched_verify(
                &standalone,
                b"borrowed-test",
                statement(),
                BasisMode::Lagrange,
            )
            .unwrap();

        let mut prover = new_prover_channel(b"borrowed-test", b"outer").unwrap();
        prover.public_message(b"prefix");
        scheme
            .batched_prove_in_session(
                &setup,
                prover_data(),
                &backend,
                &mut prover,
                BasisMode::Lagrange,
            )
            .unwrap();
        let opening_end = prover.narg_string().len();
        prover.public_message(b"suffix");
        let challenge: [u8; 32] = prover.verifier_message();
        prover.prover_message(&challenge);
        let proof = prover.narg_string().to_vec();
        let mut state = new_verifier_channel(b"borrowed-test", b"outer", &proof).unwrap();
        state.public_message(b"prefix");
        verifier
            .batched_verify_in_session(&mut state, statement(), BasisMode::Lagrange)
            .unwrap();
        // The opening returns before the outer protocol's final proof message.
        state.public_message(b"suffix");
        assert_eq!(state.verifier_message::<[u8; 32]>().unwrap(), challenge);
        assert_eq!(state.prover_message::<[u8; 32]>().unwrap(), challenge);
        state.check_eof().unwrap();

        for (session, domain, prefix) in [
            (
                b"wrong-suite".as_slice(),
                b"outer".as_slice(),
                b"prefix".as_slice(),
            ),
            (b"borrowed-test", b"wrong-domain", b"prefix"),
            (b"borrowed-test", b"outer", b"wrong-prefix"),
        ] {
            let mut state = new_verifier_channel(session, domain, &proof).unwrap();
            state.public_message(prefix);
            assert!(verifier
                .batched_verify_in_session(&mut state, statement(), BasisMode::Lagrange)
                .is_err());
            assert!(state.check_eof().is_err());
        }
        let mut corrupted = proof[..opening_end].to_vec();
        corrupted[opening_end / 2] ^= 1;
        for bad in [&proof[..opening_end - 1], corrupted.as_slice()] {
            let mut state = new_verifier_channel(b"borrowed-test", b"outer", bad).unwrap();
            state.public_message(b"prefix");
            assert!(verifier
                .batched_verify_in_session(&mut state, statement(), BasisMode::Lagrange)
                .is_err());
            assert!(state.verifier_message::<[u8; 32]>().is_err());
        }
        // EOF belongs to the outer session, even when the opening has succeeded.
        let mut trailing = proof[..opening_end].to_vec();
        trailing.push(0);
        let mut state = new_verifier_channel(b"borrowed-test", b"outer", &trailing).unwrap();
        state.public_message(b"prefix");
        verifier
            .batched_verify_in_session(&mut state, statement(), BasisMode::Lagrange)
            .unwrap();
        assert!(state.check_eof().is_err());
    });
}
