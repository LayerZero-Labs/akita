use std::panic::{catch_unwind, AssertUnwindSafe};

use super::*;
use crate::test_support::cross_mode_catalogs;

fn statement<'a>(
    selection: akita_types::OpeningScheduleSelection,
    point: &[F],
    opening: F,
    commitment: &'a CommittedGroup<F>,
) -> GroupBatchStatement<'a, F, F> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        vec![opening],
        commitment,
    )
    .expect("cross-mode verifier group")])
    .expect("cross-mode verifier claims");
    GroupBatchStatement::new(selection, claims).expect("cross-mode statement")
}

#[test]
fn proofs_cannot_replay_across_valid_quotient_and_reduced_schedules() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(|| {
            const NUM_VARS: usize = 14;
            const LABEL: &[u8] = b"test/cross-relation-mode";

            let key = akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::new(NUM_VARS, 1),
            );
            let catalogs = cross_mode_catalogs::<Cfg>(&key).expect("valid cross-mode catalogs");
            let quotient_scheme = Scheme::new(catalogs.quotient);
            let reduced_scheme = Scheme::new(catalogs.reduced);
            let quotient_row = quotient_scheme
                .schedules()
                .resolve_selection(catalogs.quotient_selection)
                .expect("valid quotient-only row");
            let reduced_row = reduced_scheme
                .schedules()
                .resolve_selection(catalogs.reduced_selection)
                .expect("valid reduced row");
            assert_eq!(quotient_row.profiles(), reduced_row.profiles());
            assert!(quotient_row
                .schedule()
                .recursive_folds
                .iter()
                .all(|fold| !fold.params.ring_relation_mode.is_reduced_evaluation()));
            assert!(reduced_row
                .schedule()
                .recursive_folds
                .iter()
                .any(|fold| fold.params.ring_relation_mode.is_reduced_evaluation()));

            let root = quotient_row.schedule().root.params.clone();
            let full_num_vars = root.position_index_bits()
                + root.block_index_bits()
                + root.d_a().trailing_zeros() as usize;
            let (poly, evals) = make_dense_poly(full_num_vars);
            let quotient_capacity = akita_config::SetupRequirements::from_catalog::<Cfg>(
                quotient_scheme.schedules(),
                full_num_vars,
                1,
            )
            .map(|requirements| requirements.matrix_capacity)
            .expect("quotient setup capacity");
            let reduced_capacity = akita_config::SetupRequirements::from_catalog::<Cfg>(
                reduced_scheme.schedules(),
                full_num_vars,
                1,
            )
            .map(|requirements| requirements.matrix_capacity)
            .expect("reduced setup capacity");
            let setup =
                if quotient_capacity.num_field_elements >= reduced_capacity.num_field_elements {
                    quotient_scheme.setup_prover(full_num_vars, 1)
                } else {
                    reduced_scheme.setup_prover(full_num_vars, 1)
                }
                .expect("cross-mode setup");
            let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
            let verifier_setup = quotient_scheme
                .setup_verifier(&setup)
                .expect("cross-mode verifier setup");
            let akita_cpu_backend::CommitOutput {
                committed_group: commitment,
                private_handle: hint,
            } = stack
                .commit(
                    quotient_scheme.schedules(),
                    &stack.import_source(vec![poly.clone()]).expect("source"),
                    akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
                )
                .expect("cross-mode commitment");
            assert_eq!(commitment.profile(), &quotient_row.profiles().final_group);

            let point: Vec<F> = (0..full_num_vars)
                .map(|i| F::from_u64((i + 2) as u64))
                .collect();
            let weights = lagrange_weights(&point).expect("cross-mode Lagrange weights");
            let opening = evals
                .iter()
                .zip(weights)
                .fold(F::zero(), |sum, (&coefficient, weight)| {
                    sum + coefficient * weight
                });

            let prove = |scheme: &Scheme, backend: &CpuBackend<F, F>, handle| {
                let group =
                    PolynomialGroupClaims::new(point.clone(), vec![opening], commitment.clone())
                        .expect("cross-mode prover group");
                let claims =
                    OpeningClaims::from_groups(vec![group]).expect("cross-mode prover claims");
                scheme
                    .batched_prove(
                        &setup,
                        selected_prover_data(scheme, claims, vec![handle])
                            .expect("cross-mode prover data"),
                        backend,
                        LABEL,
                        BasisMode::Lagrange,
                    )
                    .expect("cross-mode proof")
            };
            let quotient_proof = prove(&quotient_scheme, &stack, hint.clone());
            // The backend is family-agnostic, so one commitment handle opens under
            // either catalog without a transfer.
            let reduced_proof = prove(&reduced_scheme, &stack, hint);

            for (scheme, proof, selection, name) in [
                (
                    &quotient_scheme,
                    &quotient_proof,
                    catalogs.quotient_selection,
                    "quotient",
                ),
                (
                    &reduced_scheme,
                    &reduced_proof,
                    catalogs.reduced_selection,
                    "reduced",
                ),
            ] {
                scheme
                    .batched_verify(
                        proof,
                        &verifier_setup,
                        LABEL,
                        statement(selection, &point, opening, &commitment),
                        BasisMode::Lagrange,
                    )
                    .unwrap_or_else(|error| panic!("honest {name} proof must verify: {error:?}"));
            }

            for (scheme, proof, wrong_selection, name) in [
                (
                    &reduced_scheme,
                    &quotient_proof,
                    catalogs.reduced_selection,
                    "quotient-as-reduced",
                ),
                (
                    &quotient_scheme,
                    &reduced_proof,
                    catalogs.quotient_selection,
                    "reduced-as-quotient",
                ),
            ] {
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    scheme.batched_verify(
                        proof,
                        &verifier_setup,
                        LABEL,
                        statement(wrong_selection, &point, opening, &commitment),
                        BasisMode::Lagrange,
                    )
                }));
                assert!(
                    matches!(outcome, Ok(Err(_))),
                    "{name} replay must reject without panicking"
                );
            }
        })
        .expect("cross-mode test thread")
        .join()
        .expect("cross-mode test thread panicked");
}
