use super::*;

#[test]
fn reduced_relation_catalog_roundtrip_reaches_production_verifier() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(|| {
            const NUM_VARS: usize = 16;

            let (scheme, verifier_setup, commitment, mut proof, opening_point, opening, _) =
                make_verify_fixture(NUM_VARS);
            let key = akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::new(NUM_VARS, 1),
            );
            let selection = scheme
                .schedules()
                .resolve_key(&key)
                .expect("shipped reduced-relation schedule");
            let schedule = selection.schedule();
            let first_reduced_index = schedule
                .recursive_folds
                .iter()
                .position(|fold| fold.params.ring_relation_mode.is_reduced_evaluation())
                .expect("fixture reduced cutover");
            let reduced = &schedule.recursive_folds[first_reduced_index..];
            assert!(
                reduced.len() >= 2,
                "fixture must execute more than one reduced recursive fold"
            );
            assert!(reduced
                .iter()
                .any(|fold| fold.params.payload_mode.is_compressed()));
            assert!(reduced.iter().any(|fold| matches!(
                fold.params.payload_mode,
                akita_types::CommitmentPayloadMode::Raw
            )));
            assert!(schedule
                .recursive_folds
                .iter()
                .skip_while(|fold| !fold.params.ring_relation_mode.is_reduced_evaluation())
                .all(|fold| fold.params.ring_relation_mode.is_reduced_evaluation()));

            let commitments = [commitment];
            let openings = [opening];
            scheme
                .verifier(verifier_setup.clone())
                .and_then(|verifier| {
                    verifier.batched_verify(
                        &proof,
                        b"test/prove",
                        verifier_claims(&scheme, &opening_point, &openings, &commitments[0]),
                        BasisMode::Lagrange,
                    )
                })
                .expect("production verifier must replay the reduced-relation suffix");

            let mutation = proof.len() * 3 / 4;
            proof[mutation] ^= 1;
            scheme
                .verifier(verifier_setup.clone())
                .and_then(|verifier| {
                    verifier.batched_verify(
                        &proof,
                        b"test/prove",
                        verifier_claims(&scheme, &opening_point, &openings, &commitments[0]),
                        BasisMode::Lagrange,
                    )
                })
                .expect_err("production verifier must reject a tampered reduced stage2 proof");
        })
        .expect("reduced-relation test thread")
        .join()
        .expect("reduced-relation test thread panicked");
}

#[test]
fn verify_rejects_wrong_opening() {
    let scheme = workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout(&scheme, 16);
    let num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = scheme.setup_prover(num_vars, 1).unwrap();
    let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

    let akita_cpu_backend::CommitOutput {
        committed_group: commitment,
        private_handle: hint,
    } = stack
        .commit(
            scheme.schedules(),
            &stack.import_source(vec![poly.clone()]).expect("source"),
            akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let commitments = [commitment];

    let proof = scheme
        .batched_prove(
            &setup,
            prover_claims(
                &scheme,
                &opening_point[..],
                &[opening],
                &commitments[0],
                hint,
            ),
            &stack,
            b"test/prove",
            BasisMode::Lagrange,
        )
        .unwrap();

    let wrong_opening = opening + F::one();
    let wrong_openings = [wrong_opening];
    let result = scheme
        .verifier(verifier_setup.clone())
        .and_then(|verifier| {
            verifier.batched_verify(
                &proof,
                b"test/prove",
                verifier_claims(
                    &scheme,
                    &opening_point[..],
                    &wrong_openings[..],
                    &commitments[0],
                ),
                BasisMode::Lagrange,
            )
        });

    assert!(
        result.is_err(),
        "verify must reject an incorrect opening value"
    );
}

#[test]
fn native_spongefish_roundtrip_and_statement_binding() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(native_spongefish_roundtrip_and_statement_binding_inner)
        .expect("native test thread")
        .join()
        .expect("native test thread panicked");
}

fn native_spongefish_roundtrip_and_statement_binding_inner() {
    let scheme = workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    let layout = singleton_layout(&scheme, 16);
    let num_vars =
        layout.position_index_bits() + layout.block_index_bits() + D.trailing_zeros() as usize;
    let (poly, evals) = make_dense_poly(num_vars);
    let setup = scheme.setup_prover(num_vars, 1).unwrap();
    let stack = CpuBackend::new(setup.expanded.clone()).unwrap();
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    let akita_cpu_backend::CommitOutput {
        committed_group: commitment,
        private_handle: prover_state,
    } = stack
        .commit(
            scheme.schedules(),
            &stack.import_source(vec![poly]).unwrap(),
            akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let opening_point = (0..num_vars)
        .map(|index| F::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let weights = lagrange_weights(&opening_point).unwrap();
    let opening = evals
        .iter()
        .zip(&weights)
        .fold(F::zero(), |sum, (&value, &weight)| sum + value * weight);
    let proof = scheme
        .batched_prove(
            &setup,
            prover_claims(
                &scheme,
                &opening_point,
                &[opening],
                &commitment,
                prover_state,
            ),
            &stack,
            b"test/prove",
            BasisMode::Lagrange,
        )
        .expect("native proof");
    scheme
        .verifier(verifier_setup.clone())
        .and_then(|verifier| {
            verifier.batched_verify(
                &proof,
                b"test/prove",
                verifier_claims(&scheme, &opening_point, &[opening], &commitment),
                BasisMode::Lagrange,
            )
        })
        .expect("native verification");
    scheme
        .verifier(verifier_setup.clone())
        .and_then(|verifier| {
            verifier.batched_verify(
                &proof,
                b"test/prove",
                verifier_claims(&scheme, &opening_point, &[opening + F::one()], &commitment),
                BasisMode::Lagrange,
            )
        })
        .expect_err("native verification must bind the claimed opening");
    scheme
        .verifier(verifier_setup.clone())
        .and_then(|verifier| {
            verifier.batched_verify(
                &proof,
                b"test/different-session",
                verifier_claims(&scheme, &opening_point, &[opening], &commitment),
                BasisMode::Lagrange,
            )
        })
        .expect_err("native verification must bind the session");
    let error = scheme
        .verifier(verifier_setup.clone())
        .and_then(|verifier| {
            verifier.batched_verify(
                &proof,
                b"test/prove",
                verifier_claims(&scheme, &opening_point, &[opening, opening], &commitment),
                BasisMode::Lagrange,
            )
        })
        .expect_err("the statement must fit the setup's batch capacity");
    assert_eq!(
        error,
        AkitaError::InvalidSize {
            expected: 1,
            actual: 2
        },
        "a malformed statement keeps its own error"
    );
    let mut truncated = proof.clone();
    truncated.pop().expect("nonempty native proof");
    scheme
        .verifier(verifier_setup.clone())
        .and_then(|verifier| {
            verifier.batched_verify(
                &truncated,
                b"test/prove",
                verifier_claims(&scheme, &opening_point, &[opening], &commitment),
                BasisMode::Lagrange,
            )
        })
        .expect_err("truncated native proof must reject");
    let mut trailing = proof.clone();
    trailing.push(0);
    scheme
        .verifier(verifier_setup.clone())
        .and_then(|verifier| {
            verifier.batched_verify(
                &trailing,
                b"test/prove",
                verifier_claims(&scheme, &opening_point, &[opening], &commitment),
                BasisMode::Lagrange,
            )
        })
        .expect_err("trailing native proof bytes must reject");
    let mut mutated = proof;
    let middle = mutated.len() / 2;
    mutated[middle] ^= 1;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scheme
            .verifier(verifier_setup.clone())
            .and_then(|verifier| {
                verifier.batched_verify(
                    &mutated,
                    b"test/prove",
                    verifier_claims(&scheme, &opening_point, &[opening], &commitment),
                    BasisMode::Lagrange,
                )
            })
    }));
    assert!(matches!(outcome, Ok(Err(_))));
}
