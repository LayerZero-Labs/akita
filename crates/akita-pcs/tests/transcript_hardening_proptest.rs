#![allow(missing_docs)]
mod common;

use akita_cpu_backend::CpuBackend;
use akita_types::OpeningClaimsLayout;
use common::*;
use proptest::prelude::*;

fn batch_case(index: usize) -> (usize, usize) {
    // Keep fuzz inputs on exact generated rows so failures exercise transcript
    // replay rather than missing-schedule rejection.
    match index {
        0 => (14, 1),
        1 => (15, 2),
        2 => (17, 4),
        // Keep one recursive row for terminal-window coverage without turning
        // this transcript-semantic test into a large-prover benchmark.
        _ => (16, 1),
    }
}

fn native_dense_round_trip(shape_index: usize, basis_mode: BasisMode, seed: u64) {
    init_rayon_pool();
    let scheme = load_workspace_scheme::<DenseCfg>().expect("workspace schedule catalog");

    let (num_vars, total_claims) = batch_case(shape_index);
    let opening_batch =
        OpeningClaimsLayout::new(num_vars, total_claims).expect("valid opening batch");
    let layout = scheme
        .schedules()
        .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
            opening_batch
                .root_final_group_layout()
                .expect("batched group layout"),
        ))
        .map(|row| row.schedule().root.params.final_group())
        .expect("batched commit layout");

    let polys: Vec<DensePoly<F>> = (0..total_claims)
        .map(|poly_idx| make_dense_poly(num_vars, seed.wrapping_add(poly_idx as u64)))
        .collect();
    let opening_point = random_point(num_vars, seed.wrapping_add(0x9e37_0000));
    let poly_refs: Vec<&DensePoly<F>> = polys.iter().collect();
    let openings: Vec<F> = poly_refs
        .iter()
        .map(|poly| opening_from_poly_for_layout(*poly, &opening_point, &layout, basis_mode))
        .collect();

    let setup = scheme.setup_prover(num_vars, total_claims).unwrap();
    let stack =
        CpuBackend::<DenseCfg>::new(setup.expanded.clone(), scheme.schedules()).expect("backend");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

    let akita_cpu_backend::CommitOutput {
        committed_group: commitment,
        private_handle: hint,
    } = stack
        .commit(
            &stack.import_source(polys.to_vec()).expect("source"),
            akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let proof = scheme
        .batched_prove(
            &setup,
            prove_input::<DenseCfg>(
                &opening_point,
                &openings,
                &commitment,
                hint,
                scheme.schedules(),
            ),
            &stack,
            b"hardening/proptest/native",
            basis_mode,
        )
        .expect("prove");

    scheme
        .batched_verify(
            &proof,
            &verifier_setup,
            b"hardening/proptest/native",
            verify_input::<DenseCfg>(&opening_point, &openings, &commitment, scheme.schedules()),
            basis_mode,
        )
        .expect("verify");
    let mut trailing = proof.clone();
    trailing.push(0);
    assert!(scheme
        .batched_verify(
            &trailing,
            &verifier_setup,
            b"hardening/proptest/native",
            verify_input::<DenseCfg>(&opening_point, &openings, &commitment, scheme.schedules(),),
            basis_mode,
        )
        .is_err());
}

#[test]
fn seed_corpus_covers_nv_basis_and_batch_shapes() {
    run_on_large_stack(|| {
        for (shape_index, basis_mode, seed) in [
            (0, BasisMode::Lagrange, 0x1001),
            (1, BasisMode::Lagrange, 0x1002),
            (2, BasisMode::Lagrange, 0x1004),
            (2, BasisMode::Monomial, 0x1005),
            (3, BasisMode::Lagrange, 0x1006),
        ] {
            native_dense_round_trip(shape_index, basis_mode, seed);
        }
    });
}

proptest! {
    // Full proof construction can make a rare grind-heavy input take hours.
    // Keep CI's semantic corpus reproducible; the fixed seed also makes any
    // future runtime regression locally replayable.
    #![proptest_config(ProptestConfig {
        cases: 4,
        rng_seed: proptest::test_runner::RngSeed::Fixed(1),
        ..ProptestConfig::default()
    })]

    #[test]
    fn native_stream_fuzzes_batch_shapes(shape_index in 0usize..4, seed in any::<u64>()) {
        run_on_large_stack(move || native_dense_round_trip(shape_index, BasisMode::Lagrange, seed));
    }
}
