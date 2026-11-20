// Advice recorded during native verification replays the same proof.

use crate::common::*;
use akita_cpu_backend::{CpuBackend, GroupContext};
use jolt_field::Ring;

#[test]
fn recorded_advice_verifies_the_same_proof() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(recorded_advice_verifies_the_same_proof_inner)
        .unwrap()
        .join()
        .unwrap();
}

fn recorded_advice_verifies_the_same_proof_inner() {
    type Cfg = fp128::Dense;
    const NV: usize = 14;
    const LABEL: &[u8] = b"checked-advice-e2e";
    let scheme =
        load_workspace_scheme::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked())
            .expect("workspace schedule artifact");
    let evals: Vec<F> = (0..1u64 << NV).map(|i| F::from_u64(i * 7 + 3)).collect();
    let point = random_point(NV, 0x0adb_1ce5);
    let opening = dense_opening_lagrange(&evals, &point);
    let setup = scheme.setup_prover(NV, 1).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let output = backend
        .commit(
            scheme.schedules(),
            &backend
                .import_source(vec![DensePoly::<F>::from_field_evals(NV, &evals).unwrap()])
                .unwrap(),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let commitment = output.committed_group;
    let prover_data = prove_input::<Cfg>(
        &point,
        &[opening],
        &commitment,
        output.private_handle,
        scheme.schedules(),
    );
    let proof = scheme
        .batched_prove(&setup, prover_data, &backend, LABEL, BasisMode::Lagrange)
        .unwrap();
    let verifier_setup = scheme.setup_verifier(&setup).unwrap();
    let openings = [opening];
    let verify = || {
        scheme.verifier(verifier_setup.clone())?.batched_verify(
            &proof,
            LABEL,
            verify_input::<Cfg>(&point, &openings, &commitment, scheme.schedules()),
            BasisMode::Lagrange,
            akita_verifier::SetupPolicy::Check(None),
        )
    };

    let ((), tape) = akita_challenges::record_checked_hints(verify).unwrap();
    assert!(!tape.is_empty(), "verification drew no fold challenges");

    akita_challenges::install_checked_hints(&tape, tape.len()).unwrap();
    verify().unwrap();
    akita_challenges::finish_checked_hints().unwrap();

    // Advice with one bit flipped must not verify.
    let mut corrupted = tape;
    let last = corrupted.len() - 1;
    corrupted[last] ^= 1;
    akita_challenges::install_checked_hints(&corrupted, corrupted.len()).unwrap();
    assert!(verify().is_err());
    akita_challenges::abort_checked_hints().unwrap();
}
