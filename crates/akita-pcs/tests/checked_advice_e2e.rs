// Advice recorded during native verification replays the same proof.

use akita_config::{proof_optimized::fp128, CommitmentConfig};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_params::{lagrange_weights, BasisMode};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::{Ring, Zero};

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
    type F = <Cfg as CommitmentConfig>::Field;
    const NV: usize = 14;
    const DOMAIN: &[u8] = b"checked-advice-e2e";
    let schedules = akita_config::test_support::workspace_schedule_catalog::<Cfg>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
    let scheme = AkitaCommitmentScheme::<Cfg>::new(schedules);
    let setup = scheme.setup_prover(NV, 1).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let evals: Vec<F> = (0..(1 << NV))
        .map(|index| F::from_u64(index as u64 * 7 + 3))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(NV, &evals).unwrap();
    let output = backend
        .commit(
            scheme.schedules(),
            &backend.import_source(vec![poly]).unwrap(),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let commitment = output.committed_group;
    let point: Vec<F> = (0..NV)
        .map(|index| F::from_u64((index + 5) as u64))
        .collect();
    let opening = evals
        .iter()
        .zip(lagrange_weights(&point).unwrap())
        .fold(F::zero(), |sum, (&value, weight)| sum + value * weight);
    let group =
        PolynomialGroupClaims::new(point.clone(), vec![opening], commitment.clone()).unwrap();
    let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        OpeningClaims::from_groups(vec![group]).unwrap(),
        vec![output.private_handle],
        scheme.schedules(),
    )
    .unwrap();
    let selection = prover_data.selection();
    let proof = scheme
        .batched_prove(&setup, prover_data, &backend, DOMAIN, BasisMode::Lagrange)
        .unwrap();
    let verifier_setup = scheme.setup_verifier(&setup).unwrap();
    let verify = || {
        let group = PolynomialGroupClaims::new(point.clone(), vec![opening], &commitment)?;
        let statement =
            GroupBatchStatement::new(selection, OpeningClaims::from_groups(vec![group])?)?;
        scheme.verifier(verifier_setup.clone())?.batched_verify(
            &proof,
            DOMAIN,
            statement,
            BasisMode::Lagrange,
            akita_verifier::SetupPolicy::Check(None),
        )
    };

    let (verified, frame) = akita_challenges::record_checked_hints(verify).unwrap();
    verified.unwrap();
    assert!(frame.len() > 16, "verification drew no fold challenges");

    akita_challenges::install_checked_hints(&frame, frame.len()).unwrap();
    verify().unwrap();
    akita_challenges::finish_checked_hints().unwrap();

    // Advice from the wrong tape position must not verify.
    let mut corrupted = frame.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 1;
    akita_challenges::install_checked_hints(&corrupted, corrupted.len()).unwrap();
    assert!(verify().is_err() || akita_challenges::finish_checked_hints().is_err());
    akita_challenges::abort_checked_hints().unwrap();
}
