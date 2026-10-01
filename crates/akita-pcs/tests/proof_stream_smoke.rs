use akita_config::{
    proof_optimized::{fp128, fp32},
    CommitmentConfig,
};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext, OneHotPoly};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_types::{
    lagrange_weights, BasisMode, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
};
use jolt_field::{Ring, Zero};
use jolt_transcript::{
    Blake2b512, Keccak, ProtocolId, ProverTranscript, Sponge, VerifierTranscript,
};

#[test]
fn dense_fp128_roundtrip() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(dense_fp128_roundtrip_inner)
        .unwrap()
        .join()
        .unwrap();
}

/// The core entry points run on a caller-chosen sponge and protocol id, and
/// leave the final consumption check to the caller.
#[test]
fn onehot_fp32_roundtrip_on_caller_transcripts() {
    for run in [
        onehot_fp32_roundtrip_inner::<Blake2b512> as fn(),
        onehot_fp32_roundtrip_inner::<Keccak>,
    ] {
        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(run)
            .unwrap()
            .join()
            .unwrap();
    }
}

fn onehot_fp32_roundtrip_inner<H: Sponge>() {
    type Cfg = fp32::OneHot;
    type F = <Cfg as CommitmentConfig>::Field;
    type E = <Cfg as CommitmentConfig>::ExtField;
    const NV: usize = 14;
    let schedules = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
    let scheme = AkitaCommitmentScheme::<Cfg>::new(schedules);
    let setup = scheme.setup_prover(NV, 1).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let chunk_size = akita_config::unit_onehot_source_chunk_size::<Cfg>().unwrap();
    let indices = (0..((1 << NV) / chunk_size))
        .map(|chunk| Some(((chunk * 29 + 7) % chunk_size) as u8))
        .collect();
    let poly = OneHotPoly::<F, u8>::new(chunk_size, indices).unwrap();
    let point: Vec<E> = (0..NV)
        .map(|index| E::from_u64((index + 2) as u64))
        .collect();
    let weights = lagrange_weights(&point).unwrap();
    let opening = poly
        .indices()
        .iter()
        .enumerate()
        .fold(E::zero(), |sum, (chunk, hot)| {
            sum + hot.map_or(E::zero(), |index| {
                weights[chunk * chunk_size + usize::from(index)]
            })
        });
    let output = backend
        .commit(
            scheme.schedules(),
            &backend.import_source(vec![poly]).unwrap(),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let commitment = output.committed_group;
    let group =
        PolynomialGroupClaims::new(point.clone(), vec![opening], commitment.clone()).unwrap();
    let claims = OpeningClaims::from_groups(vec![group]).unwrap();
    let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        claims,
        vec![output.private_handle],
        scheme.schedules(),
    )
    .unwrap();
    let selection = prover_data.selection();
    let protocol = ProtocolId::new::<H>("akita-pcs/test/caller-transcript");
    let session = b"native-port-onehot-fp32";
    let mut prover = ProverTranscript::<H>::new(&protocol, session);
    prover.send(&7u32);
    scheme
        .batched_prove(
            &setup,
            prover_data,
            &backend,
            &mut prover,
            BasisMode::Lagrange,
        )
        .unwrap();
    let proof = prover.finish();
    let verify_group = PolynomialGroupClaims::new(point, vec![opening], &commitment).unwrap();
    let verify_claims = OpeningClaims::from_groups(vec![verify_group]).unwrap();
    let statement = GroupBatchStatement::new(selection, verify_claims).unwrap();
    let verifier_setup = scheme.setup_verifier(&setup).unwrap();
    let mut verifier = VerifierTranscript::<H>::new(&protocol, session, &proof);
    assert_eq!(verifier.receive::<u32>().unwrap(), 7);
    scheme
        .verifier(verifier_setup)
        .and_then(|akita| akita.batched_verify(&mut verifier, statement, BasisMode::Lagrange))
        .unwrap();
    verifier.finish().unwrap();
}

fn dense_fp128_roundtrip_inner() {
    type Cfg = fp128::Dense;
    type F = <Cfg as CommitmentConfig>::Field;
    const NV: usize = 14;
    let schedules = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
    let scheme = AkitaCommitmentScheme::<Cfg>::new(schedules);
    let setup = scheme.setup_prover(NV, 1).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let evals: Vec<F> = (0..(1 << NV))
        .map(|index| F::from_u64(index as u64))
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
        .map(|index| F::from_u64((index + 2) as u64))
        .collect();
    let weights = lagrange_weights(&point).unwrap();
    let opening = evals
        .iter()
        .zip(weights)
        .fold(F::zero(), |sum, (&value, weight)| sum + value * weight);
    let group =
        PolynomialGroupClaims::new(point.clone(), vec![opening], commitment.clone()).unwrap();
    let claims = OpeningClaims::from_groups(vec![group]).unwrap();
    let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        claims,
        vec![output.private_handle],
        scheme.schedules(),
    )
    .unwrap();
    let selection = prover_data.selection();
    let proof = scheme
        .prove_standalone(
            &setup,
            prover_data,
            &backend,
            b"native-port-smoke",
            BasisMode::Lagrange,
        )
        .unwrap();
    let verify_group = PolynomialGroupClaims::new(point, vec![opening], &commitment).unwrap();
    let verify_claims = OpeningClaims::from_groups(vec![verify_group]).unwrap();
    let statement = GroupBatchStatement::new(selection, verify_claims).unwrap();
    let verifier_setup = scheme.setup_verifier(&setup).unwrap();
    scheme
        .verifier(verifier_setup.clone())
        .and_then(|verifier| {
            verifier.verify_standalone(&proof, b"native-port-smoke", statement, BasisMode::Lagrange)
        })
        .unwrap();
}
