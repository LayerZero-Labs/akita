#![allow(missing_docs)]

#[path = "support/workspace_schedules.rs"]
mod workspace_schedules;
use workspace_schedules::load_workspace_scheme;

use akita_config::proof_optimized::fp128;
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_prover::SelectedProverOpeningData;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, BasisMode, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
};
use jolt_field::CanonicalEncoding;
use std::sync::Arc;

type Config = fp128::Dense;
type F = fp128::Field;

const NUM_VARS: usize = 14;
const TRANSCRIPT_DOMAIN: &[u8] = b"akita/book/quickstart/v1";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let evaluations: Vec<F> = (0..(1usize << NUM_VARS))
        .map(|index| F::from_u128_reduced(index as u128 + 1))
        .collect();
    let polynomial = DensePoly::from_field_evals(NUM_VARS, &evaluations)?;
    let point: Vec<F> = (0..NUM_VARS)
        .map(|index| F::from_u128_reduced(index as u128 + 2))
        .collect();
    let evaluation = evaluate_multilinear(&evaluations, &point);

    let scheme = load_workspace_scheme::<Config>()?;
    let setup = scheme.setup_prover(NUM_VARS, 1)?;
    let backend = Arc::new(CpuBackend::new::<Config>(
        setup.expanded.clone(),
        scheme.schedules(),
    )?);
    let source = backend.import_source::<Config, _>(vec![polynomial])?;
    let commit_output = backend.commit::<Config>(
        &source,
        GroupContext::scheduler_without_precommitted_groups(),
    )?;

    let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![evaluation],
        commit_output.committed_group.clone(),
    )?])?;
    let prover_data = SelectedProverOpeningData::from_committed_claims::<Config>(
        prover_claims,
        vec![commit_output.private_handle.clone()],
        scheme.schedules(),
    )?;
    let selection = prover_data.selection();

    let mut prover_transcript = AkitaTranscript::<F>::unbound_prover(TRANSCRIPT_DOMAIN);
    let proof = scheme.batched_prove(
        &setup,
        prover_data,
        &backend,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )?;

    let proof_shape = proof.shape();
    let mut proof_bytes = Vec::new();
    proof.serialize_compressed(&mut proof_bytes)?;
    let decoded_proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
        &mut std::io::Cursor::new(&proof_bytes),
        &proof_shape,
    )?;

    let verifier_setup = scheme.setup_verifier(&setup)?;
    let verifier_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point,
        vec![evaluation],
        &commit_output.committed_group,
    )?])?;
    let statement = GroupBatchStatement::new(selection, verifier_claims)?;
    let mut verifier_transcript = AkitaTranscript::<F>::unbound_verifier(TRANSCRIPT_DOMAIN);
    scheme.batched_verify(
        &decoded_proof,
        &verifier_setup,
        &mut verifier_transcript,
        statement,
        BasisMode::Lagrange,
    )?;

    println!("Akita proof verified ({} bytes)", proof_bytes.len());
    Ok(())
}

fn evaluate_multilinear(evaluations: &[F], point: &[F]) -> F {
    let mut layer = evaluations.to_vec();
    let mut active_len = layer.len();
    for &coordinate in point {
        let next_len = active_len / 2;
        for index in 0..next_len {
            let low = layer[2 * index];
            let high = layer[2 * index + 1];
            layer[index] = low + (high - low) * coordinate;
        }
        active_len = next_len;
    }
    layer[0]
}
