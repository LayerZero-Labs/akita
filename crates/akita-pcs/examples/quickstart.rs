#![allow(missing_docs)]

#[path = "support/workspace_schedules.rs"]
mod workspace_schedules;
use workspace_schedules::load_workspace_scheme;

use akita_config::proof_optimized::fp128;
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_prover::SelectedProverOpeningData;
use akita_types::{BasisMode, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
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
    let backend = Arc::new(CpuBackend::new(setup.expanded.clone())?);
    let source = backend.import_source(vec![polynomial])?;
    let commit_output = backend.commit(
        scheme.schedules(),
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

    let proof = scheme.batched_prove(
        &setup,
        prover_data,
        &backend,
        TRANSCRIPT_DOMAIN,
        BasisMode::Lagrange,
    )?;

    let verifier_setup = scheme.setup_verifier(&setup)?;
    let verifier_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point,
        vec![evaluation],
        &commit_output.committed_group,
    )?])?;
    let statement = GroupBatchStatement::new(selection, verifier_claims)?;
    scheme.batched_verify(
        &proof,
        &verifier_setup,
        TRANSCRIPT_DOMAIN,
        statement,
        BasisMode::Lagrange,
    )?;

    println!("Akita proof verified");
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
