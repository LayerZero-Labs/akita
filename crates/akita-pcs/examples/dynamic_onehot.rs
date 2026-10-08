//! Four one-hot polynomials with 2^30 evaluations each, routed PrivateCpu -> CPU -> PrivateCpu.
//!
//! Run from the workspace: cargo run -p akita-pcs --release --example dynamic_onehot
//! PrivateCpu is a distinct handle family that forwards to an inner CpuBackend;
//! the other owner is a stock CpuBackend. Bridges convert between their packets.

#[path = "support/workspace_schedules.rs"]
mod workspace_schedules;

use akita_config::{proof_optimized::fp128, unit_onehot_source_chunk_size};
use akita_cpu_backend::{CpuBackend, OneHotPoly};
use akita_params::{lagrange_weights, BasisMode};
use akita_pcs::{batched_prove, BackendRegistry, FixedFoldRoute, SelectedProverOpeningData};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::CanonicalEncoding;
use std::{error::Error, time::Instant};

// Types expected by the shared PrivateCpu test backend module.
type Cfg = fp128::OneHot;
type F = fp128::Field;
type E = F;
const NV: usize = 30;

#[path = "../tests/dynamic_backends/private_cpu.rs"]
mod private_cpu;

use private_cpu::PrivateCpu;

const NUM_VARS: usize = NV;
const NUM_POLYS: usize = 4;
const STACK_BYTES: usize = 64 * 1024 * 1024;
const DOMAIN: &[u8] = b"akita/example/dynamic-onehot/v1";

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(STACK_BYTES)
        .build_global()?;

    // Ring kernels need a larger stack on both the caller and Rayon workers.
    std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(run)?
        .join()
        .map_err(|_| std::io::Error::other("example worker panicked"))?
}

fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let scheme = workspace_schedules::load_workspace_scheme::<Cfg>()?;
    let chunk_size = unit_onehot_source_chunk_size::<Cfg>()?;
    let chunks_per_poly = (1usize << NUM_VARS) / chunk_size;
    let point: Vec<F> = (0..NUM_VARS)
        .map(|i| F::from_u128_reduced(i as u128 + 2))
        .collect();

    // Polynomial p has its one at position p in every chunk. Each represents
    // 2^30 entries using only 2^30 / chunk_size compact hot-position indices.
    // Its opening is the low-coordinate Lagrange weight for p: the identical
    // chunks are independent of the high coordinates, whose weights sum to one.
    let low_vars = chunk_size.trailing_zeros() as usize;
    let low_weights = lagrange_weights(&point[..low_vars])?;
    let evaluations = low_weights[..NUM_POLYS].to_vec();
    let polys = (0..NUM_POLYS)
        .map(|p| OneHotPoly::<F, u8>::new(chunk_size, vec![Some(p as u8); chunks_per_poly]))
        .collect::<Result<Vec<_>, _>>()?;

    println!("Batch: {NUM_POLYS} one-hot polynomials, {NUM_VARS} variables each (2^30 entries)");
    let started = Instant::now();
    let setup = scheme.setup_prover(NUM_VARS, NUM_POLYS)?;
    let private = PrivateCpu::new(CpuBackend::new(setup.expanded.clone())?);
    let cpu = CpuBackend::<F, E>::new(setup.expanded.clone())?;
    println!("Setup: {:.3}s", started.elapsed().as_secs_f64());

    let started = Instant::now();
    let (committed_group, private_handle) = private.commit_onehot(scheme.schedules(), polys)?;
    println!(
        "Commit on PrivateCpu: {:.3}s",
        started.elapsed().as_secs_f64()
    );
    let opening = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            evaluations.clone(),
            committed_group.clone(),
        )?])?,
        vec![private_handle],
        scheme.schedules(),
    )?;
    let selection = opening.selection();
    let resolved = scheme.schedules().resolve_selection(selection)?;
    let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
        resolved.schedule(),
        opening.opening_layout(),
    )?;
    let prefixes_private = private.prefixes(&setup, &prefix_ids);
    let prefixes_cpu = cpu.import_setup_prefixes(&setup.prefix_slots, &prefix_ids)?;
    let mut registry = BackendRegistry::<Cfg>::new()?;
    let private_id = registry.register(&private, &prefixes_private)?;
    let cpu_id = registry.register(&cpu, &prefixes_cpu)?;
    registry.register_bridge::<PrivateCpu, CpuBackend<F, E>>()?;
    registry.register_bridge::<CpuBackend<F, E>, PrivateCpu>()?;
    let levels = resolved.schedule().num_fold_levels();

    // Alternate owners: PrivateCpu on even folds, stock CPU on odd folds.
    let mut route = FixedFoldRoute::new(
        (0..levels)
            .map(|level| if level % 2 == 0 { private_id } else { cpu_id })
            .collect(),
    );
    for level in 0..levels {
        println!(
            "Fold {level}: {}",
            if level % 2 == 0 {
                "PrivateCpu"
            } else {
                "CpuBackend"
            }
        );
    }
    let started = Instant::now();
    let proof = batched_prove(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        opening,
        DOMAIN,
        BasisMode::Lagrange,
        &mut route,
    )?;
    println!(
        "Prove: {:.3}s, {} proof bytes",
        started.elapsed().as_secs_f64(),
        proof.len()
    );

    let verifier = scheme.verifier(scheme.setup_verifier(&setup)?)?;
    let statement = GroupBatchStatement::new(
        selection,
        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point,
            evaluations,
            &committed_group,
        )?])?,
    )?;
    let started = Instant::now();
    verifier.batched_verify(&proof, DOMAIN, statement, BasisMode::Lagrange)?;
    println!("Verified: {:.3}s", started.elapsed().as_secs_f64());
    Ok(())
}
