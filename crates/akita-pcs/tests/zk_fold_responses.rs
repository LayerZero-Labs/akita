//! One-shot fold responses of a zero-knowledge opening (W2), end to end.
//!
//! The zero-knowledge schedule wraps a clear catalog row with its cut. Its
//! plan sends no nonce at the fold responses up to the cut, and the prover
//! probes each of them once. A backend that rejects a chosen probe stands in
//! for a response that overflows its digit range.

#![allow(missing_docs)]

mod common;
#[path = "dynamic_backends/private_cpu.rs"]
mod private_cpu;

use akita_config::{proof_optimized::fp128, unit_onehot_source_chunk_size};
use akita_cpu_backend::{CpuBackend, OneHotPoly};
use akita_error::AkitaError;
use akita_params::zk::{ZkOpeningSchedule, ZK_CUT_LEVEL};
use akita_params::{lagrange_weights, BasisMode};
use akita_pcs::{BackendRegistry, FixedFoldRoute, SelectedProverOpeningData};
use akita_types::{
    AkitaSponge, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims, PROOF_STREAM_PROTOCOL,
};
use jolt_algebra::CanonicalEncoding;
use jolt_transcript::VerifierTranscript;
use private_cpu::PrivateCpu;

type Cfg = fp128::OneHot;
type F = fp128::Field;
type E = F;

const NUM_VARS: usize = 15;
const NUM_POLYS: usize = 4;
const DOMAIN: &[u8] = b"akita/test/zk-fold-responses/v1";

#[test]
fn one_shot_fold_responses_prove_verify_and_overflow() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| run().unwrap());
}

fn run() -> Result<(), AkitaError> {
    let scheme = common::load_workspace_scheme::<Cfg>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )?;
    let chunk_size = unit_onehot_source_chunk_size::<Cfg>()?;
    let chunks_per_poly = (1usize << NUM_VARS) / chunk_size;
    let point: Vec<F> = (0..NUM_VARS)
        .map(|i| F::from_u128_reduced(i as u128 + 5))
        .collect();
    // Polynomial p has its one at position p in every chunk, so its opening
    // is the low-coordinate Lagrange weight for p.
    let low_weights = lagrange_weights(&point[..chunk_size.trailing_zeros() as usize])?;
    let evaluations = low_weights[..NUM_POLYS].to_vec();
    let polys = (0..NUM_POLYS)
        .map(|p| OneHotPoly::<F, u8>::new(chunk_size, vec![Some(p as u8); chunks_per_poly]))
        .collect::<Result<Vec<_>, _>>()?;

    let setup = scheme.setup_prover(NUM_VARS, NUM_POLYS)?;
    let private = PrivateCpu::new(CpuBackend::new(setup.expanded.clone())?);
    let (committed_group, handle) = private.commit_onehot(scheme.schedules(), polys)?;
    let opening = || {
        SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                evaluations.clone(),
                committed_group.clone(),
            )?])?,
            vec![handle.clone()],
            scheme.schedules(),
        )
    };
    let selection = opening()?.selection();
    let resolved = scheme.schedules().resolve_selection(selection)?;
    let levels = resolved.schedule().num_fold_levels();
    assert!(
        levels > ZK_CUT_LEVEL as usize + 1,
        "the cut must precede the terminal fold"
    );
    let zk = ZkOpeningSchedule::for_tests(resolved.schedule().clone(), ZK_CUT_LEVEL)?;
    let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
        resolved.schedule(),
        opening()?.opening_layout(),
    )?;
    let prefixes = private.prefixes(&setup, &prefix_ids);
    let mut registry = BackendRegistry::<Cfg>::new()?;
    let id = registry.register(&private, &prefixes)?;

    let prove = |zk: Option<&ZkOpeningSchedule>| {
        common::prove_on_session(DOMAIN, |transcript| {
            let mut route = FixedFoldRoute::new(vec![id; levels]);
            match zk {
                None => akita_pcs::batched_prove(
                    setup.expanded.descriptor(),
                    scheme.schedules(),
                    &registry,
                    opening()?,
                    transcript,
                    BasisMode::Lagrange,
                    &mut route,
                ),
                Some(zk) => akita_prover::execution::batched_prove_zk(
                    setup.expanded.descriptor(),
                    scheme.schedules(),
                    &registry,
                    opening()?,
                    transcript,
                    BasisMode::Lagrange,
                    &mut route,
                    zk,
                ),
            }
        })
    };
    let verifier = scheme.verifier(scheme.setup_verifier(&setup)?)?;
    let verify = |proof: &[u8], zk: Option<&ZkOpeningSchedule>| {
        let statement = GroupBatchStatement::new(
            selection,
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                evaluations.clone(),
                &committed_group,
            )?])?,
        )?;
        let mut transcript =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, DOMAIN, proof);
        match zk {
            None => verifier.batched_verify(&mut transcript, statement, BasisMode::Lagrange)?,
            Some(zk) => {
                verifier.batched_verify_zk(&mut transcript, statement, BasisMode::Lagrange, zk)?
            }
        }
        Ok::<_, AkitaError>(transcript.finish()?)
    };

    let clear = prove(None)?;
    let one_shot = prove(Some(&zk))?;
    verify(&clear, None)?;
    verify(&one_shot, Some(&zk))?;
    // The plan digest in the instance descriptor binds the mode.
    assert_eq!(verify(&one_shot, None), Err(AkitaError::InvalidProof));
    assert_eq!(verify(&clear, Some(&zk)), Err(AkitaError::InvalidProof));

    // A rejected probe at a masked level aborts after that one probe.
    for level in 0..=ZK_CUT_LEVEL {
        let start = private.fold_probes.get();
        private.reject_fold_probe.set(Some(start + level as usize));
        assert_eq!(prove(Some(&zk)), Err(AkitaError::FoldOverflow { level }));
        assert_eq!(private.fold_probes.get(), start + level as usize + 1);
    }
    // A clear prover retries the same rejection at the next counter.
    private
        .reject_fold_probe
        .set(Some(private.fold_probes.get()));
    let retried = prove(None)?;
    assert_ne!(retried, clear);
    verify(&retried, None)?;
    Ok(())
}
