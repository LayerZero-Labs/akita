//! Audit regression Pr-06: honest proving on fp128 dense rows with
//! maximal-energy opening values and with uniform full-width coefficients.

#![allow(missing_docs)]

mod common;

use akita_cpu_backend::CpuBackend;
use common::*;
use jolt_field::Ring;
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::field::{Field as TracingField, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

#[derive(Default)]
struct FoldResponseEvent {
    selected: bool,
    group_index: Option<u64>,
    attempts: Option<u64>,
}

impl Visit for FoldResponseEvent {
    fn record_u64(&mut self, field: &TracingField, value: u64) {
        match field.name() {
            "group_index" => self.group_index = Some(value),
            "attempts" => self.attempts = Some(value),
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &TracingField, value: &dyn fmt::Debug) {
        if field.name() == "message" && format!("{value:?}") == "selected physical fold response" {
            self.selected = true;
        }
    }
}

struct CaptureLayer(Arc<Mutex<Vec<(u64, u64)>>>);

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        let mut record = FoldResponseEvent::default();
        event.record(&mut record);
        if let (true, Some(group_index), Some(attempts)) =
            (record.selected, record.group_index, record.attempts)
        {
            self.0
                .lock()
                .expect("capture lock")
                .push((group_index, attempts));
        }
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The centered value whose balanced base-`2^log_basis` digits are all `-b/2`,
/// using as many digits as fit below `2^127`.
fn all_negative_half_digits(log_basis: u32) -> F {
    let basis = 1u128 << log_basis;
    let digits = 127 / log_basis;
    let mut magnitude = 0u128;
    let mut place = 1u128;
    for _ in 0..digits {
        magnitude += (basis / 2) * place;
        place = place.saturating_mul(basis);
    }
    -F::from_u128(magnitude)
}

#[test]
fn audit_regression_pr06() {
    init_rayon_pool();
    let observed = Arc::new(Mutex::new(Vec::new()));
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&observed))),
    )
    .expect("global subscriber");
    run_on_large_stack(move || {
        let scheme = load_workspace_scheme::<DenseCfg>().expect("dense catalog");
        for num_vars in [14usize, 16] {
            let row = scheme
                .schedules()
                .resolve_key(&akita_params::ScheduleLookupKey::single(
                    akita_params::PolynomialGroupLayout::singleton(num_vars),
                ))
                .expect("dense row");
            let schedule = row.schedule().clone();
            let layout = schedule.root.params.final_group();
            let log_basis_open = schedule.root.params.open().digits.log_basis;
            let setup = scheme.setup_prover(num_vars, 1).expect("setup");
            let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
            let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
            let constant = all_negative_half_digits(log_basis_open);
            for (label, trial) in [
                ("constant_negative_half_digits", 0u64),
                ("uniform_128bit", 1),
                ("uniform_128bit", 2),
            ] {
                let n = 1usize << num_vars;
                let evals: Vec<F> = if label == "constant_negative_half_digits" {
                    vec![constant; n]
                } else {
                    let mut state = 0x0bad_5eed_0000 + trial;
                    (0..n)
                        .map(|_| {
                            let hi = u128::from(splitmix(&mut state));
                            let lo = u128::from(splitmix(&mut state));
                            F::from_u128_reduced((hi << 64) | lo)
                        })
                        .collect()
                };
                let poly = DensePoly::<F>::from_field_evals(num_vars, &evals).expect("dense poly");
                let point = random_point(num_vars, 0x77_0000 + trial);
                let openings = vec![match layout.inner_commit_matrix_params().ring_dimension() {
                    1024 => opening_from_poly_with_basis::<1024, _>(
                        &poly,
                        &point,
                        &layout,
                        BasisMode::Lagrange,
                    ),
                    _ => opening_from_poly_for_layout(&poly, &point, &layout, BasisMode::Lagrange),
                }];
                let akita_cpu_backend::CommitOutput {
                    committed_group: commitment,
                    private_handle: hint,
                } = stack
                    .commit(
                        scheme.schedules(),
                        &stack.import_source(vec![poly]).expect("source"),
                        akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .expect("commit");
                observed.lock().expect("capture lock").clear();
                let domain = format!("audit/pr06/{num_vars}/{label}/{trial}");
                let proof = scheme.batched_prove(
                    &setup,
                    prove_input::<DenseCfg>(
                        &point,
                        &openings,
                        &commitment,
                        hint,
                        scheme.schedules(),
                    ),
                    &stack,
                    domain.as_bytes(),
                    BasisMode::Lagrange,
                );
                let events = observed.lock().expect("capture lock").clone();
                let verified = proof.as_ref().ok().map(|proof| {
                    scheme
                        .verifier(verifier_setup.clone())
                        .and_then(|verifier| {
                            verifier.batched_verify(
                                proof,
                                domain.as_bytes(),
                                verify_input::<DenseCfg>(
                                    &point,
                                    &openings,
                                    &commitment,
                                    scheme.schedules(),
                                ),
                                BasisMode::Lagrange,
                            )
                        })
                        .is_ok()
                });
                println!(
                    "nv={num_vars} log_basis_open={log_basis_open} input={label} trial={trial} \
                     prove_ok={} verify_ok={verified:?} (group_index, attempts)={events:?} err={:?}",
                    proof.is_ok(),
                    proof.as_ref().err(),
                );
                assert!(proof.is_ok() && verified == Some(true));
            }
        }
    });
}
