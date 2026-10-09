//! Audit regression Pr-01: joint fold-response grinding attempts on
//! recursive dense rows whose recursive levels carry a setup-prefix group.

#![allow(missing_docs)]

mod common;

use akita_config::RecursiveCommitmentConfig;
use akita_cpu_backend::CpuBackend;
use common::*;
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::field::{Field as TracingField, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

type RecDense = RecursiveCommitmentConfig<fp128::Dense>;
const TRIALS: u64 = 8;

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

#[test]
fn audit_regression_pr01() {
    init_rayon_pool();
    let observed = Arc::new(Mutex::new(Vec::new()));
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&observed))),
    )
    .expect("global subscriber");
    run_on_large_stack(move || {
        let scheme = load_workspace_scheme::<RecDense>().expect("recursive dense catalog");
        let prefix_levels_of = |num_vars: usize| {
            let row = scheme
                .schedules()
                .resolve_key(&akita_params::ScheduleLookupKey::single(
                    akita_params::PolynomialGroupLayout::singleton(num_vars),
                ))
                .expect("recursive dense row");
            let schedule = row.schedule().clone();
            let prefix_levels: Vec<usize> = schedule
                .recursive_folds
                .iter()
                .enumerate()
                .filter_map(|(index, fold)| fold.params.setup_prefix().map(|_| index + 1))
                .collect();
            (schedule, prefix_levels)
        };
        for num_vars in [20usize, 22, 24, 26, 28] {
            println!(
                "nv={num_vars} prefix_levels={:?}",
                prefix_levels_of(num_vars).1
            );
        }
        let num_vars = [20usize, 22, 24, 26, 28]
            .into_iter()
            .find(|&num_vars| !prefix_levels_of(num_vars).1.is_empty())
            .expect("an offloading recursive dense row");
        {
            let (schedule, prefix_levels) = prefix_levels_of(num_vars);
            let layout = schedule.root.params.final_group();
            let setup = scheme.setup_prover(num_vars, 1).expect("setup");
            let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
            let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
            let mut max_attempts = 0u64;
            for trial in 0..TRIALS {
                let poly = make_dense_poly(num_vars, 0x5eed_0000 + trial);
                let point = random_point(num_vars, 0x9e37_0000 + trial);
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
                let domain = format!("audit/pr01/{num_vars}/{trial}");
                let proof = scheme
                    .batched_prove(
                        &setup,
                        prove_input::<RecDense>(
                            &point,
                            &openings,
                            &commitment,
                            hint,
                            scheme.schedules(),
                        ),
                        &stack,
                        domain.as_bytes(),
                        BasisMode::Lagrange,
                    )
                    .expect("prove");
                scheme
                    .verifier(verifier_setup.clone())
                    .and_then(|verifier| {
                        verifier.batched_verify(
                            &proof,
                            domain.as_bytes(),
                            verify_input::<RecDense>(
                                &point,
                                &openings,
                                &commitment,
                                scheme.schedules(),
                            ),
                            BasisMode::Lagrange,
                        )
                    })
                    .expect("verify");
                let events = observed.lock().expect("capture lock").clone();
                max_attempts = max_attempts.max(events.iter().map(|(_, a)| *a).max().unwrap_or(0));
                println!(
                    "nv={num_vars} trial={trial} prefix_levels={prefix_levels:?} \
                     (group_index, attempts)={events:?}"
                );
            }
            println!("nv={num_vars} max_attempts={max_attempts} trials={TRIALS} cap=4096");
        }
    });
}
