//! Complete malicious transcripts for the same bound grouped statement.

use super::fault_backend::{Fault, FaultBackend};
use akita_config::proof_optimized::fp64::ExtensionField as E;
use akita_planner::native_root::find_native_root_schedule;
use akita_planner::prime_planes::price_grouped_root;
use akita_schedules::native_root::NativeRootTransition;
use akita_types::{OpeningClaims, PolynomialGroupClaims};
use std::sync::{Arc, Mutex};
use tracing_subscriber::prelude::*;

#[allow(dead_code)]
#[path = "../../../akita-planner/src/native_test_fixture.rs"]
mod resources;
use resources::native_open_fixture::{ext, opening, Claims};

#[derive(Clone, Default)]
struct RelationRejection(Arc<Mutex<usize>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for RelationRejection {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        if event.metadata().target() == "akita_verifier::binary_root_relation" {
            *self.0.lock().unwrap() += 1;
        }
    }
}

fn reject(fault: Fault, claims: &Claims) {
    let opening = opening();
    let backend = FaultBackend::new(opening.backend.clone(), [0; 32], Vec::new(), fault);
    // A fresh proof binds exactly the claims supplied to the verifier below.
    // The backend asserts that the honest assembled oracle detects the fault,
    // then repairs its private oracle so every sumcheck round is consistent.
    let proof = opening.prove_with(&backend, claims).unwrap();
    let receipt = RelationRejection::default();
    let subscriber = tracing_subscriber::registry().with(receipt.clone());
    let result = tracing::subscriber::with_default(subscriber, || {
        opening.verify(claims, &opening.setup, &proof)
    });
    assert!(result.is_err(), "accepted {fault:?}");
    assert_eq!(
        *receipt.0.lock().unwrap(),
        1,
        "wrong rejection boundary: {fault:?}"
    );
}

#[test]
fn native_open_rejects_forged_source_for_each_binary_owner() {
    super::common::run_on_large_stack(|| {
        let opening = opening();
        for group in 0..4 {
            reject(Fault::GroupedSource { group }, &opening.claims);
        }
    });
}

#[test]
fn native_open_rejects_forged_prime_for_each_owner_point_coordinate() {
    super::common::run_on_large_stack(|| {
        let opening = opening();
        for group in 0..4 {
            for point in 0..2 {
                for coordinate in 0..2 {
                    reject(
                        Fault::GroupedPrime {
                            group,
                            point,
                            coordinate,
                        },
                        &opening.claims,
                    );
                }
            }
        }
    });
}

#[test]
fn native_open_rejects_forged_signed_reading_for_each_slot_point_coordinate() {
    super::common::run_on_large_stack(|| {
        let opening = opening();
        for group in 4..8 {
            let original = &opening.claims.groups()[group];
            let readings = original.point_evaluations().collect::<Vec<_>>();
            for slot in 0..original.evaluations().len() {
                for point in 0..2 {
                    for coordinate in 0..2 {
                        let mut groups = opening.claims.groups().to_vec();
                        let mut values = [
                            readings[0].evaluations.to_vec(),
                            readings[1].evaluations.to_vec(),
                        ];
                        let mut delta = [0; 2];
                        delta[coordinate] = 1;
                        values[point][slot] += ext::<E>(delta);
                        groups[group] = PolynomialGroupClaims::new_two_points(
                            [readings[0].point.to_vec(), readings[1].point.to_vec()],
                            values,
                            original.commitment().clone(),
                        )
                        .unwrap();
                        let claims = OpeningClaims::from_groups(groups).unwrap();
                        let row = opening
                            .fixture
                            .catalog
                            .resolve_selection(opening.fixture.bindings[0].0)
                            .unwrap();
                        akita_config::bind_grouped_root(row, opening.binding.setup(), &claims)
                            .unwrap();
                        reject(Fault::GroupedValue, &claims);
                    }
                }
            }
        }
    });
}

#[test]
fn native_open_rejects_forged_signed_segment_for_each_slot() {
    super::common::run_on_large_stack(|| {
        let opening = opening();
        for group in 0..4 {
            for slot in 0..opening.claims.groups()[4 + group].evaluations().len() {
                reject(Fault::GroupedSigned { group, slot }, &opening.claims);
            }
        }
    });
}
