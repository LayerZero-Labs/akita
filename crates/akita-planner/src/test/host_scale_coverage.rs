use super::*;

use akita_config::proof_optimized::fp128::{Dense, DenseBounded, OneHot};
use akita_config::{policy_of, CommitmentConfig, RecursiveCommitmentConfig};
use akita_params::sis::CommittedSourceClass;
use akita_params::{ChunkedWitnessCfg, MultiChunkProfileId, OpeningClaimsLayout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

type RingChallengeConfig = fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

struct Shape {
    label: String,
    key: PolynomialGroupLayout,
    policy: PlannerPolicy,
    contract: CommittedSourceContract,
    ring_challenge_config: RingChallengeConfig,
}

fn root_group(num_vars: usize, num_polys: usize) -> PolynomialGroupLayout {
    OpeningClaimsLayout::new(num_vars, num_polys)
        .and_then(|layout| layout.root_final_group_layout())
        .unwrap()
}

/// Jolt-scale trace shapes at `2^30` and `2^31` for every chunk profile.
///
/// The one-hot policies mirror Jolt's: fp128 one-hot with a K=16 or K=256
/// source, recursive setup planning, and single, W2R2, W4R2, or W8R2
/// chunking. K=16 trace rows open `log_t + 4` variables with 51 to 64
/// columns; K=256 rows open `log_t + 8` variables. The dense producers use
/// W8R2 chunking. Planning takes minutes per profile, so the check is ignored
/// by default; run it with `--ignored --nocapture`.
fn shapes() -> Vec<Shape> {
    let k16 = CommittedSourceContract::try_new(
        CommittedSourceClass::UnitOneHot {
            source_chunk_size: 16,
        },
        OneHot::decomposition(),
    )
    .unwrap();
    let k256 = OneHot::committed_source_contract().unwrap();
    let mut shapes = Vec::new();
    for (profile, witness_chunk) in [
        ("single", ChunkedWitnessCfg::default()),
        (
            "w2r2",
            ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W2R2),
        ),
        (
            "w4r2",
            ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W4R2),
        ),
        (
            "w8r2",
            ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W8R2),
        ),
    ] {
        let mut policy = policy_of::<RecursiveCommitmentConfig<OneHot>>();
        policy.witness_chunk = witness_chunk;
        let mut push = |label: String, key, contract| {
            shapes.push(Shape {
                label,
                key,
                policy,
                contract,
                ring_challenge_config: OneHot::ring_challenge_config,
            });
        };
        for num_vars in [34, 35] {
            for num_polys in 51..=64 {
                push(
                    format!("k16-{profile} trace nv{num_vars} p{num_polys}"),
                    root_group(num_vars, num_polys),
                    k16,
                );
            }
        }
        for num_vars in 40..=44 {
            for num_polys in 1..=2 {
                push(
                    format!("k16-{profile} nv{num_vars} p{num_polys}"),
                    root_group(num_vars, num_polys),
                    k16,
                );
            }
        }
        for num_vars in [43, 44] {
            for num_polys in 1..=2 {
                push(
                    format!("k256-{profile} nv{num_vars} p{num_polys}"),
                    root_group(num_vars, num_polys),
                    k256,
                );
            }
        }
        for num_vars in [38, 39] {
            for num_polys in [27, 37] {
                push(
                    format!("k256-{profile} trace nv{num_vars} p{num_polys}"),
                    root_group(num_vars, num_polys),
                    k256,
                );
            }
        }
    }
    let w8r2 = ChunkedWitnessCfg::from_profile(MultiChunkProfileId::W8R2);
    for num_vars in [34, 35] {
        for num_polys in [1, 2] {
            let mut policy = policy_of::<DenseBounded>();
            policy.witness_chunk = w8r2;
            shapes.push(Shape {
                label: format!("dense-bounded nv{num_vars} p{num_polys}"),
                key: root_group(num_vars, num_polys),
                policy,
                contract: DenseBounded::committed_source_contract().unwrap(),
                ring_challenge_config: DenseBounded::ring_challenge_config,
            });
        }
        let mut policy = policy_of::<Dense>();
        policy.witness_chunk = w8r2;
        shapes.push(Shape {
            label: format!("dense-full nv{num_vars} p1"),
            key: root_group(num_vars, 1),
            policy,
            contract: Dense::committed_source_contract().unwrap(),
            ring_challenge_config: Dense::ring_challenge_config,
        });
    }
    shapes
}

#[test]
#[ignore = "plans 190 Jolt-scale shapes; takes minutes"]
fn jolt_scale_shapes_plan_for_every_chunk_profile() {
    let shapes = shapes();
    let next = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..std::thread::available_parallelism().map_or(8, |threads| threads.get()) {
            scope.spawn(|| {
                while let Some(shape) = shapes.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let planned = find_schedule(
                        &ScheduleLookupKey::single(shape.key),
                        shape.contract,
                        &[],
                        &shape.policy,
                        shape.ring_challenge_config,
                    )
                    .and_then(|planned| {
                        planned.schedule.validate_structure()?;
                        Ok(planned)
                    });
                    match planned {
                        Ok(planned) => eprintln!(
                            "{}: folds={} proof={} setup={}",
                            shape.label,
                            planned.schedule.recursive_folds.len() + 1,
                            planned.estimate.estimated_proof_payload_bytes().unwrap()
                                + planned.estimate.nonce_max_bytes,
                            planned.estimate.estimated_num_setup_field_elements,
                        ),
                        Err(error) => failures
                            .lock()
                            .unwrap()
                            .push(format!("{}: {error}", shape.label)),
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{failures:#?}");
}
