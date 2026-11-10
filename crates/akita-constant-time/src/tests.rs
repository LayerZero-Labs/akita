use std::error::Error;
use std::thread;

use rayon::prelude::*;

use super::{in_secret_pool, run_secret, status, PoolError, TimingMode};

fn dit_enabled() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        super::DIT.is_enabled()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[test]
fn apple_silicon_enforces_timing_mode() {
    assert_eq!(status(), TimingMode::Enforced);
}

#[cfg(not(target_arch = "aarch64"))]
#[test]
fn other_architectures_have_no_timing_mode() {
    assert_eq!(status(), TimingMode::Unavailable);
}

#[test]
fn every_worker_enters_secret_mode_and_caller_stays_outside() -> Result<(), PoolError> {
    assert!(!in_secret_pool());
    let before = dit_enabled();
    let workers = run_secret(|| {
        assert!(in_secret_pool());
        rayon::broadcast(|context| {
            (
                context.index(),
                in_secret_pool(),
                dit_enabled(),
                thread::current().name().unwrap().to_owned(),
            )
        })
    })?;
    assert!(!workers.is_empty());
    for (index, inside, dit, name) in workers {
        assert!(inside);
        assert_eq!(dit, status() == TimingMode::Enforced);
        assert_eq!(name, format!("akita-secret-{index}"));
    }
    assert!(!in_secret_pool());
    assert_eq!(dit_enabled(), before);
    Ok(())
}

#[test]
fn nested_entries_run_inline_on_every_worker() -> Result<(), PoolError> {
    run_secret(|| {
        rayon::broadcast(|_| {
            let worker = thread::current().id();
            run_secret(|| {
                assert_eq!(thread::current().id(), worker);
                assert!(in_secret_pool());
                assert_eq!(dit_enabled(), status() == TimingMode::Enforced);
            })
            .unwrap();
        });
    })
}

#[test]
fn repeated_and_concurrent_calls_reuse_workers() -> Result<(), PoolError> {
    let first = run_secret(|| rayon::broadcast(|_| thread::current().id()))?;
    for _ in 0..3 {
        assert_eq!(
            run_secret(|| rayon::broadcast(|_| thread::current().id()))?,
            first
        );
    }
    thread::scope(|scope| {
        let calls: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| run_secret(|| rayon::broadcast(|_| thread::current().id()))))
            .collect();
        for call in calls {
            assert_eq!(call.join().unwrap().unwrap(), first);
        }
    });
    Ok(())
}

#[test]
fn borrowed_mutable_send_captures_and_results_cross_pool_boundary() -> Result<(), PoolError> {
    let mut coins = vec![1_u8, 2, 3];
    let mut opening = String::from("opening");
    // Model the borrowed + Send trait-object boundary required by coin draws.
    let draw: &mut (dyn FnMut() -> u8 + Send) = &mut || coins.pop().unwrap();
    let opening_ref = &mut opening;
    let result = run_secret(move || {
        assert!(in_secret_pool());
        opening_ref.push_str(" updated");
        (draw(), opening_ref)
    })?;
    assert_eq!(result.0, 3);
    assert_eq!(result.1, "opening updated");
    assert_eq!(coins, [1, 2]);
    Ok(())
}

#[test]
fn rayon_jobs_inherit_secret_mode() -> Result<(), PoolError> {
    let all_inside = run_secret(|| {
        (0..1_024)
            .into_par_iter()
            .all(|_| in_secret_pool() && dit_enabled() == (status() == TimingMode::Enforced))
    })?;
    assert!(all_inside);
    Ok(())
}

#[test]
fn panic_propagates_and_pool_remains_usable() -> Result<(), PoolError> {
    assert!(std::panic::catch_unwind(|| run_secret(|| panic!("test panic"))).is_err());
    assert!(!in_secret_pool());
    assert!(run_secret(in_secret_pool)?);
    Ok(())
}

#[test]
fn initialization_ignores_caller_pool_width() -> Result<(), Box<dyn Error>> {
    // Isolate first use and environment changes from the other tests. Running
    // this test in a child also exercises the one/N-worker benchmark policy.
    const CHILD: &str = "AKITA_CONSTANT_TIME_POLICY_TEST_CHILD";
    if let Ok(count) = std::env::var(CHILD) {
        let count: usize = count.parse()?;
        let caller = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        caller.install(|| {
            assert_eq!(rayon::current_num_threads(), 1);
            assert!(!in_secret_pool());
            assert_eq!(run_secret(rayon::current_num_threads).unwrap(), count);
            assert!(!in_secret_pool());
        });
        assert_eq!(run_secret(rayon::current_num_threads)?, count);
    } else {
        for count in ["1", "2"] {
            let output = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "tests::initialization_ignores_caller_pool_width"])
                .env(CHILD, count)
                .env("RAYON_NUM_THREADS", count)
                .output()?;
            assert!(
                output.status.success(),
                "child failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    Ok(())
}
