use crate::parallel::ProfileThreadPools;
use crate::report::report_timing;
use akita_error::AkitaError;
use std::time::Instant;

pub(crate) fn run_timings<S, P, F>(
    label: &str,
    pools: &ProfileThreadPools,
    failure_context: &str,
    prepare: P,
    verify: F,
) where
    S: Send,
    P: Fn() -> S + Copy + Send + Sync,
    F: Fn(S) -> Result<(), AkitaError> + Copy + Send + Sync,
{
    for (verify_mode, single_threaded) in [("multi threaded", false), ("single threaded", true)] {
        let statement = prepare();
        tracing::info!(label, verify_mode, "profile verification start");
        let started = Instant::now();
        let result = if single_threaded {
            pools.in_verify_single(|| verify(statement))
        } else {
            pools.in_verify_multi(|| verify(statement))
        };
        let elapsed_s = started.elapsed().as_secs_f64();
        if let Err(error) = result {
            tracing::error!(label, verify_mode, elapsed_s, error = %error, "verify FAILED");
            eprintln!("[{label}] verify {verify_mode} FAILED: {elapsed_s:.6}s ({error})");
            panic!("[{label}] {failure_context} {verify_mode} verification failed: {error}");
        }
        report_timing(label, &format!("verify {verify_mode} OK"), elapsed_s);

        let (phase_result, measurements) = crate::relation_phase_timing::capture(|| {
            let statement = prepare();
            if single_threaded {
                pools.in_verify_single(|| verify(statement))
            } else {
                pools.in_verify_multi(|| verify(statement))
            }
        });
        if let Err(error) = phase_result {
            panic!("[{label}] {failure_context} {verify_mode} phase replay failed: {error}");
        }
        assert!(
            measurements
                .iter()
                .any(|measurement| measurement.phase == "complete_stage2"),
            "[{label}] {verify_mode} phase replay produced no Stage-2 samples"
        );
        for measurement in measurements {
            let mean_elapsed_nanos = measurement
                .elapsed_nanos
                .checked_div(measurement.calls)
                .unwrap_or(0);
            tracing::info!(
                label,
                verify_mode,
                relation_mode = measurement.relation_mode,
                phase = measurement.phase,
                calls = measurement.calls,
                mean_elapsed_nanos,
                total_elapsed_nanos = measurement.elapsed_nanos,
                "verifier relation phase timing"
            );
        }
    }
}
