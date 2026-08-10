use crate::parallel::ProfileThreadPools;
use crate::report::report_timing;
use akita_field::AkitaError;
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
    }
}
