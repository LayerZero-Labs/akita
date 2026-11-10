//! CPU timing mode and a persistent pool for secret-dependent prover work.
//!
//! [`run_secret`] enables Arm FEAT_DIT on every worker when supported. DIT
//! provides data-independent timing for the instructions covered by that
//! feature; callers must still use constant-time algorithms and memory access
//! patterns. Other architectures run the same pool without a CPU timing mode.
//! [`status`] reports that distinction without changing the calling thread.
//!
//! Rayon work spawned within the closure stays in this pool. Work explicitly
//! sent to another pool or to a new OS thread does not inherit this protection.

#![forbid(unsafe_code)]

use std::cell::Cell;
use std::fmt;
use std::sync::{Arc, OnceLock};

use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

#[cfg(target_arch = "aarch64")]
static DIT: std::sync::LazyLock<aarch64_dit::Dit> =
    std::sync::LazyLock::new(aarch64_dit::Dit::init);

thread_local! {
    static SECRET_WORKER: Cell<bool> = const { Cell::new(false) };

    // Each guard stays on the worker that enabled it, until that worker exits.
    #[cfg(target_arch = "aarch64")]
    static WORKER_DIT: Cell<Option<aarch64_dit::Guard<'static>>> = const { Cell::new(None) };
}

static SECRET_POOL: OnceLock<Result<ThreadPool, PoolError>> = OnceLock::new();

// Match Akita's profiling pools and Jolt's Akita backend adapter: recursive
// Rayon kernels can overflow the default worker stack at trace-scale shapes.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Whether the secret pool can enforce a CPU data-independent timing mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimingMode {
    /// Workers enable Arm FEAT_DIT for their lifetime.
    Enforced,
    /// No supported userspace timing mode exists on this CPU or architecture.
    Unavailable,
}

/// Reports the CPU mode used by [`run_secret`], without initializing the pool.
///
/// FEAT_DIT hosts, including Apple M-series Macs, report [`TimingMode::Enforced`].
/// Other hosts, including x86, report [`TimingMode::Unavailable`]. This is a
/// capability report, not a claim about the calling thread's current DIT bit.
pub fn status() -> TimingMode {
    #[cfg(target_arch = "aarch64")]
    if DIT.is_supported() {
        return TimingMode::Enforced;
    }
    TimingMode::Unavailable
}

/// A failure to build the process's dedicated secret-work pool.
///
/// The first initialization result, including any failure, is cached. Clones
/// retain the original Rayon error and its source without retrying pool creation.
#[derive(Clone, Debug)]
pub struct PoolError(Arc<ThreadPoolBuildError>);

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to build the Akita secret-work pool: {}", self.0)
    }
}

impl std::error::Error for PoolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

/// Whether this thread is a worker of the dedicated secret-work pool.
///
/// This reports pool membership even when [`status`] is [`TimingMode::Unavailable`].
pub fn in_secret_pool() -> bool {
    SECRET_WORKER.get()
}

/// Runs `op` on the persistent secret-work pool, or inline on its own workers.
///
/// The pool is built on first use with 64 MiB worker stacks. Its worker count
/// follows Rayon's default policy (`RAYON_NUM_THREADS`, otherwise available
/// parallelism), independently of the pool from which initialization is called.
/// The count is fixed for the process lifetime. Benchmarks that compare one
/// worker with N workers must use separate processes with `RAYON_NUM_THREADS`
/// set before first use. Custom caller pools do not resize this pool.
///
/// Both `op` and its result must be `Send`, because execution crosses a worker
/// boundary. Borrowed captures need not be `'static`: mutable RNG trait objects
/// must retain `+ Send`, as must opening types captured through `&mut`. Draw
/// secret coins inside `op` so that the draw also runs under the timing mode.
///
/// Nested calls execute inline on the same worker. Closure panics propagate to
/// the caller, as with [`ThreadPool::install`]; they are not [`PoolError`]s.
pub fn run_secret<T: Send>(op: impl FnOnce() -> T + Send) -> Result<T, PoolError> {
    if in_secret_pool() {
        return Ok(op());
    }
    let pool = SECRET_POOL.get_or_init(|| {
        let pool = ThreadPoolBuilder::new()
            .stack_size(WORKER_STACK_BYTES)
            .thread_name(|index| format!("akita-secret-{index}"))
            .start_handler(|_| {
                #[cfg(target_arch = "aarch64")]
                WORKER_DIT.set(Some(DIT.enable()));
                SECRET_WORKER.set(true);
            })
            .build()
            .map_err(|error| PoolError(Arc::new(error)))?;
        tracing::debug!(
            timing_mode = ?status(),
            workers = pool.current_num_threads(),
            worker_stack_bytes = WORKER_STACK_BYTES,
            "initialized Akita secret-work pool"
        );
        Ok(pool)
    });
    match pool {
        Ok(pool) => Ok(pool.install(op)),
        Err(error) => Err(error.clone()),
    }
}

#[cfg(test)]
mod tests;
