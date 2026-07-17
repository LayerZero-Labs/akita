//! Scoped Rayon pools for independent prove vs verify thread counts in profile runs.
//!
//! Prove-side work (setup, commit, prove) uses the process-global Rayon pool, sized at
//! startup from `AKITA_PROFILE_PROVE_THREADS`. When verify needs a different thread
//! count, a dedicated verify pool is created; otherwise verify reuses the global pool.

use std::env;
use std::sync::OnceLock;

static POOLS: OnceLock<ProfileThreadPools> = OnceLock::new();

/// Per-phase thread pools for the profile harness.
pub(crate) struct ProfileThreadPools {
    #[cfg(feature = "parallel")]
    verify_pool: Option<rayon::ThreadPool>,
}

impl ProfileThreadPools {
    /// Build pools from profile env vars, size the global prove pool, and log resolved counts.
    pub(crate) fn init() -> &'static Self {
        POOLS.get_or_init(Self::from_env)
    }

    pub(crate) fn get() -> &'static Self {
        POOLS.get().expect("profile thread pools not initialized")
    }

    /// Run verifier-side work on the verify pool when it differs from the prove pool.
    pub(crate) fn in_verify<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        #[cfg(feature = "parallel")]
        {
            if let Some(pool) = &self.verify_pool {
                return pool.install(f);
            }
        }
        let _ = self;
        f()
    }

    fn from_env() -> Self {
        let prove_threads = env_thread_count("AKITA_PROFILE_PROVE_THREADS");
        let verify_threads = env_thread_count("AKITA_PROFILE_VERIFY_THREADS");

        #[cfg(feature = "parallel")]
        {
            const PROVE_STACK_SIZE: usize = 64 * 1024 * 1024;

            let prove_resolved = build_global_prove_pool(prove_threads, PROVE_STACK_SIZE);
            let verify_resolved = if verify_threads > 0 {
                verify_threads
            } else {
                prove_resolved
            };
            let verify_pool = if verify_resolved != prove_resolved {
                Some(build_pool(verify_threads, "verify"))
            } else {
                None
            };
            let verify_resolved = verify_pool
                .as_ref()
                .map(rayon::ThreadPool::current_num_threads)
                .unwrap_or(prove_resolved);

            tracing::info!(
                prove_threads = prove_resolved,
                verify_threads = verify_resolved,
                prove_env = prove_threads,
                verify_env = verify_threads,
                separate_verify_pool = verify_pool.is_some(),
                "profile thread pools"
            );
            eprintln!(
                "[profile] prove_threads={prove_resolved} verify_threads={verify_resolved} \
                 (env prove={prove_threads} verify={verify_threads}; 0 = Rayon default)"
            );

            Self { verify_pool }
        }
        #[cfg(not(feature = "parallel"))]
        {
            tracing::info!(
                prove_threads,
                verify_threads,
                "profile thread pools (parallel disabled)"
            );
            eprintln!(
                "[profile] prove_threads={prove_threads} verify_threads={verify_threads} \
                 (parallel disabled)"
            );
            Self {}
        }
    }
}

fn env_thread_count(name: &str) -> usize {
    if let Ok(value) = env::var(name) {
        if let Ok(parsed) = value.parse::<usize>() {
            return parsed;
        }
    }
    env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

#[cfg(feature = "parallel")]
fn build_global_prove_pool(num_threads: usize, stack_size: usize) -> usize {
    let mut builder = rayon::ThreadPoolBuilder::new().stack_size(stack_size);
    if num_threads > 0 {
        builder = builder.num_threads(num_threads);
    }
    builder
        .build_global()
        .unwrap_or_else(|err| panic!("failed to build profile global prove pool: {err}"));
    rayon::current_num_threads()
}

#[cfg(feature = "parallel")]
fn build_pool(num_threads: usize, label: &str) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if num_threads > 0 {
        builder = builder.num_threads(num_threads);
    }
    builder
        .build()
        .unwrap_or_else(|err| panic!("failed to build profile {label} thread pool: {err}"))
}
