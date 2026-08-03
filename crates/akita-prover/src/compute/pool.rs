//! Dedicated compute pool with worker stacks sized for the prover kernels.

/// Worker stack size for [`run_on_compute_pool`]. Thread stacks are lazily
/// committed by the OS, so oversizing costs virtual address space only.
pub const COMPUTE_WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `f` with rayon parallelism on a dedicated pool whose workers have
/// [`COMPUTE_WORKER_STACK_BYTES`] stacks.
///
/// The prover kernels recurse inside rayon parallel iterators (the bridge
/// splitter re-splits whenever a job migrates to a stealing worker) with
/// heavyweight frames — deep chains bottom out in kernels like the setup XOF
/// matrix expansion (`derive_public_matrix_flat`). On rayon's default 2 MiB
/// worker stacks this can overflow nondeterministically, because the depth
/// depends on work-stealing interleavings; the in-tree e2e suite compensates
/// by installing a 256 MiB global pool, but library callers inherit rayon's
/// defaults. Funneling the public scheme entry points through this pool makes
/// the requirement library-owned. Nested calls reuse the pool.
#[cfg(feature = "parallel")]
pub fn run_on_compute_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|index| format!("akita-compute-{index}"))
            .stack_size(COMPUTE_WORKER_STACK_BYTES)
            .build()
            .expect("the akita compute thread pool must build")
    })
    .install(f)
}

/// Serial fallback: runs `f` on the calling thread.
#[cfg(not(feature = "parallel"))]
pub fn run_on_compute_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    f()
}
