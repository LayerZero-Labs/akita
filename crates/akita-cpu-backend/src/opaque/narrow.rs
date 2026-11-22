//! Runs a proof's small fold levels on a few of the pool's workers.
//!
//! The sumchecks of a small level are a long run of short passes. Split
//! across every worker of a wide pool, each pass waits for sleeping threads
//! to wake and take a few microseconds of work; on four to six threads the
//! same workers stay awake from one pass to the next. On an 18-thread pool
//! the stage-1 and stage-2 sumchecks of a `2^17`- to `2^19`-entry level take
//! about half again as long as on six threads.
//!
//! [`NarrowedPool`] parks the surplus workers of the calling thread's pool
//! until it is dropped. It is engaged between two stages, when the calling
//! thread is the pool's only source of work and every other worker is idle.

#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(feature = "parallel")]
use std::sync::{Arc, Condvar, Mutex, PoisonError};
#[cfg(feature = "parallel")]
use std::time::{Duration, Instant};

/// Largest stage domain whose level runs narrowed.
pub(crate) const NARROW_DOMAIN: usize = 1 << 20;
/// Workers a narrowed pool keeps, the calling thread among them.
#[cfg(feature = "parallel")]
const NARROW_THREADS: usize = 6;
/// How long the surplus workers get to park before narrowing is abandoned.
#[cfg(feature = "parallel")]
const PARK_TIMEOUT: Duration = Duration::from_millis(5);

/// Set while a [`NarrowedPool`] lives: a proof's stages run on one pool,
/// and a second guard would wait for workers the first already holds.
#[cfg(feature = "parallel")]
static NARROWED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "parallel")]
struct Parking {
    released: Mutex<bool>,
    wake: Condvar,
    arrived: AtomicUsize,
}

#[cfg(feature = "parallel")]
impl Parking {
    fn release(&self) {
        *self.released.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.wake.notify_all();
    }
}

/// Holds the surplus workers of one pool parked.
pub(crate) struct NarrowedPool {
    #[cfg(feature = "parallel")]
    parking: Arc<Parking>,
}

impl NarrowedPool {
    /// Parks all but [`NARROW_THREADS`] workers of the calling worker's pool.
    /// `None` when the caller is not a pool worker, the pool is already that
    /// narrow, another guard is alive, or a worker to be parked is busy.
    ///
    /// A parked worker blocks inside a pool job. A worker that is waiting for
    /// a `join` it started would take that job on top of the join, and keep
    /// the join's caller waiting until the release: the caller must be the
    /// pool's only source of work and have no parallel pass in flight, so
    /// that every other worker is idle and parks from the top of its stack.
    /// The call returns once they all have; a worker that has not arrived by
    /// then is doing something else, and narrowing is abandoned.
    pub(crate) fn engage() -> Option<Self> {
        #[cfg(not(feature = "parallel"))]
        return None;
        #[cfg(feature = "parallel")]
        {
            let driver = rayon::current_thread_index()?;
            let surplus = rayon::current_num_threads().checked_sub(NARROW_THREADS)?;
            if surplus == 0 || NARROWED.swap(true, Ordering::AcqRel) {
                return None;
            }
            let parking = Arc::new(Parking {
                released: Mutex::new(false),
                wake: Condvar::new(),
                arrived: AtomicUsize::new(0),
            });
            let shared = Arc::clone(&parking);
            rayon::spawn_broadcast(move |context| {
                let index = context.index();
                // The caller and the first workers after it stay.
                let rank = index - usize::from(index > driver);
                if index == driver || rank < NARROW_THREADS - 1 {
                    return;
                }
                let mut released = shared
                    .released
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                let _ = shared.arrived.fetch_add(1, Ordering::Release);
                while !*released {
                    released = shared
                        .wake
                        .wait(released)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            });
            let deadline = Instant::now() + PARK_TIMEOUT;
            while parking.arrived.load(Ordering::Acquire) < surplus {
                if Instant::now() >= deadline {
                    parking.release();
                    NARROWED.store(false, Ordering::Release);
                    return None;
                }
                std::thread::yield_now();
            }
            Some(Self { parking })
        }
    }
}

impl Drop for NarrowedPool {
    fn drop(&mut self) {
        #[cfg(feature = "parallel")]
        {
            self.parking.release();
            NARROWED.store(false, Ordering::Release);
        }
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::collections::HashSet;

    fn workers_used(pool: &rayon::ThreadPool) -> usize {
        pool.install(|| {
            (0..4096)
                .into_par_iter()
                .with_max_len(1)
                .map(|_| {
                    std::thread::sleep(Duration::from_micros(50));
                    rayon::current_thread_index()
                })
                .collect::<HashSet<_>>()
                .len()
        })
    }

    /// One test: the guard is process-wide.
    #[test]
    fn a_narrowed_pool_runs_on_few_workers_and_widens_again() {
        let narrow = rayon::ThreadPoolBuilder::new()
            .num_threads(NARROW_THREADS)
            .build()
            .unwrap();
        assert!(narrow.install(NarrowedPool::engage).is_none());
        assert!(NarrowedPool::engage().is_none());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NARROW_THREADS + 6)
            .build()
            .unwrap();
        let narrowed = pool.install(NarrowedPool::engage).unwrap();
        assert!(pool.install(NarrowedPool::engage).is_none());
        assert!(workers_used(&pool) <= NARROW_THREADS);
        drop(narrowed);
        assert!(workers_used(&pool) > NARROW_THREADS);
    }
}
