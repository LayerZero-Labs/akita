//! Shared GPU test support: device serialization, a hang watchdog, and
//! deterministic inputs.

#![allow(dead_code)]

use std::fs::File;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Mutex, MutexGuard, Once};
use std::thread;
use std::time::Duration;

use akita_metal::AkitaMetal;

/// A GPU hang can take the machine down with it; abort the test process
/// instead of waiting on it.
const WATCHDOG: Duration = Duration::from_secs(120);

static IN_PROCESS: Mutex<()> = Mutex::new(());
static SCALAR_CPU: Once = Once::new();

/// Exclusive use of the GPU for one test: serialized within the process by a
/// mutex and across processes by a file lock, with a watchdog that aborts
/// the process if the test outlives [`WATCHDOG`].
pub(crate) struct GpuTest {
    pub(crate) metal: AkitaMetal,
    _disarm: mpsc::Sender<()>,
    _lock: File,
    _guard: MutexGuard<'static, ()>,
}

/// Selects the CPU's portable scalar transforms, whose words the device
/// kernels reproduce exactly. Must run before any CPU parameter set is built.
pub(crate) fn scalar_cpu() {
    SCALAR_CPU.call_once(|| {
        // Runs once, before any CPU parameter set reads the variable.
        std::env::set_var("AKITA_SCALAR_NTT", "1");
    });
}

pub(crate) fn gpu() -> GpuTest {
    scalar_cpu();
    let guard = IN_PROCESS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let lock = File::create(std::env::temp_dir().join("akita-metal-gpu.lock"))
        .expect("create the GPU lock file");
    lock.lock().expect("lock the GPU");
    let (disarm, armed) = mpsc::channel::<()>();
    thread::spawn(move || {
        if armed.recv_timeout(WATCHDOG) == Err(RecvTimeoutError::Timeout) {
            eprintln!("GPU test exceeded {WATCHDOG:?}; aborting");
            std::process::abort();
        }
    });
    let metal = AkitaMetal::new().expect("an Apple GPU with the Akita kernels");
    GpuTest {
        metal,
        _disarm: disarm,
        _lock: lock,
        _guard: guard,
    }
}

/// SplitMix64: a fixed-seed generator, so failures reproduce.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn next_i32(&mut self) -> i32 {
        self.next_u64() as i32
    }

    /// Uniform in `(-bound, bound)`.
    pub(crate) fn next_i32_below(&mut self, bound: i32) -> i32 {
        let span = 2 * u64::from(bound.unsigned_abs()) - 1;
        (self.next_u64() % span) as i32 - bound + 1
    }
}
