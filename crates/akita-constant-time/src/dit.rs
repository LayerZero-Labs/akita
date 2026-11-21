//! Checked DIT detection and thread-local register state.
//!
//! The enable sequence follows Apple's DIT guidance, using the baseline
//! `dsb nsh; isb` speculation barrier without requiring FEAT_SB:
//! <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms>.
//! `S3_3_C4_C2_5` names DIT without requiring a crate-wide `+dit` target feature.

use std::arch::asm;
use std::marker::PhantomData;
use std::rc::Rc;

pub(super) struct Dit {
    supported: bool,
}

impl Dit {
    pub(super) fn init() -> Self {
        Self {
            supported: detect_supported(),
        }
    }

    pub(super) fn is_supported(&self) -> bool {
        self.supported
    }

    pub(super) fn enable(&self) -> Guard {
        let was_enabled = self.is_enabled();
        if self.supported {
            // SAFETY: the OS reported FEAT_DIT. The barrier instructions are
            // baseline AArch64. Omitting nomem/readonly gives this asm a compiler
            // memory clobber; the CPU barriers order execution before any jobs.
            unsafe {
                asm!(
                    "msr S3_3_C4_C2_5, {enabled}",
                    "dsb nsh",
                    "isb",
                    enabled = in(reg) 1_u64 << 24,
                    options(nostack, preserves_flags),
                );
            }
        }
        Guard {
            restore_disabled: self.supported && !was_enabled,
            thread_local: PhantomData,
        }
    }

    pub(super) fn is_enabled(&self) -> bool {
        if !self.supported {
            return false;
        }
        let value: u64;
        // SAFETY: the OS reported FEAT_DIT; this only reads the current thread.
        unsafe {
            asm!(
                "mrs {value}, S3_3_C4_C2_5",
                value = out(reg) value,
                options(nostack, preserves_flags),
            );
        }
        value & (1 << 24) != 0
    }
}

// The guard must be dropped on the thread whose register it changed.
pub(super) struct Guard {
    restore_disabled: bool,
    thread_local: PhantomData<Rc<()>>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.restore_disabled {
            // SAFETY: creation checked FEAT_DIT, and the guard is !Send/!Sync.
            // A memory clobber prevents moving secret work after restoration.
            unsafe {
                asm!(
                    "msr S3_3_C4_C2_5, {disabled}",
                    disabled = in(reg) 0_u64,
                    options(nostack, preserves_flags),
                );
            }
        }
    }
}

#[cfg(target_vendor = "apple")]
fn detect_supported() -> bool {
    let name = c"hw.optional.arm.FEAT_DIT";
    // Exercise a real failed sysctl in isolated test processes. Production has
    // no environment override and always queries the fixed FEAT_DIT key.
    #[cfg(test)]
    let name = {
        DETECTION_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DETECTION_THREAD.get_or_init(|| std::thread::current().id());
        if std::env::var_os("AKITA_CONSTANT_TIME_FAILED_QUERY_CHILD").is_some() {
            c"hw.optional.arm.AKITA_NONEXISTENT_FEATURE"
        } else {
            name
        }
    };
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of_val(&value);
    // SAFETY: name is NUL terminated; the output and size pointers are valid
    // for their declared sizes. Null newp and zero newlen make this read-only.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut libc::c_int).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    result == 0 && size == std::mem::size_of_val(&value) && value == 1
}

#[cfg(not(target_vendor = "apple"))]
fn detect_supported() -> bool {
    // std's OS-specific detector returns false when the capability is unknown.
    std::arch::is_aarch64_feature_detected!("dit")
}

#[cfg(all(test, target_vendor = "apple"))]
pub(super) static DETECTION_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, target_vendor = "apple"))]
pub(super) static DETECTION_THREAD: std::sync::OnceLock<std::thread::ThreadId> =
    std::sync::OnceLock::new();
