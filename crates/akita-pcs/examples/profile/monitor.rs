use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use sysinfo::{get_current_pid, ProcessRefreshKind, ProcessesToUpdate, System};

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[must_use = "the resource monitor stops when dropped"]
pub(super) struct ResourceMonitor {
    handle: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl ResourceMonitor {
    pub(super) fn start(interval: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("akita-resource-monitor".to_string())
            .spawn(move || sample_resources(thread_stop, interval))
            .map_err(|error| {
                tracing::warn!(%error, "failed to start profile resource monitor");
                error
            })
            .ok();
        Self { handle, stop }
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn sample_resources(stop: Arc<AtomicBool>, interval: Duration) {
    let Ok(pid) = get_current_pid() else {
        tracing::warn!("resource monitor could not resolve the current process id");
        return;
    };
    let refresh_kind = ProcessRefreshKind::nothing().with_cpu().with_memory();
    let mut system = System::new();
    system.refresh_cpu_all();
    system.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, refresh_kind);
    let Some(process) = system.process(pid) else {
        tracing::warn!("resource monitor could not read the current process");
        return;
    };
    let mut previous_cpu_ms = process.accumulated_cpu_time();
    let mut previous_sample_at = Instant::now();
    let mut system_cpu_percent = system.global_cpu_usage();
    let mut previous_system_sample_at = Instant::now();

    while !stop.load(Ordering::Acquire) {
        thread::sleep(interval);
        if previous_system_sample_at.elapsed() >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
            system.refresh_cpu_all();
            system_cpu_percent = system.global_cpu_usage();
            previous_system_sample_at = Instant::now();
        }
        system.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, refresh_kind);
        if let Some(process) = system.process(pid) {
            let sampled_at = Instant::now();
            let cpu_ms = process.accumulated_cpu_time();
            let elapsed_ms = sampled_at.duration_since(previous_sample_at).as_secs_f64() * 1_000.0;
            let process_effective_cores = if elapsed_ms > 0.0 {
                cpu_ms.saturating_sub(previous_cpu_ms) as f64 / elapsed_ms
            } else {
                0.0
            };
            let process_cpu_percent = process_effective_cores * 100.0;
            let rss_gib = process.memory() as f64 / BYTES_PER_GIB;
            let virtual_memory_gib = process.virtual_memory() as f64 / BYTES_PER_GIB;
            let logical_cpus = system.cpus().len();
            let system_effective_cores =
                f64::from(system_cpu_percent) / 100.0 * logical_cpus as f64;
            tracing::trace!(
                target: "akita_profile_resources",
                counter_process_cpu_percent = process_cpu_percent,
                counter_process_effective_cores = process_effective_cores,
                counter_system_cpu_percent = system_cpu_percent,
                counter_system_effective_cores = system_effective_cores,
                counter_rss_gib = rss_gib,
                counter_virtual_memory_gib = virtual_memory_gib,
                counter_logical_cpus = logical_cpus,
                "profile_resource_sample"
            );
            previous_cpu_ms = cpu_ms;
            previous_sample_at = sampled_at;
        }
    }
}

#[cfg(unix)]
pub(super) fn peak_rss_bytes() -> Option<u64> {
    // SAFETY: `getrusage` initializes the complete output value on success.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) != 0 {
            return None;
        }
        usage
    };
    let raw = usage.ru_maxrss as u64;
    Some(if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    })
}

#[cfg(not(unix))]
pub(super) fn peak_rss_bytes() -> Option<u64> {
    None
}
