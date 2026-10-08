//! Usable CPU and memory, and the enforceable OS memory control.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Budget {
    pub cpus: u64,
    pub memory_mb: u64,
    pub hard_memory: bool,
    pub notes: Vec<String>,
}

fn cgroup_dir() -> Option<PathBuf> {
    let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let line = text.lines().next()?;
    let relative = line.strip_prefix("0::")?;
    let path = PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
    path.is_dir().then_some(path)
}

pub fn usable_cpus() -> (u64, &'static str) {
    let mut cpus = std::thread::available_parallelism().map_or(1, |n| n.get() as u64);
    let mut source = "available_parallelism (affinity-aware)";
    if let Some(cgroup) = cgroup_dir() {
        if let Ok(text) = std::fs::read_to_string(cgroup.join("cpu.max")) {
            let mut parts = text.split_whitespace();
            if let (Some(quota), Some(period)) = (parts.next(), parts.next()) {
                if let (Ok(quota), Ok(period)) = (quota.parse::<u64>(), period.parse::<u64>()) {
                    let limited = (quota / period.max(1)).max(1);
                    if limited < cpus {
                        cpus = limited;
                        source = "cgroup cpu.max";
                    }
                }
            }
        }
    }
    (cpus, source)
}

pub fn available_memory_mb() -> (u64, &'static str) {
    let mut available = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemAvailable:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
                .map(|kb| kb / 1024)
        });
    let mut source = "MemAvailable";
    if available.is_none() {
        // SAFETY: sysconf has no preconditions.
        let (pages, size) = unsafe {
            (
                libc::sysconf(libc::_SC_PHYS_PAGES),
                libc::sysconf(libc::_SC_PAGESIZE),
            )
        };
        if pages > 0 && size > 0 {
            available = Some(pages as u64 * size as u64 / (1 << 20));
            source = "physical memory";
        }
    }
    let mut available = available.unwrap_or(4096);
    if let Some(cgroup) = cgroup_dir() {
        let limit = std::fs::read_to_string(cgroup.join("memory.max")).ok();
        let current = std::fs::read_to_string(cgroup.join("memory.current")).ok();
        if let (Some(limit), Some(current)) = (limit, current) {
            if let (Ok(limit), Ok(current)) =
                (limit.trim().parse::<u64>(), current.trim().parse::<u64>())
            {
                let remaining = limit.saturating_sub(current) / (1 << 20);
                if remaining < available {
                    available = remaining;
                    source = "cgroup memory.max";
                }
            }
        }
    }
    (available, source)
}

/// Whether per-worker cgroup memory limits can be created unprivileged.
pub fn hard_memory_supported() -> bool {
    Command::new("systemd-run")
        .args([
            "--user",
            "--scope",
            "--quiet",
            "-p",
            "MemoryMax=64M",
            "true",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn budget(
    cpus: Option<u64>,
    memory_mb: Option<u64>,
    reserve_cpus: Option<u64>,
    hard_memory: Option<bool>,
) -> Budget {
    let mut notes = Vec::new();
    let (detected, cpu_source) = usable_cpus();
    let cpus = match cpus {
        Some(cpus) => {
            notes.push(format!(
                "cpus: {cpus} (explicit; detected {detected} via {cpu_source})"
            ));
            cpus
        }
        None => {
            let reserve = reserve_cpus.unwrap_or((detected / 16).max(1));
            let cpus = detected.saturating_sub(reserve).max(1);
            notes.push(format!(
                "cpus: {cpus} of {detected} ({cpu_source}), {reserve} reserved"
            ));
            cpus
        }
    };
    let (detected_mb, mem_source) = available_memory_mb();
    let memory_mb = match memory_mb {
        Some(memory) => {
            notes.push(format!(
                "memory: {memory} MiB (explicit; detected {detected_mb} MiB via {mem_source})"
            ));
            memory
        }
        None => {
            let memory = detected_mb * 4 / 5;
            notes.push(format!(
                "memory: {memory} MiB = 80% of {detected_mb} MiB ({mem_source})"
            ));
            memory
        }
    };
    let hard_memory = hard_memory.unwrap_or_else(hard_memory_supported);
    notes.push(if hard_memory {
        "memory enforcement: hard per-worker cgroup MemoryMax (systemd-run --user --scope) plus libFuzzer limits".into()
    } else {
        "memory enforcement: libFuzzer -rss_limit_mb (sampled) and -malloc_limit_mb only; no hard OS limit".into()
    });
    Budget {
        cpus,
        memory_mb,
        hard_memory,
        notes,
    }
}
