//! Persistent campaign directory: lock, identity, state, and bounded logs.

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("state"),
        std::process::id()
    ));
    let mut file = File::create(&tmp)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(tmp, path)
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

pub fn hostname() -> String {
    let mut buffer = [0u8; 256];
    // SAFETY: the buffer is valid for its full length.
    let ok = unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) } == 0;
    if ok {
        let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
        String::from_utf8_lossy(&buffer[..end]).into_owned()
    } else {
        "unknown".into()
    }
}

pub fn machine_identity() -> Value {
    let machine_id = ["/etc/machine-id", "/var/lib/dbus/machine-id"]
        .iter()
        .find_map(|path| fs::read_to_string(path).ok())
        .map(|id| id.trim().to_string())
        .unwrap_or_default();
    let cpu_model = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, model)| model.trim().to_string())
        })
        .unwrap_or_default();
    json!({
        "hostname": hostname(),
        "machine_id": machine_id,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu_model": cpu_model,
    })
}

/// Layout of one campaign output directory.
pub struct Store {
    pub root: PathBuf,
    pub corpus: PathBuf,
    pub findings: PathBuf,
    pub quarantine: PathBuf,
    pub logs: PathBuf,
    pub stats: PathBuf,
    pub exports: PathBuf,
    pub state_path: PathBuf,
    pub live_path: PathBuf,
    pub campaign_path: PathBuf,
    lock: Option<File>,
}

impl Store {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            corpus: root.join("corpus"),
            findings: root.join("findings"),
            quarantine: root.join("quarantine"),
            logs: root.join("logs"),
            stats: root.join("stats"),
            exports: root.join("exports"),
            state_path: root.join("state.json"),
            live_path: root.join("live.json"),
            campaign_path: root.join("campaign.json"),
            lock: None,
        }
    }

    pub fn ensure(&self) -> std::io::Result<()> {
        for dir in [
            &self.corpus,
            &self.findings,
            &self.quarantine,
            &self.logs,
            &self.stats,
            &self.exports,
        ] {
            fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    /// Exclusive advisory lock; a second runner on the same directory fails.
    pub fn lock(&mut self) -> Result<(), String> {
        fs::create_dir_all(&self.root)
            .map_err(|e| format!("create {}: {e}", self.root.display()))?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.root.join(".lock"))
            .map_err(|e| format!("open lock: {e}"))?;
        if file.try_lock().is_err() {
            let mut holder = String::new();
            let _ = file.read_to_string(&mut holder);
            return Err(format!(
                "{} is in use by another runner ({})",
                self.root.display(),
                holder.trim()
            ));
        }
        file.set_len(0).map_err(|e| e.to_string())?;
        file.rewind().map_err(|e| e.to_string())?;
        writeln!(
            file,
            "pid={} host={} since={}",
            std::process::id(),
            hostname(),
            now()
        )
        .map_err(|e| e.to_string())?;
        self.lock = Some(file);
        Ok(())
    }

    pub fn is_locked(&self) -> bool {
        match File::open(self.root.join(".lock")) {
            Ok(file) => match file.try_lock_shared() {
                Ok(()) => {
                    let _ = file.unlock();
                    false
                }
                Err(_) => true,
            },
            Err(_) => false,
        }
    }

    /// Create or resume the campaign identity; refuse silent machine changes.
    pub fn campaign(&self, build_id: &str, adopt: bool) -> Result<Value, String> {
        let identity = machine_identity();
        let Some(mut existing) = read_json::<Value>(&self.campaign_path) else {
            let mut random = [0u8; 6];
            File::open("/dev/urandom")
                .and_then(|mut file| file.read_exact(&mut random))
                .map_err(|e| format!("read /dev/urandom: {e}"))?;
            let id: String = random.iter().map(|b| format!("{b:02x}")).collect();
            let campaign = json!({
                "campaign_id": id,
                "created": now(),
                "machine": identity,
                "builds": [build_id],
            });
            write_json(&self.campaign_path, &campaign).map_err(|e| e.to_string())?;
            return Ok(campaign);
        };
        let previous = existing["machine"].clone();
        if previous["machine_id"] != identity["machine_id"]
            || previous["hostname"] != identity["hostname"]
        {
            if !adopt {
                return Err(format!(
                    "{} was created on {} ({}); pass --adopt to continue it here, or use a fresh --output",
                    self.root.display(),
                    previous["hostname"],
                    previous["machine_id"]
                ));
            }
            if !existing["adopted_by"].is_array() {
                existing["adopted_by"] = json!([]);
            }
            existing["adopted_by"]
                .as_array_mut()
                .expect("array")
                .push(json!({"at": now(), "machine": identity.clone()}));
            existing["machine"] = identity;
        }
        let builds = existing["builds"]
            .as_array_mut()
            .ok_or("campaign.json lacks builds")?;
        if !builds.iter().any(|b| b == build_id) {
            builds.push(json!(build_id));
        }
        write_json(&self.campaign_path, &existing).map_err(|e| e.to_string())?;
        Ok(existing)
    }
}

/// Append-only text log capped at `max_bytes` with one rotated backup.
pub struct RotatingLog {
    path: PathBuf,
    file: File,
    written: u64,
    max_bytes: u64,
}

impl RotatingLog {
    pub fn open(path: &Path, max_bytes: u64) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let written = file.metadata().map_or(0, |m| m.len());
        Ok(Self {
            path: path.to_path_buf(),
            file,
            written,
            max_bytes,
        })
    }

    pub fn line(&mut self, line: &str) {
        let _ = writeln!(self.file, "{line}");
        self.written += line.len() as u64 + 1;
        if self.written >= self.max_bytes {
            let backup = self.path.with_extension("log.1");
            let _ = fs::rename(&self.path, backup);
            if let Ok(file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
            {
                self.file = file;
                self.written = 0;
            }
        }
    }
}

pub fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => directory_bytes(&entry.path()),
            Ok(_) => entry.metadata().map_or(0, |m| m.len()),
            Err(_) => 0,
        })
        .sum()
}

pub fn count_files(path: &Path) -> u64 {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
                .count() as u64
        })
        .unwrap_or(0)
}
