//! libFuzzer command lines, output parsing, and failure signatures.

use crate::registry::Lane;
use sha1::{Digest, Sha1};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct Limits<'a> {
    pub lane: &'a Lane,
    pub timeout_s: u64,
}

pub fn base_args(binary: &Path, limits: Limits<'_>, artifacts: &Path) -> Vec<String> {
    let lane = limits.lane;
    vec![
        binary.display().to_string(),
        format!("-artifact_prefix={}/", artifacts.display()),
        format!("-timeout={}", limits.timeout_s),
        format!("-rss_limit_mb={}", lane.rss_limit_mb),
        format!("-malloc_limit_mb={}", lane.malloc_limit_mb),
    ]
}

/// A fuzzing job over `corpus` for `seconds` (0 = run the corpus once).
pub fn fuzz_args(
    binary: &Path,
    lane: &Lane,
    corpus: &[&Path],
    artifacts: &Path,
    seconds: u64,
    runs_zero: bool,
    extra: &[String],
) -> Vec<String> {
    let mut args = base_args(
        binary,
        Limits {
            lane,
            timeout_s: lane.timeout_s,
        },
        artifacts,
    );
    args.extend([
        format!("-max_len={}", lane.max_len),
        "-print_final_stats=1".into(),
        "-reload=1".into(),
        "-close_fd_mask=1".into(),
    ]);
    if runs_zero {
        args.push("-runs=0".into());
    } else {
        args.push(format!("-max_total_time={seconds}"));
    }
    if lane.value_profile {
        args.push("-use_value_profile=1".into());
    }
    args.extend(extra.iter().cloned());
    args.extend(corpus.iter().map(|dir| dir.display().to_string()));
    args
}

pub fn environment(
    lane: &Lane,
    artifacts_dir: &Path,
    stats_file: &Path,
    symbolizer: Option<&Path>,
) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.insert("AKITA_FUZZ_THREADS".into(), lane.threads.to_string());
    env.insert("RAYON_NUM_THREADS".into(), lane.threads.to_string());
    env.insert(
        "AKITA_FUZZ_ARTIFACTS".into(),
        artifacts_dir.display().to_string(),
    );
    env.insert(
        "AKITA_FUZZ_STATS_FILE".into(),
        stats_file.display().to_string(),
    );
    env.insert("RUST_BACKTRACE".into(), "1".into());
    // Rust statics are reachable, so leak checking only adds noise; aborting
    // makes libFuzzer record every sanitizer report as an artifact.
    env.insert(
        "ASAN_OPTIONS".into(),
        "abort_on_error=1:symbolize=1:detect_leaks=0:allocator_may_return_null=0:quarantine_size_mb=64:malloc_context_size=20".into(),
    );
    if let Some(symbolizer) = symbolizer {
        env.insert(
            "ASAN_SYMBOLIZER_PATH".into(),
            symbolizer.display().to_string(),
        );
    }
    env.extend(lane.env.clone());
    env
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Status {
    pub execs: u64,
    pub cov: u64,
    pub ft: u64,
    pub corpus: u64,
    pub exec_per_s: u64,
    pub rss_mb: u64,
    pub inited: bool,
}

fn field(line: &str, key: &str) -> Option<u64> {
    let rest = &line[line.find(key)? + key.len()..];
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

pub fn parse_status(line: &str, status: &mut Status) {
    if line.contains("INITED") {
        status.inited = true;
    }
    if let Some(rest) = line.strip_prefix('#') {
        if let Some(execs) = rest.split_whitespace().next().and_then(|n| n.parse().ok()) {
            if let (Some(cov), Some(ft)) = (field(line, "cov:"), field(line, "ft:")) {
                status.execs = execs;
                status.cov = cov;
                status.ft = ft;
                status.corpus = field(line, "corp:").unwrap_or(status.corpus);
                status.exec_per_s = field(line, "exec/s:").unwrap_or(status.exec_per_s);
                status.rss_mb = field(line, "rss:").unwrap_or(status.rss_mb);
            }
        }
    }
    if let Some(rest) = line.strip_prefix("stat::") {
        if let Some((key, value)) = rest.split_once(':') {
            if let Ok(value) = value.trim().parse::<u64>() {
                match key {
                    "number_of_executed_units" => status.execs = status.execs.max(value),
                    "peak_rss_mb" => status.rss_mb = status.rss_mb.max(value),
                    _ => {}
                }
            }
        }
    }
}

pub fn artifact_path(line: &str) -> Option<PathBuf> {
    let rest = &line[line.find("Test unit written to ")? + "Test unit written to ".len()..];
    rest.split_whitespace().next().map(PathBuf::from)
}

pub fn classify(lines: &[String], artifact: Option<&Path>) -> &'static str {
    let name = artifact
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let has = |needle: &str| lines.iter().any(|line| line.contains(needle));
    if name.starts_with("timeout-") || has("ERROR: libFuzzer: timeout") {
        "timeout"
    } else if name.starts_with("oom-") || has("ERROR: libFuzzer: out-of-memory") {
        "oom"
    } else if name.starts_with("leak-") || has("LeakSanitizer") {
        "leak"
    } else if name.starts_with("slow-unit-") {
        "slow"
    } else if has("ERROR: AddressSanitizer") {
        "asan"
    } else if has("panicked at") {
        "panic"
    } else if has("deadly signal") {
        "signal"
    } else {
        "crash"
    }
}

/// Keep a message's shape while dropping values that vary per input.
fn normalize(message: &str) -> String {
    let mut out = String::new();
    let mut digits = 0;
    let mut chars = message.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '0' && chars.peek() == Some(&'x') {
            chars.next();
            while chars.peek().is_some_and(char::is_ascii_hexdigit) {
                chars.next();
            }
            out.push_str("0x_");
            continue;
        }
        if c.is_ascii_digit() {
            digits += 1;
            if digits == 3 {
                // Collapse long numbers.
                out.truncate(out.len() - 2);
                out.push('N');
            } else if digits < 3 {
                out.push(c);
            }
            continue;
        }
        digits = 0;
        out.push(c);
    }
    out.chars().take(160).collect()
}

fn panic_location(line: &str) -> Option<String> {
    let line = line.trim();
    let rest = &line[line.find("panicked at ")? + "panicked at ".len()..];
    let rest = rest.trim_end_matches(':');
    // `file:line:col` -> `file:line`
    let mut parts = rest.rsplitn(2, ':');
    parts.next()?;
    parts.next().map(str::to_string)
}

fn frame_function(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let function = if let Some(rest) = trimmed.strip_prefix('#') {
        rest.split(" in ").nth(1)?.split_whitespace().next()?
    } else {
        let (index, rest) = trimmed.split_once(": ")?;
        index.parse::<u32>().ok()?;
        rest.split_whitespace().next()?
    };
    let function = match function.rfind("::h") {
        Some(at) if function.len() - at == 19 => &function[..at],
        _ => function,
    };
    (function.contains("akita") && !function.starts_with("akita_fuzz"))
        .then(|| function.to_string())
}

/// Stable `(id, text)` deduplicating findings of one root cause.
pub fn signature(kind: &str, target: &str, lines: &[String]) -> (String, String) {
    let mut key = None;
    for (index, line) in lines.iter().enumerate() {
        if let Some(location) = panic_location(line) {
            let message = lines.get(index + 1).map_or("", |m| m.trim());
            key = Some(format!("panic at {location}: {}", normalize(message)));
            break;
        }
    }
    if key.is_none() {
        key = lines
            .iter()
            .find(|line| line.starts_with("SUMMARY: AddressSanitizer"))
            .map(|line| normalize(line.split(" in ").next().unwrap_or(line)));
    }
    let key = key.unwrap_or_else(|| {
        let frames: Vec<String> = lines
            .iter()
            .filter_map(|line| frame_function(line))
            .take(3)
            .collect();
        if frames.is_empty() {
            "no stack".into()
        } else {
            frames.join(" <- ")
        }
    });
    let text = format!("{kind} in {target}: {key}");
    let digest = Sha1::digest(text.as_bytes());
    let short: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    (format!("{kind}-{target}-{short}"), text)
}

pub fn sha1_hex(data: &[u8]) -> String {
    Sha1::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
