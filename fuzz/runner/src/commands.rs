//! Commands that work while a campaign runs, plus triage helpers.

use crate::findings;
use crate::libfuzzer;
use crate::registry::Lane;
use crate::runner::{symbolizer, State};
use crate::store::{directory_bytes, now, read_json, Store};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

fn duration(seconds: f64) -> String {
    let seconds = seconds as u64;
    let (days, hours, minutes) = (seconds / 86400, seconds % 86400 / 3600, seconds % 3600 / 60);
    if days > 0 {
        format!("{days}d{hours:02}h{minutes:02}m")
    } else {
        format!("{hours}h{minutes:02}m")
    }
}

pub fn status(store: &Store, lanes: &[Lane], as_json: bool) -> i32 {
    let state: State = read_json(&store.state_path).unwrap_or_default();
    let live: Option<Value> = if store.is_locked() {
        read_json(&store.live_path)
    } else {
        None
    };
    let campaign: Value = read_json(&store.campaign_path).unwrap_or(json!({}));
    let all_findings = findings::summarize(&store.findings);
    let disk = directory_bytes(&store.root);
    if as_json {
        let report = json!({
            "campaign": campaign,
            "running": live.is_some(),
            "live": live,
            "lanes": state.lanes,
            "findings": all_findings,
            "disk_bytes": disk,
        });
        println!("{}", serde_json::to_string_pretty(&report).expect("json"));
        return 0;
    }
    println!(
        "campaign {} on {}  output {}",
        campaign["campaign_id"].as_str().unwrap_or("?"),
        campaign["machine"]["hostname"].as_str().unwrap_or("?"),
        store.root.display()
    );
    match &live {
        Some(live) => println!(
            "RUNNING pid {} for {}; budget {} cpus / {} MiB; hard memory limit: {}",
            live["pid"],
            duration((now() - live["started"].as_u64().unwrap_or(now())) as f64),
            live["budget"]["cpus"],
            live["budget"]["memory_mb"],
            if live["budget"]["hard_memory"] == true {
                "yes"
            } else {
                "no"
            }
        ),
        None => println!("not running"),
    }
    println!("disk usage {:.1} MiB\n", disk as f64 / (1 << 20) as f64);
    let jobs: Vec<Value> = live
        .as_ref()
        .and_then(|l| l["jobs"].as_array().cloned())
        .unwrap_or_default();
    let header = format!(
        "{:32} {:>10} {:>12} {:>8} {:>7} {:>8} {:>7} {:>5} {:>5}  baseline",
        "lane", "cpu-time", "execs", "exec/s", "cov", "ft", "corpus", "find", "fail"
    );
    println!("{header}\n{}", "-".repeat(header.len()));
    for lane in lanes {
        let name = lane.name();
        let entry = state.lanes.get(&name).cloned().unwrap_or_default();
        let running: Vec<&Value> = jobs
            .iter()
            .filter(|job| job["lane"] == name.as_str())
            .collect();
        let live_execs: u64 = running
            .iter()
            .filter(|job| job["purpose"] == "fuzz")
            .map(|job| job["status"]["execs"].as_u64().unwrap_or(0))
            .sum();
        let eps: u64 = running
            .iter()
            .map(|job| job["status"]["exec_per_s"].as_u64().unwrap_or(0))
            .sum();
        let cov = running
            .iter()
            .map(|job| job["status"]["cov"].as_u64().unwrap_or(0))
            .fold(entry.cov, u64::max);
        let ft = running
            .iter()
            .map(|job| job["status"]["ft"].as_u64().unwrap_or(0))
            .fold(entry.ft, u64::max);
        let baseline = match &entry.baseline {
            None => "-",
            Some(b) if b["ok"] == true => "ok",
            Some(_) => "FAILED",
        };
        let backoff = if entry.backoff_until > now() {
            format!("  backoff {}s", entry.backoff_until - now())
        } else {
            String::new()
        };
        println!(
            "{name:32} {:>10} {:>12} {eps:>8} {cov:>7} {ft:>8} {:>7} {:>5} {:>5}  {baseline}{backoff}",
            duration(entry.cpu_seconds),
            entry.execs + live_execs,
            entry.corpus_files,
            entry.findings,
            entry.startup_failures + entry.infra_failures,
        );
    }
    println!();
    if all_findings.is_empty() {
        println!("no findings");
    }
    for meta in &all_findings {
        println!(
            "[{}] {}  x{}  reproducible={}  phase={}\n    {}",
            meta["kind"].as_str().unwrap_or("?"),
            meta["id"].as_str().unwrap_or("?"),
            meta["count"],
            meta["reproducible"],
            meta["phase"].as_str().unwrap_or("?"),
            meta["signature"].as_str().unwrap_or("?")
        );
    }
    let reach: Vec<(&String, _)> = state
        .lanes
        .iter()
        .filter_map(|(name, lane)| lane.harness.get("counters").map(|c| (name, c)))
        .collect();
    if !reach.is_empty() {
        println!("\nharness reach counters (completed jobs):");
        for (name, counters) in reach {
            let text: Vec<String> = counters
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect();
            println!("  {name}: {}", text.join(", "));
        }
    }
    0
}

pub fn export(
    store: &Store,
    destination: Option<&Path>,
    with_logs: bool,
) -> Result<PathBuf, String> {
    let campaign: Value = read_json(&store.campaign_path).unwrap_or(json!({}));
    let host = campaign["machine"]["hostname"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let id = campaign["campaign_id"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let name = format!("akita-fuzz-{host}-{id}-{}.tar.gz", now());
    let path = match destination {
        Some(path) if path.to_string_lossy().ends_with(".tar.gz") => path.to_path_buf(),
        Some(dir) => dir.join(name),
        None => store.exports.join(name),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let file =
        std::fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut archive = tar::Builder::new(flate2::write::GzEncoder::new(
        file,
        flate2::Compression::default(),
    ));
    archive.follow_symlinks(false);
    let prefix = format!("{host}-{id}");
    let mut members = vec![
        "campaign.json",
        "state.json",
        "corpus",
        "findings",
        "quarantine",
        "events.log",
    ];
    if with_logs {
        members.push("logs");
    }
    for member in members {
        let source = store.root.join(member);
        // Campaign files are written atomically or appended, so a snapshot
        // taken while the campaign runs is consistent per file.
        let result = if source.is_dir() {
            archive.append_dir_all(format!("{prefix}/{member}"), &source)
        } else if source.is_file() {
            archive.append_path_with_name(&source, format!("{prefix}/{member}"))
        } else {
            Ok(())
        };
        result.map_err(|e| format!("archive {member}: {e}"))?;
    }
    archive
        .into_inner()
        .and_then(|gz| gz.finish())
        .map_err(|e| e.to_string())?;
    Ok(path)
}

fn lane_for<'a>(lanes: &'a [Lane], spec: &str) -> Result<&'a Lane, String> {
    let (target, variant) = match spec.split_once('@') {
        Some((target, variant)) => (target, Some(variant)),
        None => (spec, None),
    };
    lanes
        .iter()
        .find(|lane| {
            lane.target == target && (variant.is_none() || lane.variant.as_deref() == variant)
        })
        .ok_or_else(|| format!("unknown target {spec}"))
}

/// A finding id (its stored samples) or `TARGET[@VARIANT]` plus input files.
fn resolve<'a>(
    store: &Store,
    lanes: &'a [Lane],
    spec: &str,
    inputs: &[PathBuf],
) -> Result<(&'a Lane, Vec<PathBuf>, Option<PathBuf>), String> {
    let directory = store.findings.join(spec);
    if let Some(meta) = read_json::<Value>(&directory.join("meta.json")) {
        let lane = lane_for(lanes, meta["lanes"][0].as_str().unwrap_or(""))?;
        let paths: Vec<PathBuf> = meta["samples"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|sample| sample["input"].as_str())
            .map(|input| directory.join(input))
            .collect();
        if paths.is_empty() {
            return Err(format!(
                "finding {spec} has no stored input; see its sample reports"
            ));
        }
        return Ok((lane, paths, Some(directory)));
    }
    if inputs.is_empty() {
        return Err(format!(
            "{spec} is not a finding id; reproduce TARGET needs input files"
        ));
    }
    Ok((lane_for(lanes, spec)?, inputs.to_vec(), None))
}

fn env_for(dist: &Path, lane: &Lane) -> Vec<(String, String)> {
    let mut env = libfuzzer::environment(
        lane,
        &dist.join("artifacts/schedules"),
        Path::new("/dev/null"),
        symbolizer(dist).as_deref(),
    );
    env.insert("RUST_BACKTRACE".into(), "full".into());
    env.into_iter().collect()
}

pub fn reproduce(
    dist: &Path,
    store: &Store,
    lanes: &[Lane],
    spec: &str,
    inputs: &[PathBuf],
) -> Result<i32, String> {
    let (lane, paths, _) = resolve(store, lanes, spec, inputs)?;
    let scratch = std::env::temp_dir().join(format!("akita-fuzz-reproduce-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    let mut args = libfuzzer::base_args(
        &dist.join("bin").join(libfuzzer::BINARY),
        libfuzzer::Limits {
            lane,
            timeout_s: lane.timeout_s * 2,
        },
        &scratch,
    );
    args.extend(paths.iter().map(|path| path.display().to_string()));
    let env = env_for(dist, lane);
    let shown: Vec<String> = lane.env.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!("+ {} {}", shown.join(" "), args.join(" "));
    let status = Command::new(&args[0])
        .args(&args[1..])
        .envs(env)
        .status()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(status.code().unwrap_or(1))
}

pub fn minimize(
    dist: &Path,
    store: &Store,
    lanes: &[Lane],
    finding: &str,
    seconds: u64,
) -> Result<i32, String> {
    let (lane, paths, directory) = resolve(store, lanes, finding, &[])?;
    let directory = directory.expect("findings resolve to their directory");
    let work = directory.join("minimize-work");
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let output = directory.join(format!("minimized-{}.input", now()));
    let mut args = libfuzzer::base_args(
        &dist.join("bin").join(libfuzzer::BINARY),
        libfuzzer::Limits {
            lane,
            timeout_s: lane.timeout_s,
        },
        &work,
    );
    args.extend([
        "-minimize_crash=1".to_string(),
        format!("-max_total_time={seconds}"),
        format!("-exact_artifact_path={}", output.display()),
        paths[0].display().to_string(),
    ]);
    println!("+ {}", args.join(" "));
    let status = Command::new(&args[0])
        .args(&args[1..])
        .envs(env_for(dist, lane))
        .current_dir(&work)
        .status()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&work);
    if output.is_file() {
        println!(
            "minimized input: {} (original kept: {})",
            output.display(),
            paths[0].display()
        );
        Ok(0)
    } else {
        println!(
            "no smaller crashing input (exit {:?}); original kept: {}",
            status.code(),
            paths[0].display()
        );
        Ok(1)
    }
}

/// Static checks of a prepared distribution.
pub fn validate_dist(dist: &Path, lanes: &[Lane]) -> Vec<String> {
    let mut problems = Vec::new();
    if read_json::<Value>(&dist.join("BUILD-INFO.json")).is_none() {
        problems.push("BUILD-INFO.json is missing; run `akita-fuzz prepare`".into());
    }
    match std::fs::read_to_string(dist.join("MANIFEST.sha256")) {
        Ok(manifest) => {
            for line in manifest.lines() {
                let Some((digest, name)) = line.split_once("  ") else {
                    continue;
                };
                match std::fs::read(dist.join(name)) {
                    Ok(bytes) => {
                        let actual: String = Sha256::digest(&bytes)
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect();
                        if actual != digest {
                            problems.push(format!("checksum mismatch: {name}"));
                        }
                    }
                    Err(_) => problems.push(format!("missing {name}")),
                }
            }
        }
        Err(_) => problems.push("MANIFEST.sha256 is missing".into()),
    }
    if !dist.join("bin").join(libfuzzer::BINARY).is_file() {
        problems.push(format!(
            "missing instrumented binary bin/{}",
            libfuzzer::BINARY
        ));
    }
    for target in crate::registry::targets(lanes) {
        if !dist.join("seeds").join(&target).is_dir() {
            problems.push(format!("missing seeds/{target}"));
        }
    }
    let schedules = std::fs::read_dir(dist.join("artifacts/schedules"))
        .map(|e| e.flatten().count())
        .unwrap_or(0);
    if schedules == 0 {
        problems.push("missing artifacts/schedules/*.aks".into());
    }
    problems
}
