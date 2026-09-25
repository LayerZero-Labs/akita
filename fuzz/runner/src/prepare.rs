//! `akita-fuzz prepare`: build instrumented targets and package a distribution.
//!
//! Run from the source tree (`cargo run --release -p akita-fuzz-runner --
//! prepare --out DIR`). A plain `cargo build --release` produces no fuzzing
//! instrumentation; this command runs `cargo fuzz build` (nightly, libFuzzer,
//! AddressSanitizer, SanitizerCoverage, debug assertions, overflow checks)
//! and copies the results with everything the campaign needs offline.

use crate::registry;
use crate::store::{now, write_json};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Prepare {
    pub dist: PathBuf,
    pub sanitizer: String,
    pub sequential: bool,
    pub skip_build: bool,
}

fn fuzz_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("fuzz directory")
}

fn output(command: &mut Command) -> Result<String, String> {
    let result = command.output().map_err(|e| format!("{command:?}: {e}"))?;
    if !result.status.success() {
        return Err(format!(
            "{command:?} failed:\n{}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&result.stdout).trim().to_string())
}

fn host_triple(fuzz: &Path) -> Result<String, String> {
    let verbose = output(Command::new("rustc").arg("-vV").current_dir(fuzz))?;
    verbose
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_string)
        .ok_or_else(|| "rustc -vV reported no host".into())
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(from)
        .map_err(|e| format!("read {}: {e}", from.display()))?
        .flatten()
    {
        let target = to.join(entry.file_name());
        if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)
                .map_err(|e| format!("copy {}: {e}", entry.path().display()))?;
        }
    }
    Ok(())
}

fn files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files(root, &path, out);
        } else if let Ok(relative) = path.strip_prefix(root) {
            out.push(relative.to_path_buf());
        }
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

pub fn run(options: Prepare) -> Result<(), String> {
    let fuzz = fuzz_root();
    let repo = fuzz.join("..");
    let lanes = registry::load(&fuzz.join("campaign/targets.toml"))?;
    let registered: BTreeSet<String> = registry::targets(&lanes).into_iter().collect();
    let library: BTreeSet<String> = akita_fuzz::targets::ALL
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    let binaries: BTreeSet<String> = output(
        Command::new("cargo")
            .args(["fuzz", "list"])
            .current_dir(&fuzz),
    )?
    .lines()
    .map(str::to_string)
    .collect();
    let per_target: BTreeSet<String> = binaries
        .iter()
        .filter(|name| name.as_str() != crate::libfuzzer::BINARY)
        .cloned()
        .collect();
    if registered != library
        || registered != per_target
        || !binaries.contains(crate::libfuzzer::BINARY)
    {
        return Err(format!(
            "target sets disagree:\n  campaign/targets.toml: {registered:?}\n  akita_fuzz::targets::ALL: {library:?}\n  cargo fuzz list without fuzz_all: {per_target:?}"
        ));
    }

    let triple = host_triple(&fuzz)?;
    let mut build = Command::new("cargo");
    build
        .args([
            "fuzz",
            "build",
            "--release",
            "--debug-assertions",
            "--sanitizer",
            &options.sanitizer,
            "--target",
            &triple,
            crate::libfuzzer::BINARY,
        ])
        .current_dir(&fuzz);
    if options.sequential {
        build.args(["--no-default-features"]);
    }
    if !options.skip_build {
        println!("+ {build:?}");
        let status = build
            .status()
            .map_err(|e| format!("cargo fuzz build: {e}"))?;
        if !status.success() {
            return Err("cargo fuzz build failed".into());
        }
    }

    let dist = &options.dist;
    if dist.exists() {
        std::fs::remove_dir_all(dist).map_err(|e| format!("clear {}: {e}", dist.display()))?;
    }
    for sub in ["bin", "campaign", "artifacts/schedules", "seeds"] {
        std::fs::create_dir_all(dist.join(sub)).map_err(|e| e.to_string())?;
    }
    let built = fuzz.join("target").join(&triple).join("release");
    let binary = crate::libfuzzer::BINARY;
    std::fs::copy(built.join(binary), dist.join("bin").join(binary))
        .map_err(|e| format!("copy instrumented {binary}: {e} (was `cargo fuzz build` run?)"))?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    std::fs::copy(&exe, dist.join("akita-fuzz")).map_err(|e| format!("copy runner: {e}"))?;
    std::fs::copy(
        fuzz.join("campaign/targets.toml"),
        dist.join("campaign/targets.toml"),
    )
    .map_err(|e| e.to_string())?;
    std::fs::copy(fuzz.join("README.md"), dist.join("README.md"))
        .map_err(|e| format!("copy README: {e}"))?;
    copy_dir(
        &repo.join("artifacts/schedules"),
        &dist.join("artifacts/schedules"),
    )?;
    let symbolizer = ["llvm-symbolizer"]
        .iter()
        .find_map(|name| {
            output(Command::new("sh").args(["-c", &format!("command -v {name}")])).ok()
        })
        .filter(|path| !path.is_empty());
    if let Some(path) = &symbolizer {
        std::fs::copy(path, dist.join("bin/llvm-symbolizer"))
            .map_err(|e| format!("copy symbolizer: {e}"))?;
    }

    // Seeds use the packaged artifacts so their case selectors match the
    // catalogs the campaign will load.
    std::env::set_var(
        akita_fuzz::env::ARTIFACTS_ENV,
        dist.join("artifacts/schedules"),
    );
    println!("generating seeds");
    crate::tools::seeds(&dist.join("seeds"));

    let git = |args: &[&str]| {
        output(Command::new("git").args(args).current_dir(&repo)).unwrap_or_default()
    };
    let dirty = !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    let mut manifest_files = Vec::new();
    files(dist, dist, &mut manifest_files);
    manifest_files.sort();
    let mut manifest = String::new();
    for file in &manifest_files {
        manifest.push_str(&format!(
            "{}  {}\n",
            sha256_file(&dist.join(file))?,
            file.display()
        ));
    }
    std::fs::write(dist.join("MANIFEST.sha256"), &manifest).map_err(|e| e.to_string())?;
    let build_id: String = Sha256::digest(manifest.as_bytes())
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect();
    let info = json!({
        "build_id": build_id,
        "created": now(),
        "git_commit": git(&["rev-parse", "HEAD"]),
        "git_branch": git(&["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": dirty,
        "rustc": output(Command::new("rustc").arg("-vV").current_dir(&fuzz)).unwrap_or_default(),
        "cargo_fuzz": output(Command::new("cargo").args(["fuzz", "--version"]).current_dir(&fuzz)).unwrap_or_default(),
        "target_triple": triple,
        "sanitizer": options.sanitizer,
        "flags": "cargo fuzz build --release --debug-assertions (profile: debug=1, overflow-checks)",
        "features": if options.sequential { "sequential (no parallel)" } else { "parallel" },
        "cargo_lock_sha256": sha256_file(&fuzz.join("Cargo.lock"))?,
        "symbolizer": symbolizer,
        "targets": registered,
        "build_host": crate::store::machine_identity(),
    });
    write_json(&dist.join("BUILD-INFO.json"), &info).map_err(|e| e.to_string())?;
    println!(
        "prepared {} (build {build_id}{})",
        dist.display(),
        if dirty { ", dirty tree" } else { "" }
    );
    if symbolizer.is_none() {
        println!("note: no llvm-symbolizer found; Rust panics are still symbolized, sanitizer frames will be raw addresses");
    }
    Ok(())
}
