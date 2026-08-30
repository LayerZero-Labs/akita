use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const STACK_BASE_SHA: &str = "e473df62baa6f3491fa867c25a4b6237451737d4";

fn git_output(manifest_dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .args(args)
        .output()
        .ok()?;
    output.status.success().then(|| {
        String::from_utf8_lossy(&output.stdout)
            .trim_end()
            .to_string()
    })
}

fn watch(path: impl AsRef<Path>) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

fn watch_evidence_inputs(repo_root: &Path) {
    watch(repo_root.join("Cargo.toml"));
    watch(repo_root.join("Cargo.lock"));
    if let Ok(crates) = fs::read_dir(repo_root.join("crates")) {
        for entry in crates.flatten() {
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                watch(entry.path());
            }
        }
    }
}

fn is_baseline_backport(status: &str) -> bool {
    status
        .lines()
        .filter(|line| !line.trim().is_empty())
        .all(|line| {
            let path = line.get(3..).unwrap_or_default();
            path.ends_with("crates/akita-planner/Cargo.toml")
                || path.ends_with("crates/akita-planner/build.rs")
                || path.ends_with("crates/akita-planner/examples/quotient_free_catalog_evidence.rs")
        })
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CATALOG_EVIDENCE");
    if env::var_os("CARGO_FEATURE_CATALOG_EVIDENCE").is_none() {
        println!("cargo:rustc-env=AKITA_CATALOG_SOURCE_SHA=disabled");
        println!("cargo:rustc-env=AKITA_CATALOG_SOURCE_STATE=disabled");
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let repo_root = git_output(&manifest_dir, &["rev-parse", "--show-toplevel"])
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.clone());
    watch_evidence_inputs(&repo_root);

    for git_path in ["HEAD", "packed-refs"] {
        if let Some(path) = git_output(&manifest_dir, &["rev-parse", "--git-path", git_path]) {
            watch(path);
        }
    }
    if let Some(symbolic_ref) = git_output(&manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git_output(&manifest_dir, &["rev-parse", "--git-path", &symbolic_ref]) {
            watch(path);
        }
    }

    let source_sha = git_output(&manifest_dir, &["rev-parse", "HEAD"])
        .filter(|sha| sha.len() == 40)
        .unwrap_or_else(|| "unknown".to_string());
    let status = git_output(
        &manifest_dir,
        &["status", "--porcelain", "--untracked-files=all"],
    )
    .unwrap_or_else(|| "git-status-unavailable".to_string());
    let quotient_free = env::var_os("CARGO_FEATURE_QUOTIENT_FREE_EVIDENCE").is_some();
    let source_state = if status.is_empty() {
        "clean"
    } else if !quotient_free && source_sha == STACK_BASE_SHA && is_baseline_backport(&status) {
        "approved-base-backport"
    } else {
        "dirty"
    };
    println!("cargo:rustc-env=AKITA_CATALOG_SOURCE_SHA={source_sha}");
    println!("cargo:rustc-env=AKITA_CATALOG_SOURCE_STATE={source_state}");
}
