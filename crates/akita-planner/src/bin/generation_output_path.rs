//! Canonical output-path resolution and checked-tree isolation.

use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

use super::ExplicitRows;

pub(super) fn resolved_output_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|error| format!("read current directory: {error}"))?
            .join(path)
    };
    let mut resolved = PathBuf::new();
    let mut missing = Vec::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let removed = if missing.is_empty() {
                    resolved.pop()
                } else {
                    missing.pop();
                    true
                };
                if !removed {
                    return Err(format!(
                        "output path escapes the filesystem root: {}",
                        path.display()
                    ));
                }
            }
            Component::Normal(name) if missing.is_empty() => {
                let candidate = resolved.join(name);
                if candidate.exists() {
                    resolved = fs::canonicalize(&candidate)
                        .map_err(|error| format!("resolve {}: {error}", candidate.display()))?;
                } else {
                    missing.push(name.to_os_string());
                }
            }
            Component::Normal(name) => missing.push(name.to_os_string()),
            Component::Prefix(_) | Component::RootDir => resolved.push(component.as_os_str()),
        }
    }
    for component in missing {
        resolved.push(component);
    }
    Ok(resolved)
}

pub(super) fn validate_explicit_output_isolation(
    base_dir: &Path,
    explicit_rows: &ExplicitRows,
) -> Result<(), String> {
    if explicit_rows.final_group.is_none() {
        return Ok(());
    }
    let checked_in_generated_dir = resolved_output_path(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/schedules"),
    )?;
    let requested_dir = resolved_output_path(base_dir)?;
    if requested_dir.starts_with(&checked_in_generated_dir) {
        return Err(format!(
            "explicit schedule sweeps must use an isolated output directory outside {}",
            checked_in_generated_dir.display()
        ));
    }
    Ok(())
}
