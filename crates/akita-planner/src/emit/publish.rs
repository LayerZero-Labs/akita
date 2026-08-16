use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use super::GeneratedOutput;

static PUBLISH_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct StagedOutput {
    destination: PathBuf,
    staged: PathBuf,
    backup: Option<PathBuf>,
    published: bool,
}

fn transaction_sibling_path(destination: &Path, label: &str, nonce: u64, index: usize) -> PathBuf {
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("generated-output");
    destination.with_file_name(format!(
        ".{file_name}.akita-{label}-{}-{nonce}-{index}",
        std::process::id()
    ))
}

fn cleanup_staged_outputs(staged: &[StagedOutput]) {
    for output in staged {
        let _ = fs::remove_file(&output.staged);
    }
}

fn rollback_published_outputs(staged: &mut [StagedOutput]) -> Result<(), String> {
    let mut failures = Vec::new();
    for output in staged.iter_mut().rev() {
        if output.published {
            if let Err(error) = fs::remove_file(&output.destination) {
                let recovery = output
                    .backup
                    .as_ref()
                    .map(|path| format!("; original preserved at {}", path.display()))
                    .unwrap_or_default();
                failures.push(format!(
                    "remove {}: {error}{recovery}",
                    output.destination.display()
                ));
                continue;
            }
            output.published = false;
        }
        if let Some(backup) = output.backup.take() {
            if let Err(error) = fs::rename(&backup, &output.destination) {
                failures.push(format!(
                    "restore {} from {}: {error}; original preserved at {}",
                    output.destination.display(),
                    backup.display(),
                    backup.display(),
                ));
                output.backup = Some(backup);
            }
        }
    }
    cleanup_staged_outputs(staged);
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

/// Stage, format, and publish one complete generated-output batch.
///
/// All stage files are written before any destination is replaced. A publish
/// failure restores every destination already replaced by this batch.
pub fn publish_generated_outputs(outputs: Vec<GeneratedOutput>) -> Result<Vec<PathBuf>, String> {
    let nonce = PUBLISH_NONCE.fetch_add(1, Ordering::Relaxed);
    let mut destinations = std::collections::BTreeSet::new();
    let mut staged = Vec::with_capacity(outputs.len());
    for (index, output) in outputs.into_iter().enumerate() {
        if !destinations.insert(output.destination.clone()) {
            cleanup_staged_outputs(&staged);
            return Err(format!(
                "generated output batch contains duplicate destination {}",
                output.destination.display()
            ));
        }
        if output.destination.is_dir() {
            cleanup_staged_outputs(&staged);
            return Err(format!(
                "generated output destination is a directory: {}",
                output.destination.display()
            ));
        }
        let parent = match output.destination.parent() {
            Some(parent) => parent,
            None => {
                cleanup_staged_outputs(&staged);
                return Err(format!(
                    "generated output has no parent directory: {}",
                    output.destination.display()
                ));
            }
        };
        let staged_path = transaction_sibling_path(&output.destination, "stage", nonce, index);
        let mut staged_file = match fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staged_path)
        {
            Ok(file) => file,
            Err(error) => {
                cleanup_staged_outputs(&staged);
                return Err(format!("stage in {}: {error}", parent.display()));
            }
        };
        if let Err(error) = staged_file
            .write_all(output.body.as_bytes())
            .and_then(|()| staged_file.sync_all())
        {
            let _ = fs::remove_file(&staged_path);
            cleanup_staged_outputs(&staged);
            return Err(format!("stage {}: {error}", output.destination.display()));
        }
        staged.push(StagedOutput {
            destination: output.destination,
            staged: staged_path,
            backup: None,
            published: false,
        });
    }
    if staged.is_empty() {
        return Ok(Vec::new());
    }

    let mut rustfmt = Command::new("rustfmt");
    rustfmt.args(["--edition", "2021"]);
    // The batch may introduce both a module declaration and its source file.
    // Neither destination exists until the atomic publish below, so rustfmt
    // must not traverse children while formatting the staged module index.
    rustfmt.args(["--config", "skip_children=true"]);
    rustfmt.args(staged.iter().map(|output| &output.staged));
    let status = match rustfmt.status() {
        Ok(status) => status,
        Err(error) => {
            cleanup_staged_outputs(&staged);
            return Err(format!("format staged generated outputs: {error}"));
        }
    };
    if !status.success() {
        cleanup_staged_outputs(&staged);
        return Err(format!(
            "format staged generated outputs failed with {status}"
        ));
    }
    for output in &staged {
        let sync_result = fs::OpenOptions::new()
            .write(true)
            .open(&output.staged)
            .and_then(|file| file.sync_all());
        if let Err(error) = sync_result {
            cleanup_staged_outputs(&staged);
            return Err(format!(
                "sync staged output {}: {error}",
                output.destination.display()
            ));
        }
    }

    let backup_paths = staged
        .iter()
        .enumerate()
        .map(|(index, output)| {
            output
                .destination
                .exists()
                .then(|| transaction_sibling_path(&output.destination, "backup", nonce, index))
        })
        .collect::<Vec<_>>();
    if let Some(stale) = backup_paths.iter().flatten().find(|path| path.exists()) {
        cleanup_staged_outputs(&staged);
        return Err(format!(
            "stale generated-output backup: {}",
            stale.display()
        ));
    }

    for index in 0..staged.len() {
        if let Some(backup) = backup_paths[index].clone() {
            if let Err(error) = fs::rename(&staged[index].destination, &backup) {
                let destination = staged[index].destination.display().to_string();
                let rollback = rollback_published_outputs(&mut staged);
                return Err(match rollback {
                    Ok(()) => format!("backup {destination}: {error}"),
                    Err(rollback) => {
                        format!("backup {destination}: {error}; rollback failed: {rollback}")
                    }
                });
            }
            staged[index].backup = Some(backup);
        }
        if let Err(error) = fs::rename(&staged[index].staged, &staged[index].destination) {
            let destination = staged[index].destination.display().to_string();
            let rollback = rollback_published_outputs(&mut staged);
            return Err(match rollback {
                Ok(()) => format!("publish {destination}: {error}"),
                Err(rollback) => {
                    format!("publish {destination}: {error}; rollback failed: {rollback}")
                }
            });
        }
        staged[index].published = true;
    }

    let destinations = staged
        .iter()
        .map(|output| output.destination.clone())
        .collect();
    for output in &mut staged {
        if let Some(backup) = output.backup.take() {
            let _ = fs::remove_file(backup);
        }
    }
    Ok(destinations)
}

#[cfg(test)]
mod tests {
    use super::{publish_generated_outputs, rollback_published_outputs, StagedOutput};
    use crate::emit::GeneratedOutput;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_NONCE: AtomicU64 = AtomicU64::new(0);

    fn test_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "akita-generated-publish-{label}-{}-{}",
            std::process::id(),
            TEST_NONCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn staging_failure_leaves_every_existing_output_untouched() {
        let dir = test_dir("stage-failure");
        fs::create_dir_all(&dir).expect("create test directory");
        let first = dir.join("first.rs");
        fs::write(&first, "pub const FIRST: usize = 1;\n").expect("write first fixture");
        let blocked_parent = dir.join("not-a-directory");
        fs::write(&blocked_parent, "blocking file\n").expect("write blocking fixture");

        let error = publish_generated_outputs(vec![
            GeneratedOutput {
                destination: first.clone(),
                body: "pub const FIRST: usize = 2;\n".to_string(),
            },
            GeneratedOutput {
                destination: blocked_parent.join("second.rs"),
                body: "pub const SECOND: usize = 2;\n".to_string(),
            },
        ])
        .expect_err("second stage must fail");
        assert!(error.contains("stage in"));
        assert_eq!(
            fs::read_to_string(&first).expect("read first fixture"),
            "pub const FIRST: usize = 1;\n"
        );
        assert!(
            fs::read_dir(&dir)
                .expect("read test directory")
                .all(|entry| !entry
                    .expect("directory entry")
                    .file_name()
                    .to_string_lossy()
                    .contains("akita-stage")),
            "failed staging must clean earlier stage files"
        );
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn rollback_restores_every_replaced_output() {
        let dir = test_dir("rollback");
        fs::create_dir_all(&dir).expect("create test directory");
        let mut staged = Vec::new();
        for index in 0..3 {
            let destination = dir.join(format!("output-{index}.rs"));
            let backup = dir.join(format!("backup-{index}.rs"));
            fs::write(&destination, format!("pub const NEW_{index}: usize = 2;\n"))
                .expect("write replacement fixture");
            fs::write(&backup, format!("pub const OLD_{index}: usize = 1;\n"))
                .expect("write backup fixture");
            staged.push(StagedOutput {
                destination,
                staged: dir.join(format!("absent-stage-{index}.rs")),
                backup: Some(backup),
                published: true,
            });
        }

        rollback_published_outputs(&mut staged).expect("rollback published outputs");
        for (index, output) in staged.iter().enumerate() {
            assert!(
                fs::read_to_string(&output.destination)
                    .expect("read restored output")
                    .contains(&format!("OLD_{index}")),
                "rollback must restore output {index}"
            );
        }
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn failed_rollback_preserves_the_original_backup() {
        let dir = test_dir("rollback-preserves-backup");
        fs::create_dir_all(&dir).expect("create test directory");
        let destination = dir.join("published-directory");
        let backup = dir.join("original.rs");
        fs::create_dir(&destination).expect("create removal-resistant destination");
        fs::write(&backup, "pub const ORIGINAL: usize = 1;\n").expect("write backup");
        let mut staged = vec![StagedOutput {
            destination,
            staged: dir.join("absent-stage.rs"),
            backup: Some(backup.clone()),
            published: true,
        }];

        let error = rollback_published_outputs(&mut staged).expect_err("rollback must fail");
        assert!(error.contains(&backup.display().to_string()));
        assert_eq!(
            fs::read_to_string(&backup).expect("read preserved backup"),
            "pub const ORIGINAL: usize = 1;\n"
        );
        assert_eq!(staged[0].backup.as_ref(), Some(&backup));
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn complete_batch_publishes_every_output_together() {
        let dir = test_dir("success");
        fs::create_dir_all(&dir).expect("create test directory");
        let family = dir.join("family.rs");
        let registry = dir.join("family_multi_chunk.rs");
        let wiring = dir.join("mod.rs");
        for path in [&family, &registry, &wiring] {
            fs::write(path, "pub const OLD: usize = 1;\n").expect("write old fixture");
        }

        let published = publish_generated_outputs(vec![
            GeneratedOutput {
                destination: family.clone(),
                body: "pub const FAMILY: usize = 2;\n".to_string(),
            },
            GeneratedOutput {
                destination: registry.clone(),
                body: "pub const REGISTRY: usize = 3;\n".to_string(),
            },
            GeneratedOutput {
                destination: wiring.clone(),
                body: "pub const WIRING: usize = 4;\n".to_string(),
            },
        ])
        .expect("publish complete batch");

        assert_eq!(
            published,
            vec![family.clone(), registry.clone(), wiring.clone()]
        );
        assert!(fs::read_to_string(family)
            .expect("read family")
            .contains("FAMILY"));
        assert!(fs::read_to_string(registry)
            .expect("read registry")
            .contains("REGISTRY"));
        assert!(fs::read_to_string(wiring)
            .expect("read wiring")
            .contains("WIRING"));
        assert!(
            fs::read_dir(&dir)
                .expect("read test directory")
                .all(|entry| !entry
                    .expect("directory entry")
                    .file_name()
                    .to_string_lossy()
                    .contains(".akita-")),
            "successful publishing must clean transaction files"
        );
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn batch_can_introduce_a_module_and_its_source_together() {
        let dir = test_dir("new-module");
        fs::create_dir_all(&dir).expect("create test directory");
        let family = dir.join("new_family.rs");
        let wiring = dir.join("mod.rs");
        fs::write(&wiring, "// old module index\n").expect("write old module index");

        publish_generated_outputs(vec![
            GeneratedOutput {
                destination: family.clone(),
                body: "pub const VALUE:usize=1;\n".to_string(),
            },
            GeneratedOutput {
                destination: wiring.clone(),
                body: "pub mod new_family;\n".to_string(),
            },
        ])
        .expect("publish new module batch");

        assert_eq!(
            fs::read_to_string(&family).expect("read new family"),
            "pub const VALUE: usize = 1;\n"
        );
        assert_eq!(
            fs::read_to_string(&wiring).expect("read module index"),
            "pub mod new_family;\n"
        );
        fs::remove_dir_all(dir).expect("remove test directory");
    }
}
