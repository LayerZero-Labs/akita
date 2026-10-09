//! Corpus compaction: libFuzzer `-merge=1` jobs that keep a minimal subset
//! of a target's corpus and archive the rest.

use super::{Job, Purpose, Runner};
use crate::registry::Lane;
use crate::store::{count_files, now};
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One corpus compaction of a target: a `-merge=1` job per lane of the
/// target, run one after another into the same merged set, so inputs any
/// variant's coverage needs are kept.
pub(super) struct Compaction {
    /// The lane whose state records the compaction schedule.
    lane: Lane,
    /// Corpus file names the compaction started from.
    snapshot: Vec<String>,
    /// Lanes still to merge after the running job.
    pending: VecDeque<Lane>,
    started: Instant,
}

fn corpus_names(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
                .filter_map(|entry| entry.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default()
}

impl Runner {
    /// Private workspace of a target's compaction: `snapshot/` (the corpus
    /// it started from) and `merged/` (the minimal set so far).
    pub(super) fn compaction_dir(&self, target: &str) -> PathBuf {
        self.store.root.join("compaction").join(target)
    }

    /// Snapshot `lane`'s target corpus and merge it under every lane of the
    /// target, `lane` first.
    fn start_compaction(&mut self, lane: &Lane) -> std::io::Result<()> {
        let workspace = self.compaction_dir(&lane.target);
        if workspace.exists() {
            std::fs::remove_dir_all(&workspace)?;
        }
        // Merge a private snapshot: running jobs replace and remove corpus
        // files (`-reduce_inputs`), which aborts a merge that reads the live
        // directory.
        let snapshot_dir = workspace.join("snapshot");
        std::fs::create_dir_all(&snapshot_dir)?;
        std::fs::create_dir_all(workspace.join("merged"))?;
        let corpus = self.store.corpus.join(&lane.target);
        let snapshot = corpus_names(&corpus);
        for file in &snapshot {
            let source = corpus.join(file);
            if std::fs::hard_link(&source, snapshot_dir.join(file)).is_err() {
                let _ = std::fs::copy(&source, snapshot_dir.join(file));
            }
        }
        let pending = self
            .lanes
            .iter()
            .filter(|other| other.target == lane.target && other.name() != lane.name())
            .cloned()
            .collect();
        let compaction = Compaction {
            lane: lane.clone(),
            snapshot,
            pending,
            started: Instant::now(),
        };
        let result = self.spawn(lane, Purpose::Merge, &[]);
        match result {
            Ok(id) => {
                self.jobs.get_mut(&id).expect("job").compaction = Some(compaction);
                Ok(())
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(&workspace);
                Err(error)
            }
        }
    }

    /// After each lane's merge, start the next lane's; after the last, keep
    /// the merged set: move snapshot inputs it dropped to
    /// `corpus-archive/<target>/`. Inputs fuzz jobs added meanwhile stay. Any
    /// failure abandons the compaction and archives nothing.
    pub(super) fn finish_merge(&mut self, job: &mut Job, code: i32) {
        let Some(mut compaction) = job.compaction.take() else {
            return;
        };
        let name = compaction.lane.name();
        let target = compaction.lane.target.clone();
        let workspace = self.compaction_dir(&target);
        let interrupted = if code == 0 {
            self.stopping && !compaction.pending.is_empty()
        } else {
            job.terminated_at.is_some() && !job.watchdog_stopped
        };
        if interrupted {
            // Stopped by the runner at shutdown, not a merge failure:
            // compaction becomes due again on the next check.
            let _ = std::fs::remove_dir_all(&workspace);
            self.event(&format!("compaction of {target} interrupted"));
            return;
        }
        let failure = if code != 0 {
            Some(format!("exit {code} under {}", job.lane.name()))
        } else if let Some(next) = compaction.pending.pop_front() {
            match self.spawn(&next, Purpose::Merge, &[]) {
                Ok(id) => {
                    self.event(&format!(
                        "compacting {target}: merging under {}",
                        next.name()
                    ));
                    self.jobs.get_mut(&id).expect("job").compaction = Some(compaction);
                    return;
                }
                Err(error) => Some(format!("{} failed to start: {error}", next.name())),
            }
        } else {
            None
        };
        if let Some(reason) = failure {
            let _ = std::fs::remove_dir_all(&workspace);
            self.lane_state(&name).next_compaction_at = now() + 6 * 3600;
            self.event(&format!(
                "compaction of {target} failed ({reason}); retrying in 6h"
            ));
            return;
        }
        let kept: HashSet<String> = corpus_names(&workspace.join("merged"))
            .into_iter()
            .collect();
        let corpus = self.store.corpus.join(&target);
        let archive = self.store.root.join("corpus-archive").join(&target);
        let _ = std::fs::create_dir_all(&archive);
        let mut archived = 0u64;
        for file in &compaction.snapshot {
            if !kept.contains(file)
                && std::fs::rename(corpus.join(file), archive.join(file)).is_ok()
            {
                archived += 1;
            }
        }
        let _ = std::fs::remove_dir_all(&workspace);
        let remaining = count_files(&corpus);
        let state = self.lane_state(&name);
        state.compacted_corpus_files = remaining;
        state.corpus_files = remaining;
        state.compactions += 1;
        state.next_compaction_at = now() + 3600;
        let total = compaction.snapshot.len() as u64;
        self.event(&format!(
            "compacted {target}: {total} -> {} inputs in {:.0}s ({archived} moved to corpus-archive)",
            total - archived,
            compaction.started.elapsed().as_secs_f64()
        ));
    }

    /// Start a corpus merge for any target whose corpus doubled since its
    /// last compaction (and has at least 200 inputs), one merge per target.
    pub(super) fn schedule_compactions(&mut self) {
        let time = now();
        let merging: HashSet<String> = self
            .jobs
            .values()
            .filter(|job| job.purpose == Purpose::Merge)
            .map(|job| job.lane.target.clone())
            .collect();
        let mut seen = HashSet::new();
        for lane in self.lanes.clone() {
            if self.stopping || !seen.insert(lane.target.clone()) || merging.contains(&lane.target)
            {
                continue;
            }
            let state = self
                .state
                .lanes
                .get(&lane.name())
                .cloned()
                .unwrap_or_default();
            let awaiting_baseline = self.baseline_running.contains(&lane.name())
                || self
                    .baseline_pending
                    .iter()
                    .any(|pending| pending.target == lane.target);
            let due = state.corpus_files >= 200
                && state.corpus_files >= 2 * state.compacted_corpus_files
                && state.next_compaction_at <= time;
            if due && !awaiting_baseline && self.fits(&lane) {
                match self.start_compaction(&lane) {
                    Ok(()) => self.event(&format!(
                        "compacting {} ({} inputs)",
                        lane.target, state.corpus_files
                    )),
                    Err(error) => self.event(&format!(
                        "compaction of {} failed to start: {error}",
                        lane.target
                    )),
                }
            }
        }
    }
}
