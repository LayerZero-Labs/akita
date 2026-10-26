//! Failures of finished jobs: findings, quarantine, and replay verdicts.

use super::{Job, Purpose, Runner};
use crate::findings;
use crate::libfuzzer;
use crate::store::{read_json, write_json};
use serde_json::{json, Value};
use std::path::Path;

/// Whether a finished job's failure is stored as a finding. A replay re-runs
/// a stored finding, so its outcome goes to that finding instead.
pub(super) fn records_finding(
    purpose: Purpose,
    code: i32,
    has_artifact: bool,
    failure_output: bool,
    interrupted: bool,
) -> bool {
    match purpose {
        Purpose::Replay => false,
        _ if has_artifact => true,
        _ => code != 0 && !interrupted && failure_output,
    }
}

/// A job's artifacts directory may hold the only copy of a reproducer until
/// it is stored, so it is kept when recording one was attempted and failed.
pub(super) fn may_remove_artifacts(has_artifact: bool, recorded: Option<bool>) -> bool {
    !has_artifact || recorded != Some(false)
}

impl Runner {
    /// Store a failure as a finding; false if storage failed.
    pub(super) fn record_finding(
        &mut self,
        job: &Job,
        artifact: Option<&Path>,
        tail: &[String],
        elapsed: f64,
    ) -> bool {
        let name = job.lane.name();
        let kind = libfuzzer::classify(tail, artifact);
        let (id, text) = libfuzzer::signature(kind, &job.lane.target, tail);
        let context = json!({
            "host": self.campaign["machine"]["hostname"],
            "campaign_id": self.campaign["campaign_id"],
            "build_id": self.build_id,
            "phase": job.purpose.name(),
        });
        let recorded = findings::record(
            &self.store.findings,
            findings::Occurrence {
                id: &id,
                signature: &text,
                kind,
                lane: &name,
                target: &job.lane.target,
                artifact,
                report: tail,
                context,
            },
        );
        let (meta, new) = match recorded {
            Ok(result) => result,
            Err(error) => {
                self.event(&format!("could not record finding {id}: {error}"));
                return false;
            }
        };
        self.lane_state(&name).findings += 1;
        let moved = findings::quarantine_corpus_copy(
            artifact,
            &self.store.corpus.join(&job.lane.target),
            &self.store.quarantine.join(&job.lane.target),
        );
        self.event(&format!(
            "{}finding {id} (x{}) in {name} after {elapsed:.0}s: {text}{}",
            if new { "NEW " } else { "" },
            meta["count"],
            moved.as_ref().map_or(String::new(), |path| format!(
                "; quarantined corpus input {}",
                path.display()
            ))
        ));
        if new && self.options.replay_new_findings {
            if let Some(input) = meta["samples"][0]["input"].as_str() {
                self.pending_replays.push_back((
                    id.clone(),
                    job.lane.clone(),
                    self.store.findings.join(&id).join(input),
                ));
            }
        }
        // A merge's outcome is handled by its compaction, not by restarts.
        if job.purpose == Purpose::Merge {
            return true;
        }
        self.lane_state(&name).restarts += 1;
        if elapsed < 60.0 && !job.status.inited && moved.is_none() {
            self.backoff(
                &name,
                "failure before initialization with no corpus input to quarantine",
            );
        } else {
            self.lane_state(&name).consecutive_failures = 0;
        }
        true
    }

    pub(super) fn finish_replay(&mut self, job: &Job, code: i32, tail: &[String]) {
        let Some(id) = job.finding_id.clone() else {
            return;
        };
        let meta_path = self.store.findings.join(&id).join("meta.json");
        let Some(mut meta) = read_json::<Value>(&meta_path) else {
            return;
        };
        let verdict = if code == 0 {
            "no (input passed when replayed in a fresh process)".to_string()
        } else {
            let (replay_id, _) =
                libfuzzer::signature(libfuzzer::classify(tail, None), &job.lane.target, tail);
            if replay_id == id {
                "yes".into()
            } else {
                format!("different signature on replay: {replay_id}")
            }
        };
        meta["reproducible"] = json!(verdict);
        let _ = std::fs::write(
            self.store.findings.join(&id).join("replay.txt"),
            tail.join("\n") + "\n",
        );
        let _ = write_json(&meta_path, &meta);
        self.event(&format!("replay of {id}: reproducible={verdict}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failures_are_recorded_for_every_purpose_but_replay() {
        for purpose in [Purpose::Fuzz, Purpose::Baseline, Purpose::Merge] {
            // An artifact is a reproducer even when the process exited 0
            // (a merge survives crashes of its inner process).
            assert!(records_finding(purpose, 0, true, false, false));
            assert!(records_finding(purpose, 1, true, false, true));
            assert!(records_finding(purpose, 1, false, true, false));
            assert!(!records_finding(purpose, 0, false, true, false));
            assert!(!records_finding(purpose, 1, false, false, false));
            assert!(!records_finding(purpose, 1, false, true, true));
        }
        assert!(!records_finding(Purpose::Replay, 1, true, true, false));
    }

    #[test]
    fn unrecorded_reproducers_are_kept() {
        assert!(!may_remove_artifacts(true, Some(false)));
        assert!(may_remove_artifacts(true, Some(true)));
        assert!(may_remove_artifacts(true, None));
        assert!(may_remove_artifacts(false, Some(false)));
        assert!(may_remove_artifacts(false, None));
    }
}
