//! Deduplicated, bounded storage of findings; originals are never deleted.

use crate::libfuzzer::sha1_hex;
use crate::store::{now, read_json, write_json};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_SAMPLES: usize = 5;

pub struct Occurrence<'a> {
    pub id: &'a str,
    pub signature: &'a str,
    pub kind: &'a str,
    pub lane: &'a str,
    pub target: &'a str,
    pub artifact: Option<&'a Path>,
    pub report: &'a [String],
    pub context: Value,
}

/// Store one occurrence; returns `(meta, is_new_signature)`.
pub fn record(findings: &Path, occurrence: Occurrence<'_>) -> std::io::Result<(Value, bool)> {
    let directory = findings.join(occurrence.id);
    fs::create_dir_all(&directory)?;
    let meta_path = directory.join("meta.json");
    let existing: Option<Value> = read_json(&meta_path);
    let new = existing.is_none();
    let mut meta = existing.unwrap_or_else(|| {
        let mut meta = json!({
            "id": occurrence.id,
            "signature": occurrence.signature,
            "kind": occurrence.kind,
            "target": occurrence.target,
            "lanes": [],
            "first_seen": now(),
            "count": 0,
            "samples": [],
            "reproducible": null,
        });
        if let (Some(meta), Some(context)) = (meta.as_object_mut(), occurrence.context.as_object())
        {
            meta.extend(context.clone());
        }
        meta
    });
    meta["count"] = json!(meta["count"].as_u64().unwrap_or(0) + 1);
    meta["last_seen"] = json!(now());
    let lanes = meta["lanes"].as_array_mut().expect("lanes array");
    if !lanes.iter().any(|lane| lane == occurrence.lane) {
        lanes.push(json!(occurrence.lane));
    }
    let samples = meta["samples"].as_array_mut().expect("samples array");
    if samples.len() < MAX_SAMPLES {
        let index = samples.len();
        let report = format!("sample-{index}.txt");
        let mut sample =
            json!({"at": now(), "lane": occurrence.lane, "input": null, "report": report});
        if let Some(artifact) = occurrence.artifact.filter(|path| path.is_file()) {
            let data = fs::read(artifact)?;
            let name = format!("sample-{index}.input");
            fs::write(directory.join(&name), &data)?;
            sample["input"] = json!(name);
            sample["sha1"] = json!(sha1_hex(&data));
            sample["artifact"] = json!(artifact.file_name().and_then(|n| n.to_str()));
        }
        fs::write(directory.join(&report), occurrence.report.join("\n") + "\n")?;
        samples.push(sample);
    }
    write_json(&meta_path, &meta)?;
    Ok((meta, new))
}

/// Move a crashing input out of the corpus so restarts do not loop on it.
/// libFuzzer names corpus files by the SHA-1 of their contents.
pub fn quarantine_corpus_copy(
    artifact: Option<&Path>,
    corpus: &Path,
    quarantine: &Path,
) -> Option<PathBuf> {
    let data = fs::read(artifact?).ok()?;
    let digest = sha1_hex(&data);
    let candidate = corpus.join(&digest);
    if !candidate.is_file() {
        return None;
    }
    fs::create_dir_all(quarantine).ok()?;
    let destination = quarantine.join(&digest);
    fs::rename(&candidate, &destination).ok()?;
    Some(destination)
}

pub fn summarize(findings: &Path) -> Vec<Value> {
    let mut out: Vec<Value> = fs::read_dir(findings)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| read_json(&entry.path().join("meta.json")))
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a: &Value, b: &Value| a["id"].as_str().cmp(&b["id"].as_str()));
    out
}
