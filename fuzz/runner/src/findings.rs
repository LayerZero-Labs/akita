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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("akita-fuzz-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn records_bounded_samples_and_counts_all() {
        let root = scratch("findings");
        let artifact = root.join("crash-1");
        fs::write(&artifact, b"input").unwrap();
        for index in 0..(MAX_SAMPLES + 3) {
            let (meta, new) = record(
                &root.join("findings"),
                Occurrence {
                    id: "panic-x-1",
                    signature: "sig",
                    kind: "panic",
                    lane: "x",
                    target: "x",
                    artifact: Some(&artifact),
                    report: &["line".to_string()],
                    context: json!({"host": "h"}),
                },
            )
            .unwrap();
            assert_eq!(new, index == 0);
            assert_eq!(meta["count"], json!(index + 1));
        }
        let meta: Value = read_json(&root.join("findings/panic-x-1/meta.json")).unwrap();
        assert_eq!(meta["samples"].as_array().unwrap().len(), MAX_SAMPLES);
        assert_eq!(meta["host"], "h");
        assert_eq!(
            fs::read(root.join("findings/panic-x-1/sample-0.input")).unwrap(),
            b"input"
        );
    }

    #[test]
    fn quarantines_the_corpus_copy_by_content_hash() {
        let root = scratch("quarantine");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        fs::write(corpus.join(sha1_hex(b"boom")), b"boom").unwrap();
        let artifact = root.join("crash-2");
        fs::write(&artifact, b"boom").unwrap();
        let moved = quarantine_corpus_copy(Some(&artifact), &corpus, &root.join("q")).unwrap();
        assert!(moved.is_file());
        assert!(!corpus.join(sha1_hex(b"boom")).exists());
        assert!(quarantine_corpus_copy(Some(&artifact), &corpus, &root.join("q")).is_none());
    }
}
