//! Canonical target registry (`campaign/targets.toml`) and scheduling lanes.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

const KINDS: [&str; 3] = ["primitive", "end_to_end", "boundary"];

/// One schedulable unit: a target, or one variant of it.
#[derive(Clone, Debug)]
pub struct Lane {
    pub target: String,
    pub variant: Option<String>,
    pub kind: String,
    pub description: String,
    pub weight: f64,
    pub timeout_s: u64,
    pub rss_limit_mb: u64,
    pub malloc_limit_mb: u64,
    pub max_len: u64,
    pub threads: u64,
    pub value_profile: bool,
    pub env: BTreeMap<String, String>,
}

impl Lane {
    pub fn name(&self) -> String {
        match &self.variant {
            Some(variant) => format!("{}@{variant}", self.target),
            None => self.target.clone(),
        }
    }
}

#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
struct Spec {
    kind: Option<String>,
    description: Option<String>,
    weight: Option<f64>,
    timeout_s: Option<u64>,
    rss_limit_mb: Option<u64>,
    malloc_limit_mb: Option<u64>,
    max_len: Option<u64>,
    threads: Option<u64>,
    value_profile: Option<bool>,
    variants: Option<Vec<Variant>>,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Variant {
    name: String,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    defaults: Spec,
    target: BTreeMap<String, Spec>,
}

pub fn load(path: &Path) -> Result<Vec<Lane>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let file: File = toml::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let d = &file.defaults;
    let mut lanes = Vec::new();
    for (target, spec) in &file.target {
        let kind = spec.kind.clone().or(d.kind.clone()).unwrap_or_default();
        if !KINDS.contains(&kind.as_str()) {
            return Err(format!(
                "target {target}: kind must be one of {KINDS:?}, got {kind:?}"
            ));
        }
        let pick = |a: Option<u64>, b: Option<u64>, name: &str| {
            a.or(b)
                .ok_or_else(|| format!("target {target}: missing {name}"))
        };
        let variants: Vec<Option<Variant>> = match &spec.variants {
            Some(variants) if !variants.is_empty() => variants.iter().cloned().map(Some).collect(),
            _ => vec![None],
        };
        let weight = spec.weight.or(d.weight).unwrap_or(1.0) / variants.len() as f64;
        for variant in &variants {
            lanes.push(Lane {
                target: target.clone(),
                variant: variant.as_ref().map(|v| v.name.clone()),
                kind: kind.clone(),
                description: spec.description.clone().unwrap_or_default(),
                weight,
                timeout_s: pick(spec.timeout_s, d.timeout_s, "timeout_s")?,
                rss_limit_mb: pick(spec.rss_limit_mb, d.rss_limit_mb, "rss_limit_mb")?,
                malloc_limit_mb: pick(spec.malloc_limit_mb, d.malloc_limit_mb, "malloc_limit_mb")?,
                max_len: pick(spec.max_len, d.max_len, "max_len")?,
                threads: pick(spec.threads, d.threads, "threads")?.max(1),
                value_profile: spec.value_profile.or(d.value_profile).unwrap_or(false),
                env: variant.as_ref().map(|v| v.env.clone()).unwrap_or_default(),
            });
        }
    }
    if lanes.is_empty() {
        return Err(format!("{} declares no targets", path.display()));
    }
    Ok(lanes)
}

pub fn targets(lanes: &[Lane]) -> Vec<String> {
    let mut names: Vec<String> = lanes.iter().map(|lane| lane.target.clone()).collect();
    names.dedup();
    names
}
