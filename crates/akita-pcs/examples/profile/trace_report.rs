use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

pub(super) const ROOT_SPAN: &str = "akita_profile_run";
const SUMMARY_SCHEMA_VERSION: u32 = 1;
const TAXONOMY_VERSION: u32 = 1;

#[derive(Debug)]
struct Samples {
    points: Vec<(f64, f64)>,
}

impl Samples {
    fn summary(&self) -> CounterSummary {
        let samples = self.points.len();
        let sum = self.points.iter().map(|(_, value)| value).sum::<f64>();
        CounterSummary {
            samples,
            min: self.points.iter().map(|(_, value)| *value).reduce(f64::min),
            mean: (samples != 0).then_some(sum / samples as f64),
            max: self.points.iter().map(|(_, value)| *value).reduce(f64::max),
        }
    }
}

#[derive(Debug, Serialize)]
struct ProfileSummary {
    schema_version: u32,
    taxonomy_version: u32,
    run: RunMetadata,
    root: Option<RootSummary>,
    peak_rss_gib: Option<f64>,
    cpu_utilization: Option<CpuUtilizationSummary>,
    spans: BTreeMap<String, SpanAggregate>,
    counters: BTreeMap<String, CounterSummary>,
}

#[derive(Debug, Serialize)]
struct RunMetadata {
    mode: String,
    num_vars: usize,
    num_polys: usize,
    prove_threads: usize,
    logical_cpus: usize,
    timestamp_unix_secs: u64,
    git_rev: Option<String>,
}

#[derive(Debug, Serialize)]
struct RootSummary {
    label: &'static str,
    wall_time_ns: u64,
    dark_time_ns: u64,
    dark_time_fraction: f64,
    peak_sampled_rss_gib: Option<f64>,
}

#[derive(Debug, Serialize, PartialEq)]
struct SpanAggregate {
    count: u64,
    total_ns: u64,
    self_ns: u64,
}

#[derive(Debug, Serialize)]
struct CounterSummary {
    samples: usize,
    min: Option<f64>,
    mean: Option<f64>,
    max: Option<f64>,
}

#[derive(Debug, Serialize)]
struct CpuUtilizationSummary {
    prove_threads: usize,
    mean_effective_cores: f64,
    peak_effective_cores: f64,
    mean_prove_pool_utilization_percent: f64,
    samples_below_one_core_fraction: f64,
    samples_below_half_pool_fraction: f64,
}

pub(super) struct ReportContext<'a> {
    pub(super) mode: &'a str,
    pub(super) num_vars: usize,
    pub(super) num_polys: usize,
    pub(super) prove_threads: usize,
    pub(super) logical_cpus: usize,
    pub(super) timestamp_unix_secs: u64,
    pub(super) peak_rss_bytes: Option<u64>,
}

struct OpenSpan {
    name: String,
    start_us: f64,
    child_us: f64,
}

#[derive(Clone, Copy)]
struct Interval {
    start_us: f64,
    end_us: f64,
}

impl Interval {
    fn duration_ns(self) -> u64 {
        us_to_ns(self.end_us - self.start_us)
    }

    fn contains(self, timestamp_us: f64) -> bool {
        timestamp_us >= self.start_us && timestamp_us <= self.end_us
    }
}

struct TraceAggregate {
    spans: BTreeMap<String, SpanAggregate>,
    root: Option<(Interval, u64)>,
    counters: BTreeMap<String, Samples>,
}

pub(super) fn finalize_trace(
    trace_path: &Path,
    context: &ReportContext<'_>,
) -> Result<PathBuf, String> {
    let encoded = fs::read_to_string(trace_path)
        .map_err(|error| format!("read {}: {error}", trace_path.display()))?;
    let trace: Value = serde_json::from_str(&encoded)
        .map_err(|error| format!("parse {}: {error}", trace_path.display()))?;
    let events = match trace {
        Value::Array(events) => events,
        Value::Object(mut object) => match object.remove("traceEvents") {
            Some(Value::Array(events)) => events,
            _ => return Err("profile trace object must contain a traceEvents array".to_string()),
        },
        _ => return Err("profile trace root must be an event array".to_string()),
    };

    let converted = convert_resource_counters(events);
    let aggregate = aggregate_events(&converted);
    write_atomic(
        trace_path,
        &serde_json::to_vec(&converted)
            .map_err(|error| format!("encode {}: {error}", trace_path.display()))?,
    )?;

    let summary = build_summary(aggregate, context);
    let summary_path = trace_path.with_extension("summary.json");
    write_atomic(
        &summary_path,
        &serde_json::to_vec_pretty(&summary)
            .map_err(|error| format!("encode profile summary: {error}"))?,
    )?;
    Ok(summary_path)
}

fn build_summary(aggregate: TraceAggregate, context: &ReportContext<'_>) -> ProfileSummary {
    let root_interval = aggregate.root.map(|(interval, _)| interval);
    let root = aggregate.root.map(|(interval, dark_time_ns)| {
        let wall_time_ns = interval.duration_ns();
        RootSummary {
            label: ROOT_SPAN,
            wall_time_ns,
            dark_time_ns,
            dark_time_fraction: if wall_time_ns == 0 {
                0.0
            } else {
                dark_time_ns as f64 / wall_time_ns as f64
            },
            peak_sampled_rss_gib: peak_within(aggregate.counters.get("rss_gib"), interval),
        }
    });
    let cpu_utilization = aggregate
        .counters
        .get("process_effective_cores")
        .and_then(|samples| cpu_summary(samples, root_interval?, context.prove_threads));
    let counters = aggregate
        .counters
        .into_iter()
        .map(|(name, samples)| (name, samples.summary()))
        .collect();

    ProfileSummary {
        schema_version: SUMMARY_SCHEMA_VERSION,
        taxonomy_version: TAXONOMY_VERSION,
        run: RunMetadata {
            mode: context.mode.to_string(),
            num_vars: context.num_vars,
            num_polys: context.num_polys,
            prove_threads: context.prove_threads,
            logical_cpus: context.logical_cpus,
            timestamp_unix_secs: context.timestamp_unix_secs,
            git_rev: git_rev(),
        },
        root,
        peak_rss_gib: context
            .peak_rss_bytes
            .map(|bytes| bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
        cpu_utilization,
        spans: aggregate.spans,
        counters,
    }
}

fn convert_resource_counters(events: Vec<Value>) -> Vec<Value> {
    let mut converted = Vec::with_capacity(events.len());
    for event in events {
        let samples = counter_samples(&event);
        if samples.is_empty() {
            converted.push(event);
            continue;
        }
        let timestamp = event.get("ts").cloned().unwrap_or(Value::Null);
        let process_id = event.get("pid").cloned().unwrap_or(Value::Null);
        for (name, value) in samples {
            let mut args = Map::new();
            args.insert(name.clone(), json!(value));
            converted.push(json!({
                "name": name,
                "ph": "C",
                "ts": timestamp.clone(),
                "pid": process_id.clone(),
                "tid": 0,
                "args": args,
            }));
        }
    }
    converted
}

fn aggregate_events(events: &[Value]) -> TraceAggregate {
    let mut stacks = HashMap::<String, Vec<OpenSpan>>::new();
    let mut spans = BTreeMap::<String, SpanAggregate>::new();
    let mut root: Option<(Interval, u64)> = None;
    let mut counters = BTreeMap::<String, Samples>::new();

    for event in events {
        let phase = event.get("ph").and_then(Value::as_str).unwrap_or_default();
        let name = event
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let timestamp_us = event.get("ts").and_then(Value::as_f64).unwrap_or_default();
        let thread = event.get("tid").map(Value::to_string).unwrap_or_default();
        match phase {
            "B" => stacks.entry(thread).or_default().push(OpenSpan {
                name: name.to_string(),
                start_us: timestamp_us,
                child_us: 0.0,
            }),
            "E" => close_span(
                stacks.entry(thread).or_default(),
                name,
                timestamp_us,
                &mut spans,
                &mut root,
            ),
            "C" => {
                if let Some(args) = event.get("args").and_then(Value::as_object) {
                    for (counter, value) in args {
                        if let Some(value) = json_f64(value) {
                            counters
                                .entry(counter.clone())
                                .or_insert_with(|| Samples { points: Vec::new() })
                                .points
                                .push((timestamp_us, value));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    TraceAggregate {
        spans,
        root,
        counters,
    }
}

fn close_span(
    stack: &mut Vec<OpenSpan>,
    name: &str,
    end_us: f64,
    spans: &mut BTreeMap<String, SpanAggregate>,
    root: &mut Option<(Interval, u64)>,
) {
    let Some(matching) = stack.iter().rposition(|open| open.name == name) else {
        return;
    };
    stack.truncate(matching + 1);
    let Some(open) = stack.pop() else { return };
    let duration_us = (end_us - open.start_us).max(0.0);
    let self_us = (duration_us - open.child_us).max(0.0);
    if let Some(parent) = stack.last_mut() {
        parent.child_us += duration_us;
    }
    let span = spans.entry(open.name.clone()).or_insert(SpanAggregate {
        count: 0,
        total_ns: 0,
        self_ns: 0,
    });
    span.count += 1;
    span.total_ns += us_to_ns(duration_us);
    span.self_ns += us_to_ns(self_us);
    if open.name == ROOT_SPAN {
        let interval = Interval {
            start_us: open.start_us,
            end_us,
        };
        if root.is_none_or(|(previous, _)| interval.duration_ns() > previous.duration_ns()) {
            *root = Some((interval, us_to_ns(self_us)));
        }
    }
}

fn counter_samples(event: &Value) -> Vec<(String, f64)> {
    if !matches!(event.get("ph").and_then(Value::as_str), Some("i" | "I")) {
        return Vec::new();
    }
    event
        .get("args")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(key, value)| {
            let name = key
                .strip_prefix("counters.")
                .or_else(|| key.strip_prefix("counter_"))?;
            Some((name.to_string(), json_f64(value)?))
        })
        .collect()
}

fn json_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(encoded) => encoded.parse().ok(),
        _ => None,
    }
}

fn peak_within(samples: Option<&Samples>, interval: Interval) -> Option<f64> {
    samples?
        .points
        .iter()
        .filter(|(timestamp, _)| interval.contains(*timestamp))
        .map(|(_, value)| *value)
        .reduce(f64::max)
}

fn cpu_summary(
    samples: &Samples,
    interval: Interval,
    prove_threads: usize,
) -> Option<CpuUtilizationSummary> {
    let scoped = samples
        .points
        .iter()
        .filter(|(timestamp, _)| interval.contains(*timestamp))
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();
    let count = scoped.len();
    if count == 0 {
        return None;
    }
    let pool = prove_threads.max(1) as f64;
    let sum = scoped.iter().sum::<f64>();
    let mean = sum / count as f64;
    let peak = scoped.iter().copied().fold(0.0, f64::max);
    let below_one = scoped.iter().filter(|value| **value < 1.0).count();
    let below_half = scoped.iter().filter(|value| **value < pool / 2.0).count();
    Some(CpuUtilizationSummary {
        prove_threads,
        mean_effective_cores: mean,
        peak_effective_cores: peak,
        mean_prove_pool_utilization_percent: mean / pool * 100.0,
        samples_below_one_core_fraction: below_one as f64 / count as f64,
        samples_below_half_pool_fraction: below_half as f64 / count as f64,
    })
}

fn us_to_ns(microseconds: f64) -> u64 {
    (microseconds.max(0.0) * 1_000.0).round() as u64
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)
        .map_err(|error| format!("write {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| {
        format!(
            "replace {} with {}: {error}",
            path.display(),
            temporary.display()
        )
    })
}

fn git_rev() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_samples_accept_stringified_tracing_fields() {
        let event = json!({
            "ph": "i",
            "args": {
                "counter_process_effective_cores": "7.5",
                "message": "profile_resource_sample"
            }
        });
        assert_eq!(
            counter_samples(&event),
            vec![("process_effective_cores".to_string(), 7.5)]
        );
    }

    #[test]
    fn aggregation_reports_inclusive_and_self_time() {
        let events = vec![
            json!({"ph": "B", "name": ROOT_SPAN, "ts": 0.0, "tid": 1}),
            json!({"ph": "B", "name": "child", "ts": 2.0, "tid": 1}),
            json!({"ph": "E", "name": "child", "ts": 7.0, "tid": 1}),
            json!({"ph": "E", "name": ROOT_SPAN, "ts": 10.0, "tid": 1}),
        ];
        let aggregate = aggregate_events(&events);
        assert_eq!(
            aggregate.spans[ROOT_SPAN],
            SpanAggregate {
                count: 1,
                total_ns: 10_000,
                self_ns: 5_000,
            }
        );
        assert_eq!(aggregate.root.map(|(_, dark)| dark), Some(5_000));
    }

    #[test]
    fn counter_conversion_emits_native_perfetto_tracks() {
        let converted = convert_resource_counters(vec![json!({
            "ph": "i",
            "ts": 4.0,
            "pid": 2,
            "args": {"counter_rss_gib": "3.25"}
        })]);
        assert_eq!(converted[0]["ph"], "C");
        assert_eq!(converted[0]["name"], "rss_gib");
        assert_eq!(converted[0]["args"]["rss_gib"], 3.25);
    }

    #[test]
    fn cpu_summary_excludes_samples_outside_the_root_interval() {
        let samples = Samples {
            points: vec![(1.0, 4.0), (5.0, 6.0), (11.0, 0.0)],
        };
        let summary = cpu_summary(
            &samples,
            Interval {
                start_us: 0.0,
                end_us: 10.0,
            },
            8,
        )
        .unwrap();

        assert_eq!(summary.mean_effective_cores, 5.0);
        assert_eq!(summary.peak_effective_cores, 6.0);
        assert_eq!(summary.samples_below_one_core_fraction, 0.0);
    }
}
