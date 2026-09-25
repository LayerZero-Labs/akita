//! The unattended campaign loop.
//!
//! Workers are runner-managed: each job is one libFuzzer process for one lane,
//! running until it records a failure or its time slice ends. libFuzzer's own
//! `-fork`/`-jobs` modes are never used, so worker count and Akita's internal
//! Rayon threads are the only two concurrency settings, accounted together.

use crate::findings;
use crate::libfuzzer::{self, Status};
use crate::registry::Lane;
use crate::resources::Budget;
use crate::store::{count_files, now, read_json, write_json, RotatingLog, Store};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

const TAIL_LINES: usize = 400;
const LOG_BYTES: u64 = 4 << 20;
const STATE_INTERVAL: Duration = Duration::from_secs(15);
const CORPUS_COUNT_INTERVAL: Duration = Duration::from_secs(300);
const STARTUP_GRACE_S: u64 = 120;
const KILL_GRACE: Duration = Duration::from_secs(30);
const MAX_BACKOFF_S: u64 = 1800;

static SIGNALS: AtomicU32 = AtomicU32::new(0);

extern "C" fn on_signal(_: libc::c_int) {
    SIGNALS.fetch_add(1, Ordering::SeqCst);
}

fn install_signals() {
    // SAFETY: the handler only touches an atomic.
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as usize);
        libc::signal(libc::SIGTERM, on_signal as *const () as usize);
    }
}

pub struct Options {
    pub slice_s: u64,
    pub duration_s: Option<u64>,
    pub skip_baseline: bool,
    pub replay_new_findings: bool,
    pub hard_memory_headroom_mb: u64,
    pub extra_args: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LaneState {
    pub cpu_seconds: f64,
    pub jobs: u64,
    pub execs: u64,
    pub findings: u64,
    pub restarts: u64,
    pub startup_failures: u64,
    pub infra_failures: u64,
    pub consecutive_failures: u64,
    pub backoff_until: u64,
    pub cov: u64,
    pub ft: u64,
    pub corpus_files: u64,
    pub baseline: Option<Value>,
    pub harness: BTreeMap<String, BTreeMap<String, f64>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct State {
    pub lanes: BTreeMap<String, LaneState>,
    pub sessions: Vec<Value>,
    pub updated: u64,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Purpose {
    Fuzz,
    Baseline,
    Replay,
}

impl Purpose {
    fn name(self) -> &'static str {
        match self {
            Purpose::Fuzz => "fuzz",
            Purpose::Baseline => "baseline",
            Purpose::Replay => "replay",
        }
    }
}

struct Job {
    lane: Lane,
    purpose: Purpose,
    child: Child,
    started: Instant,
    deadline: Instant,
    log: RotatingLog,
    artifacts: PathBuf,
    stats_file: PathBuf,
    status: Status,
    tail: VecDeque<String>,
    artifact: Option<PathBuf>,
    finding_id: Option<String>,
    terminated_at: Option<Instant>,
    output_closed: bool,
    exited_at: Option<Instant>,
}

enum Message {
    Line(u64, String),
    Closed(u64),
}

pub struct Runner {
    dist: PathBuf,
    store: Store,
    lanes: Vec<Lane>,
    budget: Budget,
    options: Options,
    campaign: Value,
    build_id: String,
    jobs: HashMap<u64, Job>,
    sender: Sender<Message>,
    receiver: Receiver<Message>,
    stopping: bool,
    started: Instant,
    started_unix: u64,
    events: RotatingLog,
    sequence: u64,
    state: State,
    symbolizer: Option<PathBuf>,
    pending_replays: VecDeque<(String, Lane, PathBuf)>,
    last_state: Instant,
    last_corpus_count: Instant,
}

fn which(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|path| path.is_file())
    })
}

pub fn symbolizer(dist: &Path) -> Option<PathBuf> {
    let bundled = dist.join("bin/llvm-symbolizer");
    if bundled.is_file() {
        Some(bundled)
    } else {
        which("llvm-symbolizer")
    }
}

impl Runner {
    pub fn new(
        dist: PathBuf,
        store: Store,
        lanes: Vec<Lane>,
        budget: Budget,
        options: Options,
        campaign: Value,
        build_id: String,
    ) -> std::io::Result<Self> {
        let mut state: State = read_json(&store.state_path).unwrap_or_default();
        state
            .sessions
            .push(json!({"started": now(), "build_id": build_id, "budget": budget}));
        for lane in &lanes {
            state.lanes.entry(lane.name()).or_default();
        }
        let events = RotatingLog::open(&store.root.join("events.log"), LOG_BYTES)?;
        let (sender, receiver) = mpsc::channel();
        let symbolizer = symbolizer(&dist);
        Ok(Self {
            dist,
            store,
            lanes,
            budget,
            options,
            campaign,
            build_id,
            jobs: HashMap::new(),
            sender,
            receiver,
            stopping: false,
            started: Instant::now(),
            started_unix: now(),
            events,
            sequence: 0,
            state,
            symbolizer,
            pending_replays: VecDeque::new(),
            last_state: Instant::now() - STATE_INTERVAL,
            last_corpus_count: Instant::now() - CORPUS_COUNT_INTERVAL,
        })
    }

    fn event(&mut self, message: &str) {
        let line = format!("{} {message}", now());
        self.events.line(&line);
        println!("{line}");
    }

    fn lane_state(&mut self, name: &str) -> &mut LaneState {
        self.state.lanes.entry(name.to_string()).or_default()
    }

    fn write_state(&mut self, force: bool) {
        if !force && self.last_state.elapsed() < STATE_INTERVAL {
            return;
        }
        self.last_state = Instant::now();
        if force || self.last_corpus_count.elapsed() >= CORPUS_COUNT_INTERVAL {
            self.last_corpus_count = Instant::now();
            for lane in self.lanes.clone() {
                let count = count_files(&self.store.corpus.join(&lane.target));
                self.lane_state(&lane.name()).corpus_files = count;
            }
        }
        self.state.updated = now();
        let _ = write_json(&self.store.state_path, &self.state);
        let jobs: Vec<Value> = self
            .jobs
            .iter()
            .map(|(id, job)| {
                json!({
                    "job": id,
                    "lane": job.lane.name(),
                    "purpose": job.purpose.name(),
                    "pid": job.child.id(),
                    "running_s": job.started.elapsed().as_secs(),
                    "status": job.status,
                    "stats_file": job.stats_file,
                })
            })
            .collect();
        let live = json!({
            "pid": std::process::id(),
            "started": self.started_unix,
            "updated": now(),
            "stopping": self.stopping,
            "budget": self.budget,
            "jobs": jobs,
        });
        let _ = write_json(&self.store.live_path, &live);
    }

    fn reservation(&self, lane: &Lane) -> u64 {
        lane.rss_limit_mb
            + if self.budget.hard_memory {
                self.options.hard_memory_headroom_mb
            } else {
                0
            }
    }

    fn fits(&self, lane: &Lane) -> bool {
        if self.jobs.is_empty() {
            return true; // a small budget still makes progress
        }
        let cpus: u64 = self.jobs.values().map(|job| job.lane.threads).sum();
        let memory: u64 = self
            .jobs
            .values()
            .map(|job| self.reservation(&job.lane))
            .sum();
        cpus + lane.threads <= self.budget.cpus
            && memory + self.reservation(lane) <= self.budget.memory_mb
    }

    /// Copy shipped seeds into the writable corpus, skipping quarantined ones.
    fn seed_corpus(&self, lane: &Lane) -> std::io::Result<()> {
        let seeds = self.dist.join("seeds").join(&lane.target);
        let corpus = self.store.corpus.join(&lane.target);
        let quarantine = self.store.quarantine.join(&lane.target);
        std::fs::create_dir_all(&corpus)?;
        let Ok(entries) = std::fs::read_dir(&seeds) else {
            return Ok(());
        };
        for entry in entries.flatten() {
            if !entry.file_type()?.is_file() {
                continue;
            }
            let data = std::fs::read(entry.path())?;
            let digest = libfuzzer::sha1_hex(&data);
            if !corpus.join(&digest).exists() && !quarantine.join(&digest).exists() {
                std::fs::write(corpus.join(&digest), data)?;
            }
        }
        Ok(())
    }

    fn spawn(&mut self, lane: &Lane, purpose: Purpose, inputs: &[PathBuf]) -> std::io::Result<u64> {
        self.sequence += 1;
        let id = self.sequence;
        let job_name = format!("{}-{}-{id}", now(), std::process::id());
        let artifacts = self
            .store
            .root
            .join("artifacts")
            .join(lane.name())
            .join(&job_name);
        std::fs::create_dir_all(&artifacts)?;
        let stats_file = self
            .store
            .stats
            .join(lane.name())
            .join(format!("{job_name}.json"));
        std::fs::create_dir_all(stats_file.parent().expect("parent"))?;
        let binary = self.dist.join("bin").join(libfuzzer::BINARY);
        let corpus = self.store.corpus.join(&lane.target);
        let mut args = match purpose {
            Purpose::Fuzz => libfuzzer::fuzz_args(
                &binary,
                lane,
                &[&corpus],
                &artifacts,
                self.options.slice_s,
                false,
                &self.options.extra_args,
            ),
            Purpose::Baseline => {
                let scratch = artifacts.join("scratch-corpus");
                std::fs::create_dir_all(&scratch)?;
                libfuzzer::fuzz_args(
                    &binary,
                    lane,
                    &[&scratch, &corpus],
                    &artifacts,
                    0,
                    true,
                    &[],
                )
            }
            Purpose::Replay => {
                let mut args = libfuzzer::base_args(
                    &binary,
                    libfuzzer::Limits {
                        lane,
                        timeout_s: lane.timeout_s * 2,
                    },
                    &artifacts,
                );
                args.extend(inputs.iter().map(|path| path.display().to_string()));
                args
            }
        };
        if self.budget.hard_memory {
            let limit = lane.rss_limit_mb + self.options.hard_memory_headroom_mb;
            let mut wrapped: Vec<String> = [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "-p",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            wrapped.extend([
                format!("MemoryMax={limit}M"),
                "-p".into(),
                "MemorySwapMax=0".into(),
                "--".into(),
            ]);
            wrapped.extend(args);
            args = wrapped;
        }
        let env = libfuzzer::environment(
            lane,
            &self.dist.join("artifacts/schedules"),
            &stats_file,
            self.symbolizer.as_deref(),
        );
        let mut log = RotatingLog::open(
            &self
                .store
                .logs
                .join(lane.name())
                .join(format!("{job_name}.log")),
            LOG_BYTES,
        )?;
        log.line(&format!(
            "# {} job {job_name}: {}",
            purpose.name(),
            args.join(" ")
        ));
        let mut child = Command::new(&args[0])
            .args(&args[1..])
            .envs(env)
            .current_dir(&artifacts)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()?;
        let stderr = child.stderr.take().expect("piped stderr");
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).split(b'\n').map_while(Result::ok) {
                if sender
                    .send(Message::Line(
                        id,
                        String::from_utf8_lossy(&line).into_owned(),
                    ))
                    .is_err()
                {
                    return;
                }
            }
            let _ = sender.send(Message::Closed(id));
        });
        let budget_s = match purpose {
            Purpose::Fuzz => self.options.slice_s + 600,
            Purpose::Baseline => 3600,
            Purpose::Replay => 600,
        };
        let deadline =
            Instant::now() + Duration::from_secs(budget_s + 2 * lane.timeout_s + STARTUP_GRACE_S);
        if purpose == Purpose::Fuzz {
            self.lane_state(&lane.name()).jobs += 1;
        }
        self.jobs.insert(
            id,
            Job {
                lane: lane.clone(),
                purpose,
                child,
                started: Instant::now(),
                deadline,
                log,
                artifacts,
                stats_file,
                status: Status::default(),
                tail: VecDeque::with_capacity(TAIL_LINES),
                artifact: None,
                finding_id: None,
                terminated_at: None,
                output_closed: false,
                exited_at: None,
            },
        );
        Ok(id)
    }

    fn consume(job: &mut Job, line: String) {
        job.log.line(&line);
        libfuzzer::parse_status(&line, &mut job.status);
        if let Some(path) = libfuzzer::artifact_path(&line) {
            job.artifact = Some(if path.is_absolute() {
                path
            } else {
                job.artifacts.join(path)
            });
        }
        if job.tail.len() == TAIL_LINES {
            job.tail.pop_front();
        }
        job.tail.push_back(line);
    }

    fn pump(&mut self, timeout: Duration) {
        let mut next = self.receiver.recv_timeout(timeout);
        loop {
            match next {
                Ok(Message::Line(id, line)) => {
                    if let Some(job) = self.jobs.get_mut(&id) {
                        Self::consume(job, line);
                    }
                }
                Ok(Message::Closed(id)) => {
                    if let Some(job) = self.jobs.get_mut(&id) {
                        job.output_closed = true;
                    }
                }
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => return,
            }
            next = self.receiver.recv_timeout(Duration::ZERO);
        }
    }

    fn signal(job: &Job, signal: libc::c_int) {
        // SAFETY: the job is its own process-group leader.
        unsafe {
            libc::killpg(job.child.id() as libc::pid_t, signal);
        }
    }

    fn reap(&mut self) {
        let ids: Vec<u64> = self.jobs.keys().copied().collect();
        for id in ids {
            let job = self.jobs.get_mut(&id).expect("job");
            match job.child.try_wait() {
                // A descendant may keep the pipe open after the worker exits.
                Ok(Some(status))
                    if job.output_closed
                        || job
                            .exited_at
                            .is_some_and(|at| at.elapsed() > Duration::from_secs(10)) =>
                {
                    let code = status
                        .code()
                        .unwrap_or_else(|| -status.signal().unwrap_or(0));
                    let job = self.jobs.remove(&id).expect("job");
                    self.finish(job, code);
                }
                Ok(Some(_)) => {
                    // Let the reader drain the pipe first.
                    job.exited_at.get_or_insert_with(Instant::now);
                }
                Ok(None) | Err(_) => {
                    let now = Instant::now();
                    if job.terminated_at.is_none() && now > job.deadline {
                        let name = job.lane.name();
                        Self::signal(job, libc::SIGTERM);
                        job.terminated_at = Some(now);
                        self.event(&format!(
                            "watchdog: {name} exceeded its wall-clock deadline; terminating"
                        ));
                    } else if job.terminated_at.is_some_and(|at| now - at > KILL_GRACE) {
                        Self::signal(job, libc::SIGKILL);
                    }
                }
            }
        }
    }

    fn merge_harness_stats(&mut self, job: &Job) {
        if let Some(stats) = read_json::<Value>(&job.stats_file) {
            let state = self.lane_state(&job.lane.name());
            for group in ["counters", "seconds"] {
                if let Some(values) = stats[group].as_object() {
                    let target = state.harness.entry(group.to_string()).or_default();
                    for (key, value) in values {
                        *target.entry(key.clone()).or_default() += value.as_f64().unwrap_or(0.0);
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&job.stats_file);
    }

    fn finish(&mut self, job: Job, code: i32) {
        let name = job.lane.name();
        let elapsed = job.started.elapsed().as_secs_f64();
        let tail: Vec<String> = job.tail.iter().cloned().collect();
        let artifact = job.artifact.clone().or_else(|| {
            std::fs::read_dir(&job.artifacts)
                .ok()?
                .flatten()
                .map(|entry| entry.path())
                .find(|path| path.is_file())
        });
        if job.purpose == Purpose::Replay {
            self.finish_replay(&job, code, &tail);
            let _ = std::fs::remove_dir_all(&job.artifacts);
            return;
        }
        self.merge_harness_stats(&job);
        {
            let state = self.lane_state(&name);
            state.cpu_seconds += elapsed * job.lane.threads as f64;
            if job.purpose == Purpose::Fuzz {
                state.execs += job.status.execs;
            }
            state.cov = state.cov.max(job.status.cov);
            state.ft = state.ft.max(job.status.ft);
            if job.purpose == Purpose::Baseline {
                state.baseline = Some(
                    json!({"ok": code == 0, "seconds": elapsed.round(), "at": now(), "execs": job.status.execs}),
                );
            }
        }
        let _ = std::fs::remove_dir_all(job.artifacts.join("scratch-corpus"));
        if code == 0 {
            self.lane_state(&name).consecutive_failures = 0;
            if job.purpose == Purpose::Baseline {
                self.event(&format!(
                    "baseline {name}: ok ({} seeds, {elapsed:.0}s)",
                    job.status.execs
                ));
            }
            let _ = std::fs::remove_dir_all(&job.artifacts);
            return;
        }
        if self.stopping && job.terminated_at.is_some() && artifact.is_none() {
            let _ = std::fs::remove_dir_all(&job.artifacts);
            return;
        }
        let failure_output = tail
            .iter()
            .any(|line| line.contains("panicked at") || line.contains("ERROR:"));
        if artifact.is_some() || failure_output {
            self.record_finding(&job, artifact.as_deref(), &tail, elapsed);
        } else if !job.status.inited && elapsed < STARTUP_GRACE_S as f64 {
            self.lane_state(&name).startup_failures += 1;
            let last: Vec<&str> = tail
                .iter()
                .rev()
                .take(5)
                .rev()
                .map(String::as_str)
                .collect();
            self.backoff(
                &name,
                &format!("startup failure (exit {code}): {}", last.join(" | ")),
            );
        } else {
            self.lane_state(&name).infra_failures += 1;
            let reason = if code < 0 {
                format!("killed by signal {}", -code)
            } else {
                format!("exit {code}")
            };
            let last: Vec<&str> = tail
                .iter()
                .rev()
                .take(3)
                .rev()
                .map(String::as_str)
                .collect();
            self.backoff(
                &name,
                &format!(
                    "infrastructure failure ({reason}, no artifact; possibly an OS OOM kill): {}",
                    last.join(" | ")
                ),
            );
        }
        let _ = std::fs::remove_dir_all(&job.artifacts);
    }

    fn record_finding(
        &mut self,
        job: &Job,
        artifact: Option<&Path>,
        tail: &[String],
        elapsed: f64,
    ) {
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
                return;
            }
        };
        {
            let state = self.lane_state(&name);
            state.findings += 1;
            state.restarts += 1;
        }
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
        if elapsed < 60.0 && !job.status.inited && moved.is_none() {
            self.backoff(
                &name,
                "failure before initialization with no corpus input to quarantine",
            );
        } else {
            self.lane_state(&name).consecutive_failures = 0;
        }
    }

    fn finish_replay(&mut self, job: &Job, code: i32, tail: &[String]) {
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

    fn backoff(&mut self, name: &str, reason: &str) {
        let state = self.lane_state(name);
        state.consecutive_failures += 1;
        let delay = (5u64 << (state.consecutive_failures - 1).min(20)).min(MAX_BACKOFF_S);
        state.backoff_until = now() + delay;
        let failures = state.consecutive_failures;
        self.event(&format!(
            "{name}: {reason}; retry in {delay}s (consecutive failures: {failures})"
        ));
    }

    fn choose(&mut self) -> Option<Lane> {
        let running: HashMap<String, u64> = self
            .jobs
            .values()
            .filter(|job| job.purpose == Purpose::Fuzz)
            .fold(HashMap::new(), |mut acc, job| {
                *acc.entry(job.lane.name()).or_default() += 1;
                acc
            });
        let time = now();
        let mut best: Option<(f64, Lane)> = None;
        for lane in &self.lanes {
            let state = self
                .state
                .lanes
                .get(&lane.name())
                .cloned()
                .unwrap_or_default();
            if state.backoff_until > time || !self.fits(lane) {
                continue;
            }
            let projected = state.cpu_seconds
                + (running.get(&lane.name()).copied().unwrap_or(0)
                    * self.options.slice_s
                    * lane.threads) as f64;
            let score = projected / lane.weight;
            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score < *best_score)
            {
                best = Some((score, lane.clone()));
            }
        }
        best.map(|(_, lane)| lane)
    }

    fn schedule(&mut self) {
        while !self.stopping {
            let Some((_, lane, _)) = self.pending_replays.front() else {
                break;
            };
            if !self.fits(lane) {
                break;
            }
            let (id, lane, path) = self.pending_replays.pop_front().expect("front");
            match self.spawn(&lane, Purpose::Replay, &[path]) {
                Ok(job) => self.jobs.get_mut(&job).expect("job").finding_id = Some(id),
                Err(error) => self.event(&format!("replay of {id} failed to start: {error}")),
            }
        }
        while !self.stopping {
            let Some(lane) = self.choose() else {
                return;
            };
            if let Err(error) = self.spawn(&lane, Purpose::Fuzz, &[]) {
                self.backoff(&lane.name(), &format!("spawn failed: {error}"));
            }
        }
    }

    fn check_signals(&mut self) {
        let count = SIGNALS.load(Ordering::SeqCst);
        if count > 0 && !self.stopping {
            self.stopping = true;
            self.event("received a stop signal; stopping (send again to force)");
        }
    }

    fn baseline(&mut self) {
        self.event(&format!(
            "baseline: executing shipped seeds for {} lanes",
            self.lanes.len()
        ));
        let mut queue: VecDeque<Lane> = self.lanes.iter().cloned().collect();
        while (!queue.is_empty() || !self.jobs.is_empty()) && !self.stopping {
            while queue.front().is_some_and(|lane| self.fits(lane)) {
                let lane = queue.pop_front().expect("front");
                if let Err(error) = self.spawn(&lane, Purpose::Baseline, &[]) {
                    self.event(&format!(
                        "baseline {} failed to start: {error}",
                        lane.name()
                    ));
                }
            }
            self.pump(Duration::from_secs(1));
            self.reap();
            self.write_state(false);
            self.check_signals();
        }
        let failed: Vec<String> = self
            .lanes
            .iter()
            .filter(|lane| {
                !self
                    .state
                    .lanes
                    .get(&lane.name())
                    .and_then(|s| s.baseline.as_ref())
                    .is_some_and(|b| b["ok"] == true)
            })
            .map(Lane::name)
            .collect();
        if failed.is_empty() {
            self.event("baseline: all lanes passed");
        } else {
            self.event(&format!(
                "baseline: FAILED for {}; see findings (fuzzing continues)",
                failed.join(", ")
            ));
        }
    }

    pub fn run(mut self) -> std::io::Result<()> {
        install_signals();
        for note in self.budget.notes.clone() {
            self.event(&note);
        }
        for lane in self.lanes.clone() {
            self.seed_corpus(&lane)?;
        }
        if !self.options.skip_baseline {
            self.baseline();
        }
        self.event(&format!(
            "fuzzing {} lanes; slice {}s",
            self.lanes.len(),
            self.options.slice_s
        ));
        while !self.stopping {
            if self
                .options
                .duration_s
                .is_some_and(|limit| self.started.elapsed().as_secs() >= limit)
            {
                self.event("duration reached; stopping");
                self.stopping = true;
                break;
            }
            self.schedule();
            self.pump(Duration::from_secs(1));
            self.reap();
            self.write_state(false);
            self.check_signals();
        }
        self.shutdown();
        Ok(())
    }

    fn shutdown(&mut self) {
        for job in self.jobs.values_mut() {
            Self::signal(job, libc::SIGINT);
            job.terminated_at = Some(Instant::now());
        }
        let deadline = Instant::now() + KILL_GRACE;
        while !self.jobs.is_empty()
            && Instant::now() < deadline
            && SIGNALS.load(Ordering::SeqCst) < 2
        {
            self.pump(Duration::from_millis(500));
            self.reap();
        }
        for job in self.jobs.values() {
            Self::signal(job, libc::SIGKILL);
        }
        while !self.jobs.is_empty() {
            self.pump(Duration::from_millis(200));
            self.reap();
        }
        if let Some(session) = self.state.sessions.last_mut() {
            session["stopped"] = json!(now());
        }
        self.write_state(true);
        let _ = std::fs::remove_file(&self.store.live_path);
        self.event("stopped; state saved");
    }
}
