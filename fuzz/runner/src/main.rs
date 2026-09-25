//! `akita-fuzz`: standalone Akita fuzzing campaign runner.
//!
//! The executable lives at the root of a prepared distribution
//! (`akita-fuzz prepare`), next to `bin/` (instrumented targets),
//! `seeds/`, `artifacts/schedules/`, and `campaign/targets.toml`.

mod commands;
mod findings;
mod libfuzzer;
mod prepare;
mod registry;
mod resources;
mod runner;
mod store;
mod tools;

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "akita-fuzz", about = "Standalone Akita fuzzing campaign")]
struct Cli {
    /// Prepared distribution directory (default: this executable's directory).
    #[arg(long, global = true)]
    dist: Option<PathBuf>,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start or resume an unattended campaign.
    Run {
        #[arg(long)]
        output: PathBuf,
        /// CPU slots for workers (default: usable CPUs minus a reserve).
        #[arg(long)]
        cpus: Option<u64>,
        /// CPUs left for the system when --cpus is not given.
        #[arg(long)]
        reserve_cpus: Option<u64>,
        /// Memory budget in MiB (default: 80% of available memory).
        #[arg(long)]
        memory_mb: Option<u64>,
        /// Only these targets (comma separated).
        #[arg(long, value_delimiter = ',')]
        targets: Vec<String>,
        /// Skip these targets (comma separated).
        #[arg(long, value_delimiter = ',')]
        exclude: Vec<String>,
        /// Minutes each libFuzzer process runs before rotation.
        #[arg(long, default_value_t = 60)]
        slice_minutes: u64,
        /// Stop after this many hours (default: run until signalled).
        #[arg(long)]
        duration_hours: Option<f64>,
        /// Do not wrap workers in cgroup memory scopes even if available.
        #[arg(long)]
        no_hard_memory_limit: bool,
        /// Skip the startup execution of every shipped seed.
        #[arg(long)]
        skip_baseline: bool,
        /// Do not replay new findings in a fresh process.
        #[arg(long)]
        no_replay: bool,
        /// Continue an output directory created on another machine.
        #[arg(long)]
        adopt: bool,
        /// Extra libFuzzer flags for every fuzzing job.
        #[arg(last = true)]
        libfuzzer_args: Vec<String>,
    },
    /// Show progress, coverage, and findings (safe while running).
    Status {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Archive corpus, findings, and state (safe while running).
    Export {
        #[arg(long)]
        output: PathBuf,
        /// Archive path or directory (default: OUTPUT/exports).
        #[arg(long)]
        to: Option<PathBuf>,
        #[arg(long)]
        with_logs: bool,
    },
    /// Re-run a finding's samples, or TARGET[@VARIANT] on input files.
    Reproduce {
        #[arg(long)]
        output: PathBuf,
        spec: String,
        inputs: Vec<PathBuf>,
    },
    /// Minimize a finding's first sample with libFuzzer; the original is kept.
    Minimize {
        #[arg(long)]
        output: PathBuf,
        finding: String,
        #[arg(long, default_value_t = 300)]
        seconds: u64,
    },
    /// Build instrumented targets and package a distribution (source tree only).
    Prepare {
        /// Distribution directory to (re)create.
        #[arg(long)]
        out: PathBuf,
        /// Sanitizer for `cargo fuzz build`.
        #[arg(long, default_value = "address")]
        sanitizer: String,
        /// Build the sequential feature graph (no Rayon).
        #[arg(long)]
        sequential: bool,
        /// Package the existing `cargo fuzz build` outputs without rebuilding.
        #[arg(long)]
        skip_build: bool,
    },
    /// Check the prepared distribution (manifest, binaries, seeds, artifacts).
    Validate,
    /// List registry lanes.
    Targets,
    /// Developer: write deterministic seed corpora (uninstrumented).
    Seeds { out: PathBuf },
    /// Developer: list planned and excluded catalog cases.
    Cases {
        #[arg(default_value_t = 20)]
        log2_cost: u32,
    },
    /// Developer: run pseudo-random inputs through a target without libFuzzer.
    Smoke {
        target: String,
        #[arg(default_value_t = 100)]
        count: usize,
        #[arg(default_value_t = 1)]
        seed: u64,
    },
    /// Developer: run inputs once through a target without libFuzzer.
    Replay {
        target: String,
        inputs: Vec<PathBuf>,
    },
    /// Developer: print the library's target names.
    LibraryTargets,
}

fn dist_dir(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."))
    })
}

fn registry_path(dist: &Path) -> PathBuf {
    let shipped = dist.join("campaign/targets.toml");
    if shipped.is_file() {
        shipped
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../campaign/targets.toml")
    }
}

fn lanes(dist: &Path) -> Result<Vec<registry::Lane>, String> {
    registry::load(&registry_path(dist))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let dist = dist_dir(cli.dist);
    match execute(&dist, cli.command) {
        Ok(code) => ExitCode::from(code),
        Err(message) => {
            eprintln!("akita-fuzz: {message}");
            ExitCode::from(2)
        }
    }
}

fn execute(dist: &Path, command: Cmd) -> Result<u8, String> {
    match command {
        Cmd::Run {
            output,
            cpus,
            reserve_cpus,
            memory_mb,
            targets,
            exclude,
            slice_minutes,
            duration_hours,
            no_hard_memory_limit,
            skip_baseline,
            no_replay,
            adopt,
            libfuzzer_args,
        } => {
            let mut lanes = lanes(dist)?;
            if !targets.is_empty() {
                lanes.retain(|lane| targets.contains(&lane.target));
            }
            lanes.retain(|lane| !exclude.contains(&lane.target));
            if lanes.is_empty() {
                return Err("no targets selected".into());
            }
            let problems = commands::validate_dist(dist, &lanes);
            if !problems.is_empty() {
                return Err(format!(
                    "distribution {} is not ready:\n  {}",
                    dist.display(),
                    problems.join("\n  ")
                ));
            }
            let build: serde_json::Value =
                store::read_json(&dist.join("BUILD-INFO.json")).unwrap_or_default();
            let build_id = build["build_id"].as_str().unwrap_or("unknown").to_string();
            let mut store = store::Store::new(&output);
            store.lock()?;
            store.ensure().map_err(|e| e.to_string())?;
            let campaign = store.campaign(&build_id, adopt)?;
            let hard = if no_hard_memory_limit {
                Some(false)
            } else {
                None
            };
            let budget = resources::budget(cpus, memory_mb, reserve_cpus, hard);
            let options = runner::Options {
                slice_s: slice_minutes.max(1) * 60,
                duration_s: duration_hours.map(|hours| (hours * 3600.0) as u64),
                skip_baseline,
                replay_new_findings: !no_replay,
                hard_memory_headroom_mb: 1024,
                extra_args: libfuzzer_args,
            };
            runner::Runner::new(
                dist.to_path_buf(),
                store,
                lanes,
                budget,
                options,
                campaign,
                build_id,
            )
            .and_then(runner::Runner::run)
            .map_err(|e| e.to_string())?;
            Ok(0)
        }
        Cmd::Status { output, json } => {
            Ok(commands::status(&store::Store::new(&output), &lanes(dist)?, json) as u8)
        }
        Cmd::Export {
            output,
            to,
            with_logs,
        } => {
            let path = commands::export(&store::Store::new(&output), to.as_deref(), with_logs)?;
            println!("{}", path.display());
            Ok(0)
        }
        Cmd::Reproduce {
            output,
            spec,
            inputs,
        } => {
            let code = commands::reproduce(
                dist,
                &store::Store::new(&output),
                &lanes(dist)?,
                &spec,
                &inputs,
            )?;
            Ok(u8::try_from(code).unwrap_or(1))
        }
        Cmd::Minimize {
            output,
            finding,
            seconds,
        } => Ok(commands::minimize(
            dist,
            &store::Store::new(&output),
            &lanes(dist)?,
            &finding,
            seconds,
        )? as u8),
        Cmd::Prepare {
            out,
            sanitizer,
            sequential,
            skip_build,
        } => {
            prepare::run(prepare::Prepare {
                dist: out,
                sanitizer,
                sequential,
                skip_build,
            })?;
            Ok(0)
        }
        Cmd::Validate => {
            let problems = commands::validate_dist(dist, &lanes(dist)?);
            if problems.is_empty() {
                println!("distribution {} is ready", dist.display());
                Ok(0)
            } else {
                Err(problems.join("\n"))
            }
        }
        Cmd::Targets => {
            for lane in lanes(dist)? {
                println!(
                    "{:32} {:10} weight={:<5} threads={} timeout={}s rss={}MiB max_len={}  {}",
                    lane.name(),
                    lane.kind,
                    lane.weight,
                    lane.threads,
                    lane.timeout_s,
                    lane.rss_limit_mb,
                    lane.max_len,
                    lane.description
                );
            }
            Ok(0)
        }
        Cmd::Seeds { out } => {
            tools::seeds(&out);
            Ok(0)
        }
        Cmd::Cases { log2_cost } => {
            tools::cases(log2_cost);
            Ok(0)
        }
        Cmd::Smoke {
            target,
            count,
            seed,
        } => tools::smoke(&target, count, seed).map(|()| 0),
        Cmd::Replay { target, inputs } => tools::replay(&target, &inputs).map(|()| 0),
        Cmd::LibraryTargets => {
            tools::list();
            Ok(0)
        }
    }
}
