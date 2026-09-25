# Akita fuzzing campaign

A standalone, resumable, multi-day fuzzing campaign for Akita built on
cargo-fuzz/libFuzzer. Everything is Rust: the targets and their generators
and oracles live in the `akita-fuzz` library (`src/`); the campaign runner is
the `akita-fuzz` binary (`runner/`, which does not link Akita); developer
commands over the library are the `akita-fuzz-dev` binary (`devtool/`).

- What is tested, how, and what is not: [`COVERAGE.md`](COVERAGE.md)
- Findings from development runs: [`FINDINGS.md`](FINDINGS.md)
- Target registry (single source of truth): [`campaign/targets.toml`](campaign/targets.toml)

## Quick start

On a Linux x86_64 or aarch64 machine with `rustup`, `git`, and a C/C++
toolchain (cargo-fuzz links libFuzzer):

```bash
git clone https://github.com/LayerZero-Labs/akita && cd akita/fuzz
rustup show                       # installs the pinned nightly from fuzz/rust-toolchain.toml
cargo install cargo-fuzz --locked
cargo run --release -p akita-fuzz-runner -- prepare --out ../dist   # 20-30 min
../dist/akita-fuzz run --output /data/akita-fuzz
```

`run` never returns on its own (unless `--duration-hours` is given); run it
under `tmux`, `screen`, `nohup`, or a systemd user service. Stop it with
Ctrl-C or `SIGTERM`; rerun the same command to resume.

## 1. Preparation (build)

`prepare` is the only step that needs Cargo and the network:

1. Checks that `campaign/targets.toml`, `cargo fuzz list`, and the library's
   `akita_fuzz::targets::ALL` name exactly the same targets.
2. Runs `cargo fuzz build --release --debug-assertions --sanitizer address
   --target <host> fuzz_all`. `fuzz_all` links every target into one
   instrumented binary and selects one with `AKITA_FUZZ_TARGET` (a sanitizer
   build of the full Akita graph is about 1.4 GiB, so one binary per target
   would need about 28 GiB). This is what adds instrumentation: libFuzzer,
   SanitizerCoverage edge counters, value profiling support, and
   AddressSanitizer, applied to the harness *and every Akita crate and
   dependency it links*. The fuzz profile (`fuzz/Cargo.toml`) also enables
   `debug-assertions`, `overflow-checks`, and line tables. A plain
   `cargo build --release` produces none of this and cannot be fuzzed.
3. Copies into `--out`:
   - `akita-fuzz` (this runner, uninstrumented)
   - `bin/fuzz_all`, the instrumented libFuzzer executable
   - `artifacts/schedules/*.aks` (the shipped trusted schedule catalogs)
   - `seeds/<target>/` generated deterministically, including honest proofs
     and the shipped artifacts
   - `campaign/targets.toml`, this README
   - `bin/llvm-symbolizer` if one is installed (optional, see below)
   - `BUILD-INFO.json` (git commit/branch/dirty flag, rustc and cargo-fuzz
     versions, target triple, sanitizer, features, Cargo.lock hash, build
     host) and `MANIFEST.sha256` (every file's SHA-256)

Options: `--sequential` builds the sequential (no Rayon) feature graph, which
the repository's CI never executes; `--skip-build` repackages an existing
build; `--sanitizer none` builds without ASan (faster, less detection).

The toolchain is pinned in `fuzz/rust-toolchain.toml` and dependency versions
in `fuzz/Cargo.lock` (seeded from the workspace lock), so the same commit
yields the same build. Copy `dist/` to another machine of the same
architecture and OS family to run there without Cargo.

`akita-fuzz validate` re-checks a distribution's manifest.

## 2. Running

```bash
dist/akita-fuzz run --output /data/akita-fuzz [options] [-- extra libFuzzer flags]
```

| Option | Default | Meaning |
|---|---|---|
| `--cpus N` | usable CPUs − reserve | CPU slots for workers |
| `--reserve-cpus R` | max(1, CPUs/16) | CPUs left to the system |
| `--memory-mb M` | 80% of available | Memory budget for all workers |
| `--targets a,b` / `--exclude a,b` | all | Restrict lanes |
| `--slice-minutes M` | 60 | libFuzzer process lifetime before rotation |
| `--duration-hours H` | unbounded | Stop after H hours |
| `--no-hard-memory-limit` | auto | Skip cgroup memory scopes |
| `--skip-baseline` | off | Skip the startup seed execution |
| `--no-replay` | off | Do not re-run new findings in a fresh process |
| `--adopt` | off | Resume an output directory created on another machine |

At startup the runner:

- validates the distribution (manifest checksums, binaries, seeds,
  artifacts) and takes an exclusive lock on the output directory; a second
  runner on the same directory refuses to start;
- records the machine identity (hostname, `/etc/machine-id`, CPU model) and
  refuses to resume a directory from a different machine without `--adopt`;
- detects usable CPUs (affinity mask and cgroup `cpu.max`) and memory
  (`MemAvailable` and cgroup `memory.max`) and prints the chosen budget;
- copies seeds into the writable corpus and runs every lane once over its
  whole corpus (`-runs=0`): the **honest baseline**. For the end-to-end
  targets each seed is a complete commit/prove/verify of a shipped catalog
  row; any failure is recorded as a finding before fuzzing starts.

### Workers, threads, and scheduling

Each job is one libFuzzer process for one *lane* (a target or a variant such
as `ring_ntt@scalar`). The runner manages workers itself; libFuzzer's
`-fork`/`-jobs` are never used, so the only concurrency settings are the
number of workers and Akita's internal Rayon threads per worker
(`AKITA_FUZZ_THREADS`, from the registry's `threads`; the harness sizes the
global Rayon pool from it). A lane reserves `threads` CPU slots and its
`rss_limit_mb` of memory; jobs start only while both budgets have room.

Lanes are chosen by weighted deficit: the lane with the least CPU time per
unit `weight` runs next. Every job ends after `--slice-minutes`, which
rotates lanes and bounds any slow growth inside a process. Workers of one
target share a corpus directory (`-reload=1`).

### Limits

Per libFuzzer process, from the registry:

- `-timeout` (per input; exceeding it is a `timeout` finding),
- `-rss_limit_mb` (libFuzzer samples RSS about once per second),
- `-malloc_limit_mb` (checked on every allocation; catches unbounded
  allocation from untrusted lengths),
- `-max_len` sized to each target's input encoding.

When `systemd-run --user --scope` works (cgroup v2 with delegated memory
controller), every worker also runs in its own scope with
`MemoryMax = rss_limit_mb + 1024 MiB` and no swap: a **hard** OS limit.
Otherwise only the sampled libFuzzer limits apply and the runner says so at
startup and in `status`. No hard CPU-time limit is set per input beyond
libFuzzer's timeout; a wall-clock watchdog kills any job that outlives its
slice by more than twice the timeout plus a grace period.

### Failures and restarts

When a worker exits with a failure, the runner:

1. classifies it (`panic`, `asan`, `timeout`, `oom`, `leak`, `signal`,
   `crash`), computes a signature (panic location and message shape,
   sanitizer summary, or top Akita frames), and stores the raw libFuzzer
   artifact and the last 400 output lines under
   `findings/<kind>-<target>-<hash>/` (first five samples per signature;
   later occurrences only increment `count`). Originals are never deleted;
2. moves the crashing input out of the corpus into `quarantine/` if it is
   there, so the restarted worker does not loop on it;
3. for a new signature, replays the input once in a fresh process and records
   `reproducible` (`yes`, `no`, or `different signature`);
4. restarts the lane. Startup and infrastructure failures (no artifact, e.g.
   a missing file or an OS OOM kill) back off exponentially from 5 s to
   30 min per lane and are counted separately from findings.

Worker output is parsed as it arrives and written to per-job logs capped at
4 MiB with one rotated backup; `events.log` is capped the same way. The
output directory therefore grows with corpus and findings, not with time.

### Shutdown and resume

`SIGINT`/`SIGTERM` sends `SIGINT` to every worker (libFuzzer prints final
stats and exits), waits 30 s, then kills stragglers, saves `state.json`, and
releases the lock. A second signal forces immediate shutdown. Rerunning
`run` with the same `--output` resumes: corpora, findings, quarantine,
per-lane CPU time, and statistics persist. A new distribution build may
resume an old directory; `campaign.json` lists every build id used.

## 3. Inspecting (while running)

```bash
dist/akita-fuzz status --output /data/akita-fuzz          # table
dist/akita-fuzz status --output /data/akita-fuzz --json   # machine readable
```

`status` shows the budget and enforcement mode, per-lane CPU time,
executions, exec/s, coverage (`cov` edges, `ft` features), corpus size,
findings, failures, baseline result and backoff, every finding with its
reproducibility, and the harness **reach counters**: how often each target
reached its intended operation (for example `honest_proofs`,
`mutation_point`, `point_change_false`, `ring_outside_capacity`), plus time
spent generating inputs versus proving and verifying.

## 4. Exporting

```bash
dist/akita-fuzz export --output /data/akita-fuzz [--to DIR_OR.tar.gz] [--with-logs]
```

Writes `akita-fuzz-<host>-<campaign>-<time>.tar.gz` containing the corpus,
findings, quarantine, state, identity, and events log, rooted at
`<host>-<campaign>/` so archives from several machines unpack side by side.
It is safe while the campaign runs.

## 5. Reproducing and minimizing

```bash
dist/akita-fuzz reproduce --output /data/akita-fuzz <finding-id>
dist/akita-fuzz reproduce --output /data/akita-fuzz pcs_dense ./input.bin
dist/akita-fuzz minimize  --output /data/akita-fuzz <finding-id> [--seconds 300]
```

`reproduce` runs `bin/fuzz_all` with the finding's `AKITA_FUZZ_TARGET` on the stored samples with the
lane's environment and `RUST_BACKTRACE=full`. `minimize` runs libFuzzer's
`-minimize_crash` and writes `minimized-<time>.input` next to the original.

From a source checkout, `akita-fuzz-dev` replays any input through the same
target code without instrumentation, which is convenient under a debugger:

```bash
cd fuzz && cargo run --release -p akita-fuzz-dev -- replay pcs_dense input.bin
```

Rust panics are symbolized by the standard library from the binary's line
tables. AddressSanitizer, timeout, and out-of-memory stacks need
`llvm-symbolizer` for function names: install it on the build machine
(`apt install llvm`, then `prepare` bundles it) or the campaign machine.
Without it, stacks show `fuzz_all+0xOFFSET` frames and findings are
deduplicated by those offsets, which are stable within one build.
(binutils `addr2line` also works for manual triage but takes minutes on a
binary this size.)

## Developer commands

| Command | Purpose |
|---|---|
| `akita-fuzz targets` | Registry lanes and limits |
| `akita-fuzz-dev cases [LOG2]` | Catalog rows planned or excluded at a cost limit |
| `akita-fuzz-dev smoke TARGET [N] [SEED]` | N pseudo-random inputs, no libFuzzer |
| `akita-fuzz-dev replay TARGET FILE...` | Run inputs once, no libFuzzer |
| `akita-fuzz-dev seeds DIR` | Regenerate seed corpora |
| `akita-fuzz-dev list` | Library target names |

Run them with `cargo run --release -p akita-fuzz-dev -- <command>` from `fuzz/`.

`cargo fuzz run <target>` also works directly from `fuzz/` for quick local
sessions; it reads the schedule artifacts from the source tree.
`cargo fuzz run fuzz_all` without `AKITA_FUZZ_TARGET` fuzzes every target,
choosing one by the first input byte.

## Adding a target

1. Implement `pub fn run(data: &[u8])` in `src/targets/` and list it in
   `targets::ALL`.
2. Add `fuzz_targets/<name>.rs` forwarding to it and a `[[bin]]` entry (for
   `cargo fuzz run <name>`; the campaign uses `fuzz_all`).
3. Add `[target.<name>]` to `campaign/targets.toml`.

`prepare` refuses to package if the three disagree.

## Output directory layout

```
campaign.json   identity: campaign id, machine, build ids
state.json      per-lane totals (persisted every 15 s and at shutdown)
live.json       running jobs (present while running)
events.log      runner events (capped)
corpus/<target>/        shared, resumable corpora (libFuzzer SHA-1 names)
findings/<id>/          meta.json, sample-N.input, sample-N.txt, replay.txt
quarantine/<target>/    corpus inputs removed because they crash
logs/<lane>/            per-job libFuzzer output (capped)
exports/                archives written by `export`
```
