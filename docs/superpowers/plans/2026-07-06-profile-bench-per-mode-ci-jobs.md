# Per-Mode CI Compile Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each `profile-bench.yml` matrix job its own Cargo feature so it only compiles the `profile` example's generic mode instantiations it actually benchmarks, instead of all 8, cutting that binary's compile time from ~94s to ~29-48s per job.

**Architecture:** Split the existing blanket `profile-ci` feature into 8 per-mode features (`mode-<name>`) plus a `profile-ci-registry` marker feature that selects a Vec-based, individually-gated mode registry in `modes.rs` instead of the current single all-or-nothing array. `profile-ci` becomes shorthand for "enable the registry + every mode," preserving today's behavior for anyone still using it. The CI workflow then requests only the specific `mode-*` features each matrix group's cases need.

**Tech Stack:** Rust (Cargo features, `cfg` attributes), GitHub Actions YAML, Python (existing `check_profile_ci_features.sh` helper script).

## Global Constraints

- `cargo build --release --example profile --no-default-features --features parallel,profile-ci` (today's exact CI invocation) must keep working and keep including all 8 modes — no regressions for the umbrella feature.
- Each mode's Cargo feature must forward to the matching existing `akita-config/schedules-*` feature (do not introduce new schedule-table features; reuse what exists).
- `AKITA_MODE=<name>` selection in `main.rs`/`modes.rs` must keep working unchanged — only which entries populate the registry changes, not how lookup works.
- No new test framework — verification is: build succeeds, `cargo run` with a bad `AKITA_MODE` reports the exact expected known-modes list, and existing shell check scripts still pass.

---

## File Structure

- Modify: `crates/akita-pcs/Cargo.toml` — add 8 `mode-*` features + `profile-ci-registry` marker; redefine `profile-ci` as their union.
- Modify: `crates/akita-pcs/examples/profile/modes.rs` — rename the existing CI/dev branch predicate from `profile-ci` to `profile-ci-registry`; replace the single `PROFILE_CI_MODES` const with a per-mode-gated `Vec` builder inside `profile_modes()`.
- Modify: `scripts/check_profile_ci_features.sh` — update its `MODE_FEATURE` table to point at the new `mode-*` feature names instead of raw `akita-config/schedules-*` names.
- Modify: `.github/workflows/profile-bench.yml` — add `pcs_mode_features` to each matrix group; use it in both build steps with a 3-tier fallback for the merge-base build.

---

### Task 1: Add per-mode Cargo features to `akita-pcs`

**Files:**
- Modify: `crates/akita-pcs/Cargo.toml:48-57`

**Interfaces:**
- Produces: Cargo features `profile-ci-registry`, `mode-dense-fp128-d64`, `mode-onehot-fp128-d64`, `mode-onehot-fp128-d64-tensor`, `mode-onehot-fp128-d64-multi-chunk-w2r2`, `mode-onehot-fp128-d64-multi-chunk-w4r2`, `mode-onehot-fp128-d64-multi-chunk-w8r2`, `mode-onehot-fp32-d128`, `mode-onehot-fp64-d128` — consumed by Task 2's `modes.rs` cfg gates and Task 4's workflow feature lists.

- [ ] **Step 1: Replace the `profile-ci` feature block**

Current content at `crates/akita-pcs/Cargo.toml:48-57`:

```toml
profile-ci = [
  "akita-config/schedules-fp32-d128-onehot",
  "akita-config/schedules-fp64-d128-onehot",
  "akita-config/schedules-fp128-d64-onehot",
  "akita-config/schedules-fp128-d64-onehot-tensor",
  "akita-config/schedules-fp128-d64-full",
  "akita-config/schedules-fp128-d64-onehot-multi-chunk",
  "akita-config/schedules-fp128-d64-onehot-multi-chunk-w2r2",
  "akita-config/schedules-fp128-d64-onehot-multi-chunk-w4r2",
]
```

Replace with:

```toml
profile-ci = [
  "profile-ci-registry",
  "mode-dense-fp128-d64",
  "mode-onehot-fp128-d64",
  "mode-onehot-fp128-d64-tensor",
  "mode-onehot-fp128-d64-multi-chunk-w2r2",
  "mode-onehot-fp128-d64-multi-chunk-w4r2",
  "mode-onehot-fp128-d64-multi-chunk-w8r2",
  "mode-onehot-fp32-d128",
  "mode-onehot-fp64-d128",
]
# Selects the CI mode registry in examples/profile/modes.rs (a curated,
# individually-gated subset) instead of the full local-dev registry. CI jobs
# enable this plus only the `mode-*` features their matrix cases exercise, so
# unused generic prover/verifier instantiations are never monomorphized.
profile-ci-registry = []
# Each mirrors one CI profile mode in examples/profile/modes.rs. Forwards to
# the existing akita-config schedule-table feature so the mode's
# `CommitmentConfig::runtime_schedule` has data to look up.
mode-dense-fp128-d64 = ["akita-config/schedules-fp128-d64-full"]
mode-onehot-fp128-d64 = ["akita-config/schedules-fp128-d64-onehot"]
mode-onehot-fp128-d64-tensor = ["akita-config/schedules-fp128-d64-onehot-tensor"]
mode-onehot-fp128-d64-multi-chunk-w2r2 = [
  "akita-config/schedules-fp128-d64-onehot-multi-chunk-w2r2",
]
mode-onehot-fp128-d64-multi-chunk-w4r2 = [
  "akita-config/schedules-fp128-d64-onehot-multi-chunk-w4r2",
]
mode-onehot-fp128-d64-multi-chunk-w8r2 = [
  "akita-config/schedules-fp128-d64-onehot-multi-chunk",
]
mode-onehot-fp32-d128 = ["akita-config/schedules-fp32-d128-onehot"]
mode-onehot-fp64-d128 = ["akita-config/schedules-fp64-d128-onehot"]
```

- [ ] **Step 2: Verify the feature graph resolves**

Run: `cargo metadata --format-version=1 --no-default-features --features akita-pcs/profile-ci -q > /dev/null && echo OK`
Expected: `OK` (proves the new feature names don't have typos/cycles; this does not yet build anything since Task 2 hasn't changed `modes.rs`).

- [ ] **Step 3: Commit**

```bash
git add crates/akita-pcs/Cargo.toml
git commit -m "feat(akita-pcs): split profile-ci into per-mode Cargo features"
```

---

### Task 2: Make `modes.rs`'s CI registry selectively buildable per mode

**Files:**
- Modify: `crates/akita-pcs/examples/profile/modes.rs:45,159-198,200,280-289,348,433,445,452,466,473,480,488,502,509,522,530`

**Interfaces:**
- Consumes: Cargo features from Task 1 (`profile-ci-registry`, `mode-*`).
- Produces: `fn profile_modes() -> Vec<ProfileMode>` (signature change from `&'static [ProfileMode]`) — consumed by the existing callers `run_profile_mode` (`modes.rs:538`) and `run_all_profile_modes` (`modes.rs:556`), both of which only call `.iter()`/`for entry in ...` and work unchanged against a `Vec`.

- [ ] **Step 1: Rename the branch predicate from `profile-ci` to `profile-ci-registry`**

This is a global, exact string replacement — every occurrence of `feature = "profile-ci"` (as a `cfg` predicate string, 17 occurrences) becomes `feature = "profile-ci-registry"`. This keeps the local-dev "all 19 modes" branch (currently gated `#[cfg(not(feature = "profile-ci"))]`) mutually exclusive with the new selective CI branch, which will be gated on the new marker feature, not on `profile-ci` itself (since a job will enable specific `mode-*` features without enabling the full `profile-ci` union).

Run:
```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("crates/akita-pcs/examples/profile/modes.rs")
text = p.read_text()
old = 'feature = "profile-ci"'
new = 'feature = "profile-ci-registry"'
count = text.count(old)
assert count == 17, f"expected 17 occurrences, found {count}"
p.write_text(text.replace(old, new))
print(f"replaced {count} occurrences")
PY
```
Expected output: `replaced 17 occurrences`

- [ ] **Step 2: Add `#[derive(Clone, Copy)]` to `ProfileMode`**

At `modes.rs:159-162`, change:

```rust
struct ProfileMode {
    name: &'static str,
    run: ProfileModeRunner,
}
```

to:

```rust
#[derive(Clone, Copy)]
struct ProfileMode {
    name: &'static str,
    run: ProfileModeRunner,
}
```

(Needed for `PROFILE_ALL_MODES.to_vec()` in Step 4 below.)

- [ ] **Step 3: Delete the `PROFILE_CI_MODES` const**

Delete the entire block (now reading `#[cfg(feature = "profile-ci-registry")]` after Step 1's rename) at what was `modes.rs:164-198`:

```rust
#[cfg(feature = "profile-ci-registry")]
const PROFILE_CI_MODES: &[ProfileMode] = &[
    ProfileMode {
        name: "dense_fp128_d64",
        run: run_profile_dense_fp128_d64,
    },
    ProfileMode {
        name: "onehot_fp128_d64",
        run: run_profile_onehot_fp128_d64,
    },
    ProfileMode {
        name: "onehot_fp128_d64_tensor",
        run: run_profile_onehot_fp128_d64_tensor,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w2r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w2r2,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w4r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w4r2,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w8r2,
    },
    ProfileMode {
        name: "onehot_fp32_d128",
        run: run_profile_onehot_fp32_d128,
    },
    ProfileMode {
        name: "onehot_fp64_d128",
        run: run_profile_onehot_fp64_d128,
    },
];

```

Remove it entirely (including the trailing blank line), leaving `PROFILE_ALL_MODES`'s `#[cfg(not(feature = "profile-ci-registry"))]` block immediately following the `ProfileMode` struct.

- [ ] **Step 4: Rewrite `profile_modes()`**

Replace (what was `modes.rs:280-289`, now with the renamed predicate):

```rust
fn profile_modes() -> &'static [ProfileMode] {
    #[cfg(feature = "profile-ci-registry")]
    {
        PROFILE_CI_MODES
    }
    #[cfg(not(feature = "profile-ci-registry"))]
    {
        PROFILE_ALL_MODES
    }
}
```

with:

```rust
fn profile_modes() -> Vec<ProfileMode> {
    #[cfg(feature = "profile-ci-registry")]
    {
        let mut modes = Vec::new();
        #[cfg(feature = "mode-dense-fp128-d64")]
        modes.push(ProfileMode {
            name: "dense_fp128_d64",
            run: run_profile_dense_fp128_d64,
        });
        #[cfg(feature = "mode-onehot-fp128-d64")]
        modes.push(ProfileMode {
            name: "onehot_fp128_d64",
            run: run_profile_onehot_fp128_d64,
        });
        #[cfg(feature = "mode-onehot-fp128-d64-tensor")]
        modes.push(ProfileMode {
            name: "onehot_fp128_d64_tensor",
            run: run_profile_onehot_fp128_d64_tensor,
        });
        #[cfg(feature = "mode-onehot-fp128-d64-multi-chunk-w2r2")]
        modes.push(ProfileMode {
            name: "onehot_fp128_d64_multi_chunk_w2r2",
            run: run_profile_onehot_fp128_d64_multi_chunk_w2r2,
        });
        #[cfg(feature = "mode-onehot-fp128-d64-multi-chunk-w4r2")]
        modes.push(ProfileMode {
            name: "onehot_fp128_d64_multi_chunk_w4r2",
            run: run_profile_onehot_fp128_d64_multi_chunk_w4r2,
        });
        #[cfg(feature = "mode-onehot-fp128-d64-multi-chunk-w8r2")]
        modes.push(ProfileMode {
            name: "onehot_fp128_d64_multi_chunk_w8r2",
            run: run_profile_onehot_fp128_d64_multi_chunk_w8r2,
        });
        #[cfg(feature = "mode-onehot-fp32-d128")]
        modes.push(ProfileMode {
            name: "onehot_fp32_d128",
            run: run_profile_onehot_fp32_d128,
        });
        #[cfg(feature = "mode-onehot-fp64-d128")]
        modes.push(ProfileMode {
            name: "onehot_fp64_d128",
            run: run_profile_onehot_fp64_d128,
        });
        modes
    }
    #[cfg(not(feature = "profile-ci-registry"))]
    {
        PROFILE_ALL_MODES.to_vec()
    }
}
```

- [ ] **Step 5: Build with the full umbrella feature and verify no regression**

Run: `cargo build --release --quiet --example profile --no-default-features --features parallel,profile-ci -p akita-pcs`
Expected: builds with no errors (warnings about unused `run_profile_*` functions are fine and expected for whichever functions a later, narrower build excludes).

- [ ] **Step 6: Verify the mode list is correct for the umbrella feature**

Run:
```bash
AKITA_ALLOW_DEBUG_PROFILE=1 AKITA_MODE=bogus AKITA_PROFILE_TRACE=0 \
  cargo run --release --quiet --example profile --no-default-features --features parallel,profile-ci -p akita-pcs 2>&1 | grep known_modes
```
Expected: a line containing `known_modes="dense_fp128_d64, onehot_fp128_d64, onehot_fp128_d64_tensor, onehot_fp128_d64_multi_chunk_w2r2, onehot_fp128_d64_multi_chunk_w4r2, onehot_fp128_d64_multi_chunk_w8r2, onehot_fp32_d128, onehot_fp64_d128, all"` (all 8 modes, same set as before this change).

- [ ] **Step 7: Build with a single narrow mode feature and verify only that mode is registered**

Run:
```bash
cargo build --release --quiet --example profile \
  --no-default-features --features parallel,profile-ci-registry,mode-dense-fp128-d64 -p akita-pcs
AKITA_ALLOW_DEBUG_PROFILE=1 AKITA_MODE=bogus AKITA_PROFILE_TRACE=0 \
  cargo run --release --quiet --example profile \
  --no-default-features --features parallel,profile-ci-registry,mode-dense-fp128-d64 -p akita-pcs 2>&1 | grep known_modes
```
Expected: `known_modes="dense_fp128_d64, all"` — exactly one mode registered, confirming the per-feature gating works.

- [ ] **Step 8: Commit**

```bash
git add crates/akita-pcs/examples/profile/modes.rs
git commit -m "feat(akita-pcs): make the profile-ci mode registry per-mode selectable"
```

---

### Task 3: Update `check_profile_ci_features.sh` to the new feature names

**Files:**
- Modify: `scripts/check_profile_ci_features.sh`

**Interfaces:**
- Consumes: the `profile-ci` array shape written in Task 1 (now lists `mode-*` + `profile-ci-registry` instead of `akita-config/schedules-*`).

- [ ] **Step 1: Update the `MODE_FEATURE` table**

In `scripts/check_profile_ci_features.sh`, change:

```python
MODE_FEATURE = {
    "onehot_fp32_d128": "schedules-fp32-d128-onehot",
    "onehot_fp64_d128": "schedules-fp64-d128-onehot",
    "dense_fp128_d64": "schedules-fp128-d64-full",
    "onehot_fp128_d64": "schedules-fp128-d64-onehot",
    "onehot_fp128_d64_tensor": "schedules-fp128-d64-onehot-tensor",
    "onehot_fp128_d64_multi_chunk_w8r2": "schedules-fp128-d64-onehot-multi-chunk",
    "onehot_fp128_d64_multi_chunk_w2r2": "schedules-fp128-d64-onehot-multi-chunk-w2r2",
    "onehot_fp128_d64_multi_chunk_w4r2": "schedules-fp128-d64-onehot-multi-chunk-w4r2",
}
```

to:

```python
MODE_FEATURE = {
    "onehot_fp32_d128": "mode-onehot-fp32-d128",
    "onehot_fp64_d128": "mode-onehot-fp64-d128",
    "dense_fp128_d64": "mode-dense-fp128-d64",
    "onehot_fp128_d64": "mode-onehot-fp128-d64",
    "onehot_fp128_d64_tensor": "mode-onehot-fp128-d64-tensor",
    "onehot_fp128_d64_multi_chunk_w8r2": "mode-onehot-fp128-d64-multi-chunk-w8r2",
    "onehot_fp128_d64_multi_chunk_w2r2": "mode-onehot-fp128-d64-multi-chunk-w2r2",
    "onehot_fp128_d64_multi_chunk_w4r2": "mode-onehot-fp128-d64-multi-chunk-w4r2",
}
```

(The `profile-ci` parser in this script already strips a leading `akita-config/`-style `pkg/` prefix if present via `if "/" in line: line = line.split("/", 1)[1]`; the new `mode-*` entries in `profile-ci` have no `/`, so that branch is simply skipped — no other script logic needs to change.)

- [ ] **Step 2: Run the script and verify it passes**

Run: `./scripts/check_profile_ci_features.sh`
Expected: `profile-ci feature coverage check passed.`

- [ ] **Step 3: Commit**

```bash
git add scripts/check_profile_ci_features.sh
git commit -m "chore(ci): point profile-ci coverage check at per-mode feature names"
```

---

### Task 4: Split the CI matrix build to use per-group mode features

**Files:**
- Modify: `.github/workflows/profile-bench.yml:80-140,194-225`

**Interfaces:**
- Consumes: `mode-*` features from Task 1, `profile-ci-registry` marker from Task 1/2.

- [ ] **Step 1: Add `pcs_mode_features` to each matrix group**

At `.github/workflows/profile-bench.yml:80-113`, add a `pcs_mode_features` key to each of the 4 groups (comma-separated feature list, no spaces, matching exactly the modes each group's `cases` uses):

```yaml
        group:
          - name: 1-fp128-dense
            pcs_mode_features: mode-dense-fp128-d64
            cases: |
              dense_fp128_d64:24:1
          - name: 2-fp128-tensor
            skip_merge_base_baseline: true
            pcs_mode_features: mode-onehot-fp128-d64-tensor
            cases: |
              onehot_fp128_d64_tensor:26:1
          - name: 3-flat-onehot-suite
            pcs_mode_features: mode-onehot-fp32-d128,mode-onehot-fp64-d128,mode-onehot-fp128-d64
            cases: |
              onehot_fp32_d128:28:1
              onehot_fp64_d128:28:1
              onehot_fp128_d64:32:1
              onehot_fp128_d64:32:1:recursive
              onehot_fp128_d64:30:4
          - name: 4-distributed
            baseline_required_mode: onehot_fp128_d64_multi_chunk_w8r2
            pcs_mode_features: mode-onehot-fp128-d64-multi-chunk-w2r2,mode-onehot-fp128-d64-multi-chunk-w4r2,mode-onehot-fp128-d64-multi-chunk-w8r2
            cases: |
              onehot_fp128_d64_multi_chunk_w2r2:32:1
              onehot_fp128_d64_multi_chunk_w4r2:32:1
              onehot_fp128_d64_multi_chunk_w8r2:32:1
```

(Only the new `pcs_mode_features:` lines are additions; everything else in this block is unchanged from today.)

- [ ] **Step 2: Export the resolved feature list in "Initialize benchmark paths"**

At `.github/workflows/profile-bench.yml:119-140`, add one line after the existing `AKITA_BENCH_BASE_REQUIRED_MODE` export:

```yaml
          echo "AKITA_BENCH_BASE_REQUIRED_MODE=${{ matrix.group.baseline_required_mode }}" >> "$GITHUB_ENV"
          echo "AKITA_BENCH_PCS_MODE_FEATURES=${{ matrix.group.pcs_mode_features }}" >> "$GITHUB_ENV"
```

- [ ] **Step 3: Update "Build merge-base profile binary" with a 3-tier fallback**

At `.github/workflows/profile-bench.yml:194-217`, replace the build logic inside the `(cd ...)` subshell:

Current:
```bash
          (
            cd "$RUNNER_TEMP/bench-base" &&
            if python3 scripts/cargo_feature_exists.py akita-pcs profile-ci; then
              CARGO_TARGET_DIR="$RUNNER_TEMP/bench-base-target" \
                cargo build --release --quiet --example profile \
                  --no-default-features --features parallel,profile-ci
            else
              CARGO_TARGET_DIR="$RUNNER_TEMP/bench-base-target" \
                cargo build --release --quiet --example profile
            fi
          )
```

Replace with:
```bash
          (
            cd "$RUNNER_TEMP/bench-base" &&
            # 3-tier fallback so a merge-base commit that predates this PR's
            # per-mode features still builds a usable (if slower) baseline:
            # 1) merge-base already has the specific mode features (steady
            #    state) -> fast, narrow build.
            # 2) merge-base has profile-ci but not yet the per-mode split
            #    (the PR introducing this split) -> slow, full-8-mode build.
            # 3) merge-base predates profile-ci entirely -> default-features
            #    build, same as before this change.
            if python3 scripts/cargo_feature_exists.py akita-pcs profile-ci-registry; then
              CARGO_TARGET_DIR="$RUNNER_TEMP/bench-base-target" \
                cargo build --release --quiet --example profile \
                  --no-default-features \
                  --features "parallel,profile-ci-registry,$AKITA_BENCH_PCS_MODE_FEATURES"
            elif python3 scripts/cargo_feature_exists.py akita-pcs profile-ci; then
              CARGO_TARGET_DIR="$RUNNER_TEMP/bench-base-target" \
                cargo build --release --quiet --example profile \
                  --no-default-features --features parallel,profile-ci
            else
              CARGO_TARGET_DIR="$RUNNER_TEMP/bench-base-target" \
                cargo build --release --quiet --example profile
            fi
          )
```

- [ ] **Step 4: Update "Build profile binary" (PR side) to use the narrow features**

At `.github/workflows/profile-bench.yml:219-222`, replace:

```yaml
      - name: Build profile binary
        run: |
          cargo build --release --quiet --example profile \
            --no-default-features --features parallel,profile-ci
```

with:

```yaml
      - name: Build profile binary
        run: |
          cargo build --release --quiet --example profile \
            --no-default-features \
            --features "parallel,profile-ci-registry,$AKITA_BENCH_PCS_MODE_FEATURES"
```

(The PR build has no fallback: this PR's own checkout always has the per-mode features from Tasks 1-2.)

- [ ] **Step 5: Validate the YAML**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/profile-bench.yml'))" && echo OK`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/profile-bench.yml
git commit -m "ci(profile-bench): build each matrix group with only its needed modes"
```

---

### Task 5: Local verification of compile-time savings and correctness

**Files:**
- None (verification only, no source changes).

**Interfaces:**
- Consumes: everything from Tasks 1-4.

- [ ] **Step 1: Clean-build each matrix group's exact feature set and record timing**

Run (repeat per group, using a fresh `CARGO_TARGET_DIR` each time to force a clean build of the example unit, matching how CI always builds from a fresh checkout):

```bash
for spec in \
  "1-fp128-dense:mode-dense-fp128-d64" \
  "2-fp128-tensor:mode-onehot-fp128-d64-tensor" \
  "3-flat-onehot-suite:mode-onehot-fp32-d128,mode-onehot-fp64-d128,mode-onehot-fp128-d64" \
  "4-distributed:mode-onehot-fp128-d64-multi-chunk-w2r2,mode-onehot-fp128-d64-multi-chunk-w4r2,mode-onehot-fp128-d64-multi-chunk-w8r2" \
; do
  name="${spec%%:*}"
  features="${spec#*:}"
  target_dir="$(mktemp -d)/target"
  echo "=== $name ($features) ==="
  time CARGO_TARGET_DIR="$target_dir" cargo build --release --example profile \
    --no-default-features --features "parallel,profile-ci-registry,$features" -p akita-pcs
  rm -rf "$(dirname "$target_dir")"
done
```

Expected: each group's build completes; group 1 and 2 (1 mode) finish noticeably faster than group 3 and 4 (3 modes), and all are faster than the ~107s full-8-mode baseline measured before this change.

- [ ] **Step 2: Confirm the umbrella `profile-ci` build still produces all 8 modes**

Run: (same command as Task 2 Step 6)
Expected: same 8-mode `known_modes` list as before this change — no behavior change for the umbrella feature.

- [ ] **Step 3: Run both existing profile-ci check scripts against a per-group binary**

Run:
```bash
cargo build --release --quiet --example profile \
  --no-default-features --features parallel,profile-ci-registry,mode-dense-fp128-d64 -p akita-pcs
./scripts/check_profile_ci_linkage.sh target/release/examples/profile
./scripts/check_profile_ci_features.sh
```
Expected: `profile-ci linkage smoke check passed.` and `profile-ci feature coverage check passed.` (the linkage check's forbidden-symbol list is a subset check against schedules outside the full profile-ci union, so it stays valid — and now conservatively more true — against any narrower per-group binary).

- [ ] **Step 4: Commit** (only if any of the above surfaced a fix; otherwise skip — this task is verification-only)

---

## Self-Review

**Spec coverage:**
- "Explain where source code is, how modes are configured now" → covered directly in chat, and in Task 1-2's file/line references.
- "How will we configure them in future" → Task 1 (Cargo features) + Task 2 (registry).
- "Group per mode ... faster compile time" → Task 4 (workflow matrix `pcs_mode_features` per group) + Task 5 (timing verification).
- Backward compatibility for the umbrella `profile-ci` feature and for old merge-base commits during the transitional PR → Task 1 Step 1 (`profile-ci` = union), Task 4 Step 3 (3-tier fallback).
- Existing coverage/linkage gates must keep passing → Task 3 (features script), Task 5 Step 3 (linkage script).

**Placeholder scan:** none found — every step has literal code/commands and expected output.

**Type consistency:** `profile_modes()` signature change (`&'static [ProfileMode]` → `Vec<ProfileMode>`) is introduced once in Task 2 Step 4 and its two callers (`run_profile_mode`, `run_all_profile_modes`) are confirmed compatible without modification (both only use `.iter()`/`for ... in`, which works identically on `Vec` and `&[T]`).
