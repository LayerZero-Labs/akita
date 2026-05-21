# Offline schedule table generator: self-feedback regression

**Status**: known issue, not yet fixed.

## Symptom

Re-running the offline table generator
(`cargo run -p akita-config --features planner --bin gen_schedule_tables`)
**against an already-populated** family table can produce a strictly
worse schedule (larger proof, no verify improvement) than the first
generation from an empty table.

Observed on `fp128::D32OneHotFastVerify` at `nv = 32`:

| Source table state at regen | Suffix shape (level 1+) | Proof size at `nv = 32` |
|-----------------------------|--------------------------|--------------------------|
| Empty stub                  | `n_a = 2, log_basis ∈ {2, 4, 5}` | 63,172 B (+1,872 vs `D32OneHot`)  |
| Populated (previous good)   | `n_a = 3, log_basis ∈ {3, 4, 5}` | 67,972 B (+6,672 vs `D32OneHot`) |

Regenerating from the second state is stable, i.e. the planner
converges on a self-consistent — but suboptimal — fixed point. There
is no failure or warning; the new table simply has a different
suffix.

Reproduce on the current branch (uncommitted) with the
`fp128::D32OneHotFastVerify` preset:

```bash
# Good baseline — wipe the table first, then regen
cat > crates/akita-types/src/generated/fp128_d32_onehot_fast_verify.rs <<'EOF'
use super::{GeneratedScheduleTableEntry, GeneratedScheduleKey, GeneratedDirectStep, GeneratedFoldStep, GeneratedStep};
pub(crate) static FP128_D32_ONEHOT_FAST_VERIFY_SCHEDULES: &[GeneratedScheduleTableEntry] = &[];
EOF
cargo run --release -p akita-config --features planner --bin gen_schedule_tables -- \
    crates/akita-types/src/generated fp128_d32_onehot_fast_verify
# inspect nv=32 entry → suffix has n_a=2, log_basis 2 at the start

# Drift — regen on top of the populated table
cargo run --release -p akita-config --features planner --bin gen_schedule_tables -- \
    crates/akita-types/src/generated fp128_d32_onehot_fast_verify
# inspect nv=32 entry → suffix shifts to n_a=3, log_basis 4 at the start
```

## Root cause

`crates/akita-config/src/proof_optimized.rs`:

```155:165:crates/akita-config/src/proof_optimized.rs
pub(crate) fn proof_optimized_level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let singleton_key = AkitaScheduleLookupKey::singleton(inputs.num_vars);
    if let Ok(Some(plan)) = proof_optimized_schedule_plan::<Cfg>(singleton_key) {
        if let Ok(Some(planned_level)) =
            exact_planned_level_execution(&plan, inputs, log_basis, Cfg::stage1_challenge_config)
        {
            return planned_level.level.lp.clone();
        }
    }
    ...
```

The planner DP at a recursive level calls `Cfg::planner_current_level_layout_with_log_basis`,
which delegates to `Cfg::level_params_with_log_basis`, which calls
`proof_optimized_level_params_with_log_basis`. That hook **consults
the compiled-in offline schedule table** via
`Cfg::schedule_table()`/`Cfg::schedule_plan(key)`.

* **Production runtime** — correct behaviour. The table is the source of
  truth for the level's `LevelParams`; returning the pre-planned
  `lp` is exactly what the verifier expects.
* **Inside `bin/gen_schedule_tables`** — incorrect. The generator
  calls `find_optimal_schedule_from_scratch::<Cfg>(key)`, which
  passes `allow_offline_schedule = false` to the planner's *top-level*
  lookup, but this flag does **not** propagate down into
  `Cfg::level_params_with_log_basis`. So the DP for each recursive
  candidate is pinned to the `lp` already recorded in the table,
  and the search collapses to whatever's there.

The legacy `fp128::D32OneHot` table is unaffected only because it
was first generated from an empty stub. Any future regeneration of
that family is vulnerable to the same drift.

## Fix sketch

Two cheap options, either is acceptable:

1. **Thread-local / atomic regen flag.**
   * Add a `static OFFLINE_TABLE_LOOKUP_DISABLED: AtomicBool` in
     `crates/akita-config/src/proof_optimized.rs` plus a `pub fn
     disable_offline_schedule_lookup()`.
   * Gate the `let singleton_key = …; if let Ok(Some(plan)) = …`
     block on `!OFFLINE_TABLE_LOOKUP_DISABLED.load(Relaxed)`.
   * In `bin/gen_schedule_tables`'s `main`, call
     `disable_offline_schedule_lookup()` before any
     `emit_family_rows::<…>` call.
   * Production runtime is unaffected (flag stays `false`).

2. **Empty target tables before regen, in-process.**
   * Hard because `Cfg::schedule_table()` returns a `&'static`
     table compiled into the generator binary. Wiping the file on
     disk doesn't change the in-process static. To make this work
     we'd need to add a runtime override (e.g. `OnceLock`) on the
     `ScheduleProvider` impl, which is essentially option (1) with
     more surface area.

Option (1) is the recommended fix.

## Acceptance check

After the fix, the following must hold for every supported family:

```bash
cargo run --release -p akita-config --features planner --bin gen_schedule_tables -- \
    crates/akita-types/src/generated <family>
# Re-run with the just-written file in place
cargo run --release -p akita-config --features planner --bin gen_schedule_tables -- \
    crates/akita-types/src/generated <family>
git diff -- crates/akita-types/src/generated/<family>{,_zk}.rs
# must be empty
```

In other words: the generator must be **idempotent** with respect to
its own output. The current behaviour is not.
