# Configuration and planning

How a preset turns into a concrete recursion schedule: the single
`CommitmentConfig` trait, the `LevelParams` it produces, and the planner that
selects (or searches for) the schedule and prices its proof size.

## CommitmentConfig and presets

The single user-facing trait that defines every per-config policy hook (algebra,
exact SIS profile, decomposition, layout, schedule, transcript bind, prove params), and
the `fp32` / `fp64` / `fp128` preset families built on it.

Both field roles live on the trait: `Field` carries committed witnesses, setup
matrices, and SIS, while `ExtField` carries public opening points, claimed
evaluations, and Fiat-Shamir challenges. The protocol geometry gates on
whether the two roles coincide (`EXT_DEGREE == 1`, all `fp128` presets) or
claims live in a proper extension (`EXT_DEGREE > 1`, `fp32` / `fp64`), never
on field bit-width. See
[Fold path and field geometry](./proving/fold-path.md).

**Implementation map**

- `crates/akita-config/src/lib.rs:54-120`.
- `crates/akita-config/src/proof_optimized/`.
- [`crates/akita-planner/README.md`](../../../crates/akita-planner/README.md) for the current planner/config boundary.

## Schedule and LevelParams

What a schedule fixes per level (fold shape, decomposition depth, ring/ext
degrees), the `LevelParams` representation, and the invariants the verifier
re-derives rather than trusts.

**Implementation map**

- `crates/akita-types/src/layout/params.rs:41-97`.
- `crates/akita-types/src/schedule.rs` (`FoldStep`, `TerminalWitnessPlan`, `Schedule`).
- Paper §3.11 `sec:akita-planner` ("What the schedule fixes").
- Council architecture + newcomer reports (schedule invariants, level overload).

## The planner and proof size

The `Cfg`-free planner: catalog validation, on-demand compact→`LevelParams`
expansion, and the schedule-search DP fallback (verifier-reachable, so it must
reject malformed input, never panic). The feature-gated `akita-schedules` crate
owns shipped table data. The verifier-reachable proof-size formula.

**Implementation map**

- [`crates/akita-planner/README.md`](../../../crates/akita-planner/README.md) for the current planner overview, search model, generated tables, and supported features.
- `crates/akita-planner/src/` owns search and emission. Runtime catalog
  expansion and audit live in `crates/akita-schedules/src/`.
- `crates/akita-types/src/proof_size.rs` and `crates/akita-types/src/layout/proof_size.rs` (`level_proof_bytes`, planned witness sizing).
- Paper §3.11 `sec:akita-planner` (objective/constraints, the dynamic program, generated schedules).
- `crates/akita-planner/src/generated_families.rs`,
  `crates/akita-schedules/src/generated/`, and
  `crates/akita-config/src/lib.rs` (`resolve_schedule`).
- `book/src/usage/profiling.md`, `specs/profile-bench-coverage-matrix.md`, `.github/workflows/profile-bench.yml`.

### Selective physical L2 candidates

The coefficient `L∞` route remains the default at every fold. A preset may
also supply a squared physical `L2` cap for an exact later nonterminal fold
candidate. The cap key includes the fold level, incoming witness length,
physical response length, fold basis, and fold digit count. A cap does not
apply to another row or to a successor whose witness length changed.

The planner keeps every bounded split on a path that can still reach a measured
L2 state. At the predecessor level it keeps every candidate whose output
witness is an exact measured input, alongside the ordinary best L-infinity
candidate per secure A rank. At the measured state it keeps every L-infinity
and L2 route through complete suffix pricing. Unrelated L-infinity-only states
retain the established rank frontier, so selective search does not multiply
the whole dynamic program.

The measured rollout window starts at fold level 3 and later measured levels
are contiguous. Runtime policy validation rejects a later first level or a gap.
This bounds complete split retention to the short measured window while still
comparing every route that can enter or leave one of its exact measured states.

The suffix comparison includes the norm proof, A payload, next witness, later
folds, and terminal response. A smaller A rank can reduce the next witness
enough to remove a fold, but the planner keeps `L∞` when the extra norm proof
costs more than the suffix saves.

Generated schedule identity includes the cap policy and the separate L2 table
digest. Runtime expansion derives the route, cap, proof shape, and A rank from
that identity. A mismatch between the preset policy and generated catalog is
an error.
