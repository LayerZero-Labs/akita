# Configuration and planning

> **Status:** stub. Part of the initial Akita Book scaffold.

How a preset turns into a concrete recursion schedule: the single
`CommitmentConfig` trait, the `LevelParams` it produces, and the planner that
selects (or searches for) the schedule and prices its proof size.

## CommitmentConfig and presets

The single user-facing trait that defines every per-config policy hook (algebra,
SIS family, decomposition, layout, schedule, transcript bind, prove params), and
the `fp32` / `fp64` / `fp128` preset families built on it.

**Sources to fold in**

- `crates/akita-config/src/lib.rs:54-120`.
- `crates/akita-config/src/proof_optimized/`.
- `specs/planner-config-consolidation.md` (caveat: largely superseded — verify against code).

## Schedule and LevelParams

What a schedule fixes per level (fold shape, decomposition depth, ring/ext
degrees), the `LevelParams` representation, and the invariants the verifier
re-derives rather than trusts.

**Sources to fold in**

- `crates/akita-types/src/layout/params.rs:41-97`.
- `crates/akita-types/src/schedule.rs` (`Step`, `FoldStep`, `DirectStep`).
- Paper §3.10 `sec:akita-planner` ("What the schedule fixes").
- Council architecture + newcomer reports (schedule invariants, level overload).

## The planner and proof size

The `Cfg`-free planner: shipped table lookup, on-demand compact→`LevelParams`
expansion, and the schedule-search DP fallback (verifier-reachable, so it must
reject malformed input, never panic). The verifier-reachable proof-size formula.

**Sources to fold in**

- `crates/akita-planner/src/` (`resolve.rs`, `find_schedule`, `generated/`).
- `crates/akita-types/src/proof_size.rs` and `crates/akita-types/src/layout/proof_size.rs` (`level_proof_bytes`, planned witness sizing).
- Paper §3.10 `sec:akita-planner` (objective/constraints, the dynamic program, generated schedules).
- `AGENTS.md` (Profiling: D32/D64/D128 optimum), `specs/planner-incidence-generalization.md` (active tail).
