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

What a schedule fixes per level (decomposition depth and ring/ext degrees),
the `LevelParams` representation, and the invariants the verifier
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
- `book/src/usage/profiling.md` and `.github/workflows/profile-bench.yml`.

### Selective physical L2 candidates

The coefficient `L∞` route remains available at every fold. A production
preset may also supply empirical calibration rows for physical `L2` response
planning. The shipped fp32, fp64, and fp128 dense and one-hot families all opt
in, including their generated multi-chunk and recursive companions.
A row with a physical response length of zero opts the family into the
balanced-digit response model. An exact row binds the fold level, incoming
witness length, source digit basis, challenge ring dimension, challenge
energy, response length, fold basis, and fold digit count. A row cannot cross
challenge families or reuse a measured source state under another basis.

The planner always retains its ordinary L infinity candidate. From level 3
onward, an opted-in family also evaluates the same canonical block split with
an L2 A matrix for response bases 16 and above. Basis 8 is not admitted because
it caused deterministic stage-2 folded-oracle consistency failures in two D64
production profiles. That geometry remains unsupported until the protocol cause
is fixed and validated. The planner keeps an eligible alternative only when it
lowers the A rank. This
adds at most one modeled L2 alternative per basis and dimension state, which
keeps the suffix search bounded.

An exact calibration takes precedence when its state key matches. It multiplies
the measured source energy and exact challenge energy by 1.25, leaving more
than 18 percent margin over the largest observed response-to-mean ratio. Other
eligible states use

```text
ceil(input_len * (B^2 + 2) / 12 * challenge_l2_sq * 1.75),
```

where `B` is the source digit basis. The balanced-digit second moment supplies
the base estimate. The 1.75 multiplier covers the largest source-energy and
response deviations seen in end-to-end fp32, fp64, and fp128 dense and one-hot
samples, with at
least 15 percent remaining margin over every recorded response maximum.

The suffix comparison includes the norm proof, A payload, next witness, later
folds, and terminal response. A smaller A rank can reduce the next witness
enough to remove a fold, but the planner keeps `L∞` when the extra norm proof
costs more than the suffix saves.

This model affects completeness and schedule selection only. Once the planner
selects a route, its concrete cap is frozen into the generated schedule. The
prover rejection samples against that cap and the verifier enforces the same
value. The SIS calculation therefore still uses the public accepted cap and
does not trust the statistical model.

A clear terminal L2 candidate has no recursive norm proof. The verifier checks
the decoded response norm directly. The planner may use the certified energy
to estimate a smaller Golomb payload for candidate comparison. The scheduled
Golomb byte cap and the payload grind remain unchanged.

Generated schedule identity includes the cap policy and the separate L2 table
digest. Runtime expansion derives the route, cap, proof shape, and A rank from
that identity. A mismatch between the preset policy and generated catalog is
an error.
