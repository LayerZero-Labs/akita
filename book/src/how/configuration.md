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
  `crates/akita-schedules/src/resolve.rs` (`select_generated_schedule_row`).
- `book/src/usage/profiling.md` and `.github/workflows/profile-bench.yml`.

### Selective L2 candidates

The coefficient `L∞` route remains available at every fold. A production
preset may also enable the typed `L2` response model. Every shipped
fp32, fp64, and fp128 dense and one-hot family enables it. This includes each
generated multi-chunk and recursive companion.

The planner always retains the universal L infinity candidate. When typed
moments are available, it also evaluates the modeled L infinity depth. From
level 3 onward, an enabled family evaluates the same canonical block split with
an L2 A matrix for response bases 8 and above. The planner keeps an eligible L2
alternative only when it lowers the A rank. Basis 8 uses the same fused norm and
range-image leaf as larger bases, with its class-indexed source prepared lazily
because it has no product-stage prefix.

The planner estimates the squared norm of the actual recursive witness. It
applies the following rules.

* A dense root uses the deterministic maximum squared digit energy for every
  coefficient. A one-hot root maximizes the physical squared energy over every
  hot position allowed by its policy-owned chunk contract. This follows the
  tensor projection kernel and includes coefficient collisions.
* The Z part uses the centered residues of a rounded normal variable. Its
  variance comes from the previous source energy and the challenge energy.
* The E, T, and R parts use the centered field digit moment for every live
  scalar. The final digit plane uses its actual remaining field width.
* Negative binary compression contributes one half unit of expected energy per
  coefficient.
* Extension tensor packing multiplies the logical energy by `(2K - 1) / K`,
  where `K` is the extension degree.

The planner rounds each source estimate upward while retaining seven leading
bits. This adds less than `1/64` relative error and keeps the suffix search
small. It then multiplies the source estimate by the challenge squared energy,
a 1.03 model envelope, and a `40/39` response allowance. If the model envelope
bounds the conditional mean, Markov's inequality gives at least `1/40`
acceptance probability on each independent attempt. The protocol permits 4096
attempts, so the resulting exhaustion bound for one response is below
`2^-149`.

The 1.03 factor covers approximations in the normal, field digit, challenge
covariance, and finite mixing models. It is an empirical completeness margin,
not a soundness claim. Response-model diagnostics measure exact source and
response energy in complete production proofs. The benchmark parser joins each
measurement to the planned fold in the same run, rejects a successful run whose
response exceeds its frozen cap, and records cap utilization and nonce attempts
for every L2 fold. Historical measurements are evidence, not compiled unit
test constants.

The field digit model is exact for uniform power of two residues, apart from
the negligible pseudo-Mersenne boundary. Recursive setup values can retain
correlation. This usually lowers their E, T, and R energy, so the model is
conservative. Separate component validation found at most 2.24 percent
unfavorable error.

The suffix comparison includes the norm proof, A payload, next witness, later
folds, and terminal response. A smaller A rank can reduce the next witness
enough to remove a fold, but the planner keeps `L∞` when the extra norm proof
costs more than the suffix saves.

This model affects completeness and schedule selection only. Once the planner
selects a route, its concrete cap is frozen into the generated schedule. The
prover rejection samples against that cap and the verifier enforces the same
value. The SIS calculation therefore still uses the public accepted cap and
does not trust the statistical model.

If the typed model is disabled, the geometry is ineligible, no Euclidean SIS
row exists, or the L2 route does not lower the A rank, the planner keeps the L
infinity candidate.
Runtime expansion never reruns the model. It only checks the frozen schedule.

A clear terminal L2 candidate has no recursive norm proof. The verifier checks
the decoded response norm directly. The planner may use the certified energy
to estimate a smaller Golomb payload for candidate comparison. The scheduled
Golomb byte cap and the payload grind remain unchanged.

Generated schedule identity includes the cap policy and the separate L2 table
digest. Runtime expansion derives the route, cap, proof shape, and A rank from
that identity. A mismatch between the preset policy and generated catalog is
an error.

Source type is not part of runtime schedule identity. Dense and one-hot
presets own different offline policies and generated catalogs, but equivalent
polynomial groups have the same runtime geometry. In particular, one-hot chunk
size is an input to `UnitOneHotFoldPolicy`; it is not serialized in a
commitment, proof, opening layout, or transcript.
