# Spec: L-infinity SIS Table Cutover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-01 |
| Status        | active |
| PR            | [#255](https://github.com/LayerZero-Labs/akita/pull/255) |
| Supersedes    | `specs/sis-infinity-estimator-implementation-plan.md` Slice 7 |
| Superseded-by | |
| Book-chapter  | |

## Summary

Akita currently prices production SIS security with generated Euclidean
(`L2`) width tables at a 128-bit floor. Most planner bounds, however, are
naturally coefficient-`L∞` envelopes. The current production path converts each
coefficient bound `B` into a Euclidean key `d * B^2`, then looks up a 128-bit
Euclidean SIS floor. This spec cuts over production SIS sizing to generated
coefficient-`L∞` tables keyed directly by the bound that the protocol enforces.

The first shipped table floor is 138 bits, using the Rust SIS estimator's
infinity-norm ADPS16 + LGSA profile. The production key must still include the
minimum security floor, even though only `138` is generated in this PR, so that
future contributors can add 128-bit, 192-bit, or other floors without changing
the table API again.

## Intent

### Goal

Switch all production SIS-floor lookups used by Ajtai key sizing from
Euclidean keys to coefficient-`L∞` keys:

```text
(security_bits, family, ring_dimension, coeff_linf_bound)
    -> max secure ring width by module rank
```

The table lookup remains verifier-safe and table-only. The Rust estimator crate
is still an offline generation tool, not a runtime prover or verifier
dependency.

### Current State

The current production modules expose:

```text
(family, d, collision_l2_sq) -> widths[rank - 1]
```

`AjtaiKeyParams` stores `collision_l2_sq`, and `min_secure_rank` audits a
candidate against that key. L-infinity-originating bounds are converted by
`collision_l2_sq_for_linf_envelope`, which prefers a derived key
`d * ceil_coeff_linf_bucket(B)^2` and falls back to a power-of-two Euclidean
bucket.

The new estimator crate already has an offline table generator for the desired
planner-shaped key:

```text
(family, ring_dimension, coeff_linf_bound) -> widths[rank - 1]
```

That generator currently defaults to a 138-bit target, but its committed CSV
artifact is comparison-only and was produced with the local-minimum parity
profile. The production cutover must generate canonical Rust table modules
from the production-intended profile, not blindly reuse the comparison CSV.

### Target State

The production SIS table layer has one canonical key type. A suggested shape is:

```rust
pub struct SisTableKey {
    pub min_security_bits: u16,
    pub family: SisModulusFamily,
    pub ring_dimension: u32,
    pub coeff_linf_bound: u128,
}
```

Only `min_security_bits = 138` is supported at first. Unsupported security
floors return `None` or `InvalidSetup`, depending on caller context. They must
not silently round down to another floor.

`AjtaiKeyParams` carries the canonical audited `SisTableKey` or its compact
fields. It no longer carries a production `collision_l2_sq` field for
L-infinity-originating roles. Descriptor bytes include the security floor and
coefficient-`L∞` bucket so setup/proof descriptors change intentionally and
unambiguously under the cutover.

### Invariants

- **Security floor is part of identity.** A schedule generated for 138-bit SIS
  tables is not interchangeable with a schedule generated for 128-bit SIS
  tables. The table key, descriptor bytes, catalog identity, and generated
  schedule validation must all encode or recompute the same floor.

- **No silent fallback across floors.** If a caller asks for an unsupported
  floor, the result is a hard rejection. The first PR supports only 138.

- **No split-brain norm accounting.** A role's coefficient-`L∞` envelope is
  rounded to a coefficient bucket and audited directly against the `L∞` table.
  Production lookup must not convert through `d * B^2` or a Euclidean
  power-of-two bucket.

- **One source of truth per role.** A/B/D/F roles compute their coefficient
  envelopes once in `akita_types::sis`. Planner DP, generated schedule
  expansion, catalog validation, group-commit layout, and tests call those
  primitives rather than recomputing formulas.

- **Verifier no-panic contract.** Verifier-reachable table expansion and key
  construction reject malformed or unsupported rows with `AkitaError`, not
  `panic!`, `unwrap`, unchecked indexing, or implicit allocation.

- **Offline estimator only.** `akita-sis-estimator` may generate Rust modules,
  CSVs, metadata, and tests. Runtime crates use generated tables only.

- **Generated schedules are coherent snapshots.** If SIS floor changes alter
  optimal ranks or planner choices, every shipped schedule table is regenerated
  and the table-hit expansion guard compares against the pure DP on the same
  branch.

- **Cap-hit rows are explicit lower bounds.** A generated row where
  `max_width == search_cap` says: "this rank supports at least this width." It
  does not prove the true cutoff. Such rows are safe for production when the
  planner only checks `width <= max_width`, but they reduce measurement
  precision and may hide opportunities to lower ranks.

### Non-Goals

- Runtime on-demand lattice estimation.
- Shipping multiple security floors in the first PR.
- Preserving old descriptor bytes or generated schedule bytes.
- Keeping compatibility aliases such as `collision_l2_sq_for_linf_envelope` for
  production use after the cutover.
- Deleting the Euclidean estimator or its comparison tools. They may remain as
  offline audit/reference paths, clearly labeled as non-production.
- Proving that proof sizes improve globally. The expected result is mixed:
  direct `L∞` modeling can reduce some ranks, while a 138-bit floor can raise
  others.

## Evaluation

### Acceptance Criteria

- [ ] Production table lookup is keyed by
  `(min_security_bits, family, d, coeff_linf_bound)`.
- [ ] Only `min_security_bits = 138` is accepted by production constructors and
  lookup helpers.
- [ ] `AjtaiKeyParams` stores and serializes the new canonical SIS audit key,
  not a production `collision_l2_sq`.
- [ ] A-role committed-fold pricing returns a rounded coefficient-`L∞` bucket
  and secure rank from the `L∞` table.
- [ ] B-role, D-role, and tiered F-role pricing return rounded coefficient
  buckets from the same helper family.
- [ ] All planner DP paths, generated-table expansion paths, group-commit
  conservative layout paths, and tests use the same helpers.
- [ ] Generated SIS table modules are regenerated from the Rust infinity table
  generator with a production profile and carry provenance for security floor,
  estimator profile, target bits, search caps, and generation command.
- [ ] Shipped schedule tables are regenerated after the SIS table cutover.
- [ ] A local delta report summarizes rank/proof-byte changes for shipped
  generated families and representative lookup keys.
- [ ] Documentation describes the current planner/SIS architecture after the
  cutover and no live spec claims that production SIS floors are 128-bit
  Euclidean tables.

### Testing Strategy

Required checks:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh
```

Targeted checks:

```bash
cargo test -p akita-types sis
cargo test -p akita-sis-estimator
cargo test -p akita-config --test generated_tables --features all-schedules
cargo test -p akita-config --test generated_tables --all-features
```

Generation checks:

```bash
cargo run -p akita-sis-estimator --release --features parallel \
  --example infinity_width_table -- \
  --format rust-split \
  --profile exhaustive-parallel

cargo run --release -p akita-config --no-default-features \
  --bin gen_schedule_tables -- crates/akita-schedules/src/generated

cargo run --release -p akita-config --no-default-features \
  --bin gen_schedule_tables -- crates/akita-schedules/src/generated --wiring-only
```

If exhaustive table generation is too slow for the full table, the PR may use
the local-minimum profile only after a documented spot-check shows that
local-minimum rows are no looser than exhaustive rows on the planner-relevant
sample. The spot-check must include small, medium, and large coefficient
buckets, every modulus family, every ring dimension, and ranks that actually
appear in regenerated schedules.

### Performance And Proof Size

The cutover changes two levers at once:

1. Direct coefficient-`L∞` modeling can make the SIS floor tighter than the old
   Euclidean `d * B^2` conversion.
2. Raising the floor from 128 to 138 bits can require higher ranks or different
   planner geometry.

The net effect is expected to be mixed. The implementation PR must include a
reviewable local artifact or summary with:

- per-family proof-byte deltas for generated schedule keys;
- rank deltas for A, B, D, and F roles where they change;
- table cap-hit counts by family, dimension, bound, and rank;
- any schedule keys that become unsupported under the 138-bit `L∞` tables.

Cap-hit policy:

- A cap hit is safe if every planner width that uses that row is at or below
  the emitted `max_width`.
- A cap hit is not a reason to relax security.
- Increasing caps is useful only when it changes a planner decision, removes an
  unsupported row, or proves a tighter rank/proof-size opportunity.
- Cap increases must be targeted by evidence rather than applied globally. The
  default caps are already very large, and broad increases can make generation
  substantially slower without improving shipped schedules.

## Design

### Table Model

Generated production modules should move from:

```rust
pub(crate) fn sis_max_widths(
    family: SisModulusFamily,
    d: u32,
    collision_l2_sq: u128,
) -> Option<&'static [u64]>
```

to a floor-aware coefficient-bound lookup:

```rust
pub(crate) fn sis_max_widths(
    min_security_bits: u16,
    family: SisModulusFamily,
    d: u32,
    coeff_linf_bound: u128,
) -> Option<&'static [u64]>
```

or an equivalent `SisTableKey` wrapper. The public helper should accept a raw
coefficient bound and round it to the smallest supported coefficient bucket.
The generated modules may store only rounded bucket keys.

### Role Envelopes

The production role helpers should expose coefficient-`L∞` buckets:

```text
A: 8 * challenge_l1_mass * fold_witness_verifier_linf_bound * nu
B: 2^log_basis - 1
D: 2^log_basis - 1
F: 2^log_basis - 1
```

The A-role formula keeps the #251 digit-envelope correction: the verifier
accepts every balanced `delta_fold` digit string, so soundness prices the
verifier-public folded-witness digit envelope, not the honest-prover tail cap
directly.

### Security Floor API

`PlannerPolicy` or the SIS helper layer should have an explicit
`min_sis_security_bits` field or equivalent constant. The proof-optimized
presets set it to `138`. If this is added to `PlannerPolicy`, catalog identity
must include it so generated schedule tables fail closed when the floor changes.

The field should be narrow and integer-valued (`u16` is sufficient). Avoid
floating `target_bits` in runtime keys; floats belong in offline generation
config and provenance.

### Estimator Profile Choice

Preferred production profile:

```text
norm = infinity
reduction cost model = ADPS16 classical
shape model = LGSA
optimizer = exhaustive beta + exhaustive zeta
target_bits = 138
```

`exhaustive-parallel` is acceptable when it is deterministic and matches the
serial exhaustive profile. If full exhaustive generation is too expensive, use
local-minimum only with a committed rationale and spot-check summary. The
fallback criterion is not "close enough in general"; it is "not looser for
planner-relevant rows, or explicitly reviewed where it differs."

### Generated Schedules

The compact generated schedule rows still store geometry and ranks, not full
SIS keys. Expansion recomputes each role's coefficient bucket and exact
required rank. After the SIS table cutover, the current generated schedules are
expected to drift and must be regenerated.

The drift guard remains the authority:

```text
table-hit expansion == pure DP regeneration on this branch
```

### Descriptor And Catalog Identity

Descriptor bytes for each Ajtai key must change from:

```text
family, row_len, col_len, collision_l2_sq
```

to:

```text
family, min_security_bits, row_len, col_len, coeff_linf_bound
```

or the equivalent canonical key order. The exact byte order is a descriptor
contract and should be covered by existing descriptor tests or a new targeted
test.

Catalog identity must include the SIS security floor and any other policy field
that affects role envelopes or table lookup. A table generated at 128 bits must
not validate under a 138-bit policy.

### Backward Compatibility

This repo explicitly makes no backward-compatibility guarantees. Do a full
cutover:

- delete production L2 wrapper paths that exist only for the old table shape;
- rename fields and accessors instead of keeping aliases;
- update all call sites in one pass;
- regenerate shipped artifacts rather than supporting both shapes in runtime
  code.

Offline Euclidean comparison utilities can remain if their names, docs, and
headers make clear that they are not the production lookup path.

## Documentation

The implementation PR must treat docs as part of the cutover, not a follow-up
cleanup. Required documentation work:

- Update `book/src/how/security.md` with the production SIS table model,
  coefficient-`L∞` role envelopes, 138-bit floor, and no-panic lookup contract.
- Update `book/src/how/configuration.md` with the current planner architecture:
  policy -> DP search -> generated compact rows -> canonical expansion ->
  catalog identity and drift guard.
- Update `book/src/how/architecture.md` if the crate map or generated table
  ownership changes.
- Update `scripts/sis_golden/README.md` so infinity table generation is no
  longer described only as comparison output after production cutover.
- Update `docs/crate-graph.md` if the table-generation or crate dependency
  story changes.
- Update `AGENTS.md` only if the canonical commands or maintainer pointers
  change.

Spec lifecycle work:

- Mark this spec `implemented` and set `PR` before merge.
- Update `specs/sis-infinity-estimator-implementation-plan.md` so Slice 7 points
  to this spec and does not remain a live contradictory plan.
- Revisit `specs/sis-euclidean-estimator.md`; after this cutover, it should be
  either historical/offline-only or archived once durable Euclidean-generator
  notes are folded into docs.
- Revisit `specs/weak-binding-norm-fix.md`,
  `specs/schedule-catalog-ownership.md`, and
  `specs/fold-linf-rejection.md` for terminology that still says production
  SIS floors are `collision_l2_sq` or 128-bit L2.
- If durable planner architecture prose is added to the book, archive or
  supersede stale planner specs according to `specs/PRUNING.md`.

## Execution

Recommended implementation order:

1. Add a floor-aware `SisTableKey` and table lookup surface in
   `akita-types::sis`.
2. Extend the infinity width generator with production `rust-split` output,
   provenance headers, and full-config validation.
3. Generate 138-bit `L∞` Rust table modules from the production profile.
4. Replace L2-derived role helpers with coefficient-bucket helpers.
5. Cut over `AjtaiKeyParams`, descriptor bytes, and tests.
6. Cut over planner DP, generated schedule expansion, generated validation,
   group-commit conservative layout, and schedule-width audits.
7. Regenerate schedule tables.
8. Produce and review the schedule delta report.
9. Update book/docs/spec lifecycle.
10. Run full verification.

Risk-first checks:

- Generate a small 138-bit exhaustive table for one family/dimension and verify
  serial-vs-parallel parity.
- Compare local-minimum vs exhaustive on planner-relevant rows before deciding
  whether exhaustive full-table generation is feasible.
- Run a temporary DP regeneration against partial tables to identify unsupported
  buckets or width cap gaps before spending time on full schedule regen.
- Check descriptor-byte tests early, because the cutover intentionally changes
  setup/proof identity.

## References

- `crates/akita-types/src/sis/ajtai_key.rs`
- `crates/akita-types/src/sis/norm_bound.rs`
- `crates/akita-types/src/sis/generated_sis_table/`
- `crates/akita-sis-estimator/src/width_table.rs`
- `crates/akita-sis-estimator/src/config.rs`
- `crates/akita-planner/src/generated/expand.rs`
- `crates/akita-config/src/bin/gen_schedule_tables.rs`
- `specs/sis-infinity-estimator-crate.md`
- `specs/sis-infinity-estimator-implementation-plan.md`
- `specs/sis-euclidean-estimator.md`
- `specs/schedule-catalog-ownership.md`
- `docs/documentation.md`
