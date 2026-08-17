# Spec: Bounded Dense Committed Sources

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Omid Bodaghi                   |
| Created       | 2026-08-17                     |
| Status        | implemented                    |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  | how/configuration.md           |

## Summary

A dense Akita commitment currently prices its committed witness `s` at the full
configured field width. `fp128::Dense` declares `log_commit_bound = 128`, so the
A-role gadget decomposition uses `δ_commit = ceil(128 / log_basis_inner)` digits
per source coefficient even when the caller knows every coefficient is, say, a
64-bit value lifted into the 128-bit field. `δ_commit` multiplies the A-matrix
input width, and therefore the shared setup matrix, the root A payload, and the
level-1 witness that the whole recursion suffix inherits. Paying for 128 bits of
range that the witness never uses is pure overhead.

The extreme opposite already exists: a one-hot preset declares
`log_commit_bound = 1` and gets `δ_commit = 1`. Nothing in between is
expressible as a shipped configuration, and several code paths use
`log_commit_bound == 1` as an informal "is this one-hot" test rather than reading
a declared source class.

This feature makes the committed-source coefficient bound a first-class,
declarable schedule-generation input for dense sources — any
`1 ≤ log_commit_bound ≤ field_bits` — so a caller with a known bound gets
parameters sized to that bound. One-hot (`B = 1`) and full-field
(`B = field_bits`) become the two endpoints of one parameterized family rather
than two special cases.

## Intent

### Goal

Ship bounded dense presets whose committed-source coefficient bound `B` is
declared independently of the field width, generate their schedules from `B`
instead of `field_bits`, and reject at commit time any polynomial whose
coefficients fall outside the declared bound.

Key surfaces:

- `akita_types::DecompositionParams::log_commit_bound` becomes the documented,
  validated, general source bound (today it is documented as general but only
  ever set to `1` or `field_bits` by shipped presets).
- A new explicit source-class value in `akita-config` replaces the
  `log_commit_bound == 1` predicates in
  `crates/akita-config/src/proof_optimized.rs` and
  `crates/akita-config/src/setup_prefix_slots.rs`, and drives
  `CommitmentConfig::root_honest_fold_policy`.
- New `CommitmentConfig` presets under `crates/akita-config/src/proof_optimized/`
  with their own generated-schedule families, cargo features, and catalogs.
- A new prover-side representability check on the committed dense source,
  reusing `akita_types::sis::balanced_digit_representable_bounds` and
  `akita_algebra::ring::cyclotomic::decompose_centering_threshold` so the check
  and the decomposition agree by construction.
- `akita_planner::response_model` learns the source bound so the modeled L2
  source energy stops charging the final digit plane a full `log_basis` of
  range when the bound stops short of it.

### What the bound does and does not change

This was verified against the current planner before writing the spec (see
[Measurements](#measurements)). Recording it here because it determines the
whole shape of the change:

**The bound enters the protocol at exactly one place: `δ_commit`.** From there it
propagates to the A-matrix input width
(`decomposed_s_block_ring_count(num_positions_per_block, δ_commit)`), the SIS
rank `n_a` that width demands, the root A payload, the shared setup matrix
length, and the level-1 witness length — which the entire recursion suffix
inherits.

**No new honest-fold policy is needed.** `BalancedSignedDigitFoldPolicy` sizes
the folded response from the *post-decomposition* source plane, whose norms are
`FoldWitnessNorms::bounded(log_basis_inner, ring_dimension) = (b/2, D·b/2)`.
Those are the norms of one balanced digit plane and are independent of `B`. Under
the digit-innermost layout a challenge block spans all `δ_commit` planes of its
positions, so even a partially constrained top plane does not lower the block's
`‖s‖_∞`. `UnitOneHotFoldPolicy` is *not* the `B = 1` case of a bounded policy:
its gains come from source *sparsity* (`‖s‖_1 = 1` per logical chunk) plus a
per-source-class MGF/Chernoff argument, not from smallness. A bounded dense
source is dense and has no sparsity to exploit. Bounded dense keeps
`BalancedSignedDigit`.

**No wire-format or verifier-logic change is needed.** `DecompositionParams` is
already serialized into the instance descriptor
(`encode_decomposition` in `crates/akita-types/src/instance_descriptor/mod.rs`)
and hashed into both `policy_digest` and `identity_digest`
(`write_decomposition` in `crates/akita-schedules/src/catalog_identity.rs`).
The root `num_digits_inner` is stored in the compact catalog entry, replayed
verbatim by `expand_to_level_params_with_setup`, and covered by
`entries_key_digest`. A bounded root therefore round-trips and audits with no new
field.

**The security statement is the one-hot statement, generalized.** A bounded
family is binding and complete only for polynomials whose centered coefficients
lie in the range that `δ_commit` balanced digits represent. That is a *smaller*
accepted witness space than full-field dense, priced by exactly the digit
envelope the A-role SIS route already prices. It is not a weakening — but it does
mean an out-of-range input must be rejected rather than silently truncated, which
is what the current decomposition does (see
[Correctness gap](#correctness-gap-silent-truncation)).

### Invariants

1. **Representability.** For a bounded family, `commit` must reject any source
   polynomial with a coefficient outside the range that `δ_commit` balanced
   base-`2^log_basis_inner` digits represent under the centering rule
   `decompose_centering_threshold(δ_commit, log_basis_inner, q)` actually uses.
   Pinned by `bounded_dense_commit_rejects_a_coefficient_above_the_bound`.
2. **Decomposition exactness.** For an accepted input, the committed digits
   recompose to the original centered coefficient exactly. Pinned by extending
   the existing decomposition round-trip tests in
   `crates/akita-algebra/src/ring/cyclotomic/decomposition.rs`.
3. **Bound/depth agreement.** A generated root row's `num_digits_inner` equals
   `num_digits_inner_for_bound(decomposition @ log_basis_inner, log_commit_bound)`.
   Currently `audit_committed_params` in `crates/akita-schedules/src/audit.rs`
   only checks `num_digits_inner != 0` plus A-width consistency at the root; the
   canonical check exists for the terminal fold only (`audit_terminal`). Add the
   root check.
4. **Bound validity.** `1 ≤ log_commit_bound ≤ field_bits`, and
   `log_open_bound.is_some()` whenever `log_commit_bound < field_bits`. There is
   no such validation today; add it to `DecompositionParams` and reject in
   `SetupSection::check`, which currently only rejects `log_basis == 0`.
5. **Identity separation.** Two families differing only in `log_commit_bound`
   have distinct catalog identities and cannot resolve each other's rows.
   Already guaranteed by `write_decomposition` in `policy_digest` /
   `identity_digest`; pin it with a test asserting the digests differ.
6. **Prover/verifier consistency.** A bounded family's proof verifies under the
   same `Cfg`, and fails to verify under the corresponding full-width `Cfg`
   (different catalog identity ⇒ resolution error, not a silent accept).
7. **No effect on unbounded families.** Every existing generated table byte,
   catalog identity, and `dense_inner_basis.rs` snapshot is unchanged. The
   `fp32::Dense` nv=26 row reproduced exactly during the feasibility probe, so
   the probe harness and production agree; keep that snapshot green.
8. **Verifier no-panic.** Every new validation on a verifier-reachable path
   (`SetupSection::check`, root audit) returns `AkitaError` /
   `SerializationError`, never panics.

### Non-Goals

- **Exact non-power-of-two integer caps.** `log_commit_bound` stays a signed bit
  width. `δ_commit` is the only quantity the bound feeds, and
  `num_digits_for_linf_cap` almost never returns a smaller depth than
  `num_digits_for_bound(ceil_log2)` does for the same cap. The exact-cap
  primitive already exists if a future need appears; wiring an exact `u128` cap
  through `DecompositionParams`, the descriptor, and both digests is not worth it
  now.
- **Per-polynomial bounds inside one group.** The bound is per-`Cfg`. Distinct
  bounds across a multi-group root are already reachable today via distinct
  `Cfg`s for precommitted groups, because a `CommittedGroupProfile` freezes its
  own `inner_commit_matrix` (and therefore its own `δ_commit`). The *final* group
  uses the planning `Cfg`'s bound.
- **Changing `UnitOneHotFoldPolicy`.** One-hot presets keep their own policy and
  catalogs unchanged.
- **A bounded multi-chunk or recursive companion family.** Ship the direct
  single-chunk case first.
- **Inferring the bound from the polynomial.** The bound is declared by the
  caller's `Cfg` choice, not discovered at commit time.

## Evaluation

### Acceptance Criteria

- [x] `DecompositionParams` validates `1 ≤ log_commit_bound ≤ field_bits` and the
      `log_open_bound` pairing rule; `SetupSection::check` rejects violations.
- [x] An explicit source-class value replaces both `log_commit_bound == 1`
      predicates (`proof_optimized.rs` `supports_multi_group_root`,
      `setup_prefix_slots.rs` `recursive_group_batch_candidates_for_capacity`)
      and selects the honest fold policy in the preset macro. `rg
      'log_commit_bound == 1|log_commit_bound != 1'` over `crates/` returns
      nothing.
- [x] At least one bounded dense preset ships with a generated catalog behind its
      own cargo feature, wired through `akita-schedules`, `akita-config`,
      `akita-pcs`, and `ALL_GENERATED_FAMILIES`.
- [x] `scripts/generate-schedule-tables.sh` is idempotent with the new family
      present (the CI drift guard passes).
- [x] `commit` returns `AkitaError` for an out-of-bound coefficient on a bounded
      family, and accepts every in-bound coefficient including both signed
      endpoints.
- [x] A prove/verify round trip passes for the bounded family at every nv in its
      catalog (`bounded_dense_roundtrip_at_every_catalog_size`, nvs `[14, 24, 26]`),
      plus the mixed-bound grouped cell
      (`bounded_dense_precommit_with_onehot_final_group`), both new rows of the
      `akita_fp128_e2e.rs` coverage matrix.
- [x] The bounded family's root `num_digits_inner` is strictly smaller than the
      full-width family's at the same nv and inner basis, asserted by a snapshot
      test alongside `crates/akita-config/tests/dense_inner_basis.rs`.
- [x] Root `num_digits_inner` canonicality is audited; a hand-mutated row is
      rejected.
- [x] Setup field elements for the bounded family are strictly below the
      full-width family's at matched nv, asserted from the generated tables.
- [x] Every existing generated table and identity is byte-identical.

### Testing Strategy

Must keep passing unchanged:

- `crates/akita-config/tests/dense_inner_basis.rs` — the exact fp32/fp64/fp128
  dense nv=26 root snapshots.
- `crates/akita-config/tests/generated_tables.rs`,
  `runtime_fallback.rs`, `basis_envelope.rs`,
  `schedule_catalog_miswire.rs`, `schedule_catalog_feature_off.rs`.
- `crates/akita-pcs/tests/akita_fp128_e2e.rs` and
  `akita_small_field_e2e.rs` matrices.
- The `akita-types` SIS unit tests, in particular
  `compute_num_digits_covers_signed_range` and
  `exact_linf_cap_does_not_round_through_a_power_of_two_range`.

New tests:

| Test | Home | Asserts |
|---|---|---|
| bounded root snapshot | `crates/akita-config/tests/` (next to `dense_inner_basis.rs`) | exact `δ_commit`, `n_a`, A width, next-witness length for the bounded family |
| bound/depth audit | `crates/akita-schedules/src/audit.rs` tests | a row whose root `num_digits_inner` disagrees with `log_commit_bound` is rejected |
| identity separation | `crates/akita-schedules/src/catalog_identity.rs` tests | `policy_digest` and `identity_digest` differ across bounds |
| decomposition params validation | `crates/akita-types/src/config.rs` tests + `instance_descriptor` tests | `B = 0`, `B > field_bits`, and the missing-`log_open_bound` case are rejected |
| out-of-bound commit rejection | `crates/akita-prover` (dense backend tests) | `AkitaError`, not a truncated commitment; both signed endpoints accepted |
| bounded e2e | `crates/akita-pcs/tests/akita_fp128_e2e.rs` matrix row | prove/verify round trip, plus cross-`Cfg` resolution failure |
| response-model bound awareness | `crates/akita-planner/src/response_model_tests.rs` | bounded source moment ≤ full-width moment, and the final plane uses the residual bound width |

Feature combinations: the new family must build and test under
`--no-default-features --features transcript-blake2b` plus its own schedule
feature, and must not be pulled into `schedules-default` unless we intend it to
be a default-shipped family (open question below). Run the four CI Clippy
configurations from `AGENTS.md`, since the feature graphs differ.

### Performance

The bound buys **setup size and prover-side witness volume, not proof bytes.**
The adaptive-dimension presets select with
`SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload`, i.e.
lexicographic setup-first, so this is exactly what the objective optimizes.

#### Measurements

Planner-only probe on this branch (temporary example, since removed): existing
preset policies with `log_commit_bound` overridden and `log_open_bound` set to
the field width; `find_schedule` on singleton keys. The `fp32::Dense` nv=26
`B = 32` row reproduces the committed `dense_inner_basis.rs` snapshot exactly
(inner basis 5, `δ_commit` 7, `n_a` 3, next witness 16 205 824), which validates
the harness against production.

`fp128::Dense`, field_bits 128:

| nv | B | setup field elts | root `δ_commit` | level-1 witness | est. proof bytes |
|---|---|---|---|---|---|
| 24 | 128 | 2 818 048 | 26 | 15 614 976 | 74 772 |
| 24 | 64 | 2 097 152 (−26%) | 8 | 10 590 592 (−32%) | 74 772 (0%) |
| 24 | 32 | 1 409 024 (−50%) | 7 | 8 012 800 (−49%) | 73 136 (−2%) |
| 32 | 128 | 46 137 344 | 22 | 273 783 232 | 76 420 |
| 32 | 64 | 41 943 040 (−9%) | 10 | 193 534 336 (−29%) | 76 196 (−0.3%) |
| 32 | 32 | 22 544 384 (−51%) | 5 | 130 619 776 (−52%) | 76 084 (−0.4%) |

`fp64::Dense` (field_bits 64) and `fp32::Dense` (field_bits 32) at nv=26:

| Cfg | B | setup field elts | root `δ_commit` | level-1 witness | est. proof bytes |
|---|---|---|---|---|---|
| fp64 | 64 | 4 194 304 | 8 | 20 854 016 | 73 444 |
| fp64 | 32 | 2 883 584 (−31%) | 7 | 16 260 864 (−22%) | 73 156 (−0.4%) |
| fp64 | 16 | 2 097 152 (−50%) | 2 | 10 499 328 (−50%) | 72 980 (−0.6%) |
| fp32 | 32 | 2 883 584 | 7 | 16 205 824 | 70 196 |
| fp32 | 16 | 1 572 864 (−45%) | 2 | 8 996 864 (−44%) | 70 020 (−0.25%) |

Expectations to hold implementation to:

- Setup field elements and level-1 witness length strictly decrease with `B` at
  the nv values we ship. Assert from the generated tables.
- Estimated proof bytes are flat to slightly better at nv ≥ 24. **Do not promise
  proof-size wins.**
- Prover wall-clock should track the level-1 witness reduction. Verify with
  `book/src/usage/profiling.md` / the profile-bench workflow on the shipped
  bounded key set before merge; a bounded family that does not beat its
  full-width sibling on prover time is not worth shipping.

#### Proof-size regression risk at small nv

The setup-first objective can trade a large proof increase for a small setup
decrease. Measured, `fp128::Dense` nv=16: `B = 128` gives setup 176 128 / proof
71 552, while `B = 16` gives setup 98 304 / proof **126 308** (a 1-level
schedule). Same shape at `fp64::Dense` nv=16, `B = 32`: setup 131 072 / proof
92 052 versus 229 376 / 68 204 at `B = 48`. This is the objective behaving as
specified, not a search bug — confirmed by re-running with
`RecursiveSplitSearchPolicy::Exhaustive`, which returns byte-identical results,
so it is not the `BoundedBalancedExtremesV1` split-domain heuristic either.

Consequence: **every (nv, B) pair we ship must be reviewed for proof size before
it enters a catalog**, and the small-nv keys are where a bounded family can
regress badly. Options to decide during review: restrict the bounded family's key
set to nv ≥ 24; give bounded families
`SelectionPolicyId::MinEstimatedProofPayload`; or accept the tradeoff and record
it. Note that changing the selection policy changes the catalog identity, so it
must be decided before generation.

## Design

### Architecture

The change is a thin declaration layer plus one new safety check; the sizing
machinery already generalizes.

```
Cfg (akita-config)
  └─ DecompositionParams { log_basis, log_commit_bound: B, log_open_bound: Some(field_bits) }
  └─ source class  ──────────────────────────────┐
                                                 ▼
policy_of::<Cfg>()  ──►  PlannerPolicy.decomposition
                                                 │
akita-planner                                    │
  root_inner_basis_source(policy, B)  ──►  InnerBasisSource::RawCoefficients { log_bound: B }
        ├─ search_range        : (min, inner_max.min(max(B, min)))
        └─ num_digits_inner    : num_digits_inner_for_bound(decomp @ selected basis, B)  ──►  δ_commit
                                                 │
                        δ_commit ──► A width ──► n_a ──► A payload, setup length, level-1 witness
                                                 │
akita-schedules                                  ▼
  compact entry stores root num_digits_inner; expand replays it; audit checks it
  policy_digest / identity_digest hash DecompositionParams  ⇒  distinct family identity
                                                 │
akita-prover                                     ▼
  commit: reject coefficients outside δ_commit balanced digits  (NEW)
  dense backend: decompose to δ_commit digits (unchanged)
```

Already general, needs no change:

- `InnerBasisSource::RawCoefficients { log_bound }` in
  `crates/akita-planner/src/policy.rs` — carries an arbitrary bound and both
  caps the inner-basis sweep and computes the exact digit depth for it.
- `num_digits_for_bound` / `compute_num_digits` /
  `compute_num_digits_field_width` in
  `crates/akita-types/src/sis/decomposition_digits.rs` — the symmetric/asymmetric
  router is already correct for `1 ≤ B ≤ field_bits`, and
  `compute_num_digits_covers_signed_range` already pins coverage and minimality
  for every `log_basis` in 2..=8 and `log_bound` in 1..=120.
- `BalancedSignedDigitFoldPolicy` and
  `HonestFoldPolicySpec::witness_norms_for_inner_basis`.
- `num_digits_open` and `num_digits_setup_prefix_commit`, which correctly read
  `field_bits()` (= `log_open_bound` = the true field width) because `t̂` / `ŵ`
  and setup prefixes carry genuine full-width field elements.
- The compact-entry / expansion / identity path described above.

Needs change:

1. **`crates/akita-types/src/config.rs`** — document `log_commit_bound` as the
   general source bound; add `validate()` for invariant 4; note that
   `field_bits()` is deliberately the *open* width, not the commit bound.
2. **`crates/akita-types/src/instance_descriptor/mod.rs`** — extend
   `SetupSection::check` to enforce invariant 4 on deserialized descriptors.
3. **`crates/akita-config/src/proof_optimized.rs`** — introduce the explicit
   source class; use it for `root_honest_fold_policy` and
   `supports_multi_group_root` in `setup_capacity_scan_layouts`; extend
   `impl_proof_optimized_preset!` so a preset states its bound. The macro already
   threads `$log_commit_bound` separately from `$field_bits` and already derives
   `log_open_bound = Some(field_bits)` when they differ, so a bounded preset is
   nearly a data-only addition — the honest-fold selector's
   `if $log_commit_bound == 1` is the one branch that must become a source-class
   match.
4. **`crates/akita-config/src/setup_prefix_slots.rs`** — replace the
   `log_commit_bound != 1` gate in
   `recursive_group_batch_candidates_for_capacity` with the source-class test.
5. **New preset(s)** under `crates/akita-config/src/proof_optimized/fp128.rs`
   (and `fp64.rs` if we ship the fp64 case), with `A/B/D` ring-dimension domains
   copied from the corresponding dense preset.
6. **`crates/akita-planner/src/generated_families.rs`** — a `family_row!` plus a
   scalar key list, and a `group_batch_keys` generator (or
   `no_group_batch_keys` for the first cut).
7. **Cargo features** — `akita-schedules/<family>`,
   `akita-config/schedules-<family>`, `akita-pcs` passthrough, plus
   `all-schedules` membership. Follow the `fp128-dense-multi-chunk` pattern
   exactly.
8. **`crates/akita-schedules/src/generated/`** — the generated module plus
   `mod.rs` wiring, produced by `scripts/generate-schedule-tables.sh`.
9. **`crates/akita-schedules/src/audit.rs`** — root `num_digits_inner`
   canonicality (invariant 3), mirroring the existing `audit_terminal` check.
10. **`crates/akita-prover`** — the representability check (below).
11. **`crates/akita-planner/src/response_model.rs`** — thread the source bound
    into `bounded_field_source_moment` so the plane loop breaks on the bound, not
    on `field_bits`, and the final plane uses the residual bound width.
    `root_group_source_moments` currently takes a single `field_bits`; it needs
    the per-group source bound instead. Direction is conservative today (it
    over-charges), so this is a tightening, not a correctness fix — but it should
    land with the feature so the L2 route can actually exploit the bound.

### Correctness gap: silent truncation

`balanced_decompose_coefficients_pow2_signed_into_with_params` in
`crates/akita-algebra/src/ring/cyclotomic/decomposition.rs` peels exactly
`params.levels` digits and discards the remaining quotient. There is no residual
check. `decompose_centering_threshold` returns `q/2` whenever
`levels · log_basis != field_bits`, so a bounded commitment centers symmetrically
about `q/2` and then truncates anything that does not fit `δ_commit` digits.

Today this is unreachable in production because every shipped preset either
covers the field width or uses a structurally constrained one-hot source type.
A bounded dense family makes it reachable: an out-of-range coefficient would
produce a commitment to a *different* polynomial than the one the caller opens,
which surfaces as an opaque verification failure rather than a clear input error.

Fix: an explicit pre-commit check in `akita-prover`, gated on
`δ_commit · log_basis_inner < field_bits` so full-width families pay nothing.
Reuse `fold_witness_representable_linf_bounds(log_basis_inner, δ_commit)` for the
asymmetric `(negative_reach, positive_reach)` pair and
`decompose_centering_threshold` for the sign rule, so the check and the
decomposition cannot drift — this is the `AGENTS.md` "no split-brain between
certification and the primitive" rule. The scan is `O(n)` against an
`O(n · δ_commit)` matvec, so it is free.

Deliberately *not* making the decomposition kernel itself fallible: the inner
loops are hot and SIMD-specialized, and the bound is a property of the input, so
a boundary check is the right altitude.

### Alternatives Considered

**A new `BoundedFoldPolicy` alongside `UnitOneHotFoldPolicy`.** This is the
shape the framing suggests, but it has no content: the folded response is sized
from the post-decomposition digit plane, whose norms do not depend on `B`. One-hot
wins from sparsity and its physical source-class MGF, and a bounded dense source
has no sparsity. A bounded policy would compute the same cap as
`BalancedSignedDigit` while adding a second code path to keep in sync. Rejected.

**Exact integer `u128` cap instead of a bit width.** Strictly more general, and
`num_digits_for_linf_cap` already exists. But `δ_commit` is the only consumer,
and exact-vs-ceil-log rarely differ there; the cost is a `DecompositionParams`
shape change that ripples into the descriptor encoding, `policy_digest`, and
`identity_digest`. Deferred, not foreclosed.

**A const-generic preset `BoundedDense<const LOG_BOUND: u32>`.** Cleaner to
declare, but each generated catalog is bound to a cargo feature and a static
table, and a const-generic type cannot select one per instantiation without new
machinery. Named presets per bound match how every other family already ships.

**Deriving the bound from the polynomial at commit time.** Would make the
commitment's accepted witness space input-dependent and therefore not
transcript-bound. Rejected outright.

**Reusing `fp64::Dense` for 64-bit values in a 128-bit field.** Different field,
different SIS modulus profile, different challenge configuration — not the same
thing. The bounded case is a 128-bit field with a 64-bit witness.

## Documentation

- `book/src/how/configuration.md` — owning page. Its "Selective L2 candidates"
  section currently says "A dense root uses the deterministic maximum squared
  digit energy for every coefficient" and, at the end, "Source type is not part
  of runtime schedule identity." Both need qualifying: the *bound* is part of
  runtime schedule identity (it is inside `DecompositionParams`, hashed by
  `write_decomposition`), even though one-hot chunk size is not.
- `book/src/usage/quickstart.md` — preset table gains the bounded family row and
  guidance on when to pick it (known coefficient bound, prover-time/setup-bound
  workload) and its cost (out-of-range inputs are rejected).
- `book/src/usage/feature-flags.md` — new schedule feature.
- `book/src/foundations/gadget-decomposition.md` — state that the committed
  source bound is a declared parameter and that `δ_commit` follows from it.
- `book/src/how/security.md` — the accepted witness space of a bounded family,
  and why a smaller space is not a weakening.
- `docs/doc-blast-radius.json` — add the new config/schedules paths if the
  advisory does not already cover them.
- Run `scripts/check-doc-guardrails.sh`.

## Execution

Suggested order, each step independently reviewable:

1. **Groundwork, no behavior change.** `DecompositionParams` docs + validation;
   `SetupSection::check`; root `num_digits_inner` audit; source-class value
   replacing both `log_commit_bound == 1` predicates. All existing tables and
   identities must stay byte-identical.
2. **Prover safety.** The representability check plus its tests. Still no new
   family — exercise it with an explicitly constructed bounded parameter set.
3. **Response-model bound awareness.** Thread the source bound through
   `root_group_source_moments` / `bounded_field_source_moment`. This changes
   planner output for bounded policies only; confirm existing tables are
   unchanged before generating anything.
4. **First bounded family.** Preset, `family_row!`, features, key set, generated
   table; snapshot and e2e tests. **Decide the key set and selection policy
   before generating** — both are baked into the catalog identity.
5. **Profile.** Prover wall-clock and setup size versus the full-width sibling on
   the shipped keys. Ship only if the prover-time win materializes.

### Decisions taken

The review's open questions resolved as follows.

- **Shipped pair:** `fp128` with `B = 64`, as `fp128::Dense64`. `fp64` with
  `B = 32` is deferred — the machinery is bound-generic, so it is a preset plus a
  catalog, not new protocol work.
- **Key set:** `[14, 24, 26]`. 24 and 26 are where the bound's savings are
  measured against the matching `fp128_dense` rows. 14 is not a production size
  choice: it is the *producer* for the bounded precommit descriptor embedded in
  the `fp128_onehot` grouped catalog, which the mixed-bound end-to-end test
  needs. The spec's "nv ≥ 24 only" recommendation is honored for the sizes a
  caller would pick for real work.
- **Selection policy:** kept setup-first, consistent with every other adaptive
  family. The small-nv proof-size behavior documented above is therefore still
  reachable at nv=14; that row exists only to freeze a precommit profile.
- **`schedules-default`:** not a member. `schedules-fp128-dense64` is opt-in and
  lives in `all-schedules`, with a dedicated CI step ("Bounded dense source PCS
  e2e") following the `schedules-fp128-dense-multi-chunk` precedent.

### Deltas from the approved spec

Three things came out differently once implemented, all in the direction of
deleting rather than adding:

1. **`BalancedSignedDigitFoldPolicy` lost its `witness` field.** The preset-level
   `fold_norms = FoldWitnessNorms::…` declaration was write-only — the policy only
   `validate()`d it, while the norms actually used come from
   `HonestFoldSizingQuery::witness_norms`. Keeping it would have left a
   preset-visible knob that does nothing, right next to the bound that does. The
   presets now declare `source = balanced_digits` / `source = unit_one_hot`
   instead, which is the source-class declaration the spec asked for.
2. **Both `log_commit_bound == 1` proxies were deleted, not re-expressed.** The
   `supports_multi_group_root` branch in `setup_capacity_scan_layouts` pushed
   multi-group layouts that `proof_optimized_schedule_key` unconditionally
   rejects, so every one of them was discarded by the caller's
   `let Ok(..) else { continue }` — dead work, now removed along with
   `DEFAULT_GROUP_BATCH_MAX_PRECOMMITTED_GROUPS`. The
   `setup_prefix_slots` gate was redundant given `recursive_setup_planning()`
   (every recursive config delegates its decomposition to a one-hot base), so the
   bound test was dropped rather than replaced. No source-class predicate was
   needed anywhere.
3. **`fold_witness_representable_linf_bounds` was renamed to
   `balanced_digit_representable_bounds`.** It is the accepted envelope of any
   balanced-digit plane, and the producer-side bound check is its second caller;
   the old name read as fold-specific. A companion
   `balanced_digits_cover_centered_field` decides whether a schedule needs the
   check at all, derived from the modulus and the stored digit geometry rather
   than from a declared bit width — so the guard cannot disagree with the digits
   the commitment holds.

### The trap the guard walked into

The first implementation of the producer-side check compared each source's reach
against `balanced_digit_representable_bounds` and decided "is this schedule
bounded?" by asking whether that interval covered `[-q/2, (q-1)/2]`. Both halves
were wrong, and together they rejected **every full-width dense commitment**:

1. `balanced_digit_max` saturates `b^n` at `u128::MAX` and is documented as
   returning a conservative *lower* bound — correct for choosing a digit depth,
   wrong as an acceptance interval. A full-field fp128 dense row uses
   `ceil(128/11) = 12` digits of base `2^11`, spanning 132 bits, so the "positive
   reach" came back as `1.70e38` instead of the true `2.7e39`.
2. The coverage test assumed centering at `q/2`. The decomposition centers at
   `decompose_centering_threshold`, which for a depth where
   `num_digits · log_basis == field_bits` (base 4, 16, or 256 on a 128-bit field)
   deliberately drops *below* `q/2` so that values above the shorter positive
   reach are centered negative, where the longer negative reach covers them.
   Against `q/2` those depths look uncovered; against the real threshold every
   full-field depth in `log_basis ∈ [2, 11]` covers the field exactly.

Both are fixed by stating the check in the same terms the decomposition uses:
`checked_balanced_digit_representable_bounds` (in `akita-types`) returns `None` for
a side beyond `u128` instead of a saturated lower bound, and the coverage test and
per-coefficient comparison both live in `akita-prover` where
`decompose_centering_threshold` is reachable, taking the threshold as their sign
rule. `RootCommitSource::committed_centered_reach` accordingly receives
`(modulus, centering_threshold)` rather than assuming a convention. The naive
`balanced_digits_cover_centered_field` helper was deleted rather than patched:
a coverage predicate that cannot see the threshold cannot be correct.

This is the reason invariant 2 (decomposition exactness) is worth stating
separately from invariant 1: the accepted interval is a property of the digit
depth *and* the centering rule, and the two must be read from one place.

The response-model change landed as specified but is narrower in effect than the
spec implied: `bounded_field_source_moment` now breaks on the *source bound*
instead of the field width, which tightens the final digit plane for a bounded
final group. Precommitted groups keep the field-width bound, because a frozen
group's params do not carry its producer's declared bound; that is the previous
behavior and a valid upper bound. No existing family's output changes, since for
every full-field preset `log_commit_bound == field_bits`.

Risks:

- Generation cost. Each new family adds keys to the offline generator and the
  drift guard. Keep the first key set small.
- Adding the root `num_digits_inner` audit could reject an existing shipped row
  if any full-width family stores a non-canonical depth. Verify against all
  current tables in step 1 before relying on it.
- `min_offloaded_witness_contraction = 3` interacts with `δ_commit`: a smaller
  root A width means a smaller root output witness, which changes which deeper
  folds clear the contraction requirement. Expect different level counts, not
  just smaller widths.

## Unrelated pre-existing failure found while validating

`akita-setup`'s `expanded_setup_roundtrips_and_derives_same_verifier` fails under
the CI feature set (`--no-default-features --features
parallel,disk-persistence,schedules-default,transcript-blake2b`) with

```text
LengthLimitExceeded { len: 67633152, max: 67108864 }
```

when deserializing the expanded setup for `fp128::Dense` at `(nv = 14, 3 polys)`.
The materialized shared matrix is ~64.5 MiB and the deserializer's limit is
64 MiB.

**This is not caused by this feature.** Verified by stashing the entire branch and
re-running on the clean tree: byte-identical failure, same two numbers. It is out
of scope here and left untouched, but it should be filed separately — either the
capacity envelope for that shape needs to come down or the serialization limit
needs to admit it.

## References

- `specs/fold-linf-rejection.md` — folded-witness `L∞` sizing; why the fold cap
  is independent of the source bound.
- `specs/selective-l2-fold-security-sizing.md` — the L2 route the response-model
  tightening feeds.
- `specs/digit-innermost-layout.md` — why a bounded top digit plane does not
  lower a challenge block's `‖s‖_∞`.
- `specs/typed-schedule-topology-cutover.md` — generated-family / catalog
  identity structure a new family must follow.
- `book/src/how/configuration.md`, `book/src/foundations/gadget-decomposition.md`.
- Probe command shape used for the measurements (temporary example, removed):
  `cargo run --release -p akita-planner --features catalog-gen --example <probe>`,
  overriding `PlannerPolicy::decomposition` and calling
  `akita_planner::find_schedule` on singleton keys.
