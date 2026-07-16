# Spec: Planner Refactor — `Cfg`-Free DP, On-Demand Schedule Expansion, SIS-Audit Hardening

| Field     | Value                                  |
|-----------|----------------------------------------|
| Author(s) |                                        |
| Created   | 2026-05-27 → 2026-06-01 (consolidated) |
| Status    | archived |
| PR        |                                        |
| Book-chapter | book/src/how/configuration.md |

## Summary

This branch reworks the path from "offline planner output" to "runtime per-level
parameters" and fixes the correctness bugs that the old structure was hiding. It
is two structural changes plus a security fix:

1. **Collapse the derivation layers.** The flow used to be three overlapping
   layers (planner DP → `akita-derive` materialization → runtime `Schedule`),
   each re-deriving the same data, with two plan models and two copies of the
   SIS/proof-size math. It is now one flow: the planner stores brute-forced
   numbers in the compact `GeneratedScheduleTableEntry`, and the runtime expands
   full `LevelParams` on demand.
2. **`Cfg`-free DP + dependency inversion.** The DP search is now a pure,
   trait-free library that `akita-config` depends on, so `runtime_schedule` can
   regenerate any key on a table miss — identically on prover and verifier.
3. **SIS-audit hardening.** A silent audit bypass that shipped SIS-insecure
   layouts is closed, the planner's DP comparator bias is fixed, and all schedule
   tables are regenerated against the corrected planner.

There is **no backward-compatibility guarantee**; transcript bytes and shipped
tables change deliberately.

---

# New architecture

## Parameter flow (offline → runtime)

- **`akita-planner` is the sole producer of brute-forced numbers.** The DP emits
  a compact `GeneratedScheduleTableEntry { key, steps: &'static [GeneratedStep] }`
  (`Copy`, `'static`). `GeneratedDirectStep` carries `commit: Option<GeneratedFoldStep>`:
  `Some` is the brute-forced root-direct commit layout, `None` a terminal-direct
  handoff.
- **The runtime fills in the deterministic rest on demand.** A table hit expands
  the entry via `akita_types::schedule_from_entry_bits(...)`; there is no
  `akita-derive`, no second `AkitaSchedulePlan` model, and no runtime
  re-derivation of SIS ranks. Full `LevelParams` are computed per level from the
  stored numbers plus caller-supplied inputs.
- **One source per concern.** Exactly one proof-size formula
  (`akita_types::level_proof_bytes` / `estimate_proof_bytes`), one SIS-sizing
  implementation (`akita_planner::ajtai_params`, run at codegen *and* as the
  runtime fallback), and one schedule representation (the generated entry).
- **Transcript binds the compact effective schedule.** The Fiat-Shamir preamble
  digests the resolved schedule (key + compact steps); the old
  `SetupSection.level_params_digest` is dropped — `setup_seed_digest`,
  `decomposition`, and `sis_modulus_profile` already pin everything the expansion
  needs.

## Crate graph (dependency inversion)

```
akita-config ──► akita-planner ──► akita-types / akita-challenges / akita-field
   (presets,        (pure Cfg-free DP:
   policy_of,        find_schedule(key, &PlannerPolicy, stage1, fold_shape),
   runtime_schedule, ajtai sizing — names no Cfg type)
   gen bin)
```

- **`akita-planner`** is trait-free. `find_schedule` takes a plain
  `PlannerPolicy` value (`ring_dimension`, `decomposition`, `sis_modulus_profile`,
  `ring_subfield_norm_bound`, `claim/chal_ext_degree`, `basis_range`) plus
  `stage1` / `fold_shape` closures. It depends only on
  `akita-types`/`akita-challenges`/`akita-field`. `akita-derive` is deleted; the
  `use_lookup` flag, `offline_schedule_for_key`, and `PlannerCfg` are gone.
- **`akita-config`** now depends on `akita-planner`. `policy_of::<Cfg>()` derives
  the `PlannerPolicy` from the preset (the single source of truth stays the `Cfg`
  impl). `runtime_schedule` serves the table on a hit and calls
  `akita_planner::find_schedule(key, &policy_of::<Self>(), …)` on a miss — DP
  fallback is the default for every preset, so **any** input is supported. The
  preset family list (`ALL_GENERATED_FAMILIES`) and the `gen_schedule_tables`
  binary live here (only config can name presets).
- **DP is verifier-reachable.** The lean "verifier excludes planner" invariant is
  intentionally repealed; `find_schedule` and everything it calls are audited
  under the verifier no-panic contract (`Result`-returning, no
  `panic!`/`unwrap`/`unreachable!`/unchecked indexing/overflow-prone shape math).
  The verifier validates `key.nuposition_index_bits` against setup capacity before the DP runs.
- **Test-only helper relocated.** `akita_batched_root_layout` (used only by tests
  and the `profile` example to pre-size per-poly inputs) lives in the
  feature-gated `akita_config::test_support` module, enabled via dev-deps of
  `akita-pcs`/`akita-scheme`; it is absent from production builds.

## Invariants

1. **Prover/verifier consistency.** Both resolve the same schedule (table hit or
   deterministic DP regen); the effective schedule is Fiat-Shamir-bound, so any
   divergence is a rejected proof, not a soundness hole.
2. **Table parity.** For keys in a shipped table the DP branch is not taken;
   existing proofs and setup envelopes stay byte-identical.
3. **Single source of truth for policy.** `policy_of::<Cfg>()` derives every
   brute-force input from the `Cfg` impl — never hand-written literals.
4. **SIS security.** Expanded keys carry audited collision buckets/ranks; the
   strict `AjtaiKeyParams::try_new` audit fires at the layout boundary.
5. **Verifier no-panic**, and **determinism across `zk`** (witness-length and
   proof-byte math differ under `zk`; regeneration matches the shipped `zk`
   tables).

---

# Issues we found

## 1. SIS-audit bypass shipped insecure root layouts

The planner's root-candidate enumeration reused the SIS ranks `(n_a, n_b, n_d)`
from one inner `(m*, r*)` split for *other* `(m, r)` candidates whose secure floor
was higher. Three lenient defaults combined to hide this:

- `LevelParams::params_only` builds `AjtaiKeyParams` placeholders with
  `collision_inf = 0`;
- `with_layout` then read `collision_inf` from the **layout** argument (still `0`)
  instead of from `self` (which held the SIS-secure bucket), so the bucket was
  lost;
- `AjtaiKeyParams::sis_security_violation` short-circuited to "no violation"
  whenever `collision_inf == 0` (and also when `min_rank_for_secure_width`
  returned `None`).

Net: the per-`(m, r)` audit always saw `0` and never ran, so insecure rows
shipped. **Example** (`fp16::D32Full`, `nv=32`, `t=w=4`): outer split `(m=16,
r=11)` needs rank 17 but inherited rank 15 from the inner `(14, 13)` split and
passed the bypassed audit. The drift guard didn't catch it because both the
shipped table and the from-scratch regen consulted the same buggy code.

**Fixed by:** strict `sis_security_violation` (any zero or an uncovered
configuration is an explicit violation); `with_layout` preserves `collision_inf`
from `self`; `sis_secure_level_params` builds its intentional `col_len = 0`
placeholder via `new_unchecked`; `scale_batched_root_layout` failures fall back to
`commit_params: None` (a missing hint, not a fatal error); the planner derives
ranks per `(m, r)`. All tables regenerated; added an audit-walk guard that
reconstructs every shipped entry's `AjtaiKeyParams` via `try_new`.

## 2. DP comparator bias against `Direct` steps

The suffix DP modelled a terminal-direct step as a placeholder using the parent's
`RelationMatrixRowLayout::Intermediate` witness length, patched to `Terminal` only later. Two
compounding effects resulted:

- **Direct-vs-fold bias.** The local `min(...)` scored "direct now" on the
  inflated Intermediate shape, so it over-rejected `Direct` in favor of one more
  fold.
- **Hidden parent-formula trade-off.** The DP returned a single best suffix, but
  the parent's proof formula depends on the child's `Fold`-vs-`Direct` choice and
  on the child's first-fold `log_basis` (via `next_lp.b_key.row_len()`).
  Collapsing those options hid cases where a locally cheaper child forced a more
  expensive parent.

**Fixed by:** an eager, SIS-aware `to_direct_step` that builds the `Terminal`
shape (and runs the SIS check) up front, plus a two-shape DP returning
`SuffixResult { best_direct, best_fold_per_lb: BTreeMap<u32, …> }` keyed by
first-fold `log_basis`; the memo key gains `w_len_terminal`. Parents enumerate
both branches and pick the global minimum. `finalize_terminal_direct_witness_shape`
and `successor_level_params_from_schedule` are deleted. Result across 1,172
`(family, key)` pairs: **461 improved, 711 unchanged, 0 regressed** (worst-family
~2.6% smaller). Tables regenerated; the drift guard re-runs cleanly.

## 3. (Open) planner overcounts stage-2 degree-2 sumcheck rounds

`akita_types::level_proof_bytes` assumes every stage-2 sumcheck round ships a
degree-3 compressed univariate. The prover emits a handful of rounds at degree 2
(the y-/x-prefix micro-optimization), each one challenge element shorter, so the
estimate **overcounts** the real serialized proof by a few bytes (observed: 16 B
for `onehot_fp128_d32` nv32 np1, 48 B for nv12). This is **pre-existing** —
`origin/main` reproduces it on odd-shaped terminal folds; the refactor only
flipped which schedule a few keys resolve to.

The estimate remains a *conservative upper bound*, so it is safe for planning and
transcript binding. **Interim handling:** the `profile` example no longer asserts
byte-exact equality — it requires `proof.size() <= planned` and accepts an
overcount up to `ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES`, logging a `NOTE`
when nonzero (a real structural regression is far larger and still fails loudly).
**Proper fix (deferred):** teach the offline formula the exact per-round stage-2
degree schedule, regenerate the tables, and restore the byte-exact assertion.

---

## References

- Supersedes `specs/planner-config-consolidation.md` (materialization model).
- `specs/transcript-hardening.md` — `PlanSection` binds the effective schedule digest.
- Key sources: `crates/akita-planner/src/schedule_params.rs`,
  `crates/akita-planner/src/ajtai_params.rs`,
  `crates/akita-types/src/layout/params.rs`,
  `crates/akita-types/src/schedule.rs`,
  `crates/akita-types/src/proof_size.rs`,
  `crates/akita-config/src/lib.rs`.
- Profile command: `AKITA_MODE=onehot_fp128_d32 AKITA_NUM_VARS=32 cargo run --release --example profile`.
