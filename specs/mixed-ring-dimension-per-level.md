# Spec: Mixed Ring-Dimension-Per-Level Openings (Large-D Root, Small-D Tail)

| Field         | Value |
|---------------|-------|
| Author(s)     | Omid Bodaghi |
| Created       | 2026-07-22 |
| Status        | experimental |
| Scope         | large `nv` only (primary benchmark `nv = 36`, `np = 1`) |
| Preset        | fp128 (`p = 2^128 - 2^32 + 22537`), one-hot, `onehot_k = 256`, `log_commit_bound = 1` |
| Related       | `specs/runtime-ring-cutover.md`, `report.md` (pinned early `log_basis` study) |

## Summary

This spec documents an experiment that opens a single `nv = 36` one-hot
polynomial with a **mixed ring dimension per fold level**: the leading fold
levels run at a *larger* ring dimension `D` (cheaper commitment), then the
schedule switches down to `D = 64` for the recursive tail. It records:

1. the idea and why it can help,
2. the mixed-`D` schedule construction (single source of truth) and why the
   suffix must be **planned**, not repriced,
3. a prover cost cliff at the ring-dimension **transition** and why the switch
   must land after the first fold pair (`switch = 2`), not on the root,
4. end-to-end numbers at `nv = 36` for `D = 64` (baseline), `root D = 128`
   `switch = 1`, and `root D = 128` `switch = 2`,
5. a three-band variant whose root is `512/128/128`, followed by
   `128/64/64`, then uniform `64`: it minimizes commit time, but raises prove
   time, proof size, verify time, and setup memory versus column E, and
6. a per-role variant (`d_a = 128`, `d_b = d_d = 64`, i.e. commitment
   compression on one level): it runs and verifies, but is **dominated** by the
   per-level `switch = 2` schedule.

**Headline (nv = 36, one-hot, fp128):** committing the first two folds at
`D = 128` and the tail at `D = 64` (`switch = 2`) matches the `D = 64`
baseline's prove time and proof size, **halves commit time and setup memory,
and roughly halves verify time**.

**Update — Fix B (prover + verifier).** The transition prove cliff is now fixed
at its source: the digit range check binds the witness's real shared low ring
block (`ring_bits = log2(coeff_count)`) on every path instead of collapsing
non-uniform steps to a flat sweep. This is **byte-identical on uniform/shipped
schedules** and makes `switch = 1` (root-only `D = 128`) and per-role
compression prove *fast* too (prove ≈3.4 s, down from ≈12 s / ≈7.5 s).
Follow-on verifier changes (**Fixes C–F**) then made non-uniform-root **verify**
fast: C/D/E parallelize the relation setup scan, constraint term, and
per-column weight-building, and **Fix F** replaces the scan outright with the
succinct uniform evaluator for uniform-role levels (`opening_ring_dim < d_a`).
Together they cut non-uniform-root verify from ≈0.317 s to **≈0.026 s** — a 12×
reduction, now on par with `switch = 2` (≈0.023 s) and below the uniform `D=64`
baseline — all output-identical and byte-identical on uniform schedules.

## Background

### Akita fold recap

Akita opens a committed witness through an iterated Hachi fold. Each
non-terminal level:

1. commits the current witness under an Ajtai (SIS) matrix (done at the root
   for the first opening),
2. runs sum-checks (a stage-1 digit **range check** plus a fused relation /
   ring-switch sum-check),
3. emits a fold proof and a smaller folded witness for the next level.

The schedule terminates with a cleartext ("terminal / direct") tail once the
witness is small enough that shipping it beats another fold. The planner
(`akita_planner::find_schedule`) chooses per-level geometry to minimize proof
bytes; its recursive suffix search is `derive_optimal_suffix_schedule`.

### Ring dimension `D` and the commit/fold trade-off

The cyclotomic ring dimension `D` is per-level schedule shape metadata after
`specs/runtime-ring-cutover.md`. Increasing `D` at a level changes the cost of
that level's work in opposite directions:

- **Commit gets cheaper.** A larger `D` packs more field coefficients per ring
  element, so the root Ajtai commitment needs fewer matrix rows (`n_a`) and
  fewer position columns for the same `2^nv` witness. At `nv = 36` the `D = 128`
  root uses `n_a = 3` vs the `D = 64` root's `n_a = 6`, and half the
  positions-per-block.
- **Fold/verify get more expensive per element.** Sum-check and NTT work at
  ring dimension `D` are heavier, and the verifier does `D`-sized ring
  arithmetic per fold.

Commit dominates the `nv = 36` baseline (≈26 s of a ≈30 s commit+prove+verify
total), so trading a cheaper commit for a slightly heavier fold is attractive —
**if** the heavier fold does not blow up. This experiment tests exactly that.

## The idea: a large-D leading band, then a small-D tail

Open with the larger ring dimension for the first (largest) folds, where the
commit savings are greatest, then hand off to `D = 64` for the recursive tail,
where per-element fold cost matters more and the smaller ring is cheaper.

Concretely, for `nv = 36`:

- **Baseline (A):** `D = 64` at every level.
- **Experiment (B):** `D = 128` for fold levels `[0, switch)`, then `D = 64`
  for levels `[switch, …)` and the terminal.

The setup matrix is generated at the envelope's `gen_ring_dim = 128`, and the
`D = 64` suffix levels validate because `64 | 128` (the runtime ring-cutover
divisibility rule).

## Construction (single source of truth)

The mixed schedule is built by
`akita_pcs::test_support::mixed_d_per_level_schedule::<Env, Suffix>(num_vars,
num_polynomials, switch_at_fold)` and driven through the **normal public PCS
API** via a thin config adapter
`akita_pcs::test_support::MixedDConfig<Env, Suffix, SWITCH_AT_FOLD>`, which
delegates every policy hook to `Env` (so `Env::D` sets the setup
`gen_ring_dim`) and overrides only `runtime_schedule` / `get_params_for_prove`
to return the mixed schedule. Both `batched_prove` and `batched_verify` resolve
their schedule through `effective_batched_schedule::<Cfg>`, so this is
production plumbing, not a test-only side door. The same builder backs the
`mixed_d_per_level_e2e` acceptance test.

The builder:

1. resolves the `Env` (large-`D`) schedule via `Env::runtime_schedule`,
2. keeps its root and the `switch_at_fold - 1` recursive folds before the
   switch **verbatim** (at `Env`'s ring dimension), and
3. plans a **proof-size-optimal** `Suffix` (small-`D`) continuation from the
   prefix's output witness.

### Why the suffix must be planned, not repriced (Defect 1)

The first implementation kept the `Env` envelope's *entire* recursive tail and
merely repriced each level at the suffix ring dimension. That is wrong: the
`D = 128` envelope folds aggressively and terminates in few levels; repricing
those few levels at `D = 64` does **not** shrink the witness the same way, so
the schedule terminated far too early on a huge cleartext tail.

- Symptom (`root D = 128`, `switch = 1`, `nv = 36`): the tail's `t` (inner
  state) segment was **159,744 B** and total proof **244,155 B** — 2.6× the
  `D = 64` baseline — with only 5 non-terminal fold levels reaching a
  418,048-element terminal witness (vs the baseline's 7 folds down to 91,904).

The fix reuses the planner's own suffix DP through a new public entry point,
`akita_planner::plan_optimal_suffix` (returning `PlannedSuffix` /
`PlannedSuffixFold` / `PlannedSuffixTerminal`). It runs
`derive_optimal_suffix_schedule` from `(start_level, start_witness_len,
start_lb)` at the suffix policy's ring dimension and returns the min-bytes
continuation, which the builder splices onto the kept prefix. There is exactly
one folding-geometry authority (the planner DP); the builder no longer
hand-copies level geometry.

- Result (`root D = 128`, `switch = 1`, `nv = 36`): total proof drops
  **244,155 B → 94,415 B**, matching the `D = 64` baseline (93,400 B), with 8
  non-terminal folds down to the same ≈92k terminal.

## The transition cost cliff and why `switch = 2` (Defect 2)

After Defect 1, `switch = 1` (only the root at `D = 128`) still had a large
prove/verify regression: prove **11.97 s** (vs baseline 3.14 s) and verify
**0.317 s** (vs 0.053 s), even though the proof size now matched. The extra
cost was almost entirely one span: the **root's stage-1 digit range check**
(`digit_range_prove`) took **7.06 s** vs the baseline root's **0.42 s** — a
~17× blow-up for the same ~`2^36` range-check domain.

### Root cause

The digit range check (`LowBasisRangeCheckProver`) folds a virtual table
indexed by `[column bits] × [ring bits]`. The **ring (low) rounds** use a fast
compact-`i8` path (two-round prefix + third-round deferral + compact folding);
the **column (x) rounds** materialize field elements and are much heavier per
element. The ring/low split is set in
`crates/akita-prover/src/protocol/ring_switch/finalize.rs`:

```
digit_range_equality_low_variable_count =
    if dims == CommitmentRingDims::uniform(opening_ring_dim) {
        opening_ring_dim.trailing_zeros()   // log2(D): ring rounds enabled
    } else {
        0                                   // "mixed path": flat tau0, no ring rounds
    };
```

`dims = instance.role_dims()` is the **current** level's role dimensions, while
`opening_ring_dim` is the **next** level's ring dimension (the fold's
ring-switch target, `next_params.inner_ring_dimension()` in
`protocol/core/fold.rs`). On a uniform step (e.g. `128 → 128` or `64 → 64`)
they match and the range check gets `low = log2(D)` and the fast ring phase. On
the `D`-transition step (`128 → 64`) the current role dims (128) differ from the
next witness dim (64), so the prover takes the "mixed path", samples `tau0` in
flat physical-address order, and sets `low = 0` — the entire range-check domain
runs in the slow materialized x-phase.

Diagnostic (range-check `[col_bits, ring_bits]` at the root, `nv = 24`):

| Root configuration            | col_bits | ring_bits | root range check |
|-------------------------------|----------|-----------|------------------|
| `D = 64` (uniform)            | 15       | 6 (`log2 64`) | fast |
| `D = 128` pure (uniform)      | 14       | 7 (`log2 128`) | fast |
| `D = 128` → `D = 64` (`switch=1`) | 21   | **0**     | **slow (dense)** |

So the penalty is **inherent to the ring-dimension transition**, and it is
proportional to the witness at the transition level. At `switch = 1` the
transition sits on the **root**, whose range-check domain is ~`2^36` — the
worst possible place.

### Fix A (schedule): move the transition off the root

Switch one fold later. At `switch = 2` the root **and** fold level 1 run
uniformly at `D = 128` (fast ring phase on the two largest folds), and the
`128 → 64` transition lands on fold level 2, whose witness is ~1.4M elements —
where the flat range check is negligible. This is a pure **schedule** choice
and keeps the intent ("large `D` early, small `D` late").

### Fix B (prover + verifier): bind the shared low block, not the flat path

Fix A avoids the cliff but forbids a genuine root-only large-`D` (`switch = 1`).
The deeper fix removes the cliff itself. The witness handed to the range check
is *already* stored with a `coeff_count`-wide low ring block, where
`coeff_count = role_dims.common_relation_witness_coeff_count(opening_ring_dim)`
and `ring_bits = log2(coeff_count)`. The flat fallback (`low = 0`) simply
declined to use it. Both the prover
(`ring_switch_finalize`) and the verifier (`ring_switch.rs`) now set

```
digit_range_equality_low_variable_count = ring_bits;  // = log2(coeff_count)
```

unconditionally. On a **uniform** schedule `coeff_count == opening_ring_dim`, so
this equals the historical `log2(opening_ring_dim)` and the proof is
**byte-identical** (verified: the `D = 64` baseline proof stays exactly
93,400 B). On a **non-uniform** step it binds the same low ring block the
witness is stored in, so the digit range check runs the fast compact ring phase
instead of the dense flat x-sweep — with no change to soundness (the Stage-1
point folds the ring block first and Stage-2 reads it through the same
`col_bits`/`ring_bits` split; the `mixed_d_per_level_e2e` prove/verify +
tamper-rejection suite passes unchanged).

Effect at `nv = 36`, `switch = 1`: **prove drops 11.97 s → 3.42 s** and verify
0.317 s → 0.213 s, with proof size unchanged (94,428 B). The per-role cases
(`d_b`/`d_d = 64`) likewise drop from ~7.5 s to ~3.3 s prove.

### Fix C (verifier): parallelize the non-uniform relation scan

A second, independent non-uniform cost sat on **verify** only. When
`opening_ring_dim ≠ role_dims`, the verifier evaluates the relation weight
through `evaluate_lane_factored_relation_at_point`
(`ring_switch/mixed_relation.rs`), whose `mixed_relation_setup_scan` multiplies
the setup matrices by per-column relation weights in their native role rings
(`evaluate_weighted_setup_matrix`). That helper folded over the **row** axis,
but non-uniform levels have very few rows (the D role has `n_d = 1`) and
hundreds of thousands of columns, so the inner column loop ran serially — ~168
ms at the `D = 128` root.

The full sum `Σ_row Σ_col row_weight·col_weight·⟨ring, α⟩` is
associative/commutative, so **Fix C** folds over the **column** axis instead
(mathematically identical output, now parallel across the large dimension).
This helper is used only on the non-uniform path, so uniform schedules are
untouched. A companion change (**Fix D**) parallelizes the Z-consistency term
in `evaluate_group_constraints` over its position axis (again an
associative/commutative sum, so identical output). Combined effect: the root
scan drops **168 ms → ~44 ms**, the constraint term **~28 ms → ~12 ms**, and
non-uniform-root verify drops **0.213 s → 0.076 s** (`switch = 1`) and ~0.21 s →
~0.09 s (per-role), with the `mixed_d_per_level_e2e` tamper-rejection suite
unchanged.

### Fix E (verifier): parallelize the per-column weight-building

After Fixes C/D the setup scan's remaining serial cost was building the A/B/D
per-column weight vectors (the nested `for unit/claim/block/…` loops that fill
`{a,b,d}_column_weights`). **Fix E** rewrites each as a parallel map over the
flat native-column index: the loop→column map is a bijection, so inverting
`native_column → (claim, block, subcolumn/a_row, digit)` and computing each
column independently produces the identical vector (columns for blocks not
covered by a unit stay zero, as before). Combined with C/D this drops
non-uniform-root **verify to 0.047 s** (`switch = 1`) — a 6.7× reduction from
the original 0.317 s.

### Fix F (verifier): uniform-role fast path for `opening_ring_dim < d_a`

Fixes C–E parallelized the mixed scan, but it was still an explicit
`O(setup columns)` multiply. Fix F removes the scan entirely for the shape that
matters here — a **uniform-role** level (all roles `= d_a = D`) whose outgoing
witness ring is smaller (`opening_ring_dim < D`, i.e. the `switch = 1` root and
every ring-dimension transition). The observation: when `opening_ring_dim < D`
the relation carries `D / opening_ring_dim` lanes per ring element, but for
uniform roles the flat point is laid out `[coeff][lane][column]` (the relation
address is `witness_column · lanes + lane`, with the coefficient block below),
so the low `log2(D)` bits are *exactly* the `D`-ring coefficient block. Hence
`coefficient_eval(D) = coeff_eval · lane_eval`, the column structure is
identical to the `opening_ring_dim == D` case, and the succinct uniform
evaluator (`evaluate_uniform_columns_at_point`) returns the *same value* the
lane-factored mixed scan would — without the `O(setup)` multiply.

The change is a one-line gate relaxation in `eval_flat_at_point`
(`context.opening_ring_dim == D && role_dims == uniform(D)` →
`role_dims == uniform(D)`); the uniform evaluator never reads
`opening_ring_dim`. It is a **no-op on uniform schedules** (which always have
`opening_ring_dim == D`, so the gate already fired — proof byte-identical), and
it is validated by `mixed_d_per_level_e2e`, whose level-1 fold is precisely a
uniform-role / `opening_ring_dim = D/2` step: verifying the honest proof plus
rejecting every tamper confirms the fast path equals the trusted mixed scan.

Effect: non-uniform-**root** verify drops **0.047 s → 0.026 s** (`switch = 1`),
now on par with `switch = 2` (0.023 s) and below the uniform `D = 64` baseline
(≈0.043 s) — a **12× reduction** from the original 0.317 s.

The remaining non-uniform case is genuine per-**role** compression
(`d_b`/`d_d ≠ d_a`, e.g. A=128/B=D=64), where the roles themselves differ so the
low `log2(d_a)` bits are not a single clean coefficient block; that still uses
the (parallelized) mixed scan at ≈0.048 s. Extending the fast path there is a
harder, separately-scoped generalization (see [Future work](#future-work)).

## Results

### nv = 36, one-hot, fp128, single machine, catalog-backed (no offline DP)

Both cases resolve schedules from the shipped catalogs (`akita-pcs` default
`schedules-default` feature). Timings are commit / prove / verify wall time;
proof size is the serialized on-wire proof.

Numbers below are **after Fix B** (the prover/verifier low-block change); the
pre-fix `switch = 1` prove/verify are shown in parentheses.

| Metric                      | A: `D = 64` all | B: root-only `D = 128` (`switch = 1`) | **B: `D = 128` folds 0–1 (`switch = 2`)** |
|-----------------------------|-----------------|----------------------------------------|-------------------------------------------|
| Commit                      | 26.38 s         | 14.95 s                                | **13.87 s**  (−47%)                       |
| Prove                       | 3.14 s          | 3.42 s (was 11.97 s)                    | **3.10 s**   (≈ A)                        |
| Verify                      | 0.043 s         | **0.026 s** (was 0.317 s)              | **0.023 s**                               |
| **Commit + Prove + Verify** | 29.6 s          | 27.2 s                                 | **17.0 s**   (−42%)                       |
| Proof size                  | 93,400 B        | 94,415 B                               | 97,800 B (+4.7%)                          |
| — akita_fold / tail         | 41,012 / 52,388 | 31,144 / 52,379*                       | 34,956 / 62,844                           |
| Fold levels (incl. terminal)| 9               | 9                                      | 7                                         |
| Setup expand + prepare      | 0.66 s          | 0.50 s                                 | 0.30 s                                    |
| Setup vector / NTT cache    | 2.16 GB / 5.41 GB | 1.08 GB / 2.71 GB                    | **1.08 GB / 2.71 GB** (−50%)              |
| Verifier NTT cache          | 1.44 MB         | 1.44 MB                                | 1.44 MB                                   |

\* `switch = 1` proof was **244,155 B before the Defect 1 fix** (repriced
suffix). The 94,415 B figure is post-fix.

**Reading the table.** `switch = 2` is a strict win over the baseline on
commit, verify, and setup memory; it ties on prove and costs +4.7% proof size.
`switch = 1` is dominated by `switch = 2` because of the root transition
penalty. The commit win comes from the cheaper `D = 128` Ajtai layout
(`n_a = 3` vs `6`); the setup-memory halving comes from the `D = 128` envelope
needing ~4× fewer setup ring elements (each 2× larger), netting ~½ the bytes.

### Post-#328: fp128 D=256 and A-only D=512

The earlier fp128 `D = 256` ceiling no longer applies. The larger-D challenge
cutover (#328) added fp128 inner dispatch through `D = 512`; the regular SIS
table covers all roles at `D = 256`, while the additive
`Q128_INNER_D512` table covers **only the A role at `D = 512`**. B and D remain
capped at `256`, so a uniform `D = 512` root is neither planned nor certified.

Column F therefore uses a genuinely per-role root:
`d_a/d_b/d_d = 512/128/128`. Its A matrix is repriced with the additive D512
digest and the production D512 sparse challenge; B/D use their existing D128
rows. Setup-envelope dispatch and the relation-quotient digit decomposition
also include D512. The current experimental builder preserves the flat source
length by promoting the tableless D256 planner's root geometry to D512
(halving its source-ring and position counts), then rederives the A norm, rank,
fold-linf state, and outgoing witness length. The two suffix bands are planned
normally at D128 and D64.

## Variant: per-role compression (`d_a = 128`, `d_b = d_d = 64`)

A different question is whether a **single level** can commit at `d_a = 128`
(A-role) while its B and D roles run at `64` — mixing ring dimension *per role*
rather than *per level*.

### Is it possible? Yes.

Per-role ring dimensions are a first-class, validated concept
(`CommitmentRingDims { inner, outer, opening }`). `validate_role_dims`
(`crates/akita-types/src/layout/ring_dims.rs`) requires A to be at least as
large as B and D because A is the recursive relation-witness carrier. It does
not order B relative to D. For example, `{d_a, d_b, d_d} = {128, 32, 64}` is
valid and deliberately has `d_d > d_b`; `{128, 64, 32}` is valid as well.
Since supported dimensions are powers of two, both smaller roles divide A and
pack into its physical columns without padding or widening the witness. The
`d_a = 128` role clears the sparse-fold challenge floor, and the fp128 dispatch
table permits the B/D dimensions above.

A pre-existing E2E fixture (`crates/akita-pcs/tests/mixed_role_e2e.rs`) that
exercised `d_a/d_b/d_d = 128/64/32` is currently **disabled**
(`#![cfg(any())]`) because it targets the pre-merge `Schedule` API; the
underlying per-role support still lives on the current `FoldSchedule` path
(`CommittedGroupParams::role_dims` reads the per-matrix ring dims, and the
prover has a per-role `ensure_role_dim` path alongside the uniform
`ensure_ring_dim`).

### Construction and feasibility on the live API

`akita_pcs::test_support::compressed_role_root_schedule::<Env>` takes the
uniform `D = 128` one-hot schedule and rebuilds the root's **outer (B)** commit
matrix at `outer_d` and **open (D)** commit matrix at `open_d` (a role whose
target equals the envelope dimension is left untouched) — halving a role's
ring dimension doubles its matrix input width and re-derives its SIS output
rank from the audited table at the new dimension — while keeping the inner (A)
matrix at `128` and re-stitching the outgoing witness length. The
`CompressedRoleRootConfig<Env, OUTER_D, OPEN_D>` adapter drives it through the
public PCS API.

This **proves and verifies** end-to-end (confirmed at `nv ≤ 24`). When the B
role is compressed, at `nv = 36` the widened B matrix exceeds the uniform-`D128`
setup envelope (`B-role commit params require 2113536 setup ring elements at
d=64, but setup has 1056768`) because compressing `d_b` to `d = 64` **doubles**
the B matrix width; the adapter therefore overrides `max_setup_matrix_size` to
size the setup from the actual compressed schedule. Compressing only `d_d`
(keeping `d_b = 128`) leaves the setup at the uniform-`D128` footprint.

### Results (nv = 36; after Fix B; timings are 2-run means)

| Metric | A: `D = 64` all | A′: `D = 128` root only (`switch = 1`) | B: `D = 128` folds 0–1 (`switch = 2`) | E: fold-1 A-band (`128/128/128 → 128/64/64 → 64`) | F: three-band A=512 (`512/128/128 → 128/64/64 → 64`) | C: `d_b=d_d=64` | D: `d_b=128, d_d=64` |
|---|---|---|---|---|---|---|---|
| Commit                   | 23.99 s | 14.49 s | 14.34 s | 13.65 s | **10.83 s** | 14.93 s | 14.45 s |
| Prove                    | 2.97 s  | 3.37 s  | 3.17 s  | 3.12 s | **5.31 s** | 3.25 s (was 7.54 s) | 3.26 s (was 7.15 s) |
| Verify                   | 0.038 s | 0.027 s | 0.026 s | 0.024 s | **0.049 s** | 0.045 s (was 0.217 s) | 0.036 s (was 0.210 s) |
| Proof size               | 93,400 B | 94,428 B | 97,824 B | 95,768 B | **98,230 B** | **108,160 B** | **108,179 B** |
| Setup vector / NTT cache | 2.16 GB / 5.41 GB | 1.08 GB / 2.71 GB | 1.08 GB / 2.71 GB | 1.08 GB / 2.71 GB | **1.44 GB / 3.61 GB** | **2.16 GB / 5.41 GB** | 1.08 GB / 2.71 GB |
| Root ranks (`n_a/n_b/n_d`) | — | `3/1/1` | `3/1/1` | `3/1/1` | **`1/1/1`** | `3/2/1` | `3/1/1` |
| Fold levels (incl. terminal) | 9 | 9 | 7 | 7 | 7 | 6 | 6 |

Columns A–E, C, and D retain the prior two-run means measured on an otherwise
idle machine. Column F is the mean of its two new back-to-back passes; proof
size, setup footprint, root ranks, and fold-level counts were deterministic.
The paired E rerun used for the direct E/F comparison below had stable commit
and prove timings but a noisier second verify pass, so the table retains E's
earlier idle-machine verify mean.

### `switch = 1` (root only) vs `switch = 2`

Committing only the root at `D = 128` and dropping to `D = 64` at level 1
(`switch = 1`, column A′) keeps essentially all of the commit and setup win of a
`D = 128` leading band — commit ≈ 14.5 s and setup 1.08 GB / 2.71 GB, tied with
`switch = 2` and ~40% below the `D = 64` baseline (24.0 s, 2.16 GB / 5.41 GB) —
and it produces a **smaller proof than `switch = 2`** (94,428 B vs 97,824 B). The
proof-size edge comes from the tail: `switch = 1`'s terminal response is
52,392 B versus 62,868 B for `switch = 2`, which more than offsets its larger
folded section (42,036 B vs 34,956 B). The cost is that `switch = 1` reverts to
the `D = 64` fold geometry one level earlier, so it runs the full 9-level ladder
(vs 7) with a marginally slower prove (≈3.37 s vs ≈3.17 s) and verify (≈0.027 s
vs ≈0.026 s). Both roots are uniform-per-role (`3/1/1`), so both take the fast
fused relation evaluator on verify.

Neither strictly dominates: `switch = 1` trades ~0.2 s of prove latency and two
extra fold levels for ~3.4 KB of proof size; `switch = 2` trades proof size for
fewer levels and slightly lower prover/verifier latency. The `switch = 2`
recommendation elsewhere in this spec optimizes for prove/verify latency and
level count; pick `switch = 1` when minimizing proof size matters more.

### `E`: fold-1 A-band (role-switch, root = 128)

Column E keeps the whole root uniform at `D = 128` (like `switch = 2`) but at
**fold 1** compresses only the B/D roles to `64` while the A role stays at `128`
(`d_a/d_b/d_d = 128/64/64`); every deeper level is uniform `D = 64`. It is the
`RoleSwitchConfig<D128OneHot, D64OneHot, 64, 128>` adapter
(`AKITA_MIXED_ROLESWITCH=1 AKITA_MIXED_ROLESWITCH_ROOT_D=128`). The measured
per-level geometry is:

| Level | `d_a/d_b/d_d` | `n_a/n_b/n_d` |
|---|---|---|
| 0 (root) | 128/128/128 | 3/1/1 |
| 1 | 128/64/64 | 3/1/1 |
| 2 | 64/64/64 | 5/1/1 |
| 3+ | 64/64/64 | 5–6/1/1 |

Both levels 0 and 1 keep the A path at `128` and the B/D roles are already at
their rank floor (`n_b = n_d = 1`), so verify still takes the fast fused
evaluator (≈0.024 s — the fastest of the `D = 128`-root variants) and the setup
stays at the compressed 1.08 GB / 2.71 GB footprint. Commit (≈13.7 s) is at the
low end of the `D = 128`-root group, but since every variant here commits the
same `D = 128` root the differences are within run-to-run noise.

The useful comparison is against its two neighbors:

- **vs `switch = 2` (B):** E compresses the fold-1 B/D roles to `64` instead of
  keeping them at `128`, which shrinks the proof to **95,768 B (−2,056 B)** at the
  **same 7 levels** and the same commit / prove / verify — so E strictly improves
  on `switch = 2` on proof size here at no other cost.
- **vs `switch = 1` (A′):** E holds the A-band one level deeper, which cuts the
  ladder from 9 to **7 levels** and gives the fastest verify (0.024 s vs 0.027 s),
  at the cost of a slightly larger proof (95,768 B vs 94,428 B, +1,340 B).

So among the `D = 128`-root variants, E is the best "balanced" point: it keeps
`switch = 2`'s short 7-level ladder and fast verify while taking a smaller proof,
and it costs nothing extra on setup or commit.

### `F`: D512 A-only root vs E

Column F changes only the leading band relative to E:

| Level | E `d_a/d_b/d_d` | F `d_a/d_b/d_d` |
|---|---|---|
| 0 (root) | 128/128/128 | **512/128/128** |
| 1 | 128/64/64 | 128/64/64 |
| 2+ | 64/64/64 | 64/64/64 |

Two back-to-back F passes measured commit **10.51 / 11.15 s**, prove
**5.68 / 4.93 s**, and verify **0.0489 / 0.0487 s** (means shown in the
table). Proof size was deterministic at **98,230 B** over 7 levels:
35,372 B of fold proof plus a 62,858 B terminal response. A companion E rerun
on the same binary measured commit 14.31 / 14.51 s and prove 3.16 / 3.10 s;
verify was noisier at 0.024 / 0.045 s, while its proof stayed exactly
95,768 B.

Compared with E, F makes the intended commitment trade:

- **Commit:** 10.83 s vs 13.65 s in the table (−21%); the paired rerun gives
  −25%. The root A rank drops from `3` to `1`.
- **Prove:** 5.31 s vs 3.12 s (+70%). The larger D512 root relation and fold
  arithmetic give back most of the commitment saving.
- **Commit + prove + verify:** about **16.19 s vs 16.79 s** using the table's E
  numbers (≈4% faster), or 16.19 s vs 17.57 s in the paired rerun (≈8% faster).
- **Wire and verify:** F adds **2,462 B** (+2.6%) and its stable ≈0.049 s verify
  is about 2× E's idle ≈0.024 s result.
- **Setup:** the vector / prepared NTT cache grows by one third,
  1.08/2.71 GB → **1.44/3.61 GB**, because each D512 setup ring is four times
  the coefficient storage of D128 even though the root needs only one third as
  many setup ring elements (176,128 vs 528,384).

So F is the **commit-optimized** point and slightly improves total online
prover latency, but E remains the better balanced choice: smaller setup,
smaller proof, much faster prove, and faster verify.

### Why it is (still) dominated

After Fix B the prove penalty is gone (≈3.3 s), so per-role compression is no
longer catastrophic — but it is still **dominated by `switch = 2`** on the
remaining axes:

1. **Verify still pays the non-uniform relation scan** (even after Fixes C–E). A
   non-uniform *root* (`d_d = 64 ≠ d_a = 128`) routes verify through the
   `mixed_relation_setup_scan` (≈0.058 s at the `D = 128` root after Fixes C–E,
   down from ≈0.21 s) instead of the uniform fused evaluator (`switch = 2` root:
   ≈0.03 s). It is ~identical whether or not B is also compressed (C vs D).
2. **No commitment-size win, and B compression widens the setup.** The B/D
   commitments are already at their rank floor (`n_b = n_d = 1` at `d = 128`),
   so shrinking a role's ring only widens its columns without shrinking an
   already-minimal output rank — and for B it bumps the rank to 2 to stay
   secure. Compressing `d_b` doubles the shared setup matrix (2.16 GB, back to
   the `D = 64` footprint); compressing only `d_d` keeps it at 1.08 GB (D vs C).
   Either way the proof grows to ~108 KB (vs 97.8 KB for `switch = 2`).

So per-role compression buys nothing here even after Fix B: it ties the D128
commit, still costs the verifier's non-uniform relation scan, grows the proof,
and (if B is compressed) loses the setup saving. The per-level **uniform
bands** with a late switch (`switch = 2`) strictly dominate it. Compression's
real payoff is proof-*size* on schedules whose public B/D ring rows dominate
(see `specs/commitment-compression-cutover.md`), not prover latency on this
large-`nv` one-hot workload.

## Reproducing

```bash
# Baseline: D = 64 at every level
AKITA_NUM_VARS=36 AKITA_MODE=onehot_fp128_d64 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info \
  cargo run --release -p akita-pcs --example profile

# Mixed: D = 128 leading band, D = 64 tail. AKITA_MIXED_SWITCH selects the
# switch point (default 1); AKITA_MIXED_ROOT_D selects the leading-band D
# (default 128). Recommended:
AKITA_MIXED_SWITCH=2 AKITA_NUM_VARS=36 AKITA_MODE=onehot_fp128_d64_root_d128 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info \
  cargo run --release -p akita-pcs --example profile

# Per-role root: A = 128, B = AKITA_MIXED_OUTER_D, D = AKITA_MIXED_OPEN_D
# (both default 64). E.g. d_b = 128, d_d = 64:
AKITA_MIXED_ROLE=1 AKITA_MIXED_OUTER_D=128 AKITA_MIXED_OPEN_D=64 \
  AKITA_NUM_VARS=36 AKITA_MODE=onehot_fp128_d64_root_d128 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info \
  cargo run --release -p akita-pcs --example profile

# Column E: fold-1 A-band (root 128/128/128, fold 1 128/64/64, then 64).
# AKITA_MIXED_ROLESWITCH_ROOT_D=128 keeps the root fully uniform.
AKITA_MIXED_ROLESWITCH=1 AKITA_MIXED_ROLESWITCH_ROOT_D=128 \
  AKITA_NUM_VARS=36 AKITA_MODE=onehot_fp128_d64_root_d128 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info \
  cargo run --release -p akita-pcs --example profile

# Column F: A-only D512 root (512/128/128, then 128/64/64, then 64).
AKITA_MIXED_THREEBAND=1 AKITA_MIXED_THREEBAND_ROOT_D=512 \
  AKITA_NUM_VARS=36 AKITA_MODE=onehot_fp128_d64_root_d128 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info \
  cargo run --release -p akita-pcs --example profile

# D256 three-band regression companion:
AKITA_MIXED_THREEBAND=1 AKITA_MIXED_THREEBAND_ROOT_D=256 \
  AKITA_NUM_VARS=36 \
  AKITA_MODE=onehot_fp128_d64_root_d128 \
  cargo run --release -p akita-pcs --example profile
```

Extract metrics:

```bash
rg '\] (setup|commit|prove|verify OK|proof: total)' <log>
```

## Implementation summary

| Change | Location |
|--------|----------|
| Public suffix planner `plan_optimal_suffix` + `PlannedSuffix{,Fold,Terminal}` | `crates/akita-planner/src/schedule_params.rs`, re-exported from `crates/akita-planner/src/lib.rs` |
| Mixed-schedule builders/adapters (plans optimal suffixes) | `crates/akita-pcs/src/test_support.rs` |
| `mixed_d_per_level_e2e` refactored onto the shared builder | `crates/akita-pcs/tests/mixed_d_per_level_e2e.rs` (deleted the duplicated `tests/mixed_d_per_level/fixture.rs`) |
| Per-role builder `compressed_role_root_schedule` + `CompressedRoleRootConfig<Env, OUTER_D, OPEN_D>` | `crates/akita-pcs/src/test_support.rs` |
| Per-role E2E oracle (prove/verify + wrong-opening + tampered-commitment rejection) for a `128/64/64` root | `crates/akita-pcs/tests/compressed_role_e2e.rs` |
| Profile mode `onehot_fp128_d64_root_d128` with `AKITA_MIXED_SWITCH` / `AKITA_MIXED_ROOT_D` / `AKITA_MIXED_ROLE` / `AKITA_MIXED_OUTER_D` / `AKITA_MIXED_OPEN_D` | `crates/akita-pcs/examples/profile/modes.rs` |
| `validate_against_planner` flag so the synthetic-schedule mode skips the shipped-catalog proof-size assertion | `crates/akita-pcs/examples/profile/workload.rs` |
| Tableless `fp128::{D256OneHot,D512OneHot}` experiment presets | `crates/akita-config/src/proof_optimized/fp128.rs` |
| D512 A-only three-band root, tableless root planning, and schedule regression | `crates/akita-pcs/src/test_support.rs`, `crates/akita-pcs/tests/three_band_schedule.rs` |
| Thread SIS table digest through canonical A-role norm pricing | `crates/akita-types/src/sis/norm_bound.rs` and planner/schedule callers |
| fp128 D512 setup-envelope dispatch and D512 relation-quotient digit decomposition | `crates/akita-types/src/dispatch/policy.rs`, `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` |
| **Fix B**: `digit_range_equality_low_variable_count = ring_bits` (bind the shared low block on every path) | prover `crates/akita-prover/src/protocol/ring_switch/finalize.rs` + verifier `crates/akita-verifier/src/protocol/ring_switch.rs` |
| **Fix C**: fold `evaluate_weighted_setup_matrix` over the column axis (parallel; identical sum) | verifier `crates/akita-verifier/src/protocol/ring_switch/mixed_relation.rs` |
| **Fix D**: parallelize the Z-consistency term in `evaluate_group_constraints` over positions (parallel; identical sum) | verifier `crates/akita-verifier/src/protocol/ring_switch/mixed_relation.rs` |
| **Fix E**: build the A/B/D per-column weight vectors via a parallel map over the native-column index (bijective inversion; identical vector) | verifier `crates/akita-verifier/src/protocol/ring_switch/mixed_relation.rs` |
| **Fix F**: uniform-role fast path for `opening_ring_dim < d_a` (relax the `eval_flat_at_point` gate to `role_dims == uniform(D)`) | verifier `crates/akita-verifier/src/protocol/ring_switch.rs` |
| **Per-role succinct path**: `RoleLaneSpec` + canonical lane-summed / subcolumn-expanded eq-slices; `prepare` takes `alpha`; subcolumn-scaled column geometry | `crates/akita-types/src/setup_contribution/{weights.rs,plan/prepare.rs}` |
| Route per-role relation setup contribution to `evaluate_direct` when `coeff_count == base_ring_dim` | verifier `crates/akita-verifier/src/protocol/ring_switch/mixed_relation.rs` |
| `SetupIndexWeightEvaluator` per-role fallback to the plan's materialized weights (recursive/offloaded verifier) | `crates/akita-types/src/setup_contribution/setup_index_weight_evaluator.rs` |
| Thread `alpha` through `setup_contribution_plan` / `prepare` callers (prover stage3, verifier stages, benches, tests) | prover/verifier/bench/test call sites |

All prover/verifier protocol changes here are gated so **uniform/shipped
schedules stay byte-identical**: Fix B is byte-identical on uniform schedules;
Fixes C/D/E only reorder associative sums or reindex a bijective build on the
non-uniform relation path; Fix F routes uniform-role levels to the existing
succinct evaluator (a no-op where the gate already fired); and the per-role
succinct path (`RoleLaneSpec`, subcolumn expansion, α-lane sum, the
`SetupIndexWeightEvaluator` fallback) is gated on `a_ratio > 1`, so uniform roles
(`a_ratio = 1`) keep the pre-existing fill-interval / compact-pair fast paths
unchanged. Threading `alpha` into `prepare` is inert for uniform roles.
Validated: the `D = 64` baseline proof is unchanged at 93,400 B; the
`single_poly_e2e`, `akita_e2e`, recursive/offloaded, and `mixed_role_e2e` suites
pass; `mixed_d_per_level_e2e` prove/verify + tamper-rejection is unchanged; and
`compressed_role_e2e` (per-role `128/64/64`) verifies and rejects tampers.
Everything else rides the shipped `specs/runtime-ring-cutover.md` machinery.

## Per-role succinct setup fast path (implemented)

Fix F closed the uniform-role case (the `switch = 1` root); the remaining
non-uniform verify cost was per-role compression (A=128/B=D=64 etc.), which used
the dense mixed scan. That is now routed through the succinct
`SetupContributionPlan::evaluate_direct` as well, after the plan's column model
was generalized to per-role commitments.

**Correctness derivation (validated by a file-backed differential harness).**
For the `128/64/64` root (`coeff_count = base = 64`, `opening_ring = 128`) the
dense setup contribution is

```text
Σ_col [ Σ_lane eq(canonical_relation_lane_index(col, lane)) · α^{base·lane} ]
     · eval_role_ring(setup_col)
```

i.e. eq is evaluated at **`coeff_count` granularity** (relation-lane address
`witness_col·a_ratio + subcolumn·role_lanes + lane`). Each role covers the
`a_ratio = d_a/base` relation lanes with `role_subcolumns = a_ratio/(d_role/base)`
distinct physical setup columns, each carrying `role_lanes = d_role/base`
α-weighted lanes. The A-role (`role_subcolumns = 1`, `role_lanes = a_ratio`)
α-sums its lanes; B/D (`role_lanes = 1`, `role_subcolumns = 2`) spread across
subcolumns. The pre-existing `prepare` evaluated eq at **`opening_ring`
granularity** with **no subcolumn factor**, dropping the lane/subcolumn bit.

**The structural change** (all gated on `a_ratio > 1`, so uniform roles stay
byte-identical, keeping the fast fill-interval path):

- `SetupContributionPlan::prepare` takes `alpha` and builds `e/t/z` eq-slices via
  the unified canonical lane-summed weight (`RoleLaneSpec` in `weights.rs`): D/B
  slices are subcolumn-expanded (parallel map over the expanded column index in
  the physical `[claim][block][subcolumn][digit]` / `[claim][block][A_row][digit][subcolumn]`
  order); the A slice is α-lane-summed. `d_physical_cols`, `d_col_range`, `t_cols`,
  and the projection geometry are scaled by the role subcolumns.
- The verifier routes the per-role relation setup contribution to
  `evaluate_direct` when `coeff_count == base_ring_dim` (mixed relation path).
- `SetupIndexWeightEvaluator` (recursive/offloaded verifier) evaluates the MLE
  directly from the plan's materialized weights when `a_ratio > 1` (its
  compact-pair recurrence only models the uniform ring projection), keeping it
  consistent with the prover's `materialize_setup_index_weights` (single source
  of truth).

**Result.** On the same build, `nv = 36` onehot per-role root (A=128/B=D=64)
verify drops from **≈0.056 s (dense) to ≈0.045 s (succinct)** — the setup scan
folds B/D into one base-ring pass instead of three per-role dense multiplies. The
residual is the A-role setup read (`required ≈ 2.1M` base rings), which both
paths must perform. Validated by `compressed_role_e2e` (honest verify + two
tamper rejections), `mixed_role_e2e`, the recursive/offloaded e2e suites, and the
full `akita-types`/`akita-verifier`/`akita-prover` unit suites.

## Multi-group role ownership

A multi-group level has two geometry scopes:

- Each final or precommitted group owns its A and B matrices and therefore its
  `d_a` and `d_b`.
- The consuming level owns one D matrix and one `d_d`, shared by every group.

`CommittedGroupParams::group_role_dims` is the canonical resolver for that
contract. It combines a group's native A/B dimensions with the level-shared D
dimension and checks the A-carrier geometry. `RelationRhsLayout` records the
resolved dimensions with each group's row counts. Relation RHS sizing,
assembly, and public-claim evaluation consequently use native group widths
instead of applying the final group's A/B dimensions to every group.

The level relation witness still has one physical carrier. Its dimension is the
final group's `d_a`, and it must be at least every group-local `d_a`. Existing
uniform multi-group schedules keep the batch-wide fast path. A heterogeneous
batch uses group-local B dispatch only for its commitment segments.

This establishes the statement and public-data boundary. Group-local quotient
construction, witness emission, and setup-contribution spans remain follow-on
work; no production schedule should claim heterogeneous group A/B support until
those consumers use the same resolver.

## Future work
- **Plan the D512 mixed-role root natively.** Column F currently promotes the
  D256 planner's root geometry and rederives every D512 A/security field. A
  planner entry point that searches per-role root dimensions could determine
  whether another D512 block split improves the tradeoff.
- **Sweep the band.** Explore `switch ∈ {2, 3}` and larger `nv` to map where the
  cheaper large-`D` commit stops paying for the heavier large-`D` early folds.
