# Spec: Weak-Binding Norm Correction + Folded-Witness L∞ Optimization

| Field     | Value                          |
|-----------|--------------------------------|
| Author(s) | Omid Bodaghi, Quang Dao, Cursor agent draft |
| Created   | 2026-06-01                     |
| Status    | implemented (the original anchored A-role bound is superseded by the committed-fold reprice below) |
| PR        | https://github.com/LayerZero-Labs/akita/pull/146 |

> **Status note (2026-06-03).** The A-role weak-binding price that first
> shipped on this branch (`collision_A = 2·ω̄·β̄·ν` with a single-block `β̄`,
> the "anchored" bound described under "Superseded design" below) was itself
> still under-priced at every committed level. It has been replaced by the
> **committed-fold reprice** (`collision_A = 8·ω·num_claims·2^r·β̄·ν`). The
> corrected derivation, the new formula, and the schedule / preset fallout are
> in the two sections immediately below. Everything from "Implementation
> outcome (2026-06-02)" onward is retained as historical / superseded context.

## Correction (2026-06-03): committed-fold A-role reprice

### What was still wrong

The original fix priced the A-role (inner committed witness `s`) weak-binding
collision at the *anchored* per-block bound

```text
collision_A = 2 · ω̄ · β̄ · ν,   β̄ = min(||c||_inf·||s||_1, ||c||_1·||s||_inf)
```

i.e. one short ring product `||c·s||_inf` for a single opened block. The
weak-binding extractor does not certify that quantity for the committed witness
at a folded level. From two distinct weak openings it produces the kernel
vector

```text
z_A = c̄'·(c̄·s) − c̄·(c̄'·s'),
```

whose size is governed by the **fold-response** difference
`||z^(ℓ,i) − z^0||_inf`, not by a single per-block product. The only norm the
extractor certifies for that difference is the fold bound `2·β^resp`, and
`β^resp` sums one short product over **every** folded block, so it carries the
fold arity `num_claims · 2^r_vars`. Dividing the response by the ring unit `c̄`
does not recover `||s||_inf` (negacyclic division is not norm-preserving), and
the range / one-hot / booleanity checks bind the *honest committed table*, not
the *extracted* quotient. The anchored bound is therefore unsound at every
Ajtai-committed level (the dense root and all recursive fold levels). Only the
**terminal cleartext level** is genuinely anchored: its witness is revealed and
read directly at `||w^(t)||_inf ≤ b/2`, with no commitment and no quotient.

One-hotness does not rescue anchoring. It sets `||s||_inf = 1`, which shrinks
`β^resp`, but it does not remove the `num_claims · 2^r_vars` fold factor. The
old `is_root` / `is_onehot` regime axis was the wrong axis; the correct split
is *committed (folded)* vs *terminal (cleartext)*, and one-hotness only enters
through the witness norm.

### The corrected bound

Every committed level is priced at the fold response, then at the **verifier digit
envelope** the stage-1 range check actually certifies:

```text
β^resp = num_claims · 2^r_vars · min(||c||_inf·||s||_1, ||c||_1·||s||_inf)
       = fold_witness_beta(...)
δ_fold = num_digits_fold(..., honest cap = min(β_inf, t*) when tail-bound-with-grind)
z_verifier = balanced_digit_abs_max(log_basis, δ_fold)

collision_A_inf = 8 · ω · z_verifier · ν
collision_A     = ceil_bucket(d · collision_A_inf²)   (L2 MSIS table)
```

`fold_witness_beta` still names the fold-response kernel bound; MSIS pricing uses
`z_verifier`, not raw `β_inf`, because the verifier accepts only balanced
`δ_fold`-digit coefficients.

This is implemented in
[`crates/akita-types/src/sis/norm_bound.rs`](../crates/akita-types/src/sis/norm_bound.rs)
(with fold-linf cap policy in
[`fold_linf_cap.rs`](../crates/akita-types/src/sis/fold_linf_cap.rs)):
`rounded_up_role_a_inf_norm` prices the
`8·ω·balanced_digit_abs_max·ν` coefficient-`L∞` envelope into the audited
A-role collision bucket; each call site then derives the level's secure A-role
rank from that bucket via `min_secure_rank`. Both thread `num_claims` and
`ring_subfield_norm_bound` from each call site (the planner DP in
`schedule_params.rs`, the runtime expansion, and the verifier-reachable layout
derivation in `layout/sis_derivation.rs`). The A-role price and `δ_fold` now
share `fold_witness_digit_plan`, so the binding rank and the digit count cannot
drift.

### Public-paper basis

This is the batched-soundness form of Hachi's weak-binding lemma (Hachi,
Lemma 7), specialized to a recursive random-linear-combination fold. The
fold-as-relaxed-binding view (the fold response is the quantity SIS binding is
proven against, at a norm proportional to the fold width) follows
Nguyen and Setty's SuperNeo interactive-reductions framework (ePrint 2026/242,
Theorem 4 plus the decomposition-reduction extractor): a random-linear fold is
a *weak* reduction whose relaxed binding is Module-SIS at norm `4·T·B`, with
`T = num_claims · 2^r` the fold width and `B` the response-difference bound
(Akita's `z^(ℓ,i) − z^0`); a norm check is a *strong* reduction that makes the
output witness short for the next input but does not lower the current fold's
binding norm. The Akita-specific batched statement and proof live in the
private Akita write-up and are not reproduced here.

### Consequences

- **Proof size rises at committed levels.** The `num_claims · 2^r` factor lifts
  the A-collision into higher SIS buckets, so committed levels need a larger
  A-rank. This is the cost of matching the proven security.
- **SIS collision ladder extended `2^20−1 → 2^26−1`.** Under the heavier norm a
  fp128 D128 batched dense root folds to a collision above the old `2^20−1`
  ceiling; without the extension it fell back to a cleartext root-direct proof.
  The generated SIS-floor tables now cover coefficient-`L∞` buckets up to
  `2^26−1`.
- **All shipped schedule tables regenerated** against the corrected norm.
- **Small-D families pruned (no longer securable).** fp32 D32 was dropped
  entirely, fp16 was removed from the production and profile paths, and the
  small-D fp128 / fp32 / fp64 families that cannot fold securely under honest
  pricing were removed. The shipping families are now fp128 D128 (plus the D64
  one-hot and D64 one-hot tensor), fp64 D128 / D256, and fp32 D128 / D256
  one-hot (see `akita_config::generated_families::ALL_GENERATED_FAMILIES`); fp32
  ships no dense family, and the smallest secure small-field ring degree is now
  `D = 128`.
- **Fixtures retargeted.** The terminal recursive fold's first stage-2 round is
  now degree-2; the zk setup-envelope test moves to fp128 D128; the impossible
  fp32 Terminal-root extension-reduction fixture is retired (fp32 has no 1-fold
  regime under honest pricing).
- **CI profile benchmarks re-pointed at securable profiles.** The benchmark
  matrix now benches only families that fold securely (fp128 D128 plus the
  small-field D128 one-hot families); the non-securable D32 / D64 small-field
  cells were removed. See
  [`specs/profile-bench-coverage-matrix.md`](profile-bench-coverage-matrix.md).

### Related fix on this branch: small-field ring-challenge soundness

This branch also ships a second, independent soundness fix that this spec does
not own: a real ≥128-bit ring-challenge policy for 64-bit-and-lower fields. The
historical small-field challenge was the toy `pm1-only { weight: 8, [−1, 1] }`,
which has only ~31 bits of Fiat-Shamir support at `D = 32`, far below 128-bit
soundness. It is replaced by the shared, dimension-keyed family specified in
[`specs/bounded-l1-sparse-challenge.md`](bounded-l1-sparse-challenge.md)
("Current Proof-Optimized Policy"): `D=32` `BoundedL1Norm` (`||c||_1 = 121`,
`||c||_inf = 8`), `D=64` `signed-sparse{30,12}` (`54, 2`), `D=128` `Uniform{31}`
(`31, 1`), `D=256` `Uniform{23}` (`23, 1`).

The two fixes are coupled in the regenerated tables: those `(||c||_1, ||c||_inf)`
values are exactly the challenge norms `fold_witness_beta` and the corrected
A-role collision above are generated against. The challenge *family* is
specified there, not here; this pointer exists so the norm reprice and the
challenge-security fix are not read in isolation.

### Still valid from the original design

Two pieces of the original spec are unaffected and remain in force:

- The negacyclic ring-product inequality
  `||c·s||_inf ≤ min(||c||_inf·||s||_1, ||c||_1·||s||_inf)` is still the kernel
  of both `fold_witness_beta` and the A-role price.
- The folded-witness digit optimization (a sparse one-hot witness takes the
  cheaper `||c||_inf` side, shrinking `δ_fold`) is unchanged; dense `δ_fold` is
  identical to before.

### Resolved follow-up (negative): the one-hot outer factor cannot drop to `||c||_inf`

For a one-hot committed witness the per-block product `β̄ = ||c·s||_inf`
already takes the `||c||_inf` side of the inner `min` (`||s||_inf = ||s||_1 = 1`),
but the A-role collision still pays the outer challenge-difference factor
`κ̄ = ||c − c'||_1 = 2·ω`. The open question was whether the one-hot structure
lets the fold response `z = z^(ℓ,i) − z^0` be bounded flexibly enough for the
collision to use `||c||_inf` rather than `||c||_1` in that outer factor too,
which would cut the one-hot collision (and A-rank) by a factor of `ω`.

It cannot, and lowering it would be unsound. The extractor's kernel vector is
`z_A = c̄'·(c̄·s) − c̄·(c̄'·s')`, i.e. `c̄'·z − c̄·z'` in the two fold responses
`z = c̄·s` and `z' = c̄'·s'`. The only norm the protocol certifies for a response
is its L∞ bound `||z||_inf ≤ β^resp` (the verifier's range check is L∞-only).
Each cross term obeys the ring-product `min`:

```text
||c̄'·z||_inf ≤ min( ||c̄'||_1·||z||_inf , ||c̄'||_inf·||z||_1 ).
```

The `||c||_inf` outer factor would have to come from the second side, which
needs `||z||_1`. Two independent facts close that door:

1. **No certified L1 mass.** The protocol bounds only `||z||_inf`. The single
   generic bound the extractor can derive is `||z||_1 ≤ D·||z||_inf`, and since
   any challenge has at most `D` nonzero coefficients, `D ≥ ||c||_1 / ||c||_inf`.
   Hence `||c̄'||_inf·D·||z||_inf ≥ ||c̄'||_1·||z||_inf`: the `||c||_1` side always
   wins the `min`, and the `||c||_inf` side never helps.
2. **A tight L1 would only tie.** Even granting the structural bound
   `||z||_1 ≤ T·||c||_1` (`T = num_claims·2^r_vars`; the one-hot fold response is a
   sum of `T` rotated challenges), the second side is
   `||c̄'||_inf·||z||_1 = ||c̄'||_inf·T·||c||_1`, while the first is
   `||c̄'||_1·||z||_inf = ||c̄'||_1·T·||c||_inf`. The two challenges are drawn from
   the same family (`||c||_1 = ||c'||_1`, `||c||_inf = ||c'||_inf`), so the two
   sides are *equal*. The one-hot fold response inherits the challenge's
   `||·||_1 / ||·||_inf` ratio, so the outer `min` is a no-op: there is no
   `ω`-factor to recover, with or without an L1 range check.

So the shipped `8·ω·balanced_digit_abs_max(δ_fold)·ν` bound is already the
tight one for the one-hot case; replacing the outer `||c||_1` with `||c||_inf` would
under-price the coefficient-`L∞` collision by a factor of `ω` and select SIS
ranks below the configured floor. No code change: the conservative `||c||_1` outer factor is also the correct one. The
one-hot A-rank therefore cannot be lowered by this route; any further one-hot
proof-size win has to come from the fold / digit side (already optimized via the
`min` and the digit envelope), not from the binding collision.

---

> **Everything below this line is superseded / historical.** It documents the
> original anchored A-role design (`collision_A = 2·ω̄·β̄·ν` with a single-block
> `β̄`) and its 2026-06-02 follow-ups. Where it prices the A-role at a single
> block, read the "Correction (2026-06-03)" section above instead. The
> `min(...)` ring-product bound and the one-hot fold optimization it describes
> are still correct.

## Implementation outcome (2026-06-02)

The fix is implemented as specified, with one consequence that required
regenerating the SIS-floor security tables and one deferred follow-up:

- **SIS-floor tables regenerated.** The corrected collision norm
  `collision_A = 2·ω̄·β̄·ν` reaches ~10^6 for the densest high-`ω` levels —
  far above the old maximum tabulated collision bucket (`2047`). The
  `akita_types::sis_floor` tables were regenerated with
  `scripts/gen_sis_table.py` (lattice-estimator, BDGL16 + lgsa) for **all
  families and dimensions (D = 32/64/128/256), ranks 1..=20, and collision
  buckets `2 … 1_048_575` (`2^k − 1` up to `2^20 − 1`)**. The estimator's
  `sis_lattice.cost_euclidean` trivial-easy bound is evaluated in log-space to
  avoid a float overflow at high rank / large `q` (an exact reformulation of
  its `min(term1, term2)`); all pre-existing table values are reproduced
  bit-for-bit.
- **Q16 dense presets ship cleartext-only.** The 16-bit modulus cannot
  securely commit the dense fold-witness widths at the corrected collision
  norms (the SIS-secure widths it admits are too small), so `fp16::*Full`
  schedules degrade to cleartext (`Direct`) — sound, but non-succinct. Q16
  one-hot and all fp32/fp64/fp128 families keep folding.
- **Deferred: dense poly under a one-hot tensor config.** `D64OneHotTensor`
  has `log_commit_bound == 1`, so the corrected fold `β` sizes against one-hot
  witness sparsity. Committing a *dense* poly under it folds to a larger
  `||z||_inf` than that `β`, so the prover aborts. The affected
  `single_poly_tensor_e2e::*dense_tensor*` tests are `#[ignore]`d pending a
  follow-up (the tensor + dense-witness β interaction).

## Follow-up fix (2026-06-02): root-dense witness L∞ off-by-one

While reviewing the consolidated `akita_types::sis` module, a second, smaller
soundness bug surfaced in the *same* A-role collision norm — this one
resolves the spec's Open Question 3 below.

### The bug

The committed witness `s` is shared by two prices: the A-role weak-binding
collision (`collision_A = 2·ω̄·β̄·ν`) and the fold bound
(`β = num_claims·2^r·β̄`). Both consume `(||s||_inf, ||s||_1)`, and per the
"clean symmetry" note above they **must** use the same witness norms. They
did not:

| level / encoding | fold (`fold_witness_norms`) | A-role (`a_role_witness_infinity_norm`) |
|---|---|---|
| one-hot | `1` | `1` ✅ |
| recursive dense | `2^(lb−1) = b/2` | `2^(lb−1) = b/2` ✅ |
| **root dense** | `2^(lb−1) = b/2` | `2^(lb−1) − 1 = b/2 − 1` ❌ |

`a_role_witness_infinity_norm` modelled the *root* dense witness as a
*symmetric* range `[−(b/2−1), b/2−1]` (the opening half-range `β`), dropping
the most-negative balanced digit `−b/2`.

### Why `b/2` is correct (not `b/2 − 1`)

The committed witness is a balanced base-`b = 2^lb` gadget decomposition whose
digits the prover bounds by `b/2`, identically at every level (no root
special-case):

```text
crates/akita-prover/src/kernels/linear/common.rs
  balanced_digit_abs_bound(log_basis) = 1 << (log_basis - 1) = b/2
```

`balanced_digit_max` documents the digit range as `[−b/2, b/2 − 1]`, so the max
*absolute* value (the L∞ norm) is `b/2` even though the max *positive* value is
`b/2 − 1`. The spec's own B/D paragraph already states the same:
`γ̄ = ||t̂||_inf = b/2` for a "balanced digit in `[−b/2, b/2)`". So the true
`||s||_inf` is `b/2 = 2^(lb−1)` at every level; the root-dense `b/2 − 1` is an
off-by-one **under-estimate**. For the A-role binding collision, under-estimating
`||s||_inf` is the **unsafe** direction (it under-prices the SIS collision and
can select a sub-128-bit rank), so this is a (small) soundness gap, not just a
cosmetic inconsistency.

### The fix

Make the A-role binding reuse `fold_witness_norms` as the single source of truth
for `(||s||_inf, ||s||_1)` (`1` one-hot / `b/2` dense, with
`||s||_1 = nonzeros·||s||_inf`), and **delete** `a_role_witness_infinity_norm`
and its root/recursive split. The spec's stated symmetry — "both binding and
fold feed the same un-doubled witness norms into the same helper" — now actually
holds in code.

### Consequences

- **Only dense families shift.** One-hot `||s||_inf = 1` was already correct, so
  every `*_onehot*` schedule table is byte-identical; only the dense
  (`*_full`, `fp{32,64}_d{32,64}`) families' root collision rises one notch
  (`b/2−1 → b/2` on the `min` side), occasionally bumping the root A-rank.
- **`fp16::D32Full` now ships fully cleartext (`commit: None`) for
  `num_vars >= 6`.** Previously it root-committed (`commit: Some`); the one-notch
  collision bump pushes the dense root A-rank above the 16-bit modulus's secure
  ceiling, so the DP drops even the root commitment. It still commits at the
  single-block size `num_vars = 5`. The `akita_e2e::fp16_static_dense_round_trip`
  test was retargeted from `num_vars = 8` to `5` so it keeps exercising a real
  SIS commitment. fp32/fp64 dense are unaffected at the tested sizes.
- **SIS-floor tables unchanged.** The lattice-estimator security tables
  (`sis/generated_sis_table.rs`) do not depend on the witness norm — only the
  collision *bucket* we look them up with — so they are not regenerated; only
  `crates/akita-planner/src/generated/*.rs` is.

## Summary

Two changes to how Akita prices ring-element products `c · s` in the
coefficient embedding of the negacyclic ring, both rooted in the same
inequality:

```text
||c · s||_inf  <=  min( ||c||_inf · ||s||_1 ,  ||c||_1 · ||s||_inf )
```

1. **Soundness bug (in scope, primary).** The A-role weak-binding collision
   norm computed in
   [`crates/akita-planner/src/ajtai_params.rs`](../crates/akita-planner/src/ajtai_params.rs)
   (`WitnessType::S::binding_norm`, lines 61–67) does **not** match Hachi
   Lemma 7. The extracted A-collision is `c_i c'_i (s_i − s'_i)`, bounded by
   `2 · ω̄ · β̄`, where `ω̄ = ||c_i||_1` (the challenge **L1** norm) and
   `β̄ = ||c_i · s_i||_inf`. The current code multiplies the witness bound
   (`a_role_base_norm`) by the challenge **L∞** norm
   (`stage1.infinity_norm()`) only — the **L1 factor `ω̄` from the
   cross-multiplication is missing entirely**, and the `β̄` term uses the
   invalid product `||c||_inf · ||s||_inf` instead of
   `min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`. The result is an
   **under-estimate** of the SIS coefficient-`L∞` bound, which is the exact
   quantity the production SIS-floor tables are indexed by. Under-pricing it
   selects SIS instances below the configured security floor for the true
   collision norm, so Module-SIS binding is no longer guaranteed.

2. **Optimization (in scope, secondary).** The folded-witness digit bound in
   [`crates/akita-types/src/layout/digit_math.rs`](../crates/akita-types/src/layout/digit_math.rs)
   (`compute_num_digits_fold_with_claims`, line 148) uses only the
   `||c||_1 · ||s||_inf` side of the inequality
   (`β = challenge_l1_mass · num_claims · 2^(r_vars + log_basis − 1)`). Taking
   the full `min(...)` lets sparse (one-hot) witnesses, where `||s_i||_1` is
   small, use the much smaller `||c||_inf` side. This shrinks `δ_fold`, the
   next-level witness, and proof size for one-hot presets at no security cost.

Both changes call **one shared helper** that computes the ring-product L∞ bound
`min(||c||_inf·||s||_1, ||c||_1·||s||_inf)` from explicit per-witness L1 and L∞
bounds. The only new input is the witness **L1 bound**, which for one-hot
depends on the chunk size `K` (the reviewer asked that `K` be passed in). Both
changes also alter the generated schedule tables and require a regeneration
pass.

## Background: the theory

### The ring-product L∞ inequality

In the negacyclic ring `R = Z[X]/(X^D + 1)`, for any `a, b ∈ R`:

```text
||a · b||_inf  <=  ||a||_1 · ||b||_inf          (each output coeff is a signed
                                                 sum of <= ||a||_1 copies of
                                                 b-coefficients)
||a · b||_inf  <=  ||a||_inf · ||b||_1          (symmetric)
=> ||a · b||_inf  <=  min( ||a||_1 · ||b||_inf , ||a||_inf · ||b||_1 )
```

`||a · b||_inf <= ||a||_inf · ||b||_inf` is **false** in general (it omits the
`D`-fold convolution sum); using it is the core mistake in the current
binding-norm code.

This is already the basis of the existing fold bound — see
[`specs/bounded-l1-sparse-challenge.md`](bounded-l1-sparse-challenge.md),
which tracks the challenge L1 mass precisely *because*
`||c · s||_inf <= ||c||_1 · ||s||_inf`. This spec extends the same reasoning to
(a) the weak-binding collision norm, which was never updated to track L1, and
(b) the `min(...)` refinement for sparse witnesses.

### Hachi Lemma 7 (Weak Binding) — first screenshot

A *weak opening* of a commitment `u` is a tuple `(s_i, t̂_i, c_i)_{i∈[2^r]}`
with, for all `i`:

```text
||c_i · s_i||_inf <= β̄ ,   ||c_i||_1 <= ω̄ ,   c_i ∈ R_q^× ,   A·s_i = G_{n_B}·t̂_i
B · [t̂_1; …; t̂_{2^r}] = u ,   ||[t̂_1; …; t̂_{2^r}]||_inf <= γ̄
```

**Lemma 7.** Given two weak openings `(s_i, t̂_i, c_i)` and
`(s'_i, t̂'_i, c'_i)` of `u` with `s_j ≠ s'_j` for some `j`, a deterministic
algorithm outputs `z` with `[A | B] z = 0` and

```text
0 < ||z||_inf <= max( 2·ω̄·β̄ , 2·γ̄ ).
```

**Where each bound comes from (collision extraction).** Both openings satisfy
`B·[t̂_i] = u`, so `B·([t̂_i] − [t̂'_i]) = 0`.

- **B/D-collision (the `2γ̄` term).** If `[t̂_i] ≠ [t̂'_i]`, this is a *direct*
  witness difference with `||[t̂_i] − [t̂'_i]||_inf <= ||t̂||_inf + ||t̂'||_inf
  <= 2γ̄`. **No challenge multiplication** here, so the only factor is the
  difference factor of 2.
- **A-collision (the `2ω̄β̄` term).** Otherwise `t̂_i = t̂'_i ∀i`, so
  `A·(s_i − s'_i) = 0`. The opening only bounds `||c_i s_i||_inf`, not
  `||s_i||_inf`, so the extractor uses the **cross-multiplication**
  `z_i = c'_i·(c_i s_i) − c_i·(c'_i s'_i) = c_i c'_i (s_i − s'_i)`. Then
  `A·z_i = 0` (units factor out), `z_i ≠ 0`, and

  ```text
  ||z_i||_inf <= ||c'_i||_1·||c_i s_i||_inf + ||c_i||_1·||c'_i s'_i||_inf
              <= ω̄·β̄ + ω̄·β̄ = 2·ω̄·β̄.
  ```

So the **A matrix** must be SIS-hard for `2·ω̄·β̄` and the **B/D matrices** for
`2·γ̄`. The A half is special: it carries the challenge **L1** factor `ω̄`
because the collision is multiplied by `c_i c'_i`. The B/D half does not.

The opening's `β̄` is itself a ring product, which is where the `min(...)`
enters:

```text
β̄ = ||c_i · s_i||_inf  <=  min( ||c_i||_inf·||s_i||_1 , ||c_i||_1·||s_i||_inf ),
=> collision_A = 2 · ω̄ · min( ||c||_inf·||s||_1 , ||c||_1·||s||_inf ).
```

### The factor of 2 (confirmed by reviewer)

Lemma 7's `max(2ω̄β̄, 2γ̄)` has a factor of 2 on both halves:

- **B/D** (`WitnessType::T | W::binding_norm`): already folded into
  `2^lb − 1 = 2γ̄`. With `γ̄ = ||t̂||_inf = b/2` (balanced digit in
  `[−b/2, b/2)`), the difference `t̂_i − t̂'_i ∈ [−(b−1), b−1]`, i.e.
  `||·||_inf <= b − 1 = 2^lb − 1 = 2γ̄`. **Correct as-is — no change.**
- **A-role**: the fix makes the factor of 2 **explicit** in the collision
  formula (`collision_A = 2·ω̄·β̄·ν`), so `β̄` is computed from the
  **un-doubled** witness norm `||s||_inf`. Today `a_role_base_norm`
  (`crates/akita-types/src/sis_offline.rs`) returns `2·||s||_inf` (the
  reviewer's phrasing: "computes `||s_i||_inf` and multiplies by 2"); the fix
  exposes the un-doubled `||s||_inf` and moves that `2` into the explicit `2 ·`
  of `2·ω̄·β̄`. The un-doubled witness L∞ is:
  - root, one-hot (`log_commit_bound == 1`): `1`
  - root, dense: `2^(lb−1) − 1`
  - recursive: the balanced-digit bound `≈ 2^(lb−1)` (exact value confirmed
    against the recursive witness construction — see Open Question 3).

The fix must **not** introduce a *second* factor of 2: there is exactly one,
the paper's, now written explicitly for A and left implicit (`2γ̄`) for B/D.

### The folded-witness bound β (Section 4.2 / Figure 3) — second screenshot

The fold produces `z = Σ_{i=1}^{2^r} c_i · s_i` — a **sum, not a difference**,
so there is **no factor of 2** here. The paper's naive bound is

```text
|| Σ c_i s_i ||_inf <= Σ_{i=1}^{2^r} ||c_i s_i||_inf <= Σ ω·b = 2^r · ω · b =: β,
```

i.e. it uses the `||c||_1 · ||s||_inf` side (`ω = ||c||_1`, `b ≈ ||s||_inf`).
The implementation encodes this as
`β = challenge_l1_mass · num_claims · 2^(r_vars + log_basis − 1)` (with
`2^(log_basis−1) = b/2`, the balanced-digit L∞). The optimization replaces the
single side with the full `min(...)`, which is tighter whenever the per-block
witness `||s_i||_1` is small (sparse / one-hot).

## Design

### One shared helper that reads like the theory

Both call sites should read as the inequality above:

```rust
// crates/akita-types/src/layout/digit_math.rs

/// Worst-case `||c · s||_inf` in the negacyclic ring, from the per-element
/// L1/L∞ bounds:
///
///     ||c · s||_inf  <=  min( ||c||_inf · ||s||_1 ,  ||c||_1 · ||s||_inf ).
pub fn ring_product_infinity_norm_bound(
    challenge_infinity_norm: u64,
    challenge_l1_norm: u64,
    witness_infinity_norm: u64,
    witness_l1_norm: u64,
) -> u64 {
    (challenge_infinity_norm.saturating_mul(witness_l1_norm))
        .min(challenge_l1_norm.saturating_mul(witness_infinity_norm))
}
```

No `χ` and no ratio appears in the formula: the two products of the `min` are
written exactly as the theory states them. The challenge norms come straight
from `SparseChallengeConfig` (`crates/akita-challenges/src/config.rs`):
`challenge_infinity_norm = stage1.infinity_norm()`,
`challenge_l1_norm = stage1.l1_norm()`.

### Per-witness L1 and L∞ bounds (explicit, named)

The only quantity the codebase does not already expose is the witness **L1**
bound. It is computed by one small, clearly-named helper so it never appears
inline inside the `min`:

```rust
/// Worst-case L1 mass of one committed witness ring element (block):
///   ||s||_1 <= nonzeros · ||s||_inf,
/// where `nonzeros` is the max number of hot coefficients per block:
///   - dense / full-field        : D            (every coefficient can be hot)
///   - one-hot, chunk size K >= D : 1            (single-chunk: <= 1 hot coeff)
///   - one-hot, chunk size K < D  : D / K        (multi-chunk: <= D/K hot coeffs)
fn witness_block_l1_norm(
    witness_infinity_norm: u64,
    ring_dimension: usize,
    onehot_chunk_size: usize,
) -> u64 {
    let nonzeros = (ring_dimension as u64).div_ceil(onehot_chunk_size as u64);
    witness_infinity_norm.saturating_mul(nonzeros)
}
```

(`onehot_chunk_size = K`; dense presets pass `K = 1`, giving `nonzeros = D`.
See `crates/akita-prover/src/backend/onehot/{entries,blocks}.rs`: a
single-chunk ring element carries `<= 1` hot coefficient, a multi-chunk one
`<= D/K`.) `nonzeros` is the **only** place `D/K` appears, and it is named, not
folded into the `min`.

### A-role binding (literal Lemma 7)

Compute `β̄` from the un-doubled witness norms, then multiply by `2 · ω̄`
exactly as `collision_A = 2·ω̄·β̄·ν`:

```rust
let challenge_infinity_norm = stage1.infinity_norm();   // ||c||_inf
let challenge_l1_norm       = stage1.l1_norm();         // ||c||_1 = ω̄
let witness_infinity_norm   = a_role_witness_infinity_norm(...)?;  // ||s||_inf  (un-doubled)
let witness_l1_norm =
    witness_block_l1_norm(witness_infinity_norm, ring_dimension, onehot_chunk_size); // ||s||_1

// β̄ = ||c · s||_inf
let beta = ring_product_infinity_norm_bound(
    challenge_infinity_norm,
    challenge_l1_norm,
    witness_infinity_norm,
    witness_l1_norm,
);

// Hachi Lemma 7:  collision_A = 2 · ω̄ · β̄ · ν
// (ν = ring_subfield_norm_bound, unchanged small-field embedding factor)
let collision_a = 2u64
    .saturating_mul(challenge_l1_norm)
    .saturating_mul(beta)
    .saturating_mul(ring_subfield_norm_bound);
```

This reads exactly as the theory: `beta` is `β̄ = ||c·s||_inf`, the `2` is
Lemma 7's factor, `challenge_l1_norm` is `ω̄`, and `ring_subfield_norm_bound`
is `ν`. The diff against current code is: swap the bare `* infinity_norm` for
`ring_product_infinity_norm_bound(...)`, add the `2 *` and the
`challenge_l1_norm *` factors, and source the witness L∞ from the un-doubled
`a_role_witness_infinity_norm` (the current `a_role_base_norm` with its built-in
`× 2` removed; that `2` now lives in the explicit formula).

`ν = ring_subfield_norm_bound` is **unchanged** — per the reviewer it is a
separate small-field `psi`-embedding concern, not in the paper, and stays a
plain multiplier.

### Fold (literal Section 4.2, with the `min`)

The fold has no difference and no `ω̄`; it is a sum of `2^r` per-block products,
each bounded by the same `min`. It uses the same un-doubled witness norms
(`witness_infinity_norm = b/2` for dense multi-digit blocks, `= 1` for the
un-decomposed one-hot value), so there is **no factor of 2**:

```rust
let challenge_infinity_norm = stage1.infinity_norm();
let challenge_l1_norm       = stage1.l1_norm();
let witness_infinity_norm   = commit_block_infinity_norm(...);  // b/2 (dense) or 1 (one-hot)
let witness_l1_norm =
    witness_block_l1_norm(witness_infinity_norm, ring_dimension, onehot_chunk_size);

// max_i ||c_i · s_i||_inf
let beta_block = ring_product_infinity_norm_bound(
    challenge_infinity_norm,
    challenge_l1_norm,
    witness_infinity_norm,
    witness_l1_norm,
);
// β = Σ over the 2^r blocks (× num_claims for batched roots) — no factor of 2 (a sum, not a collision)
let fold_beta = (num_claims as u64) * (1u64 << r_vars) * beta_block;
```

- **Dense** (`witness_infinity_norm = b/2`, `nonzeros = D`): the `min` picks
  `||c||_1·||s||_inf = challenge_l1_norm · b/2`, so `fold_beta` is **identical
  to today**.
- **One-hot** (`witness_infinity_norm = 1`, `nonzeros = 1`): the `min` picks
  `||c||_inf·||s||_1 = challenge_infinity_norm`, so
  `fold_beta = num_claims · 2^r · challenge_infinity_norm` — strictly smaller
  than today's `num_claims · 2^r · challenge_l1_norm · b/2`. Both the corrected
  `witness_infinity_norm = 1` and the `min` are required to realize the full
  drop.

Note the clean symmetry: both binding and fold feed the **same un-doubled
witness norms** into the **same helper**; they differ only in the outer factor —
binding is a collision of two openings (`2 · ω̄ ·`), the fold is a sum of blocks
(`num_claims · 2^r ·`).

### Passing `K` into the planner

The planner is `Cfg`-free and reads only `PlannerPolicy`. Add the one-hot chunk
size `K` so `witness_block_l1_norm` has its `nonzeros` input (mirroring how
`ring_subfield_embedding_norm_bound` is a Cfg hook projected onto
`PlannerPolicy`):

- Add a `CommitmentConfig` hook (e.g. `onehot_chunk_size() -> usize`), returning
  `K`. Dense presets return `K = 1` (⇒ `nonzeros = D`); one-hot presets return
  their actual chunk size (single-chunk presets return `K >= D` ⇒
  `nonzeros = 1`).
- Project it onto `PlannerPolicy` (e.g. `pub onehot_chunk_size: usize`), set in
  `akita-config`'s `policy_of::<Cfg>()`.

This keeps the `min` formula identical across dense, single-chunk one-hot, and
multi-chunk one-hot — only `nonzeros` (inside `witness_block_l1_norm`) changes.

### Numeric effect (sanity check, expressed via the `min`)

Writing `||s||_inf` for the un-doubled witness L∞ (so the current code, with its
doubled `a_role_base_norm = 2·||s||_inf`, equals `2·||c||_inf·||s||_inf·ν`):

| Backend | `min(...)` picks | `collision_A` correct | current | under-estimate |
|---|---|---|---|---|
| one-hot single-chunk | `||c||_inf · ||s||_1` | `2·||c||_1·||c||_inf·||s||_inf·ν` | `2·||c||_inf·||s||_inf·ν` | `||c||_1 = ω̄` |
| dense | `||c||_1 · ||s||_inf` | `2·||c||_1²·||s||_inf·ν` | `2·||c||_inf·||s||_inf·ν` | `||c||_1²/||c||_inf` |

`ω̄ = ||c||_1 = O(D)`, so the current code under-prices `collision_inf` by a
factor of order `D` (one-hot) to `D²/||c||_inf` (dense). Because
`collision_inf` is the exact infinity-norm the SIS-floor tables were certified
for, this is a genuine Module-SIS binding gap.

### Why the two changes are related

Both compute `||c · s||_inf` and call `ring_product_infinity_norm_bound`; both
consume the same `witness_block_l1_norm` (hence the same `K`) and the same
un-doubled witness norms. They differ only in the outer factor:

- **Binding**: `collision_A = 2 · ω̄ · β̄ · ν` — the `2` because the collision
  is a difference of two openings (the paper's factor, same as B/D), and the
  `ω̄ = ||c||_1` because the collision is `c_i c'_i (s_i − s'_i)`.
- **Fold**: `β = num_claims · 2^r · β̄` — a single sum `Σ c_i s_i`, so no
  difference (no `2`) and no `ω̄`.

They are independent in effect (binding raises ranks; fold lowers digit counts)
but ship together because both edit the same generated tables and both need `K`.

### Affected code and consistency points

The A-role formula and the fold β are each computed in several places that must
stay byte-for-byte consistent (prover, verifier, planner DP, and runtime table
expansion all re-derive layouts and must agree, per AGENTS.md's verifier
no-panic + planner determinism contracts):

**A-role binding norm (`collision_inf`):**

- `crates/akita-planner/src/ajtai_params.rs` — `WitnessType::S::binding_norm`
  (the reported site). Feeds `ajtai_a_width_bucket` → both the DP
  (`compute_ajtai_key_params_a`) and runtime expansion
  (`generated/expand.rs::expand_to_level_params`).
- `crates/akita-types/src/sis_offline.rs` — `a_role_collision_raw`
  (`a_raw · stage1.infinity_norm() · ν`) is a **second copy** of the same
  formula used by `sis_derived_root_params_for_layout`. Must be updated
  identically (call the same `ring_product_infinity_norm_bound` +
  `witness_block_l1_norm`, with the explicit `2 · ω̄`) or the DP and the SIS
  derivation drift. `a_role_base_norm` is refactored here to expose the
  un-doubled witness L∞ (`a_role_witness_infinity_norm`).
- `crates/akita-planner/src/schedule_params.rs:434` — uses
  `WitnessType::S::binding_norm` for the m/r-split scoring bucket; inherits the
  fix automatically.
- B/D (`WitnessType::T | W::binding_norm`) is **unchanged** (`2^lb − 1 = 2γ̄`).

**Fold β / `δ_fold`:**

- `crates/akita-types/src/layout/digit_math.rs` —
  `compute_num_digits_fold_with_claims` (reported site) and its use inside
  `optimal_m_r_split`. Both thread the commit-block `witness_infinity_norm` and
  `K` in.
- `crates/akita-types/src/layout/params.rs:340` — `LevelParams::num_digits_fold`
  (runtime, verifier-reachable).
- `crates/akita-prover/src/protocol/ring_relation.rs:40` —
  `beta_linf_fold_bound` is a **parallel prover-side copy** used by
  `validate_decompose_fold` (the `||z||_inf > β` prover-abort check). It must
  call the same helper in lock-step, or the prover aborts on valid one-hot
  witnesses (β too small) or the planner sizes too few digits vs the prover's β.

### Generated schedule tables

Both fixes change `collision_inf` buckets and/or `δ_fold`, hence ranks, widths,
and `total_bytes`, so every shipped table under
`crates/akita-planner/src/generated/*.rs` must be regenerated with the
`gen_schedule_tables` binary (owned by `akita-config`, per AGENTS.md). The
existing `old_tables/` snapshots and `tests/regen_diff.rs` /
`generated_tables.rs` machinery (already touched on this branch) diffs and
re-pins the tables.

## Evaluation

### Acceptance Criteria

- [ ] `ring_product_infinity_norm_bound(challenge_infinity_norm,
  challenge_l1_norm, witness_infinity_norm, witness_l1_norm)` exists and returns
  `min(challenge_infinity_norm·witness_l1_norm,
  challenge_l1_norm·witness_infinity_norm)`; both binding and fold call it (no
  inlined ratio).
- [ ] `WitnessType::S::binding_norm` returns
  `2 · challenge_l1_norm · ring_product_infinity_norm_bound(...) ·
  ring_subfield_norm_bound` (i.e. `2·ω̄·β̄·ν`), using the un-doubled witness L∞,
  with exactly one factor of 2.
- [ ] `a_role_collision_raw` in `sis_offline.rs` calls the same helpers; a test
  asserts the DP bucket and the SIS-derivation bucket agree.
- [ ] B/D binding norm is unchanged; a test pins `2^lb − 1` against `2γ̄`.
- [ ] `compute_num_digits_fold_with_claims` calls the helper with the un-doubled
  commit-block norms and no factor of 2; dense `δ_fold` is unchanged; one-hot
  `δ_fold` strictly smaller.
- [ ] `beta_linf_fold_bound` (prover) calls the helper; a one-hot
  prover/verifier round-trip does not abort.
- [ ] `K` is plumbed via a `CommitmentConfig` hook and `PlannerPolicy`; every
  shipped preset sets it so `witness_block_l1_norm`'s `nonzeros` matches its
  actual sparsity (incl. any multi-chunk preset).
- [ ] All `generated/*.rs` tables regenerated; `regen_diff` is clean.
- [ ] A unit test pins `collision_inf` against a hand-computed Lemma-7 value for
  one-hot single-chunk, multi-chunk, and dense.

### Testing Strategy

- Unit tests for `ring_product_infinity_norm_bound` (each side wins;
  saturation).
- Unit tests for `witness_block_l1_norm` (dense `D`, single-chunk `1`,
  multi-chunk `D/K`).
- Unit tests for `binding_norm` (one-hot single/multi-chunk vs dense; root vs
  recursive) pinning `collision_inf` to hand-computed Lemma-7 values, plus a
  regression test that the `challenge_l1_norm` (`ω̄`) factor is present.
- `digit_math` fold tests: dense unchanged; one-hot reduced; monotonicity in
  `num_claims` preserved.
- Consistency test: planner DP bucket == runtime expansion bucket
  (`ajtai_a_width_bucket`) == `sis_offline` derivation bucket.
- `cargo test` end-to-end (`akita-pcs`: `akita_e2e`, `single_poly_e2e`, `zk`)
  on at least one dense and one one-hot preset.
- Regenerate tables; confirm `tests/generated_tables.rs` / `tests/regen_diff.rs`
  pass.

### Performance / proof-size direction

- **Dense presets:** A-role `collision_inf` rises by ~`ω̄²/||c||_inf`, bumping
  it into a higher SIS bucket → larger A-rank → larger commitments/proofs. This
  is the cost of matching the proven security; measure with the planner
  (`estimate_proof_bytes`) and the `profile` example per shipped dense preset.
- **One-hot presets:** A-role `collision_inf` rises by ~`ω̄` (rank may grow),
  but `δ_fold` shrinks substantially (one-hot fold β drops to `||c||_inf`),
  shrinking the recursive witness. Net proof-size effect must be measured per
  preset; the optimization specifically offsets the binding-norm increase for
  one-hot.
- Report before/after `total_bytes` for every shipped family with
  `scripts/profile_bench_report.py`.

## Non-Goals

- No change to challenge sampling, `SparseChallengeConfig`, or its L1/L∞
  accessors (already correct).
- No change to the SIS-floor tables (`sis_floor.rs`); only the `collision_inf`
  we look them up with.
- **No change to the B/D collision norm** (`2^lb − 1 = 2γ̄` is correct).
- No change to `ring_subfield_norm_bound` (ν).
- No new backend; one-hot/dense is distinguished by `log_commit_bound` plus `K`.

## Open Questions

1. **`K` plumbing shape.** Confirm the `CommitmentConfig` hook signature for `K`
   and the default per preset (a single `onehot_chunk_size`, with dense = `1`).
2. **Multi-chunk presets today.** Are any shipped one-hot presets multi-chunk
   (`K < D`)? If all are single-chunk, `nonzeros = 1` covers shipped configs,
   but `K` plumbing is still required to stay sound for future presets.
3. **Recursive un-doubled witness L∞.** ~~`a_role_base_norm` returns `2^lb − 1`
   for recursive levels and `2β` for the root. Confirm the un-doubled witness
   L∞ used by the new `a_role_witness_infinity_norm` at recursive levels.~~
   **Resolved (2026-06-02)** — see "Follow-up fix" above. The committed witness
   is `||s||_inf = b/2 = 2^(lb−1)` at *every* level (the prover's
   `balanced_digit_abs_bound`), so the root/recursive split was wrong: the
   root-dense `b/2 − 1` under-counted the `−b/2` digit. The A-role now reuses
   `fold_witness_norms` and the split is deleted.

## References

- Hachi paper: Lemma 7 (Weak Binding); "Basic parameters" / Section 4.2 /
  Figure 3 (the fold bound `β = 2^r · ω · b`). (Screenshots in the originating
  task.)
- [`specs/bounded-l1-sparse-challenge.md`](bounded-l1-sparse-challenge.md) —
  prior art tracking challenge L1 mass for the fold bound.
- [`specs/tensor-structured-folding-challenges.md`](tensor-structured-folding-challenges.md)
  — `challenge_l1_mass` / `effective_l1_mass` definitions.
- Code: `crates/akita-planner/src/ajtai_params.rs`,
  `crates/akita-types/src/sis_offline.rs`,
  `crates/akita-types/src/layout/digit_math.rs`,
  `crates/akita-types/src/sis_floor.rs`,
  `crates/akita-prover/src/protocol/ring_relation.rs`,
  `crates/akita-prover/src/backend/onehot/{entries,blocks,decompose_fold}.rs`,
  `crates/akita-challenges/src/config.rs`.
