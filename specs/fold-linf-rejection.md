# Spec: Folded-Witness ∞-Norm Rejection (digit-count tightening)

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| Author(s)   | Quang Dao                                                 |
| Created     | 2026-06-10                                                |
| Status      | proposed                                                  |
| PR          | #174 (spec, targets `main`); implementation #189 (stacked on #188 → #186) |

## Summary

The fold response `z = Σ c_i · s_i` enters the next recursive level only through
its balanced base-`b` digit planes `z_hat`, and the plane count
`K = num_digits_fold` fixes the next level's width (Ajtai columns, sum-check
variables) and therefore a large slice of proof size. Today `K` is sized so the
structural per-coordinate bound `balanced_digit_max(lb, K)` covers the
**worst-case** coordinate envelope `β_inf = num_fold_blocks · ω · witness_linf`,
which assumes all `num_fold_blocks · ω` challenge-coefficient products align in
sign at one output position, an alignment the honest fold never attains. This
spec replaces that worst case with a **concentration tail bound** `t* < β_inf`
(sub-Gaussian argument) and a single **challenge reroll** step (Fiat–Shamir grind)
on certified flat fold challenges, so a level commits the smallest `K` with
`balanced_digit_max(lb, K) >= min(β_inf, t*)`. The prover expects **≤ 2**
rerolls per committed fold level (hard-capped). The verifier reads the `‖z‖_inf`
cap off the committed digits, never off the prover's accepting nonce; the nonce
only replays the accepted Fiat–Shamir challenge stream.

**Glossary (first use):**

| Symbol | Meaning |
|--------|---------|
| `ω` | challenge L1 mass `‖c‖_1` |
| `β_inf` | worst-case fold coordinate envelope (today's digit sizing) |
| `t*` | tighter cap from the concentration tail bound (new sizing when certified) |
| `K` | `num_digits_fold` (digit planes for `z_hat`) |
| `c_l2_sq_max` | family worst-case `max ‖c‖_2²` (per logical block) |

This is orthogonal to the L2-MSIS cutover (#155): that stack prices the **A-role
binding rank** (operator norm + Euclidean MSIS); this spec tightens the **fold
digit count** that sizes the **next-level width**. The two never touch the same
quantity.

The concentration argument for the approved flat-family cutover is reproduced in
the Design section below. This spec is self-contained and consistent with the
"Folded-Witness ∞-Norm Rejection" section of
[`specs/l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md).

## Intent

### Goal

Size `num_digits_fold` for sign-certified flat committed fold levels from a
concentration tail bound on `‖z‖_inf` instead of the worst-case envelope
`β_inf`, and add a transcript-bound challenge reroll that re-derives the fold
challenge until the realized `‖z‖_inf <= t*`.
The first approved implementation covers the flat challenge families whose
per-coordinate sign structure is proven in this spec: `ExactShell` at `d=64` and
`Uniform{[-1,1]}` at `d=128, 256`.
`BoundedL1Norm` and tensor-shaped folds keep the deterministic `β_inf` digit
sizing until separate proofs pin their tail constants.

The feature introduces or modifies:

- A per-family **worst-case squared ℓ₂ norm** `c_l2_sq_max = max ‖c‖_2²`
  (`SparseChallengeConfig::challenge_l2_sq_max`), the only new family-level
  quantity. Exact integer for every shipping family.
- A pure **tail-bound primitive**
  `fold_linf_tail_bound_sq(num_fold_blocks, c_l2_sq_max, witness_linf_sq, ln_term)` in
  `akita-types::sis::norm_bound` (squared domain, no floats on the
  verifier-reachable path).
- A **digit-sizing path** `num_digits_fold` that takes `K` from
  `min(β_inf, t*)` only when the level's threshold policy is certified.
  Unsupported policies return the existing `β_inf` sizing, with no reroll.
- One per-level **grind nonce** (`u32`) on the wire (`AkitaLevelProof`), absorbed
  before the fold challenge is squeezed, replayed by the verifier.
  The nonce cardinality is exactly one per `sample_folding_challenges` call in an
  intermediate committed fold level.
  `u32` is ample: rerolls halt in `<= 2` attempts in expectation and are
  hard-capped far below `2^16`.
- A **prover reroll loop** replacing the `validate_decompose_fold` abort.
- Planner/schedule awareness of the lowered `K` (regenerated shipped tables).

### Invariants

1. **Extraction bound tightens, Fiat–Shamir grinding is budgeted.** The verifier
   never reads the accepting nonce as evidence of `‖z‖_inf <= t*`. The stage-1
   range check structurally forces `|z_r| <= balanced_digit_max(lb, K)` against
   the level's published `K`. A smaller `K` is a tighter extraction bound, while
   the prover's nonce search is accounted for as bounded Fiat–Shamir grinding
   (see "Why it terminates and stays sound"). Protected by: existing stage-1
   range-check relation tests; new e2e tamper test that a `z` with
   `‖z‖_inf > balanced_digit_max(lb, K)` cannot produce an accepting transcript.
2. **Prover/verifier transcript equality.** The grind nonce is absorbed
   wire-before-squeeze, identically on both sides; the verifier re-derives the
   exact same fold challenge from `(transcript, accepted_nonce)`. Protected by:
   `LoggingTranscript` event-stream equality tests (the
   `logging-transcript` feature), extended to the fold-challenge reroll.
3. **Termination (completeness).** For every certified flat family and level, the
   chosen threshold `t*` satisfies `Pr[‖z‖_inf > t* | accepted] <= 1/2`, so rerolls
   halt in `<= 2` expected attempts. A hard attempt cap makes cap
   exhaustion a terminating, no-panic, prover-only error on pathological input.
   Unsupported families do not reroll. Protected by: a prover-side statistical
   test (sampled re-fold count stays small over many transcripts) and the
   capped-loop unit test.
4. **Planner/digit consistency.** The prover's grind threshold `t*` is the same
   value the planner used to size `K` (no drift), exactly as
   `beta_linf_fold_bound_with_num_claims` mirrors `num_digits_fold` today.
   Protected by: a shared `LevelParams` accessor consumed by both, plus the
   existing `generated_schedule_tables_match_find_schedule` drift guard after
   regen.
5. **No-panic on the verifier path.** The tail-bound primitive is integer-only and
   total; a malformed nonce / shape is rejected with `AkitaError` /
   `SerializationError`. Protected by: verifier no-panic audit + shape
   deserialization tests.
6. **Descriptor binding.** The active threshold policy (formula identity,
   certified-family set, per-family `c_l2_sq_max`, attempt cap, grind-nonce
   presence) is bound into `AkitaInstanceDescriptor`; a proof produced under the
   rejection policy must not verify under the legacy `β_inf` policy. Protected by:
   pinned descriptor-bytes test.

### Non-Goals

- **Not** the L2 Euclidean certificate (S6–S13 of the L2 spec): `Z_SQUARED`,
  four-square slack, the two linked sum-checks. Those price the A-role rank and
  are a separate stack.
- **Not** a calibrated/measured threshold (the `~0.03·β_inf` regime). The spec
  uses the rigorous `t*`; a calibrated constant is left as a future opt-in with a
  documented second-moment assumption (see Alternatives).
- **Not** a change to the challenge *sampler* distribution. The reroll is an outer
  loop over fold-challenge derivation; the per-attempt distribution is unchanged.
- **No** D=64 production preset change (the shell stays `(30, 12)`, `p = 1`); this
  spec only changes how `K` is sized given the existing shell.
- **No tensor or `BoundedL1Norm` threshold cutover in the first implementation.**
  Both continue to size `K` from `β_inf`; their tighter thresholds require a
  separate proof of the accepted-challenge tail bound and descriptor policy.

## Evaluation

### Acceptance Criteria

- [ ] `SparseChallengeConfig::challenge_l2_sq_max()` returns the exact
  worst-case `‖c‖_2²` for `Uniform`, `ExactShell`, `BoundedL1Norm`, validated by
  a unit test against hand-computed values.
- [ ] `fold_linf_tail_bound_sq(...)` is integer-only, total, monotone in
  each argument; digit sizing uses `min(β_inf, t*)` (raw `t*` may exceed the
  tight `fold_witness_beta`; the applied cap is always the minimum).
- [ ] `num_digits_fold` returns `K_reject <= K_worstcase` for every certified flat
  `(family, level, nv)`, strictly smaller at the wider folds, verified by a
  table test; tensor and `BoundedL1Norm` cases are pinned to `K_worstcase`.
- [ ] Shipped schedule tables regenerated; `generated_schedule_tables_match_find_schedule`
  passes (plain + zk), and `generated_families_stay_within_audited_sis_widths`
  still passes (the A-role rank is unaffected).
- [ ] Prover reroll loop terminates with mean attempts `< 2` over `>= 100`
  transcripts for each certified flat production mode at `nv ∈ {16, 28, 30}`.
- [ ] Headerless serialization is pinned: one `u32` nonce is serialized in every
  intermediate `AkitaLevelProof`, `LevelProofShape` has no variable nonce length,
  and serialized proof bytes match `level_proof_bytes` after adding four bytes
  per intermediate committed fold level.
- [ ] e2e prove/verify passes; a tampered `z` exceeding `balanced_digit_max(lb, K)`
  is rejected by the verifier.
- [ ] `LoggingTranscript` event-stream equality holds across the reroll.
- [ ] Descriptor bytes change intentionally and are pinned; cross-policy verify
  fails.
- [ ] Net proof-size improvement at the affected modes, reported by the profile
  command (direction: smaller next-level width at wide folds).

### Testing Strategy

Must keep passing: all `akita-types` sis/digit tests, the schedule drift guards,
e2e batched/recursive/zk suites, transcript tests.

New tests:

- `akita-challenges`: `challenge_l2_sq_max` per-family values; tensor
  `effective_l2_sq_max` (flat = `c_l2_sq_max`, tensor = factor product) for future
  policy binding, while the first digit-sizing cutover enables only flat shapes.
- `akita-types::sis`: tail-bound monotonicity, overflow/no-panic,
  `min(β_inf, t*)` sizing table; `fold_linf_ln_term` reference checks for the
  largest supported `num_fold_coeffs` and for a nontrivial `p_num/p_den` case.
- `akita-prover`: reroll-loop termination (statistical, sampled re-fold count);
  capped-loop error path; `LoggingTranscript` equality.
- `akita-verifier`: malformed-nonce / shape rejection (no panic).
- e2e: digit-bound tamper test; ZK parity if the feature is enabled.

Feature combinations: default, `--no-default-features`, `--features zk`,
`--features logging-transcript`.

### Performance

Expected direction: **smaller proofs**, no prover slowdown of note.

- `K` drops by up to one base-`b` digit at the wider folds (`t*/β_inf ≈ 0.20,
  0.14` at `num_fold_blocks = 16, 32`), shrinking the next level's Ajtai columns
  and sum-check variable count. Net proof-size change is reported by
  `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`
  and the planner `total_bytes` optimum.
- Prover cost: `<= 2` expected rerolls per committed level (each reroll is one
  challenge derivation + one fold pass), a small constant overhead, only on
  levels where `t* < β_inf` crosses a digit boundary.
- No verifier cost beyond consuming one extra nonce per intermediate committed
  fold level.
- A-role rank, setup size, and the L2 pricing are unchanged.

## Design

### Architecture

The change sits across the four existing layers, mirroring how the worst-case
`β_inf` already flows from the family config through the planner into the prover
abort:

```text
SparseChallengeConfig.challenge_l2_sq_max (c_l2_sq_max)    [akita-challenges]
        │
        ▼
LevelParams (witness_linf via fold_witness_norms, num_fold_blocks,
             num_fold_coeffs, p, policy)
        │  fold_linf_tail_bound_sq → t*
        ▼
num_digits_fold = min(β_inf, t*) → K                     [akita-types::sis]
        │
        ├──────────────► planner DP / shipped tables (K sizes next-level width)
        │
        ▼
prover reroll loop: sample fold challenge (nonce) → fold → accept if ‖z‖_inf ≤ t*
        │  accepted nonce → AkitaLevelProof.fold_grind_nonce  [akita-prover]
        ▼
verifier: absorb nonce → re-derive same challenge → digit-range check enforces K
        │                                                      [akita-verifier]
        ▼
AkitaInstanceDescriptor binds the threshold policy            [akita-config]
```

### Nonce and Wire Contract

The nonce is a fixed scalar field of every intermediate `AkitaLevelProof`:

```text
AkitaLevelProof {
    y_ring,
    extension_opening_reduction,
    v,
    fold_grind_nonce: u32,
    stage1,
    stage2,
    stage3_sumcheck_proof,
}
```

It is serialized immediately after `v` and before any stage-1 proof payload.
`LevelProofShape` does not carry a variable nonce length because the cardinality
is fixed at one `u32` per intermediate committed fold level. `level_proof_bytes`
adds exactly four bytes for every intermediate `AkitaLevelProof`. Terminal direct
proofs do not get a nonce in this cutover because they do not create a committed
next witness whose width is selected by `num_digits_fold`.

The nonce is one per `sample_folding_challenges` call. Under the single-point
opening batch contract (#186), a batched root uses one shared opening point and
one nonce for the whole stage-1 fold round: the prover samples one challenge
object, builds the folded witness, and accepts only if every emitted coefficient
is at most `t*`. Flat recursive intermediate folds use the same one-nonce rule.
Tensor folds do not reroll in this first cut, so they serialize nonce `0` and the
descriptor marks the tensor threshold policy as deterministic `β_inf`.

Level counters (conservative for planner/prover/verifier accessors):

```text
num_fold_coeffs = inner_width · D
num_fold_blocks = num_claims · 2^r_vars
```

`LevelParams::fold_linf_tail_bound_sq(num_claims)` and
`LevelParams::num_digits_fold(num_claims, field_bits)` are the shared accessors
used by planner/prover/verifier paths.

### Tail bound (statement)

Let `num_fold_blocks = num_claims · 2^r_vars` logical blocks in one stage-1
challenge call; let `witness_linf = ‖s‖_inf` be the per-block committed-witness
`∞`-norm (`1` one-hot, `b/2 = 2^(lb-1)` dense), and let
`num_fold_coeffs = inner_width · D` be the number of emitted folded witness
coefficients covered by the shared nonce.

The prover rerolls the fold challenge until `‖z‖_inf <= t*`, where `t*` is the
integer square root of `fold_linf_tail_bound_sq(...)`. Digit sizing uses
`min(β_inf, t*)` so a loose raw `t*` never widens `K` beyond the existing
worst-case bound.

### Proof sketch (concentration)

**Requirement (sign structure).** Each challenge `c`'s nonzero coefficients carry
**conditionally independent, symmetric (mean-zero) signs** given the support and
magnitude pattern. Fix an output coordinate `r` of `z = Σ_{(l,i)} c_{l,i}*s_{l,i}`.
Expanding the negacyclic products,

```text
z_r = Σ_{(l,i)} Σ_{a ∈ supp(c_{l,i})} ε_{l,i,a} · m_{l,i,a} · (± s_{l,i, r⊖a}),
```

a signed linear combination with independent `ε`; each term has magnitude at most
`m_{l,i,a}·witness_linf`. Conditioned on every support/magnitude pattern, a
second-moment upper bound on the sum is

```text
V_r = Σ Σ m² s²  ≤  witness_linf² · Σ_{(l,i)} ‖c_{l,i}‖_2²
       ≤ witness_linf² · num_fold_blocks · c_l2_sq_max  =:  V,
       c_l2_sq_max := max over the family of ‖c‖_2² (per block).
```

Hoeffding's inequality for independent ±1 signs gives
`Pr[|z_r| > t] <= 2·exp(-t²/2V)` for every conditioning (hence unconditionally),
and a union bound over the `num_fold_coeffs` coordinates:

```text
Pr[‖z‖_inf > t]  ≤  2·num_fold_coeffs·exp(-t²/2V).                    (T)
```

Let `p = Pr_c[Γ(c) <= Γ]` be the operator-norm acceptance probability of the
already-applied witness-independent rejection (`p = 1` when the cap does not bind;
production `(30,12)` ships with `T = 54 >= ‖c‖_1`, so `p = 1`). Conditioning on
accepted blocks gives

```text
Pr[‖z‖_inf > t | all num_fold_blocks accepted] ≤ (2·num_fold_coeffs / p^{num_fold_blocks})·exp(-t²/2V),
```

so

```text
t* = sqrt( 2·num_fold_blocks·c_l2_sq_max·witness_linf² · ( ln 4·num_fold_coeffs + num_fold_blocks·ln(1/p) ) )
```

makes the conditional miss probability `<= 1/2`: rerolls take `<= 2` attempts in
expectation. At `p = 1` this is
`t* = sqrt(2·num_fold_blocks·c_l2_sq_max·witness_linf²·ln 4·num_fold_coeffs)`;
against the ω-envelope `β_inf = num_fold_blocks·ω·witness_linf`, the gain ratio is
`t*/β_inf ≈ sqrt(2·c_l2_sq_max·ln 4·num_fold_coeffs)/(ω·sqrt(num_fold_blocks))`.
For `(c_l2_sq_max, ω) = (78, 54)`, `num_fold_coeffs ≈ 2^16`: `≈ 0.41, 0.29, 0.20,
0.14` at `num_fold_blocks = 4, 8, 16, 32`.

**Per-family `c_l2_sq_max` (all exact integers).**

| family                      | `c_l2_sq_max = max ‖c‖_2²` | note                                            |
|-----------------------------|----------------------------|-------------------------------------------------|
| `ExactShell{k1, k2}`        | `k1 + 4·k2`                | identical for every member; `(30,12) → 78`      |
| `Uniform{w, [-1,1]}`        | `w`                        | each nonzero `±1`; `d=128 → 31`, `d=256 → 23`   |
| `Uniform{w, coeffs}`        | `w · max_{a∈coeffs} a²`    | symmetric alphabet                              |
| `BoundedL1Norm` (M=8,B=121) | `M·B = 968` (safe), `961` exact | exposed for future policy; first cut keeps `β_inf` |

**Sign-structure status per family.**

- `ExactShell`: each nonzero gets an independent uniform sign
  (`sample_exact_shell_challenge` via `XofCursor::next_sign`). Exact.
- `Uniform{[-1,1]}`: each nonzero is iid uniform on the symmetric `{-1,+1}`.
  Exact. (A general symmetric alphabet keeps the proof; an asymmetric alphabet
  would not, but no preset uses one.)
- `BoundedL1Norm`: the full ball `{‖c‖_inf ≤ M, ‖c‖_1 ≤ B}` is sign-symmetric and
  the unrank `±a` buckets are equal-size (`suffix_count(remaining, budget-|a|)` is
  sign-independent), but the production sampler uses a fixed `2^128` rank prefix.
  This spec does not claim the prefix has the conditional sign-independence needed
  for (T). Its `challenge_l2_sq_max` value is exposed and tested, but
  `fold_linf_threshold_policy` returns deterministic `β_inf` for `BoundedL1Norm`
  until a separate certificate proves a tail bound for the truncated support.

**Tensor folds.** A tensor fold materializes the product `c = α_p · β_q`; the
signs are products `ε^α·ε^β` and are no longer independent across `(p,q)`. The
clean concentration argument does not apply directly. The code may expose
`effective_l2_sq_max = c_l2_sq_max(α)·c_l2_sq_max(β)` for future descriptor
binding, matching the shape of `effective_operator_norm_cap`, but the first
digit-count cutover treats tensor as unsupported and returns deterministic
`β_inf`. A future tensor threshold needs its own dependency-aware tail proof
before it can change `K`.

### Why it terminates and stays sound (restated)

- **Termination** is the `<= 1/2` miss probability above, capped at
  `MAX_FOLD_GRIND_ATTEMPTS = 4096`; exceeding the cap is a prover-only
  `AkitaError`, never verifier-reachable. The cap is intentionally the same order
  as `MAX_OP_NORM_ATTEMPTS`, but the expected count remains `<= 2`.
- **Soundness** is structural: the verifier enforces
  `|z_r| <= balanced_digit_max(lb, K)` via the stage-1 range check against the
  published `K`; the weak-binding extractor reads only accepting transcripts and
  never how `c` was sampled. The nonce gives the prover at most
  `MAX_FOLD_GRIND_ATTEMPTS` additional Fiat-Shamir challenge streams per
  committed fold level. The security statement accounts for this exactly as
  bounded grinding: if an adversary makes `Q` protocol attempts and a proof has
  at most `L` committed fold levels, the extractor's challenge-search budget is
  multiplied by at most `MAX_FOLD_GRIND_ATTEMPTS^L`, equivalently costing
  `L·log2(MAX_FOLD_GRIND_ATTEMPTS)` bits of challenge entropy. With the fixed cap
  above this is `12L` bits. The descriptor/security docs must state that the
  accepted fold-challenge support still exceeds
  `λ + log2(Q) + 12L` bits, refining the accepted-challenge entropy invariant from
  the in-repo L2 spec.

### Precise diff surface

Crate-by-crate, smallest coherent change (no sibling `_v2` functions; the
worst-case path is generalized in place):

**`akita-challenges`**

- `src/config.rs`: add `pub fn challenge_l2_sq_max(&self) -> u128` to
  `SparseChallengeConfig` (table above). Pure; mirrors the existing `l1_norm` /
  `operator_norm_cap` accessors.
- `src/tensor.rs`: add `pub fn effective_l2_sq_max(&self, cfg) -> u128` to
  `ChallengeShape` (`Flat → c_l2_sq_max`, `Tensor → product`), mirroring
  `effective_operator_norm_cap`.
- `src/sampler/mod.rs`: extend `sample_folding_challenges` (and the inner
  `sample_sparse_challenges`) with a `grind_nonce: u32` that is folded into
  `sparse_challenge_absorb_buf` (a new field after the config domain separator),
  so an incremented nonce yields an independent transcript-derived stream while
  staying prover/verifier-replayable. Unsupported policies pass nonce `0`.

**`akita-types`**

- `src/sis/norm_bound.rs`: add
  `fold_linf_tail_bound_sq(num_fold_blocks, c_l2_sq_max, witness_linf_sq, ln_term) -> Result<u128, AkitaError>`
  returning `t*²` (squared domain, exact `u128`, saturating/no-panic). The only
  irrational input is `ln 4·num_fold_coeffs + num_fold_blocks·ln(1/p)`; pass it
  as a conservative integer `ln_term` via `fold_linf_ln_term(num_fold_coeffs,
  num_fold_blocks, p_num, p_den)` (table-bounded for `num_fold_coeffs <= 2^32`,
  `ln 4·num_fold_coeffs <= 24`). Document that the real sqrt is taken only at
  the digit-sizing boundary.
- `src/sis/decomposition_digits.rs`: `num_digits_fold` gains the tail-bound inputs
  (`c_l2_sq_max`, `num_fold_coeffs`, `p`, `policy`) and sizes `K` from
  `min(β_inf, isqrt_ceil(t*²))` only for certified flat policies. Keep the
  degenerate guards and return `β_inf` for deterministic policies.
- `src/sis/mod.rs`: re-export the new primitive.
- `src/layout/params.rs`: `LevelParams::num_digits_fold` passes the new inputs
  (`challenge_l2_sq_max` via `stage1_config`, `inner_width()·D`, the op-norm
  acceptance `p`, and the threshold policy). Add
  `fold_linf_tail_bound_sq(num_claims)` so the prover reads the identical value
  (invariant 4).
- `src/proof/levels.rs`: add `fold_grind_nonce: u32` (one `u32` per intermediate
  committed fold level) to `AkitaLevelProof`; update constructors and
  serialization. `TerminalLevelProof` is unchanged in this cutover.
- `src/proof/shapes.rs` + `src/layout/proof_size.rs`: keep `LevelProofShape`
  nonce-length-free, update shape/serialization tests, and extend the proof-size
  formula by the fixed nonce bytes.

**`akita-planner`**

- `src/schedule_params.rs` + `src/generated/expand.rs`: thread `c_l2_sq_max` /
  `num_fold_coeffs` / `p` / policy into the DP's `num_digits_fold` call so a
  lowered `K` is searched only for certified flat levels.
- Regenerate `src/generated/*.rs` (plain + zk) via the existing
  `gen_schedule_tables` binary.

**`akita-prover`**

- `src/protocol/ring_relation.rs`: replace `validate_decompose_fold`'s abort with
  a capped reroll loop around `sample_folding_challenges` →
  `build_point_decompose_fold_witness`/`decompose_fold` → accept first `z` with
  `‖z‖_inf <= t*` (read from `lp.fold_linf_tail_bound_sq(num_claims)`). Record
  the accepted nonce into the level proof. Unsupported policies skip the loop and
  use nonce `0`. The nonce is absorbed before the challenge squeeze (already the
  absorb point in `sample_folding_challenges`).

**`akita-verifier`**

- `src/protocol/batched.rs` / `ring_switch.rs`: read `fold_grind_nonce` from the
  proof, pass it to `sample_folding_challenges`, reject nonzero nonce under
  deterministic policies with `AkitaError`. No new norm check (the digit-range
  check already enforces `K`).

**`akita-config`**

- Instance-descriptor binding: add the threshold-policy identity (formula tag,
  certified-family set, deterministic-policy set, per-family `c_l2_sq_max`,
  attempt cap, nonce presence, and `12L` entropy-budget rule) to
  `bind_transcript_instance_descriptor`; pin the bytes.

### Alternatives Considered

- **Calibrated `~0.03·β_inf` threshold** from the `z_rms`/`mu2_implied` tables.
  Tighter (smaller `K`), but termination then rests on an unproven second-moment
  assumption about honest witnesses; a denser-than-calibration witness re-folds
  more and can hit the cap (completeness risk). Deferred as an opt-in policy with
  a documented assumption, gated behind the same `LevelParams` threshold accessor
  so swapping it in is a one-line policy change.
- **Keep `β_inf`, no rejection.** The status quo: correct but pessimistic, leaving
  up to a base-`b` digit of next-level width on the table at wide folds.
- **Witness-independent threshold (no nonce on the wire).** Impossible: `‖z‖_inf`
  depends on the secret `s`, so the verifier cannot replay which challenge passed.
  The nonce is the minimal wire cost (one `u32` per level).
- **Tensor: exact expanded-product `c_l2_sq_max`** instead of the factor product
  bound. Tighter but requires modeling the dependent product signs. Deferred until
  a tensor-specific tail proof exists; first cut leaves tensor at `β_inf`.
- **BoundedL1 threshold with an empirical/inflated constant.** Rejected for the
  approved cutover. The fixed `2^128` rank prefix needs a proof or certificate of
  the conditional sign tail; an inflated constant without that artifact would make
  termination a conjecture.

## Documentation

- Update the "Folded-Witness ∞-Norm Rejection" section of
  [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) to point
  at this spec, mark flat `ExactShell`/`Uniform{[-1,1]}` as certified, and mark
  tensor/`BoundedL1Norm` as deterministic `β_inf` pending separate proofs.
- Crate docs on `num_digits_fold` and the tail-bound primitive, stating the
  per-family `c_l2_sq_max` table and the sign-symmetry requirement inline.
- Public security-model docs: extend the challenge-distribution / norm-bound
  description with the rigorous `‖z‖_inf` reroll threshold and the per-family
  `c_l2_sq_max` table from this spec.

## Execution

Slices (each independently reviewable; W0 are pure and unblock the rest):

```text
W0 (pure, parallel)
  F1  challenge_l2_sq_max + effective_l2_sq_max              [akita-challenges]
  F2  threshold policy + fold_linf_tail_bound_sq + ln term   [akita-types::sis]

W1
  F3  num_digits_fold sizes K from min(β_inf, t*)            [akita-types::sis]  (F2)
      for certified flat policies only
  F4  LevelParams tail-bound accessor                        [akita-types]       (F2,F3)
  F6  grind nonce: sampler param + proof field + shape       [challenges,types]

W2
  F5  planner DP + regenerate shipped tables                 [akita-planner]     (F3,F4)
  F7  prover reroll loop + accepted nonce                    [akita-prover]      (F4,F6)

W3
  F8  verifier replay + no-panic                             [akita-verifier]    (F6,F7)
  F9  descriptor binding + pinned bytes                       [akita-config]      (F4,F6)

W4
  F10 e2e tamper / termination / ZK parity tests             (all)
```

Resolved before approval: `BoundedL1` and tensor are scoped to deterministic
`β_inf` in the first implementation; `num_fold_coeffs = inner_width · D` under
the single-point batch contract; the nonce is a fixed `u32` per intermediate
committed fold level.

## References

- [`specs/l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md)
  ("Folded-Witness ∞-Norm Rejection" section; accepted-challenge entropy
  invariant).
- `crates/akita-types/src/sis/{norm_bound,decomposition_digits,ajtai_key}.rs`
- `crates/akita-types/src/layout/{params,proof_size}.rs`,
  `crates/akita-types/src/proof/levels.rs`
- `crates/akita-prover/src/protocol/ring_relation.rs`
  (`validate_decompose_fold`, `sample_folding_challenges` call sites)
- `crates/akita-challenges/src/{config,tensor}.rs`,
  `crates/akita-challenges/src/sampler/{mod,exact_shell,uniform,bounded_l1}.rs`
- Profile: `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`
