# Spec: Folded-Witness ∞-Norm Rejection (digit-count tightening)

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| Author(s)   | Quang Dao                                                 |
| Created     | 2026-06-10                                                |
| Status      | proposed                                                  |
| PR          | stacked on #155 (`quang/s3-s5-sis-estimator-spec`)        |

## Summary

The fold response `z = Σ c_i · s_i` enters the next recursive level only through
its balanced base-`b` digit planes `z_hat`, and the plane count
`K = num_digits_fold` fixes the next level's width (Ajtai columns, sum-check
variables) and therefore a large slice of proof size. Today `K` is sized so the
structural per-coordinate bound `balanced_digit_max(lb, K)` covers the
**worst-case** coordinate envelope `β_inf = T_p · ω · σ_inf`, which assumes all
`T_p · ω` challenge-coefficient products align in sign at one output position, an
event the honest fold never attains. This spec replaces that worst case with a
**rigorous sub-Gaussian threshold** `t* < β_inf` and a single witness-dependent
rejection (grind) step on the fold challenge, so a level commits the smallest `K`
with `balanced_digit_max(lb, K) >= t*`. The grind provably terminates in `O(1)`
expected re-folds, and soundness is untouched: the verifier reads the `‖z‖_inf`
bound off the committed digits, never off the prover's accepting nonce.

This is a stacked follow-on to the L2-MSIS cutover (#155), **orthogonal** to it:
#155 / the deferred L2 certificate price the **A-role binding rank** (operator
norm + Euclidean MSIS); this spec tightens the **fold digit count** that sizes the
**next-level width**. The two never touch the same quantity.

The full sub-Gaussian argument is reproduced from first principles in the Design
section below, so this spec is self-contained. It is consistent with the
"Folded-Witness ∞-Norm Rejection" section of the in-repo
[`specs/l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md).

## Intent

### Goal

Size `num_digits_fold` for every committed fold level from a rigorous
sub-Gaussian `‖z‖_inf` threshold `t*` instead of the worst-case envelope `β_inf`,
and add a transcript-bound, witness-dependent rejection step that re-derives the
fold challenge until the realized `‖z‖_inf <= t*`, with a proven `O(1)` expected
re-fold count for every production challenge family (`ExactShell` at `d=64`,
`Uniform{[-1,1]}` at `d=128, 256`, and, with a documented looser constant,
`BoundedL1Norm` at `d=32`).

The feature introduces or modifies:

- A per-family **worst-case challenge energy** `ρ2 = max ‖c‖_2²`
  (`SparseChallengeConfig::challenge_energy_rho2`), the only new family-level
  quantity. Exact integer for every shipping family.
- A pure **threshold primitive**
  `fold_response_linf_threshold_sq(T_p, ρ2, σ_inf², N, p)` in
  `akita-types::sis::norm_bound` (squared domain, no floats on the
  verifier-reachable path).
- A **digit-sizing path** `num_digits_fold` that takes `K` from
  `min(β_inf, t*)`.
- A per-level **grind nonce** (`u32`) on the wire (`AkitaLevelProof`), absorbed
  before the fold challenge is squeezed, replayed by the verifier. `u32` is
  ample: the grind halts in `<= 2` attempts in expectation and is hard-capped far
  below `2^16`, so the nonce never approaches even a `u16` ceiling; `u32` is
  chosen only for alignment and headroom.
- A **prover grind loop** replacing the `validate_decompose_fold` abort.
- Planner/schedule awareness of the lowered `K` (regenerated shipped tables).

### Invariants

1. **Soundness unchanged.** The verifier never reads the accepting nonce as
   evidence of `‖z‖_inf <= t*`. The stage-1 range check structurally forces
   `|z_r| <= balanced_digit_max(lb, K)` against the level's published `K`. A
   smaller `K` is a *tighter* extraction bound, so binding can only strengthen.
   Protected by: existing stage-1 range-check relation tests; new e2e tamper test
   that a `z` with `‖z‖_inf > balanced_digit_max(lb, K)` cannot produce an
   accepting transcript.
2. **Prover/verifier transcript equality.** The grind nonce is absorbed
   wire-before-squeeze, identically on both sides; the verifier re-derives the
   exact same fold challenge from `(transcript, accepted_nonce)`. Protected by:
   `LoggingTranscript` event-stream equality tests (the
   `logging-transcript` feature), extended to the fold-challenge grind.
3. **Termination (completeness).** For every shipping family and level, the
   chosen threshold `t*` satisfies `Pr[‖z‖_inf > t* | accepted] <= 1/2`, so the
   grind halts in `<= 2` expected attempts; a hard attempt cap makes it a
   terminating, no-panic, prover-only error on pathological input. Protected by:
   a prover-side statistical test (sampled re-fold count stays small over many
   transcripts) and the capped-loop unit test.
4. **Planner/digit consistency.** The prover's grind threshold `t*` is the same
   value the planner used to size `K` (no drift), exactly as
   `beta_linf_fold_bound_with_num_claims` mirrors `num_digits_fold` today.
   Protected by: a shared `LevelParams` accessor consumed by both, plus the
   existing `generated_schedule_tables_match_find_schedule` drift guard after
   regen.
5. **No-panic on the verifier path.** The threshold primitive is integer-only and
   total; a malformed nonce / shape is rejected with `AkitaError` /
   `SerializationError`. Protected by: verifier no-panic audit + shape
   deserialization tests.
6. **Descriptor binding.** The active threshold policy (formula identity,
   per-family `ρ2`, attempt cap, grind-nonce presence) is bound into
   `AkitaInstanceDescriptor`; a proof produced under the rejection policy must not
   verify under the legacy `β_inf` policy. Protected by: pinned descriptor-bytes
   test.

### Non-Goals

- **Not** the L2 Euclidean certificate (S6–S13 of the L2 spec): `Z_SQUARED`,
  four-square slack, the two linked sum-checks. Those price the A-role rank and
  are a separate stack.
- **Not** a calibrated/measured threshold (the `~0.03·β_inf` regime). The spec
  uses the rigorous `t*`; a calibrated constant is left as a future opt-in with a
  documented second-moment assumption (see Alternatives).
- **Not** a change to the challenge *sampler* distribution. The grind is an outer
  loop over fold-challenge derivation; the per-attempt distribution is unchanged.
- **No** D=64 production preset change (the shell stays `(30, 12)`, `p = 1`); this
  spec only changes how `K` is sized given the existing shell.

## Evaluation

### Acceptance Criteria

- [ ] `SparseChallengeConfig::challenge_energy_rho2()` returns the exact
  worst-case `‖c‖_2²` for `Uniform`, `ExactShell`, `BoundedL1Norm`, validated by
  a unit test against hand-computed values.
- [ ] `fold_response_linf_threshold_sq(...)` is integer-only, total, monotone in
  each argument, and `<= β_inf²` for every shipping `(family, level)` (else it is
  clamped to `β_inf`, never above).
- [ ] `num_digits_fold` returns `K_reject <= K_worstcase` for every shipping
  `(family, level, nv)`, strictly smaller at the wider folds, verified by a
  table test.
- [ ] Shipped schedule tables regenerated; `generated_schedule_tables_match_find_schedule`
  passes (plain + zk), and `generated_families_stay_within_audited_sis_widths`
  still passes (the A-role rank is unaffected).
- [ ] Prover grind loop terminates with mean attempts `< 2` over `>= 100`
  transcripts for each production mode at `nv ∈ {16, 28, 30}`.
- [ ] e2e prove/verify passes; a tampered `z` exceeding `balanced_digit_max(lb, K)`
  is rejected by the verifier.
- [ ] `LoggingTranscript` event-stream equality holds across the grind.
- [ ] Descriptor bytes change intentionally and are pinned; cross-policy verify
  fails.
- [ ] Net proof-size improvement at the affected modes, reported by the profile
  command (direction: smaller next-level width at wide folds).

### Testing Strategy

Must keep passing: all `akita-types` sis/digit tests, the schedule drift guards,
e2e batched/multipoint/recursive/zk suites, transcript tests.

New tests:

- `akita-challenges`: `challenge_energy_rho2` per-family values; tensor
  `effective_energy_rho2` (flat = `ρ2`, tensor = factor product).
- `akita-types::sis`: threshold primitive monotonicity, overflow/no-panic,
  `t* <= β_inf` clamp; `num_digits_fold` reject-vs-worstcase table.
- `akita-prover`: grind-loop termination (statistical, sampled re-fold count);
  capped-loop error path; `LoggingTranscript` equality.
- `akita-verifier`: malformed-nonce / shape rejection (no panic).
- e2e: digit-bound tamper test; ZK parity if the feature is enabled.

Feature combinations: default, `--no-default-features`, `--features zk`,
`--features logging-transcript`.

### Performance

Expected direction: **smaller proofs**, no prover slowdown of note.

- `K` drops by up to one base-`b` digit at the wider folds (`t*/β_inf ≈ 0.20,
  0.14` at `T_p = 16, 32`), shrinking the next level's Ajtai columns and sum-check
  variable count. Net proof-size change is reported by
  `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`
  and the planner `total_bytes` optimum.
- Prover cost: `<= 2` expected re-folds per committed level (each re-fold is one
  challenge derivation + one fold pass), a small constant overhead, only on
  levels where `t* < β_inf` crosses a digit boundary.
- No verifier cost beyond consuming one extra nonce per level.
- A-role rank, setup size, and the L2 pricing are unchanged.

## Design

### Architecture

The change sits across the four existing layers, mirroring how the worst-case
`β_inf` already flows from the family config through the planner into the prover
abort:

```text
SparseChallengeConfig.challenge_energy_rho2 (ρ2)         [akita-challenges]
        │
        ▼
LevelParams (σ_inf via fold_witness_norms, T_p, N=inner_width·D, p)
        │  fold_response_linf_threshold_sq → t*
        ▼
num_digits_fold = min(β_inf, t*) → K                     [akita-types::sis]
        │
        ├──────────────► planner DP / shipped tables (K sizes next-level width)
        │
        ▼
prover grind loop: sample fold challenge (nonce) → fold → accept if ‖z‖_inf ≤ t*
        │  accepted nonce → AkitaLevelProof.fold_grind_nonce  [akita-prover]
        ▼
verifier: absorb nonce → re-derive same challenge → digit-range check enforces K
        │                                                      [akita-verifier]
        ▼
AkitaInstanceDescriptor binds the threshold policy            [akita-config]
```

### The sub-Gaussian threshold (all families)

Let a level fold `T_p = num_claims · 2^r_vars` blocks; let `σ_inf = ‖s‖_inf` be the
per-block committed-witness `∞`-norm (`1` one-hot, `b/2 = 2^(lb-1)` dense), and let
`N = inner_width · D` be the number of `z` coefficients.

**Requirement (sign structure).** Each challenge `c`'s nonzero coefficients carry
**conditionally independent, symmetric (mean-zero) signs** given the support and
magnitude pattern. Fix an output coordinate `r` of `z = Σ_{(l,i)} c_{l,i}*s_{l,i}`.
Expanding the negacyclic products,

```text
z_r = Σ_{(l,i)} Σ_{a ∈ supp(c_{l,i})} ε_{l,i,a} · m_{l,i,a} · (± s_{l,i, r⊖a}),
```

a zero-mean Rademacher sum in the independent signs `ε` with weights of magnitude
`m_{l,i,a}·|s| <= m_{l,i,a}·σ_inf`. Conditioned on every support/magnitude pattern,
its variance proxy is

```text
V_r = Σ Σ m² s²  ≤  σ_inf² · Σ_{(l,i)} ‖c_{l,i}‖_2²  ≤  σ_inf² · T_p · ρ2  =:  V,
       ρ2 := max over the family of ‖c‖_2² (per block).
```

Hoeffding for Rademacher sums gives `Pr[|z_r| > t] <= 2·exp(-t²/2V)` for every
conditioning (hence unconditionally), and a union bound over the `N` coordinates:

```text
Pr[‖z‖_inf > t]  ≤  2N·exp(-t²/2V).                              (T)
```

Let `p = Pr_c[Γ(c) <= Γ]` be the operator-norm acceptance probability of the
already-applied witness-independent rejection (`p = 1` when the cap does not bind;
production `(30,12)` ships with `T = 54 >= ‖c‖_1`, so `p = 1`). Bayes against (T)
on the unconditioned event over the `T_p` accepted blocks gives

```text
Pr[‖z‖_inf > t | all T_p blocks accepted] ≤ (2N / p^{T_p})·exp(-t²/2V),
```

so

```text
t* = sqrt( 2·T_p·ρ2·σ_inf² · ( ln 4N + T_p·ln(1/p) ) )
```

makes the conditional miss probability `<= 1/2`: the grind re-folds `<= 2` times in
expectation. At `p = 1` this is `t* = sqrt(2·T_p·ρ2·σ_inf²·ln 4N)`; the gain ratio
is `t*/β_inf = sqrt(2·ρ2·ln 4N)/(ω·sqrt(T_p))`, independent of `σ_inf` and growing
only as `sqrt(ln N)`. For `(ρ2, ω) = (78, 54)`, `N ≈ 2^16`: `≈ 0.41, 0.29, 0.20,
0.14` at `T_p = 4, 8, 16, 32`.

**Per-family `ρ2` (all exact integers).**

| family                      | `ρ2 = max ‖c‖_2²`         | note                                            |
|-----------------------------|---------------------------|-------------------------------------------------|
| `ExactShell{k1, k2}`        | `k1 + 4·k2`               | identical for every member; `(30,12) → 78`      |
| `Uniform{w, [-1,1]}`        | `w`                       | each nonzero `±1`; `d=128 → 31`, `d=256 → 23`   |
| `Uniform{w, coeffs}`        | `w · max_{a∈coeffs} a²`   | symmetric alphabet                              |
| `BoundedL1Norm` (M=8,B=121) | `M·B = 968` (safe), `961` exact | `‖c‖_2² ≤ ‖c‖_inf·‖c‖_1 ≤ M·B`           |

**Sign-structure status per family.**

- `ExactShell`: each nonzero gets an independent uniform sign
  (`sample_exact_shell_challenge` via `XofCursor::next_sign`). Exact.
- `Uniform{[-1,1]}`: each nonzero is iid uniform on the symmetric `{-1,+1}`.
  Exact. (A general symmetric alphabet keeps the proof; an asymmetric alphabet
  would not, but no preset uses one.)
- `BoundedL1Norm`: the full ball `{‖c‖_inf ≤ M, ‖c‖_1 ≤ B}` is sign-symmetric and
  the unrank `±a` buckets are equal-size (`suffix_count(remaining, budget-|a|)` is
  sign-independent), so conditioned on the magnitude pattern the signs are exactly
  independent uniform in the **untruncated** set. The `2^128`-rank truncation
  perturbs this by a bounded amount; since the threshold only governs termination
  (completeness, not soundness), we adopt a documented looser constant for `d=32`
  (e.g. inflate `ρ2` or the `ln` term by a small fixed factor) and note `d=32`
  bounded-L1 is not currently in production schedules. Lowest priority of the
  three.

**Tensor folds.** A tensor fold materializes the product `c = α_p · β_q`; the
signs are products `ε^α·ε^β` and are no longer independent across `(p,q)`. The
clean Rademacher argument does not apply directly, so we take the conservative
product energy `ρ2_eff = ρ2(α)·ρ2(β)` (the analogue of
`effective_operator_norm_cap`'s `Γ²` for tensor) and a looser constant, sufficient
for termination. This resolves L2-spec Open Question 5 for the digit-count axis;
the A-role pricing keeps its own answer.

### Why it terminates and stays sound (restated)

- **Termination** is the `<= 1/2` miss probability above, capped at
  `MAX_FOLD_GRIND_ATTEMPTS` (mirroring `MAX_OP_NORM_ATTEMPTS = 4096`); exceeding
  the cap is a prover-only `AkitaError`, never verifier-reachable.
- **Soundness** is structural: the verifier enforces `|z_r| <= balanced_digit_max(lb, K)`
  via the stage-1 range check against the published `K`; the weak-binding
  (special-soundness) extractor reads only accepting transcripts and never how `c`
  was sampled. The grind's bias on the
  challenge distribution is absorbed into the standard Fiat-Shamir `(Q+1)·κ`
  knowledge error, the only contract being that the accepted support retains
  `λ + log2 Q` bits (the accepted-challenge entropy invariant already stated in
  the in-repo L2 spec).

### Precise diff surface

Crate-by-crate, smallest coherent change (no sibling `_v2` functions; the
worst-case path is generalized in place):

**`akita-challenges`**

- `src/config.rs`: add `pub fn challenge_energy_rho2(&self) -> u128` to
  `SparseChallengeConfig` (table above). Pure; mirrors the existing `l1_norm` /
  `operator_norm_cap` accessors.
- `src/tensor.rs`: add `pub fn effective_energy_rho2(&self, cfg) -> u128` to
  `ChallengeShape` (`Flat → ρ2`, `Tensor → ρ2·ρ2`), mirroring
  `effective_operator_norm_cap`.
- `src/sampler/mod.rs`: extend `sample_folding_challenges` (and the inner
  `sample_sparse_challenges`) with a `grind_nonce: u32` that is folded into
  `sparse_challenge_absorb_buf` (a new field after the config domain separator),
  so an incremented nonce yields an independent transcript-derived stream while
  staying prover/verifier-replayable.

**`akita-types`**

- `src/sis/norm_bound.rs`: add
  `fold_response_linf_threshold_sq(t_p, rho2, sigma_inf_sq, n_coeffs, ln_term) -> Result<u128, AkitaError>`
  returning `t*²` (squared domain, exact `u128`, saturating/no-panic). The only
  irrational input is `ln 4N + T_p·ln(1/p)`; pass it as a conservative integer
  `ln_term` (a small helper `ceil_ln4n_term(n_coeffs, p_num, p_den, t_p)`
  table-bounded for `N <= 2^32`, `ln 4N <= 24`). Document that the real sqrt is
  taken only at the digit-sizing boundary.
- `src/sis/decomposition_digits.rs`: `num_digits_fold` gains the threshold inputs
  (`rho2`, `n_coeffs`, `p`) and sizes `K` from
  `min(β_inf, isqrt_ceil(t*²))` instead of `β_inf`. Keep the degenerate guards.
- `src/sis/mod.rs`: re-export the new primitive.
- `src/layout/params.rs`: `LevelParams::num_digits_fold` passes the new inputs
  (`challenge_energy_rho2` via `stage1_config`, `inner_width()·D`, the op-norm
  acceptance `p`). Add a `fold_linf_threshold_sq()` accessor so the prover reads
  the identical value (invariant 4).
- `src/proof/levels.rs`: add `fold_grind_nonce: Vec<u32>` (one `u32` per committed
  fold point at the level) to `AkitaLevelProof`; update constructors and
  serialization.
- `src/proof/shapes.rs` + `src/layout/proof_size.rs`: extend `LevelProofShape` and
  the proof-size formula by the nonce bytes.

**`akita-planner`**

- `src/schedule_params.rs` + `src/generated/expand.rs`: thread `rho2` / `n_coeffs`
  / `p` into the DP's `num_digits_fold` call so a lowered `K` is searched.
- Regenerate `src/generated/*.rs` (plain + zk) via the existing
  `gen_schedule_tables` binary.

**`akita-prover`**

- `src/protocol/ring_relation.rs`: replace `validate_decompose_fold`'s abort with
  a capped grind loop around `sample_folding_challenges` →
  `build_point_decompose_fold_witness`/`decompose_fold` → accept first `z` with
  `‖z‖_inf <= t*` (read from `lp.fold_linf_threshold_sq()`). Record the accepted
  nonce into the level proof. Multipoint: accept when all points pass the same `K`.
  The nonce is absorbed before the challenge squeeze (already the absorb point in
  `sample_folding_challenges`).

**`akita-verifier`**

- `src/protocol/batched.rs` / `ring_switch.rs`: read `fold_grind_nonce` from the
  proof, pass it to `sample_folding_challenges`, reject malformed nonce/shape with
  `AkitaError`. No new norm check (the digit-range check already enforces `K`).

**`akita-config`**

- Instance-descriptor binding: add the threshold-policy identity (formula tag,
  per-family `ρ2`, attempt cap, nonce presence) to
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
- **Tensor: exact expanded-product energy** instead of the `ρ2(α)·ρ2(β)` bound.
  Tighter but requires modeling the dependent product signs; the product bound is
  safe and sufficient for termination, matching the existing `Γ²` op-norm policy.

## Documentation

- Update the "Folded-Witness ∞-Norm Rejection" section of
  [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) to point
  at this spec and mark the families generalized.
- Crate docs on `num_digits_fold` and the new threshold primitive, stating the
  per-family `ρ2` and the sign-symmetry requirement inline.
- Public security-model docs: extend the challenge-distribution / norm-bound
  description with the rigorous `‖z‖_inf` rejection threshold and the per-family
  `ρ2` table from this spec.

## Execution

Slices (each independently reviewable; W0 are pure and unblock the rest):

```text
W0 (pure, parallel)
  F1  challenge_energy_rho2 + effective_energy_rho2        [akita-challenges]
  F2  fold_response_linf_threshold_sq + ceil_ln term       [akita-types::sis]

W1
  F3  num_digits_fold sizes K from min(β_inf, t*)          [akita-types::sis]  (F2)
  F4  LevelParams threshold accessor + wiring              [akita-types]       (F2,F3)
  F6  grind nonce: sampler param + proof field + shape     [challenges,types]

W2
  F5  planner DP + regenerate shipped tables               [akita-planner]     (F3,F4)
  F7  prover grind loop + accepted nonce                   [akita-prover]      (F4,F6)

W3
  F8  verifier replay + no-panic                           [akita-verifier]    (F6,F7)
  F9  descriptor binding + pinned bytes                    [akita-config]      (F4,F6)

W4
  F10 e2e tamper / termination / ZK parity tests           (all)
```

Risks to resolve first: (a) confirm the `BoundedL1` truncation constant or scope
`d=32` out of the first PR; (b) lock the tensor `ρ2_eff` policy; (c) confirm
`N = inner_width · D` matches the `coeffs` column the calibration tables use
(spot-checked: `nv=16 onehot_fp32_d64` has `rows=256, coeffs=16384=256·64`).

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
