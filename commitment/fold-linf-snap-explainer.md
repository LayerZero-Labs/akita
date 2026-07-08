# Fold L-inf Cap, Grinding, and Snap-Down

This note explains the fold-L-inf sizing logic around
`fold_witness_linf_digit_plan` and `snap_num_digits_fold_down`.

The goal is to explain the idea from high to low:

1. why the protocol wants a smaller folded-witness digit count,
2. what `beta_inf`, `t_star`, `delta_base`, `delta_fold`, and `grind_cap` mean,
3. why snap-down exists,
4. what the prover checks,
5. what the verifier checks,
6. how the code implements the flow.

The main files are:

- `crates/akita-types/src/sis/norm_bound.rs`
- `crates/akita-types/src/sis/decomposition_digits.rs`
- `crates/akita-types/src/sis/fold_linf_cap.rs`
- `crates/akita-prover/src/protocol/fold_grind.rs`
- `crates/akita-verifier/src/stages/stage1.rs`
- `crates/akita-verifier/src/protocol/core/fold.rs`

---

## 1. The High-Level Motivation

At each fold level, Akita combines many witness pieces into one folded witness:

```text
z = sum_i c_i * s_i
```

where:

- `s_i` are current-level witness pieces,
- `c_i` are Fiat-Shamir folding challenges,
- `z` is passed into the next recursive level.

The next level does not carry arbitrary integer coefficients for `z`. It carries
a balanced base-`b` digit decomposition of each coefficient, where:

```text
b = 2^log_basis
```

The number of digits per folded coefficient is called `delta_fold`.

This matters because `delta_fold` directly affects the size of the next witness:

```text
larger delta_fold
    -> more digit planes for z
    -> wider next-level witness
    -> larger proof and more prover work
```

So the protocol wants `delta_fold` to be as small as safely possible.

The old fully pessimistic way is to size `delta_fold` from a worst-case bound
`beta_inf`. That is always safe, but often too large. The fold-L-inf machinery
uses a tighter statistical target `t_star`, bounded Fiat-Shamir grinding, and an
optional snap-down step to reduce `delta_fold` when there is enough slack.

---

## 2. The Five Important Quantities

### 2.1 `beta_inf`: worst-case absolute cap

`beta_inf` is the deterministic worst-case L-inf bound for the folded witness.

It answers:

```text
In the worst case, how large can any coefficient of z be in absolute value?
```

In code, it is computed by `fold_witness_beta`:

```text
beta_inf = num_claims * 2^r_vars
         * min(||c||_inf * ||s||_1, ||c||_1 * ||s||_inf)
```

`beta_inf` is an absolute-value cap. It is not the positive range, and it is not
the negative range.

It means:

```text
for every coefficient z_j:
    |z_j| <= beta_inf

equivalently:
    -beta_inf <= z_j <= beta_inf
```

If `beta_inf = 1000`, the mental model is:

```text
z_j is somewhere in [-1000, +1000]
```

This is safe but pessimistic because it assumes the worst possible sign
alignment of all challenge terms.

### 2.2 `t_star`: statistical absolute cap

For certified challenge families, the code has a sub-Gaussian tail argument that
gives a smaller cap called `t_star`.

It answers:

```text
With random-looking Fiat-Shamir challenges, what absolute cap should honest z
usually fit under, with bounded grinding?
```

Like `beta_inf`, `t_star` is also an absolute-value cap:

```text
for every coefficient z_j:
    |z_j| <= t_star

equivalently:
    -t_star <= z_j <= t_star
```

So `t_star` is not "the positive side" or "the negative side." It is the
statistical L-inf target for the whole folded witness.

For tail-bound levels:

```text
pre_snap_cap = min(beta_inf, t_star)
```

For unsupported challenge families:

```text
pre_snap_cap = beta_inf
```

and there is no `t_star`-based grinding policy.

### 2.3 `delta_base`: first digit count

`delta_base` is the first digit count computed from `pre_snap_cap`.

The code is:

```rust
let log_cap = (128 - pre_snap_cap.leading_zeros()).saturating_add(1);
let delta_base = num_digits_for_bound(log_cap, field_bits, log_basis);
```

This starts from the absolute cap:

```text
|z_j| <= pre_snap_cap
```

Then it converts that magnitude cap into a signed bit-width. If
`pre_snap_cap = C`, the intended mental range is:

```text
-C <= z_j <= C
```

The `+ 1` accounts for the sign bit. For example, if `C = 600`, then 600 needs
10 magnitude bits because:

```text
2^9 = 512  < 600
2^10 = 1024 >= 600
```

So the signed bit range is roughly:

```text
[-1024, +1023]
```

`num_digits_for_bound` then asks:

```text
How many balanced base-b digits cover this signed range?
```

Balanced digits are asymmetric:

```text
one digit is in [-b/2, b/2 - 1]
```

So the positive side is the limiting side. Internally,
`compute_num_digits` checks whether the candidate digit count reaches the
required positive endpoint:

```rust
let required_positive = (1u128 << (log_bound - 1)).saturating_sub(1);
if balanced_digit_max(log_basis, num_digits) < required_positive {
    num_digits += 1;
}
```

So the precise answer is:

```text
delta_base is derived from a symmetric signed interval for the absolute cap,
but the minimality check is governed by the positive side because balanced
digits are shorter on the positive side.
```

### 2.4 `delta_fold`: final digit count

`delta_fold` is the final number of balanced digits allocated for each
coefficient of `z`.

If snap-down does nothing:

```text
delta_fold = delta_base
```

If snap-down succeeds:

```text
delta_fold < delta_base
```

This is the value that changes the next witness size.

### 2.5 `grind_cap`: prover acceptance cap

`grind_cap` is the final cap the prover uses while searching for an acceptable
nonce.

It is passed to prover code as `witness_linf_cap`.

The prover accepts a nonce only if the resulting folded witness satisfies the
cap and fits the digit representation.

If snap-down does nothing:

```text
grind_cap = pre_snap_cap
```

If snap-down succeeds:

```text
grind_cap = min(pre_snap_cap, positive_reach(delta_fold))
```

This alignment matters. If `delta_fold` is made smaller, the prover cap must
also be lowered so the prover does not accept a `z` that the smaller digit
representation cannot encode.

---

## 3. Why Balanced Digits Create a Positive-Side Bottleneck

Balanced base-`b` digits use:

```text
[-b/2, b/2 - 1]
```

For `delta` digits, the representable interval is:

```text
negative_abs_reach(delta) = (b/2)     * (1 + b + ... + b^(delta - 1))
positive_reach(delta)     = (b/2 - 1) * (1 + b + ... + b^(delta - 1))
```

The negative side reaches farther because each digit can be `-b/2` but only
`b/2 - 1` on the positive side.

That gives two different notions:

```text
verifier absolute digit envelope:
    negative_abs_reach(delta)

honest positive representability bottleneck:
    positive_reach(delta)
```

The verifier prices the full accepted digit language, so it uses the larger
absolute envelope. The prover, however, must be able to encode positive and
negative realized coefficients. The positive side is the tighter bottleneck.

---

## 4. Correct Intuition for Snap-Down

A common but incorrect intuition is:

```text
sample z
check whether this concrete z fits delta_base - 1 digits
if yes, use fewer digits
```

That is not what the code does.

Snap-down happens before the prover samples the final accepted `z`. It is part
of choosing the proof shape and prover acceptance cap.

The real flow is:

```text
compute beta_inf
compute t_star, if the policy supports it
pre_snap_cap = min(beta_inf, t_star), or beta_inf for worst-case-only policy
compute delta_base from pre_snap_cap
maybe snap down to a smaller delta_fold
choose grind_cap to match the chosen delta_fold
then the prover runs the nonce loop until actual z fits the final grind_cap
```

So snap-down is not a last-minute optimization after seeing `z`.

It is a planning step:

```text
Can we choose fewer digits up front and make the prover grind for a z that fits
that smaller digit representation?
```

---

## 5. The Snap-Down Rule

The implementation is:

```rust
pub fn snap_num_digits_fold_down(
    log_basis: u32,
    delta_base: usize,
    pre_snap_cap: u128,
    t_star: u128,
    retain_num: u128,
    retain_den: u128,
) -> (usize, u128) {
    if delta_base <= 1 || t_star == 0 || retain_den == 0 {
        return (delta_base, pre_snap_cap);
    }
    let floor = snap_min_tstar_retain_floor(t_star, retain_num, retain_den);
    let mut delta = delta_base;
    let mut grind_cap = pre_snap_cap;
    while delta > 1 {
        let (_, positive_lower) = fold_witness_representable_linf_bounds(log_basis, delta - 1);
        if positive_lower < floor {
            break;
        }
        delta -= 1;
        let (_, positive_at) = fold_witness_representable_linf_bounds(log_basis, delta);
        grind_cap = pre_snap_cap.min(positive_at);
    }
    (delta, grind_cap)
}
```

Read it as:

```text
Start from delta_base.
Try one fewer digit.
If one fewer digit still has positive reach >= retain floor, accept the smaller digit count.
Lower grind_cap to the positive reach of the smaller digit count.
Repeat while possible.
```

The retain floor is:

```text
floor = floor(t_star * retain_num / retain_den)
```

In production:

```text
retain_num = 1
retain_den = 2
floor = floor(t_star / 2)
```

So snap-down asks:

```text
Would using one fewer digit still keep the positive representable range at
least half of t_star?
```

If yes, snap down.

If no, keep the current digit count.

---

## 6. Why the Retained Fraction is `1/2`

The `1/2` is a protocol parameter. It is not caused by the positive/negative
asymmetry.

The asymmetry answers:

```text
Which digit reach should we compare against?
```

Answer:

```text
positive_reach, because it is the shorter side.
```

The `1/2` answers:

```text
How much of t_star must remain after snap-down?
```

Answer:

```text
at least half, by current protocol choice.
```

Why have such a parameter?

Because smaller caps are not a soundness problem, but they can be a completeness
and performance problem.

If the cap is much smaller than `t_star`, the prover may need many more nonce
attempts to find an acceptable `z`. The tail-bound calculation and grind budget
were chosen around the `t_star` acceptance target. Snap-down is allowed to make
the target tighter, but only by a bounded amount.

So `1/2` means:

```text
We are willing to tighten the prover target to save a digit,
but not by more than a 50% reduction from t_star.
```

Why not make `t_star` smaller from day one?

Because making `t_star` smaller globally would make every grind target harder,
including cases where the smaller cap does not save a digit. Snap-down is more
selective:

```text
Only tighten the cap when it actually buys a smaller delta_fold,
and only tighten it by a bounded amount.
```

So:

```text
positive/negative asymmetry:
    determines that positive_reach is the bottleneck

1/2 retained fraction:
    determines how aggressive snap-down may be
```

---

## 7. Toy Example: Base 4

Use:

```text
log_basis = 2
b = 4
one digit is in [-2, 1]
```

For 5 digits:

```text
positive_reach(5)     = 1 * (1 + 4 + 16 + 64 + 256) = 341
negative_abs_reach(5) = 2 * (1 + 4 + 16 + 64 + 256) = 682
```

So 5 digits can represent:

```text
-682 <= z_j <= 341
```

The positive side is the limiting side.

### Case A: no snap

Inputs:

```text
delta_base = 6
pre_snap_cap = 739
t_star = 739
retain floor = floor(739 / 2) = 369
```

Try to drop from 6 digits to 5 digits:

```text
positive_reach(5) = 341
341 < 369
```

Five digits would force the prover cap down to at most 341. But the snap policy
requires keeping at least 369, which is half of `t_star`.

So the code refuses to snap:

```text
delta_fold = 6
grind_cap = 739
```

Intuition:

```text
t_star says:       "try to find z below 739"
snap policy says:  "you may tighten this, but not below 369"
5 digits say:      "I can only encode positive values up to 341"
decision:          "341 is below 369, so 5 digits are too tight"
```

### Case B: snap succeeds

Inputs:

```text
delta_base = 6
pre_snap_cap = 600
t_star = 600
retain floor = floor(600 / 2) = 300
```

Try to drop from 6 digits to 5 digits:

```text
positive_reach(5) = 341
341 >= 300
```

Five digits still retain at least half of `t_star`.

So the code accepts the smaller digit count:

```text
delta_fold = 5
grind_cap = min(600, positive_reach(5))
          = min(600, 341)
          = 341
```

Intuition:

```text
t_star says:       "try to find z below 600"
snap policy says:  "you may tighten this, but not below 300"
5 digits say:      "I can encode positive values up to 341"
decision:          "341 is above 300, so 5 digits are acceptable"
```

The cap is lowered to 341 because 5 digits cannot encode `+600`.

---

## 8. What the Prover Checks

The prover computes the final digit plan before the nonce loop:

```text
FoldWitnessLinfDigitPlan {
    delta_fold,
    grind_cap,
    pre_snap_cap,
    t_star,
}
```

Then it computes exact digit bounds:

```text
(negative_abs_bound, positive_bound) =
    fold_witness_representable_linf_bounds(log_basis, delta_fold)
```

Then it loops over nonces:

```text
for nonce in probe_nonces:
    preview folding challenges from transcript + nonce
    compute z
    if accepts_fold_witness(z):
        commit the same nonce/challenges to the live transcript
        return proof data
    else:
        try next nonce
```

`accepts_fold_witness` checks:

```text
1. every centered coefficient fits the chosen digit range:
       -negative_abs_bound <= z_j <= positive_bound

2. when grind cap checking is enabled:
       |z_j| <= grind_cap

3. global centered_inf_norm <= grind_cap

4. terminal Golomb wire checks, when terminal tail encoding is active
```

Using Case B:

```text
delta_fold = 5
grind_cap = 341
digit range = [-682, +341]
```

The prover behavior is:

```text
z_j = +300  accepted by this coefficient check
z_j = +500  rejected: above positive digit reach and above grind_cap
z_j = -300  accepted by this coefficient check
z_j = -500  rejected: digit-representable, but |z_j| > grind_cap
z_j = -700  rejected: outside digit range and above grind_cap
```

Notice the asymmetry:

```text
-500 is digit-representable with 5 digits,
but the prover still rejects it because grind_cap is an absolute cap of 341.
```

So the prover checks both digit representability and the absolute grind cap.

---

## 9. What the Verifier Checks

The verifier does not run the nonce search.

It receives the proof, including the chosen `fold_grind_nonce`.

The verifier checks:

```text
1. the nonce is legal for the policy
   - WorstCaseBetaOnly: nonce must be 0
   - TailBoundWithGrind: nonce < max_nonce_exclusive

2. the same challenges are derived from the transcript and nonce

3. the ring relation / sumcheck proof verifies

4. the digit representation is valid for the proof's delta_fold
```

The verifier does not treat the nonce as proof that:

```text
|z_j| <= grind_cap
```

The prover used `grind_cap` to find an encodable honest witness. The verifier
checks the proof relation and the digit language that the proof actually
commits to.

For verifier-side binding/MSIS pricing, the relevant envelope is:

```text
fold_witness_verifier_linf_bound(log_basis, delta_fold)
```

That is the larger absolute digit reach:

```text
negative_abs_reach(delta_fold)
```

Using Case B:

```text
delta_fold = 5
digit range = [-682, +341]
verifier L-inf envelope = 682
```

The verifier prices the full accepted digit language, not just the prover's
honest cap 341.

This is the division of responsibility:

```text
prover:
    search for a z that fits grind_cap and digit bounds

verifier:
    replay the chosen nonce, verify the relation, and enforce digit membership

security accounting:
    price every accepted digit string using the verifier digit envelope
```

---

## 10. End-to-End Code Flow

### 10.1 Digit planning

`fold_witness_linf_digit_plan` does:

```text
read protocol binding
compute beta_inf
maybe compute t_star
pre_snap_cap = beta_inf or min(beta_inf, t_star)
delta_base = num_digits_for_bound(pre_snap_cap)
maybe snap down
return FoldWitnessLinfDigitPlan
```

Code:

- `fold_witness_pre_snap_linf_cap`
- `fold_witness_linf_digit_plan`
- `snap_num_digits_fold_down`

### 10.2 Prover grinding

`sample_fold_decompose_witness` does:

```text
contract = fold_witness_grind_contract(...)
witness_linf_cap = plan.grind_cap
digit bounds = representable bounds for plan.delta_fold
probe_nonces = nonce order
dispatch to sample_fold_decompose_witness_at_dim
```

Then `sample_fold_decompose_witness_at_dim` tries nonces:

```text
preview challenges
compute folded witness
accept or reject
commit the winning nonce
```

Code:

- `fold_witness_grind_contract`
- `witness_linf_cap_for_grind`
- `fold_probe_witness_kernel`
- `accepts_fold_witness`
- `sample_fold_decompose_witness_at_dim`

### 10.3 Verifier replay

`verify_fold` does:

```text
validate_fold_grind_nonce(...)
derive grouped stage-1 challenges using the nonce
verify relation and stage proofs
```

Code:

- `validate_fold_grind_nonce`
- `FoldWitnessGrindContract::validate_nonce`
- `derive_grouped_stage1_challenges`
- `verify_fold`

---

## 11. Common Misunderstandings

### "Are `beta_inf` and `t_star` positive ranges or negative ranges?"

No. They are absolute L-inf caps:

```text
|z_j| <= beta_inf
|z_j| <= t_star
```

The positive/negative distinction appears when we ask what balanced digits can
represent.

### "Is `delta_base` based on the positive side or negative side?"

It starts from an absolute cap, converts it to a symmetric signed bit range, and
then checks the positive side because positive reach is the limiting side for
balanced digits.

### "Does snap-down look at the actual sampled `z`?"

No. Snap-down happens before the accepted `z` is sampled. It chooses
`delta_fold` and `grind_cap` up front. The later nonce loop searches for a `z`
that fits those choices.

### "Is `1/2` caused by the positive/negative asymmetry?"

No. The asymmetry determines that the code compares against `positive_reach`.
The `1/2` is the retained-fraction parameter that limits how aggressive
snap-down may be.

### "Why not make `t_star` smaller directly?"

Because a smaller `t_star` would make grinding harder everywhere, including
places where the smaller cap does not save a digit. Snap-down tightens only when
it buys a smaller `delta_fold`.

### "Does the verifier check `grind_cap`?"

Not as a separate nonce-search condition. The verifier validates the nonce,
replays challenges, verifies the relation, and checks digit membership. Security
pricing uses the full verifier digit envelope for `delta_fold`.

---

## 12. One-Sentence Mental Model

`beta_inf` is the pessimistic absolute cap; `t_star` is the tighter statistical
absolute cap; `delta_base` is the first digit count for that cap; snap-down may
choose fewer digits if their positive reach still keeps at least half of
`t_star`; `grind_cap` is then lowered to match the chosen digits, and the prover
grinds Fiat-Shamir nonces until the actual `z` fits.
