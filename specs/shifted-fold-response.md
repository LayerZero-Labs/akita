# Shifted Fold-Response Decomposition

## Status

Implemented on PR #254 (`quang/shifted-z-fold-response`), stacked on
`fix/a-role-public-fold-cap`.

## Motivation

Akita range-checks recursive witness coordinates as balanced base-`b` digits with
digit alphabet

```text
D_b = {-b/2, ..., b/2 - 1}.
```

This alphabet is optimal for small integer digits, but it is half-open. With
`delta` digits, the directly representable integer interval is

```text
[-N_delta, P_delta]
N_delta = (b/2)     * L_delta
P_delta = (b/2 - 1) * L_delta
L_delta = 1 + b + ... + b^(delta - 1).
```

The interval has exactly `b^delta` integers, but its center is `-L_delta / 2`.
For full-field decompositions we already compensate for this with asymmetric
centering. The folded response `z` has not used that trick because `z` is a low
norm integer object rather than a field element whose representative can be
changed freely.

That leaves a structural inefficiency. For sign-symmetric folding challenges,
the honest folded response

```text
z = sum_i c_i * s_i
```

is centered conditional on the source witness. This remains true for one-hot
sources and for small balanced source digits: once `s_i` is fixed before the
fold challenge, multiplying by sign-symmetric `c_i` makes every nonzero
contribution zero-mean. The fold accumulates tail mass, not directional bias.

Therefore the accepted interval for semantic `z` should be as centered as an
integer power-of-two digit alphabet allows.

## Protocol Change

The protocol no longer commits to a digit decomposition of the semantic folded
response `z` directly. It commits to a public shifted integer

```text
z_comm = z - eta_delta
```

coordinate-wise, where `eta_delta` is derived only from the public level
schedule:

```text
eta_delta = ceil(L_delta / 2)
```

Equivalently one may say the prover decomposes `z + shift_delta` with
`shift_delta = -eta_delta`. This spec uses `eta_delta` because it makes the
semantic reconstruction read naturally:

```text
z = z_comm + eta_delta.
```

The stage-1 range check remains exactly the same predicate. It checks that every
digit of `z_comm` lies in `D_b`. The verifier interprets the checked segment as
`z_comm`, then applies the public affine shift whenever a relation row needs the
semantic folded response.

No half-integer witness coordinates are introduced. Exact zero bias is
impossible for an even number of integer points; the public shift removes the
large `L_delta / 2` offset and leaves only a one-integer asymmetry.

With `eta_delta = ceil(L_delta / 2)`, the accepted semantic interval is

```text
[-N_delta + eta_delta, P_delta + eta_delta]
= [ -((b - 1)L_delta - 1)/2, ((b - 1)L_delta + 1)/2 ].
```

The semantic infinity bound for security pricing is therefore

```text
Z_delta = ((b - 1)L_delta + 1) / 2.
```

This replaces the old direct negative reach `N_delta = (b/2)L_delta` wherever
we price the semantic folded response accepted by the verifier.

## Relation Rows

Today the two folded rows use semantic `z`:

```text
sum_i c_i e_i = <a, G z>
sum_i c_i t_i = A z
```

After shifting, the committed witness segment is `z_comm`, so the rows become
affine rows with public constants:

```text
sum_i c_i e_i - <a, G z_comm> = eta_delta * <a, G 1>
sum_i c_i t_i - A z_comm      = eta_delta * A 1
```

Here `1` is the all-one vector in the folded-response coordinate space. It has
the same length as `z`, namely `block_len * num_digits_commit` ring elements per
point segment. The constants are public because `eta_delta`, the opening point,
the gadget powers, and the setup matrix are public.

The right-hand-side vector `y` is no longer

```text
0 | D rows | B/F rows | B_inner zeros | A zeros.
```

It becomes

```text
z_consistency_shift | D rows | B/F rows | B_inner zeros | A_shift_rows.
```

Terminal layouts that drop the D-block still keep the consistency and A shift
constants.

## Prover Semantics

The prover still computes the honest semantic folded response:

```text
z = sum_i c_i s_i.
```

The grind / rejection rule is still applied to semantic `z`, because its purpose
is to find a sign-symmetric challenge whose semantic folded response has small
centered norm. The grind acceptance condition should remain:

```text
||z||_inf <= public semantic cap
```

After the fold is accepted, the prover forms

```text
z_comm = z - eta_delta
```

and uses `z_comm` for the recursive witness digit planes. The semantic rings
`z` are still retained for checks and terminal reporting, but the committed
segment in `w` is the digit decomposition of `z_comm`.

## Verifier Semantics

The verifier never sees `z` in committed recursive levels. It sees only a
commitment to the next witness polynomial. It must nevertheless derive the same
public constants used by the prover:

1. Compute `delta_fold = lp.num_digits_fold(num_claims, field_bits)`.
2. Compute `eta_delta = fold_response_shift(log_basis, delta_fold)`.
3. Build the nonzero folded-row RHS constants:
   - consistency row: `eta_delta * <a, G 1>`;
   - A rows: `eta_delta * A 1`.
4. Include those rows in the public relation claim `V_alpha`.
5. Evaluate the structured MLE contribution against the committed shifted
   `z_comm` segment.

The stage-1 digit range check is unchanged. The statement it certifies changes:
instead of saying "the committed segment is semantic `z`", it says "the
committed segment is `z_comm`, and semantic `z` is obtained by adding the public
shift."

## Security Pricing

There are three different bounds, and the implementation must keep them
separate.

1. **Honest semantic cap.**
   Used by fold grind and terminal Golomb parameterization:

   ```text
   cap = min(beta_inf, t*)   under TailBoundWithGrind
   cap = beta_inf            under WorstCaseBetaOnly
   ```

2. **Committed shifted-digit interval.**
   Used by stage-1 range-check soundness. The predicate still checks
   `z_comm in [-N_delta, P_delta]`.

3. **Verifier-accepted semantic interval.**
   Used by A-role committed-fold collision pricing:

   ```text
   |z| <= Z_delta = ((b - 1)L_delta + 1) / 2.
   ```

The planner must size `delta_fold` from the honest semantic cap, then price the
committed-fold A-role security from `Z_delta`, not from `N_delta` and not from
the honest cap alone.

This is the full-cutover replacement for the current
`fold_witness_verifier_linf_bound(log_basis, delta_fold)` behavior. That helper
currently returns the negative reach `N_delta`. After this protocol change, the
semantic verifier bound should return `Z_delta`, while any helper that needs the
raw committed digit reach should be named accordingly.

## Terminal Witnesses

The terminal cleartext path has two possible encodings:

1. Encode `z_comm` exactly as the committed witness segment.
2. Encode semantic centered `z`, then have the verifier subtract `eta_delta`
   before expanding the terminal witness digits.

The second choice preserves the existing Golomb model: terminal `z` is centered
and low-norm, while `z_comm` is intentionally shifted. Therefore the terminal
segment-typed witness should continue to carry semantic `z` values on the wire,
validate them against the public semantic cap, and expand them to committed
digit planes by subtracting `eta_delta`.

Packed-digits terminal witness formats, if retained, must be interpreted as
already-expanded committed witness digits. They do not get an extra semantic-z
compression benefit.

## Implementation Surface

### `akita-types`

- Add shared helpers in `sis/decomposition_digits.rs`:
  - `balanced_digit_series(log_basis, num_digits) -> u128`;
  - `fold_response_shift(log_basis, num_digits_fold) -> u128`;
  - `fold_response_committed_linf_bound(log_basis, num_digits_fold) -> u128`;
  - `fold_response_semantic_linf_bound(log_basis, num_digits_fold) -> u128`.
- Replace A-security uses of `fold_witness_verifier_linf_bound` with the
  semantic shifted bound.
- Either rename the old helper or leave it as a compatibility-free full cutover
  to the semantic meaning. If any code needs the raw digit negative reach, it
  must call the explicitly named committed-bound helper.
- Extend relation RHS construction. `generate_y` currently hardcodes zero
  consistency and A rows; it needs a path that accepts:
  - one consistency RHS ring;
  - a slice of A-row RHS rings.
- Extend `relation_claim_from_rows(_extension)` or replace it with a full-`y`
  evaluator so consistency and A RHS constants contribute to `V_alpha`.
- Update terminal segment expansion so semantic terminal `z` is shifted before
  balanced digit expansion.

### `akita-prover`

- Extend `DecomposeFoldWitness` to retain both:
  - semantic centered coefficients of `z`;
  - committed balanced digit planes of `z_comm`.
- Compute the digit expansion of `z_comm = z - eta_delta` once, immediately
  after building the semantic fold witness and after grind accepts the semantic
  norm. Do not require the shifted representative to fit in `i32`: for full
  field schedules `eta_delta` can be large even though its digit expansion is
  still just `delta_fold` small balanced digits.
- Use semantic coefficients for:
  - grind acceptance;
  - semantic `centered_inf_norm`;
  - terminal Golomb payload input.
- Use committed shifted digit planes for:
  - `ring_switch_build_w`;
  - `compute_relation_quotient`;
- When aggregating per-polynomial decompose-fold witnesses, sum semantic `z`
  first, then recompute the committed shifted digit planes once for the
  aggregate. Do not sum already-shifted witnesses, which would subtract
  `eta_delta` multiple times.
- Compute the public RHS correction before creating `RingRelationInstance`:
  - consistency correction from the opening point, commit gadget powers, and
    `eta_delta`;
  - A correction from the setup matrix applied to the all-one z vector.
- Feed the corrected `y` into quotient construction and transcript-bound
  instance construction.

### `akita-verifier`

- Recompute the same shift from `LevelParams` and `num_claims`.
- Build the same consistency and A-row RHS constants from public data.
- Include the constants in the relation claim and ring-switch replay.
- Keep stage-1 range verification unchanged; it checks committed shifted digits.
- Ensure malformed proofs cannot choose their own shift. The shift is derived
  only from the schedule already bound into Fiat-Shamir.

### Planner / Generated Tables

- `num_digits_fold` still derives the digit count from the honest semantic cap.
- A-role rank selection must price the shifted semantic verifier bound
  `Z_delta`.
- Proof-size formulas are unchanged except through any resulting smaller
  `delta_fold` / ranks.
- Regenerate schedule tables after implementation.

### Tests

Add focused tests for:

- Numeric helper examples:
  - `b=4, delta=1`: direct `[-2,1]`, shift `1`, semantic `[-1,2]` if choosing
    positive slack, semantic bound `2`.
  - `b=4, delta=2`: direct `[-10,5]`, shift `3`, semantic `[-7,8]`.
  - `b=8, delta=2`: direct `[-36,27]`, shift `5`, semantic `[-31,32]`.
- `num_digits_fold` coverage: a semantic cap must be accepted iff it lies in the
  shifted semantic interval.
- A-role pricing uses `fold_response_semantic_linf_bound`, not the old negative
  reach.
- Prover aggregation recomputes one shift for the aggregate.
- Prover/verifier relation claims agree on nonzero consistency and A-row RHS
  constants.
- End-to-end fold proof still verifies for dense and one-hot roots.
- Terminal segment-typed witness encodes semantic `z` but expands to committed
  shifted digit planes.

## Setup artifacts and performance

The affine fold rows need public constants `eta * <a, G 1>` and `eta * A 1`.
Both depend only on the public setup matrix prefix, the fold level geometry
(`n_a`, `inner_width`, ring dimension), and the schedule-derived `eta`. They do
not depend on the proof witness.

### `FoldAOnesTable`

`akita-types::proof::fold_response_rhs::FoldAOnesTable` stores **unscaled**
`A · 1` rows keyed by `(a_row_len, inner_width)` per ring-dimension bucket.
The public shift `eta` is applied at lookup time.

Warming happens once at prover setup:

1. `CommitmentConfig::warm_fold_a_ones_at_setup` scans every
   `(num_vars, num_polynomials)` with `num_polynomials` in
   `1..=max_num_batched_polys` (level-0 fold geometry depends on batch size;
   unlike setup-matrix envelope sizing, intermediate counts must be warmed).
2. For catalog presets, fold geometries come from
   `akita_planner::fold_level_params_from_entry` (no full `Schedule`
   materialization). Catalog misses fall back to `runtime_schedule`.
3. Infeasible envelope shapes are skipped (same semantics as setup-matrix
   sizing).
4. Each new geometry row uses `CyclotomicRing::mul_accumulate_all_ones_into`
   (O(D) prefix formula per setup row, row-parallel).

The table lives on `AkitaProverSetup` and `AkitaVerifierSetup`. It is **not**
part of the wire-format verifier setup blob. Jolt recursion guest/host re-warm
after decoding the expanded matrix.

`PartialEq` on setup compares only `expanded` and `prefix_slots`; the table is
a derived cache.

### Disk cache (`disk-persistence`)

With `akita-setup/disk-persistence`, `.setup` cache files append the serialized
`FoldAOnesTable` after `prefix_slots`. On load:

- trailing section present: deserialize and bind to `public_matrix_seed`;
- legacy caches (EOF after prefix slots): call `warm_fold_a_ones_at_setup`.

For large presets the warm step is ~0.2–0.3s on a fresh build; a cache hit
still spends most of setup load time deserializing the expanded matrix (hundreds
of MB for production one-hot envelopes). The fold table removes repeat warm work
on subsequent process starts, not the matrix I/O.

## PR Shape

Recommended implementation slices:

1. Shared numeric helpers and tests in `akita-types`.
2. Relation RHS shape change shared by prover and verifier.
3. Prover `DecomposeFoldWitness` split plus shifted witness packing.
4. Verifier replay constants.
5. Terminal segment-typed adjustment.
6. Planner/security pricing update and generated table regeneration.

The PR should be stacked on `fix/a-role-public-fold-cap` because it depends on
the corrected planner distinction between honest caps, verifier-certified digit
envelopes, and A-role security pricing.
