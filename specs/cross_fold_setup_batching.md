# Spec: Cross-Fold Batched Setup Contribution

| Field      | Value      |
|------------|------------|
| Created    | 2026-05-24 |
| Status     | proposed   |

## Summary

Every non-direct Akita verifier level runs a stage-2 final oracle check.
The expensive part of that oracle is the `setup_contribution`: the
alpha-evaluated shared setup matrix rows for the `D · w_hat`, `B · t_hat`,
and `A · z_hat` row blocks.

Today the verifier evaluates this contribution independently per fold:

```text
setup_i(r_i) = Σ_{row,c} eval_ring_at_alpha_i(M[row,c]) · beta_i(row,c)
```

where `beta_i(row,c)` is the fold-local row weight times the shifted-eq
column pattern for that row/column. This is already optimized inside one
fold by fusing the W/T/Z setup blocks and sharing one SIS-row scan. It does
not exploit the fact that every recursive fold reads the same backing
setup matrix and later folds touch only prefixes or smaller subranges of
the root-level setup support.

This spec proposes a verifier-only cross-fold batch:

1. During each stage-2 final check, evaluate every row-block contribution
   except `setup_contribution`.
2. Derive the setup value implied by the sumcheck transcript, but do not
   scan the setup matrix yet.
3. After all fold transcripts are bound, sample random batching scalars
   `lambda_i`.
4. Check one random linear combination of all deferred setup claims with a
   single alpha-aware scan over the shared setup matrix support.

The optimization does not change prover messages, proof encoding, setup
format, or commitment semantics. It changes only verifier scheduling and
adds verifier-local Fiat-Shamir squeezes after all checked proof messages
have been absorbed.

## Current Per-Fold Check

For fold `i`, stage 2 verifies a final sumcheck claim at challenge point
`(y_i, x_i)`:

```text
final_i = virtual_i(x_i, y_i)
        + w_eval_i(x_i, y_i) · alpha_val_i(y_i) · row_eval_i(x_i)
```

`row_eval_i` decomposes as:

```text
row_eval_i =
    w_structured_i
  + t_structured_i
  + z_structured_i
  + r_tail_i
  + setup_i
  + zk_blinding_i
```

The expensive term is:

```text
setup_i =
  Σ_{row,c} eval_ring_at_alpha_i(M[row,c])
           · beta_i(row,c)
```

with:

```text
eval_ring_at_alpha_i(M[row,c])
  = Σ_{k=0}^{D-1} M[row,c,k] · alpha_i^k
```

`beta_i(row,c)` includes:

- the row equality weight from `eq_tau1`;
- the W/T/Z-specific column shifted-eq value;
- the T group routing mask;
- the Z fold gadget and sign;
- zero outside the fold's active row/column support.

The current implementation is `compute_setup_contribution` in
`crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`.

## Deferred Claim Extraction

For each fold, define:

```text
scale_i = w_eval_i · alpha_val_i
non_setup_i =
    w_structured_i
  + t_structured_i
  + z_structured_i
  + r_tail_i
  + zk_blinding_i
```

The stage-2 final claim gives the verifier:

```text
weighted_setup_claim_i =
    final_i
  - virtual_i
  - scale_i · non_setup_i
```

For a valid proof:

```text
weighted_setup_claim_i = scale_i · setup_i
```

The verifier should store the weighted form rather than dividing by
`scale_i`. This preserves the original behavior when `scale_i = 0`: the
relation simply does not constrain `setup_i` at that challenge point.

The per-fold stage-2 verifier therefore changes from:

```text
check final_i == virtual_i + scale_i · (non_setup_i + setup_i)
```

to:

```text
store {
    weighted_setup_claim_i,
    scale_i,
    prepared_row_eval_i,
    x_i,
    alpha_i,
    layout offsets,
}
```

and immediately accepts only the non-setup part of the final oracle. The
actual setup equality is enforced by the final batch.

## Batch Soundness

After every fold's stage-2 transcript has been absorbed and every
`weighted_setup_claim_i` is fixed, sample independent verifier-only
batching scalars:

```text
lambda_0, ..., lambda_{n-1}
```

Then check:

```text
Σ_i lambda_i · weighted_setup_claim_i
  =
Σ_i lambda_i · scale_i · setup_i
```

Equivalently:

```text
rhs =
  Σ_i lambda_i · weighted_setup_claim_i

lhs =
  Σ_{row,c,k} M[row,c,k]
      · (Σ_i lambda_i · scale_i · beta_i(row,c) · alpha_i^k)
```

If any deferred setup claim is wrong, the difference is a non-zero linear
form in the `lambda_i` except with probability at most `1 / |F|` over the
batching challenge, assuming the challenges are sampled after the claims
are fixed.

### Transcript Placement

The `lambda_i` challenges must be sampled after:

- the root stage-2 rounds have been absorbed;
- every recursive stage-2 round has been absorbed;
- the terminal stage-2 rounds have been absorbed;
- all cleartext terminal witness bytes, if any, have been absorbed.

No prover message follows these challenges. Production transcript labels
do not enter sponge bytes, so this is a verifier-local scheduling change.
Logging transcript tests must either include the new verifier-only label or
avoid requiring prover/verifier event equality after verification has
already consumed the whole proof.

Use a dedicated label:

```text
CHALLENGE_SETUP_BATCH = b"ak/c/sb"
```

## Alpha-Aware Fused Matrix Scan

The backing setup matrix is shared across folds. Later folds use prefixes
or smaller active ranges of the root-level support. Define each fold's
weighted column-row pattern:

```text
theta_i(row,c) = lambda_i · scale_i · beta_i(row,c)
```

The fused scan walks the union support:

```text
r_max = max_i r_max_i
c_max = max_i n_cols_total_i
```

For each `(row,c)`:

1. Determine the active folds `i` whose setup contribution includes this
   row and column.
2. If no fold is active, skip.
3. If exactly one fold is active, use the current fast per-fold kernel:

   ```text
   eval_ring_at_alpha_i(M[row,c]) · theta_i(row,c)
   ```

4. If multiple folds are active, combine the alpha powers first:

   ```text
   combined_k(row,c) = Σ_i theta_i(row,c) · alpha_i^k
   cell = Σ_k M[row,c,k] · combined_k(row,c)
   ```

5. Accumulate `cell`.

The single-active fast path matters. Without it, a root-only cell would pay
extra multiplications to construct `combined_k`, which can erase the
benefit. For current one-hot schedules, most root-level cells are
single-active; multi-active cells are concentrated in the prefixes touched
by recursive folds.

## Pattern Construction

Each deferred fold needs the same column patterns currently constructed by
`compute_setup_contribution`:

- `w_eq_slice[c]`;
- `t_eq_slice_per_group[g][c]`;
- `z_eq_slice[c]`;
- per-row D/B/A weights.

The implementation should split the current function into:

```text
PreparedSetupPattern {
    alpha_pows,
    w_eq_slice,
    t_eq_slice_per_group,
    z_eq_slice,
    d_weights,
    b_weights_by_row,
    a_weights,
    n_cols_w,
    n_cols_t,
    n_cols_z,
    r_max,
}
```

Then provide two evaluators:

```text
compute_setup_contribution(pattern, setup)
compute_batched_setup_contribution(weighted_patterns, setup)
```

`compute_setup_contribution` remains useful for tests, direct profiles, and
as a fallback when only one setup query exists.

## Expected Cost

Let:

```text
O_i = per-fold setup matrix cell visits
U   = union setup matrix cell visits
M   = multi-active union cells
D   = ring degree
```

Current dominant coefficient work is roughly:

```text
D · Σ_i O_i
```

The fused scan with a single-active fast path is roughly:

```text
D · U                         // coefficient dot products
+ D · Σ_{multi cells} active_count(row,c)
```

The second term comes from combining `theta_i · alpha_i^k` for multi-active
cells. This term exists because each fold has its own ring-switch
challenge `alpha_i`; if all folds shared one alpha, the combined
shifted-eq table would be strictly cheaper.

For the current `AKITA_MODE=onehot AKITA_NUM_VARS=32 D=32` trace:

```text
level 0 setup cells: 851,968
level 1 setup cells:  65,536
level 2 setup cells:  16,384
level 3 setup cells:   4,096
level 4 setup cells:   2,048
level 5 setup cells:     846
level 6 setup cells:     832
total setup cells:   941,710
root union cells:    851,968
```

The absolute ceiling from eliminating every later setup span is the measured
drop from `15.2 ms` setup time to the root's `9.19 ms`, or about `27%` of
the whole native verifier in that trace. The alpha-aware fused scan cannot
reach that ceiling automatically; it must beat the extra multi-active
alpha-combination work.

## Implementation Plan

1. Add `CHALLENGE_SETUP_BATCH`.
2. Add a sumcheck verifier helper that returns the final folded claim and
   challenge point without immediately calling `expected_output_claim`.
3. Split `RingSwitchDeferredRowEval::eval_at_point` into parts so the
   verifier can evaluate `non_setup_i` without scanning the setup matrix.
4. Add `DeferredSetupCheck` and `DeferredSetupBatch` in
   `akita-verifier`.
5. Thread a mutable batch context through root and recursive fold
   verification.
6. After all folds verify, sample `lambda_i`, run
   `compute_batched_setup_contribution`, and compare to the batched claim.
7. Keep a fallback path for empty batches and single-query batches.
8. Add tests:
   - deferred batch matches per-fold setup for one query;
   - deferred batch matches sum of per-fold setup values for multiple
     real schedule-shaped queries;
   - tampering one deferred claim rejects with high probability using a
     fixed non-zero lambda in unit tests;
   - existing batched verifier tests still pass.
9. Benchmark:
   - native `AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release --example profile`;
   - recursion guest trace for the selected arity;
   - compare `akita_verify` cycles and, if instrumentation exists, the
     setup-specific markers.

## Correctness Risks

- The stage-2 sumcheck final claim must be captured before the final oracle
  equality is checked. Recomputing it outside the transcript loop risks
  diverging from `CompressedUniPoly::eval_from_hint`.
- `scale_i = 0` must not require division or introduce an artificial setup
  constraint.
- The batch challenge must be sampled after every deferred claim is fixed.
- ZK blinding terms are not part of the setup matrix prefix and should stay
  in `non_setup_i` unless a future spec explicitly folds them in.
- Different ring dimensions cannot be batched in one const-generic matrix
  scan. Current schedules use one `D`; if mixed-`D` schedules are enabled,
  maintain one batch per `D`.
- The alpha-aware fused scan may be slower for small batches or small
  recursive supports. The implementation should retain the old per-fold
  path behind a switch until profiling proves the fused path wins.
