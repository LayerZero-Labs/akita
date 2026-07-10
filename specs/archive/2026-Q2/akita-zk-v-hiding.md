# Spec: Akita ZK Prover V Hiding

| Field     | Value                  |
| --------- | ---------------------- |
| Author(s) | Amirhossein Khajehpour |
| Created   | 2026-05-08             |
| Status    | implemented on branch  |
| PR        |                        |

## Summary

This spec extends Akita's compile-time `zk` hiding layer from outer
B-commitments to the stage-1 prover message `v = D * w_hat`. In transparent
builds, `v` remains the deterministic D-matrix image of the decomposed folded
witness. In `--features zk` builds, the prover appends fresh digit-source
blinding columns to the D input and sends

```text
v = D_msg * w_hat + D_blind * r_D
```

so repeated folded proofs for the same witness re-randomize the public
`ABSORB_PROVER_V` value while preserving the existing ring-switch and sumcheck
relations.

## Intent

### Goal

Hide every wire-visible `v = D * w_hat` message in `zk` builds by adding fresh
Leftover Hash Lemma (LHL) D-blinding columns and proving the enlarged D relation
inside the recursive witness.

The feature modifies these surfaces:

- `akita-prover`: samples D-blinding digits, includes them in `v`, stores them
  in `QuadraticEquation`, and emits them into the recursive witness.
- `akita-verifier`: replays the same D-blinding segment in deferred ring-switch
  row evaluation.
- `akita-types`: sizes recursive witnesses and proof-size estimates with both
  B- and D-blinding columns under `zk`.
- `akita-config`: reserves enough shared-matrix stride for D-blinding columns.
- `akita-planner`: accounts for D-blinding in root witness search and
  shape-aware schedule sizing.
- `akita-pcs`: checks that repeated `zk` proofs re-randomize both commitments
  and folded `v`.

### Invariants

1. Transparent builds must preserve existing proof shapes, deterministic `v`
   values, schedule sizing, and setup sizing. All D-blinding code is gated by
   `cfg(feature = "zk")`.
2. In `zk` builds, every root and recursive folded proof must sample fresh
   D-blinding material before absorbing `ABSORB_PROVER_V`.
3. D-blinding width must use the same LHL target as B-blinding:

   ```text
   ceil((kappa * D * field_bits + 2 * 128 - 2) / (D * log_basis))
   ```

   where `kappa = d_key.row_len()`.
4. The prover and verifier must agree on recursive witness segment order. In
   `zk` builds use this witness segment order:

   ```text
   z_pre || w_hat || t_hat || B-blinding || D-blinding || r_hat
   ```

   D-blinding is deliberately placed after B-blinding so the existing `w_hat`
   and `t_hat` low-block alignment remains unchanged.
5. The D-blinding contribution must be included in both prover materialization
   and verifier deferred MLE evaluation. The D matrix uses local columns
   `[w_hat || D-blinding]`, while the recursive witness stores
   `[w_hat || t_hat || B-blinding || D-blinding]`.
6. D-blinding is prover witness material only. It is not serialized directly in
   folded proofs and is not part of `AkitaCommitmentHint`, because it blinds the
   proof-local `v` message rather than a reusable commitment.
7. Setup stride must cover `d_matrix_width + d_blinding_cols` under `zk`, while
   B stride continues to cover `outer_width + b_blinding_cols`.
8. The branch still must not claim full proof zero-knowledge. This change hides
   wire-visible Ajtai outputs `u` and `v`; it does not mask sumcheck messages,
   hide `y_ring`, or replace clear terminal witnesses with a sigma protocol.

### Non-Goals

- This feature does not implement committed sumcheck pads, `Com_pre`,
  `Com_aux1`, fused Spartan, LNP22 tail checks, or Gaussian tail sigma.
- This feature does not make root-direct proofs zero-knowledge. The root-direct
  path does not use folded `v` and remains a verifier-consistent shortcut.
- This feature does not introduce runtime switching between transparent and ZK
  modes.
- This feature does not change Fiat-Shamir labels or transcript ordering.
  `ABSORB_PROVER_V` still absorbs a `FlatRingVec`/ring-slice value in the same
  position; only its distribution changes under `zk`.

## Evaluation

### Acceptance Criteria

- The prover computes `v` with fresh D-blinding in both
  `QuadraticEquation::new_prover` and
  `QuadraticEquation::new_recursive_prover` when `zk` is enabled.
- The recursive witness contains a private D-blinding segment and verifier
  replay includes the matching D-row contribution.
- `w_ring_element_count`, `planned_w_ring_element_count`, proof-size helpers,
  and planner root sizing include both B- and D-blinding under `zk`.
- Setup envelope sizing reserves D-side blinding columns so D-matrix row lookups
  cannot exceed the shared setup stride.
- Repeated folded `zk` proofs for the same polynomial and commitment produce
  different public `v` values and still verify.
- Transparent clippy/tests continue to compile without needing the `zk` feature.

### Testing Strategy

Focused ZK test:

```bash
cargo test -p akita-pcs --features zk --test zk
```

This test covers D=32, D=64, and D=128 fp128 full-field configs. It verifies
that same-polynomial commitments re-randomize, proofs serialize/deserialize and
verify, and repeated folded proofs produce different blinded `v` values without
exposing the plain `D * w_hat` image.

Compile and lint checks:

```bash
cargo clippy -p akita-prover -p akita-verifier -p akita-types -p akita-config -p akita-planner -p akita-pcs --features zk --tests --message-format=short -q -- -D warnings
cargo clippy -p akita-prover -p akita-verifier -p akita-types -p akita-config -p akita-planner --tests --message-format=short -q -- -D warnings
```

Regression coverage should also keep representative transparent E2E tests,
batched opening tests, and planner validation passing. Dedicated negative tests
would further strengthen this feature:

- Corrupt a private D-blinding segment before ring-switch proving and expect
  verification to fail.
- Compare transparent and `zk` witness sizes for the same layout and assert the
  exact delta is `B_blinding_cols * num_commitment_groups + D_blinding_cols`.
- Check setup envelope sizing for a small layout where D-blinding, not ordinary
  `d_matrix_width`, determines the max stride.

### Performance

Transparent builds should have no intentional runtime, setup-size, or proof-size
regression.

`zk` builds add:

- one fresh D-blinding digit stream per folded proof level;
- extra D-matrix multiplication columns when computing `v`;
- one private recursive-witness segment of
  `blinding_column_count(d_key.row_len(), D, log_basis)` ring elements;
- larger setup stride when D-blinding exceeds the previous D matrix width.

For common fp128 profiles with `d_key.row_len() = 1`, the D-blinding cost matches
the per-output B-blinding count: for example, D=64 and `log_basis = 5` adds 27
ring elements to the recursive witness for D-blinding.

## Design

### Architecture

The public stage-1 value currently appears as:

```text
v = D * w_hat
```

where `w_hat` is the base-`2^log_basis` decomposition of the folded witness. In
`zk` builds the prover samples a direct digit-source mask:

```text
r_D <- balanced digit planes
v   = D_msg * w_hat + D_blind * r_D
```

The sampler is the shared
`crates/akita-prover/src/protocol/masking.rs::sample_blinding_digits`, also used
for B-blinding.

`QuadraticEquation` owns the D-blinding digits because they are proof-local
material tied to the sampled `v`. Unlike B-blinding, they do not belong in
`AkitaCommitmentHint`: hints represent commitment-opening material that must
survive across commitment APIs and recursive commitment caches, while
D-blinding is consumed by the same proof's ring-switch construction.

`ring_switch_build_w` consumes `d_blinding_digits` from `QuadraticEquation` and
passes them into:

- `compute_r_split_eq`, which adds the D-blinding cyclic rows to the D residual;
- `build_w_coeffs`, which emits the private D-blinding planes into the
  recursive witness after the B-blinding segment.

`compute_relation_matrix_col_evals` and
`akita-verifier::protocol::ring_switch::RingSwitchDeferredRowEval::eval_at_point`
mirror the same layout. The D-blinding segment uses D matrix local columns after
the ordinary `w_hat` columns:

```text
D local input = w_hat || D-blinding
```

but the recursive witness keeps B and D blinding together after `t_hat`:

```text
w_hat || t_hat || B-blinding || D-blinding
```

This preserves the prior low-block alignment invariant for `w_hat` and `t_hat`,
which is required by the peeled offset-eq evaluation.

### Alternatives Considered

1. Keep separate `sample_b_blinding_digits` and `sample_d_blinding_digits`
   wrappers.

   Rejected. The two functions were identical and made the sampler API noisier.
   Call sites now use the shared `sample_blinding_digits` directly, with the
   matrix side clear from the receiving variable name.

2. Store D-blinding in `AkitaCommitmentHint`.

   Rejected. Hints are attached to commitments and may be reused by later proof
   calls. D-blinding must be fresh per proof transcript because it blinds
   `ABSORB_PROVER_V`, not a reusable commitment object.

3. Place D-blinding immediately after `w_hat`.

   Rejected. That shifts `t_hat` by a non-block-aligned amount in many layouts
   and breaks the existing offset-eq fast path. Placing D-blinding after
   B-blinding keeps `w_hat` and `t_hat` offsets aligned exactly as before.

4. Change `LevelParams::d_key.col_len()` to include D-blinding.

   Rejected. The ordinary protocol layout and transparent schedules should not
   change. Under `zk`, setup envelope sizing and witness formulas explicitly add
   D-blinding columns without changing the semantic D message width.

## Documentation

This spec complements `specs/akita-zk-commitment-hiding.md`, which documents
B-side commitment hiding. The combined branch now hides both wire-visible Ajtai
outputs called out in `akita-zk-unified.md`: commitment rows `u` and prover rows
`v`. Future full-ZK specs should build on this without treating it as sufficient
for sumcheck or tail zero-knowledge.

## Execution

Implementation direction reflected by the branch:

- Expose one shared `sample_blinding_digits` helper.
- Sample D-blinding in root and recursive `QuadraticEquation` constructors.
- Compute `v` from `[w_hat || D-blinding]`.
- Store D-blinding as proof-local `QuadraticEquation` state.
- Consume D-blinding in ring-switch witness construction.
- Add D-blinding rows to prover `compute_relation_matrix_col_evals` and verifier deferred row
  evaluation.
- Update witness-size formulas in `akita-types`.
- Update planner formulas in `akita-planner`.
- Update setup envelope sizing in `akita-config`.
- Extend `zk` tests to assert `v` re-randomization and plain `D * w_hat` hiding.

Risks to resolve before treating this as a final public ZK surface:

- Recompute SIS margins with both B and D blinding columns included.
- Add negative tests for corrupted D-blinding witness material.
- Regenerate or audit dedicated ZK schedule tables rather than relying only on
  planner fallback.
- Continue with sumcheck masking, `y_ring` masking, and tail sigma before
  claiming full proof zero-knowledge.

## References

- `specs/akita-zk-commitment-hiding.md`
- `akita-zk-unified.md`
- `crates/akita-prover/src/protocol/masking.rs`
- `crates/akita-prover/src/protocol/quadratic_equation.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-types/src/schedule.rs`
- `crates/akita-types/src/layout/proof_size.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-planner/src/search.rs`
- `crates/akita-pcs/tests/zk.rs`
