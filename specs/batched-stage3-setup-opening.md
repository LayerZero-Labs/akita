# Spec: Batched Stage-3 Setup and Next-Witness Opening

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     |                                |
| Created       | 2026-06-23                     |
| Status        | proposed                       |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  |                                |

## Summary

Recursive setup-contribution mode used to prove the setup product in stage 3
after stage 2, while the next recursive witness opening stayed at the point
produced by stage 2. That left two independent opening points in the recursive
suffix: one for the next witness and one for the setup prefix/product.

This spec batches the stage-3 setup product with the stage-2 next-witness result
by carrying the stage-2 witness opening through the stage-3 sumcheck. The
batched stage-3 sumcheck runs over a common padded Boolean cube whose
projections open both the next witness and the setup prefix at points derived
from the same transcript challenges.

## Intent

### Goal

Replace standalone recursive setup-product stage 3 with a batched stage-3
sumcheck that proves the setup contribution and re-randomizes the next-witness
opening point in one transcript-bound stage.

Key surfaces:

- `akita-prover::protocol::sumcheck::setup_sumcheck` provides the batched
  stage-3 prover, combining the setup product with a carried next-witness term.
- `akita-verifier::stages::stage3` provides the verifier counterpart and final
  relation for the batched sumcheck.
- `akita-types::proof::levels::SetupSumcheckProof` carries the setup claim, the
  batched next-witness opening claim, and the batched stage-3 sumcheck proof.
- `akita-prover::protocol::core::fold::prove_fold` writes the suffix state from
  the batched stage-3 point in recursive setup mode.
- `akita-verifier::protocol::core::fold::verify_fold` mirrors the same suffix
  point derivation.

### Mathematical Model

Stage 2 ends at

```text
r2 = (r_y, r_x)
```

and produces the next-witness opening

```text
w2 = W(r2).
```

The recursive setup contribution for the same fold is

```text
setup_claim(r_x) =
  sum_{lambda, y} S(lambda, y) * setup_index_weight_{r_x}(lambda) * alpha(y),
```

where `setup_index_weight_{r_x}` is derived from the stage-2 `x` challenges and the
ring-switch/setup-contribution plan.

The batched stage-3 sumcheck samples a fresh point over a common cube:

```text
w_vars     = y_bits + x_bits
setup_vars = y_bits + lambda_bits
batched_vars = max(w_vars, setup_vars)

rho in F^batched_vars
rho_w     = rho[..w_vars]
rho_setup = rho[..setup_vars]
```

Thus one native point is a prefix/projection of the other whenever the two
native domains have different lengths.

The batched input claim is

```text
eta * w2 + setup_claim(r_x),
```

where `eta` is sampled after the stage-2 next-witness evaluation is absorbed.
The sumcheck proves the multilinear polynomial

```text
F(z) =
  eta * Lift_w(eq(r2, z_w) * W(z_w))
  +
  Lift_setup(S(z_setup) * setup_index_weight_{r_x}(lambda(z_setup)) * alpha(y(z_setup))).
```

`Lift_n_to_N` embeds an `n`-variable polynomial into the `N = batched_vars` cube
with a normalization factor:

```text
Lift_n_to_N(f)(z_0, ..., z_{N-1}) =
  2^{-(N - n)} * f(z_0, ..., z_{n-1}).
```

This preserves the original sum:

```text
sum_{z in {0,1}^N} Lift_n_to_N(f)(z) =
sum_{u in {0,1}^n} f(u).
```

At the final batched point `rho`, the verifier checks

```text
final_claim =
  eta * lift_w_scale
      * eq(r2, rho_w)
      * W(rho_w)
  +
  lift_setup_scale
      * S(rho_setup)
      * setup_index_weight_{r_x}(rho_lambda)
      * alpha(rho_y),
```

where

```text
lift_w_scale     = 2^{-(batched_vars - w_vars)}
lift_setup_scale = 2^{-(batched_vars - setup_vars)}.
```

The next suffix opening point becomes `rho_w`, not the old stage-2 point `r2`.
The setup prefix/product is opened at `rho_setup`.

### Invariants

- **Stage-2 claim is carried, not re-proved independently.** The value `w2 =
  W(r2)` absorbed after stage 2 is the input claim for the carried witness term
  in batched stage 3.
- **Common transcript point.** The next witness and setup prefix use projections
  of the same stage-3 challenge vector `rho`.
- **Sum preservation under padding.** If one native domain is shorter than the
  common cube, its lifted term is scaled by `2^{-(batched_vars - native_vars)}`.
- **Setup contribution still depends on stage-2 x.** Stage 3 is constructed
  after stage 2 has produced `r_x`, because `setup_index_weight_{r_x}` depends on those
  challenges.
- **Recursive suffix state changes only in recursive setup mode.** In
  `SetupContributionMode::Direct`, the next suffix state remains the stage-2
  opening point and claim.
- **The consumer is nonterminal.** Recursive Stage 3 may create a setup-prefix
  opening only when the successor fold can consume that prefix and still produce
  another committed witness. Direct steps and terminal folds consume exactly one
  group, as required by `specs/multi-group-batching.md`.
- **No missing-slot fallback.** Recursive mode requires the exact setup-prefix
  slot selected by the schedule. Setup construction returns `InvalidSetup` when
  it is absent; proving may not replace the offloaded opening with a direct setup
  evaluation.
- **Prover/verifier transcript symmetry.** Both sides absorb the stage-2
  next-witness opening, sample `eta`, replay batched stage-3 rounds, absorb the
  stage-3 `W(rho_w)` opening under `ABSORB_STAGE3_NEXT_W_EVAL`, and derive the
  same `rho_w`/`rho_setup` projections before the next suffix fold samples any
  challenges.
- **No verifier panics.** Mismatched proof shape, bad domain sizes, malformed
  prefix projections, or inconsistent final claims return `AkitaError`.

### Non-Goals

- This spec does not change stage 1 or the existing stage1-to-stage2 batching.
- This spec does not make recursive setup mode the default.
- This spec does not require `x_bits == lambda_bits`; the common cube handles
  unequal lengths.
- This spec does not introduce a new commitment scheme for setup prefixes. It
  reuses the current setup-prefix slot machinery when available.
- ZK masking changes are not designed here beyond preserving the existing masked
  next-witness opening convention. A ZK implementation must extend this spec
  with explicit mask accounting before it ships under `feature = "zk"`.

## Evaluation

### Acceptance Criteria

- [ ] Recursive setup mode produces and verifies a batched stage-3 proof whose
      final relation includes both the setup product and the carried
      next-witness term.
- [ ] `SuffixProverState.sumcheck_challenges` is set to `rho_w` in recursive
      setup mode.
- [ ] `SuffixProverState.opening` is set to `W(rho_w)` in recursive setup mode.
- [ ] The verifier derives the next suffix verifier state from the batched
      stage-3 point, not from the stage-2 point.
- [ ] Both prover and verifier absorb `W(rho_w)` with
      `ABSORB_STAGE3_NEXT_W_EVAL` before deriving any next-suffix challenges.
- [ ] Direct setup mode remains byte-for-byte compatible unless proof-shape
      metadata explicitly changes.
- [ ] Recursive mode is rejected when its successor would consume the setup
      prefix in a terminal fold.
- [ ] A missing or mismatched required setup-prefix slot rejects without direct
      evaluation fallback.
- [ ] Tampering with either the carried witness opening, the setup claim, or a
      batched stage-3 round polynomial is rejected.
- [ ] Tests cover both `w_vars < setup_vars` and `w_vars > setup_vars`.

### Testing Strategy

- Add unit tests for the lifting scale:
  - equal domain sizes;
  - witness domain shorter than setup domain;
  - setup domain shorter than witness domain.
- Add prover/verifier tests for the batched stage-3 final relation using small
  deterministic tables where `W`, `S`, `setup_index_weight`, and `alpha` are materialized.
- Extend recursive setup e2e tests so the next suffix level verifies against
  `rho_w`.
- Add transcript tamper tests:
  - mutate the carried `W(rho_w)` claim;
  - mutate the batched stage-3 input claim;
  - mutate one setup-product round coefficient;
  - mutate one carried-witness round coefficient.
- Existing direct-mode e2e suites must continue to pass unchanged.

### Performance

The batched proof adds the carried witness term to stage 3, but should remove
the need for a separate next-witness opening point in recursive setup mode.
Expected effects:

- Stage-3 round count becomes `max(y_bits + x_bits, y_bits + lambda_bits)`.
- Each batched round accumulates both setup-product and carried-witness
  contributions until the shorter native domain is exhausted.
- Proof size changes from standalone setup-product rounds to batched rounds plus
  the carried next-witness opening at `rho_w`.
- The implementation should avoid materializing a full common-cube table. The
  setup product remains factored/materialized as today, and the carried witness
  term should reuse the folded `W` table or a streaming fold state.

## Design

### Architecture

#### Prover Flow

For a non-terminal fold in `SetupContributionMode::Recursive` whose successor is
also nonterminal:

1. Run stage 1 as today.
2. Run stage 2 as today to obtain `r2 = (r_y, r_x)` and `w2 = W(r2)`.
3. Absorb the transcript-visible `w2` claim using the existing stage-2
   next-witness evaluation label.
4. Build the setup-contribution plan from `r_x`.
5. Compute `setup_claim(r_x)`.
6. Sample `eta`.
7. Run batched stage 3 over `batched_vars`.
8. Let `rho` be the batched stage-3 challenge vector.
9. Compute/prove `W(rho_w)`.
10. Absorb `W(rho_w)` with `ABSORB_STAGE3_NEXT_W_EVAL`.
11. Store the next suffix state as:

    ```text
    sumcheck_challenges = rho_w
    opening = W(rho_w)
    ```

The old stage-2 point `r2` remains part of the batched stage-3 witness-carry
relation through `eq(r2, rho_w)`.

#### Verifier Flow

For a non-terminal fold in `SetupContributionMode::Recursive` whose successor is
also nonterminal:

1. Verify stage 1 and stage 2 as today, obtaining `r2`.
2. Absorb the stage-2 next-witness claim.
3. Build the setup-contribution verifier plan from `r_x`.
4. Read the batched stage-3 proof and carried `W(rho_w)` claim.
5. Sample `eta`.
6. Verify batched stage 3 over `batched_vars`.
7. Split the batched challenge vector into `rho_w` and `rho_setup`.
8. Check the final relation above.
9. Absorb the verified `W(rho_w)` claim with `ABSORB_STAGE3_NEXT_W_EVAL`.
10. Thread the next suffix verifier state with `opening_point = rho_w` and
   `opening = W(rho_w)`.

#### Proof Shape

`SetupSumcheckProof` evolves from:

```text
claim
sumcheck
```

to a batched recursive-stage payload:

```text
setup_claim
carried_witness_opening
batched_sumcheck
```

The verifier computes the batched input claim as:

```text
eta * stage2_next_w_eval + setup_claim.
```

If ZK support is added, the proof shape must distinguish public/masked carried
openings and update the hidden witness cursor accounting.

#### Domain Order

The first `y_bits` variables are the ring-coordinate variables for both native
domains. The remaining tail variables are interpreted as:

```text
rho_w_tail     = rho_tail[..x_bits]
rho_setup_tail = rho_tail[..lambda_bits]
```

This order preserves the current setup-product convention that the setup proof
is over ring coordinate `y` and setup row `lambda`, while allowing the next
witness point to be a prefix/projection of the same batched point.

### Alternatives Considered

- **Run setup stage 3 before stage 2.** Rejected because `setup_index_weight_{r_x}` depends on
  stage-2 `x` challenges.
- **Only share the `y` coordinate.** This is simpler but does not make the next
  witness and setup prefix derive from one common point. It still leaves
  separate tail challenges.
- **Force `x_bits == lambda_bits`.** Rejected because it would make schedule and
  setup-prefix feasibility unnecessarily brittle. The padded common cube handles
  unequal domains.
- **Keep standalone stage 3 and batch openings later.** Rejected for this goal:
  it does not move the next suffix opening point onto the setup-prefix point.

## Documentation

If implemented, fold this into:

- `book/src/how/proving/sumcheck-stages.md`
- `book/src/how/proving/root-fold-ring-switch.md`
- `book/src/how/recursion.md`

The durable docs should describe the stage2-to-stage3 carried-claim relation in
parallel with the existing stage1-to-stage2 relation.

## Execution

Suggested implementation sequence:

1. Add pure helpers for domain projection and lifting-scale computation in the
   stage-3 module, with unit tests.
2. Refactor `SetupSumcheckProver` so its setup-product state can be advanced
   round-by-round and embedded into a batched driver.
3. Add a carried-witness stage-3 term:
   - input claim `w2`;
   - old point `r2`;
   - final opening `W(rho_w)`;
   - eq factor `eq(r2, rho_w)`;
   - lifting scale.
4. Implement the batched prover and verifier drivers.
5. Change `SetupSumcheckProof` and serialization shape.
6. Update `prove_fold` and `verify_fold` state threading.
7. Add e2e and tamper tests.
8. Revisit ZK masking separately before enabling this under `feature = "zk"`.

Risks to resolve before implementation:

- Whether the prover can evaluate `W(rho_w)` cheaply from the post-stage-2 state
  or needs to retain/refold the next-witness table.
- How to keep proof-shape compatibility for old direct-mode proofs.
- Exact mask allocation and cursor updates for ZK recursive setup mode.

## References

- `specs/setup-product-sumcheck.md`
- `specs/setup-prefix-ladder.md`
- `specs/setup-offloading-planner.md` — planner-owned per-fold offload
  selection and successor two-group scheduling.
- `crates/akita-prover/src/protocol/core/fold.rs`
- `crates/akita-prover/src/protocol/sumcheck/setup_sumcheck.rs`
- `crates/akita-verifier/src/stages/stage3.rs`
- `crates/akita-types/src/proof/levels.rs`
