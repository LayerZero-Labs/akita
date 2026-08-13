# Spec: Setup-Prefix Commitment Artifacts


| Field     | Value               |
| --------- | ------------------- |
| Author(s) |                     |
| Created   | 2026-06-08          |
| Status    | superseded          |
| PR        | setup-prefix-ladder |
| Superseded-by | full-setup-prefix-compact-tail-weights.md; book/src/how/verifying/setup_contribution.md |


## Summary

This historical PR added setup-prefix commitment artifacts for recursive setup-contribution
proofs. A setup prefix is a power-of-two flat coefficient prefix of the shared
setup vector `S`, committed as an ordinary Akita witness using a selected fold's
commitment parameters. The resulting slot stores verifier-visible commitment
metadata and prover-only hint material so later setup-offloading work can carry
an opening claim for `S_{<=N}` instead of rescanning the full setup matrix.

The missing-slot and planner-status statements in the original draft are
superseded. Current recursive Stage 3 requires a selected setup-prefix slot and
carried full-prefix opening claim; direct setup contribution does not execute
Stage 3.

## Intent

### Goal

Provide serializable setup-prefix slots and populate any slots that fit the
current recursive setup schedule.

Key surfaces:

- `akita-types::proof::setup_prefix` defines `SetupPrefixSlotId`,
`SetupPrefixSlot`, `SetupPrefixVerifierSlot`, and prover/verifier registries.
- `akita-prover::api::setup_prefix::commit_setup_prefix` commits one actual
full setup prefix and records the commitment hint material.
- `akita-setup::new_prover_setup_recursion` constructs ordinary setup, then
populates recursive setup-prefix slots.
- Stage-3 prover/verifier setup-product paths require the matching committed
prefix slot selected by the schedule.

### Invariants

- **Slot identity is deterministic.** A slot id binds the natural active setup
length and the commitment parameters used to commit that prefix. Prover and
verifier must derive the same id for a fold; the registry binds slots to the
public setup seed.
- **Prefix length is power-of-two.** A fold's natural setup footprint determines
`padded_setup_prefix_len`; the committed prefix covers that full actual setup
field length while retaining the natural length in the slot identity.
- **Commitment params must carry the prefix shape.** `setup_prefix_level_params`
returns `Some(LevelParams)` only when a valid block split exists for
`N_prefix / D_setup` ring elements under the candidate fold's A/B key widths.
If no such split exists, it returns `None`.
- **Missing or unsupported selected slots are fatal.** Prover and verifier reject
recursive Stage 3 when the selected slot is absent or mismatched.
- **Verifier-visible metadata excludes prover hints.** `SetupPrefixSlot` stores
`AkitaCommitmentHint`; `SetupPrefixVerifierSlot` stores only the public slot
metadata and commitment.
- **No verifier panics.** Malformed setup-prefix metadata or shape mismatches
must return `AkitaError` / serialization errors.

### Non-Goals

- **Planner awareness.** Superseded. The recursive planner now selects and
validates carried setup-prefix slots as part of schedule construction.
- **Full `SelectedSlots` from `STACK.md`.** This PR only implements the active
selected slots for one concrete setup schedule. It does not expose a durable
user-facing selected-slot list or the complete missing-slot policy matrix.
- `**FullLadder`.** Generating every power-of-two prefix in a range is out of
scope.
- **Carried setup openings.** Superseded. Recursive suffix openings now carry
setup-prefix opening claims.
- **Making recursive setup offloading universal.** Unsupported selected slots
reject instead of falling back inside Stage 3.

## Evaluation

### Acceptance Criteria

- Recursive setup construction populates every selected setup-prefix slot
required by the active schedule.
- Prover and verifier select the same committed slot id for folds whose
slots were populated.
- Superseded: folds with no populated compatible selected slot reject.
- The setup-prefix slot metadata serializes, validates, and round-trips.
- `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`, and `cargo test` pass.

### Testing Strategy

- Unit tests in `akita-types::proof::setup_prefix` cover slot id validation,
registry duplicate rejection, verifier metadata projection, active setup
footprint calculation, and prefix length selection.
- Unit tests in `akita-prover::api::setup_prefix` cover committing one prefix
slot and selecting the populated slot from the registry.
- End-to-end recursive setup tests exercise the setup-product path with selected
setup-prefix slots.
- For local manual verification, run the recursive setup profile:

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 AKITA_SETUP_MODE=recursive \
  cargo run --release -p akita-pcs --example profile
```

To inspect prefix-slot agreement across setup, prover, and verifier, pipe the
same command through a log filter:

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 AKITA_SETUP_MODE=recursive \
  cargo run --release -p akita-pcs --example profile
```

### Performance

This PR adds setup preprocessing work and stores extra setup metadata for
populated prefixes.

### Design Note: Prefix Contents and Cache Reuse

The old exact selected-slot policy committed `S[0..natural_len]` followed by
zeros up to `N_prefix`. That policy is superseded by
[`full-setup-prefix-compact-tail-weights.md`](full-setup-prefix-compact-tail-weights.md).
Selected slots now commit the actual full power-of-two prefix
`S[0..N_prefix]`; `natural_len` remains in the slot identity because it defines
the active setup-index weight support. Inactive coordinates contribute zero
through setup-product weights, not through synthesized commitment contents.

## Design

### Prefix Shape

For one fold, compute:

```text
N_active^F = active_setup_field_len(current_level_params, incidence, D_setup)
N_prefix = next_power_of_two(N_active^F)
ring_slots = N_prefix / D_setup
```

The prefix is committed as a witness with:

```text
ring_slots = num_live_blocks * num_positions_per_block
```

`setup_prefix_level_params(next_params, N_prefix, D_setup)` searches for a
power-of-two `num_live_blocks` divisor of `ring_slots` such that the normal Akita
commitment dimensions fit:

```text
num_positions_per_block * num_digits_inner <= a_key.col_len()
num_live_blocks * a_key.row_len() * num_digits_open <= b_key.col_len()
```

If a split fits, the function returns repacked `LevelParams` for the prefix
commitment. If not, recursive setup planning rejects the unsupported selected
slot.

### Current Scheduling Rule

Superseded. Current recursive planning selects and validates setup-prefix
commitment parameters as part of schedule construction:

```text
prefix commitment params for fold i = schedule.fold[i + 1].setup_prefix
```

Missing or incompatible selected slots are invalid for recursive Stage 3.

## Documentation

This spec records the current PR scope and its limitations relative to
`STACK.md`. `STACK.md` remains the durable stack plan; this document describes
the implemented subset.

## References

- `STACK.md`, slice 02B (`setup-prefix-ladder`) and 02D (`setup-offload-gating`)
- `specs/setup-product-sumcheck.md`
- `specs/setup-offloading-planner.md`
- `crates/akita-types/src/proof/setup_prefix.rs`
- `crates/akita-prover/src/api/setup_prefix.rs`
- `crates/akita-setup/src/recursion.rs`
- `crates/akita-prover/src/protocol/sumcheck/setup_sumcheck.rs`
- `crates/akita-verifier/src/stages/stage3.rs`
