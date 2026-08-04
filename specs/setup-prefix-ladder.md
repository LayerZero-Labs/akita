# Spec: Setup-Prefix Commitment Artifacts


| Field     | Value               |
| --------- | ------------------- |
| Author(s) |                     |
| Created   | 2026-06-08          |
| Status    | draft               |
| PR        | setup-prefix-ladder |


## Summary

This PR adds setup-prefix commitment artifacts for recursive setup-contribution
proofs. A setup prefix is a power-of-two flat coefficient prefix of the shared
setup vector `S`, committed as an ordinary Akita witness using a selected fold's
commitment parameters. The resulting slot stores verifier-visible commitment
metadata and prover-only hint material so later setup-offloading work can carry
an opening claim for `S_{<=N}` instead of rescanning the full setup matrix.

This is a partial implementation of the `SelectedSlots` mode described in
`STACK.md` slice 02B. The current branch materializes the active slots implied by
one runtime schedule and stores them in setup metadata. It does not implement
the full policy surface from the stack plan (`FullLadder`, explicit arbitrary
slot lists, strict missing-slot policy, generate-and-persist policy, or setup
artifact reuse across many workload shapes).

## Intent

### Goal

Provide serializable setup-prefix slots and populate any slots that fit the
current recursive setup schedule.

Key surfaces:

- `akita-types::proof::setup_prefix` defines `SetupPrefixSlotId`,
`SetupPrefixSlot`, `SetupPrefixVerifierSlot`, and prover/verifier registries.
- `akita-prover::api::setup_prefix::commit_setup_prefix` commits one padded
setup prefix and records the commitment hint material.
- `akita-setup::new_prover_setup_recursion` constructs ordinary setup, then
populates recursive setup-prefix slots.
- Stage-3 prover/verifier setup-product paths use a committed prefix when the
matching slot exists, and otherwise fall back to the direct setup scan.

### Invariants

- **Slot identity is deterministic.** A slot id binds the setup seed digest,
`D_setup`, `N_prefix`, and the digest of the commitment parameters used to
commit that prefix. Prover and verifier must derive the same id for a fold.
- **Prefix length is power-of-two padded.** A fold's natural setup footprint is
padded with `padded_setup_prefix_len`; the committed prefix covers the padded
field length while retaining the natural length in metadata.
- **Commitment params must carry the prefix shape.** `setup_prefix_level_params`
returns `Some(LevelParams)` only when a valid block split exists for
`N_prefix / D_setup` ring elements under the candidate fold's A/B key widths.
If no such split exists, it returns `None`.
- **Missing or unsupported slots are non-fatal in this PR.** Prover and verifier
fall back to the direct setup scan for that fold instead of failing the proof.
- **Verifier-visible metadata excludes prover hints.** `SetupPrefixSlot` stores
`AkitaCommitmentHint`; `SetupPrefixVerifierSlot` stores only the public slot
metadata and commitment.
- **No verifier panics.** Malformed setup-prefix metadata or shape mismatches
must return `AkitaError` / serialization errors.

### Non-Goals

- **Planner awareness.** The planner is not setup-prefix aware in this PR. It
does not score schedules by whether the next fold can commit the current
fold's setup prefix, and it does not synthesize next-fold parameters to make
prefix slots fit. Fixing the planner is deferred to a later PR.
- **Full `SelectedSlots` from `STACK.md`.** This PR only implements the active
selected slots for one concrete setup schedule. It does not expose a durable
user-facing selected-slot list or the complete missing-slot policy matrix.
- `**FullLadder`.** Generating every power-of-two prefix in a range is out of
scope.
- **Carried setup openings.** This PR does not yet carry setup-prefix openings
into the next recursive fold or batch them with folded-witness openings.
- **Making recursive setup offloading universal.** Unsupported folds use direct
fallback.

## Evaluation

### Acceptance Criteria

- Recursive setup construction populates every setup-prefix slot that fits
the active schedule and skips unsupported shapes.
- Prover and verifier select the same committed slot id for folds whose
slots were populated.
- Folds with no populated compatible slot continue to prove and verify via
direct setup scan fallback.
- The setup-prefix slot metadata serializes, validates, and round-trips.
- `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`, and `cargo test` pass.

### Testing Strategy

- Unit tests in `akita-types::proof::setup_prefix` cover slot id validation,
registry duplicate rejection, verifier metadata projection, active setup
footprint calculation, and prefix length selection.
- Unit tests in `akita-prover::api::setup_prefix` cover committing one prefix
slot and selecting the populated slot from the registry.
- End-to-end recursive setup tests exercise the setup-product path and direct
fallback behavior.
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
populated prefixes. It does not yet remove the setup matrix scan from every
fold: unsupported or missing slots use direct fallback. Proof-size and verifier
performance wins are expected only after the later planner-aware and carried
opening integration PRs.

### Design Note: Prefix Contents and Cache Reuse

There are two different setup-prefix commitment policies with different cache
behavior. A reusable ladder commits full power-of-two prefixes
`S[0..N_prefix]`, so one artifact can serve any active footprint up to that
prefix size. An exact selected-slot policy commits `S[0..natural_len]` followed
by zeros up to `N_prefix`; that object is cheaper only if the commitment/opening
backend exploits the zero tail, and it is specific to `natural_len`.

The current CPU path commits the padded dense block shape, so zero-padding is a
semantic choice, not a preprocessing speedup. Exact zero-padded selected slots
must include `natural_len` in the slot identity and transcript binding. A
reusable `FullLadder` should instead commit full power-of-two setup prefixes and
make inactive coordinates contribute zero through setup-product weights.

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
commitment. If not, it returns `None` and the fold uses direct setup scan.

### Current Scheduling Rule

The planner is not consulted for setup-prefix compatibility. The current
implementation blindly tries the **next fold's existing `LevelParams`** as the
commitment parameters for the current fold's setup prefix:

```text
prefix commitment params for fold i = schedule.fold[i + 1].params
```

This matches the intended direction that a setup-prefix commitment should use
the same kind of parameters as the next recursive witness commitment, but it is
not globally optimized. Some folds will fail the prefix-shape check because the
next fold's A/B key widths were chosen without considering setup-prefix
commitment needs.

Planner-aware setup-prefix scheduling is future work. That later PR should make
the planner price and constrain candidate next-fold params against the current
fold's setup-prefix footprint. The proposed per-fold mode, eligibility gates,
and two-group suffix transition are specified in
[`setup-offloading-planner.md`](setup-offloading-planner.md).

### Runtime Fallback

Setup construction only materializes slots for compatible folds. Stage-3 prover
and verifier both recompute the same candidate id from the same setup seed,
`D_setup`, padded prefix length, and prefix commitment params digest. If the
slot exists and covers the natural length, they use the padded prefix evaluation
length. Otherwise they use the full setup matrix length.

This fallback is intentionally permissive for this PR because the planner does
not yet guarantee prefix-compatible next-fold params.

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
