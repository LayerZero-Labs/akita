# Spec: Immediate Transcript Fixes

| Field       | Value                                   |
|-------------|-----------------------------------------|
| Author(s)   | Quang Dao                               |
| Created     | 2026-05-24                              |
| Status      | proposed                                |
| PR          | PR 1 of the transcript-grinding split   |

## Summary

This PR is the prerequisite layer for transcript grinding. It fixes two
transcript-shape issues without adding grinding nonces, grinding policy, or
proof-size overhead:

1. `AkitaInstanceDescriptor` binds deterministic setup identity instead of the
   expanded setup artifact.
2. Terminal folds use a terminal-specific transcript path: bind logical
   `w_hat` before sparse-seed sampling, bind the terminal witness remainder
   before `alpha`/`tau1`, and never squeeze terminal `tau0`.

The result is a smaller, cleaner first PR that can land before the grinding
proof-format cutover.

## Intent

### Goal

Land the immediate Fiat-Shamir fixes that are independent of grinding:

- Descriptor setup binding is seed/layout derived. Fiat-Shamir binds the
  canonical `AkitaSetupSeed` digest plus protocol-affecting layout, schedule,
  security, decomposition, SIS-family, and feature-mode metadata. It does not
  hash the full expanded shared matrix or prover NTT cache into the transcript.

- Terminal sparse-fold inputs are bound before the challenges that depend on
  them. In particular, the logical `w_hat` segment is absorbed before the
  sparse challenge seed, and the remaining terminal witness digits are absorbed
  before ring-switch `alpha`, grouped `tau1`, and stage-2 challenges.

- Terminal replay has no `tau0` site. Terminal folds skip stage 1, so the
  stage-1 witness-table random point has no mathematical role and must not
  advance the production transcript, consume proof data, or appear as a
  terminal logging event.

### Invariants

- Expanded setup matrices and NTT views are deterministic cache artifacts, not
  Fiat-Shamir bytes. They may be validated when setup is built or loaded, but
  descriptor equality is based on compact deterministic setup identity.

- Any deployment salt or other public setup-derivation input that changes the
  generated matrix must be part of canonical setup identity before it can affect
  verifier replay.

- Production transcript labels remain diagnostic only. New terminal event
  labels are for logging and smell checks; renaming a label must not change
  production challenge bytes when the ordered payload bytes are unchanged.

- Terminal direct witness binding is logical, not raw-byte slicing. The verifier
  must decode the canonical terminal witness into logical digits, derive the
  `w_hat` digit range from descriptor-bound layout data, absorb that segment
  before the sparse seed, and absorb the complement before later challenges.

- Verifier-reachable malformed setup, proof, descriptor, and terminal witness
  shapes reject with `AkitaError` or `SerializationError`, never with panics,
  unchecked indexing, or unchecked arithmetic.

### Non-Goals

- Do not add transcript grinding, proof nonce streams, or grinding policy in
  this PR. Those belong to `specs/transcript-grinding.md`.

- Do not preserve legacy proof or transcript byte compatibility.

- Do not migrate Akita to a NARG proof tape or make proof fields transcript
  owned. The structured proof format remains in place.

- Do not add compatibility aliases such as "artifact digest" wrappers. This is
  a full cutover to deterministic setup identity naming.

## Evaluation

### Acceptance Criteria

- [ ] `AkitaInstanceDescriptor` setup binding uses deterministic setup identity
      naming throughout, for example `SetupIdentityDigests` rather than
      `SetupArtifactDigests`.
- [ ] `SetupSection` binds `setup_seed_digest`, decomposition, SIS modulus
      family, level-parameter digest, and protocol feature mode, but contains no
      `shared_matrix_digest` or other expanded-matrix transcript digest.
- [ ] Strict expanded/verifier setup deserialization validates any materialized
      shared matrix against the serialized public-matrix seed. Recursive guest
      input decoding may use an explicit trusted cached-matrix path to skip
      this rederivation, but the name must make the trust boundary clear and
      structural/field validation must remain in place.
- [ ] Terminal fold replay uses separate terminal and non-terminal ring-switch
      helper paths. The terminal helper returns only `alpha` and grouped
      `tau1`; it must not call the `tau0` squeeze path.
- [ ] Terminal sparse-seed sampling happens only after the logical `w_hat`
      segment has been absorbed.
- [ ] Terminal ring-switch and stage-2 challenges happen only after all
      remaining terminal witness digits have been absorbed.
- [ ] Terminal prover and verifier logging streams contain no
      `CHALLENGE_TAU0` event. Tests assert this directly for terminal-root and
      recursive-terminal shapes where those shapes are reachable.
- [ ] Mutating terminal logical `w_hat`, mutating the terminal witness
      remainder, truncating the terminal witness, or changing terminal segment
      layout metadata causes verifier rejection rather than acceptance or panic.
- [ ] Existing transcript-hardening event equality and wire-before-squeeze
      smell checks remain green.

### Testing Strategy

Existing checks that must remain green:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`
- `cargo test -p akita-transcript --features logging-transcript`

New or updated checks:

- `akita-types`: descriptor serialization round trips for deterministic setup
  identity; no stale shared-matrix digest field; cached digest validation
  rejects mismatched seeds.

- `akita-scheme` / setup integration: descriptor construction uses seed and
  layout/schedule metadata, not a full shared matrix digest.

- `profile/akita-recursion` glue/guest, if this PR touches recursion input
  decoding: trusted cached-digest decode path still validates structure and
  field elements.

- Terminal transcript-order tests: assert the public event order is
  "current commitment/opening context, terminal logical `w_hat` absorb, sparse
  seed squeeze, terminal witness remainder absorb, `alpha`, grouped `tau1`,
  stage-2 challenges" and contains no terminal `tau0` squeeze.

- Terminal tamper tests: mutate logical `w_hat`, mutate the remainder, truncate
  the packed witness, and alter layout metadata; all variants reject without
  panicking.

## Design

### Deterministic Setup Identity

`SetupSection` should identify the transparent setup by the public inputs that
deterministically derive it:

```rust
pub struct SetupSection {
    pub decomposition: DecompositionSection,
    pub sis_modulus_family: SisModulusFamilySection,
    pub setup_seed_digest: [u8; 32],
    pub feature_mode: ProtocolFeatureMode,
    pub level_params_digest: [u8; 32],
}
```

`setup_seed_digest` covers the canonical `AkitaSetupSeed` bytes. The level
parameter digest covers the effective schedule and layout parameters that
determine how the generated matrix is interpreted and used: ring dimensions,
fold/direct/terminal steps, digit depths, row counts, block sizes, stage-1
configuration, public-row shape, and any other verifier branch condition.

The expanded shared matrix is not part of `SetupSection`. It is implied by the
setup seed and schedule/layout metadata in the transparent path. Strict
expanded/verifier setup decoding must validate a materialized matrix by
rederiving or checking it against the seed; that validation is separate from
Fiat-Shamir transcript input. Trusted cache consumers, such as host-produced
recursion blobs, may opt into a named cached-matrix decode path that preserves
structural/field validation while skipping the expensive rederivation.

If Akita later supports a per-deployment setup PRG salt, custom public matrix
derivation domain, or another setup-generation input, that input must become a
canonical setup-identity field. It must not be smuggled in as an unbound local
configuration knob.

### Terminal Transcript Path

Intermediate and terminal folds must have separate transcript schedules.

Intermediate fold:

1. absorb the current recursive commitment/opening context;
2. compute and absorb `v = D * w_hat` under the existing prover-value absorb;
3. absorb sparse-challenge context and squeeze the sparse seed;
4. compute `z_pre`, build the next recursive witness, and commit it;
5. absorb the next-witness commitment;
6. squeeze ring-switch `alpha`;
7. squeeze grouped `tau0` coordinates for the stage-1 witness-table point;
8. squeeze grouped `tau1` coordinates for the row-combination point;
9. run stage 1 using `tau0`;
10. absorb `s_claim`;
11. sample any needed stage-2 batching coefficient;
12. run stage 2.

Terminal fold:

1. absorb the current recursive commitment/opening context;
2. compute the decomposed terminal segment `w_hat`;
3. absorb the cleartext logical `w_hat` segment before any sparse seed is
   squeezed;
4. absorb sparse-challenge context and squeeze the sparse seed;
5. compute `z_pre`, compute `r`, and build the complete cleartext final
   witness;
6. absorb the remaining final-witness digits before ring-switch challenges;
7. squeeze ring-switch `alpha`;
8. squeeze grouped `tau1` coordinates for the row-combination point;
9. run relation-only stage 2.

Terminal folds must not squeeze `tau0`. They skip stage 1, so there is no
stage-1 witness-table point, no `s_claim`, and no stage-2 batching coefficient.

Terminal stage 2 proves:

```text
relation_claim
  = sum_{x,y} W(x,y) * a_alpha(y) * m_{tau1,alpha}(x).
```

### Terminal Witness Segmentation

The terminal direct witness remains one canonical proof object. Transcript
replay binds it in two phases by deriving a logical `w_hat` range from
descriptor-bound schedule data and the same terminal segment layout used by
stage-2 direct-witness evaluation.

All counts below are logical ring elements before converting through the packed
digit representation:

```text
num_w_vectors = descriptor-bound number of opened W vectors
num_t_vectors = descriptor-bound number of T/relation vectors
num_z_vectors = descriptor-bound number of public/folded Z rows

w_hat_ring_count = num_w_vectors * num_blocks * num_digits_open
t_hat_ring_count = num_t_vectors * num_blocks * a_key_row_len * num_digits_open
z_pre_ring_count = num_z_vectors * inner_width * num_digits_fold
z_first = m_vars >= r_vars

w_hat_digit_count = w_hat_ring_count * ring_dim
w_hat_digit_offset = if z_first { z_pre_ring_count * ring_dim } else { 0 }
```

`num_z_vectors` is the explicit public-row count carried by witness-layout
helpers. It is independent of `num_w_vectors` and must not be inferred from it.

The verifier first decodes and validates the packed terminal witness into the
canonical logical final-witness digit stream, then extracts:

```text
[w_hat_digit_offset, w_hat_digit_offset + w_hat_digit_count)
```

The verifier must not slice raw `PackedDigits` bytes. The representation is
bit-packed, and logical digit boundaries need not be byte boundaries. The
remainder is every terminal witness digit outside the logical `w_hat` range, in
canonical final-witness order. This avoids relying on a prefix convention:
current layouts may place `z_pre` before `w_hat` when `m_vars >= r_vars`.

Verifier replay rejects malformed terminal proofs whose packed witness is too
short for the derived range, whose remainder length does not match the
descriptor-bound final-witness shape, whose extracted `w_hat` digits are not
representable in the scheduled digit basis, or whose event stream contains any
terminal `CHALLENGE_TAU0` squeeze.

## Documentation

Update:

- `specs/transcript-hardening.md` with a short note that setup identity is
  deterministic and seed/layout derived, not expanded-artifact derived.
- `profile/akita-recursion/README.md`, if recursion input decoding changes, to
  describe the trusted cached-matrix fast path and the validation it preserves.
- Transcript logging docs with the terminal event order and the "no terminal
  `tau0`" invariant.

## Execution

Suggested implementation order:

1. Rename setup artifact digest types and constructors to setup identity names.
2. Remove expanded-matrix digest fields from descriptor setup binding and
   update descriptor serialization tests.
3. Update expanded setup serialization/deserialization so cached descriptor
   digests validate against the seed without becoming full-matrix transcript
   input.
4. Add terminal-specific ring-switch challenge helpers: non-terminal returns
   `alpha`, grouped `tau0`, grouped `tau1`; terminal returns only `alpha` and
   grouped `tau1`.
5. Split terminal direct-witness transcript absorption into
   logical-`w_hat`-before-sparse-seed and remainder-before-ring-switch phases.
6. Add logging and tamper tests for terminal event order, absent terminal
   `tau0`, and malformed witness rejection.
7. Run the acceptance commands.

## References

- `specs/transcript-hardening.md`
- `specs/transcript-grinding.md`
- `crates/akita-types/src/instance_descriptor.rs`
- `crates/akita-types/src/proof/setup.rs`
- `crates/akita-transcript/src/sponge.rs`
- `crates/akita-transcript/src/logging.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-sumcheck/src/types.rs`
