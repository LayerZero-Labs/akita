# Transcript and Commitment Compatibility Spec (Hachi Core)

This document specifies Hachi's protocol-layer transcript and commitment interfaces.

## Scope

- Applies to Hachi core (`src/protocol/*`).
- Uses **Hachi-native** transcript labels and ordering.
- Does **not** wire Hachi into Jolt in this phase.
- Any future cross-system interop (for example Jolt-facing adaptation) must be handled by an adapter layer outside core label definitions.

## Transcript Contract

Hachi protocol transcripts implement:

- `new(domain_label)`
- `append_bytes(label, bytes)`
- `append_field(label, x)`
- `append_serde(label, s)`
- `challenge_scalar(label)`
- `reset(domain_label)`

Current core implementations:

- `Blake2bTranscript` (Blake2b-512)
- `KeccakTranscript` (Keccak-256, matching Jolt's `sha3` crate usage)

### Byte Framing

All absorbed bytes use deterministic framing:

- `label || len_le64 || bytes`

This framing is applied uniformly for raw bytes, fields, and serializable protocol objects.

### Field Encoding

- Field elements are encoded through canonical representatives (little-endian `u128` bytes).
- Challenge derivation maps transcript digest bytes into field elements via canonical reduction.

### Label Namespace

All labels are defined in `src/protocol/transcript/labels.rs` and are Hachi-native.

Reserved core labels include:

- Domain label: `hachi/protocol`
- Commitment phase (§4.1): commitment
- Reduction phase (§4.2): evaluation-claims + linear-relation challenge
- Ring-switch phase (§4.3): ring-switch-message + ring-switch challenge
- Sumcheck phase (§4.3): sumcheck-round + sumcheck-round challenge
- Recursion stop phase (§4.5): stop-condition + stop-condition challenge

Forbidden in Hachi core transcript constants:

- Dory label literals (for example `vmv_c`, `beta`, `alpha`, `gamma`, `final_e1`, `final_e2`, `d`)

## Commitment Contract

Hachi protocol commitment interfaces include:

- `CommitmentScheme`
- `StreamingCommitmentScheme`
- `AppendToTranscript`

The commitment layer defines:

- setup split (`setup_prover`, `setup_verifier`)
- commitment/opening APIs (`commit`, `prove`, `verify`)
- homomorphic combination APIs (`combine_commitments`, `combine_hints`)
- optional streaming/chunked path for large inputs
- label-directed transcript absorption (`AppendToTranscript` call sites choose event labels)

## Determinism Requirements

- Prover and verifier must absorb the same labeled byte sequence in the same order.
- Transcript challenges must be reproducible for identical input schedules.
- Commitment/proof objects absorbed via `append_serde` must use deterministic `HachiSerialize` encoding.

## Test Requirements

Tests should enforce:

- Transcript replay determinism (same schedule => same challenges).
- Label/order sensitivity (different labels/order => diverging challenges).
- Framing stability.
- No Dory-label leakage in Hachi label constants/schedules.
- Commitment/hint combination algebraic sanity.

## Deferred Integration Note

Integration into Jolt is a separate, deferred phase tracked in `HACHI_PROGRESS.md`.
When started, an adapter should translate between external transcript conventions and Hachi core interfaces without changing Hachi-native core labels.

## Deferred Adapter Contract (Design Only)

`JoltToHachiTranscript` is deferred, but its expected behavior is fixed now:

- Owns a mutable reference to a Jolt transcript object.
- Implements Hachi `Transcript<F>` by forwarding absorption/challenge calls.
- Performs label translation at the boundary (Jolt-side naming to Hachi-side API events).
- Never mutates or extends Hachi core label constants.
- Maintains deterministic call ordering: prover and verifier adapter paths must replay identical absorb/challenge sequences.
- Supports domain initialization and explicit reset semantics.

This adapter lives outside Hachi core protocol modules and is not part of this phase's implementation.
