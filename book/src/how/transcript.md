# Transcript and instance binding

The Fiat-Shamir layer and the canonical preamble that binds the instance before
any protocol replay, so prover and verifier squeeze identical challenges.

## The transcript layer

Production code uses spongefish-backed `AkitaTranscript` with production-ZST
labels (labels are diagnostics and must **not** enter production sponge bytes).

Active hardening pillars:

| Pillar | Requirement |
|--------|-------------|
| **P0** | Bind canonical `AkitaInstanceDescriptor` bytes through spongefish `DomainSeparator.instance(...)` before protocol replay |
| **P2** | Use `AkitaTranscript` plus production-ZST labels only as diagnostics |
| **P3** | `LoggingTranscript` tests enforce prover/verifier event-stream equality and wire-before-squeeze discipline |

## Grinding plan and nonce stream

Each proof has one public `GrindingPlan`, derived from the selected schedule,
normalized opening layout, field tower, and descriptor-bound policy before the
proof shape is constructed. The plan fixes the order and bit width of every
transcript proof-of-work query and every bounded fold-response search. Its
digest is part of the instance descriptor, and its total bit count fixes the
leading headerless `TranscriptNonceStream` in the proof.

Proof-of-work and fold-response search use the same packed stream but have
different security meanings. A protected Fiat-Shamir query first checks its
scheduled nonce against a separate 32-byte predicate, then draws the protocol
challenge from the advanced transcript. A fold-response nonce is instead a
12-bit honest-prover retry value, shared by all commitment groups in that fold;
the verifier still checks the resulting response representation and norm
bound. Zero-bit sites consume no proof bits and do not change the transcript.

Prover and verifier must consume the complete plan in order. Truncation,
reordering, nonzero tail padding, or leftover nonce bits is an error.

Implementation: `crates/akita-types/src/transcript_grinding_plan.rs` and
`crates/akita-transcript/src/grinding.rs`.
Normative design: [`specs/transcript-grinding.md`](../../specs/transcript-grinding.md).

Deferred work: prover/verifier trait split, `Bound<T>`, algorithm-as-bytes digest, NARG migration.

Implementation: `crates/akita-transcript/`.
Tests: `crates/akita-pcs/tests/transcript_hardening.rs`.

## Sparse fold challenges

A fold needs one sparse ring challenge for every `(claim, live block)` pair.
Akita draws them in claim-major order. It performs only one live transcript
squeeze per commitment group, while still giving each pair an independently
forkable random-oracle coordinate.

### The group root

Before the squeeze, the transcript absorbs the complete public draw context:

- group index, number of live blocks, and number of claims;
- total number of challenge coordinates;
- challenge ring dimension;
- counts of coefficients at magnitude 1 and magnitude 2;
- the shared fold-response grinding nonce;
- the coefficient-packing method domain and challenge-subring dimension, when
  coefficient packing is selected; and
- the operator-norm rejection policy, when the selected L2 route requires it.

The transcript then squeezes one 32-byte group root. Evaluation trace preserves
its established domain encoding. Coefficient packing adds a distinct method
domain so the same transcript state cannot reinterpret a draw under the two
opening methods.

### One indexed stream per coordinate

For coordinate index \(i\), the sampler initializes a fresh SHAKE256 reader
from

```text
group_root || little_endian_u64(i).
```

Coordinate \(i\) is `claim * num_live_blocks + block`. Expanding one coordinate
does not mutate either the live transcript or another coordinate's reader.
This gives the extraction argument the required fork: one challenge can change
while every other challenge and the surrounding transcript remain fixed.

The indexed readers are an expansion of one transcript root, not additional
Fiat--Shamir squeezes and not additional proof data.

### Positions, magnitudes, and signs

Suppose the challenge ring has dimension \(D\). A configured challenge has
`count_pm1` coefficients at magnitude 1 and `count_pm2` coefficients at
magnitude 2. The sampler first chooses their distinct positions by a partial
Fisher--Yates shuffle of `0..D`.

When the challenge is very sparse, the implementation stores only the swaps
touched by that partial shuffle, using \(O(w)\) scratch for Hamming weight
\(w\). Denser cases use a fixed stack permutation for better locality. These
are two implementations of the same ordered partial shuffle and consume the
same random stream.

Every bounded integer draw uses bitmask rejection rather than `% D`, so the
position law has no modulo bias. After positions are fixed, fresh low bits
choose independent signs. The first `count_pm1` positions receive \(\pm1\);
the remainder receive \(\pm2\).

### Optional operator-norm rejection

An L2 fold may require a challenge whose negacyclic convolution operator norm
is below a scheduled threshold. For supported D64 and D128 challenge families,
the sampler tests each indexed candidate against the certified predicate and
continues reading the same coordinate stream until one is accepted. The search
is capped at 4096 candidates.

This rejection rule is part of the public challenge method. The policy and
threshold are bound before the group root is squeezed, and the verifier repeats
the same deterministic search. Coefficient-packing folds use the L-infinity
security route and reject an operator-norm policy.

Implementation:

- `crates/akita-challenges/src/fold_draw.rs` binds the group context and owns
  the single transcript squeeze.
- `crates/akita-challenges/src/sampler/xof.rs` defines the indexed SHAKE256
  stream and unbiased bounded draws.
- `crates/akita-challenges/src/sampler/position_sample.rs` implements the
  partial Fisher--Yates paths.
- `crates/akita-challenges/src/sampler/signed_sparse.rs` assigns magnitudes and
  signs.
- `crates/akita-challenges/src/sampler/op_norm.rs` checks the certified
  operator-norm predicate.

## AkitaInstanceDescriptor

The canonical descriptor binds algebra, setup, plan, and call shape.
Prover and verifier share one helper:

- `crates/akita-config/src/transcript_binding.rs` — `bind_transcript_instance_descriptor`
- `crates/akita-types/src/instance_descriptor/mod.rs` — descriptor shape and serialization

The descriptor is absorbed before any protocol message or challenge. This
binds the transcript to the selected algebra, setup, schedule, and public call
shape rather than trusting those choices from later proof bytes.

### Integrator note (Jolt / recursion hosts)

`AKITA_INSTANCE_DESCRIPTOR_VERSION` is currently **`1`**. Validation rejects
any other version. Pin an exact Akita git revision and rerun prove and verify
integration tests when upgrading. The repository does not promise
compatibility across revisions.

After the zk-strip cutover, `SetupSection.protocol_features.zk` is always
`false` on the wire. Ongoing wire regression is covered by serde roundtrips and
end-to-end prove→serialize→deserialize→verify tests in `akita-pcs` (for example
`akita_e2e.rs`, `fold_linf.rs`), not by pinned proof-byte digests.
