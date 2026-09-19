# Transcript and instance binding

Akita replaces interactive verifier randomness with Fiat-Shamir challenges.
Prover and verifier derive each challenge from the same public parameters and
the same preceding messages. The transcript records that order.

For example, suppose two claimed values $v_0,v_1$ will be checked through
$v_0+\gamma v_1$. Both values must be fixed before the transcript produces
$\gamma$. If a prover could choose $v_0$ afterward, it could cancel a
chosen error in $v_1$. This is why a transcript contract specifies both
which bytes are absorbed and when each challenge is drawn.

## The transcript layer

Production uses Spongefish's native prover and verifier states. Its domain
separator includes a backend-specific protocol tag, the caller's length-framed
session bytes, and canonical instance bytes. The selected backend is BLAKE2b
or Keccak; each has its own protocol tag.

Every logical message or challenge is preceded by a fixed public context
record. Proof values use native prover emission and verifier receipt, derived
or public values use native public messages, and verifier challenges use native
verifier messages. Field atoms are canonical. The one variable-size terminal
payload carries a checked native length atom before its body.

Production absorbs and squeezes are positional. Their callsite labels are
diagnostics and do not enter sponge bytes. The session label and explicitly
encoded protocol context do enter the cryptographic state. Renaming a
diagnostic label therefore differs from changing a session label, an encoded
domain string, or replay order.

Prover and verifier must execute the same sequence, including challenge
lengths and canonical ordering within a batch. Equal proof objects alone do
not establish that agreement.

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

For coordinate index $i$, the sampler initializes a fresh SHAKE256 reader
from

```text
group_root || little_endian_u64(i).
```

Coordinate $i$ is `claim * num_live_blocks + block`. Expanding one coordinate
does not mutate either the live transcript or another coordinate's reader.
This gives the extraction argument the required fork: one challenge can change
while every other challenge and the surrounding transcript remain fixed.

Expanding the whole challenge vector from one shared cursor would give a
different oracle dependency. The challenge context encodes the numeric nonce
separately from its compact 12-bit proof representation.

The indexed readers are an expansion of one transcript root, not additional
Fiat--Shamir squeezes and not additional proof data.

### Positions, magnitudes, and signs

Suppose the challenge ring has dimension $D$. A configured challenge has
`count_pm1` coefficients at magnitude 1 and `count_pm2` coefficients at
magnitude 2. The sampler first chooses their distinct positions by a partial
Fisher--Yates shuffle of `0..D`.

When the challenge is very sparse, the implementation stores only the swaps
touched by that partial shuffle, using $O(w)$ scratch for Hamming weight
$w$. Denser cases use a fixed stack permutation for better locality. These
are two implementations of the same ordered partial shuffle and consume the
same random stream.

Every bounded integer draw uses bitmask rejection rather than `% D`, so the
position law has no modulo bias. After positions are fixed, fresh low bits
choose independent signs. The first `count_pm1` positions receive $\pm1$;
the remainder receive $\pm2$.

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

Before replay, both parties construct an `AkitaInstanceDescriptor` from the
validated public configuration. The shared
`bind_transcript_instance_descriptor` helper binds its canonical bytes
through spongefish's `DomainSeparator.instance(...)`.

The descriptor records the following identities:

| Section | What it fixes |
| --- | --- |
| Algebra | Base modulus and field extension degrees |
| Setup | Decomposition, SIS modulus profile, compression policy, setup seed digest, protocol feature flags |
| Plan | Schedule selection and the effective schedule digest |
| Grinding | The ordered grinding plan and its wire contract |
| Call | Commitment-group counts, polynomial counts, variable counts, opening-layout digest, and basis |

The effective schedule includes the level geometry, payload modes, and each
nonterminal `RingRelationMode`. Switching between quotient lifting and
reduced evaluation changes the preamble before the ring-switch challenge
$\alpha$. The verifier uses the selected mode and rejects a mismatch.

The descriptor fixes the interpretation of the call. Actual commitment
payloads, opening-point coordinates, and claimed values are absorbed during
the protocol. Binding only the call's shape would not bind those values.

For example, two calls with identical shapes but different opening values
have the same shape metadata. Their transcripts diverge when the claimed
values are absorbed. Two calls that interpret identical payload bytes under
different schedules diverge at the descriptor.

## Root and fold replay

At the root, the descriptor fixes the batch shape. Replay binds group
commitments in canonical group order, each group's complete opening point, and
the per-polynomial claimed values as public messages. The verifier checks
commitment geometry before binding the payloads.

A nonterminal fold then follows these dependencies:

1. Prepare the opening claim. If evaluation trace over a proper extension
   requires extension-opening reduction, replay that reduction first.
2. Absorb the complete opening payload, either compressed $p_H$ or raw
   $\mathbf v_D$, then draw the application claim-batching coefficients.
3. Consume the fold-response nonce and draw the sparse fold challenges for
   each group.
4. Absorb the next-witness binding. It is a recursive commitment payload or,
   on the last edge, the terminal inner-image state.
5. Draw the ring-switch challenge $\alpha$, followed by the range and row
   points $\tau_0,\tau_1$.
6. Replay Stage 1 and Stage 2. Absorb each sumcheck round message before
   drawing its round challenge. Carry the resulting witness-opening claim
   to the successor.
7. If the schedule offloads setup, replay Stage 3 and carry its separate
   setup-prefix opening alongside the witness opening.

This order makes the next witness binding independent of the random points
that check its relations. The witness itself can remain prover-only until
the terminal. [Recursion](./recursion.md) explains what each binding commits
to, and [the sumcheck stages](./proving/sumcheck-stages.md) derive the carried
claims.

A recursive successor uses the commitment and opening claim supplied by the
previous fold. If a setup prefix is present, its group precedes the witness
group. Neither party may reorder groups based on their payload contents.

## Extension-opening reduction and terminal replay

Extension-opening reduction has an earlier batching step of its own. The
prover first binds the incoming claimed values and reduction partials. The
transcript then produces the tensor-mixing challenge and the early
coefficients that batch the reduction claims. Every reduction sumcheck
message precedes its round challenge.

The reduction's final claims are absorbed before the fold's complete opening
payload. Only after that payload is bound does the application batching in
step 2 occur. These two batching steps have different inputs and different
challenge positions; they cannot reuse a challenge merely because both form
linear combinations. The
[reduction chapter](./proving/extension-opening-reduction.md) gives the full
sequence.

At the terminal, the verifier checks that the response's inner images match
the predecessor's binding. After any required extension-opening reduction,
it absorbs the ring opening partials, consumes the fold-response nonce, and
draws the sparse challenges. It then absorbs the remaining response and
performs the direct checks. There is no outgoing commitment or replay of
Stages 1 through 3.

## Grinding plan and inline nonces

Each proof has one public `GrindingPlan`, derived from the selected
schedule, normalized opening layout, field tower, and policy. The plan fixes
the order and bit width of every proof-of-work query and bounded
fold-response search. Its digest is part of the descriptor.

Nonzero proof-of-work sites and every fold-response site carry an inline
canonical unsigned LEB128 nonce at the exact protocol position where it is
used. A decoder does not
obtain a nonce count or policy from proof bytes. The plan cursor checks sites in
order and must be exhausted when native EOF is checked.

The planner accepts only complete schedules whose expanded grinding query
count is less than `u32::MAX`. If the objective-best candidate exceeds that
capacity, planning treats it as an infeasible path and continues among schedules
admitted by the planner's existing bounded candidate-generation policies.
`UnsupportedSchedule` means that no feasible schedule was found within this
domain; it does not prove that no mathematically valid schedule exists among
layouts discarded by those policies.

Proof-of-work and fold-response nonces use distinct native message kinds and
serve different purposes.

### Protected challenge queries

At a protected query with grinding target $g>0$, the prover searches a
nonce whose accepted value must fit $g+7$ bits. Each attempt absorbs the
canonical nonce, then produces a separate 32-byte predicate. Diagnostic
metadata records the grinding site without changing the sponge. The predicate
passes when its first $g$ low-order bits are zero.

The verifier repeats that predicate check. Only after it passes does replay
draw the protocol challenge from the advanced transcript. The predicate is
not reused as the challenge. A zero-bit target consumes no proof bits and
leaves the transcript unchanged at that site.

The additional seven nonce bits provide room for honest search beyond the
expected $2^g$ attempts. Storage is self-delimiting and canonical; the semantic
width is still checked from the public plan. Schedule selection retains main's
packed estimate, `ceil(sum(semantic_nonce_widths) / 8)`, so canonical schedule
regeneration remains stable. This estimate is selection-only accounting debt;
it is not the native LEB128 wire size or a verifier safety bound.

### Fold-response search

A fold-response entry carries a canonical unsigned LEB128 nonce whose value
must fit the 12-bit search domain, shared by all commitment
groups in that fold. The prover previews candidates until the resulting
response satisfies the scheduled representation and norm bounds. It commits
the winning nonce to replay, or returns an error if the bounded search is
exhausted.

The verifier reconstructs the challenges for that nonce and enforces the
response bounds. This is separate from the proof-of-work predicate. The
[PCS binding chapter](../foundations/pcs-and-binding.md#fiat-shamir-queries-and-fold-nonces)
explains why adversarial nonce trials must be included in random-oracle query
accounting.

## Integration and regression checks

`AKITA_INSTANCE_DESCRIPTOR_VERSION` is currently `5`. Validation rejects
other versions. Pin an exact Akita revision and rerun prove and verify
integration tests when upgrading; the repository does not promise
compatibility across revisions.

Binding an instance initializes the transcript state for that instance.
Application code should use the intended session label and let the scheme's
shared binding path construct the descriptor before replay. Prepending
application messages to a transcript that will then be rebound does not
preserve those messages in the new state.

The current descriptor's `SetupSection.protocol_features.zk` is
`false`. Transcript binding does not add hiding or zero knowledge.

The native protocol uses a descriptor-bound positional grammar. Context records
capture semantic sites and widths for logging diagnostics without adding
production hashing work. Tests cover prover/verifier vectors, tampering,
truncation, statement/session binding, and EOF; they do not freeze one
proof-byte digest for all future schedules. The complete message and dependency review is in
[`docs/native-proof-stream.md`](../../../docs/native-proof-stream.md).

## Code map

- `crates/akita-config/src/transcript_binding.rs` constructs and binds the
  shared descriptor and grinding plan.
- `crates/akita-types/src/instance_descriptor/mod.rs` owns descriptor fields,
  canonical serialization, and version validation.
- `crates/akita-transcript/src/native.rs` owns native state construction,
  context framing, canonical atom codecs, bounded bytes, and EOF-compatible
  proof transport.
- `crates/akita-types/src/transcript_grinding_plan.rs` defines the ordered
  plan; `crates/akita-types/src/transcript_grinding/native_replay.rs` couples
  native nonce transport, predicate checks, challenges, and plan progress.
- `crates/akita-challenges/src/sampler/xof.rs` derives the indexed sparse
  challenge streams.
- `crates/akita-verifier/src/protocol/core/fold/mod.rs` and
  `crates/akita-verifier/src/protocol/core/suffix.rs` enforce fold and
  terminal replay order.
- `crates/akita-pcs/tests/transcript_hardening.rs` tests ordering and
  prover/verifier agreement; `crates/akita-pcs/tests/fold_linf.rs` covers
  fold-response nonce behavior and proof roundtrips.

The detailed grinding contract is recorded in
[the transcript grinding specification](https://github.com/LayerZero-Labs/akita/blob/main/specs/transcript-grinding.md).
