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

Akita pins Spongefish v0.7.4. Its digest bridge encodes squeeze counters as
fixed-width `u64` values, so Blake2b transcript bytes do not depend on whether
the implementation uses 32-bit or 64-bit pointers. Akita keeps a known-answer
vector for this boundary.

Every logical message or challenge has a fixed public diagnostic context
record. Proof values use native prover emission and verifier receipt, derived
or public values use native public messages, and verifier challenges use native
verifier messages. Field atoms are canonical. The one variable-size terminal
payload carries a checked native length atom before its body.

Production absorbs and squeezes are positional. Context records and callsite
labels are diagnostics and do not enter sponge bytes. The backend-specific
protocol identifier, length-framed session, canonical instance descriptor,
messages, and replay order do enter the cryptographic state. Renaming a
diagnostic label therefore differs from changing any of those bound values.

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
different oracle dependency. Each accepted nonce has one canonical unsigned
LEB128 representation: those exact proof bytes are absorbed into the live
transcript before the challenge context is derived. There is no second compact
nonce representation or separate nonce absorption.

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

Each field challenge site has a *loss* $L$. The security argument allows at
most $L$ of the $|E|$ possible challenge values to be bad for the verifier, so
one attempt at the challenge lands on a bad value with probability at most
$L/|E|$. The next subsection shows where a loss comes from for one family of
sites.

Grinding makes each attempt cost $2^g$ hash queries on average. The plan
chooses the least nonnegative integer $g$ with $L 2^{128} \le |E| 2^g$, which
keeps the chance per query at most $2^{-128}$. The calculation uses integers
and the actual prime power, including the deficit below a power of two.

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
width is still checked from the public plan. Schedule selection adds the
per-message native maxima, `ceil(semantic_nonce_width / 7)`. This deterministic
cost is not the realized LEB128 wire size. The verifier safety bound is derived
separately from the complete native grammar.

### The loss of a claim-batching challenge

**The problem.** A fold often opens several polynomials at once. The prover
states $N$ claimed values $v_1,\dots,v_N$, one per polynomial, and the
verifier checks them together instead of one at a time. It draws
coefficients $\eta_1,\dots,\eta_N$ from $E$, one fresh value per claim, and
the rest of the fold proves the single combined claim
$\sum_k \eta_k v_k$. The question is how many bad values this draw has, that
is, what loss to price it at.

**The tempting answer, $L=1$.** Write $u_k$ for the true value of the $k$th
polynomial and $\delta_k = v_k - u_k$ for the error in its claim. The
combined claim is correct exactly when $\sum_k \eta_k \delta_k = 0$. If some
$\delta_k$ is nonzero and the errors are fixed before the coefficients are
drawn, this is one nonzero linear equation in the $\eta_k$, and a random draw
satisfies it with probability at most $1/|E|$. That suggests $L = 1$.

**Why the security argument cannot use it.** Knowledge soundness is proved
with an extractor: a procedure that reruns a prover that makes the verifier
accept and recovers the committed polynomials from its answers. The true
values $u_k$ are evaluations of those recovered polynomials, and the
extractor only obtains them from the part of the proof that comes after
this challenge. Commitment binding does not help. It says an efficient
prover cannot produce two different openings of one commitment; it does not
hand the argument a single opening, and hence fixed errors $\delta_k$, before
the coefficients are drawn. So the argument must work from accepting runs
alone.

**What the extractor does.** It reruns the prover from this challenge with
$N+1$ coefficient vectors: a base vector $\eta$, and for each $k$ a vector
$\eta^{(k)}$ that differs from $\eta$ only in coordinate $k$. The claims
$v_k$ were sent before the challenge, so they are the same in every run.
Every run also opens the same committed polynomials, or the extractor has found a commitment
collision, so every run has the same errors $\delta_k$. The base run gives
$\sum_j \eta_j \delta_j = 0$, and run $k$ gives the same sum with $\eta_k$
replaced by $\eta^{(k)}_k$. Subtracting the two leaves
$(\eta^{(k)}_k - \eta_k)\,\delta_k = 0$, so $\delta_k = 0$.

For example, with $N = 2$ the extractor uses three runs, with coefficient
vectors $(a, b)$, $(a', b)$ and $(a, b')$ where $a' \ne a$ and $b' \ne b$.
The first two runs give $(a - a')\,\delta_1 = 0$, and the first and third
give $(b - b')\,\delta_2 = 0$. Both claims are therefore correct.

**The resulting loss.** The extractor fails only if, for some coordinate
$k$, the prover succeeds with one value of $\eta_k$ and with no other value,
the other coordinates held fixed. That costs at most one bad value per
coordinate, $N$ in all, so the site has loss $L = N$. Other sites in Akita
batch claims with powers $1, \gamma, \dots, \gamma^{N-1}$ of one scalar. That
check is a nonzero polynomial of degree at most $N-1$ in $\gamma$, so those
sites have loss $N-1$.

In the production field, $|E| = (2^{64}-59)^2$ is slightly below $2^{128}$.
One polynomial draws no coefficients and has no site. Nine polynomials give
$L = 9$, and $9 \cdot 2^{128}/|E|$ lies between 8 and 16, so the site needs
$g = 4$, where $L = 1$ would give $g = 1$.

Two sites have this form: the evaluation batch of each fold, and the batch
of reduction claims in extension-opening reduction.
`independent_batch_loss_factor` in `akita-types` prices both. The
[grinding nonce specification](../../../specs/grinding-nonce-encoding.md#claim-batching-sites)
records the contract.

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
proof-byte digest for all future schedules. The native grinding grammar and
nonce encoding are documented in
[`specs/transcript-grinding.md`](../../../specs/transcript-grinding.md) and
[`specs/grinding-nonce-encoding.md`](../../../specs/grinding-nonce-encoding.md).

## Code map

- `crates/akita-config/src/transcript_binding.rs` constructs and binds the
  shared descriptor and grinding plan.
- `crates/akita-types/src/instance_descriptor/mod.rs` owns descriptor fields,
  canonical serialization, and version validation.
- `crates/akita-transcript/src/native.rs` owns native state construction,
  context framing, canonical atom codecs, bounded bytes, and EOF-compatible
  proof transport.
- `crates/akita-types/src/transcript_grinding/plan.rs` defines the ordered
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
