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

Production uses Spongefish v0.8.0 prover and verifier states through Jolt's
`jolt-transcript` crate. The selected sponge is BLAKE2b or Keccak. Each state
starts from three inputs:

1. A session identifier derived with SHAKE128 from Akita's protocol tag.
   The tag names the protocol version (currently 9), the sponge, the field
   challenge codec, and the proof-of-work encoding. Changing any of them
   changes every challenge.
2. The caller's session bytes, with a `u64` little-endian length prefix.
3. The canonical instance descriptor bytes, also length-prefixed.

The length prefixes keep the session and the descriptor apart: the session
`ab` with descriptor `c` differs from session `a` with descriptor `bc`. Akita
keeps a known-answer vector for each sponge.

Every logical message or challenge has a fixed public diagnostic context
record. Proof values use prover emission and verifier receipt, derived
or public values use public messages, and verifier challenges use
verifier messages. Field atoms are canonical. The one variable-size terminal
payload carries a checked length atom before its body.

### Field challenges

A field challenge message draws all of its elements from one squeeze. For a
prime field with a $b$-bit modulus $p$, each candidate is $\lceil b/8\rceil$
little-endian bytes masked to $b$ bits. A candidate below $p$ is accepted; any
other is skipped. An extension element takes its coordinates from consecutive
accepted candidates. Accepted values are exactly uniform.

The squeeze has a fixed length that depends only on the field and the number
of elements. It holds enough spare candidates that running out has
probability at most $2^{-161}$. A message that runs out is an error for both
parties, so the verifier rejects. The distance from uniform for the whole
message is therefore at most $2^{-161}$, and the soundness argument charges
at most $2^{-160}$ per message. Drawing several challenges in one message
shares that budget rather than paying it per element.

The codec covers the fields of every production configuration: the
128-bit prime field, the degree-2 tower over the 64-bit prime, and the
degree-4 tower over the 32-bit prime. `ChallengeCodec` is a sealed trait, so
a configuration whose field has no codec does not compile.

### Rejecting a proof

A verifier failure poisons the Spongefish state. This covers malformed proof
bytes, failed checks, and errors such as an oversized group record that the
protocol code routes through `ProofChannel::guard`. Every later read and fold
root draw fails, and the final end-of-proof check fails, even if a caller
ignores the error that reported the first failure. Acceptance requires that end-of-proof check: the
whole proof is consumed, nothing is left over, and the state is not
poisoned.

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
Akita draws them in claim-major order: pair `claim * num_live_blocks + block`.
Each commitment group gets one squeezed message, and the message decodes
directly into that group's challenges.

### Order within one fold

Each fold runs these steps, in order:

1. The prover sends the fold-response nonce, and both sides absorb it.
2. The fold proof of work. When its bit count `g` is positive, the prover runs
   the sponge's grinding search and the verifier checks it. When `g` is zero
   nothing happens.
3. For each commitment group, in canonical order: the transcript absorbs the
   public draw context, then squeezes exactly the message length for that
   group.

No prover message lies between the proof of work and the last group message,
so the work covers every group message of the fold.

The public draw context names the whole challenge family:

- group index, number of live blocks, and number of claims;
- total number of challenges;
- challenge ring dimension;
- counts of coefficients at magnitude 1 and magnitude 2;
- the coefficient-packing method domain and challenge-subring dimension, when
  coefficient packing is selected; and
- the operator-norm rejection policy, when the selected L2 route requires it.

Coefficient packing adds a distinct method domain, so the same transcript
state cannot reinterpret a message under the two opening methods. The nonce is
not repeated in the context, because the transcript state already holds it.

### Decoding one challenge

Suppose the challenge ring has dimension $d$ and a challenge has $a$
coefficients of magnitude 1 and $b$ of magnitude 2, with $w = a + b$. The set
of all such vectors, the shell $S$, has

$$
|S| = \binom{d}{a} \binom{d-a}{b} 2^w
$$

elements. Every challenge of a group occupies $K$ consecutive bits of the
message, read as a little-endian bit stream. The decoder splits those bits
into two fields:

- the low $w$ bits are the signs. The output lists the magnitude-1 positions
  in ascending order, then the magnitude-2 positions in ascending order. Bit
  $t$, least significant first, is the sign of output coefficient $t$.
- the next $K - w$ bits are a rank $r$. Write
  $r = r_a + \binom{d}{a} r_b$ with $r_a < \binom{d}{a}$. The value $r_a$
  names the magnitude-1 positions as a subset of `0..d`, and $r_b$ names the
  magnitude-2 positions as a subset of the $d - a$ remaining positions.

A rank names a subset by colexicographic order: for $k$ from the subset size
down to 1, take the largest $c$ with $\binom{c}{k} \le r$, emit position $c$,
and subtract $\binom{c}{k}$ from $r$. Every binomial coefficient the decoder
needs fits in 128 bits, and so does every field it reads. The map from
$[0, |S|)$ to $S$ is a bijection.

**Families without a norm filter.** $K = \lfloor \log_2 |S| \rfloor$. Every
$K$-bit value is accepted, so the challenge set is the first $2^K$ elements of
the shell in this order. A subset of the shell keeps everything the fold
argument uses: the difference of two distinct challenges is still invertible,
and every challenge has the same norms. The message holds exactly one $K$-bit
slot per challenge.

**Families with operator-norm rejection.** An L2 fold may require a challenge
whose negacyclic convolution operator norm is below a scheduled threshold. The
certificate for that threshold counts accepted elements of the whole shell, so
these families decode over the whole shell: $K = \lceil \log_2 |S| \rceil$.
A candidate whose rank is at least $\binom{d}{a}\binom{d-a}{b}$ is rejected;
an in-range candidate is accepted when the certified predicate accepts it.
The first accepted candidates, in message order, are the challenges.

The message holds a fixed pool of candidates, sized so that an honest message
runs out with negligible probability. If a message does run out, the
verifier rejects and the honest prover tries the next fold-response nonce.
Running out depends only on which candidates pass, never on the values of the
accepted ones, so conditioned on not running out the challenges are
independent and exactly uniform on the accepted set. The pool size therefore
affects only completeness. Coefficient-packing folds use the L-infinity
security route and reject an operator-norm policy.

### Why the fold has a proof of work

Every Akita challenge message is priced so that one hash call buys at most
about $2^{-128}$ chance of breaking soundness. Fold extraction fails at a
fixed transcript prefix with probability at most $W / 2^K$, where $W$ is the
number of pairs in the fold and $2^K$ is the size of the smallest challenge
set among its groups. With $W$ in the thousands at the first fold, one message
would miss the target by up to 11 bits. The proof of work restores it: the
plan chooses the least $g$ with $W \cdot 2^{128} \le 2^K \cdot 2^g$, so a
fresh fold message costs about $2^g$ hash calls. The prover pays this once per
fold-response attempt; the verifier pays one check per fold.

Operator-norm families price $K$ as 128, the size their certificate proves.

Implementation:

- `crates/akita-challenges/src/fold_draw.rs` binds the group context and owns
  the one squeeze per group.
- `crates/akita-challenges/src/sampler/stream.rs` defines message lengths,
  pool sizes and the decoder.
- `crates/akita-challenges/src/sampler/op_norm.rs` checks the certified
  operator-norm predicate.
- `crates/akita-params/src/transcript_grinding.rs` prices the fold proof of
  work as its own run kind, `FoldChallengeWork`.

## AkitaInstanceDescriptor

Before replay, both parties construct an `AkitaInstanceDescriptor` from the
validated public configuration. The shared `transcript_instance_descriptor`
helper builds it, and its canonical bytes become the instance input of the
Spongefish state.

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
order and must be exhausted when EOF is checked.

The planner accepts only complete schedules whose expanded grinding query
count is less than `u32::MAX`. If the objective-best candidate exceeds that
capacity, planning treats it as an infeasible path and continues among schedules
admitted by the planner's existing bounded candidate-generation policies.
`UnsupportedSchedule` means that no feasible schedule was found within this
domain; it does not prove that no mathematically valid schedule exists among
layouts discarded by those policies.

Proof-of-work and fold-response nonces use distinct message kinds and
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

At a protected query with grinding target $g>0$, the prover searches for the
least nonce below $2^{g+7}$ that passes. Each attempt absorbs the canonical
nonce, one byte per prover message, then squeezes $\lceil g/8\rceil$ check
bytes. The attempt passes when the low $g$ bits of those bytes, read little
endian, are zero. Diagnostic metadata records the grinding site without
changing the sponge. The search and check are Jolt's `grind` and
`check_grind`; the largest target either accepts is 25 bits.

The verifier repeats the check. Only after it passes does replay draw the
protocol challenge, in one message, from the advanced transcript. The check
bytes are not reused as the challenge. Each protected site grinds once, even
when its message holds several challenges. A zero-bit target consumes no
proof bits and leaves the transcript unchanged at that site.

The additional seven nonce bits provide room for honest search beyond the
expected $2^g$ attempts. Storage is self-delimiting and canonical; the semantic
width is still checked from the public plan. Schedule selection adds the
per-message maxima, `ceil(semantic_nonce_width / 7)`. This deterministic
cost is not the realized LEB128 wire size. The verifier safety bound is derived
separately from the complete grammar.

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
response bounds. This is separate from the proof-of-work check. The
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

The protocol uses a descriptor-bound positional grammar. Context records
capture semantic sites and widths for logging diagnostics without adding
production hashing work. Tests cover prover/verifier vectors, tampering,
truncation, statement/session binding, and EOF; they do not freeze one
proof-byte digest for all future schedules. The grinding grammar and
nonce encoding are documented in
[`specs/transcript-grinding.md`](../../../specs/transcript-grinding.md) and
[`specs/grinding-nonce-encoding.md`](../../../specs/grinding-nonce-encoding.md).

## Code map

- `crates/akita-config/src/transcript_binding.rs` constructs and binds the
  shared descriptor and grinding plan.
- `crates/akita-types/src/instance_descriptor/mod.rs` owns descriptor fields,
  canonical serialization, and version validation.
- `crates/akita-transcript/src/proof_stream.rs` owns state construction,
  context framing, canonical atom codecs, bounded bytes, and EOF-compatible
  proof transport. Field challenges and proof of work come from
  `jolt-transcript`; `crates/akita-transcript/src/grinding.rs` records the
  parts of that proof-of-work format the grinding plan binds.
- `crates/akita-params/src/transcript_grinding/plan.rs` defines the ordered
  plan; `crates/akita-types/src/transcript_grinding/replay.rs` couples
  nonce transport, proof-of-work checks, challenges, and plan progress.
- `crates/akita-challenges/src/sampler/stream.rs` decodes each group's sparse
  challenge message.
- `crates/akita-verifier/src/fold/mod.rs` and
  `crates/akita-verifier/src/fold/terminal.rs` enforce fold and
  terminal replay order.
- `crates/akita-pcs/tests/transcript_hardening.rs` tests ordering and
  prover/verifier agreement; `crates/akita-pcs/tests/fold_linf.rs` covers
  fold-response nonce behavior and proof roundtrips.

The detailed grinding contract is recorded in
[the transcript grinding specification](https://github.com/LayerZero-Labs/akita/blob/main/specs/transcript-grinding.md).
