# Spec: Transcript grinding

| Field | Value |
|---|---|
| Author(s) | Quang Dao, Codex |
| Created | 2026-05-22 |
| Status | active |
| PR | [#448](https://github.com/LayerZero-Labs/akita/pull/448) |
| Supersedes | Unmerged transcript grinding design at `5057456` |
| Superseded-by | |
| Book-chapter | how/transcript.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Summary

Akita prices the local bad-set bound of each typed field challenge to a
128-bit per-query rate by applying bounded transcript proof of work where
needed. Each fold's sparse challenge messages receive the same rate from
one fold proof of work priced against the smallest challenge set. Bounded fold-response search lets an honest folded witness
satisfy the scheduled representation and norm bounds. These mechanisms
share one public, schedule-derived `GrindingPlan` and one plan cursor.

All nonzero grinding values are inline Spongefish proof messages. A
proof-of-work nonce and a fold-response nonce each use the same canonical
unsigned LEB128 codec under distinct context kinds. There is no nonce prefix,
bit-packed stream, proof shape, or separate replay transcript. Zero-bit
proof-of-work sites emit no nonce and make no grinding-specific state
transition. Receipt, absorption, challenge extraction, and EOF checking
are authoritative.

Each commitment group draws its sparse fold challenges from one squeezed
message, decoded directly into challenges. The extraction argument and its
per-fold proof-of-work price appear below.

## Security model

For a challenge site with conditional bad fraction at most `L / |E|`, the plan
assigns the least nonnegative integer `g` satisfying

```text
L * 2^128 <= |E| * 2^g.
```

Here `|E| = p^e` is the exact cardinality of the extension field. All
comparisons use exact integer arithmetic in
`jolt_transcript::grind_bits_for_loss`, called with 128 security bits. In the
supported field towers whose cardinality is just below `2^128`, a power-of-two
loss needs one more bit than its binary logarithm. The modulus bit width alone
is not a challenge-set cardinality. A successful nonce
makes the next `ceil(g/8)` squeezed check bytes have `g` zero low-order bits.
The prover searches at most `2^(g+7)` candidates. Production policy rejects
`g > 57`, the largest target `jolt_transcript::MAX_POW_BITS` accepts, so the
nonce width `g+7` always fits `u64`. Honest exhaustion is at most
`(1 - 2^-g)^(2^(g+7)) <= exp(-128)`.

The classical random-oracle bad-event bound is

```text
sum_i q_i * 2^-g_i * L_i / |E_i|,
```

for queried candidates, under the conditional bad-set premise used by the
corresponding algebraic check. Fresh checks completed by the final
verifier receive their own terms. The complete knowledge-extraction
calculation appears below. `q_i` includes adversarial candidate queries
from fold-response trials. The target is a classical random-oracle
bad-event rate. Accepting any satisfying in-range nonce is sound; the
verifier MUST NOT require the prover's first solution.

The proof-of-work check and the protected challenge are distinct random-oracle
queries. The check absorbs the candidate nonce and squeezes `ceil(g/8)` bytes;
the protected challenge is the next squeeze after it. The check bytes MUST NOT
be reused as the protocol challenge. The versioned protocol identifier and descriptor bind the positional
grammar; context records are diagnostics and are not absorbed.

Field challenges use Jolt's pooled rejection codec (`jolt_transcript::ChallengeCodec`).
One challenge message squeezes a fixed number of bytes that depends only on
the field and the element count. The bytes form candidates of the base
field's canonical byte width, masked to the modulus bit width; a candidate is
accepted exactly when it is a canonical representative, and extension
coordinates come from consecutive accepted candidates. The pool holds enough
spare candidates that exhaustion has probability at most `2^-161` for the
whole message; exhaustion is an error and the verifier rejects. Accepted
values are exactly uniform, so each message has total variation at most
`2^-161` from uniform, charged as `2^-160` per message in the soundness bound.
A multi-element challenge at one site is one message and shares that budget.
Production fields use 4-, 8-, or 16-byte base candidates.

## Public plan

`derive_transcript_grinding_plan_from_public_shape` is the canonical plan
builder. Its inputs are the trusted fold schedule, validated opening layout,
field tower, and protocol policy. The descriptor binds a digest of the complete
plan and its policy revision before any proof message or dependent challenge.

Each plan entry contains:

- a canonical `GrindingSite`;
- the query kind (`ProofOfWork`, `FoldResponse`, `FoldChallengeWork`, or
  `FoldChallengeGroup`);
- the public loss factor and derived target for proof-of-work;
- the nonce width, if any; and
- the multiplicity used for query accounting.

The plan uses checked arithmetic. Its expanded query count is a structural
count for one replay, not a bound on adversarial oracle work. A fold group
contributes one message plus its challenge count; the response and fold-work
entries contribute separately as semantic bookkeeping. Absorbing its nonce is not a
separate oracle squeeze. Failed previews, alternative prefixes, and repeated
proof attempts are not enumerated by this count. It MUST be less than
`u32::MAX`. Site fields reject `u32::MAX`, and Rust enum layout, `usize`, debug
text, source lines, and diagnostic labels MUST NOT enter canonical site bytes.

The prover and verifier each own a monotone plan cursor. Every live site MUST
match the next entry exactly. Optional protocol branches come only from the
validated public schedule. Success requires complete cursor consumption; an
omitted, duplicated, reordered, or unexpected site rejects.

## Proof-of-work transition

For every proof-of-work entry with `g > 0`:

1. Record diagnostic metadata for the canonical plan site, target, nonce width,
   and `GrindingNonce` kind when transcript logging is enabled.
2. For each candidate in `[0, 2^(g+7))`, in increasing order, clone only the
   public duplex state, absorb the canonical nonce one byte at a time, and
   squeeze `ceil(g/8)` check bytes.
3. Select the first candidate whose check bytes, read as a little-endian
   integer, have `g` zero low-order bits. Exhaustion returns an error.
4. Commit the winner as one single-byte prover message per nonce byte, then
   squeeze the check bytes. Steps 2--4 are `jolt_transcript::ProverTranscript::grind`.
   The verifier runs `VerifierTranscript::check_grind`: it reads the nonce
   byte by byte, rejects a noncanonical encoding or a value of `2^(g+7)` or
   more, squeezes the check bytes, and rejects a failed check. Any failure
   poisons the verifier state.
5. Record the protected challenge's diagnostic site and draw every challenge
   of the site in one message.

A zero-bit entry has nonce width zero, emits no proof bytes, and skips steps
1--4. It remains present in the semantic plan so query coverage can be audited.

Preview MUST NOT mutate the live sponge, private prover randomness, proof
output, or plan cursor. Raw `duplex_sponge_state` access is confined to the
reviewed transcript preview implementation and enforced by CI allowlists.

## Fold-response search

Every nonterminal and terminal fold has one `FoldResponse` entry with a 12-bit
domain and at most 4096 trials. One candidate is shared across all commitment
groups in that fold.

For candidate `c`:

1. Clone the current public sponge state and absorb the canonical unsigned
   LEB128 encoding of `c`.
2. Run the fold proof of work (`FoldChallengeWork`) on the clone. With `g = 0`
   this step does nothing.
3. In canonical group order, squeeze each group's challenge message. Nothing
   is absorbed between them; the descriptor and the group's position fix the
   family
   ([positional-word-aligned-transcript](positional-word-aligned-transcript.md)). Diagnostic context metadata records the
   expected sequence without changing the production sponge.
4. Decode every group's challenges, compute the folded response, and accept
   only if all scheduled representation and norm bounds hold. A message whose
   operator-norm pool runs out rejects the candidate.

Jolt's `verifier_message_oracle` evaluates only "absorb a candidate, then squeeze". A
fold-response candidate is followed by a proof of work and several message
squeezes, so `akita_transcript::FoldPreview` performs this
search on a detached copy of the public sponge state.

The prover commits the accepted nonce once, carries the preview's proof of
work to the live state without searching again, then repeats the group
sequence on the live state. The verifier receives and range-checks
the nonce, checks the proof of work, reproduces every message in the same
order, and performs the same response checks. No group absorbs anything, so
the nonce is bound once, before the proof of work.

Fold-response search is honest-prover rejection sampling. It does not repair a
small Fiat--Shamir challenge space and does not add 12 bits of soundness. Every
candidate remains an adversarial random-oracle query in security accounting.

## Sparse fold challenges

Each commitment group of a fold draws its sparse challenges from one squeezed
message. The message length depends only on the public family and challenge
count; the decoder, pool sizes and the full soundness argument are specified
in [one stream per sparse fold group](sparse-challenge-stream.md). This
section states the security contract that the grinding plan enforces.

The whole fold is one priced challenge message. Fold extraction can fail at a
fixed prefix with probability at most `W_l / 2^K_l`, where `W_l` is the total
number of `(claim, block)` pairs over the fold's groups and `2^K_l` is the
smallest guaranteed challenge-set size among them. The plan therefore inserts
one `FoldChallengeWork` entry per fold, priced like every other site:

```text
g_l = least g with W_l * 2^128 <= 2^K_l * 2^g.
```

The entry records `K_l`, and validation recomputes `g_l` from `W_l` and
`K_l`. It is the only run kind priced against a sparse set rather than the
challenge field.

### Geometry and the sampling denominator

For fold level `j` (including the terminal) and ordered commitment group `g`,
write `W[j,g] = num_claims[j,g] * num_live_blocks[j,g]`. Challenges are indexed
by pairs `(claim, block)`, at position `claim * num_live_blocks + block`. They
are sparse ring elements, not individual polynomial coefficients. Groups can
have different dimensions and supports. Root opening layout and recursive
group layout determine these counts; neither one group nor one challenge per
fold can be assumed. The current terminal has one evaluation-trace group and
one claim, so its challenge count is its number of live blocks.

The two current opening methods are `EvaluationTrace` and
`SubringCoefficientPacking`. Evaluation trace uses the scheduled challenge
ring and can use coefficient-Linf or selective-L2 response security. Packing
samples in its bound challenge subring and requires the Linf route. It also
has a separate packed consistency polynomial check, with degree at most
`2s - 1` for challenge-subring dimension `s`. That field-check loss is not a
sparse-challenge loss.

The nonterminal prover admits a response only when its coefficients fit the
balanced interval represented by the scheduled response digit count and base.
Selective L2 additionally checks the scheduled squared norm and currently
requires one scalar group. The verifier authenticates these predicates through
the range and physical-L2 relation checks; it does not trust the prover's
search. The terminal uses centered response coefficients with a scheduled
Linf cap, optional L2 cap, and bounded Golomb-Rice encoding, then checks the
direct A and consistency relations. These bounds constrain accepted responses,
not the norm of an extracted source after division by a challenge difference.

Let `C[j,g]` be the accepted challenge set. A signed shell in dimension `d`
with `a` magnitude-one and `b` magnitude-two coefficients has cardinality

```text
|S| = binom(d, a) * binom(d-a, b) * 2^(a+b).
```

Families without a norm filter decode `K = floor(log2 |S|)` bits per
challenge and accept every value, so `|C| = 2^K` exactly:

| Dimension | `(a,b)` | `K` |
|---|---|---|
| 64 | (31,10) | 128 |
| 128 | (31,0) | 129 |
| 256 | (23,0) | 131 |
| 512 | (19,0) | 132 |
| 1024 | (16,0) | 131 |
| 2048 | (14,0) | 131 |

Selective L2 instead uses `(31,11)` at D64 with operator threshold 18, or
`(31,0)` at D128 with threshold 13. Its accepted set is the family accepted by
the exact fixed-point predicate over the whole shell. The exact rational
certificate checks are described in
[operator support certification](../scripts/operator_norm/README.md), with
reported lower bounds of 128.062439 bits at D64 and 128.563317 bits at D128;
the plan prices these families at `K = 128`.

### Fold extraction from accepting children

The fold claim has two distinct parts: deterministic extraction from a
sufficient accepting transcript tree, and a random-oracle bound for obtaining
that tree. The following argument applies to every admitted group geometry
and to both opening methods.

Fix one fold, its prechallenge commitments, shared response nonce, and all
challenges except one pair (claim, block). A pair-wise binary family has a
central accepting child and one accepting alternative for each pair, with the
two differing only at that pair's challenge. Each child has its own complete
accepting subtree for the later ring, range, norm, relation, and recursive
checks. Later challenges and prover messages need not agree between children.
Groups are drawn in sequence, so changing a pair in one group also changes
every later group's message in that child; each group's fold relation involves
only its own challenges, so that change belongs to the child's own subtree.
The terminal authenticates its outgoing source with its direct A and
evaluation-trace checks; working backward, a successor's authenticated opening
supplies the descendant certificate for its predecessor.

Run the strong checks under each child. They recover exact native-ring fold
relations and the verifier-certified response bounds. If two branches open a
common commitment path differently, commitment binding yields a scheduled
matrix collision. Otherwise, for one changed pair with unit difference
delta, subtraction cancels every other fold term and gives

$$
\delta s = z^{(1)}-z^{(0)},\qquad
A s=G\widehat t,\qquad
e=\mathcal L(Gs).
$$

The first inversion takes place in the ambient fold ring; the opening
equation uses the selected evaluation-trace ring or coefficient-packing
subring. The production LS18 condition makes every nonzero difference of
accepted sparse challenges a unit, including its packing embedding.
Repeating the step extracts all incoming weak openings, or a scheduled
matrix collision. The extracted source need not be the honest canonical
digit decomposition or short after division. For the collision reduction,
cross-multiply the two unit differences against their accepted, bounded
responses before centering the resulting kernel vector. This uses the
verifier-enforced digit/range and, when selected, physical-L2 bounds; it
does not assume that unit division preserves a norm. The terminal seeds
the same backward induction. This proves the fold extraction step for an
accepting tree; the next step bounds failure to build that tree.

### Programming one challenge

In the ideal sponge model, the message squeezed at a fixed prefix is one
uniform answer. For a family without a norm filter, the message is `W`
independent uniform `K`-bit slots and the decoder is a bijection onto `C`, so
replacing one slot by the encoding of a fresh uniform element of `C` yields
exactly the distribution of a message that differs only at that pair.

For an operator-norm family, each candidate in the pool is accepted
independently with the same probability, and an accepted candidate is uniform
on the accepted family, independent of which candidates were accepted.
Replacing the chosen accepted candidate by a fresh accepted encoding keeps the
acceptance pattern, every other candidate, and the unused tail, so it is again
exact. A message whose pool runs out is rejected by the verifier and supplies
no accepting child. Running out depends only on the acceptance pattern, so
for any bad set of challenge vectors, `P[accept and bad] <= P_ideal[bad]`.

The outer response-nonce search is different. Its admission predicate
depends on the source and the complete challenge vector. It is never
conditioned into `C[j,g]`. Every original-prover nonce trial is charged as an
oracle query.

### Fold message bound

In the Fiat–Shamir security game, an adversary can query challenges for
many proof prefixes and submit one final proof. Interpret each
protocol-shaped random-oracle query as a verifier challenge at that
prefix. The extractor replays the adversary with selected answers changed to
build the accepting tree described above.

Fix a prefix right after the fold proof of work. For one pair in group `g`,
conditioned on every other pair and everything before, at most one accepted
value of that pair makes the extractor's fork at that pair fail (the
matching-input argument). The pair is uniform on `C[l,g]`, so its fork fails
with probability at most `1/|C[l,g]|`. A union over the pairs of the fold
gives at most `W_l / 2^K_l` for one prefix.

A fresh fold message costs the adversary `2^g_l` proof-of-work checks in
expectation, and the plan's choice of `g_l` gives

$$
2^{-g_l}\,\frac{W_l}{2^{K_l}} \le 2^{-128}.
$$

So each hash call buys at most `2^-128` of fold failure, the same rate as
every other challenge site. With `Q` the adversary's hash calls and `R` the
fold messages the final verifier completes itself, the fold term is at most
`(Q + R) * 2^-128`, summed with the other sites.

No single-message design can price below `W_l * 2^(128 - K_l)` calls per fold
for the honest prover, because an attacker can do what the honest prover does.
The fold proof of work sits at that bound.

### Fold work decision and total security accounting

For a complete classical-ROM tree, let `d_r` be the proven mixed-tree failure
debit at challenge address `r`, including any branches needed to extract all
claims protected by that address. If its conditional bad-set bound is
`d_r / |E_r|`, its proof-of-work target `g_r` gives weight
`d_r / (|E_r| 2^g_r)`. Fold messages are addresses of the same kind, with
debit `W_l`, set size `2^K_l` and target `g_l`. Once every challenge stage has
such a proven weight, the same original-run support argument gives

$$
\varepsilon_{\rm tree}
\le \mathbb E\!\left[\sum_a \alpha_a(q_a+r_a)\right]
\le \rho(Q+R),\qquad
\rho=\max_a\alpha_a.
$$

Here `Q` is a deterministic upper bound on distinct logical random-oracle
addresses touched by the original adversary, including proof-of-work check
queries for rejected work candidates and response-nonce trials, and `R`
bounds previously untouched addresses completed by final verification.

The typed GrindingPlan enforces `L_r 2^128 <= |E_r| 2^g_r` for its stored
*local* loss `L_r`. The complete-tree rate `rho <= 2^-128` follows when each
site has a proof that `d_r <= L_r`, or when its target is raised to price
`d_r`. Setup sampling distance and scheduled matrix-collision advantages enter
as additive terms.

The deterministic extractor's fold branching product is
`B_fold = prod_j(1 + sum_g W[j,g])`. Its complete tree product `B` also
includes the flat or nested factors of the other challenge stages. An explicit
expected-cost reduction has the form
`T_E <= B T_0 + (B-1)(Q+1) gamma + T_tree`, where `gamma` includes
nonmatching replays and the conditional samplers. Module-SIS advantages are
evaluated at that cost. Expected *polynomial*-time knowledge extraction
requires `B` to be polynomial in the declared explicit instance length and
security parameter. Certifying `B` is a separate admission check from the
plan's structural one-replay count.

For an aggregate target `lambda`, declare `Q_max` and the verifier-completion
bound, prove each full-tree debit, and require the weighted oracle total plus
the setup and matrix-collision terms to fit `2^-lambda`. If every address has
weight at most `2^-128`, the simpler upper bound is `(Q_max + R) 2^-128`
before those additive terms.

### Established result and implementation evidence

The fold result combines descendant-certified extraction with the binary
matching-input game at each fold message, priced by the fold proof of work to
the 128-bit per-call rate in the classical ROM. It covers both opening
methods, all scheduled groups and pairs, and the terminal.

The 12-bit fold-response search serves honest-prover response admission.
Adversarial trials enter `Q`; this query charge accounts for the nonce
freedom.

| Source | What it establishes |
|---|---|
| `crates/akita-challenges/src/fold_draw.rs` | Method, group, dimensions, family, and counts bound before each group message; packing rejects operator filtering |
| `crates/akita-challenges/src/sampler/stream.rs` | Slot width, pool size, message length, and the bijective colex decoder |
| `crates/akita-challenges/src/config.rs` and `sampler/mod.rs` in that crate | Production support ladder, fixed operator policies, runtime certificate containment |
| `crates/akita-params/src/transcript_grinding/plan.rs` and `crates/akita-params/src/transcript_grinding.rs` | Geometry-derived runs, the per-fold `FoldChallengeWork` price, structural count, response nonce width |
| `crates/akita-prover/src/protocol/fold_grind.rs` | Joint response-admission search across groups |
| `crates/akita-verifier/src/stages/stage1.rs` | Nonterminal range and physical-L2 claims tied into the recursive relation |
| `crates/akita-verifier/src/terminal/direct.rs` | Terminal representation/norm and direct relation checks |
| [Security model](../book/src/how/security.md) and [subring packing](subring-coefficient-packing.md) | Accepted response-space contract and separate packed relation loss |

The [Book's binding chapter](../book/src/foundations/pcs-and-binding.md)
explains the pair forks, the fold proof of work, and the security target for
readers.

## Encoding and proof-size accounting

Each proof-of-work or fold-response site contributes the canonical unsigned
LEB128 length of its accepted nonce. Context records are diagnostic only;
public values are absorbed with `public_message`. Neither contributes proof
bytes.

Schedule selection prices every present nonce at its canonical LEB128
maximum, `ceil(semantic_nonce_width / 7)`, and adds those per-message maxima.
This is a deterministic format objective, not an estimate of the
winning nonces' realized wire lengths. It MUST NOT be used as a nonce range or
security bound. The full parser bound is derived separately because it
also includes every non-nonce message and terminal framing. Schedule artifacts
MUST be regenerated from this objective; generated files MUST NOT be copied or
hand-edited to manufacture a desired result.

Nonce decoding MUST reject unterminated, overflowing, and redundant
unsigned LEB128 encodings and poison the verifier state, so every later read
and the EOF check fail. The verifier checks the scheduled range after receipt.
Truncation, trailing argument bytes, out-of-range nonces, wrong context/order,
failed proof-of-work checks, and incomplete
plans reject with `AkitaError`; verifier-reachable code MUST NOT panic or
allocate from a proof-controlled length.

## Ownership

| Component | Responsibility |
|---|---|
| Spongefish | State, argument bytes, nonce receipt/absorption, poisoning, EOF |
| `jolt-transcript` | Field challenge codecs, proof-of-work pricing, search and check |
| `akita-transcript` | Session tag, positional codecs, diagnostic context records, fold-response preview |
| `akita-types` | Grinding sites, policy, plan, cursor, plan-owning adapters |
| `akita-config` | Derive and descriptor-bind the public plan |
| `akita-prover` | Fold-response candidate computation and honest bounded search |
| `akita-verifier` | Nonce ranges, proof-of-work checks, response equations, plan completion |
| `akita-challenges` | Sparse message decoding and distribution checks |
| `akita-planner` | Query accounting and maximum schedule objective |

There is one production proof path. A separate packed nonce codec, nonce prefix,
structured proof replay, or alternate verifier is prohibited.

## Required tests and checks

- Exact site encoding and plan digest vectors cover every site discriminator.
- Zero-bit, nonzero, maximum-target, exhaustion, incomplete-plan, out-of-range,
  truncation, mutation, and trailing-byte cases reject correctly; diagnostic
  site sequences agree between prover and verifier.
- Preview output matches live prover and verifier replay for both Blake2b and
  Keccak, multiple groups, and multiple candidate counts.
- Unsuccessful previews leave live state and proof output unchanged.
- Prover and verifier sparse messages agree, and the decoder inverts an
  independent colexicographic rank reference.
- Changing one challenge slot changes only that challenge.
- Signed-sparse and operator-rejected marginal distribution, support, unit
  difference, and norm-policy tests remain green for both opening methods.
- Runtime plan completion and Spongefish EOF are both necessary for acceptance.
- CI rejects unchecked proof decoding, raw-state access outside the reviewed
  allowlist, and bypass state constructors in protocol code.
- Planner artifacts are regenerated and checked whenever policy, byte cost, or
  plan identity changes.

## References

- [Grinding nonce encoding explainer](grinding-nonce-encoding.md)
- [Transcript implementation](../book/src/how/transcript.md)
- [PCS binding and query accounting](../book/src/foundations/pcs-and-binding.md)
- [Verifier contract](../docs/verifier-contract.md)
- [Subring coefficient packing](subring-coefficient-packing.md)
