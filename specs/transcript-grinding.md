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

Akita applies bounded transcript proof-of-work before Fiat--Shamir queries whose
algebraic loss makes a bad challenge easier to find than the exact 128-bit
per-query target permits. It also performs bounded fold-response
search so an honest folded witness satisfies the scheduled representation and
norm bounds. These are distinct mechanisms with one public, schedule-derived
`GrindingPlan` and one plan cursor.

All nonzero grinding values are inline native Spongefish proof messages. A
proof-of-work nonce and a fold-response nonce each use the same canonical
unsigned LEB128 codec under distinct context kinds. There is no nonce prefix,
bit-packed stream, proof shape, or separate replay transcript. Zero-bit
proof-of-work sites emit no nonce and make no grinding-specific state
transition. Native receipt, absorption, challenge extraction, and EOF checking
are authoritative.

The sparse fold sampler derives every claim-major block coordinate from an
indexed SHAKE256 query. This preserves the configured coordinate law while
exposing coordinatewise forks needed by a CWSS extraction argument. This is
a sampler property; the fold extraction and indexed-address soundness
argument appear below.

## Security model

For a challenge site with conditional bad fraction at most `L / |E|`, the plan
assigns the least nonnegative integer `g` satisfying

```text
L * 2^128 <= |E| * 2^g.
```

Here `|E| = p^e` is the exact cardinality of the extension field. All
comparisons use integer arithmetic. In the supported field towers whose
cardinality is just below `2^128`, a power-of-two loss needs one more bit
than its binary logarithm. The
modulus bit width alone is not a challenge-set cardinality. A successful nonce requires a
separate 32-byte predicate whose first `g` low-order bits are zero. The prover
searches at most `2^(g+7)` candidates. Production policy rejects `g > 25`, so
the nonce width `g+7` always fits `u32`. Honest exhaustion is at most
`(1 - 2^-g)^(2^(g+7)) <= exp(-128)`.

The classical random-oracle bad-event bound is

```text
sum_i q_i * 2^-g_i * L_i / |E_i|,
```

for queried candidates, under the conditional bad-set premise used by the
corresponding algebraic check. Fresh checks completed only by the final
verifier need their own terms. This local search bound is not by itself an
end-to-end knowledge-extraction theorem. `q_i` includes adversarial candidate
queries, including fold-response trials. Grinding does not add entropy, prove uniqueness, or establish a QROM
claim. Accepting any satisfying in-range nonce is sound; the verifier MUST NOT
require the prover's first solution.

The predicate and protected challenge are distinct random-oracle queries. The
predicate transition absorbs the candidate nonce and squeezes 32 bytes; after
the accepted nonce is committed to the live state, the protected challenge is
drawn separately. The predicate bytes MUST NOT be reused as the protocol
challenge. The versioned protocol identifier and descriptor bind the positional
grammar; context records are diagnostics and are not absorbed.

Native field challenges use exact canonical rejection sampling. Each attempt
squeezes the field's canonical byte width, clears unused high bits, and accepts
only a canonical representative. Consequently there is no modular-reduction
bias or statistical-distance budget. The admitted native codec supports fields
up to 64 bytes; production fields use 4, 8, or 16 bytes.

## Public plan

`derive_transcript_grinding_plan_from_public_shape` is the canonical plan
builder. Its inputs are the trusted fold schedule, validated opening layout,
field tower, and protocol policy. The descriptor binds a digest of the complete
plan and its policy revision before any proof message or dependent challenge.

Each plan entry contains:

- a canonical `GrindingSite`;
- the query kind (`ProofOfWork`, `FoldResponse`, or `FoldChallengeGroup`);
- the public loss factor and derived target for proof-of-work;
- the nonce width, if any; and
- the multiplicity used for query accounting.

The plan uses checked arithmetic. Its expanded query count is a structural
count for one replay, not a bound on adversarial oracle work. A fold group
contributes one root plus its claim-major coordinate count; the response entry
contributes separately as semantic bookkeeping. Absorbing its nonce is not a
separate oracle squeeze. Failed previews, alternative prefixes, and repeated
proof attempts are not enumerated by this count. It MUST be less than
`u32::MAX`. Site fields reject `u32::MAX`, and Rust enum layout, `usize`, debug
text, source lines, and diagnostic labels MUST NOT enter canonical site bytes.

The prover and verifier each own a monotone plan cursor. Every live site MUST
match the next entry exactly. Optional protocol branches come only from the
validated public schedule. Success requires complete cursor consumption; an
omitted, duplicated, reordered, or unexpected site rejects.

## Native proof-of-work transition

For every proof-of-work entry with `g > 0`:

1. Record diagnostic metadata for the canonical plan site, target, nonce width,
   and `GrindingNonce` kind when transcript logging is enabled.
2. For each candidate in `[0, 2^(g+7))`, clone only the public duplex state,
   absorb the canonical native nonce, and squeeze 32 predicate bytes.
3. Select a candidate exactly when the first `g` bits, read low bit first, are
   zero. Exhaustion returns an error.
4. Commit the winner once with native `prover_message`. The verifier receives
   the nonce, range-checks it against `g+7`, reproduces the predicate, and
   rejects a failed predicate.
5. Record the protected challenge's diagnostic site and draw the challenge.

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
2. In canonical group order, absorb each group's public sparse-draw payload and
   squeeze its root. Diagnostic context metadata records the expected sequence
   without changing the production sponge.
3. Derive every indexed sparse coordinate, compute the folded response, and
   accept only if all scheduled representation and norm bounds hold.

The prover commits the accepted nonce once, then repeats the same group sequence
on the live state. The verifier receives and range-checks the nonce, reproduces
all roots in the same order, and performs the same response checks. The nonce
is not repeated inside individual group payloads because the shared native
state already binds it.

Fold-response search is honest-prover rejection sampling. It does not repair a
small Fiat--Shamir challenge space and does not add 12 bits of soundness. Every
candidate remains an adversarial random-oracle query in security accounting.

## Indexed sparse challenges

The implementation supplies separately indexed coordinate streams and adds no
fold security proof-of-work. Support size alone does not justify that choice.
The fold extraction and indexed-address argument below prove a conditional
loss of at most one accepted value per selected coordinate address, then
charge every adversarial touch and verifier completion. This is the basis for
the zero-work 128-bit per-address rate.

For one group root and zero-based claim-major block coordinate `i`, the sampler
uses exactly

```text
SHAKE256(group_root[32] || little_endian_u64(i))
```

as a fresh XOF input. The fixed input is 40 bytes. Position sampling uses
unbiased rejection; sign and magnitude extraction follows the configured
signed-sparse or operator-rejected law. Operator-norm rejection and all
configured support bounds remain mandatory.

The coordinate count is the checked product
`num_claims * num_live_blocks`. Reprogramming coordinate `i` changes only that
coordinate. Group roots, coordinate order, and distribution parameters are
bound by the public schedule and the versioned positional grammar.

### Geometry and the sampling denominator

For fold level `j` (including the terminal) and ordered commitment group `g`,
write `W[j,g] = num_claims[j,g] * num_live_blocks[j,g]`. Coordinates are pairs
`(claim, block)`, with index `claim * num_live_blocks + block`. They are sparse
ring elements, not individual polynomial coefficients. Groups can have
different dimensions and supports. Root opening layout and recursive group
layout determine these counts; neither one group nor one coordinate per fold
can be assumed. The current terminal has one evaluation-trace group and one
claim, so its coordinate count is its number of live blocks.

The two current opening methods are `EvaluationTrace` and
`SubringCoefficientPacking`. Evaluation trace uses the scheduled challenge
ring and can use coefficient-Linf or selective-L2 response security. Packing
samples in its bound challenge subring and requires the Linf route. It also
has a separate packed consistency polynomial check, with degree at most
`2s - 1` for challenge-subring dimension `s`. That field-check loss is not a
sparse-coordinate loss and is not removed by indexing.

The nonterminal prover admits a response only when its coefficients fit the
balanced interval represented by the scheduled response digit count and base.
Selective L2 additionally checks the scheduled squared norm and currently
requires one scalar group. The verifier authenticates these predicates through
the range and physical-L2 relation checks; it does not trust the prover's
search. The terminal uses centered response coefficients with a scheduled
Linf cap, optional L2 cap, and bounded Golomb-Rice encoding, then checks the
direct A and consistency relations. These bounds constrain accepted responses,
not the norm of an extracted source after division by a challenge difference.

Let `C[j,g]` be the exact output support after fixed, witness-independent
sampler filters. A signed shell in dimension `d` with `a` magnitude-one and
`b` magnitude-two coefficients has cardinality

```text
|C_raw| = binom(d, a) * binom(d-a, b) * 2^(a+b).
```

The production unfiltered ladder is:

| Dimension | `(a,b)` | Certified integer support-bit floor |
|---|---|---|
| 64 | (31,10) | 128 |
| 128 | (31,0) | 129 |
| 256 | (23,0) | 131 |
| 512 | (19,0) | 132 |
| 1024 | (16,0) | 131 |
| 2048 | (14,0) | 131 |

Selective L2 instead uses `(31,11)` at D64 with operator threshold 18, or
`(31,0)` at D128 with threshold 13. Its denominator is the family accepted by
the exact fixed-point predicate, not the raw shell. The support certificates
and runtime containment checks retain a lower bound of at least `2^128` for
these accepted families. The exact rational certificate checks are described
in [operator support certification](../scripts/operator_norm/README.md), with
reported lower bounds of 128.062439 bits at D64 and 128.563317 bits at D128.
The indexed-address theorem below turns this support bound into a
per-address fold error bound.

In the ideal independent-stream model, unbiased position/sign sampling is
uniform on the shell. Trying at most 4096 shell samples against a fixed
operator predicate is uniform on its accepted family conditional on sampler
success: every accepted element has the same first-success probability.
Sampler exhaustion rejects. This inner fixed-filter search is distinct from
the outer 4096-candidate response-nonce search. Conditioning on a response
fitting its cap need not preserve the product distribution: response admission
depends on the source and all fold challenges together. It MUST NOT be treated
as another fixed support certificate.

### Fold extraction from accepting children

The fold claim has two distinct parts: deterministic extraction from a
sufficient accepting transcript tree, and a random-oracle bound for obtaining
that tree. The following argument applies to every admitted group geometry
and to both opening methods.

Fix one fold, its prechallenge commitments, shared response nonce, group
roots, and all coordinates except one pair (claim, block). A coordinate-wise
binary family has a central accepting child and one accepting alternative for
each coordinate, with the pair differing only at that coordinate. Each child
has its own complete accepting subtree for the later ring, range, norm,
relation, and recursive checks. Later challenges and prover messages need not
agree between children. The terminal authenticates its outgoing source with
its direct A and evaluation-trace checks; working backward, a successor's
authenticated opening supplies the descendant certificate for its predecessor.

Run the strong checks under each child. They recover exact native-ring fold
relations and the verifier-certified response bounds. If two branches open a
common commitment path differently, commitment binding yields a scheduled
matrix collision. Otherwise, for one changed coordinate with unit difference
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

### Exact conditional coordinate sampling

Let C[j,g] denote the accepted family after fixed, witness-independent
filters. The indexed XOF streams are independent in the ideal-oracle model.
For an unfiltered signed shell, unbiased integer rejection makes each element
of C[j,g] equally likely. For a fixed operator predicate, let a be the
single-try acceptance probability and L be the bounded attempt count.
Every accepted element has the same first-success probability, so the
output is uniform on C[j,g] conditional on sampler success.

The tree sampler needs more than a uniformly decoded sparse value: it must
program the complete ideal XOF tape conditional on that value. Sample a raw
tape until its bounded filter succeeds, replace its first accepted shell
value with the prescribed value, and sample that shell value's bit encoding
conditional on its positions, signs, and magnitudes. Keep the rejected
prefix and unused suffix from the conditional tape. The prescribed first
success was uniform and independent of its position, so this procedure is
exact. Its expected shell work is at most L/[1-(1-a)^L]; inner unbiased
integer rejection has expected polynomial work. The unfiltered case has
L=a=1. This programs an ideal stream, not a preimage of a concrete SHAKE
seed. Failed fixed-filter draws reject; they do not change the accepted
support denominator.

The outer response-nonce search is different. Its admission predicate
depends on the source and the complete challenge vector. It is never
conditioned into C[j,g]. Every original-prover nonce trial and every
coordinate stream it touches is charged as an oracle query.

### Indexed-address fold bound

For a classical random-oracle adversary, let a denote one complete indexed
coordinate-stream address, identified by its group root and claim-major
index. Let C(a)=C[j,g] be its accepted support, q_a indicate that the
original adversary touched that address, and r_a indicate that final
verification completes it without such a touch. Set

$$
Q_{\rm coord}=\sum_a q_a,\qquad
R_{\rm coord}=\sum_a r_a\le\sum_{j,g}W[j,g],\qquad
C_{\min}=\min_{j,g}|C[j,g]|.
$$

**Indexed fold bound (classical ROM).** With the accepting-tree extraction
above and exact conditional tapes, the fold family's contribution to
failure of the matching-input tree sampler is at most

$$
\mathbb E\!\left[\sum_a\frac{q_a+r_a}{|C(a)|}\right]
\le \frac{Q_{\rm coord}+R_{\rm coord}}{C_{\min}}.
$$

Here and below a compressed group-root collision is a separate binding
event. The bound counts addresses, not complete fold vectors.

To see why adaptive root and nonce choices do not invalidate the bound, fix
the original adversary's coins, all oracle answers except the answer at
address a, and the suffix sampler's random tape. Keep the address and
context selected by the *original* run even if suffix extraction fails.
The binary matching-input game loses at most one accepted challenge value
out of C(a) on a nonempty selected-address fiber; its other answers and
later accepting subtrees are held by the conditional sampler. Let H_a mean
that some answer at a causes the original run to select a. This indicator
depends only on the fixed other answers. At any fixed oracle table, H_a can
be nonempty only if the original run queried a or its selected final proof
requires a during verification. If neither happens, changing only a
cannot change that run or its selection. Thus H_a is pointwise bounded by
q_a+r_a. Sum the binary fiber charges and average over the fixed answers.
The same original selected run is retained through successive tree stages,
so these address-weighted inequalities telescope; queries made only by
rewound extractor executions affect running time, not Q_coord.

One complete candidate at level j touches sum_g W[j,g] coordinate streams,
and V fresh complete candidates touch V times that many; a partial
evaluation counts the streams it actually touches. The fold remains one
flat coordinate-wise stage with branching factor 1+sum_g W[j,g], rather
than W[j,g] sequential binary rounds. Root queries and all response-nonce
trials also count in the adversary's *total* oracle budget. The native
expanded_query_count is only a one-replay structural count.

For comparison, charging one query per complete fold vector gives the
valid whole-fold interactive term

$$
\varepsilon_{\rm fold}^{\rm whole}
=\sum_j\sum_g\frac{W[j,g]}{|C[j,g]|}.
$$

The two conventions must keep their matching query units. Indexed
accounting moves geometry into the number of touched or verifier-completed
coordinate addresses; it does not erase geometry or permit one vector
request to count as one coordinate query.

### Work decision and complete reduction boundary

Every production accepted family has at least 2^128 elements, including
the certified operator-filtered families. Therefore each sparse address
has weight 1/|C(a)| at most 2^-128 without fold proof-of-work. This is the
proved reason for a zero-bit FoldChallengeGroup target under the protocol's
**128-bit per-address rate**. It is not an unconditional 2^-128 bound for
arbitrarily many queries.

For a complete classical-ROM tree, give each field challenge family r with
conditional numerator L_r, exact field order |E_r|, and work target g_r
the weight L_r/(|E_r| 2^g_r). Give each sparse coordinate its weight
1/|C(a)|. If every security-relevant site is in that ledger, the same
original-run support argument across stages gives

$$
\varepsilon_{\rm tree}
\le \mathbb E\!\left[\sum_a \alpha_a(q_a+r_a)\right]
\le \rho(Q+R_{\rm field}+R_{\rm coord}),\qquad
\rho=\max_a\alpha_a.
$$

Q counts original-adversary hash evaluations, including rejected work
candidates, fixed-filter requests, response-nonce trials, and each indexed
coordinate stream touched by a vector candidate. R_field and R_coord count
only previously untouched addresses completed by final verification.
Exact field pricing enforces L_r 2^128 <= |E_r| 2^g_r at the typed
proof-of-work sites; the sparse support checks enforce the same per-address
ceiling with g_fold=0. Root collisions, setup sampling distance, and the
scheduled matrix-collision advantages are separate additive terms. Distinct
logical contexts that share a 32-byte group root are a compressed-context
collision; outside that event, (root, index) identifies the selected
coordinate. In an ideal 256-bit root experiment with M distinct roots
examined, the collision probability is at most M(M-1)/2^257.

The deterministic extractor's fold branching product is
B_fold=prod_j(1+sum_g W[j,g]). Its complete tree product B also includes
the actual flat or nested factors of the other challenge stages. An explicit
expected-cost reduction has the form
T_E <= B T_0 + (B-1)(Q+1) gamma + T_tree, where gamma includes
nonmatching replays and the exact conditional samplers. Module-SIS
advantages must be evaluated at that cost. Expected *polynomial*-time
knowledge extraction additionally requires B to be polynomial in the
declared explicit instance length and security parameter. This is a
tree-size admission condition, not a soundness debit, and the current
GrindingPlan's structural u32 count does not certify it.

For any claimed aggregate target lambda with a declared budget Q_max,
the displayed rate requires the exact weighted total (and other reduction
terms) to fit 2^-lambda. Replacing that total by
(Q_max+R_field+R_coord)2^-128 is conservative. An *absolute* aggregate
128-bit bound for several completed or queried addresses cannot follow
from a per-address 128-bit rate alone; it requires additional margin.
That arithmetic qualification does not undo the indexed theorem or require
a separate 12-bit debit for the response nonce.

### Current security conclusion and evidence

The fold-specific result above resolves the old inference from support size:
the one-value loss comes from the binary matching-input tree game and
descendant-certified fold extraction, while support size supplies its
denominator. For Akita's indexed streams, each original-adversary coordinate
touch and each previously untouched final-verifier completion is counted
separately. The accepted support floor makes zero additional fold work
sufficient for the 128-bit per-address rate in the classical ROM.

This result covers both opening methods, all scheduled groups and coordinates,
and the terminal. The field challenge targets are priced with exact field
orders by the typed GrindingPlan. The plan does not itself certify a
schedule-wide extractor tree product, a public adversarial Q_max, or an
absolute aggregate 128-bit error. A concrete end-to-end claim must use the
displayed query and completion counts, the actual tree cost, root-compression
and setup terms, and the relevant Module-SIS advantages. This is an explicit
quantitative reduction boundary, not a missing fold extraction theorem.

The 12-bit fold-response search is honest-prover rejection sampling.
Every adversarial trial and its induced root and coordinate queries enter Q.
There is no separate fixed 12-bit soundness debit for work already counted
there, and the nonce range does not cap searches over other prefixes.

The implementation evidence for this result is:

| Source | What it establishes |
|---|---|
| `crates/akita-challenges/src/fold_draw.rs` | Method, group, dimensions, family, and counts bound before the group root; packing rejects operator filtering |
| `crates/akita-challenges/src/challenges.rs` and `sampler/xof.rs` in that crate | Claim-major addressing and independent indexed XOF inputs |
| `crates/akita-challenges/src/config.rs` and `sampler/mod.rs` in that crate | Production support ladder, fixed operator policies, bounded coordinate rejection, runtime certificate containment |
| `crates/akita-types/src/transcript_grinding/plan.rs` and `crates/akita-types/src/transcript_grinding.rs` | Geometry-derived runs, root-plus-coordinate multiplicity, structural count, response nonce width, and zero fold work bits |
| `crates/akita-prover/src/protocol/fold_grind.rs` | Joint response-admission search across groups |
| `crates/akita-verifier/src/protocol/core/fold/mod.rs` | Nonterminal range and physical-L2 claims tied into the recursive relation |
| `crates/akita-verifier/src/protocol/core/terminal_direct.rs` | Terminal representation/norm and direct relation checks |
| [Security model](../book/src/how/security.md) and [subring packing](subring-coefficient-packing.md) | Accepted response-space contract and separate packed relation loss |

The [Book's binding chapter](../book/src/foundations/pcs-and-binding.md)
explains CWSS and the public classical-ROM references. Distribution, replay,
and tampering tests check implementation correspondence; they do not prove
adaptive extraction or its concrete loss.

## Encoding and proof-size accounting

Each proof-of-work or fold-response site contributes the canonical unsigned
LEB128 length of its accepted nonce. Context records are diagnostic only;
public values are absorbed with `public_message`. Neither contributes proof
bytes.

Schedule selection prices every present nonce at its canonical native LEB128
maximum, `ceil(semantic_nonce_width / 7)`, and adds those per-message maxima.
This is a deterministic native-format objective, not an estimate of the
winning nonces' realized wire lengths. It MUST NOT be used as a nonce range or
security bound. The full native parser bound is derived separately because it
also includes every non-nonce message and terminal framing. Schedule artifacts
MUST be regenerated from this objective; generated files MUST NOT be copied or
hand-edited to manufacture a desired result.

Native nonce decoding MUST reject unterminated, overflowing, and redundant
unsigned LEB128 encodings without advancing the input cursor. The verifier
checks the scheduled range after receipt. Truncation, trailing argument bytes,
out-of-range nonces, wrong context/order, failed predicates, and incomplete
plans reject with `AkitaError`; verifier-reachable code MUST NOT panic or
allocate from a proof-controlled length.

## Ownership

| Component | Responsibility |
|---|---|
| Spongefish | Native state, argument bytes, nonce receipt/absorption, challenge squeeze, EOF |
| `akita-transcript` | Native positional codecs, diagnostic context records, public-state previews, predicate and bounded search primitive |
| `akita-types` | Grinding sites, policy, plan, cursor, native plan-owning adapters |
| `akita-config` | Derive and descriptor-bind the public plan |
| `akita-prover` | Fold-response candidate computation and honest bounded search |
| `akita-verifier` | Nonce ranges, predicates, response equations, plan completion |
| `akita-challenges` | Indexed sparse expansion and distribution checks |
| `akita-planner` | Query accounting and native-maximum schedule objective |

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
- Prover and verifier sparse roots agree, and the indexed SHAKE256
  implementation matches an independent 40-byte-input reference.
- Reprogramming one indexed coordinate changes only that coordinate.
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
