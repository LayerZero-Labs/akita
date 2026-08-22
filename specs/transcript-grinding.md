# Spec: Transcript Grinding

| Field | Value |
|---|---|
| Author(s) | Quang Dao, Codex |
| Created | 2026-05-22 |
| Status | active |
| PR | [#417](https://github.com/LayerZero-Labs/akita/pull/417) |
| Supersedes | Unmerged transcript grinding design at `5057456` |
| Superseded-by | |
| Book-chapter | |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Summary

Akita will add transcript proof-of-work before Fiat-Shamir challenge queries
whose algebraic loss makes a bad challenge easier to find than Akita's nominal
128-bit challenge-capacity convention allows. Each protected query has a public
zero-bit target. The prover supplies a nonce that makes that many low bits of a
separate transcript predicate zero. The verifier checks the same predicate
before drawing the protocol challenge. For a site with bad fraction at most
`L / |E|`, the policy makes one accepted bad candidate cost about
`2^g |E| / L` classical random-oracle trials. It does not change the challenge
distribution.

All transcript search nonces move into one proof-level bit stream. The
stream also replaces the former fixed `u32` fields used by fold-response
rejection sampling, but the two mechanisms remain distinct. Proof-of-work pays
for algebraic challenge loss. Fold-response rejection searches for an honest
response that fits a scheduled norm bound. The existing sparse fold challenge
families already provide at least 128 bits of challenge support, so they do not
receive an additional proof-of-work check.

The same pull request also repairs the sparse fold challenge interface used by
both `EvaluationTrace` and `SubringCoefficientPacking`. Before Slice 1, the code
squeezed one transcript seed per commitment group and expanded the complete
claim-major block vector from a shared XOF stream. That did not expose the
coordinatewise forks required by the current CWSS extraction argument. The new
sampler keeps the group root seed and each coordinate's challenge law, but
derives every claim-major block coordinate through its own fixed-width indexed
oracle query. The sampler change itself adds no proof bytes or
transcript sponge squeezes.

The nonce stream is decoded in a canonical order derived from the public
schedule, opening layout, field tower, protocol configuration, and one
descriptor-bound grinding policy. A nonce with `g` proof-of-work bits uses
exactly `g + 7` wire bits. The current fold-response nonce uses exactly 12 wire
bits. Nonces are packed without byte alignment, so small queries do not pay for
a fixed `u16` or `u32` slot.

## Current state

Slices 1 through 6 are implemented on PR #417. Akita now derives sparse fold
challenge coordinates from independent indexed SHAKE256 queries, derives one
public `GrindingPlan`, binds its digest in `TranscriptGrindingBinding`, and
stores fold-response values in one packed
proof-level nonce stream. Plan-owning prover and verifier transcript adapters
now consume that stream at each live query boundary. The transcript crate
implements the exact 32-byte proof-of-work predicate and prover preview
transition. A final cursor check rejects an omitted, duplicated, reordered, or
trailing query.

The existing fold-response search remains in
`crates/akita-prover/src/protocol/fold_grind.rs`. For each fold, the prover
tries sequential values until the sparse challenge produces a folded response
accepted by the scheduled representation and norm checks. The verifier reads
exactly 12 bits, expands the value to `u32`, absorbs its four-byte numeric
encoding into each sparse challenge context, and checks the resulting response.
The 12-bit decoder makes values at or above the exclusive bound of 4096
unrepresentable. The plan digest binds the attempt cap and packed width.

That mechanism is honest-prover rejection sampling. It does not repair a small
Fiat-Shamir challenge space and it does not add 12 bits of soundness. Every
adversarial nonce trial is already another random-oracle query. The current
accounting is described in
[`book/src/foundations/pcs-and-binding.md`](../book/src/foundations/pcs-and-binding.md).

Akita now applies transcript proof-of-work to the live protocol queries priced
by the public plan. Sumcheck rounds, ring-switch checks, multilinear points,
and power batching draw from a nominal 128-bit extension challenge field, but
some checks have a bad set larger than one field element. A degree `d`
polynomial check, for example, has conditional error at most `d / |E|`. The
feature charges about `2^ceil(log2(d))` work before that challenge, so finding
an accepted bad challenge again costs about `2^128` classical random-oracle
trials in the stated bounded online-query model.

Before Slice 1, the sparse fold path had a separate security mismatch. For each commitment
group, `FoldDraw::draw_folding_challenges_with_rejection` absorbs the group
context and the shared fold-response nonce, squeezes one 32-byte seed, and
expands `num_claims * num_live_blocks` sparse challenges. The signed-sparse
path used batches of 128 challenges per SHAKE stream. The operator-rejected
path used one SHAKE stream for the complete vector. A rewind of the seed changed
the complete vector. The CWSS argument in
[`specs/subring-coefficient-packing.md`](subring-coefficient-packing.md), and
the same argument for `EvaluationTrace`, instead requires forks that change one
coordinate while all other coordinates stay fixed. PR #394 identified this
gap. Slice 1 replaces both paths with the indexed construction below.

The old unmerged design proposed one `u16` per nonzero query, a nine-bit cap,
and a new byte-granular transcript squeeze cursor. The current code makes those
choices unnecessary:

1. `AkitaBatchedProofShape` already derives the exact headerless proof layout.
2. `challenge_block(label)` consumes one complete 32-byte sponge block with a
   fixed return type.
3. `AkitaTranscript` already has a prover-only preview path that clones the
   sponge state for fold-response search.
4. Before this cutover, the proof carried one fixed `u32` fold nonce in every
   nonterminal and terminal fold object.

This specification uses those current boundaries.

## Intent

### Goal

Add one schedule-derived `GrindingPlan`, one canonical
`TranscriptNonceStream`, and one transcript proof-of-work primitive, then route
every current Fiat-Shamir query through the plan. The same stream will carry
the existing fold-response nonces in 12 bits each. Replace the shared sparse
challenge XOF stream with indexed coordinate queries so the implemented fold
challenge structure matches the existing CWSS extraction theorem.

### Terminology

This document uses these terms precisely:

| Term | Meaning |
|---|---|
| challenge site | The smallest scalar or tuple of Fiat-Shamir draws covered by one public conditional bad-set bound |
| pre-query history | All public transcript state and prover messages fixed before a candidate nonce is selected |
| loss factor `L` | Public upper bound such that a query's conditional algebraic error is at most `L / |E|` |
| grind bits `g` | Public proof-of-work target assigned to a query |
| nonce bits `w` | Number of proof bits reserved for the bounded nonce search |
| proof-of-work query | A query that checks a zero-bit predicate before drawing the protocol challenge |
| fold-response query | Existing rejection sampling that changes the sparse challenge until the honest response fits |
| zero-bit query | A catalogued query with `g = 0`; it consumes no nonce and makes no transcript change |
| fold coordinate | One sparse challenge at a canonical `(group, claim, block)` position |

### Invariants

1. **One public plan.** Prover, verifier, proof shape, descriptor, serializer,
   and proof-size accounting MUST consume the same `GrindingPlan`. No callsite
   may reconstruct a competing nonce width or loss bound.
2. **Conditional query boundaries.** Each plan site MUST have one stated bound
   on its bad challenge fraction for every fixed pre-query history. One
   extension-field element is not split into base-field limbs. One
   multilinear point remains one tuple when one joint bound covers the point.
   Each sumcheck round is separate because its round polynomial fixes a new bad
   set. The absence of an ordinary prover message is not enough to merge two
   sites: a nonzero grinding nonce is itself an adaptive prover message.
3. **Public acceptance.** A proof-of-work query with target `g` accepts exactly
   when the first `g` bits of its public 32-byte predicate block are zero.
4. **Separate predicate and challenge.** The predicate block MUST NOT be reused
   as the protocol challenge. Checking a nonzero proof-of-work query advances
   the transcript by one full predicate squeeze before the actual challenge is
   drawn.
5. **Zero means no operation.** A zero-bit query consumes no stream bits,
   absorbs no grinding payload, emits no predicate squeeze, and leaves the
   transcript unchanged.
6. **Exact packing.** Each stream entry consumes exactly the number of bits in
   its plan entry. Entries have no wire tags, widths, padding, or alignment
   between them.
7. **Canonical tail.** The stream is little-endian at both the bit and integer
   levels. Any unused high bits in the final byte MUST be zero.
8. **Complete consumption.** Verifier replay MUST reject a truncated stream, a
   stream with nonzero tail padding, a plan mismatch, a query-kind mismatch,
   or any bits left after the final scheduled query.
9. **Bounded proving.** Every search has a public finite candidate space.
   Exhaustion returns `AkitaError`; it never loops without a bound.
10. **Verifier safety.** All verifier-reachable decode, plan, cursor, and
    predicate failures return `AkitaError` or `SerializationError`. They MUST
    NOT add `panic!`, `assert!`, `unwrap`, unchecked indexing, or allocation
    from an unvalidated length.
11. **Positional transcript.** Production labels remain diagnostics. The fixed
    proof-of-work payload enters the sponge. A semantic label does not.
12. **Fold response behavior is preserved.** The fold-response nonce remains
    one jointly searched value per fold, shared by all commitment groups. Its
    numeric value still enters each group root payload as little-endian `u32`.
    The 4096-candidate bound, prover acceptance rule, verifier response check,
    and each coordinate's marginal challenge law remain unchanged. Indexed
    coordinate derivation intentionally changes deterministic challenge test
    vectors and may change which nonce an honest prover accepts.
13. **Sparse support is not double charged.** The signed sparse challenge and
    operator-rejected sparse families retain their current certified support.
    They have no proof-of-work query merely because a fold-response nonce is
    present.
14. **No compatibility path.** The proof and descriptor wire formats change in
    place. Akita provides no backward proof compatibility, so there is no
    legacy decoder or duplicate verifier replay.
15. **Coordinatewise CWSS queries.** Every sparse fold coordinate MUST be a
    distinct fixed-width indexed oracle query. Reprogramming one coordinate
    with the group root and fold-response nonce fixed MUST leave every other
    coordinate and the live transcript state unchanged.
16. **One derivation direction.** Public schedule, normalized opening layout,
    field metadata, and protocol configuration derive the `GrindingPlan`. The
    plan derives `nonce_stream_bits`, which is then supplied to
    `AkitaBatchedProofShape`. Plan derivation MUST NOT consume the completed
    proof shape.

### Non-goals

This feature does not do any of the following:

1. It does not change the marginal sparse fold challenge distribution, its
   support certificate, the LS18 unit-difference argument, or the response
   norm policy. It does change the deterministic mapping from a group root seed
   to a challenge vector.
2. It does not use proof-of-work to price SIS or MSIS security.
3. It does not treat the fold-response nonce as evidence that a response is
   short. The verifier still checks the scheduled response representation and
   norm bound.
4. It does not reduce conditional statistical soundness. Conditioned on a
   valid proof-of-work predicate, the protocol challenge remains uniform and
   its bad fraction remains `L / |E|`.
5. It does not claim a quantum random-oracle proof. Akita's current
   Fiat-Shamir accounting uses a classical online random-oracle query bound.
   A QROM theorem, including quantum search against the predicate, is separate
   work.
6. It does not prove a new extraction theorem for a vector expanded from one
   shared XOF stream. The implementation instead exposes the product challenge
   structure assumed by the existing coordinatewise CWSS theorem.
7. It does not add proof-of-work to sparse fold challenges. Their certified
   challenge families and existing schedule accounting remain the source of
   their 128-bit target.
8. It does not add `spongefish-pow`, replace the transcript backend, or switch
   Akita to another challenger interface.
9. It does not add a general transcript journal or a byte-level squeeze cursor.
10. It does not preserve the old unmerged fixed-`u16` proposal.

## Security model

### Nominal challenge capacity

Akita uses the existing nominal challenge-capacity convention

```text
C = F::modulus_bits() * E::EXT_DEGREE.
```

The production fp128, fp64 quadratic-extension, and fp32 quartic-extension
profiles all have `C = 128`. This is the same field-width convention used by
the current protocol. The tiny difference between a pseudo-Mersenne prime and
the adjacent power of two remains part of complete concrete accounting. It
does not force one extra proof-of-work bit at every query.

### Per-query grind bits

For a challenge query with public loss factor `L >= 1`, target `T = 128`, and
nominal capacity `C`, the plan assigns

```text
loss_bits = ceil_log2(L)
g = max(0, T + loss_bits - C).
```

For current production profiles this simplifies to

```text
g = ceil_log2(L).
```

For every fixed pre-query history, one candidate passes the predicate with
probability `2^-g`, and its separate protocol challenge lands in the bad set
with probability at most `L / |E|`. A classical attacker therefore expects
about

```text
2^g * |E| / L
```

random-oracle work per accepted bad candidate. Under the nominal capacity
convention, this meets the 128-bit target when `C = 128`. Exact field
cardinality remains visible in the displayed expression and in complete
concrete accounting.

This is computational work accounting. It is not a claim that the interactive
soundness error changed from `L / |E|` to `1 / |E|`. The proof-of-work
predicate and the protocol challenge use separate random-oracle outputs so the
challenge remains uniform after predicate acceptance.

### Grinding-aware classical ROM bound

The grinding claim uses a direct online random-oracle bound. For site `i`, let
`B_i(h)` be the bad set after fixing any reachable pre-query history `h`. The
site's canonical loss helper MUST establish

```text
Pr[c_i in B_i(h) | h] <= L_i / |E_i|
```

for every such `h`, including histories containing earlier accepted grinding
nonces. The prover message that fixes `B_i(h)` MUST be absorbed before the
candidate nonce. The predicate and protocol challenge MUST be distinct
consecutive outputs in the ideal duplex or random-oracle model. Production
labels remain diagnostic; consuming the complete predicate squeeze before the
challenge squeeze provides the separation.

If an adversary makes `q_i` distinct candidate queries at site `i`, then a
union bound gives

```text
Pr[accepted predicate and bad challenge at site i]
  <= q_i * 2^(-g_i) * L_i / |E_i|.
```

Repeated or malformed candidates do not improve this bound. Summing over
protected sites gives

```text
Pr[any grinding-priced bad challenge]
  <= sum_i q_i * 2^(-g_i) * L_i / |E_i|.
```

With `g_i = max(0, 128 + ceil_log2(L_i) - C_i)`, the nominalized term
`q_i * 2^-g_i * L_i / 2^C_i` is at most `q_i * 2^-128`. The exact term keeps
`|E_i|` in the denominator and reports the small pseudo-Mersenne deficit. The
total online query count remains explicit. The policy does not hide it inside
every honest prover target and MUST NOT add a blanket
`ceil_log2(number of sites)` surcharge.

This lemma is the normative justification for the grinding gain. The existing
unground Fiat-Shamir statement `(Q + 1) * kappa` remains a valid background
bound, but it does not prove the `2^-g` admission factor and MUST NOT be cited as
if it did. The final Akita security statement lists the grinding-priced terms,
CWSS extraction error, sumcheck error, transcript collision terms, honest
exhaustion, and MSIS assumptions separately before applying any outer union
bound.

### Bounded nonce search

For every proof-of-work query with `g > 0`, set

```text
w = g + 7.
```

The prover tests canonical integers

```text
0, 1, ..., 2^w - 1.
```

The probability that an honest prover finds no passing predicate is

```text
(1 - 2^-g)^(2^w)
  = (1 - 2^-g)^(2^(g+7))
 <= exp(-128)
 < 2^-184.6.
```

This bound is independent of `g`. The plan accepts fewer than `2^32` query
entries and encodes query ordinals as `u32`. A union bound therefore puts total
proof-of-work exhaustion below `2^-152`, which is already
well below the required `2^-128`. No per-proof query-count surcharge is needed.

The first implementation supports `g <= 25`, hence `w <= 32`. This is an
implementation bound, not a security limit. It lets candidate arithmetic use a
checked `u32` while the wire still uses exactly `w` bits. A schedule requesting
more than 25 grind bits fails during plan derivation and verifier setup.

### Current loss rules and site boundaries

The central query catalog uses these rules. `ceil_log2(0)` is never evaluated.
A singleton or absent check has `L = 1` and `g = 0`.

| Query family | Conditional site boundary | Loss factor `L` | Production `g` |
|---|---|---:|---:|
| degree `d` polynomial identity | one scalar challenge | `max(1, d)` | `ceil_log2(max(1, d))` |
| sumcheck round of degree `d` | one round after its prover message | `max(1, d)` | `ceil_log2(max(1, d))` |
| multilinear point with `n` coordinates | one consecutive point draw | `max(1, n)` | `ceil_log2(max(1, n))` |
| powers of one scalar batching `m` values | one scalar | `max(1, m - 1)` | `ceil_log2(max(1, m - 1))` |
| independent random coefficients | one consecutive coefficient vector | `1` | `0` |
| one random linear merge | one scalar | `1` | `0` |
| subring packing consistency at `alpha` | one scalar | `2s - 1` | `ceil_log2(2s - 1)` |
| signed sparse fold coordinate | one indexed coordinate query after a group root | certified support at least 128 bits | `0` |

For an ordinary evaluation-trace ring relation, the alpha loss MUST come from
one canonical degree-bound helper owned by the relation description. For a
subring coefficient-packing relation, that helper MUST return the existing
`2s - 1` bound. The plan, prover, verifier, and security report all call this
same helper.

Independent coefficient vectors deserve explicit treatment. If at least one
batched claim difference is nonzero, all but one independent coefficient may
be fixed and at most one value of the remaining coefficient makes the linear
combination vanish. The loss is therefore `1 / |E|`, regardless of vector
length. It receives zero grind bits. By contrast, batching with powers
`1, gamma, ..., gamma^(m-1)` creates a degree `m - 1` polynomial and receives
`ceil_log2(m - 1)` bits.

`alpha`, `tau0`, and `tau1` are separate conditional sites even when no
ordinary prover payload lies between their current draws. If a site has
nonzero `g`, its accepted nonce becomes part of the history for every later
site. The alpha degree helper bounds the bad set after the complete relation
payload is fixed. The multilinear point helper bounds each complete `tau`
tuple for every fixed prior history. Tests MUST verify the precise order
`alpha`, `tau0`, `tau1` on paths where all three exist. A future refactor may
group them only by adding one canonical joint loss bound and revising the plan
policy.

### Sparse fold CWSS structure

Let `m_g = num_claims_g * num_live_blocks_g` for commitment group `g`. The
pre-Slice-1 sampler made one transcript query for a 32-byte group root and then
consumed the complete vector from one or more shared SHAKE streams. This gave
the required marginal sparse challenge family, but it does not give the CWSS
extractor transcripts that differ at exactly one coordinate.

The new derivation keeps the existing group-root transcript transition. For
each canonical flattened coordinate

```text
j = claim_index * num_live_blocks_g + block_index,
```

it initializes a fresh SHAKE256 cursor with the exact 40-byte string

```text
group_root_seed_32
|| j_as_le_u64.
```

`j` MUST be less than `m_g`, and every checked conversion and multiplication
MUST fail before allocation or sampling. The group root already binds the fold
level's method, group index, claim and block counts, ring dimension, sparse
configuration, optional operator-rejection policy, and shared fold-response
nonce. The root has fixed width and is dedicated to fold-coordinate expansion;
the fixed-width `j` makes the inner queries distinct without another tag.
The coordinate queries do not mutate the live transcript. Group roots continue
to be squeezed in current group order.

In the classical multi-oracle ROM, the transcript backend and indexed SHAKE
call are modeled as separate ideal oracles. The concrete transcript already
binds the protocol, session, instance, complete fold context, and replay
position before producing this dedicated root; production transcript labels
are diagnostic and are not absorbed. Conditional on a fixed group root,
distinct fixed-width coordinate inputs are independent
queries with the same sparse challenge law as one current draw. An extractor
may reprogram coordinate `j` while holding the group root, fold-response nonce,
every other coordinate query, and the live transcript state fixed. The central
vector plus one fork per coordinate therefore has the product structure
required by the existing CWSS set. The existing LS18 argument then makes every
nonzero coordinate difference a unit. This applies to both `EvaluationTrace`
and `SubringCoefficientPacking`; it does not rely on a new full-vector rank
theorem.

The proof also carries the ordinary bad events that two 32-byte group roots
collide or that a root is guessed by a prior coordinate-oracle query. With
`Q_root` root queries and `Q_coord` coordinate queries, the union-bound term is
at most

```text
(Q_root choose 2) / 2^256 + Q_root * Q_coord / 2^256.
```

It is listed separately and is below the 128-bit target for the same bounded
online-query regime used by the existing Fiat-Shamir statement.

The operator-rejected challenge path also starts one cursor per coordinate and
runs the existing bounded inner rejection loop on that cursor. This preserves
the admitted marginal distribution and its support certificate. The ordinary
signed-sparse path does the same without operator rejection. Implementations
MAY parallelize coordinates, but output order is always claim-major block
order.

The shared fold-response nonce remains an honest-prover abort and retry value,
not proof-of-work. For a fixed accepting proof it is part of the pre-challenge
prefix. Coordinate forks hold it fixed. Its online attempts remain covered by
the existing random-oracle query accounting.

### Quantum scope

The PCS is intended for post-quantum use and its SIS tables target 128-bit
quantum attack cost. The current Fiat-Shamir text, however, states a classical
online random-oracle extraction bound. This feature preserves that theorem and
does not silently relabel the proof-of-work result as QROM security.

A future QROM analysis must account for quantum nonce search and for the
Fiat-Shamir extractor. It may change the grind policy. That work is not a
reason to double every current grind target without a theorem.

## Query catalog and order

`GrindingPlan` is an ordered replay program, not just a total bit count. It
contains every conditional challenge site, including zero-bit sites, and one
compact run for each fold challenge group. Only entries
with a nonce width consume stream bits. Runs avoid allocating one plan object
or logging event per sparse coordinate; one checked range event records the
group index and exact coordinate count at the live sampler boundary.

The current protocol order for each fold level is:

1. Optional extension-opening reduction point, claim batching, and per-round
   sumcheck challenges. The coefficient-packing root omits this reduction.
2. Opening claim and evaluation batching after the opening payload is bound.
   A singleton batch returns coefficient one and has no plan entry or draw.
3. One fold-response entry for the fold, then one group run for each
   commitment group. Its multiplicity covers the root and all claim-major
   coordinates.
4. Ring-switch `alpha`, then the `tau0` and `tau1` point sites that
   exist for that fold shape.
5. Stage 1 tree sumcheck rounds. Each tree stage is followed by its powers-of
   gamma interstage batching query when child claims exist.
6. Optional physical L2 powers batching, linear merge, and sumcheck rounds.
7. Optional virtual-evaluation powers batching and compression query.
8. Stage 2 linear batching and per-round sumcheck challenges.
9. Optional Stage 3 setup-product sumcheck rounds.
10. Repeat the same scheduled sequence for the root, recursive folds, and
    terminal fold.

The current diagnostic labels map to catalog rules as follows. A label may
appear in more than one protocol context, so the plan site and public shape,
not the byte label alone, select the rule.

| Current label | Current context | Catalog rule |
|---|---|---|
| `CHALLENGE_EVAL_BATCH` | independent opening coefficients when the layout has more than one polynomial | independent coefficient vector, `g = 0`; a singleton returns coefficient one and has no query |
| `CHALLENGE_SUMCHECK_BATCH` | EOR split point | multilinear point, `g = ceil_log2(max(1, split_bits))` |
| `CHALLENGE_SUMCHECK_BATCH` | Stage 2 relation merge | one linear merge, `g = 0` |
| `CHALLENGE_EOR_CLAIM_BATCH` | independent EOR claim coefficients when there is more than one claim | independent coefficient vector, `g = 0`; a singleton has no query |
| `CHALLENGE_SUMCHECK_ROUND` | EOR, Stage 1, Stage 2, and Stage 3 rounds | full verifier-checked round degree from the canonical shape; Stage 1 uses `q_degree + 1` for the equality-factored product |
| `CHALLENGE_SPARSE_CHALLENGE` and indexed fold-coordinate queries | one sparse root followed by its claim-major block coordinates | one zero-bit group run after the fold-response entry; multiplicity is one plus the certified coordinate count |
| `CHALLENGE_RING_SWITCH` | ring-relation evaluation at `alpha` | relation degree bound |
| `CHALLENGE_TAU0` | Stage 1 multilinear point | number of point coordinates |
| `CHALLENGE_TAU1` | relation multilinear point | number of point coordinates |
| `CHALLENGE_SUMCHECK_INTERSTAGE_BATCH` | powers batching of child claims | `max(1, child_claims - 1)` |
| `CHALLENGE_L2_NORM_BATCH` | powers batching of norm subclaims | `max(1, subclaims - 1)` |
| `CHALLENGE_L2_NORM_MERGE` | range and norm merge | one linear merge, `g = 0` |
| `CHALLENGE_L2_VIRTUAL_BATCH` | powers batching of virtual evaluations | `max(1, evaluations - 1)` |
| `CHALLENGE_COMPRESSION_BINARY` | support-restricted binary relation merge | one linear merge, `g = 0` |

`CHALLENGE_LINEAR_RELATION` and `CHALLENGE_STOP_CONDITION` remain registered
diagnostic labels but are not reached by the current core replay found in this
audit. They receive no plan entries until a live protocol path draws them. The
challenge coverage test prevents either label from becoming live without a
catalog decision.

The plan MUST mirror the actual branch structure. In particular:

1. A direct or absent protocol component contributes no phantom queries.
2. Terminal replay contributes no `tau0` query because terminal ring switching
   has no Stage 1 point.
3. A consecutive extension-field tuple is one catalog entry when one
   conditional bad-set helper covers the complete tuple, even though
   `sample_ext_challenge` currently squeezes one base-field limb at a time.
4. A powers batching site is one entry before its scalar `gamma` draw.
5. A sumcheck contributes one entry per round because each round polynomial is
   absorbed before the next challenge.
6. The fold-response entry appears once per fold exactly where the existing
   shared nonce is selected. It precedes all group roots for that fold.
7. Each live group contributes one run with multiplicity
   `1 + num_claims * num_live_blocks`. Group and coordinate counts use checked
   `u32` and `u64` canonical encodings even when Rust call sites use `usize`.
8. A Stage 1 round checks the equality-factor interpolation polynomial times
   the Stage 1 round polynomial. If the stored Stage 1 shape gives the latter
   degree as `d`, the conditional bad-set bound is therefore `d + 1`. The plan
   MUST price that full degree. Production degrees 2 and 4 therefore use loss
   factors 3 and 5, which require 2 and 3 grind bits at nominal 128-bit
   capacity.
9. `EvaluationBatch` and `ExtensionOpeningClaimBatch` exist only when the
   shared row-coefficient sampler draws a challenge. The sampler owns the
   `m > 1` gate and the grind-then-draw transition, so the plan and both
   protocol directions cannot disagree about singleton behavior.

The first implementation MUST include an audit test that records every
challenge label and indexed fold-coordinate event reached by each production
schedule and proves that it matches the expanded plan in the same order. A
challenge draw that bypasses the catalog is a test failure even when its
assigned grind bits would be zero.

The `logging-transcript` build enforces this audit at the active adapter. A
proof-of-work plan entry creates one pending typed site. Every consecutive
base-field draw for that logical scalar, extension element, or tuple must have
the site's canonical diagnostic label, with extension limbs normalized to
their base label. Every draw is logged. A non-challenge transcript mutation,
the next plan action, or final exhaustion seals the site and requires at least
one matching draw. Production fp32 extension tests additionally compare the
exact limb counts for extension-opening, `tau0`, `tau1`, and every
single-extension-element query against the public geometry. Live sparse
sampling records the root squeeze and indexed coordinate range at the sampler
boundary, and later compact plan consumption must match that range exactly.
Final adapter exhaustion rejects an uncataloged challenge, a label mismatch,
a missing challenge, an omitted boundary between absorbed round messages, or
an unmatched fold root or coordinate range.

## Design

### Public policy and plan

`akita-types` owns the validated plan, fixed-width site identifiers, descriptor
section, nonce stream, proof shape, and the one canonical constructor from
field metadata, the effective `FoldSchedule`, and the normalized
`OpeningClaimsLayout`. `akita-config` supplies the concrete field metadata
from `CommitmentConfig` through its typed adapter. Schedule sizing calls the
same lower-level constructor. No protocol or planner crate rebuilds loss
factors or query order.

The semantic model is:

```rust
pub const TRANSCRIPT_SECURITY_BITS: u16 = 128;
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
pub const MAX_GRINDING_BITS: u8 = 25;
pub const FOLD_RESPONSE_NONCE_BITS: u8 = 12;

pub enum GrindingQueryKind {
    ProofOfWork,
    FoldResponse,
    FoldChallengeGroup,
}

pub enum GrindingSite {
    EvaluationBatch { level: u32 },
    ExtensionOpeningPoint { level: u32 },
    ExtensionOpeningClaimBatch { level: u32 },
    SumcheckRound {
        protocol: SumcheckProtocol,
        level: u32,
        stage: u32,
        round: u32,
    },
    FoldResponse { level: u32 },
    FoldChallengeGroup { level: u32, group: u32 },
    RingSwitchAlpha { level: u32 },
    Tau0Point { level: u32 },
    Tau1Point { level: u32 },
    Stage1InterstageBatch { level: u32, stage: u32 },
    L2SubclaimBatch { level: u32 },
    L2NormMerge { level: u32 },
    L2VirtualBatch { level: u32 },
    CompressionBinary { level: u32 },
    Stage2Batch { level: u32 },
}

pub struct GrindingRun {
    pub site: GrindingSite,
    pub loss_factor: u64,
    pub grind_bits: u8,
    pub nonce_bits: u8,
    pub multiplicity: u64,
}

pub struct GrindingPlan {
    pub runs: Vec<GrindingRun>,
    pub total_nonce_bits: usize,
}
```

`GrindingSite::kind()` is the only query-kind mapping. A run does not store a
second kind field because that would admit contradictory states. The exact Rust
layout MAY change to avoid recursive or oversized enums. The semantic fields
and one canonical derivation MUST remain. The constructor takes the public
schedule, normalized opening layout, and field tower metadata. Basis is bound
elsewhere in the call descriptor but does not change this plan. The constructor
validates every checked addition, query count,
loss factor, grind target, multiplicity, and total stream length before proving
or decoding. It returns the plan before `AkitaBatchedProofShape` is built:

```text
schedule + normalized opening layout + field/config metadata
  -> GrindingPlan
  -> nonce_stream_bits
  -> AkitaBatchedProofShape.
```

Zero-bit proof-of-work entries have `nonce_bits = 0`. Nonzero proof-of-work
entries have `nonce_bits = grind_bits + 7`. Fold-response entries have
`grind_bits = 0` and `nonce_bits = 12`. This makes the shared storage explicit
without confusing the security meanings. A fold challenge group run has zero
loss, grind, and nonce fields. Its multiplicity affects the expanded audit
schedule but not the stream size.

Proof-of-work and fold-response runs have multiplicity one. A fold challenge
group run has multiplicity `1 + num_claims * num_live_blocks`: one root draw
followed by its independently indexed coordinates. The plan checks

```text
total_nonce_bits = sum(run.nonce_bits * run.multiplicity)
```

with checked arithmetic, even though every current nonzero-width run has
multiplicity one. Only proof-of-work runs may have `loss_factor >= 1`.
Fold-response and fold challenge group runs encode `loss_factor = 0`.

### Descriptor binding

`FoldLinfProtocolBinding` is replaced by the protocol-wide
`TranscriptGrindingBinding`. `AkitaInstanceDescriptor` stores this dedicated
field immediately after `plan`; `SetupSection.fold_linf` is removed. The
binding contains exactly one field:

```text
plan digest                       = 32 raw Blake2b-256 bytes
```

The descriptor serializes these 32 bytes with no length prefix or padding. The
plan digest is recomputed from public call data and compared at the protocol
boundary.

The canonical plan digest input starts with
`b"akita/grinding-plan/v1"`, followed once by the active policy fields below,
then `run_count_as_le_u32`, then every run in replay order.

```text
encoding version                 = le_u16(1)
target security bits             = le_u16(128)
proof-of-work slack bits         = u8(7)
maximum proof-of-work bits       = u8(25)
predicate bytes                  = u8(32)
predicate bit-order tag          = u8(0)  // little-endian
fold-response attempt bits       = u8(12)
fold-response attempts           = le_u32(4096)
query-policy revision            = le_u16(1)
fold-coordinate oracle revision  = le_u16(1)
```

These fixed constants are not duplicated as independently mutable descriptor
state. They are committed once through the digest. Each run is
encoded as

```text
site_tag_as_u8
|| site_payload
|| loss_factor_as_le_u64
|| grind_bits_as_u8
|| nonce_bits_as_u8
|| multiplicity_as_le_u64.
```

The query kind is derived from the site and is not encoded a second time. Site
tags and payloads are fixed as follows.

| Tag | Site | Payload fields, all little-endian `u32` |
|---:|---|---|
| 0 | `EvaluationBatch` | level |
| 1 | `ExtensionOpeningPoint` | level |
| 2 | `ExtensionOpeningClaimBatch` | level |
| 3 | `SumcheckRound` | protocol, level, stage, round |
| 4 | `FoldResponse` | level |
| 5 | `FoldChallengeGroup` | level, group |
| 6 | `RingSwitchAlpha` | level |
| 7 | `Tau0Point` | level |
| 8 | `Tau1Point` | level |
| 9 | `Stage1InterstageBatch` | level, stage |
| 10 | `L2SubclaimBatch` | level |
| 11 | `L2NormMerge` | level |
| 12 | `L2VirtualBatch` | level |
| 13 | `CompressionBinary` | level |
| 14 | `Stage2Batch` | level |

Sumcheck protocol tags are `0` extension-opening reduction, `1` Stage 1, `2`
physical L2, `3` Stage 2, and `4` Stage 3. A site payload contains only the
listed fields in the listed order. Root queries use level zero. `u32::MAX` is
reserved and rejected for every site field. Rust `usize`, enum memory layout,
generic sequence encodings, and debug strings MUST NOT enter these bytes.
Blake2b-256 uses the existing
`digest_descriptor_bytes` primitive.

The plan itself is not serialized in the proof. Prover and verifier derive it
from public inputs and bind its digest through the descriptor preamble.
Changing a loss rule, query boundary, or encoding rule is therefore a protocol
revision, not an unbound local choice.

### Packed nonce stream

`AkitaBatchedProof` gains one leading field:

```rust
pub nonce_stream: TranscriptNonceStream,
```

The individual `fold_grind_nonce: u32` fields are removed from
`FoldLevelProof` and `TerminalLevelProof`. No replacement field is added to a
level proof.

The headerless proof body serializes the nonce stream first, then the current
root, recursive, and terminal payloads. The stream has no length header. The
canonical `AkitaBatchedProofShape` gains `nonce_stream_bits`, supplied from the
already validated `GrindingPlan` when the shape is constructed. Its byte
length is

```text
nonce_stream_bytes = ceil(nonce_stream_bits / 8).
```

The canonical serialized shape places
`nonce_stream_bits_as_le_u64` first, followed by the existing `root`,
`recursive_folds`, and `terminal` encodings in their current order. Shape
construction rejects a plan total that does not fit `u64`. Shape decoding
checks conversion from `u64` to `usize`, checked ceiling division by eight,
and `nonce_stream_bytes <= available_proof_bytes` before allocation or proof
decode. The decoded value also MUST equal the plan-derived total before the
nonce stream is read.

Bits are appended in plan order. For an entry value `x` with width `w`, stream
bit `j` receives

```text
(x >> j) & 1, for j = 0, ..., w - 1.
```

Within each byte, lower stream indices use lower bit positions. Entries begin
immediately after the preceding entry. If the final entry ends inside a byte,
all remaining high bits of that byte are zero.

The codec exposes checked sequential writer and reader operations. It does not
expose random access by site. The plan cursor checks the next expected
`GrindingSite` before reading or writing and derives the kind from that site.
The verifier checks exact completion once protocol replay ends.

The stream stores values, not attempts. A proof-of-work writer rejects a value
that does not fit in `w` bits. The verifier always decodes exactly `w` bits, so
an over-wide value is not a distinct wire condition. A fold-response value
must be less than 4096. The prover uses a checked wider loop counter so the
`w = 32` endpoint does not overflow a `u32` range expression. The enclosing
headerless proof decoder derives the exact stream byte count from context and
rejects trailing bytes at the whole-proof boundary. It cannot identify an
"extra stream byte" separately from an extra proof byte.

### Transcript proof-of-work primitive

`akita-transcript` owns the transcript transition and predicate, but not the
schedule-derived plan. For `g > 0`, the canonical absorbed payload is

```text
b"akita/transcript-grinding/v1"
|| g_as_u8
|| w_as_u8
|| nonce_as_le_u32
```

The complete payload is passed once to `append_bytes` under a diagnostic
grinding label. Akita's framed transcript absorb makes this prefix-free. The
production sponge does not absorb the semantic site label.

After the absorb, the transcript squeezes exactly 32 bytes under a diagnostic
predicate label. Predicate bit `i` is

```text
(predicate[i / 8] >> (i % 8)) & 1.
```

The predicate accepts when bits `0..g` are all zero. Because `g <= 25`, this
check needs no variable-length allocation and no partial squeeze. The caller
then invokes the existing scalar, extension, vector, or byte challenge API for
the actual protocol challenge.

For `g = 0`, callers MUST use the explicit no-op path. They do not absorb the
payload or squeeze the predicate.

The prover tests candidates against a scratch copy of the current sponge state
and commits only the accepted candidate to the live transcript. The existing
`FoldChallengeSeedPreview` path already clones the internal duplex sponge for
prover-only preview. It will be generalized into one prover preview primitive
that can execute the canonical absorb and one 32-byte squeeze. A full replay
journal and a public clone of the transcript state are not required.

The verifier decodes one scheduled nonce, applies the same live absorb and
squeeze, and rejects a nonzero predicate. It performs no search.

`LoggingTranscript` records one first-class event for each nonzero
proof-of-work check. The event contains the diagnostic site label, `g`, `w`,
the nonce value or its digest, and predicate length. Prover and verifier event
streams must remain identical. Zero-bit queries emit no grind event.

### Indexed sparse fold challenge derivation

`akita-challenges` keeps `FoldDraw` as the single owner of group-root transcript
payloads. Its live and preview implementations still squeeze one root per
group. `sample_batched_challenges_from_seed` and the one-cursor operator path
are replaced at this call site by one canonical indexed sampler. The sampler
takes the group root, checked coordinate index, ring dimension, sparse config,
and optional operator-rejection policy. It initializes the cursor from the
exact coordinate-oracle byte string specified above and samples one challenge.

`Challenges` keeps its current claim-major block layout. Sequential and
parallel execution MUST return byte-identical vectors. Both opening methods and
both ordinary and operator-rejected families call the same indexed primitive.
The implementation SHOULD reuse one buffer and `SignedSparseScratch` per
sequential sampler or Rayon worker, reinitializing only the SHAKE256 state for
each coordinate. It MUST NOT allocate the current 4 KiB XOF buffer once per
coordinate. Buffer reuse does not join the oracle streams because every reset
absorbs the complete fresh `group_root_seed_32 || j_as_le_u64` input before any
bytes are read.
The generic public `sample_sparse_challenges` API either moves to this primitive
or is removed if no live caller remains; a second legacy vector expander MUST
NOT survive beside the fold path.

The live transcript records one group-root squeeze and diagnostic coordinate
events. The coordinate events do not absorb or squeeze the transcript sponge.
The preview path collects the same group-root payloads it does today, derives
the same roots as live replay, and then runs indexed coordinate sampling. This
keeps the fold-response search interface and its transcript permutation count
unchanged.

### Fold-response migration

The current fold-response search remains in
`crates/akita-prover/src/protocol/fold_grind.rs`. It receives the next
`FoldResponse` plan entry, searches candidates `0..4096`, and writes the
accepted value into 12 stream bits. The verifier reads the same entry and
expands it to `u32` for the existing challenge sampler.

For every candidate, the sparse sampler continues to absorb in each group

```text
challenge context || nonce_as_le_u32
```

and then squeeze the current 32-byte group root. There is no proof-of-work
predicate before this root. Moving an accepted numeric nonce from a `u32` field
to 12 stream bits MUST preserve the root for an otherwise identical transcript
prefix. The indexed sampler intentionally changes the vector derived from that
root, so old sparse challenge vectors and accepted nonce values are not
compatibility fixtures.

The canonical grinding policy owns the 12-bit and 4096-attempt constants.
`FoldLinfProtocolBinding` and its fixed four-byte field are removed. The
fold-l∞ spec and Book text will be updated in the implementation slice so they
describe the shared stream while preserving their current soundness argument.

### Prover and verifier integration

The top-level prover creates one plan cursor and one stream writer before root
replay. The top-level verifier creates one plan cursor and one stream reader
after bounded proof decoding. Protocol functions receive the cursor through
their existing replay path. Type methods may assemble their fields into the
canonical operation, but no second policy or wrapper helper may recompute
grind bits.

The existing sumcheck APIs already accept a challenge closure. Their callers
will capture the shared cursor and call the canonical grind-then-sample
operation in that closure. `akita-sumcheck` therefore does not need to depend
on `akita-types` or learn Akita's schedule. Its round-message ordering remains
unchanged.

Challenge tuples use one plan entry and one optional predicate before their
first draw when one conditional bad-set helper covers the tuple. Examples are
EOR split points, independent row coefficients, `tau0`, and `tau1`. The cursor
advances once for the tuple. The current limb-level `sample_ext_challenge`
calls remain ordinary transcript squeezes inside that site. `alpha`, `tau0`,
and `tau1` remain separate sites and consume separate nonces when their targets
are nonzero.

Every prover and verifier mirror MUST consume the same site at the same point.
Tests compare both the ordinary transcript event stream and the plan-consume
event stream.

### Proof size and planning

Proof-size accounting adds exactly

```text
ceil(sum(query.nonce_bits) / 8)
```

bytes. It removes four bytes from every fold level and adds 12 packed bits for
that fold. Proof-of-work sites add their individual `g + 7` widths. Because
entries share final bytes, the total is rounded once, not once per query.

Planner and profile reports show:

1. total query count;
2. nonzero proof-of-work query count;
3. fold-response query count;
4. total stream bits and bytes;
5. a histogram of proof-of-work targets;
6. expected predicate trials, summed as `sum(2^g)`;
7. the maximum and union-bounded honest exhaustion probability;
8. every query family whose loss rule is nonzero.

Generated schedule identity includes the grinding-policy revision and the
grinding contribution to proof bytes. Schedule generation MUST fail if a
candidate cannot derive a valid plan. Existing catalog drift tests protect the
generated output.

## Evaluation

### Acceptance criteria

- [x] `akita-types` exposes one validated `GrindingPlan` whose ordered runs
      cover every conditional challenge site, sparse group root, and expanded
      fold coordinate for every production schedule and opening layout.
- [x] `akita-config` derives that plan from schedule, normalized opening
      layout, and field metadata before proof
      shape construction. `AkitaBatchedProofShape` consumes the resulting
      `nonce_stream_bits`; no reverse dependency exists.
- [x] The plan derives `g = max(0, 128 + ceil_log2(L) - C)`, uses `w = g + 7`
      for nonzero proof-of-work queries, rejects `g > 25`, and proves the stated
      `exp(-128)` per-query exhaustion bound in unit tests.
- [x] The plan catalog distinguishes degree checks, sumcheck rounds,
      multilinear points, powers batching, independent coefficient vectors,
      sparse challenge draws, and fold-response search using the loss rules in
      this specification.
- [x] A security test or executable table checks for every admitted site that
      `2^-g * L / 2^C <= 2^-128` under the nominal convention and separately
      reports the exact `2^-g * L / |E|` value and pseudo-Mersenne deficit. The
      exact deficit does not silently add a blanket grind bit. The security
      documentation states the separate
      `sum_i q_i 2^-g_i L_i / |E_i|` classical ROM bound and its conditional
      bad-set premise.
- [x] Every current challenge label reached by a production proof has exactly
      one matching logical plan entry. Zero-bit entries remain visible to the
      audit but consume no stream bits and make no transcript change.
- [x] `TranscriptGrindingBinding` replaces `FoldLinfProtocolBinding` in the
      dedicated descriptor field after `PlanSection`. It serializes only the
      Blake2b-256 plan digest. That digest commits the exact policy constants,
      oracle revision, and runs specified above. Golden bytes cover every site
      and sumcheck protocol discriminator.
- [x] `AkitaBatchedProof` carries one leading `TranscriptNonceStream`.
      `FoldLevelProof` and `TerminalLevelProof` no longer carry individual
      `u32` nonce fields.
- [x] `AkitaBatchedProofShape` carries the exact plan-derived stream bit count.
      Its canonical bytes start with `nonce_stream_bits_as_le_u64`, followed by
      the existing root, recursive-fold, and terminal fields. Checked conversion,
      byte-bound, and equality-to-plan validation occur before proof allocation.
      The codec is headerless, packs entries little-endian without per-entry
      alignment, and rejects truncation, nonzero final padding, wrong query
      kinds, wrong sites, and incomplete replay. The writer rejects over-wide
      values, and the enclosing exact proof decoder rejects trailing bytes.
- [x] Existing fold-response nonces consume exactly 12 stream bits, retain the
      4096-attempt cap, remain one shared value per fold across all groups, and
      produce the same group root for the same transcript prefix and numeric
      nonce across the wire cutover.
- [x] Every fold group derives exactly `num_claims * num_live_blocks` challenges
      from fresh indexed SHAKE256 cursors using the normative coordinate input
      and claim-major block order. Reprogramming a test oracle at coordinate
      `j` changes only `j`.
- [x] Indexed sampling preserves the certified marginal challenge law for the
      signed-sparse and operator-rejected families. Their support, LS18 unit
      difference, and norm-policy tests remain green for `EvaluationTrace` and
      `SubringCoefficientPacking`.
- [x] The subring coefficient-packing spec and Book security text no longer
      list full-vector CWSS structure as an open blocker. They state the
      coordinatewise construction and complete multi-fork accounting used by
      both opening methods.
- [x] `akita-transcript` exposes one canonical 32-byte proof-of-work predicate
      transition with the exact payload and low-bit test specified above.
- [x] The prover preview tests candidates without mutating the live transcript.
      The accepted candidate replay equals verifier replay for both Blake2b and
      Keccak transcript backends.
- [x] The predicate squeeze is distinct from the following protocol challenge.
      A regression test fails if the predicate bytes are reused as challenge
      bytes.
- [x] Every protected sumcheck round grinds after absorbing its round
      polynomial and before sampling that round's challenge.
- [x] Every protected consecutive challenge vector grinds once before its
      first coordinate when one conditional bad-set bound covers the complete
      tuple. No extension-field limb receives a separate stream entry.
- [x] Independent random coefficient vectors and single linear merges receive
      zero grind bits. Powers-of-gamma batching receives
      `ceil_log2(m - 1)` bits when `m > 2`.
- [x] Ring-switch alpha uses the canonical relation degree helper. Subring
      coefficient packing uses `L = 2s - 1`. `alpha`, `tau0`, and `tau1` are
      replayed as separate conditional sites in that order.
- [x] Sparse fold challenges receive no proof-of-work predicate. Their only
      stream entry is the existing 12-bit fold-response search value.
- [x] Logging tests show identical prover and verifier transcript events and
      identical ordered plan-consumption events. Zero-bit sites emit no grind
      event.
- [x] Verifier acceptance of a proof-of-work nonce equals recomputation of the
      public predicate. Fixed known nonpassing mutations reject. A mutated
      nonce may pass only when its own predicate passes. Fold-response mutations
      still require the checked response to match. Nonzero final padding
      rejects.
- [ ] Root, recursive, terminal, direct, extension-opening, subring-packing,
      physical-L2, compressed, and setup-prefix paths have end-to-end coverage
      for their selected query schedules.
- [ ] Proof-size and planner output report the exact packed stream contribution
      and expected prover work. Generated schedules and catalog identities are
      regenerated and drift checks pass.
- [ ] The Book transcript and security chapters, the fold-l∞ spec, the crate
      graph if dependencies change, and verifier-contract documentation are
      updated before the implementation is marked complete.
- [ ] All verifier-reachable failures obey the no-panic contract under malformed
      proof, shape, descriptor, and stream inputs.
- [ ] Base-to-head performance evidence includes
      `cargo bench -p akita-challenges --bench sparse_challenge` and the exact
      direct, recursive, fp32, fp64, fp128, and multi-group case matrix in
      `.github/workflows/profile-bench.yml`. Proof and setup bytes MUST NOT grow
      because of indexed fold derivation. Any statistically credible increase
      above 1% in complete prove or verify time requires an explicit reviewer
      decision and an optimization attempt that keeps coordinate independence.

### Testing strategy

New focused tests are grouped by owner.

`akita-transcript`:

1. valid and mutated predicate checks for every `g` boundary, including 1, 8,
   9, 16, 17, 25, and zero;
2. exact payload bytes and low-bit order;
3. zero-bit no-op state equality;
4. rejected preview candidates do not mutate live state;
5. accepted preview and live replay agree;
6. predicate output and following challenge are separate blocks;
7. Blake2b and Keccak parity at the protocol event level.

`akita-challenges`:

1. exact coordinate-oracle bytes and checked index bounds;
2. deterministic claim-major ordering in sequential and parallel modes;
3. changing one injected coordinate-oracle answer changes only that challenge;
4. marginal support and rejection-policy tests for D64 and D128;
5. one root per group and one shared fold-response nonce per fold;
6. golden vectors for both opening methods and both rejection modes.

`akita-types` and `akita-serialization`:

1. bit-stream round trips across byte boundaries;
2. all-zero and nonzero final padding behavior;
3. truncated and mismatched stream rejection, final padding rejection, and
   whole-proof trailing-byte rejection;
4. 12-bit fold-response bounds and 32-bit proof-of-work endpoint handling;
5. plan derivation snapshots for every production schedule family;
6. descriptor digest changes when any policy or plan entry changes;
7. proof-shape decode budgets reject before allocation.

Protocol crates:

1. per-round sumcheck placement after the absorbed round polynomial;
2. one entry per consecutive vector rather than per limb or coordinate;
3. query coverage for EOR, ring switching, digit range, physical L2, Stage 2,
   Stage 3, compression, root, recursion, and terminal replay;
4. same-nonce group-root regression across the fold stream migration and new
   indexed sparse challenge golden vectors;
5. proof tampering at each query kind;
6. prover and verifier plan exhaustion at the same final cursor position.

Planner and integration:

1. exact bit and byte totals against serialized proofs;
2. expected-work totals against the plan histogram;
3. generated schedule and catalog identity drift;
4. Jolt shape serialization and verifier replay;
5. all production field profiles and transcript backends.

The implementation runs the repository preflight from `AGENTS.md`, including
all four release Clippy feature graphs. Documentation slices run
`./scripts/check-doc-guardrails.sh`. The final test pass uses the current
commands in `.github/workflows/ci.yml`, including its target selection,
features, profile, and sharding.

### Performance

Verifier overhead is one 32-byte transcript squeeze for each nonzero
proof-of-work query. It performs no nonce search. Stream decoding is linear in
the number of stream bits and uses bounded storage derived before proof decode.
Indexed sparse derivation retains one transcript squeeze per commitment group.
It replaces shared XOF cursors with independently initialized SHAKE256 cursors,
which may run in parallel.

Expected prover predicate work is `2^g` per protected query. The intended
production targets are small because they pay only for public algebraic loss.
The implementation MUST report the observed target histogram before this spec
is marked implemented. Any production site above 12 grind bits requires
explicit review, not because it is unsound, but because it implies at least
4096 expected predicate trials at that site.

The packed stream is always no larger than byte-aligning each entry and is
strictly smaller than keeping one `u32` per fold. Indexed challenge derivation
adds no proof or setup bytes. The exact before and after proof sizes and timings
will be recorded with `cargo bench -p akita-challenges --bench
sparse_challenge` and the merge-base comparison matrix in
`.github/workflows/profile-bench.yml`. The matrix covers fp32, fp64, fp128,
direct, recursive, multi-group, and distributed schedules. No fixed byte or
timing estimate is normative before that measurement because current proof
shapes vary by field, schedule, opening layout, and optional security route.

## Alternatives considered

### One fixed `u16` per proof-of-work query

This was the old design. It is simple, but it caps the safe target at nine bits
under its completeness rule and wastes space for the common one-bit and
two-bit sites. Exact stream packing has one canonical codec and supports larger
targets without choosing a second proof format.

### Choose `u16` or `u32` at each query

Width classes avoid some waste, but each entry still pays byte alignment and
the policy needs a class-selection rule. The bit stream already needs a cursor
for the 12-bit fold values, so exact widths are simpler as a protocol rule.
Implementations may use `u16` internally when `w <= 16` and `u32` otherwise.

### Keep one nonce field inside each level proof

That works for fold-response search but does not scale to sumcheck rounds and
other nested challenge sites. It also ties storage to Rust proof nesting rather
than transcript replay order. One top-level stream makes ordering explicit and
lets adjacent entries share bytes.

### Serialize tags and lengths with each nonce

Tags make isolated decoding easier but duplicate public schedule information.
The verifier already requires the exact schedule and proof shape. Plan-kind
checks give the same safety without per-entry proof bytes.

### Add a byte-granular transcript cursor

The current transcript squeezes in 32-byte chunks. A 32-byte grinding
predicate consumes exactly one current chunk, so another buffering layer would
add state and replay complexity without saving a sponge call.

### Reuse predicate bytes as the protocol challenge

This would make predicate acceptance condition the challenge itself and would
invalidate the simple work calculation. A separate squeeze keeps the actual
challenge uniform.

### Add proof-of-work to sparse fold challenges

The sparse families already have at least 128 bits of certified support. Their
nonce exists to find an acceptable honest response. Adding a zero predicate
would duplicate work without restoring any missing challenge bits.

### Keep one shared sparse challenge stream and prove full-vector extraction

This preserves the current sampler, but it requires a new theorem. A rewind
changes every coordinate, so subtracting two accepted relations leaves one
equation in many unknown openings. Extraction would need enough full-vector
forks plus a rank theorem for the resulting sparse challenge-difference matrix,
with its own failure probability and random-oracle forking loss. That is more
proof risk than exposing the coordinatewise product structure already assumed
by Akita's CWSS argument.

### Squeeze one stateful transcript seed per sparse coordinate

This gives distinct transcript outputs, but a stateful fork at coordinate `j`
also changes every later sponge state and challenge. A triangular extractor may
be possible, but it is not the coordinatewise CWSS theorem Akita currently
uses. It also adds one transcript permutation per coordinate. Indexed queries
from the fixed group root keep all other coordinates and the live transcript
state unchanged without adding proof bytes or transcript squeezes.

### Retain batches larger than one in the sparse XOF

Batching amortizes XOF initialization, but every challenge after the first in a
batch shares a cursor and cannot be independently reprogrammed. Parallelizing
fresh indexed coordinate cursors is the allowed performance optimization. The
security boundary forbids restoring a shared cursor, even if a microbenchmark
is faster.

### Charge the maximum query loss to every challenge

This is easy to state but unnecessarily expensive. The schedule already knows
each query's degree, vector dimension, and batching form. Public per-site loss
rules are small, auditable, and meet the intended work target directly.

### Double grind bits for quantum search

That would be a policy guess, not a QROM proof. The current Fiat-Shamir theorem
is classical and already has explicit query accounting. Quantum policy changes
belong with a complete QROM analysis.

## Execution

Implementation proceeds in the following ordered slices. Each slice leaves the
tree buildable and testable. The same draft pull request carries all slices.

### Slice 0: Revive the normative specification

Status: complete on PR #417.

1. Add this current design record.
2. Add it to the live-spec indexes and dead-symbol guard.
3. Run documentation guardrails.

Exit condition: reviewers can evaluate the security model, wire format,
ownership, query catalog, and later slice boundaries without relying on the old
unmerged branch.

### Slice 1: Coordinatewise sparse fold challenges

Status: complete on PR #417.

1. Add the fixed-width indexed SHAKE256 cursor constructor in
   `akita-challenges`.
2. Add the one-coordinate signed-sparse and operator-rejected sampler.
3. Replace shared vector cursors in `FoldDraw` for `EvaluationTrace` and
   `SubringCoefficientPacking`, preserving one transcript root per group and
   one shared fold-response nonce per fold.
4. Add injected-oracle coordinate-fork tests, checked index tests, parallel
   determinism, marginal support tests, preview and live parity, and new golden
   vectors.
5. Run the sparse challenge microbenchmark and the profile benchmark matrix
   against the merge base. Optimize independent coordinate sampling if the
   complete prove or verify regression is statistically credible.
6. Update the active subring packing spec and Book security note to replace the
   full-vector blocker with the implemented coordinatewise CWSS argument.

Exit condition: changing one coordinate-oracle response changes only that
coordinate, both fold methods use the indexed sampler, proof and setup sizes
are unchanged, transcript sponge counts are unchanged, and the CWSS structure
blocker from PR #394 is closed without a new full-vector theorem.

### Slice 2: Canonical policy, loss helpers, and plan

Status: complete on PR #417.

1. Add the grinding policy constants, fixed-width query types, and canonical
   run encoding in `akita-types`.
2. Add the one canonical conditional loss helper for every site, including the
   relation alpha degree helper.
3. Derive the complete ordered plan in `akita-config` from schedule, normalized
   opening layout, and field metadata. Keep basis in its existing call binding;
   it does not affect the grinding plan.
4. Expand each fold challenge group run in query coverage snapshots for all
   generated production schedules.
5. Replace `FoldLinfProtocolBinding` with the dedicated
   `TranscriptGrindingBinding`, bind the canonical plan digest in
   `AkitaInstanceDescriptor`, and add descriptor golden bytes.
6. Construct `AkitaBatchedProofShape` only after the plan and pass in its exact
   `nonce_stream_bits`.

Exit condition: prover-independent code can derive and validate one plan, its
exact stream bit length, expanded audit order, and descriptor digest with no
plan and proof-shape ownership cycle. No proof wire change has landed yet.

### Slice 3: Packed stream and fold-response cutover

Status: complete on PR #417.

1. Implement `TranscriptNonceStream` plus checked reader and writer cursors.
2. Add `nonce_stream_bits` to `AkitaBatchedProofShape`.
3. Add the leading stream to `AkitaBatchedProof` serialization.
4. Remove every per-level `fold_grind_nonce: u32`.
5. Route existing fold-response proving and verification through 12-bit plan
   entries.
6. Update exact proof-size accounting and Jolt shape serialization.

Exit condition: all existing proofs use the packed stream for fold-response
nonces, equal numeric nonces produce equal group roots across the storage
cutover, and no proof-of-work query is active yet.

### Slice 4: Transcript predicate and prover preview

Status: complete on PR #417.

1. Add the canonical payload, 32-byte predicate, and low-bit checker in
   `akita-transcript`.
2. Generalize the existing prover sponge preview for this one transition.
3. Add logging events and known labels.
4. Cover zero-bit no-op, search exhaustion, backend parity, and
   predicate-versus-challenge separation.

Exit condition: a focused prover and verifier test can search, serialize,
decode, and check one proof-of-work entry without protocol integration.

### Slice 5: Protocol query integration

Status: complete on PR #417.

The implementation order was:

1. Add borrowed prover and verifier transcript adapters that own the exact
   plan and packed-bit cursors. Split transcript construction into the
   `TranscriptFactory` trait so a borrowed adapter does not need a fake
   constructor.
2. Make sumcheck round challenge callbacks fallible, so cursor and predicate
   failures propagate as `AkitaError`.
3. Integrate extension-opening reduction points, claim batching, and rounds,
   then each level's evaluation batching query.
4. Integrate the fold-response entry and audit the live fold root and indexed
   coordinates immediately after each group draw.
5. Integrate ring-switch `alpha`, `tau0`, and `tau1`, Stage 1 and physical L2
   queries, virtual L2 batching, compression, Stage 2, and Stage 3.
6. Give the root prover and batched verifier top-level cursor ownership and
   require exact exhaustion after the terminal fold.
7. Derive the diagnostic label from `GrindingSite` inside the adapter. Call
   sites supply no independent raw label that could disagree with the public
   plan.
8. Omit singleton coefficient-batching sites through the shared row sampler,
   which owns the grind-then-draw boundary, and add the feature-gated
   actual-challenge audit described above.

Prover and verifier mirrors consume every entry at the same logical boundary.
Integration exposed and corrected one older plan mismatch: evaluation batching
is per fold level and occurs after any extension-opening reduction, not once at
the global root before all folds. Planner pricing now uses the same order.

Exit condition: every live challenge draw is catalogued, all nonzero sites
check proof-of-work, every zero site is a transcript no-op, and the final plan
and stream cursors are exactly exhausted.

### Slice 6: Planner, generated schedules, and reporting

Status: complete on PR #417.

1. Add exact stream bytes and expected predicate work to candidate pricing.
2. Add plan revision to schedule and catalog identity.
3. Emit query counts, target histogram, exhaustion bound, and proof bytes in
   planner and profile reports.
4. Regenerate schedule tables and update stable snapshots.
5. Key suffix-frontier candidates by the exact successor data visible to a
   parent's grinding edge: successor kind, `d_a`, recursive opening variables,
   and recursive Stage 3 round count. Payload-only keys are unsound because
   equal-size successors can induce different packed parent nonce costs.

Exit condition: generated catalog drift checks pass and serialized proof sizes
match planner estimates for every production profile fixture.

### Slice 7: End-to-end hardening and documentation

Status: in progress on PR #417.

1. Add bit-level tamper tests, malformed shape tests, and no-panic verifier
   tests.
2. Run all transcript backend and feature-graph tests.
3. Update the Book transcript, security, binding, proof-size, and verification
   text.
4. Update `fold-linf-rejection.md` to describe its 12-bit stream entry without
   changing its soundness argument.
5. Record the grinding-aware ROM bound, the coordinatewise CWSS accounting,
   and base-to-head benchmark results in the durable security and profiling
   owners.
6. Update `AGENTS.md`, verifier contract, and crate graph only where the final
   implementation changes their owned contracts.

Exit condition: every acceptance criterion is checked, the final CI-fidelity
test commands pass, and the spec status and PR field are ready for the merge
state.

## File map

The expected primary surfaces are:

| Area | Current owner | Intended change |
|---|---|---|
| transcript trait and sponge | `crates/akita-transcript/src/lib.rs`, `sponge.rs` | predicate transition and preview |
| transcript diagnostics | `crates/akita-transcript/src/labels.rs`, `logging.rs` | grind labels and events |
| proof objects and shapes | `crates/akita-types/src/proof/levels.rs`, `shapes.rs` | top-level stream and exact bit shape |
| plan and policy types | new focused module under `crates/akita-types/src/` | validated runs, canonical bytes, stream length |
| plan derivation | `crates/akita-types/src/transcript_grinding_plan.rs`, typed adapter in `crates/akita-config/src/transcript_grinding_plan.rs` | single schedule and call-data constructor shared with exact sizing |
| descriptor | `crates/akita-types/src/instance_descriptor/` | digest-only binding for the canonical policy and plan |
| fold response search | `crates/akita-prover/src/protocol/fold_grind.rs` | 12-bit stream writer |
| sparse fold draw | `crates/akita-challenges/src/fold_draw.rs` | one root per group and indexed coordinates |
| sparse sampling | `crates/akita-challenges/src/sampler/mod.rs`, `xof.rs` | fresh cursor per coordinate for both rejection modes |
| sumcheck drivers | `crates/akita-sumcheck/src/drivers/`, callers in prover and verifier | captured grind cursor in challenge closures |
| ring switching | `crates/akita-prover/src/protocol/ring_switch/`, `crates/akita-verifier/src/protocol/ring_switch.rs` | alpha and point query integration |
| fold stages | `crates/akita-prover/src/protocol/core/`, `crates/akita-verifier/src/protocol/core/` | shared cursor replay |
| proof size | `crates/akita-types/src/proof_size.rs`, layout sizing | exact packed bytes |
| planning | `crates/akita-planner/`, `crates/akita-schedules/` | pricing, reporting, identity, generation |
| CWSS design record | `specs/subring-coefficient-packing.md`, `book/src/how/security.md` | close full-vector blocker with coordinate forks |

No new crate dependency is expected. If implementation changes that, update
`docs/crate-graph.md` in the same slice.

## Documentation

While this spec is active, it is the normative design record.
After implementation stabilizes, durable behavior belongs in:

1. `book/src/how/transcript.md` for the predicate transition, stream replay,
   and descriptor binding;
2. `book/src/foundations/pcs-and-binding.md` for classical random-oracle work
   accounting and its relation to fold-response nonces;
3. `book/src/how/security.md` for query loss rules and the explicit QROM scope;
4. `book/src/how/verification.md` for malformed stream rejection and the
   verifier no-panic boundary;
5. `book/src/usage/profiling.md` for planner and profile output.

When those chapters own the stable result, set `Book-chapter` to the primary
owner, mark this spec implemented, and archive it according to
[`specs/PRUNING.md`](PRUNING.md).

## References

1. [`book/src/foundations/pcs-and-binding.md`](../book/src/foundations/pcs-and-binding.md)
   for current Fiat-Shamir query and fold nonce accounting.
2. [`book/src/how/transcript.md`](../book/src/how/transcript.md) for positional
   transcript and descriptor binding.
3. [`specs/fold-linf-rejection.md`](fold-linf-rejection.md) for current
   fold-response rejection sampling.
4. [`specs/subring-coefficient-packing.md`](subring-coefficient-packing.md) for
   the `2s - 1` alpha bound, LS18 unit differences, and CWSS extraction
   obligation.
5. `crates/akita-transcript/src/sponge.rs` for the current 32-byte squeeze and
   prover preview mechanism.
6. `crates/akita-types/src/proof/shapes.rs` for headerless proof shape
   derivation.
7. Historical unmerged design at commit `5057456`, path
   `specs/transcript-grinding.md`.
8. [Fenzi, Moghaddas, and Nguyen, *Lattice-Based Polynomial Commitments:
   Towards Asymptotic and Concrete
   Efficiency*](https://eprint.iacr.org/2023/846.pdf), Definitions 2.29 and 2.30
   and Lemma 5.16, for the coordinatewise CWSS set and extraction premise.
9. [PR #394](https://github.com/LayerZero-Labs/akita/pull/394) for the review
   that identified the mismatch between that premise and Akita's one-seed
   full-vector implementation.
