# Spec: Transcript grinding

| Field | Value |
|---|---|
| Author(s) | Quang Dao, Codex |
| Created | 2026-05-22 |
| Status | active |
| PR | [#448](https://github.com/LayerZero-Labs/akita/pull/448) |
| Supersedes | Unmerged transcript grinding design at `5057456` |
| Superseded-by | |
| Book-chapter | |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Summary

Akita applies bounded transcript proof-of-work before Fiat--Shamir queries whose
algebraic loss makes a bad challenge easier to find than the nominal 128-bit
challenge-capacity convention permits. It also performs bounded fold-response
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
exposing coordinatewise forks for the CWSS extraction argument.

## Security model

For a challenge site with conditional bad fraction at most `L / |E|`, the plan
assigns

```text
g = max(0, 128 + ceil(log2(L)) - C),
```

where `C` is the nominal challenge capacity. A successful nonce requires a
separate 32-byte predicate whose first `g` low-order bits are zero. The prover
searches at most `2^(g+7)` candidates. Production policy rejects `g > 25`, so
the nonce width `g+7` always fits `u32`. Honest exhaustion is at most
`(1 - 2^-g)^(2^(g+7)) <= exp(-128)`.

The classical random-oracle bad-event bound is

```text
sum_i q_i * 2^-g_i * L_i / |E_i|,
```

under the conditional bad-set premise used by the corresponding algebraic
check. `q_i` includes adversarial candidate queries, including fold-response
trials. Grinding does not add entropy, prove uniqueness, or establish a QROM
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

The plan uses checked arithmetic. Its expanded query count MUST be less than
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

## Encoding and proof-size accounting

Each proof-of-work or fold-response site contributes the canonical unsigned
LEB128 length of its accepted nonce. Context records are diagnostic only;
public values are absorbed with `public_message`. Neither contributes proof
bytes.

Schedule selection deliberately retains main's packed objective,
`ceil(sum(semantic_nonce_widths) / 8)`, throughout `PackedProofCost`, suffix
search, dominance, runtime materialization, and generated artifacts. This is
not an exact estimate of the native LEB128 wire and is accepted migration debt.
It MUST NOT be used as a parser bound, nonce range, or security bound. A fresh
generation MUST reproduce main's schedule artifacts; generated files MUST NOT
be copied or hand-edited to manufacture parity.

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
| `akita-planner` | Query accounting and the frozen packed-bit schedule objective |

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

- [Native Spongefish transcript integration](spongefish-integration.md)
- [Transcript implementation](../book/src/how/transcript.md)
- [PCS binding and query accounting](../book/src/foundations/pcs-and-binding.md)
- [Verifier contract](../docs/verifier-contract.md)
- [Subring coefficient packing](subring-coefficient-packing.md)
