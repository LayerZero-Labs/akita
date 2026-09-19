# Spec: Native Spongefish transcripts and proof streams

## Summary and decision

Adopt Spongefish's native prover state, verifier state, argument serialization,
message receipt, domain separation, challenge extraction, and proof completion
as Akita's canonical transcript and proof stream. Change Akita's proof layout
and transcript encoding to fit those APIs. Akita is not in production: proof
bytes, challenge vectors, headerless layout, packed nonce storage, and the
existing transcript API have no compatibility requirement.

The purpose is to delete Akita implementations of facilities Spongefish already
provides. Native state adoption alone is an intermediate step, not completion.
The final production path uses native `prover_message` to emit or receive and
absorb proof messages, `public_message` for public/derived values, and
`verifier_message` for challenges. Verification requires `check_eof` plus
Akita's algebraic and grinding-plan completion checks.

Preserve all Akita protocol functionality, including proof-of-work and
fold-response grinding. Soundness, validated input bounds, and protocol
dependencies are hard requirements. Encoding compatibility is not. A reviewed
security argument is required for changed transcript semantics; passing tests
alone cannot establish absence of soundness issues.

## Motivation and dependency

Akita currently maintains proof serialization, transcript forwarding, separate
proof-field replay, nonce transport, and diagnostics alongside Spongefish's
native states. A native argument stream couples receipt with absorption and
removes duplicated plumbing. Protocol equations, validated schedules, field
codecs, sparse sampling, grinding policy, and semantic audits remain in Akita.

Target Spongefish revision:
`d2d190b1329d35ac9577438d05aed4f17a57b9f9`, or an explicitly reviewed successor.
Byte `Encoding` supplies blanket `NargSerialize`; native receipt absorbs the
decoded value's re-encoding. Runtime shapes are handled with schedule-bounded
fixed-size atom reads, without requiring a contextual upstream decoder.

The implementation centers on `crates/akita-transcript/src/native.rs`,
`crates/akita-types/src/proof/wire.rs`,
`crates/akita-types/src/transcript_grinding/native_replay.rs`,
`crates/akita-config/src/transcript_binding.rs`, and protocol callers in the
prover/verifier crates. Indexed sparse sampling remains in `akita-challenges`.

## Intent and invariants

1. Fix every prover-controlled value before the dependent challenge required by
   the soundness argument. Transport receipt alone is insufficient without
   absorption at the correct protocol point.
2. Codecs must be injective and canonical, with unambiguous message boundaries
   under the bound public schedule. Spongefish concatenates absorbs: it does
   not automatically tag Rust types or frame arbitrary variable byte arrays.
3. Prover and verifier execute the same public-message/challenge sequence,
   including optional branches. Proof-controlled lengths and tags cannot
   select a different accepted schedule.
4. Bind setup, field tower, effective schedule, call layout, basis, opening
   method, grinding policy, and actual commitments, points, and claims before
   their dependent randomness. Descriptor shape alone is not statement binding.
5. Preserve the security target and applicable classical random-oracle accounting,
   including nonce trials and sampling bias. This change adds no QROM theorem.
6. Preserve both backends, field profiles, heterogeneous groups, opening methods,
   recursion, terminal checks, EOR, L2 policies, setup offloading, parallelism,
   persistence, diagnostics, and Jolt integration. Format/API changes are allowed.
7. Reject malformed input, noncanonical fields, wrong shape, bad nonce ranges,
   overflow, truncation, and trailing bytes with Akita errors, without panic or
   unchecked allocation. Derive allocation bounds from validated public inputs.
8. Keep verifier construction independent of entropy, including the recursion
   guest. Spongefish's private prover RNG does not make Akita zero knowledge.
9. Keep one canonical proof codec and verification path. Inspection and
   persistence consume the new stream rather than maintaining the old format.

Non-goals: legacy proof acceptance, old challenge-vector compatibility, dual
production transcript implementations, replacing unrelated setup serialization,
changing lattice relations, or removing supported modes.

## Design

### Native states and domain separation

Construct native `ProverState` or `VerifierState` after resolving the public
schedule and descriptor. Keep one descriptor/grinding-plan builder; refactor
`bind_transcript_instance_descriptor` to return the descriptor as needed rather
than resetting an existing transcript. Remove placeholder-bound construction
and the unbound-state runtime panic from production entry points.

Use a new backend-specific protocol identifier for this transcript epoch,
Spongefish `.session(...)` for the actual application session, and
`.instance(...)` for the canonical descriptor. Frame variable session and
descriptor bytes with canonical lengths. Bind wire and challenge-codec versions.
Remove the fixed session tag plus combined session/instance workaround.

Application composition explicitly binds parent context or parent transcript
output into the session/instance. Never silently discard prior application
messages. Document the parent/child challenge dependency and test different
parent contexts. Native verifier construction must not instantiate a prover RNG.

Use native APIs directly. Delete `Transcript`, `TranscriptFactory`,
`TranscriptChallengePreview`, `AkitaTranscript`, and reset/bind methods after
migrating callers. Side-specific contexts may own native states and plan cursors
but must not republish a generic substitute transcript interface. Shared
arithmetic routines take values rather than a mock transcript.

Use upstream protocol-identifier helpers with static literals tested to fit
64 bytes. Runtime session input uses framed encoding, not a potentially panicking
protocol-identifier helper. DomainSeparator enforces construction order, not a
global prohibition on resetting: low-level construction must be confined to
reviewed initialization boundaries.

### New message grammar and codecs

The proof is the Spongefish argument string emitted in protocol order. Remove
the separate nonce prefix and old root/recursive/terminal storage serializer.
The public schedule determines message kinds and counts.

The new baseline encoding rules are:

- Base-field elements use fixed-width canonical little-endian bytes; reject
  out-of-field representatives rather than reducing proof input. Extension
  elements use ordered base-field coordinates.
- Each nonzero protected-query nonce is a native `u32` message: four
  little-endian bytes, range-checked against the plan's `g + 7` bits.
- Each fold-response nonce is one native `u32` message per fold: four
  little-endian bytes, range-checked against the 12-bit search domain.
- Fixed-count vectors are sequences of fixed-size atoms. Counts come from
  the validated bound schedule, without trusting proof-supplied lengths.
  Prefer bounded scalar receipt loops over allocation-heavy dynamic helpers.
- Genuinely variable payloads receive a fixed-width length message first.
  Check it against the schedule-derived bound before allocating or reading the
  payload. The length is absorbed through native receipt. Require canonical
  compression and valid tail contents.
- Public context records contain a versioned domain and fixed-width or
  length-prefixed fields. They use `public_message` and occupy no proof bytes.
  Diagnostic labels are not implicitly cryptographic domains.

The canonical dependency map is:

| Protocol group | Stream or public value | Trusted shape or bound | First dependent randomness or check |
| --- | --- | --- | --- |
| Root statement | Commitments, opening points, claims, basis, call layout | Validated root layout and descriptor | Evaluation batching and first fold randomness |
| Extension-opening reduction | Partial claims, round polynomials, final claims | Scheduled rounds, degree, and group counts | Per-round challenge and final oracle equality |
| Opening payload | Compressed relation payload | Schedule-derived count or bounded payload length | Fold-response search and relation checks |
| Fold response | One native `u32` nonce | 12-bit semantic range | Every group-local fold root at that level |
| Fold challenge group | Public fold-domain payload and root challenge | Scheduled group and coordinate counts | Indexed sparse coordinates and folded witness |
| Stage 1 | Range-product messages and child claims | `DigitRangePlan` and scheduled security route | Per-round and interstage batching challenges |
| Physical L2 | Norm claims, subclaims, virtual evaluations | Scheduled L2 shape and response cap | Norm batching, merge, and terminal equality checks |
| Stage 2 | Compression/relation sumcheck messages | Scheduled rounds and degree | Per-round challenge and successor witness binding |
| Stage 3 | Setup slot, product claim, rounds, prefix evaluation | Validated setup-prefix slot and schedule | Per-round challenge and setup-product equality |
| Successor witness | Inner or outer commitment binding | Public next-level transition | Ring-switch and successor-level randomness |
| Terminal response | Canonical bounded response payload | Terminal plan and schedule-derived byte cap | Reconstruction, norm, relation, and retained-binding checks |
| Proof-of-work | One native `u32` nonce for each nonzero target | Grinding plan, `g + 7` range, `g <= 25` | Distinct predicate followed by the protected challenge |

Compressed payload grammar and conditional branches are defined by the public
schedule and the site/kind records below. Incidental struct or derive ordering
is not a wire contract.

Use or derive only `Encoding` and `NargDeserialize` for proof messages.
Do not derive `Codec`: it also exposes challenge `Decoding`. Reserve
`Decoding` for distributions satisfying its contract. Add small local
newtypes for external Jolt field types where Rust orphan rules require them.
Receive fixed-size atoms with native `NargDeserialize`; do not build a second
general serializer or upstream shape-context framework. Keep Akita compression
algorithms and algebraic checks where Spongefish supplies no equivalent.

Native receipt may absorb a structurally valid atom before checking its
schedule-dependent meaning. A failed semantic check must abort verification
before subsequent challenges or use. Rollback of a failed verification is not
required. For each accepted message, require `encode(decoded) == consumed`,
where `consumed` is exactly the prefix removed from the input cursor, excluding
remaining messages. Include lengths, padding and compressed representation.
Multiple valid proofs are allowed; multiple encodings of the same message are not.

On decoding failure, leave the cursor unchanged; composite decoders stage cursor
movement and commit only on success. Property tests cover both round-trip
directions, consumed-prefix equality, failed-cursor preservation, and rejection
of altered padding/high bits.

No proof input path may use `Validate::No` or unchecked deserialization.
Add scoped CI checks for the new proof codecs and verifier paths, and review
helper/alias reachability. A source scan is a regression guard, not a proof of
reachability. Shared trusted-cache decoders outside proof input paths may remain.

### Explicit message and challenge context

Absorb a fixed-format public context record before each logical proof-message
group and each challenge group. A group can contain multiple atoms; framing the
group costs no proof bytes. Use this canonical layout:

```text
domain[32] || version:u32 || site_id[32] || kind:u32 ||
atom_count:u64 || encoded_bytes:u64 || challenge_bytes:u64
```

Integers are little endian; inactive fields are zero. Domain is a fixed 32-byte
literal. Site IDs canonically encode protocol family, fold, stage, round, group
and limb positions with checked ranges and zero unused positions. Define kind
and site assignments once in the message table. Do not use source lines,
diagnostic strings, or proof-controlled branches as identities.

For variable payloads, receive a bounded length as its own recorded fixed-width
message, validate it, then absorb the payload record with the validated size
before receiving the body. Public/derived variable messages also need framing.
Record recursive commitment and terminal inner-state binding as distinct kinds,
selected by the public schedule.

Record challenge width before every prescribed challenge group, including
predicate and field-coordinate draws. These are native `public_message`
operations, not a new transcript engine or serialized operation journal.

This requirement applies to both backends. Keccak's duplex state can forget
different nonzero squeeze lengths within one rate block after the next absorb;
the digest bridge explicitly incorporates consumed length. Neither width nor
Rust type names are implicit domain separators. Associative absorbs also do not
distinguish atom grouping. Public records make site, kind, count and width
binding explicit; they do not replace schedule binding or the soundness review.

### Protocol order and terminal bindings

Bind public statements first. Preserve the dependency order: EOR where needed,
opening payloads before batching, fold nonce before group-root draws, outgoing
witness binding before ring-switch randomness, and each sumcheck message before
its challenge. Preserve setup-prefix and witness group order.

Move terminal inner-state binding into the earlier predecessor slot where it
must be fixed. Retain that value and compare it to the terminal response's
reconstructed inner state. If any copy is transmitted later, equality is
mandatory. A terminal-only path receives its own binding at its prescribed point.
Never move a commitment after the random points checking its relations.

Every freely chosen proof value enters through native proof receipt.
`public_message` handles actual public inputs and specified derived/repeated
bindings. Audit terminal values with no later challenge separately: their direct
checks remain mandatory regardless of absorption.

The emitted argument string is the authoritative proof. Structured inspection
can decode it; builders can retain diagnostic views. Accepted proofs must use
one canonical verification path. Existing structured-proof APIs may convert
through the new codecs if retained; measure that cost instead of introducing
another verifier. Remove the old proof serializer after migrating its callers.

### Preserve grinding through native operations

Offload nonce transport, live absorption, and challenge extraction to Spongefish.
Keep policy, bounded search, predicate testing, plan enforcement, and public
preview in Akita.

For each protected query with `g > 0`:

1. Absorb a public context containing a new grinding domain, plan site identity,
   `g`, and nonce width `g + 7`.
2. Preview candidates from that public sponge state. Absorb exactly the native
   `u32` nonce encoding and squeeze a 32-byte predicate. Accept when its first
   `g` low-order bits are zero.
3. Emit the winner with native `prover_message`. The verifier receives it with
   native `prover_message`, checks its range, draws the predicate with
   `verifier_message`, and rejects unless it passes.
4. Absorb the protected challenge's distinct public context record, then draw
   its bytes. Never reuse predicate bytes as the protocol challenge.

The candidate program includes the nonce message record and predicate challenge
record. Their encodings and order are identical to live native replay.

Keep targets at most 25 bits, bounded `2^(g+7)` search, and explicit exhaustion.
Zero-bit grinding remains a no-op with no nonce or grinding-specific context;
the ordinary protocol challenge still has its mandatory context record. Consume each site
once and associate it with the intended protected challenge. Preserve semantic
plan completion and query auditing while replacing the transport cursor.
Accept any satisfying in-range nonce, not necessarily the
prover's first solution. Proof uniqueness is not required; proof hashes are not
unique statement identifiers. Range checks enforce the accepted search domain
and honest-search/exhaustion contract. Increasing the nonce range alone does
not decrease the expected work for a fixed predicate difficulty.

For fold-response search, bind a public fold-search domain and site identity.
Preview a native `u32` fold nonce followed by the prescribed group contexts and
root squeezes. Commit the winner once before those live draws. Keep one 12-bit
nonce shared across all groups and verifier enforcement of response representation
and norm bounds. The nonce can leave individual group-context encodings because
the shared state already binds it; specify the new transition exactly. Preserve
bounded exhaustion and adversarial trial accounting.

Previews must not mutate live state, private RNG, proof output, or plan cursors.
Clone public state once per candidate and advance through all groups in linear
work. Represent absorb-only steps, records, and exact-width squeezes explicitly.
Share protocol-specific sequencing and codecs between preview/live paths without
introducing a second generic transcript or production operation journal.

Prefer a supported public-preview API when available. At the pinned revision,
confine `duplex_sponge_state` access to one audited internal preview module.
Cargo feature unification exposes `yolocrypto` throughout the resolved dependency;
module privacy does not reverse that exposure. Add a CI allowlist for raw state
access and bypass constructors, with review of indirect accesses. Prohibit live
state replacement/rollback and `ProverState::default` in protocol code.
Document that candidate clones copy public state only, never private randomness.

The grinding argument must establish that conditioning on predicate success
does not raise the protected challenge's bad-event probability beyond its
stated sampler/security bound under the chosen sponge/random-oracle model.
The intervening context record gives a distinct challenge query; it does not
prove independence by itself. Statistical smoke tests are not a proof.

Keep indexed sparse expansion
`SHAKE256(group_root || little_endian_u64(coordinate_index))`, unbiased position
sampling, signs, magnitudes, and configured operator-norm rejection. Group roots
change, but the required coordinate distribution and extraction dependencies do
not disappear.

Bundled `spongefish-pow` is eligible only after a reviewed argument establishes
the required work/query guarantees, bounded honest search, independent protected
challenge, and backend behavior. The pinned seed predicate and default `u64`
search are not equivalent. This migration uses the native-state predicate above;
a later substitution must update this spec and planner/security accounting.
Maximal offloading does not justify an unreviewed cryptographic substitution.

### Native challenge extraction

Use native streaming squeeze consumption. Delete old 32-byte rounding and
discarded suffix behavior. New challenge codecs consume their specified exact
widths, and known-answer vectors change.

Use native `verifier_message::<[u8; 64]>()` for each base-field coordinate
and reduce the resulting 512-bit little-endian integer using existing field
arithmetic. This is the fixed width for supported fields of at most 128 modulus
bits. New larger fields require a budget calculation before admission. Bind the
codec version and record the width and limb identity at each draw.

Fixed-width reduction is not exactly uniform. For modulus `p`, input width
`k`, `N = 2^k`, and `r = N mod p`, total variation is exactly

```text
delta = r * (p - r) / (p * N) <= p / (4 * N).
```

Certify this using exact integer/rational arithmetic. With at most 128 modulus
bits and 512 input bits, per-coordinate distance is below `2^-386`.
Bound the relevant total draw count `B` and require
`B * delta <= 2^-192` as a separate sampling-error budget. Count extension
coordinates and relevant adversarial oracle/search queries, not just one honest
proof's schedule. Explicitly combine this term with the soundness/work argument:
a 128-bit work target is not automatically a per-proof statistical-distance
target. If the certified bound fails, increase/version the width before cutover.

Do not hide biased reduction behind field `Decoding` that promises uniformity.
Draw uniform native bytes, then apply the explicitly budgeted conversion.
Proof fields are strictly canonically decoded, never reduced. Extension
challenges use ordered coordinate draws and their joint-distribution bound.
Rejection sampling is a separate option requiring termination/query accounting;
it is not this cutover's default.

### Ownership and deletions

| Owner | Target responsibility |
| --- | --- |
| Spongefish | States, argument string, native codecs, message write/read/absorb, public messages, byte challenges, domain separation, EOF |
| `akita-transcript` if needed | Protocol domains, field codec newtypes, diagnostics, narrow public preview; no replacement engine or generic facade |
| `akita-types` | Protocol message types, validated shapes, compression semantics, grinding plan; no old proof wire serializer or packed nonce transport |
| `akita-config` | Descriptor and policy derivation from trusted public configuration |
| `akita-prover` / `akita-verifier` | Equations, message order, bounded search/checks and native state use |
| `akita-challenges` | Protocol-specific indexed sparse distribution |
| `akita-serialization` | Remaining non-proof artifacts and necessary shared primitives |

Delete obsolete code instead of compatibility aliases: runtime side dispatch,
placeholder binding, forwarding layers, old proof encoding/replay duplication,
packed nonce transport, and challenge block rounding. Preserve verifier crate
boundaries; codec placement cannot introduce a dependency cycle or prover code
in verification.

Keep logging and event inspection around native operations, prover/verifier
event equality, query-plan coverage, and missing-binding/order tests. Delete
redundant wire-record/append bookkeeping where native receipt handles both.
Continue explicit audits of public and derived bindings. Identical argument
strings do not establish identical public messages, challenge consumption, or
binding order. Retain traces covering all those operations.

Keep `digest_descriptor_bytes` for schedule, policy, artifact and grinding-plan
identities; direct instance absorption does not make those digests redundant.
Keep exact-deserialization checks for non-proof artifacts.

### Verification completion and supported dependency surface

One verifier context owns native state and semantic completion. A private
acceptance value is constructible only through a consuming `finish` method
after all algebraic checks succeed. Finish verifies schedule/grinding completion
and calls native `check_eof`. Public verification returns success only through
that value. This is lifecycle enforcement, not a forwarding transcript.
Test appended-byte rejection and incomplete plans at every entry point.

Map native failures to Akita errors with site, message kind and round. Include
byte offsets where cheaply available; do not duplicate parsing solely to obtain
an offset that the pinned native cursor does not expose.

Support only the reviewed Blake2b digest bridge and Keccak duplex backends.
Do not enable the pinned generic XOF backend with its unimplemented ratchet.
Frame session bytes directly; do not describe the helper's 64-byte session
identifier as 64 bytes of hash output.

Native prover construction needs entropy at this pin. Verify supported prover
targets accordingly; entropy-free prover support would require a reviewed
constructor extension. Test verifier/guest execution on the actual target
without invoking entropy. Akita does not use the private prover RNG, and pinned
native message methods do not reseed it with transcript input; do not infer
transcript-bound private randomness or zero knowledge from its documentation.

## Soundness review and acceptance gates

The merge gate is a reviewed transcript argument plus implementation evidence,
not byte equivalence with the old system. Required review artifacts:

- A message/dependency table maps every old binding obligation to the new stream,
  including optional paths and terminal consistency checks.
- A canonical grammar/domain map establishes injectivity, boundaries, challenge
  widths, statement/session binding, and rejection of alternate parses.
- A grinding argument connects native nonce transitions to the bounded
  random-oracle trials used by accounting, including multiplicities,
  fold-response search, and predicate/challenge separation.
- A quantitative challenge-distribution/security-budget check covers reduction
  bias and indexed sparse-coordinate dependencies.
- An implementation review covers codecs, bounds, schedule/plan completion, and
  every native API bypass. Unresolved soundness questions block cutover;
  keep the existing implementation until resolved rather than weaken a gate.

### Acceptance criteria

- Native Spongefish emission/receipt is the sole production proof path;
  obsolete transcript and wire compatibility code is removed.
- Grammar, site/kind assignments, public records, message dependencies,
  domains and soundness/accounting review have no unresolved blockers.
- Codec canonicality and cursor-on-error properties are tested, with
  unchecked decoding excluded from verifier proof input paths.
- Exact sampling-error accounting certifies the 64-byte coordinate width
  and aggregate budget including adversarial query counts.
- Raw-state/constructor allowlists and the consuming acceptance/finish
  boundary are enforced.
- Both grinding mechanisms retain bounded search, plan/site enforcement,
  exhaustion, predicate separation and verifier response checks.
- Acceptance requires EOF, algebraic validity, and complete plan consumption.
- Every supported mode, field, backend, persistence path and integration works.
- Malformed input rejects without panic or unchecked allocation.
- New traces/vectors agree across prover/verifier; logging does not alter
  challenges; verifier and guest construction need no entropy.
- A responsibility/code-size report demonstrates deletion of Akita plumbing,
  including dependency patches and maintenance of raw-state exceptions.
- Performance, memory, proof-size and planner-model changes are measured;
  affected artifacts, fixtures and documentation are updated.

### Testing

Use baseline traces to inventory semantic obligations, not require byte equality.
Commit new vectors for domains, field/extension draws, streaming consumption,
nonce/predicate/challenge transitions and sparse roots. Independently construct
prover and verifier states rather than using prover constructors on both sides.

Extend `transcript_hardening`, `transcript_hardening_proptest`, `label_schedule`,
`fold_linf`, `protocol_soundness`, field end-to-end and sparse sampler tests.
Replace packed nonce byte tests with inline range, truncation, order and plan
completion tests. Mutate omitted/swapped messages, sessions, descriptors,
statements, field encodings, lengths, terminal bindings, suffix bytes and
predicate reuse. Cover all optional paths and new stream decoder fuzzing.

Check that unsuccessful previews change neither live state nor output; compare
accepted previews to native replay for both backends, multiple groups and
multiple attempts. Exercise zero-bit, maximum-width, exhausted and boundary
cases without enumerating a complete 32-bit search. Tests support, but do not
replace, the soundness argument.

Run repository preflight from `AGENTS.md` before expensive compilation.
Final validation uses `.github/workflows/ci.yml`, all three required Clippy
graphs, and applicable portability, Jolt, guest and fuzz workflows. Test Keccak
with defaults disabled and logging on/off. Run the complete
order/omission/branch/width mutation suite under each backend, and make CI fail
if either is skipped; compiling Keccak alone is insufficient. Include a
substrate regression demonstrating squeeze-length reconvergence and protocol
tests demonstrating public records prevent corresponding cross-site ambiguity.
Retain the indexed SHAKE256 reference-comparison test to protect the fixed
40-byte coordinate input, padding, and expansion law.

### Performance and planner effects

Measure baseline/candidate release builds on the same host and inputs using
`akita_e2e` and `book/src/usage/profiling.md`, with at least five measured runs
after warmup. Include production fields, terminal/recursive paths, multiple
groups, nonzero PoW, response search and guest verification. Report proof size,
time, peak memory, allocations where available, attempts and guest cycles.
Measure preview absorb/permutation counts as group count grows; one-group
wall-time results cannot establish linear multi-group preview behavior.

Proof-size growth from inline nonces or framing is allowed. New nonce storage
is four bytes per nonzero PoW site plus four bytes per fold-response site,
replacing the bit-packed prefix. Update proof-size formulas, planner costs,
nonce metadata and catalog identities; old bit counts must not describe new
wire bytes. Security query counts remain independent of storage width.

Investigate substantial regressions (initial threshold: 5% median time or guest
cycles). Zero overhead and byte equality are not gates against native adoption.
Tradeoffs need measured cost and demonstrated maintenance benefit; none may
weaken soundness or remove supported modes.

## Rollout and alternatives

Ship one new proof/transcript epoch. Bump backend protocol identifiers and
descriptor/wire identities, update size/security metadata and fixtures, and
regenerate affected schedules/catalogs with repository workflows. Old proofs
and incompatible artifacts must fail under the new integration contract.
No fallback decoder or permanent dual mode.

1. Specify grammar, domains, dependencies and query accounting. Prototype native
   field atoms, inline grinding and early terminal binding with the pinned API.
2. Implement shared codecs and native bound states; migrate one complete path
   with response search and terminal consistency, then all supported paths.
3. Migrate integrations, persistence, logging, planner costs and artifacts;
   remove old production serialization and transcript plumbing.
4. Complete security review, robustness checks and measurements before cutover.

Alternatives considered:

- **Preserve old bytes:** rejected as a requirement because distinct encodings,
  deferred sections and compatibility adapters undermine the objective.
- **Only replace the state enum:** useful intermediate work, insufficient endpoint.
- **Whole proof as a blob:** retains a separate replay system and does not couple
  individual message receipt to dependent challenges.
- **Contextual upstream decoder extension:** unnecessary for schedule-bounded
  fixed atoms; consider only for a demonstrated capability/performance need.
- **Immediately switch to bundled PoW:** needs additional security analysis;
  retain the specified native-state extension until guarantees are established.

## Documentation and references

Implementation updates `book/src/how/transcript.md`,
`book/src/how/verification.md`, `book/src/how/architecture.md`, transcript README,
proof-format documentation, examples, and changed edges in `docs/crate-graph.md`.
Update the grinding spec's nonce transport, framing, size formulas and domains
in the same cutover. Preserve its security/response-bound obligations unless
replaced by an explicitly reviewed argument. This specification describes the
native proof-stream architecture; keep its index, documentation, and live-spec
checker synchronized with the code.

- [Pinned Spongefish codecs](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/spongefish/src/codecs.rs)
- [Pinned Spongefish domain separator](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/spongefish/src/domain_separator.rs)
- [Pinned Spongefish duplex sponge](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/spongefish/src/duplex_sponge.rs)
- [Pinned Spongefish digest bridge](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/spongefish/src/instantiations/hash.rs)
- [Pinned Spongefish argument I/O](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/spongefish/src/io.rs)
- [Pinned Spongefish verifier](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/spongefish/src/narg_verifier.rs)
- [Pinned Spongefish prover](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/spongefish/src/narg_prover.rs)
- [Pinned Spongefish PoW](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/pow/src/lib.rs)
- [Current transcript](../book/src/how/transcript.md)
- [Current grinding contract](transcript-grinding.md)
- [Verifier contract](../docs/verifier-contract.md)
