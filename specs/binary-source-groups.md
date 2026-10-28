# Spec: binary source groups

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-10-27 |
| Status | design |
| PR | Pending |
| Supersedes | None; existing LaBinius entry points remain live |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

## Summary

Binary sources need to share an authenticated source and a caller-owned
transcript with ordinary field sources. This design makes a binary source a
commitment-group kind in one ordinary Akita grouped opening. It replaces the
standalone binary root reduction and nested PCS invocation when explicitly
selected. It returns separate binary and prime readings; their agreement
requires a proof supplied by the application.

This is a **design**, with protocol implementation and certification pending.
The source contract API supplies only the descriptor, addressing, cap validation
and agreement interface. It does not select or execute a new prover/verifier
path. The checkpoint soundness argument belongs in
`specs/binary-source-checkpoint.md`, which supplies the separate checkpoint
proof. This specification states the protocol contract, not a completed security
proof or production admission.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are interpreted as
specified by BCP 14, RFC 2119 and RFC 8174, only when capitalized.

## Intent

### Goal

Represent dense binary sources as opt-in groups in the ordinary grouped fold,
with an exact public index map and an explicit conditional agreement contract.

The descriptor lives in `akita-types`, behind the existing default-off
`labinius-trinomial` feature. It describes a protocol object used by prover,
verifier and callers, rather than selecting a runtime configuration or
certifying a parameter set. Existing feature and crate names are retained.

### Invariants

- Enabling a Cargo feature MUST NOT select the extension. With it unselected,
  setup, commitment and proof bytes and transcript state MUST remain identical
  to the existing protocol, including with all extension features enabled.
- Binary and ordinary groups MUST use one caller-owned transcript and one
  ordinary grouped opening. A nested proof, nested session seed or separate
  binary grinding domain MUST NOT be introduced.
- Every source coefficient MUST have one canonical owner and address. Payload,
  committed padding and MLE-only virtual positions MUST remain distinguishable.
- All committed padding MUST be zero under the agreement contract. Virtual
  positions are zero by definition and MUST NOT become witness coefficients.
- The same canonical descriptor bytes MUST be used for transcript absorption
  and descriptor comparison. Parsing MUST reject unknown identities,
  unsupported widths, inconsistent counts, noncanonical values and trailing
  bytes using typed errors, without panics or unchecked allocations.
- Group manifests MUST bind polynomial ownership and the plane budget before
  any fold challenge. Duplicate prime points MUST share an admitted plane for
  the same source owner and tensor component.
- An agreement proof MUST bind this session, group, polynomial manifest,
  descriptor/index map, dimensions and terminal prime claims. A boolean flag
  MUST NOT discharge the agreement contract.

### Non-goals

The source contract API does not wire a root relation into proving or
verification, add arithmetic kernels, certify q24 SIS, implement a challenge
sampler or setup stream, or supply an application proof. Akita supplies no
booleanity, zero-support or prime-claim-fusion helpers. Unsupported
non-power-of-two word widths are rejected rather than implicitly repacked. The
existing LaBinius implementation continues to be described by its live
specifications until a later migration.

## Evaluation

### Acceptance criteria

- [ ] For every width 1, 2, 4, 8, 16, 32, 64 and 128, exhaustively round-trip
  small source shapes through both directions of the index map.
- [ ] Check both signs, partial last carriers and ring elements, padding words,
  full-block padding and every committed/virtual boundary in both directions.
- [ ] Verify the explicit permutation against the existing column-contiguous
  source packing kernel and publish the claim-variable split.
- [ ] Accept five admitted prime planes, reject six, and reuse identical points
  without granting extra planes to another polynomial or tensor component.
- [ ] Accept the canonical image window `[-7,126,704, 7,126,704]` and enforced
  digit interval `[-9,586,980, 7,190,235]`; reject off-by-one substitutions.
- [ ] Round-trip canonical descriptor encodings; reject each unsupported field,
  bad dimensions/counts, truncation and trailing bytes with typed errors.
- [ ] Reject agreement tokens for another session, group, polynomial manifest,
  index map, dimensions or terminal prime claims.
- [ ] Pass documentation guardrails and required feature/check matrices.

### Testing strategy

Descriptor and addressing tests live with `akita-types` and run with its default
features and with `labinius-trinomial`. Test doubles exercise the integration
trait; Akita MUST NOT ship a production implementation of that trait.
Integrating the root ring switch and grouped opening MUST add ordinary-byte
stability, mixed-group transcript replay, malformed input and pending-claim
consumption tests. Existing ordinary and LaBinius tests continue to apply to
their respective live paths.

Run `./scripts/check-doc-guardrails.sh` for this specification. Use the current
workflow and repository instructions for the workspace feature matrices.
Schedule artifacts MUST NOT change merely because descriptor types exist.

### Performance

Dense packing uses all 128 payload coordinates of a carrier, including for bit
and byte sources. The point cap limits source-plane growth before allocation.
The source contract API makes no performance or proof-size measurement claims.
The design keeps an explicit linear verifier weight term in the number of
carriers per column; it does not claim a logarithmic online verifier. All q24
ranks, digit depths, suffix dimensions and proof sizes require later admission
and measurement.

## Design

### 1. Decision and architecture

Select a small commitment modulus q, a degree-648 trinomial binary root, and
ordinary balanced base-8 digits. Let `s_b` be the source vector in column b.
The inner image is `T_b = A s_b mod q`. The public commitment is the ordinary
outer commitment `B_outer digit8(T_all)`, with optional ordinary compression.
The image table is a private hint, not another publicly opened commitment.
A acts modulo q; the outer commitment and subsequent ordinary PCS act modulo P.

The selected root uses ordinary Stage 1 range-image checking and one fused
Stage 2. Typed Stage 3 authenticates its setup contributions. After the root,
bounded coefficients are repacked into an ordinary power-of-two negacyclic
witness. This is authenticated coefficient repacking, not a ring homomorphism.

The final design replaces the A quotient by seven linked integer projections,
reduces parity kernels before checking them, and commits both parity carries K
and K_val. The U view and all prime partial planes EP are committed behind D
before the column challenges. Direct binary equality weights are the selected
profile. A committed A quotient MAY serve as a separately admitted migration
path, but it does not alter the transcript ownership contract.

### 2. Fields, source layout and descriptor

Use the following notation:

| Symbol | Meaning |
| --- | --- |
| P, F | `P = 2^128 - 275`, `F = GF(P)`; this profile must be selected explicitly |
| E | F in the planning performance profile; a certified extension requires separate admission |
| H | `GF(2^128)` with modulus `X^128 + X^7 + X^2 + X + 1` |
| B | `GF(2^162) = F2[Z]/(Z^162 + Z^81 + 1)` |
| S | `Z[Z]/Phi`, `Phi = Z^162 + Z^81 + 1` |
| R | `Z[Y]/Psi`, `Psi = Y^648 - Y^324 + 1` |
| ell, k | Word width and logical words per carrier: `k = 128/ell` |
| N, C, M, m | Padded carriers, columns, scalar rows per column `N/C`, ring rows per column `N/(4C)` |

The embedding is `iota(Z) = -Y^4`; R is a free rank-four S-module. The symbol k
here always counts logical words per carrier, never the four S-components of R.
The binary scalar map uses this module decomposition and reduction of S to B.
In particular, `R/2R` MUST NOT be treated as a field.

The source descriptor MUST bind:

1. Source kind and word basis; ell, logical word length and padded carrier count.
2. Polynomial count and fold-column count, which determine source dimensions.
3. q and the exact Psi identity.
4. Signed packing map, coordinate order and the explicit existing-kernel
   permutation.
5. Canonical image representative window and full enforced digit interval.
6. A and outer-B commitment profile identities.
7. Challenge-family identity and response interval.
8. Prime-point/plane cap and the agreement contract identity.

ell MUST be a power of two in `[1,128]`; other widths produce a typed error.
The logical length is nonzero and counts words. N and C are powers of two,
N is at least `4C`, and the logical words fit within `N*k`. Every size and offset
MUST use checked integer arithmetic. Trusted admission limits, not proof bytes,
select allocation limits, relation degrees, setup slots and projection counts.

There is one canonical byte encoding of the descriptor, in this order:

| Encoding | Content |
| --- | --- |
| Fixed bytes | ASCII `AkitaBinarySource`, then version byte `0x01` |
| Six bytes | Source, basis, Psi, signed packing/permutation, challenge family and agreement identities; each is `0x01` for this profile |
| Five little-endian u64 integers | ell, logical word length, padded carriers N, columns C, polynomial count |
| Little-endian u32 | q, exactly 14,253,409 |
| Six little-endian i32 integers | Jq lower/upper, I8 lower/upper, response lower/upper endpoints |
| One byte | Prime cap, exactly 5 |
| Two 32-byte identities | A commitment profile, then outer-B commitment profile |
| Ordered 32-byte identities | Exactly polynomial-count distinct polynomial identities |

The packing identity includes the low-column map and explicit column-contiguous
kernel transposition. The challenge identity denotes the selected degree-162,
weight-at-most-46 family with unbiased support sampling and `Shake256V1`
deterministic signs; the agreement identity denotes signed
payload bits and zero committed padding. These identify contracts; they do not
select runtime behavior or certify the profile. A/B identities are opaque
bindings to the chosen admitted profiles.

Alternative encodings of the same semantic value MUST be rejected. A parser
MUST validate every field, all dimension relations, unique polynomial identities,
exact polynomial count and exact input exhaustion. Constructed descriptors and
parsed descriptors obey the same validation. Unknown tags and off-profile
fixed parameters are errors, not extension negotiation. The transcript absorbs
exactly these bytes, without a second representation.

The opening manifest additionally binds ordered claim points and values,
linear bit weights, polynomial/group ownership and the selected fold/setup
schedule. These are opening-specific; they MUST NOT be inferred from buffers.

### 3. Planning parameters, not certification

The proposed q24 profile uses

```text
q = 14,253,409 = 1 + 23,328 * 13 * 47
Jq = [-7,126,704, 7,126,704]
S8 = (8^8 - 1)/7 = 2,396,745
I8 = [-4*S8, 3*S8] = [-9,586,980, 7,190,235]
```

q is prime and is the largest prime congruent to 1 modulo 23,328 whose centered
window fits I8. The centered-window cap is `2*3*S8+1 = 14,380,471`.
There are 141 primes below `2^24` in this congruence class; larger candidates
can fit a shifted window but cannot use this centered one. This profile fixes
Jq and I8 exactly; accepting a nearby endpoint would change the contract.

For `x` in I8, `y = x + 9,586,980` lies in `[0,2^24)`, and
`digit_i = ((y >> (3*i)) & 7) - 4` for `i=0..7`.
The image can therefore be stored as one three-byte offset word per coefficient.
The offset is even, so its low bit is the parity of x. Verification enforces
I8 through the eight digits, not canonical membership in Jq. Bounds MUST use
malicious image magnitude `BT = 9,586,980`, while A's genuinely centered
coefficients have bound `A_bound = 7,126,704`.

The comparison q25 profile has `q0 = 33,568,993` and nine digits, with magnitude
`4*(8^9-1)/7 = 76,695,844`. At C=256, q24 rank 3 has 3,981,312 live image digit
cells and 6,291,456 coefficient-padded cells; rank 4 has 5,308,416 and 8,388,608.
q25 rank 3 with nine live digits and sixteen padded digit slots uses 4,478,976
live cells and 12,582,912 physical cells. These are layout counts, not certified
rank choices. Old-q certificates do not certify q24.

The planning challenge uses scalar degree 162 and support weight at most 46,
with deterministic signs. Its support count is
`sum_(i=0)^46 binom(162,i)`, with log2 approximately 136.3002783091.
Deterministic signs add no entropy; independent random signs are not selected.
Weight 46 gives `Gamma=92`. At C=256, four response digits enforce `[-2340,1755]`, diameter
4095, and comparison radius `4*Gamma*4095 = 1,506,960`.
The sizing formula gives tail caps 1266 and 1303 for N=`2^22` and `2^23`.
Each changed width, geometry, challenge family or interval requires fresh
response, no-wrap and SIS admission. Feature enablement never admits a profile.

### 4. Exact bidirectional index map

For logical word j and bit u, with `0 <= j < logical_length` and `0 <= u < ell`,

```text
i = floor(j/k)                 carrier
s = (j mod k)*ell + u          scalar coordinate, low bit first
b = i mod C                   commitment column, low carrier-address bits
r = floor(i/C)                scalar row in that column
a = r mod 4                   S-component
jR = floor(r/4)               ring row
t = 4*s + a                   ring coefficient
sign = (-1)^s
```

The stored coefficient is `sign * bit(j,u)`. Conversely, for a valid padded
ring position `(b,jR,t)`, recover

```text
s = floor(t/4)
a = t mod 4
r = 4*jR + a
i = C*r + b
j = k*i + floor(s/ell)
u = s mod ell                 when the position is payload
```

The sign is derived from s and MUST NOT be caller-controlled. Classification
precedes a claim that j,u are payload:

| Position | Classification |
| --- | --- |
| `s < 128` and `j < logical_length` | Payload |
| `s < 128` and `j >= logical_length` | Committed padding |
| Scalar coordinates `128..161` | Committed padding |
| Scalar coordinates `162..255` | MLE-only virtual |
| Ring-coefficient slots `648..1023` | MLE-only virtual |

Here all ranges are inclusive. Low-coordinate slots in the partial last
carrier, trailing components of a partial last R element, padding words,
and padding to a full `4C` carrier block are committed padding. They participate
in the commitment and MUST be zero under the agreement contract. Virtual
classification takes precedence even for an otherwise unused padded carrier.
Invalid positions outside the admitted padded shape produce typed errors.

The claim-variable split is public: the first `log2(k)` logical address
variables select a word inside the carrier, and the remaining variables select
the carrier. A word contributes `log2(ell)` bit-coordinate variables. Thus
`log2(k)+log2(ell)=7`; intra-carrier coordinates feed field-switch weights.
Small source shapes retain their true logical length and explicit zero mask.
For a full `2^n'` bit table before small-size padding, carrier log is `n'-7`.

The existing packing in `akita-labinius-verifier/src/source.rs` uses the same
`t=4s+a` and alternating sign, but stores columns contiguously. For canonical
carrier i its existing-kernel address is

```text
h = (i mod C)*M + floor(i/C)
i = (h mod M)*C + floor(h/M)    inverse
```

The descriptor MUST carry this transposition explicitly. Carrier tables and
point coordinates MUST use the same permutation. No new coefficient/sign
permutation is needed. The low-coordinate convention is consistent with
[packed F162 arithmetic](labinius-packed-f162.md), whose folds eliminate the
least-significant table-index bit first.

### 5. Claims and prime-plane cap

A binary group admits at most five distinct prime points. Each point is
compared canonically; duplicate claims at an identical point share a plane
when their source polynomial and tensor component coincide. Their claimed
values remain bound individually and conflicting duplicate values are invalid.

For a separable functional, the prime view uses
`EP_b(Z) = sum_r e_r src_(r,b)(Z)`, fixed behind D before the column challenge.
The full response, rather than its lowest digit, enters the prime relation.
Each base-field plane uses 43 balanced digits per scalar coefficient. At C=256
this is `43*162*256 = 1,783,296` live cells, or `43*256*256 = 2,818,048`
coefficient-padded cells. Five planes cost 14,090,240 such physical cells.
An extension E/F adds its coordinate planes; tensor rank adds one plane per
component. A general four-component functional can require four components.

The group manifest MUST state polynomial owners and all plane multiplicities.
The group manifest adopts a conservative aggregate budget: both distinct prime
points and unique `(polynomial, point, tensor_component)` planes MUST be at most
five. Tensor components are numbered 0 through 3; polynomial indices must belong
to the descriptor's ordered manifest. Duplicate requests for that exact triple
reuse the existing plane. Distinct owners or components consume distinct planes
even at identical points. No polynomial receives a separate five-plane quota.
Validation returns the shared plane index for each request in manifest order.

Prime-plane validation bounds input at 4096 requests and rejects excess before
allocation. Points are canonical base-field coordinate vectors; wrong point
counts or coordinates outside `[0,P)` are typed validation errors. Different
polynomials and tensor components MUST NOT receive implicit free planes through
point deduplication. Both cap violations are typed errors. These conservative
bounds are API admission limits, not a certificate. Production admission MUST
certify D and every widened ordinary instance for the declared maximum. A caller
needing more points supplies its own reduction or selects a separately certified
profile.

When one polynomial has several binary claims, reduce them in H before field
switching using independent H batching scalars and a degree-two sumcheck.
For n carrier-address variables this costs `32n+16` bytes, including the
terminal source value. Authenticate that value even when the public multiplier
is zero; do not divide by the multiplier. Then send one array of 128 H
partials, 2048 bytes, fixed before seven B row-selector challenges. One B
frontend sumcheck leaves the source terminal linked by K_val. Claims owned by
different polynomials retain distinct terminals and ownership.

### 6. Conditional agreement output

For each polynomial and column, the checkpoint extracts a relaxed rational
source `f* = a/iota(s)` from response and challenge differences. The binary
reading `Dec_2(f*)` and coefficientwise prime reading `Dec_P(f*)` are separately
authenticated. E-valued challenges do not change the base field P in this
contract. Equality of the readings is conditional on an application proof
about this extracted source, not an honest input buffer or the folded response.

The sufficient contract requires signed payload coefficients in `{0,(-1)^s}`
and every unused committed coefficient zero. Equivalently, undoing the packing
sign yields payload bits in `{0,1}` and zero committed padding. It covers
scalar coordinates 128 through 161, partial elements, padding words and full
block padding. Virtual coefficients require no proof because they are zero by
definition. Payload booleanity alone is insufficient.

The application proves its condition through prime terminal claims consumed
by this same grouped opening. The agreement lemma then lifts the prime
relation to the integers: in the planning profile,
`2*Gamma + Delta = 2*92 + 4095 = 4279 < P`.
The full argument, extraction hypotheses and composed error ledger belong to
`specs/binary-source-checkpoint.md`.

The output exposes separate readings and an `agreement_requires(proof_token)`
operation. The proof token implements an integration trait and binds the
session, group, polynomial manifest, canonical descriptor/index map, source
dimensions and terminal prime claims. The operation MUST reject any mismatch
with a typed error. Akita supplies the trait and checks bindings, but MUST NOT
supply a production implementation of the application's proof token. Test
doubles do not constitute a proof implementation. A consumer MAY use either
reading without invoking conditional agreement.

### 7. Setup, commitment and normative message order

Setup admission fixes q, rank/width certificates, digit depths, challenge
family, point/plane cap, relation bounds and the complete ordinary suffix.
A uses uniform rejection sampling modulo q and centered coefficients. P setup
streams use exact P sampling with distinct typed domains. Typed setup prefixes
are authenticated in advance; preparation may scan A once, online verification
MUST NOT scan the root A matrix. Commit using the packing above, centered T
representatives and ordinary outer B. The public object is the ordinary
commitment plus its manifest; T remains a private hint.

| Order | Message or action | Fixation requirement |
| --- | --- | --- |
| 0 | Bind caller session, all commitments, shapes, maps, claims and schedules | One session and explicit source ownership |
| 1 | Application-owned reductions and optional H multiple-point reduction | Each sumcheck message precedes its challenge; terminals remain pending |
| 2 | One field-switch partial array and B frontend rounds/terminal | Partials precede row selector; terminal awaits this fold |
| 3 | Ordinary D commitment to U, every EP plane and other groups' segments | Before any column fold challenge |
| 4 | Ordinary response nonce and indexed per-group column challenges | Binary family only for selected binary groups |
| 5 | Outgoing ordinary witness commitment to Z/T digits, U/EP views, K, K_val and ordinary segments | All tables fixed before relation challenges; no equality tree |
| 6 | Successor-source checkpoint, fresh seven-row projection seed J, then seven signed outputs y | Candidate source fixed before J; all y precede output batching |
| 7 | Output batching, ring switch and row batching | Challenges see every message they test |
| 8 | Ordinary Stage 1 range-image sumcheck | Balanced alphabet and quartic leaves |
| 9 | One fused Stage 2 for projection link, B/D, both parity/value identities, prime relations and range binding | Public direct weights |
| 10 | Typed ordinary Stage 3 setup product and prefix claims | No online A scan; prefix claims join next level |
| 11 | Ordinary grouped recursion and small terminal verification | Consume witness/prefix claims; return live caller transcript; reject trailing bytes |

The successor-source checkpoint MUST sit immediately before J at order 6,
replacing the ordinary checkpoint before ring reduction. Its candidate is the
source of the outgoing commitment, with canonical setup sources and fixed
view/decoding maps. It is not an adaptively chosen residual after J. This is
the placement required by Theorem 8 (mixed-root checkpoint) of
`specs/binary-source-checkpoint.md`; moving the existing checkpoint preserves
the successor checkpoint count. Keeping only the later ring-reduction
checkpoint does not instantiate that theorem.

K_val is committed at order 5 even if available earlier. Projection seed J is
fresh after response fixation and MUST NOT be reused across nonce trials.
Binary-only openings omit EP; prime-only openings omit the binary frontend,
U, K and K_val. Subsequent levels use ordinary rings and challenge families.
No unconsumed setup prefix claim may remain at the terminal.

### 8. Verifier work

The verifier validates descriptors, manifests, points and plane counts before
allocating source-dependent structures. It replays the shared transcript and
frontend, checks canonical projection integers and their divisibility by q,
then authenticates their links through the ordinary fused stages.

For the projected A relation, use the exact reduced multiplication adjoint.
If rho denotes projection batching and g its signed-Psi extension,
`h_i(s) = sum_t eq(r_t,t)*g_i(s+t)` for `s,t < 648`.
The setup weight factors into a response-row equality weight and h. Typed q
coefficients embedded in P and ordinary P setup have distinct authenticated
prefix streams. This requires an exact adjoint implementation, not reuse of an
unverified negacyclic evaluation map.

There is an arithmetic correction to the design's convolution-size estimate:
g has `2D-1 = 1295` entries and the reversed response weight vector has D=648.
Their **full convolution has `3D-2 = 1942` entries**; the needed h is its middle
648 entries. Compute it as **two length-1295 products**, as proved in Theorem 7
of `specs/binary-source-checkpoint.md`: split g into its first D entries g0 and
remaining D-1 entries g1, pad g1 with a final zero, and reverse the D response
weights to v_rev. Then

```text
h(s) = (g0 * v_rev)[D-1+s] + (g1 * v_rev)[s-1],
```

with the second term defined as zero for s=0. Each product has length
`2D-1 = 1295`. A length-1024 padded Karatsuba implementation of each product
uses `2*3^10 = 118,098` recursive scalar products before overhead; the
single-product figure 59,049 is not a bound for this algorithm. A faster
specialized middle product requires its own analysis.

Binary row equality weights and their 162-bit expansions are evaluated
directly. Grouping rows in blocks of eight gives an addition budget
`(M/8)*(256+162)` when M is divisible by eight, plus M B-field weight operations.
For N=`2^22/2^23`, C=256, the additions are 856,064/1,712,128. K_val adds
`C*162 = 41,472` expanded bits. The online verifier is linear in M=N/C, with
additional adjoint, frontend and suffix work. These operation counts are not
benchmarks or a logarithmic succinctness claim.

### Alternatives considered

Using P as the inner commitment modulus loses the efficiency purpose of a
small q. Keeping the old standalone nested PCS duplicates source and session
boundaries. Treating rational-source readings as unconditionally equal omits
the application's source relation. An equality tree introduces a large witness
and a new compiler obligation; direct weights retain a measurable linear term
instead. Larger fold widths, heavier supports and more prime planes require
full joint re-admission rather than inheriting planning bounds.

## Documentation

This is the normative design record for the binary-group contract. It does not
supersede live-code descriptions in the LaBinius specs. The checkpoint proof is
separate in `specs/binary-source-checkpoint.md`. After implementation and review,
fold durable explanations into the Book's architecture, commitment, transcript
and verification chapters, update lifecycle metadata and archive obsolete root
specs under the documentation policy. No unpublished paper is needed to use
this contract.

## Execution

The mixed-ring root relations and root ring switch must use the descriptor's
exact coefficient/index maps and group ownership when repacking into ordinary
rings. The grouped opening that consumes this descriptor must bind canonical
bytes and claim manifests at transcript order 0, construct one ordinary grouped
proof, consume all terminals and preserve the caller session. Further protocol
implementation supplies carries, projections, setup offloading, plane support
and certified planning. Certification and the written checkpoint are release
gates even when arithmetic and descriptor tests pass.

## References

- [Branch and extension contract](../docs/branches.md).
- [Crate ownership](../docs/crate-graph.md) and
  [documentation policy](../docs/documentation.md).
- [Live clear opening](labinius-clear-opening.md),
  [field switch](labinius-field-switch.md) and
  [packed F162 arithmetic](labinius-packed-f162.md).
- [Live root admission](labinius-root-admission.md),
  [setup contract](labinius-setup-contract.md),
  [lowered root](labinius-lowered-root.md),
  [root sumcheck](labinius-root-sumcheck.md),
  [root reduction](labinius-root-reduction.md) and
  [PCS](labinius-pcs.md).
- `specs/binary-source-checkpoint.md`, Theorems 7 and 8 (checkpoint proof, integrated separately).
