# Spec: several binary and ordinary groups in one opening

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-11-05 |
| Status | approved |
| PR | |
| Supersedes | One-binary-group and at-most-one-ordinary-group root restriction, upon implementation |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

## Summary

Open several independently committed binary and ordinary groups in one Akita
opening. Use a vector of binary roots, one D commitment per binary group and
complete ordinary preimages. Concatenate their digit views into one outgoing
witness. Batch every coefficient quotient and other root row once with
independent E64 weights. Authenticate all setup-dependent terms with one Stage 3
product claim against one composed typed setup prefix. Pass the witness and
prefix evaluations to one ordinary successor fold and one terminal. This is an
approved design awaiting implementation and production admission.

This spec builds on the approved [64-bit root](binary-source-p64-root.md),
including its [column-challenge proof of
work](binary-source-p64-root.md#column-challenge-proof-of-work). It also
requires the one-prime-point change described below. The intended readers know
the existing binary-source opening, checkpoint and setup-offload specifications.

## Intent

### Goal

Extend the 64-bit binary root to an ordered roster of binary and ordinary group
occurrences in one caller-owned transcript.

A group occurrence is one commitment and its ordered polynomial owners in this
opening. A root is the reduction from those source claims to the outgoing
ordinary witness. A typed setup prefix is a committed table of public setup
coefficients, separated by role and authenticated by the successor. The group
plan and row plan belong in `akita-types`. Setup preparation and source
computation belong in `akita-cpu-backend`. The prover and verifier execute the
same versioned plan.

### Invariants

- Bind every group, owner, claim, setup, layout and schedule at order 0.
- Fix every D before any binary column challenge at order 3.
- Use one local response search and one priced column-challenge site per binary group.
- Commit every accepted response and coefficient quotient in one outgoing witness.
- Use one independent E64 row batch with no outer group multiplier.
- Run one Stage 1, one Stage 2 and one Stage 3 for all groups.
- Prepare one composed prefix commitment and consume its single evaluation
  with the witness evaluation in one ordinary successor.
- Keep original commitment setup labels and local source maps intact.
- Admit only the single-group geometry envelope `m_b <= 8192`, `C_b <= 256`.
- Preserve setup, commitment, proof bytes and transcript state when the extension
  is unselected, including when its Cargo feature is enabled.
- Reject malformed encodings, unsupported profiles and checked-arithmetic
  failures with typed errors before allocation or indexing.

The transcript, row-oracle, layout and tampering tests in Evaluation protect
these invariants. No Rust implementation is included in this spec change.

### Non-Goals

This change does not add separate openings, a shared D, ordinary response
partials, extra Stage 2 executions, repeated coefficient tests or extra
terminals. It does not admit several prime points per group. It does not admit
`m_b = 16384` or `32768`. Those geometries need a later
carry, response-acceptance and q24 SIS change. It does not supply the
application's conditional agreement proof.

## Evaluation

### Acceptance Criteria

- [ ] Round-trip the canonical order-0 record and reject every noncanonical variant.
- [ ] Exercise two and four binary groups with zero, one and several ordinary groups.
- [ ] Replay all frontends and enforce the all-D barrier before response work or draws.
- [ ] Change one group's nonce and retain every other group's challenges.
- [ ] Check exact `g_b` thresholds, zero-bit omission and work replay for each group.
- [ ] Compare every translated witness/setup weight with a dense E64 reference.
- [ ] Include every group's rows once in the common batch and both shared sumchecks.
- [ ] Reject each commitment, owner, point, value, D, nonce, row-range and setup mutation.
- [ ] Consume the outgoing witness and composed prefix claims through one terminal and exact EOF.
- [ ] Pin inactive-feature bytes, unchanged singleton local layouts and new-version singleton bytes separately.
- [ ] Reject `m_b > 8192` before proof execution and reject uncertified selected schedules.
- [ ] Pass both exact evidence scripts and documentation guardrails.

### Testing Strategy

Extend `root_plan_tests.rs`, `typed_prefix_tests.rs` and the CPU Stage 2 oracle
tests with heterogeneous group geometries, setup seeds, ranks and owner subsets.
Test non-power-of-two roster sizes and offsets that cross power-of-two
boundaries. Use genuinely non-base E64 points for every adjoint and translated
weight test. Test the one-point API with word-only, prime-only and both claim
kinds. Test each group's early and late response retries and sampler exhaustion.
Test all `G_b in {1,2,4}` and `G_o in {0,1,3,4}` combinations end to end. Check
transcript continuation, complete grinding-plan consumption and exact EOF. Test
full-field ordinary recomposition separately from signed-width linkage. Check
count overflow, aggregate caps, truncated payloads and work-nonce canonicality.
Keep the existing single-group and ordinary compatibility suites passing.

For this documentation change, run:

```bash
python3 -I specs/evidence/binary-source-multigroup/bounds.py
python3 -I specs/evidence/binary-source-p64-root/bounds.py
scripts/check-doc-guardrails.sh
```

The implementation PRs also run the repository's cheap gates before compilation,
then the path-specific workflows and required Clippy/test feature graphs. Use
the current workflow commands for final validation.

### Performance

The cost model below separates per-group work from the one shared execution. The
evidence script checks all table counts and exact work-bit prices. Complete
proof bytes, latency and peak memory require measurements with the selected
aggregate schedule. The historical calibration is a planning model. It is not a
p64 multigroup benchmark.

## Design

### Architecture

The current binary root drivers in `akita-prover` and `akita-verifier` select
the first binary group and at most one ordinary group. `OrdinaryNativeRootBatch`
is an arithmetic compiler seam for ordered ordinary rows; the drivers do not
select it. This extension replaces that fixed dispatch with one validated group
plan. Each binary group produces local frontend/D/response data. All groups feed
the shared relation and prefix compilers described below.

### Public statement

#### Public objects and meaning

Write `b = 0..G_b-1` for binary groups and `o = 0..G_o-1` for ordinary groups.
Canonical opening order is all binary groups, then all ordinary groups. Within
each kind, retain the application's declared group order; do not sort by
commitment bytes or setup identity. A group index is a session-local occurrence,
not a polynomial identifier or a setup seed. Commitments may have been made at
different times, under different setup identities. No group is recommitted
merely to join this opening. All commitments and claims are fixed at order 0.
Commit-event identifiers refer to the caller's existing commitment history; they
are not claims about wall-clock time. The caller's transcript remains the
session binding, including that history when the application requires it.

A binary group contains:

* Its immutable descriptor, ordered `s_b` polynomial identifiers, commitment
  payload, A/B setup identities and carrier/suffix compatibility profile. `N_b =
  4m_b C_b` padded carriers **per polynomial**. Its q image has rank three in
  `Z_q[Y]/(Y^648-Y^324+1)`, with `q = 14,253,409`.
* Either or both of: one word claim on one owned polynomial, over
  `H = GF(2^128)`; one common prime point `x_b` with a nonempty ordered
  subset of owned polynomials and one value per selected polynomial. Let `w_b`
  be the word-presence bit and `t_b` the number of claimed prime polynomials.
  Require `w_b + [t_b>0] >= 1`. Each claim owns `degree` base-coordinate EP
  planes (two for P64, one for P128) and contributes two Stage 2 identities
  (over E64 for P64), as in the [64-bit root](binary-source-p64-root.md#prime-planes). No extra
  tensor-component choice or alternate point survives into this API.
* The opening profile: quotient/carry depths, D rank and typed D setup label,
  local typed prefix layout, response family and interval, nonce policy,
  source-reading and conditional-agreement contract, and setup-prefix slot. D is
  an opening commitment, not a replacement for its earlier B commitment.

A P64 binary prime point and its claimed values are over E64. Its dimension and
reading map are those fixed by the one-point API, validated against its
descriptor; this profile caps the dimension at 32. Its two base-coordinate EP
planes recombine with basis weights 1 and u in each identity. The
GF(2^128) word point has `log2(N_b)+log2(128/ell_b)` coordinates, at most 32.
The GF(2^162) terminal is produced internally, never supplied as an initial
claim. The word contract remains a relaxed source reading; equality with the
prime reading requires the existing application proof of signed bits and zero
non-payload coefficients, qualified by group and owner.

An ordinary group contains its commitment and complete immutable p64 profile,
setup identity, ordered `s_o` polynomial identifiers, a single point in E64, and
exactly one evaluation per polynomial at that point. Its coefficient cube has
`n_o = 2^v_o` coefficients per polynomial. Complete source, A-image and
compression preimages enter the outgoing witness. The point has exactly v_o
coordinates. There is no ordinary response challenge at this root and no
per-ordinary-group D: no ordinary partial is used. The already-fixed ordinary
commitment binds those preimages. Its setup-dependent A/B/compression rows
nevertheless join Stage 3. This retains the current mixed root method,
generalized to multiple groups and owners.

Use `p = 18,446,744,073,709,551,557`, `E64 = F_p[u]/(u^2-2)`, and `e = p^2 =
340,282,366,920,938,461,286,658,806,734,041,124,249`. Testing and sumcheck
challenges use E64; setup coefficients and commitments use F_p. A's centered q
coefficients embed into F_p. The binary field switch continues to use H and `B =
GF(2^162)`.

#### Canonical order-0 encoding

Define a new explicit profile/version, `akita/binary-source/multigroup/p64/v1`.
No inference from proof length or group count selects a legacy parser. Use
little-endian u64 for counts, lengths, offsets, profile enum integers and
indices; a tag/option bit is exactly one byte, 0 or 1. A variable blob is `u64
byte_length || canonical_bytes`. All decoding checks limits before reserve or
iteration. Public fields need not be retransmitted as proof data. Use kind tag 0
for binary and 1 for ordinary. The global indices are `b` for binary and `G_b+o`
for ordinary. Every list starts with its u64 item count, including
point-coordinate lists. Counts prescribed by the descriptor still appear and
must match it. Version strings and the opaque caller context use the blob
encoding. Commit-event, polynomial, setup, schedule, admission and prefix
identities are fixed 32-byte values under their selected canonical identity
codec. Typed role labels and complete descriptors/plans use length-prefixed
canonical bytes from the selected version. Their field order is part of that
version. Their decoder rejects unused fields and noncanonical internal
encodings.

Absorb this record, in order:

1. Version/domain, caller context binding, p and the E64 polynomial identity,
   complete selected root/successor schedule identity, resource/admission
   profile identity, and the composed prefix identity and commitment slot.
2. `G_b`, `G_o`, then the ordered occurrence records. Each starts with its
   global index, kind tag, commitment-event identity, descriptor/profile blob,
   setup identity/typed role labels and ordered polynomial identifier list.
   Identifiers are fixed 32-byte values. Binary descriptor bytes are the new
   one-point/p64 descriptor, not a reinterpreted old P128 descriptor.
3. Each occurrence's commitment: role/profile binding, prescribed coefficient
   count, then canonical base coefficients as eight-byte residues `<p`. For
   binary B, the count is `64 r_B,b`. A rank/ring is never taken from an
   unvalidated payload length. Ordinary counts come from the frozen profile.
4. For each binary occurrence: word presence, and if present its owner index,
   full H point and value; prime presence, and if present one E64 point with its
   prescribed dimension, `t_b`, and strictly increasing owner indices with their
   E64 values. Reject repeated owners, conflicting values, duplicate requests
   and empty presence records. H elements are 16 little-endian polynomial-basis bytes; F_p
   elements are eight.
5. For each ordinary occurrence: prescribed point dimension and E64 coordinates,
   prescribed evaluation count and values, in owner order. E64 encoding is two
   canonical F_p residues `(a,b)` for `a+b*u`, 16 bytes. Full-field coefficient
   semantics or a selected signed-width contract is explicit.
6. The derived outgoing layout, all row-range boundaries, local prefix layouts
   and placement map, quotient/carry depths, source-reading maps, and complete
   grinding/address plan, each as a canonical length-prefixed plan blob in the
   listed order. Include a count before each range or placement list. They are
   recomputed and compared, never accepted as a proof-supplied layout.

The transcript receives every byte above, with typed message boundaries. Steps 2
and 3 each traverse all occurrences; claim lists then follow in steps 4 and 5.
Do not interleave the commitment payloads into the occurrence records. Absent
options have no following body. Alternative residue encodings, trailing bytes
and unknown tags reject. Separate groups may use the same polynomial name only
as the same explicitly registered commitment occurrence; otherwise use globally
unique polynomial IDs. The implementation's existing duplicate word-claim rule
remains in force. Setup labels retain each original setup's namespace: putting a
group at index b must not change how its earlier B was computed. The opening
index enters the composite identity and challenge addresses, not a retrospective
commitment key derivation.

#### Admission and exact integer bounds

There are two gates: the resource and arithmetic envelope below, and a selected
schedule security certificate. Passing the first does not imply the second. A
deployment may select smaller trusted limits.

| Item | Resource bound |
|---|---|
| Groups | `1 <= G_b`, `0 <= G_o`, `G_b+G_o <= 64` |
| Owners | `1 <= s_g <= 4096`; `sum_g s_g <= 4096` |
| Binary dimensions | `m_b,C_b` powers of two; `1 <= m_b <= 8192`, `1 <= C_b <= 256`; `N_b=4m_b C_b <= 2^23` |
| Binary input words | `ell_b` power of two in `[1,128]`; `1 <= logical_len_b <= N_b*(128/ell_b)` |
| Total input volume | `sum_b s_b N_b <= 2^28`; `sum_o s_o n_o <= 2^28`; `v_o <= 25` |
| Prime claims `t_b` | `0 <= t_b <= s_b`; `sum_b t_b <= 4096`; zero only if word present; physical planes `= degree*t_b` |
| Binary fold coordinates | `I=sum_b s_b C_b <= 4096*256 = 1,048,576` |
| Residual coefficients | `a=sum_b 1944s_b <= 7,962,624` |
| Setup/ring parameters | Binary q A rank 3/degree 648; B/D degree 64 and rank 1..5; ordinary native degrees at most 648, each profile otherwise independently admitted |
| Table domains | Derived outgoing `W <= 2^32`, derived composed prefix `Pfx <= 2^32`; both powers of two; successor must admit these actual tables |
| Metadata / proof | Order-0 metadata at most `2^24` bytes, each variable profile blob at most `2^20`; proof at most `2^24` bytes and additionally the exact selected schedule's maximum |
| Retries / loops | Accepted response nonce `0..4095` per binary group; no more than 4096 honest probes per group; each rejection sampler has its explicit finite cap |
| Suffix | At most 32 recursive folds; at most 32 variables per admitted witness/setup table; all terminal counts fixed by that schedule |

All caps are public, trusted selections, not encoded requests to allocate that
much memory. The proof parser uses the smaller of the global and schedule byte
maxima. Metadata identity records and their aggregate count are capped at `2^20`
entries and the metadata-byte limit, whichever is tighter. Ring vectors,
segments, typed pieces, row arrays and prefix slots are sized from those
validated records. A binary group has at most seven segments including its K; an
ordinary group has four basic segments and at most one signed-width bit segment,
hence at most `7G_b+5G_o <= 448` segments. Table allocations have exact length W
or Pfx; E64 storage is at most `16*2^32` bytes per such buffer, base storage at
most `8*2^32`. A verifier should stream large row families and use a trusted
memory budget before allocating even an admitted table. Its scratch capacity for
dense independent challenges is at most `2^18` elements per chunk, at most 4
MiB, and the total row count has the finite cap specified in Stage 2. Piece
metadata is checked during construction against the metadata cap.

Use checked u128 for the following bound arithmetic and checked size arithmetic
for allocations, converting to usize only after the platform/budget check. Every
checked sum/product/power/round-up failure rejects. For `d>=1`, define

```
Pos(d) = 3*(8^d-1)/7;  Neg(d) = 4*(8^d-1)/7
Bound_b = 972*m_b*7,126,704*2340 + C_b*92*9,586,980
a_b = 3*648*s_b
hK_b = floor(Bound_b/q)
dK_b = min {d >= 1 : Pos(d) >= hK_b}
Require Bound_b + q*Neg(dK_b) < p.
```

The quotient coefficient is `K_b,i = sum_j 8^j k_b,i,j`, with each digit in
`[-4,3]`. There is no claim that K has a canonical integer representation;
completeness uses hK and soundness uses the full Neg(dK). Owner counts do not
multiply this coefficientwise lift bound. At `(m_b,C_b)=(8192,256)`,
`Bound_b=132,788,824,280,985,600`, `dK_b=12`, and `Bound_b+q
Neg(12)=692,495,571,806,665,380 < p`.

Keep the existing parity depth 9 and value depth 6:

```
Rpar_b = 3*(162*(4m_b)*4 + C_b*46*4)
dpar_b = 9; dval_b = 6
Require floor(Rpar_b/2) <= Pos(9).
Require Rpar_b + 2*Neg(9) < p.
Require floor((3*162*C_b+1)/2) <= Pos(6).
Require 3*162*C_b*4 + 1 + 2*Neg(6) < p.
```

At the admitted maximum, the parity carry bound is 31,921,152 and its no-wrap
quantity is 217,233,992. The value carry bound is 62,208 and its no-wrap
quantity is 797,257. All 126 power-of-two `(m_b,C_b)` pairs in the admitted
envelope pass the exact script. These bounds use malicious accepted U digits in
`[-4,3]`.

The study also computed the following arithmetic outside admission, at C=256:

| m | N | Parity carry bound | Required parity depth | Parity no-wrap quantity | K depth | Quotient no-wrap quantity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16384 | 2^24 | 63,771,648 | 10 | 1,354,676,808 | 12 | 825,284,170,295,098,020 |
| 32768 | 2^25 | 127,472,640 | 10 | 1,482,078,792 | 13 | 5,008,808,600,008,735,396 |

Nine parity digits have positive reach 57,521,883 and fail at both rows. Ten
have positive reach 460,175,067. These are evidence-only rows. The script checks
144 arithmetic geometries including these two extra m levels. Their
response-acceptance and q24 SIS analysis belongs to a later change. This spec
rejects them even though their no-wrap quantities fit p.

Binary EP uses 22 balanced base-eight digits per F_p coefficient. Its equations
are field identities, so no `Neg(22)<p/2` condition is imposed. The one-point
change's exact functional and dimension remain authoritative.

For B/D resource and estimator selection, use

```
width_B,b = (s_b*C_b*3*8*648)/64 = 243*s_b*C_b
width_D,b = ceil((w_b*s_b*162*C_b + 22*degree*t_b*162*C_b)/64)
```

Select the least rank whose q64 `(degree=64, digit_difference=7)` width cap
covers that width; caps at ranks 1..5 are respectively `296, 76,306, 4,030,488,
106,021,444, 1,826,815,994` from the 64-bit root spec. Do not label this
estimator lookup a hardness proof. Outgoing and setup-prefix commitments need
their own actual-width certification. In particular, t_b up to s_b removes the
old five-claim cap, requiring new D admission. For `s_b=4096, C_b=256`, the
maxima are B width 254,803,968 and D width 119,439,360 (all owners opened and
word present, degree 2). Both require rank 5 in the copied width table; D
exceeds rank 4's cap 106,021,444. B ranks are already
fixed at commitment time; D ranks are selected before order 0.

Ordinary full-field sources may use 22 base-eight digits and field
recomposition; there is no implicit signed-integer range claim. An explicit
signed width w requires a separate integer link admission. A sufficient p64
condition is `Neg(d_source) + (2^w-1) + 2^(w-1) < p`, with `Pos(d_source) >=
2^(w-1)-1` and `Neg(d_source) >= 2^(w-1)`. Do not port the current P128
comment's `2^67` bound to p64. Compression bit counts and all map moduli are
those of the selected p64 commitment profile; 128-bit P128 compression tables
cannot be silently reused.

### Protocol order

The orders below are one state machine and one proof stream. Indexed draws are
an addressable challenge facility of that state machine, not separate openings
or independently seeded proof transcripts.

| Order | Messages and challenges, in exact group order |
|---|---|
| 0 | Bind the complete statement, all earlier commitments, setups, layouts and the one suffix schedule. No group may be appended afterward. |
| 1 | No library message. Application reductions have already produced the admitted single claims. |
| 2 | For each binary b with w_b=1, send its 128 H carrier partials; draw its seven B selector coordinates; run n_b=log2(N_b) quadratic B rounds, message before challenge; send its B terminal value. All these frontends finish before order 3. Group indices label every frontend atom. No frontend is needed for a prime-only group. |
| 3 | Construct all U and EP planes. Send D_0,...,D_(G_b-1), in order, each with exactly 64*r_D,b base coefficients. This is a barrier: every D and every frontend terminal is fixed before any response coordinate can be queried. Freeze the common post-D prefix Pi. |
| 4 | Search each group's response nonce at its Pi-bound local address. After absorbing each candidate, run its priced column proof of work, then draw its indexed challenges and check response admission. Collect accepted response/work nonces and replay them in group order. Bind canonical supports in (group,owner,column) order to the live transcript. No ordinary response draw occurs. |
| 5 | Build all accepted responses Z_b, residual quotients K_b, existing T/U/EP and carries, and all ordinary preimages. Send exactly one outgoing commitment C_W for their concatenation. No response vector or quotient is sent in clear. |
| 6 | The successor-source comparison checkpoint fixes this complete witness candidate and all canonical setup sources. There is no wire message J or y. Freeze the post-C_W transcript for all later relation tests. |
| 7 | Draw the common E64 ring-switch alpha and range-image point tau0; draw one vector of independent E64 weights in the row order specified in the Stage 2 section, including each coefficient defect R_b,i-qK_b,i. Do not multiply the quotient sub-batch by a further group scalar. The scheduled work nonce/filter precedes its challenge. |
| 8 | Run one Stage 1 digit-range proof for the entire W table, messages before each challenge. Its terminal precedes Stage 2's independent E64 range-image binding scalar gamma. |
| 9 | One cubic Stage 2 over W authenticates the combined root relation, ordinary negative-bit constraints and range-image binding. Send one E64 witness evaluation at its terminal point r. All groups use this same evaluation. |
| 10 | Rebind/compare the order-0 composed prefix identity at the Stage 3 site. Send one E64 product claim C_setup; run one quadratic Stage 3 over Pfx; send one E64 setup evaluation S(zeta). Each round polynomial precedes its E64 challenge. Check Stage 2's terminal using C_setup. |
| 11 | Pass (C_W,r,W(r)) and the one prepared composite-prefix commitment with (zeta,S(zeta)) into a single ordinary successor fold. Continue the selected ordinary recursion and one terminal; discharge every witness and setup claim. Return the caller's live transcript and its exact-EOF obligation. |

Per-group D keeps the original B/D interpretation, permits different setup seeds
and ranks, and costs `512*r_D,b` bytes. A shared D would require either a new
combined commitment key or a multi-key binding argument, and would make its rank
depend on the aggregate widths. It saves small payloads while coupling
independently committed groups. Per-group D is the simpler marginal cost; it
adds no Stage 2, Stage 3, outgoing commitment or terminal. There are no ordinary
partials to commit in this design. If a later optimization folds ordinary
sources, its D must join the same order-3 barrier; it is not included here.

The honest prover searches nonce 0,1,... independently for each group, freezes
the first accepted response, and aborts after 4096 failed probes. Retry only
that group's response computation; do not rebuild another group's planes, change
Pi, or consume another group's challenge stream. All groups' accepted responses
must exist before C_W. There is no retry against quotient batching, Stage 2 or
Stage 3 within this attempt. Restarting any earlier public prefix is a fresh
attempt counted in Q, not a free retry.

A retry of group b changes no challenge of any other group, and the honest
search for b reads only b's data. A cheating prover may choose nonce candidates
adaptively after seeing other groups' challenges. Every candidate it tries
enters the query count, as in the existing fold-response analysis.

### Column-challenge addressing

Let Pi be an immutable transcript prefix through the **last** D payload, with
the whole order-0 statement and all order-2 frontends included. Define the
challenge address, before its exact support sampler consumes XOF blocks, as

```
(domain="akita/binary-source/multigroup/column/v1",
 common_prefix=Pi, level=0, group=b,
 owner=j, column=c, response_nonce=nu_b,
 column_work_binding=omega_b, family_id, sampler_block_index)
```

At the local group site, derive a detached state from Pi, the versioned domain,
level and b. Absorb the response-nonce candidate there, perform the scheduled
`FoldChallengeWork`, then derive `omega_b` from that checked local state. For
zero work bits there is no work-nonce body. This local state is an addressable
part of the one transcript, with no independently seeded session. The coordinate
address above binds the accepted local work before drawing. All address integers
use canonical fixed-width encodings. Pi may be represented by a 256-bit
transcript-state binding only with the separate hash charge below. The owner
coordinate is essential: two polynomials in one group must not share one
coordinate accidentally. Do not use the changing live cursor after an earlier
group's nonce, or the vector of all accepted nonces, as the address prefix.
Either would make a retry change unrelated coordinates. Accepted response/work
nonces and derived supports are absorbed into the live cursor for order 5 and
later; they do not redefine Pi retroactively.

Each address draws a uniform support from all subsets of `{0,...,161}` of
cardinality at most 46. Preserve the current deterministic signs, not random
independent signs. For a completely specified finite sampler, draw a uniform
137-bit integer from the indexed XOF (discard unused high bits), reject values
at least S46, then combinatorially unrank supports by size followed by
lexicographic order. Cap rejection at 256 candidates. This is the versioned
sampler for this profile. Its output is encoded as a canonical 21-byte support
bitmap with unused high bits zero; signs are derived by the selected sign-family
algorithm. Its exhaustion probability per draw is `(1-S46/2^137)^256`, not an
extraction failure probability. Across H honest support draws, union-bound it by
H times that value. Reusing the current sampler with a different mapping
preserves distribution, not transcript bytes.

The challenge set has

```
S46 = sum_{j=0}^{46} binom(162,j)
    = 107,268,495,235,892,251,993,000,716,928,106,195,079,952.
I_b = s_b C_b; I = sum_b I_b.
```

#### Flat extraction tree and within-group subtraction

Take one central accepted full continuation, and for each `(b,j,c)` one accepted
neighbor with the same Pi, **the same response/work nonce tuple**, and all fold
coordinates except `(b,j,c)` fixed. The extractor reprograms that indexed
response coordinate; it does not implement a fork by incrementing its nonce,
which would change all columns of b. C_W, Z, carries, K and later proof messages
may differ between children. D and the input commitments may not.

The existence and running-time statement assumes that each indexed SHAKE
column stream can be modeled as an independently programmable oracle.
The 64-bit root spec records this as an
[open question](binary-source-p64-root.md#open-questions).
This spec carries that question forward. It does not establish the model
from a common seed, deterministic expansion or domain separation.
The flat-tree and priced bounds below are conditional on that model and the
checkpoint spec's composition interface and exact conditional samplers.

In each child, Stage 2 plus the quotient lift supplies, for that one owner,

```
A_b Z_b,j = sum_c iota(c_b,j,c) T_b,j,c       (mod q).
```

Subtract **only these two equations for (b,j)**. Even if another group's
response or outgoing segments change after the fork, none is a term of this
equation. B binding either gives a collision or fixes T; D binding likewise
fixes the corresponding U/EP. With `s=c_b,j,c-c'_b,j,c` and `a=Z_b,j-Z'_b,j`,
obtain `A_b a = iota(s)T_b,j,c (mod q)`. The bounds are `||a||_infinity <=
1755-(-2340)=4095` and, comparing two weak openings,
`||iota(s')a-iota(s)a'||_infinity <= 4*92*4095 = 1,506,960 < q`. Divide in the
rational source ring, never modulo q, to extract `a/iota(s)`. The 64-bit root
spec's unit proof gives the prime reading; distinct supports also give a unit in
GF(2^162). Apply the parity and prime transfer arguments separately inside the
same group. Ordinary sources come directly from their fixed preimages and their
original commitment comparison maps.

For `G_b` single-owner groups of C columns, the root has exactly `1+G_b*C`
children: one central child plus `G_b*C` neighbors. It has neither `2^(G_b*C)`
children nor `(1+C)^G_b`. With multiple owners replace G_b*C by I. Let c_s be
the ordinary suffix checkpoint count including the replaced root successor
checkpoint, and A_s the product of all remaining extraction arities. The
root-plus-suffix tree is `2^c_s * (1+I) * A_s`. If an application segment
requires the checkpoint spec's additional source-comparison checkpoint, it is
`2^(c_s+1) * (1+I) * A_s`, once for the combined application, not once per
group. The expected run envelope is `(Q+2)*Tree`. For four C=256 groups, the
root arity is 1025, versus 257 for one group; the factor is `1025/257`, not
`257^4`. This increased extractor budget also enters all SIS assumptions.

#### Column-challenge proof of work

Conditional on that indexed-oracle model, a fixed coordinate has
unextractable mass at most `1/S46 <= 1/2^136`.
The group-vector site uses the conservative local loss `I_b/2^136`.
Apply the existing fold proof-of-work rule independently at each binary group:

```
W_b = I_b = s_b*C_b
g_b = least nonnegative g with I_b * 2^128 <= 2^136 * 2^g
    = max(0, ceil(log2(I_b)) - 8)
```

Compare the displayed integer inequality exactly using checked multiword
arithmetic. These products exceed u128.
Use the family's guaranteed set size `2^136`, as the current root already does.
S46 describes the full support family and sampler; it is not the work-pricing
denominator. The current root already has one `FoldChallengeWork` entry in
[`binary_source/grinding.rs`](../crates/akita-types/src/binary_source/grinding.rs).
It uses `binary_root_family().set_bits() = 136`.
This plan gives each binary group its own group-qualified entry.
Bind its group index, guaranteed set bits 136 and loss at order 0.
Validate them on replay. The response nonce is absorbed before this work and the
column challenges follow the work. This generalizes the [64-bit root's column
site](binary-source-p64-root.md#column-challenge-proof-of-work) using the
[existing fold
rule](transcript-grinding.md#fold-work-decision-and-total-security-accounting).

| Owners s_b | C_b | I_b | g_b | Expected work checks per response candidate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | at most 256 | at most 256 | 0 | 1 |
| 2 | 256 | 512 | 1 | 2 |
| 4 | 256 | 1,024 | 2 | 4 |
| 4,096 | 256 | 1,048,576 | 12 | 4,096 |

Here the zero-bit row denotes the unit cost baseline; it sends no work nonce
and requires no positive-bit work search. The zero-to-one-bit boundary is
I=257; every I up to 256 has g=0. All six single-owner cost-table rows have
`g_b=0`, including the two arithmetic-only rows. Each of the four groups in
either combined example has `g_b=0`; there is no aggregate two-bit work entry
for those four sites.

Conditional on the indexed-oracle model above, the classical random-oracle
rule gives the column contribution

```
sum_b q_b * 2^(-g_b) * I_b/2^136 <= (sum_b q_b)*2^-128.
```

The `q_b` count follows the response-candidate rule above and includes every
rejected or discarded candidate. Count their proof-of-work check queries in the
same ledger, as the fold rule requires. Untouched addresses completed by final
verification add its R term. Do not count one vector probe again as I_b separate
vector probes. For unfiltered comparison only, a whole attempt has the conservative bound
`I/2^136`. Four C256 single-owner groups have `1024/2^136`. Four sites also contribute four units
to the query count per complete attempt. Per-query pricing does not assert
`2^-128` soundness per complete opening.

A nonzero site sends one canonical LEB128 work nonce of at most
`ceil((g_b+7)/7)` bytes under the inherited semantic width. At the owner cap
this maximum is 3 bytes. The separate response nonce remains in `0..4095` and
has one or two bytes. The plan digest includes every site's presence, order and
price. E64 batching does not change this sparse-challenge term.

### Witness layout

Name coefficient quotients K, parity carries Cpar and value carries Cval to
avoid conflating two unrelated witnesses. Store base-eight digits as F_p values;
every later relation reads one canonical view of the committed W. For binary b
retain the legacy local order:

| Segment | Elements | Digits | Natural degree | Physical stride |
|---|---:|---:|---:|---:|
| Z_b | `s_b m_b` | 4 | 648 | 1024 |
| T_b | `3s_b C_b` | 8 | 648 | 1024 |
| U_b, if word | `s_b C_b` | 1 | 162 | 256 |
| Cpar_b, if word | `s_b` | dpar_b | 162 | 256 |
| Cval_b, if word | `s_b` | 6 | 162 | 256 |
| EP_b, if prime | `degree*t_b C_b` | 22 | 162 | 256 |
| K_b, in final quotient bank | `3s_b` | dK_b | 648 | 1024 |

When a word is selected, the all-owner U/carry shapes preserve current layout;
only the selected word owner's reading rows are active. A group without a word
has no U/carry segments. An optimization to omit unused owner views
would need a new layout version. The prime subset is in increasing owner order,
then base-coordinate order within each claim.
There is no EP for an unclaimed polynomial and no per-point expansion.

Global segment order is: binary legacy blocks b=0..G_b-1; ordinary blocks
o=0..G_o-1; then the quotient bank K_0..K_(G_b-1). This places all quotients
inside C_W but outside every D. Putting the K bank last preserves all old
single-binary-plus-ordinary offsets before p64 digit/rank changes; appending K
immediately after each binary would move the old mixed ordinary segments. It is
a storage choice only, not a later commitment time for K.

Within each segment use `(element,digit,coordinate_plane,coefficient)` order.
For one base plane the address is `start + (element*digits+digit)*stride+c`,
`0<=c<natural_degree`. Starts are cumulative physical lengths, as in
`RootCoefficientRepacking`; do not insert undocumented whole-group padding. The
binary segment lengths are multiples of 256, with 1024 alignment for Z/T. All
binary blocks precede ordinary blocks; a K following ordinary blocks may start
at a non-1024-aligned address. Its direct digit weight must support that offset,
or the versioned plan must explicitly add alignment padding. This spec uses the
existing generic coefficient-address formula and **no added alignment**, because
K is not a B/D typed digit lane. Binary local block starts remain multiples of
256; group Z/T starts can be non-1024-aligned when previous carries end there.
Therefore typed B/D compilation must use explicit translated global addresses,
retaining local aligned geometry for its period copies, rather than imposing the
old global-start alignment assertion.

Each ordinary block contains S digits, T image digits, first-map and second-map
negative-bit segments, and an optional signed-width bit segment. Their element
counts, degrees and depths come from the immutable p64 profile; reuse
`OrdinaryNativeRootPlan`'s canonical order and its per-owner source map. For
owners s, live blocks h, positions v, A rank r_A, A degree n_A, B degree n_B
dividing n_A, and digit depths d_S,d_T, its physical length is

```
L_o = s*h*v*d_S*n_A
    + s*h*r_A*(n_A/n_B)*d_T*n_B
    + width_map1*degree_map1 + width_map2*degree_map2
    + [signed_width] * s*h*v*w*n_A.
```

The map widths include their prescribed bit expansion. Do not multiply them by
64 or 128 a second time. `w` is the signed bit width. Full-field sources need no
optional signed-bit segment. The binary physical length is

```
L_b = 4096*s_b*m_b + 24576*s_b*C_b
    + w_b*256*s_b*(C_b+dpar_b+6)
    + 5632*degree*t_b*C_b + 3072*s_b*dK_b.
W_min = next_power_of_two(max(destination_ring_degree, sum_b L_b + sum_o L_o)).
```

Here L_b denotes a physical length; Bound_b denotes the coefficient-residual
bound in the admission section. A selected W may exceed W_min, never fall below
it. Every root relation has support only on natural addresses. All padding still
belongs to the committed table and passes the common range/norm/successor
checks; root relations need not force it to zero. Honest construction writes
zero.

#### Composed typed setup prefix and consumption

Each binary group has its own typed local prefix, containing its unchanged
centered-q A table and p64 B/D pieces, with its own setup seed/identity and D
selection. Preserve the opening spec's hole-packing algorithm **inside each
local prefix**. Local B/D digit-to-witness maps are translated by their global
segment offsets. U and EP compact packing and local setup column numbering
remain unchanged; no setup coefficient is sampled with another group's label.

Each ordinary group contributes a typed local prefix covering its A/B and both
compression setup roles. Repeated public matrices may be represented once
locally with summed weights, only when their complete setup labels and source
maps are identical. Do not assume different group identities share matrices.

Let each local table's padded domain be `P_g=2^k_g`. In canonical group order,
place it at `o_g=ceil(end_previous/P_g)*P_g` and set `end_g=o_g+P_g`, with
initial end=0. Set `Pfx=next_power_of_two(max(1,end_last))`. Zero-fill all gaps.
Bind every `(kind,index,setup identity,local prefix identity,o_g,P_g)` and the
root layout in one composite identity and one successor setup slot. This costs
less than `4*sum_g P_g` cells: each alignment gap is strictly less than its
following P_g, and final power-of-two rounding is less than twice the positive
end. For a single local table it is exactly its old domain and offset zero.

The setup preparer deterministically constructs and authenticates **one
commitment to this composed table**, before proving, from the authenticated
local setup definitions. Merely adding local commitment payloads is not a valid
concatenation commitment unless a separately proved commitment identity
justifies it. This design makes no such assumption. Preparation can scan each
public setup once and cache by the full composite identity. Per-group local
prefixes may be independently cached; the aggregate prefix depends on the chosen
group roster and plane counts. Online verification checks the trusted prepared
slot, not a prefix commitment offered by the prover.

The successor consumes two incoming claims, one on W and one on S, using its
ordinary grouped-fold machinery. Group offsets and the committed source maps
continue to bind every root view. No group-local prefix claim remains pending
behind the composed claim. Subsequent suffix folds and the terminal are the same
protocol family, with parameters selected for aggregate widths.

#### Byte preservation boundaries

Before activation, refactors must preserve all current pins. With one binary
group, local Z/T/U/carry/EP ordering, compact B/D input order and the q image
format remain unchanged wherever p64 and the one-point prerequisite leave them
unchanged. Existing commitment objects are not rewritten by opening. A
compatibility test can pin byte-identical pure helpers and inactive-feature
execution, and compare the vectorized single-group path to the immediately
preceding one-point/p64 implementation under the same version.

An active p64 proof cannot be byte-identical to this checkout's P128 proof: base
coefficients shrink, EP depth becomes 22, ranks can change, K is added, 84
projection bytes disappear, E64 challenges differ, and setup identities and work
schedules change. An active multigroup version also changes statement, challenge
addresses and prefix identity even for one group. Do not promise old proof bytes
while changing these domains. Preserve full one-group bytes only by continuing
to select its separately identified predecessor version; that is versioned
compatibility, not a second proof or fallback. The new version must have its own
pinned single-group reference vectors.

### Stage 2 and Stage 3

#### One relation vector

All quantities below use the fixed candidate at order 6. For each binary group,
owner j and A row r, form the **integer**, degree-648 reduced coefficient vector

```
R_b,j,r = rem_Psi(sum_l A_b[r,l]*Z_b[j,l]
                 - sum_c iota(c_b,j,c)*T_b[j,c,r]).
D_b,j,r,t = coefficient_t(R_b,j,r) - q*K_b[j,r,t].
```

Here D_b,j,r,t is a scalar defect, not the group D commitment. Use a different
Rust name for the two concepts. At order 7 the canonical independent row order
is:

1. All quotient defects, in `(binary group,owner,A row,natural coefficient)`
   order, with independent E64 weights rho_b,j,r,t.
2. For each binary group: selected word parity/value rows (two ring rows),
   B authentication rows, D authentication rows, then the prime response and
   reading rows (two ring rows per claim).
3. For each ordinary group: A image rows, B rows, first/second compression rows,
   per-owner point evaluations and any signed-width linkage rows, in its
   canonical plan order; then one negative-bit predicate per physical natural
   bit address, including optional signed bits.

One ring-switch alpha evaluates all ring-valued equations using their correct
native remainder kernels. A polynomial relation's degree, not the sum of all
groups' degrees, determines its root bound; see the soundness section. Quotient
defects are already coefficients and do not need ring-switching. The K digit at
its natural address has weight `-q*8^d*rho_b,j,r,t`. No clear projection output
or per-group quotient challenge vector message is added.

Let O_o denote ordinary linear rows and H_o negative-bit addresses. The current
complete-preimage grammar, generalized to p64, has

```
O_o = s_o*h_o*r_A,o + slices_o*r_B,o + 2 + s_o
      + [signed_width]*s_o*h_o*v_o
H_o = width_map1*degree_map1 + width_map2*degree_map2
      + [signed_width]*s_o*h_o*v_o*w_o*n_A,o
Batches = sum_b (1944*s_b + 2*w_b + r_B,b + r_D,b + 2*t_b)
          + sum_o (O_o + H_o).
```

`slices_o*r_B,o` means the profile's exact logical output-row count, with its
configured ownership/slice convention, not an extra owner multiplier. Every row
family is independently indexed. Reject a count/range mismatch, overlap,
omission or overflow before sampling. Add the finite total cap `Batches <=
2^32`; that permits streaming but is not a reasonable default memory allocation.
Negative-bit support H is at most the total ordinary physical length and thus at
most W. This cap, the metadata caps and the schedule bounds cover every
challenge-controlled array and loop. Derive all counts from the same canonical
row plan, not separate prover/verifier formulas that can drift.

Independent E64 sampling uses two unbiased u64 rejection samplers, each
rejecting candidates >=p with a cap of 256. Across M E64 draws, the completeness
abort bound is `2M*(59/2^64)^256`. For quotient rows alone, M<=7,962,624, this
is less than `2^-14800`, as checked by the script. For the full row cap, use
M<=2^32 in that same exact expression. Chunk boundaries do not restart indices
or resample weights. Ordinary negative-bit tests stay independent; replacing
them by one equality tensor would be a different degree/error tradeoff.

Stage 1 proves the balanced alphabet for **every** physical cell. Stage 2
additionally proves `x(x+1)=0` on negative-bit cells. Let W(r) be the common
witness evaluation, `L_pub(r)` the setup-independent linear weight, `L_setup(r)`
the omitted setup-dependent linear weight, `N(r)` the negative-bit weight and
R_range the Stage 1 terminal. The verifier's cubic terminal identity has the
existing form

```
SC2_terminal = (L_pub(r)+C_setup)*W(r)
             + (N(r)+gamma*eq(R_range.point,r))*W(r)*(W(r)+1).
```

The initial Stage 2 target is the independently batched public right-hand sides
plus gamma times the Stage 1 range-image evaluation. Quotient targets are zero.
Prime values and ordinary commitment/evaluation targets are included with their
own row weights, even if a public value happens to be zero. A zero equality
factor is never inverted to remove a source obligation.

#### One setup-product claim for all roles

For the prepared composite table S, construct a public weight F on its Boolean
domain with

```
L_setup(r) = sum_{x in {0,1}^k} S(x)*F(x),   k=log2(Pfx).
```

The local contribution F_b contains A's quotient adjoint, B's image-digit
authentication weight and D's U/EP weight, all using b's own rho, row batches,
source-address map and setup identities. An ordinary local contribution contains
its A/B/compression-map weights. The setup-free terms, including recomposition,
evaluation targets and K, stay in L_pub. Carries and K add no setup table; EP
digits use their group's existing typed D setup role.

For block g placed at aligned offset o_g with size `2^k_g`, its multilinear
extension is exactly

```
F_g_embedded(z) = eq(z_high, binary(o_g/2^k_g))*F_g(z_low).
F(z) = sum_g F_g_embedded(z).
```

The high selector uses all `k-k_g` high bits, including fixed leading zero bits.
Bit order is the table's low-bit-first index order. Nonoverlapping prefix blocks
and zero gaps make this formula exact. Do not multiply local product claims by
additional random group coefficients: their original Stage 2 row weights already
distinguish all obligations. A separate scalar claim for each group is
unnecessary. Send only C_setup, perform one degree-two sumcheck with initial sum
C_setup, then check its terminal against `S(zeta)*F(zeta)`. The one S evaluation
goes into the successor. The Stage 3 algebraic loss is `2k/e`, not `sum_g
2k_g/e`.

This extends the existing offload's exact Boolean-cube identity. The new work is
the composed setup identity and its comparison map, plus ordinary setup roles
that the current root still evaluates online. For arbitrary translated root
addresses, evaluate `eq(r, offset+local_address)` using checked carry-aware
binary-addition/offset-equality factors. A bitwise transfer has carry state 0 or
1; each bit enforces `a_i+b_i+carry_i = out_i+2*carry_(i+1)` with initial carry
0 and no overflow beyond the W domain. It evaluates the exact sum over the
specified rectangular local lanes, without treating addition as XOR. Tests must
compare it against a dense dot product at non-base E64 points, including offsets
that cross powers of two. The old aligned product evaluator is a specialization,
not a proof that arbitrary concatenation is safe.

Verifier work sums these factored local evaluations and selectors. It must not
scan A/B/D or ordinary setup matrices online. Offline preparation authenticates
each seed/role/dimension/column address, and validates the aggregate slot under
the selected successor setup. Changing any one seed, manifest, block order,
rank, plane subset or destination ring changes that identity. A setup registry
lookup is not permission to substitute another group's table.

### Cost model

#### Exact quantities and scaling

These costs are for the opening after commitments and prepared setup exist.
Original binary/ordinary commitments and offline prefix preparation are separate
costs. Preparation reads each selected setup once; building a new aggregate
commitment is not free. Table lengths, ranks, owner/plane counts and ordinary
profiles matter: carrier/coefficient counts alone do not determine a schedule.

Let `B_shared(W,Pfx,schedule)` include C_W, Stage 1, Stage 2, one Stage 3, all
suffix messages/work nonces and the terminal. Let nu(v) be the canonical LEB128
nonce length, `max(1,ceil(bit_length(v)/7))`, with bit_length(0)=0. For v<4096
it is one or two bytes. Then, excluding public statement/commitment bytes
already supplied by the application,

```
B_open = B_shared(W,Pfx,schedule)
       + sum_b [w_b*(2048 + 42*log2(N_b) + 21)
                + 512*r_D,b + nu(response_nonce_b) + work_bytes_b]
       + framing(profile).
```

The H partials cost 2048 bytes, each B round has two 21-byte coefficients, and
the terminal costs 21 bytes. There are no raw T, Z, K, U or EP bytes.
`work_bytes_b` is zero at `g_b=0` and otherwise is the accepted work-nonce
length. Extra ordinary groups add no dedicated proof payload at this root; their
witness and setup sizes enlarge the shared schedule. Each extra binary group
adds its selected frontend, D, response nonce and priced work nonce. Stage 3
field-message bytes are exactly `32k+32` for k rounds, before framing/work
nonces. With the standard coefficient-omission cubic encoding, Stage 2 has
`48r+16` field-message bytes for r=log2(W), before framing/work nonces. Stage 1
bytes are obtained by summing its selected `DigitRangePlan` stage shapes, not by
multiplying a full proof by G_b. All E64 elements remain 16 bytes, while base
payloads shrink to eight.

Prover work has local and shared terms:

```
T_prove = sum_b [frontend_b(N_b) + plane_b(N_b,degree*t_b)
                 + Dcommit_b + attempts_b*(work_b(g_b) + response_b(s_b,N_b,C_b))
                 + residual_b(s_b,m_b,C_b) + O(1944*s_b*dK_b)]
        + sum_o preimage_o(L_o)
        + commit_W(W) + stage1(W) + stage2(W,local_row_families)
        + stage3(Pfx,local_prefix_families) + suffix(selected schedule).
```

The residual uses the accepted response once. Extra K digits cost exact
divisions, writes, range work and commitment work; there is no seven-projection
pass. Dense implementation bounds are linear in W/Pfx per multilinear folding
pass, plus local kernels/setup emission; convolution/transform costs depend on
the actual kernels. Structured profiling can avoid materializing full E64
weights. E64 A adjoints require two F_p component passes; no integer embedding
of `a+b*u` is valid. Peak memory is aggregate live tables and scratch, not the
largest group's table alone. Parallel group construction may increase peak
memory and does not imply a linear wall-time speedup.

Verifier work is

```
T_verify = sum_b frontend_verify_b(log2(N_b))
         + O(sum_b s_b*C_b + Batches)   [independent draws and row processing]
         + sum_b local_weight_b + sum_o local_weight_o
         + stage1_verify(log2(W)) + O(log2(W))
         + O(log2(Pfx)) + sum_g prefix_weight_g
         + suffix_verify(selected schedule).
```

The existing direct parity/prime evaluators may scale with m; this design claims
no constant-time replacement. Independent ordinary bit draws can be linear in
H_o, even when their terminal support evaluation is factored. Online setup scans
disappear for all groups if the ordinary offload is implemented. The
setup-prefix size can increase shared suffix work; “shared” means one execution,
not constant cost.

#### Concrete binary counts

For a concrete comparison take one polynomial per binary group, C=256, one word
claim and one E64 prime claim owning two planes, `r_B=r_D=2`. All ordinary
table examples below
take one small-coefficient polynomial per group, as in the measurement fixture;
changing to full-field coefficients can multiply source digit work by 22 before
compression overhead. The 2^24 and 2^25 rows use the evidence-only carry depth
10 and **need the later geometry change**. This spec does not admit them.

| N | m | K digits | Parity digits | Binary physical cells including K | Frontend bytes | D bytes | g_b | Admission |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2^20 | 1024 | 11 | 9 | 13,472,512 | 2,909 | 1,024 | 0 | inside envelope |
| 2^21 | 2048 | 11 | 9 | 17,666,816 | 2,951 | 1,024 | 0 | inside envelope |
| 2^22 | 4096 | 12 | 9 | 26,058,496 | 2,993 | 1,024 | 0 | inside envelope |
| 2^23 | 8192 | 12 | 9 | 42,835,712 | 3,035 | 1,024 | 0 | inside envelope |
| 2^24 | 16384 | 12 | 10 | 76,390,400 | 3,077 | 1,024 | 0 | needs later geometry change |
| 2^25 | 32768 | 13 | 10 | 143,502,336 | 3,119 | 1,024 | 0 | needs later geometry change |

The constant C=256 image cost is 6,291,456 cells per owner even at small N. One
prime claim costs `2*5632*C` = 2,883,584 cells. K costs respectively 33,792,
36,864 or 39,936 cells for depths 11,12,13. These fixed costs prevent
proportional-to-N extrapolation at 2^20. B width is 62,208 and D width is
`ceil((162*C+44*162*C)/64)` = 29,160 in every row; rank two suffices for the
copied q64 estimator caps.

#### Combined examples

For reproducible combined examples, choose ordinary exponents `(18,20,22,25)` in
the large case; in the small case choose binary `(20,21,21,22)` and ordinary
`(20,20,21,22)`. The large example requires the later geometry change; the small
example fits this spec's per-group envelope. The formulas above give other
choices without changing the protocol.

| Quantity | Large: binaries 2^22,2^24,2^24,2^25; four ordinary 2^18..2^25 | Small: binaries 2^20,2^21,2^21,2^22; four ordinary 2^20..2^22 |
|---|---:|---:|
| Per-group geometry admission | needs later change for three groups | inside envelope |
| Column work bits | `(0,0,0,0)` | `(0,0,0,0)` |
| Column work-nonce bytes | 0 | 0 |
| Binary carriers total | 71,303,168 = 17*2^22 | 9,437,184 = (9/4)*2^22 |
| Ordinary coefficients total | 39,059,456 = (149/16)*2^22 | 8,388,608 = 2*2^22 |
| Binary physical cells | 322,341,632 | 74,864,640 |
| Source-only lower bound on W | 2^29 | 2^27 |
| Four frontends, exact bytes | 12,266 | 11,804 |
| Four D payloads, exact bytes | 4,096 | 4,096 |
| Four response nonces | 4..8 bytes | 4..8 bytes |
| Quotient rows | 7,776 | 7,776 |
| Binary relation coordinates | 7,808 | 7,808 |
| Root extraction arity | 1025 | 1025 |
| Shared executions | 1 Stage 1, 1 Stage 2, 1 Stage 3, 1 ordinary successor, 1 terminal | same |

The W lower bound rounds `(binary cells + ordinary coefficients)` to a power of
two. Ordinary T/compression/optional signed-bit segments and the catalog's
padding can increase it; neither 2^29 nor 2^27 is asserted to be the selected W.
The large binary table alone uses 2,578,733,056 bytes in eight-byte base storage
or 5,157,466,112 bytes in 16-byte E64 storage, before padding and scratch. One
additional ordinary group does not induce a whole duplicate suffix, but it can
push W or Pfx across a power-of-two or schedule boundary.

#### Calibration model

The study uses historical source anchors of 668.1 ms total / 378.8 ms root at
2^22, 917.1 ms total / 622.5 ms root at 2^23 ([sixth-pass
table](binary-source-opening.md#sixth-pass-results)), 84,392 proof bytes, and
9.68 ms total / 3.98 ms root verification at 2^22 ([verifier base-profile
table](binary-source-opening.md#base-profile)). These are different measurement
campaigns; combining them is a planning model, not a paired p64 experiment.

The mixed 2^22 fixture adds 60.8 ms proving (`728.9-668.1`) and 5.81 ms
verification (`15.49-9.68`) for one ordinary polynomial of 2^22 coefficients. It
includes all shared-schedule changes in that difference. Ordinary setup offload
and p64 can change both increments.

Let X be binary carriers divided by 2^22 and Y ordinary coefficients divided by
2^22. The following two models use those anchors:

```
Prove A (seconds) = 0.2893 + 0.3788*X + 0.0608*Y
Prove B (seconds) = 0.2893 + 0.1351*G_b + 0.2437*X + 0.0608*Y
Verify A (ms) = 5.62 + 3.98*X + 5.81*Y
Verify B (ms) = 5.62 + 3.04*G_b + 0.94*X + 5.81*Y.
```

A scales each root linearly. B fits the binary root's fixed/variable cost from
the two available sizes, using the 4.92 ms verifier root at 2^23. Both pay the
single anchor suffix once and use measured ordinary marginal differences.
Neither is a lower/upper bound. They omit the change in the aggregate suffix and
Stage 1/2/3 geometry, p64/E64 kernels, quotient work, retries, carry widening
and offload. Because the root anchor itself includes shared stages, these models
also overcharge some duplicated root fixed work. Do not subtract that without
span evidence.

For a byte proxy, keep the anchor's shared portion and substitute only the
explicit frontend, D and projection messages:

```
Byte_proxy = 84392 - 2993 - 2048 - 84 + sum_b frontend_bytes_b + 4096.
B_actual = Byte_proxy + Delta_nonce + Delta_column_work + Delta_framing
           + Delta_shared_schedule_and_migration.
```

`Delta_nonce = sum_b nu(response_nonce_b)-nu(anchor_nonce)`, between 2 and 7 for
four groups with the stated nonce cap. Delta_column_work is zero for both
example rosters. This proxy already removes the old 84-byte projection message
and replaces the old one rank-two P128 D by four rank-two p64 D payloads. It
does not predict the unknown shared migration delta.

| Planning output | Large case | Small case |
|---|---:|---:|
| Byte proxy before deltas | 95,629 bytes (95.629 KB) | 95,167 bytes (95.167 KB) |
| Actual complete bytes | `95,629 + Delta_nonce + Delta_column_work + Delta_framing + Delta_shared` | `95,167 + Delta_nonce + Delta_column_work + Delta_framing + Delta_shared` |
| Prove A / B | 7.2951 s / 5.5388 s | 1.2632 s / 1.499625 s |
| Verify A / B | 127.385625 ms / 87.865625 ms | 26.1950 ms / 31.5150 ms |

These explicit estimates are useful for initial capacity planning only. The
aggregate schedule determines the complete total and latency. Required next
measurements are p64 single and multi-group runs with the actual selected
W/Pfx/suffix, same source widths, cache state, hardware, accepted retry counts
and non-base E64 challenges. The costs that can be promised now are the additive
local tables, frontends, D payloads, work nonces and rows, and the absence of
extra complete proof executions.

### Soundness

Use the existing spec's explicit classical-ROM/checkpoint interface. This spec
does not prove that interface from scratch, claim a quantum-ROM theorem, or
certify post-quantum security merely because the commitment is lattice based.
The extraction assumptions must hold at the enlarged tree/time budget.

#### Fixed candidate and exact divisibility

At order 6 the candidate is the entire W table together with all canonical setup
sources and fixed view maps. Two different candidates under the same C_W and
prefix statement either agree on every view or produce the ordinary commitment
comparison collision. This is the required checkpoint comparison map, stronger
than simply saying a commitment is computationally binding. There is one such
checkpoint for the joined witness. It replaces the root successor checkpoint,
without adding one per group.

After range checks, each integer defect satisfies `|R_b,i-qK_b,i| <= Bound_b+q
Neg(dK_b) < p`. A nonzero integer defect is therefore nonzero in F_p and E64.
For any fixed false defect vector across **all** groups, condition on every
independent order-7 weight except one nonzero defect's weight. Exactly one value
out of e can cancel it. The whole independent batch loses a defect with
probability at most `1/e`, with no factor G_b, a or t. This also batches other
fixed scalar defects after ring-switching. Do not charge a second `1/e` for the
quotient vector if it is literally part of that same independent vector.
Range-image and later independent binding batches are different sites and retain
their separate charges.

Outside these and the sumcheck/binding events, every quotient relation holds as
an integer, giving every group's R=0 mod q. K may differ between response
branches; subtraction only needs the divisibility conclusion. Equality of K
across branches is neither true nor required. Group-by-group extraction in the
extraction section then yields the relaxed source and both selected readings.
The word/prime agreement contract remains conditional on the application's proof
about each extracted source, not boolean buffers passed to the honest prover.

#### Shared checks and new bad events

A safe unfiltered ledger per complete interactive attempt is:

| Event | Charge or condition |
|---|---|
| Binary extraction addresses | Conditional unfiltered `I/2^136`; priced column charge `sum_b q_b*2^-g_b*I_b/2^136` in the query ledger |
| Field switches | `sum_b w_b*(7+2*log2(N_b))/2^162`; at N<=2^23 each term <=53/2^162 |
| Quotient integer lift | Zero additional error under each group's strict bound and digit checks |
| Common ring switch | At most `d_alpha/e`, where d_alpha is the maximum actual ring-polynomial degree across selected rows; reduced degree <=647, conservative admitted numerator 648; if any unreduced product is used, use its actual bound, up to 1294, instead |
| One independent row batch | At most `1/e` for its fixed nonzero defect vector |
| Stage 1 | Sum its actual round degrees and binding-site losses; the checkpoint spec's conservative per-level `5 log2(W)/e` bound is a planning envelope only when its stage grammar matches |
| Stage 2 | `3 log2(W)/e`, plus its separate range-image binding site's degree/e |
| Stage 3 | `2 log2(Pfx)/e`; any additional claim batch, if introduced, must be charged separately |
| Ordinary suffix/application | Their full selected-schedule ledgers, once, including terminal and application reductions |
| Binary source comparison | Every group's actual q24 MSIS instance, radius 1,506,960, at the enlarged extraction budget |
| Other binding | Each input B, each D, ordinary input/compression commitments, C_W, composed-prefix commitment and all suffix instances |
| Setup/context substitution | Trusted setup authenticity plus typed-role separation and hash assumptions; not a new algebraic batching gain |
| Sampler exhaustion / response exhaustion | Completeness aborts, with explicit bounds where proved; not free soundness improvements |

Sharing alpha does not require summing one 648/e term per group: fix a false row
before alpha, whose nonzero polynomial has degree at most d_alpha. The chance
that all false rows vanish is bounded by the roots of that chosen row.
Conditional on at least one surviving row, independent weights give 1/e. This
argument requires all rows to be fixed at the checkpoint and their ring
interpretation to be preserved. It is not valid if a prover can choose a row or
an unbound plane after seeing alpha. Giving each group independent alphas would
instead require the corresponding multi-site accounting and would complicate the
shared weight; it is unnecessary here.

Likewise Stage 3 proves one dot product, so its degree is two regardless of
group count. Disjoint typed blocks, the selector formula and the **one**
prepared composite commitment are the justification, not an assumption that
separately verified claims can be added. A group permutation, setup swap, wrong
local-to-global offset, duplicate/omitted row, or reading an unrelated owner is
a new implementation-level failure mode. The canonical identities and shared
plan prohibit it; tamper/reference tests are needed to validate the
implementation, not to replace this binding argument.

The source-comparison theorem extends as a tuple: if two extracted tuples
differ, choose one differing group/owner and use its original comparison map.
Record which induced instance receives the reduction and sum the relevant
binding advantages. Subtraction uses two equations under the same A setup.
A difference between groups does not define the required short q-SIS vector. Ordinary tuples use their
ordinary comparison maps, with no new binary response tree.

#### Fiat–Shamir, retries and security admission

The composed bound retains the form

```
Delta_setup + Pr[not Good] + epsilon_interactive
+ E[sum_(queried addresses a) epsilon(a)]
+ sum_instances Adv_MSIS(instance, actual extractor budget) + epsilon_hash.
```

An envelope may replace the address sum by `Q epsilon_max`; do not add both. The
Q definition includes every indexed coordinate/vector probe, rejected nonce,
independent work-filter query and restarted attempt in its chosen units. Honest
completeness within the single-group geometry has conditional bound `(7/8)^4096`
per group under the spec's tail/sign model. If that same 1/8 success lower bound
is re-established independently for every selected group, a union bound is
`G_b*(7/8)^4096`. Admission requires the conditional bound for the selected
owner counts. Larger geometry requires the separate later analysis specified
above.

If the prefix is hashed, collision-only reasoning is insufficient for a prover
who guesses an unopened digest. For each independent 256-bit digest type with
Q_h queries and V bound positions, the checkpoint spec's sufficient ideal-hash
charge is `(binom(Q_h,2)+Q_h*V)/2^256`. Sum over distinct roles and avoid
counting one digest twice. Canonical raw order-0 records avoid adding
unnecessary digest roles. Indexed challenge derivation and root transcript
binding must share the same documented hash/XOF security model.

E64 has slightly fewer than 128 bits of cardinality: `1/e > 2^-128`. Use the
existing work rule at field sites with each site's proved debit and exact e. The
column sites use the guaranteed `2^136` denominator above. The complete schedule
ledger adds those priced sites, setup distance, hash terms and every induced Module-SIS
advantage at the actual extractor budget. A 128-bit per-query column rate alone
does not certify that total. No stronger challenge family or repeated opening is
required by this design.

What is not a straightforward union of old arguments is therefore explicit: flat
simultaneous extraction with group-local indexed nonces; the common
candidate/comparison map; one coefficient batch with no group multiplier; one
alpha preserving a fixed defect; the composed setup commitment and Stage 3
weight; and correct p64 ordinary range/compression semantics. The rest is
per-group field-switch, carry, prime transfer and binding reasoning.

### Alternatives Considered

The completed study selected this architecture. A shared D would couple setup
keys and ranks across independently committed groups and require a new binding
argument. Separate proofs would repeat the outgoing commitment, relation stages
and suffix. One outer scalar per quotient group would weaken the direct
independent batch. One setup claim per group would add claims that the composed
prefix already covers. The approved design retains one execution of each shared
stage.

## Documentation

This spec and its evidence define the extension until implementation. The
opening and group specs link here at their scope boundary. Implementation PRs
update `book/src/how/architecture.md`, `book/src/how/security.md` and
`book/src/how/transcript.md` with implemented behavior, admission and query
accounting. They update the affected spec lifecycle headers when work begins and
completes. Fold durable content into the Book and archive this record under [the
documentation policy](../docs/documentation.md) and [spec pruning](PRUNING.md).
The frozen 64-bit root spec remains unchanged by this documentation change.

## Execution

### Implementation PRs

Target the opt-in extension on dev. Main may receive independently useful
byte-preserving seams under `docs/branches.md`; the implementation must retain
ordinary behavior when the extension is unselected. This documentation change
implements no protocol code. File names marked “new” are proposed files; other
names exist in this checkout. Each change below should stay below about 2,000
changed lines, including tests; split mechanical test fixtures if needed. Split any change that exceeds that budget.

1. **Admission and layout accounting, no driver activation (700–1,200 lines).**
   Touch `crates/akita-types/src/binary_source/root_plan.rs`, `root_request.rs`,
   `root_public.rs`, `root_plan_tests.rs`,
   `crates/akita-types/src/root_repacking.rs`, and a new
   `binary_source/group_plan.rs`. Introduce ordered group occurrence/segment
   ranges and checked totals while retaining the singleton constructor's exact
   output. Test counts, non-power-of-two roster sizes, nonaligned segment
   translations, overflow, duplicate owner IDs, zero/one/several ordinary
   descriptors and the aggregate caps. Can land **before p64**; it uses the
   existing field profile and does not relax driver admission. Independent of
   the one-point API, apart from an abstract plane-count input.

2. **Indexed group nonce/address seam (900–1,500 lines).**
   Touch `binary_source/root_transcript.rs`, `grinding.rs`,
   `crates/akita-params/src/transcript_grinding.rs` and its plan modules, plus
   the typed replay adapters in `akita-types`, `crates/akita-challenges/src/`
   indexed draw implementation, and `crates/akita-transcript/src/` only if its
   indexed-prefix abstraction needs extension. First pin the current
   single-group byte stream. Add the versioned post-D prefix/address facility
   and one `FoldChallengeWork` entry with set bits 136 per group.
   Keep old schedules unchanged. Tests change one
   group/owner/column/nonce/prefix component at a time; prove other group
   streams are unchanged on local retry, and later live transcript continuation
   changes after accepted-nonce changes. Test work thresholds I=256/257,
   zero-bit omission and the g=12 owner cap. Exhaustion, exact-support
   rank/unrank boundaries, nonce canonicality and replay cursor failures return
   typed errors. Can land **before p64** and before one-point; activation and
   its per-group work ledger use the decided fold proof-of-work rule.

3. **One-point prerequisite integration (500–1,000 lines).**
   Consume the separately landing one-point change in `binary_source/prime.rs`,
   `prime_plan.rs`, `prime_weights.rs`, `opening_claims/binary.rs`, and
   `crates/akita-cpu-backend/src/binary_root/prime.rs`. Add group-qualified
   point/value/plane mapping; one point with distinct owners gives distinct
   planes, equal values on distinct owners still have distinct planes, and a
   second point or duplicate owner rejects. Preserve supported predecessor bytes
   before version selection. Can land **before p64**, but follows the one-point
   PR; later change digit depth only through the p64 profile.

4. **Translated setup blocks and one composite prefix (1,000–1,800 lines).**
   Touch `binary_source/typed_prefix.rs`, `typed_prefix_weights.rs`,
   `a_product.rs`, `typed_prefix_tests.rs`, and
   `crates/akita-cpu-backend/src/binary_root/setup_prefix.rs`; add a small
   composite-prefix layout module. Preserve each local A/B/D packing and add
   placement selectors, identity binding and trusted preparation of the composed
   commitment. Dense-reference tests cover two/four different setup seeds,
   unequal dimensions/ranks, nonaligned root offsets, padding and role swaps.
   Ensure a one-local-table layout has offset zero and unchanged local bytes.
   Can land **before p64** as arithmetic/preparation infrastructure, tested in
   the existing field; genuine non-base E64 tests follow p64.

5. **p64 quotient migration prerequisite and aggregate compatibility
   (700–1,400 lines beyond the separate migration stack).** Follow the 64-bit
   root spec's own implementation of bounded K, 22-digit EP, E64 coefficient
   challenges, B/D rank admission and field/serialization identities. Touch
   `binary_source/projection.rs`, `root_projection_weights.rs`,
   `evaluation_bounds.rs`, `authentication.rs`, `commitment_manifest.rs`,
   `crates/akita-params/src/binary_setup.rs`, and CPU `binary_root/residual.rs`
   only for the multigroup-facing bounds and K-bank maps. Test dK boundaries,
   malicious rather than honest envelopes, non-base E64 adjoints, and strict
   equality rejection at every no-wrap boundary. **Must follow p64**, and the
   chosen one-point p64 layout. This is not permission to fit the entire p64
   migration into this PR.

6. **p64 ordinary preimage grammar and setup offload (1,000–1,800 lines).**
   Touch `binary_source/ordinary_root.rs`, `ordinary_batch.rs`,
   `ordinary_root_reference.rs`,
   `crates/akita-cpu-backend/src/opaque/binary_ordinary.rs` and its tests, plus
   the composite-prefix module. Generalize canonical ordinary setup coefficients
   away from `Prime128Offset275`; include its compression roles in the single
   prefix product. Full-field digits use field reconstruction; explicitly signed
   widths must pass the new integer bound. Test one/multiple owners,
   heterogeneous ordinary profiles, independent points, negative-bit and
   signed-width tampering, and no online ordinary setup reads. Arithmetic
   cleanup can precede p64; **p64 activation must follow it**. Depends on PRs 1
   and 4, not the binary one-point functional itself.

7. **CPU multigroup preparation and outgoing witness (1,000–1,800 lines).**
   Touch `crates/akita-cpu-backend/src/binary_root.rs`,
   `binary_root/evaluation.rs`, `binary_root/sessions.rs`, opaque binary-source
   preparation and `crates/akita-types/src/binary_source/root_request.rs`. Own
   one list of local prepared groups, all D payloads, isolated response searches
   and a single concatenated witness. Cache accepted residuals; append the
   entire K bank before commitment. Test early/late retry in each of two/four
   groups, unchanged other response streams, all option combinations, offset
   decoding and exact row oracles. **Follows p64 and one-point**, plus PRs 1–6.
   Do not expose the public multiproof driver yet.

8. **One relation and setup-weight compiler (900–1,600 lines).**
   Touch `binary_source/root_public.rs`, `root_projection_weights.rs`,
   `evaluation_weights.rs`, `prime_weights.rs`, `ordinary_batch.rs`,
   `reference_weights.rs`, and CPU `binary_root/stage2_oracle_tests.rs`. Compile
   the canonical independent row vector, one Stage 2 and one composite Stage 3.
   Dense-reference tests use two/four binary and zero/one/several ordinary
   groups at genuinely non-base E64 points; mutate K, T, U, EP, Cpar, Cval,
   ordinary S/T/compression bits in each position. Verify batching counts,
   public targets and no root row on padding. **Follows p64**, one-point, and
   PRs 4–7. It should remain an arithmetic change separate from scheduling.

9. **Shared transcript driver and successor handoff (1,000–1,900 lines).**
   Touch `crates/akita-prover/src/protocol/prove/binary_root.rs`,
   `crates/akita-verifier/src/fold/binary_root.rs`,
   `crates/akita-types/src/opening_claims/root_schedule.rs`,
   `root_transcript.rs`, and root handoff/claim selection code as required.
   Enforce the all-D barrier, nonce collection, one C_W, one shared batch/stage
   pair, one prefix evaluation and one ordinary successor. Remove the hardcoded
   first/second-group dispatch only for the new profile. End-to-end tests in
   `crates/akita-pcs/tests/binary_source_end_to_end.rs` cover the full Cartesian
   set `G_b in {1,2,4}` and `G_o in {0,1,3,4}`, including prime-only, word-only
   and both. Compare transcript continuation and exact EOF. Test each group
   commitment, claim owner, point/value, D, nonce and setup identity tampering;
   reorder/remove/duplicate groups; reject an unconsumed prefix. **Follows p64,
   one-point and PRs 1–8**. No fallback or second opening is introduced.

10. **Admission and certificate ledger (800–1,500 lines).**
    Touch `crates/akita-types/src/binary_source/evaluation_bounds.rs`,
    `grinding.rs`, descriptor/profile admission in `crates/akita-params/src/`,
    and native-root sizing/security tests in `akita-planner`. Enforce
    `m_b<=8192`, `C_b<=256` and parity depth 9. Reject 2^24/2^25 per-group
    carriers before proof allocation. Check group-local retry/tail certificates
    for the selected owner counts, exact 136-bit-set work pricing, flat `1+I` arity and
    actual extractor time. Certify the actual q24, B, D, outgoing, prefix and
    suffix instances. Test absent certificates, cap boundaries and strict
    no-wrap rejection. **Follows p64**, one-point and PRs 1–9. The larger
    evidence-only geometries require a separate later change.

11. **Byte pins, performance and user-facing specification (700–1,500 lines).**
    Touch
    `crates/akita-pcs/examples/binary_source_full_size/{run,catalog,claims}.rs`,
    PCS end-to-end tests, `specs/binary-source-opening.md`,
    `specs/binary-source-groups.md`, `specs/binary-source-checkpoint.md`, the
    owning Book architecture/security pages, and setup/offload documentation.
    Pin inactive-feature and supported predecessor single-group bytes, new p64
    singleton reference bytes, and two/four-group bytes separately. Measure the
    small roster and admitted per-group boundary shapes, selected W/Pfx,
    ordinary widths, retries, proof atoms/bytes, times and RSS. Include both
    feature-enabled/unselected ordinary runs. **Follows the integration and
    admission PRs**; the large evidence-only roster stays outside activation
    until its later response/carry/q24 SIS change.

For each implementation PR, run the cheap preflight gates before expensive
compilation. Use the current path-specific workflows and CI test-pass commands.
Scope Cargo builds by package and target before filtering tests. Use `rtk` for
Nextest and verbose developer commands. Run all required Clippy feature graphs
for final CI validation. Poll every live process session to an exit code before
reporting success. This spec-only change uses the documentation and evidence
checks in Evaluation.

### Open questions

- Can each indexed SHAKE column stream be modeled as an independently
  programmable oracle for the extraction step? This is the
  [64-bit root spec's open question](binary-source-p64-root.md#open-questions).
  The flat extraction and per-query bound here are conditional on it.
- Which trusted setup-preparation API and cache lifecycle serve a dynamic
  roster? The protocol already requires one authenticated composed commitment.
  The API must expose its preparation cost and bind the exact roster identity.
- Which p64 ordinary compression/norm profiles and aggregate W/Pfx/suffix
  schedules will production select? Their exact proof bytes and peak memory
  require measurement. Full-field ordinary costs need separate measurements from
  the small-coefficient fixture.
- Which aggregate security allocation, query budget and extractor budget will
  the production catalog admit? Per-group column work pricing is fixed here. The
  tuple checkpoint, translated setup weights and actual induced SIS instances
  still need review and certification at that budget.
- Which selected owner counts have a certified conditional response-success
  bound? The geometry remains inside the single-group envelope, but owner
  multiplicity is not a new response-completeness proof.
- Which streaming preparation and weight representations meet the trusted
  memory budget for admitted aggregate tables?

The one-point interface is a dependency with a fixed contract in this spec: one
common E64 point for P64, one canonical functional and two base-coordinate EP
planes per selected owner, contributing two identities over E64. P128 retains
one plane per claim. Its landing API must preserve that contract. Larger binary
geometry is a later extension, not an undecided admission limit.

## References

- [Exact multigroup bounds and cost evidence](evidence/binary-source-multigroup/bounds.py).
  Run `python3 -I specs/evidence/binary-source-multigroup/bounds.py` from the
  repository root. The standard-library script uses exact integers and
  fractions. It checks 126 admitted geometries, 18 extra arithmetic-only
  geometries, setup placement, work prices and every concrete table. It does not
  prove response acceptance, lattice hardness or the checkpoint theorem.
- [64-bit root and its evidence](binary-source-p64-root.md).
- [Binary source opening](binary-source-opening.md).
- [Binary source group contract](binary-source-groups.md).
- [Checkpoint and extraction argument](binary-source-checkpoint.md).
- [Typed B/D setup offload](binary-source-setup-offload.md).
- [Transcript grinding and fold proof of work](transcript-grinding.md).
- [Branch contract](../docs/branches.md).
