# Spec: packed offset-digit ordinary sources

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-11-07 |
| Status | proposed |
| PR | |
| Supersedes | Signed-bit ordinary source grammar only when this new profile is explicitly selected, after approval and implementation |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

## Summary

Replace each ordinary source's balanced decomposition and separate signed-bit
bank by one table of offset radix-eight lanes. Pack the three width-two sources
into one producer owner and the three width-nine sources into another. Keep all
original application readings. This reduces the counted source cells and the
ordinary input volume enough to leave room under the existing caps at 32,768
signatures. Actual admission still needs the complete witness, setup prefix,
row count, successor schedule and security certificate.

This draft is for maintainer review. It defines a new producer and opening
format, not a reinterpretation of old commitments. No implementation or
production security approval is included. The separate p64 challenge repair
called Ropt remains open.

The local representation and affine-reading arguments below are proved here.
Counts are derived arithmetic. Turning those arguments into an authenticated
fixed-source comparison across the producer, outgoing witness and suffix is
an open proof obligation. Resource headroom is not that proof.

## Intent and inherited contracts

Use the [multigroup root](binary-source-multigroup.md),
[p64 root](binary-source-p64-root.md),
[checkpoint](binary-source-checkpoint.md),
[padding rule](binary-source-zero-padding.md) and
the two-point extension readings proposed in draft #414 as the surrounding
protocol. This profile changes ordinary source lanes, source ownership,
reading maps and the Stage 2 polynomial degree. Binary coefficient quotients,
response rules and prime-coordinate planes retain their own contracts.

There are four ordinary groups. Their producer owner counts become `(1,1,5,1)`;
their logical source counts remain `(3,3,5,1)`. A producer owner names one
coefficient cube committed through its lane columns. A logical source names
one original application polynomial decoded from those lanes. These are
separate namespaces; reducing the first must never reduce the second.

The algebra uses `p=2^64-59` and E64 of size `e=p^2`. All stored lanes are base
field residues. Application points and sumcheck evaluations may be in E64.
The selected version must be explicit. An unselected extension keeps its
predecessor bytes as required by [the branch contract](../docs/branches.md).

## Lane grammar and comparison of integer sources

For a signed width w, the old semantic interval is
`I_w = [-2^(w-1), 2^(w-1)-1]`, inclusive at both ends. Set

```text
u = x + 2^(w-1)
d = ceil(w/3), r_top = w - 3(d-1), 1 <= r_top <= 3
u = sum_(k=0..d-1) 8^k a_k
a_k in {0,...,7}; a_(d-1) in {0,...,2^r_top-1}
y_k = a_k - 4
x = sum_k 8^k y_k + c_w
c_w = 4*(8^d-1)/7 - 2^(w-1).
```

Every lane is in `[-4,3]`. The top lane is additionally in
`[-4,2^r_top-5]`. For example, width two uses `y=x-2` in `[-4,-1]`;
`x=-2` uses `y=-4`, so the negative endpoint is not lost. Width nine
uses three full lanes. The proposed profile admits only the widths and packing
shapes below; this is not blanket admission of every width supported by the
old signed constructor. A future remainder-one width would need its two-root
top subset as well as the full-lane range proof.

<!-- check:endpoints:begin -->
| Width w | Lanes d | Top bits r_top | Minimum x | Maximum x | Affine constant c_w | Diameter |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 2 | -2 | 1 | 2 | 3 |
| 9 | 3 | 3 | -256 | 255 | 36 | 511 |
| 18 | 6 | 3 | -131,072 | 131,071 | 18,724 | 262,143 |
| 27 | 9 | 3 | -67,108,864 | 67,108,863 | 9,586,980 | 134,217,727 |
| 33 | 11 | 3 | -4,294,967,296 | 4,294,967,295 | 613,566,756 | 8,589,934,591 |
<!-- check:endpoints:end -->

**Local comparison theorem (proved here).** The admissible lane tuples for
width w are in bijection with `I_w`. Indeed `u` ranges over exactly
`[0,2^w-1]`; division with remainder in base eight gives a unique sequence of
lower digits and the stated top digit. Conversely any such tuple reconstructs
an integer in that interval. Subtracting the offset is a bijection. This is
exact equality of semantic source sets, not equality of old and new commitment
bytes. Fixed-depth balanced digits in `[-4,3]` are also unique: they form a
complete residue system modulo eight, so reduction modulo eight determines
each successive digit. The old balanced representable interval is wider than
`I_w` at the admitted widths, which is why the old grammar also needs its
signed-bit bank and linkage rows. Offset lanes with the top restriction encode
exactly `I_w` without that bank.

A coefficientwise field equality between two decoded values in an admitted
interval is an integer equality because its possible nonzero difference has
magnitude at most the table's diameter, strictly less than p. This does not
lift arbitrary random-point readings or A-image rows to integer equalities:
those are field identities, with their separate extraction and binding rules.

### The two packed narrow owners

At each common source-cube index, let the original width-two values be
`x_0,x_1,x_2` and put `u_i=x_i+2`. Define

```text
X_small = u_0 + 8*u_1 + 64*u_2 - 256
        = y_0 + 8*y_1 + 64*y_2 + 36
u_i in {0,1,2,3}; y_i=u_i-4 in {-4,-3,-2,-1}
x_i = y_i + 2.
```

This is a width-nine owner with holes, not a generic width-nine signed source
and not a contiguous width-six packing. Its set has exactly 64 values, minimum
-256 and maximum -37. There are 156 missing integers inside that hull and
448 excluded values in the full signed width-nine interval. For example -252
is a hole (`u_0=4,u_1=u_2=0`); -248 is allowed (`u_0=0,u_1=1,u_2=0`).
The contract checks all three lanes, not only the most significant lane.

For the three original width-nine values let `u_i=x_i+256` and define

```text
X_mid = u_0 + 512*u_1 + 262144*u_2 - 67108864
u_i in {0,...,511}
x_i = sum_(k=0..2) 8^k*(y_(3i+k)+4) - 256.
```

The nine lanes are three disjoint consecutive triples. This is the entire
signed width-twenty-seven interval: `512^3=2^27`, with no holes. There is no
nonlinear decoder witness and no carry between triples. Keep five width-eighteen
owners with six lanes each and the width-thirty-three multiplicity owner with
eleven lanes. Their original signed intervals remain exact.

### Affine readings, including all original claims

Let `Y_k(z)` denote the multilinear extension of lane k on one complete source
cube. For a logical source whose local lanes are k=0 through d-1, its reading is

```text
f(z) = sum_k 8^k Y_k(z) + c_w.
```

For packed owners use the disjoint subsets above, resetting powers of eight
at each logical source's first lane. The equality weights of a complete cube
sum to one even at an extension-valued point, so the constant survives exactly.
More generally a public linear functional with weights lambda reads
`sum_k 8^k <lambda,Y_k> + c_w*sum(lambda)`. For a native ring row the offset
is the coefficient mask `c_w*sum_c X^c` on the natural source coefficients,
not merely the constant ring element c_w. Restrict the mask when support is
partial. No offset is added to virtual padding.

Every original logical source, each requested point and each terminal value
gets its own canonical reading row. In particular the two packed owners still
supply six distinct application polynomials. No message or claim reads
X_small or X_mid; they only describe the packing's integer semantics. A single
evaluation of either would not determine the separate logical evaluations.
Bind lane subsets and logical identifiers before challenges; reject overlap, omission, reordering,
duplicate claims and alternative maps. Two-point S11a claims use the same lane
table and separate reading occurrences. Extension prime-plane support for
binary groups does not by itself admit ordinary multipoint readings.

## Range proof and the quintic Stage 2

Stage 1 continues to prove `[-4,3]` for every physical W cell using radix eight.
Keep compression cells' existing `z(z+1)=0` constraints. Remove only the
ordinary signed-offset bit bank and its linkage rows. For every natural lane
of X_small, add a coefficient defect

```text
P4(y) = (y+4)(y+3)(y+2)(y+1)
      = y^4 + 10*y^3 + 35*y^2 + 50*y + 24 = 0.
```

Over the base or extension field its roots are exactly `{-4,-3,-2,-1}`.
Within the larger accepted alphabet, the four forbidden values 0,1,2,3 give
24,120,360,840 respectively, all nonzero modulo p. A generic alphabet check
alone therefore fails to prove the packed owner's contract.

Add these defects, each with an independent coefficient, to the existing
common row vector after the selected ordinary group's compression-bit rows.
Use increasing natural coefficient address within each lane and canonical
owner/element/lane order. Every logical reading has a separate row before the
predicate rows. The full versioned row plan fixes all boundaries. In this
specified construction there are `3N` new quartic rows and no extra witness
cells. There is no separate subset sumcheck or additional row-batch loss.

At the common row-batch challenge, after the outgoing source is fixed, derive
one E64 weight `omega_j` per row by independent exact rejection-sampled XOF
draws. Bind the selected profile, row-batch site and canonical row index in
each draw's address; include both prime coordinates and rejection counters in
the codec. The quartic rows use their own `3N` draws within that same vector.
Powers of one challenge or equality-tensor weights do not supply this
independence and cannot inherit the fixed-vector `1/e` loss.

Let H4 be the multilinear table of those independent weights, zero outside the
natural subset addresses: `H4(rho)=sum_j omega_j*eq(addr_j,rho)` over the
quartic rows. The verifier must derive and consume all `3N` weights to evaluate
this MLE, with linear work in the supported table size (100,663,296 independent
E64 weights at 32,768 signatures), plus address/equality-weight evaluation.
Streaming can avoid storing the entire table, but does not remove those draws
or their weighted accumulation. No sublinear verifier evaluation is claimed.
Let rho denote the Stage 2 evaluation point, Nbit retain only compression-bit
weights, and all other notation be the multigroup terminal's notation. Replace
that terminal by

```text
SC2_terminal = (L_pub(rho)+C_setup)*W(rho)
             + (Nbit(rho)+gamma*eq(R_range.point,rho))*W(rho)*(W(rho)+1)
             + H4(rho)*P4(W(rho)).
```

The new defects have zero targets. The initial target retains the range-image
claim and the public linear right-hand sides, adjusted by all affine constants.
H4 already includes the support mask; do not multiply it by a second mask or
an extra equality tensor, which would change the degree. W and H4 are each
multilinear, so the last summand has individual degree at most five. The whole
Stage 2 changes from cubic to quintic, including serializer, terminal evaluator
and grinding plan. A hidden cubic parser is not compatible with this version.
Stage 1 and the setup-product stage keep their own degrees.

For one reading per logical source, the ordinary group's linear row count is
`s*h*r_A + logical_B_rows + 2 + J`, where J is its logical reading count.
Compression predicates number `width_map1*degree_map1 + width_map2*degree_map2`.
Add `3N` subset rows globally and the unchanged binary row families. With
multiple points use the actual occurrence count for J. Old signed-link rows
and signed bits disappear; compression Booleanity does not. Exact total K is
OPEN pending profiles and claims, and must satisfy the existing row cap.

## Producer, manifest and version changes

The current A producer has independent matrix columns for each position and
digit. Replacing its preimage by y changes A images and all downstream B and
compression commitments. It cannot be obtained by shifting an old commitment
to the weighted scalar. Commit y itself under the new profile's matrix, with
column order `(position,lane)` and the usual native coefficient order. The
packed scalar only describes integer semantics; it is not read by any message
or claim and has no extra committed scalar or witness segment. T and both
compression preimages remain present and are recomputed from the new A images.

Propose explicit domains `akita/ordinary-source/pack-offset/p64/v1` for the
source contract and `akita/binary-source/pack-offset/p64/v1` for the selected
opening. These are proposed identity strings, not existing API symbols. Bind
in the immutable producer descriptor: field, native geometry, lane depth,
packing kind, original widths, offsets, top subsets, physical owner IDs,
logical source IDs, ordered disjoint lane subsets and source-cube order.
Derive A/B/compression role identities and parameter identities from this
contract before committing. Reusing deterministic setup machinery does not
permit replay under the old role identity or old security certificate.

At opening order 0, use the multigroup canonical integer/blob/residue codec
and bind the complete contract plus its digest, all producer commitments and
setup identities, ordered logical reading occurrences and points, root/suffix
schedule and certificate identities, row plan, physical layout, typed prefix
placement and work/address plan. Counts and maps are derived and compared;
they are never proof-controlled allocation requests. Constants are determined
by the typed packing kind and widths, not arbitrary prover-selected fields.
Unknown versions, mixed old/new interpretations, duplicate ownership and
noncanonical encodings reject before allocation. Exact byte fixtures and
concrete enum allocation remain implementation gates.

Regenerate the composed prefix identity/commitment after changed matrix roles
or widths. Keep each original source's commitment event before the application
challenge that reads it. The outgoing commitment fixes y, T, compression and
all binary segments before range/relation challenges, using the p64 checkpoint
order. A manifest and an honest producer alone do not establish the malicious
comparison theorem.

## Exact capacity arithmetic

Let B be signatures, `N=1024B` and `M=2^20`. These counts assume complete
source cubes and the existing power-of-two A degrees without stride inflation.
They describe one opening and the selected ordinary inventory, independent of
any later Ropt replication decision.

```text
Old source digits = (3*1+3*4+5*7)N + 12M = 50N+12M
Old signed bits   = (3*2+3*9+5*18)N + 33M = 123N+33M
Old subtotal      = 173N+45M; old ordinary volume = 11N+M
New source lanes  = (3+9+5*6)N + 11M = 42N+11M
New signed bits   = 0; new ordinary volume = 7N+M
W0 = next_power_of_two(C).
```

<!-- check:counts:begin -->
| Signatures B | Grammar | N | Source cells C | Ordinary volume V | Source floor W0 | Cell headroom 2^32-C | Volume headroom 2^28-V |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | old | 262,144 | 92,536,832 | 3,932,160 | 134,217,728 | 4,202,430,464 | 264,503,296 |
| 256 | PACK-OFFSET | 262,144 | 22,544,384 | 2,883,584 | 33,554,432 | 4,272,422,912 | 265,551,872 |
| 1,024 | old | 1,048,576 | 228,589,568 | 12,582,912 | 268,435,456 | 4,066,377,728 | 255,852,544 |
| 1,024 | PACK-OFFSET | 1,048,576 | 55,574,528 | 8,388,608 | 67,108,864 | 4,239,392,768 | 260,046,848 |
| 32,768 | old | 33,554,432 | 5,852,102,656 | 370,147,328 | 8,589,934,592 | -1,557,135,360 | -101,711,872 |
| 32,768 | PACK-OFFSET | 33,554,432 | 1,420,820,480 | 235,929,600 | 2,147,483,648 | 2,874,146,816 | 32,505,856 |
<!-- check:counts:end -->

Negative headroom proves a failure before residuals. Positive cell headroom
is conditional: R must include all binary physical cells, ordinary T images,
both compression banks, layout gaps and any other witnesses. The minimal
compact-layout domain is `next_power_of_two(max(destination_degree,C+R))`.
The selected W may be larger; require `W<=2^32`, `Pfx<=2^32`, `K<=2^32`,
`V<=2^28`, and every other multigroup admission bound. No cap is raised here.
The complete original-source cube dimensions remain within the variable cap.

The current `ResearchBatchPlan` aligns each local power-of-two domain, places
binary domains first and the combined ordinary domain last. It differs from
the multigroup spec's compact physical concatenation. Let H be the actual end
of the preceding binary domains including their gaps, and let
`D=next_power_of_two(max(destination_degree,C+R_ord))` be the ordinary domain.
Its end is `ceil(H/D)*D+D`. If `0<H<=D`, the resulting W is `2D`, not D.
Neither H nor R_ord may be guessed from the source table.

<!-- check:aligned:begin -->
| B | Source floor D0 | Ordinary residual to D0 | Ordinary residual to 2^31 | Quartic rows 3N |
| --- | --- | --- | --- | --- |
| 256 | 33,554,432 | 11,010,048 | 2,124,939,264 | 786,432 |
| 1,024 | 67,108,864 | 11,534,336 | 2,091,909,120 | 3,145,728 |
| 32,768 | 2,147,483,648 | 726,663,168 | 726,663,168 | 100,663,296 |
<!-- check:aligned:end -->

At 32,768, the ordinary source floor is `2^31`. Under the aligned layout,
`R_ord<=726,663,168`, `destination_degree<=2^31`, and `0<H<=2^31` suffice
for `W=2^32`. This stricter ordinary budget must not be replaced by the compact
aggregate headroom of 2,874,146,816. The table also gives the residual that
would keep each smaller case at its own source floor. A compact implementation
needs explicit layout-version reconciliation, not an undocumented reorder.
Prefix, memory, row and successor certificates remain OPEN in either layout.

### Exact partial serialization price

For actual `l=log2 W` and composed `t=log2 Pfx` (t=0 only for a direct root),
the Stage 1 / Stage 2 / Stage 3 round-field payload is
`16*(4l+5l+2t)=144l+32t` bytes; the old cubic version is `112l+32t`.
These are headerless round payloads, not whole proofs. At the counterfactual
source floors and direct setup only:

<!-- check:payload:begin -->
| B | Old floor exponent | New floor exponent | Old partial bytes | New partial bytes | Partial delta |
| --- | --- | --- | --- | --- | --- |
| 256 | 27 | 25 | 3,024 | 3,600 | 576 |
| 1,024 | 28 | 26 | 3,136 | 3,744 | 608 |
| 32,768 | 33 | 31 | 3,696 | 4,464 | 768 |
<!-- check:payload:end -->

Actual nonces, commitments, terminal values, setup claims, suffix, framing and
all application proofs remain additional. No total bytes or runtime is claimed.

## Malicious sources, norms and the open security ledger

An accepted fixed candidate must satisfy all lane predicates, all affine
readings and every A/B/compression identity on the same source. An invalid lane
cannot be excused by an honest encoding routine. A swapped lane, omitted
logical claim, omitted offset, reused setup identity or false top subset is a
false branch obligation, even if an aggregate scalar happens to cancel.
Independent batching protects a fixed nonzero defect vector with loss `1/e`;
without the required fixed-candidate comparison theorem, use the
coordinate-wise extraction debit `K/e`. This draft proves neither the generic
extractor nor its instantiation for the new producer. P4 defects share the
one vector; they do not receive an extra factor `1/e` or an automatic square.

Stored lanes have magnitude at most four and raw pairwise difference at most
seven. Restricted lanes have difference at most three. Thus the source-bank
energy is at most `16C`, its raw collision energy at most `49C`, and the full
physical table bounds are `16W` and `49W`. Physical padding is included in W;
no saving assumes malicious padding is zero.

For uniformly distributed width-two inputs, honest mean square per source
lane rises from `(4+1+0+1)/4=1.5` to `(16+9+4+1)/4=7.5`. The worst-case
source bound `16C` is unchanged in form. Parameters sized from honest norms
must be recomputed for the actual input distribution and offset encoding.

<!-- check:norms:begin -->
| B | 16C | 49C | 16W0 | 49W0 |
| --- | --- | --- | --- | --- |
| 256 | 360,710,144 | 1,104,674,816 | 536,870,912 | 1,644,167,168 |
| 1,024 | 889,192,448 | 2,723,151,872 | 1,073,741,824 | 3,288,334,336 |
| 32,768 | 22,733,127,680 | 69,620,203,520 | 34,359,738,368 | 105,226,698,752 |
<!-- check:norms:end -->

W0 columns are floor-only arithmetic, not the actual outgoing norm. Affine
constants cancel in comparisons using the same manifest. The decoded widths
have the exact diameters above; X_small's diameter is 219. None of these is a
complete response-extraction MSIS radius. Recompute producer widths/ranks,
response bounds, operator norms, compression bounds and suffix parameters
from the actual profile. The existing response-comparison formulas
`beta_inf=4*omega_resp*Delta_z` (or `8*omega_resp*B_z` for symmetric bounds) and
`beta_2^2=64*K_op^2*B_z,2^2` require their own hypotheses. Raw lane difference
seven must not silently replace them. Here `omega_resp` denotes the inherited
response-comparison factor, distinct from the row weights `omega_j`.

For a fixed candidate, radix eight, native degree at most 648 and one root,
the unfiltered algebraic planning envelope is

```text
ring: 648/e; range point: l/e; common row batch: b_row/e
Stage 1: 5l/e; range-image binding: 1/e; Stage 2: 5l/e
Stage 3: 2t/e; columns: I/S46
b_row=1 with fixed-candidate comparison, otherwise K
S46 = sum_(j=0..46) binom(162,j); I = sum_b s_b*C_b
root subtotal envelope = (649+11l+b_row+2t)/e + I/S46.
```

This uses the degree-per-round convention and Stage 1 planning envelope in
[the multigroup ledger](binary-source-multigroup.md#shared-checks-and-new-bad-events):
the quintic Stage 2 contributes `5l/e`; Stage 3 contributes `2t/e`.
The Stage 1 `5l/e` envelope is conditional on the unchanged radix-eight stage
grammar, including its binding sites, fitting that envelope. Actual schedules
must sum their actual degrees and binding losses and verify this condition.
Do not mix in the p64 sibling's `(degree+1)/e` convention: under that convention
a quintic round would cost `6/e`, requiring a different subtotal. This is an
envelope for the stated roster, not a certified selected-schedule total.

The reduced prime-plane degree is included in the conservative ring term,
not added twice. Full unreduced identities require their actual degree.
Word field switches, suffix/EOR/terminals, application reductions, S11a, H,
any separate linkage, setup, Good, hashes and real transcript/XOF reduction
are additional ledger entries. Source-floor/direct-setup diagnostics use only
a hypothetical roster and leave the remaining terms OPEN; they establish no
whole-proof exponent.

For each actual site a use its conditional rate `d_a/(S_a*2^g_a)` only when
the work theorem applies, and sum over every adversarial and final-verifier
query. One may upper-bound by `Q*sum_a rate_a`, or by `Q*max_a rate_a` with
the addressed-site theorem; do not add both. Use one inclusive classical-ROM
Q, including retries. Hash collision/target terms use one global role census,
`sum_h (binom(Q_h,2)+Q_h*V_h)/2^256`, charged once. Real hash/XOF assumptions
and setup distances are explicit open terms. Under ideal-XOF exact rejection
with fail-closed exhaustion, the codec adds zero soundness loss; its abort
probability is completeness only. The older pooled-codec soundness charge in
the p64 sibling is not used under these exact-sampling hypotheses.

Keep the statistical sum separate from `sum_i Adv_MSIS,i(T_i)` for every
induced binding instance, with classical and quantum estimator columns and
full extraction costs. No complete statistical or MSIS number is available
without the actual parameter table and remaining theorems. No QROM claim is
made. PACK-OFFSET changes W, rows, producer comparisons, prefix and suffix
certificates, so Ropt must reprice them. It does not choose replication, repair
shared application folds, prove staged source fixation or authorize squaring
errors. Repeated chains must retain every original logical reading and every
required cross-authenticated terminal.

## Evaluation and acceptance criteria

- [ ] Maintainer review approves the integer grammar, hole predicates and logical-source map.
- [ ] Inventory every reader of the old signed-bit bank, signed-width linkage
  rows and balanced source digits, in the library and downstream integrations,
  with file:line and a migration or preservation disposition. Remove the bank
  only after every selected-profile consumer uses the new lane contract;
  preserve generic signed-digit storage, compression bits and unselected paths.
- [ ] Prove the fixed-candidate comparison for the new producer and outgoing
  source, with actual MSIS instances and full extractor cost.
- [ ] Pin new producer/manifest/schedule identities and reject old/new mixing.
- [ ] Match affine source and setup weights to dense references at base and
  non-base E64 points, including offsets, all endpoints and all logical claims.
- [ ] Implement and authenticate the quintic terminal, round shape and work plan;
  retain compression Booleanity and all binary constraints.
- [ ] Reconcile aligned versus compact layout and certify complete W, Pfx, K,
  volume, memory and suffix at every requested size.
- [ ] Audit malicious lanes, holes, padding support, missing claims, degree
  mutations, transcript continuation and exact EOF.
- [ ] Close statistical and computational ledgers separately, coordinated with
  the separate challenge-repair decision.
- [ ] Run Linux repository and path-specific gates after implementation.

## Source map

For the old grammar, inspect
`crates/akita-params/src/ordinary_source.rs` for the old signed interval and
malicious-lift admission; `crates/akita-params/src/layout/digit_range.rs` for
Stage 1 shape; `crates/akita-types/src/binary_source/ordinary_root.rs` for
independent A digit columns, bit/link segments and reading rows;
`crates/akita-types/src/root_repacking.rs` for natural addresses and padding;
the research batch driver (not yet on `dev`) for aligned domains;
`crates/akita-types/src/binary_source/root_public.rs` for signed-bit batch ranges;
`crates/akita-cpu-backend/src/commitment/source.rs` and
`crates/akita-cpu-backend/src/sources/packed_digits/mod.rs` for packed numeric
storage; and `crates/akita-params/src/sis/decomposition_digits.rs` for balanced
depths and reaches that remain in use by other roles.
The code is evidence for what must change, not an implementation of this draft.
