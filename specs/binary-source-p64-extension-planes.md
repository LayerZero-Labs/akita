# Spec: extension prime readings at several points

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-11-06 |
| Status | active |
| PR | |
| Supersedes | Base-only readings and the one-point rule, for explicit extension claims only |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

## Summary

An application needing a larger reading field may supply one or two E64 points
per binary group. For a fixed nonzero degree-d scalar test at a uniform field
point, the root-count bound is d/|F|; the degree and challenge/query model must
be specified before assigning a security budget. Each polynomial reading at
each point owns two base-coordinate evaluation planes (EP). All planes use the
same committed source, accepted response, outgoing commitment and three root
stages. This change provides authentication capacity, not an amplification
or production-security theorem. Linux validation and the obligations below
remain outstanding; this record is active until those checks complete.

The base single-point shape retains its canonical bytes and arithmetic. No
feature flag selects the new shape. P128 remains base-only. The extension
shape is explicit even if all supplied second coordinates happen to be zero.
The public PCS admits it through an explicitly selected native catalog row.
Both endpoints derive witness geometry, D rank, relation batches and the typed
prefix identity from the manifest's actual coordinate-plane count. Ordinary
catalog rows still refuse P64 with an explicit native-row requirement.

## Intent

Keep source ownership independent of reading occurrences. Do not register a
second commitment owner to obtain another point. Ordinary groups and word
claims keep their existing contracts. The full-size example prices base,
one-point E64 and two-point E64 shapes separately through the native planner.

At the predecessor, `PrimeGroupClaim` in `prime.rs` defines one base point,
its `validate` method rejects repeated polynomials, and
`PrimePlaneManifest::canonical_bytes` binds the base encoding. Here and below,
unqualified source paths refer to `crates/akita-types/src/binary_source/` at
upstream commit `23e59ec954d0286d5c59ea429e3006e4a29db2c0`.
`BinaryRootWitnessPlan::with_prime_planes` in `prime_plan.rs` and
`BinaryCommittedGroupManifest::d_label` in `commitment_manifest.rs` cap planes
by owner count; there is no fixed five-plane cap. The five-plane root-spec table
is an example.

## Design

### Public objects and encoding

Retain the base point/value record. Add a claim sum type selecting that record
or an ordered list of E64 point records. An E64 element is `(a0,a1)` for
`a0 + u*a1` in `F_p[u]/(u^2-2)`, with `p=2^64-59`. Each coordinate is a
canonical little-endian eight-byte residue strictly below p. Encode a point as
its coordinate pairs in the existing axis order: eight coefficient bits,
column bits, then scalar-row bits. Values use exactly the same pair encoding.

The extension manifest starts with `AkitaPrimePlanes/e64` followed by byte 4,
then a little-endian u64 point count. Each point record has a u64 dimension,
its coordinate pairs, a u64 evaluation count, then polynomial-index u64 / value
pair records. Sort evaluations by polynomial index, retain point order. Plane
order is point, polynomial, then coordinate 0 and 1; column and digit orders
inside each plane are unchanged. The base manifest keeps its existing domain
and byte sequence, including absent claims.

Reject empty point lists, more than two points, equal complete point vectors,
wrong dimensions, coordinates >=p, empty evaluation lists, repeated owners at
one point, and unknown owners. An owner may occur once at each distinct point.
A group cannot mix base and extension records. A base-valued extension record
still consumes two planes. P128 rejects the extension variant. Constructors
validate counts and dimensions before tensor allocation; malformed records
return typed errors. Zero values and zero weights never remove planes.

### Coordinate planes and reading equations

Let the natural source coefficient be U[r,b,c], with signed packing undone,
and let r, b, c index scalar rows, columns and the natural 162 coefficients.
The coefficient cube has 256 positions, with virtual positions 162 through 255
zero. For each supplied point x, factor its equality weights as R_r C_b O_c
in E64. Write R=R0+u R1. Commit the base polynomials

```text
EP[h,b](X) = sum_c sum_r R[h,r] U[r,b,c] X^c, h in {0,1}.
```

Represent each coefficient modulo p by a centered integer in 22 balanced
base-eight digits. Accepted digits need not give a unique integer lift: every
reading below is a field identity. With signed response components V_r and
column challenges c_b(X), authenticate both ring identities modulo Phi_243:

```text
sum_r R[h,r] V_r(X) = sum_b c_b(X) EP[h,b](X).
```

Each uses the entire four-digit response. At the common E64 ring-switch point
alpha, each nonzero reduced degree-at-most-161 defect has failure probability
at most 161/p^2, conditional on a fixed defect. No extra source response is
sampled for a new point or plane.

The E64 scalar reading is

```text
v = sum_(b,c) C_b O_c (EP[0,b,c] + u EP[1,b,c]).
```

For w=C_b O_c=w0+u*w1 its two coordinate equations are
`v0=sum(w0*EP0+2*w1*EP1)` and `v1=sum(w1*EP0+w0*EP1)`.
Use two independent E64 reading batches beta0 and beta1. Thus the weights of
EP0 and EP1 are respectively `beta0*w0+beta1*w1` and
`2*beta0*w1+beta1*w0`. Compute w0 and w1 from the coordinates of the product:
`w0=C0*O0+2*C1*O1`, `w1=C0*O1+C1*O0`. Taking the limbs of an evaluation at a
later E64 verifier point is not coordinatewise polynomial evaluation.
The implementation keeps these two tensor terms factored.

There remain two independent batch coordinates per physical plane, in plane
order: ring identity, scalar coordinate reading. A pair uses both odd-index
reading batches across both planes, with target beta0*v0+beta1*v1. This
preserves the base batch grammar. A fixed nonzero scalar coordinate defect
survives independent batching except with probability 1/p^2; this is a common
batch loss, not an additional independent error for every reading. A union
bound for ring identities is `epsilon_ring <= min(1, 161*P/p^2)` for P
fixed degree-at-most-161 defects and uniform alpha in E64. The common alpha
means these events cannot be multiplied. Conditional on those tests and fixed
nonzero residuals, independent uniform E64 batches contribute
`epsilon_batch <= 1/p^2`. Reading identities have no new random point test:
x is already fixed and their coefficients are public.

These are statistical terms for one root reduction only, with `p=2^64-59`
and `P=2*sum_j s_j` for one or two E64 points claiming `s_j` owners each.
For R root reductions a conservative interactive union bound on these two
terms is `min(1, sum_r (161*P_r+1)/p^2)`. This excludes all other root stages
and application reductions. A Fiat–Shamir bound must additionally account for
each addressed query budget `Q_r`; even `sum_r Q_r*(161*P_r+1)/p^2` is only
a conditional bound on these sites, requiring fixed-defect conditioning at
each query. It is not a composition or amplification theorem. Computational
SIS, setup and hash advantages are separate, unquantified terms here; the
full statistical and computational totals remain open.

### Transcript and extraction

All points and values are caller-supplied and bound at order zero, before D,
column challenges and C_W. Akita draws no new application point. An application
may draw points in its earlier borrowed transcript, but must complete those
reductions before supplying the manifest. Claim type, point order, owner and
coordinate order are all bound. Order three commits every EP coordinate in D;
order four draws the original owner-column challenges; order five commits one
W; order seven draws the original ring point and the expanded relation batch.
One Stage 1, Stage 2, Stage 3 and ordinary suffix follow. There is no per-point
root retry, commitment occurrence or extraction branch.

Conditional on common-source extraction, subtracting the same owner's two
response branches transfers all its ring identities simultaneously. Extension
of the base-ring inverse remains valid. For s owners and C columns, the root
branch count remains `1+s*C`, independent of the point count. One root with
s=4 owners and C=256 gives 1025 branches; a single-owner root gives 257.
Separate groups require the multigroup extraction accounting. This is
conditional accounting, not a proof that the new trace satisfies the extraction
interface. The agreement binding
must retain the entire canonical manifest, all points and both coordinates;
its exact coefficient lift remains `184+4095=4279<p` under its stated premises.

### Resource limits and cost

The selected extension cap is k<=2 and P<=4s coordinate planes for s owners;
at the descriptor maximum s=4096 this is 16384 planes. Each point claims a
nonempty subset of owners. The old base shape still admits at most s planes.
Both layout and D validation enforce the profile's maximum capacity; manifest
validation enforces the tighter selected shape. Actual schedule and allocation
limits may reject a shape below this cap. The full-size example constructs the
same manifest for pricing and proving, so one/two E64 points cost two/four
planes per participating owner. Its catalog cache key includes the field,
protocol, size and explicit claim shape. On reload it re-prices the native
row and checks exact physical witness and prefix sizes. The selected catalog
digest binds that row; a profile lookup key alone is not a geometry certificate.

At C=256, the following counts follow from `P*22*162*C` natural digits and
`P*22*256*C` physical cells. D width is
`ceil((P*22*162*C + [word]*s*162*C)/64)`. D independently selects the least
rank covering that width. The copied q64 degree-64/difference-seven width caps
are 296, 76306, 4030488, 106021444 and 1826815994 for ranks one through five.
B width is unchanged. These are estimator-table selections, not hardness proofs.

| P | Natural digits | Physical cells | Prime D width / rank | With word D width / rank (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 912384 | 1441792 | 14256 / 2 | 14904 / 2 (1) |
| 2 | 1824768 | 2883584 | 28512 / 2 | 29160 / 2 (1) |
| 4 | 3649536 | 5767168 | 57024 / 2 | 57672 / 2 (1) |
| 8 | 7299072 | 11534336 | 114048 / 3 | 115344 / 3 (2) |
| 16 | 14598144 | 23068672 | 228096 / 3 | 230688 / 3 (4) |

Rows aggregating several owners describe one group. Separate groups each
select D locally. At the maximum, prime-only D width is 233570304, rank five;
adding all-owner word U still uses rank five.

No raw EP digit is a proof atom. The direct D payload change is
`512*(new_D_rank-old_D_rank)` bytes per group. At fixed ranks and fixed
outgoing/prefix/suffix schedule, the fixed-width algebraic payload delta is
zero: extra relation weights are challenges, public points are statement input,
and there is still one witness evaluation and one typed setup product claim. Response
and work nonces can change when the manifest changes, and their canonical
LEB128 encodings can have different lengths even with the same schedule.
A zero total proof-length delta additionally requires equal encoded nonce
lengths. In general add the selected schedule's changes in C_W, range rounds,
cubic Stage 2 rounds, Stage 3 rounds, setup-prefix commitments, successor commitments, terminal
payloads and response/work nonce encodings. Public statement bytes and
application reduction messages are separate. No exact whole-proof delta or
timing is claimed without a selected schedule.

## Evaluation

- [ ] Dense source evaluation agrees with both EP coordinate readings at
  genuinely non-base points and with the optimized relation evaluator.
- [ ] Two points share one response, including repeated owners across points.
- [ ] Reject coordinate-digit, coordinate-order and owner/point mutations at
  the relation being tested; reject duplicate points and count/cap violations.
- [ ] Legacy base manifest bytes and unselected proof bytes remain identical.
- [ ] Linux scoped tests, formatting, preflight and required feature checks pass.

Run `python3 -I specs/evidence/binary-source-p64-extension-planes/bounds.py` for
exact arithmetic and coordinate-product checks. It follows the existing
bounds.py evidence convention and uses only the Python standard library.

## OPEN proof and admission obligations

- **OPEN:** candidate comparison and backward extraction for the expanded
  coordinate relation compiler, proving one weak source supplies every point.
- **OPEN:** the composition theorem for independent application reductions,
  including equality tests, round challenges, source batching, query losses and
  any mixed extraction-tree debit. Distinct points alone do not prove independence.
- **OPEN:** full addressed-query ledger, work budgets, extraction-time SIS
  advantages and setup/hash terms; no total security level follows here.
- **OPEN:** Linux generation and validation of each native catalog geometry,
  complete proof bytes and performance, including typed prefix and ordinary
  successor. Source support for research admission is implemented; this does
  not close the preceding proof obligations.
