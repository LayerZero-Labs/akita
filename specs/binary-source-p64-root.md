# Spec: binary source root over the 64-bit field

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-11-05 |
| Status | active |
| PR | |
| Supersedes | P128 root profile and exact-divisibility projection design in [binary-source-opening.md](binary-source-opening.md), upon implementation |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

## Summary

Move the binary-source grouped opening from `P = 2^128 - 275`, with
base-field challenges, to `p = 2^64 - 59`, with quadratic-extension challenges.
The current seven integer projections do not fit the smaller characteristic.
Replace them with committed coefficient quotients and independent extension-field
batching. Keep the rank-three degree-648 commitment modulo q and the
GF(2^128) word claim. This spec approves the protocol design.
Production security is not established. The research implementation selects P64
through explicitly admitted native catalog rows, the sole execution boundary.
Linux catalogs, reference pins, complete measurements and validation remain pending;
this record stays active until that evidence and the admission obligations below
are resolved.

## Intent

### Goal

Implement the binary root over `F_p` with challenges in
`E64 = F_p[u]/(u^2 - 2)` and a bounded integer quotient for every residual
coefficient.

A residual coefficient is a coefficient of the accepted-response equation
`A Z - sum_b iota(c_b) T_b`, reduced by the degree-648 minus trinomial.
Here Z is the accepted response, T is the committed source image, and c_b is
the column challenge for column b.
The map iota places the degree-162 challenge in the source ring.
The quotient table K certifies that every residual coefficient is a multiple
of q. K is an auxiliary part of the outgoing witness.
It is distinct from the parity carry called K in the existing opening spec.
This spec calls those existing carries `K_parity` and `K_value`.

The source baseline is `549f86d5d7620d2a35cd9769bd3c4725589ad543` on dev.
The [existing opening](binary-source-opening.md) describes that implementation.
The [checkpoint spec](binary-source-checkpoint.md) supplies the binding and
extraction premises used here.
This spec governs the migration where its field and projection rules differ.

### Invariants

- The outgoing commitment fixes every K digit before its testing randomness.
  The accepted response also precedes that randomness.
- Stage 1 range-checks K in the same alphabet as the other base-eight digits.
  Stage 2 authenticates all root relations against one opening of that witness.
- The coefficient lift uses the characteristic p, even when challenges lie in
  E64. It never uses `p^2` as an integer no-wrap threshold.
- Each coefficient has an independent uniform E64 batching weight.
  The coefficient equations enter the common relation batch directly.
- Root equations read natural coefficient addresses only.
  Range, successor authentication and physical-norm checks still cover all
  committed cells, including padding.
- Stage 3 authenticates the same typed A/B/D setup product used by Stage 2.
  Its extension-field point and value enter the ordinary successor.
- Prover and verifier bind identical profile, layout, setup and encoding
  identities. They reject mismatched or malformed inputs with `AkitaError` or
  `SerializationError`.
- An unselected extension preserves ordinary setup, commitment and proof bytes,
  as required by [the branch contract](../docs/branches.md).

### Non-Goals

This work does not change the GF(2^128) word claim or the rank-three q-valued
A commitment. It does not prove application-side signed-bit membership.
It does not admit every descriptor geometry at 128-bit security.
It does not define a new zero-knowledge claim.
It does not preserve binary-root proof or persisted-setup compatibility.

## Evaluation

### Acceptance Criteria

- [ ] A p64/E64 word-only, prime-only and mixed opening uses the same grouped
  PCS entry points, caller transcript and grinding cursor as its successor.
- [ ] Geometry determines K's digit depth and addresses before commitment.
  Admission checks `L + q B_K < p` using the full accepted digit bounds.
- [ ] K is committed at order 5 and range-checked in Stage 1.
  There are no order-six J rows, y messages or 95-bit y admission cap.
- [ ] Order 7 draws independent E64 coefficient weights.
  The K-digit weight is `-q * 8^j * rho_i` at its natural address.
- [ ] Prover, verifier, reference compiler and optimized compiler agree for
  challenges with a nonzero u coordinate.
- [ ] Prime base coordinates use 22 digits and eight-byte canonical residues.
  Any admitted E64 prime claim accounts for both coordinate planes.
- [ ] B/D ranks cover their actual widths under the q64 table.
  The outgoing witness, typed prefix and ordinary suffix are repriced.
- [ ] The grinding plan keeps the root's column-challenge work entry and its
  `2^136` denominator. The full security ledger includes all algebraic, retry, setup and
  binding losses.
- [ ] The evidence script passes under `python3 -I`.
  Protocol mutations and malformed-input tests reject with typed errors.

### Testing Strategy

Keep the existing binary-root, field-switch, statement-binding, setup-prefix,
prime-plane and grouped-recursion tests.
Extend `crates/akita-types/src/binary_source/root_plan_tests.rs`,
`root_projection_weights_tests.rs`, `authentication_tests.rs`,
`typed_prefix_tests.rs` and `prime_tests.rs` for the new profile.
Compare canonical relations with the independent reference compiler.
Use independent long-division adjoint checks for both base coordinates.

Extend `crates/akita-cpu-backend/src/binary_root_tests.rs`,
`binary_root/stage2_oracle_tests.rs`, `binary_root/evaluation_tests.rs` and
`binary_root_mixed_tests.rs` for quotient production and genuine E64 challenges.
Check the fixture and envelope arithmetic without allocating maximum-size
witnesses in unit tests.
Exercise negative quotients, both digit endpoints, insufficient digit depths,
wrong segment addresses, noncanonical field encodings and rank mismatches.

Extend `crates/akita-pcs/tests/binary_source_end_to_end.rs` and its
`projection.rs`, `fault_witness.rs`, `statement_binding.rs`, `prime_plane.rs`,
`stage3.rs` and `compatibility.rs` modules.
Mutate K independently of Z and T.
Check that changing an order-five commitment changes the later challenges.
Check that both extension coordinates reach Stage 2, Stage 3 and the successor.
Retain the guard against online verifier A materialization.
Retain exact transcript replay and ordinary byte-compatibility tests.

For implementation PRs, run the repository preflight before compilation.
Use the feature graphs and test-pass commands in the applicable workflows.
Run the three Clippy configurations from `AGENTS.md`.
This documentation change requires the evidence script, `typos`, documentation
guardrails, a no-Rust-change check and the privacy scan.

### Performance

K adds `a*d` natural digits and `3*s*d*1024` physical cells.
Here a is the residual coefficient count, s is the owner count, and d is K's
digit depth, as defined below.
It adds no raw proof message, separate commitment payload, witness evaluation
or Stage 3 product claim.
Removing the seven twelve-byte y messages saves exactly 84 bytes.
This is the projection-specific proof delta when the outgoing domain and
successor schedule remain fixed.

| Geometry | Owners s | K natural digits | K physical cells |
| --- | ---: | ---: | ---: |
| Fixture | 1 | 15,552 | 24,576 |
| Fixture | 4096 | 63,700,992 | 100,663,296 |
| Envelope | 1 | 23,328 | 36,864 |
| Envelope | 10 | 233,280 | 368,640 |
| Envelope | 4096 | 95,551,488 | 150,994,944 |

The following word-only layout excludes optional ordinary preimages.
Its old and new columns isolate K's addition before schedule selection.
Large owner counts are arithmetic examples, not security admissions.

| Geometry; owners | Old live cells | New live cells | Old minimum domain | New minimum domain |
| --- | ---: | ---: | ---: | ---: |
| Fixture; 1 | 57,600 | 82,176 | 65,536 | 131,072 |
| Fixture; 4096 | 235,929,600 | 336,592,896 | 268,435,456 | 536,870,912 |
| Envelope; 1 | 39,915,264 | 39,952,128 | 67,108,864 | 67,108,864 |
| Envelope; 10 | 399,152,640 | 399,521,280 | 536,870,912 | 536,870,912 |
| Envelope; 4096 | 163,492,921,344 | 163,643,916,288 | 274,877,906,944 | 274,877,906,944 |

For projection-induced scheduling changes, the proof delta is
`-84 + bytes(new selected schedule) - bytes(old selected schedule)`.
Base-field encoding changes and B/D rank changes contribute separately.
Prime planes shrink from 43 to 22 digits per base coordinate.
There is no exact complete-proof size until a p64 catalog selects the domains
and successor schedule.

The prover reuses the accepted integer residual.
It replaces seven projections and their J-to-rho combination with `a` exact
quotient divisions, `a*d` digit writes and `a` E64 draws.
At `N = 2^22`, `C = 256`, `s = 1`, twelve digits add 36,864 cells.
The resulting 23,174,912 live cells fit the existing `2^25` domain.
The addition is less than `1/900` of that domain.
At the envelope it also leaves the `2^26` minimum domain unchanged.
These counts do not establish a timing result.

The verifier draws `a` E64 weights and evaluates K's digit weights.
A direct compiler uses `O(a)` challenge work and `O(a*d)` digit-weight work.
The existing rank-many A/image adjoints avoid an online scan over m A rows.
Stream weights at large owner counts where the compiler permits it.
Benchmark tiny, `2^22` and `2^23` word, prime and mixed shapes with
`crates/akita-pcs/examples/binary_source_full_size.rs` and its catalog module.
Report complete bytes, selected domains, prefix size, time and peak memory.
Measure the two-coordinate A adjoint separately.

## Design

### Architecture

The base commitment field is `F_p`.
The random testing field is E64.
The exact field parameters are:

```text
q = 14,253,409
P = 340,282,366,920,938,463,463,374,607,431,768,211,181
p = 18,446,744,073,709,551,557 = 2^64 - 59
e = |E64| = p^2
  = 340,282,366,920,938,461,286,658,806,734,041,124,249
floor(p/2) = 9,223,372,036,854,775,778
```

The evidence checks that p is prime.
It also checks `p mod 8 = 5` and `2^((p-1)/2) = -1 mod p`.
Thus 2 is a quadratic nonresidue and `u^2 - 2` defines a field with e elements.

#### Residual and quotient bounds

Let m be the source ring-row count and C the column count.
Let s be the number of committed polynomials, also called owners.
The fixed A rank is three, so there are `a = 3*648*s = 1944*s` residual
coefficients across the group.
The fixture has `(m,C) = (1,2)`.
The envelope has `(m,C) = (8192,256)`.
The descriptor cap is 4096 owners.

Let `B_A` bound a centered public A coefficient, `B_Z` the reconstructed
response, and `B_T` the reconstructed image.
Let Gamma bound multiplication by a column challenge in infinity norm.
Use the full accepted digit bounds:

```text
B_A = (q-1)/2 = 7,126,704
B_Z = 2340                 # four base-eight digits
B_T = 9,586,980            # eight base-eight digits
Gamma = 92
L = 972*m*B_A*B_Z + C*Gamma*B_T
```

The factor 972 is the maximum absolute coefficient row sum of the reduced
multiplication kernel.
Each natural residual coefficient `R_i` satisfies `|R_i| <= L`.
The image bound here is the malicious accepted bound.
Replacing it with the smaller honest centered image bound would invalidate
the soundness lift.

For a q-divisible residual, define `K_i = R_i/q` over the integers.
Set the geometry-dependent digit depth and accepted magnitude as follows:

```text
H_K = floor(L/q)
d = min { j >= 1 : 3*(8^j - 1)/7 >= H_K }
K_i = sum_(j=0..d-1) 8^j k_(i,j),  k_(i,j) in [-4,3]
B_K = 4*(8^d - 1)/7
Admission: L + q*B_K < p
```

The positive reach `3*(8^d - 1)/7` covers every quotient up to H_K.
The negative reach has magnitude B_K.
Completeness uses H_K.
Soundness uses the larger B_K because Stage 1 accepts every digit in `[-4,3]`.
No separate comparison against H_K is required.

| Geometry | L | H_K | d | Positive reach | B_K | L + q B_K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixture | 16,211,309,718,240 | 1,137,363 | 8 | 7,190,235 | 9,586,980 | 152,858,456,733,060 |
| Envelope | 132,788,824,280,985,600 | 9,316,285,267 | 12 | 29,451,204,315 | 39,268,272,420 | 692,495,571,806,665,380 |

Both lift bounds are strictly below p.
They do not multiply by s.
Seven digits have positive reach 898,779 and fail at the fixture.
Eleven have positive reach 3,681,400,539 and fail at the envelope.
The evidence checks every power-of-two m and C within the current carry
envelope. Intermediate geometries use the minimal d from the same formula.

The old projection bound is `U = (q-1)*a*L`.
Even `2*(q-1)*L >= p` at both geometries.
Splitting the old dot product into single residual coefficients cannot repair
its lift.
Extension challenges do not change the kernel of the integer map:
integers map to zero in E64 exactly when they are multiples of p.

#### Transcript order and witness layout

Append K after the existing binary and selected ordinary witness segments.
Use one degree-648 element for each owner and A-rank row.
Each element has d digit lanes of stride 1024.
Order the natural digits by owner, rank row, digit, then coefficient.
The coefficient weights use the corresponding owner, rank row and coefficient
index i. Bind the shape, depth, segment offset and addresses in the root plan.
K is present in word-only, prime-only and mixed roots.

| Order | Required action |
| --- | --- |
| 0 | Bind the versioned p64/E64 statement, manifest, commitments, claims and selected schedule. |
| 2 | Run the unchanged binary carrier field switch when a word claim is selected. |
| 3 | Commit selected U and EP views in D with the p64 encoding, digit depth and admitted rank. |
| 4 | Draw the column challenges and accept the bounded response using the existing response rule. |
| 5 | Commit Z, T, selected U/EP, `K_parity`, `K_value`, ordinary segments and K's digits in one outgoing witness. |
| 6 | No J draw and no y messages. |
| 7 | Draw the ring-switch challenge, range-image point and independent E64 relation weights, including one rho_i per residual coefficient. |
| 8 | Stage 1 range-checks the outgoing digit table, including every K digit. |
| 9 | One fused Stage 2 authenticates the coefficient equations, other root relations and range-image binding against one witness opening. |
| 10 | Stage 3 authenticates the typed A/B/D prefix with E64 claims and evaluations. |
| 11 | Continue the p64 ordinary successor and terminal verification. |

K depends on the accepted response, so it is absent from D.
It requires no additional typed setup table.
Use the existing A/T adjoint with the explicit vector rho.
Its entries no longer come from combinations of q-valued J rows.

For base-field prime claims, let
`H_other = 2*[word] + r_B + r_D + O + 2*planes`.
Here `[word]` is one when the word frontend is selected and zero otherwise.
The values r_B and r_D are the B and D authentication ranks.
O counts the selected ordinary linear rows and negative-bit addresses.
The relation batch has `a + H_other` independent coordinates instead of
`7 + H_other`.
This changes the number of challenge coordinates, not the number of Stage 2
executions.

The coefficient contribution is

```text
sum_i rho_i * (R_i - q*K_i), with target zero in E64.
```

Put these defects directly into the common independent relation batch.
Do not add an outer random scalar to this contribution.
At K's natural address `(i,j)`, the Stage 2 weight is
`-q * 8^j * rho_i`.
The A and image weights are the existing adjoints applied to rho.
The equations remain linear in committed digits, so the fused Stage 2
polynomial degree remains three.

Encode an E64 element `a0 + a1*u` as its two canonical little-endian u64 coordinates in
basis order `(1,u)`.
Sample the coordinates independently.
Reject random u64 candidates at or above p instead of reducing them modulo p.
A cap of 256 attempts per coordinate gives the following completeness-abort
bound for the coefficient batch at 4096 owners:

```text
2*a*(59/2^64)^256 < 2^-14800.
```

Sampling failure returns a typed error.
The abort bound is separate from soundness error.

#### Soundness and extraction

Condition on authentication and extraction of a common bounded digit witness
and a fixed typed setup before order 7.
This is the candidate-comparison premise of the checkpoint argument.
It needs validation for the new relation compiler.
An honest execution test alone does not establish it.

Write the integer defect as `D_i = R_i - q*K_i`.
The digit bounds and admission imply `|D_i| <= L + q*B_K < p`.
A nonzero integer defect is therefore nonzero in F_p and in E64.
Fix one such coordinate and condition on every other independent batch weight.
Exactly one of its e possible weights makes the combined relation zero.
The failure probability is at most `1/e`.
There is no factor a and no separate q-annihilation term.
The same independent batch also contains the other fixed root rows.
Charge that batch once rather than once per coefficient or relation family.

Outside this event and the separately charged authentication and sumcheck
events, the defects vanish over F_p.
Their bounds imply that they vanish over the integers.
Thus `R_i = q*K_i` coefficientwise and the extracted residual is zero modulo q.
This restores the premise used by two-response extraction.
Subtracting openings gives `A a_b = iota(s_b) T_b mod q`.
The q-SIS comparison radius remains
`4*Gamma*4095 = 1,506,960 < q`.
K is auxiliary to each opening and need not agree across different response
challenges.

Committing K after rho would let the prover fit a single field equation.
Omitting its range check would remove the integer lift.
Both operations violate this protocol.

#### Parity and value carries

Keep nine parity digits and six value digits.
Their negative magnitudes are 76,695,844 and 149,796.
Their positive reaches are 57,521,883 and 112,347.
Let `M = 4m` be the scalar-row count and define
`R_par = 3*(162*M*4 + C*46*4)`.
The required malicious-U lifts are

```text
R_par + 2*76,695,844 < p
3*162*C*4 + 1 + 2*149,796 < p.
```

| Geometry | Parity carry bound R_par/2 | Honest value carry bound | Parity lift | Value lift |
| --- | ---: | ---: | ---: | ---: |
| Fixture | 4,440 | 486 | 153,400,568 | 303,481 |
| Envelope | 31,921,152 | 62,208 | 217,233,992 | 797,257 |

These coefficientwise bounds are unchanged and fit p.
They do not multiply by the owner count.
Keep Gamma 92 and the full malicious U interval `[-4,3]`.
At the boundary geometries, doubling m beyond 8192 or C beyond 256 fails
positive carry reach before it fails p64 no-wrap.
A stronger column-challenge family requires new carry analysis.

#### Lane period

The P in the existing typed-prefix inequality `0 <= u < P*n` denotes a lane
period, not the proof prime.
Use the name `period = 64/gcd(n,64)` in the migrated description and code.
It is 32 for degree 162 and 8 for degree 648.
Both give `period*n = 5184`.
This is a renaming with no arithmetic change.
Retain checked indexing, alignment, natural-address support and allocation
limits.
New digit depths and ranks still change the prefix layout identity.

#### Prime planes

The explicit extension claim shape and its two-point capacity are specified in
[binary-source-p64-extension-planes.md](binary-source-p64-extension-planes.md).
That amendment supersedes the base-only API discussion below for that shape;
the base single-point encoding remains unchanged. Admission is still separate.

Use 22 balanced base-eight digits for each base coordinate, replacing 43.
Centered F_p coefficients have endpoints
`+/-9,223,372,036,854,775,778`.
Twenty-one digits reach only 3,952,873,730,080,618,203 on the positive side.
Twenty-two have positive reach 31,622,989,840,644,945,627 and negative
magnitude 42,163,986,454,193,260,836.
Even every canonical nonnegative p64 residue fits that positive reach.
Keep centered representatives as the production convention.

The accepted EP digit envelope exceeds p.
EP reading equations are field identities and do not require a unique integer
representative. Do not impose `B_EP < p/2`.
Point and value validation uses p, and each canonical base coordinate occupies
eight bytes.
Public statement bytes are separate from projection proof bytes.

At `C = 256`, one base-coordinate plane has 912,384 natural digits and
1,441,792 physical cells.
Five have 4,561,920 natural digits and 7,208,960 physical cells.
Prime-only D has `ceil(planes*22*162*C/64)` columns.
Word U adds `s*162*C` compact digits before division into degree-64 columns.

A base-field prime claim keeps one plane per separable claim.
An admitted E64 claim needs two base-coordinate EP planes, each with 22 digits.
Its reading uses extension arithmetic.
The current `PrimePlaneClaim` API carries base residues and admits at most one
plane per distinct owner at the shared point: `t <= s` for t planes and s owners.
Repeated owner indices and more planes than owners are rejected.
Supporting E64 points and values therefore needs an API and layout change.
Count both coordinate planes against the resource cap, or enlarge and reprice
the cap explicitly.
The five-plane cost and rank tables in this spec refer to base-field claims.
Random testing challenges are E64 in either case.

#### Agreement and units

The binary/prime agreement inequality remains
`2*Gamma + Delta = 184 + 4095 = 4279 < p`, with `Delta = 4095`.
Its error is zero conditional on the application's proof of signed-bit
membership and zero on every committed non-payload coefficient of the
extracted source.
This migration does not provide that proof.
Boolean input buffers do not establish it.
Preserve the session, owner, manifest and terminal binding in `agreement.rs`.

The prime-unit proof changes.
The old proof uses `ord_243(P) = 6` and `P^6 > 276^81`.
The evidence checks `ord_243(p) = 162` by modular exponentiation.
Hence the degree-162 cyclotomic polynomial `Phi_243` is irreducible over F_p.
A nonzero challenge difference has degree below 162 and coefficient magnitudes
at most two.
It remains nonzero modulo p and is a unit in `F_p[X]/Phi_243`.
Scalar extension to E64 preserves its inverse even though the polynomial
factors over E64.
The alternative norm contradiction also holds because `p^162 > 276^81`.
The fixed signed-support rule continues to give nonzero reduction in
GF(2^162).

#### B and D ranks

Reprice B and D using the degree-64, coefficient-difference-bound-seven row of
[the q64 SIS table](../crates/akita-params/src/sis/generated_sis_table/q64.rs).
Its maximum widths for ranks one through five are respectively
`296`, `76,306`, `4,030,488`, `106,021,444` and `1,826,815,994`.
The corresponding old q128 rank-one and rank-two widths were 76,306 and
106,021,444.
These are estimator outputs.
The evidence checks selection against the copied row and does not certify
lattice hardness independently.

Use the widths

```text
B_columns = s*C*3*8*648/64
D_columns = ceil((t*22*162*C + s*162*C)/64), 0 <= t <= s
```

for t base planes plus all-owner word U. Each row below is admitted by the
single-point plane-count contract; five planes require at least five owners.

| Geometry; owners; planes | B columns | Minimum B rank | D columns | Minimum D rank |
| --- | ---: | ---: | ---: | ---: |
| Fixture; 1; 1 | 486 | 2 | 117 | 1 |
| Fixture; 5; 5 | 2,430 | 2 | 583 | 2 |
| Fixture; 4096; 5 | 1,990,656 | 3 | 21,293 | 2 |
| Envelope; 1; 1 | 62,208 | 2 | 14,904 | 2 |
| Envelope; 5; 5 | 311,040 | 3 | 74,520 | 2 |
| Envelope; 10; 5 | 622,080 | 3 | 77,760 | 3 |
| Envelope; 4096; 5 | 254,803,968 | 5 | 2,725,488 | 3 |

The single-owner fixture needs B rank two and D rank one; the single-owner
envelope needs rank two for both. Rank two does not cover every larger-owner width.
Each role must meet its minimum rank.
A shared-rank implementation must select the maximum of the two minima.
An uncompressed rank-r base payload occupies `64*r*8 = 512r` bytes.
The old rank-two payload occupied 2048 bytes.
K is absent from D, but its outgoing witness and all induced setup-prefix
commitments still require SIS repricing.

Keep A's degree 648, rank three and modulus q for this projection design.
The visible generated commitment-prime certificate table concerns the old q25
profile. It does not certify this q24 root.
Experimental rank-three tests do not establish security admission.
Certify the q24 instance, B/D instances, ordinary suffix and induced setup
instances under the current policy before a production claim.

#### Removal of the y encoding

Delete the seven y messages and their signed 95-bit magnitude encoding.
Delete the associated encoding-based polynomial-count cap.
At the fixture the old path admitted 4096 owners under that cap.
At the envelope it admitted ten: eleven gives
`40,473,353,320,142,885,026,607,923,200 >= 2^95`.
This restriction disappears because the new protocol sends no y.
The descriptor cap, resource limits, schedule admission and security admission
still apply independently.
Shrinking or widening y alone would not repair the old p64 lift.
Retain canonical-zero and magnitude validation at any remaining integer wire
boundaries.

#### Field assumptions and source map

The following files contain degree-one restrictions, fixed-P assumptions or
integer-only paths that the migration must replace or validate.
Paths without a crate prefix in this table are under
`crates/akita-types/src/binary_source/`.

| File | Existing assumption or boundary | Required p64/E64 behavior |
| --- | --- | --- |
| `root_transcript.rs` | The statement rejects extension degree other than one and lifts base payloads under the degree-one atom convention. | Admit the selected quadratic profile and bind base payloads canonically as base values or explicitly embedded extension values on both sides. Version the statement identity. |
| `crates/akita-prover/src/protocol/prove/binary_root.rs` | Root admission checks extension degree one; the A operand callback uses `degree_one_base`. | Admit E64 and send both base coordinates through integer adjoints. |
| `crates/akita-verifier/src/fold/binary_root.rs` | Root admission checks degree one and the P128 modulus. | Validate the paired p64/E64 profile consistently with the prover. |
| `root_plan.rs`, `root_public.rs` | Bounds and relation layout select the fixed P profile and seven projection coordinates. | Use p64 bounds, the K segment and `a` coefficient coordinates. |
| `prime.rs`, `prime_plan.rs`, `prime_weights.rs` | Base-residue claims, 43-digit geometry and P-only validation or weights. | Use p, 22 digits per coordinate and explicit two-plane accounting for any E64 claims. |
| `crates/akita-cpu-backend/src/binary_root/prime.rs` | Prime digit production checks P and centers P residues. | Produce centered p64 coordinate digits. |
| `authentication.rs`, `commitment_manifest.rs` | P-valued streams, fixed rank-two B/D shape and payload sizing. | Sample p-valued setup, select sufficient ranks and bind new sizes. |
| `crates/akita-params/src/binary_setup.rs` | Typed modulus labels, setup sampler and stream identities select P128. | Define a distinct p64 profile and sampler; preserve the q-valued A stream. |
| `crates/akita-cpu-backend/src/opaque/binary_source.rs`, `opaque/binary_source/digit_commitment.rs` | Import guards, bound construction and digit-commitment arithmetic use the old base profile. | Admit the new base field and price the new ranks and widths. |
| `grinding.rs` | Challenge order is constructed from `(128,1,P)`. | Use `(64,2,p)` and rederive each loss and query site, including dense coefficient batching. |
| `root_projection_weights.rs` | `BinaryAOperand` expects an integer representative of E. | Apply the A adjoint separately to rho's two base coordinates and recombine in E64. |
| `crates/akita-cpu-backend/src/binary_root.rs`, `binary_root/residual.rs` | Residual helpers include concrete P128 return values alongside exact integer computation. | Keep integer residuals for quotient production and embed into the selected base/extension fields explicitly. |
| `ordinary_root.rs`, `ordinary_root_reference.rs`, `reference_weights.rs` | Ordinary/reference setup and weights use concrete P128 matrices or P guards. | Use the selected p64 ordinary profile and compare canonical and optimized E64 relations. |
| `typed_prefix.rs`, `typed_prefix_weights.rs`, `a_product.rs` | Generic F/E interfaces are orchestrated only by the admitted degree-one root. | Test base setup coefficients with E64 factors, claims and successor consumption. Bind changed layouts and ranks in the prefix identity. |
| `crates/akita-types/src/proof/commitment/binary.rs` | `new_binary` requires the P128 proof modulus and the constant payload length `BINARY_COMMITTED_PAYLOAD_COEFFICIENTS`. | Require the selected p64 profile. Size the payload from the manifest's admitted B rank. |
| `crates/akita-types/src/proof/commitment/wire.rs`, `crates/akita-verifier/src/fold/admission.rs` | Validation and deserialization compare binary payloads with the same constant rank-two length. | Derive the expected payload length from the manifest's admitted B rank. |
| `crates/akita-cpu-backend/src/opaque/binary_ordinary.rs` | Signed source decoding, reached from `opaque/binary_source.rs`, subtracts canonical coefficients from `BINARY_SETUP_P`. Over p64, a valid coefficient -1 is encoded as p-1 and would fail the signed-width check. | Decode signed values against the selected field modulus. |

The canonical relation compiler remains the source of truth.
Do not add a parallel p64 compiler with duplicated bounds.
Version manifest and typed-prefix identities where the profile changes.
Invalidate incompatible persisted setup artifacts through those identities.
A remains the same centered q-valued setup stream and embeds faithfully into
F_p.
All ring-switch, Stage 1, Stage 2 tau/alpha and sumcheck challenges in the
odd-characteristic root must range over E64.
The binary field-switch challenges retain their separate binary field.
Test fixtures and byte pins that instantiate only P128 do not validate E64.
Add genuine extension values to the existing fixture and end-to-end tests.

#### Direct-A CRT kernel

There is no field-compatible integer representative for `a0 + a1*u`.
Write rho as two F_p coordinate vectors.
Apply the base-valued A adjoint to each vector and recombine the results in
E64. This is exact by F_p linearity.
Apply the same rule to every integer-only optimized extension path.

For each base coordinate, retain the current sufficient CRT reconstruction
bound for 43-bit limbs:

```text
2*3*3*648*B_A*2^43
  = 731,182,933,063,452,850,126,848
  < 67,091,329*134,217,649*268,433,353
  = 2,417,199,514,328,443,471,521,913.
```

A p64 representative needs at most two such limbs instead of three.
Do not concatenate the two extension coordinates into one integer.
The extra coordinate pass requires measurement.

The auxiliary smooth 128-bit transform prime used for integer residual
construction is separate from the proof prime.
It may remain with its existing exact-lift checks.
The new residual bound fitting p64 does not provide convenient transform
roots of unity.
Keep checked i128 accumulation where used.

#### Binary field switch

Keep `H = GF(2^128)` and `B = GF(2^162)`.
Keep 128 H carrier partials, seven B row-selector coordinates and B-round
challenges. The embedding is F2-linear and is not an odd-characteristic field
homomorphism.
The switch produces the same B terminal.
Only its surrounding parity/value authentication moves to the new proof
field.
For n field-switch rounds, the error remains `(7+2n)/2^162`.
At `n <= 23` it is at most `53/2^162 < 2^-156` per selected word frontend.
Charge multiple frontends separately.
The fixture frontend remains 2,195 bytes.

#### Mixed ordinary profile

The optional ordinary root currently admits a P128 profile with two 128-bit
compression maps.
Select an actual p64 ordinary profile with a declared compression and norm
contract. Recompute its complete preimage segments, compression bit tables,
O relation coordinates and suffix costs.
Do not reinterpret old compressed payloads or old bit tables as p64 data.
Retain the common outgoing witness and one fused Stage 2 and Stage 3.
The field migration alone does not define the mixed profile's admission.

### Security accounting

The projection relation contributes at most `1/e` at the common independent
coefficient/relation batch, conditional on its fixed-witness premise.
Its integer lift and divisibility implication contribute zero additional error.
The old `q^-7` annihilation term disappears.
The exact field-size comparison is

```text
2^-128 < 1/e < 2^-128 * (1 + 2^-56).
```

This term alone is slightly larger than `2^-128`.
It is not a whole-protocol security claim.

| Event | Algebraic charge over E64 |
| --- | --- |
| Independent coefficient and other root relation batch | At most `1/e`, charged once for the common vector |
| Conservative root ring switch | `648/e` |
| Prime-polynomial check | `161/e`, already covered by the conservative root ring-switch charge |
| Other independent range/linear binding batches | Their actual independent batch losses over e |
| Stage 1 range proof | Its actual stage degrees, round counts and interstage batching losses over e |
| Fused Stage 2 with r rounds | `3r/e` |
| Typed-prefix Stage 3 with k rounds | `2k/e` |
| Binary carrier field switch | `(7+2n)/2^162` per selected frontend |

For `r = 26`, Stage 2 alone contributes `78/e > 2^-128`.
For `k = 18`, Stage 3 contributes `36/e > 2^-128`.
Its field messages still occupy `32k + 32 = 608` bytes at k = 18,
because an E64 element occupies 16 bytes.
This excludes grinding nonces.
Stage 3 setup coefficients are base values, while equality factors, round
messages, product claims and evaluations are extension values.
Its denominator is e, not p.

Use the complete Fiat–Shamir/grinding analysis with the new challenge order
and query sites. Include retry, hash, binding, setup and ordinary-suffix losses.
Do not divide interactive error by grinding work without the corresponding
security theorem.
Two independent coefficient tests on the same fixed witness would give
`1/e^2 < 2^-255` for that step if both are authenticated.
Batching their results back to one scalar would restore a `1/e` bottleneck.
Such repetition is not part of this approved projection schedule.
A full-proof repetition would have to cover all deficient tests and retain
the fixed-commitment premises.

#### Complete research schedule ledger

Research execution is not a production security allocation. A trusted native
catalog supplies the exact outgoing physical span, the typed prefix footprint,
the ordinary producer (when selected), and the complete suffix. Ordinary catalog
artifacts cannot admit P64 roots. Missing selections return typed errors; field,
rank, strict lift, prefix identity, range and suffix checks remain mandatory.
Native-root catalog rows are the execution admission boundary for both P64 and
P128 native roots; there is no separate runtime production gate or research flag.
A caller selecting a validated native catalog can execute P64. Production use
additionally requires closing the ledger obligations below.

The measurement preset deliberately retains `fp128::DenseBounded` search
geometry, basis ranges and decomposition basis for both fields, to compare the
root migration within a fixed degree-64 suffix search space. P64 overrides the
modulus to Q64Offset59 and the opening bound to 64 bits; all suffix candidates
are audited against that field. These measurements are not a benchmark of the
canonical `fp64::Dense` adaptive search, whose suffix dimensions include 128.

For one selected row let r be the outgoing domain's log size, k the typed setup
prefix domain's log size, n the word carrier log size, I=s*C, and e=p^2.
The following ledger includes every class of loss; unresolved premises are not
assigned zero. The public grinding plan records the actual per-site degrees.

| Event | Charge and admission obligation |
| --- | --- |
| Coefficient and other fixed root rows | One `1/e` batch, conditional on the common bounded-witness checkpoint; no factor `1944*s` and no extra `q^-7` |
| Integer quotient and carry lifts | Zero additional algebraic error only after range checks and the strict characteristic-p bounds |
| Root ring switch | `648/e`; the prime polynomial's `161/e` is included, and mixed native degrees must not exceed 648 |
| Range equality point | `r/e`, at the root Tau0 site |
| Stage 1 | Sum `(degree+1)/e` over the actual `DigitRangePlan::new(8)` round shapes, plus `(child_claims-1)/e` at each nonempty interstage powers batch |
| Stage 2 | Separate range-image binding `1/e`, then `3r/e` for the fused sumcheck |
| Stage 3 | `2k/e` for the one typed A/B/D product; setup and witness evaluations continue into the ordinary successor |
| Word frontend | `(7+2n)/2^162` when selected; zero when absent |
| Column extraction | `I/2^136` before its work filter, retaining the indexed-oracle premise and query units below |
| Ordinary suffix and terminal | Every retained plan site and every scheduled binding instance, including the optional ordinary input; no discount from sharing the root |
| Field codecs | Charge pooled challenge messages under the transcript codec's `2^-160` budget per message; the separate exact P64 coefficient sampler has abort at most `2*batch_count*(59/2^64)^256`, not a soundness improvement |
| Response and work retries | Every candidate enters the adversarial query budget. The 4096 response trials are not 12 security bits. Conditional response-tail bounds and bounded work exhaustion are completeness obligations |
| Setup | Include the total sampling/model distance for typed A/B/D, outgoing, prefix and suffix setup streams, and trusted setup/context authenticity |
| Binding | Sum actual q24 A, q64 B/D, ordinary/compression, outgoing, typed-prefix and suffix matrix-collision advantages at the extractor's full cost |
| Hashes | Include each distinct identity/commitment/setup hash role and the transcript/XOF model; for a 256-bit ideal digest role use `(binom(Q_h,2)+Q_h*V)/2^256` where applicable |

The q24 comparison radius is 1,506,960. B/D use their actual degree-64 widths
and coefficient-difference bound seven, not the old q128 ranks. Generated suffix
rows audit their own ordinary instances; this does not certify A or the complete
native checkpoint. Induced prefix instances must be included in the same
certificate set, not inferred from the visible B/D lookup alone.

For every priced site a use the exact conditional debit d_a, set size S_a,
work g_a and adversarial/final-verifier query counts q_a and v_a. A candidate
aggregate bound is the sum of `(q_a+v_a)*d_a/(S_a*2^g_a)`, unpriced frontend
terms, codec/setup distances, hash terms and all induced binding advantages.
Do not also add an envelope `Q*max(rate)` for those same queries. A plan's
one-replay structural query count is not an adversarial query bound, especially
for the indexed SHAKE columns and the blockwise coefficient sampler.

Production use requires the fixed-witness checkpoint, independently
programmable indexed column streams, full-tree debits, extraction branching and
cost, query budget and every induced SIS certificate to be established together.
A local 128-bit per-query grinding target is not a whole-proof 128-bit claim.
Linux timing, byte pins and mutation tests provide implementation evidence only.

#### Column-challenge proof of work

The current weight-46 column-challenge family has support size

```text
S46 = sum_(j=0..46) binom(162,j)
    = 107,268,495,235,892,251,993,000,716,928,106,195,079,952
2^136 <= S46 < 2^137
I = s*C
extraction-address term per response candidate = I/S46
```

The current root already prices this site.
`crates/akita-types/src/binary_source/grinding.rs` places one
`FoldChallengeWork` entry after the root response nonce.
The entry counts all I column coordinates and uses the family's guaranteed
set size `2^136` as its denominator, as in
[transcript grinding](transcript-grinding.md):

```text
g = least g with I * 2^128 <= 2^136 * 2^g
```

| I | Example | g |
| ---: | --- | ---: |
| at most 256 | one owner at `C = 256` | 0 |
| 512 | two owners at `C = 256` | 1 |
| 1,024 | four owners at `C = 256` | 2 |
| 1,048,576 | 4,096 owners at `C = 256` | 12 |

The migration keeps this entry, its position and its `2^136` denominator.
The exact size S46 would allow `g = 0` up to `I = 315`.
Using it would need a new entry representation, validation, encoding and plan
identity, so the migration does not adopt it.
The field change does not alter I or the column-challenge family.
Only the plan identity changes, through the new challenge order of the other
sites.
The 4096-owner descriptor cap stays a resource limit, not a cap on I for this
event.

The per-query rate of this entry rests on the indexed extraction stage in
[the checkpoint argument](binary-source-checkpoint.md).
That stage replaces one column coordinate and keeps every other coordinate.
The root draws its coordinates by squeezing one 32-byte seed and expanding
indexed SHAKE streams (`crates/akita-challenges/src/fold_draw.rs`,
`crates/akita-challenges/src/binary/sampler.rs`).
The argument therefore needs each indexed stream to be modeled as an
independently programmable oracle, with its queries counted.
This premise is shared with the current P128 root and is listed under open
questions. The migration neither strengthens nor weakens it.
E64 batching does not affect this term. The full security ledger still adds
every other algebraic, retry, setup, binding and hash term.

### Alternatives Considered

The completed design study selected committed coefficient quotients.
This decision is fixed for implementation.
Public limbs and elementary-product chunks require many authenticated partial
sums at the envelope.
Smaller projection entries need more rows and segmentation outside the tiny
fixture. Clear coefficient quotients expose the quotient vector and add
explicit bytes per owner.
Per-operation modulo-q gadgets add intermediate witnesses even though the
complete coefficient residual already fits p64.
The committed terminal quotient keeps exact coefficientwise extraction and
adds no explicit quotient message.
The evidence retains the study's exact alternative calculations for audit.
They are not selectable protocol fallbacks.

## Documentation

This change adds this approved spec and its evidence script.
The only change to an existing spec is the short migration pointer in
[binary-source-opening.md](binary-source-opening.md).
Its current-protocol description remains intact until implementation.

Implementation PRs must update the opening and checkpoint descriptions with
the new ordering, unit proof and security ledger.
Fold durable content into `book/src/how/architecture.md`,
`book/src/how/transcript.md`, `book/src/how/security.md` and
`book/src/how/verification.md` as implementation lands.
Update the affected profile and feature usage pages with actual admission
limits. Keep this spec approved until implementation starts, then track its
lifecycle under [the pruning policy](PRUNING.md).
Archive it only after its durable content is folded into the Book.

## Execution

### Staged implementation PRs

Target dev for the selected p64 binary extension.
Land any shared extension seams on main first under the branch contract.
Each PR keeps unselected ordinary behavior byte-identical.
Stages 1–5 provide the paired-field arithmetic and typed setup machinery.
Stage 6 adds research catalog admission and the complete measurement driver.
Its Linux-generated catalogs, singleton pins and measured results are pending;
no production security allocation is admitted.

1. **Define the paired field profile and identities.** Touch
   `crates/akita-params/src/binary_setup.rs` and `binary_setup/tests.rs`;
   `crates/akita-types/src/binary_source/commitment_manifest.rs`,
   `root_transcript.rs`, `authentication.rs` and `grinding.rs`;
   `crates/akita-types/src/proof/commitment/binary.rs` and `wire.rs`;
   `crates/akita-verifier/src/fold/admission.rs`.
   Define the p64 setup sampler, rank admission and versioned identities.
   Size binary commitment payloads from the manifest's admitted B rank at
   construction, validation and deserialization.
   Audit base payload encodings and the `(64,2,p)` challenge order.
   Add degree-two profile checks to
   `crates/akita-prover/src/protocol/prove/binary_root.rs` and
   `crates/akita-verifier/src/fold/binary_root.rs` without exposing an incomplete
   runtime profile.
2. **Add the quotient layout and canonical relation.** Touch
   `crates/akita-types/src/binary_source/projection.rs`, `projection_rows.rs`,
   `root_plan.rs`, `root_public.rs`, `root_request.rs`,
   `root_projection_weights.rs`, `evaluation_bounds.rs` and `reference_weights.rs`.
   Replace old projection bounds with the coefficient lift and K digit depth.
   Add K's natural-address weights and remove J/y from the selected profile.
   Extend `root_plan_tests.rs` and `root_projection_weights_tests.rs`.
3. **Produce and authenticate K end to end.** Touch
   `crates/akita-cpu-backend/src/binary_root.rs`, `binary_root/residual.rs`,
   `opaque/binary_source.rs`, `opaque/binary_source/projection.rs` and
   `opaque/binary_source/digit_commitment.rs`;
   `crates/akita-prover/src/protocol/prove/binary_root.rs` and
   `crates/akita-verifier/src/fold/binary_root.rs`.
   Use the accepted residual cache, exact divisions and the order-five segment.
   Switch order 7 to dense independent E64 weights.
   Range-check K through the existing Stage 1 and authenticate one Stage 2.
   Extend `binary_root_tests.rs`, `binary_root/stage2_oracle_tests.rs` and the
   end-to-end projection and fault modules.
4. **Port integer adjoints and typed Stage 3.** Touch
   `crates/akita-types/src/binary_source/root_projection_weights.rs`,
   `a_product.rs`, `typed_prefix.rs`, `typed_prefix_weights.rs` and
   `authentication.rs`; `crates/akita-cpu-backend/src/binary_root/setup_prefix.rs`;
   the prover and verifier binary-root orchestration files.
   Split integer adjoints by base coordinate and retain their exact CRT bounds.
   Authenticate the repriced A/B/D prefix with extension claims.
   Extend `a_product_tests.rs`, `typed_prefix_tests.rs`, `authentication_tests.rs`
   and end-to-end `stage3.rs` and `stage3_catalog.rs`.
5. **Migrate prime and mixed profiles.** Touch
   `crates/akita-types/src/binary_source/prime.rs`, `prime_plan.rs`,
   `prime_weights.rs`, `ordinary_root.rs`, `ordinary_root_reference.rs` and
   `agreement.rs`; `crates/akita-cpu-backend/src/binary_root/prime.rs` and
   `opaque/binary_ordinary.rs`;
   `crates/akita-pcs/tests/binary_source_end_to_end/prime_plane.rs` and
   `crates/akita-cpu-backend/src/binary_root_mixed_tests.rs`.
   Use 22 digits per base coordinate and the new unit proof.
   Gate any E64 claim API on explicit two-plane resource accounting.
   Select the actual p64 ordinary compression/norm contract.
   Decode signed ordinary sources against the selected field modulus.
6. **Admit and measure the complete schedule.** Touch
   `crates/akita-types/src/binary_source/grinding.rs`,
   `crates/akita-pcs/examples/binary_source_full_size.rs` and its
   `catalog.rs`, `native_catalog.rs`, `config.rs` and `metrics.rs` modules;
   `crates/akita-pcs/tests/binary_source_end_to_end/byte_contract.rs`,
   `compatibility.rs` and `statement_binding.rs`.
   Generate the selected p64 catalog and induced setup instances.
   Keep the column-challenge work entry and complete the security ledger
   before production admission. Record complete size and timing results.
   Update the owning Book pages and spec lifecycle in the implementing PRs.

### Open questions

- Which p64 catalog and successor schedules cover the intended word, prime and
  mixed shapes? Complete proof bytes, time and memory remain unmeasured.
- What complete security allocation will production admit? The q24, B/D,
  suffix and setup instances need certification under that allocation.
- Which extraction argument covers the seed-plus-indexed-SHAKE column sampler?
  The indexed stage of the checkpoint argument replaces one coordinate at a
  time. It needs each indexed stream modeled as an independently programmable
  oracle with its queries counted, or a sampler whose coordinates are encoded
  independently in the transcript. This question applies equally to the
  current root.
- Will the prime API remain base-field-only or admit E64 points and values?
  E64 support requires two coordinate planes and an explicit cap decision.
- Will B and D keep a shared rank or expose separate ranks?
  Either choice must meet the per-role minima above.
- The research mixed root selects Q64Offset59 through the generated ordinary
  producer row, with two compression maps checked by `OrdinaryNativeRootPlan`.
  Production certification of that complete instance remains open.
- Does the generalized checkpoint comparison establish the required fixed
  witness for the extension compiler and shared setup/suffix claim?
  Validate that premise alongside negative carries, quotient mutations and
  genuine non-base challenges.

## References

- [Exact arithmetic evidence](evidence/binary-source-p64-root/bounds.py).
  Run `python3 -I specs/evidence/binary-source-p64-root/bounds.py` from the
  repository root. It uses only standard-library integer and Fraction
  arithmetic and exits zero when all assertions pass.
  It retains the completed study's alternative tables.
  It checks arithmetic and copied SIS-row rank selection, not lattice hardness
  or the checkpoint extraction theorem.
- [Binary source grouped opening](binary-source-opening.md).
- [Binary source checkpoint and adjoint](binary-source-checkpoint.md).
- [Binary source setup offload](binary-source-setup-offload.md).
- [Binary source zero padding](binary-source-zero-padding.md).
- [Documentation policy](../docs/documentation.md).
- [Verifier contract](../docs/verifier-contract.md).
