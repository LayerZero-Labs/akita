# LaBinius binary-root matrix and shape admission

Status: active
Book-chapter: book/src/foundations/security.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

The second commitment-modulus profile and its conditional lift are specified in
[`labinius-small-modulus-root.md`](labinius-small-modulus-root.md).

## Scope and profile

The opt-in `labinius-sis` feature supplies a runtime certified-width lookup and
one closed root profile. `LabiniusRootShape::derive` is the common admission
function for future prover, verifier and planner callers. Enabling the feature
does not select the extension or change the ordinary protocol.

| Profile item | `D648P128BoundedW46Delta16` |
| --- | --- |
| Scalar ring | `BinaryScalarRing::Cyclotomic243`, degree 162 |
| Commitment ring | `LabiniusRingDegree::D648`, packing degree 4 |
| Coefficient prime | `LabiniusCoefficientPrime::P128OffsetA7F7` |
| Challenge family | Bounded weight 46 on `Cyclotomic243` |
| Accepted response interval | `[-32768, 32767]` |
| Accepted response diameter | `Delta = 65535` |
| Stable wire tag | 0 |

Unknown tags reject. Adding a profile requires a new enum variant. Identity
bytes begin with `akita/labinius/root-profile/v1` and a zero byte, then encode
the tag, scalar-ring tag, exact prime (16 little-endian bytes), commitment degree
(4 little-endian bytes), challenge identity, both signed interval endpoints
(16 little-endian bytes each), and the 32-byte certified-table digest. Challenge
bytes are framed as `(1, byte)` pairs followed by zero, so variable lengths
cannot alias subsequent fields. This binds every norm, entropy, sign-rule and
accepted-range choice as well as the table version. Challenge construction and
identity accessors return errors rather than panicking on a violated profile
invariant.

## Geometry and source admission

Issue sections 2, 3, 4 (direct baseline) and 6 specify the contracts below.
Write `d=162`, `D=648`, `k=D/d=4`, `C=2^log_fold_width`, `N=2^log_num_cells`,
`M=N/C`, and `m=M/k`. N counts F162 scalar cells, M scalar cells per column,
and m commitment-ring elements per column. M must be a positive multiple of k.
Geometry and coefficient counts use checked `usize` arithmetic; norm arithmetic
uses checked `u128` arithmetic. No allocation is proportional to the geometry.

The challenge profile supplies `w=46` and `Gamma_inf=92`. Its exact support
cardinality must satisfy `|C_family| >= C * 2^lambda_fold`, through
`BinaryChallengeProfile::meets_budget`. This budget is caller allocated and is
separate from the per-instance 128-bit quantum ADPS16 table policy. Deterministic
honest response admission requires `C * Gamma_inf <= min(U, -L)` for source
coefficients of magnitude at most one (issue section 3).

The accepted response diameter, rather than an honest constructor's smaller
range, feeds a single `SourceOccurrenceBound::binary_extracted(Gamma_inf,
U-L)` into `checked_source_comparison_class_bound`. That existing ledger includes
the diagonal comparison and computes `eta_A = 4 * Gamma_inf * Delta`, enforcing
`eta_A < P` (issue sections 3 and 6). Its matrix-view digest is a fixed placeholder
here: this single-occurrence calculation does not compare root layouts or bind
an eventual protocol matrix view. The root protocol must supply that identity.

`labinius_min_secure_rank(prime, degree, eta_A, m)` chooses the smallest rank
with a certified cell of the same prime and degree whose bound and width both
dominate the request. Missing coverage rejects. There is no interpolation,
extrapolation or runtime lattice estimation. A stronger norm cell can admit a
request at the same rank after one particular cell's norm boundary is crossed.

The admitted image has `n_A*C*D` coefficients. The root reduction tests the
matrix-row remainder modulo the degree-D trinomial; each row has degree below
D and tag 0 sends no auxiliary matrix-row witness. The relation and its
`(D-1)/P` evaluation bound are owned by
[the lowered relation](labinius-lowered-root.md#polynomial-rows).

## Direct parity quotient and carry

The direct row of issue section 4 is

```text
R(X) = sum_i B_i(X) V_i(X) - sum_j U_j(X) C_j(X)
     = (X^162 + X^81 + 1) Q(X) + 2 K(X).
```

B and U have bit coefficients; V uses the full accepted interval; C has L1
norm at most w. Each product coefficient in B*V has at most d summands of
magnitude `B_V=max(|L|,|U|)`. Each coefficient in U*C has magnitude at most w.
Thus `H = d*M*B_V + C*w` bounds every coefficient of R, whose degree is at most
322. The same reasoning holds for signed C lifts: their parity reductions
have the specified binary support.

To check the honest bounds independently, write `r_t` for R's integer
coefficients, taking missing terms to be zero. Monic schoolbook division from
highest degree down gives

```text
q_k = r_(k+162)                 for 80 <= k <= 160
q_k = r_(k+162) - r_(k+243)      for  0 <= k <= 79.
```

Indeed only a quotient coefficient 81 positions higher can affect the next
high coefficient. Those higher coefficients have already been assigned their
original residual value. Consequently `|q_k| <= 2H`; Q has 161 coefficients.
The remainder coefficient at `0 <= j <= 161` is
`r_j - q_j - q_(j-81)`, ignoring quotient indices outside `[0,160]`.
Substituting the quotient formulas exposes a cancellation:

```text
rem_j = r_j - r_(j+162) + r_(j+243)    for   0 <= j <= 79
rem_80 = r_80 - r_242
rem_j = r_j - r_(j+81)                for  81 <= j <= 160
rem_161 = r_161 - r_242.
```

For `81 <= j <= 160`, `q_j + q_(j-81) = r_(j+81)`; the two copies
of `r_(j+162)` cancel. Thus the first 80 remainder coefficients have magnitude
at most 3H, and every other coefficient at most 2H. A valid F162 identity holds
exactly when all remainder coefficients are even: integer division commutes
with reduction modulo two because the divisor is monic. Then K is half the
remainder and integral, with

```text
B_Q = 2H
B_K = floor(3H/2).
```

The independent BigInt long-division oracle checks all 162 closed-form
remainder coefficients on random and extremal relation instances and arbitrary
residual polynomials, including extrema that attain the 3H remainder bound.

These are conservative honest construction bounds. They are not permission to
use tighter accepted alphabets in a proof. `check_parity_no_wrap(B_Q', B_K')`
requires each caller-supplied magnitude to cover its honest bound and enforces
the issue's full accepted-envelope inequality
`H + 3*B_Q' + 2*B_K' < P`. The coefficient sum of Phi has three terms, so this
ensures that a checked residual equality modulo P cannot conceal a nonzero
integer coefficient. Rounded digit alphabets must pass their actual enforced
magnitudes. An overflow rejects, as does equality at P.

## Enforced signed digits and root witness lengths

`LabiniusDigitBase` is a closed enum with digit widths `b=1,2,4` and stable tags
0, 1, 2 respectively. Unknown tags reject. `LabiniusRootShape::derive_encoding`
is the single derivation of the accepted integer ranges and witness lengths;
it allocates no vectors. `LabiniusSignedDigitRange` exposes the enforced bit
width e, digit count e/b, signed offset and both inclusive endpoints.
The same signed map applies to response, parity quotient and parity carry:

```text
value = sum_(i < e/b) digit_i * 2^(b*i) - 2^(e-1)
0 <= digit_i <= 2^b - 1
accepted interval = [-2^(e-1), 2^(e-1)-1]
enforced magnitude = offset = 2^(e-1).
```

The profile response interval must exactly equal that centered power-of-two
interval, with e positive and divisible by b. The current profile has e_v=16
for every supported base. For the quotient choose the smallest positive multiple
of b satisfying `2^e_Q >= 2*B_Q+1`; choose e_K by the same rule from B_K.
The +1 ensures the positive honest endpoint fits the smaller positive accepted
endpoint. These are accepted power-of-two ranges, rather than the honest
magnitudes. The derivation calls `check_parity_no_wrap` with offsets
`2^(e_Q-1)` and `2^(e_K-1)` and exposes the checked total
`H + 3*2^(e_Q-1) + 2*2^(e_K-1)`. A rounded envelope reaching P rejects even
when the honest envelope would pass. Both paths use the same total formula.

The committed successor table contains only response digits in the canonical
trinomial response layout, `akita_types::TrinomialResponseLayout`: digits are
innermost, then the coefficient index padded to a power of two, then the ring
element index. Its address rule and lengths are

```text
address = digit + digit_count * (coefficient + padded_coefficient_len * ring_element)
padded_coefficient_len = next_power_of_two(D)
digit_count = e_v/b
response_table_len = m * padded_coefficient_len * digit_count
response_table_log_len = ceil_log2(response_table_len).
```

The parity quotient and carry are `d - 1` and `d` integers, respectively, with
the enforced ranges above; they are not part of the committed response table.
The root protocol must enforce those ranges, either by sending the integers in
the clear and checking them directly or by committing digits. Whichever method
it chooses, the no-wrap condition uses the enforced offsets
`2^(e_Q-1)` and `2^(e_K-1)`.

`LabiniusRootEncoding` exposes the padded coefficient length, response table
length and log length, parity integer counts, and image table length and log
length. Coefficient padding uses checked integer rounding, products and log
rounding use `akita_error::checked`, and range and no-wrap computations use
checked u128 arithmetic. Overflow rejects.

Every position of the response table, including coefficient positions `t >= D`
and any positions above the natural extent, is subject to the same b-bit
alphabet check. Public relation weights are zero at every padding position,
and the honest prover writes the zero digit there. Thus a padding position
cannot contribute to any relation row. No selector polynomial or separate
support proof is needed. Zero here means the unsigned digit value zero, not
the lower endpoint of an entire signed integer map.

The image table uses the same padded coefficient index and has
`padded_coefficient_len * C * n_A` entries over the coefficient field, with no
range proof.

At `(log_num_cells, log_fold_width, lambda_fold) = (22,8,128)`,
`B_Q=173946199040` and `B_K=130459649280`. Rounding gives:

| b | e_Q | e_K | Accepted no-wrap total | Response table length | Response table log length |
| --- | --- | --- | --- | --- | --- |
| 1 | 39 | 38 | 1186484727296 | 67108864 | 26 |
| 2 | 40 | 38 | 2011118448128 | 33554432 | 25 |
| 4 | 40 | 40 | 2835752168960 | 16777216 | 24 |

This gives image table length 262144, log 18.

Tests exercise rounded rejection by deriving a P128 shape with a signed 32-bit
response interval and `(log_num_cells, log_fold_width)=(22,0)`, then perturbing
only its internal prime to P64. The honest total is 14591662792680407500,
below P64; the b=1 rounded total is 19905910352977592366, above P64. The
adjacent log-size 21 passes at b=1. This isolates the no-wrap gate: the perturbed
larger shape cannot itself pass P64's certified SIS lookup.

An exhaustive internal search over both supported primes, the covered centered
response intervals and dyadic D648 geometries finds no certified shape whose
honest envelope passes but rounded envelope fails. The nearest certified P64
boundary has e_v=32, M=2^16, C=2^24, b=4 and rounded total
1824239324833513472, still below `P64=18446744073709527913`. Wider response
intervals exceed the certified norm cells; larger M exceeds certified widths;
larger C violates deterministic honest response admission. This is test-only
parameter exploration, not an additional public profile.

## Admission order and boundary fixture

The canonical derivation applies these gates in order:

1. Represent both powers of two; check ring/scalar compatibility and exact,
   positive packing/column divisions.
2. Check the challenge family's exact fold budget.
3. Check the response interval and deterministic honest response envelope.
4. Use the source-comparison ledger, including its `eta_A < P` check.
5. Select the minimum dominating certified SIS rank; reject missing coverage.
6. Derive image lengths and honest parity quotient/carry bounds with checked arithmetic.

Caller-selected failures use `AkitaError::InvalidSetup`, matching neighbouring
SIS admission. Actual parity range envelopes are checked by `derive_encoding` when the
caller chooses a digit base, or by `check_parity_no_wrap` for another enforced
envelope. The sole profile's eta is fixed below P; tests
perturb internal fields to exercise the ledger boundary without exposing a
runtime profile plugin.

For `(log_num_cells, log_fold_width, lambda_fold) = (22,8,128)` the shape has
`N=4194304`, `C=256`, `M=16384`, `m=4096`, `k=4`, `n_A=1`,
`eta_A=24116880` and image length 165888. The reduction has no A-quotient message.
The direct residual bound is `162 * 2^29 + 256 * 46 = 86973099520`.
At budget 128 the bounded-weight family supports at most 315 columns, so the
largest admitted power-of-two fold width is 256 (log 8). Log 9 rejects at the
challenge gate. With a lower budget it instead fails the honest response gate.

## Table provenance and verification

The frozen CSV is
`crates/akita-sis-estimator/data/labinius_infinity_width.csv`, with 69 certified
rows. It records accepted cutoffs and rejected successors under the existing
quantum ADPS16 policy. The generator example `labinius_infinity_width_table`
can search new offline cells, or emit the runtime Rust file directly from the
checked-in CSV without repeating the lattice search:

```bash
cargo run -p akita-sis-estimator --no-default-features --features labinius-sis \
  --example labinius_infinity_width_table -- --from-csv
```

The emitted `generated_width_table.rs` has canonical sorted cells and a
SHA3-256 digest of the exact CSV bytes. Runtime consumers depend only on
`akita-params`, never on the estimator. The estimator integration test
`labinius_width_table` compares all fields in both directions and recomputes
the digest, forming the drift gate. Fast parameter unit tests cover domination,
profile identity perturbations, admission boundaries, overflow and exact no-wrap
limits. An independent BigInt convolution/long-division oracle and separate
binary reduction cover random, extremal, valid and invalid parity rows.

## Remaining protocol work

This slice defines no schedule family, proof-size model, Fiat–Shamir ledger,
proof wire grammar, representation of the parity quotient and carry, planner
integration, or root protocol. The response table's address rule and the
lengths above fix the committed table's size and index order but not the
relation over it. This is not protocol security admission and does not claim
complete PCS security admission. Those belong to the root
protocol and its composition with the ordinary tail. No unpublished paper or
unlicensed reference implementation is needed for these formulas or tests.
