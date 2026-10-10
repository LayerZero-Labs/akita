# LaBinius binary-root parameter and shape admission

Status: active
Book-chapter: book/src/foundations/security.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Scope and profile

The opt-in `labinius-sis` feature of `akita-params` supplies one closed root
profile, a certified-width lookup, and the two admission functions every
prover, verifier and sizing caller shares: `LabiniusRootShape::derive` for the
geometry and `LabiniusRootShape::derive_encoding` for a proof prime. Enabling
the feature does not select the extension or change the ordinary protocol.

| Profile item | `LabiniusRootProfile::D648Q25BoundedW46` |
| --- | --- |
| Scalar ring | `BinaryScalarRing::Cyclotomic243`, degree 162 |
| Commitment ring | `Z_q[Y]/(Y^648 - Y^324 + 1)`, `LabiniusRingDegree::D648`, packing degree 4 |
| Commitment prime | `LabiniusCommitmentModulus::Q25Plus14561`, `q = 33568993` |
| Challenge family | Bounded weight 46 on `Cyclotomic243` |

The profile fixes nothing else. The accepted response interval follows from
the geometry, and the proof prime is admitted separately; neither is a profile
field. There is one profile and one commitment modulus; adding either needs a
new enum variant.

`q = 2^25 + 14561 = 1 + 23328 * 1439` is prime and congruent to one modulo
`23328 = lcm(1944, 3888, 5832, 7776)`. The limb transform of
`akita_algebra::ring::trinomial::limb` at degree 648 uses its primitive
1944th root of unity and lists q in `TrinomialLimbDomain::ADMITTED_PRIMES`.
Its label is `q25+14561`.

`LabiniusRootProfile::identity_bytes` is

```text
"akita/labinius/root-profile/v1" || 0x00 || scalar_ring_tag:u8
  || q:u32_le || D:u32_le
  || u64_le(len(challenge_identity)) || challenge_identity
  || width_table_digest:[u8;32]
```

Scalar-ring tag 0 means `Cyclotomic243` and 1 means `Cyclotomic729`. The
challenge identity is `BinaryChallengeProfile::identity_bytes`, which binds the
family's ring, weight rule and sign rule. The digest is
`LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE_DIGEST`. Construction returns an error
rather than panicking on a violated profile invariant.

## Geometry and fold budget

Write `d = 162`, `D = 648`, `k = D/d = 4`, `C = 2^log_fold_width`,
`N = 2^log_num_cells`, `M = N/C` and `m = M/k`. N counts F162 scalar cells, M
scalar cells per column, and m commitment-ring elements per column. M must be
a positive multiple of k. Geometry and coefficient counts use checked `usize`
arithmetic; norm arithmetic uses checked `u128` arithmetic. No allocation is
proportional to the geometry.

The challenge profile supplies `w = 46` (coefficient L1 and squared L2 bound)
and the multiplication bound `Gamma = 2w = 92`. Its exact support cardinality
must satisfy `|S| >= C * 2^lambda_fold`, through
`BinaryChallengeProfile::meets_budget`. This budget is caller allocated and is
separate from the 128-bit quantum ADPS16 policy of the width table. At budget
128 the family supports at most 315 columns, so the widest admitted
power-of-two fold is 256 (log 8).

## Fold response

`LabiniusFoldResponse::derive(challenge, C, m, degree)` states one binary fold
`z = sum_{i < C} c_i * s_i` in the inputs of Akita's fold-response cap
([`fold-linf-rejection.md`](fold-linf-rejection.md),
`crates/akita-params/src/sis/fold_linf_cap.rs`) and calls it. It derives no
tail bound of its own:

- challenge family `SparseChallengeConfig::pm1_only(w)`;
- C folded blocks of one claim and `m * D` response coefficients in the union
  bound;
- source norms `(Gamma / w, (Gamma / w) * D)`, where `Gamma / w = 2` bounds the
  coefficients one signed challenge monomial produces from a ring element with
  coefficients of magnitude at most one. Scalar components that are binary in
  the power basis reach only magnitude one, so the cap is conservative for
  them.

`fold_witness_linf_cap` returns `min(beta_inf, t*)` with the deterministic
`beta_inf = C * Gamma`. `num_digits_for_linf_cap` selects the balanced digit
depth at log basis `LABINIUS_BALANCED_LOG_BASIS = 4`. The accepted interval is
the exact balanced digit range of that depth, not the cap:
`[L, U] = [-8 * (16^n - 1) / 15, 7 * (16^n - 1) / 15]` for n digits. The
extracted commitment-relation bound is `eta_A = 4 * Gamma * Delta` with
`Delta = U - L = 16^n - 1`, from
`role_a_collision_inf_norm_for_response_difference`. Akita's
`BalancedSignedDigitFoldPolicy` composes the same three calls, but its query
validation admits only the power-of-two production ring degrees, so they are
called directly.

| Quantity at weight 46, `C = 256`, `m = 4096`, `D = 648` | Value | Source function |
| --- | --- | --- |
| Honest cap `min(beta_inf, t*)` | 1266 | `fold_witness_linf_cap` |
| Balanced base-16 digits | 3 | `num_digits_for_linf_cap` |
| Accepted interval | `[-2184, 1911]` | `checked_balanced_digit_representable_bounds` |
| Diameter `Delta` | 4095 | `balanced_digit_interval_diameter` |
| `eta_A = 4 * Gamma * Delta` | 1506960 | `role_a_collision_inf_norm_for_response_difference` |
| Abort bound per attempt | 7/8 | `FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM`, `_DEN` |
| Response nonce width | 12 bits | `FOLD_RESPONSE_NONCE_BITS` |

The cap depends on `m * D` through the union bound, not on C alone. When
`beta_inf` is below the tail threshold the cap is deterministic and the abort
bound is zero. Otherwise the cap is smaller than the worst case by design:
`C * Gamma = 23552` at the sample geometry. Completeness is then
probabilistic, and the honest prover searches a fold-response nonce as
specified in [the clear opening](labinius-clear-opening.md#fold-response-nonce).
Akita's tail bound conditions on the challenge supports and uses independent
uniform signs. This family derives its signs from the support
(`BinarySignRule::Shake256V1`), so the bound and the abort probability apply
to it only with that map modelled as a random function: they are a
completeness estimate. Soundness uses only the enforced interval.

## Rank

`labinius_min_secure_rank(modulus, degree, eta_A, m)` returns the smallest rank
with a certified cell of the same modulus and degree whose bound and width
both dominate the request. Missing coverage rejects. There is no
interpolation, extrapolation or runtime lattice estimation. The certified
cells cover bound 1506960 only, so a geometry whose response needs a fourth
digit has `eta_A = 24116880` and no cell: derivation rejects it, not a list of
geometries. A stronger norm cell can admit a request at the same rank.

The cells exist only below `(q - 1) / 2`, the bound above which the SIS
instance is trivial. Explicit clear setups, which make no table lookup, check
`eta_A < q` through `checked_source_comparison_class_bound`.

## Direct parity quotient and carry

The parity row of the lowered relation is

```text
R(X) = sum_i B_i(X) V_i(X) - sum_j U_j(X) C_j(X)
     = (X^162 + X^81 + 1) Q(X) + 2 K(X).
```

B and U have bit coefficients; V uses the full accepted interval; C has L1
norm at most w. Each product coefficient in B*V has at most d summands of
magnitude `B_z = max(|L|, |U|)`. Each coefficient in U*C has magnitude at most
w. Thus `H = d*M*B_z + C*w` bounds every coefficient of R, whose degree is at
most 322.

Write `r_t` for R's integer coefficients, taking missing terms to be zero.
Monic schoolbook division from highest degree down gives

```text
q_k = r_(k+162)                 for 80 <= k <= 160
q_k = r_(k+162) - r_(k+243)      for  0 <= k <= 79.
```

Only a quotient coefficient 81 positions higher can affect the next high
coefficient, and those have already been assigned their original residual
value. Consequently `|q_k| <= 2H`; Q has 161 coefficients. The remainder
coefficient at `0 <= j <= 161` is `r_j - q_j - q_(j-81)`, ignoring quotient
indices outside `[0,160]`. Substituting the quotient formulas exposes a
cancellation:

```text
rem_j = r_j - r_(j+162) + r_(j+243)    for   0 <= j <= 79
rem_80 = r_80 - r_242
rem_j = r_j - r_(j+81)                for  81 <= j <= 160
rem_161 = r_161 - r_242.
```

The first 80 remainder coefficients have magnitude at most 3H and every other
coefficient at most 2H. A valid F162 identity holds exactly when all remainder
coefficients are even, because the divisor is monic. Then K is half the
remainder and integral, with

```text
B_Q = 2H
B_K = floor(3H/2).
```

An independent BigInt long-division oracle checks all 162 closed-form
remainder coefficients on random and extremal relation instances and on
arbitrary residual polynomials, including extrema that attain the 3H bound.

## Commitment-row carry

The commitment relation holds modulo q and is proved over the proof prime P.
Its integer form in the commitment ring is

```text
A z - sum_{i < C} c_i T_i = q K_A,
```

with A read as canonical residues in `[0, q - 1]`, the response z of m ring
elements, the image columns `T_i` and a carry `K_A` of `n_A * D` integers. A
reduced coefficient of a trinomial-ring product collects at most three
unreduced coefficients, each a sum of at most D products. An honest image
coefficient is a residue of magnitude at most `q - 1`, so
`q * |K_A| <= (q - 1) * (3 * m * D * B_z + C * Gamma)`, and with `Gamma = 2w`

```text
B_KA = 3 * (m * D * B_z + C * w).
```

`LabiniusCommitmentLift::honest_carry_bound` computes it. It is essentially
independent of q.

## Proof-prime admission and clear-integer ranges

`LabiniusRootShape::derive_encoding(P)` takes the characteristic P of the
proof field as a `u128`. It is the only source of the clear-integer ranges and
table lengths, so a caller cannot obtain ranges that were never checked
against a prime. Primality of P is the caller's premise. Three conditions are
checked, in this order; each failure is `AkitaError::InvalidSetup`.

**1. Challenge differences are units.**
`check_labinius_proof_prime_units(P, challenge)` admits P only if every nonzero
difference of two fold challenges is a unit modulo P. The scalar ring
`Z[x]/(x^deg + x^(deg/2) + 1)` is the ring of integers of the cyclotomic field
of conductor `n = 3 * deg / 2`. A difference s of two challenges of squared
norm at most w has `|N(s)| <= (1.5 * ||s||^2)^(deg/2) <= (6w)^(deg/2)`. A prime
`P != 3` is unramified and every prime ideal above it has norm `P^f`, with f
the multiplicative order of P modulo n, so a non-unit difference has norm
divisible by `P^f`. Hence `P^f > (6w)^(deg/2)` suffices. The check compares
bit lengths: it accepts when `f * (bits(P) - 1) >= (deg / 2) * bits(6w)`,
which implies the inequality in exact integers, and it rejects any P divisible
by three. It is a sufficient test, not the exact comparison; a test brackets
it against the exact one. At weight 46 on `Cyclotomic243` the right side is
729:

| Prime | f | Left side | Result |
| --- | --- | --- | --- |
| `2^64 - 59` | 162 | 10206 | admitted |
| `2^128 - 275` | 6 | 762 | admitted |
| `2^64 - 23703` | 1 | 63 | rejected |
| `2^128 - 2^32 + 22537` | 1 | 127 | rejected |

The relation that uses this condition is the prime row of the lowered
relation: it isolates one column of the prime left opening by cancelling a
difference of two fold challenges modulo P, and the object a prime claim is
about, the extracted source reduced modulo P, is defined only when that
difference is a unit
([lowered relation](labinius-lowered-root.md#prime-opening-and-the-unit-condition)).
The commitment rows and the parity row are lifted to the integers and do not
use it. Admission checks it for every opening mode, so a proof prime admitted
for a binary claim is admitted for a prime claim, which adds no condition.

**2. The commitment rows cannot wrap.** Each clear integer has the narrowest
two's-complement range covering its honest bound:
`LabiniusSignedRange::covering(bound)` has `e = bitlen(bound) + 1` bits,
interval `[-2^(e-1), 2^(e-1) - 1]` and enforced magnitude `2^(e-1)`. These
widths have nothing to do with the digit base.
`LabiniusCommitmentLift::check_no_wrap(P, B_K)` is called with the enforced
magnitude `B_K = 2^(e_KA - 1)` of the commitment carry. It rejects a magnitude
below `B_KA` and requires

```text
3 * m * D * (q - 1) * B_z + C * Gamma * B_T + q * B_K < P.
```

The first term bounds `A z`, the second the image term, the third the carry
term. `B_T` is the magnitude the image digit encoding enforces. An image
coefficient is committed as `sum_{l < k} digit_l * 16^l - B_T` with stored
digits in `[0, 15]`, so the alphabet check of the root reduction confines it
to `[-B_T, 7 * (16^k - 1) / 15]` with `B_T = 8 * (16^k - 1) / 15`. The count
k is the least whose positive reach covers the honest value, a canonical
residue in `[0, q - 1]`. `labinius_image_digit_bound(q)` returns
`(7, 143165576)` for `q = 33568993`. Every term of the inequality therefore
bounds a quantity the reduction checks: the response and image alphabets and
the carry range. A commitment row that holds modulo P between such values
holds over the integers.

**3. The parity row cannot wrap.** With the enforced magnitudes
`B_Q' = 2^(e_Q - 1)` and `B_K' = 2^(e_K - 1)`,

```text
H + 3 * B_Q' + 2 * B_K' < P.
```

The coefficient sum of Phi has three terms, so a parity identity that holds
modulo P between values in these ranges holds over the integers. Whenever
`q >= 7` condition 2 implies condition 3; both are checked, also for a
statement with a prime claim alone, which has no parity row.

`LabiniusRootEncoding` then fixes the tables. The committed response table
holds stored base-16 digits in `[0, 15]` in the canonical
`akita_types::TrinomialResponseLayout`: digits innermost, then the coefficient
index padded to a power of two, then the ring-element index.

```text
address = digit + digit_slots * (coefficient + padded_coefficient_len * ring_element)
digit_slots = next_power_of_two(digit_count)
padded_coefficient_len = next_power_of_two(D)
response_table_len = m * padded_coefficient_len * digit_slots
image_digit_slots = next_power_of_two(image_digit_count)
image_table_log_len = ceil_log2(image_digit_slots * padded_coefficient_len * C * n_A)
prime_table_log_len = ceil_log2(padded_coefficient_len * C)
```

A packed coefficient p with offset o is stored as
`p + o = sum_{l < digit_count} digit_l * 16^l`; the offset rule is owned by
[the lowered relation](labinius-lowered-root.md#response-and-image-tables).
Every table position, including digit slots `l >= digit_count`, coefficient
positions `t >= D` and positions above the natural extent, is subject to the
same alphabet check. Public relation weights are zero at every padding
position and the honest prover writes the digit zero there. No selector
polynomial or separate support proof is needed. The parity quotient, parity
carry and commitment carry are `d - 1`, `d` and `n_A * D` clear integers
outside this table.

The committed image table holds stored base-16 digits in the same order:
digit innermost in `image_digit_slots` slots, then the padded coefficient
index, then the image entry `col * n_A + i`. An image coefficient T is stored
as `T + image_offset = sum_{l < image_digit_count} digit_l * 16^l` with
`image_offset = B_T`, under the same alphabet check and padding rule.
`LabiniusRootEncoding` exposes `image_digit_count`, `image_digit_slots`,
`image_offset` and `image_table_log_len`, so a committer shapes the table
from the admitted encoding without restating a constant.

The prime left opening, bound only for a statement with a prime claim, is a
table of challenge-field elements with no digit axis: the padded coefficient
index innermost, then the fold column, `padded_coefficient_len * C` entries.
It has no alphabet and no range, so it enters no admission inequality; the
encoding exposes only its length, `prime_table_log_len`.

## Challenge-field admission

The root reduction draws its field challenges from a field E containing the
proof field, of exact cardinality `P^e` for extension degree e. Conditions 1
to 3 concern P alone and are unaffected by E. The size of E is admitted by
Akita's transcript-grinding rule
([transcript grinding](transcript-grinding.md#security-model)): a challenge
whose bad fraction is at most L over the cardinality needs the least g with
`L * 2^128 <= P^e * 2^g` bits of proof of work, and `g > 25` is rejected.

There is no separate admission test. Every field challenge of the reduction
is a site of its
[grinding plan](labinius-root-reduction.md#grinding-plan), with its own loss
factor L, and each site pays its own g through an inline nonce. E is admitted
exactly when every site in E has a target within the cap.
`LoweredRootLayout::new::<F, E, _, _>` derives those sites, by the same
function that builds the plan, and returns `AkitaError::InvalidSetup` when
one of them cannot be priced. It derives them for the opening mode with both
claims, whose sites include those of the other two modes with loss factors
at least as large, so one layout serves every mode. The loss factors are

```text
D - 1, 2 * d - 2, n_A, 1, nu, 1, 17, mu, 1, 17, 1, 2
```

for alpha, xi, gamma, the prime row's scalar, for each combined instance its
equality point, its batching scalar and one round, with `nu` and `mu` the
logarithms of the two table lengths, and for the product instance of the
prime opening its batching scalar and one round. When `D - 1` is the largest
of them, E is admitted exactly when `(D - 1) * 2^103 <= P^e`.

At the sample geometry `D - 1 = 647` is the largest, so the cardinality must
be at least `647 * 2^103`, about `2^112.34`:

| Proof prime | Extension degree | Site targets, least to largest | Result |
| --- | --- | --- | --- |
| `2^128 - 275` | 1 | 1 to 10 | admitted |
| `2^64 - 59` | 2 | 1 to 10 | admitted |
| `2^64 - 59` | 1 | 65 to 74 | rejected |

## Derivation bias

The matrix is a stream of field elements reduced modulo q
([setup contract](labinius-setup-contract.md)). For X uniform on `[0, P)` and
`r = P mod q`, the total variation distance of `X mod q` from uniform is
`r * (q - r) / (P * q) <= q / (4P)`. Over the `n_A * m * D` matrix coefficients
a hybrid gives at most `n_A * m * D * q / (4P)`, an additive loss in the SIS
reduction.

`LabiniusRootShape::check_derivation_bias(P_stream)` requires

```text
n_A * m * D * q * 2^t <= 4 * P_stream,    t = LABINIUS_MIN_DERIVATION_BIAS_BITS = 64.
```

The floor is a provisional policy, not a derived requirement or a claim of
composed security. `AdmittedRootSetup::derive::<P>` applies it to the stream
field it reduces. `2^128 - 275` achieves 82 bits at the sample geometry. A
64-bit stream is rejected at every geometry: at rank 1 and width 1 it achieves
31 bits. The reduced view and an unreduced view of the same prefix are
correlated; nothing may treat them as independent matrices.

## Admission order and sample geometry

`LabiniusRootShape::derive` applies these gates in order and is independent of
any proof prime:

1. Represent both powers of two; check ring and scalar compatibility and
   exact, positive packing and column divisions.
2. Check the challenge family's exact fold budget.
3. Derive the fold response: cap, digit count, interval and `eta_A`.
4. Select the minimum dominating certified rank; reject missing coverage.
5. Derive `H`, `B_Q`, `B_K` and `B_KA` with checked arithmetic.

Proof-prime admission and the derivation-bias check are separate calls on the
derived shape; challenge-field admission is made where the field pair enters,
at the lowered layout. Caller-selected failures use
`AkitaError::InvalidSetup`.

For `(log_num_cells, log_fold_width, lambda_fold) = (22, 8, 128)`:

| Quantity | Value |
| --- | ---: |
| N / C / M / m / k | 4194304 / 256 / 16384 / 4096 / 4 |
| Response interval, digits, digit slots | `[-2184, 1911]`, 3, 4 |
| `eta_A` | 1506960 |
| Rank `n_A` | 2 |
| H | 5796802048 |
| `B_Q`, `B_K` | 11593604096, 8695203072 |
| `B_KA` | 17390406144 |
| `(e_Q, e_K, e_KA)` | (35, 35, 36) |
| Parity no-wrap total | 91696147968 |
| Commitment no-wrap total | 1737202407392206848 |
| Response table length, log | 16777216, 24 |
| Image digits, digit slots, offset `B_T` | 7, 8, 143165576 |
| Live image coefficients; image table length, log | 331776; 4194304, 22 |
| Prime left-opening table length, log | 262144, 18 |
| Achieved derivation-bias bits, `2^128 - 275` | 82 |

Both `2^64 - 59` and `2^128 - 275` admit this geometry with the same encoding.
At budget 128, fold-width log 9 rejects at the budget gate. With the budget
lowered, widening the fold eventually needs a fourth response digit and
rejects at the rank gate. The longest certified column is `(44, 0)`; `(45, 0)`
exceeds the width cap. `(44, 0)` passes condition 2 for `2^128 - 275` and
fails it for `2^64 - 59`.

## Table provenance and verification

The certified artifact is
`crates/akita-sis-estimator/data/labinius_commitment_prime_infinity_width.csv`.
The estimator names the modulus `q25-labinius` and the source
`phi243-bounded-w46-delta12`: the fold response above, at d = 648 and bound
1506960, with candidate ranks 1 through 4 under the quantum ADPS16 target of
128 bits, the existing local-minimum discovery plus proven-pruned
certification policy, and search cap 6400000000000.

A runtime row is admitted only in these cases:

- `Exact`: `max_width` is positive, with an accepted certificate there and a
  rejected successor.
- `AtLeast`: `max_width` equals the search cap, with an accepted certificate
  at the cap and no successor. The parameter cell records
  `LabiniusWidthCutoff::SearchCap` rather than presenting the cap as an exact
  cutoff.

The boundary search in
`crates/akita-sis-estimator/src/width_table/boundary.rs` states the
precondition that the security predicate is true on a prefix of widths and
false thereafter. Under it a cap certificate covers every smaller width; it
certifies a lower bound on the cutoff, not the cutoff. Missing or malformed
certificates, zero-width rows and `AtLeast` rows below their cap reject.
`validate_infinity_width_rows` checks the security certificates and
monotonicity.

| Rank | Bound | max_width | Kind | Cost at max_width | Successor cost |
| --- | ---: | ---: | --- | --- | ---: |
| 1 | 1506960 | 0 | Exact | absent | 51.145 |
| 2 | 1506960 | 6400000000000 | AtLeast | above-target:128.26000000000002 | absent |
| 3 | 1506960 | 6400000000000 | AtLeast | above-target:128.26000000000002 | absent |
| 4 | 1506960 | 6400000000000 | AtLeast | above-target:128.26000000000002 | absent |

Rank 1 is a diagnostic candidate only; the artifact contains ranks 2 to 4.
The search does not measure the rank-2 margin: it stops at the first block
size clearing the target, `0.265 * 484 = 128.26`, a figure shared by every
instance that clears it. The estimate is made under a generic q-ary lattice
model with inputs dimension, modulus, width, bound and norm. It does not see
`Phi_D` or the factorisation of the ring, and it is not a proof. A prime q
has no coefficient-modulus divisors, so the divisor-sublattice attacks a
composite modulus would have to price do not exist.

The generator example searches the cells, or emits the runtime Rust file from
the checked-in CSV without repeating the search:

```bash
cargo run -p akita-sis-estimator --release --no-default-features --features labinius-sis \
  --example labinius_infinity_width_table
cargo run -p akita-sis-estimator --no-default-features --features labinius-sis \
  --example labinius_infinity_width_table -- --from-csv
```

It takes `--dims` from 162, 324 and 648, `--max-rank`, `--search-cap`,
`--output` and `--rust-output`. Search diagnostics include zero-width
candidates; only admitted rows enter the CSV. The emitted
`crates/akita-params/src/sis/labinius/generated_commitment_prime_width_table.rs`
has sorted `LabiniusWidthCell` entries and the SHA3-256 digest of the exact
CSV bytes. Runtime consumers depend only on `akita-params`, never on the
estimator.

The estimator integration test `labinius_width_table` regenerates the four
candidates by search, compares them with the CSV and the runtime cells, and
recomputes the digest. The example's own test compares the generator output
for the CSV with the checked-in Rust file byte for byte and checks that the
table's bound is the sample fold's `eta_A`. Parameter unit tests cover the
sample numbers, domination, budget and missing-cell rejections, the
derivation-bias boundary against big-integer evaluation, both no-wrap
inequalities, the unit condition for the four primes above, and the parity
closed forms.

## What admission does not establish

- Condition 2 makes each accepting transcript's commitment row an identity
  between bounded integers. That is all the source-binding argument asks of
  P. The inequality `4 * Gamma * B_nu < P`, with
  `B_nu = 3 * m * D * (q - 1) * Delta + q * (2^e_KA - 1)`, belonged to the
  argument for an image table that was not range-checked. It is not needed
  and no code checks it; the reason is given with
  [the lowered relation](labinius-lowered-root.md#source-binding-and-projection).
  At the sample geometry its left side is about `2^70.1`, above `2^64 - 59`.
- The image digits are not required to be those of canonical residues. Any
  alphabet-valid table satisfying the relation is accepted; only its residues
  modulo q enter the commitment relation.
- The width table prices a generic q-ary instance with modulus q, degree D,
  the admitted rank and the bound `eta_A`, plus the additive derivation bias.
  It does not establish ring-ideal security.
- Knowledge extraction across the fold challenges, the SIS reduction and
  Fiat-Shamir composition are open, as recorded in
  [the root reduction](labinius-root-reduction.md#what-acceptance-proves-and-open-obligations).
  This is parameter admission, not protocol security admission.
