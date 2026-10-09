# LaBinius small-modulus root admission and foreign-modulus lift

Status: reduction-level protocol implemented; image and root PCS deferred
Book-chapter: book/src/foundations/security.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Scope and implementation status

The off-by-default `labinius` feature supports the closed profile
`LabiniusRootProfile::D648P128Q28BoundedW46Delta16` for seed-derived setup,
clear binary openings and the root reduction with the transparent oracle.
The parameter layer supplies its typed commitment modulus, certified SIS
(short integer solution) width table, shape admission and digit-base carry
envelopes. At `(22, 8, 128)` it admits rank 3. Opening-time objects stay over
the 128-bit coefficient prime P.

Commitment and quotient construction use the existing P-field transform
kernels on the reduced matrix, followed by exact centred lifting. A dedicated
small-prime kernel, planner integration and catalogs remain deferred.
`akita-labinius-pcs` rejects this profile with `InvalidSetup` at root prover
and verifier construction and at the image methods that accept an admitted
root. Its image constructors do not take a root profile. The existing tag-0
profile, table, identities, commitments and reduction proof bytes are unchanged.

## Notation


As in `specs/labinius-lowered-root.md`: `d = 162`, `Phi(Z) = Z^162 + Z^81 + 1`, `D = 648`, `Phi_D(Y) = Y^648 - Y^324 + 1`, `k = 4`, `m` ring elements per column, `C` columns, rank `n_A`, response interval `[L, U] = [-2^15, 2^15 - 1]`, accepted diameter `Delta = U - L = 65535`, challenge weight `w = 46`, operator bound `Gamma_inf = 2w = 92`, `eta_A = 4 * Gamma_inf * Delta = 24116880`.

New: `q0` is the commitment modulus, a prime with `1944 | q0 - 1` and `2 * eta_A < q0 < 2^28`. `[x]` denotes the integer in `[0, q0)` congruent to `x`. `S = Z[Y]/(Phi_D)`.

## Choice of modulus and rank


At the sample geometry, the selected profile uses **`q0 = 268433353` (the largest prime below `2^28` with `1944 | q0 - 1`), `n_A = 3`.**

- Under Akita's SIS policy (`EstimateConfig::akita_infinity_table()`, quantum ADPS16, target 128 bits; scalar dimensions `n_A * 648` by `m * 648`, infinity bound `eta_A`), the configured search was run for six primes, the largest with `1944 | q - 1` at each bit length from 26 to 31: `67091329`, `134217649`, `268433353`, `536868649`, `1073717857`, `2147478481`. At `m = 4096` each is below target at rank 2 (94.6, 99.6, 104.7, 110.0, 115.0, 116.3 bits respectively) and is classified above target at rank 3. The rank-3 classification was also obtained at widths `2^12`, `2^16`, `2^20`, `2^24` and `2^28` ring elements for the 26-, 28-, 29-, 30- and 31-bit primes. This is a statement about these six primes, not about every splitting prime in the range: `33563161` is a 26-bit prime with `1944 | q - 1` and `(q - 1)/2 < eta_A`, for which the problem is trivial at any rank and the estimator rejects the instance.
- "Above target" means the estimator's configured ADPS16/LGSA search found no attack below 128.26 bits; that figure is the search's stopping threshold (`0.265 * 484`), shared by every instance that clears the target, so it cannot compare margins between moduli. Comparing margins needs the fully optimised minimum with the early stop disabled. It is an estimate under a generic q-ary lattice model with inputs `n`, `q`, width, bound and norm. It does not see `Phi_D` or the factorisation of the ring, exactly as for the existing `P128` and `P64` rows. It is not a proof.
- A prime `q0` has no coefficient-modulus divisors, so the divisor sublattice attacks that a composite modulus would have to price (`x = d * y` for `d | q0`) do not exist. Composite moduli are out of scope.
- Why 28 bits and not 31: in signed 32-bit lanes a prime below `2^28` leaves enough headroom for a transform with no correction between butterflies; the 28-bit and the larger tested primes all clear the target at rank 3; smaller `q0` gives more margin in the extraction inequality and a smaller derivation bias.
The six-prime survey above is reported design evidence. The artifact in this
change independently searches only the selected q0; it does not certify the
other five primes. Admission selects the smallest certified rank dominating
both eta_A and m, with no interpolation or extrapolation. The rank-2 candidate
has zero width and is absent from the runtime table, even at requested width 1.

## Commitment and derivation bias

The following commitment and reduced matrix are implemented behavior.


- Matrix: `A in Z[Y]^{n_A x m}`, each coefficient an integer in `[0, q0)`: coefficient `t` of `A[i,j]` is `canonical(prefix((i*m + j)*D + t)) mod q0`, where `prefix` is the existing `derive_public_matrix_prefix::<F>` stream.
- Distribution: for `X` uniform on `[0, P)` and `r = P mod q0`, the total variation distance of `X mod q0` from uniform is `r * (q0 - r) / (P * q0) <= q0 / (4P)`. Over `N = n_A * m * D` coefficients a hybrid gives at most `N * q0 / (4P)`: `2^-79.1` at `m = 4096`, `2^-71.1` at `m = 2^20`, `2^-63.1` at `m = 2^28`. This term enters the SIS reduction as an additive loss and must be budgeted per admitted geometry. The reduced view and the unreduced `F_P` view of the same prefix are correlated; nothing may treat them as independent matrices.
- Image: `T_(col,i) = [sum_j A_ij * s_(col,j) mod Phi_D] mod q0`, coefficientwise, `n_A * C * D` integers in `[0, q0)`.
- The image table `Y` holds these integers embedded in `F_P` and is bound by
  the transparent oracle here. Image PCS integration is deferred. Y is not
  range-checked.

### Implemented bias admission policy

`LABINIUS_MIN_DERIVATION_BIAS_BITS = 64` is a provisional policy floor,
not a derived security requirement. With `N = n_A * m * D`, shape admission
requires `N * q0 * 2^64 <= 4 * P`. The achieved value is the largest integer t
satisfying `N * q0 * 2^t <= 4 * P`; it is 79 at the sample geometry.
This bounds an additive loss and does not establish total 128-bit composed
security. The large-width bias figures above are algebraic examples, not
claims of admission or feasible matrix materialization. In particular the
width 2^28 example misses the provisional 64-bit floor.

## Lifted relation


The A row becomes an identity over `Z[Y]`, with every product unreduced:

```text
sum_j A_ij(Y) * p_j(Y) - sum_col iota(Ch_col)(Y) * T_(col,i)(Y)
    = Phi_D(Y) * QA_i(Y) + q0 * KA_i(Y),        deg KA_i < D,  deg QA_i <= D - 2.
```

`QA_i` is the integer quotient by the monic `Phi_D`; `q0 * KA_i` is the remainder, which for an honest prover is divisible by `q0` coefficientwise because the identity reduced modulo `(q0, Phi_D)` is the commitment relation. The verifier checks the identity in `F_P` at the existing challenge `alpha`:

- `QA_i`: `D - 1` unconstrained elements of `F_P` per row, canonical encoding, sent before `alpha`, as today.
- `KA_i`: new. `D` integers per row, sent in the clear before `alpha`, range-checked against the signed interval of `e_KA` bits, as the parity quotient and carry are today.
- The public constant gains one term: `c_pub += sum_i gamma^i * q0 * KA_i(alpha)`.
- `K_W` and `K_Y` are unchanged as formulas; `Abar_j` is computed from the reduced coefficients.

The parity row, the alphabet check, the response table, both sumchecks and the oracle contract are unchanged.

### Honest bound on the carry

Every coefficient of the unreduced left side is bounded by `H_A = (q0 - 1) * (m * D * 2^15 + C * w)`. For the minus trinomial with `h = 324` the remainder of `r_0..r_1294` is

```text
rem_j   = r_j - r_(j+648) - r_(j+972)   for   0 <= j <= 322
rem_323 = r_323 - r_971
rem_j   = r_j + r_(j+324)               for 324 <= j <= 646
rem_647 = r_647 + r_971,
```

so reduction multiplies a coefficient bound by at most 3. Hence `|KA| <= B_KA = floor(3 * H_A / q0)`, and `e_KA` is the smallest multiple of the digit width with `2^e_KA >= 2 * B_KA + 1`.

At `(22, 8, 128)` with `q0 = 268433353`: `m * D * 2^15 + C * w = 86973099520`, `H_A = 23346480637983191040`, `B_KA = 260919297587` (`2^37.92`), `e_KA = 39` for one-bit digits and `40` for two- and four-bit digits. `B_KA` is essentially independent of `q0`.

Honest completeness: the prover can recover the remainder through `F_P` and centre it when `3 * H_A < P / 2`; this holds here with about 61 bits to spare (`log2(P / (6 * H_A)) = 61.08`) and is an admission inequality per geometry.

### What is range-checked

Only `KA`. The two statements below are algebra; the security conclusion drawn from them is conditional and is stated separately.

**`QA` needs no range check.** It needs its degree bound, canonical encoding and binding before `alpha`. Monic division commutes with reduction modulo `P`, and a polynomial of degree below `D` that is a multiple of `Phi_D` in `F_P[Y]` is zero; so an adaptive `QA_i` cannot change the remainder of the left side modulo `Phi_D`.

**Cancellation lemma (no assumption on `T`).** Fix one table `Y*` over `F_P` (arbitrary elements, no integrality, not assumed of the form `[A b]`). Suppose three accepting children share `Y*` and the matrix, use challenges from the admitted family, and differ only in the fold challenge of column `col`: a base child with `(c, z, KA)` and two others with `(c', z', KA')`, `(c'', z'', KA'')`, each satisfying the exact polynomial A-row identity in `F_P[Y]` with its own alphabet-valid response and range-valid carry. Put `a = z - z'`, `s = c - c'`, `kappa = KA - KA'` and likewise `a2, s2, kappa2` for the second pair. Reducing modulo `Phi_D` and subtracting,

```text
s * Y*_col = nu,   s2 * Y*_col = nu2     in R_P^{n_A},
nu := (A * a - q0 * kappa) mod Phi_D     computed over Z,
|nu|_inf <= B_nu = 3 * m * D * (q0 - 1) * Delta + q0 * (2^e_KA - 1).
```

Then `s2 * nu = s * nu2` in `R_P^{n_A}` by commutativity; no inverse of `s` is taken, so zero divisors are harmless. Each side is the image of an element of `S^{n_A}` of infinity norm at most `2 * Gamma_inf * B_nu` (a challenge difference has operator norm at most `2 * Gamma_inf`; `nu` is already reduced, so no further factor 3). Under

```text
4 * Gamma_inf * B_nu < P                                   (extraction no-wrap)
```

the equality holds in `S^{n_A}`, that is

```text
A * (s2 * a - s * a2) = q0 * (s2 * kappa - s * kappa2)    in S^{n_A}.
```

Only now reduce modulo `q0`: `A * (s2 * a - s * a2) = 0 mod (q0, Phi_D)` with `|s2 * a - s * a2|_inf <= eta_A < q0 / 2` (`eta_A < q0` already makes a nonzero integer vector nonzero modulo `q0`; SIS admission separately requires `eta_A < (q0 - 1)/2`, without which the instance is trivial). Either this is a nonzero module-SIS solution for `A` modulo `q0` of infinity norm at most `eta_A`, or `s2 * a = s * a2` in `S`, which is the hypothesis from which the existing "Source binding and projection" sketch continues (common source `a / s` over `Q[Z]/(Phi)` with odd denominators, the parity row, projection modulo two). That continuation never uses `A a = s Y mod P`.

At `(22, 8, 128)`: `B_nu = 2^68.56` and `4 * Gamma_inf * B_nu = 2^77.08` for two- and four-bit digits (`2^67.96`, `2^76.49` for one-bit digits). `P128` admits this; `P64` does not.

**What this does and does not establish.** The lemma is a statement about a fixed table and exact polynomial identities. It becomes a binding theorem for the composed protocol only together with a theorem that the nested Akita openings and the sumchecks supply, on a suitable tree of accepting transcripts, one common `Y*`, one alphabet-valid `W*` per child, bounded carries and the exact identities, with a stated extraction and Fiat-Shamir loss. `specs/labinius-root-reduction.md` already lists knowledge extraction, its loss, the SIS reduction and Fiat-Shamir composition as open for the existing shared-prime root; this design inherits that obligation and uses it in one more place, because the existing root does not need the cross-comparison to bound `T`. Until that theorem exists the binding claim is conditional. A range check on `T` would not discharge it.

## Implemented admission inequalities and encodings

All size and bound calculations use checked integer arithmetic. Failure,
unsupported widths and overflow return `AkitaError::InvalidSetup`.
`LabiniusRootShape::derive` retains challenge entropy, deterministic honest
response and source-comparison admission. For the small modulus it enforces:

1. `2 * eta_A < q0`, for centered uniqueness. The certified estimator cell
   separately preserves nontriviality `eta_A < (q0 - 1)/2`.
2. `6 * H_A < P`, for honest recovery through the opening field.
3. `N * q0 * 2^64 <= 4 * P`, the provisional bias policy above.

`derive_encoding` chooses the smallest positive multiple e_KA of digit width b
with `2^e_KA >= 2*B_KA+1` and completes admission using that actual range:

4. `2^(e_KA-1)-1 >= B_KA`, so both honest endpoints fit.
5. `4 * Gamma_inf * B_nu < P`, using the enforced carry diameter `2^e_KA-1`,
   rather than twice the honest bound.

The explicit carry-check method applies the same last two inequalities to an
enforced bit width. The A-carry length is `n_A * D`; shared-prime shapes
have no A-carry range and length zero. Existing parity range derivation and
no-wrap checks are unchanged. The signed carry map is
`value = sum_i digit_i * 2^(b*i) - 2^(e_KA-1)`, with digits in `[0,2^b-1]`
and accepted interval `[-2^(e_KA-1),2^(e_KA-1)-1]`.

At `(log_num_cells, log_fold_width, lambda_fold) = (22,8,128)`:

| Quantity | Value |
| --- | ---: |
| m / C / n_A | 4096 / 256 / 3 |
| eta_A | 24116880 |
| H_A | 23346480637983191040 |
| B_KA | 260919297587 |
| Live image length | 497664 |
| A-quotient length | 1941 |
| A-carry length | 1944 |
| Achieved bias bits | 79 |

| b | e_KA | B_nu | 4*Gamma_inf*B_nu |
| --- | ---: | ---: | ---: |
| 1 | 39 | 287649523880552564791 | 105855024788043343843088 |
| 2 | 40 | 435222320333752371255 | 160161813882820872621840 |
| 4 | 40 | 435222320333752371255 | 160161813882820872621840 |

## Soundness ledger delta

- The A-row polynomial identity test remains `(2D - 2)/|F|`.
- The implemented lift inequalities use e_KA rounded to the selected base.
- SIS pricing uses q0, degree D and the admitted rank, plus the additive
  derivation bias. The width certificate rule is stated below. The generic
  estimate does not establish ring-ideal security or the open extraction and
  Fiat-Shamir composition theorem.
- The auxiliary message binds QA and KA before alpha, using the exact grammar
  below. KA is checked as an integer before field embedding.
- The closed profile and setup identity bind q0, the reduced derivation,
  carry-envelope rule and layout before challenges. Existing setup transcript
  binding absorbs the profile identity; the layout also requires the setup
  commitment modulus to match the root shape.
- Parity no-wrap remains unchanged.

## Exact auxiliary-message grammar and matrix identity

For tag 1, the auxiliary message has no count fields or framing:

```text
QA[0][0..D-1], ..., QA[n_A-1][0..D-1]  : canonical F_P, 16 bytes each
KA[0][0..D],   ..., KA[n_A-1][0..D]    : signed offset, ceil(e_KA/8) bytes each
Q[0..161]                             : signed offset, ceil(e_Q/8) bytes each
K[0..162]                             : signed offset, ceil(e_K/8) bytes each
```

Ranges above are half-open. KA is row-major, with exactly `n_A*D` integers.
For any signed e-bit value x, the bytes encode `x + 2^(e-1)` little-endian;
the value must lie in `[-2^(e-1), 2^(e-1)-1]`. An offset at least `2^e`,
including any nonzero unused high bit, rejects. QA is unconstrained beyond
canonical field encoding and its exact degree extent. All four sections
precede alpha. Tag 0 omits KA entirely, including any empty-section framing.

Let `LP(x) = u64_le(byte_length(x)) || x`. The reduced matrix-view digest
uses the existing backend-specific `matrix_digest` construction: a fresh
channel frames this header as its session, the canonical field coefficient
list's `field_digest` as its instance, and squeezes 32 bytes. Its header is:

```text
LP("akita/labinius/clear-matrix-view/reduced/v1")
|| commitment_modulus_tag:u8 || q0:u32_le
|| n_A:u64_le || m:u64_le || D:u64_le || middle_coefficient:i8
```

The coefficient list contains the reduced matrix entries in tight row-major
order, each as its canonical F_P representation. This explicitly binds the
reduction view and q0 even when the same coefficients appear in another view.
Tag 0 retains `akita/labinius/clear-matrix-view/v1` and its original header.
The nested clear identity binds this digest; the admitted identity also binds
the root profile and derivation seed. No second matrix identity is introduced.

`BinaryClearSetup::new` constructs only a shared-prime explicit setup.
Caller-supplied small-modulus setups are unrepresentable through that public
constructor; pairing one with a tag-1 shape rejects at layout admission.
Only seed-derived root admission installs the small modulus. Its internal
boundary additionally rejects any canonical matrix coefficient at least q0.

## Commitment and clear-endpoint lifting

For binary packed source coefficients in `[-1,1]`, the true reduced matrix
product coefficient has magnitude at most `3*m*D*(q0-1)`. Admission rejects
unless twice this bound is strictly below P. Both commitment paths compute
in R_P, centre each coefficient exactly, then take its integer remainder
modulo q0 in `[0,q0)`.

The root quotient kernel obtains the original-ring remainder and the
conjugate-ring residual through the existing transforms. It centres the
remainder under `6*H_A < P`, checks integer divisibility by q0, and range-checks
KA using the selected digit base. Subtracting the remainder from the conjugate
residual leaves the existing field QA recovery unchanged. Tag 0 retains its
nonzero-remainder rejection.

The clear endpoint centres each coefficient of `A*z - sum_col ch_col*T_col`
in R_P and checks integer divisibility by q0 and the admitted carry interval.
It sends no carry message. Because the clear API has no digit-base selector,
it uses the smallest admitted envelope, that for one-bit digits. Root reduction
uses the selected base's envelope. Field division by q0 is never an integer
check. T is not range-checked by either verifier.

## Consequences outside the relation (deferred)

Setup offloading cannot authenticate the reduced matrix using a commitment to
the unreduced prefix: reduction `canonical(S_t) mod q0` is not F_P-linear.
Future verification needs the materialized reduced view or a separately
authenticated derived source. Common-field opening batching still uses P.

The materializer temporarily allows 128 MiB each for the flat F_P prefix and
stored matrix for tag 1; tag 0 retains both 64 MiB caps. At rank 3 and m=4096
the matrix is 127401984 bytes as field elements or 31850496 bytes as u32.
A later storage change will derive the prefix in pages and store u32 entries,
then return both caps to 64 MiB. These are two allocation caps, not a global
verifier-memory cap. The padded setup-weight scratch vector separately has
2^23 field elements, or 128 MiB. No storage representation change is implemented.

`LoweredPublic::new` and `witness_weight_mle` scan n_A*m*D coefficients,
three times the current count at the sample geometry. Although reduced
coefficients fit in 28 bits, contractions still occur in F_P; no verifier
multiplication cost or performance claim follows from the storage size.

The image layout is unchanged: every ring element is padded to 1024, giving
`image_table_len = 3*256*1024 = 786432` and `image_table_log_len = 20`.
The final power-of-two domain has 1048576 entries, two variables more than
rank one's 2^18. Future image PCS support will still use full-field openings for small honest
entries.
Future support needs new scalar and grouped catalogs and wire-size accounting
over P. Composed PCS proof size and verifier time are not established. Reduction wire
size includes the clear KA section; the exact formula is in the
[root-reduction specification](labinius-root-reduction.md).

## Profile identity bytes

Both profiles use the domain `akita/labinius/root-profile/v1` followed by a
zero byte. The exact tag-0 layout stays:

```text
domain || 0 || profile_tag:u8 || scalar_ring_tag:u8
  || P:u128_le || D:u32_le
  || (1 || challenge_byte)* || 0
  || L:i128_le || U:i128_le || original_table_digest:[u8;32]
```

Scalar-ring tag 0 means Cyclotomic243; tag 1 means Cyclotomic729.
Challenge framing uses one marker byte before every challenge byte and a zero
terminator, so its variable length cannot alias later fields.

Tag 1 has the same common fields through U and then binds the typed small
commitment modulus and its separate table digest:

```text
common_fields_through_U || commitment_modulus_tag:u8 || q0:u32_le
  || small_modulus_table_digest:[u8;32]
```

Commitment-modulus tags are 0 for CoefficientPrime and 1 for Q28Offset2103;
only tag 1 adds this suffix. q0 is 268433353. The closed profile fixes the
reduced derivation and carry rule above. Unknown root tags reject.
The tag-0 SHA3-256 identity fixture captured before adding the new profile is
`cf1ff75929964d58bd7c04cc2b53fa0011a60e9420c2ddca7c673444e04b1570`.
The original table and digest are frozen.

## Small-modulus table rule and provenance

The independent artifact is
`crates/akita-sis-estimator/data/labinius_small_modulus_infinity_width.csv`.
It uses `q28-labinius`, source `phi243-bounded-w46-delta16`, d=648, bound
24116880, candidate ranks 1 through 4, quantum ADPS16 target 128 bits and
the existing local-minimum discovery plus proven-pruned certification policy.
The search cap is 6400000000000. A runtime row is admitted only in these cases:

- `Exact`: max_width is positive, with an accepted certificate there and a
  rejected successor.
- `AtLeast`: max_width equals search_cap, with an accepted certificate at the
  cap and no successor. The parameter cell records `SearchCap`, rather than
  presenting the cap as an exact cutoff.

A cap certificate accepts the cap. The boundary search in
`crates/akita-sis-estimator/src/width_table/boundary.rs` states the precondition
that the security predicate is true on a prefix and false thereafter. Under
that precondition, the certificate covers every smaller width; it certifies a
lower bound on the cutoff, not the cutoff itself. Missing or malformed
certificates, zero-width rows and AtLeast rows below their cap reject.
`validate_infinity_width_rows` checks the security certificates and
monotonicity. This small-modulus rule follows the production SIS policy audit's
reading of AtLeast rows. The original LaBinius artifact still accepts only
exact positive cutoffs with rejected successors, without any byte change.

The unchanged search produces these candidates:

| Rank | Bound | max_width | Kind | Cost at max_width | Successor cost |
| --- | ---: | ---: | --- | --- | ---: |
| 1 | 24116880 | 0 | Exact | absent | 37.365 |
| 2 | 24116880 | 0 | Exact | absent | 120.045 |
| 3 | 24116880 | 6400000000000 | AtLeast | above-target:128.26000000000002 | absent |
| 4 | 24116880 | 6400000000000 | AtLeast | above-target:128.26000000000002 | absent |

Ranks 1 and 2 are diagnostic candidates only; the artifact contains ranks 3
and 4. Even width 1 resolves to rank 3. Widths above the cap have no certified
coverage. The rank-3 margin above target is not measured: the search stops at
the first block size whose reduction lower bound and best visited attack
clear the target, `0.265 * 484 = 128.26`.

Search and regeneration use the existing generator:

```bash
cargo run -p akita-sis-estimator --release --no-default-features --features labinius-sis \
  --example labinius_infinity_width_table -- --profiles q28-labinius
cargo run -p akita-sis-estimator --no-default-features --features labinius-sis \
  --example labinius_infinity_width_table -- --profiles q28-labinius --from-csv
```

Search diagnostics include zero-width candidates and admitted candidates;
only admitted rows enter the CSV. The runtime output is
`crates/akita-params/src/sis/labinius/generated_small_modulus_width_table.rs`,
with sorted `LabiniusSmallModulusWidthCell` entries recording cutoff kind,
and SHA3-256 of the exact CSV bytes in
`LABINIUS_SMALL_MODULUS_WIDTH_TABLE_DIGEST`. The digest covers the recorded
Exact/AtLeast kind as well as costs and provenance. Both generated tables share
the generator without changing the original output.

The two artifact admission paths share row parsing and certificate validation,
while retaining their different cutoff rules. Tests compare runtime and CSV
cells in both directions, recompute the digest, reproduce generated Rust from
CSV and regenerate the small artifact rows by search. They also check that the
old path rejects AtLeast while the new path accepts it, and both reject zero
width and inconsistent cap rows. Parameter tests pin sample bounds, rank,
identity and envelopes, plus fallible rejection of each lift inequality and
overflow. Neither the original CSV nor its generated table changes.

## Deferred (not implemented)

- Quotient batching with a new transcript order and parity-aware ledger.
- Carry digits in response-table tails with a new address and weight contract.
- Limb folding with centered lifting and signed packing.
- A remainder relation without QA.
- Column-major image packing and its terminal weight evaluator. This retains
  the column tensor factor but changes rank/coefficient coordinates; no exact
  multiplication count is established.
