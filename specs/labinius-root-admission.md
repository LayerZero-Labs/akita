# LaBinius binary-root admission

Status: active
Book-chapter: book/src/foundations/security.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

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

The admitted image has `n_A * C * D` coefficients. Products in the A relation
have degree at most `2D-2`; division by the monic degree-D cyclotomic polynomial
therefore gives quotient degree at most `D-2`, or `n_A * (D-1)` coefficients
(issue section 5).

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
`r_j - q_j - q_(j-81)`, ignoring quotient indices outside `[0,160]`. Bounding
each quotient term by 2H gives `|remainder_j| <= 5H`. A valid F162 identity
holds exactly when all these remainder coefficients are even: integer division
commutes with reduction modulo two because the divisor is monic. Then K is
half the remainder and is integral, with

```text
B_Q = 2H
B_K = floor(5H/2).
```

These are conservative honest construction bounds. They are not permission to
use tighter accepted alphabets in a proof. `check_parity_no_wrap(B_Q', B_K')`
requires each caller-supplied magnitude to cover its honest bound and enforces
the issue's full accepted-envelope inequality
`H + 3*B_Q' + 2*B_K' < P`. The coefficient sum of Phi has three terms, so this
ensures that a checked residual equality modulo P cannot conceal a nonzero
integer coefficient. Rounded digit alphabets must pass their actual enforced
magnitudes. An overflow rejects, as does equality at P.

## Admission order and boundary fixture

The canonical derivation applies these gates in order:

1. Represent both powers of two; check ring/scalar compatibility and exact,
   positive packing/column divisions.
2. Check the challenge family's exact fold budget.
3. Check the response interval and deterministic honest response envelope.
4. Use the source-comparison ledger, including its `eta_A < P` check.
5. Select the minimum dominating certified SIS rank; reject missing coverage.
6. Derive image/quotient lengths and honest parity bounds with checked arithmetic.

Caller-selected failures use `AkitaError::InvalidSetup`, matching neighbouring
SIS admission. Actual parity range envelopes are checked separately once the
caller chooses an encoding. The sole profile's eta is fixed below P; tests
perturb internal fields to exercise the ledger boundary without exposing a
runtime profile plugin.

For `(log_num_cells, log_fold_width, lambda_fold) = (22,8,128)` the shape has
`N=4194304`, `C=256`, `M=16384`, `m=4096`, `k=4`, `n_A=1`,
`eta_A=24116880`, image length 165888, and A-quotient length 647.
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
proof encoding, successor-witness layout, planner integration, or root protocol.
It does not claim complete PCS security admission. Those belong to the root
protocol and its composition with the ordinary tail. No unpublished paper or
unlicensed reference implementation is needed for these formulas or tests.
