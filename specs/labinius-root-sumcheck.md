# LaBinius root sumchecks

Status: implemented arithmetic kernels; the enclosing composition is specified in
[`labinius-root-reduction.md`](labinius-root-reduction.md).

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY in this document
have the BCP 14 meanings defined by RFC 2119 and RFC 8174 when capitalized.

## Statement and conventions

Let `F` be the base field, a prime field, and `E` the challenge field, an
extension of `F` (`E: ExtField<F>`; `E = F` is allowed). Committed tables hold
stored base-16 digits (`LABINIUS_BALANCED_LOG_BASIS = 4`), which are small
integers of `F`; there is no other digit base. Equality points, batching
scalars, public weights, claims, round messages and round challenges are
elements of `E`. The characteristic MUST be greater than 17: interpolation
uses every node `0,...,17`, which MUST be distinct. Every proof prime
admitted by
[root admission](labinius-root-admission.md#proof-prime-admission-and-clear-integer-ranges)
satisfies this. Let `W` and `Y` be the committed response and image digit
tables on `nu` and `mu` Boolean variables. The statement requires every
`W(x)` and every `Y(x)` to belong to `[0, 15]` and

```text
sum_x W(x) K_W(x) + sum_x Y(x) K_Y(x) = c_pub.
```

`K_W`, `K_Y`, and `c_pub` are public. The prover supplies
`y_Y = sum_x Y(x) K_Y(x)`; the verifier defines `s = c_pub - y_Y`.
Tables MUST have lengths `2^nu` and `2^mu`, respectively. Multilinear
extensions use little-endian index bits: round `i` binds bit `i`, as in
`akita_algebra::poly::multilinear_eval`.

A statement with a prime claim has a third committed table, the prime left
opening `uP` on `pi` Boolean variables. It holds arbitrary elements of `E`
and has no alphabet. With public weights `K_v` and `K_P`, a public value `v`
and a public scalar `eta`, the statement becomes

```text
sum_x W(x) K_W(x) + sum_x Y(x) K_Y(x) - eta sum_x uP(x) K_P(x) = c_pub;
sum_x uP(x) K_v(x) = v.
```

The prover also supplies `y_P = sum_x uP(x) K_P(x)`, and the verifier
defines `s = c_pub - y_Y + eta*y_P`. The table and weights are those of
[the prime row](labinius-lowered-root.md#prime-row).

Write `eq(a,z) = product_i ((1-a_i)(1-z_i) + a_i z_i)` and
`A(T) = product_{a=0}^{15} (T-a)`. The alphabet polynomial has degree 16 and
vanishes exactly on the digit alphabet, over `F` and over `E`.

## Combined alphabet and relation instance

For a digit table `T` on `n` variables, public weights `K`, an equality point
`tau`, a batching scalar `beta` and a linear claim `c`, the instance proves
the sum of

```text
G(x) = eq(tau,x) A(T~(x)) + beta T~(x) K~(x)
```

with input claim `beta*c`, `n` rounds, and degree bound 17 per round. The
honest constructor accepts digits as bytes and MUST reject a byte outside the
alphabet, mismatched lengths, or a mismatched equality point.

After challenges `rho`, the verifier MUST check

```text
final_claim = eq(tau,rho) A(t_eval) + beta t_eval k_eval.
```

Here `t_eval = T~(rho)` MUST come from an authenticated opening in the later
root protocol, and `k_eval = K~(rho)` is evaluated from the public weights.
`combined_terminal` evaluates this formula, including the equality factor in
`O(n)` time; it does not authenticate the supplied evaluations.

The root reduction runs this instance twice, each with its own fresh point
and scalar:

| Invocation | Table | Weights | Point, scalar | Linear claim | Opened value |
| --- | --- | --- | --- | --- | --- |
| 0 | `W`, `nu` variables | `K_W` | `tau`, `beta` | `s = c_pub - y_Y` | `w_eval` |
| 1 | `Y`, `mu` variables | `K_Y` | `tau'`, `beta'` | `y_Y` | `y_eval` |

## Product instance

The product instance proves `sum_x T~(x) K~(x) = c` for two field tables on
`n` variables, with degree bound 2. After challenges `rho`, the verifier MUST
check

```text
final_claim = t_eval k_eval.
```

`product_terminal` takes those two evaluations as explicit inputs.

The root reduction runs this instance once, exactly when the statement has a
prime claim. It proves both linear claims on `uP` under one fresh scalar
`theta`:

| Invocation | Table | Weights | Scalar | Claim | Opened value |
| --- | --- | --- | --- | --- | --- |
| 2 | `uP`, `pi` variables | `K_v + theta*K_P` | `theta` | `v + theta*y_P` | `p_eval` |

The terminal is `product_terminal(p_eval, kv + theta*kp)`, with `kv` and `kp`
the two weights' multilinear extensions at the challenge point. There is no
equality point and no alphabet term: the table is not range-checked.

## Binding order and soundness

For each combined instance the root protocol MUST fix the table before
drawing its uniform equality point. It MUST fix the table, the point, the
weights and the linear claim before drawing the batching scalar, which MUST
be fresh, uniform conditional on the preceding transcript.
It MUST bind all tables, public weights, points, and input claims before
the round challenges. Each round message MUST precede its challenge.
These kernels take the inputs explicitly; their channel adapters supply
sumcheck diagnostic contexts and field challenges. They do not implement the
enclosing statement binding order. That order is specified in
[`labinius-root-reduction.md`](labinius-root-reduction.md).

A round challenge is drawn by the channel adapter. An adapter that replays
the reduction's
[grinding plan](labinius-root-reduction.md#grinding-plan) places the round's
proof-of-work nonce between the round message and its challenge: the round
is one visit of the plan site `Round` of that invocation, with loss factor
the round's degree bound. The round drivers and kernels do not see the
nonce. A standalone instance schedules its own round-site plan with
`RootGrindingPlan::sumcheck_rounds::<F, E>`; an adapter without a scheduled
plan rejects every round challenge.

Before the generic driver runs, both sides MUST absorb the same canonical
instance header as public bytes. `bind_root_sumcheck_instance` is the single
encoding authority. The header concatenates these fields, without lengths:

| Field | Encoding |
|---|---|
| Domain | ASCII `akita/labinius/root-sumcheck-instance/v1` |
| Kind | One byte: 0 combined, 1 product |
| Invocation | Four bytes, unsigned little-endian |
| Number of variables | Eight bytes, unsigned little-endian |

The verifier replay entry points and prover `prove_combined_rounds` and
`prove_product_rounds` entry points own this binding. It applies even when
there are no rounds. The header binds the instance kind, invocation and
dimension to all subsequent challenges. Distinct invocation
numbers alone separate nothing: diagnostic `ProtocolSiteId` records and
message contexts are not absorbed. The enclosing session and this explicit
header supply domain separation.

For the product instance the root protocol MUST fix `uP`, both weights,
`v` and `y_P` before drawing `theta`, which MUST be fresh in the same sense,
and MUST bind all of them and `theta` before the round challenges.

The statements below are written for the response instance; the image
instance is the same with `(Y, K_Y, tau', beta', y_Y, mu)`.

If some Boolean `x*` has an invalid digit, then
`Z(tau) = sum_x eq(tau,x) A(W(x))` is a nonzero multilinear polynomial in
`tau`: its Boolean evaluation at `x*` equals `A(W(x*)) != 0`.
For fixed `W`, it vanishes at uniform `tau` in `E` with probability at most
`nu/|E|`.

Let `L = sum_x W(x) K_W(x)`. The combined claim requires
`Z(tau) + beta*(L-s) = 0`. If `Z(tau) != 0` or `L != s`, at most one
`beta` satisfies this equation, so the additional error is at most `1/|E|`.
When `L=s` and `Z(tau) != 0`, no `beta` satisfies it. The claim `s` MUST
already be fixed: for a nonzero realized `beta`, the adaptive claim
`s = L + Z(tau)/beta` satisfies the equation and is not covered by this bound.

Sumcheck contributes at most `17*nu/|E|` for the response instance and
`17*mu/|E|` for the image instance. These statements require the terminal
evaluations to correspond to the fixed tables; the kernels alone do not prove
that correspondence. A union bound for these arithmetic reductions is
`(nu + 1 + 17*nu + mu + 1 + 17*mu)/|E|`, apart from opening errors. The
response instance establishes `L = c_pub - y_Y` and the image instance
`sum_x Y(x) K_Y(x) = y_Y`, for the one `y_Y` fixed before either point, so
together they give the statement's linear relation.

For the product instance, let `L_v = sum_x uP(x) K_v(x)` and
`L_P = sum_x uP(x) K_P(x)`. Its claim requires
`(L_v - v) + theta*(L_P - y_P) = 0`; unless both differences vanish, at most
one `theta` satisfies it, an error of at most `1/|E|`. Sumcheck contributes
at most `2*pi/|E|`. With `L_P = y_P` the response instance's claim
`L = c_pub - y_Y + eta*y_P` gives the first line of the statement above, and
`L_v = v` is the second.

## Weight construction

The remainder relation, dense definition, shift recurrence and terminal
closed forms are owned by [the lowered relation](labinius-lowered-root.md).

`coefficient_weights` builds the compact response-weight factor. For each
response ring column j it forms `B_j=sum_i gamma^i*A_ij` from the residue
matrix with `n_A*D` multiply-adds, runs the O(D) shift recurrence at alpha,
and adds the unchanged parity coefficient weight, scaled by gamma^n_A. There
is no transform, no transform cache and no field copy of the matrix.

Column tasks use the Rayon pool behind `parallel`, with fallible
reservations. They write directly into the compact `m*padded_coefficients`
table passed to `CombinedRootKernel::new`; coefficient tails are zero. There
is no intermediate dense `m*D` matrix-weight copy and no full digit-expanded
weight table. The phase is traced by `root_witness_weights`. At the sample
geometry `(22,8,128)` the compact table has 4,194,304 entries of `E`, or
64 MiB for 16-byte elements.

`image_weights` builds the compact image-weight factor the same way: the
recurrence runs once per fold challenge on `iota(Ch_col)`, and `-gamma^i`
times the result is written into entry `(col,i)`. The table has one entry per
padded image coefficient and keeps coefficient tails and final entry padding
with zero weights; no separate rank tensor block is introduced. The image
digit factor `pow_Y` is passed to the kernel separately. The phase is traced
by `root_image_weights`. At the sample geometry the table has 524,288
entries, or 8 MiB.

The `root_a_carry` phase computes the commitment-row remainder over the
integers, checks divisibility by the commitment prime q and enforces the
admitted carry range. It precedes alpha and does not build a quotient. The
carry is owned by [the lowered relation](labinius-lowered-root.md#polynomial-rows)
and its range by
[root admission](labinius-root-admission.md#commitment-row-carry).

With a prime claim, `coefficient_weights` also adds
`eta * eq(r_ring,j) * alpha^t` to coefficient t of column j. `prime_row_weights`
builds `K_P` by one run of the recurrence per fold challenge on
`iota(Ch_col)`, written into the first D coefficients of column `col` with no
sign and no power of gamma; the phase is traced by `root_prime_row_weights`.
`PrimeClaim::value_weights` builds `K_v` as the outer product of
`eq(r_col,.)` and `omega`. Both tables have the length of `uP`, coefficient
low and column high, with zero coefficient tails: 262,144 entries or 4 MiB
at the sample geometry. The prover forms `y_P` by one pass over `uP` and
`K_P`, and after `theta` overwrites `K_P` with `K_v + theta*K_P`.

## Messages and prover storage

The generic driver omits the linear coefficient and reconstructs it from the
running claim. A combined round sends 17 coefficients of `E`, for a total
of `17*nu` in the response instance and `17*mu` in the image instance. A
product round sends two, `2*pi` in the prime instance. An element of `E`
occupies `f = E::DEGREE * F::NUM_BYTES` bytes, its ordered canonical base
coordinates, so the round bodies occupy `17*nu*f`, `17*mu*f` and `2*pi*f`
bytes.
Public claims and the instance header are absorbed rather than transmitted.
Message contexts and site identifiers are diagnostic only; they absorb no bytes.
There are no proof-supplied lengths. Replay returns the challenge point and
final claim; it MUST be followed by the terminal check and, at the enclosing
proof boundary, the channel's EOF check.

The combined prover evaluates both terms in one pass per round and interpolates
at nodes `0,...,17`. It packs the input digits two to a byte and delays
lifting them to field elements for `r` rounds.
Equality uses two suffix tensor factors. Weights use the tensor of a digit
factor (length `2^a`, occupying the lowest variables: `digit_powers` for the
response, `image_digit_powers` for the image) and a compact coefficient table
`cw` (length `N/2^a`, coefficient-low, entry-high). Padded coefficient
positions in `cw` are zero. The prover never expands these weight factors
into a length-`N` field table.

The first `a` challenges fold only the digit factor. Its final scalar is applied
to `cw` in place, and subsequent rounds fold that table. Class rounds 0 and 1
use addition-only class histograms when their tables fit within the live pairs,
or evaluate packed pairs directly otherwise. Weight sums use compact
coefficients when they fit within the live pairs, or the two weight endpoints
otherwise.

Moment round 2 reads each pair as two 16-bit classes. With `U` the smaller
endpoint class bound, it is admitted when `8*U <= N/8`, giving `r=3`;
otherwise `r=min(nu,2)`. It accumulates seventeen moments per class of that
endpoint, using powers of the other endpoint's value. Those powers are
tabulated when its bound `Q <= 2^13` and `16*Q <= N/4`, and computed per
pair otherwise. Both root shapes qualify: `U=4096`, with `Q=4096` for the
response and `Q=65536` for the image. After the packed rounds, the digits
lift into `N/2^r` field entries for `nu>0`. Later rounds evaluate each lifted
pair on its line through the range image of the alphabet polynomial, skipping
pairs of zeros.

The peak reserved payload bound, including worker tables, is specified on
`CombinedRootKernel`. With 16-byte elements and `U=4096`, the response at
`nu=24`, `a=2`, `Q=4096` is bounded by 119,243,536 bytes for one worker and
223,450,816 bytes for sixteen; the image at `mu=22`, `a=3`, `Q=65536` by
29,082,416 and 115,726,048 bytes, respectively. These bounds exclude vector
metadata, allocator rounding and interpolation scratch. All reservations
happen in the constructor; rounds reuse them. A product prover owns `uP`
and the batched weights, 4 MiB each at `pi=18`, and constant-size round
workspace. The root prover retains `uP` and `K_P` across both combined
instances.

## Scope and validation

The lowered relation and weight construction are specified separately. The
complete root transcript binding, oracle seam, batching, and wire encodings
are specified in [`labinius-root-reduction.md`](labinius-root-reduction.md).
Polynomial commitment implementations, planner and schedule changes remain
outside this specification.
Zero-variable instances are permitted; their input claim is already terminal.

`crates/akita-labinius-prover/tests/root_sumcheck.rs` tests the
reference instances: the combined rounds against a brute-force Boolean
expansion, the product rounds through their channel entry points, and both
terminal formulas; and a witness with
a digit outside the alphabet whose linear relation is exactly satisfied,
which passes round replay and fails the terminal check.
`unscheduled_adapters_reject_the_first_round_challenge` checks both channel
roles without a plan. `tests/combined_kernel.rs` compares the packed kernel
with the dense reference in
`random_rounds_transcripts_and_terminal_checks`,
`digit_factor_shapes_and_padded_compact_entries_match_dense_reference`,
`padded_tables_reach_class_tables_and_the_moment_round`,
`extremal_digits_zero_weights_and_boolean_equality_points`, and
`false_linear_claim_preserves_polynomials_and_is_rejected`.
The product instance inside the reduction, its batched weights
and its structured terminals are exercised by the prime modes of
`tests/root_reduction.rs`. Existing clear-opening behavior is unchanged.
