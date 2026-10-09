# LaBinius root sumchecks

Status: implemented arithmetic kernels; the enclosing composition is specified in
[`labinius-root-reduction.md`](labinius-root-reduction.md).

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY in this document
have the BCP 14 meanings defined by RFC 2119 and RFC 8174 when capitalized.

## Statement and conventions

Let `F` be the coefficient prime field. For the selected digit base `b`, its
characteristic MUST be greater than `2^b + 1`: interpolation uses every node
`0,...,2^b+1`, which MUST be distinct. Both supported coefficient primes
satisfy this condition for every base. Both instances use `E = F`, without an
extension field. Let `b` be 1, 2, or 4,
and let `W` and `Y` be committed tables on `nu` and `mu` Boolean variables.
The statement requires every `W(x)` to belong to `[0, 2^b)` and

```text
sum_x W(x) K_W(x) + sum_x Y(x) K_Y(x) = c_pub.
```

`K_W`, `K_Y`, and `c_pub` are public. The prover supplies
`y_Y = sum_x Y(x) K_Y(x)`; the verifier defines `s = c_pub - y_Y`.
Tables MUST have lengths `2^nu` and `2^mu`, respectively. Multilinear
extensions use little-endian index bits: round `i` binds bit `i`, as in
`akita_algebra::poly::multilinear_eval`.

Write `eq(a,z) = product_i ((1-a_i)(1-z_i) + a_i z_i)` and
`P_b(T) = product_{a=0}^{2^b-1} (T-a)`. The alphabet polynomial has degree
`2^b` and vanishes exactly on the digit alphabet over `F`.

## Combined alphabet and relation instance

The instance proves the sum of

```text
G(x) = eq(tau,x) P_b(W~(x)) + beta W~(x) K_W~(x)
```

with input claim `beta*s`, `nu` rounds, and degree bound `2^b+1` per round.
Thus the degree bounds for `b = 1, 2, 4` are 3, 5, and 17. The honest
constructor accepts digits as bytes and MUST reject a byte outside the
selected alphabet, mismatched lengths, or a mismatched equality point.

After challenges `rho`, the verifier MUST check

```text
final_claim = eq(tau,rho) P_b(w_eval) + beta w_eval kw_eval.
```

Here `w_eval = W~(rho)` MUST come from an authenticated opening in the later
root protocol, and `kw_eval = K_W~(rho)` is evaluated from the public weights.
`combined_terminal` evaluates this formula, including the equality factor in
`O(nu)` time; it does not authenticate the supplied evaluations.

## Product instance

The second instance proves `sum_x Y~(x) K_Y~(x) = y_Y`, with `mu` rounds and
degree bound 2. After challenges `rho'`, the verifier MUST check

```text
final_claim = y_eval ky_eval.
```

`y_eval = Y~(rho')` MUST come from an authenticated opening in the later root
protocol, and `ky_eval = K_Y~(rho')` is evaluated from the public weights.
`product_terminal` takes those two evaluations as explicit inputs.

## Binding order and soundness

The root protocol MUST fix `W` and `Y` before drawing uniform `tau`.
It MUST fix `W`, `tau`, `K_W`, and `s` before drawing `beta`, which MUST be
fresh, uniform conditional on the preceding transcript.
It MUST bind all tables, public weights, points, and input claims before
the round challenges. Each round message MUST precede its challenge.
These kernels take the inputs explicitly; their channel adapters supply
sumcheck diagnostic contexts and field challenges. They do not implement the
enclosing statement binding order. That order is specified in
[`labinius-root-reduction.md`](labinius-root-reduction.md).

Before the generic driver runs, both sides MUST absorb the same canonical
instance header as public bytes. `bind_root_sumcheck_instance` is the single
encoding authority. The header concatenates these fields, without lengths:

| Field | Encoding |
|---|---|
| Domain | ASCII `akita/labinius/root-sumcheck-instance/v1` |
| Kind | One byte: 0 combined, 1 product |
| Invocation | Four bytes, unsigned little-endian |
| Number of variables | Eight bytes, unsigned little-endian |
| Digit base | One byte: 1, 2, or 4 combined; 0 product |

The verifier replay entry points and prover `prove_combined_rounds` and
`prove_product_rounds` entry points own this binding. It applies even when
there are no rounds. The header binds the instance kind, invocation, dimension
and combined alphabet to all subsequent challenges. Distinct invocation
numbers alone separate nothing: diagnostic `ProtocolSiteId` records and
message contexts are not absorbed. The enclosing session and this explicit
header supply domain separation.

If some Boolean `x*` has an invalid digit, then
`Z(tau) = sum_x eq(tau,x) P_b(W(x))` is a nonzero multilinear polynomial in
`tau`: its Boolean evaluation at `x*` equals `P_b(W(x*)) != 0`.
For fixed `W`, it vanishes at uniform `tau` with probability at most
`nu/|F|`.

Let `L = sum_x W(x) K_W(x)`. The combined claim requires
`Z(tau) + beta*(L-s) = 0`. If `Z(tau) != 0` or `L != s`, at most one
`beta` satisfies this equation, so the additional error is at most `1/|F|`.
When `L=s` and `Z(tau) != 0`, no `beta` satisfies it. The claim `s` MUST
already be fixed: for a nonzero realized `beta`, the adaptive claim
`s = L + Z(tau)/beta` satisfies the equation and is not covered by this bound.

Sumcheck contributes at most `nu*(2^b+1)/|F|` for the combined instance and
`2*mu/|F|` for the product instance. These statements require the terminal
evaluations to correspond to the fixed tables; the kernels alone do not prove
that correspondence. A union bound for these arithmetic reductions is
`(nu + 1 + nu*(2^b+1) + 2*mu)/|F|`, apart from opening errors.

## Adjoint-weight construction

The remainder relation, trace maps, dense definition and terminal closed forms
are owned by [the lowered relation](labinius-lowered-root.md). The sumcheck
instances, degrees and digit factorization above are unchanged for both root
profiles. Only the construction of the public coefficient weights changes.

`PreparedRootMatrices` retains the `PreparedCommitMatrix` own-ring transform
cache. There is no conjugate-ring transform cache. The response-weight kernel
forms G^-1(u_i) through the verifier's common trace-map owner and performs:

1. n_A forward transforms of G^-1(u_i).
2. For each response ring column j, n_A pointwise multiply-accumulates against
   the cached transforms of A_ij, followed by one inverse transform and G.
3. Addition of the unchanged parity coefficient weight, scaled by gamma^n_A.

Column tasks use the Rayon pool behind `parallel`, with per-task transform
workspaces and fallible reservations. They write directly into the compact
`m*padded_coefficients` table passed to `CombinedRootKernel::new`; coefficient
tails are zero. There is no intermediate dense `m*D` matrix-weight copy and
no full digit-expanded weight table. The matrix part is traced by
`root_a_weights`. At the first geometry the compact table has 4,194,304 field
entries, or 64 MiB for 16-byte field elements. Transformed inputs and worker
workspaces add O(n_A*D + workers*D) field storage, beyond the retained matrix
cache and this table. At rank 3 the P-field matrix cache contains 7,962,624
field entries, or 127,401,984 bytes; these slots belong to F_P, not F_q0.

The product sumcheck's dense image-weight table uses C*n_A challenge products
applied to G^-1(u_i), followed by G and negation. The shared verifier adjoint
owner keeps this construction consistent with the dense reference. Products
by weight-46 embedded challenges may use their sparse representation. The
image table retains coefficient tails and final entry padding with zero
weights; no separate rank tensor block is introduced.

Tag 0 constructs no matrix-row auxiliary witness. Tag 1's `root_a_carry`
phase uses the own-ring commitment cache to obtain the remainder, then centres
its coefficients, checks integer divisibility by q0 and enforces the admitted
carry range. This phase precedes alpha and does not build a quotient. The
carry grammar and admission remain owned by the small-modulus specification.

## Messages and prover storage

The generic driver omits the linear coefficient and reconstructs it from the
running claim. A combined round sends `2^b+1` field coefficients, for a total
of `nu*(2^b+1)` coefficients. Product rounds send two coefficients, totaling
`2*mu`. With canonical coefficient width `F::NUM_BYTES`, the round bodies
occupy `nu*(2^b+1)*F::NUM_BYTES` and `2*mu*F::NUM_BYTES` bytes, respectively.
Public claims and the instance header are absorbed rather than transmitted.
Message contexts and site identifiers are diagnostic only; they absorb no bytes.
There are no proof-supplied lengths. Replay returns the challenge point and
final claim; it MUST be followed by the terminal check and, at the enclosing
proof boundary, the channel's EOF check.

The combined prover evaluates both terms in one pass per round and interpolates
at nodes `0,...,2^b+1`. It packs the input digits and delays lifting them to
field elements for `r=min(nu,4/3/2)` rounds for one-, two-, and four-bit digits.
Equality uses two suffix tensor factors. Response weights use the tensor of
`digit_powers` (length `2^a`, occupying the lowest variables) and a compact
coefficient table `cw` (length `N/2^a`, coefficient-low, ring-element-high).
Padded coefficient positions in `cw` are zero. The prover never expands these
weight factors into a length-`N` field table.

The first `a` challenges fold only the digit factor. Its final scalar is applied
to `cw` in place, and subsequent rounds fold that table. For the first root
profile, `a=r`: packed class buckets sum compact coefficients by class and
remaining digit-pair position, then apply the digit factor once per bucket.
Other factorizations use direct weight endpoint evaluation. Once the packed
rounds finish, the lifted digit table has length `N/2^r`.

The exact reservation bound, including worker buckets and small tables, is
specified on `CombinedRootKernel`. At `nu=25`, two-bit digits, `a=3`, 16-byte
field elements and one worker, it is 148,579,168 bytes (about 141.7 MiB),
excluding caller-retained originals, allocator rounding and interpolation
scratch. The two main field tables each occupy 64 MiB. The product prover owns
two length-`2^mu` field tables and constant-size round workspace.

## Scope and validation

The lowered relation and weight construction are specified separately. The
complete root transcript binding, oracle seam, batching, and wire encodings
are specified in [`labinius-root-reduction.md`](labinius-root-reduction.md).
Polynomial commitment implementations, planner and schedule changes remain
outside this specification.
Zero-variable instances are permitted; their input claim is already terminal.

`crates/akita-labinius-prover/tests/root_sumcheck.rs` exercises the real drivers
through the local LaBinius adapters under both transcript backends, manual
round loops, independent Boolean expansions, malformed constructors, excess
degrees, terminal mutations, and test-only invalid-alphabet witnesses with
an exactly satisfied linear relation, including byte 255. The companion
`tests/root_sumcheck_binding.rs` checks every instance-header field, exhaustive
one-variable digit tables at every interpolation node for all bases and both
coefficient fields, and cancellation with a claim fixed before `beta`.
Existing clear-opening behavior is unchanged.
