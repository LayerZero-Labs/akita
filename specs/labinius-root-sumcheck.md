# LaBinius root sumchecks

Status: implemented arithmetic kernels; root protocol integration is out of scope.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY in this document
have the BCP 14 meanings defined by RFC 2119 and RFC 8174 when capitalized.

## Statement and conventions

Let `F` be the coefficient prime field, with characteristic greater than 16.
Both instances use `E = F`, without an extension field. Let `b` be 1, 2, or 4,
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
It MUST fix `W`, `tau`, `K_W`, and `s` before drawing uniform `beta`.
It MUST bind all tables, public weights, points, and input claims before
the round challenges. Each round message MUST precede its challenge.
These kernels take the inputs explicitly; their channel adapters only supply
sumcheck message contexts and uniform field challenges. They do not implement
the enclosing statement binding order.

If some Boolean `x*` has an invalid digit, then
`Z(tau) = sum_x eq(tau,x) P_b(W(x))` is a nonzero multilinear polynomial in
`tau`: its Boolean evaluation at `x*` equals `P_b(W(x*)) != 0`.
For fixed `W`, it vanishes at uniform `tau` with probability at most
`nu/|F|`.

Let `L = sum_x W(x) K_W(x)`. The combined claim requires
`Z(tau) + beta*(L-s) = 0`. If `Z(tau) != 0` or `L != s`, at most one
`beta` satisfies this equation, so the additional error is at most `1/|F|`.
When `L=s` and `Z(tau) != 0`, no `beta` satisfies it.

Sumcheck contributes at most `nu*(2^b+1)/|F|` for the combined instance and
`2*mu/|F|` for the product instance. These statements require the terminal
evaluations to correspond to the fixed tables; the kernels alone do not prove
that correspondence. A union bound for these arithmetic reductions is
`(nu + 1 + nu*(2^b+1) + 2*mu)/|F|`, apart from opening errors.

## Messages and prover storage

The generic driver omits the linear coefficient and reconstructs it from the
running claim. A combined round sends `2^b+1` field coefficients, for a total
of `nu*(2^b+1)` coefficients. Product rounds send two coefficients, totaling
`2*mu`. With canonical coefficient width `F::NUM_BYTES`, the round bodies
occupy `nu*(2^b+1)*F::NUM_BYTES` and `2*mu*F::NUM_BYTES` bytes, respectively.
Public claims and message contexts are absorbed rather than transmitted.
There are no proof-supplied lengths. Replay returns the challenge point and
final claim; it MUST be followed by the terminal check and, at the enclosing
proof boundary, the channel's EOF check.

The combined prover evaluates both terms in one pass per round and interpolates
at nodes `0,...,2^b+1`. Round zero uses a digit-pair lookup table with `2^(2b)`
rows. Later rounds use folded field values. For `N=2^nu`, peak owned storage is

```text
(2N + max(1,N/2) + nu + 2^(2b)*(2^b+2))*size_of::<F>()
    + N bytes + O(2^b) field workspace.
```

This excludes caller-retained originals and excess capacities of supplied
vectors. A dense field table at `nu=26` occupies 1 GiB when `F` occupies
16 bytes; the two dense tables and equality suffix occupy 2.5 GiB, plus the
digit copy and small lookup/workspace storage. The product prover owns two
length-`2^mu` field tables and constant-size round workspace.

## Scope and validation

The lowered relation, weight construction, enclosing transcript binding,
commitments, polynomial openings, batching, wire encodings for the complete
root, planner and schedule changes are outside this specification.
Zero-variable instances are permitted; their input claim is already terminal.

`crates/akita-labinius-prover/tests/root_sumcheck.rs` exercises the real drivers
through the local LaBinius adapters under both transcript backends, manual
round loops, independent Boolean expansions, malformed constructors, excess
degrees, terminal mutations, and test-only invalid-alphabet witnesses with
an exactly satisfied linear relation. Existing clear-opening behavior is
unchanged.
