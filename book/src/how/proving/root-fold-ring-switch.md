# Root fold and ring switching

One folding step builds a batched relation and switches it into the next ring
witness. The schedule also selects how each commitment group is opened.

## Subring coefficient packing

Let the claim field have extension degree `k`, let the A ring have dimension
`d_A`, and let the challenge subring have dimension `s`. An admitted packing
geometry satisfies

```text
d_A = k h s
```

for a power of two packing factor `h`. The prover reads source coefficient
`a + k h j`, contracts the `a` coordinate over the claim field, and keeps the
subring coefficient `j` explicit. One partial has `k s` base field
coordinates. The order is extension coordinate followed by subring
coefficient.

The fold challenge is sampled once in `K[Y]/(Y^s+1)`. The A relation uses the
same coefficients embedded by `Y -> X^(k h)`. The packing consistency row uses
the challenge at `alpha`, while the A rows use it at `alpha^(k h)`. The prover
also records the positive high half of the ordinary product as `Q_pack`.

Stage 2 keeps this check compact. It uses the normal alpha based term for the
native relation rows and separate factorized terms for each packing group's Z
weights and direct opening. These terms share one witness point. The prover and
verifier do not build a dense witness sized weight table.

Production planning considers packing only at absolute fold levels 0 and 1.
Those folds use the coefficient `L∞` A security route. Later folds and the
terminal use evaluation trace. If no complete packing assignment is feasible,
the planner retains an evaluation trace fallback instead of dropping the row.

Commitment identity records the coefficient representation and the A and B
matrices. It does not record the consuming opening method, `s`, or challenge.
The schedule and transcript descriptor record that opening plan. This lets a
later evaluation trace fold consume a flat witness or setup prefix produced by
an earlier packing fold without changing its commitment identity.

The prover binds the complete D payload, or its compressed H payload, before it
draws the subring challenge. It binds `Q_pack` and the next witness before it
draws `alpha`. The verifier replays the same order.

## The root fold

`OpeningClaimsLayout` routes polynomial groups to claims. Each group keeps its
own public point, commitment profile, and opening geometry. The relation order
is final group followed by precommitted groups. A recursive fold uses the same
group rules for its folded witness and an incoming setup prefix.

## Ring switching

The lattice fold lifts the relation `M w = h` from
`R_q = Z_q[X]/(X^D+1)` to `Z_q[X]` through its unique quotient. The prover
computes native ring quotients with paired cyclic and negacyclic NTTs. A packing
consistency row instead has modulus dimension `s` and `k` coordinate planes.
Its physical width is `k s`; it is not a ring of dimension `k s`.

This ring switch is distinct from EOR. EOR changes an extension valued opening
claim before the lattice relation. Ring switching proves the polynomial
quotients of the lattice relation itself.

## Implementation map

- `crates/akita-prover/src/protocol/ring_relation.rs`.
- `crates/akita-prover/src/protocol/ring_switch.rs`.
- `crates/akita-prover/src/protocol/coefficient_packing.rs`.
- `crates/akita-types/src/subring_coefficient_packing.rs`.
- `crates/akita-types/src/proof/coefficient_packing_relation.rs`.
- Paper section 3.5 and implementation appendix B.3.
