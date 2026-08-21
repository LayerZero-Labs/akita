# Polynomial commitments and binding

Akita is a multilinear polynomial commitment scheme. Its public interface has
four algorithms: setup, commitment, opening, and verification. Completeness
says that an honest opening verifies. Binding says that one commitment cannot
be opened to incompatible values. Knowledge soundness says that a prover which
opens successfully can be used to extract a witness, except with a stated
knowledge error.

## The two-tier Ajtai commitment

Akita commits with three public matrices. The A matrix compresses the
decomposed witness. The B matrix commits the A output. The D matrix is used by
the opening relation. This is the two-tier Ajtai structure used by LaBRADOR,
Greyhound, and Hachi.

The security reduction turns an invalid opening into a short nonzero kernel
vector for one of these matrices. The exact ring dimension, module rank, input
width, and accepted norm therefore matter. Akita stores these values in the
selected schedule and validates them again when the schedule is expanded. The
coefficient Linf and Euclidean L2 routes use different SIS tables because they
certify different kernel-vector norms.

See [Setup and commitment](../how/commitment.md) for the protocol wiring and
[Security model](../how/security.md) for the concrete SIS policy.

## Coordinate-wise special soundness

Akita's folding protocol has coordinate-wise special soundness, or CWSS. At a
challenge coordinate with support S, an extractor needs a bounded number of
distinct accepting challenge answers. If the protocol has challenge depth
`ell_i` and extraction degree `k_i` at coordinate `i`, the interactive
knowledge error is bounded by the corresponding CWSS sum. In the common form
used by Akita, each term is proportional to

```text
ell_i * (k_i - 1) / |S_i|.
```

This is the challenge-space error before Fiat-Shamir compilation. The precise
parameters and batching factors must be taken from the schedule being proved.

## Fiat-Shamir queries and fold nonces

Fiat-Shamir replaces each verifier challenge with a random-oracle answer. The
reduction accounts for the adversary's total number of oracle queries. If the
interactive CWSS knowledge error is `kappa`, the standard online extractor
bound has the form

```text
Fiat-Shamir knowledge error <= (Q + 1) * kappa,
```

where `Q` is the adversary's total random-oracle query budget. This query factor
is the correct way to account for repeated challenge trials.

Each Akita fold carries a bounded nonce. The nonce is absorbed before the fold
challenge is sampled. Trying another nonce is therefore another random-oracle
query. It does not create a separate fixed loss of `log2(max_nonce_count)` bits
at every fold. For a fixed transcript prefix and a bad-challenge set of measure
`epsilon`, `q` trials have success probability

```text
1 - (1 - epsilon)^q <= q * epsilon.
```

The same `q` trials also cost `q` oracle queries. The Fiat-Shamir theorem already
charges that work through `Q`. Adding a second static entropy debit would count
the same freedom twice. The same reasoning applies when a prover varies honest
randomness, including future zero-knowledge blinding, before asking for the
next challenge. What matters is that every distinct attempt is represented by
a distinct oracle query and that the verifier reconstructs the accepted query.

This soundness accounting is separate from honest-prover grinding. The response
model may predict that an honest response fits its cap with probability at
least `1/40`, and the protocol may allow 4096 attempts to make exhaustion
negligible. That is a completeness statement. It does not grant the adversary
4096 free soundness attempts. Adversarial attempts remain part of `Q`.

The argument depends on the random-oracle and CWSS extraction theorems. It does
not claim concrete security for an unbounded adversary. A concrete deployment
must choose a query budget, instantiate every challenge support size, and apply
the complete CWSS sum.

Primary references are the CWSS and Fiat-Shamir analysis in
[Lattice-Based Polynomial Commitments](https://eprint.iacr.org/2023/846) and the
online-extractability theorem of
[Attema et al.](https://ir.cwi.nl/pub/33324/33324.pdf).

## Implementation map

- `crates/akita-types/src/sis/` owns matrix and norm security parameters.
- `crates/akita-types/src/instance_descriptor/` binds the protocol-wide nonce
  wire contract and challenge policy.
- `crates/akita-prover/src/protocol/fold_grind.rs` performs bounded honest
  probing.
- The selected schedule binds each fold's challenge and response admission
  rule.
