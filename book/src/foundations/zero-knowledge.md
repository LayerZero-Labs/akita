# Zero-knowledge background

> **Status:** stub. Part of the initial Akita Book scaffold.

The background needed to follow Akita's zero-knowledge work: why lattice PCS
leaks witness data through sum-check rounds, commitments, and terminal openings;
the committed-pad masking idea; and the prefix / seam / suffix pipeline from
paper §6 `sec:zk`. The roadmap page tracks implementation against that design:
[Roadmap → Zero-knowledge](../roadmap/zero-knowledge.md).

## Why ZK is hard over lattices

Blindfold-style techniques (Jolt-with-Dory) lean on a commitment homomorphism
and small commitments; both are blocked over lattices. And Akita is always a PCS
inside a larger PIOP, so ZK must hold through the whole stack, not just the PCS.

**Sources to fold in**

- Paper §6 `sec:zk` ("Where the difficulty lies"; `sections/akita/6_zero_knowledge.tex`).

## The pipeline (prefix / seam / suffix)

Paper §6 `sec:zk-pipeline`: leakage inventory (sum-check rounds, level-transition
commitments, terminal opening); the three regions that close it. The seam
(`sec:zk-joint-sigma`) seals zero knowledge; the suffix is ordinary non-ZK
opening of the masked response. Modulus switching runs only in the suffix.

**Sources to fold in**

- Paper §6 `sec:zk-pipeline`, `fig:zk-pipeline`, `sec:zk-joint-sigma`.
- `crates/akita-r1cs/src/lib.rs` (deferred R1CS instance), `crates/akita-pcs/tests/zk.rs`.
- `specs/akita-zk-sumcheck-hiding-plain.md` (plain-opening implementation vs seam).

## Masking strategies (prefix detail)

Mask the output, or mask the entire polynomial. Masking the whole polynomial
requires Gaussian masking (with room to overflow the prime) to stay supported by
MSIS — building on the [discrete Gaussians and rejection sampling](./lattices-sis.md#one-shot-and-iterative-rejection-sampling)
toolkit.

**Sources to fold in**

- Paper §6 ("Two load-bearing ideas", §6.3 `sec:zk-sumcheck-mask` committed-pad masking).
- `crates/akita-prover/src/protocol/masking.rs`, `zk_hiding_commit.rs`.
- `specs/akita-zk-commitment-hiding.md`, `akita-zk-v-hiding.md`.