# Zero-knowledge background

> **Status:** stub. Part of the initial Akita Book scaffold.

The background needed to follow Akita's zero-knowledge work: why ZK is hard over
lattices, the two masking strategies, and the Gaussian + MSIS-support reasoning
they rest on. Akita's concrete full-ZK construction is mostly in-flight; see
[Roadmap → Zero-knowledge](../roadmap/zero-knowledge.md).

## Why ZK is hard over lattices

Blindfold-style techniques (Jolt-with-Dory) lean on a commitment homomorphism
and small commitments; both are blocked over lattices. And Akita is always a PCS
inside a larger PIOP, so ZK must hold through the whole stack, not just the PCS.

**Sources to fold in**

- Paper §6 `sec:zk` (motivation, "Where the difficulty lies").

## Masking strategies

Mask the output, or mask the entire polynomial. Masking the whole polynomial
requires Gaussian masking (with room to overflow the prime) to stay supported by
MSIS — building on the [discrete Gaussians and rejection sampling](./lattices-sis.md#one-shot-and-iterative-rejection-sampling)
toolkit.

**Sources to fold in**

- Paper §6 ("Two load-bearing ideas", §6.3 `sec:zk-sumcheck-mask` committed-pad masking).
- `crates/akita-prover/src/protocol/masking.rs`, `zk_hiding_commit.rs`.

## The pipeline view (prefix / seam / suffix)

How the masked recursion (prefix), the committed-response seal (seam), and the
non-zero-knowledge opening (suffix) compose, and the leakage inventory that
motivates each region.

**Sources to fold in**

- Paper §6.1 `sec:zk-pipeline`, §6.5 `sec:zk-joint-sigma` (committed-response tail).
- `crates/akita-r1cs/src/lib.rs` (deferred R1CS instance), `crates/akita-pcs/tests/zk.rs`.
- `specs/akita-zk-commitment-hiding.md`, `akita-zk-v-hiding.md`, `akita-zk-sumcheck-hiding-plain.md`.
