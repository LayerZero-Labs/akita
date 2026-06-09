# Summary

[Introduction](./intro.md)

# Usage

- [Overview](./usage/usage.md)
  - [Quickstart and configuration](./usage/quickstart.md)
  - [The commitment API](./usage/commitment-api.md)
  - [Verifier-only integration](./usage/verifier-only.md)
  - [Feature flags](./usage/feature-flags.md)
  - [Profiling](./usage/profiling.md)
  - [Troubleshooting](./usage/troubleshooting.md)
  - [Jolt recursion](./usage/jolt-recursion.md)

# How it works

- [How it works](./how/how-it-works.md)
  - [Architecture overview](./how/architecture.md)
  - [Configuration and planning](./how/configuration.md)
  - [Setup and commitment](./how/commitment.md)
  - [Transcript and instance binding](./how/transcript.md)
  - [The proving protocol](./how/proving/proving.md)
    - [Opening points and block order](./how/proving/opening-points-block-order.md)
    - [Root fold and ring switching](./how/proving/root-fold-ring-switch.md)
    - [Sumcheck stages](./how/proving/sumcheck-stages.md)
    - [Extension-opening reduction](./how/proving/extension-opening-reduction.md)
  - [Recursion and proof shape](./how/recursion.md)
  - [Verification](./how/verification.md)
  - [Security model](./how/security.md)
  - [Optimizations](./how/optimizations.md)

# Foundations

- [Foundations](./foundations/foundations.md)
  - [Cyclotomic rings and extension fields](./foundations/rings-and-fields.md)
  - [NTT, CRT, and fast ring arithmetic](./foundations/ntt-crt.md)
  - [Gadget decomposition](./foundations/gadget-decomposition.md)
  - [Lattices, Module-SIS, and discrete Gaussians](./foundations/lattices-sis.md)
  - [Multilinear extensions and sum-check](./foundations/multilinear-sumcheck.md)
  - [Equality-factored sum-check](./foundations/eq-factored-sumcheck.md)
  - [Extension-opening reduction](./foundations/extension-opening-reduction.md)
  - [Polynomial commitments and binding](./foundations/pcs-and-binding.md)
  - [Operator-norm certification](./foundations/operator-norm-certification.md)
  - [Zero-knowledge background](./foundations/zero-knowledge.md)
  - [Glossary and notation](./foundations/glossary.md)
  - [Spec index](./foundations/spec-index.md)
  - [References](./foundations/references.md)

# Roadmap

- [Roadmap](./roadmap/roadmap.md)
  - [L2-MSIS cutover](./roadmap/l2-msis.md)
  - [Modulus switching](./roadmap/modulus-switching.md)
  - [Zero-knowledge (Whiteout)](./roadmap/zk-whiteout.md)
  - [Compute backends (GPU/Metal)](./roadmap/compute-backends.md)
