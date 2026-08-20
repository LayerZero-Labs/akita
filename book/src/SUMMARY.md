# Summary

[Introduction](./intro.md)
  - [Why lattices?](./introduction/why-lattices.md)
  - [Built for production](./introduction/built-for-production.md)
  - [Reviewing and auditing Akita](./introduction/reviewing-akita.md)

# Using Akita

- [Using Akita](./usage/usage.md)
  - [Your first proof](./usage/quickstart.md)
  - [Choosing a configuration](./usage/configuration.md)
  - [Integrating the PCS](./usage/integration.md)
    - [Commitment groups and opening claims](./usage/commitment-api.md)
    - [Setup and prepared compute state](./usage/setup-runtime.md)
    - [Proof encoding and transcripts](./usage/proof-artifacts.md)
    - [Verifier only integration](./usage/verifier-only.md)
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
    - [Field-to-ring evaluation reduction](./how/proving/field-ring-reduction.md)
    - [Semantic relations in an Akita fold](./how/proving/akita-fold.md)
    - [Raw and compressed realizations of an Akita fold](./how/proving/akita-fold-realizations.md)
    - [Advanced relation layouts](./how/proving/advanced-relation-layouts.md)
    - [Opening points and digit-innermost layout](./how/proving/opening-points-layout.md)
    - [Fold path and field geometry](./how/proving/fold-path.md)
    - [Root fold and ring switching](./how/proving/root-fold-ring-switch.md)
    - [Sumcheck stages](./how/proving/sumcheck-stages.md)
    - [Extension-opening reduction](./how/proving/extension-opening-reduction.md)
  - [The distributed prover](./how/proving/distributed-prover.md)
  - [Recursion and proof shape](./how/recursion.md)
  - [Setup offloading](./how/setup-offloading.md)
  - [Verification](./how/verification.md)
    - [Matrix evaluation at a point](./how/verifying/matrix_evaluation.md)
    - [The Stage 2 fused check](./how/verifying/stage2.md)
    - [Evaluation trace](./how/verifying/evaluation_trace.md)
    - [Setup contribution and Stage 3](./how/verifying/setup_contribution.md)
    - [The distributed relation verifier](./how/verifying/distributed-relation-verifier.md)
    - [Terminal verification](./how/verifying/terminal.md)
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
  - [Glossary and notation](./foundations/glossary.md)
  - [Spec index](./foundations/spec-index.md)
  - [References](./foundations/references.md)

# Roadmap

- [Roadmap](./roadmap/roadmap.md)
  - [Zero knowledge](./roadmap/zero-knowledge.md)
  - [Compute backends (GPU/Metal)](./roadmap/compute-backends.md)
