# Foundations

> **Status:** stub. Part of the initial Akita Book scaffold.

The background a reader needs before (or alongside) the protocol chapters. The
spine of this part follows the **Preliminaries** of the Akita paper, so each
page can be folded directly from a known section, with the implementation
appendices supplying the concrete arithmetic.

Distinguish generic background (teachable from any source) from Akita-specific
choices, and flag what is liable to change (the paper is a work in progress).

Reading map (paper section → page):

- §2.1 → [Cyclotomic rings and extension fields](./rings-and-fields.md)
- Ring/extension evaluation pairing → [Trace openings](./trace-open.md)
- App B.2 → [NTT, CRT, and fast ring arithmetic](./ntt-crt.md)
- §2.2 → [Gadget decomposition](./gadget-decomposition.md)
- §2.2 + §2.7 → [Lattices, Module-SIS, and discrete Gaussians](./lattices-sis.md)
- §2.3 → [Multilinear extensions and sum-check](./multilinear-sumcheck.md)
- §2.4 → [Equality-factored sum-check](./eq-factored-sumcheck.md)
- §2.5 → [Extension-opening reduction](./extension-opening-reduction.md)
- §2.6 → [Polynomial commitments and binding](./pcs-and-binding.md)
- §6 → [Zero-knowledge background](./zero-knowledge.md)

Plus reference material: [glossary and notation](./glossary.md), the
[spec index](./spec-index.md), and the [references](./references.md).

## Sources to fold in

- Paper §2 `sec:preliminaries` (the canonical scoping for this whole part).
- Council math-foundations report (Part A generic, Part B protocol-specific).
- Crate-level: `akita-field`, `akita-algebra`, `akita-sumcheck`, `akita-challenges`, `akita-types/src/sis`.
