# Lattices, Module-SIS, and discrete Gaussians

> **Status:** stub. Part of the initial Akita Book scaffold.

The hardness assumption Akita reduces to, and the Gaussian toolkit used by the
zero-knowledge layer. Generic background here; the Akita-specific parameter
sizing lives in [How it works → Security model](../how/security.md).

## Module-SIS

The \\( \mathsf{MSIS}_{q,d}(n,m,\beta) \\) problem (find a short nonzero kernel
vector of a uniform \\( M \in R_q^{n\times m} \\)), the two norm models used
(\\( \ell_\infty \\) and \\( \ell_2 \\)), and the dimension factor
\\( \lVert z \rVert_\infty \le \lVert z \rVert_2 \le \sqrt{md}\,\lVert z \rVert_\infty \\)
that lets a role be re-priced between them.

**Sources to fold in**

- Paper §2.2 `def:msis` (Module-SIS; the two-norm interchange).
- `crates/akita-types/src/sis/` (`mod.rs`, `ajtai_key.rs`, `generated_sis_linf_table/`).
- `specs/akita-sis-consolidation.md`.
- External: Hachi (Lemma 5-7), SuperNeo, LaBRADOR. See [References](./references.md).

## Discrete Gaussians and the tail bound

The discrete Gaussian \\( D_{\Lambda,\sigma,c} \\), the standard tail bound, and
the regime in which it is negligible — the foundation for Gaussian masking.

**Sources to fold in**

- Paper §2.7 `sec:prelim-gaussian` (`eq:gaussian-tail`).
- External: Lyubashevsky trapdoors (Thm 4.6).

## One-shot and iterative rejection sampling

The rejection-sampling lemma (output within negligible statistical distance of a
centered Gaussian, success \\( \ge 1/M \\)) and the iterative variant whose width
scales with a single summand's norm rather than the total — a \\( \kappa \\)-fold
width reduction used by the masked sum-check tail.

**Sources to fold in**

- Paper §2.7 (`lem:rejection-sampling`, `lem:iterative-rejection`).
- Cross-link: [Zero-knowledge background](./zero-knowledge.md), [Roadmap → Zero-knowledge](../roadmap/zero-knowledge.md).
