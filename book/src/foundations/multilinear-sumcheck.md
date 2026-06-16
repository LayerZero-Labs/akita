# Multilinear extensions and sum-check

> **Status:** stub. Part of the initial Akita Book scaffold.

The sum-check protocol is the engine of every Akita fold. This page is the
generic background (paper §2.3): multilinear extensions, the compressed
protocol, its soundness, and how multiple claims batch into one run.

## Multilinear extensions

The MLE \\( \widetilde{f}(X) = \sum_{i} f(i)\,\mathrm{eq}(i,X) \\), the factored
equality polynomial \\( \mathrm{eq} \\), and identifying a vector with its MLE.

**Sources to fold in**

- Paper §2.3 `sec:prelim-sumcheck` (MLE definition).
- `crates/akita-algebra/src/poly.rs`, `eq_poly.rs`, `split_eq.rs`, `uni_poly.rs`.

## The compressed sum-check protocol

The round-by-round protocol with the **linear coefficient omitted** (the
verifier recovers it from the running claim \\( T_j = s_j(0)+s_j(1) \\)), and the
final oracle check.

**Sources to fold in**

- Paper §2.3 (`fig:sumcheck`, the compressed message; LFKN/Lund et al.).
- `crates/akita-sumcheck/src/` (`traits.rs`, `drivers/`, `compact_fold.rs`).

## Soundness and special soundness

Round-by-round soundness \\( \le \mu\ell/|\mathbb{F}| \\) and the
\\( (\ell+1) \\)-special-soundness used by the CWSS knowledge-error analysis.

**Sources to fold in**

- Paper §2.3 (`thm:sumcheck-soundness`; Hachi Lemma 9).
- Cross-link: [Polynomial commitments and binding](./pcs-and-binding.md) (CWSS).

## Batched sum-check

Folding \\( t \\) claims of possibly different arities into one run via a random
linear combination and a prefix (or suffix, or arbitrary-subset) variable
embedding; how inactive instances contribute a constant.

**Sources to fold in**

- Paper §2.3 ("Batching"; prefix vs suffix embedding).
- `crates/akita-sumcheck/src/batched_sumcheck.rs`, `accum.rs`.
- External: Thaler PAZK, Gruen eq-poly (ePrint 2024/1210). See [References](./references.md).
