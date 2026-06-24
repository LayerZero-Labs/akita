# Polynomial commitments and binding

> **Status:** stub. Part of the initial Akita Book scaffold.

What it means to be a multilinear PCS, the two-tier Ajtai template Akita
instantiates, and the coordinate-wise special-soundness framework that yields
the fold's knowledge error. Folds from paper §2.6.

## The multilinear PCS definition

The four algorithms (`Setup` / `Commit` / `Eval.P` / `Eval.V`) and the three
properties: completeness, binding, and knowledge soundness (extractor with
rewinding access).

**Sources to fold in**

- Paper §2.6 `def:pcs`.
- `crates/akita-types/src/proof/scheme.rs`, `crates/akita-prover/src/api/scheme.rs`.

## The two-tier Ajtai commitment

The \\( \mathbf{A}/\mathbf{B}/\mathbf{D} \\) matrices and their roles: inner
commitment \\( \mathbf{t} = \mathbf{A}\,\mathbf{G}^{-1}(\mathbf{f}) \\), outer
commitment via \\( \mathbf{B} \\), opening commitment via \\( \mathbf{D} \\) — the
LaBRADOR/Greyhound/Hachi template whose binding reduces to Module-SIS. The Akita
wiring is in [How it works → Setup and commitment](../how/commitment.md).

**Sources to fold in**

- Paper §2.6 ("Two-tier Ajtai commitment").
- `crates/akita-types/src/sis/ajtai_key.rs`.

## Coordinate-wise special soundness (CWSS)

The CWSS framework: special-sound tree structure, the CWSS protocol definition,
and the knowledge-error bound \\( \sum_i \ell_i k_i / |\mathcal{C}_i|^{\ell_i} \\),
which carries to the Fiat-Shamir-compiled protocol in the ROM.

**Sources to fold in**

- Paper §2.6 (`def:cwss`, `def:cwss-protocol`, `thm:cwss-knowledge-error`; Hachi Def 3 / Lemma 3, Attema et al.).
- Applied to Akita in Paper §3.11 `sec:akita-cwss`, §3.12 `sec:batched-soundness`.
