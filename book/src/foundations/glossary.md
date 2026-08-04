# Glossary and notation

> **Status:** stub. Part of the initial Akita Book scaffold.

Two reference tables a newcomer keeps open while reading: the glossary of Akita
terms, and the symbol table mapping paper notation to code names. Keep both in
sync with `specs/archive/2026-Q2/w-to-e-notation.md`.

## Glossary

Plain-language definitions of the load-bearing terms: root vs recursive level,
fold, ring switch, extension-opening reduction, digit-innermost layout, one-hot vs dense,
folded-only proof family, schedule, `LevelParams`, the A/B/D
roles, weak binding, fold price.

**Sources to fold in**

- `specs/archive/2026-Q2/w-to-e-notation.md:49-68`,
  `book/src/how/proving/opening-points-layout.md`.
- Council newcomer report (full glossary table).

## Notation

The symbol table: \\( q, d, k, R_q, \mathbf{G}_{b,n}, \delta, \beta, \Gamma(c),
\mathrm{eq}(\tau,x), \mu, \ell \\), and the paper ↔ code naming (`w` / `e` / `v`,
`M`, the level index). The canonical post-cutover symbols.

**Sources to fold in**

- Paper §2 (notation introduced section by section), `preamble.tex` (macro definitions).
- `specs/archive/2026-Q2/w-to-e-notation.md` (canonical post-cutover symbols).
- `crates/akita-types/src/layout/params.rs`, `crates/akita-prover/src/protocol/ring_relation.rs`.
