# Cyclotomic rings and extension fields

> **Status:** stub. Part of the initial Akita Book scaffold.

The algebraic substrate of Akita: the power-of-two cyclotomic ring
\\( R_q = \mathbb{Z}_q[X]/(X^d+1) \\), how it splits into extension fields, the
subfield where extension-field points live, and the norms that control folding.
This page folds directly from paper §2.1, with the concrete low-degree
arithmetic from implementation appendix B.1.4.

## The ring and partial splitting

\\( R_q = \mathbb{Z}_q[X]/(X^d+1) \\) with \\( d = 2^\alpha \\); when
\\( q \equiv 2k+1 \pmod{4k} \\), \\( X^d+1 \\) splits into \\( k \\) irreducible
factors of degree \\( d/k \\), giving \\( R_q \cong \prod \mathbb{F}_{q^{d/k}} \\)
by CRT. Larger \\( k \\) means faster NTT arithmetic at a tighter invertibility
threshold.

**Sources to fold in**

- Paper §2.1 `sec:prelim-ring` ("Partial splitting", `eq:partial-split`; LS18 Cor 1.2).
- `crates/akita-algebra/src/ring/cyclotomic.rs`.

## Extension-field embedding (cyclotomic ring-subfield coordinates)

The trace/Galois embedding of \\( \mathbb{F}_{q^{k'}} \\) into \\( R_q \\),
the explicit cyclotomic ring-subfield basis \\( e_j = X^{jm}+X^{-jm} \\), and
the concrete degree-2 and degree-4 arithmetic in the implementation
(`fp_ext2`, `fp_ext4`, `fp_ext8`; no separate power/tower quartic path).

**Sources to fold in**

- Paper §2.1 ("Extension-field embedding", "Ring-subfield coordinates"; Hachi Thm 2 / Lemma 4), App B.1.4 `sec:akita-ext-fields` (degree-2/4 multiplication tables, tower squaring/inversion, the \\( K=4 \\) trace map).
- `crates/akita-field/src/ext/` (`fp_ext2.rs`, `fp_ext4.rs`, `fp_ext8.rs`, `lift.rs`, `native_algebra.rs`).

## Base-field coefficients vs extension evaluation points

The two roles of an extension \\( E = \mathbb{F}_{q^{k'}} \\): coefficient field
of the committed polynomial vs the challenge/evaluation field used by sum-check.
Akita commonly commits \\( \mathbb{F}_q \\)-valued tables but evaluates at points
in \\( E \\) for negligible soundness — the mismatch the extension-opening
reduction later resolves.

**Sources to fold in**

- Paper §2.1 ("Base-field coefficients and extension-field points").
- `crates/akita-field/src/ext/lift.rs`, `ext/mod.rs`.

## Challenge subrings and coefficient packing

For extension degree `k`, coefficient packing uses three rings:

```text
R = K[X]/(X^d_A+1),
S = K[Y]/(Y^s+1),
C = E[Y]/(Y^s+1).
```

The A ring `R` holds committed data. The challenge subring `S` holds sparse
fold challenges. The extension opening ring `C` holds partial evaluations.
The dimensions satisfy `d_A = k h s`.

The challenge subring embeds into the A ring by `Y -> X^(k h)`. This embedding
acts only on one A coefficient axis. The partial evaluation contracts the
other axis and keeps `s` coefficients in `E`. A packed partial therefore has
`k` base field coordinate planes of length `s`. Its physical width is `k s`,
but its polynomial modulus still has dimension `s`.

The schedule selects `s`. It does not change the extension degree or the
committed field. The implementation uses one canonical extension basis and one
canonical coefficient order.

See [Root fold and ring switching](../how/proving/root-fold-ring-switch.md#subring-coefficient-packing)
for the coefficient grid, the commutation rule that makes packing valid, and a
worked fp32 example.

**Implementation map**

- `crates/akita-types/src/subring_coefficient_packing.rs`.
- `crates/akita-types/src/proof/coefficient_packing_relation.rs`.

## Norms, invertibility, and challenge families

The centered \\( \ell_\infty, \ell_1, \ell_2 \\) norms on \\( R_q \\), the
invertibility bound \\( \lVert c \rVert_\infty < q^{1/k}/\sqrt{k} \\), the
challenge family with bounded \\( \ell_1 \\)-norm and invertible pairwise
differences, and challenge L1 mass \\( \omega = \lVert c \rVert_1 \\) for
folded-witness collision sizing.

**Sources to fold in**

- Paper §2.1 ("Norms, invertibility, and challenges").
- `crates/akita-challenges/src/` (challenge sampling), `crates/akita-types/src/sis/norm_bound.rs`.
