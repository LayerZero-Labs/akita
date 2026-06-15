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
- `crates/akita-algebra/src/ring/cyclotomic.rs`, `ring/partial_split_ntt.rs`.

## Extension-field embedding and ring-subfield coordinates

The trace/Galois embedding of \\( \mathbb{F}_{q^{k'}} \\) into \\( R_q \\) (the
Hachi packing), the explicit ring-subfield basis \\( e_j = X^{jm}+X^{-jm} \\),
and the concrete degree-2 and degree-4 arithmetic (Karatsuba, the tower form,
the closed-form trace decode).

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

## Norms, invertibility, and challenge families

The centered \\( \ell_\infty, \ell_1, \ell_2 \\) norms on \\( R_q \\), the
invertibility bound \\( \lVert c \rVert_\infty < q^{1/k}/\sqrt{k} \\), and the
challenge family with bounded \\( \ell_1 \\)-norm and invertible pairwise
differences.

**Sources to fold in**

- Paper §2.1 ("Norms, invertibility, and challenges").
- `crates/akita-challenges/src/` (challenge sampling), `crates/akita-types/src/sis/norm_bound.rs`.

## The operator norm of a ring element

\\( \Gamma(c) = \lVert M_c \rVert_{2\to2} = \max_k |c(\zeta_k)| \\), the largest
singular value of the negacyclic multiplication matrix; it is submultiplicative
and controls Euclidean folding \\( \lVert c\,u \rVert_2 \le \Gamma(c)\lVert u \rVert_2 \\).
Certifying the cap \\( \Gamma(c) \le \Gamma \\) deterministically is its own
topic: see [Operator-norm certification](./operator-norm-certification.md).

**Sources to fold in**

- Paper §2.1 ("Operator norm of a ring element").
- `crates/akita-challenges/src/sampler/op_norm.rs`.
