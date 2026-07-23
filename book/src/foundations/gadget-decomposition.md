# Gadget decomposition

> **Status:** stub. Part of the initial Akita Book scaffold.

The digit machinery that keeps committed vectors short: the gadget matrix, its
inverse (balanced base-\\( b \\) decomposition), and the asymmetric representable
range. Folds from paper §2.2 (first half) with implementation notes from
appendix B.1.3.

## The gadget matrix

\\( \mathbf{G}_{b,n} = I_n \otimes (1, b, \dots, b^{\delta-1}) \\) reconstructs a
vector from its digits; \\( \mathbf{G}_{b,n}^{-1} \\) decomposes each coefficient
into balanced base-\\( b \\) digits in \\( \mathcal{D}_b = \{-b/2,\dots,b/2-1\} \\).

**Sources to fold in**

- Paper §2.2 `sec:prelim-gadget` (gadget matrix, balanced digits).
- `crates/akita-algebra/src/ring/cyclotomic/decomposition.rs`.

## Balanced digits and representable range

Why balanced digits halve the digit \\( \ell_\infty \\) bound (and the worst-case
\\( \ell_2 \\) mass), the asymmetric range \\( [-M_k, T_k] \\), and the centering
threshold that avoids an extra digit.

The storage width follows the mathematical digit interval. For `L <= 8`,
`[-2^(L-1), 2^(L-1)-1]` uses i8. For `9 <= L <= 16`, it uses signed i16;
in particular base 10 maps to `[-512, 511]` and base 11 maps to
`[-1024, 1023]`. The first basis requiring i16 is `L=9`. This is independent of whether later protocol decompositions
are range checked: the inner `f -> s` source basis is an arithmetic choice, and
its NTT capability is validated from its actual signed bound.

**Sources to fold in**

- Paper §2.2 (the \\( T_k, M_k \\) range; centering threshold \\( T \\)).
- `crates/akita-types/src/sis/decomposition_digits.rs` (`decomp_depths`, `num_digits_*`).

## Commitment vs opening bases

Different protocol components use different bases: commitment depth
\\( \delta_{\mathsf{com}} = \lceil \log_b q \rceil \\) vs opening depth
\\( \delta_{\mathsf{open}} = \lceil \log_{b_1} q \rceil \\); when to keep the base
explicit.

**Sources to fold in**

- Paper §2.2 (commitment vs opening bases).
- `crates/akita-types/src/sis/decomposition_digits.rs`.
