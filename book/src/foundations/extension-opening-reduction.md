# Extension-opening reduction

> **Status:** stub. Part of the initial Akita Book scaffold.

The reduction that resolves the base-field-polynomial / extension-field-point
mismatch (raised in [rings and fields](./rings-and-fields.md#base-field-coefficients-vs-extension-evaluation-points)):
a sum-check that turns an evaluation claim on a base-field multilinear
\\( f \\) at an extension point into a single claim on a **packed** polynomial
\\( g \\) over the extension with \\( \kappa \\) fewer variables. This is the
generic protocol and its soundness (paper §2.5); Akita's prover paths and
scheduling are in
[How it works → Extension-opening reduction](../how/proving/extension-opening-reduction.md).

> Terminology: Diamond-Posen call this "ring switching" (FRI-Binius); Hashcaster
> uses the Frobenius-orbit form. The book reserves "ring switching" for Hachi's
> lattice fold and uses "extension-opening reduction" for this tensor bridge.

## The packed polynomial

Splitting \\( f \\)'s variables into \\( \kappa \\) head + \\( \ell-\kappa \\)
tail and packing the head into an \\( \mathbb{F}_q \\)-basis
\\( (\beta_y) \\) of \\( \mathbb{F}_{q^{2^\kappa}} \\):
\\( g(X) = \sum_y f(y,X)\,\beta_y \\).

**Sources to fold in**

- Paper §2.5 `sec:prelim-ext-opening` ("Packed polynomial", input/output instances).

## Column and row partials, and the tensor algebra

Why the naive \\( \sum_y \beta_y S_y \\) shortcut is insecure, and how the
verifier works in the tensor algebra
\\( \mathbb{F}_{q^{2^\kappa}} \otimes_{\mathbb{F}_q} \mathbb{F}_{q^{2^\kappa}} \\)
(column partials \\( S_y \\), row partials \\( \mathrm{row}_u \\)) to bind the
\\( S_y \\) to \\( g \\) over \\( \mathbb{F}_q \\).

**Sources to fold in**

- Paper §2.5 ("Column and row partials").
- `crates/akita-types/src/extension_opening_reduction.rs`.

## The reduction sum-check and soundness

Row-batching with \\( \eta \\), the degree-2 sum-check on
\\( A_\eta(w)\,g(w) \\), the transparent factor \\( A_\eta(\rho) \\) from the
tensor-algebra equality, and the soundness bound
\\( \kappa/q^{2^\kappa} + 2(\ell-\kappa)/q^{2^\kappa} \\).

**Sources to fold in**

- Paper §2.5 ("Reduction sum-check", `fig:ext-opening-reduction`, `thm:ext-opening-soundness` / Binius2 Thm 3.5).
- External: FRI-Binius (Diamond-Posen), Hashcaster.
