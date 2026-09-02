# Extension-opening reduction

Akita sometimes commits to a polynomial whose coefficients lie in the base
field $\mathbb{F}$, then opens that polynomial at a point whose
coordinates lie in a larger extension field $\mathbb{E}$. Extension-opening reduction bridges
those two field roles. It converts the original opening into an opening of a
smaller polynomial whose values already lie in $\mathbb{E}$.

This chapter gives the protocol idea needed to follow the implementation. For
Akita's scheduling and concrete prover storage, see
[How it works → Extension-opening reduction](../how/proving/extension-opening-reduction.md).

## Packing several Boolean variables into one value

Suppose $\mathbb{E}$ has degree $2^\kappa$ over $\mathbb{F}$. Choose an
$\mathbb{F}$-basis $(\beta_y)_{y\in\{0,1\}^\kappa}$ for $\mathbb{E}$.
If the original multilinear polynomial $f$ has $n$ variables, split its
input into $\kappa$ leading variables $y$ and $n-\kappa$ remaining
variables $x$. Akita defines the packed polynomial

$$
g(x)=\sum_{y\in\{0,1\}^\kappa} f(y,x)\,\beta_y.
$$

The new polynomial $g$ has only $n-\kappa$ variables. Each value of $g$
packs one full block of $2^\kappa$ base-field values into a single
extension-field value. The implementation uses little-endian Boolean order
for this basis packing.

## Column partials and row partials

At the claimed opening point, the verifier needs to connect two views of the
same table:

- The column view fixes the first $\kappa$ Boolean variables and evaluates
  the remaining variables. This produces one base-field value for each basis
  position.
- The row view transposes those column values through the chosen extension
  basis. This produces the extension-field values used in the packed opening.

The row values are not extra witness data chosen independently by the prover.
They are a deterministic basis transpose of the column values. This detail is
what binds the packed polynomial back to the original base-field polynomial.

## The reduction check

Akita checks the connection with a degree-two sum-check. The sum-check reduces
the many table positions to one randomly selected point. At that point, the
verifier checks that the packed witness value and the transparent tensor
factor reproduce the claimed opening. The transcript samples the random
coefficients only after the relevant claims have been absorbed, so the prover
cannot adapt the claims to those coefficients.

Akita runs this reduction only when a scheduled fold uses an evaluation trace
and the extension degree is greater than one. Configurations whose base and
extension fields are the same do not need it. A fold that uses coefficient
packing also opens its extension-valued claim directly and skips this
reduction.

## Implementation map

- `crates/akita-types/src/extension_opening_reduction.rs` contains the shared
  tensor algebra, field split, packing, and transpose operations.
- `crates/akita-prover/src/protocol/extension_opening_reduction/` contains the
  prover implementation.
- `crates/akita-verifier/src/protocol/core/fold/extension_claim.rs` replays the
  resulting claim on the verifier side.
- [Fold path and field geometry](../how/proving/fold-path.md) explains which
  schedules select this path.
