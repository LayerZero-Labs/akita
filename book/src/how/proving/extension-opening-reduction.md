# Extension-opening reduction

How Akita wires in suffix and terminal extension-opening reduction (EOR). It turns a base-field
evaluation claim at an extension-field point into a single claim on a packed
polynomial over the extension, with fewer variables. A fold uses this path only
when its scheduled opening method is `EvaluationTrace` and
`CommitmentConfig::EXT_DEGREE > 1`. Single-field presets never run EOR.
Subring coefficient packing also skips EOR because it opens the extension
valued claim directly. See
[Fold path and field geometry](./fold-path.md). The generic reduction and its
field algebra live in
[Foundations → Extension-opening reduction](../../foundations/extension-opening-reduction.md);
this page is about Akita's prover paths, scheduling, and efficiency.

The current prover materializes one dense packed witness table for each claim
and one transparent factor table for each group. Claims in the same group reuse
that factor. A group with fewer variables is extended to the common sumcheck
domain virtually, without repeating either table.

## Multi-group openings

A multi-group evaluation trace fold emits one EOR proof and runs one degree-two
sumcheck. A coefficient packing fold emits neither.
For every evaluation-trace group `g` and claim `i`, the reduction uses the
group's complete public point, native packed witness, and transparent factor:

\\[
A_{\eta,g}(x) W_{g,i}(x).
\\]

The public points are independent; equal, nested, and unrelated values use the
same per-group preparation path.
The reduction embeds all claims in one maximum-arity Boolean domain. After the
partials and their input claims are fixed, the transcript samples an early
coefficient vector. The prover linearly combines the claim polynomials with
those coefficients and sends one degree-two polynomial per round.
If a group has fewer variables, Akita treats its witness as independent of the
additional high variables and multiplies it by equality to a fixed zero point
on those variables.
That equality factor has Boolean sum one.
The prover stores this cylindrical extension as folding state; it does not
allocate repeated witness evaluations.
After sumcheck, the prover and verifier truncate the internal challenge vector
to each group's native tail before preparing that group's resulting relation
point.
This internal shared reduction challenge is not an ambient public opening point.

The proof also carries the unweighted terminal claim for every opening. The
ordinary sumcheck terminal value must equal their early random combination.
The transcript absorbs these terminal claims before the prover builds the
opening payload.

The application uses a second, independent coefficient vector. It samples
these coefficients only after the complete opening payload is absorbed. Stage
2 checks the resulting combination of the terminal claims against the
committed witness relation. The early combination binds the logical EOR input
claims to the terminal vector. The later combination binds that vector to the
committed witness.

## What the prover stores

For a group `g`, let `W_(g,i)` be the packed extension-field witness table for
claim `i`, and let `A_(eta,g)` be the transparent tensor factor derived from the
group's public point and the reduction challenge `eta`. The group contributes

$$
\sum_i \lambda_{g,i}
\sum_x W_{g,i}(x)A_{\eta,g}(x)
$$

to the batched EOR claim. The coefficients `lambda_(g,i)` are sampled after the
input claims and partials have been absorbed.

The implementation stores:

- one dense `W_(g,i)` table for every claim;
- one dense `A_(eta,g)` table shared by all claims in group `g`; and
- one scalar coefficient `lambda_(g,i)` per claim.

The packed witness tables are already smaller than the original base-field
tables: the first `log2([E:F])` Boolean variables have been transposed into one
extension-field value. The implementation does not allocate another factor
table for every claim.

## Put unequal groups in one sumcheck

Suppose the largest group leaves `m` Boolean variables after packing, while a
smaller group leaves only `m-t`. Akita views the smaller witness as constant in
the additional `t` high variables and pins those variables to zero in the
transparent factor. Its virtual contribution is

$$
W_{g,i}(x)
A_{\eta,g}(x)
\prod_{j=0}^{t-1}(1-y_j).
$$

The Boolean sum of the added equality factor is one, so this extension preserves
the group's claim. The prover retains the native witness and factor tables and
tracks only the accumulated value of the extra equality factor as the new
challenges arrive. It does not expand a table of length `2^(m-t)` into one of
length `2^m`.

After the shared sumcheck finishes, each group uses only the prefix of the
challenge vector belonging to its native tail. The remaining challenges account
for its virtual zero-pinned variables.

## Fuse folding with the next round

In one ordinary round, each term needs the constant and quadratic coefficients
of

$$
W_{g,i}(0)+T\bigl(W_{g,i}(1)-W_{g,i}(0)\bigr)
$$

times the analogous affine interpolation of `A_(eta,g)`. After the challenge
`r` is sampled, both tables must be folded at `r` before the next round.

For a group with at least four live entries, the CPU path combines these jobs.
It folds the shared factor and the first witness together, while also computing
the first claim's next-round coefficients. It then folds each remaining witness
against the already-folded factor and computes that claim's next-round
coefficients. Thus the group factor is folded once per round, not once per
claim, and the next scan is avoided.

Some field profiles can accumulate a bounded sum of wide products before
reduction. The prover uses that path only when the field's `SUM_IS_EXACT`
contract proves that the delayed sum is identical to reducing each product.
Other profiles reduce every product immediately. Both paths produce identical
proof coefficients.

## Implementation map

- `crates/akita-prover/src/protocol/extension_opening_reduction/`.
- `crates/akita-prover/src/protocol/core/extension_opening_reduction.rs`.
- `crates/akita-prover/src/protocol/extension_opening_reduction/dense.rs`
  contains the fused fold-and-accumulate kernels.
- `crates/akita-verifier/src/protocol/core/fold/extension_claim.rs`.
- `crates/akita-types/src/extension_opening_reduction.rs`.
- Historical records under `specs/archive/2026-Q3/` document the removed root
  EOR implementations and the surviving suffix machinery's origin.
