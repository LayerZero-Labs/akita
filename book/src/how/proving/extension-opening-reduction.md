# Extension-opening reduction

How Akita wires in the extension-opening reduction (EOR): it turns a base-field
evaluation claim at an extension-field point into a single claim on a packed
polynomial over the extension, with fewer variables. This path is used only
when `CommitmentConfig::EXT_DEGREE > 1`, meaning that the claim field is a
proper extension of the coefficient field. Single-field presets
(`EXT_DEGREE == 1`, including production fp128) never run EOR; see
[Fold path and field geometry](./fold-path.md). The generic reduction and its
soundness live in
[Foundations → Extension-opening reduction](../../foundations/extension-opening-reduction.md);
this page is about Akita's prover paths, scheduling, and efficiency.

The implemented prover has dense-packed and sparse one-hot paths, a lazy tensor
factor for early rounds, and a streamed form that keeps small balanced
representatives visible to the hot loop.

## Multi-group openings

A multi-group root still emits one EOR proof and runs one degree-two sumcheck.
Group `g` contributes its own complete public point, native packed witness, and
transparent factor:

\\[
\sum_g A_{\eta,g}(x)
  \left(\sum_{i \in g}\gamma_i\,W_{g,i}(x)\right).
\\]

The public points are independent; equal, nested, and unrelated values use the
same per-group preparation path.
The reduction embeds all terms in one maximum-arity Boolean domain and samples
one internal sumcheck challenge vector.
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

**Sources to fold in**

- `crates/akita-prover/src/protocol/extension_opening_reduction/`.
- `crates/akita-prover/src/protocol/core/extension_opening_reduction.rs`.
- `crates/akita-verifier/src/protocol/core/fold/mod.rs`.
- `crates/akita-types/src/extension_opening_reduction.rs`.
- Paper App B.4.1 `sec:akita-eor-sumcheck` (implemented prover paths, prefix-suffix tensor weight, streamed/staged prover).
- `specs/extension-field-opening-batching.md` (trim stale `akita-scheme` refs), `specs/eor-streamed-prover.md` (active).
