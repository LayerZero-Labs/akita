# Extension-opening reduction

How Akita wires in suffix and terminal extension-opening reduction (EOR). It turns a base-field
evaluation claim at an extension-field point into a single claim on a packed
polynomial over the extension, with fewer variables. A fold uses this path only
when its scheduled opening method is `EvaluationTrace` and
`CommitmentConfig::EXT_DEGREE > 1`. Single-field presets never run EOR.
Subring coefficient packing also skips EOR because it opens the extension
valued claim directly. See
[Fold path and field geometry](./fold-path.md). The generic reduction and its
soundness live in
[Foundations → Extension-opening reduction](../../foundations/extension-opening-reduction.md);
this page is about Akita's prover paths, scheduling, and efficiency.

The implemented prover consumes recursive witness sources through dense-packed
or sparse extension-opening terms, a lazy tensor factor for early rounds, and a
streamed form that keeps small balanced representatives visible to the hot loop.

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

## Implementation map

- `crates/akita-prover/src/protocol/extension_opening_reduction/`.
- `crates/akita-prover/src/protocol/core/extension_opening_reduction.rs`.
- `crates/akita-verifier/src/protocol/core/fold/extension_claim.rs`.
- `crates/akita-types/src/extension_opening_reduction.rs`.
- Historical records under `specs/archive/2026-Q3/` document the removed root
  EOR implementations and the surviving suffix machinery's origin.
