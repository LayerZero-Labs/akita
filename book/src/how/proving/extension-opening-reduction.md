# Extension-opening reduction

Akita uses extension-opening reduction, or EOR, to reduce a base-field
evaluation claim at an extension-field point to a product claim on a packed
polynomial with fewer variables. A fold uses this path only
when its scheduled opening method is `EvaluationTrace` and
`CommitmentConfig::EXT_DEGREE > 1`. Single-field presets never run EOR.
Subring coefficient packing also skips EOR because it opens the extension
valued claim directly. See
[Fold path and field geometry](./fold-path.md). For the two-dimensional
example, row identity, and prover/verifier message flow, start with
[Foundations → Extension-opening reduction](../../foundations/extension-opening-reduction.md).
This page describes Akita's scheduling and multi-group execution.

The implemented prover consumes recursive witness sources through dense-packed
or sparse extension-opening terms, a lazy tensor factor for early rounds, and a
streamed form that keeps small balanced representatives visible to the hot loop.

## Multi-group openings

A multi-group evaluation trace fold emits one EOR proof and runs one degree-two
sumcheck. A coefficient packing fold emits neither.
For every evaluation-trace group $a$ and claim $i$, the reduction uses the
group's complete public point, native packed witness, and transparent factor:

$$
\widetilde A_{\eta,a}(x)\widetilde g_{a,i}(x).
$$

Here $g_{a,i}$ is the packed table called $g$ on the foundations page.
$A_{\eta,a}$ is its group's public coordinate factor. The tilde denotes
multilinear extension of a Boolean table.

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

Let $m_a$ be group $a$'s tail length, $m_{\max}$ the maximum tail length,
and $\rho_a$ the first $m_a$ coordinates of the shared sumcheck point $\rho$.
The proof carries one terminal product for every opening:

$$
h_{a,i}=\theta_a\widetilde g_{a,i}(\rho_a),
\qquad
\theta_a=\widetilde A_{\eta,a}(\rho_a)
\prod_{j=m_a}^{m_{\max}-1}(1-\rho_j).
$$

The extra factors come from equality to zero on the additional variables.
An empty product is one. Each $h_{a,i}$ excludes its early claim-batching
coefficient, but includes $\theta_a$. The sumcheck terminal value must equal
the early random combination of these products.
The transcript absorbs these terminal claims before the prover builds the
opening payload.

The application uses a second, independent coefficient vector. It samples
these coefficients only after the complete opening payload is absorbed. Stage
2 checks the resulting combination of the terminal claims against the
committed witness relation. The early combination binds the logical EOR input
claims to the terminal vector. The later combination binds that vector to the
committed witness.

Akita keeps the products rather than dividing by $\theta_a$. If the later
application coefficients are $\alpha_{a,i}$, the trace target and weights
express the remaining relation

$$
\sum_{a,i}\alpha_{a,i}h_{a,i}
=
\sum_{a,i}\alpha_{a,i}\theta_a\widetilde g_{a,i}(\rho_a).
$$

The verifier computes every $\theta_a$ and multiplies it into that group's
evaluation-trace coefficients. A singleton claim uses coefficient one and
needs no claim-batching draw. On the terminal path, Akita checks the
single-group trace relation directly against the terminal response instead
of running Stage 2.

## Implementation map

- `crates/akita-prover/src/protocol/extension_opening_reduction/`.
- `crates/akita-prover/src/protocol/core/extension_opening_reduction.rs`.
- `crates/akita-verifier/src/protocol/core/fold/extension_claim.rs`.
- `crates/akita-types/src/extension_opening_reduction.rs`.
- Historical records under `specs/archive/2026-Q3/` document the removed root
  EOR implementations and the surviving suffix machinery's origin.
