# Roadmap

> **Status:** stub. Part of the initial Akita Book scaffold.

In-flight and planned work. The larger threads get their own pages
([L2-MSIS cutover](./l2-msis.md), [Modulus switching](./modulus-switching.md),
[Zero-knowledge / Whiteout](./zk-whiteout.md),
[Compute backends](./compute-backends.md)); the shorter ones are recorded here
as sections. Keep each item honest about what has already landed versus what is
still a spec.

## Streaming prover

A small-space prover for the extension-opening reduction (and the broader fold):
the staged prefix-suffix construction that streams base-field slots instead of
materializing the packed table.

**Sources to fold in**

- Paper App B.4.1 (`sec:akita-eor-sumcheck`, "Small-space staged prover"; streaming-Jolt App A).
- `specs/eor-streamed-prover.md` (ready for implementation), `specs/eor-sumcheck-prover-acceleration.md` (in progress).

## Setup-claim offloading

Shrinking the per-level verifier cost by reducing the setup matrix's MLE
contribution to an inner-product claim on a preprocessed prefix commitment, via
an extra reduction sum-check. Partially landed (the product sum-check exists);
the full offloaded verifier is in progress.

**Sources to fold in**

- Paper §4 `sec:verifier-offloading` (the full construction), §4.3 `sec:claim-reduction`.
- `specs/setup-layout-repack.md`, `specs/setup-product-sumcheck.md`, `specs/planner-incidence-generalization.md`.

## Recursion in production

Open follow-ups for running the verifier inside Jolt at scale: cycle-count
results, remaining glue work, and the prerequisites tracked in the recursion
sub-workspace.

**Sources to fold in**

- `profile/akita-recursion/README.md` (open follow-ups, cycle results).

## On-chain verifier

A succinct on-chain verifier path; prerequisites and the gap from the current
recursion target.

**Sources to fold in**

- `profile/akita-recursion/README.md` (recursion prerequisites).
