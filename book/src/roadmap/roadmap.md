# Roadmap

> **Status:** current status index. Detailed reader guidance belongs in the
> reader-path follow-up.

This section separates implemented work, active work, and paper-only designs.
[Recursive setup-prefix offloading](./verifier-offloading.md) is implemented.
[Compute backends](./compute-backends.md) tracks the active Metal work.
[Zero-knowledge](./zero-knowledge.md) records a paper design that is not
implemented in the current repository.

## Streaming prover

PR [#398](https://github.com/LayerZero-Labs/akita/pull/398) is implementing
streamed inputs for suffix extension-opening reduction. Root folds use subring
coefficient packing and are outside that work. The archived design below is
historical and does not define the active implementation.

**Sources to fold in**

- Paper App B.4.1 (`sec:akita-eor-sumcheck`, "Small-space staged prover"; streaming-Jolt App A).
- Historical records:
  `specs/archive/2026-Q3/eor-streamed-prover.md` and
  `specs/archive/2026-Q3/eor-sumcheck-prover-acceleration.md`.

## Recursion in production

Open follow-ups for running the verifier inside Jolt at scale: cycle-count
results, remaining glue work, and the prerequisites tracked in the recursion
sub-workspace.

**Sources to fold in**

- `profile/akita-recursion/README.md` (open follow-ups, cycle results).
