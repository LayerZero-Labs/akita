# Roadmap

> **Status:** current status index. Detailed reader guidance belongs in the
> reader-path follow-up.

This section tracks capabilities that are not part of the current production
implementation. [Compute backends](./compute-backends.md) tracks the active
Metal work. [Zero knowledge](./zero-knowledge.md) states the current privacy
boundary and the requirements for any future implementation.

Implemented work belongs under [How it works](../how/how-it-works.md). See
[Setup offloading](../how/setup-offloading.md) for the recursive setup path that
has moved out of this roadmap.

## Streaming prover

PR [#398](https://github.com/LayerZero-Labs/akita/pull/398) is implementing
streamed inputs for suffix extension-opening reduction. Root folds use subring
coefficient packing and are outside that work. The archived design below is
historical and does not define the active implementation.

**Sources to fold in**

- Historical records:
  `specs/archive/2026-Q3/eor-streamed-prover.md` and
  `specs/archive/2026-Q3/eor-sumcheck-prover-acceleration.md`.

## Recursion in production

Open follow-ups for running the verifier inside Jolt at scale: cycle-count
results, remaining glue work, and the prerequisites tracked in the recursion
sub-workspace.

**Sources to fold in**

- `profile/akita-recursion/README.md` (open follow-ups, cycle results).
