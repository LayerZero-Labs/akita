# Roadmap

> **Status:** index of remaining implementation and integration work.

This section tracks capabilities that are not part of the current production
implementation. [Compute backends](./compute-backends.md) tracks the active
Metal work. [Zero knowledge](./zero-knowledge.md) states the current privacy
boundary and the requirements for any future implementation.

Implemented work belongs under [How it works](../how/how-it-works.md).
[Setup offloading](../how/setup-offloading.md) explains the recursive setup
path. [Prover optimizations](../how/optimizations.md) covers the current
commitment tiling, matrix streaming, and
[streaming suffix tensor inputs](../how/optimizations.md#streaming-suffix-tensor-inputs).

## Recursion in production

Open follow-ups for running the verifier inside Jolt at scale: cycle-count
results, remaining glue work, and the prerequisites tracked in the recursion
sub-workspace.

**Sources to fold in**

- `profile/akita-recursion/README.md` (open follow-ups, cycle results).
