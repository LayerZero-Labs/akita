# Roadmap

> **Status:** stub. Part of the initial Akita Book scaffold.

In-flight and planned work. The larger threads get their own pages
([Verifier offloading](./verifier-offloading.md),
[Modulus switching](./modulus-switching.md),
[Zero-knowledge](./zero-knowledge.md),
[Compute backends](./compute-backends.md)); shorter items stay here as sections.
Keep each item honest about what has already landed versus what is still a spec.

## Streaming prover

A future small-space prover for suffix and terminal extension-opening reduction
may stage the prefix-suffix construction instead of materializing the packed
table. Root folds use subring coefficient packing and are outside this work.

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
