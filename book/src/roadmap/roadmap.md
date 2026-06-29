# Roadmap

> **Status:** stub. Part of the initial Akita Book scaffold.

In-flight and planned work. The larger threads get their own pages
([Verifier offloading](./verifier-offloading.md),
[Modulus switching](./modulus-switching.md),
[Zero-knowledge](./zero-knowledge.md),
[Compute backends](./compute-backends.md)); shorter items stay here as sections.
Keep each item honest about what has already landed versus what is still a spec.

## Streaming prover

A small-space prover for the extension-opening reduction (and the broader fold):
the staged prefix-suffix construction that streams base-field slots instead of
materializing the packed table.

**Sources to fold in**

- Paper App B.4.1 (`sec:akita-eor-sumcheck`, "Small-space staged prover"; streaming-Jolt App A).
- `specs/eor-streamed-prover.md` (ready for implementation), `specs/eor-sumcheck-prover-acceleration.md` (in progress).

## Recursion in production

Open follow-ups for running the verifier inside Jolt at scale: cycle-count
results, remaining glue work, and the prerequisites tracked in the recursion
sub-workspace.

**Sources to fold in**

- `profile/akita-recursion/README.md` (open follow-ups, cycle results).
