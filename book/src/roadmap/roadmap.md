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

## Small-space extension-opening prover

The production EOR prover materializes one packed witness table per claim and
one transparent factor table per group. It reuses the group factor and avoids
expanding smaller groups to the maximum arity, but its largest tables are still
linear in the native EOR domain. The [implementation
chapter](../how/proving/extension-opening-reduction.md) documents that path.

The separable representation of the [transparent
factor](../foundations/extension-opening-reduction.md#structure-of-the-transparent-factor)
suggests a different prover organization. Partition an EOR domain of size `N`
into `C` stages. At each stage, fold the small basis-coordinate prefix tables
and accumulate suffix contractions against the current witness source. A
fully streamed form could read the original base-field coordinate tables
directly instead of first retaining the packed extension-field table.

For extension degree `K`, this organization targets

```text
working memory: O(C K N^(1/C))
field work:     O(C K N).
```

It would change only how the prover supplies the existing degree-two sumcheck
terms. The transcript and verifier need not change. No current protocol path
implements this staged construction, and there is no active PR for it. The
closed PR [#398](https://github.com/LayerZero-Labs/akita/pull/398) and the
records in `specs/archive/2026-Q3/eor-streamed-prover.md` and
`specs/archive/2026-Q3/eor-sumcheck-prover-acceleration.md` are historical
experiments, not the production design.

## Recursion in production

Open follow-ups for running the verifier inside Jolt at scale: cycle-count
results, remaining glue work, and the prerequisites tracked in the recursion
sub-workspace.

**Sources to fold in**

- `profile/akita-recursion/README.md` (open follow-ups, cycle results).
