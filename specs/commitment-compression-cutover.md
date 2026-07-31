# Spec: Commitment Compression Cutover

Status: diagnostic implementation; protocol cutover proposed.

## Decision

Akita will evaluate commitment compression in two distinct phases.

The first phase computes real compressed commitments while proving and then
discards them. The prover still sends the current B-side and D-side commitment
images, constructs the current proof, and uses the current transcript. The
verifier and all protocol-facing types remain unchanged.

The second phase is a full protocol cutover. It will replace the public raw
commitment images with their compressed images and add the relations needed to
bind those images. It will not preserve both protocols behind compatibility
wrappers.

This order lets us validate the kernels, setup geometry, batching, security
parameters, and cost on real proof data before changing the wire protocol.

## Current Protocol Images

The compression inputs are the public images already produced by the current
protocol:

- A B-side image is one group commitment. Multi-group openings retain one
  compression source per group.
- The D-side image is the opening commitment `v = D * e_hat` when the current
  level has a D block.

The diagnostic path operates on the flat field coefficients of these images.
It does not reinterpret mixed-dimension groups as one common ring vector.
Groups may have different protocol B dimensions; compression starts after each
group's B image has been computed.

Main does not currently slice B or D images for compression. The diagnostic
path does not introduce slicing semantics: it compresses each complete image
once and rejects images above its explicit maximum.

## Compression Map

For a source image `y`, decompose each canonical field coefficient into
negative-binary digits:

```text
y_j = -sum_k bit_k(q - y_j) * 2^k mod q
bit_k(q - y_j) in {0, 1}
```

The resulting matrix input has coefficients in `{-1, 0}`. Digits are ordered
bit-major across the source coefficients and then packed into compression ring
elements.

One compression map is:

```text
y_(i+1) = F_i * negbin(y_i)   // B-side source
y_(i+1) = H_i * negbin(y_i)   // D-side source
```

`F_i` and `H_i` name distinct logical uses. Equal-shape maps may read the same
physical universal-setup prefix during the diagnostic phase because no
protocol identity is assigned to them yet. The protocol cutover must decide
and bind physical view reuse explicitly.

The CPU kernel batches right-hand sides with the same:

```text
(field profile, ring dimension, input width, output rank)
```

Such a batch scans one shared matrix prefix. Different shapes remain separate
batches. This supports mixed-dimension and multi-group openings without forcing
their B images into one common shape.

## Input Bound And Deferred Slicing

The ordinary case is an approximately 1 KiB B or D image. The diagnostic
accepts one complete image of at most 16 KiB of canonically encoded field
elements. Sixteen KiB is an extreme maximum, not a standard slice size. Larger
images are rejected rather than divided automatically.

The bound is in bytes, not field elements or ring elements:

| Profile | Field bytes | Source coefficients at the maximum |
| ------- | ----------- | ---------------------------------- |
| q128 | 16 | 1,024 |
| q64 | 8 | 2,048 |
| q32 | 4 | 4,096 |

The decomposition basis and compression ring dimension determine the matrix
width. The ordinary 1--8 KiB ladder starts at the standard dimension. The
16 KiB maximum doubles only its first dimension so that the first image remains
rank one:

| Profile | Standard first `D` | 16 KiB first `D` | 16 KiB first width |
| ------- | ------------------ | ---------------- | ------------------- |
| q128 | 16 | 32 | 4,096 |
| q64 | 32 | 64 | 2,048 |
| q32 | 64 | 128 | 1,024 |

Any future slicing policy belongs to the protocol planner. It should be
considered only when a B or D matrix is longer than the corresponding A matrix,
and only at the first few folds where this imbalance matters. Candidate sizes
should grow through the relevant power-of-two cases, normally 1, 2, 4, and
8 KiB, with 16 KiB as the hard ceiling. The protocol cutover must bind any
selected slice boundaries and ordering. The diagnostic phase does none of this.

## Compression Ladder

The diagnostic path permits at most three maps and targets one 128-byte
terminal image.

| Input size | First image | Terminal image | Maps |
| ---------- | ----------- | -------------- | ---- |
| At most 8 KiB | 256 B | 128 B | 2 |
| Over 8 KiB through 16 KiB | 512 B | 128 B via 256 B | 3 |

The ring dimensions are profile-specific and halve at each map:

| Profile | At-most-8-KiB ladder | Over-8-KiB ladder |
| ------- | --------------------- | -------------------- |
| q128 | 16, 8 | 32, 16, 8 |
| q64 | 32, 16 | 64, 32, 16 |
| q32 | 64, 32 | 128, 64, 32 |

The 1 KiB standard case is therefore `1 KiB -> 256 B -> 128 B`. Each stage
decomposes its complete input into negative-binary digits before applying the
next F/H matrix. Selection stops at the first exact 128-byte image.
Undershooting the target or failing to reach it within three maps is an error.

These are small negacyclic module-SIS maps, not unstructured scalar maps. Their
small ring dimensions are intentional and are part of the kernel surface. All
reachable maps have output rank one.

## Security Contract

Every compression map instance must independently meet the 128-bit quantum
ADPS16 floor. The coefficient infinity bound is exactly one, matching the
negative-binary kernel.

The compression table is separate from production A/B/D sizing and contains
only the nine rank-one cells exercised by the ladder:

| Profile | `D` | Certified rank-1 width | Required max width |
| ------- | --- | ---------------------- | ------------------ |
| q128 | 32 | at least 4,096 | 4,096 |
| q128 | 16 | 7,077 | 4,096 |
| q128 | 8 | 508 | 256 |
| q64 | 64 | at least 2,048 | 2,048 |
| q64 | 32 | 3,538 | 2,048 |
| q64 | 16 | 254 | 128 |
| q32 | 128 | at least 1,024 | 1,024 |
| q32 | 64 | 1,769 | 1,024 |
| q32 | 32 | 127 | 64 |

The planner selects the minimum secure rank for the exact width. Width zero,
width above the required maximum, another coefficient bound, another ring
dimension, or another modulus profile is outside this table and must fail.

Security certification and parameter selection call the same compression SIS
API. The general production SIS surface is not widened for speculative
compression shapes.

## Diagnostic Phase

The diagnostic phase is opt-in through the `compression-diagnostics` feature.
Its parameter selection is diagnostics-local:

- one private prover module validates the input bound and selects dimensions,
  widths, ranks, and map count;
- one feature-gated backend extension owns diagnostic execution and cache
  metrics, so ordinary digit-row backends and prepared setups remain unchanged;
- schedule search, candidate derivation, suffix dynamic programming, generated
  schedule tables, catalog identity, and proof-size scoring are untouched;
- the selector runs only after the prover has the real B/D source lengths.

During proving:

1. Record each live B group image as an independent source.
2. Compute the current D image and absorb it exactly as the current protocol
   requires.
3. Reject a source above 16 KiB rather than slicing it.
4. At each ladder stage, batch complete sources with identical map shape.
5. Execute the negative-binary compression kernel over the shared setup
   prefix.
6. Retain timing and size metrics, then discard all terminal images.
7. Continue the existing prover unchanged.

The aggregate diagnostic report records source and terminal bytes, map and
equal-shape batch counts, total elapsed time, and compression-cache bytes before
and after the shadow computation when the backend exposes them. Each batch
also records its map index, matrix shape, batch size, input/output bytes,
negative-binary digitization time, and kernel time. Kernel time includes a cold
cache build when that exact matrix shape has not been prepared before.

The diagnostic must not:

- replace a commitment carried by the proof;
- append a transcript message;
- add a proof or verifier field;
- alter a schedule or descriptor digest;
- add compression witness data to the recursive witness;
- claim that the compressed image is protocol-bound.

An error in diagnostic execution fails the feature-enabled proof rather than
silently reporting success without computing the image.

The CPU backend prepares one cache slot for each exact compression shape:

```text
(field profile, ring dimension, output rank, input width)
```

The slot covers exactly `output rank * input width` setup ring elements and
contains only the negacyclic transform required by the compression kernel.
Compression slots have a separate namespace from the generic setup cache,
whose entries cover the full envelope and contain both cyclic and negacyclic
transforms. This separation is required even when the two prefixes have equal
length: a negacyclic-only slot must never satisfy a later cyclic lookup.
Concurrent first use of one compression shape is single-flight.

## Protocol Cutover

After diagnostic measurements are satisfactory, implement the protocol change
as a full cutover from the current raw images.

The cutover must address these surfaces together:

1. Define the public flat compressed payload for each B group and the D image.
2. Bind the compression plan and map views to protocol identity. If the planner
   activates slicing under the narrow B/D-versus-A policy, also bind the
   selected boundaries and ordering.
3. Add hidden decomposition and intermediate images to the witness with one
   canonical layout.
4. Enforce raw B/D image consistency, every intermediate decomposition, and
   every terminal F/H image in the existing relation machinery.
5. Absorb only the new compressed public payloads under fixed transcript
   labels.
6. Update prover, verifier, serialization, proof-size accounting, setup
   contribution, planner pricing, and tests in the same cutover.
7. Delete the raw public commitment paths and the diagnostic-only integration
   once the new protocol is live.

Mixed-dimension multi-group openings are in scope. Each group keeps its own raw
B image and compression chain. Batching is an execution optimization over equal
map shapes, not a change to semantic group ownership.

The cutover should use measurements from the diagnostic mode to decide whether
every eligible source should be compressed or whether protocol-owned planning
needs an explicit threshold. That decision belongs in the protocol planner,
not in kernel dispatch.

## Verification

The diagnostic implementation must include:

- schoolbook equivalence tests for q128, q64, and q32 compression kernels;
- rejection of mixed-shape batches, non-negative-binary digits, and
  undersized setup prefixes;
- exact-prefix compression-cache accounting, single-flight first use, and
  isolation from full-envelope both-transform cache slots;
- diagnostic selector tests that 1, 2, 4, and 8 KiB inputs use two maps;
- diagnostic selector tests that 16 KiB uses three maps and larger inputs are
  rejected;
- an end-to-end proof with diagnostics enabled that verifies with the current
  verifier;
- an extension-field or mixed-dimension proof exercising the same shadow path;
- default-feature compilation proving that the diagnostic module and hook
  disappear from the normal build.

Protocol-cutover tests will additionally need malformed planner-owned shape,
transcript binding, tampered intermediate image, tampered terminal image,
mixed-group ordering, and cross-protocol rejection coverage.
