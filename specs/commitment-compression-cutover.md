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
path introduces compression-only slicing as described below. This slicing does
not change protocol groups, schedule levels, commitment objects, or transcript
messages.

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

## Slice Bound

Each compression chain accepts at most 16 KiB of canonically encoded source
field elements. Larger B or D images are split into independent consecutive
slices. Each slice receives its own per-instance security guarantee and
terminal compressed image.

The bound is in bytes, not field elements or ring elements:

| Profile | Field bytes | Source coefficients per full slice |
| ------- | ----------- | ---------------------------------- |
| q128 | 16 | 1,024 |
| q64 | 8 | 2,048 |
| q32 | 4 | 4,096 |

The decomposition basis and compression ring dimension determine the matrix
width. With the negative-binary basis selected here, a full slice has the
following first-map width:

| Profile | First map ring dimension | First-map width |
| ------- | ------------------------ | --------------- |
| q128 | 16 | 8,192 |
| q64 | 32 | 4,096 |
| q32 | 64 | 2,048 |

The last slice may be shorter. The planner computes its exact width rather than
padding it to 16 KiB. Zero padding exists only inside the final packed ring
element.

The protocol cutover must bind slice boundaries and ordering in the instance
descriptor or another canonical protocol-owned shape. The diagnostic phase
does not add those fields.

## Compression Ladder

The diagnostic path permits at most three maps and targets a 128-byte terminal
image per slice.

| Profile | First map `D` | Later map `D` | Full-slice image sizes |
| ------- | ------------- | ------------- | ---------------------- |
| q128 | 16 | 8 | 512 B -> 256 B -> 128 B |
| q64 | 32 | 16 | 512 B -> 256 B -> 128 B |
| q32 | 64 | 32 | 512 B -> 256 B -> 128 B |

A shorter source may reach 128 bytes in fewer maps. Selection stops at the
first exact 128-byte image. Undershooting the target or failing to reach it
within three maps is an error.

These are small negacyclic module-SIS maps, not unstructured scalar maps. Their
small ring dimensions are intentional and are part of the kernel surface.

## Security Contract

Every compression map instance must independently meet the 128-bit quantum
ADPS16 floor. The coefficient infinity bound is exactly one, matching the
negative-binary kernel.

The compression table is separate from production A/B/D sizing and contains
only the six cells exercised by the ladder:

| Profile | `D` | Rank-1 max width | Rank-2 max width | Required max width |
| ------- | --- | ---------------- | ---------------- | ------------------ |
| q128 | 16 | 7,077 | 8,192 | 8,192 |
| q128 | 8 | 508 | 512 | 512 |
| q64 | 32 | 3,538 | 4,096 | 4,096 |
| q64 | 16 | 254 | 256 | 256 |
| q32 | 64 | 1,769 | 2,048 | 2,048 |
| q32 | 32 | 127 | 128 | 128 |

The planner selects the minimum secure rank for the exact width. Width zero,
width above the required maximum, another coefficient bound, another ring
dimension, or another modulus profile is outside this table and must fail.

Security certification and parameter selection call the same compression SIS
API. The general production SIS surface is not widened for speculative
compression shapes.

## Diagnostic Phase

The diagnostic phase is opt-in through the `compression-diagnostics` feature.
Its planner involvement is deliberately quarantined:

- one standalone module selects slices, dimensions, widths, ranks, and map
  count;
- schedule search, candidate derivation, suffix dynamic programming, generated
  schedule tables, catalog identity, and proof-size scoring are untouched;
- the prover calls the standalone planner only after it has the real B/D
  source lengths.

During proving:

1. Record each live B group image as an independent source.
2. Compute the current D image and absorb it exactly as the current protocol
   requires.
3. Plan at most 16 KiB slices for every source.
4. At each ladder stage, batch slices with identical map shape.
5. Execute the negative-binary compression kernel over the shared setup
   prefix.
6. Retain timing and size metrics, then discard all terminal images.
7. Continue the existing prover unchanged.

The diagnostic must not:

- replace a commitment carried by the proof;
- append a transcript message;
- add a proof or verifier field;
- alter a schedule or descriptor digest;
- add compression witness data to the recursive witness;
- claim that the compressed image is protocol-bound.

An error in diagnostic execution fails the feature-enabled proof rather than
silently reporting success without computing the image.

## Protocol Cutover

After diagnostic measurements are satisfactory, implement the protocol change
as a full cutover from the current raw images.

The cutover must address these surfaces together:

1. Define the public flat compressed payload and canonical slice ordering for
   each B group and the D image.
2. Bind the compression plan, map views, and slice boundaries to protocol
   identity.
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
B image, slice sequence, and compression chain. Batching is an execution
optimization over equal map shapes, not a change to semantic group ownership.

The cutover should use measurements from the diagnostic mode to decide whether
every eligible source should be compressed or whether protocol-owned planning
needs an explicit threshold. That decision belongs in the protocol planner,
not in kernel dispatch.

## Verification

The diagnostic implementation must include:

- schoolbook equivalence tests for q128, q64, and q32 compression kernels;
- rejection of mixed-shape batches, non-negative-binary digits, and
  undersized setup prefixes;
- planner tests for full and partial 16 KiB slicing;
- planner tests that every full slice reaches 128 bytes in at most three maps;
- an end-to-end proof with diagnostics enabled that verifies with the current
  verifier;
- an extension-field or mixed-dimension proof exercising the same shadow path;
- default-feature compilation proving that the diagnostic dependency and hook
  disappear from the normal build.

Protocol-cutover tests will additionally need malformed slice shape, transcript
binding, tampered intermediate image, tampered terminal image, mixed-group
ordering, and cross-protocol rejection coverage.
