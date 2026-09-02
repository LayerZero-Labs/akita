# Setup and commitment

Akita uses one packed public setup for the commitment matrices at every fold
level. A polynomial becomes an Ajtai commitment through an inner A matrix, an
outer B matrix, and a compressed public payload.

## Setup

The shared setup is one vector of field elements. Each level interprets a
prefix of that vector as its A, B, and D matrices. Compression matrices use the
same setup at their own smaller ring dimensions. The setup envelope is the
largest physical matrix requirement across the selected schedule. Generated
schedules and setup-prefix identifiers bind the exact geometry that uses this
vector.

### Exact public-stream derivation

`AkitaSetupSeed` contains two values: a 32-byte public seed and a versioned
derivation method. The current method is `Shake256PagedV1`. Versioning the
method prevents a change to page size, domain separation, or field sampling
from silently changing the setup identified by an existing seed.

The derivation splits the infinite field stream into pages of 4096 elements.
For page index \(i\), it initializes one SHAKE256 stream with the following
length-prefixed fields, in order:

| Label | Value |
| --- | --- |
| `domain` | `akita/commitment/public-field-stream` |
| `derivation` | `shake256-paged-v1` |
| `page_field_elements` | 4096 as little-endian `u64` |
| `seed` | the 32-byte public seed |
| `field` | the protocol-field modulus as 32 big-endian bytes |
| `page` | \(i\) as little-endian `u64` |

A length-prefixed field is encoded as the little-endian `u64` length of its
label, the label, the little-endian `u64` length of its value, and the value.
The explicit field modulus prevents equal seed bytes from identifying the same
coefficient stream in two different fields.

The page then calls `Field::random` repeatedly on that SHAKE256 reader. The
production fields use exact rejection sampling, so this is a uniform field
stream rather than a fixed-width integer stream reduced modulo \(q\). Pages
may be generated in parallel; concatenating them by page index gives the same
prefix as sequential generation.

### One stream, several matrix views

Ring dimensions do not enter the derivation. A request for \(L\) field
elements therefore returns the same prefix under every schedule that uses the
same setup seed and field. A schedule gives a finite prefix a matrix meaning
only when it constructs a view.

A view with \(r\) rows, \(c\) columns, and ring dimension \(D\) reads exactly

\[
r c D
\]

field elements. It groups each consecutive \(D\) coefficients into one ring
element and stores ring elements in row-major order. A, B, and D are
role-local views beginning at field index zero. They overlap rather than
occupying disjoint regions, so the materialized setup capacity is the maximum
role footprint required by the schedule, not the sum of all role footprints.

This prefix sharing does not require the three roles to use the same ring
dimension. For example, two views that each use 4096 field coefficients read
the same random prefix even if one groups it into 64-coefficient rings and the
other into 128-coefficient rings.

The witness column order is defined by the relation layout, not by setup
generation. Akita keeps digits consecutive within each logical value. Exact A,
B, and D column orders, including coefficient packing and sliced B layouts,
are documented in [Advanced relation layouts](./proving/advanced-relation-layouts.md)
and [Opening points and digit-innermost layout](./proving/opening-points-layout.md).

A recursive setup schedule may require commitments to selected power of two
prefixes of this vector. Setup construction materializes exactly those prefix
slots and checks that no required slot is missing. The prover later opens a
slot when a fold defers its setup contribution. See
[Setup offloading](./setup-offloading.md) for the complete lifecycle.

## Ajtai commitment mechanics

The commitment path decomposes each witness block into commit digits `s_hat`.
It computes the inner commitment `t = A * s_hat`, then uses `B` for the outer
relation and `D` for the opening relation. Neither full image is public. Every
B and D image is negative-binary decomposed through exactly two rank-one maps;
the wire carries only the 128-byte terminal payload `p_F` or `p_H`. The source
image is capped at 8 KiB. The binding argument reduces to Module-SIS for the
base relation and for each compression map.

The compression ladder is profile-owned (`q128: 16/8`, `q64: 32/16`, `q32:
64/32`) and is separate from A/B/D matrix dimensions. All A/B/D dimensions are
at least 64; the smaller compression rows stay in the shared witness tail and
cannot change ordinary relation alignment.

## Dyadic B slicing

At absolute commitment levels zero and one, a compressed commitment may reuse
one smaller physical B matrix across `S` consecutive block ranges, where `S`
is 1, 2, 4, or 8. D is never sliced. Raw commitments and deeper levels require
`S = 1`. Setup prefixes are separate frozen precommitments and may use slicing
at any consumer level.

The block ranges are the proportional dyadic ranges

```text
[floor(i * F / S), floor((i + 1) * F / S))
```

for `F` live blocks. Sliced commitments require `S <= F`, so every B slice is
nonempty. Each slice is assembled in the ordinary B column order. Within each
polynomial-major segment, a shorter slice receives its own zero suffix before
the next polynomial segment begins. The prover then applies the same physical
B matrix to every slice.

The resulting B images remain logically separate. If B has `n_B` rows, the
relation has `S * n_B` B rows in slice-major order. Akita stacks the complete
image and runs one canonical two-map F compression chain over it. The full
stack, not one physical image, must fit the unchanged 8 KiB compression-source
limit.

This separates physical and logical cost:

- SIS rank and setup storage use the smaller physical B width.
- Relation rows, compression work, and proof sizing use the complete logical
  stack.
- Direct and recursive setup contribution evaluation combine all logical
  slice weights before scanning the physical B matrix, so each physical entry
  is evaluated once.

The planner checks every admitted slice count. Proof-focused selection keeps
the counts through complete schedule scoring. Setup-focused selection keeps
the smallest count at the exact local setup floor. The selected count belongs
to the commitment group and is frozen in standalone profiles, setup-prefix
metadata, descriptors, and generated catalog identity.

Relevant implementation sources:

- `crates/akita-types/src/commitment_slicing.rs`
- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-types/src/setup_contribution/plan/physical_b.rs`
- `specs/archive/2026-Q3/commitment-slicing.md`

Public-stream and view sources:

- `crates/akita-types/src/proof/setup.rs`
- `crates/akita-types/src/layout/flat_matrix.rs`
- `crates/akita-types/src/dispatch/mod.rs`

## Polynomial backends: dense vs one-hot

When the dense (CRT+NTT digit) mat-vec is used versus the one-hot backend that
iterates only nonzero monomial positions. One-hot at **fp128 D64** is the usual
production choice; **D128** remains a comparison / legacy profile (see
`usage/quickstart.md`).

Both backends use the same checked commitment geometry and sliced B executor.
They differ only in how they produce the inner A image. Prepared setup and NTT
caches remain keyed by the physical matrix, so increasing the logical slice
count does not create extra stored B matrices.
