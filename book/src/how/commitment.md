# Setup and commitment

Akita commits to a polynomial by arranging its table entries into ring
elements, decomposing those elements into small digits, and applying public
matrices. The result binds the table that the prover will later open. This
chapter follows one polynomial through that computation before describing
matrix reuse and the dense and one-hot implementations.

## Setup

The public setup starts from a seed. Akita deterministically expands the seed
into a vector of base-field elements, then interprets checked ranges of this
vector as matrices of ring elements. There is no secret trapdoor to retain or
discard. Both parties must use the same seed and matrix geometry.

The [instance descriptor](./transcript.md#akitainstancedescriptor) binds the
setup seed identity and selected schedule before protocol replay. Expanded
matrix files and prepared NTT views are caches derived from that public
identity. Loading a cached coefficient matrix checks that it matches the seed.

### Exact public-stream derivation

`AkitaSetupSeed` contains two values: a 32-byte public seed and a versioned
derivation method. The current method is `Shake256PagedV1`. Versioning the
method prevents a change to page size, domain separation, or field sampling
from silently changing the setup identified by an existing seed.

The derivation splits the infinite field stream into pages of 4096 elements.
For page index $i$, it initializes one SHAKE256 stream with the following
length-prefixed fields, in order:

| Label | Value |
| --- | --- |
| `domain` | `akita/commitment/public-field-stream` |
| `derivation` | `shake256-paged-v1` |
| `page_field_elements` | 4096 as little-endian `u64` |
| `seed` | the 32-byte public seed |
| `field` | the protocol-field modulus as 32 big-endian bytes |
| `page` | $i$ as little-endian `u64` |

A length-prefixed field is encoded as the little-endian `u64` length of its
label, the label, the little-endian `u64` length of its value, and the value.
The explicit field modulus prevents equal seed bytes from identifying the same
coefficient stream in two different fields.

The page then calls `Field::random` repeatedly on that SHAKE256 reader. The
production fields use exact rejection sampling, so this is a uniform field
stream rather than a fixed-width integer stream reduced modulo the field
modulus. Pages
may be generated in parallel; concatenating them by page index gives the same
prefix as sequential generation.

### One stream, several matrix views

Ring dimensions do not enter the derivation. Requests for the same number of
field elements return the same prefix whenever the setup seed and field agree.
The schedule determines how to read that prefix as a matrix.

A matrix view with `r` rows and `c` columns contains `r × c` ring elements.
Each ring element holds `D` field coefficients, so the view reads
`r × c × D` coefficients from the start of the stream. It groups every `D`
coefficients into one ring element, then fills the matrix one row at a time.

The matrices $\mathbf A$, $\mathbf B$, and $\mathbf D$ each read from field
index zero. Their prefixes overlap. The setup therefore needs enough storage
for the largest matrix view required by the schedule, not the sum of all views.

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

## From a table to inner commitments

Begin with one polynomial, one chunk, and a common ring
$R=F[X]/(X^D+1)$, where $F$ is the base field and $D$ is the ring
dimension. An element of $R$ stores $D$ field coefficients. Ring
multiplication wraps powers with the rule $X^D=-1$.

The polynomial's table supplies coefficients $f[\ell,p,b]$. Here $\ell$
selects a ring coefficient, $p$ a position within a block, and $b$ a block.
Pack one ring element per position:

$$
F_{p,b}(X)=\sum_{\ell=0}^{D-1}f[\ell,p,b]X^\ell.
\tag{1}
$$

This is a representation of the table. It does not evaluate the application
polynomial at $X$. [Field to ring reduction](./proving/field-ring-reduction.md)
explains how an evaluation of the table becomes a claim about these rings.

The inner commitment needs short inputs. Let $G_a^{\mathrm{in}}$ be the
public recomposition weight for inner digit $a$. Decompose each position
coefficientwise into digit rings $s_{b,p,a}$:

$$
F_{p,b}=\sum_aG_a^{\mathrm{in}}s_{b,p,a}.
\tag{2}
$$

The configured digit range bounds integer representatives of the
coefficients. Small digits keep the subsequent commitment and folding
relations within their certified norm bounds.

Collect the digits of block $b$ into $\mathbf s_b$, in position order
with digit index innermost. For $L$ positions and $k_{\mathrm{in}}$ digits
per position, the public inner matrix has shape
$\mathbf A\in R^{n_A\times Lk_{\mathrm{in}}}$. Its image is

$$
\mathbf t_b=\mathbf A\mathbf s_b\in R^{n_A}.
\tag{3}
$$

For a small algebra example, take $R=\mathbb F_{17}[X]/(X^2+1)$, one
A row $(2+X,\;1-X)$, and a digit block $\mathbf s_b=(1,-1)$.
Equation (3) gives $t_b=(2+X)-(1-X)=1+2X$. A second block
$(0,1)$ gives $1-X$ using the same A row. These tiny parameters only
illustrate the computation; production schedules choose security-sized
matrices.

Every block now has a short vector of A outputs, but the number of such
outputs still grows with the number of blocks. The outer matrix compresses
that collection.

## The outer commitment and public payload

Let $\rho$ index an A output row. Decompose each $t_{b,\rho}$ using
outer weights $G_h^{\mathrm{out}}$, then concatenate the resulting digits
in the outer matrix's column order:

$$
t_{b,\rho}=\sum_hG_h^{\mathrm{out}}\hat t_{b,\rho,h},
\qquad
\mathbf u=\mathbf B\hat{\mathbf t}.
\tag{4}
$$

The hat distinguishes the short digit representation $\hat{\mathbf t}$
from the recomposed inner images $\mathbf t_b$. The two matrix applications
in (3) and (4) form the two-tier Ajtai commitment.

A standalone commitment transmits a compressed payload $p_F$ for
$\mathbf u$. Two rank-one maps form the compression chain. Each map
decomposes its input coefficients into digits in $\{-1,0\}$, packs those
digits into its native ring, and applies its public matrix. The digit weights
are positive powers of two, interpreted in the base field. The first map
produces 256 bytes and the second produces the 128-byte payload $p_F$.
The complete source image must fit within 8 KiB.

The ring dimensions are profile-owned:

| Modulus profile | First map | Second map |
| --- | ---: | ---: |
| q128 | 16 | 8 |
| q64 | 32 | 16 |
| q32 | 64 | 32 |

These smaller rings belong to compression. A, B, and D retain their own
dimensions, each at least 64. Repacking coefficients between these rings is
a specified coefficient map, so it must not be treated as an arbitrary ring
homomorphism. The [physical fold relations](./proving/akita-fold-realizations.md)
explain how the proof checks recomposition and every compression map.

Later recursive commitments can use a schedule-selected raw payload
$\mathbf u$ instead. Standalone commitments and setup-prefix
precommitments remain compressed. [Recursion](./recursion.md) describes where
the raw suffix is admitted.

## What the prover retains

Commitment produces a public committed group and a prover-only hint. The
group associates its payload with the frozen commitment profile. That profile
fixes the geometry needed to interpret the commitment, including the inner
and outer decompositions and B slicing.

The hint retains the semantic inner images $\mathbf t_b$ for each
polynomial. For compressed commitments it also retains the packed compression
digits and, when quotient lifting is used, the compression quotient images.
It does not store a second copy of the public commitment or the complete
outer digit table $\hat{\mathbf t}$. The prover can reconstruct those
digits from the retained inner images.

The hint avoids repeating work during opening. It is not evidence that the
verifier trusts, and it does not replace access to the committed polynomial.
The verifier receives the public commitment and opening proof, then checks
the corresponding relations.

## Where the opening matrix enters

D depends on the opening geometry and is used when proving an evaluation.
For the evaluation-trace opening method, the prover computes a ring partial
$E_b$ for each block, decomposes these partials into
$\hat{\mathbf e}$, and forms

$$
\mathbf v_D=\mathbf D\hat{\mathbf e}.
\tag{5}
$$

The opening proof carries the compressed payload $p_H$, or the raw
$\mathbf v_D$ in an admitted raw fold. Equation (5) binds the opening
digits. A separate scalar relation recovers the requested evaluation from
them, as derived in [the Akita fold](./proving/akita-fold.md).

This is why a standalone commitment profile freezes A and B without requiring
D. The query is not needed to commit. Once the query is known, the opening
schedule selects D and the proof connects its opening digits to the same
source used by A and B.

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
metadata, descriptors, and trusted external catalog identity.

Relevant implementation sources:

- `crates/akita-types/src/commitment_slicing.rs`
- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-types/src/setup_contribution/plan/physical_b.rs`

Public-stream and view sources:

- `crates/akita-types/src/proof/setup.rs`
- `crates/akita-types/src/layout/flat_matrix.rs`
- `crates/akita-types/src/dispatch/mod.rs`

## Dense and one-hot backends

The dense backend decomposes the source coefficients and evaluates the inner
matrix product with CRT and NTT arithmetic. The one-hot backend uses the
source's sparse structure to visit only its nonzero monomial positions. A
one-hot source has at most one nonzero entry, equal to one, in each consecutive
chunk of `onehot_k` entries. An all-zero chunk is allowed.

Both backends use the same checked commitment geometry and sliced B executor.
They differ only in how they produce the inner A image. Prepared setup and NTT
caches remain keyed by the physical matrix, so increasing the logical slice
count does not create extra stored B matrices.

## Code map

- `crates/akita-setup/src/lib.rs` constructs setup and validates cached public
  matrices against their seed. Its tests cover seed and cache mismatches.
- `crates/akita-prover/src/api/commitment.rs` checks standalone commitment
  profiles, computes inner images, and executes sliced outer commitments.
- `crates/akita-types/src/proof/hints.rs` defines the retained prover state and
  tests its canonical serialization and shape checks.
- `crates/akita-prover/src/protocol/ring_switch/commit.rs` commits recursive
  witnesses, selects raw or compressed output, and prepares the A-only terminal
  handoff.

The production layout can use different native ring dimensions and several
polynomials per group. Equations (1) through (5) give the common-ring,
single-polynomial meaning; the checked layout determines the coefficient
conversions and concatenation in those larger cases.
