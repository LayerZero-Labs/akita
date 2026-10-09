# Standalone clear binary opening

Status: active
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Scope

The two standalone crates `akita-labinius-verifier` and
`akita-labinius-prover` implement a clear binary opening differential oracle.
Their `labinius` feature is off by default. Enabling it exposes this separate
API; nothing on the Akita proof path selects or depends on the extension.

This oracle performs no SIS admission, width-table lookup, or security lookup.
It admits no production parameter set. It has no outer commitment, setup
offloading, quotient or carry rows, bit dropping, recursion, performance
contract, or zero knowledge. Its proof encoding is not the production Akita
proof format. The field-switch projection and composition obligations recorded
in [the field-switch specification](labinius-field-switch.md) remain open for
production integration. Arithmetic and protocol agreement in this oracle do
not discharge them.

## Statement and geometry

Let `H: SwitchField` be one of the sealed host profiles:

| Host profile | Source word | Live partial rows | Batching coordinates |
| --- | --- | --- | --- |
| `BinaryField128` | `u128` | 128 | 7 |
| `BinaryField192` | `u64` | 192 | 8 |

For a source table `w` of `N = 2^n` words, the claim is
`sum_j eq_H(r,j) * H::embed_source(w_j) = t`, with `r` in `H^n`.
The first point coordinate selects the table index's least-significant bit.
The map `phi(w_j) = embed_source::<H>(w_j)` copies the source bits into F162's
low polynomial coordinates. It is binary-linear, not a field homomorphism.

The odd coefficient field is `Prime64Offset23703` or `Prime128OffsetA7F7`.
Write `R = TrinomialRing<F,D,M>`, where `D = 162*k`, `k` is 1, 2, or 4.
`k=1` uses `PlusTrinomial`; the larger degrees use `MinusTrinomial`.
The scalar ring is always `Z[Z]/(Z^162 + Z^81 + 1)`. Its reduction modulo two
is exactly F162. Scalar packing and embedding use the existing
`pack_scalar_components` and `embed_scalar` contracts.

There are `M_rows = k*m` scalar rows and `C` columns. Both `m` and `C` are
powers of two. Address word `(row,col)` as `row + M_rows*col`.
Packed element `e` in a source column contains scalar components
`row = e*k + comp`. The first `log2(M_rows)` binary evaluation coordinates
select rows; the remaining `log2(C)` select columns. Rows and columns need
not have equal size.

The caller supplies an explicit row-major matrix `A` with `n_A*m` elements
of `R`. There is no seed expander. The commitment stores
`Y_col = A * W_col` for every packed source column. Images are ordered column
first, then matrix row: `images[col*n_A+a]`.

## Setup admission

`BinaryClearSetup::new` returns an error instead of clamping parameters.
It requires:

1. Nonzero `n_A`, power-of-two geometry, supported degree, matching packing
   rank and modulus sign, exactly `n_A*m` matrix entries, and a coefficient
   prime matching the field modulus. All allocation and wire extents use
   checked size arithmetic. The trinomial NTT domain must be constructible.
2. `lower <= 0 <= upper` for the accepted `i64` response interval.
3. A `BinaryChallengeProfile` over `BinaryScalarRing::Cyclotomic243` that
   satisfies `profile.meets_budget(C, lambda_fold)`.
4. Successful `checked_source_comparison_class_bound` on a singleton
   occurrence from `SourceOccurrenceBound::binary_extracted`, using the
   profile multiplication bound and the entire accepted diameter
   `Delta = upper-lower`. This includes the diagonal comparison and requires
   the resulting bound to be strictly smaller than the coefficient prime.

The source comparison identity binds the coefficient prime, ring degree, and
matrix-view digest. These are geometry, entropy, and no-wrap checks, not SIS
security admission.

The matrix-view digest binds the length-prefixed version tag
`akita/labinius/clear-matrix-view/v1`, `n_A`, `m`, `D` as u64 little-endian,
the one-byte signed middle coefficient, and every canonical coefficient of
`A` in row-major order. The exported `field_digest` hashes the coefficient
list. A fresh backend channel frames the header as its session and that
32-byte digest as its instance, then squeezes 32 bytes. This construction is
independent of the opening channel and binds both components unambiguously.
It uses the selected transcript backend, so setup digests are backend-specific.

## Transcript and byte encoding

All protocol code uses `ClearChannel`. Only `channel.rs` uses
`akita-transcript`, `ProverFoldDraw`, or `VerifierFoldDraw`. A channel public
operation absorbs bytes; a message operation emits or reads an exact-length
buffer. A challenge operation squeezes a uniform 32-byte block. Fold sampling
uses the existing sampler adapters with level and group both zero.
Diagnostic context labels do not provide cryptographic domain separation.

A uniform F162 challenge consists of the first 21 bytes of a block, with the
six high bits of byte 20 cleared, then `BinaryField162::from_bytes`.
Every transmitted F162 value is exactly 21 little-endian bytes; nonzero unused
high bits reject. Source words are 16 bytes for `u128` or 8 bytes for `u64`.
Host elements use their low-coordinate-first 64-bit words: 16 bytes for F128
and 24 bytes for F192. Prime coefficients use their canonical fixed-width
little-endian field encoding. Counts and message lengths come from setup;
there are no proof-supplied lengths or shape headers.

The prover and verifier execute this order:

0. Absorb the domain `akita/labinius/clear-binary-opening/v1` with a u64
   little-endian byte-length prefix. Absorb the setup identity in this order:
   length-prefixed host tag (`f128-source-u128` or `f192-source-u64`), scalar
   degree, `D`, modulus sign, length-prefixed prime label, `k`, `m`, `C`,
   `n_A`, `lower`, `upper`, `lambda_fold`, length-prefixed
   `profile.identity_bytes()`, and the 32-byte matrix-view digest. Degrees and
   dimensions are u64, interval endpoints i64, budget u32, and sign i8, all
   little-endian. Absorb every commitment coefficient in image order, the
   host point, and the host claim. All public bytes precede any challenge.
1. Send exactly `H::ROWS` source-valued partials. Reconstruct F192's remaining
   64 rows as zeros before `SwitchPartials::try_from_values`. Reject unless
   the reconstructed host evaluation equals `t`.
2. Draw `H::BATCH_BITS` F162 coordinates `s`. Start with
   `h = partials.batch(s)`.
3. Run `n` product-sumcheck rounds for
   `h = sum_j phi(w_j) * batched_weights(eq_H(r,.),s)_j`.
   Each message sends `(c0,c2)`, two F162 values. The verifier sets
   `c1 = claim+c2`, draws `z_i`, and updates
   `claim = c0+c1*z_i+c2*z_i^2`. This is characteristic-two arithmetic;
   there is no odd-field interpolation. The prover uses
   `PackedBinary162::refill_binary_words`, `round_product`, and
   `fold_in_place`. The current-claim kernel argument is a prover hint,
   not a verifier check.
4. Send the F162 terminal source evaluation `t_prime = phi(w)(z)`. Require
   `claim = t_prime * transparent_weight(r,z,s)`. There is no division and
   no special-case acceptance at a zero transparent weight.
5. Send `C` F162 values `U_col = sum_row eq(z_row,row)*phi(w_row,col)`.
   Require `sum_col eq(z_col,col)*U_col = t_prime`.
6. After binding the entire `U`, draw `C` binary fold challenges using label
   `akita/labinius/clear-fold/v1` and the admitted profile.
7. Send each coefficient of each scalar response row in row-major order.
   The response is `v_row = sum_col c_col * W_row,col` over signed integers
   in the scalar trinomial ring. Convolution and reduction use checked i128
   accumulators; final coefficients must fit i64 and the accepted interval.
   Reduction uses `Z^162 = -Z^81-1` before any prime embedding or parity read.
   The prover returns an error, without retry, if a coefficient leaves the
   interval.

For step 7, encode `v-lower` in
`max(1, ceil(bit_length(upper-lower)/8))` little-endian bytes. Decode exactly
that width and reject offsets above the diameter. Both interval endpoints
have canonical encodings; unused offsets and extra bytes reject.

## Direct endpoint checks and completion

All three checks use the same decoded integer response:

- **Prime:** map signed coefficients into `F`, pack every contiguous group of
  `k` scalar rows, and require
  `A*pack(v) = sum_col embed_scalar(c_col)*Y_col` in every matrix row.
  Products use `TrinomialNttDomain`.
- **Binary:** take each coefficient's `rem_euclid(2)` parity, including
  negative odd coefficients, and require
  `sum_row eq(z_row,row)*(v_row mod 2) = sum_col U_col*(c_col mod 2)` in F162.
- **Range:** the canonical offset decoder enforces the accepted interval.

The challenge's parity image decodes
`BinaryChallenge::canonical_support_encoding(profile)` directly as F162.
That API validates profile weight, ordered support, positions, and deterministic
signs and returns the same 21-byte little-endian coordinate bitset. The parity
conversion accepts only scalar degree 162; other degrees return `InvalidInput`.
Endpoint verification maps invalid challenge encodings to `InvalidProof`.
The local arbitrary-dimension F162 equality expansion duplicates the algebra
crate's private fixed-batch helper until a checked public expansion is exposed
under tracking issue #45.

A zero transparent weight leaves every source-opening check mandatory.
`prove_binary_clear` and `verify_binary_clear` operate on an existing channel
without finishing it. The byte wrappers create the standard backend channels;
the verifier calls `check_eof`, so truncation and trailing bytes reject.
Proof-dependent failures return `AkitaError::InvalidProof`; caller setup or
statement failures use `InvalidSetup`, `InvalidInput`, or dimension errors.

## Reusable frontend boundary

Steps 1-4 live in `frontend.rs` and return
`BinaryEvaluationClaim { point: z, value: t_prime }`. This is an obligation to
open the binary source against its original commitment, not an accepted
opening by itself. The later mixed-root integration can consume this boundary
without changing the host-field reduction. The standalone crates supply only
the clear source-opening consumer.

## Validation

The prover crate contains small deterministic integration fixtures for both
host profiles, both primes, and scalar and packed trinomial degrees, including
asymmetric row/column geometry. Independent oracles cover the matrix action,
field-switch partials, every sumcheck round, transparent weights, left
expansion, integer folding, parity projection, and scalar packing action.
Malformed encoding, statement and setup identity changes, proof-message
tampering, range failures, and a forced zero transparent weight are tested.
Both transcript backends run the same protocol suite. The verifier crate also
checks canonical interval encodings at their exact endpoints.

Direct endpoint regressions isolate the binary comparison while keeping range,
prime action, and left expansion valid for both scalar and packed geometry.
Packed response mutations also isolate the prime check while preserving parity
and range. Constructor tests exercise the adjacent no-wrap boundary intervals;
roundtrips include bounded weight 46 with the degree-648 P128 coefficient field,
and matrix-action oracles include degree 324.
