# NTT, CRT, and fast ring arithmetic

> **Status:** current narrative. This page documents the implemented CRT and NTT paths, including the AVX-512IFMA exact cache.

Akita computes ring products by mapping each ring to several smaller prime
fields. It performs a negacyclic NTT in each field, multiplies matching
evaluations, applies an inverse NTT, and reconstructs the centered result with
the Chinese remainder theorem. The same representation supports matrix
matvecs over balanced signed digits.

## Pseudo-Mersenne fields and Solinas reduction

The protocol prime fields use moduli of the form
\(p = 2^k - c\), where \(c\) is small. Elements use canonical integers in
\([0,p)\) rather than Montgomery form. Addition and subtraction use a small
number of conditional corrections. A two fold Solinas reduction brings a wide
product back into the canonical range.

The field code provides the protocol primes at 32, 64, and 128 bits, together
with smaller field types used by tests and auxiliary paths. The CRT NTT primes
are separate primes. They are chosen for the roots of unity required by the
active ring degree.

**Code:** `crates/akita-field/src/prime/` and
`crates/akita-algebra/src/ntt/tables.rs`.

## Deferred reduction and balanced digits

The commitment matvec first decomposes field coefficients into balanced base
\(2^L\) digits. A digit has the exact interval

\[
[-2^{L-1}, 2^{L-1}-1].
\]

Bases through \(L=8\) fit in `i8`. Bases from \(L=9\) through \(L=16\) fit
in `i16`. The NTT cache selector receives the resulting absolute coefficient
bound directly. It does not infer the bound from a label such as `L10`.

**Code:** `crates/akita-algebra/src/ring/cyclotomic/decomposition.rs` and
`crates/akita-types/src/ntt_cache/`.

## CRT and NTT representation

For a ring of degree \(D\), each CRT prime supplies a primitive root of order
\(2D\). The forward transform multiplies by the negacyclic twist and then
performs a decimation in frequency transform. The inverse performs the inverse
decimation in time stages, applies the inverse scale, and removes the twist.
The forward and inverse order avoids a separate bit reversal.

The ordinary production profiles use 30 bit CRT primes stored in `i32` limbs:

| Field tier | CRT primes | Base representation |
| --- | ---: | --- |
| Q32 | 2 | 2 `i32` residues |
| Q64 | 3 | 3 `i32` residues |
| Q128 | 5 | 5 `i32` residues |

The pointwise products and inverse transforms run independently for each CRT
prime. Garner reconstruction then combines the residues into the protocol
field. The optional prime 12289 is an exactness tail. It supports every
protocol ring degree through `D = 2048` and is added only when the base CRT
product does not meet the requested bound.

## What AVX-512 means in the current implementation

Akita has two different AVX-512 paths. They must not be described as one
backend.

The ordinary i32 NTT path uses the scalar reference implementation or the
runtime selected AVX2 implementation on x86. `AKITA_SCALAR_NTT=1` forces the
scalar path. A width aware AVX-512 i32 transform also exists. It uses 16 i32
lanes on stages with a half length of at least 16, 8 lanes at half length 8,
4 lanes at half length 4, and scalar work for the remaining small stages.
Production runtime dispatch does not select this wide i32 transform. It is
kept for direct architecture tests and benchmark experiments because the
measured AVX2 transform is faster on the target workloads.

The second path is AVX-512IFMA. It is used for exact signed NTT caches,
including selected dense q128 commitments whose digits fit in `i8`. The
selector enables it only when all of the following hold:

- the process has `avx512f`, `avx512dq`, and `avx512ifma`;
- `AKITA_SCALAR_NTT` is not set to `1`;
- the ring degree is `D64` through `D512`; and
- the exact cache request can be represented by the selected IFMA CRT
  product, with an optional exactness tail where that profile supports one.

The IFMA kernels use 512 bit vectors with eight `u64` lanes. The instruction
family uses a 52 bit radix internally. The selected CRT primes are each below
\(2^{50}\), so the stored canonical residues are 50 bit values. They are NTT
primes, not the protocol field moduli:

| Prime | Value | Distance below \(2^{50}\) |
| --- | ---: | ---: |
| \(p_0\) | `1125899906826241` | `16383` |
| \(p_1\) | `1125899906629633` | `212991` |
| \(p_2\) | `1125899905744897` | `1097727` |

The exact cache uses the smallest CRT representation that meets the strict CRT bound:

| Field tier | IFMA residues | IFMA selection rule |
| --- | ---: | --- |
| Q32 | 1 `u64` residue | Use the base residue when it fits. Add 12289 when the base does not fit but the mixed product does. Otherwise use the ordinary Q32 profile. |
| Q64 | 2 `u64` residues | Use the IFMA form only when the two base residues fit. Otherwise use the ordinary Q64 profile, which can add 12289. |
| Q128 | 3 `u64` residues | Use the base residues when they fit. Add 12289 when the mixed product fits. Otherwise use the ordinary Q128 profile. |

For the full signed `i16` bound, the one prime Q32 base is not sufficient at
the eligible degrees, so Q32 exact caches use the mixed tail. The two prime Q64
base supports much larger widths without a tail. The three prime Q128 base has
less capacity relative to its larger field, so sufficiently wide exact requests
use the tail. This is why one AVX-512IFMA host can use different cache
representations for different field and schedule shapes.

This representation is used by `ExactNegacyclic` cache requests. Ordinary
`Negacyclic`, `Cyclic`, and `BothTransforms` requests use the selected i32
profile. The exact selector uses the field modulus, ring degree, matrix width,
and signed RHS bound. It does not select IFMA merely because the host has
AVX-512.

Dense q128 commitments add one further performance rule. If the complete row
exceeds the three-prime IFMA capacity but fits after adding 12289, an eligible
AVX-512IFMA host accumulates the row exactly. This avoids repeatedly
transforming and reconstructing many small chunks. Scalar, AVX2, and NEON
hosts keep the chunked `i8` path, which is faster and uses less prepared-cache
memory on those architectures. The rule depends on CPU capability and the
capacity bound, not on a machine name or a fixed problem size.

The IFMA matrix stores one transformed negacyclic matrix for each selected
50 bit prime. When a tail is needed, the cache stores a shorter prefix of the
same matrix under the 14 bit prime 12289. The matvec transforms the signed
`i16` RHS, accumulates pointwise products, reconstructs with all selected
residues, and returns canonical protocol field elements. The tail does not
change setup bytes, proof bytes, transcript bytes, or setup digests.

**Code:** `crates/akita-algebra/src/ntt/ifma52.rs`,
`crates/akita-algebra/src/ntt/ifma52/x86.rs`,
`crates/akita-types/src/ntt_cache/exact.rs`, and
`crates/akita-algebra/src/ntt/avx/wide512.rs`.

## Accumulation capacity and chunking

Let `q` be the protocol field modulus, `D` the ring degree, `W` the number of
matrix columns accumulated before reconstruction, `B` the maximum absolute
value of a signed RHS coefficient, and `P` the product of the active CRT
primes. Each output coefficient is a signed sum of at most `D` products per
matrix column. The strict condition for centered reconstruction is

```text
2 * W * D * floor(q / 2) * B < P
```

If the base product does not satisfy this condition, exact preparation adds
the 12289 tail when the combined product does satisfy it. If neither product
is sufficient, preparation rejects the request. The same capacity calculation
is used by cache preparation, verifier warming, runtime matvec checks, and
tests.

Large matrix operations can split their work into chunks and reconstruct after
each chunk. This keeps every intermediate result inside the same bound. The
cache key records the ring degree, transform domain, and required prefix. A
stronger exact request can retain a tail prefix for a later weaker request.

On portable, AVX2, and NEON hosts, exact caches use the field selected i32
profile and add the 12289 tail when needed. On eligible AVX-512IFMA hosts, the
selector may use the 50 bit u64 profiles described above. These are different
storage choices for the same centered CRT contract.

**Code:** `crates/akita-algebra/src/ntt/crt.rs`,
`crates/akita-types/src/ntt_cache/`, and
`docs/crt-ntt-capacity-profile.md`.

## Smooth subgroup mixed radix FFT

The protocol field modulus does not provide a large power of two subgroup for
all evaluation sizes. Reed Solomon evaluation and interpolation therefore use
an iterative mixed radix Cooley Tukey FFT over a smooth subgroup. The
implementation includes radix 2, 3, 5, and 7 kernels and evaluates on a coset
when the requested domain requires it.

**Code:** `crates/akita-field/src/fft.rs`.

## Further reading

- `specs/large-digit-ntt-infrastructure.md` records the implemented exact
  signed digit and terminal NTT contract.
- `book/src/usage/profiling.md` documents the NTT matvec benchmarks and their
  cache labels.
- `docs/crt-ntt-capacity-profile.md` records the generated portable i32
  capacity table. It does not replace the host dependent IFMA rules on this
  page.
