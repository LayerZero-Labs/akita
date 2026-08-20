# Why lattices?

Akita uses structured lattices because they offer more than post quantum
security. Their algebra fits polynomial commitments. Linear operations can
preserve sparse data, while ring structure supports the fast arithmetic needed
by a large prover.

This chapter explains that choice in the wider commitment landscape.

## The commitment carries the assumption

Many proof systems check polynomial identities using algebra and random
challenges. Those checks do not need an elliptic curve hardness assumption. The
PCS fixes the polynomials and later proves their evaluations. In these systems,
the PCS is the part that introduces the computational assumption.

This division gives system designers a focused post quantum upgrade path. They
can replace a curve based PCS with Akita while preserving much of the protocol
that constructs and checks the polynomial identities.

When the PCS is the host system's main computational assumption, replacing it
with Akita changes the security of the complete proof. Any additional
computational assumptions in the host must provide their own post quantum
security.

## Three commitment approaches

The main approaches make different choices about proof size, setup, and prover
work.

### Elliptic curves and pairings

Curve based commitments support very small proofs and fast verification. This
has made them a common choice in deployed systems such as Groth16 and Plonk.
Their binding rests on discrete logarithm assumptions that a large quantum
computer could break. Some pairing based systems also require a trusted setup
ceremony.

### Hash functions

Hash based commitments use a conservative post quantum foundation and need no
trusted setup. A prover usually encodes the complete object and commits to it
with a hash tree. An opening includes authentication paths through that tree.

This design is general, but it moves substantial data. The prover must hash the
dense encoding, and the proof carries the paths needed to authenticate opened
positions. These costs are a major part of the proof size and memory traffic in
many post quantum proof systems.

### Structured lattices

A lattice commitment applies a public linear map to a representation with
small coefficients. Akita performs that map over polynomial rings. The ring
structure turns large matrix products into fast polynomial operations, while
the small coefficient bounds provide the binding statement priced by
Module-SIS.

This gives Akita a transparent commitment with post quantum security and useful
algebra. Every layer agrees on the exact dimensions, decomposition ranges,
challenge distributions, and accepted response bounds. Akita makes those
choices explicit in generated schedules and validates them at the verifier
boundary.

## Sparse data stays sparse

Akita first decomposes a field value into small signed digits. It then
multiplies those digits by a public matrix. A zero digit adds nothing to the
result, so the prover does not need to process it as a nonzero matrix input.

This property is important for a one hot polynomial. Divide its evaluation
table into blocks. Each block contains one value equal to one, and every other
value is zero. A dense encoding stores every zero. A hash tree also hashes every
encoded position. Akita can store the hot positions and perform commitment work
for the nonzero digits.

Jolt memory checking creates tables with this structure. Akita provides one hot
polynomial sources and generated one hot configurations for them. The same
principle applies to other applications whose committed data has a sparse
representation.

Sparse input does not make every later proof operation sparse. Folding combines
values and can produce dense intermediate witnesses. The gain begins at the
commitment boundary and continues wherever the operation can preserve the
sparse representation. The prover code makes each transition explicit instead
of hiding it behind a dense polynomial interface.

## Ring structure supports fast arithmetic

Akita groups coefficients into elements of a polynomial ring. Public matrices
then become matrices of ring elements rather than unstructured scalar matrices.
The implementation uses number theoretic transforms to multiply these ring
elements quickly.

The selected schedule may use different ring dimensions for different matrix
roles and fold levels. Larger rings reduce the number of independent matrix
entries, but they also change transform cost, exact arithmetic bounds, and
available parallelism. The offline planner compares those costs and emits a
concrete schedule. Normal proving and verification consume the generated result
instead of searching again.

The [NTT, CRT, and fast ring arithmetic](../foundations/ntt-crt.md) chapter
explains the arithmetic. [Configuration and planning](../how/configuration.md)
explains how generated schedules select it.

## Part of the post quantum direction

Module lattices already support major post quantum standards. NIST standardized
[ML-KEM](https://csrc.nist.gov/pubs/fips/203/final) for key establishment and
[ML-DSA](https://csrc.nist.gov/pubs/fips/204/final) for signatures. Akita uses
module lattice algebra for a different purpose: binding polynomial
commitments.

The shared algebra creates a natural path for proof systems that need to prove
statements about lattice based encryption, signatures, and related protocols.
The host can represent those computations using the same broad family of ring
and module operations instead of translating them into an unrelated commitment
model.

Akita therefore serves two roles. It can replace a curve based commitment in a
proof system that needs post quantum security. It can also provide the
commitment layer for proving the post quantum primitives that modern
cryptographic systems are beginning to use.

## From practical commitments to repeated folding

Akita follows a line of work that includes LaBRADOR, Greyhound, and Hachi.
These systems established compact lattice proofs and practical polynomial
openings. Greyhound and Hachi produced small Module-SIS based proofs, but their
verifiers still performed work that grew roughly with the square root of the
polynomial size.

Akita makes the reduction repeatable. It replaces one large opening relation
with a smaller relation of the same general form, then folds again. A generated
schedule continues this process until the remaining claim is small enough for
direct verification.

Repeated folding is what turns the lattice commitment into a PCS with a small
verifier. It also creates the need for exact transcript binding, per-level
parameters, setup contribution checks, and terminal verification. The [How it
works](../how/how-it-works.md) section develops those mechanisms.

## Multilinear today, broader tensor structure tomorrow

The current API commits to multilinear polynomials represented by evaluation
tables. A table with $2^n$ entries has one axis for each of its $n$ variables,
so it already has tensor structure.

Akita's folds use that tensor structure. They do not rely on multilinearity as
the only possible interpretation of the data. The coefficients of a univariate
polynomial can also be arranged as a tensor and folded one axis at a time.

The repository does not yet expose a univariate commitment interface. The
protocol architecture can support one without changing the central reason for
using lattices: efficient linear commitments over structured ring data.
