# Introduction

Akita is a lattice based polynomial commitment scheme, or PCS, implemented in
Rust. It lets a prover commit to a large table and later prove an evaluation of
the multilinear polynomial represented by that table. The verifier checks the
proof without receiving the whole table. Akita uses transparent setup, so it
needs no trusted ceremony. Its production parameters provide 128-bit post
quantum security under Module-SIS.

The codebase implements a standalone PCS that other proof system frameworks can
use. Its first major integration is
[Jolt](https://jolt.a16zcrypto.com/), a zero knowledge virtual machine, or zkVM,
that proves the correct execution of 64-bit RISC-V programs.

> **Our claim:** Akita is the first production ready lattice based polynomial
> commitment scheme. Earlier systems such as Greyhound showed that lattice
> commitments could be practical, though not without downsides. Akita improves
> on that line of work and adds the implementation, validation, portability,
> and integration work needed for deployment.

## Why build Akita?

Zero knowledge proofs have a quantum problem. Many proof systems deployed today
commit to their data with elliptic curve cryptography. Groth16, Plonk, and
Bulletproofs are prominent examples. Their commitments are compact and well
understood, but a sufficiently powerful quantum computer could break the
assumptions behind them.

Many proof systems split the work into two parts. One part checks polynomial
identities using algebra and random challenges. These checks do not rely on the
difficulty of an elliptic curve problem. The PCS fixes the polynomials and lets
the verifier check them without reading every coefficient. For these systems,
the PCS carries the elliptic curve assumption. Replacing it with a post quantum
PCS can make the whole proof post quantum without replacing the algebraic
protocol around it.

Most post quantum proof systems in use today rely on hash functions. That path
is well developed. It also tends to produce larger proofs and to move more data
through the prover. Akita takes the lattice path. It uses structured lattice
problems to build a compact commitment with no trusted setup.

## Why the commitment layer matters

A proof system often represents a computation as one or more large tables. The
prover has the tables. The verifier wants confidence in a calculation over them
without downloading the tables or repeating the work.

The word *polynomial* describes how the proof system gives those tables useful
algebraic structure. An application starts with a list of field elements. A
field is a number system in which addition, subtraction, multiplication, and
division by a nonzero value have consistent results. Akita treats the list as
the values of a multilinear polynomial.

A table with $2^n$ entries becomes a polynomial with $n$ variables. Each
variable appears with degree at most one. For example, a table with eight
entries becomes a polynomial in three variables because three bits select one
of eight positions. The polynomial also has a value at points that are not made
only of zeroes and ones. That larger domain lets a proof system combine many
table checks into a small number of algebraic claims.

> **Univariate commitments fit the same design.** Akita's folding machinery
> needs tensor structure, not multilinearity itself. The coefficients of a
> univariate polynomial can also be arranged as a tensor and folded one axis at
> a time. The current code exposes only multilinear commitments, but the
> protocol architecture can support a univariate interface as well.

A polynomial commitment scheme gives the two parties a smaller interface:

1. The prover commits to the table. A commitment is a public value that fixes
   the data the prover is claiming to use and is much smaller than the table in
   the intended applications.
2. An evaluation point is selected, often by the larger proof system that uses
   Akita.
3. The prover sends a claimed value and an opening proof.
4. The verifier checks that the claimed value is consistent with the original
   commitment.

The commitment is not the data itself. It fixes the table for later checks. A
valid opening should not let the prover change that table after learning the
evaluation point.

The [multilinear extensions and sumcheck](./foundations/multilinear-sumcheck.md)
chapter develops this representation and the main checking protocol from small
examples.

## The three commitment paths

The choice of commitment scheme changes the security assumptions, setup, proof
size, prover work, and verifier work of the complete proof system.

| Approach | Why systems use it | The post quantum question |
| --- | --- | --- |
| Elliptic curves and pairings | Very small proofs and fast verification | A large quantum computer can solve the underlying discrete logarithm problems. Some schemes also require a trusted setup. |
| Hash functions | Conservative assumptions and transparent setup | Proofs often contain many hash tree paths, which increases proof size and data movement. |
| Structured lattices | Post quantum assumptions with useful algebraic structure | Parameters, norm bounds, and optimized ring arithmetic must all agree exactly. |

Akita chooses lattices for both security and performance. Akita first writes
each field value as small signed digits. It then multiplies those digits by a
public matrix. This structure fits the polynomial arithmetic already present in
the proof system.

This construction also changes the cost of sparse data. A hash based commitment
usually encodes the full table and builds a hash tree over the result. The
prover must process every position, including the zeroes. Zero digits add
nothing to Akita's matrix product, so its sparse commitment path can skip them.
This is especially useful for a one hot table, in which each block contains one
value equal to one and all other values are zero. Jolt's memory checks produce
tables of this form, and Akita provides dedicated one hot configurations for
them.

Large public matrix operations can be streamed. Fast number theoretic
transforms can use wide CPU instructions without changing the proof.

Module lattices already anchor the post quantum standards that the rest of
cryptography is adopting. NIST standardized the module-lattice-based
[ML-KEM](https://csrc.nist.gov/pubs/fips/203/final) for key establishment and
[ML-DSA](https://csrc.nist.gov/pubs/fips/204/final) for digital signatures.
Akita brings module lattice algebra to polynomial commitments. That shared
algebra makes Akita a natural commitment layer for proof systems that need to
prove statements about lattice based encryption, signatures, and other post
quantum protocols. Akita is both a post quantum replacement for curve based
commitments and a foundation for proving the post quantum primitives that
modern cryptographic systems are beginning to use.

Akita follows the line of work from LaBRADOR through Greyhound and Hachi. That
work established compact lattice proofs and practical polynomial openings.
Greyhound and Hachi produced small proofs under Module-SIS, but their verifiers
still did work that grew roughly with the square root of the polynomial size.
Akita makes the fold repeatable. It reduces the opening again and again until
the remaining claim is small enough to check directly. Akita also adds a
complete system that another proof project can operate, profile, upgrade, and
audit.

## Built for production

Akita treats deployment engineering as part of the cryptographic design.
Related public implementations such as
[LaBRADOR](https://github.com/lazer-crypto/labrador),
[LaZer](https://github.com/lazer-crypto/lazer),
[RoKoko](https://github.com/lattice-arguments/rokoko), and
[Jindo](https://github.com/SNUCP/jindo) made important lattice proof techniques
reproducible. Their code is organized around research experiments, benchmark
configurations, or narrow machine targets. Akita takes the next step. It is
built for adoption as infrastructure and is the first implementation in this
line to combine a recursive lattice PCS with the complete system another proof
project needs to integrate, operate, profile, and audit it.

That production system includes:

- The planner searches for proof schedules offline. Normal verification reads
  generated and reviewed schedule tables instead of running a search.
- Each schedule carries the ring dimensions, decomposition ranges, response
  bounds, and SIS parameters needed by its proof.
- The prover and verifier bind the same configuration, setup identity, claim
  layout, and schedule into the transcript before deriving challenges.
- The verifier has its own dependency path and must reject malformed public
  input with a structured error instead of a panic.
- Portable scalar arithmetic remains available beside AVX2, AVX-512, and NEON
  paths. Tests compare the implementations on supported hosts.
- The prover preserves sparse inputs and streams selected matrix operations to
  control memory beyond the polynomial itself.

Current repository profiles provide one concrete result of this work.
Representative dense, one hot, and grouped statements across the supported
field sizes have produced Akita proof payloads of roughly 65 to 80 KB. This is
the Akita commitment proof, not the complete proof produced by a host system.
The measurements come from specific profiles and are not a promise for every
application. The [profiling chapter](./usage/profiling.md) explains how the
repository measures current configurations.

Jolt is the first demanding host for these boundaries. It exercises a verifier
inside another proof system and forces Akita to account for proof bytes,
verification work, memory, serialization, and guest failures as one integration
problem. The same boundaries are meant to serve other hosts.

## How Akita gets a small opening proof

Akita uses lattice based commitments and recursive folding. These choices
affect both its security argument and its implementation.

### Lattice binding

The commitment is built from public matrices over polynomial rings. Security is
tied to the Module-SIS assumption. Informally, this assumption says that it is
hard to find a short, nonzero input that a public matrix maps to zero.

This lattice assumption gives Akita its post quantum design goal. The goal is
based on the absence of a known efficient quantum attack against the selected
problem and parameters. It is not a proof that future attacks cannot improve.

The current parameter tables target at least 128 bits of attack cost under the
specific lattice attack model used by the repository. This is a concrete model
and a set of parameter checks. It is not an unconditional statement that every
use of Akita has 128 bits of security. A deployment must use a supported
configuration, preserve the transcript and verifier checks, and account for the
larger proof system around the commitment scheme.

The [security model](./how/security.md) explains the exact assumptions, norm
bounds, challenge spaces, and generated tables. The [lattices and Module-SIS](./foundations/lattices-sis.md)
chapter introduces the mathematical terms.

### Transparent setup

Akita does not require a trusted ceremony or a secret trapdoor. Its setup
contains public matrices and shape information. The prover may expand and cache
more setup data for speed, but correctness does not depend on somebody keeping
a setup secret.

Transparent does not mean private. The current repository does not implement a
zero knowledge mode. An Akita opening proves consistency with a commitment, but
applications must not assume that the current proof hides every fact about the
witness. The [zero knowledge roadmap](./roadmap/zero-knowledge.md) records that
boundary.

### Recursive folding

A direct check of one large opening relation would still be expensive. Akita
therefore replaces it with a sequence of smaller relations. Each replacement is
called a fold. A schedule records the parameters and opening method for each
level. The prover follows that schedule, and the verifier derives and checks the
same schedule before accepting the proof.

The process ends with a small terminal witness that the verifier can check
directly. The folds do not mean that Akita verifies another proof inside itself.
Here, recursion means that one opening relation is reduced to another relation
of the same general kind until a direct check is practical.

The [architecture overview](./how/architecture.md) follows this full path from
configuration through setup, commitment, proving, and verification.

## The four operations

All of this becomes four public operations.

| Operation | What it receives | What it produces |
| --- | --- | --- |
| Setup | A supported configuration and capacity | Public setup data for commitments and proofs |
| Commit | Setup data and one or more polynomials | A commitment plus private data needed by the prover |
| Prove | A commitment, the original data, opening points, and claimed values | An opening proof |
| Verify | Public setup data, the commitment, opening claims, and the proof | Acceptance or a structured error |

The implementation can batch several polynomials and several committed groups
into one proof. It also chooses a fold schedule from the exact shape of the
claim. Batching and scheduling make the implementation more complex. They do
not change the contract. Commitment fixes the data, proving explains an
evaluation, and verification checks that explanation.

Start with [Quickstart and configuration](./usage/quickstart.md) if you want to
run these operations. Read [The commitment API](./usage/commitment-api.md) for
the ownership and data flow around each call.

## What the verifier relies on

The verifier does more than replay arithmetic. It also checks the shape and
identity of the statement before deriving challenges. In particular, the
current implementation binds the configuration, setup, schedule, claim layout,
and transcript descriptor into the proof session.

Verifier facing bytes may be controlled by an attacker. Code on that path must
return `AkitaError` or `SerializationError` for malformed input. It must not
panic, allocate without a checked bound, or trust a length merely because it
appears in the proof. This is the verifier no panic contract described in
[Verification](./how/verification.md).

For an audit, the important boundaries are:

- setup and schedule validation;
- canonical serialization of fields, rings, proofs, commitments, and claims;
- transcript order and challenge derivation;
- accepted response and norm bounds;
- terminal consistency checks;
- unsafe arithmetic and vector kernels.

The repository records the full review boundary in
`docs/security-posture.md`. Tests show that known cases behave as expected, but
they do not replace an argument for why the protocol checks are sufficient.

## How the repository is divided

Cryptographic boundaries that exist only in prose are easy to violate in code.
Akita therefore uses small crates with one direction of dependency between
them. The split keeps the verifier from pulling in prover only polynomial
backends and keeps offline schedule search out of verification.

At a high level:

- `akita-field` and `akita-algebra` implement fields, rings, transforms, and
  polynomial operations;
- `akita-transcript`, `akita-challenges`, and `akita-sumcheck` implement shared
  proof machinery;
- `akita-types` owns proof, setup, schedule, commitment, and claim shapes;
- `akita-planner` searches for schedules offline, while `akita-schedules`
  stores generated schedules used by normal builds;
- `akita-config` selects concrete policies and resolves generated schedules;
- `akita-setup`, `akita-prover`, and `akita-verifier` own their respective
  execution paths;
- `akita-pcs` provides the end to end `AkitaCommitmentScheme` interface and
  broad public exports.

The [architecture overview](./how/architecture.md) has the complete crate map,
core types, and entry points. Verifier only applications should use
`akita-verifier`, `akita-types`, and `akita-config` directly when they do not
need prover code.

## Choose a reading path

You do not need to read the Book in one fixed order.

| If you are... | Start here | Then read... |
| --- | --- | --- |
| New to proof systems | [Foundations](./foundations/foundations.md) | [How it works](./how/how-it-works.md), beginning with the lifecycle |
| Integrating Akita | [Usage](./usage/usage.md) | [Configuration](./how/configuration.md) and [Troubleshooting](./usage/troubleshooting.md) |
| Integrating only the verifier | [Verifier only integration](./usage/verifier-only.md) | [Verification](./how/verification.md) and the no panic contract |
| Contributing code | [Architecture](./how/architecture.md) | The chapter for the crate or protocol stage you plan to change |
| Reviewing the cryptography | [Polynomial commitments and binding](./foundations/pcs-and-binding.md) | [Security model](./how/security.md), [Transcript](./how/transcript.md), and [Verification](./how/verification.md) |
| Auditing the implementation | [Architecture](./how/architecture.md) | Follow each protocol claim to its validation boundary, source path, and tests |
| Studying performance | [Profiling](./usage/profiling.md) | [Optimizations](./how/optimizations.md), [NTT and CRT](./foundations/ntt-crt.md), and the relevant backend code |
| Maintaining documentation | `docs/documentation.md` | [Spec index](./foundations/spec-index.md) and the source paths in each completed chapter |

Some chapters are complete narratives and others are still short guides. Each
landing page states what is ready, what is still being developed, and which
current source owns the behavior in the meantime.

## How to read implementation claims

The Book explains the system, but the running code is the authority for current
behavior. Live specifications record accepted designs that have not yet been
fully folded into narrative chapters. Tests and generated tables provide
evidence that specific cases and parameter identities are preserved.

When those sources disagree, treat the mismatch as a documentation or
implementation defect. Do not silently combine parts from different versions.
The [spec index](./foundations/spec-index.md) explains which records are live and
which are retained only as history.

The rest of the Book follows that rule. It first explains why a mechanism
exists, then introduces the terms and examples needed to understand it, and
finally points to the code and checks that enforce the current design.
