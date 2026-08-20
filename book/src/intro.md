# Introduction

Akita is a polynomial commitment scheme written in Rust. It is designed for
applications that need to prove facts about very large tables of values without
sending the whole table to the verifier. Its main intended integration is the
[Jolt](https://jolt.a16zcrypto.com/) virtual machine.

The word *polynomial* can make the job sound more abstract than it is. An
application starts with a list of field elements. A field is a number system in
which addition, subtraction, multiplication, and division by a nonzero value
have consistent results. Akita treats the list as the values of a multilinear
polynomial, commits to it, and later proves that the polynomial has a claimed
value at a chosen point. The commitment stays the same across many opening
claims.

This chapter explains that process before introducing the implementation terms
used by the rest of the Book.

> **Project status:** Akita is under active development. The repository makes
> no promise that its public interfaces or proof format will remain compatible
> with earlier versions. The current security process uses specifications,
> code review, strict continuous integration, and focused tests. It is not a
> substitute for an independent security audit.

## The problem Akita solves

Suppose a prover has a large table and wants to convince a verifier that a
calculation over that table is correct. Sending the entire table would let the
verifier repeat the calculation, but it would also remove the main performance
benefit of a proof system.

A polynomial commitment scheme gives the two parties a smaller interface:

1. The prover commits to the table. A commitment is a public value that fixes
   the data the prover is claiming to use and is much smaller than the table in
   the intended applications.
2. An evaluation point is selected, often by the larger proof system that uses
   Akita.
3. The prover sends a claimed value and an opening proof.
4. The verifier checks that the claimed value is consistent with the original
   commitment.

The commitment is not the data itself. It acts more like a binding handle for
later checks. A valid opening should not let the prover change the committed
table after learning the point.

Akita supports multilinear polynomials. A table with $2^n$ entries becomes a
polynomial with $n$ variables. Each variable appears with degree at most one.
For example, a table with eight entries becomes a polynomial in three variables
because three bits select one of eight positions. The polynomial is defined at
every field point, not only at bit values. This extension beyond the original
table is what lets a proof system combine many table checks into algebraic
claims.

The [multilinear extensions and sumcheck](./foundations/multilinear-sumcheck.md)
chapter develops this representation and the main checking protocol from small
examples.

## What makes Akita different

Akita uses lattice based commitments and folding. These choices affect both its
security assumptions and the shape of its implementation.

### Lattice based commitments

The commitment is built from public matrices over polynomial rings. Security is
tied to the Module Short Integer Solution assumption, usually shortened to
Module SIS. Informally, this assumption says that it is hard to find a short,
nonzero input that a public matrix maps to zero.

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
bounds, challenge spaces, and generated tables. The [lattices and Module SIS](./foundations/lattices-sis.md)
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

The public story can be organized around four operations.

| Operation | What it receives | What it produces |
| --- | --- | --- |
| Setup | A supported configuration and capacity | Public setup data for commitments and proofs |
| Commit | Setup data and one or more polynomials | A commitment plus private data needed by the prover |
| Prove | A commitment, the original data, opening points, and claimed values | An opening proof |
| Verify | Public setup data, the commitment, opening claims, and the proof | Acceptance or a structured error |

The implementation can batch several polynomials and several committed groups
into one proof. It also chooses a fold schedule from the exact shape of the
claim. Those details matter for integration, but they do not change the basic
contract: commitment fixes the data, proving explains an evaluation, and
verification checks the explanation.

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

Akita uses small crates with one direction of dependency between them. The
split keeps the verifier from pulling in prover only polynomial backends and
keeps offline schedule search out of verification.

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
