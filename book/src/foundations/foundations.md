# Foundations

These chapters explain the mathematical and cryptographic ideas that the rest
of the Book uses. They are written for readers who may know Rust but have not
studied polynomial commitments, as well as for cryptographers who want to check
how each abstract rule becomes code.

You do not need to finish every foundation before using Akita. An integrator can
start with [Usage](../usage/usage.md) and return here when a term or design
choice needs more explanation. A contributor or reviewer should follow the
paths below until every claim they are changing has a clear implementation and
validation boundary.

> **Current depth:** this landing page is complete. The rings, NTT and CRT,
> gadget decomposition, and Module-SIS chapters are full implementation
> narratives. The commitment and binding chapter gives a current security
> overview. Several sumcheck chapters are still short guides and are marked as
> such below.

## What these foundations connect

Akita sits at the meeting point of four subjects:

1. **Tables and polynomials.** A table with $2^n$ entries determines one
   multilinear polynomial in $n$ variables that agrees with the table at every
   input made of zeroes and ones. This gives the proof system an algebraic way
   to ask for one value without sending the full table.
2. **Fields and polynomial rings.** A field is a number system with addition,
   subtraction, multiplication, and division by nonzero values. A ring keeps
   addition, subtraction, and multiplication but may not support every
   division. Akita's polynomial rings group many field values into one object
   for lattice commitment operations.
3. **Checks with derived challenges.** Sumcheck turns one large polynomial
   identity into a sequence of smaller claims. A transcript is the ordered
   record of public inputs and proof messages. The Fiat Shamir transformation
   derives the verifier's challenges from that record, so the proof can be
   checked without a live conversation between prover and verifier.
4. **Lattice binding.** Public matrices commit to short ring vectors. Module-SIS
   is the hardness assumption used to argue that a prover cannot open one
   commitment in incompatible ways.

The implementation adds a fifth concern: all of these objects must have exact
sizes, layouts, encodings, and fast arithmetic paths. A mathematically correct
description is not enough for an implementation audit. A reviewer must also
check where parameters enter, where malformed input is rejected, and whether an
optimized kernel computes the same result as its reference path.

## A first path for new readers

If this is your first polynomial commitment scheme, use the following order.
It starts with the object being proved and ends with the security claim.

1. Read [Multilinear extensions and sumcheck](./multilinear-sumcheck.md) to see
   how a table becomes a polynomial and how a large identity is reduced to one
   evaluation.
2. Read [Cyclotomic rings and extension fields](./rings-and-fields.md) to learn
   where Akita stores coefficients and why some evaluation points live in a
   larger field.
3. Read [Gadget decomposition](./gadget-decomposition.md) to see how one large
   coefficient becomes a list of small signed digits. That chapter includes
   concrete examples and the path into the current code.
4. Read [Lattices and Module-SIS](./lattices-sis.md) for the hard problem behind
   the commitment.
5. Read [Polynomial commitments and binding](./pcs-and-binding.md) to connect
   setup, commitment, opening, verification, and the security assumption.
6. Return to [How it works](../how/how-it-works.md) and follow one Akita proof
   from configuration to terminal verification.

[NTT, CRT, and fast ring arithmetic](./ntt-crt.md) can wait until you want to
understand performance or audit the arithmetic kernels. The specialized
sumcheck and opening reduction chapters are easier after the general sumcheck
chapter.

## The chapter map

The order below matches the order in the Book sidebar. The status column
describes the current documentation, not the maturity of the corresponding
code.

| Chapter | The question it answers | Current depth |
| --- | --- | --- |
| [Cyclotomic rings and extension fields](./rings-and-fields.md) | What number systems does Akita use, and how does it move between coefficient and evaluation views? | Full introduction with examples, field roles, ring arithmetic, embeddings, norms, and an implementation review map. |
| [NTT, CRT, and fast ring arithmetic](./ntt-crt.md) | How does Akita multiply and transform ring elements efficiently across its supported prime sizes? | Current implementation narrative, including scalar, AVX2, and AVX 512 paths. |
| [Gadget decomposition](./gadget-decomposition.md) | How does Akita replace a coefficient with bounded signed digits, and how is the exact range enforced? | Full introduction with examples, formulas, layouts, code paths, and an audit checklist. |
| [Lattices and Module-SIS](./lattices-sis.md) | What hard problem supports binding, and why do short vectors and norm bounds matter? | Full introduction from integer lattices through commitment collisions, generated security tables, and verifier enforcement. |
| [Multilinear extensions and sumcheck](./multilinear-sumcheck.md) | How does a table become a polynomial, and how can a verifier check a large sum with little work? | Short guide. It still needs a complete small example and a round by round verifier view. |
| [Equality factored sumcheck](./eq-factored-sumcheck.md) | How does Akita exploit the structure of an equality polynomial while avoiding verifier inversions? | Short implementation guide. It assumes the general sumcheck chapter. |
| [Extension opening reduction](./extension-opening-reduction.md) | How is an opening over an extension field reduced to claims over the base field? | Short protocol guide. It still needs an end to end numerical example and a fuller code map. |
| [Polynomial commitments and binding](./pcs-and-binding.md) | What do setup, commitment, opening, and verification promise, and how is incompatible opening tied to Module-SIS? | Current security overview with implementation anchors. More introductory examples can still be added. |
| [Glossary and notation](./glossary.md) | What do recurring Akita names and symbols mean? | Developing reference. Use it as a lookup aid, not as the first explanation of a concept. |
| [Spec index](./spec-index.md) | Which design records are live, implemented, or retained as history? | Current maintainer index. |
| [References](./references.md) | Where can a reader find stable external background for the ideas used here? | Developing bibliography. |

## Paths for different jobs

The same chapter can be read at different depths. Use the path that matches the
decision you need to make.

### Application integration

An application integrator needs the public contract before the internal proof.
Start with [Quickstart](../usage/quickstart.md), then use these foundations to
answer specific questions:

- multilinear extensions explain the required polynomial shape and opening
  points;
- fields and rings explain which value types belong to a configuration;
- polynomial commitments and binding explain what acceptance means;
- the [zero knowledge roadmap](../roadmap/zero-knowledge.md) explains what the
  current proof does not hide.

### Protocol contribution

A contributor changing a proof stage should read the general foundation first,
then the matching chapter under [How it works](../how/how-it-works.md). For
example, a change to a sumcheck driver should be checked against both the
generic sumcheck rules and the transcript order used by the proving and
verification paths.

The important question is not only where a function is called. It is which
protocol fact the function owns. One canonical function should enforce each
size formula, range, transcript label, or acceptance rule.

### Cryptographic review

A cryptographer can use the Foundations chapters to separate a general
argument from an Akita parameter choice. For each claim, identify:

1. the mathematical statement;
2. the public parameters that instantiate it;
3. the code that validates those parameters;
4. the verifier code that relies on the statement;
5. the tests that cover accepted and rejected cases.

The [security model](../how/security.md) then connects these local checks to the
complete knowledge and binding claims.

### Implementation and performance audit

An implementation auditor should follow both the logical value and its physical
representation. The same ring element may appear as coefficients, transformed
values, packed lanes, or a flat field vector at different points in the code.
The NTT and gadget chapters state where those changes happen and which scalar
path can be used as a reference.

Pay special attention to dispatch by field and ring dimension, integer capacity
before deferred reduction, canonical decoding, checked allocation sizes, and
unsafe vector code. These are places where a correct formula can still become
an incorrect implementation.

## From concepts to crates

The main implementation owners are:

| Concept | Primary code owner |
| --- | --- |
| Prime fields, extension fields, packed values, and FFTs | `akita-field` |
| Cyclotomic rings, NTTs, modules, and gadget decomposition | `akita-algebra` |
| Multilinear polynomial views shared by proof code | `akita-witness` |
| Generic sumcheck proof types and drivers | `akita-sumcheck` |
| Transcript construction and challenge sampling | `akita-transcript` and `akita-challenges` |
| SIS parameters, commitments, proof shapes, and schedules | `akita-types` |
| Concrete configuration policy | `akita-config` |
| Commitment and proof execution | `akita-prover` |
| Verifier replay and rejection | `akita-verifier` |

This table gives ownership, not a substitute for explanation. A completed
chapter should define the concept first, show a small example, state the rule,
and then identify the exact source and tests that enforce it.

## What to remember

The Foundations section has two jobs. It should let a new reader understand the
ideas without opening the source, and it should let a reviewer find the source
without guessing what each function is meant to prove.

When a chapter is still short, treat it as a map rather than a complete lesson.
When a chapter is complete, its prose should agree with current code, live
specifications, generated data, and tests. Any disagreement is a defect to
resolve, not a choice between equally valid descriptions.
