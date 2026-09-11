# Introduction

Akita is a lattice-based polynomial commitment scheme, or PCS, implemented in
Rust. Akita lets a prover commit to a large table and later prove an evaluation
of the multilinear polynomial represented by that table. The verifier checks
the proof without receiving the whole table. Akita uses transparent setup, so
it needs no trusted ceremony. Its production parameters provide 128-bit
post-quantum security under Module-SIS.

Akita's small proofs are one of its strongest advantages. Current production
benchmarks produce opening proofs of roughly 65 to 80 KB. This contrasts with
hash-based post-quantum polynomial commitment schemes, which produce proofs of
hundreds of kilobytes at the same polynomial sizes Akita supports.

The codebase implements a standalone PCS that other proof systems can use. Its
first major integration is [Jolt](https://jolt.a16zcrypto.com/), a
zero-knowledge virtual machine, or zkVM, that proves the correct execution of
64-bit RISC-V programs.

> **Our claim:** Akita is the first production-ready lattice-based polynomial
> commitment scheme. Earlier systems such as Greyhound showed that lattice
> commitments could be practical. Akita improves on that line of work and adds
> the validation, portability, verifier isolation, and integration work needed
> for deployment.

## Why Akita exists

Many proof systems deployed today commit to their data with elliptic curve
cryptography. Groth16, Plonk, and Bulletproofs are prominent examples. Their
commitments are compact and well understood, but a sufficiently powerful
quantum computer could break the assumptions behind them.

The commitment is often the part of the proof system that carries this
assumption. The surrounding protocol checks polynomial identities using
algebra and random challenges. Replacing the PCS with a post-quantum PCS can
therefore protect the complete proof without replacing every other protocol in
the system.

Hash-based commitments provide one post-quantum path. They are well developed,
but they tend to produce larger proofs and require more prover memory. Akita
takes the lattice path. It uses structured lattice problems to obtain
post-quantum security, compact proofs, and a commitment operation that can
preserve sparse inputs.

## What a polynomial commitment proves

A proof system often represents a computation as one or more large tables. The
prover has the tables. The verifier wants confidence in a calculation over them
without downloading the tables or repeating the calculation.

The word *polynomial* describes how the proof system gives a table useful
algebraic structure. A table with $2^n$ entries defines a multilinear
polynomial with $n$ variables. Each variable appears with degree at most one.
For example, eight table entries define a polynomial in three variables because
three bits select one of eight positions.

The polynomial also has values away from those eight table positions. A larger
proof system uses evaluations at such points to combine many table checks into
a small number of claims.

A PCS gives the prover and verifier four operations:

| Operation | What it does |
| --- | --- |
| Setup | Creates the public information needed by commitments and proofs |
| Commit | Fixes one or more polynomials and produces a compact public commitment |
| Prove | Produces evidence for claimed evaluations of the committed polynomials |
| Verify | Checks the claims against the commitment and public setup |

The commitment fixes the data before the evaluation point is known. A valid
opening should not let the prover change that data after seeing the point.

Akita currently exposes multilinear commitments. Its folding protocol needs
tensor structure rather than multilinearity itself, so the same architecture
can support a future univariate interface.

## Why Akita uses lattices

The choice of commitment changes the security, proof size, prover work, and
verifier work of the complete proof system.

| Approach | Main advantage | Main cost |
| --- | --- | --- |
| Elliptic curves and pairings | Very small proofs and fast verification | The security assumptions are not post-quantum, and some schemes require a trusted setup |
| Hash functions | Post-quantum security and transparent setup | Proofs often contain many hash tree paths, which increases proof size and data movement |
| Structured lattices | Post-quantum security with useful algebraic structure | Parameters, norm bounds, and optimized ring arithmetic must agree exactly |

Akita writes each field value as small signed digits and multiplies those digits
by a public matrix. A zero digit contributes nothing to that product. The
prover can therefore keep sparse polynomials sparse instead of first expanding
them into a dense encoding. This is useful for one-hot tables such as those
created by Jolt memory checks.

Module lattices also connect Akita to the direction taken by post-quantum
cryptography more broadly. NIST standardized module lattice systems
[ML-KEM](https://csrc.nist.gov/pubs/fips/203/final) for key establishment and
[ML-DSA](https://csrc.nist.gov/pubs/fips/204/final) for signatures. Akita brings
module lattice algebra to polynomial commitments, which makes it a natural
commitment layer for proof systems that need to reason about post-quantum
protocols.

The [Why lattices?](./introduction/why-lattices.md) chapter develops this
comparison, the sparse advantage, and the protocol lineage in more detail.

## How Akita makes the proof small

Akita combines lattice commitments with repeated folding. A fold replaces one
large claim with a smaller claim that has the same form. The proof records why
the replacement is valid. Akita can then apply the same step again instead of
sending evidence proportional to the original table.

First, the prover commits to small signed digits using public matrices over
polynomial rings. Binding rests on Module-SIS. Informally, Module-SIS says that
it is hard to find a short nonzero input that a public matrix maps to zero.

Second, Akita follows a fold schedule. This is a plan that says how large each
intermediate claim will be and which security limits the verifier will enforce.

The benchmark suite currently tests 13 cases. Each case fixes the arithmetic
field and the shape of the polynomial data. It also fixes how many polynomials
are opened and how the prover handles the public setup calculation. Twelve
cases use seven fold levels. A fold level means one reduction step. The seventh
level is the terminal fold, where the verifier checks the remaining response
directly. One benchmark case divides the work into eight chunks during its
first two steps and uses eight fold levels in total.

For a concrete example, consider the benchmark that opens one dense polynomial
over Akita's 128-bit field. Dense means that every position in the polynomial's
table may contain an arbitrary element of that field. This polynomial has 28
variables, so its table contains $2^{28} = 268,435,456$ field values. The fold
schedule reduces the data that remains to be checked as follows:

| Fold level | Field values entering the level |
| --- | ---: |
| L0 | 268,435,456 |
| L1 | 55,332,544 |
| L2 | 3,892,032 |
| L3 | 1,064,256 |
| L4 | 488,384 |
| L5 | 255,488 |
| L6, terminal | 128,768 |

The first six levels send a new commitment and a few small field values. They
also send short algebraic proofs, called sumchecks, that connect the next claim
to the previous one. Those messages occupy 19,912 bytes in this benchmark case.
L6 performs the terminal check, adds a 4-byte number used when choosing the
folding challenge, and sends the remaining response directly. The complete
fold payload is therefore 19,916 bytes.

That complete proof is 73,018 bytes. Its 53,102-byte terminal response contains
three parts:

| Terminal part | What it contains | Size |
| --- | --- | ---: |
| $z$ | The folded witness response, 16,384 coefficients stored with a compact integer encoding called Golomb coding | 20,334 bytes |
| $e$ | The 512 opening values needed by the final check | 8,192 bytes |
| $t$ | The 1,536 inner commitment values needed by the final check | 24,576 bytes |

Another fold is useful only when the bytes it removes from this remaining
response exceed the bytes needed to prove the extra fold. The planner stops at
the least expensive valid point. Across the latest complete profile run, all
13 production cases passed with proofs from 66,600 to 79,527 bytes. The
[benchmark report guide](./usage/benchmark-reports.md) explains how to inspect
the current totals and the byte cost of every fold.

Finally, the verifier checks the selected schedule, replays the same transcript,
checks every fold, and verifies the clear terminal response before it accepts
the claimed evaluation.

Akita does not need a trusted ceremony or a secret trapdoor. The prover may
cache expanded public setup for speed, but correctness does not depend on
anybody keeping a setup secret. The current PCS proof is not itself zero
knowledge. Applications that require witness privacy must account for that
boundary. The [zero-knowledge roadmap](./roadmap/zero-knowledge.md) describes
the planned privacy layer.

The [How it works](./how/how-it-works.md) section follows the complete protocol
from configuration through setup, commitment, folding, and verification. The
[security model](./how/security.md) gives the exact Module-SIS assumptions and
parameter checks.

## Built as a complete system

Akita treats deployment engineering as part of the cryptographic design. The
repository ships external schedule artifacts, checked security parameters, a
separate verifier package, canonical proof encoding, and portable optimized
arithmetic. The verifier rejects malformed public input with structured errors
instead of panicking. The prover preserves sparse inputs and streams selected
matrix operations in bounded chunks.

Memory is another scaling advantage. An arbitrary dense source still occupies
one field element per table entry, but Akita can take ownership of that buffer
instead of copying the complete table. Its folding schedules replace the
source with progressively smaller intermediate data. For large supported
tables, the prover's working memory beyond the source polynomial grows more
slowly than the polynomial itself. This is in contrast to standard hash-based
PCS provers, which usually materialize an encoded word and a hash tree during
commitment, then retain both to answer later openings. The memory for those
objects grows linearly with the polynomial size.

The 65 to 80 KB figures cover the Akita commitment proof. The host proof system
contributes its own proof data.

Jolt is the first demanding host for these boundaries. It exercises Akita
inside another verifier and accounts for proof bytes, verification work,
memory, serialization, and guest failures as one integration problem. The same
interfaces are designed for other proof systems to adopt.

The [Built for production](./introduction/built-for-production.md) chapter
explains the engineering evidence behind the production-ready claim. The
[Reviewing and auditing Akita](./introduction/reviewing-akita.md) chapter maps
the security boundaries to the code, generated artifacts, and tests that
enforce them.

## Continue through the Book

Choose the next section by what you want to do:

| Goal | Continue with |
| --- | --- |
| Run Akita or integrate the PCS | [Usage](./usage/usage.md) |
| Learn the mathematical ideas from the beginning | [Foundations](./foundations/foundations.md) |
| Follow the protocol and implementation | [How it works](./how/how-it-works.md) |
| Review the assumptions and verifier boundary | [Reviewing and auditing Akita](./introduction/reviewing-akita.md) |
| See planned work rather than current behavior | [Roadmap](./roadmap/roadmap.md) |

The Book explains each mechanism before pointing to its implementation. Current
code is the authority for runtime behavior. Live specifications record accepted
designs that have not yet been fully folded into the Book, while archived
specifications record history rather than current requirements.
