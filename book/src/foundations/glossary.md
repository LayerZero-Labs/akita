# Glossary and notation

This page is a quick reference for terms and symbols used throughout the Akita
Book. Individual chapters introduce each idea in more detail.

## Glossary

| Term | Meaning |
|---|---|
| **Polynomial commitment scheme (PCS)** | A protocol that commits to a polynomial and later proves a claimed evaluation without sending the whole polynomial. |
| **Multilinear polynomial** | A polynomial that has degree at most one in each variable. Its values on the Boolean cube determine it everywhere. |
| **Evaluation table** | The values of a multilinear polynomial at all Boolean inputs. A polynomial in $n$ variables has a table of length $2^n$. |
| **Commitment** | A short public value that binds the polynomial for later opening checks. This does not promise hiding; see Akita's [current privacy boundary](../roadmap/zero-knowledge.md). |
| **Opening** | A proof that a committed polynomial evaluates to a claimed value at a specified point. |
| **Opening claim** | The polynomial identity being proved: a commitment, an opening point, and the claimed value. |
| **Commitment group** | One or more related commitments that are opened together under one generated schedule. |
| **Precommitted group** | A commitment produced earlier and carried into a later grouped opening. Its profile is fixed when the commitment is created. |
| **Fold** | One step that replaces a large opening relation with a smaller one while preserving the claim the verifier must check. |
| **Root fold** | The first fold. It starts from the user's polynomial commitments and opening claims. |
| **Recursive fold** | An intermediate fold that consumes the relation produced by the preceding fold. A schedule may contain none. |
| **Terminal fold** | The final step, where the remaining witness is small enough to reveal and check directly. |
| **Generated schedule** | A precomputed choice of fold dimensions, bounds, opening methods, and proof layout for one supported request shape. |
| **Catalog row** | The generated record selected for a particular configuration and group shape. The prover and verifier read the same row. |
| **Proof shape** | The exact number, order, dimensions, and byte layout of every proof component required by a schedule. |
| **Public setup** | Public matrices and prepared data used to commit, prove, and verify. Akita does not require a secret trapdoor. |
| **Transcript** | The ordered record of public inputs and proof messages from which Fiat-Shamir challenges are derived. |
| **Base field** | The field that stores committed coefficients and supports the lattice relation. |
| **Extension field** | A larger field used for opening points, claimed evaluations, and transcript challenges in some configurations. |
| **Cyclotomic ring** | The quotient ring in which Akita's lattice commitments and fast polynomial arithmetic operate. |
| **Dense polynomial** | A polynomial whose table positions may each contain an arbitrary field element. |
| **One-hot polynomial** | A polynomial whose table has at most one `1` in each consecutive chunk of `onehot_k` entries; all other entries are zero. An all-zero chunk is allowed. Akita can use smaller schedules for this structure. |
| **Gadget decomposition** | Writing a field or ring value as a short vector of bounded signed digits in a power-of-two base. |
| **Module-SIS** | The lattice assumption used to bind Akita commitments. Informally, it says that finding a short nonzero relation among the public commitment columns is hard. |
| **Sum-check** | An interactive reduction that turns a sum over a Boolean cube into one polynomial evaluation at a random point. |
| **Extension-opening reduction (EOR)** | The bridge from a base-field polynomial opened at an extension-field point to a packed extension-field polynomial with fewer variables. |
| **Ring switching** | Akita's lattice step that lifts a relation from a quotient ring to an integer polynomial relation with an explicit quotient. It is distinct from extension-opening reduction. |
| **Setup offloading** | Carrying a committed setup prefix into a later fold so that recursive verification does less setup work directly. |

## Protocol terms by topic

The tables below define terms that the protocol chapters use. Each entry
gives one sentence and links to the chapter that explains the term.

### Layout

| Term | Meaning |
|---|---|
| **Block** | A group of consecutive source ring elements that the A matrix commits together. The block size is the number of positions per block, a power of two. See [Setup and commitment](../how/commitment.md). |
| **Live block** | A block that holds real source data. A source of $N$ ring elements with $M$ positions per block has $\lceil N/M\rceil$ live blocks. |
| **Witness chunk** | One of a power-of-two number of contiguous slices of a fold's witness, used on the leading folds of a multi-chunk schedule. The distributed prover gives each chunk to one machine. See [The distributed prover](../how/proving/distributed-prover.md). |
| **Commitment profile** | The layout metadata frozen when a commitment group is created. Every later opening of that commitment must match it. The Book also says "profile" for this. |
| **Owner** | The prover backend instance that executes a fold and holds its handles. See [Architecture overview](../how/architecture.md). |

### One fold

| Term | Meaning |
|---|---|
| **Roles A, B, and D** | The fixed jobs of the three commitment matrices of a fold. A carries the relation witness, B commits the next witness, and D commits the opening digits. The code names the roles `Inner`, `Outer`, and `Opening`. |
| **Inner image** | The vector $\mathbf t_b=\mathbf A\mathbf s_b$ for the digits $\mathbf s_b$ of one block. |
| **Outer image** | The B matrix applied to the digits of the inner images. The Book also calls it the outer commitment. See [Polynomial commitments and binding](./pcs-and-binding.md). |
| **Fold challenge** | A ring element with a few nonzero coefficients of magnitude 1 or 2, drawn from the transcript for one pair of claim and live block. The Book also calls it a sparse challenge. See [Transcript and instance binding](../how/transcript.md). |
| **Fold response $\mathbf z$** | The vector $\mathbf z=\sum_b c_b\mathbf s_b$ that combines the blocks with the fold challenges. A nonterminal fold commits its digits, and the terminal fold sends it in the clear. The Book also calls it the folded response. |
| **Next witness** | The digit witness that a nonterminal fold produces and commits, and that the following fold opens. The Book also calls it the successor witness. See [Recursion and proof shape](../how/recursion.md). |
| **Payload** | The bytes that carry a fold's B or D commitment image in the proof. A payload is raw, which is the image itself, or compressed. See [Raw and compressed payload realizations](../how/proving/akita-fold-realizations.md). |
| **Opening method** | The schedule-selected procedure that opens one committed group: evaluation trace or subring coefficient packing. |
| **Evaluation trace** | The opening method that opens full A-ring partial evaluations with evaluation-trace weights. Over a proper extension field it requires EOR. See [Evaluation trace](../how/verifying/evaluation_trace.md). |
| **Subring coefficient packing** | The opening method that packs coefficients over a challenge subring, from which the fold challenges are drawn. The Book also says "coefficient packing". See [Root fold and ring switching](../how/proving/root-fold-ring-switch.md). |
| **Quotient lift** | The ring-relation mode that turns an equality modulo $X^D+1$ into a polynomial identity with explicit quotient rows. The other mode, reduced evaluation, checks the relation after reduction and has no quotient rows. |
| **Grinding** | A nonce search that the prover runs at a fixed transcript position. Akita has two kinds: a proof-of-work nonce before a priced Fiat-Shamir query, and the fold-response nonce. See [Rejection sampling the fold response](../how/proving/akita-fold.md#rejection-sampling-the-fold-response). |

### Sum-check stages

| Term | Meaning |
|---|---|
| **Stage 1** | The sum-check that proves every digit of the new witness lies in its scheduled range. On an L2 route it also proves the response norm. See [Sumcheck stages](../how/proving/sumcheck-stages.md). |
| **Stage 2** | The fused sum-check that ties the ring relation, the Stage 1 output, and the incoming opening claim to one evaluation of the next witness. See [The Stage 2 fused check](../how/verifying/stage2.md). |
| **Stage 3** | The sum-check for the setup contribution. It runs only when a fold offloads its setup, and it reduces the contribution to one evaluation of the committed setup prefix. See [Setup contribution and Stage 3](../how/verifying/setup_contribution.md). |
| **Range image** | The pointwise value $w(x)\bigl(w(x)+1\bigr)$ of a witness digit $w(x)$. It lowers the degree of the digit range check. |

### Setup

| Term | Meaning |
|---|---|
| **Setup prefix** | A power-of-two prefix of the flat public setup vector that has its own commitment. A later fold opens it in place of a direct setup scan. See [Setup offloading](../how/setup-offloading.md). |
| **`SetupPolicy`** | The verifier argument that says whether direct setup evaluations are checked now or exported to an accumulator. Stage 3 is checked in both cases. See [Verification](../how/verification.md#per-level-replay). |
| **Terminal NTT cache** | The prepared NTT form of the terminal A matrix prefix. A verifier computes it at construction or installs trusted bytes. See [Verifier only integration](../usage/verifier-only.md#install-a-prepared-terminal-matrix). |

### Schedules and admission

| Term | Meaning |
|---|---|
| **Trusted catalog** | A validated schedule catalog bound to one configuration and chunking policy. It is a verifier parameter and never proof input. |
| **Row digest** | The 32-byte identity of one complete expanded schedule row. A proof names its row by this digest. |
| **Admission** | The checks that accept a schedule row or profile before setup access and proof replay. A proof that names a row the verifier does not admit is rejected. See [Verification](../how/verification.md#schedule-and-profile-admission). |
| **Instance descriptor** | The canonical bytes, absorbed after the protocol tag, that bind the algebra, the setup identity, the schedule, the grinding plan, and the call shape of one proof. |
| **Committed source** | The coefficient data that a commitment binds and a fold opens, together with its declared class and bound. See [Security model](../how/security.md#the-accepted-committed-source-space). |

### Security analysis

| Term | Meaning |
|---|---|
| **Operator norm** | The largest factor by which multiplication by a ring element can stretch the Euclidean length of a vector. L2 folds at ring dimension 64 or 128 reject fold challenges above a scheduled threshold. |
| **Selective L2** | The per-fold security route that sizes the A matrix from a Euclidean bound on the response and adds a scheduled norm proof. The default route uses the coefficient $L_\infty$ bound. See [Configuration and planning](../how/configuration.md). |
| **CWSS** | Coordinate-wise special soundness: the extraction property that asks, for each fold coordinate, for another accepting transcript that changes only that coordinate's challenge. See [Polynomial commitments and binding](./pcs-and-binding.md). |
| **Weak opening** | The opening that the extractor obtains for one fold coordinate from two accepting transcripts that differ only in that coordinate's challenge. It divides the difference of the two fold responses by the difference of the two fold challenges. The result satisfies the verified relations, but it need not be the honest digit decomposition and need not be short. See [Polynomial commitments and binding](./pcs-and-binding.md#coordinate-wise-special-soundness). |

## Scalar and grouped catalog rows

A trusted external catalog ships one row for each supported request shape.

- A **scalar row** describes a polynomial group opened without precommitted
  groups. An independent commitment always uses this row, including a
  commitment that will later be supplied as a precommitted group.
- A **grouped row** describes a final group together with an exact ordered
  prefix of precommitted group profiles. The complete batch is opened under
  this row.

Each precommitted descriptor in a grouped row must equal the scalar-row profile
under which that commitment was produced. The external-catalog audit test
`every_grouped_artifact_precommit_has_a_shipped_scalar_producer` enforces this
invariant.

## Notation

| Symbol | Meaning |
|---|---|
| $\mathbb{F}$ | The base field. |
| $\mathbb{E}$ | The extension field used for public challenges and evaluations. It may equal $\mathbb{F}$. |
| $[\mathbb{E}:\mathbb{F}]$ | The extension degree, or the number of base-field coordinates in one extension-field value. |
| $q$ | The prime modulus of the field in the surrounding chapter. |
| $R_q$ | A cyclotomic polynomial ring with coefficients reduced modulo $q$. |
| $n$ | The number of variables in a multilinear polynomial. Its evaluation table has $2^n$ entries. |
| $r$ | A public opening point. |
| $v$ | A claimed polynomial value at the opening point. |
| $b=2^\ell$ | A power-of-two gadget-decomposition base. |
| $\ell$ | The base-two logarithm of the decomposition base, called the basis exponent in code. |
| $\delta$ | The number of signed digits in a gadget decomposition. |
| $d_j$ | The signed digit at position $j$. |
| $\operatorname{eq}(r,x)$ | The multilinear equality polynomial. On Boolean inputs, it is one at $x=r$ and zero elsewhere. |
| $\rho$ | A random point produced by a sum-check transcript. |

Some implementation chapters use short names such as A, B, and D for matrix
roles. Those letters identify protocol roles, not universal mathematical
symbols. The entry "Roles A, B, and D" above gives their jobs, and each
chapter defines them before use.
