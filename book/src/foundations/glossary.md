# Glossary and notation

This page is a quick reference for terms and symbols used throughout the Akita
Book. Individual chapters introduce each idea in more detail.

## Glossary

| Term | Meaning |
|---|---|
| **Polynomial commitment scheme (PCS)** | A protocol that commits to a polynomial and later proves a claimed evaluation without sending the whole polynomial. |
| **Multilinear polynomial** | A polynomial that has degree at most one in each variable. Its values on the Boolean cube determine it everywhere. |
| **Evaluation table** | The values of a multilinear polynomial at all Boolean inputs. A polynomial in $n$ variables has a table of length $2^n$. |
| **Commitment** | A short value that fixes the polynomial while hiding the full evaluation table from the verifier. |
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
| **One-hot polynomial** | A polynomial whose table has a single nonzero entry. Akita can use smaller schedules for this structure. |
| **Gadget decomposition** | Writing a field or ring value as a short vector of bounded signed digits in a power-of-two base. |
| **Module-SIS** | The lattice assumption used to bind Akita commitments. Informally, it says that finding a short nonzero relation among the public commitment columns is hard. |
| **Sum-check** | An interactive reduction that turns a sum over a Boolean cube into one polynomial evaluation at a random point. |
| **Extension-opening reduction** | The bridge from a base-field polynomial opened at an extension-field point to a packed extension-field polynomial with fewer variables. |
| **Ring switching** | Akita's lattice step that lifts a relation from a quotient ring to an integer polynomial relation with an explicit quotient. It is distinct from extension-opening reduction. |
| **Setup offloading** | Carrying a committed setup prefix into a later fold so that recursive verification does less setup work directly. |

## Scalar and grouped catalog rows

A trusted external catalog ships one row for each supported request shape.

- A **scalar row** describes a polynomial group opened without precommitted
  groups. An independent commitment always uses this row, including a
  commitment that will later be supplied as a precommitted group.
- A **grouped row** describes a final group together with an exact ordered
  prefix of precommitted group profiles. The complete batch is opened under
  this row.

Each precommitted descriptor in a grouped row must equal the scalar-row profile
under which that commitment was produced. The generated-catalog audit test
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
symbols. Each chapter defines them before use.
