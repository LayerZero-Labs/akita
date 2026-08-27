# Lattices and Module-SIS

Akita's commitments are binding because opening one commitment in two
incompatible ways would reveal a short solution to a Module-SIS problem. This
chapter defines that problem and shows the reduction with a small example.

The chapter explains the general assumption. The [security
model](../how/security.md) explains how Akita prices each concrete matrix and
norm bound.

## From integer vectors to lattices

A lattice is a set of vectors formed by integer combinations of fixed basis
vectors. If the basis vectors are the columns of a matrix $B$, the lattice is

$$
\mathcal{L}(B)=\{Bz:z\in\mathbb{Z}^m\}.
$$

The integer coefficients make this set discrete. A lattice can contain many
vectors, but finding a nonzero vector that is unusually short can still be
hard in a high dimension.

Lattice cryptography builds security arguments around such short vector
problems. Akita uses a structured version over a polynomial ring.

## Ring vectors and modules

Let

$$
R_q=\mathbb{F}_q[X]/(X^D+1).
$$

A vector in $R_q^m$ contains $m$ ring elements. Since each ring element has
$D$ coefficients, the vector also represents $mD$ base field coordinates.

A module is closed under addition and multiplication by elements of the ring.
For this chapter, it is enough to think of $R_q^m$ as the set of length $m$
ring vectors. Matrix multiplication follows the usual row by column rule, but
the entries are ring elements.

## The Module-SIS problem

Module-SIS stands for the Module Short Integer Solution problem. Its public
input is a matrix

$$
M\in R_q^{n\times m}.
$$

The task is to find a nonzero vector $z\in R_q^m$ such that

$$
Mz=0
$$

and $z$ is short.

There are usually more columns than rows, so a nonzero kernel vector exists as
an algebraic fact. The hard part is finding one whose centered coefficients
all satisfy the required norm bound.

Akita uses two versions of shortness. In the coefficient infinity norm problem,

$$
\lVert z\rVert_\infty\leq\beta.
$$

In the coefficient Euclidean norm problem,

$$
\lVert z\rVert_2\leq\beta.
$$

Both norms are applied to the complete list of $mD$ centered coefficients.

## A small kernel example

Take the deliberately small ring

$$
R_{17}=\mathbb{F}_{17}[X]/(X^2+1)
$$

and the one row matrix

$$
M=\begin{pmatrix}1 & X\end{pmatrix}.
$$

The vector

$$
z=\begin{pmatrix}X\\-1\end{pmatrix}
$$

is in its kernel because

$$
Mz=X-X=0.
$$

Its centered coefficient list is $(0,1,-1,0)$. Therefore
$\lVert z\rVert_\infty=1$ and $\lVert z\rVert_2=\sqrt{2}$.

This example is far too small for security. It only shows what a Module-SIS
solution looks like. Production matrices use much larger scalar dimensions and
bounds selected from generated security tables.

## Why commitment collisions reveal a solution

An Ajtai commitment has the form

$$
C=Ms,
$$

where $M$ is public and $s$ is a short ring vector derived from the committed
data.

Suppose a prover finds two different short vectors $s$ and $s'$ with the same
commitment:

$$
Ms=Ms'.
$$

Subtracting the equations gives

$$
M(s-s')=0.
$$

Since $s\neq s'$, the difference $z=s-s'$ is nonzero. Since both openings are
short, the triangle inequality gives a bound on $z$. The collision has produced
a short nonzero kernel vector for the public matrix.

This is the basic binding argument. The full Akita proof uses weak openings and
fold challenges, so its collision bound includes the challenge norm and the
accepted response norm. The same structure remains. Two incompatible accepted
openings produce a bounded Module-SIS solution.

## How the two norms relate

For any vector with $mD$ centered coefficients,

$$
\lVert z\rVert_\infty
\leq
\lVert z\rVert_2
\leq
\sqrt{mD}\,\lVert z\rVert_\infty.
$$

The first inequality holds because the Euclidean norm includes the square of
the largest coordinate. The second holds because every one of the $mD$
coordinates is at most $\lVert z\rVert_\infty$ in magnitude.

The two bounds are related, but they define different security queries. Akita
keeps separate generated tables for them.

- The infinity norm route is the standard route and is always available.
- A Euclidean route is used only when the verifier proves or directly checks
  the complete physical squared norm of the response.

The word physical is important. The security table applies to the actual ring
coefficients multiplied by the public matrix. It does not apply to a smaller
logical vector before extension coordinates or packed lanes have been expanded.

## Matrix rank, width, and security

For $M\in R_q^{n\times m}$, Akita calls $n$ the module rank and $m$ the ring
width. As scalar dimensions, the matrix has $nD$ rows and $mD$ columns.

A wider matrix gives the commitment more input capacity, but it also gives an
attacker a larger space in which to search for a short kernel vector. Increasing
the module rank makes that search harder. The required rank therefore depends
on all of the following public values:

- the modulus $q$;
- the ring dimension $D$;
- the ring width $m$;
- the accepted collision norm;
- the security model and target.

Akita has three matrix roles. The A matrix commits to the inner witness. The B
matrix compresses or contains later commitment data. The D matrix commits to
opening information. Each role has its own width and bound, so the planner
prices each role separately.

## The production security tables

Akita targets 128-bit quantum security under the ADPS16 quantum LGSA attack
model. The estimator runs offline. It searches scalar lattice attack
parameters, certifies the accepted width boundary for each supported query,
and emits compact tables for the runtime.

The runtime lookup has the following shape:

```text
(security policy, modulus profile, matrix role, ring dimension, norm bound)
    -> maximum secure width for each module rank
```

The planner asks for the smallest rank whose certified width covers the matrix
it needs. The generated schedule records that choice. Proving and verification
do not run the estimator. If a required table cell is missing, schedule
selection fails.

This split keeps expensive floating point attack estimation out of the proof
path. It also makes every accepted parameter choice reproducible from checked
in generated data and its digest.

The 128-bit value is a claim inside a specific attack model. It is not a count
of physical qubits or a claim that every possible future attack has been
classified. The exact policy name is `Quantum128BitADPS16`.

## What the verifier enforces

The security argument uses the same public bounds that verification enforces.
A schedule fixes the matrix parameters, challenge family, digit ranges, and any
response norm cap. These values are part of the schedule and instance identity.

For an infinity norm route, digit range checks and the fold response cap give
the coefficient collision bound. For a Euclidean route, the proof also checks
the squared norm of the complete physical response. A terminal response can be
checked directly because the verifier receives its coefficients.

The verifier never accepts a looser bound and then asks the security estimator
to price it. The approved table lookup happens before the proof is accepted.

## Implementation and review map

| Question | Primary source |
| --- | --- |
| Which security policy and modulus profiles exist? | `crates/akita-types/src/sis/ajtai_key.rs` |
| How is the minimum infinity norm rank selected? | `min_secure_rank` in `ajtai_key.rs` |
| How is the minimum Euclidean rank selected? | `crates/akita-types/src/sis/l2_table.rs` |
| Where do generated infinity norm widths live? | `crates/akita-types/src/sis/generated_sis_table/` |
| Where do generated Euclidean widths live? | `crates/akita-types/src/sis/generated_l2_sis_table/` |
| Where are fold collision bounds computed? | `crates/akita-types/src/sis/norm_bound.rs` |
| Where are physical Euclidean proof shapes defined? | `crates/akita-types/src/sis/physical_l2.rs` |
| Where is the offline estimator implemented? | `crates/akita-sis-estimator/` |

A security review should follow one generated schedule row from its matrix
width and accepted response bound to the exact generated table key. It should
then confirm that the verifier checks the same response coordinates and the
same cap. A separate table or a duplicated local formula would break that
connection.
