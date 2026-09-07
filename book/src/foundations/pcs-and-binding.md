# Polynomial commitments and binding

A polynomial commitment lets a prover fix a polynomial now and prove an
evaluation of it later. The verifier receives a short commitment and an
opening proof instead of reading the complete polynomial.

Akita's main interface commits to multilinear tables. For example, over
$\mathbb F_{17}$, the table

| $(x,y)$ | Value |
| --- | ---: |
| $(0,0)$ | 1 |
| $(1,0)$ | 3 |
| $(0,1)$ | 2 |
| $(1,1)$ | 4 |

defines the multilinear polynomial $\widetilde f(x,y)=1+2x+y$.
An opening at $r=(2,3)$ claims the value $v=8$. The point need not lie
on the Boolean table domain.

## The statement and its guarantees

Setup produces public parameters. Commitment maps an admissible table $f$
to a public value $C$, together with prover state retained for opening.
Opening takes the committed table and a point $r$, and produces a proof
$\pi$ for the claimed value $v$. Verification checks

$$
\operatorname{Verify}(\mathrm{pp},C,r,v,\pi)=\mathrm{accept}.
\tag{1}
$$

Here $\mathrm{pp}$ includes the public setup and the parameters that
determine the table's interpretation. The
[instance descriptor](../how/transcript.md#akitainstancedescriptor) binds
those parameters before challenge generation.

Completeness means that honest commitment and opening computations produce
an accepted claim. Akita also has bounded honest-prover searches for fold
responses and transcript proof of work. A search can return an exhaustion
error instead of a proof. Its probability is a completeness consideration,
separate from whether a dishonest proof can be accepted.

Evaluation binding rules out efficiently producing two accepted openings of
the same commitment at the same point with different values, except with the
allowed failure probability. In the example, accepting both $v=8$ and
$v=9$ at $(2,3)$ would violate binding. Opening the same polynomial at
a different point is ordinary use.

Knowledge soundness asks for more. An extractor with the prescribed access
to a successful prover should be able to recover a witness explaining its
claim, except with a stated knowledge error. An extractor is an algorithm
used in the security argument. It is not the ordinary verifier, and the
verifier does not reconstruct the full table during each opening.

Hiding asks whether a commitment conceals its contents, and zero knowledge
asks whether a proof reveals information beyond its public statement.
Neither follows from binding. Akita's current commitment and opening path
does not promise either property. See the
[current privacy boundary](../roadmap/zero-knowledge.md).

## Why short vectors matter

An Ajtai commitment applies a public matrix to a short vector. In the ring
case, let $R=F[X]/(X^D+1)$, let
$\mathbf A\in R^{n\times m}$, and let
$\mathbf s\in R^m$ have small integer coefficient representatives. Its
image is $\mathbf t=\mathbf A\mathbf s$.

If two distinct admitted vectors give the same image, subtract their
equations:

$$
\mathbf A\mathbf s=\mathbf A\mathbf s'
\quad\Longrightarrow\quad
\mathbf A\boldsymbol\delta=\mathbf 0,
\qquad
\boldsymbol\delta=\mathbf s-\mathbf s'\ne\mathbf 0.
\tag{2}
$$

If each input has coefficient norm at most $B$, the triangle inequality
gives

$$
\lVert\boldsymbol\delta\rVert_\infty\le 2B.
\tag{3}
$$

The analogous statement for Euclidean bounds is
$\lVert\boldsymbol\delta\rVert_2\le 2B_2$. Module-SIS asks for a short
nonzero vector in such a matrix kernel. Ordinary linear algebra can find
kernel vectors when the matrix has more columns than rows; the hardness
requirement is finding one within the specified bound.

This is why an accepted digit range or response norm is part of the binding
argument. An equation modulo the field does not by itself certify a small
integer representative. The prover's storage type is not a substitute for
a verifier-enforced bound either.

Equations (2) and (3) describe the elementary collision argument. A full
folding extraction can introduce additional factors from challenges,
decomposition, and ring arithmetic. The concrete SIS bounds must account for
those factors rather than using $2B$ as a universal Akita bound.

## The two-tier Ajtai commitment

Akita first computes an inner image $\mathbf t_b=\mathbf A\mathbf s_b$
for each source block. It decomposes those images into short digits
$\hat{\mathbf t}$ and computes the outer image
$\mathbf u=\mathbf B\hat{\mathbf t}$. A standalone commitment then
compresses $\mathbf u$ through two more short-input matrix maps to a
128-byte payload.

This creates several binding obligations. If two different sources share a
commitment, follow their images through the chain. Where distinct admitted
inputs first acquire the same output, their difference gives a short kernel
vector for that map. If inner images differ, the outer and compression maps
must authenticate them; if an inner image is shared by distinct source
digits, the A relation supplies the collision.

D enters during opening. It binds the opening partials through
$\mathbf v_D=\mathbf D\hat{\mathbf e}$, with its own compression chain
when the fold uses compressed payloads. The fold's consistency relation
connects these partials to the same source used by A. The scalar opening
relation then connects them to $v$.

Later recursive folds can use raw B and D images. That removes their payload
compression maps but preserves the A, B, D, consistency, and scalar-opening
obligations. The terminal uses a direct A and consistency check instead of
creating another outer commitment. The complete data path is described in
[setup and commitment](../how/commitment.md) and
[recursion](../how/recursion.md).

Ring dimension, module rank, input width, and accepted norm therefore matter
for each matrix actually used. Akita records these in the selected schedule
and validates them when expanding it. The coefficient Linf and Euclidean L2
routes use different SIS tables because they certify different norms.
[Security model](../how/security.md) owns the concrete sizing policy.

## Coordinate-wise special soundness

Folding compresses many source blocks into a response
$\mathbf z=\sum_b c_b\mathbf s_b$. One response alone cannot recover all
the $\mathbf s_b$. Extraction instead considers accepting answers to
related challenges while keeping the earlier messages fixed.

For intuition, suppose two answers differ only at coordinate $j$. Honest
responses satisfy

$$
\mathbf z-\mathbf z'
=(c_j-c'_j)\mathbf s_j.
\tag{4}
$$

If the challenge difference is invertible, division recovers
$\mathbf s_j$. In a ring, a nonzero difference need not be invertible.
An actual folding argument must handle that issue and the bounds on the
extracted coefficients. Equation (4) explains the role of coordinate forks;
it does not establish those additional properties.

Coordinate-wise special soundness, or CWSS, formalizes extraction from a
structured collection of accepting transcripts. A challenge in
$S^\ell$ has $\ell$ coordinates. For one challenge round, an
$\ell$-coordinate-wise $k$-special-sound protocol extracts from a central
challenge and $k-1$ alternatives along each coordinate, keeping the other
coordinates fixed.
Across $\mu$ rounds, these sets form a transcript tree with
$(1+\ell(k-1))^\mu$ leaves. Efficient extraction requires controlling this
cost. With common parameters, the interactive knowledge-error bound is

$$
\kappa\le\frac{\mu\ell(k-1)}{|S|},
\tag{5}
$$

Here $\ell$ counts coordinates, and $\mu$ counts challenge rounds. See
Definitions 2.29 and 2.30 and Lemma 2.31 of
[Lattice-Based Polynomial Commitments](https://eprint.iacr.org/2023/846.pdf).

Akita's sampler exposes coordinate inputs explicitly. It squeezes one group
root, then derives each claim-major block coordinate from an indexed
random-oracle input. A coordinate fork fixes the root and the other
coordinate answers. The selected support, group counts, and response
admission rules must still be matched to the extraction argument. Equation
(5) is the common-parameter case, not a complete error estimate for a
heterogeneous Akita schedule.

## Fiat-Shamir queries and fold nonces

Fiat-Shamir makes verifier challenges deterministic random-oracle answers.
For the special-sound protocols covered by
[Attema, Fehr, and Klooß](https://ir.cwi.nl/pub/33324/33324.pdf), Theorem 2
bounds the compiled knowledge error by

$$
\kappa_{\mathrm{FS}}(Q)\le(Q+1)\kappa,
\tag{6}
$$

where $Q$ is the adversary's oracle-query budget. The CWSS extension appears
in Lemma 2.32 and Section 8 of
[Lattice-Based Polynomial Commitments](https://eprint.iacr.org/2023/846.pdf).
These are classical random-oracle extraction results. Applying them requires
the specified special-soundness and efficiency premises; (6) alone is not a
quantum-random-oracle bound or an end-to-end theorem for Akita.

Each Akita fold uses a 12-bit response nonce from the packed proof-level
stream. Changing the nonce changes the sparse-challenge oracle inputs. For
a fixed prefix and a bad-challenge set of measure $\epsilon$, $q$
independent trials succeed with probability

$$
1-(1-\epsilon)^q\le q\epsilon.
\tag{7}
$$

Those trials also require oracle queries. When a security analysis already
counts them in $Q$, charging an additional fixed 12-bit loss for that same
nonce freedom would count the same work twice. Conversely, the bounded nonce
field does not justify omitting adversarial trials from $Q$. An adversary
can also vary earlier messages and start from other transcript prefixes.

Honest search measures how often the response fits the scheduled cap.
Under an independent-trial model with per-trial acceptance probability $p$,
exhausting $N$ attempts has
probability $(1-p)^N$. This models honest proving failure, not adversarial
soundness. A response-model estimate must be assessed for its actual source
and schedule; it is not an unconditional lower bound on $p$.

A concrete security claim must combine the matrix hardness estimates with
the applicable folding, ring-relation, sumcheck, and Fiat-Shamir bounds.
Choosing a large sparse challenge support or passing an SIS table check
alone does not complete that composition.

## Implementation map

- `crates/akita-types/src/sis/` owns matrix and norm security parameters.
  Schedule validation checks the selected geometry against those parameters.
- `crates/akita-prover/src/protocol/fold_grind.rs` performs bounded honest
  response search.
- `crates/akita-challenges/src/sampler/xof.rs` gives sparse challenge
  coordinates distinct oracle inputs.
- `crates/akita-types/src/instance_descriptor/` binds schedule identity and
  the protocol-wide grinding contract.
- `crates/akita-verifier/src/protocol/core/terminal_direct.rs` checks the
  final response norm and direct relations.
- `crates/akita-pcs/tests/fold_linf.rs` and
  `crates/akita-pcs/tests/transcript_hardening.rs` provide regression checks
  for nonce replay, bounds, and transcript tampering. Such tests support the
  correspondence with the implementation; they do not prove extraction.
