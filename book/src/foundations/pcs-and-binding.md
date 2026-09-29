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
$\mathbf z=\sum_b c_b\mathbf s_b$. Extraction considers accepting answers
to related challenges while keeping the earlier messages fixed.

For intuition, suppose two answers differ only at coordinate $j$. Honest
responses satisfy

$$
\mathbf z-\mathbf z'
=(c_j-c'_j)\mathbf s_j.
\tag{4}
$$

When the challenge difference is a unit, division recovers
$\mathbf s_j$. The full ring argument establishes the unit property
and controls the bounds needed for a matrix-collision reduction.

Coordinate-wise special soundness, or CWSS, describes the accepting
transcripts needed for this extraction. At one fold, take the original
selected accepting run as the central child. For each coordinate, obtain
another accepting child that changes only that coordinate. All children
share the prover messages sent before the challenge, and each has its own
accepting proof for the later checks. Repeating this construction at every
fold produces an accepting transcript tree.

Akita's sampler exposes each coordinate separately. At fold level $j$,
commitment group $g$ has
$W_{j,g}=\text{num\_claims}_{j,g}\text{num\_live\_blocks}_{j,g}$
coordinates. A coordinate is one sparse ring element for a claim and source
block. The sampler squeezes
one 32-byte group root, then uses
`SHAKE256(root || little_endian_u64(index))`, with
`index = claim * num_live_blocks + block`. The group payload binds the method,
dimension, challenge family, group index, and counts. In the ideal-oracle
model, each indexed input has its own random answer. The extractor can
therefore change one coordinate while keeping the others fixed.

There are two opening methods. Evaluation trace uses the scheduled challenge
ring and permits coefficient-Linf or selective-L2 response security. Subring
coefficient packing samples in the specified challenge subring and requires
the Linf route. Selective L2 currently requires one scalar group; the
terminal has one evaluation-trace group with one claim. Packing also has a
separate consistency polynomial check of degree at most $2s-1$, where
$s$ is the challenge-subring dimension. Its field error enters the
relation-check ledger.

Let $C_{j,g}$ be the support after fixed sampler filters. For a shell in
dimension $d$ with $a$ coefficients of magnitude one and $b$ of magnitude two,
its unfiltered cardinality is
$\binom d a\binom{d-a}b2^{a+b}$.
The production shell ladder has at least 128 support bits per coordinate.
Selective L2 at D64 and D128 uses a fixed operator-norm predicate; its
certified accepted subset also retains at least 128 support bits. The
[grinding specification](../../../specs/transcript-grinding.md#geometry-and-the-sampling-denominator)
records the exact families. The accepted-support bound is the denominator
used below, including for filtered samplers.

Binary CWSS uses a common prechallenge prefix and, for each coordinate,
a pair of accepting children that agree everywhere else. Each child has
its own complete accepting subtree for the later ring, range, norm, relation,
and recursive checks. Later challenges may differ. These strong
checks recover exact native-ring identities and verifier-certified response
bounds below each child. If the children open a common commitment path
differently, the commitment maps give a scheduled matrix collision.
Otherwise, subtraction cancels every unchanged fold term. The production
LS18 check makes the remaining nonzero challenge difference a unit in
both opening methods, so division recovers that coordinate's weak opening.
The extractor accepts any source opening that satisfies the verified
relations. To bound a collision, it cross-multiplies accepted responses and
challenge differences before centering the resulting kernel vector.

The terminal's direct relations authenticate its incoming opening.
Applying the preceding step backward through the folds extracts the root
openings or a short collision in a scheduled matrix view. A fold with
$W_{j,g}$ coordinates in group $g$ requires
$1+\sum_g W_{j,g}$ children. These factors multiply across folds and
with the other challenge stages. The
[grinding specification](../../../specs/transcript-grinding.md#fold-extraction-from-accepting-children)
gives the extraction and collision steps in more detail.

## Fiat-Shamir queries and fold nonces

Fiat–Shamir derives each verifier challenge from the public statement and the
proof prefix through a random oracle. A dishonest prover can try several proof
prefixes, including different fold-response nonces, before submitting one
proof. The security bound therefore counts the oracle inputs that the prover
actually explores.

### What one coordinate query means

A fold coordinate address is the pair `(group root, claim-major index)`.
Requesting any bytes from its indexed SHAKE256 stream touches that address.
Further reads from the same stream use the same answer. A different root or
index gives a different address. The 32-byte group root is also an oracle
output; it belongs in the total query budget. Distinct transcript contexts
that produce the same root are covered by the separate root-collision term.

For each possible address $a$, let $q_a=1$ when the original dishonest
prover touches it during any attempted proof, and let $q_a=0$ otherwise.
Let $r_a=1$ when verification of the final submitted proof first needs
that address and the prover did not touch it. Thus $q_a$ and $r_a$
mark disjoint sets. The extractor's replay queries affect its running
time; the security charge uses the original prover's queries.

$Q_{\mathrm{coord}}$ bounds $\sum_a q_a$ in every execution of the
dishonest prover. $R_{\mathrm{coord}}$ is the maximum number of previously
untouched addresses in one final proof under the selected schedule. One
final proof uses at most one address for each scheduled fold coordinate, so

$$
R_{\mathrm{coord}}\leq\sum_{j,g}W_{j,g}.
\tag{5}
$$

For example, suppose the final proof has three fold coordinates. If the
prover has already read two streams for that proof, final verification
completes one address: those three addresses contribute two to the
observed query count and one to the completion count. Reading all three
coordinates under a second candidate root adds three more query addresses.

### The fold bound

Let $C(a)$ be the set of possible values at address $a$ after the
sampler's fixed filters, and let $C_{\min}$ be the smallest such set in
the schedule. Fixed operator filtering changes this set; the
source-dependent test that a fold response fits its bounds changes which
proof the prover submits and is charged through its oracle queries.
The indexed streams are uniform on their respective accepted sets in
the ideal-oracle model, conditional on sampler success.

The fold extraction argument gives

$$
\varepsilon_{\mathrm{fold}}
\leq
\mathbb E\!\left[\sum_a\frac{q_a+r_a}{|C(a)|}\right]
\leq
\frac{Q_{\mathrm{coord}}+R_{\mathrm{coord}}}{C_{\min}}.
\tag{6}
$$

The expectation covers the dishonest prover's choices and the random
oracle. To understand the one-value numerator, keep the original selected
accepting run, fix every other oracle answer, and vary the answer at one
selected coordinate address. Two suitable accepted values at that address
provide the coordinate fork used by the accepting-tree extractor.
The remaining matching-input failure can therefore occupy at most one
accepted value at that address. An address contributes only when the
original prover touched it or final verification completes it. Applying
this charge through the tree gives (6). The
[grinding specification](../../../specs/transcript-grinding.md#indexed-address-fold-bound)
states the exact conditional sampling and tree argument.

Every production accepted coordinate set contains at least $2^{128}$
values. Equation (6) therefore assigns at most $2^{-128}$ to each
charged fold address with zero additional fold proof of work. A
12-bit fold-response nonce lets the prover try more prefixes; the
coordinate streams read during those trials increase
$Q_{\mathrm{coord}}$. This accounts for that choice directly.

### From the fold bound to a security claim

For a particular attacker budget, substitute its maximum number of
distinct coordinate addresses and the schedule's verifier-completion
bound into (6). For example, $Q_{\mathrm{coord}}+R_{\mathrm{coord}}=2^{12}$
gives a fold term of at most $2^{-116}$
when $C_{\min}=2^{128}$. An aggregate 128-bit target requires enough
support margin to cover the declared query and completion counts.

The full knowledge-soundness calculation also includes the other
challenge checks, setup distance, compressed-root collisions, and
scheduled matrix-collision advantages. Each field challenge must use
the loss of its actual position in the complete extraction tree when
pricing the full calculation. The current grinding plan prices its
typed local field checks; a complete tree calculation supplies their
combined extraction debits. The fold tree itself contributes
$\prod_j(1+\sum_gW_{j,g})$ to extractor cost. The complete tree,
including its other challenge stages, must have polynomial size in
the declared instance length and security parameter for an expected
polynomial-time knowledge extractor.

This analysis is in the classical random-oracle model. The
[grinding specification](../../../specs/transcript-grinding.md#fold-work-decision-and-total-security-accounting)
gives the detailed reduction ledger and implementation evidence.

Honest search measures how often the response fits the scheduled cap.
Under an independent-trial model with per-trial acceptance probability $p$,
the chance of exhausting $N$ attempts is $(1-p)^N$. This is an
honest-proving completeness measure. Any estimate of $p$ is tied to its
measured source and schedule.

A concrete security claim combines matrix hardness estimates with the
folding, ring-relation, sumcheck, and Fiat–Shamir bounds.

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
