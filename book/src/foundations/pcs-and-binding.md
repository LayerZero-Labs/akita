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

Akita's sampler exposes coordinate inputs explicitly. At fold level $j$,
commitment group $g$ has
$W_{j,g}=\text{num\_claims}_{j,g}\text{num\_live\_blocks}_{j,g}$
coordinates. Each coordinate is one sparse ring element indexed by a claim
and source block, not one coefficient of that element. The sampler squeezes
one 32-byte group root, then uses
`SHAKE256(root || little_endian_u64(index))`, with
`index = claim * num_live_blocks + block`. The group payload binds the method,
dimension, challenge family, group index, and counts. In the ideal-oracle
model, changing one coordinate answer leaves the other coordinate streams
unchanged. This supplies a place to fork; it does not prove extraction.

There are two opening methods. Evaluation trace uses the scheduled challenge
ring and permits coefficient-Linf or selective-L2 response security. Subring
coefficient packing samples in the specified challenge subring and requires
the Linf route. Selective L2 currently requires one scalar group; the
terminal has one evaluation-trace group with one claim. Packing also has a separate consistency polynomial check of
degree at most $2s-1$, where $s$ is the challenge-subring dimension. Its field
error belongs to the relation-check ledger, not the sparse support bound.

Let $C_{j,g}$ be the support after fixed sampler filters. For a shell in
dimension $d$ with $a$ coefficients of magnitude one and $b$ of magnitude two,
its unfiltered cardinality is
$\binom d a\binom{d-a}b2^{a+b}$.
The production shell ladder has at least 128 support bits per coordinate.
Selective L2 at D64 and D128 uses a fixed operator-norm predicate; its
certified accepted subset also retains at least 128 support bits. The
[grinding specification](../../../specs/transcript-grinding.md#geometry-and-the-sampling-denominator)
records the exact families. The raw shell cardinality cannot replace the
accepted-support bound for a filtered sampler.

Binary CWSS uses a common prechallenge prefix and, for each coordinate,
a pair of accepting children that agree everywhere else. Each child has
its own complete accepting subtree for the later ring, range, norm, relation,
and recursive checks. The later challenges need not agree. These strong
checks recover exact native-ring identities and verifier-certified response
bounds below each child. If the children open a common commitment path
differently, the commitment maps give a scheduled matrix collision.
Otherwise, subtraction cancels every unchanged fold term. The production
LS18 check makes the remaining nonzero challenge difference a unit in
both opening methods, so division recovers that coordinate's weak opening.
The extracted source need not be the honest canonical digit decomposition.
To bound a collision, cross-multiply accepted responses and challenge
differences before centering; division alone does not preserve norms.

The terminal's direct relations authenticate its incoming opening.
Applying the preceding step backward through the folds extracts the root
openings or a short collision in a scheduled matrix view. For one fold
with $W_{j,g}$ coordinates in group $g$, its flat accepting family has
$1+\sum_g W_{j,g}$ children. Charging one query per *complete fold vector*
gives the valid whole-fold interactive error

$$
\varepsilon_{\mathrm{fold}}^{\mathrm{whole}}
=\sum_j\sum_g\frac{W_{j,g}}{|C_{j,g}|}.
\tag{5a}
$$

The sum includes every group and the terminal. Its tree factors multiply
across folds and with the other challenge stages. The
[grinding specification](../../../specs/transcript-grinding.md#fold-extraction-from-accepting-children)
gives the extraction and collision steps in more detail.

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

Each Akita fold uses a bounded response nonce encoded inline as canonical
unsigned LEB128. Changing those proof bytes changes the sparse-challenge oracle
inputs. For
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

A large challenge support alone does not give a one-value bad-set bound.
The needed bound comes from the binary matching-input game for the accepting
fold tree. Fixed operator rejection is uniform on its accepted family
conditional on sampler success. The tree sampler can program the *complete*
ideal XOF tape conditional on a prescribed accepted coordinate: it samples
a successful raw tape, replaces its first accepted shell value, and
conditionally samples that value's bit encoding. Response admission is
source dependent and is never treated as another fixed sampler filter.

Let $Q_{\mathrm{coord}}$ count distinct indexed streams touched by the
original adversary and $R_{\mathrm{coord}}$ count streams completed only by
final verification. A selected stream has one address determined by its
group root and claim-major index. Fix the other oracle answers and the
suffix sampler's coins, retaining the address selected by the *original*
execution even if suffix extraction fails. At a nonempty selected-address
fiber, the binary game loses at most one accepted value out of
$C_{j,g}$. That fiber can be nonempty only if the original execution
queried the address or its final proof selects it for verification: if
neither occurs, changing that answer cannot change the execution.
Summing these address charges proves

$$
\varepsilon_{\mathrm{fold}}^{\mathrm{indexed}}
\le
\mathbb E\!\left[\sum_a\frac{q_a+r_a}{|C(a)|}\right]
\le
\frac{Q_{\mathrm{coord}}+R_{\mathrm{coord}}}{C_{\min}},
\qquad
R_{\mathrm{coord}}\le\sum_{j,g}W_{j,g}.
\tag{8}
$$

Here $q_a$ and $r_a$ mark original queries and untouched verifier
completions, and $C_{\min}=\min_{j,g}|C_{j,g}|$. The same original selected
proof is retained through later tree stages, so their weighted bounds
telescope. One complete fold candidate touches $\sum_gW_{j,g}$ streams;
indexing changes the query unit without hiding the geometry. A collision
of two distinct contexts compressed to the same 32-byte root is a
separate binding event.

All production accepted supports have at least $2^{128}$ elements.
Consequently each indexed fold address has error weight at most
$2^{-128}$ with **zero additional fold proof of work**. This proves the
128-bit *per-address rate* used to price fold work in the classical random
oracle model. It does not assert that an arbitrary number of addresses
has aggregate error at most $2^{-128}$.

For the complete challenge tree, give each field-family address its
conditional error weight $L_r/(|E_r|2^{g_r})$ and each sparse address
$1/|C(a)|$. Exact field pricing makes the former at most $2^{-128}$ at
the typed proof-of-work sites. If the ledger includes every relevant
stage, the matching-input bound is

$$
\varepsilon_{\mathrm{tree}}
\le
\mathbb E\!\left[\sum_a\alpha_a(q_a+r_a)\right]
\le \rho(Q+R_{\mathrm{field}}+R_{\mathrm{coord}}),
\qquad \rho=\max_a\alpha_a\le2^{-128}.
\tag{9}
$$

$Q$ includes failed work attempts, response-nonce trials, and each
coordinate stream touched. The final two terms count previously
untouched verifier completions. Setup distance, root-compression
collisions, and matrix-collision advantages are additional terms, with
the latter evaluated at the extractor's actual expected cost. The fold
tree contributes the factor
$\prod_j(1+\sum_gW_{j,g})$ to that cost. Expected polynomial extraction
also requires the full tree product, including nonfold stages, to be
polynomial in the declared explicit instance length and security
parameter. The current plan's one-replay query count is not that
tree-size certificate.

Thus zero fold work meets the same 128-bit per-address rate as the
priced field challenges. For a particular aggregate target, one must
insert a declared adversarial query budget and the verifier completion
counts into (9), then include the other reduction terms. Counting
response-nonce trials in $Q$ entails no separate fixed 12-bit debit.
These are classical random-oracle statements, not a QROM theorem.
The [grinding specification](../../../specs/transcript-grinding.md#indexed-address-fold-bound)
contains the full fold argument and the exact conditional sampler.

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
