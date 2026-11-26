# The distributed prover

This chapter describes a distributed variant of the Akita prover: how the witness
is split across many machines, which parts of one folding step are local, and the
single protocol change — keeping a *per-machine* folded response — that lets the
machines avoid shuffling large witnesses between each other. The machines can be
separate GPUs, accelerators, or nodes; nothing below is specific to GPUs.

> **Status.** The protocol change is implemented as the multi-chunk witness
> layout: a chunk is a contiguous block range, and each chunk keeps its own
> folded response. The prover in this repository computes all chunks in one
> process. It contains no cross-machine transport or aggregation service. Read
> "machine $P_j$" below as "the worker that owns chunk $j$", and read each
> exchange between machines as a statement of which values that worker needs
> from the others. [Multiple witness chunks](./advanced-relation-layouts.md#multiple-witness-chunks)
> derives the same relations without the machine model.

## What "distributed-prover friendly" means

We hold the design to three concrete criteria:

1. **Distributable witness.** The witness $\mathbf s$ can be split across
   independent machines so that each machine owns and processes only its share.
2. **Low cross-machine communication.** Machines must not exchange large payloads.
   Only short commitment vectors and small round-polynomial messages may cross
   the interconnect; the large per-machine witnesses stay put.
3. **Net speed-up.** The resulting scheme must stay efficient enough that the
   per-machine work plus the small aggregation actually realises the parallel
   speed-up the hardware promises.

Throughout, assume $\mathcal M = 2^N$ machines $P_0,\dots,P_{\mathcal M-1}$ (take
$N=3$, so $\mathcal M = 8$ machines). We divide the witness into $\mathcal M$
contiguous block ranges. When the number of live blocks is not divisible by
$\mathcal M$, a machine owns either the floor or the ceiling of the average;
when there are fewer blocks than machines, some ranges are empty.

## Setup: partitioning the witness

We spell the design out in the single-opening case (one opening point
$\mathbf r$, one commitment, one committed slot). Let $N$ be the exact number
of live source ring elements, choose $M := 2^{r_{\mathsf{pos}}}$ positions per
block, and set $B := \lceil N/M\rceil$. The block-index Boolean domain has size
$B_{\mathsf{dom}} := 2^{r_{\mathsf{blk}}} = \operatorname{nextpow2}(B)$, but only
the prefix $[B]$ is live. Reshape the coefficient table into $B$ blocks
$\mathbf f_i \in R_q^M$, padding only the final block when $M \nmid N$.

Every multi-chunk fold uses subring coefficient packing. For each block, the
opening point defines the public linear packing map $L$ of
[the fold page](./akita-fold.md#subring-coefficient-packing-consistency). It
sends the digit vector $\mathbf s_i$ of block $i$ to a short packed partial
$e_i := L(\mathbf s_i)$ in $E[U]/(U^s+1)$. The claimed evaluation
$F(\mathbf r)$ is a public linear function of the partials $e_1,\dots,e_B$,
given by the
[direct packed opening relation](./field-ring-reduction.md#subring-coefficient-packing-shorter-partials).

We partition the block index set $[B]$ into contiguous ranges
$[B] = \mathcal I_0 \sqcup \dots \sqcup \mathcal I_{\mathcal M-1}$, where
$\mathcal I_j := \{\, i \in [B] : \lfloor jB/\mathcal M\rfloor \le i <
\lfloor (j+1)B/\mathcal M\rfloor \,\}$, and assign $\mathcal I_j$ to machine
$P_j$. For $B=10$ and $\mathcal M=4$, these ranges are
$[0,2),[2,5),[5,7),[7,10)$. This is the
[canonical proportional partition](./opening-points-layout.md#chunks-and-fold-challenges)
owned by `WitnessLayout`. It applies to root sources. A recursive level does
not repartition: it inherits the block ranges of the chunk bodies that the
previous level produced, so its ranges need not be balanced. Larger index
spaces are always partitioned at the block level: for any $k \ge 1$ the block partition induces
$[kB] = \mathcal I_0^{[k]} \sqcup \dots \sqcup \mathcal I_{\mathcal M-1}^{[k]}$.

The public key is still the three uniformly random matrices
$\mathbf A \in R_q^{n_A \times M\delta}$,
$\mathbf B \in R_q^{n_B \times B n_A\delta}$, and
$\mathbf D \in R_q^{n_D \times B\delta}$ with $\delta := \lceil\log_b q\rceil$. All
three are prefix views of one seed-expanded matrix, so **every machine regenerates
the same public columns locally** — no setup is broadcast. In the distributed
setting we view them column-blocked,
$\mathbf B = [\mathbf B_0 \mid \dots \mid \mathbf B_{\mathcal M-1}]$ and
$\mathbf D = [\mathbf D_0 \mid \dots \mid \mathbf D_{\mathcal M-1}]$, where
$\mathbf B_j,\mathbf D_j$ hold exactly the columns indexed by $P_j$'s blocks.

## Component-by-component locality

We now walk each component of one folding step and check it against the three
criteria.

### The outer commitment $\mathbf u$

Machine $P_j$ stores its blocks $\{\mathbf f_i : i \in \mathcal I_j\}$ and performs
the ordinary inner commit on each, restricted to its block set. For every
$i \in \mathcal I_j$ it digit-decomposes the block witness
$\mathbf s_i := \mathbf G_{b,M}^{-1}(\mathbf f_i) \in R_q^{M\delta}$, computes the
inner commitment $\mathbf t_i := \mathbf A\mathbf s_i \in R_q^{n_A}$, and digitises
it to $\widehat{\mathbf t}_i := \mathbf G_{b_1,n_A}^{-1}(\mathbf t_i)$. Writing
$\widehat{\mathbf t}^{(j)} := (\widehat{\mathbf t}_i)_{i \in \mathcal I_j}$ for the
inner-commitment digits it owns, $P_j$ uses **only its column block** $\mathbf B_j$
to form the partial outer commitment

$$
  \mathbf u_j := \mathbf B_j\,\widehat{\mathbf t}^{(j)} \in R_q^{n_B}.
$$

The raw relation image is then the sum
$\mathbf u = \sum_{j=0}^{\mathcal M-1}\mathbf u_j$. The large local witnesses
$\widehat{\mathbf t}^{(j)}$ never leave $P_j$; only the short $n_B$-vectors
$\mathbf u_j$ are summed. The aggregation machine negative-binary decomposes
that sum through the two F maps and publishes only the 256-byte payload
$p_F$ (criteria 1 and 2).

### The opening commitment $\mathbf v$ and the claim $y$

The opening side is computed the same way. For every $i \in \mathcal I_j$, $P_j$
computes the local packed partial $e_i := L(\mathbf s_i)$, digitises its
base-field coordinates to $\widehat{\mathbf e}_i$, so that
$e_i = \operatorname{Rec}_{\mathrm{open}}(\widehat{\mathbf e}_i)$,
and — using only its column block $\mathbf D_j$ — forms the partial opening
commitment $\mathbf v_j := \mathbf D_j\,\widehat{\mathbf e}^{(j)} \in R_q^{n_D}$,
with $\widehat{\mathbf e}^{(j)} := (\widehat{\mathbf e}_i)_{i\in\mathcal I_j}$. The
global opening relation image is $\mathbf v = \sum_j \mathbf v_j$. The aggregation
machine compresses that sum through the two H maps and publishes only the
256-byte payload $p_H$. Each machine also
forms its partial claim $y_j$, the part of the linear opening function that
reads the partials $e_i$ with $i\in\mathcal I_j$. The claimed evaluation is
$y = F(\mathbf r) = \sum_j y_j$.

So the raw $\mathbf u$, raw $\mathbf v$, and $y$ all follow the same pattern: each machine does
the same inner/outer/opening computation as the single-machine prover, but only on
its own blocks. The machines aggregate only the small relation images and scalar
claims; one aggregation machine derives the shared compression digits and public
terminal payloads.

### The folded response $\mathbf z$ — the hard part

The verifier samples one folding challenge $c_i$ per block from the challenge
subring $S=F[U]/(U^s+1)$. The source fold uses its embedding
$\iota(c_i)\in R_q$. The single-machine prover would form **one** global
folded response

$$
  \mathbf z = \sum_{i=1}^{B} \iota(c_i)\,\mathbf s_i \in R_q^{M\delta}.
$$

But each machine only holds the blocks in $\mathcal I_j$, so it can only compute a
*partial* fold $\mathbf z_j := \sum_{i\in\mathcal I_j} \iota(c_i)\,\mathbf s_i \in R_q^{M\delta}$
from its partial challenges $\mathbf c^{(j)} = \{c_i\}_{i\in\mathcal I_j}$.
Crucially, each $\mathbf z_j$ lives in the **full** ambient dimension $R_q^{M\delta}$
— the same size as the entire folded witness. Summing the $\mathbf z_j$ into one
global $\mathbf z$ would require an all-reduce of this large folded-witness payload
across every machine, which directly violates criterion 2.

**The protocol change.** Instead of reconciling the $\mathbf z_j$, we keep them
*separate*. Each $\mathbf z_j$ becomes its own piece of the next-level witness:
machine $P_j$ carries forward its own $\mathbf z_j$ and never exchanges it. In
effect, the next recursive witness now contains $\mathcal M$ folded responses
instead of one, and the machines need no $\mathbf z$-interaction at all.

### Security accounting for the shared response

The aggregate commitment-consistency rows act on
$\mathbf z_\Sigma := \sum_j \mathbf z_j$. This preserves the algebraic identity
of the ordinary fold, but it changes the weak-binding envelope. The digit range
check is applied separately to every $\mathbf z_j$ using one common basis and
one common digit depth $\delta$. Thus a large positive coefficient in one chunk
cannot be hidden by a negative coefficient in another. If the digit basis is
$b=2^\ell$, the exact difference interval for one chunk has diameter
$b^\delta-1$, while the difference interval for $\mathbf z_\Sigma$ has diameter

$$
  C(b^\delta-1),
$$

where $C=\mathcal M$ is the response-chunk count. With $\kappa_1$ denoting the
physical $L_1$ mass of the folding challenge, the A-role collision target is

$$
  4\kappa_1 C(b^\delta-1).
$$

The planner, generated schedule, and verifier admission all derive this exact
raw target and select an audited A-role cell no smaller than it. A matrix sized
for one response cannot be reused for several response chunks. The current SIS
table does not add a separate coverage axis for the chunk count, so a two- or
four-chunk schedule may use a conservative next-higher cell rather than the
tightest possible collision-bound search. Every chunk retains the same
full-width $\mathbf z_j$ buffer—even an
empty block range contributes a full-width zero response—whereas the local
$\widehat{\mathbf e}^{(j)}$ and $\widehat{\mathbf t}^{(j)}$ ranges follow the
actual number of blocks. The number of response chunks is therefore a protocol
parameter, not merely a deployment choice. Production schedules expose only a
small fixed set of counts, and verifier validation caps the count so that a
malicious schedule cannot force arbitrarily fragmented setup evaluation.

## Why only the first few rounds keep per-machine $\mathbf z$

Keeping a separate $\mathbf z_j$ per machine is *not* free, so we do it only for a
constant number of leading rounds. Two reasons:

1. **Proof size and fold depth.** The $\mathcal M$ separate $\mathbf z_j$ enlarge
   the next witness relative to a single $\mathbf z$. Carrying that growth into the
   tail and final rounds would inflate the proof size and the number of fold levels,
   so those rounds keep the single-response protocol.
2. **Where the cost actually is.** Only the first few rounds are large enough to be
   worth distributing. Once the scheduled recursive folds have reduced the full
   witness, the remaining rounds are cheap and run fine on a single machine.
   This does not require the root fold itself to shrink the witness.

So the schedule fixes a constant number of distributed rounds. Those leading
rounds retain one $\mathbf z_j$ per machine; the later rounds return to the
single-response Akita protocol on one machine.

The opening method follows the same boundary. Every distributed round uses
subring coefficient packing. A later single-response round may use packing or
`EvaluationTrace`, as the schedule selects; its relations are the single-chunk
relations of [Semantic relations in an Akita fold](./akita-fold.md). The first
single-response round inherits the per-machine witness bodies of the last
distributed round and merges them into one owner.

## The next-round witness and the relation change

After the fold of a distributed round, each machine $P_j$ holds its own
$(\widehat{\mathbf t}^{(j)},\,\widehat{\mathbf e}^{(j)},\,\mathbf z_j)$
and can proceed to the next round on its own. The quotient is not part of this
local data: the witness holds one shared quotient range, as the ring-switch
section below explains. Supporting this requires changing the
witness format and the relation matrix $\mathbf M$ so the relation is stated over the
*concatenation* of the per-machine witnesses rather than over one global folded
response. The rest of this chapter makes that change precise.

### The per-machine partial root relation

Each machine holds its partial witness and the matching partial commitments, plus
the fold of its own blocks. Write $\mathbf z^{(j)}$ for the partial fold
$\mathbf z_j$ as a witness segment. The folded-evaluation consistency row is the
packed relation, because every multi-chunk fold uses subring coefficient
packing. It holds in $E[U]/(U^s+1)$ and uses the challenges $c_i\in S$. The
other three rows hold over $R_q$, and the folded-commitment row uses the
embedded challenges $\iota(c_i)$. Machine $P_j$ satisfies:

$$
\begin{aligned}
  \mathbf B_j\,\widehat{\mathbf t}^{(j)} &= \mathbf u_j
    && \text{(partial outer-commitment)} \\
  \mathbf D_j\,\widehat{\mathbf e}^{(j)} &= \mathbf v_j
    && \text{(partial opening-commitment)} \\
  \sum_{i\in\mathcal I_j} c_i\operatorname{Rec}_{\mathrm{open}}(\widehat{\mathbf e}_i)
    &= L(\mathbf z^{(j)})
    && \text{(partial folded-evaluation consistency)} \\
  \sum_{i\in\mathcal I_j} \iota(c_i)\,\mathbf G_{b,n_A}\widehat{\mathbf t}_i
    &= \mathbf A\,\mathbf z^{(j)}
    && \text{(partial folded-commitment consistency)}
\end{aligned}
$$

The third row follows from $L(\iota(c_i)\mathbf s_i)=c_ie_i$ and the linearity
of $L$. Write $\mathbf c^{(j)\,\top}\otimes\operatorname{Rec}_{\mathrm{open}}$
for the map on its left side and
$\iota(\mathbf c^{(j)})^\top\otimes\mathbf G_{b,n_A}$ for the map on the left
side of the fourth row. Stacking the four rows, each in its own ring, $P_j$
has a local partial root relation

$$
\begin{bmatrix}
  \mathbf D_j & 0 & 0 \\
  0 & \mathbf B_j & 0 \\
  \mathbf c^{(j)\,\top}\otimes\operatorname{Rec}_{\mathrm{open}} & 0 & -L \\
  0 & \iota(\mathbf c^{(j)})^\top\otimes\mathbf G_{b,n_A} & -\mathbf A
\end{bmatrix}
\begin{bmatrix}
  \widehat{\mathbf e}^{(j)} \\ \widehat{\mathbf t}^{(j)} \\ \mathbf z^{(j)}
\end{bmatrix}
=
\begin{bmatrix}
  \mathbf v_j \\ \mathbf u_j \\ 0 \\ \mathbf 0
\end{bmatrix}.
$$

Denote the block matrix by $\mathbf M_j$, the local witness by
$\mathbf w_j := (\widehat{\mathbf e}^{(j)},\widehat{\mathbf t}^{(j)},\mathbf z^{(j)})$,
and the right-hand side by $\mathbf h_j := (\mathbf v_j,\mathbf u_j,0,\mathbf 0)$, so
the local relation is compactly $\mathbf M_j\mathbf w_j = \mathbf h_j$.

### The virtually shared root relation

The verifier never sees the local $\mathbf h_j$; it sees only the public output
$\mathbf h = (\mathbf v,\mathbf u,0,\mathbf 0)$, whose entries are the sums of the
partial values, $\mathbf v = \sum_j\mathbf v_j$ and $\mathbf u = \sum_j\mathbf u_j$.
So instead of exposing the local right-hand sides, the machines jointly prove **one
virtually shared root relation**. With
$\mathbf w := (\mathbf w_0,\dots,\mathbf w_{\mathcal M-1})$ and the horizontal
concatenation $\mathbf M := [\mathbf M_0 \mid \dots \mid \mathbf M_{\mathcal M-1}]$,

$$
  \mathbf M\mathbf w
  = \sum_{j=0}^{\mathcal M-1}\mathbf M_j\mathbf w_j
  = \sum_{j=0}^{\mathcal M-1}\mathbf h_j
  = \mathbf h.
$$

The relation is *virtually* shared because no machine materialises the full witness
$\mathbf w$ or the full matrix $\mathbf M$: $P_j$ holds only $\mathbf w_j$ and its
column blocks $\mathbf B_j,\mathbf D_j$, while the verifier checks the single public
right-hand side obtained by summing the partial commitments.

## Ring-switch lift and next-level commitment

The displays in this section and the next one show the lift for rows over one
ring $R_q$ with one modulus $X^d+1$. The complete relation has more row kinds,
and each is lifted in its own ring. The packed consistency row is lifted in
$E[U]$ with modulus $U^s+1$; its quotient is the
[packing quotient](./root-fold-ring-switch.md#the-packing-consistency-quotient)
$Q_{\mathrm{pack}}$. A compressed payload adds the F and H rows with their own
quotients, and mixed ring dimensions give each row its native modulus. The
schedule gives every multi-chunk fold the quotient-lift relation mode; it
admits reduced evaluation only for single-chunk folds. In every case the
quotient is a linear function of the witness once the challenges are fixed.
The single-ring displays therefore show the complete structure: the global
quotient is the sum of the per-machine contributions, and the witness holds
one shared quotient range, not one per machine.

Each machine lifts its partial relation $\mathbf M_j\mathbf w_j = \mathbf h_j$ from
$R_q = \mathbb Z_q[X]/(X^d+1)$ to $\mathbb Z_q[X]$: there is a unique quotient
$\mathbf r_j \in (\mathbb Z_q^{<d}[X])^n$, with $n = n_A + n_B + n_D + 1$, such that

$$
  \mathbf M_j\mathbf w_j = \mathbf h_j + (X^d+1)\,\mathbf r_j
  \qquad\text{over } \mathbb Z_q[X].
$$

Summing the local identities, the global quotient is the coefficient-wise modular
sum $\mathbf r = \sum_j \mathbf r_j$, just as the global commitments are sums of the
partial commitments. The aggregated quotient is digit-decomposed as
$\widehat{\mathbf r} := \mathbf G_{b,n}^{-1}(\mathbf r)$, recovering the same
ring-switch witness shape as the single-machine protocol — except the root witness
stays written as the concatenation of the per-machine witnesses.

Let $\mathbf w' := (\mathbf w \,\|\, \widehat{\mathbf r})$ be the extended witness.
The lifted relation is $\mathbf M_{\mathrm{ext}}\mathbf w' = \mathbf h$ over
$\mathbb Z_q[X]$, with

$$
  \mathbf M_{\mathrm{ext}} := \bigl[\,\mathbf M \mid -(X^d+1)(I_n\otimes\mathbf g^\top)\,\bigr],
  \qquad \mathbf g^\top := (1,b,\dots,b^{\delta-1}).
$$

The extended witness is assembled into the coefficient table of the next-level
multilinear polynomial: each machine $P_j$ contributes its segment $\mathbf w_j$, and
a pre-assigned aggregation machine additionally contributes the quotient segment
$\widehat{\mathbf r}$. The machines then run the ordinary inner/outer commit at the
next level's parameters on their assigned portions and sum the **partial next-level
relation images** into one $\mathbf u'$. The aggregation machine compresses
$\mathbf u'$ and retains the two shared F digit vectors. The verifier sees exactly
the single-machine recursive interface: one 256-byte extended-witness payload and
one next-level opening claim.

## Ring-switch and row-batching

For $\mathbf M_{\mathrm{ext}}\mathbf w' = \mathbf h$ the verifier samples the
ring-switching challenge $\alpha \gets \mathbb F_{q^k}$ and the row-batching point
$\tau \gets \mathbb F_{q^k}^{\lceil\log_2 n\rceil}$. Viewing the machines as one
virtual prover, both sides evaluate the extended matrix at $X=\alpha$, obtaining
$\mathbf M_{\mathrm{ext}}(\alpha)$ and the target $\mathbf h(\alpha)$, i.e. $n$ row
equations. Writing the extended matrix as
$\mathbf M_{\mathrm{ext}} = [\mathbf M_0 \mid \dots \mid \mathbf M_{\mathcal M-1} \mid \mathbf M_\Delta]$
with the quotient block $\mathbf M_\Delta := -(X^d+1)(I_n\otimes\mathbf g^\top)$, the
row equations at $X=\alpha$ become

$$
  \sum_{j=0}^{\mathcal M-1}\mathbf M_j(\alpha)\mathbf w_j(\alpha)
  + \mathbf M_\Delta(\alpha)\,\widehat{\mathbf r}(\alpha)
  = \mathbf h(\alpha).
$$

The verifier batches the $n$ rows with the equality weights
$\widetilde{\mathrm{eq}}(\tau,i)$. For machine $P_j$ define its local
$\tau$-weighted row combination
$m_\tau^{(j)}(x) := \sum_i \widetilde{\mathrm{eq}}(\tau,i)\,\mathbf M_j(\alpha)(i,x)$,
and similarly $m_\tau^{(\Delta)}(x)$ for the quotient block; the batched target is
$V_\alpha := \sum_i \widetilde{\mathrm{eq}}(\tau,i)\,h_i(\alpha)$. Let
$\widetilde\alpha(y)$ be the multilinear extension of $(1,\alpha,\dots,\alpha^{d-1})$,
and let $\Omega_j$, $\Omega_\Delta$ denote the next-level witness-table coordinates
of $\mathbf w_j$ and $\widehat{\mathbf r}$. The full batched claim is

$$
  \sum_{j=0}^{\mathcal M-1}\sum_{(x,y)\in\Omega_j}
    \widetilde{\mathbf w}_j(x,y)\,\widetilde\alpha(y)\,m_\tau^{(j)}(x)
  + \sum_{(x,y)\in\Omega_\Delta}
    \widetilde{\widehat{\mathbf r}}(x,y)\,\widetilde\alpha(y)\,m_\tau^{(\Delta)}(x)
  = V_\alpha.
$$

Compared with the single-machine claim
$\sum_{(x,y)} \widetilde{\mathbf w}'(x,y)\,\widetilde\alpha(y)\,m_\tau(x) = V_\alpha$,
the distributed claim simply partitions both witness-dependent factors. Machine
$P_j$ owns a subset $\Omega_j$ of the next-level witness table and the matching
restriction $m_\tau^{(j)}$ of the row-batched weight, so no machine materialises the
full witness table or the full weight function. In the subsequent sum-check the same
decomposition holds round by round: each machine computes the round-polynomial
contribution for the coordinates it owns, and the machines aggregate only the small
round-polynomial coefficients. The verifier sees the same batched claim as in the
single-machine protocol.

## Recap against the three criteria

- **Distributable witness.** The block partition gives each machine a
  disjoint contiguous share, and every step — inner/outer/opening commit, the partial
  fold, the quotient lift, the next-level commit, and the sum-check — is computed
  per machine over its own blocks.
- **Low communication.** Only short objects ever cross machines: the $n_B/n_D$-sized
  partial commitments $\mathbf u_j,\mathbf v_j$, the scalar claims $y_j$, the
  coefficient-wise quotient sum, and the small per-round sum-check messages. The
  large items — the inner/opening digits and, decisively, the folded responses
  $\mathbf z_j$ — never leave their machine. Keeping a per-machine $\mathbf z_j$ is
  exactly the change that removes the one all-reduce that would otherwise dominate
  communication.
- **Net speed-up.** Per-machine $\mathbf z_j$ is kept only for the constant block of
  leading (large) rounds; afterward the prover reverts to the single-machine
  protocol, so the witness-growth and extra fold levels are bounded and the
  distributed parallelism translates into an end-to-end win.
