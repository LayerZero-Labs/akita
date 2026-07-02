# The distributed prover

This chapter describes a distributed variant of the Akita prover: how the witness
is split across many machines, which parts of one folding step are local, and the
single protocol change — keeping a *per-machine* folded response — that lets the
machines avoid shuffling large witnesses between each other. The machines can be
separate GPUs, accelerators, or nodes; nothing below is specific to GPUs.

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
$N=3$, so $\mathcal M = 8$ machines). Each machine owns $1/\mathcal M$ of the
witness — one eighth when $N=3$.

## Setup: partitioning the witness

We spell the design out in the single-opening case (one opening point
$\mathbf r$, one commitment, one committed slot). Write $\mu = m + r$ and reshape
the coefficient table into $B := 2^r$ blocks of $M := 2^m$ ring elements,
$\mathbf f_i \in R_q^M$ for $i \in \{0,1\}^r$. The opening point induces the inner
evaluation vector $\mathbf a \in R_q^M$ and the block vector $\mathbf b \in R_q^B$,
and the evaluation is
$F(\mathbf r) = \mathbf b^\top[\mathbf a^\top\mathbf f_1,\dots,\mathbf f_B]$.

We partition the block index set $[B]$ into contiguous ranges
$[B] = \mathcal I_0 \sqcup \dots \sqcup \mathcal I_{\mathcal M-1}$, where
$\mathcal I_j := \{\, i \in [B] : \lfloor i\,\mathcal M / B \rfloor = j \,\}$, and
assign $\mathcal I_j$ to machine $P_j$. Larger index spaces are always partitioned
at the block level: for any $k \ge 1$ the block partition induces
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
$\mathbf B_j,\mathbf D_j$ hold exactly the columns indexed by $P_j$'s blocks, and
we restrict the block vector as
$\mathbf b^\top = [\mathbf b_0^\top \mid \dots \mid \mathbf b_{\mathcal M-1}^\top]$.

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

The global commitment is then just the sum
$\mathbf u = \sum_{j=0}^{\mathcal M-1}\mathbf u_j$. The large local witnesses
$\widehat{\mathbf t}^{(j)}$ never leave $P_j$; only the short $n_B$-vectors
$\mathbf u_j$ are summed (criteria 1 and 2).

### The opening commitment $\mathbf v$ and the claim $y$

The opening side is computed the same way. For every $i \in \mathcal I_j$, $P_j$
computes the local block evaluation $e_i := \langle\mathbf a,\mathbf f_i\rangle$,
digitises it to $\widehat{\mathbf e}_i := \mathbf G_{b,1}^{-1}(e_i) \in R_q^\delta$,
and — using only its column block $\mathbf D_j$ — forms the partial opening
commitment $\mathbf v_j := \mathbf D_j\,\widehat{\mathbf e}^{(j)} \in R_q^{n_D}$,
with $\widehat{\mathbf e}^{(j)} := (\widehat{\mathbf e}_i)_{i\in\mathcal I_j}$. The
global opening commitment is $\mathbf v = \sum_j \mathbf v_j$. Each machine also
forms its partial claim $y_j := \sum_{i\in\mathcal I_j} b_i e_i$, and the claimed
evaluation is $y = F(\mathbf r) = \sum_j y_j$.

So $\mathbf u$, $\mathbf v$, and $y$ all follow the same pattern: each machine does
the same inner/outer/opening computation as the single-machine prover, but only on
its own blocks, and the machines aggregate only the small commitment vectors and
scalar claims.

### The folded response $\mathbf z$ — the hard part

After the verifier samples the folding challenges, the single-machine prover would
form **one** global folded response

$$
  \mathbf z = \sum_{i=1}^{B} c_i\,\mathbf s_i \in R_q^{M\delta}.
$$

But each machine only holds the blocks in $\mathcal I_j$, so it can only compute a
*partial* fold $\mathbf z_j := \sum_{i\in\mathcal I_j} c_i\,\mathbf s_i \in R_q^{M\delta}$
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

## Why only the first few rounds keep per-machine $\mathbf z$

Keeping a separate $\mathbf z_j$ per machine is *not* free, so we do it only for a
constant number of leading rounds. Two reasons:

1. **Proof size and fold depth.** The $\mathcal M$ separate $\mathbf z_j$ enlarge
   the next witness relative to a single $\mathbf z$. Carrying that growth into the
   tail and final rounds would inflate the proof size and the number of fold levels,
   so those rounds keep the single-response protocol.
2. **Where the cost actually is.** Only the first few rounds are large enough to be
   worth distributing. After they shrink the witness, the remaining rounds are cheap
   and run fine on a single machine.

So the prover fixes a **constant cutover round**: the first rounds run the
distributed, per-machine-$\mathbf z$ protocol across the machines; after the cutover
the prover collapses to a single machine, which keeps the current single-response
Akita protocol unchanged.

## The next-round witness and the relation change

After the distributed rounds, each machine $P_j$ holds its own
$(\widehat{\mathbf t}^{(j)},\,\widehat{\mathbf e}^{(j)},\,\mathbf z_j,\,\mathbf r_j)$
and can proceed to the next round on its own. Supporting this requires changing the
witness format and the relation matrix $\mathbf M$ so the relation is stated over the
*concatenation* of the per-machine witnesses rather than over one global folded
response. The rest of this chapter makes that change precise.

### The per-machine partial root relation

Each machine holds its partial witness and the matching partial commitments, plus
the fold of its own blocks. Over $R_q$, machine $P_j$ satisfies:

$$
\begin{aligned}
  \mathbf B_j\,\widehat{\mathbf t}^{(j)} &= \mathbf u_j
    && \text{(partial outer-commitment)} \\
  \mathbf D_j\,\widehat{\mathbf e}^{(j)} &= \mathbf v_j
    && \text{(partial opening-commitment)} \\
  (\mathbf c^{(j)\,\top}\otimes\mathbf G_{b,1})\,\widehat{\mathbf e}^{(j)}
    &= \langle\mathbf a,\,\mathbf G_{b,|\mathcal I_j|}\mathbf z^{(j)}\rangle
    && \text{(partial folded-evaluation consistency)} \\
  (\mathbf c^{(j)\,\top}\otimes\mathbf G_{b,n_A})\,\widehat{\mathbf t}^{(j)}
    &= \mathbf A\,\mathbf z^{(j)}
    && \text{(partial folded-commitment consistency)}
\end{aligned}
$$

Stacking these, $P_j$ has a local partial root relation

$$
\begin{bmatrix}
  \mathbf D_j & 0 & 0 \\
  0 & \mathbf B_j & 0 \\
  \mathbf c^{(j)\,\top}\otimes\mathbf G_{b,1} & 0 & -\mathbf a^\top\mathbf G_{b,M} \\
  0 & \mathbf c^{(j)\,\top}\otimes\mathbf G_{b,n_A} & -\mathbf A
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
commitments** into one $\mathbf u'$. The verifier sees exactly the single-machine
recursive interface: one extended-witness commitment and one next-level opening claim.

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

- **Distributable witness.** The block partition gives each machine a disjoint
  $1/\mathcal M$ share, and every step — inner/outer/opening commit, the partial
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
  leading (large) rounds; after the cutover the prover reverts to the single-machine
  protocol, so the witness-growth and extra fold levels are bounded and the
  distributed parallelism translates into an end-to-end win.
