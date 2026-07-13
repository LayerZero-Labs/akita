# GPU-oriented witness layout

The witness layout determines how we view and index the witness, and therefore
how the relations we plan to prove and verify must be structured. This chapter
proposes two witness layouts and compares their prover and verifier costs. It
then presents a third, layout-independent approach that keeps verification
efficient regardless of the chosen witness layout.

## Assumptions

We only analyze the case of one machine. With multiple machines, each machine
is assigned its own blocks and uses the same layout for its local part of the
witness. It is therefore enough to understand how the prover and verifier are
implemented for one machine under each layout, how efficient each
implementation is, and what trade-offs it introduces.

Let:

- $B$ be the number of blocks in the relevant witness segment. We assume that
  $B$ is always a power of two, as it is in Akita;
- $C$ be the number of claims (equivalently, the number of polynomials);
- $\delta$ be the number of digits into which each ring element must be
  decomposed.

When we write
$\widehat w[\mathsf{digit},\mathsf{claim},\mathsf{block}]$, the block axis is
innermost. When we write
$\widehat w[\mathsf{claim},\mathsf{block},\mathsf{digit}]$, the digit axis is
innermost. The question is which of these views of $\widehat w$ should define
the witness layout. This choice also changes the corresponding relation matrix
$M$, with important consequences for both prover and verifier efficiency. This
chapter compares those consequences.

## Proposal 1: digit, claim, block

The first proposal has block axis as innermost (Current implementation). Its flat index is

$$
  \mathsf{block}
  + B\bigl(\mathsf{claim} + C\cdot\mathsf{digit}\bigr).
$$

### Strength: fast verifier

The general optimization is explained in
[Matrix evaluation at a point](../verifying/matrix_evaluation.md). Here we
illustrate it with a small example.

Assume $B=4$, $C=1$, and $\delta=3$. Consider the consistency-row portion of the
relation matrix. At a verifier challenge $r$, its multilinear-extension
evaluation has the form

$$
  \sum_i \operatorname{eq}(i,r)\,M[\mathsf{row},i],
$$

where the matrix coefficient at the column for `(digit, claim, block)` is

$$
  M[\mathsf{row},i]
  =
  \mathsf{challenge}[\mathsf{claim}][\mathsf{block}]
  \cdot g^{\mathsf{digit}}.
$$

There are $B C \delta=12$ live columns, embedded in a four-bit column
hypercube. Writing each index as two high bits followed by the two low block
bits, the contribution is

$$
\begin{aligned}
&\mathsf{challenge}[0][0]g^0\operatorname{eq}(0000,r)
 +\mathsf{challenge}[0][1]g^0\operatorname{eq}(0001,r)
 +\mathsf{challenge}[0][2]g^0\operatorname{eq}(0010,r)\\
{}+{}&\mathsf{challenge}[0][3]g^0\operatorname{eq}(0011,r)
 +\mathsf{challenge}[0][0]g^1\operatorname{eq}(0100,r)
 +\mathsf{challenge}[0][1]g^1\operatorname{eq}(0101,r)\\
{}+{}&\mathsf{challenge}[0][2]g^1\operatorname{eq}(0110,r)
 +\mathsf{challenge}[0][3]g^1\operatorname{eq}(0111,r)
 +\mathsf{challenge}[0][0]g^2\operatorname{eq}(1000,r)\\
{}+{}&\mathsf{challenge}[0][1]g^2\operatorname{eq}(1001,r)
 +\mathsf{challenge}[0][2]g^2\operatorname{eq}(1010,r)
 +\mathsf{challenge}[0][3]g^2\operatorname{eq}(1011,r).
\end{aligned}
$$

Because the block occupies the two low bits, first compute

$$
\begin{aligned}
S={}&\mathsf{challenge}[0][0]\operatorname{eq}(00,r_{\mathrm{low}})
 +\mathsf{challenge}[0][1]\operatorname{eq}(01,r_{\mathrm{low}})\\
&+\mathsf{challenge}[0][2]\operatorname{eq}(10,r_{\mathrm{low}})
 +\mathsf{challenge}[0][3]\operatorname{eq}(11,r_{\mathrm{low}}).
\end{aligned}
$$

Computing these block summaries for all claims costs $O(B C)$. The full sum is
then

$$
  S\Bigl(
      g^0\operatorname{eq}(00,r_{\mathrm{high}})
    + g^1\operatorname{eq}(01,r_{\mathrm{high}})
    + g^2\operatorname{eq}(10,r_{\mathrm{high}})
  \Bigr).
$$

Combining the high-bit equality weights and gadget powers for all digits and
claims costs $O(\delta C)$. The total cost is therefore
$O(B C+\delta C)$.

This example assumes that the witness segment starts at offset zero. A nonzero
offset gives the same asymptotic computation, but the low-bit addition may
carry into the high bits. The verifier then keeps two summaries: one for carry
zero and one for carry one. This detail is not important for the purpose of
this chapter; see
[Matrix evaluation at a point](../verifying/matrix_evaluation.md) for the full
explanation.

### Challenge: GPU implementation

In this layout, two consecutive witness elements do not belong to the same
block. The decomposed digits of one block are separated rather than stored
together. For a fixed claim and block, the gap between consecutive digits is

$$
  \mathsf{index}(\mathsf{digit}+1)
  -
  \mathsf{index}(\mathsf{digit})
  = B C.
$$

The natural unit of work for a GPU thread is one block, so this separation
introduces several challenges.

1. **Decomposition must be materialized.** When a thread decomposes a block, its
   output digits belong to memory locations separated by $B C$. Unlike Proposal
   2, the thread cannot simply decompose a ring element and consume all its
   adjacent digits on the fly. The prover must retain a decomposed-witness
   buffer. In the simple memory model where both the original witness and its
   decomposed representation remain resident, this results in roughly twice the
   witness memory usage.

2. **Sum-check binding reads the stored decomposition.** An efficient GPU
   sum-check assigns each thread two adjacent evaluations, binds them with the
   verifier challenge, and writes the result. To expose the witness in this
   pairwise layout, the decomposed values must already be stored in witness
   order. Each thread therefore reads the materialized decomposed witness from
   memory instead of reading an undecomposed ring element and decomposing it on
   the fly, as in Proposal 2.

3. **Relation-matrix binding needs non-sequential preprocessing.** When binding
   the relation matrix $M$, the required preprocessed values are not laid out
   sequentially for each thread. Threads may therefore perform irregular reads.
   Keeping this step efficient may require multiple preprocessed
   representations, each stored in the order needed by a particular binding
   phase. This increases preprocessing and storage requirements.

## Proposal 2: claim, block, digit

The second proposal changes the ordering to:

```text
claim → block → digit
```

The digit axis is innermost. Its flat index is

$$
  \mathsf{digit}
  + \delta\bigl(\mathsf{block} + B\cdot\mathsf{claim}\bigr).
$$

For a fixed `(claim, block)` pair, all $\delta$ digits form one contiguous
range. Moving to the next block advances by $\delta$ cells, and moving to the
next claim advances by $B\delta$ cells. Each GPU is assigned a contiguous range
of blocks and processes the ring elements in that range.

### Strength: natural GPU execution

This layout is designed around the following GPU execution model.

1. **Contiguous work per thread.** Each GPU thread is assigned a contiguous
   range of ring elements from one block.

2. **On-the-fly decomposition.** A thread reads the ring elements it needs
   directly from memory in their undecomposed form. When digit decomposition is
   required, the thread decomposes each ring element on the fly and immediately
   consumes its contiguous digits. The prover therefore does not need to
   materialize and store the fully decomposed blocks. This reduces memory usage
   and avoids the extra reads and writes of a digit-expanded witness.

3. **Efficient sum-check binding.** Each thread owns a contiguous portion of the
   witness. During a sum-check binding round, it reads neighboring ring elements,
   binds each pair using the verifier challenge, and writes the resulting
   evaluation. This contiguous pairwise access makes witness binding natural
   and efficient on a GPU.

4. **Natural path to streaming.** The same contiguous block range is both the
   distribution unit and the processing unit. A GPU can receive ring elements,
   decompose them only when needed, and consume them without first constructing
   a global digit-major representation. This makes a future streaming prover
   more natural and easier to reason about.

### Weakness: verifier cost and tensor challenges

Use the same example: $B=4$, $C=1$, and $\delta=3$. Under the
`claim → block → digit` layout, the three digits of each block are adjacent.
The contribution is now

$$
\begin{aligned}
&\mathsf{challenge}[0][0]g^0\operatorname{eq}(0000,r)
 +\mathsf{challenge}[0][0]g^1\operatorname{eq}(0001,r)
 +\mathsf{challenge}[0][0]g^2\operatorname{eq}(0010,r)\\
{}+{}&\mathsf{challenge}[0][1]g^0\operatorname{eq}(0011,r)
 +\mathsf{challenge}[0][1]g^1\operatorname{eq}(0100,r)
 +\mathsf{challenge}[0][1]g^2\operatorname{eq}(0101,r)\\
{}+{}&\mathsf{challenge}[0][2]g^0\operatorname{eq}(0110,r)
 +\mathsf{challenge}[0][2]g^1\operatorname{eq}(0111,r)
 +\mathsf{challenge}[0][2]g^2\operatorname{eq}(1000,r)\\
{}+{}&\mathsf{challenge}[0][3]g^0\operatorname{eq}(1001,r)
 +\mathsf{challenge}[0][3]g^1\operatorname{eq}(1010,r)
 +\mathsf{challenge}[0][3]g^2\operatorname{eq}(1011,r).
\end{aligned}
$$

In Proposal 1, incrementing the block changed only the two low bits. This let
the verifier factor every equality weight into a reusable block-dependent low
part and a digit-dependent high part. That factorization is no longer available
here. Incrementing the block advances the flat index by $\delta=3$, so the
block boundaries cross the binary low/high-bit boundary: block 1, for example,
occupies indices `0011`, `0100`, and `0101`. Neither a fixed low-bit window nor
a fixed high-bit window identifies the block independently of the digit.

As a result, the verifier cannot reuse one block summary across all digits in
the same way as Proposal 1. Without another layout-specific optimization, it
must evaluate the equality weight separately for every
`(claim, block, digit)` cell. The verifier cost is therefore

$$
  O(B C\delta),
$$

instead of $O(B C+\delta C)$. When $B$ is the dominant dimension, this is a
constant factor of approximately $\delta$ slower.

This has several practical consequences:

1. **More verifier work.** For this portion of the relation-matrix evaluation,
   verification is roughly $\delta$ times slower because the verifier loses the
   reusable block/digit split from Proposal 1. This factor applies to the
   affected matrix-evaluation work, not necessarily to the entire verifier.

2. **Uncertain fourth-root optimization.** It is not yet clear whether the
   fourth-root verifier optimization based on tensor-structured challenges can
   be preserved in this layout, or how its factorization should be expressed.
   A pessimistic analysis should assume that this property is lost until a
   compatible tensor-challenge evaluation is derived.

3. **Measured overhead.** For `nv = 36`, current benchmarks show about 10 ms of
   additional verifier time: verification increases from approximately 35 ms
   to 45 ms. There are still several possible optimizations that may reduce this
   overhead.

4. **Recursion impact.** The additional verifier work may matter more when
   proving verification, such as when a RISC-V zkVM executes the verifier for
   recursive aggregation. In that setting, the extra matrix-evaluation work can
   translate into a noticeable increase in RISC-V cycles.

## Proposal 3: offload structured relation work to the prover

The first two proposals change the witness layout and then ask the verifier to
evaluate the resulting relation matrix efficiently. A third option is to change
the relation sum-check itself. Instead of making the verifier evaluate a
structured matrix entry that already combines the challenge, gadget digit, and
witness layout, expose those components as separate polynomials in the
sum-check.

This approach can reduce verifier work regardless of which witness layout is
selected.

### Current relation sum-check

Schematically, for each row of the relation the current protocol proves a claim
of the form

$$
  \sum_{i\in\mathsf{cols}}
  \operatorname{eq}(r,i)\,M[\mathsf{row},i]\,W[i]
  = Y[\mathsf{row}].
$$

The matrix polynomial $M[\mathsf{row},i]$ already contains the structured
factors associated with that row. The prover uses it inside the sum-check, and
the verifier must evaluate its multilinear extension at the final random point.
The cost and shape of that evaluation depend on the witness layout.

Proposal 3 does not keep every structured factor inside one matrix polynomial.
It gives each factor its own polynomial and lets the prover execute the
corresponding product sum-check. At the final random point, the verifier
evaluates the individual public factors instead of reconstructing the combined
matrix polynomial.

### Commitment rows remain unchanged

The two commitment relations remain in their current form:

$$
  B\widehat t=u,
  \qquad
  D\widehat e=v.
$$

They contribute two sum-check claims, one for each commitment check. This
proposal does not change their relation matrices or move additional work into
them.

### Split each consistency relation into two claims

Akita has two consistency relations:

$$
\begin{aligned}
  (c^\top\otimes G)\widehat e
    &= \langle a,G\widehat z\rangle, \\
  (c^\top\otimes G_{n_A})\widehat t
    &= A\widehat z.
\end{aligned}
$$

Instead of representing each equality as one matrix relation $MW=Y$, prove its
left-hand side and right-hand side with separate sum-check claims. The verifier
then checks that the two resulting claims are equal. This produces four
consistency claims:

1. the challenge/gadget contribution over $\widehat e$;
2. the opening-point/gadget contribution over $\widehat z$;
3. the challenge/gadget contribution over $\widehat t$;
4. the $A$-matrix contribution over $\widehat z$.

### Example: the challenge component over `t_hat`

Under Proposal 1, relate a flat witness index $i$ to digit $j$, claim $c$, and
block $b$ by

$$
  i_1(j,c,b)=b+B(c+Cj).
$$

The challenge-side `t_hat` claim is then

$$
  \sum_{j=0}^{\delta-1}
  \sum_{c=0}^{C-1}
  \sum_{b=0}^{B-1}
    \operatorname{eq}\bigl(r,i_1(j,c,b)\bigr)
    \,\mathsf{challenge}[c][b]
    \,g^j
    \,\widehat t\bigl[i_1(j,c,b)\bigr]
  =Y_{\widehat t}.
$$

Under Proposal 2, only the index map changes:

$$
  i_2(j,c,b)=j+\delta(b+Bc).
$$

The challenge, gadget, equality, and witness factors remain separate
polynomials in the sum-check:

$$
  \operatorname{eq}(r,i)
  \cdot \widetilde{\mathsf{challenge}}(i)
  \cdot \widetilde g(i)
  \cdot \widetilde{\widehat t}(i).
$$

The same construction applies to the other structured consistency components.
The prover performs the product sum-check. At the final random point, the
verifier checks:

- the witness evaluation $\widetilde W(r)$;
- the equality-polynomial evaluation;
- the gadget-digit evaluation;
- the challenge-polynomial evaluation.

As in Akita today, the witness evaluation is carried into and proved by the
next recursive round. The other factors are public and can be evaluated
directly by the verifier. In particular, the challenge and digit factors are no
longer fused into one layout-dependent matrix evaluation. This is what makes
the verifier computation potentially independent of whether block or digit is
the innermost witness axis.

That independence is a design requirement, not an automatic consequence of
writing the factors separately. With Proposal 2 and a non-power-of-two
$\delta$, the flat map $i_2(j,c,b)=j+\delta(b+Bc)$ still mixes the binary block
and digit coordinates. Padding the challenge table to the witness length does
not by itself guarantee that its multilinear extension can be evaluated
succinctly. A complete construction must provide either:

- an efficient virtual-polynomial evaluator for each padded factor in the flat
  witness domain; or
- a sum-check over explicit `(digit, claim, block)` variables together with an
  efficient wiring argument that links that domain to the single witness
  evaluation.

Without one of these constructions, the verifier may simply recover the
layout-dependent work that Proposal 3 is intended to remove.

### Six claims, batched into one sum-check

Assuming rank one and one batched claim for each row family, the proposal has
six logical sum-check claims:

- two unchanged commitment claims;
- two claims for the two sides of the opening consistency relation;
- two claims for the two sides of the commitment consistency relation.

All six can use the same sum-check challenges and be combined with transcript
batching coefficients. Consequently, the number of logical claims does not
require six independent round transcripts or six independent proofs.

To obtain literally no proof-size overhead, the batched construction must also
preserve the current maximum per-variable degree and avoid serializing extra
terminal evaluations. Splitting $M$ into several multiplicative factors can
increase the degree of a naive product sum-check, even though batching the six
claims itself is free. A concrete protocol must therefore use a product
sum-check specialization or equivalent formulation that keeps the existing
round-message size.

### Padding to one witness domain

Using one shared witness evaluation requires all six claims to run over the
same domain and receive the same sum-check point. The consistency factors—
including challenge, gadget digit, and any structured selectors—must therefore
be padded to the full length of $W$.

The padded entries are zero and need not be materialized. The prover can skip
the inactive ranges when accumulating each round polynomial. If the structured
factor evaluators satisfy the layout-independence requirement above, padding
does not add asymptotic prover or verifier work and allows every claim to finish
with the same single evaluation $\widetilde W(r)$.

## Current design status

No design is selected in this discussion yet. Proposal 1 prioritizes the
existing verifier evaluation strategy. Proposal 2 prioritizes GPU-local
execution, on-the-fly decomposition, and a future streaming prover, at the cost
of a slower verifier under the current relation sum-check. Proposal 3 changes
that sum-check so the structured factors are checked separately, with the goal
of retaining Proposal 2's prover layout while avoiding its verifier penalty.
Its remaining design requirement is to show a concrete batching construction
that preserves the current sum-check degree and proof size, while also giving
the verifier succinct evaluations of the separated factors under either
witness layout.
