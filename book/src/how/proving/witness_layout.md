# GPU-oriented witness layout

The witness layout determines how we view and index the witness, and therefore
how the relations we plan to prove and verify must be structured. This chapter
proposes two witness layouts and compares their prover and verifier costs. It
then considers a third approach that keeps the original relation but proves its
structured matrix evaluation with an auxiliary sum-check.

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
   digits on the fly. The prover must retain a decomposed-witness
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

## Proposal 3: prove the structured matrix evaluation

The third proposal keeps the original relation and its relation sum-check
unchanged. In particular, it does not split the relation into separate
challenge, gadget, and witness claims. Instead, it adds an auxiliary sum-check
whose purpose is to prove the structured relation-matrix evaluation that the
verifier would otherwise compute directly.

Let $j\in[0,\delta)$ be the digit index, $c\in[0,C)$ the claim index, and
$b\in[0,B)$ the block index. Write $\mathsf{dig}[j]$ for the gadget weight of
digit $j$; for the usual gadget vector, $\mathsf{dig}[j]=g^j$. The local flat
index depends on the selected witness layout:

$$
\begin{aligned}
  i_1(j,c,b) &= b+B(c+Cj)
    &&\text{for Proposal 1},\\
  i_2(j,c,b) &= j+\delta(b+Bc)
    &&\text{for Proposal 2}.
\end{aligned}
$$

If the witness segment begins at column $\mathsf{offset}$, its global column is
$\mathsf{offset}+i_\ell(j,c,b)$ for layout $\ell\in\{1,2\}$. At the random
column point $r$ produced by the original relation sum-check, define

$$
\begin{aligned}
Q_\ell(r)
  = \sum_{j=0}^{\delta-1}
    \sum_{c=0}^{C-1}
    \sum_{b=0}^{B-1}
      &\mathsf{challenge}[c][b]\,
       \mathsf{dig}[j] \\
      &{}\cdot
       \operatorname{eq}\bigl(
         \mathsf{offset}+i_\ell(j,c,b),r
       \bigr).
\end{aligned}
$$

This is exactly the structured challenge-and-digit contribution of the
relation matrix at $r$. The prover claims a value $q=Q_\ell(r)$, and the
original relation sum-check uses $q$ in its terminal check instead of requiring
the verifier to compute this matrix contribution directly. The auxiliary
sum-check proves that the claimed $q$ equals the sum above.

### Batching with the original relation sum-check

The original relation claim and the auxiliary matrix-evaluation claim are
transcript-batched with a random coefficient. This binds the value $q$ used by
the original terminal check to the value proved by the auxiliary sum-check, so
the prover cannot choose them independently.

There is an important ordering constraint: the auxiliary claim depends on $r$,
and $r$ is only known after the original relation sum-check has sampled its
round challenges. Therefore, the auxiliary claim cannot naively share the same
rounds as the original sum-check from the beginning. The protocol first fixes
$r$, then proves $Q_\ell(r)$, and batches the resulting verification
obligations at the terminal-claim level.

The original relation, witness layout, and relation sum-check polynomial remain
unchanged. The added prover work is the auxiliary sum-check; the intended
benefit is that the verifier no longer performs the full direct structured
matrix evaluation.

## Current design status

No design is selected in this discussion yet. Proposal 1 prioritizes the
existing verifier evaluation strategy. Proposal 2 prioritizes GPU-local
execution, on-the-fly decomposition, and a future streaming prover, at the cost
of a slower verifier under the current relation sum-check. Proposal 3 keeps
that relation unchanged and adds an auxiliary proof of its structured matrix
evaluation; its final-point evaluator remains to be specified.
