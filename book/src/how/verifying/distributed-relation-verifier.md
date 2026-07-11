# The distributed-relation verifier

The verifier is not distributed. It checks the relation produced by the
[distributed prover](../proving/distributed-prover.md), whose witness is
machine-major globally and block-fast inside every machine.

## Relation shape

For (W) machines, the verifier checks the horizontal concatenation

\[
M=[M_0\mid M_1\mid\cdots\mid M_{W-1}].
\]

Every (M_j) uses the same relation-row layout. Its columns address machine
(j)'s contiguous local witness

```text
[ z_j | e_j | t_j | r_j ].
```

The row axis is shared. Machine contributions add into the same consistency,
(A), (B), and (D) rows and the same public right-hand side.

## Opening-domain factorization

Let every local witness have live length (C) and virtual stride
(P=\operatorname{nextPowerOfTwo}(C)). The global opening index has local bits
followed by machine bits:

```text
[local witness bits | machine bits].
```

At a verifier point ((r_{\mathrm{local}},r_{\mathrm{machine}})), machine
(j)'s column weight factors as

\[
\operatorname{eq}(j,r_{\mathrm{machine}})
\cdot
\operatorname{eq}(x,r_{\mathrm{local}}).
\]

The local evaluator is the ordinary block-fast evaluator over machine (j)'s
blocks. Structural positions from (C) to (P-1) contribute zero without being
materialized.

## Component costs

The (e_j) and (t_j) segments partition the original blocks, so their total
heavy work is unchanged after summing across machines. The (z_j) segments are
full-size partial folds and therefore scale with (W); this is the intended
cost of avoiding the folded-response all-reduce.

Every machine also has a same-shaped local quotient segment. The verifier adds
all local quotient contributions to the shared rows. These quotient values are
not replicas and are not expected to agree.

The setup matrices are common seed-derived views. The verifier first combines
the structured weights contributed by all machines, then scans each setup entry
once. Repeating the setup scan per machine would turn cheap ownership
bookkeeping into a dominant regression and is forbidden.

The trace term follows the same rule: build one weight per machine-local
(e_j) support, multiply by the machine equality factor, and sum. No trace path
may reinterpret the machine-major witness as globally block-fast or digit-fast.

## Verifier work contract

Distribution must not multiply the dominant verifier work by the machine count.
The verifier contracts the machine-prefix equality weights before running the
structured local evaluators:

- identical \(z\) and quotient coefficient formulas are evaluated once after
  their machine weights are summed;
- the \(e/t\) windows tile the original block axis, so their combined scan is
  linear in the original block count rather than \(W\) times that count;
- machine-weighted setup coefficients are combined before one shared
  \(A/B/D\) scan;
- trace support remains structured and never becomes a dense per-machine table.

Consequently, the number of setup entries scanned and setup alpha evaluations
must be exactly the single-chunk count. Tests record these operation counts.
The verifier may allocate structured tables proportional to one local stride,
the machine count, and the relation rows; it must not allocate the full padded
\(W P\) table or a dense distributed relation table.

The unavoidable overhead is small: contracting the machine prefix, replaying
any genuinely additional sum-check rounds, and handling the short local
quotient and range-check terms. The implementation is benchmark-gated against
the block-fast verifier on `origin/main`, which is the primary baseline. The
large universal digit-fast regressions are only a secondary comparison and are
not treated as an inherent cost of distributed proving.

## Native evaluation contract

The implementation derives relation, quotient, setup, and trace columns directly
in machine-major order. A conceptual column permutation proves equivalence to the
single-machine relation, but no production path constructs or applies such a
permutation.

The verifier layout authority must provide:

- ordered machine witnesses;
- their common live length and virtual stride;
- local `z/e/t/r` ranges;
- local block count and block length;
- input and output machine counts at the cutover;
- work and allocation bounds.

Malformed geometry is rejected before allocation with `AkitaError`.

## Sum-check replay

The machines jointly produce one round polynomial by summing local polynomial
contributions before each challenge. The verifier therefore runs the ordinary
single transcript and ordinary sum-check verifier. Its final relation evaluation
uses the same machine-weighted sum that defined the prover's round tables.

Correctness requires every machine to restrict the same padded global MLE.
Independent local padding conventions or local transcripts would define a
different polynomial and are rejected by the descriptor-bound layout.

## Cutover

The cutover fold has hierarchical input and single-machine output. The verifier
evaluates its input with this chapter's machine-major relation, but derives the
next commitment and opening shape from the ordinary single-machine layout. It
must not infer both from one ambiguous chunk-count field.

## Multi-group status

The future product layout uses one outer machine axis and group-local segments
inside each machine. Transcript group order remains independent of relation
group order. Until a dense `groups x machines` oracle and end-to-end test land,
the verifier rejects multi-group combined with (W>1).

The normative implementation requirements and acceptance tests are in
[`machine-major-distributed-prover.md`](../../../../specs/machine-major-distributed-prover.md).
