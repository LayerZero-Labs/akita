# The distributed prover

Akita distributes the leading fold levels across (W=2^k) machines. Each
machine owns complete relation blocks, keeps its large witnesses local, and
communicates only short commitments and sum-check messages. The implementation
contract is specified in
[`machine-major-distributed-prover.md`](../../../../specs/machine-major-distributed-prover.md).

## Block ownership

Write the committed table as (B=2^r) blocks
(mathbf f_iin R_q^M). The machines partition the block indices into equal
contiguous windows

\[
[B]=\mathcal I_0\sqcup\cdots\sqcup\mathcal I_{W-1}.
\]

Machine (P_j) holds every coefficient and digit belonging to the blocks in
(mathcal I_j). The public setup is seed-expanded, so each machine regenerates
the (A), (B), and (D) columns it needs.

For the outer and opening commitments, each machine computes partial images

\[
u_j=B_j\hat t_j,
\qquad
v_j=D_j\hat e_j,
\]

and the short public images add:

\[
u=\sum_j u_j,
\qquad
v=\sum_j v_j.
\]

The large (hat t_j) and (hat e_j) vectors never move between machines.

## Why the folded response stays local

After the fold challenge is sampled, machine (P_j) can compute only its
partial folded response

\[
z_j=\sum_{i\in\mathcal I_j}c_i s_i.
\]

Each (z_j) lives in the full ambient fold space. Summing the responses would
all-reduce the largest intermediate payload in the protocol. Akita instead
retains one (z_j) per machine during the leading distributed levels.

The norm bound is applied to each partial response separately. A partial sum is
not automatically bounded by the norm of the global sum because contributions
from different machines may cancel. The planner prices the partial-fold bound,
and the prover grinds or rejects until every (z_j) satisfies it.

## Machine-major, block-fast recursion

The next recursive witness is hierarchical:

```text
machine 0: [ z_0 | e_0 | t_0 | r_0 ]
machine 1: [ z_1 | e_1 | t_1 | r_1 ]
...
machine W-1: [ z_{W-1} | e_{W-1} | t_{W-1} | r_{W-1} ]
```

Every row above is one contiguous machine-local buffer. Within that buffer,
the local block index varies fastest. If a machine owns (Q) local blocks,
its address is

```text
local_index(position, local_block) = position * Q + local_block.
```

The machine axis is outside this local block-fast order. The implementation
stores a batch of local recursive witnesses. It does not concatenate them and
reinterpret the result as one monolithic block-fast witness.

For example, with two machines and two blocks per machine:

```text
machine 0: x_A x_B | y_A y_B | z_A z_B
machine 1: x_C x_D | y_C y_D | z_C z_D
```

Both machines own complete blocks and one contiguous local buffer. This differs
from a universal digit-fast layout: only the machine axis is outermost; blocks
remain the fastest axis inside a machine.

## Native local relations

Machine (P_j) satisfies a local relation with the complete global row set and
only its local columns:

\[
M_jw_j=h_j \pmod{X^d+1}.
\]

The partial right-hand sides sum to the public statement,
(sum_jh_j=h). The protocol proves one horizontally concatenated relation:

\[
[M_0\mid\cdots\mid M_{W-1}]
\begin{bmatrix}w_0\\[-2pt]\vdots\\[-2pt]w_{W-1}\end{bmatrix}
=h.
\]

The relation matrices are defined directly in machine-major order. There is no
materialized permutation or compatibility wrapper around the single-machine
matrix.

## One local quotient per machine

Machine (P_j) lifts its own relation before reduction:

\[
M_jw_j=h_j+(X^d+1)r_j.
\]

It digit-decomposes (r_j) locally and appends (hat r_j) to its witness. The
global lifted relation is

\[
\sum_j\left(M_jw_j-(X^d+1)G\hat r_j\right)=h.
\]

The quotient vectors have identical shapes but different values. They are local
quotient contributions, not replicas of a global quotient.

This construction relies only on linear gadget recomposition,
(G\hat r_j=r_j). It does not assume that digit decomposition is additive.
Indeed,

\[
G^{-1}\!\left(\sum_jr_j\right)
\ne
\sum_jG^{-1}(r_j)
\]

in general.

Keeping local quotients multiplies the short quotient segment by (W), but
removes quotient communication and makes all machine witnesses uniform. At the
large distributed levels this cost is small relative to the replicated folded
responses and the (e/t) data.

## Commitment and sum-check

Each machine recursively commits to its complete local blocks. The public next
commitment is the sum of the short partial commitments.

The machines run one sum-check transcript. In every round each machine computes
the polynomial contribution of its local witness, the coordinator sums the
coefficient vectors, and only the sum is absorbed before the next challenge is
sampled. Local transcripts are not permitted.

The verifier treats the witness as a machine-prefix domain followed by one local
block-fast domain. Setup weights from all machines are combined before the
shared setup is scanned, so the expensive matrix scan is not repeated per
machine.

## Execution model

The old multi-chunk path is a single-host simulation: the cluster backend types
delegate to the CPU backend, one process loops over all chunks, and that process
can retain the complete flattened witness. It remains useful as a correctness
oracle, but it is not the distributed prover.

The shipped distributed path has one coordinator and \(W\) worker processes.
The coordinator owns the Fiat--Shamir transcript, public schedule, ordered
collective results, and serialized proof. Worker \(j\) alone owns machine
\(j\)'s root blocks and recursive witnesses. Before cutover, the coordinator
must not materialize the union of the worker witnesses.

The initial runtime is intentionally small. The coordinator spawns persistent
worker processes and exchanges binary-framed messages over a minimal byte-channel
interface; the reference channel uses portable child stdin/stdout.
It broadcasts challenges, sums short commitment vectors and sum-check round
polynomials, combines acceptance bits during grinding, and gathers the smaller
output only at cutover. Phase tags and checked lengths prevent a worker message
from being consumed in the wrong protocol step.

Communication before cutover is limited to short commitments, scalar claims,
grinding decisions, range-subtree roots, constant-degree round polynomials, and
transcript challenges. The large \(z_j/e_j/t_j/r_j\) buffers remain local. A
witness-sized collective before cutover violates the protocol's implementation
contract.

Message encoding and the worker state machine do not depend on process handles
or operating-system-specific IPC, so another transport can be substituted
without changing prover arithmetic. This is enough to demonstrate separate ownership and measure serialization,
synchronization, communication volume, and coordinator memory without building
a general network stack. TCP, remote deployment, retries, discovery, transport
plugins, and Byzantine-worker robustness are deferred. The integration test
must nevertheless use separate operating-system processes and produce the same
proof bytes as the single-host machine-major oracle.

## Compact local domains

Machine buffers have one common live length (C), which need not be a power of
two. Their local opening stride is (P=\operatorname{nextPowerOfTwo}(C)); the
remaining positions are structural zero. Because (W) is a power of two,

\[
\operatorname{nextPowerOfTwo}(WC)=WP,
\]

so equal local zero suffixes do not enlarge the global opening domain. They are
not serialized or committed as witness data.

## Cutover

Only the leading expensive levels remain distributed. The cutover fold still
consumes the (W) native local witnesses and runs its large work distributively.
It then aggregates the smaller folded output and emits one ordinary
single-machine recursive witness. The suffix prover continues from that point.

The cutover is explicit. A machine-major concatenation is never passed to the
ordinary monolithic recursive backend.

At cutover, workers gather only the now-smaller output. Folded responses are
summed, partitioned output segments are assembled, and quotient coefficients
are summed before the ordinary global quotient is digit-decomposed. The planner
prices these bytes when choosing the cutover level.

## Multi-group openings

Commitment groups and machines are separate axes. The eventual combined layout
places every group's local segments inside each machine witness, in canonical
relation order, followed by that machine's local quotient. Digit-fast root hints
are read by block window and normalized at ring-switch assembly; they do not
force recursive witnesses to become digit-fast.

The current implementation rejects multi-group openings combined with more than
one machine. This guard remains until the singleton distributed construction is
complete and a dense `groups x machines` reference test passes.
