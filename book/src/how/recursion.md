# Recursion and proof shape

A fold turns an opening of a committed source into an opening of a new digit
witness. Akita repeats this reduction until the remaining response is small
enough for the verifier to read and check directly. The recursion here is a
sequence of polynomial-opening reductions. Each step proves relations about
the previous source and passes one new witness claim to its successor.

This chapter follows a direct edge with one source group first. A grouped
root and an edge that offloads setup use the same handoff, with the additional
claims described below.

## What crosses a recursive edge

Suppose level $i$ starts with a commitment to a source table $f^{(i)}$,
an opening point $r^{(i)}$, and a claimed value $v^{(i)}$. The statement
to prove is

$$
\widetilde f^{(i)}(r^{(i)})=v^{(i)}.
\tag{1}
$$

The tilde denotes the multilinear extension of the table in this base case.
The commitment fixes the source before the challenges used to check (1).
At the root, the application supplies this claim. At later levels, the
preceding fold supplies it.

To see where the new witness comes from, recall the common-ring case of
[the Akita fold](./proving/akita-fold.md). Block $b$ has inner digits
$\mathbf s_b$ and inner commitment
$\mathbf t_b=\mathbf A\mathbf s_b$. Its ring opening partial is $E_b$.
After the opening payload has been bound, the transcript supplies sparse
ring challenges $c_b$, and the prover forms

$$
\mathbf z=\sum_b c_b\mathbf s_b.
\tag{2}
$$

For two blocks, this is simply
$\mathbf z=c_0\mathbf s_0+c_1\mathbf s_1$. Linearity gives
$\mathbf A\mathbf z=c_0\mathbf t_0+c_1\mathbf t_1$. The fold also checks
that the same $\mathbf z$ is consistent with the two opening partials.
Thus the commitment and opening checks use a common folded source.

The prover decomposes the response, partials, and inner images into short
digits. Write these segments as $\hat{\mathbf z}$, $\hat{\mathbf e}$,
and $\hat{\mathbf t}$. They begin the next logical witness:

$$
w^{(i+1)}
=
\hat{\mathbf z}\ \Vert\
\hat{\mathbf e}\ \Vert\
\hat{\mathbf t}\ \Vert\
\text{auxiliary digits}.
\tag{3}
$$

The auxiliary digits depend on the physical relation. Quotient lifting adds
digits for polynomial-modulus quotients. Compressed payloads add the digits
used by their compression chains, and quotient lifting also adds the
compression quotients. A raw reduced-evaluation fold has neither kind of
auxiliary data. The
[physical realizations](./proving/akita-fold-realizations.md) define the exact
segments for each admitted case.

Equation (3) is prover state. A nonterminal fold does not send this full
witness to the verifier. Instead, it binds the witness and reduces its
relations to a single opening.

## Bind the witness before checking it

If the successor is another recursive fold, the prover commits to
$w^{(i+1)}$ using that successor's commitment parameters. Call the public
payload $P^{(i+1)}$. If the successor is terminal, the prover prepares only
its inner images $\mathbf t_b^{(i+1)}$. Those images replace the outer
commitment on the final edge.

In both cases, the transcript absorbs the outgoing binding before drawing
the ring-switch challenge $\alpha$, the range and row points
$\tau_0,\tau_1$, or any Stage 1 and Stage 2 challenges. A prover must
therefore choose the witness binding before learning the points that test it.

[The sumcheck stages](./proving/sumcheck-stages.md) then reduce the fold's
constraints. Stage 1 carries claims about the digit range image and, for
the L2 route, the response norm. Stage 2 combines those checks with the ring
relations and the scalar opening relation. Its final check leaves an
evaluation of the same witness:

$$
\widetilde w^{(i+1)}(r_2)=v_w.
\tag{4}
$$

Here $r_2$ is the final Stage 2 point and $v_w$ is the prover's claimed
witness evaluation. Sumcheck alone has not authenticated $v_w$ against
the outgoing witness binding. That is the next fold's task. In the canonical
coefficient representation, the handoff is

$$
f^{(i+1)}=w^{(i+1)},\qquad
r^{(i+1)}=r_2,\qquad
v^{(i+1)}=v_w.
\tag{5}
$$

The successor opens the already-bound source at this point. Repeating (5)
connects each fold's final scalar check to the next fold's commitment check.
The final direct check closes the chain.

Some extension-field schedules store a deterministic tensor subfield
projection of the logical witness. The successor then opens that physical
representation with the corresponding point and value conversion. Equation
(5) describes the logical handoff; it does not assert byte identity between
the logical digit table and every committed representation.

## Exact layout and padding

The layout is part of the statement being committed. For $N$ source ring
elements and $L$ positions per block, the address rule is

```text
source = block * L + position
live_blocks = ceil(N / L)
```

$L$ is a power of two. The final block may be partial, and its live entries
stay tight. Digits for one source element are adjacent. Recursive witness
construction consumes canonical `WitnessLayout` units in order, including
the scheduled chunk and auxiliary segments.

For example, $N=10$ and $L=4$ give three live blocks of lengths
$4,4,2$. This does not create two extra live source elements in the final
block. Operations that need a rectangular block treat the missing positions
as zero. The multilinear commitment domain may require further zero padding,
whose length is derived from the exact witness length and commitment ring
dimension.

Prover and verifier use that same domain when interpreting $r_2$. Treating
padding as live witness data, or changing the flattening order, would change
the polynomial in (4).

## Root groups and setup-prefix claims

The root can open several polynomials and several commitment groups. Each
group has its own source geometry and opening point. The root batches their
relations into one fold and constructs one outgoing digit witness. It remains
a nonterminal fold even when the schedule sends that witness directly to the
terminal.

A direct recursive edge carries one witness group. An offloaded edge carries
two groups in this order:

```text
setup prefix | witness
```

The setup prefix is a committed prefix of the public setup vector. When
Stage 2 leaves a setup contribution for later verification, Stage 3 reduces
that contribution to a separate opening
$\widetilde S(\rho_3)=v_S$ of the selected prefix. It does not replace or
change the witness claim $\widetilde w(r_2)=v_w$.

The successor authenticates these two groups at their own points, then
constructs its own outgoing witness. The schedule determines whether the
extra prefix group exists; a mismatched prefix claim and prefix parameter set
is invalid. [Setup offloading](./setup-offloading.md) derives this additional
claim. The terminal accepts only the witness group, so the final edge does
not carry a deferred setup-prefix opening.

## Payloads and relation modes

A nonterminal fold carries an opening payload and binds its successor.
Their representations depend on the schedule:

| Public value | Compressed representation | Raw representation |
| --- | --- | --- |
| Current commitment | $p_F$, the end of the B-image compression chain | B image $\mathbf u$ |
| Opening payload | $p_H$, the end of the D-image compression chain | D image $\mathbf v_D$ |
| Recursive successor binding | Successor's $p_F$ | Successor's $\mathbf u$ |

Every compressed payload in this table is 128 bytes. Raw payload lengths
follow the selected matrix geometry. The successor's payload mode determines
the outgoing binding, so it need not match the current fold's mode at a
transition.

The root and first recursive fold use compressed payloads. Later folds that
consume a setup prefix also remain compressed. A later fold without an
incoming prefix may start a raw suffix; subsequent folds cannot return to
compressed mode. A schedule may remain compressed throughout.

Ring-relation mode is a separate choice. `QuotientLift` carries quotient
digits to check native ring equations after switching to field evaluations.
An admitted `ReducedEvaluation` suffix checks the reduced equations without
those digits. This suffix starts no earlier than absolute level two, uses
evaluation-trace openings and direct setup evaluation, and has no incoming
setup prefix. It cannot return to quotient lifting. Both choices preserve the
handoff in (4).

The root and first recursive fold use subring coefficient packing for their
opening relations. Later recursive folds use evaluation trace. In a proper
extension field, evaluation-trace openings also run
[extension-opening reduction](./proving/extension-opening-reduction.md).
These opening methods change how the current scalar claim enters the fold,
not which witness evaluation Stage 2 passes onward.

## The last edge and terminal

When the successor is terminal, its inner images
$\{\mathbf t_b^{(i+1)}\}_b$ are the outgoing binding. The predecessor
absorbs their canonical bytes before its ring-switch and sumcheck challenges.
The terminal response carries those same images, the ring opening partials,
and the folded response $\mathbf z$.

The terminal checks the response bounds, opening consistency, A relation,
and scalar opening directly. It creates no new witness commitment and has
no B or D relation. Even a one-block terminal follows this path.
[Terminal verification](./verifying/terminal.md) derives the checks and
explains how they use the predecessor's binding.

## Proof anatomy and schedule ownership

`AkitaBatchedProof` stores one packed nonce stream, one root
`FoldLevelProof`, zero or more recursive `FoldLevelProof` records, and one
`TerminalLevelProof`. Each nonterminal record contains its opening payload,
Stage 1 and Stage 2 data, and optional Stage 3 data. Extension-opening
reduction appears only where the field and opening method require it.

The next-witness binding has two variants. `OuterPayload` carries a
recursive commitment, which may be compressed or raw. `TerminalInnerState`
selects the final A-only handoff and carries no duplicate commitment payload.

The selected schedule fixes level count, group geometry, decomposition,
payload modes, and terminal response shape. The verifier validates this
shape before replay. It does not run the offline planner or try a different
mode when a proof fails.

The offline planner runs one root search. Root contraction can change candidate
order, but it is not a feasibility rule or part of the final objective.
Contractive and noncontractive roots share the same suffix memo and frontier.
The configured `SelectionPolicyId` comparator selects the final complete
schedule. Recursive folds still require strict progress, and offloaded edges
still enforce their explicit minimum contraction policy.

## Code map

- `crates/akita-prover/src/protocol/ring_switch/commit.rs` prepares the
  successor's physical witness, commitment, or terminal inner state.
- `crates/akita-types/src/proof/levels.rs` defines the level records and
  successor-binding variants.
- `crates/akita-verifier/src/protocol/core/fold/mod.rs` binds the successor
  before replaying ring switching and the sumcheck stages.
- `crates/akita-verifier/src/protocol/core/suffix.rs` carries the resulting
  point and value into the next fold and validates terminal-state identity.
- `crates/akita-pcs/tests/transcript_hardening.rs` checks transcript agreement
  and the ordering of the final witness binding.
