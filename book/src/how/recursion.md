# Recursion and proof shape

Akita uses the same digit-innermost source and witness geometry at every fold.
An intermediate fold emits one 128-byte recursive witness payload. The final fold
instead hands its predecessor-bound inner `t` state to a scalar terminal
checker, which consumes the cleartext witness without another payload.

## Intermediate vs terminal levels

For each group, source elements use `source = fold * L + position`, where `L`
is a power of two and `F = ceil(N / L)` is exact. A partial final block
stays tight. Recursive witness construction consumes canonical
`WitnessLayout` units and emits the next source in that same order; it does not
transpose through a column-major intermediate.

A grouped root fold is nonterminal. Its successor contains exactly one witness
group and one setup-prefix group. Setup-prefix materialization consumes the same
canonical ranges as witness emission. At the terminal, the single group is
consumed through the scalar direct path, including a scalar `F = 1` handoff.
Its physical relation is `consistency | A`: the terminal has no outer `u`, B
block, or D block.

The setup prefix appears only when the selected recursive schedule offloads the
preceding setup contribution. [Setup offloading](./setup-offloading.md)
explains how Stage 3 creates this second opening claim and how the successor
authenticates it beside the folded witness.

The transcript binds the schedule and exact group geometry before challenges
that depend on them. Changing a terminal or recursive handoff is therefore a
protocol change, not a serialization-only change.

Every nonterminal fold carries `opening_payload = p_H` for its D relation and
binds its successor with `OuterPayload(p_F)` for the successor B relation. The
raw D and B images remain internal prover relation values. A terminal successor
uses `TerminalInnerState` and carries no duplicate compression payload.

## Proof anatomy

`AkitaBatchedProof` stores one `FoldLevelProof` root, zero or more recursive
`FoldLevelProof` records, and one `TerminalLevelProof`. The root or final
recursive fold binds the terminal state as its successor. A schedule with no
recursive folds uses the same proof types and transcript rules. Each level's
descriptor binds the resolved `L`, exact `F`, chunk count, and
decomposition parameters. Singleton openings and terminal folds are ordinary
one-group, one-chunk cases; there is no alternate block order.

The offline planner runs one root search. Root contraction can change candidate
order, but it is not a feasibility rule or part of the final objective.
Contractive and noncontractive roots share the same suffix memo and frontier.
The configured `SelectionPolicyId` comparator selects the final complete
schedule. Recursive folds still require strict progress, and offloaded edges
still enforce their explicit minimum contraction policy.
