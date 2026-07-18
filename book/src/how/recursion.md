# Recursion and proof shape

Akita uses the same digit-innermost source and witness geometry at every fold.
An intermediate fold emits one recursive witness commitment. The final fold
instead hands its predecessor-bound inner `t` state to a scalar terminal
checker, which consumes the cleartext witness without another commitment.

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

The transcript binds the schedule and exact group geometry before challenges
that depend on them. Changing a terminal or recursive handoff is therefore a
protocol change, not a serialization-only change.

## Proof anatomy

`AkitaBatchedProof` stores one `FoldLevelProof` root, zero or more recursive
`FoldLevelProof` records, and one `TerminalLevelProof`. Supported schedules
always contain at least two fold records, so the terminal state is bound by its
predecessor and there is no root-terminal proof variant. Each level's
descriptor binds the resolved `L`, exact `F`, chunk count, challenge shape, and
decomposition parameters. Singleton openings and terminal folds are ordinary
one-group, one-chunk cases; there is no alternate block order.
