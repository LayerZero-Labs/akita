# Recursion and proof shape

Akita uses the same digit-innermost source and witness geometry at every level.
A nonterminal fold emits one recursive witness commitment. A direct fold emits
no successor setup claim, and the terminal fold is scalar and direct.

## Intermediate vs terminal levels

For each group, source elements use `source = fold * L + position`, where `L`
is a power of two and `F = ceil(N / L)` is exact. A partial final block
stays tight. Recursive witness construction consumes canonical
`WitnessLayout` units and emits the next source in that same order; it does not
transpose through a column-major intermediate.

A grouped root fold is nonterminal. Its successor contains exactly one witness
group and one setup-prefix group. Setup-prefix materialization consumes the same
canonical ranges as witness emission. At a terminal step, the single group is
consumed through the scalar direct path, including a scalar `F = 1` handoff.

The transcript binds the schedule and exact group geometry before challenges
that depend on them. Changing a terminal or recursive handoff is therefore a
protocol change, not a serialization-only change.

## Proof anatomy

The serialized structure is rooted at `AkitaBatchedProof` and
`AkitaBatchedRootProof`, followed by `AkitaLevelProof` / `AkitaProofStep`
records for the suffix. Each level's descriptors bind the resolved `L`, exact
`F`, chunk granule `S`, challenge shape, and decomposition parameters.
Singleton openings and terminal folds are ordinary one-group, one-chunk cases;
there is no alternate block order.
