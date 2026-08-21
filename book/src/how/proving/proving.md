# The proving protocol

One Akita fold takes a committed opening claim and produces a smaller witness
that carries the claim forward. It proves that the new witness comes from the
old committed data, satisfies the fold relation, uses bounded digits, and is
ready to be opened by the next level.

The root and recursive nonterminal levels use the same fold engine. They differ
in the source they consume and the schedule parameters assigned to them. The
terminal level is different: it checks the final clear witness directly and
does not create another recursive witness.

This chapter is the map for the proving section. Its child chapters derive the
relations, physical layouts, opening methods, and sum-checks in detail.

## Inputs and output of one fold

A nonterminal fold starts with four kinds of information.

| Input | What it means |
| --- | --- |
| Public opening claims | Ordered commitments, opening points, and claimed polynomial values |
| Private source data | Polynomial or recursive witness values plus the hints created when they were committed |
| Public setup | The A, B, and D matrix entries required by this scheduled level |
| Scheduled parameters | Ring dimensions, block geometry, digit bases, opening method, norm route, and proof shape |

The fold produces:

- one `FoldLevelProof` containing its commitment payloads and sum-check proofs;
- one smaller digit witness for the successor;
- one opening claim that binds that witness at the final Stage 2 challenge; and
- when setup offloading is selected, a separate opening claim for one prepared
  setup prefix.

The witness claim and setup-prefix claim remain distinct. A successor may open
both in one grouped fold, but they keep separate commitments, points, values,
and source meanings.

## One nonterminal fold in order

The prover follows this sequence.

### 1. Validate the scheduled geometry

The level reads its group parameters from the resolved schedule. It checks
matrix roles, ring dimensions, block counts, digit depths, opening methods,
payload modes, and any incoming setup prefix before building large tables.

The proof does not get to override these values. They determine the expected
message widths and transcript structure on both prover and verifier paths.

### 2. Prepare each opening

Each commitment group has one complete opening point shared by the polynomial
claims in that group. The scheduled opening method determines how the prover
turns those claims into fold inputs.

- **Subring coefficient packing** contracts selected coefficient coordinates
  and opens the extension-valued claim directly in a smaller challenge ring.
- **Evaluation trace** expresses the claim through a field-valued trace. When
  the opening point lives in a proper extension field, extension-opening
  reduction first converts it to the base-field relation consumed by the fold.

The method belongs to the level schedule. A proof carries extension-opening
data only when evaluation trace and a proper extension field require it.

See [Fold path and field geometry](./fold-path.md),
[Field-to-ring evaluation reduction](./field-ring-reduction.md), and
[Extension-opening reduction](./extension-opening-reduction.md).

### 3. Fold the source into a response

Transcript challenges combine the source blocks into a shorter response
$\mathbf z$. The opening point also produces partial evaluations that will
bind the claimed value to those same blocks.

The response, opening values, and inner commitment values are decomposed into
small signed digits. Akita stores those digits in one canonical
digit-innermost witness layout. Every later range check, matrix evaluation,
recursive handoff, and verifier calculation uses that layout.

See [Opening points and digit-innermost layout](./opening-points-layout.md).

### 4. Build the semantic fold relations

The core fold statement has four ring-valued relation families:

1. the folded response agrees with the source blocks and fold challenges;
2. the inner A matrix agrees with the source digits;
3. the outer B matrix agrees with the public commitment; and
4. the opening D matrix agrees with the opening digits.

The field-valued opening claim is kept conceptually separate, then fused into
Stage 2 beside these ring rows.

The basic derivation is in [Semantic relations in an Akita
fold](./akita-fold.md). [Advanced relation layouts](./advanced-relation-layouts.md)
extends the same four families to multiple groups, chunks, and role-specific
ring dimensions.

### 5. Choose the physical realization

The semantic B and D commitment values can appear directly as raw payloads, or
they can remain hidden behind smaller compressed payloads. Compressed mode adds
the two-map commitment chains, their digit witnesses, and their physical rows.
It does not change what the four semantic relations mean.

Every physical row is lifted from its native ring before the rows are combined
for Stage 2. This is how one fold supports distinct A, B, and D ring dimensions
without pretending that all matrices live in one ring.

See [Raw and compressed realizations of an Akita
fold](./akita-fold-realizations.md).

### 6. Form and commit the successor witness

The relation witness contains the response digits, opening digits, inner
commitment digits, quotient rows, and any compression witness required by the
selected realization. Ring switching lays these values out as the next
field-valued witness. The prover commits to that witness with the parameters of
the successor level.

At the root, subring coefficient packing also controls how sparse challenge
coefficients enter the A ring and how shortened partial evaluations enter the
relation. See [Root fold and ring switching](./root-fold-ring-switch.md).

### 7. Run the sum-check cascade

Every nonterminal fold runs two required stages and one optional stage.

| Stage | What it proves | What it leaves behind |
| --- | --- | --- |
| Stage 1 | Every new witness entry is in the scheduled balanced digit range; an L2 route also proves the physical response norm | One range-image evaluation used by Stage 2 |
| Stage 2 | The ring-switched physical relation, Stage 1 output, and incoming opening claim all agree with the committed successor witness | The successor witness opening claim |
| Stage 3 | The claimed A, B, and D setup contribution is the correct product sum for a prepared setup prefix | A separate setup-prefix opening claim |

Stage 3 appears only on a scheduled offloaded edge. In direct mode, the
verifier evaluates the setup contribution itself and no Stage 3 proof is
present.

See [Sum-check stages](./sumcheck-stages.md) and
[Setup offloading](../setup-offloading.md).

### 8. Hand the claims to the successor

The Stage 2 challenge and expected witness evaluation become the statement for
the next fold. If Stage 3 ran, its setup-prefix point and evaluation enter as a
second commitment group. The transcript binds this exact handoff before the
successor samples dependent challenges.

The process repeats until the schedule reaches its terminal level.

## Root, recursive, and terminal responsibilities

| Level | Source | Main responsibility | Output |
| --- | --- | --- | --- |
| Root | Application polynomial groups | Batch the requested openings and enter the recursive fold path | First recursive witness claim, plus an optional setup claim |
| Recursive nonterminal | Witness and optional setup group from the predecessor | Authenticate the handoff and shrink it again | Next witness claim, plus an optional setup claim |
| Terminal | Final clear witness bound by its predecessor | Check consistency, A relation, evaluation trace, encoding, and scheduled norm directly | Accept or reject |

The terminal has no outer B commitment, no opening D commitment, and no
sum-check cascade. Its predecessor already binds the terminal witness through
the opening claim that enters the final check.

[Recursion and proof shape](../recursion.md) explains how these records are
encoded and connected.

## Reading the child chapters

The most useful order depends on what you need.

### First protocol reading

1. [Field-to-ring evaluation reduction](./field-ring-reduction.md)
2. [Semantic relations in an Akita fold](./akita-fold.md)
3. [Raw and compressed realizations](./akita-fold-realizations.md)
4. [Opening points and digit-innermost layout](./opening-points-layout.md)
5. [Fold path and field geometry](./fold-path.md)
6. [Sum-check stages](./sumcheck-stages.md)

Read [Root fold and ring switching](./root-fold-ring-switch.md) when you want
the complete coefficient-packing derivation. Read [Extension-opening
reduction](./extension-opening-reduction.md) for the evaluation-trace path over
a proper extension field.

### Multi-group or distributed implementation

[Advanced relation layouts](./advanced-relation-layouts.md) adds commitment
groups, chunks, and different role dimensions to the basic fold.
[The distributed prover](./distributed-prover.md) then shows which source,
commitment, and sum-check work can stay local to separate machines.

## Prover and verifier map

| Responsibility | Prover | Verifier |
| --- | --- | --- |
| Top-level schedule walk | `akita-prover/src/protocol/core/prove.rs` | `akita-verifier/src/protocol/core/verify.rs` |
| Root and recursive fold orchestration | `akita-prover/src/protocol/core/fold/` | `akita-verifier/src/protocol/core/fold/` |
| Extension-opening reduction | `akita-prover/src/protocol/extension_opening_reduction/` | `akita-verifier/src/protocol/core/fold/extension_claim.rs` |
| Stage 1 range and norm proof | `akita-prover/src/protocol/sumcheck/digit_range/` and `physical_l2_norm.rs` | `akita-verifier/src/stages/stage1.rs` |
| Stage 2 fused relation | `akita-prover/src/protocol/sumcheck/relation_range_image/` | `akita-verifier/src/stages/stage2.rs` |
| Stage 3 setup product | `akita-prover/src/protocol/sumcheck/akita_stage3/` | `akita-verifier/src/stages/stage3.rs` |
| Terminal path | `akita-prover/src/protocol/core/suffix.rs` | `akita-verifier/src/protocol/core/suffix.rs` and `terminal_direct.rs` |

The exact source layout can evolve, so begin with the public core entry points
and follow the shared types they call. The Book's Verification child pages
provide a value-by-value map of the current verifier calculations.

## Review checklist for a fold change

Before accepting a change to this path, check all of the following:

1. The generated schedule and proof shape still describe every message.
2. Prover and verifier absorb public data and proof messages in the same order.
3. Source, witness, relation, and challenge indexes use the same physical
   layout.
4. Every relation row is interpreted in its scheduled native ring before
   lifting or batching.
5. Stage 1 output and the incoming opening claim are both bound inside Stage 2.
6. Stage 3 is present exactly on offloaded setup edges.
7. The successor receives the exact witness and setup claims produced by its
   predecessor.
8. Malformed sizes, encodings, schedules, and terminal values return an error
   on the verifier path rather than panicking.

A fold is not correct merely because its prover can produce a proof that its
verifier accepts. Its schedule, transcript, relation layout, final oracle
checks, and independent rejection tests must all describe the same statement.
