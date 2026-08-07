# Root fold and ring switching

> **Status:** stub. Part of the initial Akita Book scaffold.

The heart of one folding step: building the batched root relation and switching
it into the next-level ring witness.

## The root fold

The batched root relation (`OpeningClaimsLayout` routing polynomial groups to
claims), the commitment/claim/fold rows, and how the level-0 root fold differs
from a recursive fold.

**Sources to fold in**

- `crates/akita-prover/src/protocol/flow/root_fold.rs:352-544` (`prove_root`, terminal root fold).
- `crates/akita-prover/src/protocol/ring_relation.rs`.
- `crates/akita-types/src/opening_claims.rs`.
- Paper §3.2 `sec:akita-layout` (the batched root relation; `eq:batched-root-*`), §3.5 `sec:akita-one-step`.
- `specs/archive/2026-Q2/w-to-e-notation.md` (w / e / v naming).

## Ring switching

The lattice fold proper: lifting the root relation `M·w = h` from
`R_q = Z_q[X]/(X^D+1)` to `Z_q[X]` via the unique quotient, computed with paired
cyclic and negacyclic NTTs. Distinguish this from the FRI-Binius
extension-opening reduction (a different "ring switch"; see
[Foundations → Extension-opening reduction](../../foundations/extension-opening-reduction.md)).

**Sources to fold in**

- `crates/akita-prover/src/protocol/ring_switch.rs:48-80`, `ring_switch/finalize.rs`.
- Paper §3.5 (`fig:akita-ring-switch`), App B.3 `sec:akita-ring-switching` (quotient via cyclic/negacyclic NTT, `eq:akita-quotient-identity`).
- `specs/terminal-fold-cutover.md` (D-block at intermediate vs terminal).
- Council math report B4 (lattice fold vs EOR distinction).
