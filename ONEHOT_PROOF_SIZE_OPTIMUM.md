# Onehot Proof-Size Optimum

## Scope

This note answers the following design question:

> If the codebase is allowed to evolve, what parameter schedule minimizes
> serialized proof size for the sparse onehot setting with 1-out-of-256
> sparsity, for `nv in {30, 38, 44}`?

Assumptions:

- The witness is sparse onehot throughout.
- The first level uses the onehot-aware folded bound.
- Later levels use the generic folded bound for balanced digits.
- `N_A = 1` is fixed.
- `N_B` and `N_D` may vary by level.
- The first Hachi level and recursive `w` levels may use different digit bases.
- The handoff to Labrador is allowed to move; we optimize for bytes, not for the
  current hard-coded heuristics.
- For `D = 64`, the challenge family uses max coefficient `2` and conservative
  L1 mass `54`, matching the rigorous split family discussed in
  `scripts/estimate_hachi_d64_k256_onehot_sis.py:384-433`.

The byte model is taken from the exact serializers and layout code:

- `PackedDigits` tail size:
  `src/protocol/proof.rs:22-140`
- `HachiLevelProof` / `HachiProof` size:
  `src/protocol/proof.rs:1333-1467`
- `w`-polynomial ring-element recurrence:
  `src/protocol/ring_switch.rs:434-495`
- Current Hachi stop heuristic:
  `src/protocol/commitment_scheme.rs:479-489`
- Current Hachi-vs-Labrador tail comparison:
  `src/protocol/labrador_handoff.rs:432-453`
- Handoff witness row construction:
  `src/protocol/labrador_handoff.rs:222-232`
- Stage-1 round degree:
  `src/protocol/sumcheck/hachi_stage1.rs:1765-1770`
- Stage-2 round degree:
  `src/protocol/sumcheck/hachi_stage2.rs:1861-1866`

## Executive Summary

The best proof-size design in the searched family is:

- `D = 64`
- level-0 `log_basis = 2`
- recursive `w_log_basis = 4`
- adaptive outer ranks:
  - `nv = 30`: `N_B = N_D = 1` throughout
  - `nv = 38`: `N_B = N_D = 2` on `L0`, then `1`
  - `nv = 44`: `N_B = N_D = 2` on `L0` and `L1`, then `1`
- no Labrador handoff

Best total proof sizes:

| `nv` | best total bytes | best design |
| --- | ---: | --- |
| `30` | `106,125` | `D=64`, `b0=2`, `wb=4`, rank-1 throughout |
| `38` | `115,441` | `D=64`, `b0=2`, `wb=4`, rank-2 on `L0` only |
| `44` | `118,449` | `D=64`, `b0=2`, `wb=4`, rank-2 on `L0-L1` |

Fair cross-`D` comparison, allowing `D=128` and `D=256` to also choose their
own best first-level basis and recursive basis:

| `nv` | best `D=64` | best `D=128` | best `D=256` |
| --- | ---: | ---: | ---: |
| `30` | `106,125` | `167,821` | `306,909` |
| `38` | `115,441` | `181,905` | `328,241` |
| `44` | `118,449` | `185,105` | `329,489` |

Best higher-`D` schedules found in the same search:

| `D` | `nv=30` | `nv=38` | `nv=44` |
| --- | --- | --- | --- |
| `128` | `b0=3`, `wb=5`, rank-1 | `b0=2`, `wb=5`, rank-1 | `b0=4`, `wb=5`, rank-1 |
| `256` | `b0=2`, `wb=5`, rank-1 | `b0=2`, `wb=5`, rank-1 | `b0=3`, `wb=5`, rank-1 |

Even after giving the larger rings those extra degrees of freedom, `D = 64`
still wins by a wide margin.

## Why `D = 64` Wins

Three effects line up:

1. Lower ring dimension shrinks every flat ring payload in the proof.
2. The onehot first fold wants a smaller digit basis than the recursive fixed
   point; `b0 = 2` is best at level 0, but `wb = 4` is best after the witness
   becomes a balanced-digit object.
3. Allowing `N_B` and `N_D` to be `2` only where needed keeps the early SIS
   instances secure without paying that byte cost forever.

The recursive fixed point for the winning design is:

- `D = 64`
- `wb = 4`
- `(m, r, delta_fold) = (8, 4, 4)`
- `num_u = 12`
- `w_ring = 2245`
- `w_len = 143,680` field digits
- direct packed tail `= 71,849 B`

That fixed point is what all three winning schedules converge to.

## Security Schedule

Per-level minimum SIS estimate for the winning schedule:

| `nv` | level | basis | outer rank | min security bits |
| --- | --- | ---: | ---: | ---: |
| `30` | `L0` | `2` | `1` | `178.37` |
| `30` | `L1` | `4` | `1` | `183.94` |
| `30` | `L2` | `4` | `1` | `202.69` |
| `30` | `L3` | `4` | `1` | `224.36` |
| `38` | `L0` | `2` | `2` | `181.60` |
| `38` | `L1` | `4` | `1` | `167.82` |
| `38` | `L2` | `4` | `1` | `202.69` |
| `38` | `L3` | `4` | `1` | `202.69` |
| `38` | `L4` | `4` | `1` | `224.36` |
| `44` | `L0` | `2` | `2` | `139.38` |
| `44` | `L1` | `4` | `2` | `184.82` |
| `44` | `L2` | `4` | `1` | `183.94` |
| `44` | `L3` | `4` | `1` | `202.69` |
| `44` | `L4` | `4` | `1` | `224.36` |

Two immediate consequences:

- `nv = 38` really does need rank `2` at `L0`, but not after that.
- `nv = 44` needs rank `2` at `L0` and `L1`; after that, rank `1` is fine.

## Fine-Grained Hachi Accounting

Each row below shows the exact Hachi contribution of one fold level:

- `y` = serialized `y_ring`
- `v` = serialized opening-side payload
- `stage1` = stage-1 sumcheck
- `stage2` = stage-2 sumcheck
- `next_commit` = commitment to the next `w`
- `level_total` = full serialized `HachiLevelProof`
- `total_if_stop` = cumulative Hachi bytes up to that level plus a direct packed
  `PackedDigits` tail at that point

### `nv = 30`

| level | basis | outer rank | `(m, r, df)` | `num_u` | `w_len` | `y` | `v` | `stage1` | `stage2` | `next_commit` | `level_total` | `total_if_stop` |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L0` | `2` | `1` | `(14, 10, 7)` | `18` | `15,880,512` | `1,036` | `1,036` | `1,352` | `1,352` | `1,036` | `5,844` | `3,975,981` |
| `L1` | `4` | `1` | `(10, 8, 5)` | `15` | `1,419,584` | `1,036` | `1,036` | `3,200` | `1,184` | `1,036` | `7,524` | `723,169` |
| `L2` | `4` | `1` | `(9, 6, 4)` | `13` | `411,968` | `1,036` | `1,036` | `2,896` | `1,072` | `1,036` | `7,108` | `226,469` |
| `L3` | `4` | `1` | `(8, 5, 4)` | `12` | `211,264` | `1,036` | `1,036` | `2,744` | `1,016` | `1,036` | `6,900` | `133,017` |
| `L4` | `4` | `1` | `(8, 4, 4)` | `12` | `143,680` | `1,036` | `1,036` | `2,744` | `1,016` | `1,036` | `6,900` | `106,125` |

Optimal stop: after `L4`.

### `nv = 38`

| level | basis | outer rank | `(m, r, df)` | `num_u` | `w_len` | `y` | `v` | `stage1` | `stage2` | `next_commit` | `level_total` | `total_if_stop` |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L0` | `2` | `2` | `(17, 15, 9)` | `23` | `348,156,352` | `1,036` | `2,060` | `1,632` | `1,632` | `1,036` | `7,428` | `87,046,525` |
| `L1` | `4` | `1` | `(13, 10, 5)` | `17` | `6,957,376` | `1,036` | `1,036` | `3,504` | `1,296` | `1,036` | `7,940` | `3,494,065` |
| `L2` | `4` | `1` | `(10, 7, 5)` | `14` | `878,912` | `1,036` | `1,036` | `3,048` | `1,128` | `1,036` | `7,316` | `462,149` |
| `L3` | `4` | `1` | `(9, 5, 4)` | `13` | `276,800` | `1,036` | `1,036` | `2,896` | `1,072` | `1,036` | `7,108` | `168,201` |
| `L4` | `4` | `1` | `(8, 5, 4)` | `12` | `211,264` | `1,036` | `1,036` | `2,744` | `1,016` | `1,036` | `6,900` | `142,333` |
| `L5` | `4` | `1` | `(8, 4, 4)` | `12` | `143,680` | `1,036` | `1,036` | `2,744` | `1,016` | `1,036` | `6,900` | `115,441` |

Optimal stop: after `L5`.

### `nv = 44`

| level | basis | outer rank | `(m, r, df)` | `num_u` | `w_len` | `y` | `v` | `stage1` | `stage2` | `next_commit` | `level_total` | `total_if_stop` |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L0` | `2` | `2` | `(20, 18, 11)` | `26` | `2,919,264,704` | `1,036` | `2,060` | `1,800` | `1,800` | `2,060` | `8,788` | `729,824,973` |
| `L1` | `4` | `2` | `(14, 12, 6)` | `19` | `23,607,744` | `1,036` | `2,060` | `3,808` | `1,408` | `1,036` | `9,380` | `11,822,049` |
| `L2` | `4` | `1` | `(11, 8, 5)` | `15` | `1,747,264` | `1,036` | `1,036` | `3,200` | `1,184` | `1,036` | `7,524` | `899,333` |
| `L3` | `4` | `1` | `(9, 6, 4)` | `13` | `411,968` | `1,036` | `1,036` | `2,896` | `1,072` | `1,036` | `7,108` | `238,793` |
| `L4` | `4` | `1` | `(8, 5, 4)` | `12` | `211,264` | `1,036` | `1,036` | `2,744` | `1,016` | `1,036` | `6,900` | `145,341` |
| `L5` | `4` | `1` | `(8, 4, 4)` | `12` | `143,680` | `1,036` | `1,036` | `2,744` | `1,016` | `1,036` | `6,900` | `118,449` |

Optimal stop: after `L5`.

## Labrador Accounting

For the winning `D = 64` schedule, Labrador never wins on bytes.

At the recursive fixed point,

- `(m, r, df) = (8, 4, 4)`
- `delta_open = 33`
- the Hachi->Labrador witness rows are
  `[33 * 2^4, 33 * 2^4, 4 * 2^8] = [528, 528, 1024]`

This comes directly from `row0 = w_hat`, `row1 = t_hat`, `row2 = decomposed z_pre`
in `src/protocol/labrador_handoff.rs:222-232`.

The direct tail here is just:

- `PackedDigits(num_elems = 143,680, bits_per_elem = 4)`
- serialized size `= 8 + 1 + ceil(143680 * 4 / 8) = 71,849 B`

The modeled Labrador recursion from that same fixed point is:

| lab level | tail | `k` | `virtual_row_len` | config | payload bytes | next witness bytes |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `L0` | `false` | `4` | `520` | `wb=7`, `ap=18x7`, inner rank `3`, outer rank `2` | `7,350` | `938,012` |
| `L1` | `false` | `3` | `306` | `wb=11`, `ap=12x11`, inner rank `3`, outer rank `2` | `7,334` | `497,692` |
| `L2` | `false` | `3` | `162` | `wb=15`, `ap=9x14`, inner rank `4`, outer rank `3` | `9,382` | `331,804` |
| `L3` | `false` | `2` | `162` | `wb=18`, `ap=7x18`, inner rank `5`, outer rank `4` | `11,430` | `259,100` |
| `L4` | `false` | `2` | `127` | `wb=22`, `ap=6x21`, inner rank `6`, outer rank `5` | `13,478` | `222,236` |
| `L5` | `false` | `2` | `109` | `wb=25`, `ap=5x26`, inner rank `7`, outer rank `6` | `15,526` | `198,684` |
| `L6` | `true` | `4` | `49` | `wb=30`, tail mode, inner rank `9` | `50,342` | `50,192` |

So:

- Labrador recursive proof bytes `= 165,038`
- plus handoff-side `v = 1,036`
- plus `y_ring = 1,036`
- plus witness norm bound `= 16`
- full Labrador tail `= 167,126`

Comparison at the fixed point:

| tail choice | bytes |
| --- | ---: |
| direct packed Hachi tail | `71,849` |
| Labrador tail | `167,126` |
| Labrador overhead vs direct | `95,277` |

Earlier handoffs are strictly worse because the handoff witness is larger before
the fixed point. So the byte-optimal policy for these three `nv` values is
Hachi-only.

## What This Means For The Codebase

To evolve the codebase toward the true proof-size optimum, the important changes
are architectural, not cosmetic:

1. Make `D`, `log_basis` at level 0, recursive `w_log_basis`, and `N_B/N_D`
   level-dependent parameters rather than fixed properties of a single
   `CommitmentConfig`.
2. Replace the current shrink-ratio stop rule in
   `src/protocol/commitment_scheme.rs:479-489` with a byte objective.
3. Replace the current direct-vs-Labrador comparison in
   `src/protocol/labrador_handoff.rs:432-453` with a 3-way dynamic-programming
   choice:
   `best(state) = min(direct_tail(state), labrador_tail(state), hachi_level_cost(state -> next) + best(next))`.
4. Keep `N_A = 1`.
5. For `D = 64`, use the rigorous split challenge family with max coefficient
   `2` and conservative mass `54`.

The practical recommendation is:

- optimize for bytes, not for witness shrink ratio
- use `D = 64`
- use `b0 = 2`
- use `wb = 4`
- use rank `2` only in the short early window where the SIS floor requires it

That is the schedule that minimizes proof size in the explored family.
