# Spec: binary source root zero padding

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-11-03 |
| Status | implemented |
| PR | |
| Supersedes | None |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

> **Accepted and implemented: the deletion rule.** The opening spec,
> [binary-opening.md](binary-opening.md), describes the current
> protocol and its measurements; this note keeps the argument. The Summary and
> Sections 1–6 describe the protocol before the change, and their code
> citations are at commit `8f69246`. The padding-point rule was not
> implemented. The shape and measurement rows also predate the single-point
> prime statement: current groups have one shared point and one plane per
> claimed polynomial. Every binary group now carries its word claim, with prime
> claims optional on top. Those historical rows retain their original values.

## Summary

The binary root checks that every outgoing-witness cell without a natural
coefficient is zero. The check uses one independent batching weight per such
cell, so verification is linear in the witness. At N=2^22 `binary-and-prime-claims` the
verifier draws 17,129,105 order-7 batching scalars, of which 17,129,090 are
padding weights.

This note asks which parts of the protocol need those cells to be zero. The
answer is **none**. No Stage 2 relation, the typed Stage 3 A product or any
source reading reads a padding address. The range check reads every address but
needs only an in-range digit. The ordinary successor folds read physical
padding, because it changes their responses, evaluations and norms. They
authenticate and bound whatever table was committed, and they do not need
padding to vanish. The cheapest correct rule is therefore to **delete the
padding rows entirely** (the *deletion rule*). It adds no error term, no
verifier work and no proof field. The range check, successor authentication and
the physical-norm proof stay exactly as they are.

The note also specifies and proves the replacement the brief proposed, kept in
case reviewers want the zero-table invariant as a defensive property (the
*padding-point rule*). It uses one multilinear point `tau_pad` and one batching
slot. Its extra interactive error is n/P, where n is the number of domain
variables: 25/P = 2^-123.356 or 26/P = 2^-123.300 at the measured shapes. Its
verifier cost is at most 802 field multiplications per opening, computed in
factored form from the segment geometry.

**Recommendation:** the deletion rule. The padding-point rule is the fallback.

## Notation

Each symbol is defined once and used with this meaning throughout.

| Symbol | Meaning |
| --- | --- |
| `P` | The proof prime `2^128 - 275`. Challenges are base-field elements (`E = F`). |
| `n` | Number of Stage 2 variables. The outgoing witness domain is `{0,1}^n`, which has `2^n` addresses. |
| `a` | An address in `[0, 2^n)`, identified with its bits `a_0` (low) through `a_(n-1)`. Coordinate `i` of every point binds bit `a_i` ([`FlatBooleanDomain`](../crates/akita-params/src/layout/digit_range.rs) lines 10-14). |
| `x` | The outgoing digit table, `x : {0,1}^n -> F`. It is fixed by the order-5 outgoing commitment. |
| `eq(u, a)` | `prod_i (u_i a_i + (1-u_i)(1-a_i))` for `u` in `F^n`. |
| Segment `k` | One [`RootRepackingSegment`](../crates/akita-types/src/root_repacking.rs): base `B_k`, `L_k = elements * digits` lanes, natural degree `D_k`, stride `S_k = 2^(s_k)` (padded coefficient count), width `L_k S_k`. |
| `Lambda` | `live_len`: the end of the last segment (root_repacking.rs lines 57-76). |
| `Nat` | Addresses `B_k + l S_k + t` with `t < D_k`. These hold natural coefficients. |
| `Pad` | All other addresses: stride tails `D_k <= t < S_k` and the domain tail `[Lambda, 2^n)`. This is exactly [`is_padding_address`](../crates/akita-types/src/root_repacking.rs) (lines 162-180). |
| `W_j` | Public weight vector of relation row `j`. Row `j` asserts `<W_j, x> = c_j`. |
| `K` | Number of rows, which is [`BinaryRootPublic::batch_count`](../crates/akita-types/src/binary/root_public.rs) (lines 123-137). |
| `alpha` | Ring-switch scalar (order 7). |
| `tau0` | Range-check point in `F^n` (order 7). |
| `beta` | The K independent row-batching scalars (order 7, `Tau1Point`). The first seven are `mu`. |
| `zeta` | The Stage 1 output point. |
| `gamma` | The Stage 2 batching scalar. |
| `r` | The Stage 2 output point in `F^n`. |

Historical shape sizes (C=256, rank three, one polynomial, first successor degree
512), reproduced exactly from the code's segment rules. The columns match the
opening spec's layout table:

| N | Shape | n | Natural `|Nat|` | Physical `Lambda` | Stride tails | Domain tail | `|Pad|` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | binary-claim | 25 | 14,642,046 | 23,138,048 | 8,496,002 | 10,416,384 | 18,912,386 |
| 2^22 | binary-and-prime-claims | 25 | 16,425,342 | 25,956,096 | 9,530,754 | 7,598,336 | 17,129,090 |
| 2^22 | binary-and-prime-claims-plus-ordinary | 25 | 21,393,790 | 30,924,544 | 9,530,754 | 2,629,888 | 12,160,642 |
| 2^22 | five-prime-claims | 26 | 23,514,624 | 37,158,912 | 13,644,288 | 29,949,952 | 43,594,240 |
| 2^23 | binary-claim | 26 | 25,258,878 | 39,915,264 | 14,656,386 | 27,193,600 | 41,849,986 |
| 2^23 | binary-and-prime-claims | 26 | 27,042,174 | 42,733,312 | 15,691,138 | 24,375,552 | 40,066,690 |
| 2^23 | binary-and-prime-claims-plus-ordinary | 26 | 36,172,158 | 51,863,296 | 15,691,138 | 15,245,568 | 30,936,706 |
| 2^23 | five-prime-claims | 26 | 34,131,456 | 53,936,128 | 19,804,672 | 13,172,736 | 32,977,408 |

Two different notions are both called "padding", and this note concerns only
the second. The *source's committed padding* (scalar coordinates 128..161,
padding carriers, partial last components; [binary-groups.md](binary-groups.md))
consists of natural coefficients of source rings. Those coefficients occupy
`Nat` cells of the response segment, and the application's agreement contract
governs them. This note does not change that contract. The *root-witness
padding* `Pad` consists of the cells to which no coefficient map points.

## 1. Who needs zero padding

### Weight support lemma

Every Stage 2 weight term that the root compiles, except the padding block
itself, lies on an address in `Nat`.

*Proof.* Each term's address comes from
[`RootCoefficientRepacking::coefficient_address`](../crates/akita-types/src/root_repacking.rs)
(lines 127-156), either directly or through `coefficient_weights`
(lines 265-288). That function rejects any coefficient index outside the
natural width (lines 140-143). It maps a natural index `(plane, t)` with
`t < D_k` to an offset `plane*S_k + t` (via
[`physical_coefficient_index`](../crates/akita-params/src/layout/relation_layout.rs),
lines 110-130), so `offset mod S_k < D_k`. That is exactly the complement of
`is_padding_address`. Every raw `terms.push` in `crates/akita-types/src/binary/`
pushes such an address:

- `authentication.rs` lines 151 and 193;
- `evaluation_weights.rs` lines 102-109;
- `ordinary_root.rs` lines 292-293, 348-349, 404-405, 436-437 and 496-497.

The only exception is the padding loop itself (`root_projection_weights.rs`
lines 121-131). The unit test `mixed_ring_repacking_matches_i128_and_has_no_address_aliases`
(root_repacking.rs lines 349-402) checks on a small layout that no natural
address is a padding address. ∎

### Consumers

| Consumer | Reads `Pad`? | If a padding cell is nonzero |
| --- | --- | --- |
| A projection rows, Z and T terms (Stage 2) | No. `binary_direct_a_weights` emits Z terms for `t < 648` (root_projection_weights.rs line 103) and T terms (line 116) through `coefficient_weights`. | No effect. |
| A part offloaded to Stage 3 | No. The verifier's factor uses `EqPolynomial::evals_prefix(r_t, 648)` (a_product.rs line 174), so the weight's multilinear extension is zero at `t >= 648`. The response segment starts at address 0 (checked at a_product.rs line 152). | No effect. |
| Typed A prefix, `m*1024*4` setup cells | Not witness cells. Prefix padding (coefficients 648..1023, rank row 3, extra capacity) is zero in the deterministic setup table (`binary_a_setup_coefficients`, a_product.rs lines 66-100). The setup commitment is authenticated once at preparation. | Not applicable: the prover cannot choose these cells. |
| Parity and value | No. Response low digit at packed index `4c + row mod 4 < 648` (evaluation_weights.rs lines 97-109); U and carries at coefficients `< 162` (lines 136-141 and 150-155). | No effect. |
| Prime response and reading rows | No (prime_weights.rs lines 96-104 and 125-130). | No effect. |
| B and D authentication | No. The compact address list contains natural coefficients only (authentication.rs lines 148-158). Its chunks of 64 index that list (lines 186-193). | No effect. |
| Mixed ordinary A/B/compression/evaluation rows and the `{-1,0}` rows | No. Ordinary segments use `RelationRowGeometry::native`, which has `S_k = D_k` and therefore no tails (ordinary_root.rs lines 77-98). Negative-bit addresses are natural (lines 139-165). | No effect. |
| Stage 1 range check and the Stage 2 range-image term | Yes, every address. `tau0` has n coordinates (verifier binary_root.rs lines 177-178 and 217-220). | Passes if the digit is in `[-4,3]`. Zero is not required, and a digit outside the range is rejected as before. |
| Ordinary successor folds and terminal | Yes. They consume the committed table through the one claim `x~(r) = witness_eval` (binary_root.rs lines 234 and 265-268), and committed padding enters their responses, evaluations and norms. | The table's evaluation, responses and norms change, and they are authenticated and checked as before. The successor proves an evaluation of the table it committed. Its norm envelope already allows every ring coefficient to be nonzero (`crates/akita-params/src/sis/norm_bound.rs` lines 324-328). The physical L2 norm of each response is checked against the scheduled cap (`crates/akita-verifier/src/stages/physical_l2_norm.rs` lines 162-164 and 233-234). Extra padding can make a response fail these checks, but it cannot bypass them. |
| Authenticated readings and the agreement contract | No. Readings come from claims whose relations are listed above. Agreement concerns source coefficients, which are `Nat` cells. | No effect. |

**The domain suffix.** Let `Q = 2^n`, and let the successor commitment have
`B` live blocks, `M` positions per block and ring dimension `D`. Set
`C = min(Q, B·M·D)`. The suffix splits into two parts.

- `[Lambda, C)` lies inside committed blocks. Its cells are committed, prover-chosen
  digits. The honest prover truncates its table to `logical_len` before import
  (`crates/akita-cpu-backend/src/opaque/binary.rs` lines 446-457), and
  successor preparation fills the rest of those blocks with zeros
  (`crates/akita-params/src/layout/ring_dims.rs` lines 168-171). A dishonest
  prover may commit any in-range digits there.
- `[C, Q)` consists of absent blocks and is virtual. The commitment computes
  only its live block ranges
  (`crates/akita-cpu-backend/src/arithmetic/commitment.rs` lines 365-380). The
  verifier's opening weights keep every position but only the live blocks
  (`crates/akita-params/src/layout/opening_point.rs` lines 210-214), so these
  cells are zero in the evaluation the successor proves.

Arbitrary nonzero values in `[C, Q)` are therefore not interchangeable valid
successor witnesses. Theorem 1 below is stated narrowly for this reason.

The ordinary Stage 2 verifier has no zero-suffix relation either. Its relation
lanes are confined to the live prefix
(`crates/akita-verifier/src/relation/evaluation.rs` lines 448-459), so the
ordinary protocol on `main` already constrains its committed suffix only by the
range, norm and evaluation checks.

### Conclusion

No consumer needs zero padding. No root equation or source reading reads
padding. The range check reads it and requires only `x(a) in [-4,3]`. The
ordinary successor folds read it and authenticate and bound it, but they do not
require it to vanish. The cheapest correct rule is to **remove the padding
rows**. `K` drops by `|Pad|`, and the order-7 row-batching vector keeps every
other slot.

**Theorem 1 (deletion rule).** Suppose `x` and `x'` agree on `Nat`. Then:

1. `<W_j, x> = <W_j, x'>` for every root row `j` other than the padding rows.
   So the decoded root equations hold for `x` if and only if they hold for `x'`.
2. `x` and `x'` decode to the same source readings.

*Proof.* Part 1 follows from the support lemma. For part 2, every reading is a
linear function of `Nat` cells. ∎

Theorem 1 does **not** say that `x` and `x'` are interchangeable witnesses. The
range check, successor authentication and the physical-norm proof read the
committed table, padding included, and all three remain required. For example,
let `x = 0`, and let `x'` equal `x` except for a one at a committed padding
address `a`. At the Boolean point `r = a`, the successor claim `x~(r) = 0` holds
for `x` and fails for `x'`. The verifier consumes that evaluation at
binary_root.rs line 234. Deletion leaves all three checks unchanged: a prover
that commits nonzero in-range padding must still pass them for the table it
committed. Theorem 1 shows only that the root's equations and readings place no
constraint on `Pad`, so the per-cell rows add nothing that soundness needs.

The checkpoint argument in [binary-checkpoint.md](binary-checkpoint.md)
already lists the branch relation without padding: digit membership, view
consistency, `RA = 0 mod q`, the two parity identities, prime rows and the
ordinary B/D/compression/setup identities. Deleting the padding rows therefore
changes no hypothesis of that argument (checkpoint spec lines 576-591). Its
comparison step, in which two differing candidates under one checkpoint input
yield a collision, does not depend on where the candidates differ.

## 2. The padding-point rule (fallback)

Use this rule if reviewers want `x|Pad = 0` as an enforced invariant.

**Claim added to Stage 2.** The rule adds one row, with weight and target

```text
W_pad(a) = 1_Pad(a) * eq(tau_pad, a),   c_pad = 0,
```

where `tau_pad` is in `F^n`. It replaces the `|Pad|` per-cell rows, and it has
batching slot `beta_pad` at the position of the old padding range (after D,
before the ordinary rows; root_public.rs lines 196-215).

**Order 7, in draw order:**

| Draw | Site | Fixed before this draw |
| --- | --- | --- |
| `alpha` | `RingSwitchAlpha` | Orders 0-6: statement, terminal, D, response, outgoing commitment (so `x`), J, y. |
| `tau0` (n coordinates), then `tau_pad` (n coordinates) | `Tau0Point`, one grinding query, `2n` challenges (`crates/akita-types/src/transcript_grinding/replay.rs` lines 226-251) | The same messages. `x` is fixed, so the padding polynomial `g` below is fixed. |
| `beta`, with `K' = K - |Pad| + 1` entries | `Tau1Point` | Every row residual, including `g(tau_pad)`. |

Drawing `tau_pad` at the existing `Tau0Point` site leaves the `GrindingSite`
enum, which lives on `main` in `crates/akita-params/src/transcript_grinding.rs`,
unchanged. No new extension seam is needed. Reusing `tau0` itself as `tau_pad`
is also sound (both tests are fixed before `tau0`) and costs the same `2n/P`
at that site. I specify fresh coordinates so that the two tests are visibly
independent.

## 3. Soundness

### 3.1 The current analysis

The analysis is in the fixed-candidate model of the checkpoint argument: the
candidate `x` is fixed at order 5, before every order-7 challenge. Each row
`j` has a residual `delta_j = <W_j, x> - c_j`. For a ring row, `delta_j` is the
value at `alpha` of a reduced residual polynomial of degree below 648. A
nonzero residual survives `alpha` with probability at most 648/P. This is the
`RingSwitchAlpha` loss (`crates/akita-types/src/binary/grinding.rs`
lines 41-45).

The residual vector `delta` is fixed before `beta`. If `delta != 0`, then
`sum_j beta_j delta_j` is a nonzero linear form in independent uniform `beta`,
so it vanishes with probability at most 1/P. This is the "independent
row-batching loss 1" (grinding.rs lines 50-51). It is charged once for the
whole vector, however many rows are defective.

Per-cell padding rows have `delta_a = x(a)` with no `alpha` dependence.
Therefore the current rule adds **zero** error beyond that shared 1/P.

The batched claim `sum_a W(a) x(a) = T`, with `W = sum_j beta_j W_j` and
`T = sum_j beta_j c_j`, is joined with the Stage 1 range-image claim by
`gamma` (loss 1/P). Stage 2 then proves it with an n-round degree-three
sumcheck (loss 3n/P). The final check (binary_root.rs lines 265-268) is

```text
claim_out = (W~(r) + A3) * x~(r) + (N~(r) + gamma * eq(zeta, r)) * x~(r) * (x~(r) + 1).
```

Here `W~(r)` is the multilinear extension of the non-A weight table at `r`
(computed by `relation.evaluate`), `A3` is the Stage 3 A product, and `N` is
the negative-bit weight table. Padding is a row of `W`, so its contribution to
`W~(r)` is `sum_(a in Pad) W_pad(a) eq(r, a)`.

### 3.2 Deletion rule

By Theorem 1, the root equations without padding rows are the equations the
checkpoint argument already uses. Range, successor and norm checks are
unchanged. The only change to the error ledger is that
the `Tau1Point` vector is shorter. In the one-shot model its loss stays 1/P.
The deletion rule adds **no error term**.

### 3.3 Padding-point rule

**Theorem 2.** Fix `x` before `tau_pad`. If `x(a) != 0` in F for some `a` in
`Pad`, then `Pr[g(tau_pad) = 0] <= n/P`, where

```text
g(X) = sum_(a in Pad) eq(X, a) x(a).
```

*Proof.* `g` is multilinear in `X` in `F^n`, so its total degree is at most n.
On the Boolean cube, `g(a) = x(a)` for `a` in `Pad` and `g(a) = 0` elsewhere,
because `eq(b, a) = [a = b]` on cube points. The map from cube values to
multilinear polynomials is injective, so `g` is the zero polynomial only if
`x|Pad = 0`. If it is not, the Schwartz–Zippel lemma gives the bound. ∎

A nonzero integer digit `d` in `[-4,3]` is nonzero in F because `|d| < P`.
Theorem 2 needs only `x(a) != 0` in F, so it holds without the range check.

**Composition.** If `g(tau_pad) != 0`, then `delta_pad = g(tau_pad)` is a
nonzero entry of the residual vector fixed before `beta`. The existing 1/P
batching charge covers it, so no new batching term arises. `W_pad` is a row of
the same linear claim that the Stage 2 sumcheck reduces to `r`, so the
existing `gamma` and `3n/P` sumcheck terms cover it. The verifier must compute
the padding row's share of `W~(r)` exactly. Section 4 shows how, with no error.
Honest provers have `x|Pad = 0`, so `g = 0` and completeness is unchanged.

The padding-point rule adds exactly **n/P**.

### 3.4 Error terms

The interactive root terms over P below are the ones the root grinding plan
prices:

- `alpha`: 648 (grinding.rs lines 41-45);
- `tau0`: n (lines 46-49);
- `beta`: 1 (lines 50-51);
- Stage 1: n rounds of degree 5 (lines 55-85);
- `gamma`: 1 (lines 86-90);
- Stage 2: n rounds of degree 3 (lines 91-102);
- Stage 3: `2k` (`k = 24` at 2^22 and `k = 25` at 2^23).

The Stage 3 round counts are confirmed by the measured Stage 3 partition:
824 = 32·24 + 32 + 24 bytes and 74 = 3·24 + 2 atoms; 857 = 32·25 + 32 + 25
bytes and 77 atoms.

The per-root sum is `648 + 9n + 2 + 2k` over P.

| N | Shapes | n | k | Current root sum | Padding term (padding-point rule) | With padding point | Deletion rule |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2^22 | binary-claim, binary-and-prime-claims, binary-and-prime-claims-plus-ordinary | 25 | 24 | 923/P = 2^-118.1498 | 25/P = 2^-123.3561 | 948/P = 2^-118.1113 | 923/P (unchanged) |
| 2^22 | five-prime-claims | 26 | 24 | 932/P = 2^-118.1358 | 26/P = 2^-123.2996 | 958/P = 2^-118.0961 | 932/P |
| 2^23 | all four | 26 | 25 | 934/P = 2^-118.1327 | 26/P = 2^-123.2996 | 960/P = 2^-118.0931 | 934/P |

The frontend term `(7 + 2 log2 N)/2^162`, the divisibility term `q^-7`, the
ordinary suffix terms, binding and hashing are unchanged by either rule.

**Against 2^-128.** Even the one-shot sum over P is already about 2^-118.1,
dominated by the ring switch's 648/P = 2^-118.66. The schedule relies on
per-site grinding: each site's loss times `2^-bits / P` must be at most
2^-128. Under the padding-point rule, the `Tau0Point` site's loss rises from
n to 2n. Its grinding target rises from 5 to 6 bits (loss 50 or 52; computed
from the exact inequality `loss * 2^128 <= P * 2^bits`). The `alpha` site
stays at 10 bits and `beta` at 1 bit. No new site is added, so the per-site
union in the Fiat–Shamir ledger keeps the same number of terms. Under the
deletion rule, no grinding run changes. This experimental path still makes no
overall Fiat–Shamir or SIS claim.

### 3.5 Pricing of the row-batching vector

The ordinary grinding policy prices a vector of independent batching
coefficients at `values/|E|`, through `independent_batch_loss_factor`
(`crates/akita-params/src/transcript_grinding.rs` lines 736-748). Its
documentation says the one-shot `1/|E|` bound "does not price this site"
under a coordinate-wise tree extractor, which recovers openings below the
challenge. The ordinary protocol avoids the issue: its `Tau1Point` is a
multilinear row point priced at its coordinate count
(`transcript_grinding/plan.rs` lines 207-211). The binary root instead draws
`K` independent scalars and prices them at 1 (grinding.rs lines 50-51).

**Conclusion.** The root's loss-1 charge is correct under the checkpoint's
candidate-comparison premise
([binary-checkpoint.md](binary-checkpoint.md) lines 576-591).
Two candidates under the same checkpoint input share the outgoing commitment
and typed manifest. If they differ, the ordinary source comparison map returns
an induced P-ring SIS collision. So the extracted source may be fixed before
`J` and every later batching draw, and a nonzero residual vector survives
`beta` with probability at most 1/P. Computational binding alone would not
justify fixing the candidate; the comparison map is what does.

The root's security admission must therefore adopt one of two premises:

- the checkpoint's candidate-comparison premise, stated as an explicit
  hypothesis together with the comparison map it relies on, under which
  `Tau1Point` costs 1/P; or
- the coordinate-wise tree model, under which `Tau1Point` costs `K/P` and must
  be ground to the bits in the table below.

This note does not choose between them. The checkpoint argument shows only the
first premise's conclusion for the fixed candidate it describes. It does not
show that the tree pricing is unnecessary in general. The root grinding plan is
also expressly an experimental execution plan, "not catalog or security
admission" (grinding.rs lines 10-15).

The table gives `K` for each rule. The fallback adds one slot to the deletion
count. The mixed rows also keep one independent coefficient per `{-1,0}` bit
address and the ordinary linear rows. The current column applies the existing
one-bit `Tau1Point` grind, so it shows `K/(2P)`; the unfiltered `K/P` exponent
is one larger. The bit columns are the smallest `b` with
`K·2^128 <= P·2^b` (loss 16 needs five bits, because `P < 2^128`).

| N | Shape | Current K | Deletion K | Fallback K | Current `K/(2P)` | Tree-model bits: current / deletion / fallback |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 2^22 | binary-claim | 18,912,399 | 13 | 14 | 2^-104.8272 | 25 / 4 / 4 |
| 2^22 | binary-and-prime-claims | 17,129,105 | 15 | 16 | 2^-104.9701 | 25 / 4 / 5 |
| 2^22 | binary-and-prime-claims-plus-ordinary | 12,230,332 | 69,690 | 69,691 | 2^-105.4560 | 24 / 17 / 17 |
| 2^22 | five-prime-claims | 43,594,261 | 21 | 22 | 2^-103.6224 | 26 / 5 / 5 |
| 2^23 | binary-claim | 41,849,999 | 13 | 14 | 2^-103.6813 | 26 / 4 / 4 |
| 2^23 | binary-and-prime-claims | 40,066,705 | 15 | 16 | 2^-103.7441 | 26 / 4 / 5 |
| 2^23 | binary-and-prime-claims-plus-ordinary | 30,973,624 | 36,918 | 36,919 | 2^-104.1155 | 25 / 16 / 16 |
| 2^23 | five-prime-claims | 32,977,429 | 21 | 22 | 2^-104.0250 | 25 / 5 / 5 |

`K = 7 + 2 [word] + 2 planes + 2·2 (B/D rows) + |Pad| + O`. Here `O` is the
mixed shapes' `{-1,0}` bit addresses plus their ordinary linear rows
(root_public.rs lines 123-137). The `{-1,0}` addresses are every coefficient of
the two compression segments (ordinary_root.rs lines 138-165): 65,536 + 4,096 =
69,632 at 2^22 and 32,768 + 4,096 = 36,864 at 2^23. The ordinary linear rows
number `live_blocks · A rank + logical B rows + 3` (ordinary_root.rs lines
100-111). They come to 43 at 2^22 and 39 at 2^23 for the selected ordinary
profiles (see the verification record for how these two counts were obtained).

Under the premise the root uses today, the batching loss is 1/P for every row of
the table and every rule. Under the tree model, the current per-cell rule
misses the 2^-128 per-site target by a factor of about 2^22.5 to 2^24.4. The
figures of about 2^-103.6 to 2^-105.5 are not demonstrated attack
probabilities. Under the tree
model, deletion needs four or five grinding bits for the non-mixed shapes. The
mixed shapes need 16 bits at 2^23 and 17 at 2^22, because their `{-1,0}` block
remains. Unfiltered, deletion's mixed loss is 69,690/P = 2^-111.9113 at 2^22 and
36,918/P = 2^-112.8280 at 2^23. **This choice of premise exists independently of
this note.**

## 4. Verifier evaluation of the padding row

This section applies only to the padding-point rule. The verifier needs

```text
Sigma_pad(tau_pad, r) = sum_(a in Pad) eq(tau_pad, a) eq(r, a),
```

multiplied by `beta_pad`. It must compute this without visiting `|Pad|`
addresses.

### 4.1 Layout and alignment

Segment order is fixed by the plan constructors:

1. Z and T, with stride 1024 (root_plan.rs lines 61-76);
2. U, K and K_val, with stride 256;
3. EP, with stride 256 (prime_plan.rs lines 21-32);
4. in mixed shapes, the native ordinary segments (root_plan.rs lines 108-123;
   ordinary_root.rs lines 78-98). These have `S_k = D_k` and no tails.

Each width `L_k S_k` is a multiple of its own stride (root_repacking.rs lines
63-67). Across the padded segments 1-3, the strides are non-increasing powers
of two. Every base is a sum of earlier widths, so **every padded segment's base
is a multiple of its own stride**: `B_k = h_k S_k`. This claim does not extend
to the native segments. In mixed shapes the first native segment has stride 512
and follows stride-256 segments. Its bases, 25,956,096 at 2^22 and 42,733,312 at
2^23, are both 256 modulo 512. Native segments contribute no padding, so the
factorization below never uses them, and the fallback is unaffected. At 2^22
`binary-and-prime-claims`:

| Segment | `B_k` | `S_k` | `B_k / S_k` | Lanes `L_k` | `D_k` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Z | 0 | 1024 | 0 | 16,384 | 648 |
| T | 16,777,216 | 1024 | 16,384 | 6,144 | 648 |
| U | 23,068,672 | 256 | 90,112 | 256 | 162 |
| K | 23,134,208 | 256 | 90,368 | 9 | 162 |
| K_val | 23,136,512 | 256 | 90,377 | 6 | 162 |
| EP | 23,138,048 | 256 | 90,383 | 11,008 | 162 |

Every padded segment is aligned in all eight measured shapes and in all six
small differential layouts. Alignment holds by construction, but **no code
checks it**:
`RootCoefficientRepacking::new` (root_repacking.rs lines 42-84) accepts any
segment order. The padding-point rule must add a typed `InvalidSetup` check
that `B_k mod S_k = 0` for every segment with `S_k > D_k`. Plane count does
not matter, because `is_padding_address` tests `(a - B_k) mod S_k >= D_k`.

### 4.2 Factorization

Write `e_i(1) = tau_i r_i`, `e_i(0) = (1 - tau_i)(1 - r_i)` and
`c_i = e_i(0) + e_i(1)`, so that `eq(tau, a) eq(r, a) = prod_i e_i(a_i)`.

If `B_k = h_k S_k`, then every address in segment `k` splits as `a = h S_k + t`.
Here `h` lies in `[h_k, h_k + L_k)` and occupies bits `s_k..n`, and `t` lies in
`[0, S_k)` and occupies bits `0..s_k`. The segment's padding is
`{h in [h_k, h_k+L_k)} x {t in [D_k, S_k)}`, so

```text
Sigma_k = ( sum_(h = h_k)^(h_k+L_k-1) prod_(i>=s_k) e_i(h_(i-s_k)) )
        * ( sum_(t = D_k)^(S_k-1)     prod_(i<s_k)  e_i(t_i) ),
Sigma_pad = sum_k Sigma_k + ( prod_(i<n) c_i - Pre_[0,n)(Lambda) ).
```

Each interval sum is a difference of two prefix sums. For a bit range of width
`w` starting at bit `p`, and `0 <= b < 2^w`,

```text
Pre_[p,p+w)(b) = sum_(y<b) prod_(j<w) e_(p+j)(y_j)
               = sum_(j : b_j = 1) [ prod_(j'>j) e_(p+j')(b_j') ] * e_(p+j)(0) * [ prod_(j'<j) c_(p+j') ],
```

and `Pre(2^w) = prod_(j<w) c_(p+j)`.

*Proof.* `y < b` if and only if at the highest bit `j` where they differ,
`b_j = 1` and `y_j = 0`. Above `j`, `y` equals `b`. Below `j`, `y` is
arbitrary, and the sum over arbitrary low bits factors into the product of the
`c`. ∎

### 4.3 Operation count

The reference evaluator does the following:

- **Shared work:** compute `e_i(0)`, `e_i(1)` and `c_i` for all n coordinates,
  which takes n multiplications and 4n additions or subtractions.
- **Per prefix sum** of width `w` and bound `b`: `w - 1` multiplications for the
  prefix products of `c`, `w - 1` for the running product, and two per set bit
  of `b`.
- **Per padded segment:** two high prefix sums of width `n - s_k`, two low prefix
  sums of width `s_k`, and one product. This is at most `8n + 1`
  multiplications.

Exact counts from that evaluator, including shared work:

| N | Shape | n | Padded segments | Multiplications | Additions/subtractions |
| --- | --- | ---: | ---: | ---: | ---: |
| 2^22 | binary-claim | 25 | 5 | 641 | 176 |
| 2^22 | binary-and-prime-claims | 25 | 6 | 765 | 198 |
| 2^22 | binary-and-prime-claims-plus-ordinary | 25 | 6 | 777 | 204 |
| 2^22 | five-prime-claims | 26 | 3 | 425 | 144 |
| 2^23 | binary-claim | 26 | 5 | 665 | 180 |
| 2^23 | binary-and-prime-claims | 26 | 6 | 793 | 202 |
| 2^23 | binary-and-prime-claims-plus-ordinary | 26 | 6 | 801 | 206 |
| 2^23 | five-prime-claims | 26 | 3 | 429 | 146 |

At 2^22 `binary-and-prime-claims`, the per-segment multiplications are Z 92, T 98, U 106,
K 112, K_val 120 and EP 124. The domain tail costs 88, and the shared work
costs 25. The final multiplication by `beta_pad` adds one. The worst case is
therefore **802 multiplications** (2^23 `binary-and-prime-claims-plus-ordinary`).

The current rule instead squeezes `|Pad|` challenges (17,129,090 and
40,066,690 for `binary-and-prime-claims`) and evaluates the same number of
`weight * eq(r, a)` terms. It also performs an O(2^n) `is_padding_address`
scan in `BinaryRootPublic::new` (root_public.rs lines 111-122). That scan can
be replaced by one comparison of the last natural address with `logical_len`.

The deletion rule needs none of this work.

## 5. Prover cost and bytes

**Prover.** The deletion rule removes `|Pad|` challenge squeezes and `|Pad|`
weight terms, leaving the Stage 2 weight table zero on `Pad`.

The padding-point rule replaces the squeezes with `|Pad|` products
`eq_hi(h) * eq_lo(t)`. Per segment, it builds an `S_k`-entry low table scaled
by `beta_pad` and an `L_k`-entry high table. That is about `|Pad| + Σ_k (S_k + L_k)`
multiplications (about 17.1M at 2^22 `binary-and-prime-claims`), against 17.1M transcript
squeezes today. The degree-three Stage 2 sumcheck over `2^n` cells is
unchanged by either rule.

**Proof fields.** Neither rule adds, removes or resizes a proof field.
Challenges are not transmitted. The padding-point rule raises the `Tau0Point`
nonce width by one bit (5 to 6 grinding bits). Nonces are encoded with
variable length up to `nonce_max_bytes(nonce_bits)` (`replay.rs` lines
116-128), so the total proof length can move by a few bytes. The same holds
for later variable-length encodings after the challenges change.

**Transcript and digests.** Both rules change `W` on `Pad`. Even with honest
`x|Pad = 0`, the Stage 2 round polynomials read `W` at partially bound points
that mix padding and live addresses. Every Stage 2 message therefore changes,
and with it every later challenge and digest. The padding-point rule also
changes the root grinding plan, whose digest is bound into the instance
descriptor. The full-size driver reports the resulting proof sizes and
digests as measurements and verifies each proof in a separate process. The
proof-size tables in [binary-opening.md](binary-opening.md)
("Measured proof sizes" and "Full-size measurements") must be re-measured.

**Ordinary bytes do not move.** Every file either rule changes is absent from
`origin/main` (checked with `git cat-file` for the root public, plan, weight
and grinding modules, `root_repacking.rs`, both `binary_root.rs` drivers and
the digest TSV). The binary root runs only for a group of kind
`GroupClaimKind::Binary` (binary_root.rs lines 44-52). The ordinary
`Tau1Point` and `Tau0Point` runs in `transcript_grinding/plan.rs` are
untouched. The 65,567-byte pin in
[`ordinary_grouped_root_bytes.rs`](../crates/akita-pcs/tests/ordinary_grouped_root_bytes.rs)
(line 173; 70,541 bytes before the v8 merge) is the gate, run with the extension features both disabled and
enabled.

## 6. Alternatives

**(a) Drop checks no consumer needs (the deletion rule).** Section 1 shows that
this covers all padding.

| Measure | Deletion rule | Padding-point rule |
| --- | --- | --- |
| Error | 0 | n/P |
| Verifier work | none | ≤ 802 multiplications |
| Grinding change | none | `Tau0Point` 5 → 6 bits |
| Alignment check needed | no | yes |
| Grinding plan change | none | yes |

**(b) Dense packing.** Removing the coefficient strides would leave only the
natural cells, `Lambda = |Nat|`. With successor degree 512, the domains and
remaining padding would be:

| N | Shape | Domain | Remaining padding |
| --- | --- | ---: | ---: |
| 2^22 | binary-claim | 2^24 | 2,135,170 |
| 2^22 | binary-and-prime-claims | 2^24 | 351,874 |
| 2^22 | binary-and-prime-claims-plus-ordinary | 2^25 | 12,160,642 |
| 2^22 | five-prime-claims | 2^25 | 10,039,808 |
| 2^23 | binary-claim | 2^25 | 8,295,554 |
| 2^23 | binary-and-prime-claims | 2^25 | 6,512,258 |
| 2^23 | binary-and-prime-claims-plus-ordinary | 2^26 | 30,936,706 |
| 2^23 | five-prime-claims | 2^26 | 32,977,408 |

Dense packing halves the domain, removing one Stage 2 round, in five of the
eight shapes. However, it leaves a domain tail that is still millions of cells
at 2^23. It also breaks the 1024- and 256-stride tensor axes that the factored
A product and the parity and prime weights use. It is out of scope here and
does not by itself remove the linear term.

**(c) Sparse random linear combinations.** Weights from a small set, such as
random subsets, lose a constant factor per test and need about 128 repetitions.
Per-cell randomness of any kind still costs `Omega(|Pad|)` verifier work to
evaluate `sum w(a) eq(r, a)`. Structured weights avoid that cost. Powers
`beta^a` factor over bits but have degree `2^n`, giving a `2^25/P = 2^-103`
loss. The eq weight in Section 2 is the structured choice with degree n.

**Recommendation: (a), the deletion rule.** It is the only option with zero
error, zero verifier work and no new alignment invariant. Under the tree model
of Section 3.5, it also removes the padding share of the `Tau1Point` vector.
The mixed shapes' `{-1,0}` block remains. Keep the padding-point rule as the
specified fallback.

## 7. Implementation plan

Stack on `quang/binary-source-verifier-weights`. That branch streams the
padding weights by run from the repacking geometry and adds a padding weight
family. This change deletes that family and its run iterator, and keeps
`natural_end` in `root_repacking.rs`. Steps 1–7 of the deletion rule are
implemented as written.

**Deletion rule.**

1. `BinaryRootPublic` stays the single compiler for both drivers. Remove
   `padding_count` from its fields and from `batch_count`, remove the padding
   range in `relation` (root_public.rs lines 111-137 and 196-224), and replace
   the domain scan with a `natural_end() <= logical_len` check.
2. Remove the `padding_weights` parameter from `binary_direct_a_weights` and
   `BinaryRootWitnessPlan::relation_weights`, and update their docs in the
   opening spec.
3. Fix the `Tau1Point` comment in grinding.rs (lines 50-51). Its loss stays 1.
4. Change every test that requires a padding cell to be rejected. Theorem 1
   makes these cases valid for the root equations, and review has accepted that
   inversion.
   - `Fault::Padding` in
     `crates/akita-pcs/tests/binary_end_to_end/fault_backend.rs`
     (lines 64-68) moves from must-reject to must-accept, with unchanged
     readings, provided the mutated cell stays within `[-4,3]`.
   - The same file's relation-violation assertion (lines 253-261) currently
     requires every fault except `ForeignCommitment` and `ForeignPartials` to
     violate the public relation. It
     must exempt `Fault::Padding`.
   - `crates/akita-types/src/binary/root_plan_tests.rs` (lines 90-99)
     sets the padding address 131071 to one and requires `check_table` to
     reject. That address must leave the must-reject list. The natural
     addresses in the same list stay.

   Keep every natural-cell fault as must-reject.
5. Keep `validate_digit_table` as a local prover check, with its doc reworded,
   because honest construction still writes zero padding. Its rejection tests
   in `root_repacking.rs` may stay. Rejection tests of
   `NativeRootRelation::ring_switch` may also stay, because that separate API
   still includes per-cell padding rows.
6. Add a regression that materializes the complete compiled weight table, using
   the branch's reference materializer, and asserts that it is zero on every
   padding address.
7. Re-pin the digests listed in Section 5 and re-measure the opening spec's
   tables. Confirm that the ordinary pin (65,567 bytes after the v8 merge,
   70,541 before it) is unchanged with features off and on.

**Padding-point rule, if selected instead.**

1. Draw `2n` coordinates at `Tau0Point` in both drivers.
2. Replace the per-cell slots with one `beta_pad` slot.
3. Add the alignment check to `RootCoefficientRepacking::new`.
4. Add a factored `Sigma_pad` evaluator next to the run iterator. The
   verifier's `relation.evaluate(r)` gains `beta_pad * Sigma_pad`.
5. Give the prover the dense padding row.
6. Raise the `Tau0Point` loss in grinding.rs to `2n`.
7. Keep the `Fault::Padding` rejection.
8. Because values now differ from the current construction, test the factored
   evaluator against a brute-force sum over `is_padding_address` for the
   fixture layouts, odd row counts, two owners and mixed native segments,
   including zero and one coordinates.
9. Extend `projection_orchestrators_fix_witness_then_rows_then_outputs_then_batches`
   to check that `tau_pad` precedes `beta` in both drivers.

## Verification record

The following were checked with exact integer arithmetic modulo P, using a
script outside the repository:

- **Layout counts.** The segment rules above reproduce every `Natural`,
  `Physical`, domain and padding count in the opening spec's layout table. The
  small N=8, C=2 word-plus-prime layout has 79,616 cells in a 131,072 domain.
- **Alignment.** Every padded segment's base is a multiple of its stride in
  all eight measured shapes. The mixed shapes' first native segment, with
  stride 512 and bases 25,956,096 and 42,733,312, is not aligned
  (256 mod 512) and has no padding.
- **Factored evaluator.** The evaluator in Section 4 equals the brute-force
  `sum_(a in Pad) eq(tau,a) eq(r,a)` on 24 cases. These cover six small
  layouts (word-only, word plus prime, two prime planes without a word claim,
  which binary groups no longer admit, 16 and 32 carriers, two owners, mixed native segments) with successor degree 64 and
  512, using random points and points with coordinates 0 and 1.
- **Batch counts and grinding.** The `K` values, the error exponents and the
  grinding bits follow from the formulas above. All eight rows of the
  Section 3.5 table were recomputed. The mixed rows decompose as
  15 + `|Pad|` + 69,632 + 43 at 2^22 and 15 + `|Pad|` + 36,864 + 39 at 2^23.
  The `{-1,0}` counts follow from the compression segment sizes in the opening
  spec, which the layout reproduction confirms. The ordinary linear-row counts,
  43 and 39, depend on the planner-selected ordinary profiles, which no
  checked-in file records. They are the remainder of the independent review's
  totals and were not recomputed from a profile.
- **Stage 3 rounds.** `k` is confirmed from the measured Stage 3 bytes and
  atoms.

These items were settled by the independent review and checked against the
cited lines:

- The successor norm argument does not assume zero padding (Section 1,
  successor row).
- The suffix `[Lambda, C)` is committed and `[C, Q)` is virtual (Section 1).

No number in [binary-opening.md](binary-opening.md) was found to
be wrong. The brief's figure of 17,129,090 counts padding weights only; the
complete order-7 `Tau1Point` vector at 2^22 `binary-and-prime-claims` has 17,129,105
entries.
