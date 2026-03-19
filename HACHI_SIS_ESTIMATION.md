# Hachi SIS Estimation

This note explains how the current repo turns a Hachi commitment configuration
into a concrete Module-SIS hardness estimate, and records a Sage-backed set of
numbers for the current 128-bit field profiles.

The point is not to re-explain SIS from scratch. The point is to make the
current code path legible, show where the actual estimator call happens, and
separate three questions that are easy to conflate:

- challenge entropy,
- witness geometry / norm growth,
- SIS binding hardness.

Main code anchors for the current tree:

- Current 128-bit field and modulus: `src/algebra/fields/fp128.rs:892-901`,
  `src/algebra/ntt/tables.rs:76-80`, `examples/profile.rs:33`
- Current commitment configs and layout formulas:
  `src/protocol/commitment/config.rs:13-18`,
  `src/protocol/commitment/config.rs:54-114`,
  `src/protocol/commitment/config.rs:116-167`,
  `src/protocol/commitment/config.rs:243-279`,
  `src/protocol/commitment/config.rs:663-814`
- Current sparse challenge sampling and stage-1 `z` bound:
  `src/protocol/challenges/sparse.rs:36-46`,
  `src/protocol/quadratic_equation.rs:55-67`,
  `src/protocol/quadratic_equation.rs:161-170`
- Current `w` / `M`-side width formulas:
  `src/protocol/ring_switch.rs:485-495`,
  `src/protocol/ring_switch.rs:661-687`,
  `src/protocol/ring_switch.rs:817-819`
- Current in-tree flattened SIS mirror and the profile-style debug summary:
  `src/protocol/labrador/config.rs:116-153`,
  `src/protocol/labrador/config.rs:173-274`,
  `src/protocol/labrador/config.rs:990-1043`
- Actual Sage estimator entrypoints:
  `../lattice-estimator/hachi_estimator.py:42-57`,
  `../lattice-estimator/estimator/sis_lattice.py:31-78`
- Reproducer for the prospective `D=64, K=256` onehot candidate and the
  `N_B = N_D = 1` cutoff sweep:
  `scripts/estimate_hachi_d64_k256_onehot_sis.py`
- Default estimator cost model knobs:
  `../lattice-estimator/estimator/conf.py:11-17`
- D=64 challenge-family discussion:
  `LOWERING_TO_D64.md`,
  `K64_EXACT_CHALLENGE_FAMILY.md`

## Scope

The numbers below use the current Hachi field, not the generic P13 scripts.

- The current repo instantiates `Prime128M8M4M1M0`, i.e.
  `q = 2^128 - 275`; see `src/algebra/fields/fp128.rs:900-901`,
  `src/algebra/ntt/tables.rs:79-80`, and `examples/profile.rs:33`.
- The runtime protocol infrastructure supports
  `D in {64, 128, 256, 512, 1024}`; see `src/protocol/dispatch.rs:1-18`,
  `src/protocol/dispatch.rs:27-52`.
- The production-oriented 128-bit config family is currently the
  `Fp128BoundedCommitmentConfig<..., 3, 3>` family, which fixes
  `D = 256`, `N_A = N_B = N_D = 1`, and `omega = 23`; see
  `src/protocol/commitment/config.rs:663-746`.
- The halving-D config uses `D = 256` at level 0 and `D = 128` later, with
  `omega = 23` at `D = 256` and `omega = 31` at `D = 128`; see
  `src/protocol/commitment/config.rs:751-814`.

The Sage numbers in this note were produced with the actual estimator via
`SIS.lattice(...)`, but with `BDGL16` and `lgsa` forced explicitly so they line
up with the older Hachi scripts and with the in-tree Rust mirror. This matters
because the estimator defaults are `MATZOV` and `gsa`; see
`../lattice-estimator/estimator/conf.py:11-17`.

So:

- `hachi_estimator.py` is still the right mental model for a one-off SIS call,
  but it defaults to a different prime (`P13`) and a simplified width model;
  see `../lattice-estimator/hachi_estimator.py:29-35`,
  `../lattice-estimator/hachi_estimator.py:42-57`.
- For current-Hachi numbers, the exact geometry should come from the live Rust
  layout formulas in `src/protocol/commitment/config.rs` and
  `src/protocol/ring_switch.rs`, then fed into the actual Sage estimator.

Before getting to the prospective `D = 64` story, there is one important
clarification.

### Two `D=64` Prime / Challenge Cases

There are two distinct 128-bit prime cases floating around in the `D = 64`
discussion, and they should not be conflated:

1. The **current in-tree field** is

   ```text
   q_current = 2^128 - 275.
   ```

   Under that field, the old `D = 64` challenge discussion was about the raw
   direct full-ring shells `(31, 10)` and `(28, 11)`: these were good entropy /
   blowup points, but they did not come with a rigorous split proof.

2. The **new future `D = 64` field choice** is

   ```text
   p32 = 2^128 - 5823,
   ```

   which satisfies `p32 ≡ 65 (mod 128)` and therefore supports the `k = 32`
   partial split. Under that field, the relevant rigorous challenge spaces are
   the split families `C_{21,<=6}` and `C_{22,<=6}` from
   `K64_EXACT_CHALLENGE_FAMILY.md`.

Going forward, the prospective `D = 64` candidate and the dedicated
`scripts/estimate_hachi_d64_k256_onehot_sis.py` reproducer should be read in
the second regime, i.e. with `p32 = 2^128 - 5823`. The existing production
`D >= 128` sections below still describe the current in-tree `q_current`
configurations.

## Prospective Locked-In Candidate

The current tree does **not** yet encode this as a top-level config, but after
the onehot `D=64` sweeps the most plausible candidate direction is now:

- field `p32 = 2^128 - 5823`,
- ring degree `D = 64`,
- onehot chunk size `K = 256`,
- onehot witness with `nv = 44`,
- `N_A = 1`,
- `N_B = N_D = 2`,
- and a rigorous split `k=32` `D=64` challenge family from
  `LOWERING_TO_D64.md`, with `C_{21,<=6}` as the best strict Pareto point and
  `C_{22,<=6}` as the same-budget higher-entropy variant.

The challenge-family reason for moving away from pure ternary is unchanged, but
the relevant family depends on which `D = 64` field story we mean:

- pure exact ternary does not reach `2^128` at `D = 64`,
- under the older direct full-ring `q_current = 2^128 - 275` exploration, the
  shells `(28, 11)` and `(31, 10)` were the best non-rigorous blowup points,
- the rigorous split family `C_{21,<=6}` already gives about `128.54`
  challenge bits at `L1` mass `54` under `p32 = 2^128 - 5823`,
- `C_{22,<=6}` gives about `129.38` bits at `L1` mass `56` if keeping the old
  support / `L1` / `L2^2` budgets matters more than shaving one unit of mass.

The important new point is that for this exact `K = 256 > D = 64` onehot
regime, the generic folded-witness bound

```text
beta_inf = 2^r_vars * challenge_mass * 2^(log_basis - 1)
```

is too pessimistic for the onehot `z_pre` path.

Why:

- `map_onehot_to_sparse_blocks` allows `K > D` as long as one divides the
  other, and here each onehot chunk contributes one nonzero ring element out of
  `K / D = 4`; the other three are zero; see
  `src/protocol/commitment/onehot.rs:25-110`.
- In the sparse onehot fold path, each active ring element is a monomial and
  `accum_onehot_coeff` just shifts and signs the sparse challenge coefficients;
  see `src/protocol/hachi_poly_ops/mod.rs:212-237`.

So for the mixed `{±1, ±2}` exact family, the natural onehot-side
single-witness bound is:

```text
||z_pre||_inf <= 2^r_vars * max_abs_challenge_coeff = 2^r_vars * 2
```

and the collision bound for SIS is:

```text
collision_inf(M) = 2^(r_vars + 2).
```

Under that onehot-aware model, the `nv = 44` candidate lands at:

- `(m_vars, r_vars) = (20, 18)`,
- onehot-aware `delta_fold = 7`,
- `A ~= 2^139.38` even under the conservative full-width proxy,
- `B/D ~= 2^185.90`,
- `M ~= 2^155.74` with the code-compatible `delta_fold = 9` width,
- `M ~= 2^156.33` with the tighter onehot-aware `delta_fold = 7` width.

So the `N_B = N_D = 2` candidate is comfortably above 128 bits on the SIS
side. The conservative overall floor in that readout is the `A` side, not
`B/D` or `M`.

For comparison, if `N_B = N_D = 1` are kept fixed in the same `D = 64`,
`K = 256` onehot regime, then the largest `nv` that still clears 128 bits under
the same onehot-aware model is:

```text
nv = 34.
```

The threshold is:

| `nv` | `(m_vars, r_vars)` | `delta_fold` | `B/D` bits | `M` bits | Overall min |
| --- | --- | ---: | ---: | ---: | ---: |
| `33` | `(15, 12)` | `5` | `128.24` | `207.36` | `128.24` |
| `34` | `(16, 12)` | `5` | `128.24` | `203.56` | `128.24` |
| `35` | `(16, 13)` | `6` | `118.84` | `181.58` | `118.84` |
| `44` | `(20, 18)` | `7` | `84.46` | `106.50` | `84.46` |

So:

- with `N_B = N_D = 1`, the practical onehot cutoff is `nv = 34`,
- with `N_B = N_D = 2`, the `nv = 44` candidate is back above 128 bits,
- and the gain comes mostly from the rank increase itself:
  `B/D: 84.46 -> 185.90`, `M: 106.50 -> 156.33`.

One caveat worth recording explicitly: the current code path in
`compute_num_digits_fold` still uses the generic folded-witness proxy from
`src/protocol/commitment/config.rs:97-114`. If that older dense proxy is used
instead of the onehot-aware bound above, the `N_B = N_D = 1` cutoff is stricter
than `nv = 34`. The table in this subsection is therefore best read as the
prospective candidate-parameter story, not as a statement about what the
current code already proves automatically.

## Symbol Table

| Symbol | Meaning in this note | Current code anchor |
| --- | --- | --- |
| `D` | Ring degree | `src/protocol/dispatch.rs:1-18` |
| `alpha = log2(D)` | Number of inner ring-opening variables | `src/protocol/commitment/config.rs:719-730` |
| `m_vars` | Block-size exponent, so `block_len = 2^m_vars` | `src/protocol/commitment/config.rs:116-167`, `src/protocol/commitment/config.rs:243-279` |
| `r_vars` | Block-count exponent, so `num_blocks = 2^r_vars` | `src/protocol/commitment/config.rs:116-167`, `src/protocol/commitment/config.rs:243-279` |
| `delta_commit` | Digit depth for committed coefficients | `src/protocol/commitment/config.rs:13-18`, `src/protocol/commitment/config.rs:54-95` |
| `delta_open` | Digit depth for opened folded values | `src/protocol/commitment/config.rs:41-52`, `src/protocol/commitment/config.rs:54-95` |
| `delta_fold` | Digit depth for stage-1 folded witness `z_pre` | `src/protocol/commitment/config.rs:97-114` |
| `inner_width` | Width of the inner `A` matrix | `src/protocol/commitment/config.rs:257-259` |
| `outer_width` | Width of the outer `B` matrix | `src/protocol/commitment/config.rs:260-263` |
| `d_matrix_width` | Width of the prover-side `D` matrix | `src/protocol/commitment/config.rs:264-266` |
| `beta_inf` | Current `l_infty` bound for stage-1 `z_pre` | `src/protocol/commitment/config.rs:684-687`, `src/protocol/quadratic_equation.rs:62-67` |
| `collision_inf` | Conservative collision bound used for SIS | Derived below; matches `2 * beta_inf` in `src/protocol/labrador/config.rs:1002-1007` |
| `rank` | Module rank after flattening the relevant layer | `src/protocol/labrador/config.rs:998-1001`, `src/protocol/ring_switch.rs:817-819` |
| `width_ring_elems` | Number of ring columns before flattening | `src/protocol/labrador/config.rs:999-1001` |

## The Pipeline In One Picture

```text
config alias
    |
    | choose D, N_A, N_B, N_D, omega, log_basis, log_commit_bound
    v
decomposition depths
    |
    | delta_commit, delta_open, delta_fold
    v
layout search
    |
    | optimal (m_vars, r_vars)
    v
matrix widths
    |
    | inner_width, outer_width, d_matrix_width, M-width
    v
collision bound
    |
    | digit layers: 7
    | folded M-side: 2 * beta_inf
    v
flatten Module-SIS -> SIS
    |
    | n = rank * D
    | m = width_ring_elems * D
    | B_l2 = sqrt(m) * collision_inf
    v
SIS.Parameters(...)
    |
    | Sage picks attack dimension d_att
    | solves for delta_req, beta_BKZ, rop
    v
security estimate
```

This is exactly the current in-tree Rust mirror of the Euclidean SIS path:
`src/protocol/labrador/config.rs:173-274`.

## 1. From Hachi Config To Matrix Widths

The current config family exposes three main top-level 128-bit aliases:

- `Fp128FullCommitmentConfig = <128, 3>`:
  `src/protocol/commitment/config.rs:734-735`
- `Fp128OneHotCommitmentConfig = <1, 3>`:
  `src/protocol/commitment/config.rs:737-741`
- `Fp128LogBasisCommitmentConfig = <3, 3>`:
  `src/protocol/commitment/config.rs:743-746`

All three use base-8 digits because `log_basis = 3`.

The first thing the code computes is the digit depth:

```text
delta_commit = num balanced base-8 digits needed for committed coefficients
delta_open   = num balanced base-8 digits needed for opened folded values
delta_fold   = num balanced base-8 digits needed for z_pre
```

The exact code path is:

- `compute_num_digits(log_bound, log_basis)`:
  `src/protocol/commitment/config.rs:54-95`
- `compute_num_digits_fold(r_vars, challenge_weight, log_basis)`:
  `src/protocol/commitment/config.rs:97-114`

For the current aliases:

- `Full`: `delta_commit = 43`, `delta_open = 43`
- `OneHot`: `delta_commit = 1`, `delta_open = 43`
- `LogBasis`: `delta_commit = 1`, `delta_open = 43`

That `delta_open = 43` for `OneHot` and `LogBasis` is deliberate. The code sets
`log_open_bound = 128` whenever `log_commit_bound < 128`, because folding with
arbitrary field weights can still create full-width coefficients; see
`src/protocol/commitment/config.rs:703-712`.

Next, the planner searches `(m_vars, r_vars)` using the current witness-size
proxy

```text
C(r, m) = (delta_open + N_A * delta_commit) * 2^r
        + delta_commit * delta_fold(r) * 2^m
```

which is exactly the logic in `optimal_m_r_split`:
`src/protocol/commitment/config.rs:116-167`.

Finally, `HachiCommitmentLayout::new_with_decomp` turns that into the actual
matrix widths:

```text
num_blocks    = 2^r_vars
block_len     = 2^m_vars
inner_width   = block_len * delta_commit
outer_width   = N_A * delta_open * num_blocks
d_matrix_width = delta_open * num_blocks
```

See `src/protocol/commitment/config.rs:243-279`.

## 2. Where The Norm Bound Comes From

There are two different norm stories in the current code.

### 2.1 Digit-layer collisions: `A`, `B`, and `D`

At the digit layers, the witness entries are balanced base-8 digits in
`[-4, 3]`. A collision vector is a difference of two such witnesses, so each
coordinate lies in `[-7, 7]`.

So the conservative per-coordinate collision bound for `A`, `B`, and `D` is:

```text
digit_collision_inf = 8 - 1 = 7
```

The Euclidean bound fed to the estimator is then:

```text
B_l2 = sqrt(width_ring_elems * D) * 7
```

### 2.2 Folded witness / handoff side: `M`

For the stage-1 folded witness `z_pre`, the code enforces

```text
||z_pre||_inf <= beta_inf = 2^r_vars * omega * 2^(log_basis - 1)
```

for the pure ternary family; see
`src/protocol/commitment/config.rs:684-687` and the actual runtime check in
`src/protocol/quadratic_equation.rs:55-67`.

Since the current top-level sparse challenge sampler uses the exact ternary
alphabet `{-1, +1}`, `omega` is exactly the challenge Hamming weight; see
`src/protocol/quadratic_equation.rs:161-170` and the domain separation in
`src/protocol/challenges/sparse.rs:36-46`.

For SIS binding we care about a collision between two admissible witnesses, so
the conservative collision bound is

```text
collision_inf = 2 * beta_inf.
```

This is also what the current Rust summary test uses:
`src/protocol/labrador/config.rs:1002-1007`.

For a `{±1, ±2}` `D=64` challenge family, the current code does not yet have a
native formula, because the sampler and `beta` bound assume ternary exact
weight. For a conservative SIS estimate, the natural substitution is to charge
by challenge `L1` mass:

```text
challenge_mass = n1 + 2*n2
beta_inf       = 2^r_vars * challenge_mass * 2^(log_basis - 1)
collision_inf  = 2 * beta_inf
```

That is the convention I use below for the rigorous split families from
`LOWERING_TO_D64.md`: use `54` for `C_{21,<=6}` and `56` for `C_{22,<=6}` under
the future split field `p32 = 2^128 - 5823`. For comparison, the old raw direct
full-ring shells `(28, 11)` and `(31, 10)` from the `q_current = 2^128 - 275`
discussion have mass `50` and `51`. In the current `D=64` layouts all four
masses still give the same code-compatible `delta_fold = 7`, so the downstream
SIS floor does not materially change once the challenge entropy is repaired.

## 3. Flattening Module-SIS To The Actual Sage Call

Once the width and norm bound are fixed, the current estimator path is:

```text
n = rank * D
m = width_ring_elems * D
B_l2 = sqrt(m) * collision_inf
```

and then

```python
params = SIS.Parameters(
    n=n,
    q=q,
    m=m,
    length_bound=B_l2,
    norm=2,
)
cost = SIS.lattice(
    params,
    red_cost_model=RC.BDGL16,
    red_shape_model="lgsa",
)
```

The generic one-off script `hachi_estimator.py` does the same kind of mapping,
just with a simplified width model:
`../lattice-estimator/hachi_estimator.py:42-57`.

Inside the estimator, the Euclidean path does three things:

1. Pick an attack dimension `d_att`, defaulting to the optimizer's heuristic
   `d_att ~= sqrt(n log q / log delta)`; see
   `../lattice-estimator/estimator/sis_lattice.py:39-47`,
   `../lattice-estimator/estimator/sis_lattice.py:61-63`.
2. Solve for the root-Hermite factor required to make a `q`-ary lattice vector
   short enough:

   ```text
   delta_req^(d_att - 1) * q^(n / d_att) <= B_l2
   ```

   which is exactly `_solve_for_delta_euclidean`; see
   `../lattice-estimator/estimator/sis_lattice.py:31-36`.
3. Convert `delta_req` to a BKZ block size `beta`, then apply the chosen cost
   model; see `../lattice-estimator/estimator/sis_lattice.py:64-78`.

The estimator also has an important feasibility gate:

```text
lb = min(sqrt(n ln q), sqrt(d_att) * q^(n / d_att))
```

and if `B_l2 <= lb`, it treats the instance as having no relevant short
solution in that regime. That is why some parameter sweeps return `rop = inf`;
see `../lattice-estimator/estimator/sis_lattice.py:75-78`.

The in-tree Rust mirror exposes the same pieces more transparently as
`attack_dimension`, `required_delta`, `bkz_beta`, `solution_exists`, and
`log2_rop_bdgl16`; see `src/protocol/labrador/config.rs:125-153`,
`src/protocol/labrador/config.rs:173-274`.

## 4. Current Field And Config Facts

There are three current facts worth keeping separate.

### 4.1 The current Hachi field is not the P13 field

The generic estimator scripts in `../lattice-estimator/` often use
`P13 = 2^128 - 2^13 - 2^4 + 1`; see
`../lattice-estimator/hachi_estimator.py:29-35`.

The current Hachi code path instead uses:

```text
q = 2^128 - 275
```

via `Prime128M8M4M1M0`; see `src/algebra/fields/fp128.rs:900-901`.

### 4.2 The current 128-bit production config is really a `D = 256` profile

`Fp128BoundedCommitmentConfig<..., 3, 3>` fixes

```text
D = 256
N_A = N_B = N_D = 1
omega = 23
```

See `src/protocol/commitment/config.rs:663-746`.

### 4.3 `D = 128` is currently a challenge-entropy boundary, not an SIS boundary

The halving-D config explicitly says:

- it halves from `D = 256` to `D = 128`,
- it stops there,
- and the reason is that `D = 128` is the minimum ring dimension for which the
  sparse ternary challenge family still has enough entropy.

See `src/protocol/commitment/config.rs:751-760`.

That comment should be read literally. Under the current flattened SIS model,
the binding hardness at `D = 128` is still far above 128 bits. The thing that
breaks first is the exact ternary challenge family, not SIS.

## 5. Actual Sage Numbers For Current `D = 256` Profiles

I anchored the first table at `max_num_vars = 25` because the repo already has
an in-tree summary test at exactly that point:
`src/protocol/labrador/config.rs:990-1043`.

That Rust test printed

```text
log2(rop_bdgl16) = 883.17
```

for the `Full` handoff-style `M` instance, and the direct Sage run below matches
it exactly.

### 5.1 `max_num_vars = 25`

All rows use the current field `q = 2^128 - 275`, base 8, and the actual Sage
estimator with `BDGL16/lgsa`.

| Config | `(m_vars, r_vars)` | `(delta_commit, delta_open, delta_fold)` | `(inner_width, outer_width, M_width)` | `collision_inf(M)` | `A` bits | `B/D` bits | `M` bits | Overall min |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `Full` | `(8, 9)` | `(43, 43, 6)` | `(11008, 22016, 110080)` | `94208` | `791.43` | `725.67` | `883.17` | `725.67` |
| `OneHot / LogBasis` | `(10, 7)` | `(1, 43, 5)` | `(1024, 5504, 16128)` | `23552` | `1094.43` | `866.23` | `1188.78` | `866.23` |

Here:

- `A` means the inner commitment layer with rank `N_A = 1` and width
  `inner_width`.
- `B/D` means the outer commitment layers with rank `1` and width
  `outer_width = d_matrix_width`.
- `M` means the handoff-style extracted instance with
  `rank = N_D + N_B + 2 + N_A = 5` and width
  `M_width = d_matrix_width + outer_width + inner_width * delta_fold`; see
  `src/protocol/labrador/config.rs:998-1001`. This is the same handoff-style
  summary used by `print_profile_style_handoff_sis_summary`; it is not the
  slightly wider full `w` witness that still includes the quotient tail from
  `src/protocol/ring_switch.rs:485-495`.

The two important takeaways are:

- `OneHot` and `LogBasis` have identical SIS geometry here. That is not a typo.
  Under the current planner, they both have `delta_commit = 1` and
  `delta_open = 43`, so the layout and all SIS instances coincide.
- For the current `D = 256` profiles, the bottleneck is not the folded `M`
  side. It is the narrower `B` / `D` commitment layers.

### 5.2 `max_num_vars = 30`

This is the same calculation at a larger outer dimension.

| Config | `(m_vars, r_vars)` | `(delta_commit, delta_open, delta_fold)` | `(inner_width, outer_width, M_width)` | `collision_inf(M)` | `A` bits | `B/D` bits | `M` bits | Overall min |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `Full` | `(10, 12)` | `(43, 43, 7)` | `(44032, 176128, 660480)` | `753664` | `667.22` | `568.43` | `634.19` | `568.43` |
| `OneHot / LogBasis` | `(12, 10)` | `(1, 43, 6)` | `(4096, 44032, 112640)` | `188416` | `901.01` | `667.22` | `813.04` | `667.22` |

The trend is exactly what the live formulas predict:

- larger `max_num_vars` pushes both widths and folded norms up,
- security drops,
- but the gap to 128 bits is still enormous,
- and `B/D` remains the bottleneck layer in the current `D = 256` profiles.

## 6. Ring-Dimension Sweep With Current-Style Full-Field Parameters

To isolate the effect of ring dimension, I ran the same current-style full-field
profile at `max_num_vars = 25` for the runtime-supported `D` values, using:

- `q_current = 2^128 - 275` for the current `D >= 128` rows,
- `p32 = 2^128 - 5823` for the prospective split `D = 64` row,
- `N_A = N_B = N_D = 1`,
- base 8,
- `delta_commit = delta_open = 43`,
- the minimum exact ternary weight meeting about 128 bits of challenge entropy
  for `D >= 128`,
- and for `D = 64`, the rigorous split `k=32` family `C_{21,<=6}` from
  `LOWERING_TO_D64.md`, with `C_{22,<=6}` as a same-budget higher-entropy
  variant.

For the pure ternary family, the minimum weights are:

- `D = 128`: `omega = 31`, about `129.62` challenge bits
- `D = 256`: `omega = 23`, about `131.08` challenge bits
- `D = 512`: `omega = 19`, about `132.76` challenge bits
- `D = 1024`: `omega = 16`, about `131.58` challenge bits

For `D = 64`, pure ternary exact-weight never reaches `2^128` challenges. Under
the future split field `p32 = 2^128 - 5823`, the rigorous split `k=32` family
already repairs this:

- `C_{21,<=6}`: about `128.54` challenge bits, `L1` mass `54`
- `C_{22,<=6}`: about `129.38` challenge bits, `L1` mass `56`

The old raw direct full-ring shells `(28, 11)` and `(31, 10)` from the
`q_current = 2^128 - 275` exploration still have smaller mass `50` and `51`,
but they do not yet have a proof. Using the rigorous mass `54` in the
conservative `beta_inf` formula gives:

| `D` | Challenge family | Challenge mass used in `beta_inf` | `(m_vars, r_vars)` | `delta_fold` | `B/D` bits | `M` bits | Overall min | Bottleneck |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `64` | split-field `p32`, rigorous `k=32` `C_{21,<=6}` | `54` | `(9, 10)` | `7` | `151.11` | `148.46` | `148.46` | `M` |
| `128` | ternary exact weight `31` | `31` | `(8, 10)` | `6` | `315.58` | `349.80` | `315.58` | `B/D` |
| `256` | ternary exact weight `23` | `23` | `(8, 9)` | `6` | `725.67` | `883.17` | `725.67` | `B/D` |
| `512` | ternary exact weight `19` | `19` | `(7, 9)` | `6` | `1526.70` | `2018.30` | `1526.70` | `B/D` |
| `1024` | ternary exact weight `16` | `16` | `(7, 8)` | `6` | `3454.90` | `4931.76` | `3454.90` | `B/D` |

Here `C_{22,<=6}` gives the same downstream row in this profile as well,
because both masses `54` and `56` still induce `delta_fold = 7`.

Switching the prospective `D = 64` row from the older `q_current` proxy to the
actual split field `p32` does not change the displayed SIS numbers at the
current two-decimal precision.

This is the cleanest current answer to the "minimum secure ring dimension"
question, under the current flattened SIS model:

- `D = 128` is not remotely close to the SIS floor.
- `D = 64` is not immediately ruled out by SIS either, provided the challenge
  family changes first.
- The current code comment that "D=128 is the minimum" is about exact sparse
  ternary challenge entropy, not about binding hardness.

## Fixed `D * log q` Comparison

The next natural question is what happens if we keep the rough "bit budget per
ring element"

```text
D * log2(q)
```

roughly fixed, but trade larger fields against smaller ring dimension.

I ran the same current-style full-field profile with:

- base 8,
- `N_A = N_B = N_D = 1`,
- `max_num_vars = 25`,
- about `2^128` challenge entropy,
- and the largest primes below `2^32` and `2^64` that are `5 mod 8`.

Those primes are:

- `q32 = 2^32 - 99 = 4294967197`
- `q64 = 2^64 - 59 = 18446744073709551557`

For context I also include the prospective split-field choice
`p32 = 2^128 - 5823` at `D = 64`, using the rigorous split `k=32` family
`C_{21,<=6}` with conservative `L1` mass `54`. The same row also covers
`C_{22,<=6}` in this profile because both masses still give `delta_fold = 7`.

All three rows have essentially the same total ring-element bit budget:

```text
D * log2(q) ~= 8192.
```

| Field / ring choice | `log2(q)` | `D` | Challenge family | `(delta_commit, delta_open, delta_fold)` | `(m_vars, r_vars)` | `A` bits | `B/D` bits | `M` bits | Overall min |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| `q32, D=256` | `32` | `256` | ternary exact weight `23` | `(11, 11, 6)` | `(8, 9)` | `180.42` | `164.60` | `183.34` | `164.60` |
| `q64, D=128` | `64` | `128` | ternary exact weight `31` | `(22, 22, 6)` | `(8, 10)` | `180.42` | `150.82` | `160.19` | `150.82` |
| `p32, D=64` | `128` | `64` | rigorous split `k=32` `C_{21,<=6}` | `(43, 43, 7)` | `(9, 10)` | `164.89` | `151.11` | `148.46` | `148.46` |

The most revealing coincidence is the `A` layer:

- `q32, D = 256` and `q64, D = 128` both land at exactly `180.42` bits.

That is not magic. It comes from the geometry:

- `q32, D = 256` has `delta_commit = 11`, `inner_width = 2816`, so
  `A_coords = inner_width * D = 2816 * 256 = 720896`.
- `q64, D = 128` has `delta_commit = 22`, `inner_width = 5632`, so
  `A_coords = inner_width * D = 5632 * 128 = 720896`.

So the `A` instances have:

- the same SIS width in coordinates,
- the same `n * log2(q)` product, because `D * log2(q)` is fixed,
- and therefore essentially the same Euclidean SIS estimate.

This is the cleanest explanation for why "larger field, smaller ring dimension"
does not immediately look worse in the estimator.

But the other layers are not invariant, because the rest of the protocol is not
just "one flat q-ary vector."

The main effects are:

- `delta_commit` and `delta_open` scale roughly like `log2(q) / log_basis`, so
  larger fields really do cost more digits.
- challenge entropy depends on `D`, not on `q`, so smaller `D` needs a heavier
  exact sparse challenge family.
- smaller `D` also means `alpha = log2(D)` is smaller, which leaves more outer
  variables available and can move the layout search toward larger `r_vars`.

That is exactly what happens here:

- `q32, D = 256` uses ternary weight `23` and lands at `r_vars = 9`.
- `q64, D = 128` uses ternary weight `31` and lands at `r_vars = 10`.
- the larger `r_vars` and larger `delta_open` double the `B/D` coordinate count
  from `1441792` to `2883584`, which is why `B/D` drops from `164.60` to
  `150.82` bits.

On the folded `M` side the same effect is stronger:

- `q32, D = 256`: `M_coords = 7208960`, `collision_inf = 94208`
- `q64, D = 128`: `M_coords = 10092544`, `collision_inf = 253952`
- `p32, D = 64`: `M_coords = 15499264`, `collision_inf = 409600`

So the real lesson is not "bigger fields let us cheat and use fewer bits."
The real lesson is:

```text
the estimator mostly cares about a q-ary volume term like n * log(q),
while the protocol geometry separately cares about digit depth, challenge
entropy, and folded-norm growth.
```

Those move differently when you trade `q` against `D`.

## 7. What Actually Drives The Hardness

There are four knobs doing almost all the work.

### 7.1 `D` enters twice, and both effects matter

When you flatten a rank-`r` module instance over degree `D`, you get:

```text
n = r * D
m = width_ring_elems * D
```

So dropping `D` shrinks both the SIS row dimension and the total coordinate
count. That is why the security numbers fall quickly as `D` goes from `1024` to
`64`.

### 7.2 `delta_commit` is a huge knob

The biggest non-`D` effect in the current configs is simply:

```text
delta_commit: 43 -> 1
```

when you move from `Full` to `OneHot` or `LogBasis`; see
`src/protocol/commitment/config.rs:734-746`.

That shrinks:

- `inner_width`,
- the dominant `z_pre` width term,
- and, indirectly, the planner's preferred `(m_vars, r_vars)`.

That is why `OneHot / LogBasis` are not just a little stronger than `Full`.
They are dramatically stronger in the current SIS model.

### 7.3 `omega` hurts through the folded norm, not through the digit layers

The digit-layer collisions still pay only the base-8 collision cost `7`. The
challenge weight only enters the `M`-side through

```text
beta_inf = 2^r_vars * omega * 4
collision_inf = 2 * beta_inf.
```

So the `D = 64` mixed-family story is:

- exact ternary entropy fails first,
- mixed exact families repair entropy,
- and after that repair, the conservative SIS estimate is still above 128 bits.

That is exactly the split already hinted at in `LOWERING_TO_D64.md` and now
made rigorous in `K64_EXACT_CHALLENGE_FAMILY.md`.

### 7.4 The weakest layer is not always the folded layer

At current `D >= 128`, the smallest-security layer is usually `B` / `D`, not
the wider folded `M` instance. Intuitively:

- the `M` side has a much larger collision bound,
- but it also gets a larger rank after flattening,
- while `B` / `D` stay at rank 1 and can end up geometrically weaker.

At `D = 64` with the repaired `{±1, ±2}` challenge families, the bottleneck
finally flips to `M`.

## 8. Bottom Line

For the current 128-bit field and current Hachi geometry:

- the exact current `D = 256` profiles are nowhere near the 128-bit SIS floor,
- `OneHot` and `LogBasis` are much stronger than `Full` because `delta_commit`
  drops from `43` to `1`,
- the current `D = 128 minimum` claim is a challenge-family statement, not an
  SIS statement,
- and under the same conservative flattened-SIS model, a `D = 64` repaired
  `{±1, ±2}` family still lands around `148` bits.

So the current evidence is:

```text
challenge entropy breaks before SIS binding.
```

That was already the right high-level intuition. The actual Sage numbers now
make it explicit.

## Reproducing The Main Checks

Current in-tree Rust mirror:

```bash
cargo test print_profile_style_handoff_sis_summary -- --nocapture
```

That currently prints the `Full`, `D = 256`, `max_num_vars = 25` handoff-style
summary in `src/protocol/labrador/config.rs:990-1043`, including the
`883.17`-bit `M` estimate.

Actual Sage estimator shape:

```python
from estimator import SIS
from estimator.reduction import RC

params = SIS.Parameters(
    n=rank * D,
    q=2**128 - 275,
    m=width_ring_elems * D,
    length_bound=(width_ring_elems * D) ** 0.5 * collision_inf,
    norm=2,
)
cost = SIS.lattice(
    params,
    red_cost_model=RC.BDGL16,
    red_shape_model="lgsa",
)
```

For quick generic sweeps, `../lattice-estimator/hachi_estimator.py` remains the
right place to start. For exact current-Hachi numbers, derive the live layout
from `src/protocol/commitment/config.rs` and `src/protocol/ring_switch.rs`
first, then call Sage.
