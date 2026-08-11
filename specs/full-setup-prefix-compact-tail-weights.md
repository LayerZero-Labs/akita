# Spec: Full Setup Prefix with Compact Zero-Tail Weights


| Field         | Value                  |
| ------------- | ---------------------- |
| Author(s)     | Amirhossein Khajehpour |
| Created       | 2026-08-10             |
| Status        | implemented            |
| PR            | #386                   |
| Supersedes    | Setup-prefix zero-padding and natural-capacity portions of `flat-public-matrix-and-exact-ntt-cache.md`, `distributed-setup-offloading.md`, and `setup-prefix-ladder.md` |
| Superseded-by |                        |
| Book-chapter  | book/src/how/verifying/setup_contribution.md |


## Summary

Recursive Stage 3 currently commits to the natural flat setup prefix followed
by explicit zeros up to its power-of-two commitment domain. This spec moves the
zero tail from the committed setup polynomial to the structured setup-weight
polynomial. The committed polynomial becomes the actual power-of-two prefix of
the public setup matrix, including genuine setup coefficients after the active
Stage-2/Stage-3 footprint. The setup-index weight remains supported only on the
natural footprint and is zero outside it.

The Boolean setup-product claim is unchanged because every newly included setup
coefficient is multiplied by zero. The off-hypercube Stage-3 polynomial does
change: its terminal setup claim is now an evaluation of the full committed
prefix. The verifier checks that claimed evaluation against the compact
setup-weight MLE and carries the setup opening into the next fold, where it is
batched with the next-witness opening and proven against the full-prefix
commitment.

This design lets one power-of-two setup-prefix commitment serve directly as the
recursive opening source. It does not require the verifier to scan or reconstruct
the natural prefix. The verifier's local setup-dependent work remains the compact
evaluation of a zero-tail sum of shifted paired-equality tensors.

## Intent



### Goal

Commit the actual setup coefficients in the complete power-of-two prefix and
make the compact setup-index weight polynomial the sole owner of natural-prefix
selection.

The primary protocol surfaces are:

- `SetupPrefixSlotId`, `SetupPrefixSlot`, and `SetupPrefixVerifierSlot`, which
identify and bind an **actual full setup prefix** rather than a zero-padded
natural prefix;
- `commit_setup_prefix` and the recursive `SetupPrefix` source, which consume
all `n_prefix` setup coefficients;
- `RectangularSetupProductTerm`, which uses the full setup table but an
active-support setup-index factor;
- `SetupContributionPlan::evaluate_setup_index_weight_mle`, which evaluates the
zero-tail setup weight without materializing its padded vector;
- `SetupSumcheckProof::setup_prefix_eval`, which becomes a deferred evaluation
claim about the full setup prefix; and
- the suffix grouped opening, which must discharge that exact claim against the
exact full-prefix commitment.



### Terminology

For one Stage-3 setup product, write:

```text
d0              common base ring dimension
R               required setup rows in the natural A/B/D footprint
N               next_power_of_two(R)
natural_len     R * d0 flat field coefficients
n_prefix        N * d0 flat field coefficients
S_full          actual public setup coefficients [0, n_prefix)
W_R             structured setup-index weights on [0, N), zero on [R, N)
```

Because `d0` is a power of two,

```text
next_power_of_two(R * d0) = next_power_of_two(R) * d0 = N * d0.
```

`natural_len` remains protocol metadata describing the active setup footprint.
It no longer describes the number of setup coefficients copied into the
committed polynomial. `n_prefix` is both the committed flat coefficient length
and the amount of actual shared setup that must be available during
preprocessing.

### Invariants

- **Actual-prefix commitment.** A full-prefix slot commits to
`shared_setup[0..n_prefix]`. No coefficient in that interval is synthesized
as zero.
- **Natural support belongs to the weight.** For every Boolean setup-row index
`i >= R`, `W_R[i] == 0`, even though `S_full[i, y]` may be nonzero.
- **Input-claim preservation.** Replacing the zero-padded setup source with the
actual full source does not change the Stage-3 input claim.
- **Full-prefix terminal semantics.** `setup_prefix_eval` equals
`S_full_tilde(rho_setup_idx, rho_y)`, not the MLE of a source zero-padded at
`natural_len`.
- **Deferred claim is mandatory.** An accepted nonterminal recursive proof may
not drop the tuple `(slot_id, commitment, stage3_setup_point, setup_prefix_eval)`. The successor grouped opening must authenticate it.
- **One content identity.** Zero-padded-natural and actual-full prefix artifacts
must never share a cache key or serialized identity. The cutover must add a
content-semantics tag or bump the slot/cache format so stale artifacts reject.
- **Canonical variable order.** The full-prefix commitment, Stage-3 table, and
recursive opening source use the same little-endian flat coefficient order:
coefficient variables first, then setup-index variables.
- **Compact verifier.** The verifier does not allocate `N` setup-index weights
or an `N`-entry setup-index equality table to evaluate `W_R_tilde`.
- **No-panic verifier boundary.** Invalid lengths, missing slots, mismatched
content modes, out-of-domain addresses, and excessive recurrence work return
`AkitaError` or `SerializationError`.
- **Terminal discharge.** A terminal level cannot finish with an unverified
full-prefix opening claim.



### Non-Goals

- Changing the algebraic A/B/D setup contribution or its packed address map.
- Making setup tails secret. `S_full` is derived from the public setup seed.
- Adding a second generic range-selector API when
`SetupContributionPlan::evaluate_setup_index_weight_mle` already owns the
protocol weight.
- Deriving a truncated setup evaluation from an ordinary full-prefix opening at
the same point without a sumcheck.
- Changing the Stage-3 proof degree or adding a separate prefix-range sumcheck.
- Changing group-local opening points or merging the setup and witness groups.



## Mathematical Design



### Current and Target Committed Tables

Let the active setup use `R` rows and let `N = next_power_of_two(R)`. Suppress
the coefficient coordinate temporarily.

The current committed table is

```text
S_zero[i] = S[i]  for 0 <= i < R
S_zero[i] = 0     for R <= i < N.
```

The target committed table is

```text
S_full[i] = S[i]  for 0 <= i < N,
```

where every `S[i]` is the actual public setup entry. The target weight is

```text
W_R[i] = W[i]  for 0 <= i < R
W_R[i] = 0     for R <= i < N.
```

Therefore, on the Boolean cube,

```text
sum_{i=0}^{N-1} S_full[i] * W_R[i]
    = sum_{i=0}^{R-1} S[i] * W[i]
    = sum_{i=0}^{N-1} S_zero[i] * W_R[i].
```

The protocol statement being proved is unchanged.

### Complete Stage-3 Relation

Restore the coefficient coordinate `y in [0, d0)`. Define the power table

```text
A_alpha[y] = alpha^y.
```

The Stage-3 claim is

```text
C_setup
  = sum_{i=0}^{N-1} sum_{y=0}^{d0-1}
      S_full[i, y] * W_R[i] * A_alpha[y]
  = sum_{i=0}^{R-1} sum_{y=0}^{d0-1}
      S[i, y] * W[i] * alpha^y.
```

Let `S_full_tilde`, `W_R_tilde`, and `A_alpha_tilde` be the multilinear
extensions of the three tables on their power-of-two domains. Stage 3 runs
sumcheck on

```text
F(X, Y)
  = S_full_tilde(X, Y) * W_R_tilde(X) * A_alpha_tilde(Y).
```

`X` has `log2(N)` variables and `Y` has `log2(d0)` variables. At the terminal
point `(rho_i, rho_y)`, the verifier checks

```text
final_claim
  == s_rho * W_R_tilde(rho_i) * A_alpha_tilde(rho_y),
```

where the prover supplies the deferred claim

```text
s_rho = S_full_tilde(rho_i, rho_y).
```

The verifier evaluates `W_R_tilde(rho_i)` locally from the prepared tensor plan
and evaluates

```text
A_alpha_tilde(rho_y)
  = product_j ((1 - rho_y[j]) + rho_y[j] * alpha^(2^j)).
```



### Why the Off-Hypercube Change Is Sound

`S_zero_tilde` and `S_full_tilde` normally differ at a non-Boolean point:

```text
S_zero_tilde(rho_i, rho_y) != S_full_tilde(rho_i, rho_y).
```

Consequently, the current and target Stage-3 round polynomials and terminal
claims also differ. They nevertheless start from the same Boolean sum because
`W_R` is zero on the tail.

This is valid sumcheck usage. Soundness does not require two multilinear
polynomials with the same Boolean sum to agree off the cube. It requires:

1. prover and verifier to use the same target polynomial;
2. the sumcheck terminal relation to use `S_full_tilde`;
3. the claimed `s_rho` to be authenticated against the commitment to exactly
  `S_full`; and
4. every transcript challenge to be sampled after the values it binds have
  been absorbed.

It is incorrect to generate a proof using `S_zero_tilde` and close it with an
opening of `S_full_tilde`.

### Deferred Full-Prefix Opening

Stage 3 does not independently prove `s_rho`. It reduces the setup-product
claim to an opening claim:

```text
(Com(S_full), (rho_i, rho_y), s_rho).
```

The recursive suffix places that claim in the setup-prefix polynomial group
alongside the independent next-witness group. The successor opening protocol
samples its batching challenges after both ordered claims are transcript-bound
and proves both against their corresponding commitments.

The soundness chain is:

```text
Stage-3 sumcheck
    -> conditional terminal equation using s_rho
    -> carried full-prefix opening claim
    -> successor grouped opening proof
    -> Com(S_full) authenticates s_rho.
```

If the successor opening is missing, references another prefix identity, uses a
different point, or omits `s_rho`, verification fails. Random batching must be
sampled after the ordered setup and witness claims are fixed so a malicious
prover cannot arrange cancellation between them.

## Compact Truncated Shifted Equality



### Equality Basis and Zero-Tail Table

For an `n`-variable little-endian Boolean domain of size `N = 2^n`, define

```text
chi_i(r)
  = product_{b=0}^{n-1}
      (r[b] if bit_b(i) == 1 else (1 - r[b])).
```

The simplest truncated shifted-equality weight is

```text
e_R(i; r) = chi_i(r) * 1[i < R].
```

Its MLE at a verifier point `z` is the paired prefix contraction

```text
E_R(r, z)
  = sum_{i=0}^{N-1} e_R(i; r) * chi_i(z)
  = sum_{i=0}^{R-1} chi_i(r) * chi_i(z).
```

The verifier must compute `E_R(r, z)` without constructing either equality
table. Two equivalent compact algorithms are specified below. The two-state
comparator is the canonical scalar algorithm; dyadic interval decomposition is
the useful connection to the general shifted tensor evaluator.

### Two-State Most-Significant-Bit Recurrence

For bit `b`, define the weight of assigning the shared Boolean index bit to
zero or one:

```text
q0[b] = (1 - r[b]) * (1 - z[b])
q1[b] = r[b] * z[b].
```

Process bits from most significant to least significant while comparing the
candidate index `i` with the public bound `R`. Maintain:

```text
equal = total paired-equality weight of processed prefixes equal to R's prefix
less  = total paired-equality weight of processed prefixes below R's prefix.
```

Initialize

```text
equal = 1
less  = 0.
```

For bit `b` from `n - 1` down to `0`, save `old_equal` and `old_less` and apply:

If `bit_b(R) == 0`:

```text
less  = old_less * (q0[b] + q1[b])
equal = old_equal * q0[b].
```

If `bit_b(R) == 1`:

```text
less  = old_less * (q0[b] + q1[b]) + old_equal * q0[b]
equal = old_equal * q1[b].
```

After the final bit,

```text
E_R(r, z) = less.
```

The final `equal` state represents the single excluded index `i == R`.

#### Recurrence invariant

After processing bits `n - 1` through `b`, `equal` is

```text
sum over assignments whose processed prefix equals R's processed prefix
    product of the corresponding q-bit weights,
```

and `less` is the analogous sum for assignments whose processed prefix is
strictly smaller. Once a prefix is smaller, either value of every remaining
bit stays smaller, which explains the factor `q0 + q1`. When `R[b] == 1`, an
equal prefix followed by candidate bit zero becomes smaller, which explains
the added `old_equal * q0` term. No other transition can enter `less`.

The recurrence uses `O(log N)` field operations and constant auxiliary state.
It supports every `R in [1, N]`. The `R == N` case is handled as the untruncated
full-domain paired equality

```text
product_b (q0[b] + q1[b]);
```

because `N` itself needs `n + 1` bits and is not an in-domain excluded index.

#### Worked example: `R = 5`, `N = 8`

The three-bit bound is `R = 101₂`. Abbreviate `qv[b]` as the paired-equality
weight for candidate bit `v` at bit `b`. The recurrence processes bits `2`,
`1`, then `0`:

```text
start:
  less  = 0
  equal = 1

bit 2, R[2] = 1:
  less  = q0[2]
  equal = q1[2]

bit 1, R[1] = 0:
  less  = q0[2] * (q0[1] + q1[1])
  equal = q1[2] * q0[1]

bit 0, R[0] = 1:
  less  = q0[2] * (q0[1] + q1[1]) * (q0[0] + q1[0])
        + q1[2] * q0[1] * q0[0]
  equal = q1[2] * q0[1] * q1[0].
```

The first term in `less` covers every index whose high bit is zero, namely
`0, 1, 2, 3`. The second term covers `100₂ = 4`. The final `equal` term is
`101₂ = 5` and is excluded. Hence `less` is exactly

```text
sum_{i=0}^{4} chi_i(r) * chi_i(z).
```



### Dyadic Prefix Decomposition

The integer interval `[0, R)` is the disjoint union of at most `popcount(R)`
aligned power-of-two intervals. For example,

```text
R = 13
[0, 13) = [0, 8) union [8, 12) union [12, 13).
```

For an aligned block

```text
B = [h * 2^k, (h + 1) * 2^k),
```

the low `k` bits vary freely and the high bits are fixed to `h`. Its paired
equality contraction factors as

```text
sum_{i in B} chi_i(r) * chi_i(z)
  = chi_h(r[k..]) * chi_h(z[k..])
      * product_{b=0}^{k-1} (q0[b] + q1[b]).
```

Summing the at-most-`log2(N)` block evaluations gives `E_R(r, z)`. Prefix and
suffix products of `(q0 + q1)` make the total scalar cost `O(log N)` rather
than `O(log^2 N)`.

The dyadic view is also how an arbitrary-length affine stream is reduced to
power-of-two carry recurrences: decompose its exact live length into disjoint
power-of-two blocks and shift the address seed for each block. No block covers
an index outside the live range.

### Shifted Affine Paired Equality

The setup-weight evaluator needs more than the diagonal prefix above. One
affine stream has the form

```text
T(z, x)
  = c * sum_{j=0}^{m-1} a[j]
      * chi_{L0 + sL * j}(z)
      * chi_{R0 + sR * j}(x).
```

`m` need not be a power of two. `L0` and `R0` may be nonzero, and multiplying
the coordinate by `sL` or `sR` may create binary carries. Padding `m` to its
next power of two and evaluating the extra coordinates would be incorrect.

The logarithmic carry recurrence applies only when the stream weights are
implicit unit weights, or when `a[j]` has its own compact factorization whose
state is carried with the equality addresses. An arbitrary dense vector
`a[0..m)` cannot be evaluated in `O(log m)` by this recurrence alone: the
verifier must either read/enumerate the dense weights, paying their explicit
cost, or use a separately specified commitment/factorization for them. This
cutover uses the unit-axis case inside `EqPairTensorFamily`; it does not add a
general dense affine-stream evaluator.

The compact evaluator instead:

1. Computes the exact live length after clipping addresses to the ambient left
  and right equality domains.
2. Decomposes that live length into disjoint power-of-two blocks using its
  binary expansion.
3. Gives each block shifted seeds
  `L0 + sL * block_start` and `R0 + sR * block_start`.
4. Processes the block's coordinate bits from least significant to most
  significant.
5. Tracks `(left_carry, right_carry, weight)` states.
6. At equality bit `b`, multiplies by `z[b]` or `1-z[b]` according to the low
  bit of the left carry, and by `x[b]` or `1-x[b]` according to the low bit of
   the right carry.
7. Shifts both carries right and merges states with the same carry pair.

For `m = 13`, the evaluator processes blocks of lengths `8`, `4`, and `1`.
Coordinates `13`, `14`, and `15` are never introduced. The ambient equality
domains remain power-of-two; only the stream support is non-power-of-two.

### General Tensor Family

A setup role is represented as a sum of families

```text
T(z, x)
  = c * sum_{0 <= j_t < m_t}
      product_t a_t[j_t]
      * chi_{L(j)}(z)
      * chi_{R(j)}(x),

L(j) = L0 + sum_t sL_t * j_t,
R(j) = R0 + sum_t sR_t * j_t.
```

The axes encode role digits, subcolumns, blocks, rows, fold digits, and
optionally affine units/chunks. Dense axes carry row equality weights,
`-G_fold`, or projection powers; unit axes carry coefficient one.

The evaluator chooses among three exact paths:

- multiple power-of-two unit axes use the simultaneous binary carry
recurrence;
- one large unit axis, of arbitrary length, uses the affine-stream block
decomposition; and
- remaining normally-small or dense axes are enumerated over their exact
lengths to seed the recurrence.

The second path is logarithmic in the live length because every stream weight
is one. Dense axes are intentionally outside that claim; their cost is
proportional to the number of explicit weights unless another compact
representation is specified.

The left setup addresses generated by these families are exactly the natural
A/B/D footprints. Their maximum determines `R`; no family emits a left address
in `[R, N)`. Consequently the same tensor description is already an implicit
zero-padded representation of `W_R` on the `N`-point ambient domain.

### Mixed Ring-Dimension Projection

For a role dimension `d_role`, define

```text
q = d_role / d0
beta = alpha^d0.
```

The low `log2(q)` setup-index variables select the projected setup subring.
Their contribution factors as

```text
P_beta(z_low)
  = sum_{u=0}^{q-1} beta^u * chi_u(z_low)
  = product_b ((1 - z_low[b]) + z_low[b] * beta^(2^b)).
```

If every relation address and stride in the batch is divisible by `q`, the
relation low variables factor identically and the paired tensor runs on the
high variables. Otherwise the evaluator appends an explicit `q`-lane dense
axis with weights `(1, beta, ..., beta^(q-1))` on the relation address. In both
cases the setup-side zero tail is unchanged: projection adds weights to live
addresses but does not create padded setup addresses.

### Complexity

For the scalar truncated diagonal equality, verification costs `O(log N)`
field operations and `O(1)` auxiliary state.

For the full setup weight, cost is

```text
O(number of tensor families
  * equality bit depth
  * bounded carry-state work
  + residual dense-axis work).
```

Thus the evaluator is logarithmic in the padded setup domain only for fixed
tensor descriptions whose long axes are unit-weight or otherwise compactly
factored, and whose carry state remains bounded. It is not unconditionally
`O(log N)` for arbitrary affine streams: group count, family count, dense
fold/row axes, explicit dense weights, and carry-state multiplicity remain
explicit work. The required invariant is that it never performs an `O(N)`
padded setup-index scan or allocation for Akita's structured setup weights.

## Implementation Design



### 1. Prefix Content Identity

`SetupPrefixSlotId` currently identifies a natural length and a padded
commitment geometry whose content is defined as natural setup coefficients
followed by zeros. The target content must be domain-separated from every
existing artifact.

Add one canonical content-semantics field, for example:

```rust
pub enum SetupPrefixContent {
    FullPublicSetupV1,
}
```

or replace the old wire/cache version with a new version whose only valid
meaning is `FullPublicSetupV1`. The field/tag must participate in:

- descriptor bytes, ordering, hashing, serialization, and validation;
- setup-prefix registry keys;
- setup cache and prefix-registry cache validation;
- generated schedule/catalog identity; and
- transcript binding wherever the slot id is absorbed.

Do not retain a thin compatibility constructor or infer content semantics from
lengths. This repository makes no backward-compatibility guarantee; stale
zero-padded artifacts should reject and regenerate.

`natural_len` remains part of the identity because it defines `R` and therefore
the support of `W_R`, even if multiple natural lengths share the same `n_prefix`
and the same full setup coefficients.

### 2. Setup Availability and Planning

Preprocessing currently requires only `natural_len` shared setup coefficients.
It must instead require at least `n_prefix` actual coefficients. Planner and
setup-capacity calculations must account for this at every recursive edge.

`select_setup_prefix_slot` continues to validate:

```text
natural_len == active_setup_projection_geometry.natural_field_len()
n_prefix == next_power_of_two(natural_len).
```

It additionally validates that a prover-side shared setup has at least
`n_prefix` fields. `setup_eval_len` remains `n_prefix / source_ring_dimension`.

### 3. Prefix Commitment Construction

Change the canonical setup-prefix commitment constructor in
`crates/akita-prover/src/api/setup_prefix.rs` to:

1. require `expanded.shared_matrix().num_field_elements() >= n_prefix`;
2. extract exactly `fields[..n_prefix]`;
3. reshape those fields into the commitment ring dimension without injecting
  zeros; and
4. commit the resulting full source.

The existing `extract_setup_prefix_ring_elems` starts from zero rings and copies
only `fields[..natural_len]`; that behavior must be replaced. Tests asserting
zeros after `natural_len` must be replaced by tests asserting equality with
`fields[natural_len..n_prefix]`.

### 4. Recursive Opening Source

Change `setup_prefix_field_evals` in
`crates/akita-prover/src/backend/recursive/setup_prefix_source.rs` to return
`fields[..n_prefix]`. It must check full-prefix source availability and content
identity. `setup_prefix_rings`, evaluation/folding, and decomposition then
operate on the same polynomial that was committed.

This is security-critical: committing the full prefix but reconstructing a
zero-padded source during the successor opening would make honest proofs fail
and would split the source of truth for commitment versus opening.

### 5. Stage-3 Prover Table

Refactor `RectangularSetupProductTerm` so the following lengths are not
conflated:

```text
active_weight_rows = R
source_rows        = N
row_capacity       = N.
```

The setup-index factor has length `N` and is zero on `[R, N)`. The setup source
contains `N * d0` actual coefficients. The coefficient-pass optimization may
still accumulate only the first `R` rows because the tail factor is zero, but
the later index table must contain the coefficient-folded setup value for every
row in `[0, N)`. It must not call `resize(..., zero)` after materializing only
`R` setup rows.

The transition after coefficient rounds therefore constructs

```text
index_table[i] = S_full_tilde(i, rho_y)  for every i in [0, N).
```

Subsequent setup-index rounds fold this full table against the zero-tail weight
table. This produces the target off-hypercube polynomial and final
`setup_prefix_eval`.

### 6. Stage-3 Verifier

The recursive/offloaded path interprets `proof.setup_prefix_eval` as

```text
S_full_tilde(rho_setup_idx, rho_y).
```

It evaluates `W_R_tilde(rho_setup_idx)` through the existing canonical
`SetupContributionPlan` and checks the unchanged multiplicative terminal
shape.

Any local-scan fallback must also evaluate the full `N`-row source. It may not
scan only `R` rows, because that would evaluate `S_zero_tilde`. The fallback
must either:

- read exactly the actual `N` rows and use the `N`-point equality domain; or
- reject when the full source is not resident.

`required` remains the setup-weight support length. A distinct name such as
`source_rows` or `setup_index_len` must govern setup-source evaluation.

### 7. Carried Claim and Successor Batch

The existing fold output already carries

```text
(setup_prefix_point, setup_prefix_eval)
```

and the successor suffix constructs a setup-prefix polynomial group before the
witness group. Preserve that order. Validate that:

- the group commitment comes from the selected full-prefix slot;
- the opening source passed to the prover is the full-prefix source;
- the point equals the complete Stage-3 challenge vector in canonical order;
- the scalar equals the Stage-3 terminal `setup_prefix_eval`; and
- the verifier constructs the identical ordered claim group.

No new proof scalar is required.

### 8. Compact Evaluator

The existing `EqPairTensorFamily` and
`eval_boolean_pair_tensor_families` representation already supports exact
non-power-of-two affine lengths. Implementation work should be limited to any
missing explicit prefix-bound primitive needed by tests or future callers.

If a standalone truncated diagonal/shifted equality evaluator is added, it
must be the canonical evaluator for

```text
sum_{i < R} chi_i(r) * chi_i(z)
```

and should implement the two-state recurrence above without allocating a
dyadic block vector or equality table. Do not add a wrapper that forwards to a
dense equality-prefix materialization.

For the Stage-3 setup weight itself, continue to call
`SetupContributionPlan::evaluate_setup_index_weight_mle`; do not multiply a
second range-selector polynomial into its result. Pointwise multiplication of
MLEs would change the off-hypercube factor and unnecessarily raise the
sumcheck degree.

### 9. Serialization and Cache Cutover

The content change invalidates:

- serialized setup-prefix prover registries;
- verifier prefix registries;
- setup cache artifacts containing prefix slots;
- generated catalog identities that include prefix metadata; and
- any fixture containing a serialized prefix slot or commitment.

Use explicit version/content-tag rejection. Do not silently reinterpret an old
commitment as a full-prefix commitment merely because its lengths match.

## Evaluation



### Acceptance Criteria

- [x] For non-power-of-two `R`, preprocessing commits exactly
  ```
  `shared_setup[..N * d0]` and the recursive opening source reconstructs
  byte-for-byte the same polynomial.
  ```
- [x] Setup generation rejects a requested full prefix when fewer than
  ```
  `N * d0` actual shared setup coefficients are available.
  ```
- [x] The dense Boolean Stage-3 claim with `S_full` and zero-tail `W_R` equals
  ```
  the existing natural-prefix setup contribution.
  ```
- [x] The Stage-3 prover and verifier agree on the full-prefix terminal
  ```
  relation for random non-Boolean challenges.
  ```
- [x] `setup_prefix_eval` equals a dense oracle evaluation of `S_full`, including
  ```
  fixtures where every coefficient in `[natural_len, n_prefix)` is nonzero.
  ```
- [x] The successor grouped opening accepts the honest carried full-prefix
  ```
  claim and rejects changes to its commitment, point, scalar, slot identity,
  or group order.
  ```
- [x] The compact setup-weight evaluator equals the MLE of a dense length-`N`
  ```
  weight vector whose tail is explicitly zero.
  ```
- [x] A scalar truncated shifted-equality fixture covers every `R` for small
  ```
  domains and matches dense
  `sum_{i<R} chi_i(r) * chi_i(z)` evaluations.
  ```
- [x] The spec states that arbitrary-length unit affine streams are logarithmic
  ```
  through dyadic block decomposition.
  ```
- [x] The spec states that arbitrary dense affine weights are not logarithmic
  ```
  unless represented by a separate compact factorization; otherwise they cost
  work proportional to their explicit length.
  ```
- [x] Shifted/strided tests exercise carries on both equality addresses and
  ```
  confirm that no rounded-up stream coordinates contribute.
  ```
- [x] Mixed-ring, multigroup, and multichunk Stage-3 fixtures retain dense versus
  ```
  compact setup-weight equality under the full-prefix regime.
  ```
- [x] Old zero-padded prefix registries and caches fail validation instead of
  ```
  being silently reused.
  ```
- [x] Transcript logging shows identical prover/verifier events and binds the
  ```
  full-prefix content identity before relevant batching challenges.
  ```
- [x] No verifier path allocates `O(N)` setup-index weights or equality values.



### Testing Strategy

Add focused tests in:

- `crates/akita-prover/src/api/setup_prefix.rs` for actual-tail commitment
extraction and insufficient setup capacity;
- `crates/akita-prover/src/protocol/sumcheck/akita_stage3/product_table.rs` for
full-tail sumcheck round parity against a dense product oracle;
- `crates/akita-types/src/setup_contribution/tests/` for dense zero-tail weight
MLE parity;
- `crates/akita-algebra/src/offset_eq/tests/` for the scalar truncated paired
equality recurrence and arbitrary-length shifted streams;
- `crates/akita-pcs/tests/recursive_setup_e2e.rs` for the carried full-prefix
opening and tamper cases; and
- setup serialization/cache tests for explicit old-content rejection.

The most important adversarial fixture uses two setup vectors that agree on
`[0, R)` and differ everywhere on `[R, N)`. It must demonstrate:

1. their Boolean input claims against `W_R` are equal;
2. their Stage-3 round polynomials or terminal setup evaluations generally
  differ;
3. each proof verifies only against its own full-prefix commitment; and
4. swapping the carried setup evaluation or commitment fails.

Run the repository documentation guardrails because this spec changes a
verifier-reachable proof contract. Implementation work must also run the
path-specific recursive setup tests and the exact Clippy feature graphs from
`AGENTS.md`.

### Performance

Expected verifier behavior:

- setup-weight evaluation cost is unchanged asymptotically;
- no local setup scan occurs on the carried-opening path;
- proof round count and proof scalar count are unchanged; and
- prefix commitment/opening geometry remains `n_prefix`.

Expected prover/setup behavior:

- preprocessing needs actual setup capacity through `n_prefix`, which is less
than twice `natural_len`;
- commitment work is unchanged in domain size because the current artifact
already commits an `n_prefix`-sized polynomial;
- Stage-3 coefficient accumulation may retain the `R`-row optimization;
- Stage-3 index folding must materialize/fold `N` actual row evaluations rather
than `R` evaluations plus zeros, increasing that pass by less than 2x; and
- recursive opening already operates over the `n_prefix` commitment domain, so
its shape and proof size do not increase.

Benchmark the existing recursive multi-group profile with a deliberately
non-power-of-two natural footprint. Report separately:

- setup-prefix preprocessing;
- Stage-3 coefficient pass;
- Stage-3 setup-index pass;
- verifier compact setup-weight evaluation; and
- successor grouped opening prove/verify.



## Alternatives Considered



### Keep Zero-Padded Prefix Commitments

This preserves current semantics but creates a distinct committed polynomial
for every natural length, even when the actual setup already provides a common
power-of-two prefix. It also makes the recursive opening source differ from the
literal setup prefix named by `n_prefix`.

### Commit Full Prefix but Prove a Truncated Setup Opening Separately

One can define

```text
e_R(i; r) = chi_i(r) * 1[i < R]
```

and prove

```text
sum_i S_full[i] * e_R(i; r)
```

with another product sumcheck whose verifier evaluates `E_R` compactly. This
is mathematically valid but redundant in Stage 3 because `W_R` already contains
the exact natural support. It adds another claim/reduction boundary without
reducing the required full-prefix opening work.

### Multiply by a Range-Selector MLE at the Terminal Point

Rejected. In general,

```text
MLE(S * 1[i<R])(r) != S_full_tilde(r) * MLE(1[i<R])(r).
```

The product agrees on Boolean vertices but changes the off-hypercube
polynomial. Adding it to Stage 3 would raise the per-variable degree and would
duplicate the zero-tail semantics already owned by `W_R`.

### Materialize the Padded Weight Vector

Rejected. It gives the same mathematics but costs `O(N)` verifier memory and
time and discards the affine paired-equality structure already present in the
setup contribution plan.

## Documentation

When implemented:

- update `book/src/how/proving/sumcheck-stages.md` with the full-prefix Stage-3
relation and deferred opening semantics;
- update `book/src/how/verification.md` to state that the carried setup claim is
an opening of the actual full prefix;
- update or supersede zero-padding statements in `setup-prefix-ladder.md`,
`setup-offloading-planner.md`, `setup-layout-repack.md`, and
`distributed-setup-offloading.md`;
- update `AGENTS.md` only if the verifier-contract summary or required commands
change; and
- set this spec's `Book-chapter` and archive it after the durable explanation is
folded into the book.



## Execution

Suggested implementation order:

1. Add and bind the full-prefix content identity; make stale artifacts reject.
2. Increase setup-capacity planning to guarantee actual `n_prefix` coverage.
3. Cut prefix commitment extraction and recursive opening reconstruction over
  together so they always name the same polynomial.
4. Refactor Stage-3 product-table lengths and switch the index table to actual
  full-prefix rows.
5. Switch verifier fallback semantics and document `setup_prefix_eval` as a
  full-prefix claim.
6. Add dense mathematical oracles and compact truncated-equality tests.
7. Add successor grouped-opening tamper tests.
8. Run recursive profiles, full documentation guardrails, and CI-fidelity
  verifier/prover feature graphs.

Risks to resolve before implementation:

- setup generation may currently allocate only `natural_len` public setup
coefficients for some schedules;
- existing slot IDs and generated catalogs do not domain-separate prefix
content semantics;
- `RectangularSetupProductTerm::required_rows` currently controls both source
materialization and active weight support;
- local verifier fallback currently reads only `required` rows; and
- cached setup-prefix registries may survive across binaries unless their
format/identity is explicitly invalidated.



## References

- `[setup-product-sumcheck.md](setup-product-sumcheck.md)`
- `[setup-prefix-ladder.md](setup-prefix-ladder.md)`
- `[setup-offloading-planner.md](setup-offloading-planner.md)`
- `[setup-layout-repack.md](setup-layout-repack.md)`
- `[distributed-setup-offloading.md](distributed-setup-offloading.md)`
- `[archive/2026-Q3/group-local-opening-points.md](archive/2026-Q3/group-local-opening-points.md)`
- `crates/akita-types/src/setup_contribution/plan/setup_index_weight.rs`
- `crates/akita-algebra/src/offset_eq/tensor_pair/evaluate.rs`
- `crates/akita-prover/src/api/setup_prefix.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage3/product_table.rs`
- `crates/akita-prover/src/backend/recursive/setup_prefix_source.rs`
- `crates/akita-verifier/src/stages/stage3.rs`
