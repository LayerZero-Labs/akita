# Spec: Akita ZK Commitment Hiding


| Field     | Value                  |
| --------- | ---------------------- |
| Author(s) | Amirhossein Khajehpour |
| Created   | 2026-05-06             |
| Status    | proposed tightening    |
| PR        | #67                    |


## Summary

This spec tightens the compile-time `zk` commitment-hiding design for Akita's Ajtai
commitment path. Transparent builds keep the existing deterministic commitment
layout, while `--features zk` builds reserve extra outer B-matrix columns,
sample fresh balanced digit-source masks for every commitment group, append
those digits to the committed witness relation, and replay the same enlarged
relation in the verifier. The immediate problem solved is commitment
re-randomization:
two commitments to the same polynomial under the same setup should differ in
`zk` builds, while still opening and verifying through the existing Akita fold,
ring-switch, and batched proof machinery.

## Intent

### Goal

Implement compile-time commitment hiding for Akita commitments by adding fresh
Leftover Hash Lemma (LHL) blinding columns to the outer Ajtai commitment and
threading the resulting witness segment through setup sizing, schedule planning,
prover hints, recursive witnesses, root-direct proofs, ring-switch prover logic,
and verifier replay.

The feature introduces or modifies these surfaces:

- `akita-types`: feature-gated `zk` helpers, proof/hint payloads that carry
blinding witness material, direct witness variants, recursive witness-size
formulas, and proof-size accounting.
- `akita-field`: `CanonicalField::modulus_bits()` so LHL entropy can be computed
from concrete field modulus sizes.
- `akita-config` and `akita-planner`: feature-aware setup stride, generated
schedule handling, planner fallback, and root witness sizing that include
blinding columns in `zk` builds.
- `akita-prover`: fresh mask sampling, commitment kernels, recursive hint cache
persistence, quadratic-equation assembly, ring-switch witness construction,
root-direct proof payloads, and recursive `w` commitment handling.
- `akita-verifier`: root-direct commitment recomputation and ring-switch
M-evaluation over the same enlarged witness layout.
- `akita-scheme` and `akita-pcs`: public feature plumbing, end-to-end tests,
examples, profiles, and CI coverage for both transparent and `zk` builds.

### Invariants

1. Transparent builds must preserve the existing public API behavior, proof
  shapes, deterministic commitments, generated schedule tables, and setup
   sizing. The `zk` feature must be opt-in through Cargo feature propagation in
   `akita-pcs`, `akita-scheme`, `akita-setup`, `akita-config`, `akita-prover`,
   `akita-verifier`, `akita-planner`, and `akita-types`.
2. `zk` commitments must sample fresh B-blinding material for each commitment
  group. The sampler in `crates/akita-prover/src/protocol/masking.rs` must use
   `OsRng` and sample the decomposed B-input digits directly from the balanced
   base-`2^log_basis` digit alphabet.
3. The blinding width must satisfy the 128-bit LHL statistical-distance target:
  for nonzero output ring length `kappa`, ring dimension `D`, field modulus bit
   width `field_bits`, and digit base `beta = log_basis`, the number of
   blinding digit-ring planes is:

   ```text
   ceil((kappa * D * field_bits + 2 * 128 - 2) / (D * beta)).
   ```

   This should replace the coarser full-ring formula that first sampled
   `kappa + 1` full ring elements and then decomposed them.
4. Setup sizing must reserve enough shared-matrix stride for ordinary outer
   input columns plus the ZK blinding columns.
   `accumulate_matrix_envelope_for_level` in
   `crates/akita-config/src/proof_optimized.rs` must include the digit-source
   blinding column count when compiled with `zk`.
5. Prover and verifier must agree on recursive witness width and segment order.
  In `zk` builds the recursive witness contains, in order:
   `z_pre`, `w_hat`, `t_hat`, `B-blinding`, and `r_hat`.
6. The ring-switch M-table must include the B-blinding contribution with the
  same offsets on both sides. Prover materialization in
   `crates/akita-prover/src/protocol/ring_switch.rs` and verifier evaluation in
   `crates/akita-verifier/src/protocol/ring_switch.rs` must use the same
   `blinding_segment_len`, `blinding_segment_offset`, group-local B column
   indexing, and `group_poly_counts`.
7. `AkitaCommitmentHint` must keep blinding digits as prover witness material.
  Folded proof paths should not reveal those digits directly; they are proven
   through the same commitment relation as the ordinary `t_hat` segment.
8. Root-direct ZK proofs are a special compatibility path. Because the root
  witness is revealed directly, `zk` root-direct proofs must also reveal the
   commitment blinding digits needed for verifier recommitment. This is not a
   full ZK opening path; it is only a verifier-consistent direct proof shortcut.
9. Generated transparent schedule tables must not be reused blindly in `zk`
  builds. `zk` witness sizes include extra blinding columns, so fp128 configs
   return no generated plan under `zk` and use planner fallback until generated
   ZK tables are audited.
10. Serialization shape data must distinguish packed digit tails from direct
  field-element witnesses. `DirectWitnessProof`, `DirectWitnessShape`, and
    `AkitaBatchedRootProof::Direct` must serialize/deserialize deterministically
    and validate both transparent and `zk` payloads.
11. The implementation must not claim full proof zero-knowledge. This branch
  hides Ajtai commitments and keeps prover/verifier consistency for proofs
    containing hidden commitments. It does not mask all sumcheck messages or
    replace the terminal clear witness with a sigma protocol.

Existing and new tests that protect these invariants include
`crates/akita-pcs/tests/zk.rs`, the transparent end-to-end
tests in `crates/akita-pcs/tests/akita_e2e.rs`, multipoint and aggregated batch
tests in `crates/akita-pcs/tests/multipoint_batched_e2e.rs` and
`crates/akita-pcs/tests/batched_aggregated_e2e.rs`, scheme-level tests in
`crates/akita-scheme/src/tests.rs`, and CI's transparent, planner, and
all-features test jobs.

### Non-Goals

- This branch does not implement the full Jolt + Akita zero-knowledge protocol:
sumcheck pad commitments, tail Gaussian sigma proofs, LNP22 residual-quadratic
handling, and end-to-end simulator arguments remain out of scope.
- This branch does not make root-direct proofs zero-knowledge. Root-direct
proofs reveal field witness values, and in `zk` builds they additionally
reveal blinding digits so the verifier can recompute the commitment.
- This branch does not introduce runtime switching between transparent and ZK
commitments. The selected behavior is compile-time, through the `zk` feature.
- This branch does not publish a stable public ZK API or compatibility promise.
The repository explicitly allows breaking changes.
- This branch does not generate audited ZK schedule tables. Planner fallback is
acceptable until the schedule table generator is updated and validated for the
enlarged ZK witness formulas.
- This branch does not change Fiat-Shamir labels, transcript ordering, or the
algebraic statement of Akita's fold and ring-switch protocols except for the
extra blinding columns in the committed witness relation.

## Evaluation

### Acceptance Criteria

- The workspace exposes a top-level `zk` feature that propagates to all
crates that need ZK-aware types, setup sizing, prover logic, verifier
replay, planner sizing, or config policy.
- `CanonicalField::modulus_bits()` is implemented for Akita's concrete base
fields and is used by `akita_types::zk` to compute LHL mask width.
- `akita_types::zk` defines the 128-bit statistical security target and
computes blinding digit-plane counts from output ring length, ring dimension,
field bit width, and `log_basis`.
- Root and recursive commitment paths append sampled blinding digit planes
to the outer B input under `cfg(feature = "zk")`.
- `AkitaCommitmentHint` and `RecursiveCommitmentHintCache` preserve
blinding digits across typed and D-erased prover paths.
- Recursive witness length formulas in `akita-types`, proof-size formulas in
`akita-types/src/layout/proof_size.rs`, and planner formulas in
`akita-planner/src/schedule_params.rs` include blinding columns under
`zk`.
- Setup matrix sizing in `akita-config` accounts for ZK outer width.
- fp128 generated schedules are disabled under `zk` so transparent schedule
rows are not used for enlarged ZK witness sizes.
- Prover ring-switch witness construction and verifier M-evaluation include
the blinding segment with matching group-local B-column offsets.
- Root-direct proofs verify in `zk` builds by carrying the blinding payload
needed to recompute the public commitment.
- ZK dense end-to-end tests prove that same-polynomial commitments
re-randomize and still commit, prove, serialize, deserialize, and verify
for D=32, D=64, and D=128 fp128 dense configs.
- CI runs transparent tests, transparent planner tests, all-features tests,
clippy with all features, clippy without default features, docs with all
features, and planner validation.

### Testing Strategy

Existing transparent checks must continue passing:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

CI additionally runs:

```bash
cargo nextest run
cargo nextest run --all-features
```

Focused ZK checks:

```bash
cargo test -p akita-pcs --features zk --test zk
cargo test -p akita-pcs --all-features --test zk
cargo test -p akita-pcs --features zk --test zk zk_dense_d32_hides_folded_v_and_verifies -- --exact
cargo test -p akita-pcs --features zk --test zk zk_dense_d64_hides_folded_v_and_verifies -- --exact
cargo test -p akita-pcs --features zk --test zk zk_dense_d128_hides_folded_v_and_verifies -- --exact
```

Regression coverage should also include representative transparent direct and
folded proof cases in `crates/akita-pcs/tests/akita_e2e.rs`, multi-group batched
cases in `crates/akita-pcs/tests/batched_aggregated_e2e.rs`, multipoint cases in
`crates/akita-pcs/tests/multipoint_batched_e2e.rs`, and scheme API tests in
`crates/akita-scheme/src/tests.rs`.

New invariant tests that would strengthen this feature:

- Unit tests for the blinding digit-plane count across small and large `D`
values, including `output_ring_len = 0`.
- A negative verifier test that corrupts one root-direct blinding digit and
expects `InvalidProof`.
- A folded-root negative test that corrupts the prover hint blinding segment
before proving and expects the ring-switch relation to fail.
- A schedule-planner test that compares transparent and `zk` planned witness
widths for the same shape and asserts the ZK width increases by exactly
`num_commitment_groups * blinding_digit_plane_count`.

### Performance

Transparent builds should have no intentional proof-size, setup-size, or runtime
regression. Any transparent changes should be limited to refactoring and
compile-time conditional code that disappears without `zk`.

`zk` builds intentionally add work and size:

- commitment time increases by sampling
`blinding_digit_plane_count(n_B, D, log_basis)` fresh digit-ring planes per
commitment group;
- B-matrix multiplication uses the same number of extra columns;
- setup stride may increase to fit those extra B columns;
- recursive witness length and stage-1/stage-2 sumcheck dimensions increase by
the same blinding segment;
- root-direct proof size increases by the serialized blinding digits required
for recommitment.

For fp128 production dimensions, direct digit-source blinding is materially
tighter than the full-ring source. For example, with `n_B = kappa = 1`,
`D = 64`, and `log_basis = 5`, the previous full-ring source used
`2 * 26 = 52` digit-ring columns, while the digit-source formula uses
`ceil((64 * 128 + 254) / (64 * 5)) = 27` columns. Before adding generated ZK
schedule tables, compare planner outputs with:

```bash
cargo run -p akita-planner --bin akita-planner -- --validate
```

and profile representative proving cases with:

```bash
AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release --example profile
```

## Design

### Architecture

The final design is deliberately compile-time. The `zk` Cargo feature controls
whether blinding columns exist in public type shapes and witness formulas. This
keeps transparent builds simple and avoids carrying a runtime mode enum through
hot paths that are already monomorphized by field, ring dimension, and config.

At commitment time, the prover computes the ordinary message-dependent outer
input:

```text
t_hat = G^{-1}(A * G^{-1}(message blocks))
```

In transparent builds, the public commitment remains:

```text
u = B_msg * t_hat
```

In `zk` builds, the prover samples fresh balanced digit planes directly and
appends them to the B input:

```text
r <- (A_b^D)^s
u = B_msg * t_hat + B_blind * r
```

Implementation references:

- `crates/akita-types/src/zk.rs` defines the digit-source blinding width and the
128-bit LHL slack calculation.
- `crates/akita-prover/src/protocol/masking.rs` samples fresh balanced digit
planes into `FlatDigitBlocks`.
- `crates/akita-prover/src/api/commitment.rs` appends those digits to
`t_hat_flat` before the outer B matrix multiplication in `commit_with_params`.
- `crates/akita-prover/src/protocol/ring_switch.rs` does the same for recursive
`commit_w` commitments and passes the blinding segment into recursive witness
construction.
- `crates/akita-types/src/proof/mod.rs` extends `AkitaCommitmentHint` with
feature-gated `b_blinding_digits` and extends direct root proofs with
feature-gated blinding payloads.
- `crates/akita-prover/src/backend/recursive_hint.rs` preserves blinding digits
across D-erased recursive commitment hints.

The recursive witness includes the blinding digits because later sumchecks must
prove that the public commitment's B-row contribution matches the committed
witness. In the witness layout the witness builder in `build_w_coeffs`
emits `z_pre` first, then the `{w_hat, t_hat, blinding}` group, so the blinding
planes follow `t_hat`. The verifier mirrors this layout in
`RingSwitchDeferredRowEval::eval_at_point`.

Schedule, setup, and proof-size accounting all consume the same formula:

```text
W(lp; K, G, P) =
    K * 2^r * delta_open
  + K * 2^r * n_A * delta_open
  + G * blinding_digit_plane_count(n_B, D, log_basis)
  + P * 2^m * delta_commit * delta_fold
  + (n_D + n_B * G + P + 1 + n_A) * delta_R
```

where `K` is total claims, `G` is commitment groups, and `P` is distinct opening
points. This formula appears in runtime schedule sizing
(`crates/akita-types/src/schedule.rs`), proof-size helpers
(`crates/akita-types/src/layout/proof_size.rs`), and planner root sizing
(`crates/akita-planner/src/schedule_params.rs`).

Root-direct proofs need special handling. Since they reveal direct field
witnesses rather than folding through ring-switch, the verifier cannot infer the
hidden B-mask from a private recursive witness. In `zk` builds,
`prove_root_direct` serializes the blinding digits from the commitment hints,
and `verify_root_direct_commitments_with_params` appends them when recomputing
each direct commitment. This keeps root-direct verification correct but does not
provide zero-knowledge for direct openings.

### Hiding Proof

For one wire-visible outer Ajtai commitment in a `zk` build, the prover
computes:

```text
u = B_msg * t_hat + B_blind * r
```

where:

- `B_msg` and `B_blind` are public uniformly sampled Ajtai columns over `R_q`,
produced from the shared setup matrix and sized by `LevelParams::b_key`;
- `t_hat` is the decomposed message-dependent opening witness produced by
`AkitaPolyOps::commit_inner_witness`;
- `r` is sampled freshly in `sample_b_blinding_digits` as decomposed
base-`2^log_basis` B-input digit-ring planes;
- `u` is the public `RingCommitment` in `R_q^kappa`, where
`kappa = params.b_key.row_len()`.

The claim is that, for every fixed message witness `t_hat`, the joint
distribution consisting of the public setup and `u` is negligibly statistically
close to the same public setup together with an independent uniform element of
`R_q^kappa`.

The proof has two independent parts:

1. The family `r |-> B_blind * r` is two-universal on the sampled digit-source
  domain.
2. The number of sampled digit-ring planes gives enough min-entropy for the
  Leftover Hash Lemma.

SIS is not used for this hiding proof. SIS remains the separate binding
assumption for a fixed public matrix.

#### Setting and Assumptions

Let:

```text
R_q = F_q[X] / (X^D + 1)
```

where `q` is prime and `D` is a power of two. The proof is conditional on the
Lyubashevsky-Seiler short-invertibility invariant for the concrete parameter
set. Specifically, for the selected prime and ring dimension, assume there is a
power-of-two factorization parameter `k <= D` such that:

```text
q = 2k + 1 mod 4k.
```

Then LS18 Corollary 1.2 gives:

```text
0 < ||c||_inf < q^(1/k) / sqrt(k)  =>  c is a unit in R_q.
```

The implementation objects are:

- `B_blind` is the suffix of the public B matrix addressed after the ordinary
message columns. Prover and verifier compute the same local column offset in
`crates/akita-prover/src/protocol/ring_switch.rs` and
`crates/akita-verifier/src/protocol/ring_switch.rs`.
- `beta = log_basis`.
- `b = 2^beta`.
- `s` is the number of sampled blinding digit-ring planes.

The balanced digit alphabet is:

```text
A_b = { -b/2, ..., b/2 - 1 }.
```

The digit-source blinding distribution samples:

```text
r = (r_1, ..., r_s) in (A_b^D)^s.
```

Equivalently, each `r_j` is a ring-shaped digit plane whose `D` coefficients
are sampled independently and uniformly from `A_b`. Hence:

```text
H_min(r) = s * D * beta.
```

Assumption A2: short digit differences are units. For any two decompositions,
every nonzero digit-ring difference is short enough to satisfy the LS18
coefficientwise bound. With base `b = 2^log_basis`, each coefficient of a digit
difference lies in `[-(b - 1), b - 1]`, so it is enough to require:

```text
b - 1 < q^(1/k) / sqrt(k).
```

The two-universality argument below uses A2 in exactly one place: it must turn
one nonzero digit-ring coordinate into a unit so that conditioning on the other
matrix entries leaves exactly one colliding value for the remaining entry.

For the default fp128 prime:

```text
q = 2^128 - 2^32 + 22537 = 9 mod 16,
```

so LS18 applies with `k = 4` for `D in {32, 64, 128}`. The implementation uses
`log_basis <= 6`, hence `b - 1 <= 63`, while `q^(1/4) / 2` is about `2^31`.
The supported fp128 presets are therefore comfortably inside the LS18
short-invertibility range.

If future parameter sets use primes where this condition is not available, the
two-universality argument must be replaced by a rank/ideal-size bound or by a
different blinding matrix family.

#### Hash Family

Define:

```text
H = { h_B : (A_b^D)^s -> R_q^kappa }
h_B(r) = B * r
```

where the seed `B` is sampled uniformly from `R_q^{kappa x s}`. This is the
LHL hash family. The output `B * r` acts as the random pad for the commitment,
and the full commitment adds the fixed offset `B_msg * t_hat`.

#### Two-Universality

For all `r != r'`, prove:

```text
Pr_B[h_B(r) = h_B(r')] <= 1 / |R_q^kappa|.
```

Fix distinct `r, r' in (A_b^D)^s`. Let:

```text
z = r - r' in R_q^s.
```

Since `r != r'`, `z != 0`. Therefore some digit-ring coordinate `z_j` is
nonzero. By A2 and the short-invertibility lemma, this `z_j` is a unit in
`R_q`.

Write one row of `B` as:

```text
b = (b_1, ..., b_s) in R_q^s.
```

The event that this row collides on `r` and `r'` is:

```text
<b, z> = sum_i b_i z_i = 0 in R_q.
```

Condition on all row entries except `b_j`. Since `z_j` is a unit, there is
exactly one value of `b_j` that satisfies the equation:

```text
b_j = -z_j^{-1} * sum_{i != j} b_i z_i.
```

Because `b_j` is uniform in `R_q`, the probability for this row is exactly:

```text
Pr_b[<b, z> = 0] = 1 / |R_q|.
```

The `kappa` output rows are sampled independently, so:

```text
Pr_B[B * z = 0] = (1 / |R_q|)^kappa
                = 1 / |R_q^kappa|.
```

Since:

```text
h_B(r) = h_B(r')  <=>  B * (r - r') = 0,
```

the family is two-universal in the standard universal-hashing sense; see
Stinson's survey of universal hash families and the Leftover Hash Lemma, and
Tomamichel-Schaffner-Smith-Renner for the same collision-family notion in the
leftover-hashing setting.

#### LHL Statistical Distance

Let `R` be the random digit-source mask sampled uniformly from `(A_b^D)^s`,
and let `U` be uniform over `R_q^kappa`. Because every digit coefficient has
`beta = log_basis` bits of min-entropy:

```text
H_min(R) = s * D * beta.
```

For a two-universal hash family from the source domain to `R_q^kappa`, the
Leftover Hash Lemma gives the following public-seed extraction bound; see
Stinson for the classical statement and applications, and
Tomamichel-Schaffner-Smith-Renner for a modern generalized statement:

```text
Delta((B, h_B(R)), (B, U))
    <= 1/2 * sqrt(|R_q^kappa| / 2^{H_min(R)})
    =  1/2 * sqrt(q^{D * kappa} / 2^{s * D * beta}).
```

The `B` appears in both tuples because `B` is public. The ideal distribution is
the same public seed `B` together with a value `U` that is uniform in
`R_q^kappa` and independent of `B`.

To make the distance at most `2^{-lambda}`, it is enough to have:

```text
H_min(R) >= log2(|R_q^kappa|) + 2 * lambda - 2
```

or:

```text
s * D * beta >= kappa * D * log2(q) + 2 * lambda - 2.
```

So:

```text
s >= ceil((kappa * D * log2(q) + 2 * lambda - 2) / (D * beta)).
```

Akita sizes this with `lambda = 128` and the field modulus bit width
`field_bits`. Since `log2(q) <= field_bits`, it is conservative to require:

```text
s = ceil((kappa * D * field_bits + 2 * 128 - 2) / (D * beta)).
```

Equivalently:

```text
s = ceil((kappa * D * field_bits + 254) / (D * log_basis)).
```

The helper returns zero when `kappa = 0`, because a zero-width output has no
public commitment coordinates to hide. The closed form above is the nonzero
`kappa` case used by commitment rows.

For the default fp128 prime and `kappa = 1`, this gives:

```text
D = 64,  log_basis = 5: s = 27 digit-ring planes.
D = 128, log_basis = 5: s = 26 digit-ring planes.
D = 64,  log_basis = 4: s = 33 digit-ring planes.
```

Substituting the selected `s` into the LHL inequality gives:

```text
Delta((B, B * R), (B, U)) <= 2^{-128}.
```

#### Adding the Message Offset

The public commitment is:

```text
u = B_msg * t_hat + B_blind * R.
```

For a fixed message witness `t_hat` and public `B_msg`, the term:

```text
c = B_msg * t_hat
```

is fixed in `R_q^kappa`. Adding a fixed offset is a bijection on `R_q^kappa`, so
it preserves statistical distance from uniform:

```text
Delta(c + B_blind * R, U)
  = Delta(B_blind * R, U).
```

Therefore, for every fixed message witness, the joint view containing the
public setup and the revealed commitment `u` is within `2^{-128}` statistical
distance of the same public setup and an independent uniform value in
`R_q^kappa`.

For multiple commitment groups under the same public setup matrix, each group
samples an independent mask vector. The same seed `B` is reused, so the joint
statement is a standard hybrid over independent source samples rather than a
claim that the setup is freshly sampled for every commitment.

For a fixed public seed `B`, let `P_B` be the pad distribution `B * R` for one
fresh digit-source mask and let `U` be uniform over `R_q^kappa`.
The strong LHL bound above gives:

```text
E_B[Delta(P_B, U)] <= epsilon
```

with `epsilon = 2^{-128}` for the configured per-commitment target. For `Q`
fresh independent commitment masks under the same seed, the product-distance
hybrid gives:

```text
E_B[Delta(P_B^Q, U^Q)] <= Q * epsilon.
```

The fixed message offsets may depend on the same public setup, but adding them
coordinate-wise is a bijection for every fixed `B` and therefore does not change
the distance. Thus independent masks do address the same-seed reuse issue, with
the usual additive loss in the number of hidden commitments. In the
implementation, grouped commitments each receive their own `FlatDigitBlocks`
blinding payload through `AkitaCommitmentHint::with_recomposed_inner_rows`, and
batched root/ring-switch code tracks those groups through `group_poly_counts`.

#### Why SIS Is Not the Hiding Argument

It is tempting to argue that `B * r` hides because SIS makes collisions hard,
but that is the wrong implication. Two-universality says:

```text
for every fixed nonzero difference z,
Pr_B[B * z = 0] <= 1 / |R_q^kappa|.
```

The probability is over the random setup matrix `B`.

SIS says:

```text
given one fixed public B, it is computationally hard to find
a short nonzero z such that B * z = 0.
```

The probability is over an adversary's computation after seeing `B`. The hiding
proof above is information-theoretic and uses only the random linear map's
two-universality on the masked source domain. SIS is still needed for binding,
but not for the LHL statistical hiding bound.

### Alternatives Considered

1. Hybrid runtime and compile-time ZK mode.
  This was tried on the branch as a runtime/type-level hiding mode layered on
   top of Cargo feature gating. It was rejected because it created too much
   diff across config, setup, planner, scheme, prover, verifier, tests, benches,
   and generated schedule policy. The final compile-time-only `zk` feature keeps
   the code simpler: transparent builds compile away ZK fields and sizing, and
   `zk` builds make the enlarged witness shape explicit.
2. Runtime enum selection between transparent and ZK commitments.
  Runtime selection would let one binary choose behavior at runtime, but every
   affected path already depends on monomorphized field/config/ring-dimension
   types and schedule formulas. A runtime enum would add branches in hot paths
   and increase the chance that prover and verifier derive different shapes.
3. Always reserve ZK blinding columns and use zero masks for transparent mode.
  This would reduce conditional code but would permanently increase
   transparent setup stride, witness size, planner outputs, proof-size
   estimates, and root-direct proof handling. It would also break tests that
   depend on deterministic transparent commitments.
4. Reuse transparent generated schedule tables in `zk` builds.
  Transparent tables do not account for blinding columns. Reusing them could
   undercount recursive witness lengths, select invalid shrink steps, and size
   setup matrices too narrowly. The branch disables generated fp128 plans under
   `zk` and falls back to planner search.
5. Hide root-direct proofs without revealing blinding digits.
  The direct path reveals the witness and does not run the folded relation that
   would otherwise bind private blinding digits. Revealing the blinding payload
   is the smallest verifier-consistent fix. Full zero-knowledge direct openings
   require a separate sigma/tail design and are out of scope.
6. Serialize blinding material in folded proofs.
  This was rejected because folded proofs already prove the blinding segment as
   private witness data through the commitment relation. Revealing it would undo
   commitment hiding. Only root-direct proofs carry it, because that path is
   already non-ZK and needs recommitment data.

## Execution

Implementation direction already reflected by the branch:

- Add `CanonicalField::modulus_bits()` and implement it for fp32, fp64, and
fp128 fields.
- Add `akita_types::zk` with the LHL blinding-count formula and propagate the
`zk` feature through workspace crates.
- Extend setup sizing and planner/runtime witness formulas to include
feature-gated blinding columns.
- Sample fresh masking factors in prover commitment paths and store them in
`AkitaCommitmentHint`.
- Preserve blinding hints through recursive hint cache conversion.
- Extend root folded witness construction and verifier M-evaluation with the
same blinding segment and offsets.
- Extend root-direct proof construction and verification to carry and use the
blinding digits in `zk` builds.
- Disable generated fp128 schedule lookup under `zk` until generated ZK tables
are available.
- Add ZK dense end-to-end tests for commitment re-randomization and proof
verification across D=32, D=64, and D=128.

Risks to resolve before treating this as a final public ZK surface:

- Audit the short-invertibility assumption for every supported field/ring
parameter set.
- Add explicit negative tests for corrupted root-direct blinding payloads and
folded blinding witness mismatches.
- Generate and validate dedicated ZK schedule tables, or document planner
fallback as the intended production path.
- Specify the full proof zero-knowledge layer that hides sumcheck messages and
terminal witnesses; commitment hiding is only the first layer.

## References

- `specs/TEMPLATE.md`
- `crates/akita-types/src/zk.rs`
- `crates/akita-types/src/proof/mod.rs`
- `crates/akita-types/src/schedule.rs`
- `crates/akita-types/src/layout/proof_size.rs`
- `crates/akita-field/src/arithmetic.rs`
- `crates/akita-prover/src/protocol/masking.rs`
- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-prover/src/backend/recursive_hint.rs`
- `crates/akita-prover/src/protocol/quadratic_equation.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/protocol/batched.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-pcs/tests/zk.rs`
- Nguyen, O'Rourke, and Zhang, "Hachi: Efficient Lattice-Based Multilinear
Polynomial Commitments over Extension Fields," ePrint 2026/156.
- Lyubashevsky and Seiler short-invertibility lemma for power-of-two
cyclotomic rings, as used in the commitment-hiding proof above.
- Douglas R. Stinson, "Universal Hash Families and the Leftover Hash Lemma,
and Applications to Cryptography and Computing,"
[https://cs.uwaterloo.ca/~dstinson/papers/leftoverhash.pdf](https://cs.uwaterloo.ca/~dstinson/papers/leftoverhash.pdf).
- Marco Tomamichel, Christian Schaffner, Adam Smith, and Renato Renner,
"Leftover Hashing Against Quantum Side Information,"
[https://arxiv.org/pdf/1002.2436](https://arxiv.org/pdf/1002.2436).
