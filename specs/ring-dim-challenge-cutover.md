# Spec: per-role ring dimensions, CRT limits, and fold-challenge cutover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-04 |
| Status        | draft |
| PR            | #268 (stacked on #249 `quang/runtime-ring-full-cutover`) |
| Supersedes    | `specs/crt-ntt-prime-profiles.md` (prime tables); replaces draft `ring-dim-prime-extension.md` |
| Superseded-by | |
| Book-chapter  | book/src/foundations/ntt-crt.md |

## Summary

One PR on #249. Together these changes mean:

1. **Each commitment matrix keeps its own ring size** — fold/witness work on A
   at `d_a ≥ 64`; B and D may still use D=32 when nested (`d_d | d_b | d_a`).
2. **CRT matvec scales to larger rings** — new primes, per-field-size caps, NTT
   cache dispatch through D=2048.
3. **Ring fold challenges unify** — drop D=32 bounded-L₁ and D=32 production
   presets; one `SparseChallengeConfig { count_pm1, count_pm2 }` struct, a
   single signed-sparse sampler, production ladder through D=2048, and
   witness-fold transcript labels (`ak/c/wf`, …).

No backward compatibility. Removed families, old enum variants, and legacy label
bytes are deleted, not aliased.

## Ring fold challenges (protocol placement)

After the prover's folded witness message `v = D · ŵ` is absorbed
(`ABSORB_PROVER_V`), the protocol samples sparse ring elements `c` used to fold
the witness toward the next commitment. This is **not** sumcheck "stage 1"; it is
witness folding between `v` and the next `u` commitment.

| Rust name | Role |
|-----------|------|
| `LevelParams::fold_challenge_config` | `(count_pm1, count_pm2)` family at this fold level |
| `witness_fold_challenge_labels()` | Fiat–Shamir absorb-buffer labels for flat/tensor draws |
| `CHALLENGE_WITNESS_FOLD` | `b"ak/c/wf"` — flat witness-fold draw |
| `CHALLENGE_TENSOR_FOLD_LEFT` | `b"ak/c/wfl"` |
| `CHALLENGE_TENSOR_FOLD_RIGHT` | `b"ak/c/wfr"` |
| `ABSORB_TENSOR_FOLD_LEFT` | `b"ak/a/wtl"` — binds right factor to left digest |

Production [`AkitaTranscript`](crates/akita-transcript) sponges are **positional**:
label bytes are **not** absorbed into the live sponge (diagnostics/logging only).
They **are** included in the sparse-challenge Fiat–Shamir absorb buffer that
derives fold-challenge seeds.

## Investigation: dispatch — extend everything, or dispatch tightly?

### Cost model

Runtime `match` on ring degree is cheap. The real cost is **how many const-D
monomorphizations** each call site pays for at compile time.

Today one macro (`dispatch_ring_dim_result!`) served every role with the same
four arms `{32, 64, 128, 256}`. Role × field-tier dispatch removes unused arms.

### Design: two axes for fold / ring-switch dispatch

Fold and ring-switch paths dispatch on **both**:

1. **Matrix role** — inner (A) vs outer (B) vs opening (D).
2. **PCS field tier** — base prime width (128-bit / 64-bit / 32-bit `F`).

NTT cache dispatch uses **field tier only** (CRT profile caps).

**Heuristic:** as base prime width **decreases**, the **ceiling ring dimension
rises** — fp128 modest `d_a`, fp64 higher D, fp32 ladder top (512/1024/2048).

**Legacy exception:** on fp32/fp64, **keep** dispatch arms for shipped small ring
dims (64, 128, 256). High-D inner arms are additive later.

### Protocol arm sets by role × field tier (Slice B)

Inner (`d_a`, no D=32):

| Field tier | Inner arms (this PR) | Notes |
|------------|----------------------|-------|
| fp128 (128-bit) | 64, 128 | 256+ deferred |
| fp64 (64-bit) | 64, 128, 256 | legacy small D kept |
| fp32 (32-bit) | 64, 128, 256 | legacy small D kept; 512+ deferred |

Outer / opening (`d_b`, `d_d`, tier-specific floor):

| Field tier | B/D arms (this PR) | Notes |
|------------|-------------------|-------|
| fp128 | 16, 32, 64, 128, 256 | D=16 for finer nested opening |
| fp64 | 32, 64, 128, 256 | legacy small D kept |
| fp32 | 64, 128, 256 | no D=32 on fp32 tier |

### NTT dispatch by field tier

| Field tier | CRT profile | NTT min D | NTT max D |
|------------|-------------|----------:|----------:|
| fp128 | Q128 | 16 | 512 |
| fp64 | Q64 | 32 | 1024 |
| fp32 | Q32 | 64 | 2048 |

### Sampler

`MAX_STACK_RING_DIM = 2048`; stack tiers through 2048 in `position_sample.rs`.
Single **signed-sparse** sampler (`signed_sparse.rs`): `count_pm2 == 0` is
pm1-only (±1); `count_pm2 > 0` adds ±2 coefficients (production D=64).

## Intent

### 1. CRT prime tables

```text
1073692673, 1073668097, 1073707009, 1073738753, 1073732609
observed v₂ = 14,        13,        11,        10,        10
```

| Profile | K | Assignment |
|---------|--:|------------|
| Q32     | 2 | first two |
| Q64     | 3 | first three |
| Q128    | 5 | full list |

### 2. Per-role ring-dimension constants

| Field / CRT profile | max D |
|---------------------|------:|
| Q128 (fp128)        | 512   |
| Q64 (fp64)          | 1024  |
| Q32 (fp32)          | 2048  |

```rust
pub const SUPPORTED_CHALLENGE_RING_DIMS: [usize; 6] =
    [64, 128, 256, 512, 1024, 2048];

pub const MIN_A_ROLE_FOLD_CHALLENGE_RING_D: usize = 64;
```

`validate_role_dims_match_keys` also calls
`fold_challenge_config.validate_for_ring_dim(d_a)`.

### 3. Role- and field-tier protocol dispatch

`dispatch_for_field!` with an explicit [`ProtocolDispatchSlot`] (role, envelope, or NTT).

### 4. D=32: A-role cutover vs B/D support

**Remove:** `d_a = 32` presets; `proof_optimized_ring_challenge_config(32)`.

**Keep:** D=32 on outer/opening dispatch and NTT; nested `{128, 64, 32}`.

### 5. Unified fold-challenge config (replaces enum families)

```rust
pub struct SparseChallengeConfig {
    pub count_pm1: usize,
    pub count_pm2: usize,
}
```

- **Sampler:** one signed-sparse path (`signed_sparse.rs`).
- **Production ladder:** `SparseChallengeConfig::production_for_ring_dim(d)` —
  `(64→(31,10), 128→(31,0), 256→(23,0), 512→(19,0), 1024→(16,0), 2048→(14,0))`.
- **Validation:** `validate_for_ring_dim(d_a)` = structural + 128-bit entropy floor.
- **Removed:** `BoundedL1Norm`, `Uniform`, `ExactShell` enum variants and separate
  sampler modules.

#### Descriptor and domain separator (single encoding)

| Tag | Payload |
|----:|---------|
| 0 | `count_pm1`, `count_pm2` (usize wire encoding in descriptor; u64 LE in domain separator) |

Tag 2 (`BoundedL1Norm`) and the old tag-0/tag-1 split (`Uniform` vs
`ExactShell`) are **deleted**. Schedule tables regenerated.

Archive bounded-L₁ design: `specs/archive/bounded-l1-sparse-challenge.md`.

### 6. Fold-challenge ladder (≥128-bit entropy per draw)

| D    | (pm1, pm2) |
|-----:|-----------:|
| 64   | (31, 10) |
| 128  | (31, 0) |
| 256  | (23, 0) |
| 512  | (19, 0) |
| 1024 | (16, 0) |
| 2048 | (14, 0) |

Live E2E at `d_a > 256` waits on inner dispatch + backends + preset.

## Non-Goals (this PR)

- fp32/fp64 **inner** protocol arms for 512/1024/2048.
- Backends monomorphized at D > 256 on any tier.
- New shipped presets with `d_a = 512+`.

## Implementation checklist

### Slice A — ring dims + CRT

- [x] Prime tables + capacity goldens
- [x] Per-role `validate_role_dims`
- [x] Challenge ladder in `production_for_ring_dim`
- [x] D32 fp128 presets removed

### Slice B — role × field-tier dispatch + NTT

- [x] Protocol + NTT tier dispatch through D=2048 cache variants

### Slice B′ — tier-specific NTT floors + fp128 D=16

- [x] Tier floors; `validate_role_dims_for_field`

### Slice C — sampler tiers

- [x] `MAX_STACK_RING_DIM = 2048`; seed tests at D=512/1024/2048

### Slice D — BoundedL1Norm deletion

- [x] Delete bounded-L₁ sampler and enum variant

### Slice F — unified config + naming cutover

- [x] `SparseChallengeConfig { count_pm1, count_pm2 }` struct
- [x] Single signed-sparse sampler; `position_sample.rs` for distinct positions
- [x] `fold_challenge_config` on `LevelParams` (was `stage1_config`)
- [x] `witness_fold_challenge_labels`, `CHALLENGE_WITNESS_FOLD = b"ak/c/wf"`
- [x] Tensor labels `ak/c/wfl`, `ak/c/wfr`, `ak/a/wtl`
- [x] Descriptor/domain separator: tag 0 + two counts
- [x] Schedule table regen + catalog digests

### Slice E — docs guardrails

- [ ] `./scripts/check-doc-guardrails.sh`

## Evaluation

### Acceptance Criteria

- [x] No `BoundedL1Norm`; single descriptor tag 0 encoding.
- [x] `fold_challenge_config.validate_for_ring_dim(d_a)` at role-dim boundary.
- [x] Witness-fold transcript labels use `ak/c/wf` family (not `s1f`).
- [ ] Prime tables and dispatch acceptance criteria from Slice A/B.
- [ ] CI green.

### Testing Strategy

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh
```
