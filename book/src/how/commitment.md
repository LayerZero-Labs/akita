# Setup and commitment

> **Status:** stub. Part of the initial Akita Book scaffold.

How public parameters are built and how a polynomial becomes an Ajtai
commitment, including the two backends (dense and one-hot) that compute the
commitment mat-vec.

## Setup

The shared setup vector of field elements, interpreted (packed tightly) as the
A/B/D matrices at every level, plus how setup is constructed and optionally
cached.

**Sources to fold in**

- `crates/akita-setup/src/lib.rs:39-67`.
- Paper §3.9 `sec:akita-setup` (packed shared setup), §3.8 `Setup`.
- `specs/setup-layout-repack.md` (packed-setup direction — roadmap).

## Ajtai commitment mechanics

The two-tier template decomposes each witness block into commit digits `s_hat`.
It computes the inner commitment `t = A * s_hat`, then uses `B` for the outer
relation and `D` for the opening relation. Neither full image is public. Every
B and D image is negative-binary decomposed through exactly two rank-one maps;
the wire carries only the 128-byte terminal payload `p_F` or `p_H`. The source
image is capped at 8 KiB. The binding argument reduces to Module-SIS for the
base relation and for each compression map.

The compression ladder is profile-owned (`q128: 16/8`, `q64: 32/16`, `q32:
64/32`) and is separate from A/B/D matrix dimensions. All A/B/D dimensions are
at least 64; the smaller compression rows stay in the shared witness tail and
cannot change ordinary relation alignment.

**Sources to fold in**

- `crates/akita-prover/src/api/commitment.rs` (role-aware `commit` and its shared kernel).
- `crates/akita-prover/src/backend/onehot/inner_ajtai.rs`.
- `crates/akita-types/src/sis/ajtai_key.rs`.
- Paper §2.6 `sec:prelim-pcs` (two-tier Ajtai), §3.2 `sec:akita-layout` (commitment matrices, inner/outer commitments).

## Polynomial backends: dense vs one-hot

When the dense (CRT+NTT digit) mat-vec is used versus the one-hot backend that
iterates only nonzero monomial positions. One-hot at **fp128 D64** is the usual
production choice; **D128** remains a comparison / legacy profile (see
`usage/quickstart.md`).

**Sources to fold in**

- `crates/akita-prover/src/backend/dense/mod.rs`, `backend/onehot/mod.rs`.
- `crates/akita-pcs/src/lib.rs:1-72`.
- Paper App B.2.5 (one-hot commitment optimization), `sec:akita-crt-matvec`.
- `specs/simd-ring-subfield-fp8.md` (technique note; primary consumer removed).
