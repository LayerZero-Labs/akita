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

The two-tier template: inner commitment `t = A·G⁻¹(f)`, outer commitment via `B`,
and the opening commitment via `D`. How binding reduces to Module-SIS.

**Sources to fold in**

- `crates/akita-prover/src/api/commitment.rs:529-721` (`commit`, `batched_commit`).
- `crates/akita-prover/src/backend/onehot/inner_ajtai.rs`.
- `crates/akita-types/src/sis/ajtai_key.rs`.
- Paper §2.6 `sec:prelim-pcs` (two-tier Ajtai), §3.2 `sec:akita-layout` (commitment matrices, inner/outer commitments).

## Polynomial backends: dense vs one-hot

When the dense (CRT+NTT digit) mat-vec is used versus the one-hot backend that
iterates only nonzero monomial positions. One-hot at **fp128 D64** is the usual
production choice; smaller **D32** and larger **D128** are alternates (see
`usage/quickstart.md`).

**Sources to fold in**

- `crates/akita-prover/src/backend/dense.rs`, `backend/onehot/mod.rs`.
- `crates/akita-pcs/src/lib.rs:1-72`.
- Paper App B.2.5 (one-hot commitment optimization), `sec:akita-crt-matvec`.
- `specs/simd-ring-subfield-fp8.md` (technique note; primary consumer removed).
