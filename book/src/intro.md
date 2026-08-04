# Introduction

Akita is a high-performance, modular, lattice-based polynomial commitment scheme
(PCS) with transparent setup and post-quantum design goals, written in Rust. It
is intended to replace Dory as the PCS inside the [Jolt](https://jolt.a16zcrypto.com/)
zkVM.

> **Status:** this book is an initial scaffold. Most pages are stubs that name
> the source files, specs, and paper sections their content should be folded
> from. See [How to read this book](#how-to-read-this-book) below.

This book has four top-level parts:

1. **[Usage](./usage/usage.md)** — how to build, configure, commit, prove,
   verify, profile, and integrate Akita (including the Jolt recursion path).
2. **[How it works](./how/how-it-works.md)** — the architecture and the
   commit → fold → recurse → verify protocol, end to end.
3. **[Foundations](./foundations/foundations.md)** — the field, ring, sumcheck,
   and lattice background, plus the glossary, notation, spec index, and
   references.
4. **[Roadmap](./roadmap/roadmap.md)** — in-flight and planned work.

## What is Akita?

A multilinear PCS whose binding and knowledge soundness reduce to Module-SIS,
with generated production tables targeting at least 128 bits under a scalar
infinity-norm LGSA estimate using the ADPS16 quantum cost model. This is an
attack-cost model, not an unqualified post-quantum security claim.
It commits to
base-field (or extension-field) multilinear polynomials, then proves evaluation
claims by a recursive fold whose witness shrinks roughly from `N` to `N^{1/2}`
ring elements per step. No trusted setup, no pairing, post-quantum target.

**Sources to fold in**

- `crates/akita-pcs/src/lib.rs:1-16` — umbrella crate module docs.
- Paper §1 `sec:introduction`, §1.1 `sec:contributions` (Akita's contributions).
- `specs/akita-pcs-crate-decomposition.md` (lineage, naming).
- Council note: post-quantum is currently asserted, not argued — keep the claim honest.

## Lineage and naming

Akita descends from the LaBRADOR → Greyhound → Hachi line of lattice folding
arguments. Naming maps (paper ↔ code) and the "what each predecessor
contributed" story belong here.

**Sources to fold in**

- Paper §3 `sec:akita-recap` ("From Hachi to Basic Akita": Greyhound relation
  matrix, Hachi's three contributions, Akita's contributions).
- `crates/akita-types/src/sis/norm_bound.rs:1-2` (Hachi Lemma 7 reference).
- `specs/archive/2026-Q2/w-to-e-notation.md` (paper ↔ code naming).
- External: Hachi, Greyhound, LaBRADOR, SuperNeo (ePrint 2026/242). See [References](./foundations/references.md).

## Security status (honest)

State the audited reality separately from the marketing claim: which hardness
assumption is used, why the production tables use coefficient-`L∞` SIS bounds,
and what is asserted vs proven. The canonical narrative lives in
[How it works → Security model](./how/security.md).

**Sources to fold in**

- `crates/akita-types/src/sis/` (`mod.rs`, `ajtai_key.rs`, `norm_bound.rs`).
- `docs/security-posture.md`.
- Paper §3.12 `sec:batched-soundness`, §3.11 `sec:akita-cwss` (audited soundness).
- `specs/security-hardening.md`, `specs/sis-quantum128-scalar-n-table.md`,
  `specs/fold-linf-rejection.md`.
- `specs/sis-quantum128-scalar-n-table.md` (implemented policy, role coverage,
  generated tables, and schedule identity).

## How to read this book

Reading orders by audience:

- **Application developer (integrating Jolt or another host):** start with
  [Usage](./usage/usage.md). While chapters are stubs, use
  [`profile/akita-recursion/README.md`](../../profile/akita-recursion/README.md),
  [`specs/single-point-opening-batch.md`](../../specs/single-point-opening-batch.md),
  and [`AGENTS.md`](../../AGENTS.md) for the current API contracts.
- **Contributor:** [How it works](./how/how-it-works.md), lead with the lifecycle.
- **Reviewer:** [Foundations](./foundations/foundations.md) + [Security model](./how/security.md).

Surface the hardest newcomer questions early.

**Sources to fold in**

- Council newcomer report (hardest questions, reading order).
- Paper §1.3 `sec:organization` (paper's own reading guide).
