# Spec: NIST Category 3 Module-SIS Policy

| Field         | Value |
|---------------|-------|
| Author(s)     | sumchecker |
| Created       | 2026-10-10 |
| Status        | active |
| PR            | |
| Supersedes    | the 128-bit target and `Quantum128BitADPS16` policy ID of [`sis-quantum128-scalar-n-table`](sis-quantum128-scalar-n-table.md) |
| Superseded-by | |
| Book-chapter  | book/src/how/security.md |

## Summary

Akita priced every Module-SIS instance at 128 bits under the ADPS16 quantum
Core-SVP model. That is above ML-DSA-44 (NIST category 2) but below the
category 3 parameter sets that many deployments now choose as a buffer against
progress in lattice cryptanalysis. This spec raises the single SIS gate to the
category 3 level of ML-DSA-65. The Fiat-Shamir and challenge-space targets are
out of scope and stay at 128 bits; see [Non-goals](#non-goals).

## NIST category 3 reference parameters

NIST defines category 3 as "any attack ... must require computational
resources comparable to or greater than those required for key search on a
block cipher with a 192-bit key (e.g. AES192)". The call for proposals prices
that reference at `2^207` classical gates or `2^233 / MAXDEPTH` quantum gates,
with `MAXDEPTH` between `2^40` and `2^96`. Category 4 is SHA3-384 collision
search at `2^210` classical gates.

Lattice schemes are not priced in gates directly. Their designers publish
Core-SVP block sizes, and NIST accepted the following ones as category 3.

ML-DSA-65 parameters (FIPS 204, Table 1):

| Parameter | Value |
| --- | --- |
| `q` | 8,380,417 |
| `d` (dropped bits of `t`) | 13 |
| `tau` (challenge weight) | 49 |
| `lambda` (collision strength of `c~`) | 192, so `c~` is 384 bits |
| `gamma1` | `2^19` |
| `gamma2` | `(q - 1) / 32 = 261,888` |
| `(k, l)` | `(6, 5)` |
| `eta` | 4 |
| `beta = tau * eta` | 196 |
| `omega` | 55 |
| public key / private key / signature | 1,952 / 4,032 / 3,309 bytes |

Core-SVP hardness of the same lattice parameters (Dilithium3, round-3
specification, Table 1). Classical Core-SVP prices block size `b` at
`2^(0.292 b)`; quantum Core-SVP at `2^(0.265 b)`.

| Assumption | BKZ block size | Classical | Quantum |
| --- | --- | --- | --- |
| MLWE (key recovery) | 624 | 182 | 165 |
| SelfTargetMSIS, `zeta = max(gamma1 - beta, 2 gamma2 + 1 + 2^(d-1) tau)` (UF-CMA) | 638 | 186 | 169 |
| MSIS, `zeta' = max(2 (gamma1 - beta), 4 gamma2 + 2)` (SUF-CMA) | 602 | 176 | 159 |

ML-KEM-768 (Kyber768, round-3 specification, Table 4) is the other category 3
lattice standard: primal attack block size 626, Core-SVP 183 classical and 166
quantum.

## Calibration against Akita's estimator

`crates/akita-sis-estimator/examples/mldsa_calibration.rs` builds the three
ML-DSA MSIS instances as scalar SIS (`n = 256 k` rows, `256 (k + l + 1)` UF or
`256 (k + l)` SUF columns, coefficient `L-infinity` bound `zeta` or `zeta'`)
and prices them with the production ADPS16/LGSA proven-pruned search, with no
early classification. It reproduces the published table:

| Instance | Akita block size | Akita quantum | Akita classical | Published |
| --- | --- | --- | --- | --- |
| ML-DSA-44 UF | 423 | 112.10 | 123.52 | 423, 112 / 123 |
| ML-DSA-44 SUF | 417 | 110.51 | 121.76 | 417, 110 / 121 |
| ML-DSA-65 UF | 638 | 169.07 | 186.30 | 638, 169 / 186 |
| ML-DSA-65 SUF | 603 | 159.80 | 176.08 | 602, 159 / 176 |
| ML-DSA-87 UF | 909 | 240.89 | 265.43 | 909, 241 / 265 |
| ML-DSA-87 SUF | 868 | 230.02 | 253.46 | 868, 230 / 253 |

Akita's former 128-bit gate needs block size 484, between ML-DSA-44 and
ML-DSA-65.

## Decision

The policy is `Quantum169BitADPS16` with wire tag `2`. Its only hard gate is an
ADPS16 quantum LGSA score of at least 169 bits, so every Akita SIS instance
needs BKZ block size at least 638. That is the largest Core-SVP figure among
NIST's category 3 lattice standards, and every Akita SIS instance is at least
as hard as each of them under the same model. Tag `1` named the retired
128-bit policy and is rejected; no SIS table, schedule, setup prefix, or proof
can alias across the two policies.

The coefficient `L-infinity` table, the Euclidean `L2` table, and the narrow
compression cells all read the same policy constraint. The estimator,
proven-pruned certificate domain, LGSA shape, `0.2650` exponent, rank cap 20,
and search cap are unchanged from
[`sis-quantum128-scalar-n-table`](sis-quantum128-scalar-n-table.md).

### Compression ladder

The rank-one negative-binary compression maps lose about a factor three in
admissible width at 169 bits. The 128-bit ladder no longer fits for any
profile:

| Profile | Map | Needed width | 128-bit cap | 169-bit cap |
| --- | --- | --- | --- | --- |
| q128 | D=16, 8 KiB source | 4,096 | 7,077 | 2,400 |
| q128 | D=8, terminal | 256 | 508 | 165 |
| q64 | D=32 / D=16 | 2,048 / 128 | 3,538 / 254 | 1,200 / 82 |
| q32 | D=64 / D=32 | 1,024 / 64 | 1,769 / 127 | 600 / 41 |

A 128-byte terminal fixes the terminal map at `D = 128 bytes / field bytes`.
To compress at all, that map must absorb the output of a larger rank-one map,
so at least `2 D` field elements, which is a width of at least twice the field
bit width: 256 for q128 against a cap of 165, 128 for q64 against 82, and 64
for q32 against 41. No number of maps fixes this. The ladder therefore doubles
every dimension and the terminal image becomes 256 bytes:

| Profile | First map: needed / cap | Terminal map: needed / cap |
| --- | --- | --- |
| q128 | D=32: 2,048 / 60,460 | D=16: 256 / 2,400 |
| q64 | D=64: 1,024 / 30,230 | D=32: 128 / 1,200 |
| q32 | D=128: 512 / 15,115 | D=64: 64 / 600 |

The compression policy becomes `NegativeBinaryTwoMapExactMonotoneCutover8KiBV4`
(tag `4`). Every compressed payload costs 128 more proof bytes.

## Intent

### Goal

Every generated SIS instance (A, B, D, terminal A, setup-prefix and
precommitted groups, and both compression maps) meets the ADPS16 quantum
169-bit gate.

### Invariants

- One policy constraint feeds every table: `SisSecurityPolicy::adps16_quantum_constraint`.
- Runtime lookups for any other policy, table digest, or tag fail closed.
- `catalog_security --check` reports every schedule at or above 169 bits.

### Non-goals

These parameters bound statistical or random-oracle soundness, not lattice
hardness. They remain at the 128-bit level:

- `TRANSCRIPT_SECURITY_BITS = 128`, the per-query Fiat-Shamir grinding target.
- The challenge fields: fp128 uses the base field, fp64 a degree-2 extension,
  fp32 a degree-4 extension. Each has about `2^128` elements, so a sumcheck or
  Schwartz-Zippel check cannot exceed about 128 bits per query without a larger
  extension.
- The 128-bit entropy floor of the sparse fold challenge families.
- 32-byte transcript challenge blocks and fold seeds (128-bit collision
  resistance; ML-DSA-65 uses a 384-bit `c~` for its 192-bit collision target).

A full category 3 claim needs 256-bit challenge sets (fp128 `Ext2`, fp64
degree 4, fp32 degree 8), a 192-bit grinding target and challenge entropy
floor, and 48-byte transcript digests. That is a protocol change, not a
parameter change, and it is not part of this spec.

## Evaluation

### Acceptance criteria

- [x] The calibration example reproduces the published ML-DSA Core-SVP figures.
- [ ] `generated_sis_table`, `generated_l2_sis_table`, and the compression
      cells are regenerated under `Quantum169BitADPS16`.
- [ ] Every checked-in schedule artifact is regenerated, and
      `catalog_security --check` passes.
- [ ] The CI test pass is green.
- [ ] Proof size and runtime are measured against `main`.

### Performance

See [the evidence report](evidence/sis-nist-level3-catalog.md).
