# Binary-source q24 SIS certificate

Status: active
Book-chapter: book/src/how/security.md

The binary-source commitment's A matrix has rank three over the degree-648
trinomial ring `Z_q[Y]/(Y^648 - Y^324 + 1)` with
`q = 2^24 - 2,523,807 = 14,253,409` (`A_RANK` in
`akita-types` `binary_source/commitment_manifest.rs`, `Q24` in `akita-algebra`).
This record is the offline SIS evidence for that rank. The estimator uses
`AkitaModulusProfileId::Q24Minus2523807`, label `q24`, under the `labinius-sis`
feature of `akita-sis-estimator`. No catalog or protocol reads the certificate.

The artifact
`crates/akita-sis-estimator/data/q24_commitment_prime_infinity_width.csv`
uses the `local-minimum+proven-pruned-certification` mode, the quantum
ADPS16/LGSA model, the 169-bit policy and a 6,400,000,000,000 ring-column
search cap. Local-minimum search discovers a boundary; proven-pruned search
certifies it. An optimizer probe is never an admission certificate.

For `D = 648`, scalarization is `n = rank * 648`, `m = width * 648`.
The requested widths 4096, 8192 and 16384 are ring columns, corresponding
to 2,654,208, 5,308,416 and 10,616,832 scalar columns. The three bounds
are exactly `eta = 4 * (2w) * 4095` for weights 46, 64 and 77. They are
source-comparison bounds, not honest response radii.

Every entry below applies separately to **each** width in
`{4096, 8192, 16384}`. The independently evaluated 27 fixed-width cells,
including rejected rank-two cells, are recorded in
`crates/akita-sis-estimator/data/q24_infinity_cells.csv`.

| Rank | eta | Quantum bits | Certificate status | Clears 128 bits |
| --- | --- | --- | --- | --- |
| 2 | 1,506,960 | 126.14 | Finite attack, beta 476 | No |
| 2 | 2,096,640 | 119.78 | Finite attack, beta 452 | No |
| 2 | 2,522,520 | 116.60 | Finite attack, beta 440 | No |
| 3 | 1,506,960 | >=169.07 | Proven above target, beta threshold 638 | Yes |
| 3 | 2,096,640 | >=169.07 | Proven above target, beta threshold 638 | Yes |
| 3 | 2,522,520 | >=169.07 | Proven above target, beta threshold 638 | Yes |
| 4 | 1,506,960 | >=169.07 | Proven above target, beta threshold 638 | Yes |
| 4 | 2,096,640 | >=169.07 | Proven above target, beta threshold 638 | Yes |
| 4 | 2,522,520 | >=169.07 | Proven above target, beta threshold 638 | Yes |

All six rank-three/four width certificates reach the search cap with
`AtLeast` status and no rejected successor. Their `above-target:169.07`
values are certified lower bounds, not actual attack optima; 638 is a
block-size threshold, not a selected optimal attack. Under the width
search's monotone-prefix precondition, they cover all requested widths.
The exact cutoff above the cap remains unknown. Rank three therefore has
certificate support for the 128-bit planning target. Rank two fails that
target at every requested cell; this conclusion comes from its finite
fixed-width attacks, not merely failure of the stronger 169-bit policy.

The completed reference discovery runs at pinned lattice-estimator revision
`c667a48546f140c3a5454c7503c3ca44a264cce2` used the same quantum ADPS16,
LGSA, infinity norm and scalarization. Rust's quantum local-minimum
cross-check reproduces 126.14 bits/beta 476 at rank two, eta 1,506,960,
widths 4096 and 8192; 212.795 bits/beta 803 at rank three with the same
bound and widths; and 197.425 bits/beta 745 at rank three, eta 2,522,520,
width 4096. These discovery values agree with the certified lower bounds,
but are not certificates. The reference chose zeta 1; the finite Rust
rank-two certificate chooses zeta 0 with the same bits and beta. The
quantum reduction exponent is 0.265 per block dimension; using the default
classical exponent 0.292 would price a different model.

**The structured-ring assumption remains open.** The estimator prices
module/ring SIS for the degree-648 trinomial ring as unstructured SIS over
`Z_q`. It does not assess attacks using the ring structure. These
certificates establish parameter support in that model, not full protocol
security, extractor-time admission, or whole-schedule admission.

Regenerate both q24 artifacts, with cells in parallel, using:

```bash
rtk cargo run -p akita-sis-estimator --release --no-default-features \
  --features labinius-sis,parallel --example q24_infinity_width_table
```

The optional `--output-dir PATH` chooses a comparison directory.
The `q24_width_table` integration tests regenerate both CSV files byte for
byte and compare quantum discovery with the five reference cells. Without
`parallel`, the same generation and tests run serially. Offline lookup uses
`q24_width_table::certified_rows` or `certified_max_width`.
