# NTT, CRT, and fast ring arithmetic

> **Status:** stub. Part of the initial Akita Book scaffold.

How ring multiplication in \\( R_q \\) is actually computed: the CRT+NTT
double-transform domain, the pseudo-Mersenne base fields the digits reduce into,
the smooth-subgroup FFT, and the accumulation-capacity bound that governs
chunking. This is the bridge from the §2.1 algebra to the hot prover loops, and
folds from implementation appendix B.1-B.2.

## Pseudo-Mersenne fields and Solinas reduction

Every prime field is \\( p = 2^k - c \\) with small \\( c \\); elements are
canonical integers in \\( [0,p) \\) (no Montgomery form), enabling branchless
add/sub and a two-fold Solinas reduction. Includes the prime table across bit
widths (16/31/32/64/128) and the fused multiply-add.

**Sources to fold in**

- Paper App B.1.1-B.1.2 (`alg:akita-solinas`, `tab:akita-primes`, prime selection).
- `crates/akita-field/src/prime/` (`fp32.rs`, `fp64.rs`, `fp128/`, `pseudo_mersenne.rs`).

## Deferred reduction and balanced digits

Unreduced product and linear accumulators (lane-wise, carry-free), and the
signed balanced-digit extraction that feeds the commitment mat-vec.

**Sources to fold in**

- Paper App B.1.3 (`sec:akita-deferred-acc`), B.1.3 `sec:akita-balanced-digits`.
- `crates/akita-field/src/unreduced/`, `crates/akita-algebra/src/ring/cyclotomic/decomposition.rs`.

## CRT+NTT representation

Representing \\( a \in R_q \\) by negacyclic NTT evaluations modulo several
NTT-friendly primes, pointwise multiply, inverse NTT + Garner reconstruction,
and the per-profile CRT prime choices (Q16/Q32/Q64/Q128). The DIF-forward /
DIT-inverse pairing that avoids bit-reversal.

**Sources to fold in**

- Paper App B.2.1-B.2.2 (`sec:akita-crt-profiles`, `tab:akita-crt-profiles`).
- `crates/akita-algebra/src/ring/crt_ntt_repr.rs`, `ntt/`.
- `specs/crt-ntt-prime-profiles.md`.

## Accumulation capacity and chunking

The safe accumulation width \\( n_{\mathrm{safe}} \\) before a pointwise
accumulator can overflow the CRT range, and capacity-aware chunking with
intermediate Garner reconstruction.

For a matrix width `W`, ring degree `D`, centered matrix coefficients bounded
by `floor(q/2)`, and signed RHS coefficients bounded by `B`, centered CRT
reconstruction is unique exactly under the implemented strict bound

```text
2 * W * D * floor(q/2) * B < product(CRT primes).
```

Balanced base `2^L` digits have `B = 2^(L-1)`. Bases through 8 use i8. The
large-basis arithmetic path uses i16 for bases 9 through 16. The exact cache
selector receives the coefficient bound directly as `rhs_abs_bound`; it does
not rederive the bound from a basis label.

On portable, AVX2, and NEON hosts, exact caches use the field-selected 30-bit
i32 profile and append a 12289 residue only when that profile fails the bound.
On AVX-512IFMA hosts at D64 through D512, exact caches may instead use canonical
50-bit u64 residues: one limb plus the tail for Q32, two limbs for Q64, and
three limbs plus the tail when needed for Q128. The prepared IFMA matrix owns
the CRT parameters used to create it, so prime order, twiddles, reconstruction
constants, and execution cannot diverge. All representations use the same
strict capacity inequality and centered Garner reconstruction. Cache layout
and byte count are therefore host-dependent derived state; proof and setup
bytes are unchanged.

Prover caches store cyclic and negacyclic transforms in separate exact-prefix
slots. Commitment kernels request only negacyclic entries. Ring-switch kernels
join their D/B/A cyclic requirements by maximum and request the A negacyclic
prefix separately. `ExactNegacyclic { width, log_basis }` prepares the terminal
verifier's minimum exact negacyclic form: base residues alone when they fit,
otherwise the selected base plus the 12289 tail. Verifier warming coalesces terminal
groups into one strongest prefix per ring degree. The base prefix covers every
group, while the tail prefix covers only groups whose exact bound requires it.

**Sources to fold in**

- Paper App B.2.4 `sec:akita-crt-capacity`.
- `specs/crt-ntt-accumulation-safety.md`, `docs/crt-ntt-capacity-profile.md`.

## Smooth-subgroup mixed-radix FFT

Because pseudo-Mersenne moduli have no large power-of-two subgroup, Reed-Solomon
and interpolation use an iterative mixed-radix Cooley-Tukey DIT FFT over a smooth
subgroup, with low-multiplication Winograd radix-{2,3,5,7} kernels and coset
evaluation.

**Sources to fold in**

- Paper App B.1.5 `sec:akita-fft` (mixed-radix DIT, Winograd kernels, coset Reed-Solomon).
- `crates/akita-field/src/fft.rs`.
