# Uniform D128 weight-35 challenge candidate

This directory certifies the support and multiplication norm of a challenge
candidate that uses all 128 coefficient positions:

```text
C_raw = {c in {-1, 0, 1}^128 : wt(c) = 35}.
```

The sampler draws uniformly from `C_raw` and rejects unless Akita's strict
q=48 fixed-point operator-norm predicate accepts at runtime threshold 14.
Every challenge has squared coefficient norm and L1 norm 35. Its raw support
has `139.669567620535` bits.

The exact degree-30 moment certificate proves

```text
q1 <= 0.007901145126365772
Pr[Gamma(c) < 14 - 396 / 2^48] >= 0.494326711912590588
log2(N_raw * p0) >= 138.653104393085
```

Thus the certified sampler takes at most `1 / p0 < 2.023` trials in
expectation. Compared with S72, its coefficient energy drops from 58 to 35,
its certified operator threshold drops from 18 to 14, and its support changes
from 72 selected positions to the full D128 shell.

Run the standalone checks from this directory:

```bash
python3 validate_moment_generator.py
python3 -u check_cert.py
python3 -u check_cert.py --check-bernstein
python3 analyze_candidate.py
```

The first command compiles the canonical D128 modular moment generator and
checks it against exhaustive enumeration of a small shell. `check_cert.py`
uses only the Python standard library. It verifies all eight modular residue
files, the exact moments, the rational dual expectation, Sturm positivity,
the q=48 containment margin, and the `2^136` support floor. The optional
Bernstein replay checks all 4096 exact subintervals.

## Remaining collision theorem

This is not yet a complete approximate-strong-sampling certificate. The
remaining obligation is much weaker than proving that every pairwise
difference is a unit.

Let `L_rho` be the BN254 evaluation kernel at one primitive 256th root. If

```text
min {||v||_2^2 : 0 != v in L_rho} >= 76,
```

then every evaluation fiber of the weight-35 shell contains at most 12
challenges. Indeed, distinct challenges `c_i,c_j` in one fiber have
`||c_i-c_j||_2^2 >= 76`, hence

```text
<c_i,c_j> = (35 + 35 - ||c_i-c_j||_2^2) / 2 <= -3.
```

For a fiber of size `m`, positivity of the squared norm of its sum gives

```text
0 <= ||sum_i c_i||_2^2 <= 35m - 3m(m-1) = m(38 - 3m),
```

so `m <= 12`. For any fixed anchor challenge, at most 11 other accepted
challenges collide at one root. A union bound over all 128 roots then gives

```text
epsilon_C <= 128 * 11 / |C| < 2^-128.194.
```

All roots have isometric evaluation lattices because the full shell is
Galois invariant. One exact shortest-vector lower bound through squared radius
75 would therefore complete the approximate-strong-sampling result.

This bounded-fiber argument is the important change from the S72 proof. S72
excludes the entire difference ball through squared radius 232 to obtain zero
collision probability. The uniform candidate only needs to exclude the much
smaller radius 75, and uses nine extra support bits to absorb a bounded number
of exceptional partners.

The Gaussian volume heuristic for the radius-75 ball is about `2^-45.252`
lattice points, so the target looks plausible, but this is not a proof. A
short BKZ-40 screen found no obstruction (its shortest displayed basis vector
had squared norm 504), while its basis profile still projects about `2^94`
nodes for an exact radius-75 enumeration. The remaining step therefore needs
substantially stronger reduction or a sharper exact lower-bound certificate;
the 30-second exploratory basis is not adequate.

This distinction also matters when comparing with PikkuFold's almost-splitting
heuristic. PikkuFold conjectures slotwise near-uniformity when the challenge
set contains at least as many elements as a slot. Here BN254 has roughly 254
bits while this shell has roughly 139, so that hypothesis does not apply. The
bounded-fiber reduction above is tailored to the sparse, sub-field-size
regime and isolates an exact statement that can be certified independently.

## Reproduction pipeline

`moments_mod.cpp` is the canonical generator in
`scripts/operator_norm/d128/`. Regenerate degree-30 residues with arguments
`PRIME GENERATOR 30 128 35` and the following pairs:

| prime | primitive generator |
|---:|---:|
| 2305843009213689601 | 11 |
| 2305843009213689089 | 3 |
| 2305843009213687297 | 15 |
| 2305843009213683713 | 3 |
| 2305843009213682689 | 17 |
| 2305843009213675777 | 11 |
| 2305843009213673729 | 3 |
| 2305843009213666049 | 3 |

Then run:

```bash
python3 reconstruct_moments.py
python3 solve_dual.py
python3 make_cert.py
python3 -u check_cert.py
```

Only `reconstruct_moments.py` and `solve_dual.py` need SymPy, NumPy, and
SciPy. Floating point is used only to find a candidate dual polynomial. The
emitted certificate is rational and the standalone checker replays every
load-bearing claim exactly.
