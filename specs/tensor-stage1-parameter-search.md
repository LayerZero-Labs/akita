# Tensor Stage-1 Parameter Search

## Goal

Re-run the tensor stage-1 challenge search so the default fp128 challenge
families keep at least 128 bits of overall support while avoiding the old
"square the flat-side mass" configuration that inflated prover work.

The tensor sampler draws independent left/right sparse challenges, so the search
targets roughly 64 bits of support per side instead of preserving the old
128-bit flat-side mass on both sides.

## Runtime Model

The shipped code uses two different tensor metrics:

- Honest folding mass uses `omega^2`, i.e. the tensor product of the side L1
  masses.
- SIS extraction uses the conservative two-level CWSS proxy `4 * omega *
  ||c||_inf`, not `omega^2`.

That split is why "small side mass with small coefficient magnitudes" is the
right optimization target for tensor challenges.

## Search Script

The support search used to choose the defaults is kept in:

```bash
python3 scripts/tensor_stage1_param_search.py
```

It reports:

- the current defaults before this change,
- the absolute per-side minima at `>= 64` bits of support,
- exploratory bounded-L1 comparisons for `D=32` and `D=64`,
- the agreed safe defaults that add a small margin above the minima.

## Planner Results

The rerun produced four takeaways:

1. `D=64` and `D=128` do not need anything close to the old side masses.
   Exact-shell weight `18` for `D=64` and uniform `+-1` weight `13` for
   `D=128` already leave a real margin above the minimum.
2. `D=32` is the subtle case. Very light bounded-L1 candidates can beat the
   exact-shell candidates on honest L1 mass alone, but they lose once the
   extraction proxy is included.
3. A direct probe against the current schedule/audit logic showed that the
   low-mass `D=32` exact-shell candidates are not shippable under the existing
   audited ranges: they under-provision inner/outer ranks on some recursive
   levels. So the final shipped defaults keep `D=32` on `BoundedL1Norm`.
4. A "small but real" margin above the absolute minima keeps tensor efficient
   while avoiding a knife-edge choice.

## Current vs Minimum vs Safe

### Previous defaults

- `D=32`: `BoundedL1Norm` with side L1 `121`, `||c||_inf = 8`
  - tensor honest mass `14641`
  - extraction proxy `3872`
- `D=64`: `ExactShell { count_mag1: 30, count_mag2: 12 }`
  - side L1 `54`
  - tensor honest mass `2916`
  - extraction proxy `432`
- `D=128`: `Uniform { weight: 31, nonzero_coeffs: [-1, 1] }`
  - side L1 `31`
  - tensor honest mass `961`
  - extraction proxy `124`

### Absolute minima found at `>= 64` bits per side

- `D=32`: `ExactShell { count_mag1: 12, count_mag2: 8 }`
  - side support `64.693` bits
  - tensor honest mass `784`
  - extraction proxy `224`
- `D=64`: `ExactShell { count_mag1: 16, count_mag2: 0 }`
  - side support `64.795` bits
  - tensor honest mass `256`
  - extraction proxy `64`
- `D=128`: `Uniform { weight: 12, nonzero_coeffs: [-1, 1] }`
  - side support `66.397` bits
  - tensor honest mass `144`
  - extraction proxy `48`

### Final shipped defaults

- `D=32`: keep `BoundedL1Norm`
  - no lighter exact-shell candidate passed the current audited schedule checks
  - side L1 `121`
  - tensor honest mass `14641`
  - extraction proxy `3872`
- `D=64`: `ExactShell { count_mag1: 18, count_mag2: 0 }`
  - side support `69.678` bits
  - overall support about `139.355` bits
  - tensor honest mass `324`
  - extraction proxy `72`
- `D=128`: `Uniform { weight: 13, nonzero_coeffs: [-1, 1] }`
  - side support `70.555` bits
  - overall support about `141.110` bits
  - tensor honest mass `169`
  - extraction proxy `52`

These shipped defaults keep the large reductions for the representative
benchmark targets (`D=64` onehot and `D=128` full) while leaving `D=32` on the
existing bounded-L1 family until a lighter audited replacement is available.
