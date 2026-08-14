# Operator-norm accepted-support certificates

This directory contains exact offline certificates for the sparse challenge
families used by operator-norm rejection sampling.

- `d64/` certifies the selective-L2 shell with 31 magnitude-one and 11
  magnitude-two coefficients for the strict runtime threshold 18.
- `d128/` certifies the production signed weight-31 shell for the strict runtime
  threshold 13.

The runtime predicates use 48-bit fixed-point roots. Each certificate accounts
for the root-table error and proves that a true-norm subset just below the
integer threshold is contained in the strict runtime predicate. The exact
certificates prove that these contained subsets have at least 128 bits of
support. They do not rely on Monte Carlo data.

Run both exact checkers through repository test discovery:

```bash
python3 -m unittest scripts.tests.test_operator_norm_certificates
```

The D128 directory also contains the modular moment generator, CRT
reconstruction, numerical dual search, rational certificate emitter, and a
small exhaustive validation of the moment generator. Floating point is used
only to search for a candidate dual polynomial. Both final checkers use exact
rational arithmetic for every load-bearing inequality.
