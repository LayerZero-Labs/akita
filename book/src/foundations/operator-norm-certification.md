# Operator-norm certification

> **Status:** stub. Part of the initial Akita Book scaffold.

The Euclidean security model prices the folded response by an operator-norm cap
\\( \Gamma(c) \le \Gamma \\) on every accepted folding challenge (see
[rings and fields](./rings-and-fields.md#the-operator-norm-of-a-ring-element)).
Enforcing that cap inside Fiat-Shamir rejection sampling is delicate, because
\\( \Gamma(c) \\) is a real-analytic singular value but the accept/reject
decision must be reproduced bit-for-bit by the verifier. This page folds from
paper appendix C: the deterministic, integer-only predicate and the family-level
acceptance floor.

## Why a deterministic predicate

A floating-point FFT is inadmissible as the source of truth: IEEE results are
not reproducible across platforms, and a single prover/verifier disagreement on
acceptance desynchronizes the transcript. The predicate must be integer-only,
consume transcript randomness identically, and never accept a challenge with
\\( \Gamma(c) > \Gamma \\).

**Sources to fold in**

- Paper App C `sec:opnorm-certification`, `sec:opnorm-problem`.

## The integer enclosure and acceptance predicate

Fixed-point cosine/sine tables (certified once, offline, via a Machin
\\( \pi \\)-enclosure + interval Taylor + outward rounding) give exact integer
accumulators \\( R_k, I_k \\) that enclose the spectrum. The predicate compares
each frequency's enclosure against \\( \Gamma^2 2^{2q} \\) and returns
Accept / Reject / Indeterminate; the strict predicate treats Indeterminate as a
rejection and is sound up to a thin certified band.

**Sources to fold in**

- Paper App C `sec:opnorm-accumulators` (`lem:opnorm-enclosure`), `sec:opnorm-tables`, `sec:opnorm-predicate` (`proc:opnorm-decide`, `thm:opnorm-sound`).
- Paper App C `sec:opnorm-params` (the \\( q=48 \\), 128-bit scale window) and the rational-LDL ground-truth oracle.
- `crates/akita-challenges/src/sampler/op_norm.rs`.

## The accepted-support floor

A family-level obligation: after rejection, the accepted distribution must
retain enough min-entropy to keep the challenge set \\( \ge \lambda \\) bits. The
moment method (exact rational moments → dual majorant → Bernstein
nonnegativity certificate → union bound) discharges it; worked for the
\\( d=64 \\) shell at \\( \Gamma=18 \\).

**Sources to fold in**

- Paper App C `sec:opnorm-support`, `sec:opnorm-moment-method` (`thm:opnorm-floor`), `sec:opnorm-worked`.
- `experiments/operator-norm-acceptance/` (ancillary certificate + checker, in the paper repo).
- `specs/fold-linf-rejection.md` (interim op-norm rejection product scope).
