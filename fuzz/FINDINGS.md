# Findings

Findings from development and validation runs of this campaign. Each entry
states what was observed, why it is (or is not) a defect, and how the
harness treats it. No finding here was hidden by weakening an assertion: where
a harness expectation was wrong it is recorded as a harness correction, and
where production behavior disagrees with its documentation both are stated.

Severity: **High** (soundness, completeness, or memory safety), **Medium**
(public-boundary misbehavior), **Low** (documentation, error classification,
or unreachable API corners).

## F-1 (Low, documentation): `EqPolynomial::evals_cached` returns suffix tables

`crates/akita-algebra/src/eq_poly.rs` documents
`result[j][x] = eq(r[..j], x)` (prefixes). The implementation builds layer
`j + 1` from `r[n - 1 - j]`, so `result[j] = eq(r[n-j..], ·)` (suffixes).
`result[n] == evals(r)` holds either way, which is all the existing unit test
checks. The only production caller, `GruenSplitEq::with_initial_scalar`,
documents and depends on suffix semantics, so behavior is correct and the
doc comment is wrong.

Found by `multilinear` (counterexample: one Boolean coordinate `r = [1]`,
expected prefix table `[0, 1]`, actual `[1, 0]` for `result[1]`, which is
`eq(r[1..], ·)` padded). The target now asserts the relied-upon suffix
semantics. Suggested fix: correct the doc comment.

## F-2 (Low, unreachable API corner): `log_basis = 1` over-width decomposition is incorrect

`BalancedDecomposePow2Params::new(levels, 1, q)` is accepted (and
`SignedDigitKernel::for_log_basis(1)` returns `I8`), but base-2 balanced
digits are `{-1, 0}` and cannot represent positive values. When
`levels * log_basis > field_bits` the centering threshold is `q/2`, positive
coefficients are not folded to `c - q`, and the dropped carry makes
recomposition wrong modulo `q`. Example (fp64, `q = 2^64 - 59`,
`levels = 65`): `524289` recomposes to `524171` (the result is correct
modulo `2^65`, off by `2 * 59`). At exactly the field width the threshold
trick folds every value and the result is correct.

No shipped configuration selects `log_basis < 3`
(`opening_basis_range`/`inner_basis_range` start at 3), so production proofs
are unaffected. The `decompose` target now draws bases from the configs'
ranges. Suggested fix: reject `log_basis < 2` in
`BalancedDecomposePow2Params::new`, or fold positive values at over-width
depth for base 2.

## F-3 (Low, error classification): prover-side shape errors report `InvalidProof`

`ProverOpeningData::from_parts` (`crates/akita-prover/src/opening.rs`) and
several checks in `crates/akita-types/src/opening_claims.rs` return
`AkitaError::InvalidProof` when prover-supplied claims do not match the
layout (for example a point longer than the group's variable count), before
any proof exists. `AkitaError::InvalidProof` is documented as "Proof
verification failed" and `InvalidPointDimension` exists for this case. The
checks are shared with verifier paths, where `InvalidProof` is the intended
class, so this is a consistent convention rather than a slip, but a prover
caller cannot distinguish "bad request" from "verification failure".

The `prover_boundary` target accepts `InvalidProof` for these shape errors
and documents why; it still requires that the request fails before a proof
is produced.

## Harness corrections (not Akita defects)

- **Cross-class precommitted groups.** The one-hot backend rejects a dense
  source even in explicit mode (`InvalidInput: committed source is not a unit
  one-hot representation…`). This is the documented class check. The harness
  now commits such groups on the owning dense family's backend and transfers
  them with `import_commitment`, as `tests/akita_fp128_e2e/heterogeneous.rs`
  does.
- **Uncataloged shapes vs setup capacity.** A probe that exceeded the
  prepared setup's polynomial capacity was correctly rejected with
  `InvalidInput` before the schedule lookup. The probe now stays within
  capacity (expecting `UnsupportedSchedule`) and capacity overflow is a
  separate check.
- **`basis_weights_prefix(point, basis, 0)`** is deliberately rejected with
  `InvalidSize`; the target now checks both the zero and over-capacity
  boundaries.
