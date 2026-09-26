# Findings

Findings from development and validation runs of this campaign. Each entry
states what was observed, why it is (or is not) a defect, and how the
harness treats it. No finding here was hidden by weakening an assertion: where
a harness expectation was wrong it is recorded as a harness correction, and
where production behavior disagrees with its documentation both are stated.

Severity: **High** (soundness, completeness, or memory safety), **Medium**
(public-boundary misbehavior), **Low** (documentation, error classification,
or unreachable API corners).

## Validation runs

Host: 32-core x86_64 (AVX2, AVX-512 IFMA), Ubuntu 26.04, 121 GB RAM; campaign
budget 24 CPUs with hard per-worker cgroup memory limits; ASan build.

| Run | Duration | Result |
|---|---|---|
| 1 (build `e6de32ee`) | 30 min, stopped with `SIGTERM` | All startup baselines passed. Exposed runner bugs (slow-unit reports and an interrupted baseline treated as failures); fixed before run 2. |
| 2 (build `850f2a96`, resumed run 1's directory) | 1 h | Baselines passed for all 20 lanes. End-to-end lanes executed 1.2k–2.8k fresh cases each (`pcs_dense` 1241, `pcs_onehot` 2770, `pcs_batch` 1241, `pcs_reject` 1185, `prover_boundary` 1457) plus 357k `verifier_boundary` inputs; primitive lanes 1.6M–16M executions. Findings: F-4 (x526, three field variants) and F-5 (two sites, x18). No completeness, determinism, soundness-mutation, sanitizer, timeout, or memory findings. Zero startup or infrastructure failures. |
| Finding-pipeline check | 2 min, `pcs_dense` with an injected `-malloc_limit_mb=16` | OOM findings were recorded, deduplicated (x4), their inputs quarantined, and replayed in a fresh process ("not reproducible", as expected without the injected limit). `reproduce` and `export` worked on the result. |

Observation (not a defect): under ASan, proving gets slower as Rayon threads
increase (one `pcs_parallel` seed: 11 s at 1 thread, 23 s at 4), so the
parallel lane uses 2 internal threads.

## F-5 (Medium, robustness): schedule-artifact admission panics on a zero terminal `log_basis`

`TrustedScheduleCatalog::from_artifact_bytes` panics instead of returning an
error when a row's `schedule.terminal.inner.digits.log_basis` is `0`:

```
panicked at crates/akita-types/src/sis/decomposition_digits.rs:223:5: invalid log_basis
  compute_num_digits <- num_digits_for_bound <- audit_terminal
  <- audit_resolved_schedule <- ResolvedScheduleRow::try_new
  <- ValidatedScheduleCatalog::from_artifact_bytes
```

The terminal audit passes the artifact's `log_basis` to `num_digits_for_bound`
before any range check, and `compute_num_digits` asserts `0 < log_basis < 128`.
A second site of the same defect: `audit_committed_params` reaches
`compute_num_digits_field_width` (`decomposition_digits.rs:248`, same
assertion) with a zero `opening.log_basis_open` or `profile.outer.digits
.log_basis` in a recursive fold's group (found in `fp32_dense_recursive`).
Artifact admission is documented as the validating boundary for approved
parameter bytes ("Akita validates them before setup or proof work begins";
`crates/akita-config/tests/trusted_schedule_artifact.rs` checks rejection of
tampered artifacts), so a malformed artifact must yield an `AkitaError`, not
an abort. Artifacts are not proof-controlled, so this is not
verifier-reachable from a proof.

Found by `schedule_artifact` within minutes (three samples, every one with a
zero terminal `log_basis`; reproducible in a fresh process). Minimal
reproduction: `regressions/schedule_artifact/f5-terminal-log-basis-zero.bin`
is the shipped `fp128_dense.aks` with only `rows[0].schedule.terminal.inner
.digits.log_basis` set to `0` (leading selector byte `0` = `fp128_dense`):

```bash
cd fuzz && cargo run --release -p akita-fuzz-dev -- replay schedule_artifact \
  regressions/schedule_artifact/f5-terminal-log-basis-zero.bin
```

Suggested fix: validate every artifact-supplied `log_basis` (as
`GadgetDigits::validate` does) before computing digit counts in
`audit_terminal` and `audit_committed_params`, or make the digit-count
helpers return `Option`. Not fixed here: this change set is test infrastructure
only. Until it is fixed the `schedule_artifact` lane keeps rediscovering it;
occurrences only increment the finding's count.

## F-4 (Low, encoding): `CommittedGroup` decoding accepts non-canonical coefficient bounds

`CommittedGroup::deserialize_with_mode` (`crates/akita-types/src/proof/
commitment.rs`) reads each matrix's `coeff_linf_bound` and passes it to
`InnerCommitMatrixParams::try_new` / `OuterCommitMatrixParams::try_new`,
which resolve it through `sis_table_key_for_linf_bound` to the audited table
row and store that row's bound. Any stored bound that resolves to the same row
is accepted and silently replaced, so distinct byte strings decode to the same
commitment (observed: bytes 116..132, the inner bound, `0x0cbf33` re-encoded
as `0x0cbf34`, and larger rewrites) for all three fields.

The decoded object equals the canonical one, so proofs and transcript binding
(which use the decoded value) are unaffected; the defect is encoding
malleability. It matters to applications that identify or deduplicate
commitments by their serialized bytes, and it contrasts with the proof stream,
which rejects non-canonical atoms. Reproduction:
`regressions/public_deserialize/f4-committed-group-bound-normalized.bin`.
Suggested fix: reject a stored bound that differs from the resolved key's.

`public_deserialize` now counts these acceptances
(`noncanonical_committed_group` in `status`) and still requires the decoded
value to be a stable encode/decode fixed point; every other type keeps the
strict byte-for-byte check.

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
- **Full-width sources and the centering threshold.** An overnight run
  reported that `fp64_dense` accepted a "bounded" coefficient past
  `accepted_bounds`. Production is right: a schedule decomposing at exactly
  the field width centers at `decompose_centering_threshold`, which folds
  large values to their negative representative, so every field element is
  admissible and the commitment skips the bound check. The harness modeled
  the interval with `q/2` centering; it now uses production's threshold, so
  such profiles get the full domain and out-of-range probes stay on their own
  side of the threshold.
- **`basis_weights_prefix(point, basis, 0)`** is deliberately rejected with
  `InvalidSize`; the target now checks both the zero and over-capacity
  boundaries.
