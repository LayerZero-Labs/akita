# Spec: External setup accumulation

| Field | Value |
| --- | --- |
| Author(s) | Akita maintainers and Codex |
| Created | 2026-10-20 |
| Status | implemented |
| PR | Unpublished implementation branch |
| Supersedes | None |
| Superseded-by | None |
| Book-chapter | how/recursion.md |

## Summary

A recursive application can defer direct Stage 2 setup evaluations. Both
verification modes check ordinary and accumulated Stage 3 reductions completely.
A direct outer fold handles inherited claims in Stage 2 and evaluates its own
setup contribution through ordinary Stage 2 verification. An offloaded outer
fold closes the collection together with its own setup claim using its existing
degree-two Stage 3. Neither path requires a standalone sumcheck. Public
operations live in `akita-prover` and
`akita-verifier` and require no Cargo feature.

## Protocol

The exact polynomial identity binds the public matrix seed and derivation,
base modulus, canonical extension-field degree/encoding, actual power-of-two
flat prefix length, and coefficient-first LSB ordering. It does not include the
rectangular split, so compatible accumulators can be inherited with another
split. Stream identity is independent of the verifier's materialized matrix:
offloaded prefixes may exceed that matrix and are authenticated by their
installed commitments. Direct deciders and backend preparation enforce
`SetupPolynomialId::validate_materialized` before compiling or scanning queries.
Extension fields use Akita's canonical `FpExtEncoding` contract.

For query i, the candidate answer is c_i = sum S(x) W_i(x). A contribution
uses the raw direct-scan result. Quotient lifting retains its common alpha
factor outside this query. Reduced evaluation uses the exact native terminal
residue kernel. Physical B slicing and overlapping role contributions remain
additive on the same flat setup addresses.

The complete inherited statement is bound before the outer opening transcript.
At a direct fold, Stage 2 samples coefficients for the inherited claims only.
`Check` checks their weighted equation with a setup scan and computes the fold's
own setup contribution normally. `Defer` preserves the inherited claims and
obtains the current contribution from advice, exporting its exact query and
answer. The prover sends no candidate setup answer and creates no Stage 3 state
or transcript events for a direct fold.

An offloaded fold sends its own scalar, then binds the combined ordered
statement before independent batching coefficients lambda_i are sampled.
Stage 2 uses only the outer scalar. Offloaded closure uses the existing
Stage 3 transcript and successor. Every term of query i inherits the same
lambda_i. A query is represented by native coefficient factors and
paired-equality index tensors. The backend compiles these into rectangular
terms A_a(y) B_a(k), using lane views when native dimensions differ.

First form T_a(y) = sum_k S(k,y) B_a(k). The coefficient rounds prove
sum_y sum_a lambda_i(a) T_a(y) A_a(y), sending one degree-two polynomial per
round and sharing every challenge. At the resulting coefficient point r_y,
merge B_batch(k) = sum_a lambda_i(a) A_a(r_y) B_a(k). Contract the actual
setup once into S_ry(k) = sum_y S(k,y) eq(r_y,y), then prove one index-product
sumcheck. Terminal verification checks output = s * sum_i lambda_i W_i(r).
The existing successor checks S(r) = s. A standalone reduction, when explicitly
requested, instead returns this condition to its caller.

For interactive uniform field challenges, random batching contributes at most
1/|E| error for a fixed false vector of claims, and degree-two sumcheck adds
at most 2n/|E| per reduction. Recursive composition must account for every
layer and the underlying execution proof. These interactive bounds alone do
not establish Fiat-Shamir accumulation security in the quantum random oracle
model; that composition requires protocol review before production deployment.

## Implementation and invariants

- Setup accumulation is unconditional in the types, prover, verifier, CPU
  backend and PCS crates. `SetupMode` is a plain `Check`/`Defer` enum. The mode
  changes only direct setup verification policy, never proof grammar or schedules.
- Relation preparation owns the query before setup traversal. Reduced
  structured contractions share the existing arithmetic without requiring
  direct-scan segment construction.
- Native advice preparation returns backend-owned reusable coefficient
  contractions. Owner, domain, query encoding, split and answer are checked
  before consuming that state in a batch.
- Retain mode keeps structured index factors; regenerate mode releases them
  between advice preparation and the index phase. For offloaded folds, the
  integrated outer prover compiles all inherited and outer queries before one coefficient-contraction
  traversal shared by every available factor. Parallel builds fuse row updates;
  sequential builds reuse bounded setup blocks with the singleton contraction
  kernel so they retain its small arithmetic working set.
  One further shared setup contraction follows the coefficient challenges;
  neither traversal is repeated per claim. Sequential advice discovery remains
  supported by the standalone API.
- Ordinary and accumulated Stage 3 share one preparation function, proving loop
  and product-sumcheck kernel. The ordinary case supplies one term with
  coefficient one and retains its existing transcript bindings. Preparation
  owns the combined statement; the backend retains only the prepared arithmetic
  state and borrows that statement when binding coefficients, without cloning it.
- The original rectangular contraction, base-by-extension products, delayed
  reduction, direct scans and terminal NTT paths are retained. There is no
  production coefficient-by-index weight table or extension-field setup copy.
- Setup coefficients after a query's support remain actual public coefficients.
  Query support alone is zero-extended. Smaller prefix evaluations embed with
  zero selectors on added high coordinates.
- Batch statements accept empty input, and reject more than 65536 claims and
  invalid coefficient splits. An empty inherited input uses the ordinary singleton
  Stage 3 encoding. A nontrivial domain requires at least two coefficient
  entries so its contracted index state is smaller than the full setup.
- Decoding accumulators uses an expected polynomial context and bounded point
  arity. Batch replay rejects incomplete arguments and trailing bytes.
- A direct decider scans the authenticated setup with split equality weights.
  Commitment closure uses `SetupAccumulator::prefix_opening_claim` to resolve
  the trusted prefix commitment, then the ordinary grouped opening APIs with a
  compatible admitted schedule and retained prefix handle. A final schedule
  does not automatically authenticate an externally carried accumulator.

## Scope

Conditional inner verification defers only direct setup evaluations. Direct
folds obtain candidate answers from advice attached to their accumulator and
check their Stage 2 endpoint with those answers. They export exact functionals
and values without scanning the setup. Both modes verify ordinary and batched
Stage 3 sumcheck rounds, evaluate the combined public weight, check the terminal
product equation, and retain the ordinary successor setup-prefix opening.
Checked offloaded input claims are consumed in both modes. A later direct
successor may export new obligations, so a recursive outer proof's pending
output depends on its own direct folds rather than its inherited history.
Direct deferred folds preserve inherited inputs until a checked reduction or
direct discharge consumes them.

`SelectedProverOpeningData::with_setup_batch` selects one nonterminal fold.
`AkitaVerifier::batched_verify` takes an optional mutable `SetupClaimAccumulator`
and a `SetupMode` value. Ordinary verification passes `None` and
`SetupMode::Check`. `SetupClaimAccumulator::new(...).with_advice(...)` starts
collection for archived direct folds; `from_batch` attaches bound inputs to
the same selected fold as the prover. Empty inputs select ordinary proof grammar.
All other Akita checks and the ordinary terminal checks remain enabled.
The offloaded path requires the batch domain to match the selected successor's
actual prefix. It does not change the admitted schedule or add a sumcheck.
The standalone reduction helpers remain optional utilities. They are not part
of the integrated outer flow. SDK preparation, guest epilogues, authenticated
execution-output envelopes and guest-cycle measurements belong to Jolt integration.

No schedule artifact changes are needed. Broader security certification and
performance acceptance remain review gates rather than claims inferred from
implementation.

## Evaluation

- [x] Existing direct proofs pass complete and conditional verification.
- [x] Lifted/reduced mixed-role weights match direct scans and dense interpolation.
- [x] Singleton product-sumcheck rounds match an independent dense reference in base and extension fields.
- [x] Base/extension fields and retain/regenerate policies agree.
- [x] Chaining, domain embedding, accumulator transport, tampering and final rejection pass.
- [x] Empty inherited input preserves ordinary deterministic proof bytes.
- [x] Recursive deferral consumes inherited batches and exports only current direct-fold claims.
- [x] Cheap gates, scoped tests, Clippy, and ordinary CI configurations pass.
- [x] Native benchmarks include preparation and equivalent separate reductions.

The small dense test references are confined to test code. The ignored
`profile_setup_batch_preparation_and_shared_rounds` test reports total native
cost and argument bytes for separate and shared reductions. It does not claim
to measure application guest cycles or complete recursive execution proving.

## Earlier standalone implementation validation

Both CI shards passed (2001 tests). All three standard CI Clippy feature graphs
and parallel/sequential extension Clippy checks passed. All 23 schedule artifacts
regenerated without byte changes. The public integration tests cover base-field,
fp64 extension and fp32 extension arithmetic, identical retained/regenerated
arguments, mixed inherited/new obligations, zero weights, and zero-round domains.

A separate untouched-main comparison used seed `[37; 32]`, constant-one sources,
Lagrange points filled with 2 and session `external-setup-byte-compat`. The cases
were fp128 direct nv14, fp128 recursive nv20, fp32 direct nv20, and fp32 recursive
nv24 (including actual setup offloading). Setup, commitment and proof bytes
matched for all four cases in the earlier implementation with accumulation support disabled and enabled.

One local release/parallel fp128 microbenchmark used 65536 setup coefficients
and a 64-coefficient split. For 32 evaluation claims, retained preparation plus
separate reductions took 54.4 ms versus 19.8 ms for one shared reduction; argument
bytes were 16896 versus 528. Regeneration took 50.1 ms versus 20.8 ms. These are
single-run native measurements, not guest-cycle measurements or fixed regression
thresholds. Run the ignored profiling test in the CPU backend's `setup_batch`
integration target to reproduce the scenario.
