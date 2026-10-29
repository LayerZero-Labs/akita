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

A recursive application can defer direct Stage 2 setup evaluations and offloaded
Stage 3 input claims. The outer proof closes the complete collection together
with its own setup claim. A direct outer fold uses one combined setup scan;
an offloaded outer fold uses its existing degree-two Stage 3. Neither path
requires a standalone sumcheck. Public operations live in `akita-prover` and
`akita-verifier`; `akita-pcs` forwards the optional feature.

## Protocol

The exact polynomial identity binds the public matrix seed and derivation,
base modulus, canonical extension-field degree/encoding, actual power-of-two
flat prefix length, and coefficient-first LSB ordering. It does not include the
rectangular split, so compatible accumulators can be inherited with another
split. Extension fields use Akita's canonical `FpExtEncoding` contract.

For query i, the candidate answer is c_i = sum S(x) W_i(x). A contribution
uses the raw direct-scan result. Quotient lifting retains its common alpha
factor outside this query. Reduced evaluation uses the exact native terminal
residue kernel. Physical B slicing and overlapping role contributions remain
additive on the same flat setup addresses.

The complete inherited statement is bound before the outer opening transcript.
The outer fold sends its own scalar, then binds the combined ordered statement
before independent batching coefficients lambda_i are sampled. Stage 2 uses
only the outer scalar. Direct closure checks the combined equation in one setup
scan. Offloaded closure uses the existing Stage 3 transcript and successor. Every term of query i inherits
the same lambda_i. A query is represented by native coefficient factors and
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

- `external-setup-accumulation` is off by default in the types, prover,
  verifier, CPU backend, and PCS crates. Enabling it never selects deferred
  verification, modifies schedules, or changes ordinary proof grammar.
- Relation preparation owns the query before setup traversal. Reduced
  structured contractions share the existing arithmetic without requiring
  direct-scan segment construction.
- Native advice preparation returns backend-owned reusable coefficient
  contractions. Owner, domain, query encoding, split and answer are checked
  before consuming that state in a batch.
- Retain mode keeps structured index factors; regenerate mode releases them
  between advice preparation and the index phase. The integrated outer prover
  compiles all inherited and outer queries before one coefficient-contraction
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
- Batch statements reject empty batches, more than 65536 claims and invalid
  coefficient splits. A nontrivial domain requires at least two coefficient
  entries so its contracted index state is smaller than the full setup.
- Decoding accumulators uses an expected polynomial context and bounded point
  arity. Batch replay rejects incomplete arguments and trailing bytes.
- A direct decider scans the authenticated setup with split equality weights.
  Commitment closure uses `SetupAccumulator::prefix_opening_claim` to resolve
  the trusted prefix commitment, then the ordinary grouped opening APIs with a
  compatible admitted schedule and retained prefix handle. A final schedule
  does not automatically authenticate an externally carried accumulator.

## Scope

Conditional inner verification preserves internal Stage 3 prefix consumption.
It replays the Stage 3 transcript and grinding checks but omits its algebra.
It exports the original input functional and answer, not the unchecked endpoint.
Direct folds obtain their exact setup answer from advice and omit setup weight
evaluation. An outer proof can itself defer its combined collection; in that
case the original inherited and outer claims are carried forward.

`SelectedProverOpeningData::with_setup_batch` selects one nonterminal outer fold.
With the feature enabled, `AkitaVerifier::batched_verify` takes an optional
mutable `SetupClaimAccumulator` and a mutable `SetupMode`. `Defer` leaves the
original claims in the accumulator; `Check` closes a bound batch. Ordinary
verification passes `None` and `&mut SetupMode::Check`. `SetupClaimAccumulator::from_batch` selects the matching fold and
keeps its bound input fixed while other folds contribute pending claims.
The offloaded path requires the batch domain to match the selected successor's
actual prefix. It does not change the admitted schedule or add a sumcheck.
The standalone reduction helpers remain optional utilities. They are not part
of the integrated outer flow. SDK preparation, guest epilogues, authenticated
execution-output envelopes and guest-cycle measurements belong to Jolt integration.

The feature targets `dev`; shared math moves and behavior-preserving verifier
seams can be reviewed independently for `main`. No schedule artifact changes
are needed. Broader security certification and performance acceptance remain
review gates rather than claims inferred from implementation.

## Evaluation

- [x] Existing direct proofs pass complete and conditional verification.
- [x] Lifted/reduced mixed-role weights match direct scans and dense interpolation.
- [x] Singleton product-sumcheck rounds match an independent dense reference in base and extension fields.
- [x] Base/extension fields and retain/regenerate policies agree.
- [x] Chaining, domain embedding, accumulator transport, tampering and final rejection pass.
- [x] Feature-off behavior and existing deterministic proof bytes are preserved.
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
matched for all four cases with the extension disabled and enabled.

One local release/parallel fp128 microbenchmark used 65536 setup coefficients
and a 64-coefficient split. For 32 evaluation claims, retained preparation plus
separate reductions took 54.4 ms versus 19.8 ms for one shared reduction; argument
bytes were 16896 versus 528. Regeneration took 50.1 ms versus 20.8 ms. These are
single-run native measurements, not guest-cycle measurements or fixed regression
thresholds. Run the ignored profiling test in the CPU backend's `setup_batch`
integration target to reproduce the scenario.
