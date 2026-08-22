# Spec: HOL Light Proofs for Verifier Kernels

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-22 |
| Status | proposed |
| PR | [#436](https://github.com/LayerZero-Labs/akita/pull/436) |
| Supersedes | |
| Superseded-by | |
| Book-chapter | |

## Summary

Akita's verifier relies on a small group of low-level arithmetic routines. The
largest group implements exact ring matrix products with CRT and NTT arithmetic.
Other important routines implement deferred fp128 dot products, sparse challenge
sampling, and a fixed-point operator-norm test. These routines are tested against
scalar references, but most of them do not yet have a machine-checked proof for
the instructions that run in a production binary.

This specification defines a proof program for those routines. It uses HOL Light
and the processor models from s2n-bignum. Each proof starts from the exact bytes of
an object file. It states the mathematical result, the required input bounds, the
allowed memory changes, and the application binary interface. The build then
checks that the proved object is the object linked into the verifier binary.

This is not a proof of the whole Akita protocol. It covers the low-level kernels
that the protocol calls. The existing protocol argument, transcript rules, setup
validation, and security estimates remain separate obligations.

## Intent

### Goal

Build a reviewable and reproducible HOL Light proof boundary for every
performance-sensitive arithmetic kernel that can affect an Akita verifier result.

### Invariants

1. A proof must refer to the exact bytes that production links. A proof of a copy,
   handwritten transcription, test-only object, or compiler approximation is not
   a production proof.
2. A theorem must state the full mathematical result. A theorem about an internal
   carry, mask, residue, or butterfly is useful only as a lemma for the full
   operation theorem.
3. Every theorem must state input ranges, alignment and non-aliasing conditions,
   output ranges, allowed register changes, allowed flag changes, and allowed
   memory changes.
4. Every CRT theorem must prove both congruence and exact reconstruction. The
   capacity inequality must exclude wraparound in the integer result before the
   result is reduced into the protocol field.
5. Every deferred-reduction theorem must prove that the wide accumulator does not
   wrap for every supported dimension and operand bound.
6. Runtime dispatch must select only a proved object in the verified verifier
   profile. An unsupported processor must use a proved scalar object or fail with
   a typed error before verification starts.
7. The verified x86 profile must not select the AVX-512 IFMA52 exact cache path.
   The prover may keep that path. The verifier's arithmetic result, transcript,
   proof format, and accepted inputs must remain unchanged.
8. The x86 verified fast path uses AVX2. The AArch64 verified fast path uses NEON.
   A proved scalar path remains available for x86 machines without AVX2 and for
   supported targets without a proved vector path.
9. The proof build pins the HOL Light and s2n-bignum revisions. A revision change
   requires a clean proof run and review of the model diff.
10. The verifier no-panic contract remains in force. Proof-oriented refactors must
    not move validation after an unsafe call or replace typed errors with panics.
11. Proof-oriented refactors must not add release-mode checks inside hot loops.
    Preconditions are validated at the existing boundary or discharged by a
    caller theorem.
12. Prover and verifier challenge derivation must remain identical. No proof
    refactor may change transcript labels, absorbed bytes, draw order, rejection
    order, or domain separation.
13. Prepared cache bytes remain derived state. Their field identity, setup seed,
    schedule row, ring dimension, matrix prefix, CRT profile, and kernel profile
    must be bound and checked before use.
14. Performance claims require measurements on named hardware. A proof does not
    imply that a replacement kernel is fast enough.
15. Constant-time execution is not an acceptance criterion for this proof
    program. Verifier kernels may branch, index tables, or repeat public work
    based on public verifier inputs and public intermediate values. The proof
    report must not present functional correctness as a side-channel claim.

### Non-Goals

- This work does not prove the complete PCS or its Fiat-Shamir security theorem.
- This work does not formalize the planner, schedule generator, proof parser, or
  high-level verifier orchestration as machine code.
- This work does not add AVX-512 or IFMA52 semantics to the first verified
  production profile.
- This work does not remove AVX-512 code used by the prover or by benchmarks.
- This work does not treat differential tests as formal proofs.
- This work does not claim that XOF output is random. Sampling theorems are
  conditional on an input stream of independent uniform bits or bytes.
- This work does not prove the external lattice estimators. It proves only the
  exact integer and fixed-point predicates that consume their certified bounds.
- This work does not prove constant-time execution or make constant-time
  execution a requirement for verifier kernels. A dependency's separate
  constant-time claim remains separate from Akita's functional-correctness
  claim.
- This work does not duplicate fp128 add, subtract, or multiply proofs in Akita
  after Jolt becomes the production owner. Akita records those theorems as linked
  dependencies and proves the ring kernels that call them.

## Current verifier arithmetic

The following diagram shows the low-level part of the current path. A box does
not mean that the whole module belongs in one theorem. It identifies the points
where values cross a proof boundary.

```text
transcript seed
      |
      v
SHAKE256 cursor -> unbiased range draws -> partial Fisher-Yates -> signed sparse c
                                                               |
                                                               v
                                             fixed-point operator-norm predicate

public matrix A ------> prepared CRT and NTT cache
                              |
signed i16 vector z ----------+-> forward NTT -> pointwise dot -> inverse NTT
                                                       |               |
                                                       +------ CRT reconstruction
                                                                      |
                                                                      v
                                                        exact terminal A * z

ring coefficients + powers of alpha -> wide fp128 product sum -> one reduction
```

The main source paths are these.

| Surface | Current source |
|---|---|
| Terminal verifier matrix product | `crates/akita-verifier/src/protocol/core/terminal_ntt.rs` |
| Exact cache selection and reconstruction | `crates/akita-types/src/ntt_cache.rs`, `crates/akita-types/src/ntt_cache/exact.rs` |
| CRT and NTT representation | `crates/akita-algebra/src/ring/crt_ntt_repr/` |
| Scalar, AVX2, and NEON NTT operations | `crates/akita-algebra/src/ntt/` |
| AVX-512 IFMA52 exact route | `crates/akita-algebra/src/ntt/ifma52.rs`, `crates/akita-algebra/src/ring/ifma52.rs` |
| Deferred ring evaluation | `crates/akita-algebra/src/ring/eval.rs` |
| Sparse challenge sampling | `crates/akita-challenges/src/sampler/` |
| Operator-norm arithmetic | `crates/akita-challenges/src/sampler/op_norm.rs`, `crates/akita-challenges/src/sampler/op_norm_accumulate.rs` |
| Prepared verifier artifact | `crates/akita-verifier/src/prepared_cache.rs`, `crates/akita-types/src/ntt_cache/prepared_artifact.rs` |

The default fp128 presets use `Prime128OffsetA7F7`. The ring dimensions that may
appear are selected by the generated schedule. The proof manifest must therefore
be generated from the shipping schedules instead of assuming that one observed
schedule covers the whole supported verifier.

The ordinary fp128 exact route uses five roughly 30-bit i32 CRT primes. It adds
the 14-bit prime 12289 when the exact-capacity check requires a tail. The current
IFMA52 route uses three wider primes and may also add that tail. The verified x86
route keeps the five-prime representation because its AVX2 instruction family is
already close to the s2n-bignum proof surface.

## Processor support and dispatch

### What s2n-bignum already provides

The pinned s2n-bignum revision is
`ac31a43db30953037abd1b64b540e65cf31f4c67`. It contains complete HOL Light
proofs for AVX2 ML-DSA NTT and pointwise kernels, AArch64 NEON ML-KEM and ML-DSA
NTT kernels, and variable-time rejection samplers on both architectures. It
also contains the byte import, instruction simulation, ABI, memory framing, and
SIMD arithmetic support that Akita needs.

This is strong prior evidence, but it is not a proof of Akita. Akita uses
different primes, transforms, layouts, bounds, and instruction sequences.

### Known instruction gaps

The current Akita intrinsics mostly map to instructions already present in the
s2n-bignum models. A source and model audit found the following gaps at the
pinned revisions.

| Akita operation | Likely instruction | Current model result | Required action |
|---|---|---|---|
| AVX2 i16 lane shuffles | `VPSHUFHW`, `VPSHUFLW` | No instruction or decoder entry was found | Add and test semantics, or replace with an already modeled byte shuffle and benchmark it |
| AVX2 i8 to i16 widening | `VPMOVSXBW` | No instruction or decoder entry was found | Add semantics, or keep this prover-only path outside the first verifier target |
| NEON four-way deinterleaved load and store | `LD4`, `ST4` | The model has `LD2`, `LD3`, `ST2`, and `ST3`, but no four-register form was found | Add `LD4` and `ST4` semantics, decoding, and simulator tests, or rewrite the layout code |
| NEON halving subtract | `SHSUB` | No instruction or decoder entry was found | Add and test semantics, or express the reduction with modeled shifts and subtracts |
| AVX-512 IFMA52 multiply-add | EVEX `VPMADD52` family | No EVEX or IFMA52 instruction semantics were found | Keep it outside the verified verifier profile |

`VPMOVSXBD`, AVX2 vector multiply, add, subtract, compare, blend, permute,
logical operations, and vector shifts are present. The AArch64 model already
contains the ordinary NEON add, subtract, multiply, widening multiply, narrowing,
compare, maximum reduction, load, and store operations used by most Akita code.

The implementation must audit the final disassembly of every object. Intrinsics
are not a stable instruction contract. A compiler may choose a different but
equivalent instruction after an upgrade.

### Verified verifier policy

The current ordinary x86 NTT plan already chooses AVX2 or scalar. AVX-512 is
selected separately by the exact IFMA52 cache plan. The proof-oriented change
must make that distinction explicit at the verifier boundary.

Introduce one verifier kernel policy with these production choices.

| Target | Proved fast path | Proved fallback | Forbidden in verified mode |
|---|---|---|---|
| x86-64 with AVX2 | AVX2 | scalar | AVX-512 IFMA52 and unproved compiler output |
| x86-64 without AVX2 | scalar | none | AVX2 and AVX-512 |
| AArch64 | NEON | scalar where retained | unproved compiler output |
| RISC-V verifier profile | scalar | none | host-prepared data without a verified binding |

The policy applies only to verifier-reachable cache construction and execution.
It must not use a process-wide environment variable as the security boundary.
Tests may still request scalar execution. Production constructs the policy once,
records the selected object identity, and uses it for every later call.

If every supported x86 deployment is willing to require AVX2, the scalar row may
be removed in a later spec. This specification keeps it because removing older
x86 support would reduce the current feature surface.

## Proof claim structure

A production claim has six layers. The project must report the highest complete
layer for each target instead of saying only that a routine is verified.

| Layer | Claim |
|---|---|
| 1. Mathematical | The algorithm implements the stated integer, field, ring, transform, sampler, or norm function |
| 2. Machine body | Starting from stated registers and memory, the exact instruction body returns the mathematical result and changes only its frame |
| 3. ABI subroutine | The complete callable symbol follows the platform ABI, preserves callee-saved state, returns correctly, and permits the stated aliasing |
| 4. Object identity | `define_assert_from_elf` or the matching object importer checks the theorem bytes against the built object |
| 5. Final linkage | The production binary contains the same symbol bytes and all verifier call sites resolve to that symbol |
| 6. Dispatch coverage | Every supported verifier configuration selects a symbol with layers 1 through 5, or fails before processing a proof |

A helper lemma about one vector lane completes none of these layers by itself.
It becomes production evidence only after it is composed into the symbol theorem.

## Target registry

The proof tree must contain a machine-readable manifest. Each row records its
architectures, object names, source revision, theorem names, input bounds,
supported ring dimensions, CRT primes, compiler or assembler, and completion
through the six claim layers.

### Tier 0 dependencies

| ID | Target | Required result |
|---|---|---|
| `F32-CORE` | `Prime32Offset99` add, subtract, negate, multiply, square, and inverse | Canonical field results for every canonical input, with inverse specified only for nonzero input |
| `F32-EXT4` | the fp32 degree-four extension operations used by the verifier | The extension result agrees with the defining polynomial over `Prime32Offset99` |
| `F64-CORE` | `Prime64Offset59` add, subtract, negate, multiply, square, and inverse | Canonical field results for every canonical input, including every selected widening-multiply path |
| `F64-EXT2` | the fp64 quadratic extension operations used by the verifier | The extension result agrees with the defining polynomial over `Prime64Offset59` |
| `F128-ADD` | `Prime128OffsetA7F7` addition | Canonical `(a + b) mod p` for canonical inputs |
| `F128-SUB` | `Prime128OffsetA7F7` subtraction | Canonical `(a - b) mod p` for canonical inputs |
| `F128-MUL` | `Prime128OffsetA7F7` multiplication | Canonical `(a * b) mod p` for canonical inputs |
| `F128-MULADD` | multiply-add and reduction helpers used below | The exact modular sum with stated product-count bounds |
| `HASH-XOF-*` | the selected transcript hash and SHAKE256 expansion | The linked object implements the named hash or XOF byte function |

These proofs belong with the production field implementation. If Jolt owns the
implementation, Akita pins and checks the linked Jolt objects and theorem bundle.
The closed Akita prototype in PR #433 is evidence about the method, not a proof
of a later Jolt binary.

`F32-CORE`, `F32-EXT4`, `F64-CORE`, and `F64-EXT2` are required before Akita
makes a full low-level claim for those optional presets. The first production
milestone remains fp128. The generated manifest must keep the other presets
visible instead of silently counting the fp128 result as workspace-wide proof.

`HASH-XOF-*` is a named dependency rather than an NTT target. The manifest records
the transcript feature used by each binary. A challenge-sampling claim stops at
deterministic input bytes until the selected hash and XOF objects have their own
proof and final linkage evidence.

### Tier 1 exact ring arithmetic

| ID | Target | Mathematical contract |
|---|---|---|
| `NTT-MONT-I32` | i32 Montgomery conversion and reduction | Each lane represents the same integer modulo its CRT prime and stays in the next kernel's range |
| `NTT-MONT-I16` | i16 tail Montgomery conversion and reduction | Same contract for the tail prime |
| `NTT-CENTER-I16-I32` | centered signed input conversion | Each signed i16 coefficient maps to the correct Montgomery residue in every i32 limb |
| `NTT-CENTER-I16-I16` | centered signed tail conversion | Each signed i16 coefficient maps to the correct Montgomery residue modulo 12289 |
| `NTT-ADD-SUB-I32` | vector add, subtract, negate, and range reduction | Each lane has the stated residue and the numeric range required by the next operation |
| `NTT-ADD-SUB-I16` | i16 add and range reduction | Same contract for the tail limb |
| `NTT-FWD-I32-D*` | negacyclic forward NTT | Output is the selected negacyclic transform in the documented order and Montgomery scale |
| `NTT-INV-I32-D*` | negacyclic inverse NTT | Output is the inverse transform with the stated scale removed |
| `NTT-FWD-I16-D*` | tail-prime forward NTT | Same transform contract for i16 lanes |
| `NTT-INV-I16-D*` | tail-prime inverse NTT | Same inverse contract for i16 lanes |
| `NTT-DOT-I32` | prepared pointwise dot | Each output lane is the modular sum of all row products, with no native overflow |
| `NTT-DOT-I16` | tail-prime pointwise dot | Same contract for the tail limb |
| `NTT-PREPARE-D*` | prepared matrix transform | Every cached row is the stated transform of the bound public-matrix row |
| `CRT-GARNER` | mixed-radix reconstruction | Residues reconstruct the unique centered integer inside the proved capacity interval |
| `CRT-TAIL` | wide profile plus i16 tail | The extra residue extends capacity and is combined with the same centered integer |
| `CRT-MATVEC-D*` | full prepared matrix product | Output equals exact negacyclic integer matrix multiplication followed by reduction into `F` |

`D*` means every ring dimension reachable in a shipping schedule for the field
and verifier role. A generated manifest must list the concrete values. The first
fp128 milestone should cover the D64 terminal path, then the remaining reachable
dimensions.

The full matrix theorem must include the current capacity rule. In mathematical
form the CRT modulus product must be greater than twice the largest possible
absolute coefficient. The bound depends on matrix width, ring dimension, field
centering, and the signed right-hand-side bound. Testing this inequality at
runtime does not replace proving that the formula bounds the actual loops.

### Tier 2 deferred field and sparse ring operations

| ID | Target | Mathematical contract |
|---|---|---|
| `F128-DOT-D*` | `eval_ring_at_pows_fast` and flat form | One final reduction equals the sum of all `D` field products |
| `F128-ACCUM-HEADROOM` | `Fp128ProductAccum` addition | Every accumulator word and carry remains exact for the supported term count |
| `RING-SPARSE-MUL` | signed sparse challenge multiplication | Output equals negacyclic multiplication by the listed positions and coefficients |
| `RING-MULADD` | ring scaling and accumulation | Output equals coefficient-wise field multiply-add with negacyclic wrap and sign |

The current comment in `ring/eval.rs` and its randomized D64 test are not enough
for `F128-DOT-D*`. The theorem must cover all inputs within the type's canonical
range and the exact production term count.

### Tier 3 challenge sampling

| ID | Target | Mathematical contract |
|---|---|---|
| `SAMPLE-RANGE` | `XofCursor::next_usize_mod` | Given uniform source words, the returned value is uniform in `0..m` and no accepted value has modulo bias |
| `SAMPLE-FY-DENSE` | stack partial Fisher-Yates | Output positions are distinct and match the first `w` values of the specified permutation |
| `SAMPLE-FY-SPARSE` | virtual sparse permutation | For the same range draws, output and draw consumption equal the dense algorithm |
| `SAMPLE-SIGNS` | sign assignment | Each coefficient has the configured magnitude and an independent sign bit |
| `SAMPLE-REJECT` | bounded operator-norm rejection | The first accepted candidate is returned, draws are consumed in order, and failure occurs only after the configured limit |
| `SAMPLE-TRANSCRIPT` | seed and batch derivation | Prover and verifier absorb the same bytes and derive the same cursor streams |

These are primarily functional HOL theorems. The current Rust code contains
allocation, hashing, and error handling, so proving its whole compiler output is
not the first useful machine-code target. The first implementation should extract
fixed-buffer pure cores for range sampling and the two Fisher-Yates forms. HOL
then proves the cores. Tests prove that the existing public API calls them without
changing transcript bytes or output.

An exact machine-code theorem may follow for those cores. The s2n-bignum
variable-time ML-KEM and ML-DSA rejection proofs show that data-dependent loops
and rejection are supported. Variable time is acceptable here because challenge
candidates and acceptance are public. It must still be stated in the security
review.

The SHAKE256 implementation is a separate boundary. A complete claim must either
link a proved SHAKE object or prove the exact Rust SHAKE implementation. Until
then, `SAMPLE-TRANSCRIPT` proves deterministic byte mapping and assumes that the
XOF implementation meets SHAKE256.

`SAMPLE-FY-SPARSE` must cover the scratch table's open addressing, bounded probe,
capacity rule, missing-key identity value, generation counter, and counter-wrap
clear. `SAMPLE-RANGE` assumes the public sampler limit `D <= 2048`. It must prove
that every mask and shift is defined for that range. The composed theorem must
also preserve the exact byte-consumption order because the sign bytes follow the
position draws in the same XOF stream.

### Tier 4 operator norm

The runtime predicate uses integers. It does not execute floating-point cosine
or sine. It currently builds fixed-point sine and cosine enclosures with Machin's
formula, alternating series, and outward-rounded `i128` interval arithmetic.
The SIMD accumulator then adds or subtracts table entries according to a sparse
challenge.

| ID | Target | Mathematical contract |
|---|---|---|
| `OPNORM-PI` | fixed-point Machin formula | The returned interval contains the real value of pi |
| `OPNORM-TABLE-NOWRAP` | table builder integer arithmetic | Every release-mode i128 multiply, add, divide, and shift stays in range for D64 and D128 |
| `OPNORM-ROOTS-D64` | D64 sine and cosine table | Every entry is within `eps_root` units of the scaled real root coordinate |
| `OPNORM-ROOTS-D128` | D128 sine and cosine table | Same contract for D128 |
| `OPNORM-ACC-SCALAR` | scalar transposed accumulator | Integer real and imaginary sums equal evaluation at each stored root coordinate |
| `OPNORM-ACC-AVX2` | four-frequency x86 accumulator | Same sums as scalar, with no i64 wrap |
| `OPNORM-ACC-NEON` | four-frequency AArch64 accumulator | Same sums as scalar, with no i64 wrap |
| `OPNORM-UPPER` | error and square bound | The checked value is an upper bound on the true squared operator norm |
| `OPNORM-DECIDE` | strict accept predicate | Every accepted challenge has true operator norm below the configured threshold |
| `OPNORM-SUPPORT` | accepted-support certificate replay | The certified subset is contained in the runtime accepted set and has the recorded size |

The preferred implementation embeds generated D64 and D128 root tables and their
error bounds. A generator produces canonical bytes and a digest. HOL Light proves
the real-number enclosure and emits or checks those exact integers. The verifier
links the integers instead of rebuilding trigonometric intervals at startup. A
differential test requires byte-for-byte equality with the current generator
before the cutover.

This split keeps the hard real analysis out of the SIMD proof. HOL Light can
express real sine, cosine, pi, alternating-series bounds, integer intervals, and
the final inequality. The s2n-bignum SIMD helpers can prove the accumulator. The
accepted-support JSON certificates under `scripts/operator_norm/` remain separate
artifacts until `OPNORM-SUPPORT` is complete.

The current table builder performs some ordinary release-mode `i128` arithmetic
before its later checked accumulator validation. `OPNORM-TABLE-NOWRAP` is therefore
a required theorem even if the embedded-table cutover removes that arithmetic
from the production verifier. It certifies the generator that defines the bytes.

## Production object design

### Coarse standalone symbols

The NTT and matrix operations should become ABI-correct standalone assembly
symbols. One call performs enough work that its call cost should be small relative
to the transform. This gives the proof a stable byte sequence and avoids proving
a new compiler wrapper for every Rust release.

The symbols should be coarse. A Rust loop must not call one assembly function for
each butterfly or each field operation. A suitable boundary performs at least one
complete transform, one complete pointwise row dot, or one complete matrix row.

The operator-norm assembly boundary should process the complete list of sparse
coefficients for one four-frequency chunk. This avoids a call for each coefficient.
If that boundary regresses the measured sampler, retain the inline path until a
larger proved symbol is ready.

Every symbol needs these artifacts.

1. Source assembly with explicit architecture and ABI.
2. The built ELF or Mach-O object and a deterministic byte extraction step.
3. A body theorem.
4. A SysV or AAPCS subroutine theorem. Windows needs a separate theorem before it
   is included in the claim.
5. Rust `extern` declarations and one safe wrapper that validates dimensions and
   bounds before the call.
6. A final-binary audit that resolves every production call site to the symbol and
   compares its bytes with the proved object.

### Model extensions

When a required instruction is missing, prefer a small reviewed model extension
when the instruction is natural and widely used. The extension must include the
instruction datatype, decoder, execution semantics, lane-level simplification
rule, positive simulator vectors, invalid-encoding tests, and real-machine
cosimulation. For loads and stores, tests must cover the exact memory width,
register wraparound, addressing modes, writeback, and fault or invalid-encoding
conditions. The mathematical definition should be short enough to review by
hand. An independently stated lane or memory-layout lemma must connect that
definition to the expanded form used by proof automation.

The first expected x86 extensions are `VPSHUFHW`, `VPSHUFLW`, and, if the
verifier's final object contains it, `VPMOVSXBW`. The first expected AArch64
extensions are `LD4`, `ST4`, and `SHSUB`. The exact production disassembly is the
source of truth. Compiler intrinsic names alone do not establish which model
extensions are needed.

Every extension must be proposed upstream to s2n-bignum as a generic processor
model change. Akita may pin a reviewed fork while the upstream change is pending,
but the production proof profile must record that fork and must move back to an
upstream revision after acceptance. The model review is part of the trust
boundary: a theorem using a new instruction proves behavior according to the new
HOL definition, so an incorrect definition can support an incorrect theorem.

A kernel rewrite is acceptable when it uses an already modeled sequence and does
not regress the relevant benchmark. A rewrite made only to avoid proof work is not
accepted without measurement and review.

## Theorem form

Each machine theorem follows the s2n-bignum structure.

```text
Assume
  the exact object bytes are loaded at pc,
  the ABI arguments point to valid input and output regions,
  the input arrays have the stated mathematical values,
  the dimensions and coefficient bounds hold,
  and code, stack, inputs, outputs, and constant tables obey the stated overlap rules.

Then
  execution returns to the caller,
  the output region contains the specified mathematical result,
  the output satisfies the stated canonical or redundant range,
  and only the listed registers, flags, events, stack bytes, and output bytes may change.
```

Intermediate NTT theorems must state transform order and Montgomery scale. Saying
only that the output is congruent modulo a prime is insufficient when the next
kernel assumes a numeric range.

## Proof and source layout

The repository should use this structure.

```text
proofs/hol-light/
  README.md
  pins.env
  check.sh
  dev.sh
  manifest/
    verifier-kernels.toml
  common/
    fp128.ml
    crt.ml
    negacyclic_ntt.ml
    fixed_point.ml
    sparse_sampling.ml
  x86_64/
    objects/
    proofs/
  aarch64/
    objects/
    proofs/
  riscv64/
    objects/
    proofs/
  linkage/
    compare-object-bytes.sh
    audit-final-binary.sh
```

Object import, execution rules, shared mathematics, and final theorem bodies must
be separate files. The development runner loads the architecture model once and
reloads only the theorem body after an edit. The clean runner rebuilds every
object from pinned inputs.

## Evaluation

### Acceptance criteria

- [ ] A generated manifest lists every low-level kernel reachable from every
  shipping verifier schedule and field preset.
- [ ] The manifest records one proved dispatch route for each supported target.
- [ ] Verified x86 execution never selects IFMA52 or another AVX-512 path.
- [ ] The scalar x86 fallback is proved, or the supported-target contract is
  changed in a separately approved spec.
- [ ] Every required AVX2 instruction decodes in the pinned x86 model. Missing
  semantics have simulator tests and review.
- [ ] Every required NEON instruction decodes in the pinned AArch64 model. Missing
  semantics have simulator tests and review.
- [ ] Every Tier 1 target has mathematical, body, ABI, object, linkage, and
  dispatch evidence for its claimed architecture.
- [ ] `CRT-MATVEC-D64` proves the terminal fp128 `A * z` path for the exact
  shipping width and signed i16 bound.
- [ ] All remaining verifier-reachable dimensions have the same composed theorem.
- [ ] `F128-DOT-D*` proves deferred accumulator headroom for every production term
  count.
- [ ] Dense and sparse Fisher-Yates cores have a same-draw equivalence theorem.
- [ ] Range rejection proves absence of modulo bias under the uniform-stream
  assumption.
- [ ] D64 and D128 root tables have HOL Light enclosure theorems and exact-byte
  linkage to the embedded production tables.
- [ ] AVX2 and NEON operator-norm accumulators have whole-symbol theorems and
  no-overflow preconditions discharged by the production configurations.
- [ ] The strict operator-norm decision theorem implies the configured real norm
  bound for every accepted challenge.
- [ ] The prepared RISC-V cache artifact is bound to the setup seed, schedule row,
  field, dimension, prefix, CRT profile, and proved consumer.
- [ ] A final-binary job compares proved symbol bytes and rejects missing,
  duplicated, interposed, or redirected verifier symbols.
- [ ] The full clean proof run succeeds from pinned HOL Light and s2n-bignum
  revisions on a fresh checkout.
- [ ] The Akita Book explains the final claim and its remaining trust boundary
  before the spec is marked implemented.

### Testing strategy

The implementation keeps scalar differential tests for every vector kernel and
adds adversarial boundary vectors for zero, one, prime minus one, signed extrema,
maximum width, maximum ring dimension, maximum accumulator length, and every
conditional correction boundary.

The proof manifest is generated and checked with a repository-owned command.
The implementation should add this target.

```text
cargo run -p akita-config --release --example verifier_kernel_manifest -- \
  --check proofs/hol-light/manifest/verifier-kernels.toml
```

The generator enumerates all rows in every shipping schedule table, resolves the
field and role dimensions, and records the exact terminal width and signed input
bound. CI fails if a schedule change makes the checked manifest stale.

The following local proof modes are required.

```text
./proofs/hol-light/check.sh bytes <architecture>
./proofs/hol-light/dev.sh <architecture> <target>
./proofs/hol-light/check.sh target <target>
./proofs/hol-light/check.sh all
./proofs/hol-light/check.sh all --clean
./proofs/hol-light/check.sh linkage <verifier-binary>
```

The byte check must not start HOL Light. The development mode must reuse a loaded
processor model. The clean mode is the release and CI gate.

The existing repository checks remain required. Any code implementation also
runs the exact release Clippy feature matrices in `AGENTS.md`, verifier no-panic
tests, schedule tests, terminal NTT tests, selective L2 tests, and documentation
guardrails.

### Performance

Proof work must not silently trade away verifier performance. Before changing a
kernel boundary, record the current commit, CPU model, operating system, compiler,
frequency policy, command, sample count, median, and dispersion.

At minimum measure these paths.

| Path | Existing harness or required addition |
|---|---|
| Exact prepared NTT matrix product | `crates/akita-pcs/benches/ntt_matvec.rs` |
| Transform and cache construction | `crates/akita-pcs/benches/ring_ntt.rs` |
| Terminal verifier relation | `crates/akita-pcs/benches/root_kernels.rs` and one full verifier profile |
| Operator-norm accumulation and rejection | Add a focused `akita-challenges` benchmark plus one selective L2 verifier profile |
| Deferred fp128 ring evaluation | Add a focused D64 and maximum-reachable-dimension benchmark |
| Challenge sampling | Add dense-tier and sparse-tier draws for every production family |

A verified replacement must not regress the median of its owning end-to-end
verifier profile by more than two percent unless maintainers approve the measured
tradeoff. A microbenchmark improvement cannot excuse an end-to-end regression.
The PR must include raw before and after results. It must also report machines
where IFMA52 was previously selected so the cost of the verifier-only AVX2 policy
is visible.

## Execution

### Milestone 1

Create the proof harness, generated target manifest, exact object imports, final
binary linkage audit, and explicit verified verifier policy. Prove the scalar and
AVX2 D64 terminal matrix path. Add the small x86 model extensions needed by the
shipping object.

### Milestone 2

Prove the AArch64 D64 terminal matrix path. Add reviewed `LD4`, `ST4`, and `SHSUB`
semantics if the final object uses them. Complete all remaining schedule-reachable
ring dimensions and the scalar fallbacks.

### Milestone 3

Prove deferred fp128 ring evaluation and the sparse ring multiply-add kernels.
Compose their field dependencies with the linked Jolt theorem bundle.

### Milestone 4

Extract the pure challenge-sampling cores. Prove unbiased range sampling, dense
partial Fisher-Yates, sparse equivalence, sign assignment, and bounded rejection.
Preserve transcript bytes and draw order with golden tests.

### Milestone 5

Embed the D64 and D128 fixed-point root tables. Prove their real enclosures, the
scalar decision rule, AVX2 and NEON accumulators, and accepted-support certificate
replay.

### Milestone 6

Compose the kernel claims at each verifier call site, publish the final coverage
matrix, add CI, and write the Akita Book chapter. Mark this spec implemented only
when every acceptance criterion is complete or explicitly moved to a successor
spec.

## Risks and review points

1. The largest risk is proof and production divergence. Final-binary byte checks
   are required even when object import already succeeds.
2. Compiler intrinsics do not promise a fixed instruction sequence. Standalone
   symbols reduce this risk for large kernels.
3. CRT congruence without an exact capacity theorem can accept a wrong integer
   that happens to have the same residues.
4. SIMD lane ranges may be wider than canonical residues. Every downstream
   theorem must accept the actual upstream range.
5. A prepared matrix can be mathematically correct for the wrong setup or
   schedule. Artifact identity is part of the verifier theorem's precondition.
6. The operator-norm accumulator proof does not certify the root table or the
   security size of the accepted set. Those are separate named targets.
7. A deterministic sampler theorem does not prove that SHAKE behaves as a random
   oracle. The cryptographic assumption must remain visible.
8. Constant-time execution is out of scope. Public challenge rejection and other
   verifier kernels may be variable time. The documentation must state this so a
   functional-correctness theorem is not mistaken for a side-channel theorem.
9. Adding instruction semantics expands the trusted model. Upstream review,
   decoder tests, simulator vectors, and revision pins are required.
10. Disabling IFMA52 for verifier work may have a measurable cost on capable
    servers. The decision is provisional until end-to-end benchmarks are reviewed.

## Alternatives considered

### Prove compiler-generated intrinsic wrappers

This preserves current inlining. It also binds the proof to a compiler version,
flags, surrounding code generation, and register allocation. It remains suitable
for tiny operations when a call boundary is too expensive. It is not the default
for whole NTTs because one out-of-line call can cover substantial work.

### Add full AVX-512 IFMA52 semantics first

This would preserve the current fastest exact route on capable x86 machines. It
requires EVEX decoding, mask semantics, 512-bit arithmetic, IFMA52 semantics, and
new proof automation before any Akita theorem can finish. AVX2 already covers the
ordinary production NTT and has strong s2n-bignum precedent. The first verified
verifier therefore uses AVX2 and measures the cost.

### Force AVX2 on every x86 machine

This is simple but rejects older x86 processors that the current scalar fallback
supports. The proposed policy uses AVX2 when available and a proved scalar symbol
otherwise.

### Prove only scalar reference implementations

This proves useful mathematics but does not cover the instructions that fast
production verifiers run. Scalar proofs are fallbacks and composition lemmas, not
the endpoint.

### Prove all Rust compiler output

This would give the broadest byte-level claim, but allocation, error handling,
hashing, and monomorphized control flow create a large compiler-specific object.
The plan instead extracts stable arithmetic cores, proves their objects, and keeps
the high-level protocol boundary explicit.

## Documentation

When implementation starts, add a formal verification chapter under
`book/src/how/`. It must explain the six claim layers, the verified dispatch
policy, each target's preconditions, object linkage, the remaining trust boundary,
and how a contributor reruns or extends a proof. The chapter becomes the durable
owner when this spec is complete and archived.

Update `AGENTS.md` only after proof commands or verifier-reachable dispatch rules
exist. Do not document proposed commands there before they work.

## References

- [s2n-bignum at the audited revision](https://github.com/awslabs/s2n-bignum/tree/ac31a43db30953037abd1b64b540e65cf31f4c67)
- [x86 ML-DSA NTT proof](https://github.com/awslabs/s2n-bignum/blob/ac31a43db30953037abd1b64b540e65cf31f4c67/x86/proofs/mldsa_ntt.ml)
- [x86 ML-DSA pointwise proof](https://github.com/awslabs/s2n-bignum/blob/ac31a43db30953037abd1b64b540e65cf31f4c67/x86/proofs/mldsa_pointwise.ml)
- [AArch64 ML-KEM NTT proof](https://github.com/awslabs/s2n-bignum/blob/ac31a43db30953037abd1b64b540e65cf31f4c67/arm/proofs/mlkem_ntt.ml)
- [AArch64 ML-DSA NTT proof](https://github.com/awslabs/s2n-bignum/blob/ac31a43db30953037abd1b64b540e65cf31f4c67/arm/proofs/mldsa_ntt.ml)
- [x86 variable-time ML-KEM rejection proof](https://github.com/awslabs/s2n-bignum/blob/ac31a43db30953037abd1b64b540e65cf31f4c67/x86/proofs/mlkem_rej_uniform_VARIABLE_TIME.ml)
- [AArch64 variable-time ML-KEM rejection proof](https://github.com/awslabs/s2n-bignum/blob/ac31a43db30953037abd1b64b540e65cf31f4c67/arm/proofs/mlkem_rej_uniform_VARIABLE_TIME.ml)
- [s2n-bignum program equivalence notes](https://github.com/awslabs/s2n-bignum/blob/ac31a43db30953037abd1b64b540e65cf31f4c67/doc/program_equivalence.md)
- [HOL Light at the audited revision](https://github.com/jrh13/hol-light/tree/433477862bb90b328a593e012e09390e99b2439b)
- [Akita fp128 proof prototype PR #433](https://github.com/LayerZero-Labs/akita/pull/433)
- `specs/flat-public-matrix-and-exact-ntt-cache.md`
- `specs/large-digit-ntt-infrastructure.md`
- `specs/selective-l2-fold-security-sizing.md`
- `book/src/foundations/ntt-crt.md`
