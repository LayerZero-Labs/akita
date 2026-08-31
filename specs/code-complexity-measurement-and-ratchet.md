# Spec: Code complexity measurement and maintainability ratchet

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-08-31 |
| Status        | proposed |
| PR            | |
| Supersedes    | |
| Superseded-by | |
| Book-chapter  | |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Akita has strong correctness, verifier-safety, file-size, formatting, lint, and
dependency-hygiene checks. It does not yet have a shared vocabulary for the
internal complexity of a function. Reviewers can identify a large function or
a deeply nested state machine, but they cannot compare it with the rest of the
repository, distinguish inherited debt from a regression, or measure whether a
refactor improved the intended property.

This specification introduces a reproducible complexity report and a gradual
maintainability ratchet. The report combines cyclomatic complexity, cognitive
complexity, Halstead difficulty, physical lines, test coverage, and the Change
Risk Anti-Patterns (CRAP) score. It classifies generated code, tests, arithmetic
kernels, and ordinary production code separately. It reports both the absolute
grade of changed functions and their change from the selected comparison base.

The first implementation is advisory. It MUST NOT turn the prototype
thresholds in this document into blocking CI until the parser accepts the full
tracked Rust syntax, the team has reviewed representative reports, and the
baseline and exception process are checked in. Later enforcement is a ratchet:
new or materially changed code cannot introduce an unexplained critical
hotspot. It is not a demand to rewrite the repository at once.

The associated refactoring campaign targets a small set of measured protocol
orchestrators and state machines. It preserves protocol behavior, proof bytes,
transcript order, schedule selection, verifier rejection behavior, and hot-path
performance. Dense arithmetic kernels are reviewed in context rather than
rewritten to make a score smaller.

## Decision at a glance

| Question | Decision |
|---|---|
| Initial status | Advisory report only |
| Primary unit | Named production Rust function or method |
| Comparison | Absolute grade plus base-to-head delta |
| Primary metrics | Cyclomatic, cognitive, Halstead difficulty, physical lines |
| Composite metric | CRAP, when compatible function coverage is available |
| File policy | Keep the existing 1,500-line hard cap; add 500- and 1,000-line review signals |
| Generated code | Report separately; never mix into handwritten-code percentiles |
| Tests and examples | Report separately; do not use them to grade production structure |
| Arithmetic kernels | Permit documented exceptions backed by focused tests and benchmarks |
| Initial threshold behavior | Annotation, not failure |
| Eventual hard failure | New unexplained critical hotspot or regression after explicit team approval |
| Existing debt | Baseline and reduce deliberately; do not require immediate cleanup |
| Metric gaming | Thin wrappers and single-use pass-through extraction are forbidden |
| Durable policy home | `docs/code-quality.md`, with a short `AGENTS.md` pointer after approval |

## Motivation

### Prototype evidence

A prototype audit measured quotient-free PR #445 at
`b09d69bfdab4c5ffc689bcb6b32327c9f5e7f622`. The production scope excluded
dedicated tests, examples, benches, generated tables, fuzz targets, and
profiling programs. It contained 472 files, 168,753 physical lines, and 6,118
named functions and methods.

| Metric | Median | p90 | p95 | p99 | Maximum | Prototype warning count |
|---|---:|---:|---:|---:|---:|---:|
| Cyclomatic complexity | 2 | 8 | 12 | 24.83 | 96 | 80 at or above 22 |
| Cognitive complexity | 0 | 5 | 9 | 22 | 99 | 63 at or above 22 |
| Halstead difficulty | 11.00 | 26.85 | 34.00 | 50.73 | 120.83 | 5 at or above 80 |

The typical function is simple. The useful signal is the upper tail: protocol
orchestrators, nested round-state transitions, setup/layout cross-products,
and dense arithmetic kernels. A repository-wide rule that every function must
remain below each prototype threshold would reject inherited code without
explaining which risk matters or how to improve it.

The same audit found 128 of 472 strictly classified production files above 500
physical lines. Across all 737 Rust files, 184 exceeded 500 lines, 21 exceeded
1,000, and none exceeded the existing 1,500-line cap. A 500-line hard cap would
therefore reject roughly one quarter of the current tree. It is useful as a
review prompt, not as an immediate build failure.

Coverage joined to 5,909 of the 6,118 named production functions in the
prototype run. The resulting CRAP distribution had median 2, p95 21.30, and
270 functions at or above 25. Sixty-two functions had cyclomatic complexity at
or above 25, which makes a strict CRAP score below 25 impossible even at 100%
coverage. CRAP therefore cannot replace a complexity grade.

These numbers are motivation, not a checked-in baseline. The prototype used
`rust-code-analysis-cli 0.0.25`, whose bundled Rust grammar produced 48 parse
error nodes across 14 of 737 files. An implementation MUST regenerate the
baseline with the selected pinned analyzer and MUST record parser completeness.

### Problems this policy solves

The policy gives maintainers a shared answer to five review questions:

1. Is this function complicated relative to the rest of Akita?
2. Is the difficulty caused by many independent paths, deep nesting, dense
   arithmetic, low coverage, or several of these together?
3. Did the PR introduce the hotspot, inherit it, or improve it?
4. Is decomposition appropriate, or is the function a bounded performance
   kernel whose shape should be protected by tests and benchmarks?
5. What evidence is required before an exception becomes accepted debt?

### What a score cannot prove

A complexity score does not prove incorrectness, insecurity, unreadability, or
poor performance. A lower score does not prove a better abstraction. In
particular:

- generated match tables can have extreme cyclomatic scores but no human
  maintenance burden;
- validation code can have many paths because it rejects malformed inputs;
- state machines can be cognitively difficult despite a moderate path count;
- branch-light arithmetic kernels can have high Halstead difficulty;
- extracting one-line wrappers can lower a function score while making the
  architecture worse; and
- incomplete coverage can inflate CRAP for code that a selected test profile
  never instantiates.

Reviewers MUST use the metric, source context, tests, performance role, and PR
delta together.

### Historical lineage and intended contexts

These measures come from different research and engineering traditions. They
are not independent estimates of one universal property called “complexity,”
and agreement among them is not a proof of poor quality.

| Measure | Origin and original concern | Useful interpretation here |
|---|---|---|
| Physical lines | Industrial size limits predate modern static analysis. McCabe's 1976 paper explicitly contrasted its control-flow measure with rules such as IBM's 50-line and TRW's two-page module limits. | Ownership and review-surface prompt |
| Cyclomatic complexity | Thomas McCabe's 1976 graph-theoretic measure counted a basis of linearly independent paths through a program control-flow graph. | Decision structure and minimum basis-path test burden |
| Halstead difficulty | Maurice Halstead's 1977 *Elements of Software Science* derived a family of lexical measures from distinct and total operators and operands. | Operator/operand density within one pinned lexical model |
| Test coverage | Structural coverage criteria measure which specified program elements a test suite exercises. Coverage has many forms, including statement, branch, condition, and path coverage. | Evidence of test exposure, not proof that assertions are effective |
| CRAP | Alberto Savoia and Bob Evans introduced the Change Risk Analysis and Predictions metric in 2007; the later name is Change Risk Anti-Patterns. It combines cyclomatic complexity with coverage. | Prioritization of complicated, weakly exercised functions |
| Cognitive complexity | SonarSource introduced Cognitive Complexity in December 2016 and published the governing technical report in 2017 to distinguish understandability from testability. | Relative difficulty of following nested and nonlinear control flow |

#### Cyclomatic complexity: paths, modularization, and testing

[McCabe's original paper](https://doi.org/10.1109/TSE.1976.233837) asked how
to divide programs into modules that remain testable and maintainable. It
replaced total path count, which can be infinite in the presence of loops,
with the dimension of a basis set of control-flow paths. The paper connected
that number to basis-path testing and stressed that testing the basis does not
prove correctness. The relationship was later developed into the
[NIST structured-testing methodology](https://www.nist.gov/publications/structured-testing-testing-methodology-using-cyclomatic-complexity-metric).

McCabe reported an operational upper bound of 10, but called it “reasonable,
but not magical” and allowed exceptions for large selection statements. That
history explains the familiar 10-point rule and also why it must not be copied
without context. Modern languages add short-circuit expressions, pattern
matches, error propagation, closures, and macro-generated control flow that
different analyzers count differently.

For Akita, cyclomatic complexity is therefore a testability and decomposition
signal. It is not a literal count of all executable paths, a defect
probability, or a universal module-size law.

#### Halstead difficulty: lexical software science

[Halstead's 1977 book](https://books.google.com/books?id=rRIpAQAAMAAJ)
attempted to build quantitative “software science” from a program's operator
and operand vocabulary. Difficulty is one member of that larger family,
alongside vocabulary, length, volume, effort, and other derived values. Its
historical ambition was broader than the narrow use in this specification.

Difficulty depends on what a tool calls an operator or operand and on the
lexical span it assigns to a function. Rust traits, qualified paths, generic
bounds, closures, macros, and overloaded operators make those choices
material. The value is neither computational complexity in the asymptotic
sense nor a direct measure of mathematical sophistication. Halstead did not
establish 80 as a universal difficulty limit; the 40, 80, and 100 bands in
this specification are local review bands that require calibration.

#### Cognitive complexity: an understandability heuristic

SonarSource introduced the metric in a
[2016 explanation](https://www.sonarsource.com/blog/cognitive-complexity-because-testability-understandability/),
then documented its rules in
[G. Ann Campbell's technical report](https://www.sonarsource.com/docs/CognitiveComplexity.pdf)
and a [2018 overview](https://doi.org/10.1145/3194164.3194186). Its central
choice is to charge structures that interrupt linear reading and to add a
nesting penalty, while leaving ordinary well-named calls free. This is a
designed heuristic, not a graph invariant.

A 2020 meta-analysis of about 24,000 evaluations of 427 snippets found that
the measure correlated positively with comprehension time and subjective
understandability ratings, but reported mixed results for comprehension
correctness and physiological measures
([paper](https://doi.org/10.1145/3382494.3410636),
[replication package](https://doi.org/10.5281/zenodo.3949828)). The appropriate
claim is therefore narrow: a pinned implementation can help locate code whose
control flow may be harder to follow. Different “cognitive complexity”
implementations MUST NOT be treated as one interchangeable scale.

#### Coverage and CRAP: exposure and change triage

Coverage belongs to the older family of structural test-adequacy criteria.
The current ISO/IEC/IEEE testing vocabulary defines coverage as the degree to
which specified coverage items have been exercised; those items can be
statements, branches, states, or other structures
([ISO/IEC/IEEE 29119-1:2022](https://www.iso.org/obp/ui/#iso:std:iso-iec-ieee:29119:-1:ed-2:v1:en)).
The denominator and test profile are therefore part of the measurement, not
incidental tool settings.

Coverage can demonstrate absence of test execution: uncovered code was not
executed by that test profile. It cannot demonstrate the quality of inputs or
assertions. In a 2014 study of 31,000 generated suites for five large Java
systems, coverage had only low-to-moderate correlation with mutation-based
effectiveness after controlling for suite size; the authors cautioned against
using a fixed coverage value as a quality target
([paper](https://doi.org/10.1145/2568225.2568271)). This does not make coverage
useless. It defines its role as an exposure diagnostic rather than a proof of
correctness.

Savoia and Evans's CRAP metric joined that exposure signal to McCabe's path
signal. Savoia's
[2011 retrospective](https://testing.googleblog.com/2011/02/this-code-is-crap.html)
records its 2007 origin and acknowledges that the formula does not include
higher-order design concerns such as cohesion and coupling. The often-cited
CRAP threshold of 30 is a practical convention, not a mathematical boundary.
This specification does not adopt 25, 30, or any other CRAP value as an
initial hard gate.

#### Measurement theory: why this specification uses several signals

Later software-metrics research made the limitations of single-number policy
explicit. Weyuker's
[1988 evaluation](https://doi.org/10.1109/32.6178) compared statement count,
cyclomatic number, Halstead effort, and data-flow measures against proposed
properties; no examined measure possessed all of them. Kaner and Bond's
[2004 construct-validity framework](https://kaner.com/pdfs/metrics2004.pdf)
argued that a metric must be validated against the attribute it claims to
measure and warned that poorly understood targets distort behavior.

Akita consequently uses a small dashboard rather than a synthetic quality
number:

- cyclomatic complexity prompts questions about decisions and test paths;
- cognitive complexity prompts questions about nesting and reading order;
- Halstead difficulty prompts questions about lexical and arithmetic density;
- physical lines prompt questions about ownership and review surface;
- coverage reports observed execution under a named test profile; and
- CRAP orders candidates where path complexity and missing exposure coincide.

The grades below are Akita policy proposals, not constants inherited from the
literature. Green cyclomatic complexity ending at 10 acknowledges McCabe's
historical practice. The yellow, red, and critical bands are shaped by the
prototype Akita distribution and reviewer intent. The cognitive and Halstead
bands are likewise local calibration hypotheses. Any transition from
advisory reporting to enforcement requires the explicit Phase 2 policy change
defined below.

## Intent

### Goal

Introduce a versioned, reproducible complexity report and use it to reduce
Akita's highest-risk handwritten production hotspots without changing protocol
semantics or encouraging shallow abstraction.

### Invariants

1. The report MUST identify its source SHA, comparison SHA, tool versions,
   configuration, target language, file classification, and parser status.
2. Production, test, generated, example, bench, fuzz, and profiling code MUST
   have separate aggregates.
3. A PR report MUST distinguish existing functions, added functions, removed
   functions, and functions whose identity cannot be matched reliably.
4. A changed function MUST be shown with both its absolute grade and metric
   delta when a reliable base match exists.
5. Generated code MUST NOT affect handwritten production percentiles or gates.
6. A parser error MUST be visible. CI MUST NOT silently treat an unparsed
   function as complexity zero.
7. Metric computation MUST NOT expand the verifier's trust boundary, introduce
   runtime dependencies, or affect produced artifacts.
8. A complexity refactor MUST preserve proof serialization, transcript order,
   schedule selection, verifier acceptance and rejection behavior, and security
   sizing unless a separate approved specification authorizes a change.
9. Hot-path refactors MUST include before-and-after performance evidence when
   they change loop structure, allocation, dispatch, inlining, SIMD, NTT, or
   dense sum-check kernels.
10. The implementation MUST follow Akita's single-source-of-truth rule. It MUST
    NOT reduce a score by adding thin wrappers, pass-through aliases, duplicate
    formulas, or single-use indirection.
11. Any blocking threshold or exception MUST be reviewable in the repository;
    it MUST NOT depend on an unversioned service default.

### Non-goals

This work does not:

- impose the prototype thresholds as immediate repository-wide hard limits;
- require 100% line or branch coverage;
- require zero duplicated lines across architecture-specific kernels;
- require all files to contain fewer than 500 lines;
- make Halstead difficulty or CRAP a security claim;
- rewrite generated schedules, SIS tables, vendored code, or macro expansions;
- change the PCS protocol, proof wire format, transcript, planner objective,
  setup artifacts, generated schedules, or verifier equations;
- merge semantically distinct phases merely to reduce aggregate file count;
- split performance kernels without benchmark evidence; or
- replace human review with one composite score.

## Metric definitions

The implementation MUST pin the exact analyzer and configuration that define
each metric. A different tool or version can implement a different counting
model under the same metric name, so a tool change starts a new baseline epoch.

### Cyclomatic complexity

Cyclomatic complexity estimates the number of linearly independent control-flow
paths through one function. The selected Rust analyzer's model SHOULD start at
one and add for decision-producing constructs such as conditionals, loops,
match arms, fallible propagation, and short-circuit Boolean operators.

Cyclomatic complexity is most useful for identifying:

- functions that combine validation with construction;
- large dispatch cross-products;
- parsers and argument processors with many alternatives; and
- functions whose test matrix grows with independent branches.

The report MUST document the exact counted Rust constructs. Macro expansion and
compiler-generated branches MUST be identified as included or excluded.

### Cognitive complexity

Cognitive complexity estimates how difficult the control flow is to follow.
It penalizes nested conditionals, loops, matches, Boolean sequences, and
nonlinear control flow more heavily than a flat list of checks.

This metric is most useful for protocol state machines. A function with 20 flat
input checks can have more cyclomatic paths but be easier to understand than a
function with six mutually dependent nested state transitions.

The report MUST use one pinned cognitive-complexity algorithm. It MUST NOT
compare values from different algorithms as though they shared a scale.

### Halstead difficulty

For a lexical function span, let:

- `n1` be the number of distinct operators;
- `N1` be the total operator occurrences;
- `n2` be the number of distinct operands; and
- `N2` be the total operand occurrences.

Halstead difficulty is:

```text
D = (n1 / 2) * (N2 / n2)
```

High Halstead difficulty often identifies dense arithmetic that repeatedly
combines a comparatively small vocabulary of values. It does not necessarily
identify branch complexity. The report MUST disclose whether nested closures
or functions contribute to the enclosing lexical span.

### Physical lines

Physical lines count source lines, including formatting and comments according
to the selected counter. The report SHOULD also preserve instruction-line and
logical-statement counts when the analyzer provides them.

File size is an ownership prompt. Function size is a review aid. Neither is a
substitute for a responsibility analysis.

### CRAP

For cyclomatic complexity `C` and compatible function coverage fraction `p`,
the report uses:

```text
CRAP = C^2 * (1 - p)^3 + C
```

The coverage fraction MUST be defined in the report. The prototype used
covered executable source lines within the function's inclusive lexical span.
Branch coverage and LLVM function counts are separate measurements and MUST
NOT be substituted into the formula without starting a new baseline epoch.

CRAP combines test exposure with path complexity. It is useful for ordering
test and refactoring work. It MUST remain advisory during the initial rollout.

## Scope and classification

The report MUST start from tracked files at the selected SHA. It MUST NOT scan
untracked build products or developer-local files.

At minimum, Rust files use these mutually exclusive report categories:

| Category | Typical paths or evidence |
|---|---|
| Production | Handwritten crate `src/` code used by shipped libraries or binaries |
| Test | `tests/`, test-support modules, dedicated `tests.rs`, or `_tests.rs` |
| Generated | Generated-file header, generated schedule/table directory, or explicit manifest |
| Example | Cargo examples and repository examples |
| Bench | Cargo benches and benchmark-only source |
| Fuzz | Fuzz targets and fuzz-only support |
| Profile | Profiling harnesses and production-size measurement programs |

Inline `#[cfg(test)]` modules inside a production file are a known
classification challenge. The implementation MUST either classify lexical
spaces individually or state that inline tests contribute to the enclosing
file category.

Non-Rust languages MAY use their ecosystem's pinned analyzer. Their scores
MUST be reported separately because implementations and scales are not
comparable across languages.

## Grades and review signals

The initial grades are calibration bands. They are not blocking requirements.

| Grade | Cyclomatic | Cognitive | Halstead difficulty | Intended response |
|---|---:|---:|---:|---|
| Green | 1-10 | 0-10 | below 40 | Ordinary review |
| Yellow | 11-21 | 11-21 | 40-79.99 | Inspect responsibilities, nesting, and tests |
| Red | 22-49 | 22-49 | 80-99.99 | Require explicit reviewer discussion |
| Critical | 50 or more | 50 or more | 100 or more | Refactor or record a reviewed exception |

The overall function grade is its highest metric grade. Reports MUST retain the
individual metrics so reviewers can distinguish path, nesting, and arithmetic
density.

File-size signals are:

| Physical lines | Signal |
|---:|---|
| 500 or more | Advisory ownership review |
| 1,000 or more | Strong decomposition review |
| More than 1,500 | Existing hard failure |

An eventual ratchet SHOULD focus on new critical functions and material
regressions in changed functions. It SHOULD NOT fail a PR solely because an
unchanged inherited function remains red or critical.

## Initial hotspot inventory

The prototype audit identified three different classes of hotspot. These rows
are evidence from `b09d69b`, not permanent golden values.

| Function | Cyclomatic | Cognitive | Halstead | Primary cause |
|---|---:|---:|---:|---|
| `compute_multi_group_relation_quotient` | 96 | 51 | 54.58 | Validation, group/layout cross-product, and row construction in one function |
| `RingRelationProver::new` | 82 | 50 | 50.85 | Entire prepare, bind, grind, instance, and witness pipeline in one constructor |
| `prepare_compact_factors` | 77 | 43 | 70.24 | Geometry validation and compact-factor construction across layouts |
| `RelationRangeImageProver::ingest_challenge` | 45 | 99 | 54.35 | Deeply nested round-state and fused-path transitions |
| `balanced_decompose_coefficients_pow2_signed_into_with_params` | 24 | 80 | 100.83 | Unrolled overflow and non-overflow decomposition paths |
| `fuse_folded_partial_lane_and_compute_next_round` | 10 | 32 | 120.83 | Repeated dense arithmetic inside blocked loops |
| `try_eval_aligned_family` | 47 | 63 | 80.91 | Layout dispatch plus dense tensor evaluation |

Generated SIS lookup tables reached higher cyclomatic and Halstead maxima but
had low cognitive complexity. They are excluded from the handwritten hotspot
program.

## Rollout

### Phase 0: agree on semantics

Before CI integration:

1. Select and pin the analyzer versions.
2. Record the exact metric semantics and parser version.
3. Achieve zero unexplained parser errors, or approve an explicit temporary
   parser-error baseline with file and span inventory.
4. Check in file-classification rules and fixtures.
5. Generate a baseline from a named main or stack SHA.
6. Review representative green, yellow, red, and critical functions as a team.

No threshold blocks a PR in this phase.

### Phase 1: advisory PR report

CI or a maintainer command produces:

- repository production distributions;
- the highest changed-function grades;
- base and head values for reliably matched functions;
- added and removed critical functions;
- parser and join-quality diagnostics;
- changed files crossing 500 or 1,000 physical lines; and
- CRAP only when compatible coverage is available.

The report SHOULD be concise by default and link to a complete JSON artifact.
It MUST NOT post every unchanged hotspot on every PR.

Phase 1 runs for a calibration period selected during spec approval. The team
records false positives, parser defects, unstable identities, platform
differences, and cases where the grade correctly exposed an ownership problem.

### Phase 2: changed-code ratchet

After a separate reviewed policy change, CI MAY fail when a PR:

- adds a critical production function without an approved exception;
- moves a reliably matched production function into critical grade;
- materially worsens an existing critical function without reducing another
  dimension or documenting why the change is necessary;
- hides complexity through a thin wrapper or duplicate implementation; or
- introduces an unaccounted parser error.

The exact delta considered material MUST be chosen from calibration evidence.
This specification deliberately does not guess it before Phase 1 data exists.

### Phase 3: focused reduction

Refactoring proceeds in small behavior-preserving slices. Each slice SHOULD:

1. Name the complexity mechanism being reduced.
2. Preserve or improve focused tests before moving code.
3. Establish behavioral equivalence at the relevant public boundary.
4. Record before-and-after function metrics.
5. Run performance evidence when the changed path is hot.
6. Avoid mixing unrelated protocol or feature work.

The first campaign SHOULD address orchestration and state-transition hotspots.
Arithmetic kernels follow only when the team agrees that structural change is
worth the performance and audit cost.

## Architecture

### Reproducible report

The implementation SHOULD expose one repository command, for example:

```text
scripts/code-quality-report.sh --base <sha> --head <sha>
```

The exact name is an implementation choice. The command MUST:

1. resolve literal base and head SHAs;
2. inventory tracked files at each SHA;
3. run pinned analyzers in isolated source trees;
4. normalize results into one versioned JSON schema;
5. validate parser and file coverage;
6. match functions conservatively;
7. compute distributions and deltas; and
8. emit a human-readable summary.

The report artifact SHOULD include one row per function with:

- repository path;
- qualified function identity;
- inclusive source span;
- file category;
- each raw metric;
- tool and schema versions;
- base-match status and confidence;
- changed-file status; and
- coverage and CRAP fields when available.

Function identity is not stable under every rename, move, macro expansion, or
signature change. An ambiguous match MUST be reported as unmatched rather than
silently joined to the wrong function.

### Baseline epochs

A baseline epoch is identified by:

- report schema version;
- analyzer names and versions;
- analyzer configuration;
- parser version;
- classification rules; and
- coverage definition used by CRAP.

Changing any of these requires regenerating the baseline. CI MUST NOT compare
raw scores across epochs.

### Exceptions

An eventual critical-grade gate requires a checked-in exception registry. Each
exception MUST include:

- function identity and owning area;
- metric and observed value;
- why decomposition would reduce clarity, safety, or performance;
- tests that protect the behavior;
- benchmark evidence when performance is part of the rationale;
- an owner or review group; and
- an expiry condition or explicit statement that the exception is structural.

Moving or renaming a function MUST NOT make its exception apply accidentally to
another function.

## Refactoring program

### Orchestrators

`compute_multi_group_relation_quotient` and `RingRelationProver::new` combine
several protocol phases. Their refactors SHOULD introduce typed, validated
phase outputs rather than one-line forwarding helpers. Candidate boundaries
include:

- validated per-group quotient inputs;
- compression-source preparation;
- role-specific A, B, D, and consistency row construction;
- opening preparation;
- payload binding and grinding; and
- final instance and witness assembly.

### State transitions

`RelationRangeImageProver::ingest_challenge` SHOULD make the selected transition
explicit before mutating prover state. A transition enum or equivalent typed
decision can separate reduced-dense, deferred compact-prefix, partial-lane,
full-lane, and coefficient-round paths. Handlers MUST preserve the current
challenge order and fused-kernel semantics.

### Preparation cross-products

Coefficient-packing preparation SHOULD separate geometry validation from
factor or witness materialization. Canonical layout types remain the source of
truth; the refactor MUST NOT duplicate sizing formulas locally.

### Arithmetic kernels

Balanced decomposition and fused prefix kernels require a different review.
The implementation SHOULD first add or identify scalar equivalence tests,
boundary tests, and representative benchmarks. It MAY retain a critical metric
exception when unrolling or branch specialization is deliberate and a clearer
abstraction would harm performance or constant-factor auditability.

## Evaluation

### Acceptance criteria

#### Specification and calibration

- [ ] Maintainers approve the metric definitions and advisory grades.
- [ ] The team reviews at least one representative function from each grade and
      each hotspot class.
- [ ] The calibration period and any future material-delta rule are recorded by
      a follow-up policy change.

#### Measurement implementation

- [ ] Analyzer and parser versions are pinned and reproducible.
- [ ] Every tracked Rust file is represented or rejected with a visible error.
- [ ] Parser errors are zero or covered by an explicit reviewed baseline.
- [ ] Production, tests, generated code, examples, benches, fuzz, and profiling
      code are classified separately.
- [ ] Fixture tests cover classification, nested functions, closures, matches,
      short-circuit conditions, fallible propagation, macros, and parse errors.
- [ ] Base-to-head matching reports ambiguous identities as unmatched.
- [ ] The report emits a versioned complete JSON artifact and a concise summary.
- [ ] Repeated runs at one SHA on supported CI hosts are byte-stable after
      removing explicitly documented timestamps or paths.

#### Advisory rollout

- [ ] Phase 1 reports run without blocking unrelated PRs.
- [ ] Reports show absolute grades and reliable deltas for changed functions.
- [ ] Generated-code maxima do not affect production percentiles.
- [ ] CRAP output states its coverage profile, completeness, and join quality.
- [ ] Maintainers record false positives and exception candidates during the
      calibration period.

#### Refactoring

- [ ] Each refactoring slice records before-and-after metrics.
- [ ] Existing focused correctness and malformed-input tests remain green.
- [ ] Verifier-reachable code continues to return typed errors for malformed
      input and introduces no unchecked panic path.
- [ ] Protocol descriptors, proof serialization, transcript events, generated
      schedule identity, and planner selections remain unchanged unless a
      separate approved specification says otherwise.
- [ ] Hot-path changes include before-and-after wall time and an appropriate
      lower-variance measurement such as retired instructions when available.
- [ ] No refactor introduces a thin wrapper, duplicate formula, or parallel
      source of truth solely to improve a metric.

### Testing strategy

The measurement implementation requires small analyzer fixtures with known
control-flow shapes. Golden files SHOULD assert normalized schema rows and
grades, not incidental analyzer formatting. Integration tests SHOULD run the
report twice against a fixed fixture tree and compare normalized artifacts.

Repository validation for refactoring PRs is path-specific:

- protocol orchestrators require the owning prover, verifier, and PCS tests;
- verifier-reachable changes require malformed-input and no-panic coverage;
- transcript-adjacent changes require logging-transcript or schedule-event
  comparison where applicable;
- planner-adjacent changes require production-versus-oracle agreement and
  generated replay; and
- arithmetic kernels require scalar differential tests across supported field
  and ring dimensions.

The implementation PR MUST copy current feature graphs and selectors from the
live CI workflow rather than freezing them in this specification.

### Performance

The report itself SHOULD fit within the cheap-preflight budget after caches are
warm. During Phase 1, it MAY run as a separate advisory job if runtime or tool
installation is too expensive for the main merge gate. The implementation
proposal MUST measure cold and warm runtime before choosing placement.

Complexity refactors are expected to be performance-neutral. For hot code,
"neutral" means the observed change is within the agreed noise threshold of a
named benchmark on the same machine and configuration. The benchmark command,
hardware, source SHAs, sample count, and summary statistic MUST accompany the
claim.

## Alternatives considered

### Enforce the prototype thresholds immediately

Rejected. The prototype parser was not fully current, the repository contains
inherited hotspots, and each metric has domain-specific false positives.
Immediate enforcement would create cleanup pressure before the team agrees on
semantics or exceptions.

### Use only cyclomatic complexity

Rejected. It identifies path count but under-describes nested state machines
and dense arithmetic. The prototype's highest cognitive and Halstead functions
were not the highest cyclomatic functions.

### Use only CRAP

Rejected. CRAP depends on a complete compatible coverage profile, and its
minimum equals cyclomatic complexity. It cannot distinguish uncovered simple
code from inherently high-path code without exposing both inputs.

### Use only file length

Rejected. File length is easy to reproduce but weakly connected to local
control-flow difficulty. Akita already has a 1,500-line hard cap. Advisory 500-
and 1,000-line signals are sufficient for ownership review.

### Require a universal zero-exception policy

Rejected. Generated tables and tuned arithmetic kernels have legitimate shapes
that ordinary orchestration code should not copy. A visible exception with
tests, evidence, and ownership is more honest than metric-specific code games.

### Outsource the policy to a hosted quality service

Rejected as the source of truth. A hosted presentation MAY consume the checked
report, but contributors and CI must be able to reproduce the governing values
locally from pinned tools.

## Documentation

During the proposed and advisory phases, this specification owns the design and
acceptance criteria. After approval and implementation:

- `docs/code-quality.md` becomes the durable maintainer reference for metric
  semantics, grades, exceptions, and local commands;
- `AGENTS.md` gains only the essential command and a pointer to that reference;
- `docs/documentation.md` remains the authority for documentation quality and
  MUST NOT claim that prose quality can be reduced to a complexity score; and
- this spec is archived when its durable content has been folded into the
  maintainer reference and the acceptance criteria are complete.

No Book chapter is required because this policy governs repository maintenance,
not PCS behavior or integrator-facing architecture.

## Execution

Recommended implementation slices:

1. Pin candidate analyzers and close parser-fidelity gaps.
2. Define the normalized schema, classification manifest, and fixture tests.
3. Generate and review a fresh baseline from the then-current stack head.
4. Add the advisory local command and CI artifact.
5. Run the agreed calibration period.
6. Approve or revise grades, material deltas, and exception policy.
7. Refactor orchestration and state-transition hotspots in small stacked PRs.
8. Evaluate arithmetic-kernel exceptions with differential tests and benchmarks.
9. Propose the changed-code ratchet as a separate reviewed policy change.

## References

- `AGENTS.md` — current file-size, verifier no-panic, and single-source-of-truth contracts.
- `docs/documentation.md` — documentation lifecycle and quality policy.
- `specs/SPEC_REVIEW.md` — model-agnostic spec approval rubric.
- `specs/PRUNING.md` — live-spec lifecycle and archive policy.
- PR #445 prototype complexity evidence at
  `b09d69bfdab4c5ffc689bcb6b32327c9f5e7f622`.
- Thomas J. McCabe, [“A Complexity Measure”](https://doi.org/10.1109/TSE.1976.233837),
  *IEEE Transactions on Software Engineering*, 1976.
- Maurice H. Halstead,
  [*Elements of Software Science*](https://books.google.com/books?id=rRIpAQAAMAAJ),
  Elsevier, 1977.
- Elaine J. Weyuker,
  [“Evaluating Software Complexity Measures”](https://doi.org/10.1109/32.6178),
  *IEEE Transactions on Software Engineering*, 1988.
- Dolores R. Wallace, Arthur H. Watson, and Thomas J. McCabe,
  [*Structured Testing: A Testing Methodology Using the Cyclomatic Complexity Metric*](https://www.nist.gov/publications/structured-testing-testing-methodology-using-cyclomatic-complexity-metric),
  NIST Special Publication 500-235, 1996.
- Cem Kaner and Walter P. Bond,
  [“Software Engineering Metrics: What Do They Measure and How Do We Know?”](https://kaner.com/pdfs/metrics2004.pdf),
  METRICS 2004.
- Alberto Savoia,
  [“This Code is CRAP”](https://testing.googleblog.com/2011/02/this-code-is-crap.html),
  Google Testing Blog, 2011 retrospective on the 2007 metric.
- G. Ann Campbell,
  [“Cognitive Complexity, Because Testability != Understandability”](https://www.sonarsource.com/blog/cognitive-complexity-because-testability-understandability/),
  SonarSource, 2016.
- G. Ann Campbell,
  [“Cognitive Complexity: An Overview and Evaluation”](https://doi.org/10.1145/3194164.3194186),
  TechDebt 2018.
- Laura Inozemtseva and Reid Holmes,
  [“Coverage Is Not Strongly Correlated with Test Suite Effectiveness”](https://doi.org/10.1145/2568225.2568271),
  ICSE 2014.
- Marvin Muñoz Barón, Marvin Wyrich, and Stefan Wagner,
  [“An Empirical Validation of Cognitive Complexity as a Measure of Source Code Understandability”](https://doi.org/10.1145/3382494.3410636),
  ESEM 2020; [replication package](https://doi.org/10.5281/zenodo.3949828).
