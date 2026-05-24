# Akita Crate Graph

Akita is split into small workspace crates so verifier-oriented consumers can
depend on public proof replay without pulling prover-only polynomial backends,
setup expansion, offline planner search, examples, or benchmark harnesses.

## Dependency Layers

```mermaid
graph TD
  Ser["akita-serialization"]
  Field["akita-field"]
  Algebra["akita-algebra"]
  Transcript["akita-transcript"]
  Challenges["akita-challenges"]
  Sumcheck["akita-sumcheck"]
  Types["akita-types"]
  Planner["akita-planner"]
  Config["akita-config"]
  Verifier["akita-verifier"]
  Prover["akita-prover"]
  Setup["akita-setup"]
  Scheme["akita-scheme"]
  Pcs["akita-pcs"]

  Field --> Ser
  Algebra --> Field
  Algebra --> Ser
  Transcript --> Field
  Transcript --> Ser
  Challenges --> Algebra
  Challenges --> Field
  Challenges --> Transcript
  Sumcheck --> Algebra
  Sumcheck --> Field
  Sumcheck --> Ser
  Sumcheck --> Transcript
  Types --> Algebra
  Types --> Field
  Types --> Ser
  Types --> Sumcheck
  Types --> Transcript
  Planner --> Algebra
  Planner --> Field
  Planner --> Types
  Config --> Algebra
  Config --> Field
  Config --> Types
  Config -. "planner feature" .-> Planner
  Verifier --> Algebra
  Verifier --> Challenges
  Verifier --> Field
  Verifier --> Sumcheck
  Verifier --> Transcript
  Verifier --> Types
  Prover --> Algebra
  Prover --> Challenges
  Prover --> Field
  Prover --> Ser
  Prover --> Sumcheck
  Prover --> Transcript
  Prover --> Types
  Setup --> Algebra
  Setup --> Config
  Setup --> Field
  Setup --> Prover
  Setup --> Ser
  Setup --> Types
  Scheme --> Algebra
  Scheme --> Config
  Scheme --> Field
  Scheme --> Prover
  Scheme --> Setup
  Scheme --> Ser
  Scheme --> Transcript
  Scheme --> Types
  Scheme --> Verifier
  Pcs --> Algebra
  Pcs --> Challenges
  Pcs --> Config
  Pcs --> Field
  Pcs --> Prover
  Pcs --> Scheme
  Pcs --> Ser
  Pcs --> Setup
  Pcs --> Sumcheck
  Pcs --> Transcript
  Pcs --> Types
  Pcs --> Verifier
```

## Ownership Rules

- `akita-verifier` must stay planner-free and prover-free. It may use
  `akita-types`, `akita-sumcheck`, `akita-challenges`, `akita-transcript`,
  `akita-algebra`, and `akita-field`. Its internals are grouped into public
  proof-shape preparation, protocol replay, and stage verifier modules.
- `akita-config` owns concrete runtime presets and generated-schedule lookup.
  Offline planner search is available only through the `planner` feature.
- `akita-planner` owns DP search, proof-size exploration, SIS planning, and
  planner inspection binaries. Runtime verifier/prover crates must not depend
  on it.
- `akita-prover` owns polynomial backends, prover setup artifacts, NTT/matrix
  kernels, explicit compute-backend operation traits, recursive witness
  construction, ring-switch witness construction, proving orchestration, and
  its Akita-specific sumcheck stage provers.
- `akita-types` owns inert shared protocol data: proof/setup/claim shapes,
  opening-point and layout math, schedule contracts, generated table shapes,
  and transcript append traits. It should not grow planner search or prover
  algorithms.
- `akita-scheme` wires config, setup, prover, and verifier into the end-to-end
  `AkitaCommitmentScheme`.
- `akita-pcs` is the broad umbrella crate for examples and applications that
  want the full public surface. Verifier-only integrations should not use it.

CI runs `scripts/check-crate-deps.sh` to guard the important one-way
boundaries. Add new forbidden edges there whenever a crate gets split further.
