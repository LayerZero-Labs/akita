# Operating Akita

Akita exposes its deployment choices through explicit Cargo features, prepared
compute state, and one end to end profile harness. A host can build the exact
prover or verifier it needs and measure the resulting proof under the same
generated schedule used in production.

The operating path has three layers:

1. Choose the feature set and generated catalogs that belong in the binary.
2. Run a complete proof workload and inspect time, memory, and proof size.
3. Use focused reports or kernel benchmarks when a complete run identifies a
   component that needs deeper analysis.

## Start with the complete build

The default `akita-pcs` features provide parallel CPU execution, the standard
generated schedule catalogs, and the Blake2b transcript backend. This is the
right build for the [first proof](./quickstart.md) and most local development.

Use [Feature flags](./feature-flags.md) when the host needs a smaller schedule
set, sequential execution, disk persistence, another transcript backend, or
diagnostic instrumentation.

## Measure a real proof first

Akita's profile harness runs setup, commitment, proof generation, encoding, and
verification. It reports the complete public statement and the generated
schedule selected for that statement.

Start with [Profiling a workload](./profiling.md). It gives one canonical
command and explains every output. This complete run shows whether a local
optimization improves the time and memory of the host proof.

## Read CI performance results

Profile CI compares a pull request with its merge base on the same machine. It
runs dense, one hot, grouped, recursive setup, and partitioned prover cases.

[Reading benchmark reports](./benchmark-reports.md) explains the public claim
behind each case, the timing and memory phases, and the generated report files.

## Inspect arithmetic kernels

When an end to end trace points to public matrix multiplication or transform
work, use the focused NTT benchmarks. They compare ring dimensions, input
widths, exact CRT paths, and available SIMD implementations without setup or
transcript work.

[Arithmetic microbenchmarks](./arithmetic-benchmarks.md) explains those shapes
and how to interpret the 50 bit AVX-512 IFMA path.

## Diagnose failures

[Troubleshooting](./troubleshooting.md) covers unsupported schedule requests,
setup cache recovery, equality table limits, thread selection, verification
rejection, and the Jolt guest path. Each entry begins with the public error or
observable behavior and gives the next concrete check.

Akita treats these operating tools as part of the PCS. The same generated
schedule identity connects setup, proving, verification, profiling, and CI
comparison.
