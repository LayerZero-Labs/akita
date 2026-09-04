# Troubleshooting

This page covers failures that can occur before or around a proof run. When an
error comes from verification, first read [Verifier only
integration](./verifier-only.md) and the [verifier no-panic
contract](../how/verification.md#the-verifier-no-panic-contract).

## The profile example exits before running

The profile example requires a release build because debug timings are not
useful. If it prints this message, add `--release` to the Cargo command.

```text
examples/profile must be run with --release for meaningful timings.
```

Set `AKITA_ALLOW_DEBUG_PROFILE=1` only when you need to debug the harness. Do
not compare timings from that run with release results.

An unknown `AKITA_MODE` also stops the process. The error lists the modes that
were compiled into the current binary. A build with a narrow feature set can
contain only one profile family, so use the matching mode and feature. See
[Profiling](./profiling.md) for the canonical command and [Feature
flags](./feature-flags.md) for the available profile modes.

## The requested schedule is unsupported

`AkitaError::UnsupportedSchedule` means the catalog supplied to the scheme does
not contain a row for the exact request. Runtime code does not invoke the
planner to fill a missing row.

Check these causes in order:

1. Confirm that the application loaded the intended family artifact and that
   its catalog identity matches the setup or preprocessing package.
2. Confirm that the chosen preset supports the requested polynomial arity,
   group shape, and opening method.
3. For a recursive grouped opening, commit each precommitted group with the
   base configuration. Use the recursive configuration for the final grouped
   commitment and proof.
4. If the request should be supported, record the complete error text and the
   exact configuration, `num_vars`, polynomial counts, and ordered group
   layout. A catalog row may need to be generated and shipped.

See [Configuration and planning](../how/configuration.md) for catalog ownership
and [Precommitting under a recursive
configuration](./commitment-api.md#precommitting-under-a-recursive-configuration)
for the grouped setup pattern.

## Equality table allocation is rejected

Akita checks each materialized equality table allocation against a 1 GiB
budget. A request over that limit returns `AkitaError::InvalidInput` before
the allocation. The message includes the table label, element count, byte
count, limit, and source location.

Lower `num_vars` when testing a workload that does not need the larger domain.
For a production request, keep the complete error and configuration details.
Some protocol paths use split equality weights and do not materialize the full
table. Do not raise the limit as a first fix. It is an allocation safety
boundary, not a tuning knob.

This check covers equality tables only. The witness, setup matrix, retained NTT
entries, and proof workspace use additional memory. Use the RSS counters and
summary described in [Profiling](./profiling.md) to locate the phase where
memory grows.

## A setup cache cannot be loaded or saved

The `disk-persistence` feature stores the public setup matrix and setup prefix
registry. It does not store backend NTT caches.

Cache entries use versioned filenames, and prefix registry names include a
digest of the resolved schedule. Old, truncated, corrupt, or mismatched entries
are not accepted as current setup. Akita logs the load failure and regenerates
the setup. If saving the replacement fails, Akita logs a warning and continues
with the in-memory setup.

Akita chooses the cache directory from environment variables in this order:

1. If `LOCALAPPDATA` is set, use `$LOCALAPPDATA/akita`.
2. Otherwise, if `HOME/Library/Caches` exists, use
   `$HOME/Library/Caches/akita`.
3. Otherwise, if `HOME` is set, use `$HOME/.cache/akita`.

This order is independent of the operating system. If neither environment
variable is set, disk persistence has no cache path.

Normal upgrades do not require manual cache deletion. If every run reports a
write failure, check that the directory is writable or build without
`disk-persistence`. If every run rejects a newly written entry, preserve the
warning and affected cache file for diagnosis before removing that file.

## Thread counts do not match the profile request

The profile harness reads `AKITA_PROFILE_PROVE_THREADS` and
`AKITA_PROFILE_VERIFY_THREADS`. Each value falls back to `RAYON_NUM_THREADS`
when it is missing or invalid. The prover pool resolves first. A prover value
of `0` lets Rayon choose the global pool size. A verifier value of `0` reuses
that resolved prover size. A positive verifier value creates a separate pool
only when it differs from the prover size. The harness prints both resolved
counts at startup.

Set the profile variables before the process starts. Rayon global pool size is
fixed during initialization. When the `parallel` feature is disabled, prover
and verifier work is sequential and all reported thread counts are one.

## Verification rejects an artifact

A verifier error does not always mean the proof was wrong. Decode and shape
errors mean the public artifact was malformed. `AkitaError::InvalidProof`
means the well-formed proof failed a protocol check.

Confirm that prover and verifier use the same Akita revision, configuration,
transcript backend, transcript domain, ordered commitment groups, opening
points, and claimed evaluations. Akita does not preserve proof bytes across
revisions. Regenerate the proof and verifier setup after an upgrade.

## A Jolt recursion run fails

The Jolt host performs strict native decoding and Akita verification before it
starts guest preprocessing. Fix that preflight error first.

The guest output has three defined values:

| Value | Meaning |
|-------|---------|
| `0` | Akita verification succeeded |
| `1` | Input decoding or statement construction failed |
| `2` | The Akita verifier rejected the proof |

A guest panic is reported separately as `guest_panic`. Start a new schedule or
larger arity with `--trace-only` and confirm that it fits the guest trace limit
before running the full Jolt prover.

For a panic, temporarily change the guest attribute to
`backtrace = "dwarf"`, rebuild the guest, and run with
`JOLT_BACKTRACE=full`. The exact commands, cache paths, memory limits, and
current trace limit live in the
[`profile/akita-recursion` runbook](https://github.com/LayerZero-Labs/akita/blob/main/profile/akita-recursion/README.md).

## What to include in a bug report

Include the full error, Akita commit, Cargo feature set, configuration type,
ordered group layout, `num_vars`, polynomial counts, operating system, CPU
architecture, and resolved thread counts. For performance or memory failures,
also include the profile summary JSON and the command that produced it. Do not
attach private witness data or application inputs.
