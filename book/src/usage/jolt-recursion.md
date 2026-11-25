# Jolt recursion

[Jolt](https://jolt.a16zcrypto.com/) is a zero-knowledge virtual machine that
proves the correct execution of 64-bit RISC-V programs. Akita's Jolt
integration turns the Akita verifier into one of those programs. Jolt can
therefore produce an outer proof that says an Akita opening proof was accepted.

This is recursive verification: one proof system proves the execution of
another proof system's verifier. It lets a host combine Akita's compact,
post-quantum polynomial commitments with the execution and composition tools
provided by a zkVM.

The live integration is the standalone workspace under
`profile/akita-recursion/`. Its
[runbook](../../../profile/akita-recursion/README.md)
contains current commands, environment variables, and measured trace notes.

## The complete path

The integration has four main crates. The workspace also holds two
`inline-parity` crates, which this chapter does not use.

| Crate | Runs where | Responsibility |
| --- | --- | --- |
| `artifact` | Native host | Creates a real Akita commitment, opening proof, and verifier bundle |
| `glue` | Host and guest | Defines the bounded `AkitaJoltInputs` wire format |
| `guest` | Jolt RISC-V guest | Decodes the bundle and calls `AkitaVerifier::batched_verify` |
| `host` | Native host | Compiles the guest, runs Jolt, and checks the outer proof |

The data flow is direct:

```text
Akita artifact generator
        │
        │ AkitaJoltInputs bytes
        ▼
strict native Akita verification
        │
        ▼
Jolt RISC-V guest
        │ decode input
        │ rebuild statement
        │ run Akita verifier
        ▼
guest result: accepted or rejected
        │
        ▼
Jolt proves and verifies the guest execution
```

Before compiling or proving the guest, the host strictly decodes the bundle
and runs the native Akita verifier. This makes failures easy to locate. If the
native check fails, the public statement or Akita proof is wrong. If it passes,
the remaining work belongs to guest compilation, execution, or the Jolt prover.

## What crosses into the guest

`AkitaJoltInputs<F, D>` is one versioned verifier bundle. It contains:

- the transcript domain;
- the polynomial arity and opening point;
- the claimed value and commitment;
- the exact generated schedule selection;
- the verifier setup;
- the canonical native Akita proof bytes.

The decoder checks a format marker, the fixed source-view dimension, bounded
lengths, complete consumption of the input, and a schedule-derived upper bound
on the proof stream before allocation. The guest then rebuilds the same
opening statement used by the native verifier.

The guest depends on `akita-verifier` rather than the complete PCS package. It
does not carry the polynomial backend, setup generator, or planner into the
RISC-V program.

## The current configurations

The harness runs a fixed list of cases. A case fixes the field, the polynomial
arity, the polynomial count, the setup mode, and the root envelope. The root
envelope is the source-view dimension $D$ that the guest is compiled with. The
generated schedule still chooses the dimensions used at each later fold.

| Case | Field | Variables | Setup mode | Root envelope |
| --- | --- | --- | --- | --- |
| `onehot_fp32:30:1` | fp32 with degree four extension claims | 30 | direct | $D = 2048$ |
| `onehot_fp64:30:1` | fp64 with degree two extension claims | 30 | direct | $D = 512$ |
| `onehot_fp128:36:1:direct` | fp128 | 36 | direct | $D = 512$ |
| `onehot_fp128:36:1:recursive` | fp128 | 36 | recursive | $D = 512$ |
| `onehot_fp128_multi_group_recursive:32:4:recursive` | fp128 | 32 | recursive | $D = 512$ |

The first four cases each open one structured one hot polynomial. Their names
use the case syntax of the profile harness,
`MODE:NUM_VARS:NUM_POLYS[:SETUP_MODE]`. The last case opens several
commitment groups in one proof. It is the default when no case is selected.

Select a case with the `--case` flag of the artifact generator, or with the
`AKITA_RECURSION_CASE` environment variable. A case does not accept separate
overrides for its field, arity, or setup mode. `AKITA_NUM_VARS` applies only to
the default grouped case, and that case accepts only 32.

The verifier bundle stores the case identity. The guest declares one Jolt
entrypoint for each case, and the host dispatches to the entrypoint that
matches the bundle. A bundle cannot be replayed through the entrypoint of a
different case.

The recursion workspace is separate from the main Cargo workspace. It pins
Rust 1.99.1, the RISC-V targets, and one exact Jolt revision. This keeps Jolt's
toolchain and patched dependencies out of the main Akita dependency graph.

## Run the native artifact and guest trace

Install the Jolt CLI from the same revision pinned by the recursion workspace.
Then run these commands from `profile/akita-recursion/`:

```bash
cargo build --release

AKITA_RECURSION_BLOB=target/onehot_fp128_nv36_recursive.bin \
    ./target/release/akita-recursion-artifact \
    --case onehot_fp128:36:1:recursive

ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info \
    ./target/release/akita-recursion-host \
    --trace-only \
    --trace-output /dev/null \
    --input target/onehot_fp128_nv36_recursive.bin
```

`AKITA_RECURSION_BLOB` names the file that the artifact generator writes. The
host reads the same file through `--input`. Both default to
`target/akita_recursion_inputs.bin`.

The artifact command proves and verifies natively before publishing the input
blob. The trace command compiles and executes the guest without running the
full Jolt prover. It is the fastest way to confirm that a new Akita revision or
a new case fits the guest.

The guest reports three cycle regions:

| Marker | Work measured |
| --- | --- |
| `deserialize_input` | Decode and validate the verifier bundle |
| `prepare_verifier` | Admit the selected row and install or prepare its terminal matrix |
| `akita_verify` | Build the public statement and run the Akita verifier |

At large arities, decoding the expanded verifier setup can dominate the guest
trace. Measuring it separately makes that transport cost visible instead of
attributing it to the Akita verifier.

## Run the full recursive proof

The full pipeline runs only when the guest trace fits Jolt's current limit of
$2^{32}$ instructions. Measure the selected case with `--trace-only` first.
When the trace length is under the limit, run the same case without
`--trace-only`:

```bash
./target/release/akita-recursion-artifact --case onehot_fp64:30:1

ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info \
    ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin
```

The full command performs Jolt preprocessing, proves the guest execution, and
verifies the resulting Jolt proof. Success ends with
`Akita-in-Jolt proof OK`, `is_valid=true`, and `guest_panic=false`.

The larger cases are intended for a server with enough memory for their input
and trace.

## Rejection is part of the proved execution

The guest returns a small status value:

| Value | Meaning |
| --- | --- |
| `0` | Akita verification succeeded |
| `1` | Input decoding or statement construction failed |
| `2` | The Akita verifier rejected the proof |

Malformed public input produces a defined nonzero guest result. Jolt proves
that result in the same way it proves a successful result. A guest panic is
reported separately.

## Trusted benchmark setup

The host always performs strict setup decoding and Akita verification before
the guest runs. For cycle measurement, the RISC-V benchmark guest then uses the
already checked expanded setup matrix directly. This keeps the benchmark
focused on transport and verifier execution.

A deployment can preserve the same trust boundary by authenticating the setup
package outside the guest. Strict setup decoding inside the guest re-derives
the public matrix from the seed, so it removes the need to trust the matrix
bytes. It does not recompute the setup-prefix commitments, and it does not
establish where the schedule catalog came from. Those remain inputs that the
deployment must authenticate, for example through the guest program or its
public input. Authenticating them fixes the bytes. It still relies on whoever
derived them. The normal guest
feature keeps strict decoding enabled; the host opts into the benchmark path
only for the ELF it builds through the pinned Jolt SDK.

## Keep the integration current

The smoke workflow tests the glue and compiles the artifact and host against
the current Akita APIs. It does not run the expensive Jolt trace or proof. After an
Akita or Jolt upgrade:

1. Run the recursion workspace tests and release build.
2. Generate a fresh artifact and confirm native verification.
3. Run `--trace-only` and compare the three cycle markers.
4. Run the full Jolt proof for a case whose trace fits the limit.
5. Measure the intended production case on its deployment class machine.

Pin the complete integration to one Akita revision. Regenerate the verifier
bundle after every protocol upgrade because setup, schedule identity, proof
shape, and proof encoding advance together.
