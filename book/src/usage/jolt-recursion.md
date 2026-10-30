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

The integration has four small crates.

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

## The current configuration

The harness uses `proof_optimized::fp128::OneHot`. Its input is one structured
one hot polynomial, and the guest is compiled with a source-view dimension of
$D = 256$. This fixed type is the root envelope for the current adaptive
configuration. The generated schedule still chooses the dimensions used at
each later fold.

`AKITA_NUM_VARS` chooses the polynomial arity. The artifact generator resolves
the approved row for that exact arity and rejects a row that does not match the
guest's fixed source view. The default is 20 variables. The 32 variable target
uses the same $D = 256$ source view and is the large recursion benchmark.

The recursion workspace is separate from the main Cargo workspace. It pins
Rust 1.95, the RISC-V targets, and one exact Jolt revision. This keeps Jolt's
toolchain and patched dependencies out of the main Akita dependency graph.

## Run the native artifact and guest trace

Install the Jolt CLI from the same revision pinned by the recursion workspace.
Then run these commands from `profile/akita-recursion/`:

```bash
cargo build --release

AKITA_NUM_VARS=20 \
    ./target/release/akita-recursion-artifact

ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info \
    ./target/release/akita-recursion-host \
    --trace-only \
    --trace-output /dev/null
```

The artifact command proves and verifies natively before publishing the input
blob. The trace command compiles and executes the guest without running the
full Jolt prover. It is the fastest way to confirm that a new Akita revision or
larger arity fits the guest.

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

Begin with the 20 variable target, whose trace fits the guest's current
$2^{32}$ instruction limit:

```bash
AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact

ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info \
    ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin
```

The full command performs Jolt preprocessing, proves the guest execution, and
verifies the resulting Jolt proof. Success ends with
`Akita-in-Jolt proof OK`, `is_valid=true`, and `guest_panic=false`.

The 32 variable target is intended for a server with enough memory for its
large input and trace. Measure it with `--trace-only` first. The current
follow-up is to record its adaptive trace length and raise Jolt's trace limit
if that measured execution exceeds $2^{32}$ instructions.

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

This profiling harness passes the verifier setup through guest input, so it is
outside the guarantees of pinned releases: it measures transport and verifier
execution, not a production trust boundary. The host performs strict setup
decoding and Akita verification before the guest runs, and the benchmark guest
then uses the already checked expanded setup matrix directly.

A production guest reads no preprocessing from its input. It embeds the trust
anchor, the release manifest with its set record, the set's registry entries,
the catalog view of the row it verifies, the matrix prefix that row reads, and
the row's terminal caches. It loads them with `load_guest_verifier_setup`, a
single-row load that checks the anchor, the set record, and the embedded
release files like a host loader, but takes the embedded matrix prefix instead
of deriving the matrix from the seed.

Hashing the matrix prefix and terminal caches would cost the guest too many
cycles, so it issues them through the guest-only `unsafe fn`s
`PinnedMatrixPrefix::embedded_unchecked` and
`PinnedTerminalCache::embedded_unchecked`. Their soundness contract is that
the embedded bytes are the matrix prefix derived from the release's seed and
the terminal caches the release determines. A guest crate denies every other
unchecked constructor with `clippy::disallowed_methods`, for example:

```toml
disallowed-methods = [
  { path = "akita_types::AkitaVerifierSetup::from_parts", reason = "build verifiers with the release loader" },
  { path = "akita_verifier::AkitaVerifier::new", reason = "use AkitaVerifier::from_pinned" },
  { path = "akita_verifier::AkitaVerifier::for_selection", reason = "use AkitaVerifier::from_pinned" },
]
```

## Keep the integration current

The smoke workflow compiles and tests the glue, artifact, and host against the
current Akita APIs. It does not run the expensive Jolt trace or proof. After an
Akita or Jolt upgrade:

1. Run the recursion workspace tests and release build.
2. Generate a fresh artifact and confirm native verification.
3. Run `--trace-only` and compare the three cycle markers.
4. Run the full Jolt proof at a known arity.
5. Measure the intended production arity on its deployment class machine.

Pin the complete integration to one Akita revision. Regenerate the verifier
bundle after every protocol upgrade because setup, schedule identity, proof
shape, and proof encoding advance together.
