# Setup and prepared compute state

Akita uses transparent public setup. The setup contains no secret trapdoor and
needs no ceremony. A deployment can generate it from the public setup seed,
store it for reuse, and derive a smaller verifier package for each supported
proof shape.

The API separates protocol identity from local compute state. This lets a host
cache expensive transforms without changing commitments or proof bytes.

## The four setup objects

| Object | What it means | Can local copies differ? |
| --- | --- | --- |
| `AkitaSetupSeed` | Identity of the deterministic public matrix stream | No |
| Prover setup capacity | How much of that stream the prover materialized | Yes, if every proof requirement is covered |
| Setup prefix registry | Public commitments used by recursive setup offloading | It must cover the same supported schedules |
| Prepared backend state | Local transformed matrix prefixes and compute resources | Yes |

The public matrix stream has one identity across ring dimensions. A generated
schedule takes the exact rows, widths, and dimensions needed by each operation
from a prefix of that stream.

## Choosing a setup identity

Setup is a deterministic function of the public seed and the requested shape, so
the seed is what makes two machines agree. A deployment picks one of two ways to
get it.

`AkitaSetupSeed::DEFAULT` is the stable protocol default. Deployments that do
not need their own identity share it, and it never changes.

`AkitaSetupSeed::from_rng` mints a deployment-specific identity. The caller
supplies the entropy source, so Akita never reaches for OS randomness itself and
the verifier's dependency graph stays free of it.

```rust
let setup_seed = AkitaSetupSeed::from_rng(&mut OsRng);
```

Sample once, at deployment time, and record the result. Do not sample on the
proving path: cache entries are keyed by seed digest, so a fresh seed always
misses its cache and regenerates the whole public matrix. The
`generate_setup` example in `akita-pcs` shows the intended shape — sample,
print the seed and its digest for recording, then build setup.

Both prover and verifier then read that recorded seed from their own trusted
configuration. Akita logs the seed when it builds setup, which gives operators
one value to compare across machines. No error names the seed, so a mismatch is
otherwise invisible until an algebraic check fails for no stated reason.

The seed is public but it is not proof data. A verifier must take it from
trusted configuration and never from a proof, because whoever chooses the seed
chooses the public matrix.

## Build prover setup once

The normal entry point takes the largest number of variables and largest group
size that the host plans to support, plus the setup identity to derive from.

```rust
let setup = AkitaCommitmentScheme::<Config>::setup_prover(
    max_num_vars,
    max_polynomials_in_one_group,
    setup_seed,
)?;
```

A setup carries its own identity, so code holding only the artifact can still
report which seed produced it:

```rust
let setup_seed = setup.setup_seed();
```

A larger covering setup can serve a smaller proof under the same public seed.
The commitment and proof bind the public setup identity, not the amount of
matrix data that one prover happened to store.

For a recursive configuration, setup construction also prepares the setup
prefix commitments required by its generated catalogs. The host should build
setup with the configuration that will produce the final proofs.

## Prepare the compute backend

The CPU backend turns public setup into reusable execution state.

```rust
let backend = CpuBackend::DEFAULT;
let prepared = backend.prepare_setup(&setup)?;
let stack = UniformProverStack::uniform(
    &backend,
    &prepared,
    setup.expanded.as_ref(),
)?;
```

The prepared state starts with empty transform caches. Commitment and proving
kernels build exact entries as needed. A larger cached prefix can serve a
smaller request with the same field, ring dimension, and transform domain.

Keep `backend`, `prepared`, and `stack` alive across repeated work. This makes
later commitments and proofs reuse the matrix transforms already built by the
first one.

## Build verifier setup

The simplest conversion keeps the complete materialized public matrix prefix.

```rust
let verifier_setup =
    AkitaCommitmentScheme::<Config>::setup_verifier(&setup)?;
```

This is a good starting point for local verification and the first integration.

A deployment that knows the exact proof schedule can derive a smaller verifier
setup:

```rust
let resolved = Config::resolve_schedule_selection(selection)?;
let opening_layout = verifier_claims.committed_layout()?;
let verifier_setup =
    AkitaCommitmentScheme::<Config>::setup_verifier_for_schedule(
        &setup,
        resolved.schedule(),
        &opening_layout,
    )?;
```

The narrowed setup retains the terminal matrix and every public matrix prefix
that verification still reads directly. A setup contribution carried by a
recursive setup proof does not need the same direct prefix in the verifier hot
path.

The complete setup prefix commitment registry remains part of verifier setup.
Those commitments authenticate the offloaded public setup contributions.

## Verifying on a separate machine

A verifier that never receives a setup artifact rebuilds it from the recorded
seed. There is no separate entry point: it constructs setup exactly as the
prover does, then narrows it.

```rust
let setup = AkitaCommitmentScheme::<Config>::setup_prover(
    max_num_vars,
    max_polynomials_in_one_group,
    configured_seed,
)?;
let verifier_setup =
    AkitaCommitmentScheme::<Config>::setup_verifier(&setup)?;
```

Two costs are worth knowing before choosing this over shipping a setup package.

Construction is a one-time startup cost. With `disk-persistence` the public
matrix and prefix registry are cached per seed, so later runs load instead of
rederiving.

For a recursive configuration, that first construction also commits to the
required setup prefixes. It runs a backend prepare and a commitment pass over
the prover-sized matrix before the narrowed verifier view is taken, so peak
memory during setup construction is prover-sized even though the retained
verifier setup is smaller. A deployment that cannot afford that peak should
distribute an authenticated verifier package instead.

## Reuse and release CPU caches

Prepared state stays warm by default. That is the right policy for a service
that proves many statements with the same setup.

Some hosts need to lower peak memory between the root commitment and later
folds. `ReleaseRootNttAfterFold` wraps a prover stack and releases large shared
matrix transform entries after the root fold. The entries rebuild when a later
proof needs them again.

The release policy changes local memory and compute time. It does not change
setup identity, commitment identity, or proof bytes. Measure the complete host
before choosing it.

## Disk persistence

The `disk-persistence` feature stores public matrix coefficients and setup
prefix artifacts. It does not store backend transform caches.

Stored setup entries include their public identity and provisioning limits.
Versioned filenames prevent an older setup format from being accepted as a
current entry. If a cache entry is missing or invalid, Akita can regenerate the
public data from the seed.

Cache identity includes the setup seed, so two seeds on one machine keep
separate entries and neither invalidates the other. Loading also rechecks the
seed recorded inside each file, so a stale or mismatched entry is rejected and
regenerated rather than used.

A deployment that distributes verifier setup should authenticate the package
or regenerate it from the public seed. This gives every verifier the same
public setup identity while allowing each machine to choose its own local
storage and compute caches.

## Setup offloading

Recursive setup offloading commits to an actual power of two prefix of the
public matrix stream. The prefix may be longer than the active setup weight.
The extra coefficients are still real public setup data, while the weight is
zero outside its natural support.

The planner chooses the commitment parameters for that prefix independently of
the fold that produced or consumed it. The
[setup offloading chapter](../how/setup-offloading.md) explains this lifecycle
from planning through verifier replay.

## Operational rule

Share protocol identity and choose compute state locally.

- Prover and verifier packages must agree on the setup seed and prefix
  commitments.
- The seed is trusted configuration on both sides. A verifier must never take it
  from a proof.
- A prover may materialize a larger covering matrix prefix.
- Each machine may build its own prepared transform caches.
- Cache release and persistence policies may follow the host's measured memory
  budget.

The [profiling guide](./profiling.md) reports setup size and prepared cache size
separately so these costs remain visible.
