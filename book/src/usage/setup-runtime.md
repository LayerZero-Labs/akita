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
| Setup prefix registry | Public commitments used by recursive setup offloading, published in a pinned release | No: every entry must match the anchored manifest |
| Prepared backend state | Local transformed matrix prefixes and compute resources | Yes |

The public matrix stream has one identity across ring dimensions. A trusted
schedule row takes the exact widths and dimensions needed by each operation from
a prefix of that stream.

The current identity is `Shake256PagedV1`: independently derived 4096-element
SHAKE256 pages followed by exact field rejection sampling. [Setup and
commitment](../how/commitment.md#exact-public-stream-derivation) gives the byte
encoding. A host that regenerates setup must use that versioned derivation,
not an application-specific PRG expansion of the 32-byte seed.

## Build prover setup once

The normal entry point takes the largest number of variables and largest total
polynomial count across the final and all precommitted groups in one opening
batch that the host plans to support.

```rust
let catalog = TrustedScheduleCatalog::<Config>::from_artifact_bytes(&artifact_bytes, akita_params::ChunkedWitnessCfg::default_non_chunked())?;
let scheme = AkitaCommitmentScheme::<Config>::new(catalog);
let setup = scheme.setup_prover(
    max_num_vars,
    max_total_batched_polys,
)?;
```

A larger covering setup can serve a smaller proof under the same public seed.
Of the setup, the commitment and proof bind only the seed, not the amount of
matrix data that one prover happened to store.

For a recursive configuration, setup construction also prepares the setup
prefix commitments required by its supplied catalog. The host should build
setup with the configuration that will produce the final proofs.

## Share one setup across families

Configurations with the same field can share one setup, and those that also
share the extension field can share one backend. Compute each catalog's
requirements at the same bound, combine them, and build the setup from the
combined requirements:

```rust
let requirements = SetupRequirements::from_catalog::<DenseConfig>(
    dense_scheme.schedules(),
    max_num_vars,
    max_total_batched_polys,
)?
.union(SetupRequirements::from_catalog::<OneHotConfig>(
    onehot_scheme.schedules(),
    max_num_vars,
    max_total_batched_polys,
)?)?;
let setup = akita_pcs::new_prover_setup::<Field>(&requirements)?;
```

The combined setup uses the same public seed as each per-family setup, so a
proof produced with it is byte-identical to one produced with the smaller
per-family setup. The transcript's instance descriptor binds the field
algebra, the family's decomposition and SIS profile, the setup seed, the
selected schedule row, the grinding plan, and the call layout. It binds no
provisioned capacity: neither the setup's variable and polynomial bounds nor
its materialized matrix length. Commitments read the same seed-derived matrix
prefix under either setup. Each commitment call uses its producer catalog.
Each opening selects a row from, and is admitted against, its opening catalog.

`SetupRequirements<F>` is tied to the field and has no public constructor
other than `from_catalog` and `union`, so every requirement passed to
`new_prover_setup` comes from validated catalog metadata.

## Prepare the compute backend

The CPU backend turns public setup into reusable execution state.

```rust
let backend = std::sync::Arc::new(CpuBackend::new(setup.expanded.clone())?);
```

The prepared state starts with empty transform caches. Commitment and proving
kernels build exact entries as needed. A larger cached prefix can serve a
smaller request with the same field, ring dimension, and transform domain.

These caches are derived from a matrix view. They are not part of the public
stream and are not covered by `AkitaSetupSeed`. Two machines may retain
different transform domains and prefix lengths while producing identical
commitments and proofs.

Keep the shared `backend` alive across repeated work. It owns prepared setup
resources and caches. Reusable commitment handles retain their immutable sources,
while every proof creates separate temporary state. Finishing or abandoning one
proof leaves other proofs and commitments valid.

## Build verifier setup

A production verifier loads its setup from a pinned release that its compiled
trust anchor names; see
[Verifier only integration](./verifier-only.md#supply-verifier-setup). A
prover of the same parameter set loads its setup with `load_prover_setup`,
which admits the same rows and checks every commitment against the same
manifest.

For local verification of proofs a process made itself, the scheme can convert
a prover setup it derived. These conversions are `unsafe fn`s: their contract
is that the caller derived every setup-prefix commitment, which holds for a
setup built in the same process and never for one read from storage.

```rust
// SAFETY: `setup` was derived from its seed in this process.
let verifier_setup = unsafe { scheme.setup_verifier(&setup) }?;
```

A local check that knows the exact proof schedule can derive a smaller
verifier setup:

```rust
let resolved = scheme.schedules().resolve_selection(selection)?;
let opening_layout = verifier_claims.committed_layout()?;
// SAFETY: as above.
let verifier_setup = unsafe {
    scheme.setup_verifier_for_schedule(&setup, resolved.schedule(), &opening_layout)
}?;
```

The narrowed setup retains the terminal matrix and every public matrix prefix
that verification still reads directly. A setup contribution carried by a
recursive setup proof does not need the same direct prefix in the verifier hot
path. The release loader sizes its matrix the same way, from the rows it
admits.

Every setup-prefix commitment a proof uses remains part of verifier setup.
Those commitments authenticate the offloaded public setup contributions, and
protocol Stage 3 binds each one by its prefix commitment digest.

## Reuse and release CPU caches

Prepared state stays warm by default. That is the right policy for a service
that proves many statements with the same setup.

Applications can call `backend.trim_caches()` to release reusable transform
entries. Active operations retain the storage they still need. Proof completion
releases proof-owned temporary state while shared caches may survive for later
proofs. Cache entries rebuild lazily when needed.

The release policy changes local memory and compute time. It does not change
setup identity, commitment identity, or proof bytes. Measure the complete host
before choosing it.

## Disk persistence

The `disk-persistence` feature enables `akita_setup::ReleaseCache`, a disk
cache of pinned release files keyed by `bundle_id`. It does not store backend
transform caches or locally derived setups.

The cache is untrusted transport. `ReleaseCache::manifest` returns a manifest
only when its bytes hash to the requested `bundle_id`, and every file read
through `ReleaseCache::files` is checked by a loader against that manifest. A
corrupted or planted file is rejected rather than trusted, so a cached setup
can no longer bypass setup validation.

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

- Prover and verifier load the same anchored release, so they agree on the
  setup seed and prefix commitments.
- A prover may materialize a larger covering matrix prefix.
- Each machine may build its own prepared transform caches.
- Cache release and persistence policies may follow the host's measured memory
  budget.

The [profiling guide](./profiling.md) reports setup size and prepared cache size
separately so these costs remain visible.
