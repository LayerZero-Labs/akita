# Preprocessing releases

An Akita verifier is only sound if the setup-prefix commitments it uses are
honest commitments to the public matrix. A **release** is how Akita makes
that checkable: a deterministic generator turns a short description into
every preprocessing file a prover, verifier, or guest needs, and a manifest
names each file by hash. A binary trusts a release through a **trust
anchor** compiled into it, and Akita's loaders check every file against the
manifest before any prover or verifier sees it.

[What the verifier trusts](../how/verification.md#pinned-releases) explains
the trust model and the loader's checks. This page is the operator's guide:
when a deployment needs its own release, how to write and generate one, how
to anchor and publish it, and how provers, verifiers, and guests load it.

## Terms

- **Parameter set**: the unit a loader builds. It fixes a compiled
  `CommitmentConfig` (field, decomposition, SIS profile), a schedule catalog,
  the witness chunking, and limits on the largest supported shape. Each set
  has a name, such as `fp128-dense`.
- **Profile**: the complete recipe of one setup-prefix commitment. A release
  stores one commitment per stream and profile; slot ids that differ only in
  their natural length share it.
- **Manifest**: the list of streams, commitments, set records, and files of
  a release, with a hash for each. Its hash is the release's `bundle_id`.
- **Trust anchor**: the `(bundle_id, set name)` pairs a binary accepts,
  written as compile-time constants.
- **Program rows**: grouped schedule rows an application plans for one
  program after the release is published. A set that admits them publishes a
  **profile universe**, the profiles program rows may use.

## When a deployment needs its own release

Akita ships one release, `params/akita-v1`, anchored by the `akita-anchors`
crate. Its sets cover Akita's own configurations with the default seed. A
deployment needs a release of its own when it uses:

- another seed;
- its own catalogs, or limits beyond Akita's sets;
- its own `CommitmentConfig` types, as Jolt does (see
  [Integrations](#integrations));
- program rows, which need a set that publishes a universe.

The release is generated with the same generator, and the deployment
compiles its own anchor. Production and custom deployments use the same
loaders; only the anchor differs.

## The description

A release starts from `params/<release>/description.json`, written by hand.
Akita's own description is a good template:

```json
{
  "format_version": 1,
  "name": "akita-v1",
  "kind": "akita",
  "generator_version": "akita-preprocess 0.1.0",
  "seed": "bebafecaefbeadde000000000000000000000000000000000000000000000000",
  "targets": ["riscv64"],
  "max_file_part": 1073741824,
  "hints": true,
  "sets": [
    {
      "name": "fp128-dense",
      "config": "fp128_dense",
      "catalog": "artifacts/schedules/fp128_dense.aks",
      "num_chunks": 1,
      "num_activated_levels": 0,
      "max_num_vars": 22,
      "max_num_batched_polys": 1
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `format_version` | Always `1`. |
| `name` | The release name. The description must live in `params/<name>/`. |
| `kind` | `"akita"` or `"jolt"`. A loader rejects a manifest of another kind. |
| `generator_version` | A label recorded in the manifest. |
| `seed` | 64 lowercase hex digits: the 32-byte seed of the public matrix stream. Every set of the release uses this seed. |
| `targets` | Guest targets to build files for. `riscv64` is the only one; the generator then writes each admitted row's selected-row catalog view and, for fp128 sets, its RISC-V terminal cache. |
| `max_file_part` | The largest stored part of one file, in bytes. Larger files are split into `<name>.part-0`, `<name>.part-1`, and so on. |
| `hints` | Also publish prover hints, so a CPU prover loads its setup-prefix slots instead of exporting them. |
| `sets[].name` | The set name binaries anchor. |
| `sets[].config` | The `schedule_family_name()` of a compiled configuration, such as `fp128_dense_recursive`. |
| `sets[].catalog` | The catalog artifact, relative to the repository root. |
| `sets[].num_chunks`, `sets[].num_activated_levels` | The witness chunking the catalog was planned for. |
| `sets[].max_num_vars`, `sets[].max_num_batched_polys` | The set limits: rows within them are admitted. They are upper bounds, not what a loader builds. |
| `sets[].program_rows.universe_slots` | Optional. Files listing setup-prefix slot ids (one hex-encoded uncompressed id per line, `#` comments allowed); their profiles join the catalog's in the set's universe, and the set admits program rows. |

## Generate a release

```bash
cargo run --release -p akita-preprocess -- \
  generate params/akita-v1 out/akita-v1 --write-manifest
```

`generate` derives each stream's matrix once, at the largest capacity any
set or profile needs, commits every profile with the CPU reference path, and
writes the files and the manifest to the output directory. It prints the
release name and `bundle_id`. `--write-manifest` also writes the manifest
back to `params/<release>/manifest`, the file the repository checks in.

The output depends only on the description, the code, and the digest-hash
selection (`keccak256-digests` selects Keccak-256 instead of SHA3-256): not
on thread count, machine, or instruction set, and never on a cache or an
earlier bundle. `--repo-root DIR` resolves catalog paths against another
directory.

`--role` selects which files to write; every role computes the same
`bundle_id`.

| Role | Files written |
| --- | --- |
| `all` (default) | Every file. |
| `prover` | Registry payloads, prover hints, catalogs. |
| `verifier` | Registry payloads and catalogs. |
| a target name, such as `riscv64` | Registry payloads, catalog views, and that target's terminal caches. |

The files are named by kind:

| File | Content |
| --- | --- |
| `registry/<stream>/<profile>` | A setup-prefix commitment. |
| `hints/<stream>/<profile>` | The CPU prover's hint for that commitment. |
| `catalog/<set>` | The set's catalog artifact. |
| `catalog-view/<set>/<row>` | A selected-row view of the catalog, for a guest. |
| `terminal-cache/<stream>/<row>/<target>/<capacity>` | A guest's prepared terminal cache. |

## Anchor a release

```bash
cargo run --release -p akita-preprocess -- anchor params/akita-v1
```

prints one `AnchorEntry::new("<bundle_id>", "<set>"),` line per set. Paste
them into the anchor crate's `TrustAnchor::new(&[...])` constant;
`crates/akita-anchors/src/lib.rs` shows the layout, including the manifest
compiled in with `include_bytes!`. An anchor crate must stay outside the
generator's dependency closure, so adding a release never changes a
`bundle_id`; `scripts/check-crate-deps.sh` and
`scripts/check-trust-anchors.sh` enforce this for Akita.

A release is published by one commit that adds its description and manifest
and its anchor pairs. The large files are untrusted transport: upload them
anywhere, for example as release assets, because a loader checks each one
against the manifest.

## Load a release

Every load starts by opening the manifest under the binary's anchor and
naming the sets it will use:

```rust
use akita_pcs::preprocessing::{open_anchored_manifest, DirectoryReleaseFiles, ReleaseKind};

let release = open_anchored_manifest(
    &akita_anchors::AKITA_ANCHOR,
    ReleaseKind::Akita,
    akita_anchors::AKITA_V1_MANIFEST,
    &["fp128-dense-recursive"],
)?;
let files = DirectoryReleaseFiles::new("release/akita-v1");
```

Any `ReleaseFileSource` serves the files: `DirectoryReleaseFiles` reads a
directory written by `generate` (and reassembles `.part-N` files),
`MemoryReleaseFiles` serves bytes in memory, and with the `disk-persistence`
feature `akita_setup::ReleaseCache` keeps downloaded files keyed by
`bundle_id`. A loader reads each file once, into a private buffer, and checks
its hash before decoding it.

**Prover.** `load_prover_setup::<Cfg>(&release, &files, set, program)`
admits the set's rows, derives the matrix for them, and loads each
setup-prefix slot from its registry file and hint, or, when the release has
no hints, exports it from the derived matrix and compares it with the
manifest. It returns a `PinnedProverSetup`, whose `into_parts` gives the
prover setup and catalog.

**Verifier.** `load_verifier_setup::<Cfg>(&release, &files, set, selection,
program, source)` returns a `PinnedVerifierSetup` for every admitted row, or
for the one row `selection` names; `AkitaVerifier::from_pinned` builds the
verifier. `source` chooses where commitments come from:

- `PrefixCommitmentSource::Registry` reads registry files;
- `PrefixCommitmentSource::SeedDerived(&CpuPrefixDeriver)` derives each
  commitment from the seed instead (hosts only; it reads no registry or
  hint file and costs one derivation per profile). A prover setup of the
  same set is also a deriver, so an in-process verifier can take the
  commitments its prover already checked.

Either way every commitment must match its manifest digest.

**Guest.** A recursion guest embeds its files in an approved image and
never reads preprocessing from its input.
`load_guest_verifier_setup::<Cfg>(&release, &files, set, selection,
program, matrix)` is the same single-row load over the embedded registry
files and catalog view, except that it takes the embedded matrix prefix,
issued with the guest-only `PinnedMatrixPrefix::embedded_unchecked`. The
row's terminal cache comes from `load_terminal_cache` or, embedded, from
`PinnedTerminalCache::embedded_unchecked`;
`AkitaVerifier::from_pinned_with_terminal_cache` uses it. The two
`embedded_unchecked` issuers are `unsafe fn`s: their contract is that the
bytes are embedded in an approved image, and an integration denies them
outside its guest crate with `clippy::disallowed_methods`.

## Program bounds and program rows

Every loader takes an optional `ProgramRows` argument, `program`. Its
bounds (`max_num_vars`, `max_num_batched_polys`) select the rows a
loader admits within the set limits, so a small program never derives the
matrix of the set's largest row. For a set without program rows, `rows`
must be empty.

For a set that admits program rows, `rows` holds the grouped rows one
program planned for itself. They arrive as an argument, authenticated by
the program's own preprocessing digest, so an application stores them, with
the release's `bundle_id` and the merged catalog digest, in its program
preprocessing. The loader merges them with the base catalog into the
**merged catalog** and requires:

- every setup-prefix profile a program row uses to be in the set's
  universe;
- every precommitted group of a program row to carry the frozen profile of
  its layout in the producer set the row names, which the application loads
  into `ProgramRows::producers` with `ProducerProfiles::load::<Cfg>` from the
  same release;
- the merged catalog to have the digest stored in `merged_catalog_digest`,
  when given, so a host and a single-row guest bind the same catalog.

To plan rows that pass, restrict the planner to the universe: pass the
universe as the `setup_prefix_profiles` filter of `find_adapted_schedule`,
and take each precommitted group's profile from
`ProducerProfiles::frozen_profile`.

## The merge check

```bash
cargo run --release -p akita-preprocess -- check
```

regenerates every release under `params/` from scratch and requires its
manifest to equal the checked-in one byte for byte. CI runs it whenever a
description, manifest, anchor, `Cargo.lock`, `rust-toolchain.toml`, or any
crate in the generator's dependency closure changes
(`.github/workflows/preprocessing-releases.yml`). It is also the audit mode:
anyone can run it.

`check` also requires every profile a release publishes to fall in a class
of the **known-answer corpus**, `crates/akita-preprocess/corpus/known-answer-v1.txt`.
The corpus pins the behaviour of the commitment code on small profiles; its
digest, one per field, is compiled into `akita-types` and recomputed in CI on
every supported instruction set (`akita-preprocess known-answer` prints the
current values). When a release publishes a profile of a new class, `check`
fails and prints the corpus lines to append. Appending them changes the
known-answer digest of that field, and with it every prefix commitment
digest of that field, so it is a code transition: update the compiled
digests, regenerate every affected release, and replace its anchor pairs.

## Transitions

| Change | What to do |
| --- | --- |
| Description only: sets, catalogs, limits, universes | Generate a new release under a new name. Its existing commitments are byte-identical. Add its pairs to the anchor; remove the old pairs one software version later. |
| Code that changes commitment digests, catalog bytes, or row digests | Regenerate every release; the old manifests no longer regenerate. Replace the anchor pairs in the same change. |
| Seed | Every commitment changes. Generate a release with the new seed and anchor it, alongside or instead of the old one. |

A proof stays verifiable under a new release when everything its transcript
binds is unchanged: the setup section (seed digest, decomposition, SIS
profile, protocol features), the selected row, and the prefix commitment
digests protocol Stage 3 absorbs. A release with a custom seed works with
the same loaders and its own anchor.

## Integrations

An application with its own `CommitmentConfig` types builds its own
generator binary on the `akita-preprocess` library. It implements
`ConfigRegistry`, which maps a description's `config` name to a compiled
configuration by calling `plan_set::<Cfg>`, and runs the same command line
with `akita_preprocess::cli_main(&its_registry)`. `generate` and
`check_release` take the registry as their first argument; `AkitaConfigs`
covers Akita's own configurations.

A custom host deployment whose program rows need profiles outside every
anchored universe can opt into **local profile derivation**:
`Admission::admit_any_profile` admits such rows, and
`akita_preprocess::derive_local_verifier_setup` derives the verifier's
matrix and every commitment from the seed in the verifier's own process. No
manifest digest checks those commitments, so only the `unsafe`
`AkitaVerifier::from_locally_derived` accepts the result, and it is never
available in a guest.
