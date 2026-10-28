# Spec: Pinned Preprocessing Releases

| Field         | Value |
|---------------|-------|
| Author(s)     | sumchecker |
| Created       | 2026-10-26 |
| Status        | proposed |
| PR            | #287 |
| Supersedes    | The cache-loading policy question of #210 |
| Superseded-by | |
| Book-chapter  | |

## Summary

An Akita verifier is only sound if its setup-prefix commitments are honest
commitments to the public matrix. Today nothing checks that. The commitment
registry is accepted after structural checks only
(`AkitaVerifierSetup::from_parts`, setup deserialization, the `akita-setup`
disk cache, `AkitaProverSetup::to_verifier_setup`), a supplied terminal NTT
cache is accepted after identity and shape checks
(`AkitaVerifier::for_selection`), and the recursion guest reads its catalog,
matrix, and registry from untrusted input. Whoever can replace a commitment
can open it to any vector.

This spec turns preprocessing into a published **release** with a single
identity, `bundle_id`. A generator produces the release deterministically
from a description in the repository. A binary trusts only the
`(bundle_id, set name)` pairs compiled into it, and a loader checks every
file it uses against the release manifest before anything reaches a prover
or verifier. A required merge check regenerates every published release from
scratch, so a published release is always what the code at that commit
produces.

The key words MUST, MUST NOT, SHOULD, and MAY are used as described in
RFC 2119.

## Intent

### Goal

```text
release description ──generator──▶ release files + manifest
                                          │
                           bundle_id = H(tag || manifest)
                                          │
binary: trust anchor {(bundle_id, set name)}
                                          │
                     loader: anchor check, file hashes, commitment digests
                                          │
                     PinnedVerifierSetup / PinnedTerminalCache / PinnedProverSetup
                                          │
                     protocol Stage 3 binds the commitment digest of each prefix used
```

The generator is the only component that computes setup-prefix commitments.
Provers, verifiers, and guests load them, check them against a manifest
their binary trusts, and construct their setups only from what the loader
issues. The rest of this section defines what a release contains, how each
part is hashed, how a loader checks it, and how releases change over time.

### Threat model

The adversary can replace any preprocessing bytes that a prover or verifier
reads at run time, from storage or a network, including while they are being
read: setup, registry, cache, and catalog files, release assets, and the
input of a recursion guest.

The adversary cannot modify the build environment, the source tree at the
source commit, the resulting executables, the trust anchor compiled into an
integration, an approved guest image, or the program preprocessing digest a
Jolt deployment trusts. Anyone who controls the build can change the
verifier's code, and no data check defends against that.

| Object replaced | Effect at the verifier | Effect at the prover |
|---|---|---|
| Setup-prefix commitment | Soundness break: the adversary opens it to any vector | Honest proofs fail |
| Public matrix, on a path that skips re-derivation (`borrow_trusted`, `from_trusted_seed_derived_parts_unchecked`, `Validate::No`) | Soundness break: a matrix with a known short vector breaks binding | Honest proofs fail; may increase what the prover discloses |
| Terminal NTT cache | As the matrix it transforms | Not used |
| Catalog, program rows, or the chosen parameter set | Policy failure: rows still pass the admission check and use only anchored commitments, but the deployment did not approve them | Honest proofs fail; a hostile row may send more of the witness in clear |
| Prover hints | Not used | Honest proofs fail |

Akita does not claim witness confidentiality, and this spec does not add it.
Prover inputs are still checked, because hostile preprocessing can change
what an honest prover discloses before any verifier rejects the proof.

### Release structure

A release has three parts: the **description** a maintainer writes, the
**files** the generator produces from it, and the **manifest** that names
every file by hash. The trust anchor and Jolt's program preprocessing are not
part of the release; they point at it.

```text
release                                  bundle_id = H(tag || manifest)
├─ description                           written by hand, in params/<release>/
│   ├─ seed                              fixes the public matrix
│   ├─ packaging header                  name, kind (akita|jolt), generator version
│   └─ parameter set "jolt-onehot-k16-w1r0"   one or more, each named
│       ├─ CommitmentConfig              how commitments are computed and sized
│       ├─ catalog                       518 pre-planned rows, one per shape
│       ├─ chunking policy
│       ├─ limits                        max_num_vars, max_num_batched_polys
│       └─ profile universe              ~36 commitment profiles (program-row sets)
├─ files                                 written by the generator
│   ├─ registry/<stream>/<profile>       one commitment per profile any set needs
│   ├─ hints/<stream>/<profile>          prover opening material for it
│   ├─ catalog/<set>, catalog-view/…     catalogs and guest views
│   └─ optional: matrix/…, page-hashes/…, terminal-cache/…
└─ manifest
    ├─ packaging header, stream identities
    ├─ registry: (stream_id, profile_key, commitment_digest) per commitment
    ├─ set records: one per parameter set
    └─ files: (name, length, file_hash) per file

binary
└─ trust anchor                          (bundle_id, set name) pairs, compiled in

Jolt ProgramPreprocessing                covered by preprocessing_digest
├─ program commitments
├─ bundle_id, set names
└─ program rows                          grouped rows planned for this program
```

**Parameter set.** The unit a loader builds. It fixes how one kind of
commitment is computed (`CommitmentConfig`: field, decomposition, SIS modulus
profile and tables), which shapes are supported (the catalog), how witnesses
are chunked, and the largest supported shapes (the limits). A loader builds
one parameter set at a time, by name, and only if `(bundle_id, set name)` is
in its anchor.

**Catalog and schedule rows.** A schedule row says how polynomials of one
shape are committed and opened: the full fold schedule, and which levels use
which setup-prefix commitment. Planning a row is expensive and
security-relevant, so catalogs are planned offline and checked in as the
existing schedule artifacts. A release names each catalog by
`catalog_digest()`. Catalogs are enumerated: no row is added after
publication. `PlanSection` binds the row a proof uses.

**Limits and program bounds.** The limits are upper bounds, not what gets
built. A loader sizes the setup from the rows it admits (Loading), so a small
program never derives the matrix of the largest one. For Jolt, the program
bounds (its `max_num_vars`, polynomial counts, and largest precommitted
arity) select the rows and MUST fit within the limits.

**Commitment profile.** The complete recipe of one setup-prefix commitment:
group shape, block layout, inner and outer matrix parameters, and route. A
commitment depends only on the stream, the commitment algorithm, and the
profile. The `natural_len` of a `SetupPrefixSlotId` is validated against the
profile but does not change the commitment, because the committed source is
always the full prefix `S[0..n_prefix]`. A release therefore stores one
commitment per `(stream, profile)`, and every slot id that differs only in
`natural_len` shares it.

**Profile universe and program rows.** Jolt's grouped rows depend on each
program's shape and cannot be published (Jolt). Each program plans its own
rows, called program rows. Those rows still need commitments that only the
generator may compute, so a parameter set that admits program rows declares
a profile universe: the profiles program rows may use. The universe contains
every profile the set's catalog uses, and the release publishes a commitment
for each. A program row that needs any other profile is rejected.

**Commitment registry.** All commitments in a release, one per
`(stream, profile)`. Today's verifier registry is keyed by the full
`SetupPrefixSlotId`; a loader expands each registry entry into every slot id
its admitted rows require, keeps the existing `natural_len` validation, and
keeps the full slot id for transcript binding.

**What changes between releases.** Production releases normally keep the
same seed and commitment algorithm. Commitments for existing profiles are
then byte-identical across releases, and only parameter sets, catalogs,
limits, and universes change. A new seed changes every commitment (Anchors
and transitions).

### Encodings and hashes

Every hash is computed over fully specified bytes, and every value has
exactly one valid encoding: a decoder MUST reject input that does not
re-encode to the same bytes. The encodings reuse what the code already
writes wherever it can.

```text
H(x)       Blake2b-256, unkeyed, 32 bytes (the existing digest_descriptor_bytes)
u8, u32,   little-endian, as push_u32 and push_usize
u64
bytes(x)   u64(len(x)) || x
str(s)     bytes(UTF-8 of s)
list(…)    u64(count) || items; each item fixed-width or length-prefixed
tag(t)     bytes(ASCII of t); every hash input starts with its own tag
ser(x)     the existing uncompressed AkitaSerialize encoding
hex(d)     lowercase hex of a digest, used in file names
```

**Stream.** The stream identity fixes the public field stream; page `i` is a
function of the stream identity and `i`.

```text
stream_identity = prime_modulus_be          32 bytes, as in AlgebraSection
               || ser(setup_seed)           33 bytes: u8(derivation tag) || seed[32]
               || u64(page_field_elements)  8 bytes; 4096 today

stream_id = H(tag("akita/preprocessing/stream/v1") || stream_identity)
```

`ser(setup_seed)` is also the input of the existing `setup_seed_digest`, so
the transcript's `SetupSection` binds the same seed bytes.

**Commitment algorithm.** The algorithm identity fixes how a commitment is
computed from a profile and the stream. Versions that only affect the
transcript (protocol and descriptor versions) are not part of it.

```text
algorithm_identity = u8(COMPRESSION_POLICY.tag())          4 today
                  || u32(GroupCommitPhaseParams::VERSION)   4 today
                  || u32(SETUP_PREFIX_PAYLOAD_VERSION)      new constant
                  || known_answer_digest                    32 bytes

known_answer_digest = H(tag("akita/preprocessing/known-answer/v1")
                        || u32(corpus_version)
                        || test_stream_identity
                        || list(bytes(profile_descriptor_i) || bytes(payload_i)))
```

The known-answer digest pins the behaviour of the commitment code on a
versioned corpus of small profiles over a fixed test stream, checked into the
repository. It is a compiled constant. CI MUST recompute it on every
supported instruction set (x86-64 with and without IFMA52, AArch64). A second
CI test MUST map every profile in every anchored release to a corpus class
(field, A and B ring dimensions, inner and outer route, slice count,
compression chain length, multi-block layout) and fail if a class is missing.
The corpus catches a kernel or encoding change that forgot a version bump;
size-specific paths are covered by full regeneration in the merge check.
Extending the corpus changes every commitment digest, so it is a code
transition.

**Profile.**

```text
profile_descriptor = u8(route)    0 = L-infinity inner route, 1 = L2
                  || bytes(p)     p = GroupCommitPhaseParams::canonical_descriptor_bytes()

profile_key = H(tag("akita/preprocessing/profile/v1") || profile_descriptor)
```

The route byte states the inner security route explicitly instead of leaving
it to be inferred from the layout of `p` (after its SIS modulus, security
policy, and matrix role tags, the inner matrix descriptor writes
`akita-l2-route-v1` on the L2 route or the table digest on the L-infinity
route).

**Commitment.**

```text
payload = ser(SetupPrefixPublicCommitment<F>)

commitment_digest = H(tag("akita/preprocessing/commitment/v1")
                      || stream_identity
                      || algorithm_identity
                      || bytes(profile_descriptor)
                      || bytes(payload))
```

The commitment digest binds a commitment to the stream, algorithm, and
profile it was computed for, so it cannot be presented under another
context. The loader recomputes it from the payload it loaded; protocol
Stage 3 absorbs it.

**Files.** Each file is hashed with its name, so content cannot be moved to
another file. Matrix pages are hashed individually so that a loader can check
a range without hashing the whole matrix. With `P = page_field_elements`,
`page_i = ser(S[iP]) || … || ser(S[(i+1)P − 1])`, with no length prefix or
padding. For a stream of `N` elements, `page-hashes` holds exactly
`ceil(N/P)` hashes; the final page is derived in full even if it extends
past `N`. A range loader authenticates every complete page the range touches
before keeping the requested elements.

```text
file_hash   = H(tag("akita/preprocessing/file/v1") || str(name) || bytes(content))
page_hash_i = H(tag("akita/preprocessing/page/v1") || stream_id || u64(i) || page_i)
```

| File | Content | Used by |
|---|---|---|
| `registry/<hex(stream_id)>/<hex(profile_key)>` | `payload` | Both |
| `hints/<hex(stream_id)>/<hex(profile_key)>` | the hint part of the exported `SetupPrefixSlot<F>`, serialized with `Compress::No`, without the slot id and public commitment | Prover |
| `catalog/<set>` | catalog artifact bytes | Both |
| `catalog-view/<set>/<hex(row digest)>` | selected-row verifier view | Guest |
| `matrix/<hex(stream_id)>/<first>-<last>` | optional complete pages `page_first || … || page_last` | Guest, constrained provers |
| `page-hashes/<hex(stream_id)>` | `page_hash_i` concatenated in page order | Loaders of matrix ranges |
| `terminal-cache/<hex(stream_id)>/<hex(row digest)>/<target>/<capacity>` | prepared terminal cache with its binding header | Guest |
| `certificate/<hex(stream_id)>/<hex(profile_key)>` | reserved for a future proof of derivation | None yet |

In file names, `row digest` is the row's `ScheduleRowDigest` (the
`row_digest` of `OpeningScheduleSelection`); page numbers and `<capacity>`
are ASCII decimal without leading zeros; `<target>` is a guest target name
exactly as listed in the manifest's `target` list.

**Set record and manifest.** The set record is everything a loader needs to
accept or reject a parameter set before reading other files. The manifest
collects every set record, every commitment digest, and every file hash; its
hash is the release identity.

```text
set_record = str(set name) || str(config family) || bytes(decomposition)
          || u8(sis_modulus_profile_tag) || SisTableDigest || SisL2TableDigest
          || algorithm_identity || stream_id
          || u64(num_chunks) || u64(num_activated_levels)
          || u64(max_num_vars) || u64(max_num_batched_polys) || catalog_digest
          || list(profile_key)                     catalog profiles
          || u8(admits program rows)
          || list(profile_key)                     universe; empty if flag is 0

manifest = tag("akita/preprocessing/manifest/v1") || u32(format_version)
        || str(release name) || u8(kind: 0 = akita, 1 = jolt)
        || str(generator version) || list(str(target)) || u64(max file part)
        || list(stream_identity || u64(element count))
        || list(stream_id || profile_key || commitment_digest)
        || list(bytes(set_record))
        || list(str(name) || u64(length) || file_hash)

bundle_id = H(tag("akita/preprocessing/bundle/v1") || manifest)
```

Fields not defined above:

- `config family`: `CommitmentConfig::schedule_family_name()`.
- `decomposition`: the bytes `encode_decomposition` writes for `SetupSection`.
- `num_chunks`, `num_activated_levels`: the set's `ChunkedWitnessCfg`.
- Flags are `0` or `1`; any other value is invalid. Digests are their raw 32
  bytes.
- `format_version` is `1`.
- `target`: a guest target the release provides files for, as named in the
  generator's role argument.
- `max file part`: the largest size in bytes of one stored part of a release
  file; larger files are split into parts of this size for transport.
- `element count`: the number of field elements of the stream the release
  covers, the largest capacity any set or profile needs.

Lists are strictly sorted, bytewise: streams by `stream_id`, the registry by
`(stream_id, profile_key)`, set records by set name and files by file name,
comparing the UTF-8 bytes of the name without its length prefix. A duplicate, an
unsorted list, an unknown version or kind, trailing bytes, or a manifest over
the compiled length cap MUST be rejected. The manifest does not list itself.
The length cap is fixed with headroom over the largest known release;
exceeding it is a code transition.

### Loading

A loader turns release files into the values a prover or verifier is built
from. It checks the cheapest and most decisive things first, and never
decodes data it has not yet authenticated.

```text
1. check the length cap, hash the manifest,                           before decoding
   (bundle_id, set name) in anchor
   then decode the manifest; release kind matches the integration
2. code-derived set-record fields == compiled code                   config family,
                                                                      decomposition, SIS
                                                                      profile and digests,
                                                                      algorithm_identity
3. for each file used: file_hash == manifest entry                    on the exact bytes used
4. for each commitment: recompute commitment_digest == manifest
5. admit rows, size the setup, derive the matrix, re-admit            A1–A4
6. issue PinnedVerifierSetup / PinnedTerminalCache / PinnedProverSetup
```

Admission sizes the setup from what is actually needed:

- **A1.** Admit rows with `TrustedScheduleCatalog::rows_within_setup_capacity`
  against the set limits (for a single-row load, resolve the selection and
  apply the same capacity check; `for_selection` is constructed after A3).
  For a set with program rows, admit (a) the base rows whose final group
  equals the program's shape and (b) the program's rows, both against the
  program bounds. A program row is admitted only if every slot that
  `required_setup_prefix_slot_ids_for_schedule` derives from its audited
  schedule maps to a universe profile. The loader MUST NOT accept a
  separately supplied list of slot ids.
- **A2.** Size the matrix: for a verifier, the largest
  `verifier_setup_matrix_capacity_for_schedule` over the admitted rows; for a
  prover, `SetupRequirements::from_catalog(…).matrix_capacity()` over the
  rows it may prove.
- **A3.** Derive that prefix from the seed, and build the setup descriptor
  from the set limits (the program bounds, for a set with program rows) and
  the derived length.
- **A4.** `verifier_admits` on the resulting setup MUST return exactly the A1
  rows.

The matrix is derived, not shipped: at the largest measured shape it is
4 GiB, and deriving it is about 3% of setup time. Prover hints are loaded
only for the profiles the admitted rows require. Without a supplied cache,
`for_selection` prepares the terminal cache itself, so native hosts load
none. Admission and the set-record check live in `akita-config`, which both
loaders depend on, so prover and verifier run the same code.

### Anchors and transitions

Production and custom deployments use the same loader; only the anchor
differs.

- Akita compiles its anchor from `akita-anchors`, Jolt from a crate of its
  own. Each lists the pairs of the releases committed under `params/`. Anchor
  crates MUST stay outside the generator's dependency closure, so adding a
  release never changes a `bundle_id`; `scripts/check-crate-deps.sh` enforces
  this.
- Anchors MUST be declared as compile-time constants from a `&'static` list,
  and production loaders MUST accept only those values. `TrustAnchor` has no
  deserializer and no `FromStr`; a lint rejects run-time construction in
  production code. An anchor read from storage voids every guarantee here.
- The integration MUST derive the set name from its own state (Jolt: flavor,
  K, chunk profile), never from transported bytes alone. Jolt also carries
  the `bundle_id` and set names in its program preprocessing and requires
  them to match.
- Every file of one construction MUST come from the single manifest that
  passed the anchor check. A construction MAY load several sets of that
  manifest; Jolt loads a one-hot set with the bounded-dense and full-dense
  sets its producers need.
- A deployment with its own seed, catalogs, or limits generates its own
  release with the same generator and passes its own anchor.

Releases change in three ways:

| Change | What happens | Anchor update |
|---|---|---|
| Description only: sets, catalogs, limits, universe | Existing commitments stay byte-identical; both releases regenerate under the same code | Add the new pairs; remove the old ones one software version later |
| Commitment algorithm or catalog encoding (for example `AKITA_INSTANCE_DESCRIPTOR_VERSION`, which sets the catalog `protocol_epoch`) | Commitment digests or catalog bytes change; old releases no longer regenerate | Replace in one step |
| Seed | Every commitment changes | Replace, or keep both releases anchored |

An old proof stays verifiable under a new release when everything its
transcript binds is unchanged: the `SetupSection` fields (seed digest,
decomposition, SIS profile, compression policy, protocol features), the row
in `PlanSection`, and the commitment digests absorbed at protocol Stage 3.
For standalone Akita, description-only changes preserve all three as long as
the row is still admitted. Jolt also binds `preprocessing_digest`, which
covers `bundle_id`, so an old Jolt proof needs its original program
preprocessing.

Revocation removes pairs from the anchor in a new software version; older
binaries keep accepting them. Key-based anchors could revoke without
rebuilding but need freshness and rollback protection, and are out of scope.

### Invariants

1. **Determinism.** Every file, the manifest, and `bundle_id` MUST depend only
   on the description and the code, not on thread count, machine, operating
   system, instruction set, backend, or build profile. Hints come from the
   CPU reference path. Test: generate one release at two thread counts on
   x86-64 and AArch64 and compare `bundle_id`.
2. **From scratch.** Generation, including regeneration in audit mode, MUST
   NOT use an ambient disk cache, a previous bundle, or the candidate being
   audited as input. Audit mode reads the candidate only to compare it with
   the regenerated output. Test: plant a poisoned
   registry in the `akita-setup` cache directory; regeneration ignores it.
3. **Hashed bytes are used bytes.** A loader MUST read each file into a
   private, immutable buffer while hashing it, check the manifest length
   before allocating, decode exactly that buffer, and never reopen or reread
   the file. Matrix ranges MAY be read through private page-sized snapshots,
   each checked against `page-hashes`. A loaded value MUST NOT be refreshed
   from storage except through the loader; this covers lazy caches (such as
   Jolt's first-use setup regeneration) and memory-mapped files. Program
   rows are the one non-file input: they arrive as an argument,
   authenticated by the program preprocessing digest. Test: modify a file
   between hashing and use, and after loading.
4. **Anchor first.** Nothing is decoded before the anchor check. Manifest
   decoding MUST be total and bounded under the verifier no-panic contract.
5. **Code matches release.** Before using any file, the loader MUST compare
   the code-derived set-record fields (config family, decomposition, SIS
   modulus profile and table digests, algorithm identity) with its compiled
   configuration. Limits, catalog identity, and profile lists are release
   data, authenticated through the manifest; admission limits never come from
   unauthenticated bytes.
6. **Only loader values reach production constructors.** `PinnedVerifierSetup`
   and `PinnedTerminalCache` (in `akita-verifier`) and `PinnedProverSetup` (in
   `akita-cpu-backend`) have private fields, no public constructor, and no
   deserializer. With program rows, the pinned catalog is
   `ValidatedScheduleCatalog::try_new` over the admitted base and program
   rows; `catalog_digest()` authenticates the base catalog, and Jolt's
   transcript binds the merged catalog's digest. Every path that builds a
   setup from loaded bytes becomes loader-private or an `unsafe fn` whose
   contract is that the caller guarantees derivation:
   - `AkitaVerifierSetup` fields, `from_parts`, and its deserializer (it
     becomes the unchecked type for tests, benchmarks, and the loader);
   - `AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked`,
     `Validate::No` setup decoding;
   - `borrow_trusted`, `TrustedTerminalCache` in `for_selection`;
   - `AkitaProverSetup` public fields, `to_verifier_setup`;
   - `AkitaCommitmentScheme::setup_verifier`, `setup_verifier_for_schedule`,
     `verifier`;
   - the `AkitaVerifier` constructor taking a `LocallyDerivedVerifierSetup`.

   Jolt and the production guest deny these with
   `clippy::disallowed_methods`, and compile-fail tests check that
   production constructors reject anything else. Here `unsafe` marks a
   soundness contract, not a memory-safety one, so every call site is
   visible and deniable.
7. **Prover slots.** The loader rejects any file that fails its hash. A
   backend adopts a slot without recomputing it only if the slot came from
   this process or arrived in a `PinnedProverSetup` whose payload and hints
   matched the manifest; otherwise it recomputes. A backend other than the
   CPU reference path MUST reproduce the reference payload byte for byte.
8. **Audit mode.** The generator crate can regenerate the registry and set
   records of any release and compare them with its manifest. It runs on
   hosts that can link `akita-cpu-backend`.

### Non-Goals

- A proof of derivation (the certificate chain of
  [`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md#setup-provenance-and-validation)).
  This spec provides the trust root that spec asks for and reserves a file
  for a certificate.
- Witness confidentiality.
- Authenticated catalog subsets for recursion (the deferred layer of
  [`external-schedule-catalog-ownership.md`](external-schedule-catalog-ownership.md#deferred-authenticated-recursion-layer)).
- Key-based anchors, running old and new commitment algorithms side by side,
  and backward compatibility.

## Evaluation

### Acceptance Criteria

- [ ] The generator writes every file and the manifest, and `bundle_id` is
      identical across thread counts on x86-64 and AArch64.
- [ ] Generation ignores a poisoned ambient cache.
- [ ] The known-answer digest passes on every supported instruction set and
      fails when a compression-policy or kernel change alters a corpus
      output; the coverage test fails for a profile with no corpus class.
- [ ] A loader-built verifier loads its catalog and registry entries,
      derives its matrix for its admitted rows, and rejects: a manifest
      outside its anchor, an unanchored set name, the wrong release kind, a
      one-byte change in any file it uses, and a set record that differs from
      its compiled code.
- [ ] Compile-fail tests show production constructors reject non-loader
      values; production code cannot construct a `TrustAnchor` at run time;
      Clippy fails on Jolt and the guest when they call a denied `unsafe fn`.
- [ ] Protocol Stage 3 absorbs a commitment digest recomputed in the proof,
      and a proof fails when a commitment it uses differs.
- [ ] Both releases of a description-only transition load; the old one is
      rejected once its pairs are removed.
- [ ] A release generated from a different seed works with a custom anchor
      and the same loader.
- [ ] The repository holds at least one manifest, and a required merge check
      regenerates it when its description, manifest, anchor, `Cargo.lock`,
      `rust-toolchain.toml`, or any crate in the generator's dependency
      closure changes.
- [ ] Jolt loads through the loader, admits program rows only within the
      universe and with the anchored producer profiles, and no longer
      regenerates setup-prefix commitments on first use or accepts
      serde-transported verifier setups.
      Swapping a program row changes `preprocessing_digest`.
- [ ] The production guest checks its embedded preprocessing with the loader
      and reads none from its input; its approved program commitment comes
      from the same ELF snapshot the approval checked.

### Testing Strategy

- Determinism at two thread counts and on two instruction sets.
- Known-answer digest and corpus coverage on every supported instruction set.
- Poisoned cache (Invariant 2).
- Tampering: one byte in each file, truncation, extension, reordered or
  duplicated manifest entries, modification after hashing.
- Anchors: wrong set name, wrong kind, files from two releases.
- Config and algorithm mismatch.
- Admission: A1 and A4 agree; matrix ranges ending mid-page.
- Program rows: profile outside the universe, wrong precommitted profile,
  duplicate base key, and planner determinism (`find_adapted_schedule` on
  sampled keys at two thread counts and on two instruction sets).
- Audit mode and the loader build identical verifier state.
- End-to-end tests against generated releases; `.github/workflows/ci.yml`
  stays the source of truth for feature sets.

### Performance

Measured on one Jolt recipe (one-hot, K=16, chunk profile w1r0, 57
polynomials); details in Measurements.

- **Generation.** Slot export is about 90% of setup time and runs one slot
  after another, so all cores speed setup up only about 3.5x. A Jolt release
  per recipe (about 36 commitments plus dense sets) is estimated at the cost
  of one largest-shape setup; a release spans several recipes.
- **Generation memory** SHOULD stay near the matrix plus the largest entry.
  The measured revision peaked at about 11x the matrix; current slot export
  streams its sources, so this needs remeasuring.
- **Verifier construction** hashes only the registry entries it uses (about
  21 KB at the largest shape) and its catalog, and derives its matrix. It no
  longer regenerates setup-prefix commitments on first use, which at the
  measured Jolt revision cost 7 to 55 times the verification time of a
  2^22-cycle program.
- **Prover construction** loads hints (up to about 93 MiB) instead of
  exporting slots.
- **Program preprocessing.** The full-search fallback, hit by 6% to 35% of
  sampled rows, is four to five orders of magnitude slower than the guided
  path. It is paid once per program.
- **Proof size** is unchanged; the Stage 3 digest is absorbed, not sent.
- **Per-proof digests** hash a few kilobytes; negligible natively, unmeasured
  in a guest.

## Design

### Crates

```text
akita-types        manifest, set record, anchor types; manifest decoding;
                   hash and anchor checks
akita-config       set-record check against CommitmentConfig, commitment_digest,
                   row admission; shared by both loaders
akita-verifier     verifier loader → PinnedVerifierSetup, PinnedTerminalCache
akita-cpu-backend  prover loader → PinnedProverSetup
akita-setup        disk cache of release files keyed by bundle_id (settles #210)
akita-pcs          pinned replacements for the AkitaCommitmentScheme entry points
akita-preprocess   generator library and binary
akita-anchors      Akita's compiled anchor; used by binaries, never by the generator
jolt               Jolt generator binary (links akita-preprocess and jolt-akita)
                   and Jolt's anchor crate
```

Admission lives in `akita-config` because `akita-cpu-backend` may not depend
on `akita-verifier`. The generator depends on `akita-config`,
`akita-planner` (to enumerate universes), `akita-cpu-backend`, and the
verifier's public `build_riscv64_terminal_ntt_cache`. It MUST NOT depend on
`akita-setup`, whose cache Invariant 2 forbids it to read, or on an anchor
crate. `scripts/check-crate-deps.sh` gains a case for it and adds it to the
forbidden list of every verifier-facing crate.

The generator takes a role (`all`, `prover`, `verifier`, or a guest target).
The role only filters which files are written; every role computes the same
`bundle_id`.

### Jolt

Jolt splits its preprocessing by what it depends on. What is independent of
the program is published; what depends on the program travels with the
program.

```text
published, per flavor / K / chunk profile        per program
├─ one-hot set                                   ProgramPreprocessing
│   ├─ base catalog                              ├─ program commitments
│   └─ profile universe                          ├─ bundle_id, set names
├─ bounded-dense set                             └─ program rows
│   advice, committed bytecode and image,             (find_adapted_schedule)
│   bounded fused increment
└─ full-dense sets
    one per field-increment arity (14..31)
```

Grouped rows cannot be published. A grouped row's key is the final group
plus the ordered list of producers (untrusted and trusted advice, committed
bytecode and image, the bounded fused increment, and the optional full-width
field increment), each with its own arity. That is about
10^8 keys for one recipe, every key plans to a distinct row, and a row is
about 15.6 KB. Commitment profiles, by contrast, saturate: about 36 cover
every sampled row, for about 70 MB of hints and 18 KB of commitments.

Rules:

- Program rows, the `bundle_id`, and set names MUST be stored in
  `ProgramPreprocessing` (full and committed variants), so
  `preprocessing_digest` covers them. They MUST NOT be stored in `pcs_setup`,
  which the digest does not cover.
- A construction loads the one-hot set and the dense sets it needs from one
  manifest. Every precommitted group profile in a program row MUST equal the
  frozen profile for that layout in its anchored producer set: bounded-dense
  for bounded producers, full-dense for full-width field increments.
- A program row never supplies a commitment; every commitment it uses comes
  from the anchored registry.
- The serde `AkitaVerifierSetup` in `pcs_setup`, its transport, and the
  first-use regeneration of setup-prefix commitments are removed. Jolt keeps
  its own transcript bindings (bounds, `one_hot_k`, chunk profile, layout
  digest, merged catalog digest).
- A CI test plans rows over a deterministic sample of the producer grid and
  over Jolt's example programs, and fails if a row needs a profile outside
  the universe.

The universe must be complete for this to be live, and sampling does not
prove that: the committed-program sample covers about 0.0015% of its keys,
and the field-increment, K=256, and w8r2 cases were not sampled. Therefore
`find_adapted_schedule` MUST choose only universe profiles, so a program
fails only if no feasible schedule exists within the universe. The generator
SHOULD enumerate the universe from the planner's candidate space. A program
that needs a wider universe waits for a release that adds it.

**Local profile derivation** (opt-in, host only). A custom deployment whose
rows need profiles outside every anchored universe can have both sides
compute the missing commitments themselves. The verifier MUST NOT import a
commitment from the prover or a file. The result is a distinct
`LocallyDerivedVerifierSetup`, produced only by a host-only crate downstream
of `akita-cpu-backend` and `akita-planner`, and accepted only by an
`unsafe fn` `AkitaVerifier` constructor (Invariant 6). It is never available
in the guest.

### Transcript binding

Protocol Stage 3 already absorbs the compressed slot id through
`stage3_public_slot` (from `bind_setup_prefix_slot` in
`crates/akita-verifier/src/stages/stage3.rs`, and at the matching point in
`crates/akita-prover/src/protocol/prove/fold/sumcheck.rs`). At that call it
MUST now absorb `serialize_compressed(slot id) || commitment_digest`, the
digest as 32 raw bytes. Prover and verifier MUST
recompute the digest in every proof from the registry entry the fold reads,
never from a stored value. Every fold with a setup prefix passes Stage 3
once, so every use is bound once.

The binding makes each proof name the commitments it was checked against.
Soundness comes from the load-time checks: once loading fixes a commitment,
the prover cannot choose it. The binding costs a transcript change
(`AKITA_INSTANCE_DESCRIPTOR_VERSION` and protocol version bumps, regenerated
catalogs) and a Jolt revision bump, so it SHOULD land before the first
release.

Two things are deliberately not absorbed. The anchor is local policy and
changes whenever any release is added, so it would break proofs between
binaries that trust the same release. `bundle_id` would tie proofs to one
release and break them across description-only transitions with identical
commitments; Jolt already binds it through `preprocessing_digest`.

### Recursion guest

A guest cannot plan rows or compute commitments, and must not read
preprocessing from its input. Everything it needs is embedded in an approved
image.

```text
approved guest image
├─ trust anchor
├─ manifest, set record, catalog view, registry entries   checked by the loader at startup
├─ matrix prefix, terminal caches                          vouched for by image approval
└─ inner ProgramPreprocessing with its program rows        admitted, never planned
```

At startup the guest runs the same loader as a host on the small embedded
files. Hashing the large ones may cost too many cycles, so they MAY be issued
through a guest-only `unsafe fn` whose contract is that the bytes are
embedded in an approved image and match the manifest; only the guest crate
may call it. A guest verifying grouped Jolt proofs embeds the inner program's
`ProgramPreprocessing` and admits its rows with the same loader call as the
host. One approved image covers one inner program.

Approval happens in the trusted build environment: a reproducible build
produces one immutable ELF snapshot; the approval step checks its embedded
files against the manifest and the inner program preprocessing against its
digest; Jolt's program commitment for the guest is computed from the same
snapshot, and the outer verifier trusts that commitment.

### Publication and audit

A release is published by one commit that adds its description and manifest
under `params/<release>/` and its pairs to the anchor crate. Large files are
uploaded as GitHub release assets; they are untrusted transport, since the
loader checks them against the manifest.

A required merge check regenerates every anchored release from scratch and
compares bytes whenever its description, manifest, or anchor changes,
`Cargo.lock` or `rust-toolchain.toml` changes, or any crate in the
generator's dependency closure changes (from `cargo metadata`). That closure
covers nearly all of Akita, so most changes trigger it. In Jolt, the closure
includes the Akita crates. A scheduled job also regenerates every release,
and anyone can run audit mode locally.

### Security argument

Assume `H` is collision resistant; the build environment and source are
authentic; the integration compiles in an authentic anchor and derives the
set name itself; a Jolt verifier holds the authentic program preprocessing
digest; and the merge check ran on every production release.

1. Production provers and verifiers are built only from loader values
   (Invariant 6).
2. Every file the loader used has the bytes named by the one manifest that
   passed the anchor check (Invariants 3 and 4). In the guest, large files
   rely instead on the approval of the image that embeds them.
3. Those bytes are the generator's from-scratch output at the source commit
   (merge check, Invariant 2). For custom releases this rests on whoever
   generated them.
4. The parameter set matches the running code (Invariant 5), and the
   algorithm identity is tied to the compiled commitment code by the
   known-answer test and full regeneration.
5. Hence every commitment satisfies `C = Commit(S[0..n_prefix], profile)` for
   the loader's stream `S`; the verifier's matrix is derived from `S` (or
   embedded in an approved image); every catalog row passes the admission
   check and belongs to the named set; every program row is covered by the
   trusted program digest, passes the same check, and uses only anchored
   commitments. These are the preprocessing hypotheses of Akita's soundness
   claim.

Still trusted: the repository, build, and merge checks (audit mode is the
independent check); the completeness of the row admission check, which is
the only check on program rows beyond the program digest; the integration's
set-name derivation; Jolt's program digest; guest image approval; and the
author of any custom release.

### Alternatives Considered

| Alternative | Why not |
|---|---|
| Hash the executable instead of the data | Depends on platform and toolchain; prover data would bloat every binary |
| Compile registry and catalogs into the verifier | Only the compiled seed and catalogs would work; anchors give the same guarantee and allow custom releases |
| Anchor of bare `bundle_id`s | An adversary could select any set in an anchored bundle, including one the integration did not approve |
| Cargo feature gate on unchecked constructors | Features are additive; any dependency could enable it |
| Verifier recomputes commitments at load | Needs `akita-cpu-backend` and the full source prefix; the known-answer digest and regeneration cover drift |
| Set-level identity in `SetupSection` | Stage 3 binds exactly the commitments used; `PlanSection` binds the row |
| Publish every Jolt grouped row | Advice alone is about 129,000 rows and 2 GB, 8x the catalog row cap and 30x its byte cap; about 10^8 keys overall |
| One registry file | Every verifier would hash the whole registry |
| Ship the matrix by default | 4 GiB at the largest shape; deriving it is about 3% of setup |
| Merkle tree over matrix pages | Flat page hashes give the same range checks without padding or sibling rules |
| Derive longer-prefix commitments from shorter ones | The outer gadget decomposition and compression chain are not linear |
| Regenerate on a schedule only | A manifest is opaque in review and an algorithm change does not touch it |

## Documentation

- `docs/security-posture.md`: the anchor model; in-place setup views are used
  only on bytes the loader or an approved guest image vouches for. This
  updates its existing Verifier Preprocessing section.
- `book/src/how/verification.md`: what the verifier trusts, anchors, the
  loader, audit mode, Stage 3 binding.
- `book/src/usage/verifier-only.md`, `book/src/usage/jolt-recursion.md`:
  loading a release, custom anchors, the guest's embedded preprocessing.
- `specs/external-schedule-catalog-ownership.md`: allow a guest to embed a
  catalog view checked against an anchored release; describe program rows.
- Doc comments on the unchecked constructors point to the loader.

## Execution

### Measurements

One Jolt recipe (one-hot, K=16, chunk profile w1r0, 57 polynomials), on the
Akita revision Jolt main depends on, on one multi-core x86-64 machine,
release build, `parallel` on, disk persistence off. `max_num_vars` is
`log2(trace length) + 4`, so `nv = 26` is a 2^22-cycle program; the matrix
stops growing at `nv = 35`. Each configuration ran once. Times are ratios
because absolute times depend on the machine.

| nv | Matrix | Slots | Slot export share of setup | Setup time vs `nv = 26` | Prover hints | Verifier commitments |
|---|---|---|---|---|---|---|
| 24 | 32 MiB | 0 | none | 0.4x | none | none |
| 26 | 64 MiB | 3 | 55–70% | 1x | 1.5 MiB | 1.5 KB |
| 28 | 256 MiB | 9 | 78–85% | 5x | 7.3 MiB | 4.6 KB |
| 30 | 512 MiB | 17 | 88–91% | 20x | n/a | n/a |
| 32 | 1 GiB | 25 | 92–94% | 45x | 38 MiB | 12.7 KB |
| 40 | 4 GiB | 41 | 90–92% | 150x | 93 MiB | 20.8 KB |

- Matrix derivation is 3% to 10% of setup from `nv = 28` up, and about 3% at
  `nv = 40`; it is about 6x faster on all cores than on one.
- All cores speed setup up only about 3.5x, because `export_setup_prefixes`
  handles slots one after another. Within a slot, 1 to 8 threads gives about
  3.4x and 8 to 32 threads another 1.15x.
- At the measured Jolt revision, first-use regeneration of setup-prefix
  commitments cost 7 to 55 times the verification time of a 2^22-cycle
  program, and about one full setup at `nv = 40`. Jolt's serialized
  verifier setup is 4.2 MiB at every size, almost all of it the catalog.
- `validate_public_matrix_matches_seed` is sequential and about 6x slower
  than parallel derivation.
- Dense and one-hot share a seed, and the dense matrix is a prefix of the
  one-hot one; a setup with both derives the shared prefix twice.

At the measured revision, peak setup memory was 8x to 14x the matrix (about
11x at `nv = 40`), because slot export kept a copy of every slot source until
the end. Freeing each source after use was estimated to bring this to 2x to
2.5x. Current slot export streams its sources and builds them lazily, so
these figures need remeasuring.

Grouped rows were sampled with a reimplementation of Jolt's `plan_row`, with
10 to 24 rows planned concurrently (so per-row times are inflated):

| Grid | Keys | Sampled | Errors | Distinct rows | Full-search fallback |
|---|---|---|---|---|---|
| Advice | 129,360 | 6,000 | 0 | 6,000 | 5.7% |
| Committed program | 57,171,240 | 871 | 0 | 871 | 35% |

- Profiles: 16 in the base catalog of the measured revision; the advice
  sample reached 33 by row 4,500 and stayed flat; about 36 across both
  samples, all with groups of at most 2^28 elements.
- Full slot ids do not saturate (661 after 6,000 rows), but ids differing
  only in `natural_len` share a commitment.
- Hints per profile grow from about 0.5 MB (group `nv` 20) to 10.5 MB
  (`nv` 28): about 70 MB for 36 profiles.
- Full-dense matrices for arities 14 to 31 are 9.5 MB to 268 MB, with no
  setup-prefix slots.
- Not sampled: field-increment grids, K=256, w8r2.

### Generation strategy

1. Derive each stream's matrix once, at the largest capacity any set or
   profile needs, hashing pages in the same pass.
2. Collect profiles across all sets; slot ids that differ only in
   `natural_len` map to one entry.
3. Share exactly shared work: one digit decomposition per
   `(ring dimension, basis)`, one prepared A transform per
   `(ring dimension, width)`.
4. Generate commitments in parallel, bounded by memory. Each borrows its
   source from the derived matrix and frees it once its payload and hints are
   written.
5. Reuse inner commitment rows between nested profiles only if stream, A ring
   dimension and rank, positions per block, digit basis and count, and block
   boundaries all match; recompute outer layers and compare with independent
   generation. Measure how often this applies first.
6. Build guest caches per `(stream, row, target, capacity)`.
7. Write the files, then the manifest.

### Steps

1. Types and encodings in `akita-types`; set-record check and admission in
   `akita-config`; known-answer corpus and coverage tests; tamper tests.
2. Generator crate and binary; determinism and poisoned-cache tests.
3. Loaders and `Pinned*` types; pinned entry points in `akita-pcs`;
   loader-private or `unsafe fn` conversions; compile-fail tests; replace
   the `akita-setup` cache.
4. Stage 3 binding with its version bumps (before the first release).
5. First release, anchor crate, required merge check.
6. Jolt: generator, sets and universes, program rows in
   `ProgramPreprocessing`, planner restriction, coverage and determinism
   tests, removal of the rebuild and serde paths, local profile derivation.
7. Recursion guest: embedded files and inner program preprocessing, in-guest
   loader, guest-only issuance, approval from one ELF snapshot.

### Open questions

1. **Universe completeness.** Can the planner's candidate space be
   enumerated directly? What does restricting the planner cost in schedule
   quality? What are the universes for K=256, w8r2, and field increment? Do
   bounded-dense rows need setup-prefix slots?
2. **Guest hashing.** Which embedded files to hash at startup, and the cost
   of per-proof digests, from guest cycle counts.
3. **Seed.** Proposed: keep the fixed `sample_akita_setup_seed()`. A
   per-release seed adds little and stops releases from sharing commitments.
4. **Transition window.** Proposed: one software version.
5. **Merge-check runner.** Size it from a full multi-recipe release, not yet
   measured.
6. **Planner fallback.** Make it cheaper before program preprocessing
   depends on it?

## References

- [`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md), "Setup provenance and validation"
- [`external-schedule-catalog-ownership.md`](external-schedule-catalog-ownership.md)
- #223, with #233, #234, #235: verifier preprocessing trust model
- #210: recomputing setup-prefix commitments when loading disk caches
- #186: trusted in-place setup and terminal-cache views
- #187: selected-row catalog views
- RFC 2119: requirement levels
