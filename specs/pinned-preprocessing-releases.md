# Spec: Pinned Preprocessing Releases

| Field         | Value |
|---------------|-------|
| Author(s)     | sumchecker |
| Created       | 2026-10-26 |
| Status        | proposed |
| PR            | #287 |
| Supersedes    | The cache-loading policy question of #210 |
| Superseded-by | |
| Book-chapter  | how/verification.md |

## Summary

The verifier accepts setup-prefix commitments after structural checks only
(`AkitaVerifierSetup::from_parts`, setup deserialization, the `akita-setup`
disk cache, `AkitaProverSetup::to_verifier_setup`). A substituted commitment
lets a prover open it to any vector. The recursion guest reads its catalog,
matrix, and registry from untrusted input.

This spec makes preprocessing a published release with one identity,
`bundle_id`. A binary trusts only the `(bundle_id, set name)` pairs compiled
into it. A loader checks every file against the release manifest before use.
A merge check regenerates every published release from scratch.

The key words MUST, MUST NOT, SHOULD, and MAY are used as in RFC 2119.

## Intent

### Goal

```text
release description ──generator──▶ release files + manifest
                                         │
                          bundle_id = H(manifest)
                                         │
binary: anchor {(bundle_id, set name)} ──▶ loader ──▶ Pinned* setup types
                                                          │
                                       protocol Stage 3 binds commitment_digest
```

### Threat model

The adversary can replace any preprocessing bytes read at run time: setup,
registry, cache, and catalog files, release assets, guest input.

Trusted: the build environment, the source tree, the executables, the
compiled anchor, approved guest images, and the program preprocessing digest
a Jolt deployment trusts.

| Replaced | Verifier | Prover |
|---|---|---|
| Setup-prefix commitment | Soundness break | Honest proofs fail |
| Matrix, on a path that skips re-derivation | Soundness break | Honest proofs fail; may leak more |
| Terminal NTT cache | As the matrix | Not used |
| Catalog, program rows, or chosen set | Policy failure (rows still pass the admission check) | Honest proofs fail; may leak more |
| Prover hints | Not used | Honest proofs fail |

### Release structure

```text
release                                  bundle_id = H(manifest)
├─ description                           input, in params/<release>/
│   ├─ seed                              fixes the public matrix
│   ├─ packaging header                  name, kind (akita|jolt), generator version
│   └─ parameter set "jolt-onehot-k16-w1r0"   (one or more)
│       ├─ CommitmentConfig              how commitments are computed and sized
│       ├─ catalog                       338 pre-planned rows, one per shape
│       ├─ chunking policy
│       ├─ limits                        max_num_vars, max_num_polys
│       └─ profile universe              ~36 commitment profiles (program-row sets)
├─ files                                 written by the generator
│   ├─ registry/<stream>/<profile>       one commitment per profile any set needs
│   ├─ hints/<stream>/<profile>          prover opening material
│   ├─ catalog/<set>, catalog-view/…
│   └─ optional: matrix/…, page-hashes/…, terminal-cache/…
└─ manifest
    ├─ header, stream identities
    ├─ registry: (stream_id, profile_key, commitment_digest)
    ├─ set records: one per parameter set
    └─ files: (name, length, file_hash)

binary
└─ trust anchor                          (bundle_id, set name) pairs, compiled in

Jolt ProgramPreprocessing                covered by preprocessing_digest
├─ program commitments
├─ bundle_id, set names
└─ program rows                          planned per program
```

Terms:

- **Schedule row**: how polynomials of one shape are committed and opened,
  including which levels use which setup-prefix commitment.
- **Commitment profile**: the recipe of one setup-prefix commitment (group
  shape, blocks, inner and outer matrices, route). Excludes `natural_len`.
- **Limits**: largest shapes a set supports. The setup built is sized from
  the admitted rows, not the limits.
- **Profile universe**: profiles that program rows may use. Every one has a
  published commitment.
- **Program rows**: Jolt grouped rows, planned per program, outside the
  release.

Between production releases the seed and commitment algorithm stay fixed, so
existing commitments are byte-identical. Sets, catalogs, limits, and
universes change. A new seed changes every commitment.

### Encodings and hashes

```text
H(x)       Blake2b-256 (digest_descriptor_bytes)
u32, u64   little-endian
bytes(x)   u64(len) || x
str(s)     bytes(utf8(s))
list(...)  u64(count) || items
tag(t)     bytes(ascii(t))
ser(x)     AkitaSerialize, uncompressed
```

A decoder MUST reject any input that does not re-encode to the same bytes.

```text
stream_identity    = modulus_be[32] || ser(seed)[33] || u64(page_elems)   73 bytes
stream_id          = H(tag("akita/preprocessing/stream/v1") || stream_identity)

algorithm_identity = u8(COMPRESSION_POLICY.tag())
                  || u32(GroupCommitPhaseParams::VERSION)
                  || u32(SETUP_PREFIX_PAYLOAD_VERSION)                    new
                  || known_answer_digest                                  41 bytes

known_answer_digest = H(tag("akita/preprocessing/known-answer/v1")
                      || u32(corpus_version) || test_stream_identity
                      || list(bytes(profile_descriptor_i) || bytes(payload_i)))

profile_descriptor = u8(route: 0 = L-inf, 1 = L2)
                  || bytes(GroupCommitPhaseParams descriptor bytes)
profile_key        = H(tag("akita/preprocessing/profile/v1") || profile_descriptor)

payload            = ser(SetupPrefixPublicCommitment)
commitment_digest  = H(tag("akita/preprocessing/commitment/v1")
                      || stream_identity || algorithm_identity
                      || bytes(profile_descriptor) || bytes(payload))

file_hash   = H(tag("akita/preprocessing/file/v1") || str(name) || bytes(content))
page_hash_i = H(tag("akita/preprocessing/page/v1") || stream_id || u64(i) || page_i)

set_record = str(name) || str(config family) || bytes(decomposition)
          || u8(sis profile) || SisTableDigest || SisL2TableDigest
          || algorithm_identity || stream_id || bytes(chunking policy)
          || u64(max_num_vars) || u64(max_num_polys) || catalog_digest
          || list(catalog profile_key)
          || u8(program rows) || list(universe profile_key)

manifest   = tag("akita/preprocessing/manifest/v1") || u32(format version)
          || str(release name) || u8(kind) || str(generator version)
          || list(str(target)) || u64(max file part)
          || list(stream_identity || u64(elements))
          || list(stream_id || profile_key || commitment_digest)
          || list(bytes(set_record))
          || list(str(name) || u64(length) || file_hash)

bundle_id  = H(tag("akita/preprocessing/bundle/v1") || manifest)
```

Notes:

- `ser(seed)` is also the input of `setup_seed_digest`, which `SetupSection`
  already binds.
- The route byte is required: the L-inf and L2 descriptor layouts can produce
  equal bytes.
- `commitment_digest` covers the stream, algorithm, and profile, so a
  commitment cannot be presented under another context.
- `file_hash` covers the name, so content cannot move to another file.
- Lists are strictly sorted (streams by `stream_id`, registry by
  `(stream_id, profile_key)`, sets and files by name). Duplicates, unknown
  versions, trailing bytes, or exceeding the compiled length cap MUST be
  rejected.
- File names use lowercase hex of `stream_id` and `profile_key`.

### Loading

```text
1. H(manifest) and set name in anchor, right kind          before decoding
2. set record == compiled config + algorithm_identity
3. each file used: file_hash == manifest entry             on the bytes then used
4. each commitment: recompute commitment_digest == manifest
5. admit rows, size and derive the matrix, re-admit
6. issue PinnedVerifierSetup / PinnedTerminalCache / PinnedProverSetup
```

Admission:

- **A1.** Admit rows with `TrustedScheduleCatalog::rows_within_setup_capacity`
  against the set limits, or for Jolt against the program bounds (which MUST
  fit the limits). A program row is admitted only if every slot from
  `required_setup_prefix_slot_ids_for_schedule` maps to a universe profile.
- **A2.** Size the matrix from the admitted rows
  (`verifier_setup_matrix_capacity_for_schedule` for a verifier).
- **A3.** Derive that prefix from the seed.
- **A4.** `verifier_admits` MUST return exactly the A1 rows.

The matrix is not shipped by default: deriving it is about 3% of setup time.
A small program never derives the largest matrix. Admission lives in
`akita-config`, shared by both loaders.

### Anchors and transitions

- Akita's anchor lives in `akita-anchors`, Jolt's in its own crate. Both MUST
  be outside the generator's dependency closure.
- `TrustAnchor` MUST be built only by `const fn` from a `&'static` list: no
  deserializer, no `FromStr`.
- The integration derives the set name from its own state (Jolt: flavor, K,
  chunk profile). Jolt also checks it against the name in program
  preprocessing.
- Files from two releases MUST NOT be combined. One construction MAY load
  several sets of one manifest.
- Custom deployments generate their own release and pass their own anchor.

| Change | Effect | Anchor update |
|---|---|---|
| Sets, catalogs, limits, universe | Existing commitments unchanged | Add new pairs; drop old after one software version |
| Commitment algorithm or catalog encoding | Every digest or catalog changes; old releases stop regenerating | Replace in one step |
| Seed | Every commitment changes | Replace, or keep both |
| Revocation | None | Remove pairs; older binaries still accept them |

### Invariants

1. **Determinism.** Every file, the manifest, and `bundle_id` depend only on
   the description and the code. Test: two thread counts, x86-64 and AArch64,
   same `bundle_id`.
2. **From scratch.** Generation MUST NOT read an ambient cache, a previous
   bundle, or the candidate. Test: poisoned `akita-setup` cache is ignored.
3. **Hashed bytes are used bytes.** Read each file into a private buffer while
   hashing; decode that buffer; never reread. No refresh from storage outside
   the loader, including lazy caches and memory maps. Test: modify a file
   after hashing.
4. **Anchor first.** Nothing is decoded before the anchor check. Decoding is
   total under the no-panic contract.
5. **Code matches release.** The set record MUST equal what the compiled code
   produces.
6. **Only loader values reach production constructors.** `Pinned*` types
   have private fields and no deserializer. These become loader-private or
   `unsafe fn` (soundness contract, denied in Jolt and the guest with
   `clippy::disallowed_methods`):
   - `AkitaVerifierSetup` fields, `from_parts`, deserializer
   - `from_trusted_seed_derived_parts_unchecked`, `Validate::No` decoding
   - `borrow_trusted`, `TrustedTerminalCache`
   - `AkitaProverSetup` fields, `to_verifier_setup`
   - `AkitaCommitmentScheme::setup_verifier{,_for_schedule}`, `verifier`
   - the `LocallyDerivedVerifierSetup` constructor
7. **Prover slots.** Adopt a slot without recomputing only if it came from
   this process or matched the manifest. Non-reference backends MUST match
   the CPU reference payload byte for byte.
8. **Audit mode.** The generator can regenerate and compare any release.

### Non-goals

- Proof of derivation (certificate chain of
  [`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md#setup-provenance-and-validation)).
- Witness confidentiality.
- Authenticated catalog subsets
  ([`external-schedule-catalog-ownership.md`](external-schedule-catalog-ownership.md#deferred-authenticated-recursion-layer)).
- Key-based anchors; old and new algorithms side by side; backward
  compatibility.

## Evaluation

### Acceptance Criteria

- [ ] Same `bundle_id` across thread counts, x86-64, and AArch64.
- [ ] Generation ignores a poisoned cache.
- [ ] Known-answer digest passes on every instruction set and fails on a
      kernel or policy change; coverage test fails on a profile with no corpus
      class.
- [ ] Loader rejects: manifest outside the anchor, unanchored set name, wrong
      kind, one-byte change in any used file, set record that differs from
      the code.
- [ ] Compile-fail tests: production constructors reject non-loader values;
      `TrustAnchor` has no run-time constructor. Clippy denies the `unsafe fn`s
      in Jolt and the guest.
- [ ] Protocol Stage 3 absorbs `commitment_digest`, recomputed per proof.
- [ ] Both releases of a description-only transition load; the old one is
      rejected once removed.
- [ ] A custom seed works with a custom anchor.
- [ ] A merge check regenerates every published release when its inputs,
      `Cargo.lock`, `rust-toolchain.toml`, or the generator's dependency
      closure change.
- [ ] Jolt loads through the loader, admits program rows only within the
      universe and with the anchored dense profiles, and drops the rebuild and
      serde paths. Swapping a program row changes `preprocessing_digest`.
- [ ] The guest checks embedded preprocessing with the loader and reads none
      from input.

### Testing Strategy

- Determinism: thread counts, instruction sets.
- Known-answer digest and corpus coverage on every instruction set.
- Tampering: byte flip, truncation, extension, reordered or duplicate entries,
  change after hashing.
- Anchor: wrong set, wrong kind, mixed releases.
- Admission: A1 and A4 agree; mid-page matrix ranges.
- Program rows: profile outside universe, wrong precommitted profile,
  duplicate base key, planner determinism.
- Audit mode and loader build identical state.

### Performance

From the measurements (one recipe: one-hot, K=16, w1r0, 57 polynomials):

| Item | Value |
|---|---|
| Slot export share of setup | ~90% (sequential today) |
| Matrix derivation share of setup | ~3% at the largest shape |
| Verifier registry, largest shape | ~21 KB |
| Prover hints, largest shape | ~93 MiB |
| Jolt first-use rebuild (removed) | 7–55x verification time at 2^22 cycles |
| Peak setup memory today | ~11x the matrix (keeps every slot source) |
| Proof size | Unchanged |

## Design

### Crates

```text
akita-types        manifest, set record, anchor types; hash and anchor checks
akita-config       set-record check, commitment_digest, row admission (shared)
akita-verifier     verifier loader → PinnedVerifierSetup, PinnedTerminalCache
akita-cpu-backend  prover loader → PinnedProverSetup
akita-setup        disk cache keyed by bundle_id, through the prover loader
akita-pcs          pinned entry points for Jolt
akita-preprocess   generator lib + binary; no akita-setup, no akita-anchors
akita-anchors      compiled anchor; nothing depends on it
jolt               Jolt generator binary (akita-preprocess + jolt-akita)
```

The generator takes a role (`all`, `prover`, `verifier`, guest target). The
role only filters output files; `bundle_id` is the same.

### Jolt

```text
published (per flavor, K, chunk profile)        per program
├─ one-hot set: base catalog + universe         ProgramPreprocessing
├─ bounded-dense set: advice, bytecode, image   ├─ bundle_id, set names
└─ full-dense sets: field-increment arities     └─ program rows
```

- Grouped rows cannot be published: ~10^8 keys per recipe, each a distinct
  ~15.6 KB row.
- Profiles saturate: ~36 cover every sampled row, ~70 MB of hints, ~18 KB of
  commitments.
- Program rows MUST live in `ProgramPreprocessing`, not `pcs_setup`, so
  `preprocessing_digest` covers them.
- Precommitted profiles in a row MUST equal the anchored bounded-dense set's.
- `find_adapted_schedule` MUST pick only universe profiles. A program outside
  the universe fails until a release widens it.
- A row never brings a commitment; all come from the anchored registry.
- Removed: the serde `AkitaVerifierSetup` in `pcs_setup` and its first-use
  rebuild.

Local profile derivation (opt-in, host only): for custom profiles outside any
universe, both sides compute the missing commitments themselves through an
`unsafe fn` constructor. Never in the guest.

### Transcript binding

Protocol Stage 3 absorbs the slot id (today) and its `commitment_digest`
(new), recomputed every proof from the registry entry the fold reads. It
names the commitments a proof used; soundness comes from the load checks.
Cost: a descriptor and protocol version bump, regenerated catalogs, a Jolt
revision bump. Land it before the first release.

The anchor is not absorbed: it is local policy and changes whenever any
release is added. `bundle_id` is not absorbed: proofs would stop carrying
across releases with identical commitments. Jolt binds `bundle_id` through
`preprocessing_digest`.

### Recursion guest

```text
approved guest image
├─ anchor
├─ manifest, set record, catalog view, registry entries   hashed at startup
├─ matrix prefix, terminal caches                          vouched by approval
└─ inner ProgramPreprocessing + program rows               admitted, not planned
```

- Reads no preprocessing from input.
- Large embedded files MAY skip hashing through a guest-only `unsafe fn`.
- Approval: one reproducible ELF snapshot; embedded files checked against the
  manifest; Jolt's program commitment computed from the same snapshot.

### Publication and audit

- Publish: commit description + manifest under `params/<release>/` and the
  anchor pairs, in one commit. Large files go to release assets (untrusted).
- Audit: a required merge check regenerates every anchored release when its
  inputs, `Cargo.lock`, `rust-toolchain.toml`, or any crate in the generator's
  closure change. Anyone can run it.

### Security argument

Assume `H` is collision resistant, the build and source are authentic, the
anchor is authentic, and Jolt holds the authentic program digest.

1. Production constructors take only loader values (Invariant 6).
2. Every used file has the bytes the anchored manifest names (Invariants 3, 4).
3. Those bytes are the generator's from-scratch output (merge check,
   Invariant 2).
4. The set matches the running code (Invariant 5, known-answer digest).
5. So every commitment is `Commit(S[0..n_prefix], profile)`, the matrix is
   derived from `S`, and every row passes admission and uses only anchored
   commitments.

Still trusted: the repo and CI; completeness of the row admission check;
the set-name derivation; Jolt's program digest; guest image approval; the
author of a custom release.

### Alternatives Considered

| Alternative | Rejected because |
|---|---|
| Hash the executable | Platform-dependent; bloats binaries |
| Compile registry and catalogs in | No custom seeds or catalogs |
| Anchor of bare `bundle_id` | Lets an adversary pick any set or mix releases |
| Cargo feature gate on unchecked constructors | Features are additive across the graph |
| Verifier self-test of commitments | Needs `akita-cpu-backend` and the full source prefix |
| Set-level identity in `SetupSection` | Stage 3 already binds exactly what is used |
| Publish all Jolt grouped rows | ~129k rows and 2 GB for advice alone; ~10^8 keys overall |
| One registry file | Every verifier would hash everything |
| Ship the matrix | 4 GiB; deriving is ~3% of setup |
| Merkle tree over pages | Flat page hashes give the same checks |
| Derive longer-prefix commitments from shorter | Outer gadget and compression layers are not linear |
| Scheduled regeneration only | Manifests are opaque in review |

## Documentation

- `docs/security-posture.md`: anchor model; in-place views only on
  loader-vouched bytes.
- `book/src/how/verification.md`: what the verifier trusts, the loader,
  audit mode, Stage 3 binding.
- `book/src/usage/verifier-only.md`, `book/src/usage/jolt-recursion.md`:
  loading releases, custom anchors, guest embedding.
- `specs/external-schedule-catalog-ownership.md`: allow anchored guest
  catalog views; describe program rows.

## Execution

### Measurements

One recipe (one-hot, K=16, w1r0, 57 polynomials), one multi-core x86-64
machine, release build. `nv = 26` is a 2^22-cycle program. Times are ratios.

| nv | Matrix | Slots | Slot export share | Setup vs `nv = 26` | Hints | Commitments |
|---|---|---|---|---|---|---|
| 26 | 64 MiB | 3 | 55–70% | 1x | 1.5 MiB | 1.5 KB |
| 28 | 256 MiB | 9 | 78–85% | 5x | 7.3 MiB | 4.6 KB |
| 32 | 1 GiB | 25 | 92–94% | 45x | 38 MiB | 12.7 KB |
| 40 | 4 GiB | 41 | 90–92% | 150x | 93 MiB | 20.8 KB |

- Using all cores speeds setup only ~3.5x: slots export one after another.
- Peak memory: 8–14x the matrix, because every slot source is kept. Releasing
  each after use: ~2–2.5x (estimated).
- Parallel BLAKE3 hashes 4 GiB ~12x faster than deriving it.

Grouped rows (sampled with a reimplementation of Jolt's `plan_row`):

| Grid | Keys | Sampled | Distinct rows | Slow fallback |
|---|---|---|---|---|
| Advice | 129,360 | 6,000 | 6,000 | 5.7% |
| Committed program | 57,171,240 | 871 | 871 | 35% |

- Profiles: 16 in the base catalog, ~36 total, flat after ~1,200 rows.
- Fallback rows are 10^4–10^5x slower than guided ones.
- Not sampled: field increment, K=256, w8r2.

### Generation

1. Derive each stream's matrix once, hashing pages in the same pass.
2. Collect profiles across all sets; merge slots that differ only in
   `natural_len`.
3. Generate commitments in parallel, bounded by memory; borrow sources, free
   them after use.
4. Write files, then the manifest.

### Steps

1. Types, encodings, admission, known-answer corpus, tamper tests.
2. Generator, determinism and poisoned-cache tests.
3. Loaders, `Pinned*` types, `unsafe fn` conversions, compile-fail tests.
4. Stage 3 binding (before the first release).
5. First release, anchor crate, merge check.
6. Jolt: sets, universes, program rows, planner restriction, coverage test.
7. Recursion guest.

### Open questions

1. Can the universe be enumerated from the planner instead of sampled? What
   does restricting the planner cost? Universes for K=256, w8r2, field
   increment. Do bounded-dense rows need slots?
2. Guest: which embedded files to hash; cost of per-proof digests.
3. Seed: keep the fixed `sample_akita_setup_seed()` (proposed).
4. Transition window: one software version (proposed).
5. Regeneration runner size: unmeasured for a full multi-recipe release.
6. Should the planner fallback be made cheaper first?

## References

- [`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md)
- [`external-schedule-catalog-ownership.md`](external-schedule-catalog-ownership.md)
- #223 (with #233, #234, #235), #210, #186, #187
- RFC 2119
