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

The soundness of an Akita verifier depends on preprocessing that the verifier
does not currently check. The setup-prefix commitment registry is accepted
after structural checks only (`AkitaVerifierSetup::from_parts`, setup
deserialization, the disk cache in `akita-setup`, and
`AkitaProverSetup::to_verifier_setup`). A supplied terminal NTT cache is
accepted after identity and shape checks (`AkitaVerifier::for_selection`). The
recursion guest reads its catalog, matrix, and registry from untrusted input. A
registry entry that is not the commitment to its stated setup prefix allows a
prover to open that entry to a vector of its choice. A party able to replace
these inputs can therefore cause a verifier to accept false openings.

This specification defines preprocessing as a published, pinned artifact. A
deterministic generator transforms a release description into content-hashed
release files and a manifest, whose hash is the release's `bundle_id`. A loader
accepts a parameter set only if its `(bundle_id, set name)` pair is contained
in the caller's trust anchor, verifies every file it uses against that
manifest, and issues the only values from which production provers and
verifiers can be constructed. Production builds compile in the anchors
published in the repository; other deployments, such as one using a different
seed, supply their own. A required merge check regenerates every published
release from scratch whenever its inputs, or the code that derives it, change.

## Conventions

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
interpreted as described in BCP 14 (RFC 2119, RFC 8174) when, and only when,
they appear in all capitals.

Section names in parentheses, such as (Admission), refer to sections of this
document.

## Intent

### Goal

This specification defines a **preprocessing release**, consisting of a
commitment registry keyed by commitment profile, named parameter sets with
their catalogs, prover hints, and a manifest. It further defines the release's
`bundle_id`, the trust anchors that accept its parameter sets, the loader and
the values it issues, and the rules by which provers and verifiers load, check,
and bind preprocessing.

All preprocessing on which soundness depends, or which is expensive to derive,
is program-independent and published; this comprises the setup-prefix
commitments. Jolt's per-program schedule rows are the exception: they are
carried in the program's preprocessing, are covered by its digest, and are
admitted against a published parameter set.

### Threat model

The adversary can replace any preprocessing bytes that a prover or verifier
reads at run time from storage or from a network, including while those bytes
are being read. This includes setup files, registry files, cache files,
catalog artifacts, release assets, and the input of a recursion guest.

The adversary cannot modify the environment that builds executables and guest
images, the source tree at the source commit, the resulting executables, the
trust anchor compiled into an integration, an approved guest image, or the
program preprocessing digest that a Jolt deployment trusts for the program it
verifies. These are the trust roots. A party controlling the build environment
can modify the verifier's code, and no check on data can defend against such a
party. Jolt program preprocessing may itself be replaced in storage; the
trusted program digest is the mechanism that detects such a replacement.

Impact of substitution:

| Object replaced | Effect at the verifier | Effect at the prover |
|---|---|---|
| Setup-prefix commitment | Soundness failure: the adversary chooses `C` and opens it to an arbitrary vector | Honest proofs fail |
| Public matrix, on a path that omits re-derivation (`borrow_trusted`, `from_trusted_seed_derived_parts_unchecked`, `Validate::No` decoding) | Soundness failure: a matrix with a known short vector breaks binding | Honest proofs fail; a hostile matrix may increase what the prover's messages disclose |
| Terminal NTT cache | As for the matrix it transforms | Not used |
| Catalog, a program's schedule rows, or the selected parameter set | Assuming a complete row admission check, substituted rows still pass that check and use only anchored commitments; the failure is one of policy: the verifier accepts rows or limits not approved for the deployment | Honest proofs fail; a hostile row may change how much of the witness is sent in clear |
| Prover hints | Not used | Honest proofs fail |

Akita does not claim witness confidentiality, and this specification does not
add such a guarantee. Prover inputs are nevertheless checked, because hostile
preprocessing can alter what an honest prover discloses before any verifier
rejects the proof.

### Definitions

**Stream identity.** The data that determines the public field stream: the
base field, the `AkitaSetupSeed` (derivation algorithm and seed bytes), and
the page size `PublicMatrixDerivation::page_field_elements()`. Page `i` of the
stream is a function of the stream identity and `i`. The encoding is specified
in (Encodings and hashes).

**Commitment algorithm identity.** The data that determines how a setup-prefix
commitment is computed from a commitment profile and the stream:

- `COMPRESSION_POLICY` and `GroupCommitPhaseParams::VERSION`;
- a new constant, `SETUP_PREFIX_PAYLOAD_VERSION`, the version of the
  setup-prefix payload encoding;
- the **known-answer digest**.

Versions that affect only the transcript (protocol version, descriptor
version) are not part of this identity, because they do not change
commitments. A descriptor version change nevertheless changes the catalog
`protocol_epoch` and therefore catalog bytes; it is consequently a code
transition for every release (Trust anchors and transitions).

**Known-answer digest.** A digest over a versioned, ordered corpus of small
commitment profiles evaluated on a fixed test stream. The corpus is stored in
the repository, and the digest is a compiled constant (Encodings and hashes).
A CI test MUST recompute the digest on every supported instruction set (x86-64
with and without IFMA52, and AArch64) and MUST fail on mismatch. A second CI
test MUST map every commitment profile in every anchored release to a corpus
class (field, A ring dimension, B ring dimension, inner and outer route, slice
count, compression chain length, multi-block layout) and MUST fail if a class
is not represented.

The corpus provides regression coverage at small sizes. It detects changes to
the gadget decomposition, transforms, block split, or payload code made without
a version change, and instruction-set-specific kernel defects. Size-specific
code paths are covered by full regeneration in the merge check, on the
instruction set of the runner. Extending the corpus changes the known-answer
digest, and therefore every commitment digest; it is a code transition. The
known-answer digest is most significant for custom releases, which the
repository's merge check does not regenerate.

**Commitment digest.** A setup-prefix commitment depends only on the stream,
the commitment algorithm, and the slot's `commitment_profile`. The
`natural_len` of a `SetupPrefixSlotId` is validated against the commitment
profile but does not affect the commitment; the committed source is always the
full prefix `S[0..n_prefix]`. A release therefore stores one commitment per
`(stream identity, commitment profile)` pair, together with a commitment digest
that binds the commitment to that context (Encodings and hashes). Slot
identifiers that differ only in `natural_len` share one commitment digest. A
loader MUST NOT accept a commitment digest without verification: it MUST
recompute the digest from the payload it loaded, the stream identity of the set
record, its own compiled algorithm identity, and the commitment profile, and
MUST compare the result with the manifest.

**Commitment registry.** The set of `(stream identity, commitment profile)`
entries covered by a release. The manifest stores one
`(stream_id, profile_key, commitment_digest)` triple per entry. The verifier
registry in the current code is keyed by the full `SetupPrefixSlotId`, which
includes `natural_len`. A loader MUST expand each entry into every slot
identifier required by its admitted rows, MUST retain the existing validation
of `natural_len` against the commitment profile, and MUST retain the full slot
identifier for transcript binding.

**Parameter set.** A named set consisting of one `CommitmentConfig`, one
catalog, one chunking policy, and one pair of provisioning limits. The limits
are upper bounds; the setup actually constructed is sized from the admitted
rows (Admission and matrix capacity). The catalog of a parameter set is
enumerated: it contains every row the set supports on its own, and no row is
added after publication. Catalogs are inputs to the release description,
identified by `catalog_digest()`, and are taken from the existing checked-in
schedule artifacts and their regeneration checks. A parameter set MAY
additionally admit **program rows** (Jolt's grouped rows; see Jolt), which are
planned per program and are not part of the release.

**Commitment profile universe.** For a parameter set that admits program rows,
the list of commitment profiles that those rows may use. The universe MUST
contain every commitment profile required by the set's catalog, and the
commitment registry MUST cover the universe in full. A program row that
requires a commitment profile outside the universe MUST be rejected.

**Set record.** The manifest's description of one parameter set: its name,
config family name, decomposition, SIS modulus profile, `SisTableDigest`,
`SisL2TableDigest`, commitment algorithm identity, stream identifier, chunking
policy, set limits, `catalog_digest()`, the commitment profiles required by its
catalog, and, if it admits program rows, its commitment profile universe.

**Release kind.** Either `akita` or `jolt`, recorded in the manifest. A loader
MUST reject a manifest of a different kind.

**Packaging header.** The release name, release kind, generator version, file
representation versions, verifier targets, and the rules by which large
release files are split into parts. Capacities of guest caches are derived
from the set records and are not declared.

**Manifest.** The canonical encoding of the packaging header, the stream
identities and their element counts, the commitment registry, the set records,
and the list of release files with their lengths and hashes, excluding the
manifest itself (Encodings and hashes). The encoding has a compiled maximum
length, fixed in Step 1 with documented headroom over the largest known
release. A release that would exceed this maximum requires a code transition.

**Bundle identity.** `bundle_id`, the hash of the manifest bytes (Encodings
and hashes).

**Trust anchor.** A non-empty list of `(bundle_id, set name)` pairs, typed by
release kind. A loader constructs one parameter set at a time, and only if the
requested pair is contained in the anchor. `TrustAnchor` MUST be constructible
only by a `const fn` from a `&'static` list. It MUST NOT have a deserializer, a
`FromStr` implementation, or any constructor from run-time bytes; a
compile-fail test MUST verify this. An anchor read from storage voids every
guarantee of this specification.

### Encodings and hashes

Every hash defined in this specification is computed over fully specified
bytes. A decoder MUST accept an encoding only if re-encoding the decoded value
yields exactly the input bytes, so that every value has exactly one valid
encoding.

**Primitives.**

| Notation | Meaning |
|---|---|
| `H(x)` | Blake2b-256 of `x`, unkeyed, with a 32-byte output; the existing `digest_descriptor_bytes` |
| `u8(v)`, `u32(v)`, `u64(v)` | `v` encoded as 1, 4, or 8 bytes, little-endian, as in the existing `push_u32` and `push_usize` |
| `bytes(x)` | `u64(len(x)) \|\| x` |
| `str(s)` | `bytes(UTF-8 encoding of s)` |
| `list(e_1, …, e_n)` | `u64(n) \|\| e_1 \|\| … \|\| e_n`; each element is either fixed-width or length-prefixed |
| `tag(t)` | `bytes(ASCII encoding of t)`; every hash input begins with a distinct tag, so inputs of different kinds cannot coincide |
| `ser(x)` | the existing uncompressed `AkitaSerialize` encoding of `x` |
| `hex(d)` | the lowercase hexadecimal encoding of a 32-byte digest, used in file names |

**Stream identity and stream identifier.** The stream identity is 73 bytes,
fixed width:

```text
stream_identity = prime_modulus_be          32 bytes: base field modulus,
                                            big-endian, zero-padded, as in
                                            AlgebraSection
               || ser(setup_seed)           33 bytes: u8(derivation tag,
                                            1 = Shake256PagedV1) || seed[32]
               || u64(page_field_elements)  8 bytes; currently 4096

stream_id = H(tag("akita/preprocessing/stream/v1") || stream_identity)
```

`ser(setup_seed)` is the input of the existing `setup_seed_digest`. The
`SetupSection` digest in the transcript is therefore `H(ser(setup_seed))` over
the same seed bytes that the stream identity contains.

**Commitment algorithm identity.** 41 bytes, fixed width:

```text
algorithm_identity = u8(COMPRESSION_POLICY.tag())          currently 4
                  || u32(GroupCommitPhaseParams::VERSION)   currently 4
                  || u32(SETUP_PREFIX_PAYLOAD_VERSION)      new constant
                  || known_answer_digest                    32 bytes
```

**Commitment profile descriptor and profile key.**

```text
profile_descriptor = u8(route)        0 = L-infinity inner route, 1 = L2
                  || bytes(p)         p = the existing GroupCommitPhaseParams
                                      descriptor bytes of the profile

profile_key = H(tag("akita/preprocessing/profile/v1") || profile_descriptor)
```

`p` is the output of the existing
`GroupCommitPhaseParams::append_descriptor_bytes`: version, group shape, block layout, outer slice count, and the inner and outer
matrix descriptors. On the L-infinity route the inner matrix descriptor writes
its table digest directly; on the L2 route it first writes the string
`akita-l2-route-v1`. Because these two layouts can coincide as raw bytes, the
leading `route` byte is REQUIRED to make the interpretation unambiguous. The
corresponding encoders in `akita-params` become public.

**Commitment payload and commitment digest.**

```text
payload = ser(SetupPrefixPublicCommitment<F>)

commitment_digest = H(tag("akita/preprocessing/commitment/v1")
                      || stream_identity        73 bytes
                      || algorithm_identity     41 bytes
                      || bytes(profile_descriptor)
                      || bytes(payload))
```

A loader MUST decode the payload with validation, re-encode it, and require
identical bytes before computing the digest. Protocol Stage 3 absorbs this
32-byte value (Transcript binding).

**Known-answer digest.**

```text
known_answer_digest = H(tag("akita/preprocessing/known-answer/v1")
                        || u32(corpus_version)
                        || test_stream_identity
                        || list(bytes(profile_descriptor_i) || bytes(payload_i)
                                for each corpus entry i, in corpus order))
```

The known-answer digest does not include the algorithm identity, of which it
is a component.

**Release files and file hashes.** Each release file is hashed together with
its name, so that content cannot be presented under a different name:

```text
file_hash = H(tag("akita/preprocessing/file/v1")
              || str(name) || bytes(content))
```

| File | Content |
|---|---|
| `registry/<hex(stream_id)>/<hex(profile_key)>` | `payload` |
| `hints/<hex(stream_id)>/<hex(profile_key)>` | `ser` of the slot's prover opening material, as produced by `export_setup_prefixes` |
| `catalog/<set>` | the catalog artifact bytes, as written by its existing encoder |
| `catalog-view/<set>/<hex(row digest)>` | the bytes of the selected-row verifier view |
| `matrix/<hex(stream_id)>/<first page>-<last page>` | the pages, each consisting of 4096 field elements in canonical little-endian encoding |
| `page-hashes/<hex(stream_id)>` | the 32-byte page hashes, concatenated in page order |
| `terminal-cache/<hex(stream_id)>/<hex(row digest)>/<target>/<capacity>` | the prepared terminal cache, including its binding header |

The hash of page `i` of a stream is:

```text
page_hash_i = H(tag("akita/preprocessing/page/v1")
                || stream_id || u64(i) || page_i bytes)
```

**Set record.**

```text
set_record = str(set name)
          || str(config family name)
          || bytes(decomposition descriptor bytes)
          || u8(sis_modulus_profile_tag)
          || SisTableDigest || SisL2TableDigest       32 bytes each
          || algorithm_identity                       41 bytes
          || stream_id                                32 bytes
          || bytes(chunking policy descriptor bytes)
          || u64(max_num_vars) || u64(max_num_polys)  set limits
          || catalog_digest                           32 bytes
          || list(profile_key)                        catalog profiles, sorted
          || u8(admits program rows: 0 or 1)
          || list(profile_key)                        universe, sorted; empty
                                                      when the flag is 0
```

**Manifest and bundle identity.**

```text
manifest = tag("akita/preprocessing/manifest/v1")
        || u32(manifest_format_version)
        || str(release name)
        || u8(release kind: 0 = akita, 1 = jolt)
        || str(generator version)
        || list(str(target))                              verifier targets
        || u64(maximum file part size)
        || list(stream_identity || u64(element count))    sorted by stream_id
        || list(stream_id || profile_key || commitment_digest)
                                                  sorted by stream_id,
                                                  then profile_key
        || list(bytes(set_record))                        sorted by set name
        || list(str(name) || u64(length) || file_hash)    sorted by name

bundle_id = H(tag("akita/preprocessing/bundle/v1") || manifest)
```

All orderings are bytewise ascending and strict. A duplicate key, an unsorted
list, an unknown version or kind, trailing bytes, or a length exceeding the
compiled maximum MUST cause a decoding error. The manifest does not list
itself.

**Trust anchor entries.** An entry `(bundle_id, set name)` is compared as the
32-byte digest and the exact UTF-8 bytes of the name.

### Release files

The exact names and contents of release files are specified in (Encodings and
hashes). The following table states the purpose of each file.

| File | Purpose | Used by |
|---|---|---|
| `manifest` | Lists every other file; authenticated by `bundle_id` and not listed in itself | Both |
| `catalog/<set>` | The enumerated catalog, whose `catalog_digest()` the set record stores | Both |
| `catalog-view/<set>/<row>` | A selected-row verifier view of that catalog, hashed as a separate file | Guest builds |
| `registry/<stream>/<profile>` | One setup-prefix commitment payload | Both |
| `hints/<stream>/<profile>` | Prover opening material for one commitment, produced by the CPU reference path | Prover |
| `matrix/<stream>/<range>` | OPTIONAL. Complete derived pages, split by page range | Guest builds, constrained provers |
| `page-hashes/<stream>` | One hash per page of the stream | Loaders of matrix ranges, auditors |
| `terminal-cache/<stream>/<row>/<target>/<capacity>` | One prepared terminal cache, named by every field of its binding header | Guest builds |
| `certificate/<stream>/<profile>` | Reserved for a future proof of derivation | Future use |

Commitments are stored as separate files so that a verifier hashes only those
used by its admitted rows. Matrix files contain complete pages. The manifest
records the usable element count of each stream separately. A loader that
requires a range ending within a page MUST authenticate the whole page before
truncating it.

### Admission and matrix capacity

By default, matrix files are not loaded, because deriving the matrix is
inexpensive relative to the remainder of setup (Measurements). Capacity is
sized from the rows that a construction admits, in four admission steps:

- **A1.** Admit rows with `TrustedScheduleCatalog::rows_within_setup_capacity`
  (a single row with `for_selection`). For a parameter set without program
  rows, the applicable bounds are the set limits. For a parameter set with
  program rows, the applicable bounds are the program bounds (for Jolt,
  `max_num_vars`, polynomial counts, and the largest precommitted arity, taken
  from its program preprocessing), which MUST NOT exceed the set limits. The
  admitted rows are the base rows whose final group equals the program's shape,
  together with its program rows. A program row MUST be admitted only if, in
  addition, every setup-prefix slot that
  `required_setup_prefix_slot_ids_for_schedule` derives for it maps to a
  commitment profile in the set's commitment profile universe. Slot
  identifiers MUST NOT be read from fields of the row.
- **A2.** Size the matrix: for a verifier, the maximum of
  `verifier_setup_matrix_capacity_for_schedule` over the admitted rows; for a
  prover, the canonical requirement of the rows it may prove.
- **A3.** Derive that prefix, and construct the setup descriptor from the set
  limits and the derived element count.
- **A4.** Run `verifier_admits` against the resulting setup, and require that
  it admits exactly the rows admitted in A1.

A small program therefore never derives the matrix of the largest supported
program. In the current code, Jolt's verifier retains the prover's full
capacity (`setup_verifier`), which is 4 GiB at the largest measured shape.
Prover hints are loaded only for the slots that the admitted rows require.
When no cache is supplied, `for_selection` prepares the terminal cache
internally, so native hosts load none.

Admission and the set-record check are implemented in `akita-config`, on
which both loaders already depend, so that prover and verifier execute the
same code.

### Trust anchors and transitions

Production and custom deployments use the same loader with different anchors.

- **Production.** Akita exports a compiled anchor from a crate on which no
  other crate depends (working name `akita-anchors`), and Jolt exports one
  from a crate of its own. Each anchor lists the `(bundle_id, set name)` pairs
  of the releases committed under `params/` in its repository. Anchor crates
  MUST be outside the generator's dependency closure, so that adding a release
  does not affect any `bundle_id`; `scripts/check-crate-deps.sh` MUST enforce
  this for the generator and for its Jolt counterpart.
- **Parameter set selection.** The integration MUST derive the set name from
  its own state (for Jolt, its flavor, K, and chunk profile), and MUST NOT
  derive it from transported bytes alone. Jolt additionally carries the
  `bundle_id` and set names in its program preprocessing, covered by the
  trusted program digest, and the integration MUST require the carried names
  to equal the derived names. The loader MUST reject a set name that the
  anchor does not list for the given `bundle_id`.
- **One manifest per construction.** Every file used to construct a prover or
  verifier MUST be verified against the single manifest whose hash passed the
  anchor check. Files from different releases MUST NOT be combined. A
  construction MAY load several parameter sets from that manifest; Jolt loads
  a one-hot set together with the bounded-dense and full-dense sets required
  by its producers.
- **Custom deployments.** A deployment with its own seed, catalogs, or limits
  generates its own release with the same generator and supplies its own
  anchor. The checks are identical; the deployment owns the root of trust.
- **Description-only transitions.** When a release changes only in its
  description (catalogs, limits, or parameter sets), the new release is added
  under `params/` and its pairs are added to the anchor in a new software
  version. The previous description and manifest remain in the repository, and
  the previous pairs remain in the anchor, for the duration of the transition
  window; a later software version removes both. Both releases regenerate
  bit-identically under the same code and therefore both remain valid under
  the merge check.
- **Code transitions.** A change to the commitment algorithm identity, or to
  anything that changes catalog bytes (for example the catalog
  `protocol_epoch`, which reuses `AKITA_INSTANCE_DESCRIPTOR_VERSION`), causes
  previous releases to fail regeneration. The anchor is then replaced in a
  single step, in the same software version. Supporting old and new algorithms
  concurrently would require version dispatch, which this specification does
  not provide.
- **Revocation.** Pairs are revoked by removing them from the anchor in a new
  software version. Binaries built earlier continue to accept them. A
  key-based anchor could support revocation without rebuilding, but would
  additionally require freshness or rollback protection; it is out of scope.

### Invariants

1. **Determinism.** Every release file, the manifest, and `bundle_id` MUST be a
   pure function of the release description and the code at the source commit.
   They MUST NOT depend on thread count, machine, operating system,
   instruction set, backend, or build profile. Hints are produced by the CPU
   reference path, and any other backend that produces them MUST reproduce
   those bytes exactly. Test: generate one release at two thread counts, on
   x86-64 and on AArch64, and compare `bundle_id`.
2. **Generation from scratch.** Generation and audit MUST derive every object
   from the release description alone. They MUST NOT read an ambient disk
   cache, a previous bundle, or the candidate under audit. Test: place a
   poisoned registry in the `akita-setup` cache directory and require
   regeneration to ignore it.
3. **Hashed bytes are used bytes.** A loader MUST read each file into a
   private, immutable buffer while hashing it, MUST check the manifest length
   before allocating, and MUST pass exactly that buffer to the decoder. It MUST
   NOT reopen a path or reread a file after hashing. Registry, catalog, and
   hint files MUST be copied. Matrix ranges MAY be read through private
   page-sized snapshots, each verified against `page-hashes` before use. A
   loaded value MUST NOT be refreshed, reloaded, or extended from storage
   except through the loader, which hashes again. This requirement applies to
   lazily populated caches (such as Jolt's first-use rebuilds) and to
   memory-mapped files; a lazy cache MAY be populated only by derivation from
   values the loader has already issued. Program rows are the only input that
   is not a release file: the loader receives them as an explicit argument,
   authenticated by the program preprocessing digest, and admits them
   (Admission and matrix capacity). A stored hash MUST be compared only against
   bytes hashed during the same load. Test: modify a file between hashing and
   use, and after loading.
4. **Anchor first.** A loader MUST hash the manifest bytes and MUST require the
   pair `(hash, requested set name)` to be in its anchor, with a matching
   release kind, before decoding anything else. Manifest decoding MUST be total
   and bounded under the verifier no-panic contract. Unknown representation
   versions MUST be rejected without fallback.
5. **The running code matches the release.** Before using any release file, a
   loader MUST reconstruct the expected set-record fields (config, SIS digests,
   commitment algorithm identity including the known-answer digest, field, and
   page size) from its compiled code and MUST require equality with the
   selected set record. Admission limits MUST be taken from the verified set
   record and never from loaded bytes.
6. **No data path reaches a production constructor.** The loader issues
   `PinnedVerifierSetup` and `PinnedTerminalCache` (in `akita-verifier`) and
   `PinnedProverSetup` (in `akita-cpu-backend`). These types MUST have private
   fields, no public constructor, and no deserializer. `PinnedVerifierSetup`
   carries the setup of the admitted rows, the `TrustedScheduleCatalog`, and
   the chunking policy from the set record. When program rows are present, its
   catalog is `ValidatedScheduleCatalog::try_new` over the admitted base rows
   and program rows, which rejects duplicate keys and row digests; the set
   record's `catalog_digest()` authenticates the base catalog file only, and
   Jolt's transcript binds the digest of the merged catalog.
   `PinnedProverSetup` is constructed in the same manner. The production
   constructors of `AkitaVerifier` and `CpuBackend`, and the pinned
   replacements for the `AkitaCommitmentScheme` verifier entry points in
   `akita-pcs`, MUST accept only these values.

   Every path that constructs a setup from loaded or caller-supplied bytes MUST
   either become private to the loader or become an `unsafe fn` whose contract
   states that the caller guarantees derivation. This applies to:
   - `AkitaVerifierSetup` itself (its fields, `from_parts`, and its
     deserializer, which cannot be made private), which becomes the unchecked
     type used only by test, benchmark, and loader-internal code;
   - `AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked` and
     `Validate::No` setup decoding;
   - `borrow_trusted`, and `TrustedTerminalCache` in `for_selection`;
   - the public fields of `AkitaProverSetup`, and `to_verifier_setup`;
   - `AkitaCommitmentScheme::setup_verifier`, `setup_verifier_for_schedule`,
     and `verifier`;
   - the `AkitaVerifier` constructor that accepts a
     `LocallyDerivedVerifierSetup` (contract: the caller computed every
     commitment locally from the derived stream).

   Jolt and the production guest MUST deny these functions with
   `clippy::disallowed_methods`, and negative compile tests MUST verify that a
   production constructor rejects any value not issued by the loader. This
   prevents loaded bytes from reaching a production constructor by accident or
   through a decoder; it is not a cryptographic guarantee, since the threat
   model already trusts the executable. Here `unsafe` denotes a soundness
   contract rather than a memory-safety obligation. This use is intentional,
   so that every call site is visible and can be denied; it does not affect
   the verifier no-panic contract.
7. **Prover slots are recomputed unless their bytes are verified.** The loader
   MUST reject any release file that does not match the manifest. A prover MAY
   then be constructed without the hints of a `PinnedProverSetup`, in which
   case the backend recomputes the affected slots. The backend MUST adopt a
   slot without recomputation only if the slot was produced by a backend in the
   same process, or if it is contained in a `PinnedProverSetup` whose registry
   entry and hints both matched the manifest. This requirement protects
   liveness and limits disclosure; it does not affect verifier soundness. A
   slot produced in-process by a backend other than the CPU reference path MUST
   match the reference payload byte for byte, because the verifier checks
   proofs against the published commitment.
8. **Audit mode.** The generator crate MUST be able to derive the commitment
   registry and the set records from the release description and compare them
   with the manifest. Audit mode runs on hosts able to link
   `akita-cpu-backend`; the verifier crate does not link it.

### Non-Goals

- A proof of derivation (the certificate chain of
  [`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md#setup-provenance-and-validation)).
  This specification provides the operational trust root that the referenced
  specification requires and reserves a per-entry file for a certificate.
- Witness confidentiality.
- Authenticated catalog subsets for recursion (the deferred authenticated
  recursion layer of
  [`external-schedule-catalog-ownership.md`](external-schedule-catalog-ownership.md#deferred-authenticated-recursion-layer)).
- Key-based anchors and revocation without rebuilding.
- Concurrent support for old and new commitment algorithms.
- Backward compatibility.

## Evaluation

### Acceptance Criteria

- [ ] The generator writes every release file and the manifest of a release,
      and `bundle_id` is identical across thread counts and on x86-64 and
      AArch64.
- [ ] Generation ignores a poisoned ambient cache.
- [ ] The known-answer digest test passes on every supported instruction set
      and fails when a change to the compression policy or to a commitment
      kernel alters any corpus output; the coverage test fails when an anchored
      commitment profile has no corpus class.
- [ ] A verifier constructed through the loader loads catalog and registry
      entries, derives its matrix for its admitted rows, and rejects: a
      manifest outside its anchor, a set name not anchored for that manifest, a
      manifest of the other release kind, and a one-byte change in any release
      file it uses.
- [ ] A verifier whose compiled code differs from the set record rejects the
      release before using any release file.
- [ ] Compile-fail tests show that production constructors reject every value
      not issued by the loader, and Clippy runs on Jolt and on the guest fail
      when they call a denied `unsafe fn`.
- [ ] Protocol Stage 3 absorbs a commitment digest recomputed during the proof
      from the setup-prefix commitment it uses, and a proof fails when a
      commitment it uses differs.
- [ ] A production anchor accepts both releases of a description-only
      transition and rejects the previous release once it is removed.
- [ ] A custom anchor with a release generated from a different seed operates
      with the same loader.
- [ ] The repository contains the manifest of at least one release, and a
      required merge check regenerates it whenever its description, manifest,
      or anchor, `Cargo.lock`, `rust-toolchain.toml`, or any crate in the
      generator's dependency closure changes.
- [ ] `TrustAnchor` cannot be constructed from run-time bytes (compile-fail
      test).
- [ ] Jolt loads Akita preprocessing through the loader; admits program rows
      only when every commitment profile they require is in the set's universe
      and every precommitted profile equals that of the anchored bounded-dense
      set; passes a profile-coverage test; and no longer rebuilds the prover
      setup or accepts serde-transported verifier setups. Replacing a program
      row changes the program preprocessing digest.
- [ ] The production guest verifies its embedded preprocessing with the loader
      and reads no preprocessing from its input; its approved program
      commitment is computed from the same ELF snapshot that the approval step
      checked. A Jolt guest admits the embedded inner program's rows with the
      same loader call as the host.

### Testing Strategy

- Determinism at two thread counts and on two instruction sets.
- Known-answer digest and corpus coverage on every supported instruction set.
- Poisoned cache (Invariant 2).
- Tampering: a one-byte change in each release file, truncation, extension,
  reordered or duplicated manifest entries, and modification after hashing.
- Anchors: wrong set name, wrong release kind, and files from two releases.
- Config and algorithm mismatch.
- Admission: steps A1 to A4 produce the same admitted rows before and after
  sizing.
- Prefix loads: matrix ranges that end within a page.
- Audit equivalence: audit mode and the loader construct identical verifier
  state.
- Program rows: a row requiring a commitment profile outside the universe, a
  row with an incorrect precommitted profile, a row duplicating a base key,
  and planner determinism (`find_adapted_schedule` on sampled keys at two
  thread counts and on two instruction sets).
- End-to-end tests against generated releases. The CI test matrix in
  `.github/workflows/ci.yml` remains authoritative for feature sets.

### Performance

The measurements cover one Jolt recipe (one-hot, K=16, chunk profile w1r0, 57
polynomials) and a sampled sweep of its grouped rows.

- **Generation.** For the measured recipe, slot export accounts for about 90%
  of setup time at the largest shape, and slots are currently processed
  sequentially, so using all cores accelerates setup by only about 3.5x. A
  Jolt release per recipe comprises about 36 registry entries together with
  bounded-dense and full-dense sets; its generation cost is estimated to be of
  the same order as that largest shape. A release spans several recipes (K,
  chunk profiles), so total regeneration time is a multiple of that estimate.
  It has not been measured.
- **Generation memory.** Peak memory SHOULD remain close to the size of the
  matrix plus the largest entry. The current setup peaks at about 11 times the
  matrix at the largest shape, because it retains a copy of every slot's
  source (Measurements).
- **Program preprocessing.** Planning a grouped row takes milliseconds on the
  guided path. The full-search fallback, which applies to 5% to 35% of sampled
  rows, is four to five orders of magnitude slower. It is incurred once per
  program.
- **Verifier construction.** A loader-constructed verifier hashes the registry
  entries it uses (about 21 KB in total at the largest measured shape) and its
  catalog, and derives the matrix for its admitted rows. It no longer incurs
  Jolt's first-use rebuild, which costs about 7 to 55 times the verification
  time for a 2^22-cycle program and about one full setup at the largest
  measured shape.
- **Prover construction.** A loader-constructed prover loads hints (up to
  about 93 MiB at the largest measured shape) instead of exporting slots, which
  accounts for about 90% of setup time at that shape.
- **Proof size** is unchanged. Protocol Stage 3 absorbs a 32-byte commitment
  digest per setup-prefix slot; the proof does not carry it.
- **Per-proof commitment digest recomputation** hashes only the commitments of
  the selected row, a few kilobytes. The cost is negligible on native hosts and
  has not been measured in a guest (Open questions).

## Design

### Architecture

```text
 release description (in repo)
   packaging header + stream identities + named parameter sets
                │
                ▼
     generator (host only, from scratch, deterministic, parallel)
                │
   ┌────────────┼──────────────────────────────┐
   ▼            ▼                              ▼
 registry/*  catalog/<set>               prover hints per commitment;
 (one per    + set records               optional matrix ranges, page
  profile)                               hashes, and guest caches
   └────────────┴───────────► manifest ◄───────┘
                                  │
                                  ▼
       bundle_id = H(manifest)  ──►  anchor: {(bundle_id, set name)}
                                  │
                                  ▼
       loader ──► PinnedVerifierSetup / PinnedTerminalCache / PinnedProverSetup
```

**Crates.**

- **`akita-types`**: manifest, commitment digest, set record, and anchor types;
  manifest decoding; and the hash and anchor checks. Its output is evidence
  that a release file matched an anchored manifest. It cannot check a set
  record against `CommitmentConfig`, because `akita-config` depends on it.
- **`akita-config`**: the set-record check against `CommitmentConfig`,
  commitment digest recomputation, and row admission, including the commitment
  profile universe check. Both loaders call this single implementation.
  `akita-cpu-backend` may not depend on `akita-verifier`, so this logic cannot
  reside in the verifier crate.
- **`akita-verifier`**: the verifier loader. It runs the `akita-types` and
  `akita-config` checks, constructs the `TrustedScheduleCatalog`, sizes the
  setup, and issues `PinnedVerifierSetup` and `PinnedTerminalCache`, which wrap
  its `pub(crate)` cache types.
- **`akita-cpu-backend`**: the prover loader and `PinnedProverSetup`.
  `CpuBackend` is the prover-side constructor that accepts it. `akita-prover`
  does not reference it, since `akita-cpu-backend` depends on `akita-prover`.
  The disk cache of `akita-setup` becomes a cache of release files keyed by
  `bundle_id` and calls this loader, which resolves #210.
- **`akita-pcs`**: pinned replacements for the `AkitaCommitmentScheme`
  verifier and setup entry points used by Jolt, accepting loader-issued
  values.
- **Generator** (working name `akita-preprocess`): a new host-only crate with a
  library and a thin binary. It depends on `akita-config`, `akita-planner` (to
  enumerate commitment profile universes), `akita-cpu-backend`, and the
  verifier's public `build_riscv64_terminal_ntt_cache`. It MUST NOT depend on
  `akita-setup`, whose disk cache Invariant 2 forbids it to read.
  `scripts/check-crate-deps.sh` gains a case for it, adds it to the forbidden
  list of every verifier-facing crate, and forbids it from depending on
  `akita-anchors`; the preflight list in `AGENTS.md` is extended accordingly.
- **Jolt generator**: a Jolt binary that links the `akita-preprocess` library
  and `jolt-akita`, supplies Jolt's catalogs, dense layouts, and commitment
  profile universes, and runs in Jolt's merge check.
- **Anchor crates**: `akita-anchors` in Akita and an equivalent crate in Jolt,
  both outside the generator's dependency closure. The loaders receive the
  anchor as a parameter from the integration and do not link an anchor crate,
  so the generator's use of `akita-verifier` introduces no dependency cycle.
- **Local profile derivation**: `LocallyDerivedVerifierSetup` and its opt-in
  constructor reside in a host-only crate downstream of `akita-cpu-backend` and
  `akita-planner`, which constructs an `AkitaVerifier` through a dedicated
  `unsafe fn` constructor (Invariant 6). `akita-verifier` does not reference
  the type.

**Generator roles.** The binary takes a release description and a role
(`all`, `prover`, `verifier`, or a guest target). The role only selects which
files are written; every role computes the same `bundle_id`.

### Jolt

A Jolt release separates preprocessing according to its dependencies.

- **Program-independent, published.** One one-hot parameter set per flavor, K,
  and chunk profile, carrying the base catalog (currently 338 rows for K=16
  and w1r0, about 4.4 MB) and a commitment profile universe covering every
  grouped row that Jolt can plan for it; one bounded-dense parameter set
  covering the arities that Jolt commits (advice and committed bytecode and
  image, up to `DIRECT_PROGRAM_MAX_PHYSICAL_VARS`), whose profiles are the
  frozen precommitted profiles used by grouped rows; and one full-dense
  parameter set per field-increment arity (18 arities, 14 to 31, over the K=16
  trace range), which requires no setup-prefix slots. Whether bounded-dense
  rows require setup-prefix slots has not been determined. The commitment
  registry holds one commitment per profile in the universe. For the measured
  recipe this amounts to about 36 profiles, including the 16 used by the base
  catalog, about 18 KB of commitments, and about 70 MB of prover hints
  (Measurements).
- **Per program.** The key of a grouped row is the final group together with
  the ordered list of producers (untrusted and trusted advice, committed
  bytecode and image, field increment), each with its own arity. For one
  recipe this key space contains about 10^8 keys, every key yields a distinct
  row, and a row occupies about 15.6 KB; grouped rows therefore cannot be
  enumerated (Measurements). Jolt plans the rows required by a program during
  its program preprocessing, as it does in the current code with
  `find_adapted_schedule`, and stores them, together with the `bundle_id` and
  set names, in `ProgramPreprocessing` (in both its full and committed
  variants). They are consequently covered by `preprocessing_digest`, the
  Fiat-Shamir program binding that deployments trust. They MUST NOT be stored
  in `pcs_setup`, which the digest does not cover. A replaced row changes the
  digest, and a verifier holding the trusted program digest rejects it.
- **Admission.** A construction loads the one-hot set and the bounded-dense
  and full-dense sets it requires from the same manifest, and then admits the
  program rows with the program bounds and the one-hot set's commitment
  profile universe (Admission and matrix capacity). Every precommitted group
  profile in a program row MUST equal the frozen profile resolved from the
  anchored bounded-dense set for that layout. A program row never supplies a
  commitment: every setup-prefix commitment it uses is taken from the anchored
  registry. A row that requires a commitment profile outside the universe MUST
  be rejected with an explicit error.
- **Selection.** The Jolt verifier derives the set names from flavor, K, and
  chunk profile, checks them against the names carried in the program
  preprocessing, and retains its own transcript bindings of application data
  (bounds, `one_hot_k`, chunk profile, layout digest, and the digest of the
  merged catalog). `PlanSection` already binds the row used. The serialized
  `AkitaVerifierSetup` in `pcs_setup`, its transport, and its first-use rebuild
  are removed.
- **Coverage check.** A CI test MUST plan rows over a deterministic sample of
  the producer grid and over Jolt's example programs, and MUST fail if any row
  requires a commitment profile outside the set's universe.

The commitment profile universe must be complete for this design to be live.
Sampled sweeps saturate quickly (Measurements), but the committed-program
sample covers about 0.0015% of its keys, and the field-increment, K=256, and
w8r2 cases were not sampled; saturation within a sample therefore does not
establish completeness. The design consequently restricts the planner to the
universe: `find_adapted_schedule` MUST filter its candidates to profiles in the
universe, so that a program fails only if no feasible schedule exists within
the universe. The generator SHOULD additionally enumerate the universe from
the planner's candidate space rather than from samples. A gap in the universe
blocks the affected program until a new software version adds a release with
a larger universe, because anchors are compiled in.

**Local profile derivation (opt-in).** For a custom deployment whose rows
require commitment profiles outside every anchored universe, both parties
start from an anchored release and compute the missing setup-prefix
commitments themselves. The verifier MUST NOT import a commitment from the
prover or from a file, since a prover-chosen commitment constitutes a
soundness failure. The result is a distinct `LocallyDerivedVerifierSetup`,
which production constructors accept only through an explicit opt-in
constructor. It requires the commitment code in the verifier's build and is
therefore available only to host verifiers, never to the production guest.
Verifiers that do not opt in reject proofs relying on it.

### Transcript binding

Protocol Stage 3 already absorbs the encoded slot identifier
(`stage3_public_slot`, from `bind_setup_prefix_slot` in
`akita-verifier/src/stages/stage3.rs`; on the prover side at the corresponding
point in `akita-prover/src/protocol/prove/fold/sumcheck.rs`). It additionally
absorbs the slot's commitment digest. Prover and verifier MUST recompute the
commitment digest at protocol Stage 3 of every proof, from the registry entry
of the setup used by that proof that the fold relation reads in the same call,
and MUST NOT use a stored commitment digest. The commitment digest bound by a
proof therefore always describes the commitment that proof used. Only the
commitments of the selected row are hashed (Performance). Every fold with a
setup prefix passes protocol Stage 3 exactly once, so every use of a setup
prefix is bound exactly once; the root fold has no setup prefix, and the
terminal level rejects one. In the current code the payload enters the fold
relation (`akita-verifier/src/fold/suffix.rs`) but is not absorbed. The
catalog row and its effective schedule are already bound by `PlanSection`, so
no set-level identity is required.

This binding provides naming, not soundness. Once the load-time checks fix
each commitment, the prover cannot choose it, and a proof under a different
commitment already fails the opening relation. The binding causes each proof
to name the commitments against which it was checked. It requires a
transcript change (`AKITA_INSTANCE_DESCRIPTOR_VERSION` and protocol version
changes, which regenerate every checked-in catalog and every release) and a
Jolt revision update. It depends only on the commitment digest definition and
the code that recomputes commitment digests, and can therefore be delivered
separately. Delivering it before the first published release avoids a code
transition for that release.

### Recursion guest

Guest parameter sets are named sets in an anchored release. The guest embeds
the manifest, its catalog view, its registry entries, the set record, its
matrix prefix, and its terminal caches, and compiles in its anchor. At
startup it runs the same loader on the embedded bytes: it hashes the manifest,
checks the anchor and set name, checks the set record, and checks its catalog
view and registry entries against the manifest and their commitment digests.
Hashing the large embedded files (matrix prefix, terminal caches) in the guest
may cost too many cycles. These MAY instead be issued through a guest-only
`unsafe fn` whose contract is that the bytes are embedded in an approved image
and match the manifest. Only the guest crate may call this function
(`clippy::disallowed_methods` elsewhere), and the embedding guarantees its
alignment. Bytes that passed either check satisfy the canonical-word
obligation of the in-place views, because the generator writes only canonical
words.

A guest that verifies Jolt proofs with program rows (a grouped case, such as
the current multi-group recursive case) embeds the inner program's
`ProgramPreprocessing`, including its program rows, in the approved image. At
startup it admits those rows against the set limits and the commitment
profile universe with the same loader call as the host, and checks the
registry entries for their profiles against the manifest. The guest admits
rows but never plans them. One approved image covers the rows of one inner
program. Modes that read inner preprocessing from input are not production
modes.

The guest image is approved in the trusted build environment. A reproducible
build produces one immutable ELF snapshot; the approval step checks the
embedded files in that snapshot against the manifest and any embedded inner
program preprocessing against its trusted program digest; and Jolt's program
preprocessing and commitment are computed from the same snapshot. The outer
verifier trusts that commitment. The production guest reads no preprocessing
from its input. Which large files are hashed in the guest remains open, as no
guest cycle counts are available.

### Publication and audit

**Publication.** A release is published by committing its description and
manifest under `params/<release>/` in the owning repository and adding its
`(bundle_id, set name)` pairs to the anchor crate in the same commit. Large
release files are published as GitHub release assets, split so that each part
remains under the asset size limit. Release assets are untrusted transport.

**Audit.** A required merge check MUST regenerate every anchored release from
scratch whenever its description, manifest, or anchor changes, whenever
`Cargo.lock` or `rust-toolchain.toml` changes, or whenever any crate in the
generator's dependency closure changes (computed from `cargo metadata`). That
closure covers nearly all of Akita, so most Akita changes trigger the check,
and the runner is sized accordingly. In the Jolt repository the closure
includes Akita crates, so an Akita update that changes derivation triggers
the check. A scheduled job also regenerates all releases. Any party can run
the generator in audit mode.

### Security argument

Assume that `H` is collision resistant; that the build environment and the
source tree at the source commit are authentic; that the integration compiles
in an authentic trust anchor and derives the set name from its own state; that
a Jolt verifier holds the authentic program preprocessing digest; and that the
required regeneration check ran on every production release at that commit.
Then the following claims hold.

1. By Invariant 6, a production prover or verifier is constructed only from
   loader-issued values.
2. By Invariants 4 and 3, every release file used by the loader has the bytes
   named by the single manifest whose `(hash, set name)` passed the anchor
   check. In the guest, files issued through the guest-only path instead rely
   on the approval of the ELF snapshot that embeds them.
3. For production releases, by the regeneration check and Invariant 2, those
   bytes are the generator's from-scratch output on the release description
   at the source commit. For custom releases, this claim is the responsibility
   of the party that generated the release.
4. By Invariant 5, the selected parameter set matches the config and
   commitment algorithm identity of the running loader. That identity is tied
   to the compiled commitment code by the known-answer and coverage tests in CI
   and by full regeneration, not by the loader itself.
5. Consequently, every commitment in the registry satisfies
   `C = Commit(S[0..n_prefix], profile)` for the loader's stream `S`; the
   verifier's matrix prefix equals `S`, because the verifier derived it or, in
   the guest, because the approved image embeds it (claim 2); every catalog row
   passes the row admission check and belongs to the parameter set named by the
   integration; and every program row is the row covered by the trusted
   program digest, passes the same row admission check, and uses only
   commitments from that set's anchored registry. These are the preprocessing
   hypotheses of Akita's soundness claim.
6. Protocol Stage 3 binding causes each proof to name the commitments against
   which it was checked.

Residual trust assumptions:

- the repository, the build environment, and the merge checks; a party able
  to modify the anchor, the manifest, and the check together defeats the
  scheme, and independent audit mode is the countermeasure;
- the completeness of the row admission check, including against adversarial
  program rows: program rows are authenticated by the program digest rather
  than by the release, so the row admission check is their only further check;
- the integration's derivation of the set name;
- for Jolt, the trusted program preprocessing digest, which covers the program
  rows as it covers the program's own commitments;
- for the guest, the reproducible build, the approval of one ELF snapshot, and
  the outer verifier's trusted guest program commitment;
- for custom anchors, the party that generated the custom release;
- for local profile derivation, the local computation performed by each party.

### Alternatives Considered

**Hashing the executable instead of the data.** Executable hashes depend on
platform and toolchain, so no single published value could cover all builds,
and embedding prover data would enlarge every binary.

**Compiling the verifier's registry and catalog into the binary.** The
verifier would load nothing, but Akita would then operate only with the
compiled seed and catalogs. Anchors provide the same production guarantee
while permitting custom deployments to supply their own releases.

**Anchors consisting of bare `bundle_id` values.** These would allow an
adversary to select any parameter set within an anchored bundle, or to combine
files of two releases.

**A Cargo feature gate on the unchecked constructors.** Cargo features are
additive across a build graph, so any dependency could enable the gate for
every crate.

**A load-time self-test in the verifier.** Recomputing commitments requires
`akita-cpu-backend` and the full source prefix, neither of which is available
to the verifier. The known-answer digest and full regeneration in CI address
algorithm drift instead.

**A set-level identity in `SetupSection`.** Protocol Stage 3 commitment digest
binding binds exactly the commitments used, and `PlanSection` already binds
the row.

**Enumerating Jolt grouped rows in published catalogs.** Publishing one
parameter set per program shape, with every grouped row enumerated, does not
scale. At maximum bounds the advice grid alone comprises about 129,000 rows
and 2 GB, which is 8 times the trusted catalog's row limit and 30 times its
byte limit, and committed-program shapes extend the key space to about 10^8.
Commitment profiles saturate whereas rows do not; the release therefore
publishes profiles and leaves rows to program preprocessing.

**A single registry file.** Every verifier would hash the entire registry of
the release. Per-entry files allow it to hash only the entries it uses.

**Shipping the matrix by default.** At the largest measured shape the matrix
is 4 GiB, and deriving it accounts for about 3% of setup time.

**A Merkle tree over matrix pages.** A flat page-hash file provides the same
range checks without padding, sibling, or partial-leaf rules.

**Deriving commitments to longer prefixes from commitments to shorter ones.**
Inner Ajtai commitments are linear, but the outer layer applies a gadget
decomposition and a compression chain, which are not.

**Regeneration on a schedule only.** A manifest is opaque in review, and an
algorithm change does not modify it. The regeneration check is therefore
required at merge.

## Documentation

- `docs/security-posture.md`: describe the trust anchor model, and replace
  "Trusted in-place setup views" with the loader rule: in-place views are used
  only on bytes vouched for by the loader or by the guest's approved image. If
  #223 is merged first, this edits its Verifier Preprocessing section.
- `book/src/how/verification.md`: what the verifier trusts, anchors, the
  loader, audit mode, and protocol Stage 3 binding.
- `book/src/usage/verifier-only.md` and `book/src/usage/jolt-recursion.md`:
  loading a release, custom anchors, and the guest's embedded bytes.
- `specs/external-schedule-catalog-ownership.md`: permit a recursion guest to
  embed a catalog view checked against an anchored release, which that
  specification currently forbids, and describe program rows, which are
  planned during Jolt's program preprocessing, are outside any published
  catalog, and are admitted against an anchored set's commitment profile
  universe.
- Doc comments on the unchecked constructors refer to the loader.

## Execution

### Measurements

Measurements used the Akita revision on which Jolt main depends, which is
identical to Akita main on every setup path, on a single multi-core x86-64
machine, with a release build, `parallel` enabled, and disk persistence
disabled. The recipe is Jolt's one-hot setup (K=16, chunk profile w1r0, 57
polynomials). `max_num_vars` equals `log2(trace length) + 4`, so `nv = 26`
corresponds to a 2^22-cycle program; the recipe's matrix stops growing at
`nv = 35`. Each configuration was run once. Timings are reported as ratios,
because absolute times depend on the machine; sizes are properties of the
data.

| nv | Matrix | Slots | Slot export, share of setup time | Setup time relative to `nv = 26` |
|---|---|---|---|---|
| 24 | 32 MiB | 0 | none | about 0.4x |
| 26 | 64 MiB | 3 | about 55% to 70% | 1x |
| 28 | 256 MiB | 9 | about 78% to 85% | about 5x |
| 30 | 512 MiB | 17 | about 88% to 91% | about 20x |
| 32 | 1 GiB | 25 | about 92% to 94% | about 45x |
| 40 | 4 GiB | 41 | about 90% to 92% | about 150x |

| nv | Prover registry with hints | Verifier registry (commitments only) |
|---|---|---|
| 26 | 1.5 MiB | 1.5 KB |
| 28 | 7.3 MiB | 4.6 KB |
| 32 | 38 MiB | 12.7 KB |
| 40 | 93 MiB | 20.8 KB |

Further results:

- **Matrix derivation** accounts for 3% to 10% of setup time from `nv = 28`
  upward, and about 3% at `nv = 40`. Using all cores, it is about 6x faster
  than on a single core.
- **Setup parallelizes poorly.** Using all cores instead of one accelerates
  setup by only about 3.5x, because `export_setup_prefixes` processes slots
  sequentially. Within one slot, increasing from 1 to 8 threads yields about
  3.4x, and from 8 to 32 threads a further 1.15x.
- **Jolt's first-use rebuild** (`setup_prover`, `setup_verifier`, and
  `verifier`, excluding Jolt's own setup overhead) costs about 7 to 55 times
  the verification time for a 2^22-cycle program, and about one full setup at
  `nv = 40`.
- **Jolt's serialized verifier setup** is 4.2 MiB at every size, almost
  entirely the one-hot catalog. It contains no matrix.
- **Hashing compared with deriving the 4 GiB matrix:** parallel BLAKE3 is
  about 12x faster than parallel derivation, and single-threaded BLAKE3 about
  2.5x faster; SHA-256 is about 1.4x slower. The existing
  `validate_public_matrix_matches_seed` loop is sequential and runs at the
  speed of single-threaded derivation, about 6x slower than parallel
  derivation. Blake2b was not measured.
- **Dense and one-hot setups** use the same seed, and the dense matrix is a
  byte-identical prefix of the one-hot matrix. A setup with both flavors
  derives the shared prefix twice, which adds about one third to matrix
  derivation at `nv = 40`.
- **Not measured:** guest cycle counts.

**Peak memory of setup** (one-hot recipe as above; the thread count changes it
by less than 5%):

| nv | Matrix | Peak memory relative to the matrix |
|---|---|---|
| 28 | 256 MiB | about 8x |
| 32 | 1 GiB | about 14x |
| 35 | 4 GiB | about 10x |
| 40 | 4 GiB | about 11x |

Slot export proceeds through one temporary `CpuBackend`, whose
`setup_prefix_cache` retains a copy of `S[0..n_prefix]` for every slot until
the backend is dropped. Peak memory is approximately the matrix plus the sum of
all slot sources, plus 10%. Exporting one entry at a time without retaining its
source is estimated, but not measured, to reduce the peak at `nv = 40` to about
2 to 2.5 times the matrix.

**Grouped rows** (K=16, w1r0). Random keys were planned with a
reimplementation of Jolt's row-planning procedure (`plan_row`), with 10 to 24
rows planned concurrently; per-row times are therefore inflated. Each
concurrent full search held about half the size of the largest matrix in
memory. The reimplementation was not cross-checked against the output of
Jolt's `extend_catalog`.

| Grid | Key space | Sampled | Errors | Distinct rows | Full-search fallback |
|---|---|---|---|---|---|
| Advice | 129,360 | 6,000 | 0 | 6,000 | 5.7% of rows |
| Committed program | 57,171,240 | 871 | 0 | 871 | 35% of rows |

- **Fallback cost.** A row on the guided path is planned in milliseconds; a
  fallback row is four to five orders of magnitude slower.
- **Rows do not repeat.** Every sampled key yielded a distinct row. A catalog
  row occupies about 15.6 KB, against a trusted catalog limit of 16,384 rows
  and 64 MiB (about 4,300 rows of this size).
- **Commitment profiles saturate.** The base catalog uses 16 profiles. The
  advice sample reached 32 profiles by row 1,200 and 33 by row 4,500, and then
  remained constant; the committed-program sample reached 32 by row 600 and
  then remained constant. Across both samples there are about 36 distinct
  profiles, all with groups of at most 2^28 elements.
- **Full slot identifiers do not saturate** (661 after 6,000 advice rows), but
  identifiers that differ only in `natural_len` share one commitment, so they
  cost an index entry rather than a registry entry.
- **Registry size.** Hints occupy about 0.5 MB per entry at group `nv` 20 to
  22, 1 MB at 23 to 24, 2 MB at 25 to 26, 4 MB at 27, and 10.5 MB at 28: about
  70 MB for 36 entries, with about 18 KB of commitments.
- **Full-dense matrices** for field-increment arities 14 to 31 range from
  9.5 MB to 268 MB and have no setup-prefix slots.
- **Not sampled:** the field-increment grids, K=256, and the w8r2 profiles.

### Generation strategy

1. For each stream identity, derive the matrix once at the largest required
   capacity: the maximum of every parameter set's canonical matrix requirement
   and every covered profile's matrix footprint (its A, B, and compression
   matrices, which is distinct from its `n_prefix`). Derive pages in parallel
   and hash them in the same pass.
2. Collect the covered profiles: the union of every parameter set's required
   slots, with slot identifiers that differ only in `natural_len` mapped to a
   single entry.
3. Share only work that is exactly shared: one digit decomposition of the
   stream per `(ring dimension, basis)`, and one prepared A transform per
   `(ring dimension, width)`.
4. Generate entries concurrently, bounded by memory. Each entry reads
   `n_prefix` field elements borrowed from the derived matrix rather than
   copied, and releases them once its payload and hints are written. In the
   current code, entries are processed sequentially and every source remains
   resident until setup completes; this is the principal cost of generation in
   both time and memory.
5. Reuse inner commitment rows between two nested profiles only when the
   stream, A ring dimension, A rank, positions per block, digit basis and
   count, and block boundaries all match. Recompute the outer layer,
   compression, and hints per entry, and compare payloads and hints against
   independent generation before relying on the reuse. The frequency of this
   case SHOULD be measured before it is implemented.
6. Build guest caches per `(stream, row, target, capacity)`.
7. Hash release files, and write the manifest last.

### Steps

1. Manifest, commitment digest, set record, and anchor types in `akita-types`
   (including the manifest length limit); set-record check, commitment digest
   recomputation, and admission in `akita-config`; known-answer corpus,
   digest, and coverage tests; tampering tests.
2. Generator crate and binary, with determinism and poisoned-cache tests.
3. Verifier and prover loaders and their issued types; pinned entry points in
   `akita-pcs`; conversion of the raw paths into loader-private code or
   `unsafe fn`s; negative compile tests; replacement of the `akita-setup` disk
   cache.
4. Protocol Stage 3 commitment digest binding, with its version changes. This
   step depends only on Steps 1 and 3; delivering it before the first release
   avoids a code transition.
5. First published release, anchor crate, and the required regeneration
   check.
6. Jolt: Jolt generator; one-hot sets with commitment profile universes;
   bounded-dense and full-dense sets; program rows in `ProgramPreprocessing`;
   planner restricted to the universe; profile-coverage and
   planner-determinism tests; loader-based preprocessing; removal of the
   rebuild and serialization paths; opt-in local profile derivation.
7. Recursion guest: embedded release files and inner program preprocessing,
   in-guest loader, guest-only issuance for large files, and approval from one
   ELF snapshot.

### Open questions

1. **Commitment profile universe.** Whether the planner's candidate space can
   be enumerated directly, so that the universe is complete by construction
   rather than by sampling (the committed-program sample covered about 0.0015%
   of its keys); the cost in schedule quality of restricting the planner to the
   universe; the universes for K=256, w8r2, and the field-increment producers,
   which were not sampled; and whether bounded-dense rows require setup-prefix
   slots.
2. **Guest hashing.** Which embedded files the guest hashes at startup, which
   it receives through the guest-only path, and the cost of per-proof
   commitment digest recomputation, to be determined from guest cycle counts.
3. **Seed.** Proposed default: retain the fixed `sample_akita_setup_seed()`
   constant. A per-release seed provides little additional security and
   prevents releases from sharing a matrix.
4. **Transition window.** Proposed default: one software version.
5. **Regeneration runner.** One recipe is estimated to regenerate in about the
   time of one largest-shape setup, and a release spans several recipes.
   Memory requirements are about 11 times the largest matrix in the current
   code, and an estimated 2 to 2.5 times once entries no longer retain their
   sources. Any anchor change, including the addition of one release,
   regenerates every anchored release.
6. **Fallback planning cost.** Whether the planner's full-search fallback
   should be made less expensive before program preprocessing depends on it.

## References

- [`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md), section "Setup provenance and validation"
- [`external-schedule-catalog-ownership.md`](external-schedule-catalog-ownership.md)
- #223, with #233, #234, and #235: the verifier preprocessing trust model
- #210: recomputation of setup-prefix commitments when loading disk caches
- #186: trusted in-place setup and terminal-cache views
- #187: selected-row catalog views
- BCP 14: RFC 2119 and RFC 8174, requirement level key words
