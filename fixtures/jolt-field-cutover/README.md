# Jolt field cutover golden fixtures

These files were emitted by Akita's pre-cutover field serializers and
deterministic proof/setup constructors at commit
`03e3f2f96988c251f40608fd8735d011377c37b4`, the second parent of cutover
merge `7d8fca7554f2942cf7b4321a20f2051d6c0b108c`. At that commit the
production 128-bit field was `Prime128OffsetA7F7`.

## Fixtures that tests read

`crates/akita-serialization/src/field_impls.rs` reads these two files. Their
sources are intentionally independent of the post-cutover `jolt-field`
implementation:

- `fields.txt`: fixed-width scalar encodings for zero, one, the largest
  canonical representative, and deterministic random representatives of the
  production `Prime32Offset99`, `Prime64Offset59`, and 128-bit fields, plus
  `Ext2<Prime64Offset59>`, `FpExt4<Prime32Offset99>`, and
  `FpExt8<Prime32Offset99>` encodings. The random representatives were sampled
  in field-width order with rand 0.8.6's `StdRng::seed_from_u64(0x447a7f7)` and
  are recorded as literal bytes.
- `malformed.txt`: non-canonical scalar, `FpExt2`, and `FpExt8` inputs together
  with the pre-cutover results of unvalidated reduction. The tests require
  validated decoding to reject each input and unvalidated decoding to reproduce
  the pinned reduced bytes.

The 128-bit rows (`prime128_6a23801f_*`) were re-pinned for the
`Prime128Offset6A23801F` field switch (akita#326): the maximum canonical
representative and the non-canonical modulus encoding are the new field's, and
the random representatives remain canonical and unchanged.

## Historical fixtures

No test reads these files. They record pre-cutover outputs under
`Prime128OffsetA7F7` and the label schedule of that time, and they are not
evidence that current code rejects old proofs, setups, or transcripts:

- `proof.hex`: the uncompressed `AkitaBatchedProof` from
  `proof::tests::direct_terminal_relation_proof_serde_round_trip`.
- `setup.hex`: the compressed `AkitaVerifierSetup` from
  `proof::setup::tests::verifier_setup_prefix_slots_roundtrip`.
- `transcript.txt` and `transcript-keccak.txt`: the three challenges from
  `label_schedule::schedule_is_replayable_with_akita_labels` under the Blake2b
  and Keccak transcript backends.

## Changing the fixtures

Do not regenerate the files that tests read from the current implementation.
An intentional wire or transcript change must replace this fixture set with a
newly named protocol-epoch fixture and document the compatibility break.
