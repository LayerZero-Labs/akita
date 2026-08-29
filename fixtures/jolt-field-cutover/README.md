# Jolt field cutover golden fixtures

These files were emitted by Akita's pre-cutover field serializers and
deterministic proof/setup constructors at commit
`03e3f2f96988c251f40608fd8735d011377c37b4`, the second parent of cutover
merge `7d8fca7554f2942cf7b4321a20f2051d6c0b108c`. The proof and setup
constructors were instantiated with the production `Prime128OffsetA7F7` field.

The fixture sources are intentionally independent of the post-cutover
`jolt-field` implementation:

- `fields.txt`: fixed-width scalar encodings for zero, one, the largest
  canonical representative, and deterministic random representatives of the
  production `Prime32Offset99`, `Prime64Offset59`, and
  `Prime128OffsetA7F7` fields, plus `Ext2<Prime64Offset59>`,
  `FpExt4<Prime32Offset99>`, and `FpExt8<Prime32Offset99>` encodings. The
  random representatives were sampled in field-width order with rand 0.8.6's
  `StdRng::seed_from_u64(0x447a7f7)` and are recorded as literal bytes;
- `malformed.txt`: non-canonical scalar, `FpExt2`, and `FpExt8` inputs together
  with the pre-cutover results of unvalidated reduction. The post-cutover tests
  require validated decoding to reject each input and unvalidated decoding to
  reproduce the pinned reduced bytes;
- `proof.hex`: the uncompressed `AkitaBatchedProof` from
  `proof::tests::direct_terminal_relation_proof_serde_round_trip`, instantiated
  with `Prime128OffsetA7F7`;
- `setup.hex`: the compressed `AkitaVerifierSetup` from
  `proof::setup::tests::verifier_setup_prefix_slots_roundtrip`, instantiated
  with `Prime128OffsetA7F7`;
- `transcript.txt`: the three challenges from
  `label_schedule::schedule_is_replayable_with_akita_labels` under the
  Blake2b transcript backend.

Post-cutover tests consume these literal bytes and values. Do not regenerate
them from the current implementation. An intentional wire or transcript
change must replace this fixture set with a newly named protocol-epoch fixture
and document the compatibility break.
