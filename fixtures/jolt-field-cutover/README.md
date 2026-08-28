# Jolt field cutover golden fixtures

These files were emitted by the existing deterministic Akita tests at
pre-cutover commit `03e3f2f96988c251f40608fd8735d011377c37b4`, the second parent
of cutover merge `7d8fca7554f2942cf7b4321a20f2051d6c0b108c`.

The fixture sources are intentionally independent of the post-cutover
`jolt-field` implementation:

- `fields.txt`: compressed scalar encodings for `Prime32Offset99`,
  `Prime64Offset59`, and `Prime128Offset275`, followed by `FpExt4` over
  `Prime32Offset99` with coefficients `[1, 2, 3, 4]`;
- `proof.hex`: the uncompressed `AkitaBatchedProof` from
  `proof::tests::direct_terminal_relation_proof_serde_round_trip`;
- `setup.hex`: the compressed `AkitaVerifierSetup` from
  `proof::setup::tests::verifier_setup_prefix_slots_roundtrip`;
- `transcript.txt`: the three challenges from
  `label_schedule::schedule_is_replayable_with_akita_labels` under the
  Blake2b transcript backend.

Post-cutover tests consume these literal bytes and values. Do not regenerate
them from the current implementation. An intentional wire or transcript
change must replace this fixture set with a newly named protocol-epoch fixture
and document the compatibility break.
