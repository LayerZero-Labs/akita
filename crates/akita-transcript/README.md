# akita-transcript

Native Spongefish proof-stream support for Akita.

## Active Hardening Pillars

Akita's transcript hardening has three active pieces:

1. `AkitaInstanceDescriptor` bytes are bound into the spongefish preamble through `DomainSeparator.instance(...)`.
2. Spongefish `ProverState` and `VerifierState` are the only production transcript implementations. The default backend is Blake2b; `--no-default-features --features transcript-keccak` selects Keccak instead. Exactly one backend must be selected.
3. Native proof receipt, public-message absorption, challenge extraction, and EOF checking operate on the same state and argument string.

The versioned instance descriptor binds a fixed positional grammar. Public
context records capture message kind, site, atom count, encoded width, and
challenge candidate width for diagnostics; they are not hashed in production.
Rust type names and diagnostic labels are not cryptographic domains.

## Logging Checks

With `logging-transcript`, the native context helpers record the exact
`ProtocolContextRecord` sequence executed around Spongefish operations. End-to-end tests require
the prover and verifier context streams to be non-empty and identical.

The PCS integration tests enable this with:

```bash
cargo test -p akita-pcs --features logging-transcript --test transcript_hardening
cargo test -p akita-pcs --features logging-transcript --test transcript_hardening_proptest
```

For the full design, see [`book/src/how/transcript.md`](../../book/src/how/transcript.md).
