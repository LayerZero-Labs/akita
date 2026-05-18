# akita-transcript

Fiat-Shamir transcript support for Akita.

## Active Hardening Pillars

Akita's transcript hardening has three active pieces:

1. `AkitaInstanceDescriptor` bytes are bound into the spongefish preamble through `DomainSeparator.instance(...)`.
2. `AkitaTranscript` is backed by spongefish. The default backend is Blake2b; `transcript-keccak` selects Keccak instead.
3. `LoggingTranscript` is available behind `logging-transcript` for tests and schedule inspection.

Production labels are diagnostics only. `Label` is a zero-sized type when `logging-transcript` is disabled, and labels are never absorbed into the production sponge. Positional order plus the instance descriptor preamble are the protocol transcript domain.

## Logging Checks

`LoggingTranscript` records:

- descriptor preamble events;
- transcript absorbs;
- challenge squeezes;
- verifier wire-use events registered by tests or verifier harnesses.

Its smell checks assert:

- the first event is a non-empty descriptor preamble;
- absorbs are non-empty;
- labels are in `labels::ALL_LABELS`;
- each tracked verifier wire use is followed by a matching absorb before the next squeeze;
- declared wire-coverage manifest labels are actually recorded.

The PCS integration tests enable this with:

```bash
cargo test -p akita-pcs --features logging-transcript --test transcript_hardening
cargo test -p akita-pcs --features logging-transcript --test transcript_hardening_proptest
```

For the full design and deferred follow-ups, see `specs/transcript-hardening.md`.
