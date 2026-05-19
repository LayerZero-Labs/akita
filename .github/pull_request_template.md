## Summary

<!-- What does this PR change and why? Link any spec or issue. -->

## Testing

- [ ] Ran tests for modified crates
- [ ] `cargo fmt --all --check` passes
- [ ] Relevant `cargo clippy` mode passes

## Security Considerations

<!-- Every Akita change can affect proof soundness, verifier correctness, dependency trust, or private witness handling. -->

- [ ] Verifier acceptance behavior is unchanged, or the intended change is specified.
- [ ] Verifier-reachable malformed inputs return typed errors instead of panicking.
- [ ] Transcript labels, challenge order, and domain separation are unchanged, or the intended change is specified.
- [ ] Serialization changes preserve canonical decoding and bounded untrusted input handling.
- [ ] New dependencies, Git dependencies, or CI actions are justified and pass supply-chain policy.
- [ ] Unsafe code is unchanged, or each new unsafe block has a local safety argument.

## Breaking Changes

<!-- List any breaking changes to public APIs, proof formats, setup formats, transcripts, or serialization. Write "None" if not applicable. -->

None
