## Summary

<!-- What does this PR change? Keep this description current with the latest commit. Link any spec or issue. -->

## Motivation

<!-- What problem or unmet need does this solve, and who or what benefits? Explain why the existing code or a smaller change is insufficient, and why the benefit justifies any added complexity, maintenance, performance, or security cost. Use concrete evidence where relevant; a short explanation is enough for a small fix. -->

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

<!-- Prefer compatible changes. For any break to public APIs, proof/setup formats, transcripts, serialization, or supported behavior, explain the concrete longer-term goal, why a compatible alternative is insufficient, the affected consumers, and the migration or coordinated cutover. Permission to break compatibility is not justification. Write "None" if not applicable. -->

None
