# Zero knowledge

> **Status:** Akita does not currently provide zero knowledge and has no `zk`
> Cargo feature.

Zero knowledge means that a verifier learns no private witness information
beyond what follows from the public statement and whether the proof is valid.
The current Akita proof does not make that promise. It proves that an opening is
consistent with a commitment, but its proof messages may reveal information
about the committed values.

Applications must therefore treat the current PCS as transparent. They must not
place a private witness in an Akita commitment and assume that an opening hides
it. A host proof system may provide privacy through a separate construction,
but that property does not come from the current Akita API.

## Why this requires an end to end design

Hiding only the initial commitment would not be enough. An Akita proof exposes
messages at several later points:

- each sumcheck sends round polynomials derived from the witness;
- each fold commits to values derived from the preceding witness; and
- terminal verification receives the final short witness in clear form.

A complete zero knowledge construction must hide every one of these messages
while preserving the same opening claim. It must also show that the masks used
at one level remain valid after folding and do not break the norm bounds behind
Module-SIS security.

## Historical implementation work

An earlier experiment implemented part of commitment rerandomization and part
of sumcheck masking. It did not hide the complete proof, so it was removed from
the production code. The historical source remains on the `zk-wip` branch and
the `zk-prefix-snapshot-2026-06` tag.

That experiment must not be restored as a partial `zk` mode. A feature named
zero knowledge must protect the complete proof or clearly be an internal test
of one component.

## Requirements for future work

There is no approved implementation plan at present. A new proposal must cover
the following boundaries before code is treated as a production feature:

1. It must define what is public and what remains private.
2. It must account for every transcript message from the first commitment to
   the terminal check.
3. It must preserve the range and norm bounds used by the binding proof.
4. It must specify how the prover samples masks and what happens when sampling
   rejects.
5. It must give the verifier one complete acceptance rule and reject malformed
   hiding data without panicking.
6. It must include tests that compare the hidden and transparent statements and
   show that no partial mode can be selected by mistake.

Host integration is a separate layer. If a zkVM combines Akita with its own
privacy protocol, that integration must document which component hides each
message and which public values cross the boundary.
