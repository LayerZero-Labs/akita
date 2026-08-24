# ZK Sigma Failed First Attempt

This note records the failed `quang/zk-sigma-protocol` first attempt so the next implementation does not repeat it. It is intentionally self-contained because the exploratory worktree may be deleted.

## What Went Wrong

The first attempt implemented a generic field-vector Schnorr-style Sigma protocol:

- dense matrix commitment `A * witness`
- external mask sampler
- response `z = c * witness + mask`
- direct checks for generic linear and product-of-linear quadratic relations
- a simple centered `L_inf` response cutoff

That is a toy scaffold, not the Hachi/Jolt ZK protocol we need.

## What The Attempt Actually Built

The attempt added a standalone `protocol::zk_sigma` module with:

- `MatrixCommitmentKey`, a dense field-matrix commitment backend where `Commit(w) = A * w`
- `LinearExpression`, `LinearRelation`, and `QuadraticRelation` as generic field-vector relation types
- `ZkSigmaStatement`, containing the matrix key, public commitment, relation lists, and an optional response norm bound
- `ZkSigmaWitness`, just a vector of committed field coordinates
- `ZkSigmaProof`, containing attempt index, mask commitment, relation mask evaluations, and full response vector
- a `MaskSampler` trait, with tests using a scripted sampler
- prover logic that samples `mask`, computes `mask_commitment`, absorbs a first message, samples challenge `c`, and returns `z = c*w + mask` if the response bound passes
- verifier logic that checks `Commit(z) = c*Commit(w) + Commit(mask)`, linear equations, and product-of-linear quadratic equations
- custom serialization/validation impls and unit tests for accept/reject/abort/determinism/roundtrip behavior

The modularized version split that scaffold into `commitment`, `relation`, `statement`, `proof`, `prover`, `verifier`, `transcript`, `aborts`, and `serialization` modules. That made the toy implementation easier to read, but did not make it the right protocol.

## Why It Is A Stub

Do not continue this design as the real implementation. It misses the actual architecture:

- no Hachi Ajtai/ring commitment backend or LHL-hiding commitment path
- no `Com_pre` with pre-committed pads for all Jolt/Hachi/Spartan sumcheck rounds
- no masked sumcheck residual recording or batched tail discharge
- no `Com_aux1`, no level `L - 1` merge, and no fused Spartan placement
- no replacement for the leaking `PackedDigits` tail witness
- no D=64 tail promotion, sparse challenge family, or extra folding-level integration
- no Gaussian masking distribution or Nguyen/Lyubashevsky rejection sampling
- no LNP22 single-quadratic add-on with `g_quad`, `h`, `mu`, and `z_q`
- no `y_ring` coefficient masking or residual pinning
- no integration with Hachi proof objects, opening claims, transcripts, verifier logic, or tail proof layout

Modularizing the file only made the scaffold cleaner. It did not make it faithful.

## Intended Future Architecture

The next implementation should be a Hachi-native ZK path, not a standalone algebra demo. It should be built after crate decomposition so the prover, verifier, transcript, challenge, algebra, and proof-type boundaries are clean.

The intended shape is:

- one upfront Hachi commitment `Com_pre` that contains all witness material plus all pad slots needed to mask Jolt, Hachi, and Spartan sumcheck rounds
- masked sumcheck rounds that reveal only one-time-padded round polynomials, while recording the original verifier equations as deferred residuals
- a residual accumulator that batches masked-sumcheck identities, evaluation claims, ring-switch bindings, and other tail checks into the final sigma relation
- a mid-recursion auxiliary commitment `Com_aux1` for Jolt verifier/R1CS auxiliary variables, merged into Hachi's level `L - 1` mega-polynomial layout
- a fused level `L - 1` Spartan/Hachi sumcheck where Spartan's inner sumcheck shares the Hachi variable space and is combined by a verifier random linear combination
- a final fold that shrinks `Com_aux1`'s contribution before the tail
- a tail proof shape that replaces the leaking `PackedDigits` witness disclosure
- a D=64 Gaussian-masked sigma response using Hachi's ring/challenge machinery rather than dense field-matrix toy commitments
- rejection sampling with explicit Gaussian parameters and distributional tests, not a simple deterministic norm cutoff
- LNP22-style single-quadratic discharge for residual quadratic identities, with the required garbage ring element and scalar messages
- `y_ring` coefficient masking so ring-switch messages no longer reveal non-public coefficients
- verifier logic that checks the new proof object against Hachi commitments, folded openings, transcript labels, residual batches, and the tail sigma equations

## What To Do Next Time

Implementation should begin from real Hachi/Jolt protocol boundaries:

1. Inventory every current non-ZK transcript leak and assign it to committed pad masking, coefficient masking, residual batching, or tail sigma discharge.
2. Define the new tail proof object and verifier path that replaces `PackedDigits`.
3. Add the committed-pad layout to the Hachi witness/commitment schedule instead of creating a separate toy commitment.
4. Thread residual recording through the existing Jolt/Hachi/Spartan sumcheck boundaries.
5. Implement the tail sigma over Hachi ring commitments and sparse challenges.
6. Add Gaussian sampling and rejection sampling with parameter tests and simulator-oriented distribution tests.
7. Add the LNP22 quadratic add-on only for the actual residual quadratics that remain after the masking audit.

Until then, treat the existing standalone `zk_sigma` module as disposable scaffolding.
