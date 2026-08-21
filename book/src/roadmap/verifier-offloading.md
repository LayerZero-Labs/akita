# Verifier offloading

> **Status:** recursive setup-prefix offloading is implemented. The broader
> construction in paper §4 is not claimed as a separate production feature.

Akita can replace a direct setup matrix contribution at a nonterminal fold with
an opening of a precommitted setup prefix and a Stage 3 product sumcheck. The
generated recursive schedule selects this mode. The caller cannot add or remove
it after schedule selection.

The current support envelope is limited to the generated fp128 one-hot and
fp128 one-hot W8R2 recursive catalogs. Setup offloading uses the supported
uniform D64 shape. Other configurations and setup ring dimensions do not expose
a recursive offloading catalog.

The prover emits `SetupSumcheckProof` for each selected producer. The proof
contains the setup claim, the setup-prefix evaluation, and a degree-two
sumcheck. The verifier resolves the same generated schedule, authenticates the
carried setup-prefix opening, and checks the Stage 3 proof. Missing or extra
Stage 3 payloads are rejected.

This implemented path covers recursive setup-prefix offloading, including the
generated distributed profile. Paper §4 describes a broader verifier
offloading construction. The repository does not present every part of that
paper construction as implemented.

**Sources to fold in**

- `specs/setup-offloading-planner.md`
- `specs/archive/2026-Q3/distributed-setup-offloading.md`
- `book/src/how/proving/sumcheck-stages.md`
- `book/src/how/verifying/setup_contribution.md`
- Paper §4 `sec:verifier-offloading` and §4.3 `sec:claim-reduction`
