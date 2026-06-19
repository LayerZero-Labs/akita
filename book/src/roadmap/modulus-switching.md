# Modulus switching

> **Status:** stub. Part of the initial Akita Book scaffold.

The "switchfold": starting from a large prime (as Jolt requires) but moving the
terminal tail into a much smaller prime, where it is shorter. The basic relation
is an inner product of an extension-field public vector with a small-inf-norm
private witness over \\( q_{\mathrm{hi}} \\), reduced to a longer inner product
over \\( q_{\mathrm{lo}} \\) plus auxiliary decomposition/overflow witnessing.

Folds from paper §5. Key points to surface: the transport gadget and its
identity, the directional asymmetry, switching one axis (commitment vs claim) at
a time, fusing the two folds into the switchfold, and why digit decomposition
destroys the tensor structure (so the switch is best placed after the first one
or two folds, and cannot combine with verifier offloading).

**Sources to fold in**

- Paper §5 `sec:modulus-switching` (`def:semantic-switchfold`, `sec:transport-technique` / `thm:transport-soundness`, `sec:switchfold-reduction` / `thm:switchfold-soundness`, `fig:switchfold-protocol`).
- Open question (paper §6.6 `sec:zk-open`): interaction with zero-knowledge.
