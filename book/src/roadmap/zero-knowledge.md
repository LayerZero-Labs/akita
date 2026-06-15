# Zero-knowledge

> **Status:** stub. Part of the initial Akita Book scaffold.

End-to-end zero-knowledge for Akita is defined in the lattice-jolt paper
(§6, `sec:zk`): close three leakage channels (sum-check round messages,
level-transition commitments, terminal witness opening) via a **prefix / seam /
suffix** pipeline. Zero knowledge is **sealed at the seam**; everything after is
an ordinary non-ZK opening of a masked response.

| Region | Paper | Role |
|--------|-------|------|
| **Prefix** | `sec:zk-commitments`, `sec:zk-sumcheck-mask` | Single-modulus masked recursion: `Com_pre` binds all sum-check pads; per-round pads + LHL blinding columns hide transcript-visible messages. |
| **Seam** | `sec:zk-joint-sigma` | Committed-response tail: rejection-sampled masked response `Z`, long **linear** bundle discharged by ordinary Akita, small **quadratic** interface proved by a native lattice quadratic proof (LNP22-style). |
| **Suffix** | `sec:zk-pipeline` (suffix paragraph) | Open the committed response with transparent Akita; modulus switching (`sec:modulus-switching`) runs here only. |

**Implementation status (repo today).** The `zk` feature implements large parts of
the **prefix** (commitment rerandomization, sumcheck/`y_ring` masking, deferred
R1CS recording) and currently discharges tail rows by a **plain opening** of the
masking witness, not the paper's committed-response seam. Replacing that plain
opening with `sec:zk-joint-sigma` is the main remaining ZK step.

**Out of scope for this PCS chapter.** Jolt-specific outer-PIOP glue (e.g.
Spartan placement, `Com_aux1`) lives in the companion lattice-jolt paper, not in
the core Akita PCS ZK construction.

**Sources to fold in**

- Paper §6 `sec:zk` (`sections/akita/6_zero_knowledge.tex` in lattice-jolt);
  especially `sec:zk-pipeline`, `fig:zk-pipeline`, `sec:zk-joint-sigma`,
  `sec:zk-open`.
- [Foundations → Zero-knowledge background](../foundations/zero-knowledge.md)
  (leakage + masking background).
- `specs/akita-zk-commitment-hiding.md`, `specs/akita-zk-sumcheck-hiding-plain.md`,
  `specs/akita-zk-v-hiding.md` (what is implemented vs plain-opening gap).
