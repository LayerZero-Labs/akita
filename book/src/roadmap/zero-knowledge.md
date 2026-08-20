# Zero-knowledge

> **Status:** paper design only. Akita currently has no zero-knowledge
> implementation or `zk` Cargo feature.

End-to-end zero-knowledge for Akita (Paper §6, `sec:zk`) closes three leakage
channels: sum-check round messages, level-transition commitments, and terminal
witness opening. The construction is a **prefix / seam / suffix** pipeline. Zero
knowledge is **sealed at the seam**; everything after is an ordinary non-ZK
opening of a masked response.

| Region | Paper | Role |
|--------|-------|------|
| **Prefix** | `sec:zk-commitments`, `sec:zk-sumcheck-mask` | Single-modulus masked recursion: `Com_pre` binds all sum-check pads; per-round pads + LHL blinding columns hide transcript-visible messages. |
| **Seam** | `sec:zk-joint-sigma` | Committed-response tail: rejection-sampled masked response `Z`, long **linear** bundle discharged by ordinary Akita, small **quadratic** interface proved by a native lattice quadratic proof (LNP22-style). |
| **Suffix** | `sec:zk-pipeline` (suffix paragraph) | Open the committed response with transparent Akita. |

**Implementation status.** The production code is transparent only. The former
prefix experiment implemented parts of commitment rerandomization and sumcheck
masking, but it did not implement the paper's committed-response seam. That
experiment was removed from the main codebase and is preserved on the
`zk-wip` branch and the `zk-prefix-snapshot-2026-06` tag.

There is no approved implementation plan for restoring zero knowledge. Any new
work must start from a fresh specification that covers the complete prefix,
seam, and suffix construction. It must not restore the historical prefix as a
partial production feature.

**Out of scope for this chapter.** Host zkVM / outer-PIOP integration (extra
auxiliary commitments, fused outer sumchecks) is not part of the standalone PCS
ZK construction; it belongs in host integration docs, not here.

**Sources to fold in**

- Paper §6 `sec:zk`, `sec:zk-pipeline`, `fig:zk-pipeline`, `sec:zk-joint-sigma`,
  `sec:zk-open`.
- [Foundations → Zero-knowledge background](../foundations/zero-knowledge.md)
  (leakage + masking background).
- Archived prefix specs preserved on `zk-wip` and
  `zk-prefix-snapshot-2026-06`.
