# Hachi for Jolt (128-bit field arithmetization + lattice PCS commitments)

This document is a **Jolt-first** design spec for integrating a Hachi-like lattice PCS into Jolt’s architecture, under the hard requirement that Jolt’s sumchecks and transcript challenges live in a **128-bit prime field** \(F_{q'}\).

It incorporates:

- Jolt’s Stage 8 interface shape (open many polynomials at one point in \(F_{q'}\)),
- a recommended **field-to-ring embedding** choice (SuperNeo-style coefficient embedding) for one-hot/bit tables,
- a two-modulus “ring switching + modulus switching” opening reduction that stays verifier-native in \(F_{q'}\),
- concrete parameter crunching, including an estimate for the extra modulus quotient witness \(s\).

This is intentionally “implementation-ready”: it names concrete witness objects, indicates what is public vs committed vs proven, and highlights which parts require changing Jolt’s PCS trait boundary.

---

## 0. Executive summary (what we’re building)

Jolt’s prover does:

1. **Commit** to a collection of multilinear polynomials (witness tables).
2. Run application sumchecks (Stages 1–7) over a field \(F_{q'}\) and accumulate evaluation claims.
3. In **Stage 8**, prove a batch opening: many claims \(P_i(\mathbf r)=v_i\) at a **single point** \(\mathbf r\in F_{q'}^m\).

We want:

- commitments that are fast for one-hot/bit-heavy tables (so we prefer **small-modulus ring arithmetic** / NTT-friendly layouts), and
- an opening proof that can satisfy Jolt Stage 8’s interface (open at \(\mathbf r\in F_{q'}^m\)) while avoiding expensive ring multiplication.

The core idea is:

- Commit over a cyclotomic ring \(R_q = Z_q[X]/(X^d+1)\) with a **small** prime \(q\) (e.g. 32-bit).
- Verify openings using **ring switching evaluated in the large field** \(F_{q'}\) (128-bit) so all “point-derived” arithmetic is native to Jolt’s world.
- Add one extra quotient witness \(s\) so that “equalities mod \(q\)” become honest integer equalities when checked inside \(F_{q'}\).

---

## 1. Requirements and constraints

### 1.1 Jolt requirement: 128-bit prime field

Jolt’s arithmetization / sumchecks require that the field characteristic exceeds certain \(u64\cdot u64\) accumulation bounds, so we must work over a **128-bit prime**:

\[
F_{q'}\quad\text{with } q' \approx 2^{128}.
\]

This means we cannot “cheat” by using a small prime and moving to an extension \(F_{q^k}\), because the characteristic bound is on the base characteristic.

### 1.2 Jolt Stage 8 interface shape

Jolt’s accumulated opening claims are of the form:

\[
(\mathbf r,\ v)\quad\text{with }\mathbf r\in F_{q'}^m,\ v\in F_{q'}.
\]

See:

- `../jolt/jolt-core/src/poly/opening_proof.rs`: `pub type Opening<F> = (OpeningPoint<..., F>, F);`
- `../jolt/jolt-core/src/zkvm/prover.rs`: Stage 8 calls `PCS::prove(..., &opening_point.r, ...)`.

So any PCS that plugs into Stage 8 must accept opening points in \(F_{q'}\).

### 1.3 Witness regime: one-hot / bits

Many Jolt witness tables are one-hot or bit-heavy (see `docs/ONE_HOT_COMMITMENT_COST_AND_GPU_PRG.md`). We want “pay-per-bit” style cost where possible, so we strongly prefer embeddings that keep bits sparse in the ring representation.

---

## 2. Notation

- **Small prime**: \(q \approx 2^{32}\).
- **Large prime**: \(q' \approx 2^{128}\).
- **Cyclotomic degree**: \(d=2^\alpha\) (e.g. \(d=1024\)).
- **Ring**: \(R_q := Z_q[X]/(X^d+1)\).
- **Integer ring**: \(Z\) is the integers, and \(Z[X]\) is the integer polynomial ring.
- **Digit base**: \(b := 2^{\mathrm{LOG\_BASIS}}\) (e.g. \(b=16\)).
- **Digit length**: \(\delta := \lceil \log_b q\rceil\).
- **Redecomposition length**: \(\tau\) (Hachi §4.2).
- **Jolt opening point**: \(\mathbf r\in F_{q'}^m\) (Stage 8).
- **Hachi split**: \(\ell=m+r\) where \(m\approx r\) and the coefficient table has length \(2^\ell\).
- **Block count / block size**: there are \(2^r\) blocks, each of length \(2^m\).

---

## 3. Embedding choice for Jolt: SuperNeo coefficient embedding (recommended)

### 3.1 Why not use an extension-field embedding here?

Jolt’s constraint on the **field characteristic** forces the interactive proof algebra to live in a large *prime* field \(F_{q'}\).
So we do not get to pick a small prime \(q\) and move to an extension \(F_{q^k}\); that would change the characteristic.

Therefore, any embedding machinery whose purpose is “handle points/values in \(F_{q^k}\)” is simply orthogonal to the Jolt setting.

### 3.2 SuperNeo coefficient embedding preserves bit sparsity

SuperNeo’s coefficient embedding (Def. 7 in SuperNeo) maps a length-\(d\) field block into one ring element by placing entries directly in the coefficient slots. In our power-of-two cyclotomic:

\[
v=(v_0,\dots,v_{d-1})\in F_q^d
\ \mapsto\
\mathbf v(X)=\sum_{i=0}^{d-1} v_i X^i\in R_q.
\]

For one-hot/bit tables, coefficients remain \(\{0,1\}\), which enables:

- sparse gadget decomposition (often effectively \(\delta=1\) for bit witnesses), and
- “rot+add” multiplication by sparse polynomials if needed.

See `docs/FIELD_EMBEDDINGS_SUPERNEO_VS_HACHI.md` for the math-first comparison and the explicit \(X^d+1\) inner-product transform.

---

## 4. Two-field opening protocol (Option A): commit over \(R_q\), open at \(\mathbf r\in F_{q'}^m\)

**Compatibility warning (important):** for the genuine two-field setting with coefficient packing \(d>1\) (e.g. \(d=1024\)), the clean correctness-first spec lives in `docs/TWO_FIELD_OPENING_D_GT_1_CLEAN.md`.

The remainder of this Section 4 is an exploratory draft that mixes the original Hachi Step-B template (which assumes the evaluation point lives in the same algebraic domain as the ring arithmetic) with the Jolt requirement \(\mathbf r\in F_{q'}^n\). Use it for intuition only; do not treat it as a coherent d>1 protocol without reconciling the typing issues called out in that clean spec.

### 4.1 High-level picture

We implement a PCS with:

- **Commitment-time arithmetic** over \(R_q\) (small \(q\)),
- **Verifier-side checking** (sumcheck challenges, equality polynomials, batching weights) over \(F_{q'}\),
- **Ring switching** to move ring equations to field equations at a random \(\alpha\in F_{q'}\),
- plus **modulus switching** slack \(q\cdot s\) so “mod-\(q\)” equalities make sense as integer equalities when checked in \(F_{q'}\).

### 4.2 One opening claim

Input statement (Stage 8 style):

\[
\text{Given commitment } C \text{ to } P,\ \mathbf r\in F_{q'}^m,\ v\in F_{q'},\ \text{prove } P(\mathbf r)=v.
\]

Here \(P(\mathbf r)\) means: interpret the committed coefficients as integers (bitness makes this canonical), and evaluate the multilinear extension in \(F_{q'}\).

### 4.3 Full “one opening proof step” (commit structure + full relation + witness)

This section is the **self-contained core**: it writes the full relation/witness that the opening proof reduces to.

#### 4.3.1 Commitment objects (Hachi §4.1 core, ring-side)

Fix public commitment matrices (Ajtai-style) over \(R_q\):

\[
A\in R_q^{n_A\times (\delta\cdot 2^m)},\quad
B\in R_q^{n_B\times (n_A\delta\cdot 2^r)},\quad
D\in R_q^{n_D\times (\delta\cdot 2^r)}.
\]

Let \(f\in R_q^{\le 1}[X_1,\dots,X_\ell]\) be the committed multilinear, with coefficient table
\((f_{i\parallel j})_{i\in\{0,1\}^r,\ j\in\{0,1\}^m}\subset R_q\).
Define the block slices:

\[
f_i^\top := (f_{i\parallel j})_{j\in\{0,1\}^m}\in R_q^{2^m}.
\]

Gadget decomposition (base \(b\)) gives:

\[
s_i := G^{-1}_{2^m}(f_i)\in R_q^{2^m\delta}.
\]

Inner and outer commitments:

\[
t_i := A s_i \in R_q^{n_A},\quad
\hat t_i := G^{-1}_{n_A}(t_i)\in R_q^{n_A\delta},
\]

\[
u := B[\hat t_1;\dots;\hat t_{2^r}] \in R_q^{n_B}.
\]

This \(u\) is the PCS commitment.

#### 4.3.2 The opening-derived monomial vectors (Jolt field-side)

Jolt Stage 8 provides a single opening point \(\mathbf r\in F_{q'}^{n}\) (after claim reductions).
In the special case \(d=1\) (no packing) we have \(n=\ell=m+r\). If \(d>1\), then typically \(n=\ell+\log_2(d)\); see Section 4.3.2A.
Define the usual multilinear monomial vectors in the **field** \(F_{q'}\):

\[
b^\top := (r_1^{i_1}\cdots r_r^{i_r})_{i\in\{0,1\}^r}\in (F_{q'})^{2^r},
\qquad
a^\top := (r_{r+1}^{j_1}\cdots r_\ell^{j_m})_{j\in\{0,1\}^m}\in (F_{q'})^{2^m}.
\]

These are the same objects as Hachi Eq. (12), but they are computed in \(F_{q'}\) (not in \(R_q\)).

#### 4.3.2A Critical scope caveat: this Step-B template assumes *no packing* (\(d=1\)), or an additional “inner evaluation” reduction

The rest of Section 4.3 copies Hachi’s Step-B structure (aux commitment to partial evaluations, then folding, then a stacked linear system)
and was written with the “ring-level variable count” \(\ell=m+r\) in mind.

If we are in the packed regime \(d>1\) (e.g. \(d=1024\)), then Jolt’s scalar polynomial has:

- \(N = d\cdot 2^\ell\) scalar coefficients, hence
- \(n := \log_2 N = \ell + \alpha\) variables, where \(\alpha := \log_2 d\).

So the *opening point* naturally has \(n=\ell+\alpha\) coordinates \(\mathbf r\in F_{q'}^{\ell+\alpha}\), and the last \(\alpha\) coordinates index
the “inner coefficient slots” inside each packed ring element.

As written, however, the monomial vectors \(a,b\) are defined using only \(\ell\) coordinates (the “outer” ones), and the protocol implicitly treats
each ring element’s \(d\) coefficients as if they were “already evaluated” or irrelevant. This is **not correct** for the Jolt setting when \(d>1\):

- Multilinear evaluation over the inner \(\alpha\) coordinates uses **multilinear monomials** \(\prod_t r_{\ell+t}^{k_t}\).
- Ring switching at \(X=\alpha\) produces **power monomials** \(\alpha^k\) from the coefficient basis \(X^k\).

These are algebraically different in general, so ring switching does not “automatically” account for the inner \(\alpha\) coordinates.

Separately, the “aux commitment” step below defines \(w_i := a^\top G_{2^m}s_i\in R_q\) while \(a\in F_{q'}^{2^m}\) and \(G_{2^m}s_i\in R_q^{2^m}\).
That dot product is not an \(R_q\)-operation unless the weights \(a\) live in \(R_q\) (as in Hachi’s original extension-field-in-ring setting).
In the genuine two-field setting (\(F_{q'}\) independent of \(R_q\)), \(w_i\) would live in \(F_{q'}[X]/(X^d+1)\), not in \(R_q\), and the
subsequent digit decomposition and Ajtai commitment \(v := D\hat w\) over \(R_q\) are not well-defined.

**Conclusion**: the Step-B template in Sections 4.3.3–4.3.6 is internally consistent in the special case \(d=1\) (no packing),
or if we first add a separate reduction that turns an \(n=\ell+\alpha\)-variate scalar opening into an \(\ell\)-variate ring-level statement
with ring-native weights (analogous in spirit to Hachi’s \(\psi/\mathrm{Tr}\) machinery, but not provided by ring switching alone).

We keep the Step-B material below because it is still the right template for the **unstructured linear relation + ring/modulus switching + sumcheck**
pipeline once the statement is made well-typed. But for \(d>1\) in Jolt, an additional “inner evaluation” mechanism is required.

#### 4.3.2B The fix for \(d>1\): inner evaluation reduction via sumcheck + batching into ring switching

##### 4.3.2B.1 Why the algebraic "inner product to linear functional" bridge is unavailable

In the original Hachi (extension-field setting), the psi/trace identity (Paper Theorem 2)
\(\mathrm{Tr}_H(\psi(\mathbf a)\cdot\sigma_{-1}(\psi(\mathbf b))) = (d/k)\langle \mathbf a,\mathbf b\rangle\)
algebraically converts the multilinear inner product (which IS the evaluation) into a ring operation
that can be committed to and ring-switched. This works because \(F_{q^k}\) embeds into \(R_q\) as
the fixed subring \(R_q^H\), giving the algebraic bridge.

In the two-field setting, \(F_{q'}\) is an independent prime field with **no embedding into \(R_q\)**.
There is no psi map, no trace identity, and no algebraic way to convert "inner product of an
\(F_{q'}\)-vector with a coefficient vector" into a ring operation over \(R_q\).

**What we use instead**: an \(\alpha\)-round inner sumcheck (Phase 0) that reduces the multilinear
inner-coordinate evaluation to a random-point claim, followed by a batching trick that converts
random-point claims into polynomial-evaluation claims (i.e. ring switching). This is the
**computational substitute** for the missing algebraic identity. It costs \(\alpha\) extra sumcheck
rounds and \(d\) extra field elements in the proof.

##### 4.3.2B.2 Setup and notation

Let \(\alpha := \log_2 d\) (e.g. \(\alpha=10\) for \(d=1024\)), \(n := \ell + \alpha\) (total scalar variable count),
opening point \(\mathbf r = (\mathbf r_{\mathrm{out}},\mathbf r_{\mathrm{in}}) \in F_{q'}^{\ell} \times F_{q'}^{\alpha}\),
scalar coefficients \(p_{I,k}\in\{0,1\}\) for \(I\in\{0,1\}^\ell,\; k\in\{0,1\}^\alpha\),
and ring element \(f_I(X) := \sum_{k=0}^{d-1} p_{I,k}\,X^k \in R_q\) (coefficient embedding).

The correct scalar evaluation claim is:

\[
\boxed{
v = \sum_{I\in\{0,1\}^\ell}\;\sum_{k\in\{0,1\}^\alpha}
\iota(p_{I,k})\cdot \mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot \mathrm{eq}(\mathbf r_{\mathrm{in}},k)
\;\in F_{q'}.
}
\]

This is well-typed (pure \(F_{q'}\)), linear in the \(p_{I,k}\), and accounts for all \(n\) coordinates.

##### 4.3.2B.3 Phase 0: inner evaluation sumcheck (\(\alpha\) rounds)

Reorganize the evaluation as
\(v = \sum_{I} \mathrm{eq}(\mathbf r_{\mathrm{out}},I) \cdot f'_I\)
where \(f'_I := \sum_{k} \iota(p_{I,k})\cdot \mathrm{eq}(\mathbf r_{\mathrm{in}},k)\) is the "inner multilinear evaluation"
of ring element \(f_I\)'s coefficient vector at \(\mathbf r_{\mathrm{in}}\).

Run an \(\alpha\)-round sumcheck on the inner sum. Define
\(c_k := \sum_I \mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot\iota(p_{I,k})\) (the "coefficient-plane partial evaluations").
The sumcheck proves \(\sum_k c_k\cdot\mathrm{eq}(\mathbf r_{\mathrm{in}},k) = v\).
After \(\alpha\) rounds the verifier obtains a random point \(\mathbf t^*\in F_{q'}^\alpha\) and needs the oracle value
\(g^* := \sum_k c_k\cdot\mathrm{eq}(\mathbf t^*,k)\).

##### 4.3.2B.4 Prover sends coefficient-plane values; verifier checks and batches

The prover sends the \(d\) values \(c_k\) for \(k\in\{0,1\}^\alpha\).
The verifier checks \(\sum_k c_k\cdot\mathrm{eq}(\mathbf t^*,k) \stackrel{?}{=} g^*\) (cost: \(O(d)\) field ops).

Now **batch** the \(d\) claims \(c_k = \sum_I \mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot\iota(p_{I,k})\) into one.
The verifier samples \(\gamma\leftarrow F_{q'}\) and computes:

\[
V_\gamma := \sum_{k=0}^{d-1} \gamma^k \cdot c_k
= \sum_{I\in\{0,1\}^\ell} \mathrm{eq}(\mathbf r_{\mathrm{out}},I)
\cdot \underbrace{\sum_k \iota(p_{I,k})\gamma^k}_{=\;\mathrm{ev}_\gamma(f_I)}.
\]

The right-hand side is: **"evaluate the ring-switched table
\((\mathrm{ev}_\gamma(f_I))_I\) as a multilinear at \(\mathbf r_{\mathrm{out}}\)."**
This is an \(\ell\)-variable scalar evaluation where each table entry is obtained by ring switching at \(\gamma\).

##### 4.3.2B.5 Phase 1: modified Hachi Step B (ring-switched at \(\gamma\))

The claim is now \(V_\gamma = \sum_I \mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot \mathrm{ev}_\gamma(f_I)\).
Using the split \(\ell=m+r\) and \(I=(i,j)\):

\[
V_\gamma = \sum_i b_i \cdot \Big[\sum_j a_j \cdot \mathrm{ev}_\gamma(f_{i\|j})\Big],
\]

where \(b_i,a_j\in F_{q'}\) are the outer monomial weights. All products are well-typed: \(\mathrm{ev}_\gamma\)
maps \(R_q\)-elements to \(F_{q'}\), and \(a_j\in F_{q'}\).

**Key issue**: the prover cannot precompute \(w'_i := \sum_j a_j\cdot \mathrm{ev}_\gamma(f_{i\|j})\) before \(\gamma\) is known.
Although \(\sum_j a_j\cdot f_{i\|j}(X)\) is well-defined over \(F_{q'}[X]\), it lives in \(F_{q'}[X]/(X^d+1)\), not \(R_q\),
so it cannot be committed using \(R_q\)-Ajtai matrices.

**Resolution**: we do **not** need an auxiliary commitment to partial evaluations in the \(d>1\) regime.
Instead, the evaluation equation enters the final sumcheck as an **additional linear constraint**
on the witness table, alongside the ring/modulus-switched commitment constraints and range constraints.

##### 4.3.2B.6 Modified constraint set for Phase 1

1. **Main commitment consistency**: \(B\hat t = u\). (Pure ring; ring+mod-switch at \(\gamma\).)
2. **Fold consistency (inner commit)**: \(Az = (c^\top\otimes G_{n_A})\hat t\). (Pure ring; ring+mod-switch at \(\gamma\).)
3. **Evaluation equation** (scalar, at \(\gamma\)):
   \(V_\gamma = \sum_I \mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot\sum_k \iota(p_{I,k})\gamma^k\).
   Linear in committed digits with public weights \(\mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot\gamma^k\).
4. **Range / bitness constraints**: all digit entries bounded.

**Removed**: constraint (3) of old template (\(b^\top w = \mathrm{open\_value}\)),
constraint (4) (\(a^\top Gz = \ldots\)), the auxiliary commitment \(v=D\hat w\), and the \(\hat w\) witness.

##### 4.3.2B.7 How the evaluation equation enters the sumcheck

The witness table \(\tilde w\) encodes digit-level entries. The scalar coefficient is recovered as
\(p_{I,k} = \sum_\delta b^\delta\cdot\tilde w(I,\delta,k)\). The evaluation constraint becomes:

\[
V_\gamma = \sum_{I,k} \mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot\gamma^k\cdot\sum_\delta b^\delta\cdot\tilde w(I,\delta,k).
\]

The public weights \(\mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot\gamma^k\cdot b^\delta\) **factorize**:
\(\mathrm{eq}(\mathbf r_{\mathrm{out}},I)\) has a closed-form MLE (Section 4.5.3),
\(\gamma^k\) is a size-\(d\) power table, and \(b^\delta\) is a geometric sequence.
The verifier evaluates public encodings at any sumcheck point in \(O(\ell+d+\delta)\) field ops.
This is batched into the existing sumcheck, adding **no extra rounds**.

##### 4.3.2B.8 Cost summary

| Component | Cost |
|---|---|
| Phase 0 inner sumcheck | \(\alpha\) extra rounds (10 for \(d=1024\)) |
| Coefficient-plane values \(c_k\) | \(d\) field elements in proof (16 KiB for \(d=1024\)) |
| Verifier batch check | \(O(d)\) field multiplications |
| Phase 1 eval constraint | no extra rounds (folded into existing sumcheck) |

##### 4.3.2B.9 Alternative: lift to integers and work over \(R_q\)

One could lift the \(F_{q'}\) evaluation claim to the integers and prove using \(F_q/R_q\) arithmetic
(the "Option B" approach from Section F.0.6 of `docs/HACHI_DIGEST.md`). Caveats:
(1) every \(F_{q'}\) multiplication becomes limb-by-limb schoolbook over \(Z_q\) at cost \(O(L^2)\) with \(L\approx 4\) limbs;
(2) the \(\mathrm{eq}(\tilde r,I)\) weights involve products of up to \(n\) 128-bit numbers, creating deep non-native chains;
(3) the quotient witness \(\kappa\) for mod-\(q'\) can be enormous;
(4) the expensive part is non-native simulation, which doesn't benefit from NTT-friendly ring structure.
The cross-prime approach (Phase 0 + modified Phase 1) is typically much cheaper.

#### 4.3.3 Intermediate partial evaluations and auxiliary commitment (ring-side objects)

Define the ring partial evaluations:

\[
w_i := a^\top\cdot G_{2^m}\, s_i \in R_q\quad\text{for each }i\in\{0,1\}^r,
\]

and digitize:

\[
\hat w_i := G^{-1}_1(w_i)\in R_q^\delta.
\]

Stack \(\hat w := (\hat w_i)_{i\in\{0,1\}^r}\in R_q^{\delta\cdot 2^r}\) and define:

\[
v := D\hat w \in R_q^{n_D}.
\]

The prover sends \(v\) as the first opening message (as in Hachi Fig. 3).

#### 4.3.4 Fold challenge and folded witness (ring-side)

The verifier derives a short/sparse ring challenge vector:

\[
c=(c_1,\dots,c_{2^r})\in R_q^{2^r}.
\]

The prover folds:

\[
z := \sum_{i=1}^{2^r} c_i s_i \in R_q^{2^m\delta}.
\]

Because coefficients of \(z\) grow, the prover performs a redecomposition using a second gadget \(J\):

\[
\hat z := J^{-1}_{2^m}(z)\in R_q^{2^m\delta\tau}.
\]

#### 4.3.5 The full Step-B relation (six constraints, verbatim structure)

The prover’s remaining task is to prove existence of **small** objects \((\hat w,\hat t,\hat z)\) such that all of the following hold (this is the B.4.1 relation, rewritten):

1. **Aux-commitment consistency**:
   \[
   D\hat w = v.
   \]

2. **Main commitment consistency**:
   \[
   B\hat t = u.
   \]

3. **Evaluation equation** (in field form):
   - let \(w := G_{2^r}\hat w\) (recompose digits to ring elements), then the claimed opening value is:
   \[
   b^\top\cdot w = \mathrm{open\_value}\quad(\text{interpreted in }F_{q'}\text{ after ring switching}).
   \]

4. **Fold consistency (partials)**:
   - let \(z := J\hat z\), then
   \[
   a^\top\cdot G_{2^m}\, z = (c^\top\otimes G_1)\hat w.
   \]

5. **Fold consistency (inner commit)**:
   \[
   A z = (c^\top\otimes G_{n_A})\hat t.
   \]

6. **Smallness / range constraints**:
   - coefficients of \(\hat w,\hat t,\hat z\) lie in the intended digit ranges (and for bit/one-hot regimes, coefficient constraints can often be strengthened to \(\{0,1\}\) on specific parts of the witness).

At this point we have reduced the opening claim to a structured “prove knowledge of bounded digits satisfying a public linear system” instance.

#### 4.3.6 The same Step-B relation as one explicit stacked matrix equation (Hachi Eq. (20))

It is easy to lose track of how the opening-point-derived weights \((a,b)\) actually enter the *public* linear system.
The paper makes this explicit by stacking constraints (1)–(5) into a single matrix equation (their Eq. (20)).

First recall the evaluation rewrite (paper Eq. (15)–(19) style), specialized to our split \(\ell=m+r\):

- Define partial evaluations (ring-side):

  \[
  w_i := a^\top G_{2^m} s_i \in R_q,\qquad i\in\{0,1\}^r.
  \]

- Stack \(w := (w_i)_i \in R_q^{2^r}\) and digitize/recompose via the gadget map:

  \[
  \hat w := G^{-1}_{2^r}(w),\qquad w = G_{2^r}\hat w.
  \]

- The claimed opening value is the bilinear form:

  \[
  b^\top w = \mathrm{open\_value}.
  \]

- The fold constraint (partials) is:

  \[
  a^\top G_{2^m} z = (c^\top\otimes G_1)\hat w.
  \]

- The fold constraint (inner commit) is:

  \[
  A z = (c^\top\otimes G_{n_A})\hat t.
  \]

Now, writing \(z = J_{2^m}\hat z\) (so \(\hat z := J^{-1}_{2^m}(z)\)), constraints (1)–(5) can be written as the stacked linear system:

\[
\begin{bmatrix}
D & 0 & 0 \\
0 & B & 0 \\
b^\top G_{2^r} & 0 & 0 \\
(c^\top\otimes G_1) & 0 & -\,a^\top G_{2^m} J_{2^m} \\
0 & (c^\top\otimes G_{n_A}) & -\,A J_{2^m}
\end{bmatrix}
\cdot
\begin{bmatrix}
\hat w \\
\hat t \\
\hat z
\end{bmatrix}
\;=\;
\begin{bmatrix}
v \\
u \\
\mathrm{open\_value} \\
0 \\
0
\end{bmatrix}.
\]

**Key dependency note (why the verifier “needs \(a,b\)”)**:

- The *public matrix* on the left contains the rows \(b^\top G_{2^r}\) and \(-a^\top G_{2^m}J_{2^m}\).
  These are functions of the opening point \(\mathbf r\) via the monomial vectors \(a=a(\mathbf r)\), \(b=b(\mathbf r)\).
- Therefore, when we later write “package (1)–(5) as \(M\cdot Z = Y\)”, the packaged \(M\) **depends on** \((a,b)\),
  and the packaged \(Y\) contains \(\mathrm{open\_value}\) (in addition to \(u,v\)).

### 4.4 Ring switching + modulus switching (make the relation checkable in \(F_{q'}\))

All equalities above are ring equalities in \(R_q\) except the weights \(a,b\) which are in \(F_{q'}\).
We therefore check the relation by moving to \(F_{q'}\) via ring switching.

#### 4.4.1 Evaluation map into \(F_{q'}\)

Verifier samples \(\alpha\leftarrow F_{q'}\) and defines

\[
\mathrm{ev}_\alpha: R_q \to F_{q'}
\]

as: choose a lift convention \(\mathrm{lift}_q\) for coefficients of \(Z_q\), lift each coefficient to an integer, and then evaluate the polynomial at \(X=\alpha\) in \(F_{q'}\).

Apply \(\mathrm{ev}_\alpha\) entrywise to vectors/matrices over \(R_q\).

#### 4.4.2 The unstructured linear relation and its full witness

Package constraints (1)–(5) into a single public linear system over \(R_q\):

\[
M\cdot Z = Y \quad\text{in }R_q^n,
\]

where:

- \(Z\) is the stacked unknown vector whose coordinates are exactly the digits in \((\hat w,\hat t,\hat z)\) (plus any auxiliary digit blocks introduced by the gadget decompositions),
- \(Y\) is the stacked public right-hand side built from \((u,v)\), the claimed open value, and the fold challenge \(c\),
- \(M\) is the stacked public matrix built from \((A,B,D)\), the gadget matrices, and the fold wiring.

Now lift this ring equation to an **integer polynomial identity** with two quotient witnesses:

\[
\mathrm{lift}_q(M)\,\mathrm{lift}_q(Z) - \mathrm{lift}_q(Y)
=
(X^d+1)\,r \;+\; q\cdot s
\quad\text{over } Z[X],
\]

where:

- \(r\in (Z[X]_{<d})^n\) is the cyclotomic quotient witness,
- \(s\in (Z[X]_{<d})^n\) is the modulus quotient witness.

This is the “full witness” for the ring-switch+modulus-switch step: the prover’s hidden witness is

\[
W := \big(Z,\ (r\text{ digits}),\ (s\text{ digits})\big).
\]

Evaluating at \(X=\alpha\) gives a checkable field equation in \(F_{q'}\):

\[
\mathrm{lift}_q(M)(\alpha)\,\mathrm{lift}_q(Z)(\alpha)
=
\mathrm{lift}_q(Y)(\alpha) + (\alpha^d+1)\,r(\alpha) + q\,s(\alpha).
\]

### 4.5 Witness-table encoding + sumcheck (verifier-native in \(F_{q'}\))

To prove both:

- the ring-switch equality at \(\alpha\), and
- the range constraints on all digit witnesses,

we do exactly what Hachi §4.3 does: commit to a single multilinear “witness table” and run sumcheck.

#### 4.5.1 Witness table layout (self-contained)

Let:

- \(\mu := |Z|\) be the number of digit coordinates in the stacked witness vector \(Z\),
- \(n\) be the number of linear rows,
- \(\delta_r,\delta_s\) be the digit lengths used to represent coefficients of \(r\) and \(s\) in base \(b\).

Define a table with rows:

\[
\text{rows} = \mu \;+\; n\delta_r \;+\; n\delta_s,
\qquad
\text{cols} = d.
\]

Row semantics:

- rows \(1.. \mu\) encode coefficients of the digit witness \(Z\),
- next \(n\delta_r\) rows encode coefficients of the digit blocks of \(r\),
- next \(n\delta_s\) rows encode coefficients of the digit blocks of \(s\).

Flatten this \((\text{rows}\times \text{cols})\) table into one multilinear polynomial \(\tilde w\) over \(F_q\)-coefficients (with padding to a power of two), exactly as Hachi’s Eq. (21) “witness table embedding” does.

#### 4.5.2 Constraints proved by sumcheck (shape)

We prove two families of constraints, batched by equality polynomials and checked via sumcheck over \(F_{q'}\):

1) **Linear constraints at \(\alpha\)** (ring switching + modulus switching), which enforce the evaluated identity using public encodings of \(\mathrm{lift}_q(M)(\alpha)\) and the extra \(-(\alpha^d+1)\) and \(-q\) coefficients on the \((r,s)\) blocks.

2) **Range constraints** on every table entry:

- for generic base-\(b\) digits: use a root-check polynomial whose roots are the allowed digit set,
- for bit tables (one-hot regimes): use \(T(T-1)=0\) where appropriate.

Sumcheck reduces this to one final evaluation claim:

\[
\tilde w(\mathbf r^\*) = v^\*,
\quad \mathbf r^\*\in F_{q'}^{m'}.
\]

#### 4.5.3 What the verifier actually needs from the “public matrix” (and why we should not materialize \(a,b\))

It is tempting to think the verifier must explicitly build the monomial vectors
\(a\in F_{q'}^{2^m}\), \(b\in F_{q'}^{2^r}\) (hence “square-root work”) in order to “know \(M\)” in the packaged relation
\(M\cdot Z=Y\).

However, the **sumcheck verifier never needs the full matrix \(M\) (nor the full vectors \(a,b\))**.
What it needs, at the end of sumcheck, is the ability to evaluate a small set of **public multilinear encodings**
at the final random point \(\mathbf r^\*\) (and at intermediate round points as the transcript is checked). Concretely:

- The “linear-constraint” family only requires evaluating the public encodings of \(\mathrm{lift}_q(M)(\alpha)\) (including the extra
  \(-(\alpha^d+1)\) and \(-q\) coefficients on the quotient blocks) at the relevant sumcheck points.
- The “range-constraint” family only requires evaluating the public range-check polynomial (or bitness polynomial) at those same points.

The dependence on the **opening point** \(\mathbf r\) enters these public encodings only through the two monomial-weight tables
\(a=a(\mathbf r)\) and \(b=b(\mathbf r)\) (Section 4.3.2 / Eq. (20)). Critically, their multilinear extensions have a closed form,
so the verifier can evaluate them at arbitrary points in time **without enumerating** \(2^m\) or \(2^r\) entries.

##### Closed-form MLE evaluation for monomial weights

Let \(\mathbf r^{(b)}:=(r_1,\dots,r_r)\in F_{q'}^r\) and define the Boolean-indexed table
\(b_i := \prod_{j=1}^r (r_j)^{i_j}\) for \(i\in\{0,1\}^r\).
Let \(\tilde b : F_{q'}^r \to F_{q'}\) be its multilinear extension:

\[
\tilde b(t) := \sum_{i\in\{0,1\}^r} b_i \cdot \mathrm{eq}(t,i),
\quad
\mathrm{eq}(t,i):=\prod_{j=1}^r \big(t_j i_j + (1-t_j)(1-i_j)\big).
\]

Then \(\tilde b(t)\) factorizes as:

\[
\boxed{
\tilde b(t) = \prod_{j=1}^r \big((1-t_j) + t_j\cdot r_j\big).
}
\]

The same holds for \(\tilde a\) over the “inner” coordinates \(\mathbf r^{(a)}:=(r_{r+1},\dots,r_\ell)\in F_{q'}^m\):

\[
\boxed{
\tilde a(t) = \prod_{j=1}^m \big((1-t_j) + t_j\cdot r_{r+j}\big).
}
\]

Therefore, **any time the verifier needs “the monomial weights” inside the public MLEs**, it can compute the needed value using
\(O(r)\) (resp. \(O(m)\)) field multiplications and additions, rather than materializing an exponential-size vector.

Practical implication for implementation:

- Do **not** build arrays `a[0..2^m)` and `b[0..2^r)` unless you truly need them explicitly.
- Instead implement a helper like `eval_monomial_mle(r_slice, t)` that returns the boxed product above.

##### What “public encodings of \(\mathrm{lift}_q(M)(\alpha)\)” actually contain (structured vs unstructured)

For the linear-constraint family, the verifier conceptually needs access to a multilinear encoding of the *row-by-row coefficient function*
of the ring-switched matrix \(\mathrm{lift}_q(M)(\alpha)\). This is the Jolt-adapted analogue of Hachi’s paper definition
\(e_{M_\alpha}(i,u)\) (paper Eq. (21) discussion, right after defining \(e_\alpha(\ell)=\alpha^\ell\)), where \(i\) indexes a constraint row
and \(u\) indexes a witness coordinate block.

Because the protocol batches the row index \(i\) using an equality polynomial (paper Eq. (22) style), the verifier does **not**
need to query this encoding at many points. Concretely, after the verifier samples the batching point \(\tau_1\) (Fiat–Shamir),
it reduces the row dimension to a single *batched coefficient function*
\[
e_{M_\alpha,\tau_1}(u) := \sum_{i\in[n]} \mathrm{eq}(\tau_1,i)\cdot e_{M_\alpha}(i,u),
\]
and the subsequent sumcheck only ever needs evaluations of this batched function at **one** final random point in the \(u\)-variables
(plus the trivial ability to compute \(\sum_i \mathrm{eq}(\tau_1,i)\,y_i(\alpha)\) from public \(y_i(\alpha)\)).
So, per recursion iteration, the expensive “public encoding” workload is a **small constant number of MLE evaluations** of objects derived from \(M(\alpha)\).

In our setting, the packaged Step-B matrix \(M\) (Section 4.4.2) is exactly the stacked block matrix in Section 4.3.6:

- the **top-left blocks** are the Ajtai matrices \(D\), \(B\), and the fold wiring blocks \((c^\top\otimes G_1)\), \((c^\top\otimes G_{n_A})\),
- the **right blocks** include \(-a^\top G_{2^m}J_{2^m}\) and \(-AJ_{2^m}\),
- and the **opening equation row** contains \(b^\top G_{2^r}\).

After applying ring switching at \(X=\alpha\), each ring element coefficient becomes a field element in \(F_{q'}\).
So the verifier’s “public encoding” is, informally:

> a function that, given a (row index) and a (witness-coordinate index), returns the corresponding coefficient of \(M(\alpha)\) in \(F_{q'}\).

This coefficient function has two qualitatively different sources:

- **Structured (cheap to evaluate at a point)**:
  - anything derived from the opening point weights \(a,b\) (use the closed-form MLE evaluations above),
  - gadget matrices \(G_{\cdot}\), \(J_{2^m}\) (fixed, highly structured),
  - the sparse fold challenge \(c\) (explicit/sparse),
  - the quotient-block coefficients \(-(\alpha^d+1)\) and \(-q\) (single scalars),
  - the power table \(e_\alpha(\ell)=\alpha^\ell\) (size \(d\), cheap).

- **Unstructured (the real remaining “square-root” bottleneck)**:
  - the Ajtai matrices \(A\), \(B\), \(D\) (and any other “uniformly random over \(R_q\)” matrices).

Even though these matrices have **constant height** (\(n_A,n_B,n_D=O(1)\)), their *widths* are exponential in the split parameters:

\[
\mathrm{width}(A)=\delta\cdot 2^m,\qquad
\mathrm{width}(B)=n_A\delta\cdot 2^r,\qquad
\mathrm{width}(D)=\delta\cdot 2^r.
\]

So, when the verifier needs an evaluation of the multilinear extension of (say) a row of \(A(\alpha)\) at some transcript-derived point,
it is essentially asking for an evaluation of a length-\(\delta 2^m\) *pseudorandom table* at a random point. Without additional structure,
computing that value costs \(\Theta(\delta 2^m)\) field operations (generate the PRG outputs and take the eq-weighted sum).

This is the key “next assist-proof target”:

> **Assist-proof goal (candidate)**: the prover supplies the needed evaluations of the unstructured public encodings
> (those derived from \(A,B,D\)) at the few sumcheck points, together with a proof that these values are consistent with the public PRG seeds
> that define \(A,B,D\) (or with an explicit commitment to those matrices, if we choose that route).

##### A “seeding granularity” knob for \(A,B,D\) (public input size vs assist-proof work)

In practice we almost never want to publish the full Ajtai matrices \(A,B,D\) explicitly; we derive them from a PRG.
But “derive from a PRG” is not a binary choice: there is a continuum between “tiny seed, huge expansion” and
“large seed, small expansion”. This gives a useful **engineering knob** for the assist-proof approach.

Let the verifier’s public description of (say) \(A\) be parameterized by:

- a PRG (or hash-to-ring) specification \(\mathsf{PRG}\),
- a seed schedule (which seeds exist, and which sub-blocks of \(A\) they define),
- and optionally explicit overrides for some sub-blocks.

Example knob: **block-seeded matrices**. Fix a block width \(W\) (in ring elements) and split the wide dimension of \(A\)
into blocks of width \(W\). Publish one seed per block. Then:

- **Public input size** grows like \(\#\text{blocks}\cdot|\text{seed}|\;\approx\;\left(\frac{\delta 2^m}{W}\right)\cdot|\text{seed}|\).
- The amount of PRG expansion that must be “accounted for” in an assist proof drops proportionally, because each seed only expands to a \(W\)-sized slice rather than the full \(\delta 2^m\) width.

Two extremes:

- **One global seed per matrix** (\(W=\delta 2^m\)): smallest public input, but the assist proof must certify the largest PRG expansion.
- **Fully explicit matrix** (\(W=1\) and “seed” is just the entry itself): huge public input, but there is no PRG computation left to certify.

This knob is useful because the sumcheck verifier only needs a **constant number of MLE evaluations** of these unstructured objects per iteration
(after batching, as above). So the best choice of granularity is a trade-off between:

- cheaper public inputs / transcript (smaller seeds), versus
- cheaper assist proofs (less PRG expansion to certify), versus
- implementation constraints (streaming PRG, vectorization, GPU friendliness, etc.).

Concretely, if the assist proof is implemented using a general-purpose zkVM (e.g. Jolt), you may prefer fewer/larger seeds (smaller inputs).
If instead you build a specialized “PRG-consistency proof” (tailored to \(\mathsf{PRG}\)), it can be beneficial to increase seeding granularity
so that the certified computation is smaller and more parallel.

##### Assist proof sketch: certify unstructured MLE evaluations of PRG-seeded tables

This subsection sketches a concrete “assist proof” interface for the unstructured matrices \(A,B,D\).
The key observation (above) is that, after batching, the verifier only needs a **constant number** of multilinear-extension evaluations
of coefficient tables derived from these matrices.

**What field is this over?** In the Jolt setting, sumcheck challenges and all verifier-side checks live in \(F_{q'}\), and ring switching evaluates
ring elements at \(X=\alpha\in F_{q'}\). Therefore:

- the *values the verifier needs* (e.g. entries of \(M(\alpha)\), and the batched MLE values derived from them) are elements of \(F_{q'}\);
- the underlying PRG outputs are small residues mod \(q\) (32-bit) which we embed into \(F_{q'}\) via the chosen lift convention
  (e.g. \([0,q)\subset \mathbb Z \hookrightarrow F_{q'}\), or symmetric \([-(q-1)/2,(q-1)/2]\)).

So the assist proof naturally proves arithmetic **in \(F_{q'}\)**, with additional range/boundedness checks only if needed to pin down the lift.

**Primitive we want to prove.** Abstract the unstructured part as a PRG-seeded table
\(\mathsf{T}:\{0,1\}^s\to \mathbb Z_q\) (think “one row of \(A\)”, or one of the few wide blocks that matter after batching).
For a transcript-derived point \(t\in F_{q'}^s\), define the multilinear extension value:

\[
\tilde{\mathsf{T}}(t) := \sum_{i\in\{0,1\}^s} \iota(\mathsf{T}(i))\cdot \mathrm{eq}(t,i)\ \in F_{q'},
\]

where \(\iota:\mathbb Z_q\to F_{q'}\) is the lift/injection convention and \(\mathrm{eq}\) is the standard equality polynomial.

In the Hachi-for-Jolt verifier, the needed “public encoding” values reduce (after batching) to a small constant number of such \(\tilde{\mathsf{T}}(t)\)
queries for different tables \(\mathsf{T}\) and points \(t\) (and with \(\alpha\)-dependent scalars already folded in).

**Assist-proof statement (one query).** Public:

- PRG spec \(\mathsf{PRG}\) and seed schedule for \(\mathsf{T}\),
- the point \(t\in F_{q'}^s\),
- the claimed value \(v\in F_{q'}\).

Prove:

1) \(\mathsf{T}\) is the table generated by \(\mathsf{PRG}\) from the published seeds (with the chosen “seed granularity” policy), and
2) \(v = \tilde{\mathsf{T}}(t)\) in \(F_{q'}\).

**Witness / computation pattern.** The prover can compute \(v\) in one streaming pass:

- expand PRG outputs \(\mathsf{T}(i)\) for all \(i\in\{0,1\}^s\) (or per-block),
- compute \(\mathrm{eq}(t,i)\) weights incrementally (Gray-code / tensor-product recurrence),
- accumulate \(v := \sum_i \iota(\mathsf{T}(i))\cdot \mathrm{eq}(t,i)\).

This is \(\Theta(2^s)\) time for the prover, which is expected: the whole point is to make the verifier avoid this scan.

**Most practical instantiation today (baseline)**: use a general-purpose zkVM SNARK (e.g. Jolt) to prove the above streaming computation.
This is attractive because:

- it is easy to implement first (no custom IOP design),
- it composes naturally (the assist proof becomes just another proof object checked by the verifier),
- and it can be amortized: if a verifier needs \(k\) such MLE values for the same seeds, the prover computes all \(k\) accumulators in one pass
  (PRG expansion once, \(k\) dot products in parallel).

**Where specialization can help (beyond a zkVM)**. Two big levers:

- Choose \(\mathsf{PRG}\) to be *arithmetic-friendly in \(F_{q'}\)* (e.g. a field sponge / permutation), so proving PRG expansion is far cheaper than proving
  bitwise SHAKE/AES-style code in a VM.
- Design a dedicated “PRG-consistency proof” that proves many PRG steps and the final dot product using a custom arithmetization (e.g. a small fixed gate set),
  rather than a full VM.

Either way, this assist proof lives entirely “above” the base PCS: it only needs to convince the verifier of a few \(F_{q'}\)-valued claims that the verifier
would otherwise compute by scanning a pseudorandom table.

### 4.6 Recursion boundary (why the output is foldable again)

The output claim has the same form as the input claim:

- a commitment to a multilinear object (now \(\tilde w\) instead of \(P\)) still built over the small-modulus commitment domain, and
- an opening point \(\mathbf r^\*\) sampled in \(F_{q'}\) from the sumcheck transcript.

So the same 4.3–4.5 pipeline can be repeated on \(\tilde w\) until the instance is small enough to hand off to a base PCS.

---

## 5. Concrete parameter crunching (Hachi-like params + \(q\approx 2^{32}\), \(q'\approx 2^{128}\))

Take the benchmark-ish tuple already recorded in `docs/HACHI_DIGEST.md` (Fig. 9 style):

- \(q\approx 2^{32}\), \(d=1024\), \(m=r=10\), \(b=16\), \(\delta=8\), \(\tau=4\), \(n_A=n_B=n_D=1\).

### 5.1 How many polynomials are in the modulus quotient witness \(s\)?

\(s\) is per row of the stacked linear system in Step B. With \(n_A=n_B=n_D=1\), the conceptual stack has \(O(1)\) rows (roughly 5 “macro” ring equations), so:

\[
n \approx 5,\quad s\in (Z[X]_{<d})^n.
\]

Raw coefficient count: \(\approx n\cdot d \approx 5\cdot 1024 \approx 5120\) integers.

### 5.2 Coefficient magnitude of \(s\) (back-of-the-envelope)

The “widest” object is typically the redecomposed folded witness \(\hat z\):

\[
|\hat z| = 2^m\cdot \delta\cdot \tau = 1024\cdot 8\cdot 4 = 32768 = 2^{15}.
\]

Redecomposition contributes weights up to about \(b^\tau = 16^4=2^{16}\). Ring convolution contributes a factor about \(d=2^{10}\). Summing \(2^{15}\) terms and dividing by \(q\approx 2^{32}\) yields a rough bound:

\[
|s[\cdot]|
\lesssim
2^{15}\cdot 2^{10}\cdot 2^{16}
=2^{41}.
\]

So \(s\) coefficients are on the order of **41 bits** (not 128 bits), and with \(b=16\) the digit length is about:

\[
\delta_s \approx \left\lceil \frac{41}{4}\right\rceil = 11.
\]

### 5.3 Witness-table overhead from \(s\)

Digitizing \(s\) adds about \(n\cdot \delta_s \approx 5\cdot 11 = 55\) extra ring elements’ worth of rows, which is negligible next to the main stacked witness size (tens of thousands of ring elements in the same parameter regime).

### 5.4 Large one-hot polynomials (Jolt RA scale): \(2^{38}\) and packed \(2^{44}\)

This is the scale that shows up in Jolt’s RA families:

- domain is \((k,c)\in\{0,1\}^{\log K}\times\{0,1\}^{\log T}\), so total size is \(K\cdot T\),
- in practice \(K\in\{16,256\}\) and (for large traces) \(T=2^{30}\),
- each RA polynomial is “one-hot in \(k\) for each cycle \(c\)”, i.e. Hamming weight \(\approx T\).

We care about **recursion/folding** cost: after one opening-proof iteration, we output a new committed witness-table polynomial \(\tilde w\) (Section 4.5) whose size determines how much the instance shrinks.

#### 5.4.1 General one-iteration size estimate (conservative, dense-table view)

Let the original scalar polynomial have size \(N=2^n\) (so \(n\) Boolean variables).
With coefficient packing into ring elements of degree \(d=2^{10}=1024\), the ring-element coefficient table has size:

\[
2^\ell = \frac{N}{d} = 2^{n-10},
\qquad \ell := n-10.
\]

Choose the Hachi split \(\ell=m+r\) (Section 2) and keep the same Hachi-ish parameters
\((b=16,\ \delta=8,\ \tau=4,\ n_A=1)\).

The dominant witness blocks contributing to \(\mu=|Z|\) are:

- \(\hat w\in R_q^{\delta\cdot 2^r}\),
- \(\hat t\in R_q^{n_A\delta\cdot 2^r}\),
- \(\hat z\in R_q^{2^m\delta\tau}\),

so the witness-table row count is approximately:

\[
\text{rows} \approx \delta\big((1+n_A)2^r + \tau 2^m\big)
= 8\big(2\cdot 2^r + 4\cdot 2^m\big),
\]

and the flattened scalar table size is:

\[
N' \approx d\cdot \text{rows}.
\]

To minimize \(N'\) for fixed \(\ell=m+r\), pick \(m\) and \(r\) near-balanced (within 1). With \(\tau=4\) and \(n_A=1\) the objective is very flat around \(m\approx r\).

#### 5.4.2 Case A: single RA polynomial size \(2^{38}\) (trace \(2^{30}\), \(K=256\))

Here \(n=38\) and \(d=2^{10}\) gives \(\ell=n-10=28\). Choose:

\[
m=r=14.
\]

Then:

\[
\text{rows} \approx 8\big(2\cdot 2^{14} + 4\cdot 2^{14}\big)
=
8\cdot 6\cdot 2^{14}
=786{,}432,
\]

and:

\[
N' \approx 1024\cdot 786{,}432 = 805{,}306{,}368 \approx 2^{29.6}.
\]

After padding to a power of two for a clean multilinear embedding, this is essentially a \(2^{30}\)-sized polynomial.

So a **single iteration** shrinks \(2^{38}\to \approx 2^{30}\) (about \(256\times\) after padding; about \(341\times\) before padding).

#### 5.4.3 Case B: pack/concatenate 64 one-hot polys into one size \(2^{44}\)

Packing 64 polynomials is “add a 6-bit selector dimension”, i.e. \(n=38+6=44\), so \(\ell=n-10=34\).
Choose:

\[
m=r=17.
\]

Then:

\[
\text{rows} \approx 8\cdot 6\cdot 2^{17} = 6{,}291{,}456,
\]

and:

\[
N' \approx 1024\cdot 6{,}291{,}456 = 6{,}442{,}450{,}944 \approx 2^{32.6},
\]

which pads to essentially \(2^{33}\).

So a **single iteration** shrinks \(2^{44}\to \approx 2^{33}\) (about \(2048\times\) after padding; about \(2731\times\) before padding).

#### 5.4.4 Why packing helps beyond “one proof instead of 64”

If you do **64 separate** openings of size \(2^{38}\) and recurse once, you conceptually produce about 64 witness-table polynomials of size \(\approx 2^{30}\), i.e. total next-instance mass \(\approx 2^{36}\).

If you **pack** and recurse once, you get one next-instance polynomial of size \(\approx 2^{33}\).

That is an **additional ~8× reduction** in the “total next instance size” even before accounting for proof-object overheads.

---

### 5.5 Proof size (precise) and verifier work per recursion iteration

This subsection pins down, for the **two-field** Jolt setting:

- exactly **what is sent** in one Hachi recursion iteration (Fiat–Shamir / non-interactive view),
- a **precise proof-size formula** (bytes / field elements / ring elements),
- and the verifier’s concrete work, highlighting the **square-root** term from building monomial vectors \((a,b)\).

#### 5.5.1 What is sent in one recursion iteration (Fiat–Shamir view)

One “opening-proof iteration” is Sections 4.3–4.6 applied to an input opening claim
\(\text{open }P(\mathbf r)=v\), producing a smaller output opening claim \(\tilde w(\mathbf r^\*) = v^\*\).

In Fiat–Shamir we only count **prover-sent** messages. Concretely, one iteration sends:

- **Aux commitment**: \(v := D\hat w \in R_q^{n_D}\). (Section 4.3.3)
- **Witness-table commitment**: \(t := \mathrm{Com}(\tilde w) \in R_q^{n_B}\). (Section 4.5)
- **Sumcheck transcript**: in each round \(i\in[m']\), the prover sends a univariate polynomial \(g_i\) (coefficients in \(F_{q'}\)) for a batched constraint polynomial. (Section 4.5.2)
- **Final value**: the claimed \(v^\* := \tilde w(\mathbf r^\*) \in F_{q'}\) at the final sumcheck point \(\mathbf r^\*\). (Section 4.5.2)

Everything else (fold challenge \(c\), ring-switch point \(\alpha\), batching points, and sumcheck challenges) is
verifier-derived from the transcript under Fiat–Shamir and is not sent.

Here \(m' := \log_2(\text{padded\_len}(\tilde w))\) is the sumcheck round count.

#### 5.5.2 Proof-size formula (bytes) for one iteration

Let:

- \(q\) be the small modulus (e.g. ~32-bit prime),
- \(q'\) be the Jolt field modulus (e.g. ~128-bit prime),
- \(d\) be the cyclotomic degree (e.g. 1024),
- \(b\) be the digit base (e.g. 16),
- \(n_B,n_D\) be the commitment “heights”.

Serialization model (simple, uncompressed, coefficient form):

- one ring element in \(R_q=\mathbb Z_q[X]/(X^d+1)\) costs
  \[
  \mathrm{bytes}(R_q)\;=\; d\cdot \left\lceil\frac{\log_2 q}{8}\right\rceil,
  \]
  i.e. \(d\) coefficients mod \(q\).
- one field element in \(F_{q'}\) costs
  \[
  \mathrm{bytes}(F_{q'})\;=\;\left\lceil\frac{\log_2 q'}{8}\right\rceil.
  \]

Then the **non-interactive proof size per iteration** is:

\[
\boxed{
\mathrm{bytes/iter}
=
(n_D+n_B)\cdot \mathrm{bytes}(R_q)
\;+\;
\mathrm{bytes(sumcheck\ transcript)}
\;+\;
\mathrm{bytes}(F_{q'})
}
\]

The only subtlety is the **per-round sumcheck payload**, which depends on how you implement batching and how you encode the univariate.

##### Variant A (paper-style conservative accounting): two separate sumchecks, full coefficients

If you run *two* sumchecks “in parallel” (one for the linear constraints, one for the range constraints) and send full coefficient vectors each round, you get:

- linear-constraint univariate degree 1 ⇒ **2** field elements/round,
- range-check univariate degree \(D_{\mathrm{range}}\) ⇒ **\(D_{\mathrm{range}}+1\)** field elements/round.

So:
\[
\mathrm{field\ elems/round} = 2 + (D_{\mathrm{range}}+1) = D_{\mathrm{range}}+3.
\]

In the common “nonnegative digit” choice \(T\in\{0,\dots,b-1\}\), a root-check polynomial can have degree \(D_{\mathrm{range}}=b\),
so this becomes **\(b+3\)** field elements/round (matching the paper’s “\(2\) (resp. \(b{+}1\))” phrasing).

Total bytes:
\[
\boxed{
\mathrm{bytes/iter}
=
(n_D+n_B)\cdot \mathrm{bytes}(R_q)
\;+\;
(D_{\mathrm{range}}+3)\cdot m'\cdot \mathrm{bytes}(F_{q'})
\;+\;
\mathrm{bytes}(F_{q'})
}
\]

##### Variant B (optimized): one combined sumcheck, and omit one coefficient using \(g_i(0)+g_i(1)\)

There are two independent optimizations that reduce “\(b+3\)” down to “\(\approx b\)”:

1) **Combine constraint families into one sumcheck**: prove one random linear combination of the linear-constraint polynomial and the range-check polynomial, so you only send one univariate per round.

2) **Exploit the sumcheck round constraint** \(g_i(0)+g_i(1)=z_{i-1}\): this is one linear constraint on the univariate coefficients, so you can omit one field element per round (e.g. reconstruct the constant term).

With these, if \(D_{\mathrm{range}}=b\), you can target:
\[
\mathrm{field\ elems/round} \approx D_{\mathrm{range}} = b.
\]

Total bytes:
\[
\boxed{
\mathrm{bytes/iter\ (opt)}
=
(n_D+n_B)\cdot \mathrm{bytes}(R_q)
\;+\;
D_{\mathrm{range}}\cdot m'\cdot \mathrm{bytes}(F_{q'})
\;+\;
\mathrm{bytes}(F_{q'})
}
\]

#### 5.5.3 Concrete numbers for the recommended Jolt regime

Use the “Hachi-like” regime in Section 5:

- \(q\approx 2^{32}\) ⇒ \(\mathrm{bytes}(R_q)=d\cdot 4\).
- \(q'\approx 2^{128}\) ⇒ \(\mathrm{bytes}(F_{q'})=16\).
- \(d=1024\), \(n_B=n_D=1\), \(b=16\), and (for nonnegative digits) \(D_{\mathrm{range}}=b=16\).

So:

- \(\mathrm{bytes}(R_q)=1024\cdot 4=4096\) bytes,
- commitments per iter = \((1+1)\cdot 4096 = 8192\) bytes,
- per-round sumcheck payload:
  - Variant A: \((b+3)\cdot 16 = 304\) bytes/round,
  - Variant B: \(b\cdot 16 = 256\) bytes/round,
- plus 16 bytes for \(v^\*=\tilde w(\mathbf r^\*)\).

Therefore:

\[
\mathrm{bytes/iter\ (A)} = 8192 + 304\,m' + 16,
\qquad
\mathrm{bytes/iter\ (B)} = 8192 + 256\,m' + 16.
\]

For the RA-scale post-iteration sizes in Section 5.4:

- **Case A** (one RA poly, after 1 iter \(\tilde w\) pads to \(\approx 2^{30}\)): \(m'=30\)
  - Variant A: \(8192 + 304\cdot 30 + 16 = 17{,}328\) bytes (~16.9 KiB)
  - Variant B: \(8192 + 256\cdot 30 + 16 = 15{,}888\) bytes (~15.5 KiB)
- **Case B** (pack 64 polys, after 1 iter pads to \(\approx 2^{33}\)): \(m'=33\)
  - Variant A: \(8192 + 304\cdot 33 + 16 = 18{,}240\) bytes (~17.8 KiB)
  - Variant B: \(8192 + 256\cdot 33 + 16 = 16{,}656\) bytes (~16.3 KiB)

#### 5.5.4 Verifier work per iteration (exact operations and where the square-root term comes from)

The opening point \(\mathbf r\in F_{q'}^\ell\) induces the multilinear monomial weight tables
\(a\in F_{q'}^{2^m}\) and \(b\in F_{q'}^{2^r}\):

\[
b^\top := (r_1^{i_1}\cdots r_r^{i_r})_{i\in\{0,1\}^r},
\qquad
a^\top := (r_{r+1}^{j_1}\cdots r_\ell^{j_m})_{j\in\{0,1\}^m}.
\]

Why this is necessary:

- \(a\) and \(b\) are the **public weights** that connect the committed coefficient blocks (hidden behind Ajtai commitments)
  to the public evaluation claim. Concretely, they appear in the Step-B constraints:
  - \(w_i := a^\top G_{2^m}s_i\),
  - the opening equation becomes \(b^\top w = \mathrm{open\_value}\),
  - and fold-consistency uses \(a^\top G_{2^m}z\).
  Without constructing these weights, the verifier cannot even *form* the public linear relation \(MZ=Y\) that the proof is about.

##### Two ways to handle \(a,b\) on the verifier

**(Naive / explicit materialization)**. If the verifier explicitly materializes the full vectors \(a\) and \(b\), then a standard DP build costs:

- Build \(a\): \(2^m-1\) multiplications in \(F_{q'}\).
- Build \(b\): \(2^r-1\) multiplications in \(F_{q'}\).

So the “square-root” term is:
\[
\boxed{
\mathrm{mul}_{a,b} = (2^m-1) + (2^r-1).
}
\]

Given \(\ell=m+r=\log_2(N/d)\) at the ring-element table level, balancing \(m\approx r\approx \ell/2\) gives:
\[
\mathrm{mul}_{a,b} \approx 2\cdot 2^{\ell/2} = 2\sqrt{2^\ell} = 2\sqrt{N/d},
\]
which is the concrete source of the “square-root verifier time” term.

**(Recommended / no materialization)**. The sumcheck verifier typically only needs \(a,b\) through evaluations of their multilinear
extensions at a few transcript-derived points (Section 4.5.3). Those evaluations have closed forms:
\(\tilde b(t)=\prod_j((1-t_j)+t_j r_j)\) and similarly for \(\tilde a\), so this part can be done in \(O(m+r)\) field operations.
In that implementation, the square-root cost above is **avoidable for the \(a,b\) contribution**; remaining verifier work then depends on how
the other public encodings (notably those derived from unstructured Ajtai/PRG matrices) are evaluated.

Other verifier costs per iteration (typically smaller than building \(a,b\) in the large-RA regime):

- **Ring-switch powers**: compute \(e_\alpha(\ell)=\alpha^\ell\) for \(\ell\in[0..d-1]\) in \(F_{q'}\): \(d-1\) multiplications.
- **Sumcheck checks**: for each round, verify \(g_i(0)+g_i(1)=z_{i-1}\) and evaluate \(g_i(a_i)\).
  - If the sent univariate has degree \(D\), evaluating \(g_i(a_i)\) costs \(D\) multiplications via Horner’s rule.
  - In Variant B with \(D=D_{\mathrm{range}}=b\), that is \(b\) multiplications per round.

Concrete counts for the RA cases in Section 5.4:

- **Case A** uses \(m=r=14\):
  \[
  \mathrm{mul}_{a,b} = 2(2^{14}-1) = 32{,}766.
  \]
- **Case B** uses \(m=r=17\):
  \[
  \mathrm{mul}_{a,b} = 2(2^{17}-1) = 262{,}142.
  \]

For reference, with \(d=1024\) the \(\alpha^\ell\) table is only \(1023\) multiplications.

#### 5.5.5 Multi-iteration recursion: next-instance sizes and when shrinking stops (fixed-parameter model)

Using the conservative size model in Section 5.4 (fixed \(d,\delta,\tau,b,n_A\), and balancing \(m\approx r\) each level),
recursing tends to shrink quickly for a few steps and then can hit a “padding/overhead” plateau.

Example (starting from Case A, where the first output is \(\approx 2^{30}\) scalars after padding):

- Iter 1 output: \(\approx 2^{30}\)  (Section 5.4.2)
- Iter 2 output: \(\approx 2^{26}\)
- Iter 3 output: \(\approx 2^{24}\)
- Iter 4 output: \(\approx 2^{23}\)
- Iter 5+: can stop shrinking materially if you keep the same digit parameters and witness-table overheads.

This is one reason the original Hachi paper composes with a base PCS at small sizes; in a “Hachi-only” design,
you should stop iterating once the next-instance size no longer decreases (since each extra iteration adds ~15–18 KiB of transcript at \(b=16\)).

### 5.6 P13 regime: using a 128-bit modulus directly (\(q = q' \approx 2^{128}\))

Sections 5.1–5.5 assume a small modulus \(q \approx 2^{32}\) with ring switching into the 128-bit Jolt field \(F_{q'}\).
An alternative is to set \(q = q'\) directly—i.e. commit over \(R_q\) where \(q\) is itself a 128-bit prime.
This eliminates the modulus-switching quotient witness \(s\) entirely and simplifies the protocol (no modulus embedding, \(k=1\)).

We use **P13** = \(2^{128} - 2^{13} - 2^4 + 1\) (see `docs/SPARSE_2P128_PRIMES_AND_IMPL.md`).

The cost: \(\delta = \lceil\log_b q\rceil\) is 4× larger than with a 32-bit prime (32 vs 8 at \(b=16\)),
so all commitment matrices widen by 4×. The question is whether security still holds with smaller \(d\) and larger \(n_A\).

#### 5.6.1 Target: \(n_{\mathrm{sis}} = d \cdot n_A = 512\)

The paper uses \(d=1024, n_A=1\) (so \(n_\mathrm{sis}=1024\)).
For practical reasons (smaller ring elements → smaller proofs, see Section 5.6.6), we target \(n_\mathrm{sis}=512\) via:

| Config | \(d\) | \(n_A\) | \(n_\mathrm{sis}\) | \(\mathrm{bytes}(R_q)\) |
|---|---|---|---|---|
| A | 512 | 1 | 512 | \(512 \cdot 16 = 8192\) |
| B | 256 | 2 | 512 | \(256 \cdot 16 = 4096\) |

Config B is preferred for proof size: each ring element is half the size, and commitments contribute \((n_B+n_D)\) ring elements per iteration regardless of \(n_A\).

Config \(d=256, n_A=3\) (\(n_\mathrm{sis}=768\)) was evaluated and gives 175–283 bits depending on the norm model—massively over-secured and wasteful.

#### 5.6.2 Challenge space: deriving \(\omega\) from \(d\)

Hachi uses sparse ternary challenges: each \(c \in R_q\) has exactly \(c_\mathrm{nz}\) non-zero coefficients, each \(\pm 1\), so \(\|c\|_1 = c_\mathrm{nz} =: \omega\).

The challenge space is:

\[
\mathcal C = \big\{ c \in R_q : \mathrm{hw}(c) = c_\mathrm{nz},\; c[l] \in \{-1,+1\} \text{ for } l \in \mathrm{supp}(c) \big\},
\qquad |\mathcal C| = \binom{d}{c_\mathrm{nz}} \cdot 2^{c_\mathrm{nz}}.
\]

For knowledge-soundness with security parameter \(\lambda = 128\), we need \(|\mathcal C| \geq 2^{128}\). Minimum \(c_\mathrm{nz}\) (computed via `math.comb` + `log2`):

| \(d\) | min \(c_\mathrm{nz}\) | \(\omega\) | \(\log_2|\mathcal C|\) |
|---|---|---|---|
| 1024 | 16 | 16 | 131.6 |
| 512 | 19 | 19 | 132.8 |
| 256 | 23 | 23 | 131.1 |

Note: \(d=1024\) recovers the paper's \(\omega=16\). For \(d=256\), \(\omega\) increases to 23—about 44% larger, which inflates the norm bound on \(z\). This is the price for smaller rings.

#### 5.6.3 Norm bound on the folded witness \(z\): sub-Gaussian analysis

After the fold step (paper §4.2), the prover computes:

\[
z = \sum_{i=1}^{2^r} c_i \cdot s_i \in R_q^{2^m \delta},
\]

where \(c_i \in \mathcal C\) are random challenges and \(s_i\) are the base-\(b\) decomposed blocks (coefficients in \(\{0,\ldots,b{-}1\}\)).

**Naive bound** (paper line 1742, triangle inequality):

\[
\|z\|_\infty \leq \sum_{i=1}^{2^r} \|c_i\|_1 \cdot \|s_i\|_\infty \leq 2^r \cdot \omega \cdot (b{-}1).
\]

For the paper's Fig 9 parameters: \(2^{10} \cdot 16 \cdot 15 = 245{,}760\). But the paper uses \(z = 30{,}583\)—over **8× smaller**.

The paper implicitly uses a tighter, concentration-based bound. We now derive it.

##### Sub-Gaussian concentration bound

**Setup.** Fix a coordinate \((j,k)\), where \(j \in [2^m \delta]\) indexes a ring element of \(z\) and \(k \in [d]\) indexes a scalar coefficient. Then:

\[
z_j[k] = \sum_{i=1}^{2^r} (c_i \cdot s_i^{(j)})[k].
\]

The ring product \((c \cdot s)[k]\) in \(R_q = \mathbb Z_q[X]/(X^d+1)\) is a negacyclic convolution:

\[
(c \cdot s)[k] = \sum_{l \in \mathrm{supp}(c)} c[l] \cdot s[(k{-}l) \bmod d] \cdot \sigma_{k,l},
\]

where \(\sigma_{k,l} \in \{+1,-1\}\) is the negacyclic sign (\(-1\) when \(l > k\), i.e. the product wraps around \(X^d = -1\)).

**Key observation.** The signs \(c[l] \in \{-1,+1\}\) are independent Rademacher random variables (by construction of \(\mathcal C\)), and \(s[(k{-}l) \bmod d] \cdot \sigma_{k,l}\) is a fixed bounded scalar. So conditioned on \(\mathrm{supp}(c)\), each \((c \cdot s)[k]\) is a **Rademacher sum**:

\[
(c \cdot s)[k] = \sum_{l \in \mathrm{supp}(c)} \varepsilon_l \cdot a_l, \qquad \varepsilon_l \stackrel{\mathrm{iid}}{\sim} \mathrm{Uniform}\{-1,+1\}, \quad |a_l| \leq b{-}1.
\]

By Hoeffding's lemma (sub-Gaussian tail for bounded independent r.v.s), this sum is sub-Gaussian with parameter:

\[
\sigma_i^2 = \sum_{l \in \mathrm{supp}(c_i)} a_l^2 \leq c_\mathrm{nz} \cdot (b{-}1)^2 = \omega \cdot (b{-}1)^2.
\]

**Summing over \(2^r\) independent challenges.** In the random oracle model (Fiat-Shamir), each \(c_i\) is an independent draw from \(\mathcal C\). So the \(2^r\) terms in \(z_j[k]\) are independent sub-Gaussian random variables. Their sum is sub-Gaussian with:

\[
\sigma^2_{\mathrm{total}} \leq 2^r \cdot \omega \cdot (b{-}1)^2.
\]

**Tail bound.** For a sub-Gaussian random variable with parameter \(\sigma^2\):
\(\Pr[|X| > t] \leq 2\exp(-t^2/(2\sigma^2))\).

**Union bound.** There are \(N_\mathrm{coords} = \delta \cdot 2^m \cdot d\) scalar coordinates in \(z\). Setting the union-bounded failure probability to \(2^{-\lambda}\):

\[
2 \cdot N_\mathrm{coords} \cdot \exp\!\left(-\frac{z_\infty^2}{2\sigma^2_\mathrm{total}}\right) \leq 2^{-\lambda}.
\]

Solving:

\[
\boxed{
z_\infty = (b{-}1) \cdot \sqrt{2 \cdot 2^r \cdot \omega \cdot \ln 2 \cdot (\lambda + 1 + \log_2(\delta \cdot 2^m \cdot d))}
}
\]

##### Verification against Figure 9

Paper parameters: \(b=16, r=10, \omega=16, \lambda=128, \delta=8, m=10, d=1024\).

\[
z_\infty = 15 \cdot \sqrt{2 \cdot 1024 \cdot 16 \cdot 0.693 \cdot (129 + 23)} = 15 \cdot 1858 = 27{,}870.
\]

Rounding up with \(b\) instead of \(b{-}1\): \(16 \cdot 1858 = 29{,}728\). The paper uses \(z = 30{,}583\), within **3%**.

##### Assumptions and provenance

This is a **standard application of Hoeffding/sub-Gaussian concentration** to the Hachi fold operation.
The same technique appears throughout lattice cryptography (Lyubashevsky's signature schemes, CRYSTALS-Dilithium parameter selection, Banaszczyk's transference theorems).

Three assumptions are required:

1. **Rademacher signs**: the \(\pm 1\) coefficient signs of each challenge \(c_i\) are independent and uniform.
   This holds by construction of the sparse ternary challenge distribution.

2. **Independence across \(i\)**: challenges \(c_1, \ldots, c_{2^r}\) are independent.
   In the **random oracle model** (ROM), each \(c_i = H(\mathrm{transcript} \| i)\) is an independent sample.
   This is the standard Fiat-Shamir assumption used universally in lattice-based cryptography.

3. **Fixed \(s_i\)**: the decomposed blocks are determined before the challenges are drawn.
   This holds by protocol structure (prover commits, then challenges are derived from the commitment).

**What if the ROM fails?** The worst case reverts to the naive bound \(2^r \cdot \omega \cdot b\). But:
- MSIS binding is unconditional (no ROM needed)—so commitment security is unaffected.
- Only the prover's \(z\) values might exceed the sub-Gaussian bound, causing a prover abort (liveness issue, not soundness).
- In practice, SHAKE-256 / Keccak provides an excellent ROM approximation.

**The paper does not explicitly derive this bound**, but Figure 9 uses \(z = 30{,}583\) (consistent with the sub-Gaussian formula within 3%) rather than the naive bound \(2^r \omega b = 262{,}144\). The analysis is implicit.

#### 5.6.4 MSIS security estimation

The commitment's binding property reduces to Module-SIS. Unrolling the module structure to plain SIS over \(\mathbb Z_q\):

\[
n_\mathrm{sis} = n_A \cdot d, \qquad m_\mathrm{sis} = \delta \cdot 2^m \cdot d, \qquad \beta_{L_2} = \sqrt{m_\mathrm{sis}} \cdot z_\infty.
\]

Security is estimated via the Lattice Estimator ([APS15], `SIS.lattice`) in the \(L_2\) norm.

Results for \(q = \text{P13}\), \(d = 256\), \(n_A = 2\), \(n_\mathrm{sis} = 512\), \(b=16\) (\(\delta=32\)), with balanced \(m \approx r\):

| \(\ell\) | \(m\) | \(r\) | \(z_\infty\) (subgauss) | \(\tau\) | security (bits) | \(z_\infty\) (naive) | security (bits) |
|---|---|---|---|---|---|---|---|
| 22 (\(n{=}30\)) | 11 | 11 | \(\approx 63\text{k}\) | 4 | **\(\approx 226\)** | \(\approx 754\text{k}\) | **\(\approx 210\)** |
| 30 (\(n{=}38\)) | 15 | 15 | \(\approx 205\text{k}\) | 5 | **\(\approx 176\)** | \(\approx 12\text{M}\) | **\(\approx 151\)** |
| 36 (\(n{=}44\)) | 18 | 18 | \(\approx 588\text{k}\) | 5 | **\(\approx 176\)** | \(\approx 96\text{M}\) | **\(\approx 131\)** |

Under the sub-Gaussian model, all cases are comfortably above 128 bits with \(\geq 48\) bits of headroom.
Under the naive model, all cases still exceed 128 bits, though \(\ell=44\) (Jolt packed regime) is tight at 131 bits.

Full sweep data: `lattice-estimator/hachi_p13_sweep_v3.csv`.
Sweep script: `lattice-estimator/hachi_p13_optimizer.py`.

#### 5.6.5 Recommended concrete parameters

**Primary recommendation: \(d=256,\ n_A=2,\ b=16\).**

| Parameter | Symbol | Value | Rationale |
|---|---|---|---|
| Prime modulus | \(q\) | P13 = \(2^{128} - 2^{13} - 2^4 + 1\) | Sparse pseudo-Mersenne, fast reduction |
| Ring degree | \(d\) | 256 | Minimizes \(\mathrm{bytes}(R_q) = 4096\) at 128-bit \(q\) |
| Inner matrix height | \(n_A\) | 2 | \(n_\mathrm{sis} = 512\) |
| Outer matrix height | \(n_B\) | 1 | |
| Aux matrix height | \(n_D\) | 1 | |
| Decomposition base | \(b\) | 16 | Sumcheck degree 16 per round (256 bytes/round) |
| Digit length | \(\delta\) | 32 | \(\lceil 128/4 \rceil = 32\) |
| Challenge weight | \(\omega\) | 23 | Min for \(|\mathcal C| \geq 2^{128}\) with \(d=256\) |
| Extension degree | \(k\) | 1 | \(q \approx 2^{128}\) already large enough |

Per-\(\ell\) fold parameters (balanced split, \(\alpha = \log_2 d = 8\)):

| Jolt regime | \(n\) | \(\ell = n - \alpha\) | \(m\) | \(r\) | \(\tau\) (subgauss) | sec (subgauss) | sec (naive) |
|---|---|---|---|---|---|---|---|
| Single RA | 38 | 30 | 15 | 15 | 5 | 176 bits | 151 bits |
| Packed 64 | 44 | 36 | 18 | 18 | 5 | 176 bits | 131 bits |
| Benchmark | 30 | 22 | 11 | 11 | 4 | 226 bits | 210 bits |

#### 5.6.6 Proof size per iteration (P13 regime)

A pleasant surprise: **the proof size is virtually identical to the \(q \approx 2^{32}\) regime**.

Ring element size:

\[
\mathrm{bytes}(R_q) = d \cdot \left\lceil\frac{\log_2 q}{8}\right\rceil = 256 \cdot 16 = 4096 \text{ bytes}.
\]

This is the same as \(d=1024,\, q \approx 2^{32}\) (where \(1024 \cdot 4 = 4096\)). The two regimes produce identically-sized ring elements.

Sumcheck field elements: \(q = q'\) so \(\mathrm{bytes}(F_{q'}) = 16\). Per round (Variant B): \(b \cdot 16 = 256\) bytes. Same as before.

The only difference is the number of sumcheck rounds \(m'\), which depends on the next-witness size:

\[
\mathrm{witness\_scalars} \approx d \cdot \delta \cdot \big((1+n_A) \cdot 2^r + \tau \cdot 2^m\big),
\qquad m' = \lceil \log_2(\mathrm{witness\_scalars}) \rceil.
\]

Concrete proof sizes (Variant B, \(d=256,\, n_A=2,\, b=16,\, \delta=32,\, n_B=n_D=1\)):

| Jolt regime | \(m,r\) | \(\tau\) | witness scalars | \(m'\) | commitments | sumcheck | total |
|---|---|---|---|---|---|---|---|
| Single RA (\(n{=}38\)) | 15, 15 | 5 | \(\approx 2^{31}\) | 31 | 8,192 B | 7,936 B | **16.1 KiB** |
| Packed 64 (\(n{=}44\)) | 18, 18 | 5 | \(\approx 2^{34}\) | 34 | 8,192 B | 8,704 B | **16.5 KiB** |
| Benchmark (\(n{=}30\)) | 11, 11 | 4 | \(\approx 2^{27}\) | 27 | 8,192 B | 6,912 B | **14.8 KiB** |

Compare with the \(q \approx 2^{32}, d=1024\) regime (from Section 5.5.3):

| | P13 (\(d{=}256\)) | Fig 9 (\(d{=}1024\)) | Difference |
|---|---|---|---|
| \(\mathrm{bytes}(R_q)\) | 4,096 | 4,096 | identical |
| Proof (RA, \(n{=}38\)) | 16.1 KiB | 15.5 KiB | +0.6 KiB (+1 round) |
| Proof (packed, \(n{=}44\)) | 16.5 KiB | 16.3 KiB | +0.2 KiB (+1 round) |

The P13 regime adds at most 1–2 extra sumcheck rounds (256–512 bytes) compared to the 32-bit regime,
because the larger \(\delta\) (32 vs 8) is compensated by the smaller \(d\) (256 vs 1024).

**Why \(b=16\) over \(b=32\):** Setting \(b=32\) reduces \(\delta\) from 32 to 26, narrowing the commitment matrices.
But each sumcheck round sends \(b\) field elements, so the per-round payload doubles from 256 to 512 bytes,
adding \(\approx m' \cdot 256 \approx 8\) KiB to the proof. Since we're optimizing for proof size, \(b=16\) is preferred.

#### 5.6.7 Comparison: P13 vs Fig 9 regime

| | \(q \approx 2^{32}\) (Fig 9) | \(q = \text{P13} \approx 2^{128}\) |
|---|---|---|
| Modulus size | 32 bits | 128 bits |
| Ring degree \(d\) | 1024 | 256 |
| \(n_A\) | 1 | 2 |
| \(n_\mathrm{sis}\) | 1024 | 512 |
| \(\delta\) at \(b{=}16\) | 8 | 32 |
| \(\omega\) | 16 | 23 |
| Ring element size | 4,096 B | 4,096 B |
| Proof size (RA) | ~15.5 KiB | ~16.1 KiB |
| Security (subgauss) | 157 bits | 176 bits |
| Needs ring switching | yes (\(k=4\)) | no (\(k=1\)) |
| Needs modulus quotient \(s\) | yes | no |
| NTT in \(R_q\) | yes (\(2048 \mid q{-}1\)) | no (\(512 \nmid q{-}1\)) |

**Advantages of P13**: no ring switching, no modulus quotient, simpler protocol, same per-iteration proof size.

**Disadvantages**:

1. P13 does not admit NTT for \(d=256\) (since \(512 \nmid P13{-}1\); the 2-adic valuation of \(P13{-}1\) is only \(2^4\)).
   Ring multiplication must use schoolbook \(O(d^2)\) or Karatsuba \(O(d^{1.585})\) instead of \(O(d \log d)\).
   For \(d=256\): Karatsuba costs \(\approx 16{,}000\) operations per ring multiply (vs \(\approx 2{,}000\) with NTT).
   This affects **prover cost** but not proof size or verifier cost.
   See `docs/SPARSE_2P128_PRIMES_AND_IMPL.md` for NTT-friendly alternatives.

2. P13 converges slower during recursion (Section 5.6.8 below).

#### 5.6.8 Recursion convergence: P13 needs more iterations

Section 5.6.6 showed that per-iteration proof size is nearly identical. But the **number of iterations** required differs, because P13's larger \(\delta\) and \(n_A\) inflate the next-witness size.

##### Recurrence for next-instance size

After one balanced-split iteration (\(m \approx r \approx \ell/2\)), the number of scalar variables evolves as:

\[
n' \;\approx\; \frac{n}{2} \;+\; \underbrace{\frac{\alpha}{2} + \log_2\!\big(\delta(1+n_A+\tau)\big)}_{\text{overhead constant } C}
\]

where \(\alpha = \log_2 d\). The fixed point of this recurrence is \(n^* = 2C\): the instance can never shrink below \(\sim 2^{n^*}\) scalars no matter how many iterations are applied.

| | Fig 9 | P13 |
|---|---|---|
| \(d,\; \alpha\) | 1024, 10 | 256, 8 |
| \(\delta\) | 8 | 32 |
| \(n_A\) | 1 | 2 |
| \(\tau\) | 4 | 5 |
| \(\delta(1{+}n_A{+}\tau)\) | 48 | 256 |
| overhead \(C\) | 10.6 | 12.0 |
| **fixed point \(n^*\)** | **21.2** | **24.0** |

P13's overhead is 5.3× larger, driven by \(\delta\) (4×), \(1{+}n_A\) (1.5×), and \(\tau\) (1.25×).

##### Recursion trace from \(n=38\) (single RA poly)

| Iteration | Fig 9 (\(n\)) | Fig 9 (\(N'\)) | P13 (\(n\)) | P13 (\(N'\)) |
|---|---|---|---|---|
| 0 (input) | 38 | \(2^{38}\) | 38 | \(2^{38}\) |
| 1 | 30 | \(2^{30}\) | 31 | \(2^{31}\) |
| 2 | 26 | \(2^{26}\) | 28 | \(2^{28}\) |
| 3 | 24 | \(2^{24}\) | 26 | \(2^{26}\) |
| 4 | 23 | \(2^{23}\) | 25 | \(2^{25}\) |
| 5 | 22 | \(2^{22}\) | 25 | \(2^{25}\) (stuck) |

##### Recursion trace from \(n=44\) (packed 64 polys)

| Iteration | Fig 9 (\(n\)) | Fig 9 (\(N'\)) | P13 (\(n\)) | P13 (\(N'\)) |
|---|---|---|---|---|
| 0 (input) | 44 | \(2^{44}\) | 44 | \(2^{44}\) |
| 1 | 33 | \(2^{33}\) | 34 | \(2^{34}\) |
| 2 | 27 | \(2^{27}\) | 29 | \(2^{29}\) |
| 3 | 24 | \(2^{24}\) | 27 | \(2^{27}\) |
| 4 | 23 | \(2^{23}\) | 25 | \(2^{25}\) |
| 5 | 22 | \(2^{22}\) | 25 | \(2^{25}\) (stuck) |

##### Number of Hachi iterations before terminal-PCS handoff

| Handoff threshold | Fig 9 iters (\(n{=}38\)) | P13 iters (\(n{=}38\)) | Extra | Fig 9 iters (\(n{=}44\)) | P13 iters (\(n{=}44\)) | Extra |
|---|---|---|---|---|---|---|
| \(2^{28}\) | 2 | 2 | 0 | 2 | 2 | 0 |
| \(2^{26}\) | 2 | 3 | **+1** | 3 | 3 | 0 |
| \(2^{24}\) | 3 | \(\infty\) | — | 3 | \(\infty\) | — |

##### Total Hachi proof size (excluding terminal PCS)

| Handoff | Fig 9 (\(n{=}38\)) | P13 (\(n{=}38\)) | Overhead |
|---|---|---|---|
| \(2^{28}\) | 31.0 KiB | 32.2 KiB | +1.2 KiB (+4%) |
| \(2^{26}\) | 31.0 KiB | 48.3 KiB | +17.3 KiB (+56%) |

| Handoff | Fig 9 (\(n{=}44\)) | P13 (\(n{=}44\)) | Overhead |
|---|---|---|---|
| \(2^{28}\) | 31.0 KiB | 32.2 KiB | +1.2 KiB (+4%) |
| \(2^{26}\) | 46.5 KiB | 48.3 KiB | +1.8 KiB (+4%) |

##### Takeaway

The per-iteration proof size is identical, but **P13 converges to a higher floor** (\(n^* \approx 24\) vs \(21\)).

- If the terminal PCS can handle \(2^{28}\) efficiently (e.g. Greyhound / Brakedown): **both regimes use the same number of iterations**, and P13's total overhead is negligible (+4%).
- If you need to recurse down to \(2^{26}\): P13 may need **one extra iteration** for \(n=38\) (+56% Hachi proof cost, though the terminal proof at \(2^{26}\) is cheaper than at \(2^{28}\)).
- If you need to reach \(2^{24}\) or below: P13 **cannot converge there** — its fixed point is \(n^* = 24\). The small-\(q\) regime is necessary.

The choice reduces to: **is the terminal PCS cheap enough at \(2^{28}\) to make the extra iteration moot?** If yes, P13 wins on simplicity (no ring switching, no modulus quotient). If not, the small-\(q\) regime's faster convergence justifies the protocol complexity.

## 6. Implementation-ready mapping (this repo + required Jolt changes)

### 6.1 What exists in this repo already

- Commitment core (Hachi §4.1 shape): see `src/protocol/commitment/*` and `docs/ONE_HOT_COMMITMENT_COST_AND_GPU_PRG.md`.
- Sparse fold challenges \(c\): `src/protocol/opening/ring_switch/challenges.rs`.
- Opening orchestration namespace: `src/protocol/opening/mod.rs`.
- Lift helper for extension towers: `src/algebra/fields/lift.rs` (note: this is for \(F_{q^k}\) towers, not cross-prime).

### 6.2 What must be added for Jolt integration

1. **Two-field PCS interface**: Jolt currently assumes one field type `F: JoltField` for both polynomial coefficients and challenges. For Hachi-for-Jolt we need:

   - commitment/witness modulus domain: \(F_q\) / \(R_q\),
   - challenge/evaluation domain: \(F_{q'}\).

   This likely requires splitting Jolt’s PCS trait into:

   - `CommitField` and `ChallengeField` (or `StatementField`),
   - plus explicit serialization/transcript rules for sampling \(\mathbf r\in F_{q'}^m\).

2. **Ring-switch evaluation map into \(F_{q'}\)**:

   - implement \(\mathrm{ev}_\alpha: R_q \to F_{q'}\) (with a chosen lift convention),
   - implement quotient witnesses for both \((X^d+1)\) and \(q\) (the \(r\) and \(s\) slacks),
   - digitize and range-check those witnesses in the same sumcheck framework.

3. **Coefficient embedding path for one-hot/bit tables**:

   - choose an embedding layout consistent with Jolt’s polynomial storage and with \(d\),
   - ensure bitness constraints are enforced (cheap: \(T(T-1)=0\)).

### 6.3 Suggested milestone plan

- MVP: support bit/one-hot polynomials only (so coefficient lift is canonical).
- Add a single-polynomial opening proof at one point; batch later (Jolt already does RLC batching).

---

## 7. Security (self-contained) in the two-field setting

This section summarizes the **security argument shape** for the PCS described in Sections 4–5, without relying on external documents.

We separate:

- **Computational binding** of the commitment (Module/Ring-SIS),
- **Knowledge soundness** of the opening proof (special soundness / extraction),
- and the only substantive change in the two-field setting: checks happen in \(F_{q'}\) and must include the extra modulus quotient witness \(s\).

### 7.1 Security goal (PCS binding / evaluation soundness)

Informal PCS binding goal:

> After seeing a commitment \(C\) to a multilinear table \(P\), no PPT prover can produce two different valid openings at the same point \(\mathbf r\in F_{q'}^m\) (or produce an opening to a wrong value) except with negligible probability, under Module/Ring-SIS.

Equivalently: if an adversary convinces the verifier of \(P(\mathbf r)=v\) for some \(v\) not consistent with *any* “small” committed witness, then we can extract a short Module/Ring-SIS solution.

### 7.2 Commitment assumption: Module/Ring-SIS (binding of Ajtai commitments)

The commitment core is Ajtai/Module-SIS style over the cyclotomic ring \(R_q\):

\[
\mathrm{Com}(x) := A x \in R_q^{n}
\]

for a uniformly random public matrix \(A\) over \(R_q\) (or a “virtual matrix” derived from a PRG).

**Module/Ring-SIS assumption (informal)**: given uniformly random \(A\), it is hard to find a nonzero “short” vector \(z\) such that:

\[
A z = 0 \quad\text{in } R_q^n,
\qquad 0 < \|z\|_\infty \le B,
\]

for the relevant norm bound \(B\) induced by the protocol’s digit/range constraints.

### 7.3 Weak openings and weak binding (what binding we actually need)

Hachi-style inner/outer commitments use digit decompositions and fold challenges. The security proof typically relies on a **weak binding** notion: it is enough that the commitment is binding with respect to openings that satisfy:

- the stated linear equations (e.g. \(A s_i = G\hat t_i\) and \(B[\hat t_i]=u\)), and
- the stated shortness bounds (either directly or after multiplying by a short challenge element).

Weak binding has the standard reduction:

> Given two distinct valid weak openings to the same commitment, one can efficiently derive a short nonzero kernel vector for the public commitment matrix, hence solve Module/Ring-SIS.

(This is exactly the lemma used for Greyhound-style commitments; the algebra of the commitment itself is unchanged by whether later checks occur in \(F_{q'}\).)

### 7.4 Knowledge soundness outline: special soundness + extraction

The opening proof is an interactive, public-coin protocol consisting of:

1. **Split-and-fold layer** (Section 4.3): prover sends an auxiliary commitment \(v\), verifier samples a short/sparse fold challenge vector \(c\), prover forms folded witnesses and digitizations.
2. **Ring switching layer** (Section 4.4): verifier samples \(\alpha\in F_{q'}\), checks field-native equalities derived from ring equations at \(X=\alpha\).
3. **Constraint batching + sumcheck layer** (Section 4.5): verifier samples random points to batch constraints, and the parties run sumcheck to reduce everything to one evaluation of the committed witness-table multilinear \(\tilde w\).

To argue knowledge soundness, we use the standard “special soundness” pattern:

- from many accepting transcripts under a special “structured challenge set” (the SS/CWSS notion), an extractor can recover a valid witness,
- or else it obtains two different valid openings to the same commitment, contradicting weak binding.

The only algebraic ingredient needed for special soundness is polynomial identity testing:

> If a univariate polynomial of degree \(\le D\) agrees with a claimed value at \(D+1\) distinct points, then it must be the zero polynomial.

This holds over any field, including \(F_{q'}\).

### 7.5 The key delta in the two-field setting: ring switching needs the extra \(q\cdot s\) slack

In a one-modulus setting (checks in characteristic \(q\)), “\(E=0\) in \(R_q\)” can be checked by lifting to \(Z_q[X]\) and adding only the cyclotomic quotient witness:

\[
E = (X^d+1)\,r \quad\text{over } Z_q[X].
\]

In the two-field Jolt setting, checks occur in a different characteristic (\(q'\)), so “\(E\equiv 0\pmod q\)” must be made explicit as an integer multiple of \(q\).

That is why the correct lifted identity is:

\[
\mathrm{lift}_q(E) \;=\; (X^d+1)\,r \;+\; q\cdot s \quad\text{over } Z[X].
\]

Here \(s\) is the **modulus quotient witness**.

Without \(s\), a malicious prover could exploit the fact that equality mod \(q\) does not imply equality over integers (and hence does not imply equality in \(F_{q'}\) after any chosen lift).

### 7.6 Special soundness of ring+modulus switching over \(F_{q'}\) (the extracted witness)

Fix a single linear row of the stacked linear system (Section 4.4.2) and define the lifted difference polynomial:

\[
\Delta(X) := \mathrm{lift}_q(M)\,\mathrm{lift}_q(Z) - \mathrm{lift}_q(Y) - (X^d+1)\,r - q\,s \in Z[X]_{< 2d}.
\]

Degree bound: \(\deg(\Delta)\le 2d-1\).

In the *real protocol*, the verifier samples a single \(\alpha\) and checks the identity at that point.
In the *special-soundness / extraction* view, an extractor rewinds (or, in the Fiat–Shamir ROM, reprograms / resamples) the transcript to obtain \(2d\) accepting transcripts whose only difference is \(2d\) distinct points \(\alpha_1,\dots,\alpha_{2d}\in F_{q'}\).

If all these checks pass, then for each \(\alpha_i\):

\[
\Delta(\alpha_i)=0 \quad\text{in } F_{q'}.
\]

Since \(\Delta\) has degree \(\le 2d-1\), this forces \(\Delta\equiv 0\) as a polynomial (over the integers, hence also mod \(q\)), yielding:

\[
\mathrm{lift}_q(M)\,\mathrm{lift}_q(Z) - \mathrm{lift}_q(Y) = (X^d+1)\,r + q\,s.
\]

So a special-soundness extractor can recover a valid integer witness \((Z,r,s)\), unless it finds two different accepted openings to the same commitment, in which case it breaks weak binding.

Soundness error from the ring-switch step is bounded by:

\[
\Pr[\text{false statement passes}] \le \frac{2d-1}{|F_{q'}|}.
\]

For \(d=1024\) and a 128-bit prime \(q'\), this is negligible.

### 7.7 Sumcheck layer: special soundness and final opening

The sumcheck layer is standard:

- the prover’s messages are low-degree univariates,
- the verifier samples random field challenges in \(F_{q'}\),
- and at the end the verifier reduces correctness to a small number of evaluations of the committed witness-table polynomial \(\tilde w\) at a random point \(\mathbf r^\*\).

Special soundness of sumcheck does not depend on the field being an extension of \(F_q\); it holds over any field used for challenges, including \(F_{q'}\).

The only protocol-specific requirement is that the committed witness table includes **all** witness coordinates whose values appear in the sumcheck constraints:

\[
\tilde w \text{ encodes } (Z,\ \text{digits}(r),\ \text{digits}(s)).
\]

and that the range constraints enforce that those coordinates correspond to genuine integers (bounded digits) so that extraction yields “short” openings required by weak binding.

### 7.8 Fiat–Shamir (non-interactive PCS)

In practice, Jolt uses Fiat–Shamir: all verifier challenges (\(c\), \(\alpha\), batching points, and sumcheck challenges) are derived from a transcript.

The security argument follows the standard ROM pattern:

- special soundness of the interactive protocol implies knowledge soundness in the ROM after Fiat–Shamir,
- the knowledge error is essentially the interactive soundness error (dominated by the degree/field-size terms),
- and any successful forgery implies either a violation of weak binding (hence a Module/Ring-SIS solution) or a violation of the random-oracle model assumptions.

### 7.9 Why additive quotient witnesses \((r,s)\) are inherently non-unique (and why we do not care)

In Sections 4.4–4.5, we introduce *additive* quotient witnesses \(r\) and \(s\) to express ideal membership:

\[
\mathrm{lift}_q(E) = (X^d+1)\,r + q\,s \quad\text{over } Z[X],
\]

for some known polynomial/vector \(E\) derived from the protocol’s linear system.

Even when \(\mathrm{lift}_q(E)\) is fixed, the pair \((r,s)\) is **not unique** in general, because it is a
representation of an element in the ideal \(\langle X^d+1,\ q\rangle\subset Z[X]\), and the generators \((X^d+1)\) and \(q\) are not a basis.
Concretely, for any \(t\in Z[X]\) (or vector of polynomials, row-wise), define:

\[
r' := r + q\,t,
\qquad
s' := s - (X^d+1)\,t.
\]

Then:

\[
(X^d+1)\,r' + q\,s'
=(X^d+1)(r+qt)+q(s-(X^d+1)t)
=(X^d+1)r + q s.
\]

So there is an *affine family* of valid quotient witnesses.

Additionally, \(s\) also absorbs **lift-convention differences**:
changing \(\mathrm{lift}_q(\cdot)\) (e.g. canonical vs centered representatives) changes \(\mathrm{lift}_q(E)\) by a multiple of \(q\), which can be reabsorbed into \(s\).

Why we do not care:

- The verifier’s job is to check **existence** of some bounded-digit witness that makes the relation true.
- Extraction for knowledge soundness only needs to output *some* valid \((r,s)\) (with bounded digits), not a canonical one.
- The “semantic” object we care about for Jolt is the committed polynomial’s coefficient table (one-hot/bit values), not the quotient slack.

### 7.10 What “extracting Jolt one-hot polynomials” should mean in this PCS setting

Your goal is exactly the right one: you do **not** need uniqueness of witnesses; you need that a prover who can convince the verifier of openings must “know” an underlying committed one-hot polynomial table.

Formally, define a PCS opening relation \(R_{\mathrm{open}}\) whose instance is:

\[
u,\ \mathbf r\in F_{q'}^\ell,\ v\in F_{q'},
\]

and whose witness includes:

- the committed coefficient table \(P\) (for Jolt RA this is a one-hot/bit table),
- and whatever auxiliary objects are required by the opening protocol (digits, fold wiring, and quotient slacks).

Knowledge soundness for the opening proof says:

> for any (expected) poly-time prover that outputs an accepting proof for \((u,\mathbf r,v)\), there exists an extractor that outputs a witness \((P,\ldots)\) such that \((u,\mathbf r,v; P,\ldots)\in R_{\mathrm{open}}\).

In our recommended embedding (Section 3.2), the coefficient table \(P\) is *bit/one-hot*, so there is a **canonical** interpretation “as integers”:
coefficients are literally \(0/1\) and the lift \(\mathrm{lift}_q\) is unambiguous on them.
Therefore, once extraction recovers a valid bounded-digit witness table for the opening protocol, it also recovers the original Jolt one-hot polynomial coefficients (up to trivial padding/packing conventions).

The quotient witnesses \((r,s)\) are not unique and are not “Jolt semantics”; they are merely bookkeeping for checking ring/modulus equalities inside \(F_{q'}\).

### 7.11 Strong vs weak interactive reductions (precise definitions; why they matter here)

The SuperNeo paper introduces a clean way to reason about exactly the phenomenon you pointed out:
extractors in lattice protocols can often recover only **relaxed/weak** openings unless additional structure is present.
We restate their definitions here because they are the right abstraction layer for “do we extract the *actual* committed witness-table values?”

#### 7.11.1 Weak interactive reduction (extract relaxed witness + “single-valuedness” on \(\phi\)-classes)

Let \(R_1\subseteq R_1'\) be a relation and its relaxation (same instance space, weaker witness condition), and let \(R_2\) be another relation.
Let \(U_1\) be the ambient instance space of \(R_1\) (i.e. the set that contains all possible instances \(u_1\)).

An interactive reduction \(\Pi: R_1\to R_2\) (as in Section 7.4’s “public-coin transcript” view) is **weak** if it is complete and public-coin, and there exists a function \(\phi:U_1\to C\) (for some space \(C\)) such that:

1) (**Relaxed extraction**) for any (expected) poly-time adversary \((\mathcal A,\mathcal P^\*)\), there exists an (expected) poly-time extractor \(\mathcal E\) such that, letting \(\epsilon(\mathcal A,\mathcal P^\*)\) be the adversary’s acceptance probability (non-negligible),

\[
\Pr\Big[
(pp,s,u_1,w_1)\in R_1'
\ \Big|\ 
pp\leftarrow G(1^\lambda,sz),\ (s,u_1,st)\leftarrow \mathcal A(pp),\ (pk,vk)\leftarrow K(pp,s),\ w_1\leftarrow \mathcal E(pp,s,u_1,st)
\Big]
\ \ge\
\epsilon(\mathcal A,\mathcal P^\*)-\mathrm{negl}(\lambda).
\]

2) (**Single-valuedness on \(\phi\)-classes**) for any adversary \(\mathcal A=(\mathcal B,\mathcal B')\) that samples a shared internal state \(st^\*\) and then produces two instances \(u_1,u_1'\) with \(\phi(u_1)=\phi(u_1')\),

\[
\Pr\Big[
w_1\neq w_1'\ \wedge\ w_1\neq\bot\ \wedge\ w_1'\neq\bot
\Big]
\le \mathrm{negl}(\lambda),
\]

where the probability is over:

\[
pp\leftarrow G(1^\lambda,sz),\ (s,st^\*)\leftarrow \mathcal B(pp),\ (u_1,st)\leftarrow \mathcal B'(st^\*),\ (u_1',st')\leftarrow \mathcal B'(st^\*),
\]

and the extracted witnesses are:

\[
w_1\leftarrow \mathcal E(pp,s,u_1,st),\qquad w_1'\leftarrow \mathcal E(pp,s,u_1',st').
\]

Intuition: a weak reduction may only let you extract a witness for a **relaxed** relation (e.g. “weak openings” / “relaxed openings”), but the extractor is essentially “single-valued” once \(\phi(\cdot)\) is fixed.

#### 7.11.2 Strong interactive reduction (output instance is pinned by \(\phi\) + extraction of the original witness)

Let \(R_2\subseteq R_2'\) be a relation and its relaxation, and let \(U_2\) be the ambient instance space of \(R_2\).

An interactive reduction \(\Pi:R_1\to R_2\) is **strong** if it is complete and public-coin, and there exists \(\phi:U_2\to C\) such that:

1) (**\(\phi\)-consistency of the reduced instance**) for any adversary \((\mathcal A,\mathcal P^\*)\), if we run the prover/verifier interaction twice on the same initial adversary state, producing reduced instances \(u_2,u_2'\), then:

\[
\Pr\big[\phi(u_2)=\phi(u_2')\big]=1.
\]

2) (**Extraction**) for any adversary \((\mathcal A,\mathcal P^\*)\), there exists an extractor \(\mathcal E\) such that, defining the adversary’s success probability on the *relaxed output* relation:

\[
\epsilon'(\mathcal A,\mathcal P^\*) :=
\Pr\Big[
(pp,s,\langle \mathcal P^\*,\mathcal V\rangle((pk,vk),u_1,st))\in R_2'
\ \Big|\ 
pp\leftarrow G(1^\lambda,sz),\ (s,u_1,st)\leftarrow \mathcal A(pp),\ (pk,vk)\leftarrow K(pp,s)
\Big],
\]

if \(\epsilon'(\mathcal A,\mathcal P^\*)\ge 1/\mathrm{poly}(\lambda)\) and additionally:

\[
\Pr\Big[
w_2\neq w_2'\ \wedge\ w_2\neq\bot\ \wedge\ w_2'\neq\bot
\Big]
\le \mathrm{negl}(\lambda),
\]

where \((u_2,w_2)\leftarrow \langle \mathcal P^\*,\mathcal V\rangle((pk,vk),u_1,st)\) and \((u_2',w_2')\leftarrow \langle \mathcal P^\*,\mathcal V\rangle((pk,vk),u_1,st)\) are two independent runs of the interaction from the same initial adversary state, then:

\[
\Pr\Big[
(pp,s,u_1,w_1)\in R_1
\ \Big|\ 
pp\leftarrow G(1^\lambda,sz),\ (s,u_1,st)\leftarrow \mathcal A(pp),\ (pk,vk)\leftarrow K(pp,s),\ w_1\leftarrow \mathcal E(pp,s,u_1,st)
\Big]
\ge \epsilon'(\mathcal A,\mathcal P^\*)-\mathrm{negl}(\lambda).
\]

#### 7.11.3 Why this matters for Hachi-for-Jolt

Hachi’s own security discussion explicitly notes that *naive* extraction in lattice settings may yield only “weak openings” (multiplicative relaxation), because inverses of challenge differences can have large norm.
That is exactly the setting where weak/strong interactive reductions are useful:

- A **weak** step corresponds to extracting something like a weak opening (or a relaxed opening) in a way that is essentially unique for a fixed commitment transcript.
- A **strong** step corresponds to ensuring that rewinding does not change the “semantic” reduced instance, so that composing with a weak step yields a genuine reduction of knowledge.

For our purposes, the “semantic object” is the Jolt one-hot/bit coefficient table embedded into the committed polynomial.
So the design target is:

- ensure the protocol’s *extracted* witness table entries are range-checked to be actual bits/digits (so they correspond to a real committed table), and
- ensure the extractor’s output is tied to the commitment (weak binding / uniqueness),

which together imply that the prover “knows” the underlying Jolt one-hot polynomial(s), even though the additive quotient witnesses \((r,s)\) remain non-unique and unimportant.

### 7.12 Mapping SuperNeo’s strong/weak framework onto Hachi-for-Jolt

This subsection makes the correspondence concrete: *which parts* of the Hachi-for-Jolt opening proof behave like “weak interactive reductions” (relaxed extraction) versus “strong interactive reductions” (full extraction), and why this is enough for your goal: **an extractor can recover the underlying Jolt one-hot/bit coefficient table**.

#### 7.12.1 The semantic instance class \(\phi\): “same commitment means same polynomial”

For PCS knowledge soundness, the natural semantic equivalence class is: “instances that share the same commitment should correspond to the same underlying polynomial table”.

So we take the SuperNeo-style class function:

\[
\phi(u,\mathbf r,v) := u.
\]

This choice matches what we actually want from extraction:

- it is *fine* if the prover opens at many different points \(\mathbf r\),
- but all those openings should be consistent with a single committed polynomial determined by \(u\).

In Jolt’s one-hot regime (Section 1.3), the coefficient table is range-restricted (bits / small digits), so “same \(u\)” computationally pins a unique small-norm witness-table, under Module/Ring-SIS.

#### 7.12.2 Where Hachi is “weak”: the fold/RLC step and weak openings

The “fold challenge” step (Section 4.3.4) is exactly a random linear combination:

\[
z := \sum_{i=1}^{2^r} c_i s_i.
\]

In lattice settings, extracting the individual \(s_i\) from two accepting transcripts typically requires dividing by \((c-c')\), and inverses of short ring elements can have large norm.
Hachi’s paper explicitly embraces this by extracting **weak openings**: tuples \((s_i,\hat t_i,c_i)\) that satisfy the commitment equations but only guarantee \(\|c_i\cdot s_i\|\) is small (not necessarily \(\|s_i\|\)).

So this step naturally matches a **weak interactive reduction** viewpoint:

- extraction may land in a *relaxed* relation (“weak openings” / “relaxed openings”), and
- single-valuedness is enforced by **weak binding**: producing two distinct weak openings for the same commitment \(u\) yields a short nonzero Module/Ring-SIS solution.

This is the key conceptual parallel to SuperNeo’s discussion of “relaxed openings \(\Delta\cdot c = Az\)” and why weak reductions can still be safe when the extractor is forced to be essentially single-valued.

#### 7.12.3 Where Hachi-for-Jolt is “strong”: range checks make the extracted table a real one-hot witness

Your requirement is stronger than “extract some relaxed opening”: you want extraction to recover the **original Jolt one-hot polynomials** (i.e., the bit/one-hot coefficient table \(P\)).

In our Hachi-for-Jolt design, we achieve this by ensuring the extracted witness is not merely a weak opening but a **bounded-digit witness table**:

- The witness table \(\tilde w\) (Section 4.5.1) explicitly encodes the digits of \(\hat w,\hat t,\hat z\) and the quotient witnesses’ digits.
- The sumcheck constraints include **range/bitness constraints** for every table entry (Section 4.5.2).

Because the coefficient embedding for Jolt tables is literal coefficient placement (Section 3.2), and coefficients are in \(\{0,1\}\), extraction of a satisfying witness table implies:

- the extracted digits decode to actual bits / small integers,
- recomposition yields the ring elements \(f_{i\parallel j}\) (hence the committed coefficient table),
- and therefore the extractor recovers a valid one-hot/bit polynomial \(P\) consistent with the commitment \(u\).

This is the “strong” part: the protocol does not just prove linear relations; it also proves the **property** that the extracted witness lies in the small domain where the commitment is binding and where “the polynomial table is well-defined”.

In SuperNeo terms, this is exactly why they insist on combining “linear-opening extraction” with an information-theoretic norm/range check inside the same composed argument: it upgrades relaxed/weak extraction into extraction of an actual witness for the intended relation.

#### 7.12.4 Extractor sketch (what is actually recovered)

At a high level, an extractor for Hachi-for-Jolt proceeds like this:

1) Use special soundness of the ring-switching check (Section 7.6) to extract a valid integer witness \((Z,r,s)\) for the lifted identity (or else violate weak binding).

2) Use special soundness of sumcheck (Section 7.7) to extract a witness-table polynomial \(\tilde w\) whose entries satisfy:
   - the linear constraints (encoding the Step-B relation + ring/modulus switching at \(\alpha\)),
   - and the range constraints (digits/bits).

3) Decode the digit rows corresponding to \(\hat w\) (and the implied \(s_i\) blocks) to reconstruct the coefficient table \(P\) (bit/one-hot values under the coefficient embedding).

The quotient witnesses \((r,s)\) need not be unique (Section 7.9), but the reconstructed bit table \(P\) is exactly the semantic object Jolt cares about.

### 7.13 Formal knowledge soundness (SuperNeo “reduction of knowledge” view)

#### 7.13.1 Theorem statement

**Theorem 7.13 (Knowledge soundness of the Hachi-for-Jolt opening reduction).**
Fix parameters \((q,q',d,b,\delta,\tau)\) and the commitment-time ring \(R_q = Z_q[X]/(X^d+1)\), together with the opening protocol described in Sections 4.3–4.5 (and its non-interactive Fiat–Shamir form in Section 7.8).
Assume the following conditions hold:

1) (**Weak binding**) The Ajtai-style inner/outer commitment used to form \(u\) is computationally binding with respect to weak openings (Section 7.3), under the relevant Module/Ring-SIS assumption.

2) (**CWSS extraction for the split-and-fold layer**) The “Figure 3”-type split-and-fold protocol used to relate the commitment opening \((s_i,\hat t_i)_i\) to the folded witness admits a coordinate-wise special soundness extractor that outputs a valid weak opening satisfying the evaluation equation, or else breaks binding / solves the underlying SIS problem.

3) (**Ring+modulus switching extraction**) The ring-switching check (Section 7.6), including the cross-prime modulus-switch slack \(q\cdot s\), admits a \(2d\)-special-soundness extractor for the lifted identity, or else breaks weak binding.

4) (**Sumcheck + range-check extraction**) The sumcheck layer (Section 7.7) for the witness-table polynomial \(\tilde w\), including all range/bitness constraints required to interpret witness-table entries as bounded digits, admits a special-soundness extractor that outputs a valid \(\tilde w\) (and hence a decoded bounded witness satisfying the Step-B relation), or else breaks binding.

5) (**Terminal PCS is a proof of knowledge**) The final opening proof used to justify the terminal claim \(\tilde w(\mathbf r^\*)=v^\*\) (either by recursion or by handoff to a base PCS) is a proof of knowledge for the relation \(t=\mathrm{Com}(\tilde w)\wedge \tilde w(\mathbf r^\*)=v^\*\).

Then the composed opening proof (one or more iterations of Sections 4.3–4.5, followed by the terminal PCS) is a **proof of knowledge** for the PCS opening relation \(R_{\mathrm{JoltPCS}}\) defined in Section 7.13.3.
In particular, there exists an (expected) polynomial-time extractor that, given oracle access to any prover that convinces the verifier with non-negligible probability, outputs:

- a bit/one-hot coefficient table \(P\) consistent with the commitment \(u\), and
- auxiliary witnesses (digits, fold wiring, and quotient slacks),

such that the reconstructed polynomial satisfies the claimed evaluation \(v=\mathrm{Eval}_{F_{q'}}(P,\mathbf r)\).

For the original Hachi setting (single modulus and checks in \(F_{q^k}\)), Conditions (1)–(4) are established by Lemmas 7–11 of `paper/hachi.pdf` (weak binding, CWSS of the split-and-fold layer, and special soundness of ring switching / random evaluation / sumcheck).
For the cross-prime Jolt setting, the only substantive change is Condition (3): the extracted identity must include the additional slack term \(q\cdot s\) (Section 7.6); the special-soundness argument remains a bounded-degree root-counting argument over \(F_{q'}\).

#### 7.13.2 SuperNeo definitions we rely on

We use the following objects from SuperNeo:

- **Reduction of knowledge definition** (Definition 5): an interactive reduction \(\Pi:R_1\to R_2\) is a *reduction of knowledge* if it is complete, public-coin, and knowledge-sound (extractor exists).

```1071:1118:/Users/quang.dao/Documents/SNARKs/hachi/paper/superneo.txt
Definition 5 (Interactive Reductions [50, 52]).Consider relations R1 and
R2 ...
Areduction of knowledge[51] is an interactive reduction,( G,K,P,V ),
that satisfies ...
(ii) Knowledge soundness: ... there exists an
EPT extractor E ...
```

- **Sequential composition** (Lemma 2): composing two reductions of knowledge yields a reduction of knowledge.

```1124:1129:/Users/quang.dao/Documents/SNARKs/hachi/paper/superneo.txt
Lemma 2 (Sequential composition [50,52]).For reductions of knowledge
Π1 ... and Π2 ... we have that Π2 ◦ Π1 ... is a reduction of knowledge ...
```

#### 7.13.3 The PCS opening relation \(R_{\mathrm{JoltPCS}}\) and the reduced relation \(R_{\mathrm{next}}\)

**Relation \(R_{\mathrm{JoltPCS}}\)**:

Instance:

\[
u,\ \mathbf r\in F_{q'}^\ell,\ v\in F_{q'}.
\]

Witness:

\[
P\in\{0,1\}^N\quad\text{(the underlying bit/one-hot coefficient table)}
\]

plus auxiliary variables required by the opening protocol (digits, fold wiring, quotient slacks).

Membership requires:

1) \(u\) is a valid commitment to \(P\) under the commitment construction in Section 4.3.1 (using the coefficient embedding of Section 3.2 for bit/one-hot tables), and
2) \(v = \mathrm{Eval}_{F_{q'}}(P,\mathbf r)\) under the intended evaluation semantics (Section 4.2), and
3) there exist auxiliary witnesses satisfying:
   - the split-and-fold consistency constraints (Section 4.3.5),
   - the ring+modulus switching identity (Section 4.4.2),
   - and the range/bitness constraints on all committed digit entries (Section 4.5.2).

Crucially, because \(P\) is bit/one-hot, the “integer interpretation” and lift are canonical (Section 7.10).

**Relation \(R_{\mathrm{next}}\)**:

Instance:

\[
t,\ \mathbf r^\*\in F_{q'}^{m'},\ v^\*\in F_{q'}.
\]

Witness:

\[
\tilde w \quad\text{(the committed witness-table multilinear from Section 4.5.1)}.
\]

Membership requires:

\[
t = \mathrm{Com}(\tilde w)\quad\text{and}\quad \tilde w(\mathbf r^\*)=v^\*,
\]

with \(\tilde w\) ranging over the same bounded-entry space enforced by the range checks.

#### 7.13.4 The protocol as an interactive reduction \(\Pi_{\mathrm{Hachi}}\)

The “one opening proof step” of Sections 4.3–4.5 can be viewed as an interactive reduction:

\[
\Pi_{\mathrm{Hachi}}:\ R_{\mathrm{JoltPCS}}\ \to\ R_{\mathrm{next}}.
\]

Intuitively:

- the prover’s witness \(w_1\) for \(R_{\mathrm{JoltPCS}}\) includes (at minimum) the bit table \(P\) and the bounded-digit objects \((\hat w,\hat t,\hat z)\) plus quotient witnesses \((r,s)\),
- the interaction checks (via ring switching + sumcheck + range checks) that these objects exist and are consistent,
- and the interaction *outputs* the reduced instance \(u_2=(t,\mathbf r^\*,v^\*)\), i.e. the final witness-table opening claim (Section 4.5.2).

This exactly matches the “interactive reduction outputs a new instance” interface in Definition 5.

Completeness is immediate from the protocol definition: an honest prover with a valid witness can form the required auxiliary witnesses and pass all checks.
Public-coin holds in the interactive form by construction of the verifier challenges; in the non-interactive form it holds in the random-oracle model via Fiat–Shamir (Section 7.8).

#### 7.13.5 Proof of knowledge soundness (extractor construction)

Assume an adversary \((\mathcal A,\mathcal P^\*)\) produces accepting transcripts for \(\Pi_{\mathrm{Hachi}}\) with non-negligible probability.
We construct an extractor \(\mathcal E\) for \(R_{\mathrm{JoltPCS}}\) from such a prover.

Let:

\[
\epsilon := \Pr\big[\text{Verifier accepts the transcript of } \Pi_{\mathrm{Hachi}} \text{ on input }(u,\mathbf r,v)\big].
\]

Assume \(\epsilon \ge 1/\mathrm{poly}(\lambda)\).
The extractor \(\mathcal E\) uses standard rewinding (interactive setting) or random-oracle reprogramming / resampling (Fiat–Shamir setting) to obtain the transcript trees required by the special-soundness and CWSS extractors in Conditions (2)–(4).

The extraction is by composition of the deterministic sub-extractors guaranteed by Conditions (1)–(4) of Theorem 7.13.

**Step A (extract a bounded witness table).**
Use the sumcheck + terminal-opening extractor (Condition (4) together with Condition (5)) to obtain an explicit witness-table multilinear \(\tilde w\) that satisfies:

- all linear constraints encoding the Step-B relation (Section 4.3.5) as checked via ring+modulus switching at \(\alpha\) (Section 4.5.2), and
- all range/bitness constraints on witness-table entries (Section 4.5.2).

If extraction yields two distinct openings for the same commitment during rewinding, Condition (1) (weak binding) yields an SIS solution.

**Step B (extract a weak opening for the original commitment).**
From the extracted \(\tilde w\), decode the prover-side witness values that serve as the final message of the split-and-fold protocol (Section 4.3.5).
Then apply the CWSS extractor for the split-and-fold layer (Condition (2)) to obtain a **weak opening**:

\[
(\bar s_i,\ \bar{\hat t}_i,\ \bar c_i)_{i\in[2^r]}
\]

for the commitment \(u\), together with the evaluation equation relating \((\bar s_i)_i\) to the claimed opening value.
If the CWSS extraction produces two incompatible weak openings for the same \(u\), Condition (1) yields an SIS solution (weak binding).

**Step C (extract quotient witnesses for ring+modulus switching).**
Apply the \(2d\)-special-soundness extractor for ring+modulus switching (Condition (3)) to obtain quotient witnesses \((r,s)\) satisfying the lifted identity as an identity of polynomials (not merely at a single evaluation point).
Non-uniqueness of \((r,s)\) is irrelevant (Section 7.9); extraction only requires outputting *some* bounded-digit solution.

**Step D (decode the underlying coefficient table \(P\)).**
From the extracted weak opening \((\bar s_i)_i\), reconstruct the ring coefficient table and then unpack it to the scalar bit/one-hot table \(P\) (Section 7.13.6).

By construction of decoding and the extracted constraints, the resulting \((P,\ldots)\) satisfies the relation \(R_{\mathrm{JoltPCS}}\).
Furthermore, under the Module/Ring-SIS assumption (Condition (1)), any extraction failure mode that would require producing two distinct valid extracted openings for the same commitment occurs with negligible probability.
Therefore:

\[
\Pr\big[\mathcal E \text{ outputs a witness } (P,\ldots)\text{ such that } (u,\mathbf r,v;P,\ldots)\in R_{\mathrm{JoltPCS}}\big]
\ge
\epsilon - \mathrm{negl}(\lambda),
\]

which is exactly the knowledge soundness requirement of SuperNeo Definition 5 for the interactive reduction \(\Pi_{\mathrm{Hachi}}\).

#### 7.13.6 Explicit decoding procedure \(\mathrm{DecodeOpenings}((\bar s_i)_i)\to P\)

This section specifies the decoding map used in Step D.

Input is a weak opening \((\bar s_i,\bar{\hat t}_i,\bar c_i)_{i\in[2^r]}\) extracted in Step B.
Only the \((\bar s_i)_i\) are used to decode \(P\).

1) **Recompose the ring coefficient table.**
For each outer index \(i\in\{0,1\}^r\), compute:

\[
\bar f_i := G_{2^m}\,\bar s_i \in R_q^{2^m},
\]

and interpret \(\bar f_i = (\bar f_{i\parallel j})_{j\in\{0,1\}^m}\) as the ring coefficient block for that outer index.

2) **Unpack ring coefficients to scalar coefficients.**
Each ring element \(\bar f_{i\parallel j}\in R_q\) is (by coefficient embedding) an encoding of a length-\(d\) field block:

\[
\bar f_{i\parallel j}(X) = \sum_{k=0}^{d-1} p_{i\parallel j\parallel k}\,X^k,\qquad p_{i\parallel j\parallel k}\in F_q.
\]

Define:

\[
\mathrm{Unpack}(\bar f_{i\parallel j}) := (p_{i\parallel j\parallel 0},\dots,p_{i\parallel j\parallel (d-1)})\in F_q^d
\]

to be the coefficient vector of the ring element (well-defined and canonical for bit/one-hot tables).

Finally, define the extracted scalar coefficient table \(P\) by concatenating these blocks in the fixed packing order:

\[
P := \big(p_{i\parallel j\parallel k}\big)_{(i,j,k)\in\{0,1\}^r\times\{0,1\}^m\times[d]} \in \{0,1\}^N.
\]

Here \(N := d\cdot 2^{m+r}\) is the total number of scalar coefficients packed into the ring coefficient table of length \(2^{m+r}\).

The range/bitness constraints guarantee \(p_{i\parallel j\parallel k}\in\{0,1\}\) for the intended one-hot tables.

#### 7.13.7 Sequential composition with the terminal opening proof

\(\Pi_{\mathrm{Hachi}}\) outputs an instance in \(R_{\mathrm{next}}\): an opening claim \(\tilde w(\mathbf r^\*)=v^\*\).
To obtain a *complete* non-interactive opening proof, we must also prove this final opening using some base PCS (or recurse until small enough).

If that terminal opening proof is itself a reduction of knowledge, then by SuperNeo’s sequential composition lemma, the whole composed protocol is a reduction of knowledge (hence a proof of knowledge) for \(R_{\mathrm{JoltPCS}}\).

This is the formal justification for recursion / handoff: each iteration is a reduction of knowledge, and sequential composition preserves knowledge soundness (Lemma 2).


