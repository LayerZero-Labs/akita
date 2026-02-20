# Field embeddings in SuperNeo vs Hachi (math-first notes)

This note focuses **only** on the math in the two papers:

- `docs/superneo.pdf` (“Neo and SuperNeo: Post-quantum folding with pay-per-bit costs over small fields”)
- `paper/hachi.pdf` (“Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments over Extension Fields”)

The shared theme is: lattice commitments naturally live over **cyclotomic rings**, but we want the *interactive proof logic* (sum-check, norm checks, etc.) to live over a **(small) field or a small extension field**. Both works build *embeddings/reductions* that let you:

- commit in a ring/module world (Ajtai/Module-SIS commitments), while
- proving the needed algebraic statements using field arithmetic, and
- keeping norms under control (for binding) and enabling linear-combination/folding operations.

---

## 1) Common background and notation (as used in SuperNeo)

SuperNeo sets up a base field \(F = \mathbb{F}_q\), an extension field \(K/F\) of minimal degree such that \(1/|K| = \mathrm{negl}(\lambda)\), and a cyclotomic ring

\[
R_F := F[X]/(\Phi(X)), \quad R_K := K[X]/(\Phi(X)),
\]

where \(\Phi(X)\) is an \(\eta\)-th cyclotomic polynomial of degree \(d\). It explicitly treats

\[
F \subseteq R_F \subseteq R_K, \qquad F \subseteq K
\]

as nested substructures (SuperNeo, Def. 1; `docs/superneo.pdf`, p. 19–20: “-- 19 of 60 --”, “-- 20 of 60 --”).

Two coefficient maps show up everywhere:

- coefficient vector: \(\mathrm{cf}(a)\in F^d\) for \(a\in R_F\),
- constant term: \(\mathrm{ct}(a)\in F\) for \(a\in R_F\),

and similarly over \(R_K\) (SuperNeo, Def. 2; `docs/superneo.pdf`, p. 20: “-- 20 of 60 --”).

### 1.1 What I mean by the “Gram operator” in this note

Fix:

- an \(F\)-basis \(\{e_0,\dots,e_{d-1}\}\) of the \(F\)-vector space \(R_F\) (most often \(e_i = X^i\), the coefficient basis), and
- a bilinear form \(B: R_F\times R_F \to F\).

Typical examples in these papers are:

- \(B(u,v)=\mathrm{ct}(u\cdot v)\) (SuperNeo’s constant-term functional), possibly with an automorphism inserted, or
- \(B(u,v)=\mathrm{Tr}_H(u\cdot \sigma_{-1}(v))\) (Hachi’s trace-to-subfield functional).

Then the **Gram matrix** of \(B\) in the basis \(\{e_i\}\) is the \(d\times d\) matrix
\[
G_{ij} := B(e_i,e_j).
\]
This matrix encodes the pairing:
if \(u=\sum_i a_i e_i\) and \(v=\sum_j b_j e_j\), then
\[
B(u,v) = a^\top\, G\, b.
\]

The corresponding **Gram operator** is just the linear map \(g: F^d\to F^d\) given by
\[
g(b) := G\,b,
\]
so that \(B(u,v)=a^\top g(b)\).

When SuperNeo says “there exists a linear transform \(T\) such that \(\mathrm{ct}(T(a)\cdot b)=\langle a,b\rangle\)” (Thm. 3),
one way to interpret it is: choose a bilinear form \(B_0(u,v)=\mathrm{ct}(u\cdot v)\) (or a close variant), write down its Gram matrix \(G\) in the coefficient basis, and take \(T = G^{-{\top}}\). This makes the pairing become the standard dot product in coordinates.

---

## 2) What “field embedding” means in SuperNeo (the core problem)

### 2.1 The Ajtai-commitment mismatch

Ajtai/Module-SIS style commitments are *ring-module* commitments:

- Commit to \(z \in R_F^n\) via a linear map \(c = A z\) over \(R_F\).

But CCS witnesses (and CCS arithmetic) are naturally vectors over \(F\). So you need a map:

\[
\iota: F^{N} \longrightarrow R_F^n
\]

that is compatible with:

- **norm constraints** (binding only holds for “small-norm” openings),
- **field constraint checking** via sum-check over \(F\) or \(K\),
- **folding** (random linear combinations of commitments and claims).

SuperNeo frames this as “embed field vectors (CCS witnesses) into the ring vectors that Ajtai commitments operate over” and calls out that the embedding must preserve **norm bounds** and an **evaluation homomorphism** needed for sum-check-based folding (SuperNeo, §1.2; `docs/superneo.pdf`, p. 5–6: “-- 5 of 60 --”, “-- 6 of 60 --”).

### 2.2 What went wrong before (NTT embedding)

Prior lattice folding used an NTT/SIMD isomorphism that maps ring elements into a product of extension fields; this makes field-constraint checking look “ring-native”, but:

- the NTT map is **not norm-preserving**, so small bit-width witnesses become arbitrary-norm ring elements,
- the commitment must then decompose regardless of bit-width ⇒ **no pay-per-bit**,
- packing efficiency is limited by the factor \(t\) in \(F_{q^t}\) (SuperNeo, §1.2.1; `docs/superneo.pdf`, p. 6–7: “-- 6 of 60 --”, “-- 7 of 60 --”).

---

## 3) SuperNeo’s key innovation: *norm-preserving embeddings + evaluation homomorphism*

SuperNeo’s abstract summarizes the core as:

> “two new norm-preserving embeddings of field vectors into ring vectors that respect an evaluation homomorphism required for folding”
(`docs/superneo.pdf`, p. 1: “-- 1 of 60 --”).

The paper describes both a “Neo embedding” (SIMD-friendly) and a “SuperNeo embedding” (general, non-SIMD), and then focuses the rest of the paper on SuperNeo (see `docs/superneo.pdf`, p. 10–11: “-- 10 of 60 --”, “-- 11 of 60 --”).

### 3.1 Neo embedding (high level: “coefficients-as-SIMD lanes”)

Neo’s embedding idea: pack **\(d\) field vectors** \(z^{(1)},\dots,z^{(d)}\in F^n\) into the **coefficient slots** of a ring vector \(z \in R_F^n\), so the coefficient matrix \(\mathrm{cf}(z)\in F^{d\times n}\) literally equals those \(d\) vectors (SuperNeo, §1.2.2 “Contribution 1”; `docs/superneo.pdf`, p. 8–10: “-- 8 of 60 --”, “-- 9 of 60 --”, “-- 10 of 60 --”).

Key consequences:

- **norm-preserving**: small field entries ⇒ small ring coefficients ⇒ binding is aligned with bit-width.
- **optimal SIMD packing**: achieves \(d\cdot n\) field elements per length-\(n\) ring vector (under a SIMD constraint system).
- **evaluation homomorphism for folding**: if you fold commitments with a *short ring* challenge \(\delta\in R_F\), the embedded evaluation claims can be folded consistently by embedding the \(d\) field evaluations as a ring element \(y=\sum_i y^{(i)}X^{i-1}\) (SuperNeo, p. 9: “-- 9 of 60 --”).

But Neo still needs SIMD constraints: “the same constraint system must be applied to all \(d\) underlying field vectors” (SuperNeo, p. 10: “-- 10 of 60 --”).

### 3.2 SuperNeo embedding (formal: coefficient embedding of **one** length-\(d n_R\) vector)

SuperNeo removes SIMD by embedding a **single** long field vector \(z\in F^{n_F}\) where \(n_F = d\cdot n_R\), by chunking it into \(n_R\) blocks of length \(d\), and mapping each block to one ring element’s coefficient vector.

This is defined formally as “Coefficient Embedding” (SuperNeo, §5, Def. 7; `docs/superneo.pdf`, p. 23: “-- 23 of 60 --”):

- element: \(v\in F^d \mapsto \mathbf{v}\in R_F\) with \(\mathrm{cf}(\mathbf{v})=v\)
- vector: \(z\in F^{d n_R}\mapsto \mathbf{z}\in R_F^{n_R}\) by splitting into \(d\)-sized blocks
- matrix: \(M\in F^{m\times d n_R}\mapsto \mathbf{M}\in R_F^{m\times n_R}\) row-wise.

**Why this matters:**

- **optimal packing without SIMD**: you pack \(d\cdot n_R\) field elements into \(n_R\) ring elements.
- **norm-preserving**: the committed object’s coefficients are exactly the witness entries.
- **field-native checks become possible**: you can write constraints directly over the underlying field vector \(z\) and use sum-check over \(F\) or \(K\).

### 3.3 The nontrivial part: lifting *field* products to *ring* products while keeping folding linear

SuperNeo’s obstacle is: commitments/folding live over the ring, but sum-check outputs **field** multilinear evaluation claims like

\[
M z \;\widetilde{}\; (r) \in K
\]

for some random \(r\), and folding wants to take ring-linear combinations \(z'' = z + \delta z'\) with \(\delta \in R_F\) and have the **claims** fold “the same way”.

SuperNeo’s main embedding tool is Section 5 “Embedding products with evaluation homomorphism”:

1. Use a cyclotomic **inner-product automorphism trick** to turn coefficient inner products into ring constant terms.

SuperNeo states an “Inner Product Transform” (Thm. 3; `docs/superneo.pdf`, p. 23: “-- 23 of 60 --”):

> there exists a linear transform \(\bar{\cdot}: F^d \to F^d\) such that for all \(a,b\in F^d\),
> \[
> \mathrm{ct}(\overline{a}\cdot \mathbf{b}) = \langle a,b\rangle.
> \]

Conceptually: cyclotomic rings have many Galois automorphisms (e.g. “conjugation” \(X\mapsto X^{-1}\)), and by applying an appropriate automorphism/linear transform to one operand, the constant coefficient of a ring product recovers a dot product of coefficient vectors. (SuperNeo explicitly attributes this to “(Galois, conjugation, or inner product) automorphism trick” in §1.2.2; `docs/superneo.pdf`, p. 10–11: “-- 10 of 60 --”, “-- 11 of 60 --”.)

2. Extend this transform blockwise to vectors/matrices (Def. 8; `docs/superneo.pdf`, p. 23: “-- 23 of 60 --”).

3. Obtain a **matrix-vector product transform** (Thm. 4; `docs/superneo.pdf`, p. 23–24: “-- 23 of 60 --”, “-- 24 of 60 --”):

> For \(M\in F^{m\times n_F}\), \(z\in F^{n_F}\),
> \[
> M z = \mathrm{ct}(\overline{M}\,\mathbf{z}),
> \]
> i.e. the field product equals the vector of constant terms of a ring product.

4. Lift this to evaluation claims and prove the **evaluation homomorphism** (Thm. 5; `docs/superneo.pdf`, p. 24: “-- 24 of 60 --”):

Roughly: if you linearly combine committed ring vectors with ring scalars \(\rho_i\in R_F\), then the lifted ring-evaluation objects combine linearly as well, and constant terms track the underlying field evaluations.

This is the formal engine that makes “field-native sum-check + ring-linear folding” composable.

### 3.3.1 Explicit inner-product transforms for two cyclotomics you care about

SuperNeo’s Theorem 3 is an *existence* statement: there is a linear map \(T: F^d \to F^d\) (write \(T(a)=\bar a\)) such that for all \(a,b\in F^d\),

\[
\mathrm{ct}(\mathbf{\bar a}\cdot \mathbf{b}) = \langle a,b\rangle
\]

where \(\mathbf{v}\in R_F\) denotes the coefficient embedding of \(v\in F^d\).

Below are **concrete closed forms** for \(T\) in two important special cases.

#### (A) Power-of-two cyclotomic: \(\Phi(X)=X^d+1\) (negacyclic ring)

Let \(R_F = F[X]/(X^d+1)\), and write
\(a(X)=\sum_{i=0}^{d-1} a_i X^i\), \(b(X)=\sum_{i=0}^{d-1} b_i X^i\).

Define \(\bar a(X)\) by the coefficient rule:

- \(\bar a_0 := a_0\)
- for \(i=1,\dots,d-1\): \(\bar a_i := -a_{d-i}\)

Equivalently,
\[
\bar a(X)=a_0 - \sum_{i=1}^{d-1} a_i X^{d-i}.
\]

Then in \(R_F\),
\[
\mathrm{ct}(\bar a(X)\,b(X)) = \sum_{i=0}^{d-1} a_i b_i.
\]

Reason (one-line): the term \((-a_i X^{d-i})(b_i X^i)=-a_i b_i X^d\) contributes \(+a_i b_i\) to the constant term since \(X^d=-1\); other cross-terms cannot reduce to constants without leaving a nonzero power of \(X\).

This is exactly the classical “conjugation/inversion automorphism” trick specialized to \(X^d+1\).

#### (B) Trinomial cyclotomic: \(\Phi_{81}(X)=X^{54}+X^{27}+1\)

Let \(R_F = F[X]/(X^{54}+X^{27}+1)\) (so \(d=54\)). Write
\(a(X)=\sum_{i=0}^{53} a_i X^i\), \(b(X)=\sum_{i=0}^{53} b_i X^i\).

One valid “inner product transform” \(T(a)=\bar a\) with
\(\mathrm{ct}(\bar a(X)\,b(X))=\sum_i a_i b_i\)
is:

- \(\bar a_0 := a_0\)
- for \(i=1,\dots,26\): \(\bar a_i := -(a_{27-i} + a_{54-i})\)
- for \(i=27,\dots,53\): \(\bar a_i := -a_{54-i}\)

(indices are in \(\{0,\dots,53\}\)).

Notable features:

- **extremely sparse**: each output coefficient depends on at most 2 input coefficients,
- **\(O(d)\)** time with only adds + sign flips, matching SuperNeo’s efficiency remark for power-of-two / trinomial cyclotomics.

---

### 3.4 Why this is the “field-embedding innovation” (in one sentence)

SuperNeo’s innovation on the embedding side is:

- **embed a single field witness vector into ring coefficients in a norm-preserving way**, and
- **systematically lift field matrix products/evaluations to ring expressions whose constant terms recover the field values**, so that
- **ring-linear folding preserves the field evaluation claims** (evaluation homomorphism),

thereby enabling a HyperNova-like folding architecture where *sum-check and norm checks run over \(K\)* rather than over the ring, while commitments still live over \(R_F\).

---

## 4) What “field embedding” means in Hachi (the PCS perspective)

Hachi is not a folding scheme; it is a **multilinear PCS**. But the verification bottleneck is similar: classic lattice PCS machinery lives in cyclotomic rings \(R_q = \mathbb{Z}_q[X]/(X^d+1)\), whereas sum-check is naturally over a field.

Hachi’s abstract states its two embedding/reduction ideas (Hachi, Abstract; `paper/hachi.pdf`, p. 1: “-- 1 of 33 --”):

1. **Ring-switching + sum-check**: integrate Greyhound with ring-switching so the verifier avoids ring multiplication.
2. **Generic reduction (extension field → ring)**: convert evaluation proofs over \( \mathbb{F}_{q^k}\) into statements over cyclotomic rings \(R_q\).

### 4.1 Embedding extension fields inside cyclotomic rings (fixed rings under automorphisms)

In Hachi’s technical overview (Hachi, §1.3; `paper/hachi.pdf`, p. 4–5: “-- 4 of 33 --”, “-- 5 of 33 --”), it identifies finite fields \( \mathbb{F}_{q^k}\) *inside* \(R_q\) using fixed subrings under a subgroup of the Galois group:

- Let \(R = \mathbb{Z}[X]/(X^d+1)\) and \(R_q = R/(q)\) with \(d=2^\alpha\).
- For automorphisms \(\sigma_i: X\mapsto X^i\), define the fixed ring
  \[
  R_q^H := \{x\in R_q : \forall \sigma\in H,\;\sigma(x)=x\}.
  \]

Then (Lemma 1, informal; `paper/hachi.pdf`, p. 5: “-- 5 of 33 --”):

> for suitable \(q\) (notably \(q\equiv 5\pmod 8\)) and \(k\mid d/2\), there exists \(H\) such that \(R_q^H\) is a **subfield** of \(R_q\) isomorphic to \(\mathbb{F}_{q^k}\).

This is a literal *field embedding into the ring*: \(\mathbb{F}_{q^k}\hookrightarrow R_q\) realized as a fixed subring.

### 4.2 Inner products via trace + automorphisms (a close cousin of SuperNeo’s transform)

Hachi then uses a trace map \(\mathrm{Tr}_H:R_q\to R_q^H\) and an automorphism (notably \(\sigma_{-1}\)) to turn ring products into **field inner products** (Theorem 1, informal; `paper/hachi.pdf`, p. 5: “-- 5 of 33 --”):

> there exists a bijection \(\psi:(R_q^H)^{d/k}\to R_q\) such that
> \[
> \mathrm{Tr}_H\big(\psi(a)\cdot \sigma_{-1}(\psi(b))\big) = (d/k)\cdot \langle a,b\rangle.
> \]

This is structurally very similar to SuperNeo’s “inner product transform → constant term” idea, except Hachi uses:

- a **trace to a subfield** \(R_q^H \cong \mathbb{F}_{q^k}\),
- whereas SuperNeo phrases it as a **linear transform** on coefficient vectors whose ring-product constant term recovers the dot product.

### 4.3 Ring-switching: from ring equations to extension-field equations (so the verifier stays field-native)

Hachi’s “ring switching and sum-check over extension fields” overview (Hachi, §1.3; `paper/hachi.pdf`, p. 6–7: “-- 6 of 33 --”, “-- 7 of 33 --”) sketches:

- lift a relation over \(R_q\) to an identity over \(\mathbb{Z}_q[X]\) with an explicit multiple of \((X^d+1)\),
- sample \(\alpha \leftarrow \mathbb{F}_{q^k}\) and substitute \(X=\alpha\),
- reducing the ring relation to a **field inner product / sum-check-type claim** over \(\mathbb{F}_{q^k}\),
- then run sum-check over the field and recurse.

This is the PCS analogue of “avoid ring operations during sum-check”: Hachi’s verifier can avoid cyclotomic ring multiplications even though the underlying assumption/commitments are lattice/ring-based.

---

## 5) Same vs different (embedding viewpoint)

### 5.1 The same (high-level mathematical pattern)

- **Cyclotomic rings + automorphisms are the bridge.**
  - SuperNeo: an automorphism/linear transform makes \(\mathrm{ct}(\bar{a}\cdot b)\) become \(\langle a,b\rangle\) (SuperNeo Thm. 3; `docs/superneo.pdf`, p. 23: “-- 23 of 60 --”).
  - Hachi: trace + automorphisms make \(\mathrm{Tr}_H(\psi(a)\cdot \sigma_{-1}(\psi(b)))\) become \(\langle a,b\rangle\) (Hachi Thm. 1; `paper/hachi.pdf`, p. 5: “-- 5 of 33 --”).

- **Field-native sum-check is the goal.**
  - SuperNeo explicitly targets “field-native arithmetic” where “sum-check and norm checks run purely over a small field” (SuperNeo Abstract; `docs/superneo.pdf`, p. 1: “-- 1 of 60 --”).
  - Hachi’s verifier similarly reduces to sum-check over \(\mathbb{F}_{q^k}\) after ring switching (Hachi §1.3; `paper/hachi.pdf`, p. 6–7: “-- 6 of 33 --”, “-- 7 of 33 --”).

- **Linearity matters.**
  Both constructions rely on the fact that the commitment operation is linear in the ring/module, and they build embeddings/reductions so the *claimed evaluations* transform linearly under the same combinations (SuperNeo Thm. 5; `docs/superneo.pdf`, p. 24: “-- 24 of 60 --”).

### 5.2 The different (what each paper is optimizing for)

- **Direction of “embedding”:**
  - **SuperNeo**: embeds *field witnesses* into *ring vectors* to make Ajtai commitments “pay-per-bit” and folding-friendly.
  - **Hachi**: embeds *extension-field evaluation statements* into *ring statements* (and back), to make PCS verification fast via sum-check.

- **Primary object being preserved:**
  - **SuperNeo**: preserves **norm** (for binding/pay-per-bit) *and* preserves **evaluation homomorphism** (for folding).
  - **Hachi**: preserves the **truth of evaluation claims** (over \(\mathbb{F}_{q^k}\)) when translated into ring relations; norm constraints are handled via sum-check after ring switching.

- **Ring family emphasis:**
  - **Hachi**’s core ring is explicitly power-of-two cyclotomic \(X^d+1\) (Hachi Abstract; `paper/hachi.pdf`, p. 1: “-- 1 of 33 --”).
  - **SuperNeo** broadens to more general cyclotomics to support fields like Goldilocks without “full splitting” issues, and explicitly mentions supporting trinomials in parameter sets (SuperNeo intro + parameters discussion; `docs/superneo.pdf`, p. 1–4 and later “concrete parameters” sections).

- **Protocol role:**
  - **SuperNeo** needs an embedding compatible with **folding** (random linear combination of instances/commitments).
  - **Hachi** needs a reduction compatible with **PCS recursion and verifier-time reduction** (ring switching + sum-check).

### 5.3 Are the embeddings “the same”, mathematically?

It depends what you mean by “same”. There are two distinct layers:

- **Layer 1: the raw embedding map(s) \(F^{dn}\to R_F^n\) vs \(F_{q^k}\to R_q\)**  
  These are **not the same functions**.
  - SuperNeo’s core embedding is literally the **coefficient embedding** (Def. 7 in their §5), i.e. *place field coordinates into ring coefficients* (`docs/superneo.pdf`, “Definition 7”; see the quote range in this repo at lines 1222–1238 of the extracted text).
  - Hachi’s key “field embedding” is to realize \(F_{q^k}\) as a **fixed subfield** \(R_q^H \subseteq R_q\), then use a **basis-dependent bijection** \(\psi:(R_q^H)^{d/k}\to R_q\) (`paper/hachi.pdf`, lines 290–300 in the extracted text).

- **Layer 2: the algebraic *mechanism* (automorphisms/trace giving inner products, and linearity of evaluations)**  
  At this layer, they are **the same underlying idea**: both are exploiting a canonical bilinear pairing on cyclotomic rings derived from Galois automorphisms (e.g. \(\sigma_{-1}\)) plus a linear functional (constant term or trace).
  - SuperNeo packages it as “there exists a linear transform \(T\) so that \(\mathrm{ct}(T(a)\cdot b)=\langle a,b\rangle\)” (Thm. 3).
  - Hachi packages it as “there exists a bijection \(\psi\) so that \(\mathrm{Tr}_H(\psi(a)\cdot\sigma_{-1}(\psi(b)))=(d/k)\langle a,b\rangle\)” (Thm. 1).

Concretely: SuperNeo’s \(T\) is (mathematically) the **inverse Gram operator** of the ring’s trace/constant-term pairing *expressed in the coefficient basis*. Hachi’s \(\psi\) is a **choice of basis** that identifies \(R_q\) as a free module over a subfield \(R_q^H\cong F_{q^k}\), so that the same pairing looks like a scaled dot product over that subfield.

So the correct crisp statement is:

- **Not identical as maps** (different domains/codomains and different basis choices), but
- **equivalent in the sense that both instantiate the same cyclotomic “automorphism + linear functional = inner product” backbone**, just presented in different coordinates.

---

## 6) Practical mental model (how to read both papers through one lens)

Both papers can be read as building “interfaces” between three layers:

1. **Ring/module commitment layer** (Ajtai / Module-SIS commitments)
2. **Field arithmetic layer** (sum-check, equality tests, low-degree checks)
3. **Embedding/reduction layer** (the math glue)

SuperNeo’s embedding layer is primarily:

- coefficient embedding \(F^{d n_R} \leftrightarrow R_F^{n_R}\),
- inner-product transform + constant-term extraction,
- evaluation homomorphism for ring-linear combinations.

Hachi’s embedding layer is primarily:

- fixed-field embedding \( \mathbb{F}_{q^k} \cong R_q^H \subseteq R_q\),
- trace + automorphisms for inner products,
- ring-switching by evaluation \(X=\alpha\) to move verifier work into \( \mathbb{F}_{q^k}\).

If you want a single phrase:

> **SuperNeo**: “make folding happen over small fields even though commitments live over rings.”  
> **Hachi**: “make PCS verification happen over extension fields even though commitments live over rings.”


---

## 7) SuperNeo's bilinear form, explicitly identified

SuperNeo's Theorem 3 states existence of a linear transform \(T\) but never identifies it as a Galois automorphism. Here we close that gap.

### 7.1 Claim: \(T = \sigma_{-1}\) for \(\Phi(X) = X^d + 1\)

In \(R_q = F_q[X]/(X^d+1)\), we have \(X^d = -1\), so \(X^{-1} = -X^{d-1}\). The automorphism \(\sigma_{-1}: X \mapsto X^{-1}\) acts on monomials as:

\[
\sigma_{-1}(X^0) = 1, \qquad \sigma_{-1}(X^i) = X^{-i} = -X^{d-i} \;\text{ for } 1 \le i \le d-1.
\]

So on a polynomial \(a(X) = \sum_i a_i X^i\):

\[
\sigma_{-1}(a) = a_0 - \sum_{i=1}^{d-1} a_i\, X^{d-i}.
\]

This is *exactly* the transform \(\bar{a}(X)\) from §3.3.1(A).

### 7.2 Why the Gram matrix is the identity

Consider the bilinear form \(B(a,b) = \mathrm{ct}(\sigma_{-1}(a)\cdot b)\) in the monomial basis \(\{1, X, \dots, X^{d-1}\}\). The Gram matrix is:

\[
G_{ij} = \mathrm{ct}\big(X^{j-i} \bmod (X^d+1)\big).
\]

For \(j = i\): \(\mathrm{ct}(1) = 1\).

For \(j \ne i\): the exponent \(j-i\) satisfies \(-(d-1) \le j-i \le d-1\) and \(j-i \ne 0\).
- If \(j > i\): \(\mathrm{ct}(X^{j-i}) = 0\) since \(1 \le j-i \le d-1\).
- If \(j < i\): \(X^{j-i} = X^{j-i+d}\cdot X^{-d} = -X^{d+j-i}\) where \(1 \le d+j-i \le d-1\), so \(\mathrm{ct}(-X^{d+j-i}) = 0\).

Therefore \(G = I\) (the identity matrix). This means:

\[
\mathrm{ct}(\sigma_{-1}(a)\cdot b) = \mathrm{cf}(a)^\top\, I\, \mathrm{cf}(b) = \langle \mathrm{cf}(a),\,\mathrm{cf}(b)\rangle.
\]

No Gram correction is needed. The pairing \((\sigma_{-1}, \mathrm{ct})\) gives the standard dot product directly.

### 7.3 For the trinomial \(\Phi_{81}(X) = X^{54}+X^{27}+1\), the Gram matrix is NOT the identity

Here \(X^{81} = 1\) but \(X^{27} \ne 1\) in \(R_F\). So \(X^{-27} = X^{54} = -X^{27}-1\), which gives:

\[
\mathrm{ct}(X^{-27}) = \mathrm{ct}(-X^{27}-1) = -1.
\]

The Gram matrix \(G_{ij} = \mathrm{ct}(X^{j-i} \bmod \Phi_{81})\) therefore has off-diagonal entries \(G_{i,\,i+27} = -1\) (and their transposes). The inner-product transform for the trinomial (§3.3.1(B)) is \(T = G^{-1}\circ \sigma_{-1}\), which is why its formula involves sums like \(-(a_{27-i}+a_{54-i})\) rather than a simple sign flip.

### 7.4 Bottom line

SuperNeo's bilinear form is:

\[
B_{\mathrm{SuperNeo}}(a,b) = \mathrm{ct}\big(\sigma_{-1}(a)\cdot b\big) \in F_q.
\]

Hachi's bilinear form is:

\[
B_{\mathrm{Hachi}}(a,b) = \mathrm{Tr}_H\big(a\cdot \sigma_{-1}(b)\big) \in R_q^H \cong \mathbb{F}_{q^k}.
\]

Same involution \(\sigma_{-1}\). Different linear functional (\(\mathrm{ct}\) vs \(\mathrm{Tr}_H\)). Different target (\(F_q\) vs \(\mathbb{F}_{q^k}\)).

---

## 8) Classification of non-degenerate bilinear forms on cyclotomic rings

### 8.1 The general template

For a cyclotomic ring \(R_q = F_q[X]/(\Phi(X))\) of degree \(d\), with Galois group \(\mathrm{Gal} = (\mathbb{Z}/\eta\mathbb{Z})^\times\) acting by \(\sigma_i: X\mapsto X^i\), the general bilinear form from an automorphism + linear functional is:

\[
B_{\sigma,\lambda}(a,b) = \lambda\big(a\cdot \sigma(b)\big)
\]

where \(\sigma \in \mathrm{Gal}\) and \(\lambda: R_q \to T\) is a linear functional into some target \(T\).

### 8.2 When is this non-degenerate? (Frobenius algebra theory)

\(R_q = F_q[X]/(\Phi(X))\) is a **Frobenius algebra** over \(F_q\) (since cyclotomic polynomials are squarefree). This gives a clean classification:

**Theorem (Frobenius classification)**: A linear functional \(\lambda: R_q \to F_q\) makes the bilinear form \(B(a,b) = \lambda(a\cdot b)\) non-degenerate if and only if \(\lambda\) is a **generating functional**, meaning the induced map \(R_q \to R_q^\vee\) (sending \(a\) to the functional \(b\mapsto \lambda(ab)\)) is an isomorphism.

Moreover: **the set of all generating functionals is \(\{\lambda(g\,\cdot\,{-}) : g \in R_q^\times\}\)**. Once you have one non-degenerate form, all others come from pre-multiplying by a unit.

For the form \(B_{\sigma,\lambda}(a,b)=\lambda(a\cdot\sigma(b))\), non-degeneracy reduces to: \(\lambda\circ(\text{mult by }\sigma(\cdot))\) is generating. Since \(\sigma\) is an automorphism (hence maps units to units), \(B_{\sigma,\lambda}\) is non-degenerate iff \(\lambda\) is generating.

### 8.3 Complete menu of linear functionals

The dual space \(R_q^\vee = \mathrm{Hom}_{F_q}(R_q, F_q)\) is \(d\)-dimensional. Here are the "natural" families:

#### (A) Coefficient extraction: \(\mathrm{ct}_j(a) = [X^j](a)\)

Target: \(F_q\). There are \(d\) such functionals (one per coefficient index \(j\)).

\(\mathrm{ct} = \mathrm{ct}_0\) is the constant term. For \(\Phi = X^d+1\) with \(\sigma_{-1}\), the Gram matrix of \(\mathrm{ct}_0\) is the identity (§7.2). Other \(\mathrm{ct}_j\) give non-degenerate forms too, but with permuted/signed Gram matrices.

*Generating?* Yes for many cyclotomics including \(X^d+1\). Can fail for pathological factorizations.

#### (B) Absolute trace: \(\mathrm{Tr}_{R_q/F_q}(a) = \sum_{\sigma\in\mathrm{Gal}} \sigma(a)\)

Target: \(F_q\). This is the canonical choice from algebraic number theory.

*Generating?* Always yes for separable algebras (which cyclotomic rings are).

Relation to \(\mathrm{ct}\): since both are generating, \(\mathrm{Tr}(a) = \mathrm{ct}(g\cdot a)\) for some explicit unit \(g\in R_q^\times\) (related to the "different ideal" of the cyclotomic extension).

#### (C) Relative/partial trace: \(\mathrm{Tr}_H(a) = \sum_{\sigma\in H}\sigma(a) \in R_q^H\)

Target: \(R_q^H \cong \mathbb{F}_{q^k}\) (the fixed subfield under a subgroup \(H \subseteq \mathrm{Gal}\)).

This is what Hachi uses. The key upgrade: the target is an extension field \(\mathbb{F}_{q^k}\), not just \(F_q\). You get an \(\mathbb{F}_{q^k}\)-valued bilinear form, which means inner products live natively in the sumcheck domain.

*Generating?* As a map \(R_q \to R_q^H\), yes: the induced \(R_q^H\)-bilinear form on \(R_q\) (viewed as a free \(R_q^H\)-module of rank \(d/k\)) is non-degenerate.

Trace tower: \(\mathrm{Tr}_{R_q/F_q} = \mathrm{Tr}_{R_q^H/F_q}\circ \mathrm{Tr}_H\). So the relative trace "refines" the absolute trace.

#### (D) Evaluation at a root: \(\mathrm{ev}_\zeta(a) = a(\zeta)\) for \(\zeta\) a root of \(\Phi\) in some \(\mathbb{F}_{q^t}\)

Target: \(\mathbb{F}_{q^t}\).

*Generating?* If \(\Phi\) is irreducible over \(F_q\) (so \(R_q \cong \mathbb{F}_{q^d}\)), every nonzero functional is generating. If \(\Phi\) factors, \(\mathrm{ev}_\zeta\) projects onto one CRT slot and annihilates the others, so it is **not** generating.

**Fatal for norms**: the evaluation map is equivalent to NTT, which is not norm-preserving. This is exactly the problem SuperNeo identifies with prior approaches (§2.2).

#### (E) CRT-slot projection: \(\pi_s: R_q \twoheadrightarrow \mathbb{F}_{q^t}\)

When \(\Phi = \prod_i \phi_i\) factors over \(F_q\) (each \(\phi_i\) irreducible of degree \(t\)), \(R_q \cong \prod_i \mathbb{F}_{q^t}\). The projection \(\pi_s\) onto one factor is:

Target: \(\mathbb{F}_{q^t}\).

*Generating?* **No** --- it annihilates the other factors.

**Same norm problem as (D)**: the CRT isomorphism is an NTT-like map.

#### (F) Arbitrary \(F_q\)-linear combinations of the above

Any \(\lambda = \sum_j c_j \,\mathrm{ct}_j\) for scalars \(c_j \in F_q\) gives a functional. It is generating iff the associated element \(g = \sum_j c_j X^j \in R_q\) is a unit.

### 8.4 Practical takeaway

If you want **norm preservation**, you must stay in "coefficient-land" (functionals A, B, C, F). Evaluation-based functionals (D, E) destroy norms.

If you want **extension-field-valued inner products** (for native sumcheck over \(\mathbb{F}_{q^k}\)), the only natural coefficient-land option is **(C) the relative trace** \(\mathrm{Tr}_H\).

---

## 9) The evaluation homomorphism (SuperNeo Theorem 5), spelled out

This is the key property that makes SuperNeo's folding work. The paper states it abstractly; here is the fully explicit version.

### 9.1 Setup

Fix:
- Base field \(F = \mathbb{F}_q\), extension \(K = \mathbb{F}_{q^2}\) (or any extension with \(1/|K| = \mathrm{negl}(\lambda)\)).
- Cyclotomic ring \(R_F = F[X]/(\Phi(X))\), \(R_K = K[X]/(\Phi(X))\), degree \(d\).
- A CCS matrix \(M \in F^{m\times n_F}\) with \(n_F = d\cdot n_R\).
- Witness vectors \(z_1, \dots, z_\ell \in F^{n_F}\), coefficient-embedded as \(\mathbf{z}_i \in R_F^{n_R}\).
- Folding challenges \(\rho_1, \dots, \rho_\ell \in C \subset R_F\) (short ring elements).
- An evaluation point \(r \in K^{\log m}\) (from a prior sumcheck round).
- Any \(R_F\)-module homomorphisms \(L: R_F^{n_R}\to C\) (commitment) and \(\mathrm{Lin}: R_F^{n_R}\to R_F^{n_{R,\mathrm{in}}}\) (input extraction).

### 9.2 Per-instance "lifted" claims

For each instance \(i\in[\ell]\), define three objects:

- **Commitment**: \(c_i := L(\mathbf{z}_i) \in C\) (Ajtai commitment of the ring vector).
- **Input**: \(x_i := \mathrm{Lin}(\mathbf{z}_i) \in R_F^{n_{R,\mathrm{in}}}\) (public input part).
- **Lifted evaluation**: \(y_i := \overline{M}\,\widetilde{\mathbf{z}_i}(r) \in R_K\) (apply the transformed matrix \(\overline{M}\) to the multilinear extension of the ring vector \(\mathbf{z}_i\), evaluated at \(r\)).

The lifted evaluation \(y_i\) is a **ring element in \(R_K\)** (not a field element in \(K\)). Its constant term recovers the field-level evaluation (by Theorem 4 / Remark 2):

\[
\mathrm{ct}(y_i) = M\,\widetilde{z_i}(r) \in K.
\]

In other words: \(y_i\) has \(d\) coefficients in \(K\); the constant coefficient is the "real" field evaluation; the other \(d-1\) coefficients carry extra information that is needed for folding consistency.

### 9.3 The evaluation homomorphism (Theorem 5)

**Statement**: Define the folded objects:

\[
\mathbf{z} := \sum_{i\in[\ell]} \rho_i\,\mathbf{z}_i \in R_F^{n_R}, \quad
c := \sum_{i\in[\ell]} \rho_i\, c_i \in C, \quad
x := \sum_{i\in[\ell]} \rho_i\, x_i \in R_F^{n_{R,\mathrm{in}}},
\quad
y := \sum_{i\in[\ell]} \rho_i\, y_i \in R_K.
\]

Then **all three of the following hold simultaneously**:

1. **Commitment homomorphism**: \(c = L(\mathbf{z})\).
2. **Lifted-evaluation homomorphism**: \(y = \overline{M}\,\widetilde{\mathbf{z}}(r)\).
3. **Field-evaluation consistency**: \(\mathrm{ct}(y) = M\,\widetilde{z}(r) \in K\) where \(z = \sum_i \rho_i z_i \in F^{n_F}\).

(1) holds because \(L\) is \(R_F\)-linear. (2) holds because multilinear extension is \(K\)-linear and matrix multiplication is \(R_K\)-linear. (3) follows from applying \(\mathrm{ct}\) to (2) and invoking Theorem 4.

### 9.4 Why ring-level claims are essential (the subtle point)

The folding challenge \(\rho_i \in R_F\) is a **ring** scalar, not a field scalar. At the field level, multiplying by \(\rho_i\) acts on the coefficient vector \(z_i \in F^{n_F}\) by **convolution** (polynomial multiplication scrambles coefficients), not by coordinate-wise scaling.

This means:
- **At the field level**: after folding, the individual CCS constraints \(Mz = 0\) do NOT hold in any simple sense. The "meaning" of the folded witness as a CCS satisfier is lost.
- **At the ring level**: the commitment, lifted evaluation claims, and input extraction all fold cleanly. The accumulator stores \((c, x, r, y)\) as ring-level objects.

This is why SuperNeo's accumulator relation is **CE (CCS Evaluation)**, not CCS itself: it stores ring-level evaluation claims \(y_j \in R_K\), and the decider checks them by opening the commitment and verifying ring-level equalities. The \(\mathrm{ct}\) extraction (field-level check) happens only once, at decision time.

### 9.5 Contrast with Hachi

Hachi does not need an "evaluation homomorphism" in SuperNeo's sense because Hachi does not fold multiple instances. Instead:

- Hachi's protocol reduces a *single* opening claim by splitting the coefficient table into blocks, folding with a sparse ring challenge \(c\), and then using ring-switching + sumcheck.
- The analogue of "linearity under ring multiplication" in Hachi is the fold relations (Eq. (18)--(19)): \(a^\top G_{2^m} z = (c^\top\otimes G_1)\hat{w}\) and \(Az = (c^\top\otimes G_{n_A})\hat{t}\), which are linear in the sparse challenge \(c\).

Both papers exploit the same algebraic fact (ring multiplication is \(R_F\)-linear), but for different protocol purposes: folding (SuperNeo) vs opening reduction (Hachi).

---

## 10) Unified embedding framework (best of both worlds)

### 10.1 The core observation

The two bilinear forms use the **same involution** \(\sigma_{-1}\) and differ only in the "readout functional":

| | SuperNeo | Hachi |
|---|---|---|
| Form | \(\mathrm{ct}(\sigma_{-1}(a)\cdot b)\) | \(\mathrm{Tr}_H(\sigma_{-1}(a)\cdot b)\) |
| Target | \(F_q\) | \(\mathbb{F}_{q^k}\) |
| Gram (for \(X^d+1\)) | \(I\) (identity) | Non-trivial (absorbed into \(\psi\)) |

These are **compatible**: they use the same product \(\sigma_{-1}(a)\cdot b\) in \(R_q\) and differ only in what you project onto. The trace tower connects them:

\[
\mathrm{Tr}_{R_q/F_q} = \mathrm{Tr}_{\mathbb{F}_{q^k}/F_q} \circ \mathrm{Tr}_H.
\]

### 10.2 The unified pairing abstraction

**Definition (Cyclotomic Pairing).** Given:
- cyclotomic ring \(R_q\) of degree \(d\),
- a subfield \(S \cong \mathbb{F}_{q^k}\) realized as \(R_q^H \subseteq R_q\) (or \(S = F_q\) as the trivial case \(k=1\)),
- the involution \(\sigma_{-1} \in \mathrm{Aut}(R_q)\),
- the relative trace \(\mathrm{Tr}_H: R_q \to S\),

define the **level-\(k\) pairing**:

\[
B_k(a,b) := \mathrm{Tr}_H\big(\sigma_{-1}(a)\cdot b\big) \in S \cong \mathbb{F}_{q^k}.
\]

Special cases:
- **\(k = 1\)** (\(H = \mathrm{Gal}\), \(S = F_q\)): \(B_1\) is (up to unit scaling) the absolute-trace pairing. Closely related to SuperNeo's \(\mathrm{ct}\)-based form.
- **\(k = d\)** (\(H = \{1\}\), \(S = R_q\)): \(B_d(a,b) = \sigma_{-1}(a)\cdot b\) (the ring product itself, no projection).
- **\(1 < k < d\)** (\(H\) a proper subgroup): Hachi's setting. \(B_k\) is an \(\mathbb{F}_{q^k}\)-valued non-degenerate bilinear form on the free \(\mathbb{F}_{q^k}\)-module \(R_q \cong \mathbb{F}_{q^k}^{d/k}\).

### 10.3 Combined embedding pipeline

```
Field witness z ∈ F_q^{d·n_R}
  │  coefficient embedding (SuperNeo Def. 7, norm-preserving)
  ▼
Ring vector z ∈ R_q^{n_R}       ← ‖z‖_∞ preserved
  │  Ajtai commit: c = Az
  ▼
Commitment c ∈ R_q^κ
  │
  ├── MODE A (SuperNeo-style folding):
  │   Use B_1: ct(σ_{-1}(a)·b) = ⟨cf(a), cf(b)⟩ ∈ F_q
  │   → sumcheck over K = F_{q²}, fold with ρ ∈ R_F, track ring-level claims y ∈ R_K
  │
  └── MODE B (Hachi-style PCS opening):
      Use B_k: Tr_H(σ_{-1}(a)·b) ∈ F_{q^k}
      → ring-switch at α ∈ F_{q^k}, sumcheck over F_{q^k}, recurse on smaller instance
```

The committed object \(\mathbf{z}\in R_q^{n_R}\) is the same in both modes. The choice of functional only affects how you read out inner products and run the interactive proof.

### 10.4 What you gain

| Property | SuperNeo alone | Hachi alone | Unified |
|---|---|---|---|
| Norm-preserving embedding | Yes | N/A (PCS, not CCS) | Yes |
| Pay-per-bit commitments | Yes | No | Yes |
| Extension-field inner products | No (\(\mathrm{ct}\to F_q\)) | Yes (\(\mathrm{Tr}_H\to \mathbb{F}_{q^k}\)) | Yes |
| Sumcheck natively over \(\mathbb{F}_{q^k}\) | No (needs \(K\) externally) | Yes | Yes |
| Power-of-two cyclotomics | Supported | Required | Supported |
| General cyclotomics (trinomials) | Supported | Not discussed | Supported (Gram \(\ne I\)) |

### 10.5 The Gram cost of switching from \(\mathrm{ct}\) to \(\mathrm{Tr}_H\)

For \(\Phi = X^d+1\) with \(\sigma_{-1}\):
- \(\mathrm{ct}\): Gram = \(I\). No correction needed. Beautifully simple.
- \(\mathrm{Tr}_H\): Gram \(\ne I\). Hachi absorbs this into the packing map \(\psi:(R_q^H)^{d/k}\to R_q\) (Theorem 2, Eq. (8)).

If you use \(\mathrm{Tr}_H\) instead of \(\mathrm{ct}\), you need:
- the packing map \(\psi\) (one \(O(d)\) linear map per block),
- its inverse \(\psi^{-1}\) (for verification).

Both are explicit, \(O(d)\)-time, and defined once per parameter set. The Galois subgroup \(H\) and the map \(\psi\) are static public data.

### 10.6 When you would want each mode

- **Folding** (SuperNeo-style IVC) with pay-per-bit: use \(\mathrm{ct}\) mode. The evaluation homomorphism (§9) works cleanly because the Gram is trivial.
- **PCS opening** (Hachi-style recursive reduction): switch to \(\mathrm{Tr}_H\) mode when you need ring-switching at \(\alpha \in \mathbb{F}_{q^k}\).
- **Hybrid fold-then-open**: fold in \(\mathrm{ct}\) mode (accumulating ring-level claims), then at the end, open the accumulated commitment using \(\mathrm{Tr}_H\) mode for efficient PCS verification.

The handoff between modes is seamless because the committed object \(\mathbf{z}\in R_q^{n_R}\) is the same either way.
