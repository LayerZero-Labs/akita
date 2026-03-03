# Hachi with Key Delegation: Two-Level Protocol Spec

This note works out, equation by equation, a **two-level** Hachi protocol where the Level-0 key-evaluation bottleneck is absorbed into the Level-1 opening proof. The goal is to reduce the verifier's dominant cost from \(O(\sqrt{N})\) to \(O(N^{1/4})\).

We work in the **single-field setting** throughout:

- \(q\) is a large prime with \(q \approx 2^{128}\).
- All verifier challenges, sumchecks, and ring-switch checks happen over \(F_q\) (so \(k=1\)).

This choice makes commitments and proofs larger (coefficients are 128-bit), but it avoids all cross-field typing issues and keeps the delegation story purely “within one field”.

---

## 1. Hachi Core Equations Recalled

### 1.1 Setting

- \(R_q = \mathbb{Z}_q[X]/(X^d+1)\) with \(d = 2^\alpha\) (e.g. \(d=1024\)).
- Digit base \(b\) (e.g. \(b=16\)), digit length \(\delta = \lceil \log_b q \rceil\) (e.g. \(\delta = 8\)).
- Gadget matrix \(G_n \in R_q^{n \times n\delta}\) satisfying \(G_n \cdot G_n^{-1}(x) = x\) for any \(x \in R_q^n\), with \(G_n^{-1}(x)\) having coefficients in \(\{0,\ldots,b-1\}\).
- Public Ajtai matrices \(A \in R_q^{n_A \times 2^m\delta}\), \(B \in R_q^{n_B \times n_A\delta \cdot 2^r}\), \(D \in R_q^{n_D \times \delta \cdot 2^r}\).

### 1.2 Step A: Commitment (Eqs. 13–14)

The committed polynomial \(f\) has \(2^\ell\) ring-element coefficients, with \(\ell = m + r\).

Split into \(2^r\) blocks: \(f_i = (f_{i \| j})_{j \in \{0,1\}^m} \in R_q^{2^m}\) for each \(i \in \{0,1\}^r\).

**Digit decomposition** (Eq. 13):
\[
s_i := G_{2^m}^{-1}(f_i) \in R_q^{2^m \delta}, \quad \text{so } G_{2^m} \cdot s_i = f_i.
\]
Each coordinate of \(s_i\) has coefficients in \(\{0,\ldots,b-1\}\).

**Inner commitment** (per block):
\[
t_i := A \cdot s_i \in R_q^{n_A}.
\]

**Digitize the inner commitment**:
\[
\hat{t}_i := G_{n_A}^{-1}(t_i) \in R_q^{n_A \delta}.
\]

**Outer commitment** (Eq. 14):
\[
u := B \cdot [\hat{t}_1; \ldots; \hat{t}_{2^r}] \in R_q^{n_B}.
\]

The commitment is \(u\). The opening witness is \((s_i, \hat{t}_i)_i\).

### 1.3 Step B: Evaluation Reduction (Eqs. 12–20)

**Opening claim**: given commitment \(u\) and point \(x \in R_q^\ell\), prove \(f(x) = v \in R_q\).

#### B.1 Bilinear decomposition (Eq. 12)

Define monomial vectors from the opening point \(x = (x_1, \ldots, x_\ell)\):
\[
b = \big(\textstyle\prod_{j=1}^r x_j^{i_j}\big)_{i \in \{0,1\}^r} \in R_q^{2^r}, \qquad
a = \big(\textstyle\prod_{j=1}^m x_{r+j}^{i_j}\big)_{i \in \{0,1\}^m} \in R_q^{2^m}.
\]

The evaluation factors as:
\[
\boxed{f(x) = \sum_{i \in \{0,1\}^r} b_i \cdot \underbrace{\Big(\sum_{j \in \{0,1\}^m} a_j \cdot f_{i\|j}\Big)}_{w_i} = b^\top w.}
\tag{Eq. 12}
\]

#### B.2 Partial evaluations and auxiliary commitment (Eqs. 15–16)

Define the partial evaluations:
\[
w_i := a^\top \cdot G_{2^m} \cdot s_i = a^\top \cdot f_i \in R_q.
\tag{Eq. 15}
\]

Digitize and commit:
\[
\hat{w}_i := G_1^{-1}(w_i) \in R_q^\delta, \qquad
\hat{w} := (\hat{w}_1, \ldots, \hat{w}_{2^r}) \in R_q^{\delta \cdot 2^r},
\]
\[
v := D \cdot \hat{w} \in R_q^{n_D}.
\tag{Eq. 16}
\]

**Prover sends \(v\)** (first message).

#### B.3 Fold (Eqs. 18–19)

Verifier sends a short/sparse challenge vector \(c = (c_1, \ldots, c_{2^r})\), with \(\|c_i\|_1 \le \omega\).

Prover folds:
\[
z := \sum_{i=1}^{2^r} c_i \cdot s_i \in R_q^{2^m \delta}.
\]

Two linear identities hold:
\[
a^\top G_{2^m}\, z = (c^\top \otimes G_1)\, \hat{w},
\tag{Eq. 18}
\]
\[
A\, z = (c^\top \otimes G_{n_A})\, \hat{t}.
\tag{Eq. 19}
\]

#### B.4 Redecompose and stack (Eq. 20)

The fold increases coefficient magnitudes. Redecompose:
\[
\hat{z} := J_{2^m}^{-1}(z) \in R_q^{2^m \delta \tau}, \qquad z = J_{2^m} \hat{z}.
\]

The prover must prove knowledge of short \((\hat{w}, \hat{t}, \hat{z})\) satisfying a stacked linear system \(M \cdot (\hat{w}, \hat{t}, \hat{z})^\top = (v, u, f(x), 0, 0)^\top\), written row-by-row as the five constraints below (Eq. 20).

The five constraint families:

| # | Name | Equation | Ties... |
|---|------|----------|---------|
| 1 | Aux commitment | \(D\hat{w} = v\) | \(\hat{w}\) to sent \(v\) |
| 2 | Main commitment | \(B\hat{t} = u\) | \(\hat{t}\) to public \(u\) |
| 3 | Opening equation | \(b^\top G \hat{w} = f(x)\) | evaluation to \(\hat{w}\) |
| 4 | Fold consistency (partials) | \((c \otimes G_1)\hat{w} = a^\top G J \hat{z}\) | \(\hat{w}\) to \(\hat{z}\) |
| 5 | Fold consistency (inner) | \((c \otimes G_{n_A})\hat{t} = AJ\hat{z}\) | \(\hat{t}\) to \(\hat{z}\) |

Plus: **range constraints** — all entries of \(\hat{w}, \hat{t}, \hat{z}\) lie in digit range \(\{0,\ldots,b-1\}\).

### 1.4 Step C: Ring Switch + Sumcheck (Eqs. 21–23)

Package the stacked system as \(M Z = Y\) in \(R_q\), where \(Z = (\hat{w}, \hat{t}, \hat{z})\) and \(M\) is the public matrix from Eq. 20.

**Ring switching** (Figure 4): The ring equation \(MZ = Y\) in \(R_q\) is equivalent to existence of a quotient witness \(\rho\) such that:
\[
MZ = Y + (X^d+1)\rho \quad \text{over } \mathbb{Z}_q[X].
\]

The quotient \(\rho\) is digit-decomposed: \(\rho = \sum_u b^u \rho_u\), and each \(\rho_u\) has bounded coefficients.

**Witness table** \(\tilde{w}\) (Eq. 21): Encode the coefficient tables of \(Z\) and \((\rho_u)_u\) as a single multilinear polynomial. The table has:
- \(\mu\) rows from the stacked witness \(Z = (\hat{w}, \hat{t}, \hat{z})\),
- \(n \cdot \delta_\rho\) rows from the quotient digit vectors,
- \(d\) columns (one per ring-element coefficient).

**Sample** \(\alpha \leftarrow F_q\). Evaluate the ring-switch identity at \(X = \alpha\).

**Sumcheck** over \(F_q\): Batch the linear constraint \(H_\alpha\) and the range constraint \(H_0\) into two sumcheck claims (Eqs. 22–23), then prove via sumcheck. After all rounds, the sumcheck reduces to:

\[
\boxed{\text{Open } \tilde{w} \text{ at random point } r^* \in F_q^{m'},\quad \tilde{w}(r^*) = v^*.}
\]

**This is the recursion boundary**: the output is a new, smaller opening claim of the same shape.

### 1.5 Where the Verifier's Bottleneck Arises

At the end of sumcheck, the verifier must evaluate the **public encodings** \(e_{M_\alpha}(i, u)\) at the final random point \(r^*\). These encodings include:

- **Structured parts** (cheap): \(e_\alpha(\ell) = \alpha^\ell\) (size \(d\), cost \(O(d)\)); equality polynomial weights (cost \(O(\log n)\)); gadget/fold challenge weights (cost \(O(1)\) per entry); monomial vectors \(\tilde{a}(r^*), \tilde{b}(r^*)\) (closed-form, cost \(O(m+r)\)).

- **Unstructured parts** (expensive): The entries of the Ajtai matrices \(A, B, D\) evaluated at \(\alpha\). These are pseudorandom ring elements; their MLE at \(r^*\) costs:
\[
\boxed{O(\mu) = O(\delta \cdot 2^m + n_A \delta \cdot 2^r + 2^m \delta \tau) \approx O(\delta \cdot 2^{\ell/2})}
\]
field operations. This is the **square-root verifier bottleneck**.

---

## 2. Level 0: End-to-End Trace

### 2.1 Input

- Committed polynomial \(P\) with \(N = d \cdot 2^\ell\) scalar coefficients (equivalently \(2^\ell\) ring elements).
- Commitment \(u_0 \in R_q^{n_B}\).
- Opening point \(x_0 \in R_q^\ell\) (or \((R_q^H)^\ell \subset R_q^\ell\)).
- Claimed value \(v_0 \in R_q\).

### 2.2 Level-0 parameters

Split: \(\ell_0 = m_0 + r_0\) with \(m_0 \approx r_0\).

Public matrices:
\[
A_0 \in R_q^{n_A \times 2^{m_0}\delta}, \quad
B_0 \in R_q^{n_B \times n_A\delta \cdot 2^{r_0}}, \quad
D_0 \in R_q^{n_D \times \delta \cdot 2^{r_0}}.
\]

### 2.3 Level-0 protocol execution

1. **Step A**: Commitment \(u_0\) already computed.

2. **Step B**: Prover computes partial evaluations \(w_i\), sends aux commitment \(v_0 = D_0 \hat{w}_0\). Receives fold challenge \(c_0\). Folds: \(z_0 = \sum c_{0,i} s_{0,i}\). Redecomposes: \(\hat{z}_0 = J^{-1}(z_0)\).

3. **Step C**: Prover commits to the witness table \(\tilde{w}_0\) encoding \((\hat{w}_0, \hat{t}_0, \hat{z}_0, \text{quotient digits})\).

   Commitment to \(\tilde{w}_0\):
   \[
   t_{\tilde{w}_0} := \text{Com}(\tilde{w}_0) \in R_q^{n_B}.
   \]
   (This is a fresh Hachi commitment to the witness table, using its own Ajtai matrices \(A_1', B_1'\).)

   Ring-switch at \(\alpha_0\), then sumcheck over \(F_q\).

### 2.4 Level-0 output

After sumcheck, the verifier holds:

- **Claim 1** (standard recursion): \(\tilde{w}_0(r_1^*) = v_1^* \in F_q\), where \(r_1^* \in F_q^{m_0'}\) is the sumcheck's random point and \(v_1^*\) is the claimed value.

  The verifier has checked the sumcheck rounds, so this claim is all that remains of the original opening.

- **Claim 2** (key evaluation bottleneck): To finish verifying the sumcheck's final identity, the verifier needs:
  \[
  V_{\text{key}} := \sum_{u=1}^{\mu_0} \text{eq}(r_1^*, u) \cdot e_{M_0}(i^*, u)(\alpha_0),
  \]
  where \(e_{M_0}\) encodes the Level-0 Ajtai matrices \(A_0, B_0, D_0\) (after ring switching at \(\alpha_0\)), and \(i^*\) is the batched row index from the equality-polynomial batching.

  Cost to compute \(V_{\text{key}}\) directly: \(O(\mu_0) \approx O(\delta \cdot 2^{m_0})\) field operations.

The verifier **defers** Claim 2. Instead, the prover sends the claimed value \(V_{\text{key}}\), and the verifier defers checking it by reducing it to a single opening claim against a *precommitted, \(\alpha\)-independent* key table (Section 3.2–3.3). That resulting opening claim is then resolved at Level 1.

---

## 3. Level 1 without homomorphic commitments

### 3.1 What we have after Level 0

Two pending claims:

| Claim | Object | Point | Value | Object size |
|-------|--------|-------|-------|-------------|
| 1 (witness) | \(\tilde{w}_0\) (committed) | \(r_1^* \in F_q^{m_0'}\) | \(v_1^*\) | \(\mu_0' \cdot d\) scalars |
| 2 (key) | \(\mathsf{KeyBase}_0\) (precommitted base table) | \(\rho_1^*\) (key-reduction point) | \(k_1^* \in R_q\) | \(\mu_0^{\text{key}} \cdot d\) scalars |

**Key sizing observation**: Both objects have size on the order of \(\sqrt{N}\).

- \(\tilde{w}_0\) has \(\mu_0' = \mu_0 + n_0 \delta_\rho\) rows, each with \(d\) coefficients. Total: \(\mu_0' \cdot d \approx O(\delta \cdot 2^{m_0} \cdot d)\).
- \(\mathsf{KeyBase}_0\) is an \(\alpha\)-independent commitment-time representation of the Level-0 public matrices (e.g. flattened \(A_0,B_0,D_0\), or a smaller batched slice). Total size is the same \(\Theta(\sqrt N)\) order.

So both have \(\Theta(\delta \cdot 2^{m_0} \cdot d)\) scalar entries. They are the **same order of magnitude**.

### 3.2 Preprocessing: commit to an \(\alpha\)-independent base key table

During setup, we commit to an \(\alpha\)-independent **base key table** that does *not* depend on any runtime Fiat–Shamir challenges (in particular, it cannot depend on \(\alpha_0\) or on transcript-derived row-batching points like \(i^*\)).

Concretely, define \(\mathsf{KeyBase}_0\) as the multilinear polynomial whose coefficient table is the flattened list of ring elements in \(A_0,B_0,D_0\) (or whatever smaller subset suffices for the verifier’s final public-encoding check). For the maximal/simple choice, there are:
\[
\mu_0^{\text{key}} = \underbrace{2^{m_0}\delta}_{A_0\text{ width}} + \underbrace{n_A \delta \cdot 2^{r_0}}_{B_0\text{ width}} + \underbrace{\delta \cdot 2^{r_0}}_{D_0\text{ width}}
\]
ring elements (each with \(d\) coefficients), giving \(\mu_0^{\text{key}} \cdot d\) scalar entries total.

The setup computes a Hachi commitment to this table (using the **Level-1** commitment parameters; see Section 6.1):
\[
u_{\text{key}} := \text{HachiCommit}(\mathsf{KeyBase}_0) \in R_q^{n_B}.
\]

This \(u_{\text{key}}\) goes into the **verification key**.

### 3.3 Reducing the runtime key evaluation to one opening claim

The runtime quantity \(V_{\text{key}}\) depends on verifier challenges like \(\alpha_0\) and on transcript-derived batching indices. Therefore, it is generally **not** the evaluation of an \(\alpha\)-independent precommitted polynomial at a fixed point.

Instead, we apply a lightweight claim-reduction step (Jolt-style) that reduces correctness of \(V_{\text{key}}\) to a **single opening claim** against the precommitted base table \(\mathsf{KeyBase}_0\):

\[
\boxed{\mathsf{KeyBase}_0(\rho_1^*) = k_1^*}
\]

for a verifier-sampled random point \(\rho_1^*\) (over the indexing variables of \(\mathsf{KeyBase}_0\)), and a prover-sent value \(k_1^*\).

Intuitively: the verifier’s needed key evaluation is a fixed linear functional of the base table once \((r_1^*,\alpha_0,i^*,\ldots)\) are fixed; a short sumcheck/claim-reduction turns that linear functional check into one random-point evaluation check.

If desired, a second claim-reduction layer can make \(\rho_1^*\) match \(r_1^*\) (same point) so that later batching/stacking is slightly simpler. This is an optimization, not a requirement.

### 3.4 No commitment homomorphism: prove both openings inside one Level-1 proof

The earlier “homomorphic batching” idea (forming \(u_{\text{combined}} = t_{\tilde w_0} + \eta u_{\text{key}}\)) is **not valid** for Hachi commitments, because the commitment map includes digit decompositions and is not linear.

Instead, Level 1 proves **both** opening claims simultaneously, without combining commitments:

\[
\boxed{
\text{Open }\tilde w_0(r_1^*) = v_1^*
\quad\text{and}\quad
\mathsf{KeyBase}_0(\rho_1^*) = k_1^*,
\text{ given commitments } t_{\tilde w_0}, u_{\text{key}}.
}
\]

This can be done with a single “stacked” Hachi instance:

- Run Hachi Step B (evaluation reduction) for \(\tilde w_0\) at \(r_1^*\) and for \(\mathsf{KeyBase}_0\) at \(\rho_1^*\).
- Form the corresponding stacked linear systems (paper Eq. 20 shape) for each, and **stack** them into one larger linear system.
- Commit once to a single Level-1 witness table \(\tilde w_1\) that encodes the digits/quotients for *both* sub-instances.
- Run one ring-switch + sumcheck to reduce everything to one recursive opening claim about \(\tilde w_1\).

The verifier’s work stays \(O(1)\) in the number of deferred claims (constant many), and the “delegated key evaluation” is now checked by opening \(\mathsf{KeyBase}_0\) under the Level-1 commitment scheme — no homomorphism required.

---

## 4. Level 1: Hachi on a stacked relation (two openings, one proof)

### 4.1 Level-1 parameters

Level 1 must open two committed multilinears (not necessarily at the same point):

- \(\tilde w_0\) (the Level-0 witness table polynomial), committed as \(t_{\tilde w_0}\).
- \(\mathsf{KeyBase}_0\) (the \(\alpha\)-independent key base table), committed as \(u_{\text{key}}\).

Let their (ring-element) lengths be \(L_w, L_k\) (each \(\approx \Theta(\sqrt N)\)). Pad each to a power of two and choose a common split \(\ell_1=m_1+r_1\) that supports both (or use separate splits and stack; common split is just simpler to describe).

Split: \(\ell_1 = m_1 + r_1\) with \(m_1 \approx r_1 \approx \ell_1 / 2\).

Level-1 Ajtai matrices (public):
\[
A_1 \in R_q^{n_A \times 2^{m_1}\delta}, \quad
B_1 \in R_q^{n_B \times n_A\delta \cdot 2^{r_1}}, \quad
D_1 \in R_q^{n_D \times \delta \cdot 2^{r_1}}.
\]

### 4.2 Step A (Level 1): Inputs are two commitments

There is no “combined commitment”. The Level-1 verifier input includes both commitments:

- \(t_{\tilde w_0} \in R_q^{n_B}\)
- \(u_{\text{key}} \in R_q^{n_B}\)

Both must be commitments under the **same Level-1 commitment parameters** (i.e. derived from the same Level-1 public seed and matrix shapes), but they remain separate commitments.

### 4.3 Step B (Level 1): Evaluation reduction (two copies, stacked)

Apply Hachi’s Step B equations twice (with potentially different opening points):

- Once to \(\tilde w_0\) at point \(r_1^*\) (with commitment RHS \(t_{\tilde w_0}\) and claimed value \(v_1^*\)).
- Once to \(\mathsf{KeyBase}_0\) at point \(\rho_1^*\) (with commitment RHS \(u_{\text{key}}\) and claimed value \(k_1^*\)).

Each application produces its own Step-B witness objects \((\hat w,\hat t,\hat z)\) and its own five constraint families (Eq. 20 shape). For clarity, denote these with superscripts \((w)\) and \((k)\).

Digit decomposition and range constraints are identical in both copies: all digit vectors lie in \(\{0,\ldots,b-1\}\), regardless of whether the underlying coefficients were “small” (digits) or “pseudorandom” (key entries modulo \(q\)).

Monomial vectors (from \(r_1^*\)) live in \(F_q\):
\[
b_1 \in F_q^{2^{r_1}}, \qquad a_1 \in F_q^{2^{m_1}}.
\]

Each copy also has its own aux commitment (to its partial-eval digits), its own fold challenge, and its own redecomposition witness. These can be sampled with independent transcript labels (recommended), or shared if you are willing to analyze correlated challenges.

### 4.4 Stack the two Eq. 20 systems into one

Let the two Step-B outputs yield two ring-linear systems:

\[
M^{(w)} Z^{(w)} = Y^{(w)}
\qquad\text{and}\qquad
M^{(k)} Z^{(k)} = Y^{(k)}
\quad\text{in }R_q.
\]

Define the stacked system (block diagonal, conceptually):
\[
M^{(\mathrm{tot})} :=
\begin{bmatrix}
M^{(w)} & 0 \\
0 & M^{(k)}
\end{bmatrix},\quad
Z^{(\mathrm{tot})} := (Z^{(w)}, Z^{(k)}),\quad
Y^{(\mathrm{tot})} := (Y^{(w)}, Y^{(k)}).
\]

The public right-hand side \(Y^{(\mathrm{tot})}\) contains *both* commitments and *both* claimed values:

- one commitment-consistency constraint uses RHS \(t_{\tilde w_0}\),
- the other uses RHS \(u_{\text{key}}\),
- one opening equation uses RHS \(v_1^*\),
- the other uses RHS \(V_{\text{key}}\).

All range constraints are appended for both sub-instances’ digit witnesses.

### 4.5 Step C (Level 1): One ring switch + one sumcheck

Package the stacked system as \(M^{(\mathrm{tot})} Z^{(\mathrm{tot})} = Y^{(\mathrm{tot})}\) in \(R_q\).

Ring switch (single \(\alpha_1 \leftarrow F_q\)): prove existence of quotient witness \(\rho^{(\mathrm{tot})}\) such that:
\[
M^{(\mathrm{tot})} Z^{(\mathrm{tot})}
=
Y^{(\mathrm{tot})} + (X^d+1)\rho^{(\mathrm{tot})}
\quad\text{over } \mathbb{Z}_q[X].
\]

Commit to a single Level-1 witness table \(\tilde w_1\) encoding:

- both sub-instances’ digit witnesses \((\hat w,\hat t,\hat z)\),
- both quotient-digit witnesses (or one concatenated quotient-digit block),
- all range-check witness data.

Run one sumcheck over \(F_q\) that batches the linear constraints and the range constraints exactly as in standard Hachi, but over the stacked table.

### 4.6 Level-1 output

After sumcheck, the verifier holds:

- **Claim 3** (stacked witness): \(\tilde{w}_1(r_2^*) = v_2^*\).
- **Pending Level-1 key evaluation**: the verifier still must evaluate the Level-1 public encodings (derived from \(A_1,B_1,D_1\)) at the final point. This is the new bottleneck, of size \(\approx N^{1/4}\).

The Level-1 key has width:
\[
\mu_1 = O(\delta \cdot 2^{m_1}).
\]

Since \(2^{m_1} \approx \sqrt{2\mu_0} \approx \sqrt{\mu_0}\), the Level-1 key is:
\[
\mu_1 \approx \delta \cdot \sqrt{\mu_0} \approx \delta \cdot \sqrt{\delta \cdot 2^{m_0}} = \delta^{3/2} \cdot 2^{m_0/2}.
\]

This is \(\approx N^{1/4}\) (up to \(\delta\) factors), confirming the fourth-root reduction. Stacking two openings increases constants but does not change the exponent: the Level-1 witness/key widths remain \(\tilde O(\sqrt{\mu_0})\).

---

## 5. Size Analysis (single-field \(q \approx 2^{128}\))

### 5.1 Concrete parameters

Use the benchmark regime: \(N = 2^{38}\), \(d = 1024\), \(q \approx 2^{128}\), \(b = 16\), \(\delta = \lceil \log_{16} q\rceil \approx 32\), \(\tau = 4\), \(n_A = n_B = n_D = 1\).

### 5.2 Level 0

\(\ell_0 = 38 - 10 = 28\) (ring-level variables). Split: \(m_0 = r_0 = 14\).

**Witness table \(\tilde{w}_0\)**:
- \(\hat{w}_0\): \(\delta \cdot 2^{r_0} \approx 32 \cdot 2^{14} = 2^{19}\) ring elements.
- \(\hat{t}_0\): \(n_A \delta \cdot 2^{r_0} \approx 32 \cdot 2^{14} = 2^{19}\) ring elements.
- \(\hat{z}_0\): \(2^{m_0} \delta \tau \approx 2^{14} \cdot 32 \cdot 4 = 2^{21}\) ring elements.
- Quotient digits: \(\approx 5 \cdot \delta_\rho \approx 5 \cdot 11 = 55\) ring elements (negligible).
- **Total \(\mu_0' \approx 2^{19} + 2^{19} + 2^{21} \approx 2^{21.6}\)** ring elements.
- Scalar entries: \(\mu_0' \cdot d \approx 2^{21.6} \cdot 2^{10} = 2^{31.6}\), padded to \(\approx 2^{32}\).

**Level-0 key** (Ajtai matrices \(A_0, B_0, D_0\)):
- \(A_0\) width: \(2^{m_0} \delta \approx 2^{14} \cdot 32 = 2^{19}\) ring elements.
- \(B_0\) width: \(n_A \delta \cdot 2^{r_0} \approx 2^{19}\) ring elements.
- \(D_0\) width: \(\delta \cdot 2^{r_0} \approx 2^{19}\) ring elements.
- Heights: all \(n_A = n_B = n_D = 1\).
- **Total key entries: \(\mu_0^{\text{key}} \approx 3 \cdot 2^{19} \approx 2^{20.6}\)** ring elements.
- Scalar entries: \(\mu_0^{\text{key}} \cdot d \approx 2^{20.6} \cdot 2^{10} = 2^{30.6}\).

**Verifier bottleneck** (without delegation): \(O(\mu_0^{\text{key}}) \approx O(2^{20.6})\) field ops.

### 5.3 Level 1 (stacked openings)

At Level 1 we prove two openings (potentially at different points):

- open \(\tilde w_0\) at \(r_1^*\) (size \(\mu_0'\) ring elements),
- open \(\mathsf{KeyBase}_0\) at \(\rho_1^*\) (size \(\mu_0^{\text{key}}\) ring elements for the maximal choice).

Choose \(\ell_1 \approx \lceil \log_2(\max(\mu_0', \mu_0^{\text{key}}))\rceil\). With the above numbers, \(\max(\mu_0',\mu_0^{\text{key}})\approx 2^{21.6}\), so \(\ell_1 \approx 22\) and a balanced split suggests \(m_1=r_1\approx 11\).

For one sub-instance at \((m_1,r_1)\approx(11,11)\), the Level-1 witness-table row count is on the order of:

- \(\hat w_1\): \(\delta \cdot 2^{r_1} \approx 32 \cdot 2^{11} = 2^{16}\),
- \(\hat t_1\): \(n_A\delta \cdot 2^{r_1} \approx 2^{16}\),
- \(\hat z_1\): \(2^{m_1}\delta\tau \approx 2^{11}\cdot 32\cdot 4 = 2^{18}\),

so per sub-instance \(\mu_{1,\text{sub}}' \approx 2^{18.6}\) ring elements. Since Level 1 stacks two such sub-instances, the stacked witness table satisfies:

\[
\mu_1' \approx 2 \cdot \mu_{1,\text{sub}}' \approx 2^{19.6}\ \text{ring elements.}
\]

**Level-1 key** (Ajtai matrices \(A_1, B_1, D_1\)):
- \(A_1\) width: \(2^{m_1}\delta \approx 2^{11}\cdot 32 = 2^{16}\) ring elements.
- \(B_1\) width: \(\approx 2^{16}\). \(D_1\) width: \(\approx 2^{16}\).
- **Total key: \(\mu_1^{\text{key}} \approx 3 \cdot 2^{16} \approx 2^{17.6}\)** ring elements.

**Verifier bottleneck after Level 1**: \(O(\mu_1^{\text{key}})\) field ops (a constant-factor larger due to stacking, but the same exponent).

### 5.4 Level 2 (continued delegation)

After Level 1, the same pattern repeats: the verifier has one recursive witness-table opening claim (for \(\tilde w_1\)) and one deferred Level-1 key-encoding claim. Level 2 stacks those two openings and proves them together.

(The exponent drop continues exactly as before; constants depend on \(\delta,\tau\), and the fact that we always stack two sub-instances.)

### 5.5 Summary (shape only)

At each level \(t\), the verifier’s dominant “public encoding evaluation” work is \(O(\mu_t^{\text{key}})\), and delegation reduces \(\mu_t^{\text{key}}\) roughly by a square root each time (up to \(\delta,\tau\) factors and a constant-factor overhead from stacking two sub-instances).

### 5.6 Communication overhead of delegation (vs no delegation)

**Without delegation**: Level 1 proves only the opening for \(\tilde w_0\).

**With delegation (this note)**: Level 1 proves a *stacked* opening: one sub-instance for \(\tilde w_0\) and one sub-instance for \(\mathsf{KeyEnc}_0\). This increases the Level-1 witness table size by roughly an additive \(\approx |\mathsf{KeyEnc}_0|\) term (same order as \(|\tilde w_0|\)).

Overhead sources:

- The prover must send \(V_{\text{key}}\) (one \(F_q\) element).
- The proof includes a short **key-claim reduction transcript** (log-sized) that reduces checking \(V_{\text{key}}\) to the single opening claim \(\mathsf{KeyBase}_0(\rho_1^*)=k_1^*\).
- The Level-1 proof is larger because the stacked witness table is larger (potentially +0–1 sumcheck rounds after padding, plus larger per-round univariates if you implement “one sumcheck over one stacked table”).

The important point is that proof growth is **polylog / constant-factor**, while verifier key-evaluation work drops by an exponent (square-root \(\to\) fourth-root).

### 5.7 Proof size breakdown

Per-level proof consists of:

- a constant number of ring commitments (note Level 1 has *two* aux commitments \(v_1^{(w)},v_1^{(k)}\) plus one stacked-table commitment \(t_{\tilde w_1}\)),
- a sumcheck transcript over \(F_q\),
- a constant number of deferred key-evaluation scalars (one per level).

Exact byte counts depend heavily on \(d\), \(q\), and the univariate encoding used in sumcheck. Since this note fixes \(q\approx 2^{128}\), ring elements are substantially larger than in small-\(q\) benchmark regimes; treat any small-\(q\) byte tables as non-applicable here.

**Accumulated proof size vs verifier improvement**:

Delegation trades **larger proofs** for reduced verifier “public encoding evaluation” work. The reduction is by an exponent (square-root per delegation level), while proof growth is by constant factors per level (stacking adds constants).

Each extra level reduces verifier key-evaluation work by roughly \(\approx 4\times\) in the balanced regime (up to \(\delta,\tau\) factors).

---

## 6. Potential Issues and Resolutions

### 6.1 Commitment-parameter compatibility (still required, but no homomorphism)

Even though we do **not** use homomorphic batching, we do require that both commitments \(t_{\tilde w_0}\) and \(u_{\text{key}}\) are commitments under the **same Level-1 commitment parameters**:

- same ring \(R_q\),
- same \((m_1,r_1)\) block layout / widths,
- same public matrix seed (so the verifier knows which \(A_1,B_1\) define the commitment scheme).

This is a standard recursion requirement: \(t_{\tilde w_0}\) is produced online during Level 0 using the Level-1 commitment setup, and \(u_{\text{key}}\) is produced during preprocessing using that *same* Level-1 setup. Since all Level sizes are deterministic functions of the public parameters, the setup can precompute \(u_{\text{key}}\) consistently.

### 6.2 Different number of variables

The witness table \(\tilde{w}_0\) has \(\mu_0'\) rows and the key table has \(\mu_0^{\text{key}}\) rows. These are generally different, so the two multilinear polynomials have different numbers of variables.

**Resolution**: Pad both tables to compatible power-of-two lengths and either:

- choose a common \((m_1,r_1)\) and run two Step-B reductions of that same shape, or
- allow different shapes and stack the resulting systems (the stacked witness table just has two differently-indexed regions).

In both cases, the final “open at \(r_1^*\)” statements must refer to the same MLE domain.

### 6.3 Coefficient magnitudes (key entries are not digits)

The key-encoding table coefficients are arbitrary mod-\(q\) values (pseudorandom-looking ring elements), not small digits.

This is fine: in the Step-B reduction for \(\mathsf{KeyEnc}_0\), the prover digit-decomposes these values with \(G^{-1}\) exactly as usual, producing \(\delta\) base-\(b\) digits in \(\{0,\ldots,b-1\}\). So the same range-constraint machinery applies.

### 6.4 Soundness of stacking (no new Schwartz–Zippel term needed)

There is no separate “batching challenge” \(\eta\) at the commitment level. Soundness is inherited from:

- Hachi’s knowledge soundness for each sub-instance’s Step-B+Step-C reduction, and
- the fact that the stacked system is just the conjunction of two relations, proven together by the same ring-switch + sumcheck machinery.

If you additionally use a lightweight claim-reduction sumcheck to align evaluation points/indices (Section 3.3), that introduces the usual Schwartz–Zippel soundness term over \(F_q\), which is negligible for \(q \approx 2^{128}\).

### 6.5 The key-claim point is shared (or reduced to be shared)

The witness opening claim uses the Level-0 sumcheck’s final point \(r_1^*\).

The key-related claim-reduction produces an opening point \(\rho_1^*\) for \(\mathsf{KeyBase}_0\). In favorable packings, \(\rho_1^*\) can be chosen to match \(r_1^*\) (same point). In general, \(\rho_1^*\) may live in a different variable set and thus be different; Level 1 handles this by simply running two Step-B reductions with their respective points, then stacking the resulting systems.

### 6.6 Binding across levels

The security argument for the two-level protocol composes two Hachi instances:

1. **Level 0** is a reduction of knowledge: it reduces the opening claim \((u_0, x_0, v_0)\) to the pair of claims (Claim 1 + Claim 2). Knowledge soundness follows from Hachi's Lemmas 7–11.

2. **Level 1** proves the conjunction of the two deferred claims by a single stacked Hachi instance. Knowledge soundness follows from Hachi's lemmas applied to each sub-instance plus the soundness of the shared sumcheck/ring-switch layer over the stacked relation.

By sequential composition (SuperNeo Lemma 2), the composed protocol is a reduction of knowledge for the original opening relation.

The only new ingredient is the preprocessing commitment \(u_{\text{key}}\) to \(\mathsf{KeyEnc}_0\). This commitment is part of the public parameters. Its binding under Module-SIS ensures that the prover cannot substitute a different key encoding without breaking the underlying assumption.

---

## 7. Summary: The Two-Level Protocol

### Message flow (Fiat–Shamir view)

```
PREPROCESSING:
  Setup commits to Level-0 key base table: u_key = HachiCommit(KeyBase₀)   (under Level-1 commitment params)
  Verification key includes: u_key, Level-1 commitment seed/params

LEVEL 0 (standard Hachi):
  P → V:  v₀ (aux commitment)
  V → P:  c₀ (fold challenge)
  P → V:  t_{w̃₀} (witness-table commitment)
  V → P:  α₀ (ring-switch challenge)
  P ↔ V:  sumcheck rounds
  P → V:  V_key (claimed key evaluation)
  (Optional/needed) P ↔ V: key-claim reduction transcript, producing (ρ₁*, k₁*)

LEVEL 1 (one stacked proof for two openings):
  Goal: prove w̃₀(r₁*) = v₁* and KeyBase₀(ρ₁*) = k₁* given commitments (t_{w̃₀}, u_key)
  P → V:  v₁^(w), v₁^(k)           (aux commitments for the two sub-instances)
  V → P:  c₁^(w), c₁^(k)           (fold challenges; can be independent transcript labels)
  P → V:  t_{w̃₁}                  (one commitment to the stacked Level-1 witness table)
  V → P:  α₁                       (ring-switch challenge)
  P ↔ V:  one sumcheck transcript  (for the stacked system)
  
OUTPUT:
  Claim: w̃₁(r₂*) = v₂*   (Level-1 witness opening for the stacked table)
  Pending: Level-1 key evaluation (cost \(O(N^{1/4})\) for verifier; can be delegated again)
```

### Verifier's dominant costs (two-level configuration)

| Operation | Cost | Notes |
|-----------|------|-------|
| Level-0 sumcheck verification | \(O(m_0') \approx 30\) rounds | \(\approx 600\) field ops |
| Receive \(V_{\text{key}}\) | \(O(1)\) | one \(F_q\) element |
| Level-1 sumcheck verification | \(O(m_1')\) rounds | larger constants due to stacking |
| **Level-1 key evaluation** (remaining bottleneck) | \(O(\mu_1^{\text{key}})\) | down from \(O(\mu_0^{\text{key}})\) |

**Delegation overhead in proof**: one extra deferred scalar \(V_{\text{key}}\in F_q\) per level, plus constant-factor growth from stacking.

**Delegation overhead in rounds**: constant-factor; the stacked witness table is larger, which may add 0–a few sumcheck rounds depending on padding choices.
