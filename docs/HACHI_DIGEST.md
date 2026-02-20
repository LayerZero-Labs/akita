# Hachi Digest (for side-by-side comparison with SuperNeo)

This file captures the parts of `paper/hachi.pdf` that are most actionable for understanding Hachi’s **parameterization** and **protocol shape**, in a compact AI-readable format.

Primary source: `paper/hachi.pdf` (“Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments over Extension Fields”).

## Canonical parameter tuples (from paper)

The paper gives a concrete benchmark parameter set (ℓ = 30) in Figure 9 and uses it to estimate proof size (~55KB) in §5.2.

```yaml
hachi_parameter_tuples:
  - id: l30_benchmark_first_round
    witness_num_vars_ell: 30
    base_field:
      q: 4294967197  # ~2^32, prime (paper Fig. 9)
      ring_coeffs: "Z_q (aka F_q)"
    extension_field_for_sumcheck:
      k: 4
      field: "F_{q^4}" # paper §5.1, §5.4
    cyclotomic_ring:
      phi: "X^d + 1"
      alpha: 10
      d: 1024
      ring: "R_q = Z_q[X]/(X^d + 1)"
    split_and_fold_params:
      m: 10
      r: 10
    commitment_matrices_heights:
      nA: 1
      nB: 1
      nD: 1
    decomposition_and_challenges:
      decomposition_base_b: 16
      delta: 8   # decomposition length (Fig. 9)
      tau: 4     # expansion factor for decomposing z (Fig. 9)
      omega_L1: 16
      c_nonzero_coeffs_in_challenge: 16
    norm_bounds:
      z_Linf_bound: 30583
      next_witness_Linf_bound: 8
    next_round_witness:
      size: 226
    proof_size_estimate:
      first_round_sumcheck: "~7.3KB"
      adaptation_overhead: "~4.8KB"
      greyhound_subproof: "~43KB"
      total: "~55.1KB"
    timings_reported:
      verify_ms_server: 227  # paper §5.2 narrative (first round), plus Fig. 8 context
      verify_ms_server_greyhound: 130  # cited as Greyhound estimate in §5.2 narrative
```

Notes:

- The paper’s 55.1KB estimate is explicitly derived in §5.2 (“To conclude, the total evaluation proof can be estimated to be 7.3 + 4.8 + 43 KB = 55.1KB.”).
- This tuple is the “Hachi + compose with Greyhound” estimate for ℓ = 30, not “Hachi alone forever”; Hachi’s design explicitly allows switching to Greyhound/LaBRADOR at small witness sizes.

## Exact counts (from the paper’s concrete section)

- Total explicit concrete parameter tables in the paper: **1** (Figure 9; ℓ = 30).
- Concrete benchmark variable counts shown: **3** (ℓ ∈ {26, 28, 30} in Figure 8 timings), but only ℓ = 30 is fully parameterized in Figure 9.
- Unique cyclotomic family used: **1** (`X^d + 1` with `d` power of two).

## What Hachi suggests (for its purpose)

High-level message (from the abstract + technical overview):

- Use **sum-check** to get fast verification, but avoid doing sum-check “over the ring”.
- Use **ring switching** (evaluate at a random α in an extension field) so that the verifier’s checks are field-native and do not require expensive \(R_q\) multiplication.
- Use a **generic reduction** to convert evaluation proofs over extension fields \(F_{q^k}\) into equivalent ring statements over \(R_q\), enabling extension-field evaluation support for lattice PCS.

Concrete implication (paper §5.3–§5.4):

- Hachi can pick **larger ring dimensions** (e.g. \(d=1024\)) than Greyhound’s typical \(d=64\) and still keep verification efficient; larger \(d\) helps commitment time (fewer ring mults, NTT-friendly structure) and enables very sparse challenges.

## Fit into this repo design (easy vs harder)

- Easy fit (already aligned):
  - Hachi’s core ring family is power-of-two cyclotomic \(R_q = Z_q[X]/(X^d+1)\), which matches this repo’s existing algebra/ring direction.
- Harder pieces (protocol-level, not yet fully implemented here):
  - Ring switching pipeline (lifting ring equations to \(Z_q[X]\), evaluate at random α in \(F_{q^k}\), sum-check over \(F_{q^k}\)).
  - The “compose with Greyhound” handoff (treating the reduced witness as a short ring instance for an existing PCS/proof system).

## Notation glossary (from `paper/hachi.pdf`)

Hachi’s paper reuses common lattice-proof-system symbols; here are the ones that tend to confuse on first read.

### Base objects

- **\(q\)**: prime modulus; base ring/field is \(Z_q\) (paper treats \(Z_q\) and \(F_q\) interchangeably).
- **\(d = 2^\alpha\)**: cyclotomic ring dimension (power of two).
- **\(\alpha\)**: shorthand for \(\log_2 d\) (used heavily in §3’s “variable count after transformation” formulas).
- **\(R\)**: integer cyclotomic ring \(Z[X]/(X^d+1)\).
- **\(R_q\)**: cyclotomic ring mod \(q\): \(Z_q[X]/(X^d+1)\).
- **\(F_{q^k}\)**: extension field used to run sum-check with negligible soundness error.
- **\(\kappa\)**: shorthand for \(\log_2 k\) when \(k\) is a power of two (this is the \(\kappa\) used in §3.2 and in the “\(\ell-\alpha+\kappa\)” variable-count formulas).
- **\(\sigma_i\)**: Galois automorphism \(X \mapsto X^i\) on \(R\) / \(R_q\) (with \(i\in (\mathbb{Z}/2d\mathbb{Z})^\times\)).
- **\(R_q^H\)**: fixed ring under a subgroup \(H\) of automorphisms; becomes a subfield isomorphic to \(F_{q^k}\) under conditions (Lemma 1 informal in §1.3).
- **\(\mathrm{Tr}_H\)**: trace map \(R_q \to R_q^H\).
- **\(\psi\)**: an efficiently computable bijection \((R_q^H)^{d/k} \to R_q\) used to turn trace-of-product into inner products (Theorem 1 informal in §1.3).

### Multilinear polynomials / sizes

- **\(\ell\)**: number of variables of the multilinear polynomial (so witness length is \(2^\ell\)).
- **\(L := 2^\ell\)**: number of coefficients / evaluation-table length.

### Split-and-fold / decomposition parameters (Figure 9)

The paper uses **\(m, r\)** as “folding parameters” (they are *not* the same \(m\) as “#constraints” in SuperNeo’s CCS section).

- **\(m, r\)**: split-and-fold parameters controlling the shape of the quadratic relation after one reduction step.
- **\(b\)**: decomposition base (e.g. 16 in Fig. 9).
- **\(\delta\)**: decomposition length for base witness (e.g. 8 in Fig. 9; essentially \(\lceil \log_b q\rceil\) in spirit).
- **\(\tau\)**: expansion factor / extra length parameter for decomposing intermediate vectors (e.g. 4 in Fig. 9).
- **\(\omega\)**: \(\ell_1\)-norm bound on a sparse challenge (Fig. 9).
- **\(c\)**: number of nonzero coefficients in a sparse challenge polynomial (Fig. 9).

### Commitment matrix “heights” (Figure 9)

Hachi uses commitment matrices \(A,B,D\) (not the same “Ajtai A” notation as in SuperNeo).

- **\(n_A, n_B, n_D\)**: heights (row counts) of the corresponding commitment matrices in the composed relation (Fig. 9 uses all 1).

## Protocol overview (what is proved, and what is sent)

This is an end-to-end walkthrough of the **full Hachi protocol**, in the order it’s built in `paper/hachi.pdf`.

### What Hachi is (PCS statement + design goal, **field-first**)

The digest previously jumped straight to the §4 “ring PCS statement”. That is **not the full picture**: the paper’s *natural* PCS interface is the usual one where the **witness coefficients are in the base field** \(Z_q \cong F_q\) but the **evaluation point is in an extension field** \(F_{q^k}\) (because sumcheck / batching wants negligible soundness error).

Concretely, the “true” opening statement the paper starts from (Intro + §3.2) is:

- **Witness polynomial (true witness)**: \(f \in Z_q^{\le 1}[X_1,\dots,X_\ell]\) (equivalently \(F_q^{\le1}[\cdot]\)), with coefficient table \((f_i)_{i\in\{0,1\}^\ell}\subset Z_q\).
- **Claim (extension-field point/value)**: for a public point \(x=(x_1,\dots,x_\ell)\in F_{q^k}^\ell\), prove \(f(x)=y\in F_{q^k}\).

Hachi is engineered so that the verifier’s heavy checking runs as **sumcheck over \(F_{q^k}\)** (fast), while commitments / Module-SIS structure remain over the cyclotomic ring \(R_q = Z_q[X]/(X^d+1)\).

### Step 0 (paper §3.2 → §3.1): the missing “embedding” step (from **\(F_q\)-witness @ \(F_{q^k}\)-point** to a ring statement)

The paper’s §3 is exactly the bridge from the “true” statement above to the §4 ring PCS statement.

#### 0.A (paper §3.2): reduce **\(f\in Z_q[\cdot]\)** at **\(x\in F_{q^k}^\ell\)** to one evaluation over \(F_{q^k}\)

Assume \(k\) is a power of two and write \(k = 2^\kappa\) (this \(\kappa\) is the one used in §3). Split variables into the first \(\kappa\) and the remaining \(\ell-\kappa\). The evaluation can be rewritten (paper Eq. (11)) as:

\[
y \;=\;\sum_{i\in\{0,1\}^\kappa}\Big(\prod_{t=1}^{\kappa} x_t^{i_t}\Big)\cdot y_i,
\quad\text{where}\quad
y_i \;:=\; f_{i}(\,x_{\kappa+1},\dots,x_\ell\,)\in F_{q^k}.
\]

So the prover can send the \(k=2^\kappa\) *partial evaluations* \((y_i)_i\) (and the verifier checks the recombination); what remains is to prove each \(y_i\) is well-formed.

Paper detail (§3.2, right after Eq. (11)): the verifier can compute \(y_{0\ldots 0}\) from the claimed \(y\) and the other \(y_i\), so in principle only **\(k-1\)** of the partial evaluations need to be transmitted.

To make that “prove all \(y_i\)” look like **one** extension-field evaluation, §3.2 defines \(F_{q^k} := F_q[Z]/\varphi(Z)\) and builds an \((\ell-\kappa)\)-variate multilinear polynomial \(f' \in F_{q^k}[X_{\kappa+1},\dots,X_\ell]\) by *embedding* the \(k\) slices \((f_i)_i\) into the \(F_q\)-basis \(1,Z,Z^2,\dots,Z^{k-1}\) (paper §3.2):

\[
f'(X_{\kappa+1},\dots,X_\ell)
:=\sum_{i\in\{0,1\}^\kappa}
f_i(X_{\kappa+1},\dots,X_\ell)\cdot Z^{\sum_{t=1}^{\kappa} i_t 2^{t-1}}.
\]

Then \(f'(x_{\kappa+1},\dots,x_\ell)=\sum_i y_i\cdot Z^{(\cdot)}\) holds as an algebraic identity.

**Critical caveat (security / binding):** it is **not** generally sound to claim that “proving the single packed value \(f'(x_{\kappa+1},\dots,x_\ell)\)” suffices to prove that *each* \(y_i\) is correct, because the coefficients \(y_i\) live in \(F_{q^k}\) (not in the ground field \(F_q\)).

Concretely, the linear map

\[
(y_i)_{i\in\{0,1\}^\kappa}\in (F_{q^k})^{2^\kappa}
\;\longmapsto\;
\sum_i y_i \cdot Z^{(\cdot)} \in F_{q^k}
\]

is **\(F_{q^k}\)-linear** and therefore has a large kernel whenever \(2^\kappa>1\). So many distinct tuples \((y_i)_i\) produce the *same* packed sum. A toy example for \(k=2\) (basis \(\{1,Z\}\)): the two different pairs \((y_0,y_1)\) and \((y_0+Z,\; y_1-1)\) satisfy

\[
(y_0+Z) + (y_1-1)\cdot Z \;=\; y_0 + y_1\cdot Z,
\]

so the packed value alone does not pin down \(y_0,y_1\).

This is exactly the “basis is only independent over the ground field” pitfall called out in `paper/fri-binius.pdf`, where they explain that a basis \((\beta_v)\) is linearly independent over \(K\) but **not** over its extension \(L\); hence basis-combining \(L\)-valued claims is insecure. See the strawman discussion around Figure 1 in:

- `paper/fri-binius.pdf` §1.3 “Ring-Switching”, “A strawman approach”, Figure 1, and the paragraph beginning “While this protocol is complete, it’s not secure.” (pages 4–6 in this PDF copy).

**How to fix (high-level, sumcheck-style):** you must reduce the “\(y_i\) are well-formed” constraints into **ground-field (\(F_q\)) constraints** before basis-combining / packing them.

The standard way (as in Fri-Binius ring-switching) is:

1. **Basis-decompose each \(y_i\) over \(F_q\)**: write \(y_i = \sum_{u=0}^{k-1} y_{u,i}\,Z^u\) with \(y_{u,i}\in F_q\).
2. Also basis-decompose the extension-field weights (the equality-polynomial weights / monomial weights) into \(F_q\) coefficients.
3. Check the resulting family of \(F_q\)-valued equalities “slice-wise” (over the \(u\) index), and then **batch them with an additional sumcheck** so the verifier only pays polylog overhead.

After this extra sumcheck layer pins down the \(F_q\)-slices, packing becomes injective again (because it is now combining **\(F_q\)-vectors** against an \(F_q\)-basis), and the reduction from “many partial claims” to “one packed claim” becomes sound.

##### 0.A.1 Concrete “extra sumcheck” shape (Fri-Binius Eq. (12) style), and how many rounds it costs

This subsection spells out the *exact* reason the extra sumcheck has \(\ell\) rounds (and how it fits the objects already in §3.2).

Let \(K:=F_q\) and \(L:=F_{q^k}\), and assume \(k=2^\kappa\) for some \(\kappa\) (as in §3.2). Fix a \(K\)-basis \((\beta_u)_{u\in\{0,1\}^\kappa}\) of \(L\) (in §3.2 the paper chooses \(1,Z,\dots,Z^{k-1}\), which is just one such basis).

The goal is to prove the family of claims (paper after Eq. (11)):

\[
\forall v\in\{0,1\}^\kappa:\quad
y_v \stackrel{?}{=} f_v(x_{\kappa+1},\dots,x_\ell)\in L,
\]

where \(f_v\in K^{\le1}[X_{\kappa+1},\dots,X_\ell]\) is the “slice” of \(f\) with the first \(\kappa\) variables fixed to \(v\).

The unsafe step is to basis-combine these \(L\)-valued equalities directly. The safe replacement is to basis-decompose everything so the equalities become \(K\)-valued first.

1) **Basis-decompose the prover’s \(L\)-claims**: for each \(v\in\{0,1\}^\kappa\), write

\[
y_v = \sum_{u\in\{0,1\}^\kappa} y_{u,v}\,\beta_u,
\quad\text{with } y_{u,v}\in K.
\]

2) **Expand \(f_v(x_{\kappa+1},\dots,x_\ell)\) as a \(K\)-weighted sum.** Let \(\ell':=\ell-\kappa\) and index \(w\in\{0,1\}^{\ell'}\). Then

\[
f_v(x_{\kappa+1},\dots,x_\ell)
=
\sum_{w\in\{0,1\}^{\ell'}}
\mathrm{eq}(x_{\kappa+1},\dots,x_\ell;\,w)\cdot f(v,w),
\]

where \(f(v,w)\in K\) is the \((v,w)\) Lagrange coefficient of \(f\), and \(\mathrm{eq}(\cdot;\,w)\in L\) is the multilinear equality indicator value.

3) **Basis-decompose the (public) weights**: for each \(w\), decompose the \(L\)-element \(\mathrm{eq}(x_{\kappa+1},\dots,x_\ell;\,w)\) in the same basis:

\[
\mathrm{eq}(x_{\kappa+1},\dots,x_\ell;\,w)
=
\sum_{u\in\{0,1\}^\kappa} A_{w,u}\,\beta_u,
\quad\text{with } A_{w,u}\in K,
\]

where the \(A_{w,u}\) are deterministically computable from the public point \((x_{\kappa+1},\dots,x_\ell)\) and the chosen basis.

4) **Now each coordinate is a \(K\)-statement**: equating coefficients of \(\beta_u\) yields the family of \(K\)-equalities

\[
\forall u,v\in\{0,1\}^\kappa:\quad
y_{u,v} \stackrel{?}{=} \sum_{w\in\{0,1\}^{\ell'}} A_{w,u}\cdot f(v,w).
\]

5) **Pack the \(f(v,w)\) table into one \(L\)-multilinear** (this is the same “packing” idea as §3.2, but applied to a \(K\)-table so it is information-preserving):

\[
f'(w) := \sum_{v\in\{0,1\}^\kappa} f(v,w)\,\beta_v \in L,
\]

so \(f'\) has \(\ell'=\ell-\kappa\) variables over \(L\).

6) **Batch and sumcheck.** Define the combined \(L\)-valued claims (these are the secure analog of the strawman’s linear combination step):

\[
\hat y_u := \sum_{v\in\{0,1\}^\kappa} y_{u,v}\,\beta_v \in L.
\]

Then the \(K\)-equalities above imply the \(L\)-equalities

\[
\forall u\in\{0,1\}^\kappa:\quad
\hat y_u \stackrel{?}{=} \sum_{w\in\{0,1\}^{\ell'}} A_{w,u}\cdot f'(w).
\]

Finally, batch over \(u\) with a random point \(r''\in L^\kappa\) and apply sumcheck to the identity

\[
\sum_{u\in\{0,1\}^\kappa} \mathrm{eq}(u;\,r'')\cdot \hat y_u
\stackrel{?}{=}
\sum_{w\in\{0,1\}^{\ell'}}
\Big(\sum_{u\in\{0,1\}^\kappa}\mathrm{eq}(u;\,r'')\cdot A_{w,u}\Big)\cdot f'(w).
\]

This is exactly the structural form of Fri-Binius Eq. (12). The sum ranges over \((u,w)\in\{0,1\}^\kappa\times\{0,1\}^{\ell'}\), so the sumcheck has **\(\kappa+\ell'=\ell\) rounds**, i.e. it costs **\(+\kappa=\log_2 k\)** more rounds than a hypothetical scheme that only needed to open \(f'\) (which has \(\ell'\) variables).

Asymptotically, prover time for this added sumcheck is linear in the domain size \(2^{\kappa+\ell'}=2^\ell\) (times poly\((\ell,k)\) factors), i.e. \(\Theta(2^\ell)\) field operations in \(L\) for the sumcheck layer, plus one opening of the \(L\)-PCS on \(f'\) at the final point.

#### 0.B (paper §3.1): embed \(F_{q^k}\) inside \(R_q\) and turn extension-field inner products into trace statements over \(R_q\)

Now treat the remaining claim as an evaluation over \(F_{q^k}\) (equivalently over the subfield \(R_q^H \cong F_{q^k}\) from Lemma 5). §3.1 then provides:

- **A subfield of \(R_q\)**: for \(q \equiv 5 \pmod 8\) and \(k\mid d/2\), define \(H=\langle\sigma^{-1},\sigma^{4k+1}\rangle\subset\mathrm{Aut}(R)\). Then the fixed ring \(R_q^H\) is a field and \(R_q^H \cong F_{q^k}\). (Lemma 5.)
- **A packing bijection** \(\psi:(R_q^H)^{d/k}\to R_q\). (Eq. (8), Theorem 2.)
- **A trace/inner-product identity** (Theorem 2):

  \[
  \mathrm{Tr}_H\big(\psi(a)\cdot \sigma^{-1}(\psi(b))\big) \;=\; \frac{d}{k}\cdot \langle a,b\rangle,
  \quad a,b \in (R_q^H)^{d/k}.
  \]

Concretely (paper around Eq. (10)), pick \(\alpha:=\log_2 d\) and split the \(\ell\) variables into an “outer” prefix of length \(\ell-\alpha+\kappa\) and an “inner” suffix of length \(\alpha-\kappa\). Write indices as \(i\in\{0,1\}^{\ell-\alpha+\kappa}\) and \(j\in\{0,1\}^{\alpha-\kappa}\). Define:

- **Packed coefficient blocks**: \(F_i := \psi\big((f_{i\parallel j})_{j\in\{0,1\}^{\alpha-\kappa}}\big)\in R_q\).
- **Packed monomial block at the suffix**:

  \[
  v := \psi\Big(\big(\prod_{t=1}^{\alpha-\kappa} x_{\ell-\alpha+\kappa+t}^{\,j_t}\big)_{j\in\{0,1\}^{\alpha-\kappa}}\Big)\in R_q.
  \]

Then the extension-field evaluation is reduced to checking a **single trace equation** in \(R_q\) involving:

\[
Y \;:=\; \sum_{i\in\{0,1\}^{\ell-\alpha+\kappa}}
\Big(\prod_{t=1}^{\ell-\alpha+\kappa} x_t^{i_t}\Big)\cdot F_i
\;\in\; R_q,
\]

namely \(\mathrm{Tr}_H\big(Y\cdot \sigma^{-1}(v)\big)=\tfrac{d}{k}\,y\). The prover sends this single ring element \(Y\).

What remains is to prove that \(Y\) is well-formed, i.e. that it is *exactly* the evaluation of the \((\ell-\alpha+\kappa)\)-variate ring polynomial \(F := (F_i)_i \in R_q^{\le 1}[X_1,\dots,X_{\ell-\alpha+\kappa}]\) at the point \((x_1,\dots,x_{\ell-\alpha+\kappa})\) (viewing those \(x_t\) as elements of the subfield \(R_q^H\subset R_q\)). This is the “smaller multilinear evaluation claim over \(R_q\)” that becomes the input to the §4 core PCS.

### Step 1 (paper §4): the internal ring PCS statement Hachi actually proves

After §3’s transformation, Hachi reduces extension-field evaluation claims to the §4 “core” ring PCS statement:

- **Witness polynomial (ring core)**: \(f \in R_q^{\le 1}[X_1,\dots,X_{\ell'}]\) with coefficients in \(R_q\) (this \(f\) is the reduced ring polynomial constructed in Step 0.B; i.e., it is the \(F := (F_i)_i\) from above, just returning to the paper’s §4 notation),
- **Claim**: for a public point \(x\in R_q^{\ell'}\) (in the reduction flow, typically \(x\in (R_q^H)^{\ell'}\subset R_q^{\ell'}\)), prove \(f(x)=u\in R_q\),

where (per §3.2) \(\ell' = \ell - \alpha\) in the important special case “coefficients in \(Z_q\), point in \(F_{q^k}\)”, with \(d=2^\alpha\).

From here on, we are *inside* §4. The paper continues to write the ring instance as \(\ell\)-variate; in digest notation, you can read the \(\ell\) used in Steps A/B/C below as \(\ell'\) (the post-§3 reduced variable count).

### Step A (paper §4.1): Commit to the coefficient table (inner + outer commitments)

Hachi’s commitment structure is Greyhound-style: it commits to the coefficient table in blocks using two Ajtai commitments.

Split \(\ell = m + r\) with \(m \approx r\). Define the block slices \(f_i^\top := (f_{i\|j})_{j\in\{0,1\}^m}\in R_q^{2^m}\) for each outer index \(i\in\{0,1\}^r\). (This is the “\(f_i\)” notation right below Equation (12).)

Commitment construction (Equations (13)–(14)):

- **Decompose each slice**: \(s_i := G^{-1}_{2^m}(f_i)\in R_q^{2^m\delta}\), where \(\delta=\lceil \log_b q\rceil\). (Eq. (13).)
- **Inner Ajtai commit**: \(t_i := A s_i \in R_q^{n_A}\) for all \(i\in[2^r]\).
- **Decompose the inner commit**: \(\hat t_i := G^{-1}_{n_A}(t_i)\).
- **Outer Ajtai commit**: stack all \(\hat t_i\) and commit:
  - \(u := B[\hat t_1;\dots;\hat t_{2^r}] \in R_q^{n_B}\). (Eq. (14).)

So:

- **Commitment** is \(u \in R_q^{n_B}\).
- **Opening witness** (conceptually) is \((s_i,\hat t_i)_i\).
- Knowledge/binding is phrased in terms of “weak openings” (definition right after Eq. (14), Lemma 7).

### Step B (paper §4.2): Reduce “open \(f(x)=u\)” to “prove knowledge of short vectors satisfying a public linear system”

This is the core reduction pipeline of Hachi’s opening proof.

#### B.0. Descriptive names for the “hat” variables (implementation-minded)

The paper uses “hats” (\(\hat{\cdot}\)) for **digit-decomposed** objects (small coefficients), and uses a few single-letter vectors that are easy to lose track of. Here is a naming map that matches the paper’s definitions in §4.1–§4.2 and is meant to be used as docstring text during implementation.

- **\(s_i\)** = **block_digits** (digits of the \(i\)-th coefficient block):
  - \(s_i = G^{-1}_{2^m}(f_i)\). (Eq. (13).)
  - Shape: vector of ring elements, length \(2^m\cdot\delta\).

- **\(t_i\)** = **inner_commitment_block_i** (Ajtai commit of the \(i\)-th block digits):
  - \(t_i := A s_i \in R_q^{n_A}\).

- **\(\hat t_i\)** = **inner_commitment_digits_block_i** (digits of \(t_i\)):
  - \(\hat t_i := G^{-1}_{n_A}(t_i)\).

- **\(u\)** = **outer_commitment** (the actual PCS commitment output):
  - \(u := B[\hat t_1;\dots;\hat t_{2^r}] \in R_q^{n_B}\). (Eq. (14).)

- **\(w_i\)** = **block_partial_eval_i** (block \(i\)’s contribution after plugging the “\(a\)” half of the opening point):
  - \(w_i := a^\top G_{2^m} s_i \in R_q\). (Defined right before Eq. (16).)

- **\(\hat w_i\)** = **block_partial_eval_digits_i** (digits of \(w_i\)):
  - \(\hat w_i := G^{-1}_1(w_i) \in R_q^\delta\).

- **\(v\)** = **aux_commitment_to_block_partials** (commitment to all \(\hat w_i\)):
  - \(v := D\hat w \in R_q^{n_D}\). (Eq. (16).)

- **\(c\)** = **fold_challenge_vector** (short/sparse ring challenge used to fold blocks):
  - \(c=(c_1,\dots,c_{2^r})\in C^{2^r}\), with \(\|c_i\|_1\le \omega\).

- **\(z\)** = **folded_block_digits** (folded witness over block digits):
  - \(z := \sum_{i=1}^{2^r} c_i s_i\). (Eq. (18)/(19) discussion.)
  - This is the key “compress all blocks into one witness” object.

- **\(\hat z\)** = **folded_block_digits_redecomposed** (extra decomposition of \(z\) after coefficient growth):
  - \(\hat z := J^{-1}_{2^m}(z)\), where \(J\) is the gadget matrix sized for \(\tau\) digits (so \(\hat z\) has length \(2^m\cdot\delta\cdot\tau\)). (Right after Eq. (20).)

- **\(r\)** = **modulus_quotient_witness** / **slack_witness** (ring-switching quotient):
  - \(Mz = y + (X^d+1)\cdot r\) over \(Z_q[X]\). (§4.3.)
  - In the “real” protocol, \(r\) is digit-decomposed as \(r=\sum_u b^u r_u\) and the prover commits to the digit vectors \(r_u\). (See §4.3 and our C.2.1.)

#### B.1. Rewrite evaluation as a bilinear form (Eq. (12))

Define the monomial vectors from the evaluation point \(x\):

- \(b^\top := (x_1^{i_1}\cdots x_r^{i_r})_{i\in\{0,1\}^r}\in R_q^{2^r}\)
- \(a^\top := (x_{r+1}^{j_1}\cdots x_\ell^{j_m})_{j\in\{0,1\}^m}\in R_q^{2^m}\)

Then the evaluation can be written as Equation (12), which motivates everything that follows.

#### B.2. Introduce intermediate values \(w_i\) and commit to them (Eq. (16))

Using the decomposed slices \(s_i\), Equation (15) rewrites the opening equation using \(s_i\).

Define the intermediate ring elements:

- \(w_i := a^\top G_{2^m} s_i \in R_q\). (Defined right before Eq. (16).)

Then:

- \(u = b^\top w\). (Eq. (17).)

The prover also commits to \(w\) using another Ajtai-style commitment (Eq. (16)):

- decompose \(w_i\) to \(\hat w_i := G^{-1}_1(w_i)\in R_q^\delta\),
- stack \(\hat w := (\hat w_1,\dots,\hat w_{2^r})\),
- compute **\(v := D \hat w \in R_q^{n_D}\)**. (Eq. (16).)

This is the first prover message in Figure 3:

- **P → V**: send \(v\).

#### B.3. Fold all the \(s_i\) using a short challenge vector \(c\) (Eqs. (18)–(19))

The verifier samples a short/sparse challenge vector:

- **V → P**: \(c=(c_1,\dots,c_{2^r}) \leftarrow C^{2^r}\), where \(C \subset \{c\in R_q : \|c\|_1 \le \omega\}\). (See the paragraph introducing \(c\).)

The prover folds the witness:

- \(z := \sum_{i=1}^{2^r} c_i s_i \in R_q^{2^m\delta}\). (Immediately after defining \(c\).)

Two crucial linear identities then hold:

- \(a^\top G_{2^m} z = (c^\top \otimes G_1)\hat w\). (Eq. (18).)
- \(A z = (c^\top \otimes G_{n_A})\hat t\). (Eq. (19).)

These relate the folded witness \(z\) to the already-committed objects \(\hat w,\hat t\).

#### B.4. Decompose \(z\) further and form one big “unstructured linear relation” (Eq. (20))

Because coefficients of \(z\) are larger, the prover further decomposes \(z\) using another gadget matrix \(J\):

- pick a bound \(\beta\) on \(\|z\|_\infty\), define \(\tau:=\lceil \log_b \beta\rceil\),
- compute \(\hat z := J^{-1}_{2^m}(z)\in R_q^{2^m\delta\tau}\).

At this point, the prover’s remaining task becomes:

> prove knowledge of a short vector \((\hat w,\hat t,\hat z)\) satisfying a public linear system over \(R_q\). (Eq. (20).)

Equation (20) is exactly that public linear system:

- it includes the matrices \(D,B,A\),
- it includes the evaluation-derived vectors \(a,b\),
- it includes the verifier challenge \(c\),
- and it enforces: (i) consistency with the commitments \(u,v\), (ii) the opening equation, and (iii) the fold relations (18)–(19).

##### B.4.1 Eq. (20) rewritten as a list of constraints (no hats, descriptive meaning)

Mentally, Eq. (20) is just a *bundling* of several checks into one linear system \(M_{\text{big}}\cdot \text{witness} = \text{statement}\). Written as explicit constraints, the prover is proving existence of **small** objects:

- **`block_partial_eval_digits`** = \(\hat w\)
- **`inner_commitment_digits`** = \(\hat t\)
- **`folded_block_digits_redecomposed`** = \(\hat z\) (and implicitly \(z = J\hat z\))

such that all of the following hold:

1. **Aux-commitment consistency (ties \(\hat w\) to the sent commitment \(v\))**:
   - \(D\hat w = v\). (Eq. (16).)

2. **Main commitment consistency (ties \(\hat t\) to the sent commitment \(u\))**:
   - \(B\hat t = u\). (Eq. (14).)

3. **Evaluation equation (ties the claimed opening value \(u\) to the partials \(w\))**:
   - If \(w := G_{2^r}\hat w\) then \(b^\top w = u\). (Eq. (17).)

4. **Fold-consistency: partial-eval folding matches the folded witness \(z\)**:
   - Let \(z := J\hat z\) (so \(z\) is the “recomposed” folded block-digits witness).
   - Then \(a^\top G_{2^m} z = (c^\top\otimes G_1)\hat w\). (Eq. (18).)

5. **Fold-consistency: inner-commitment folding matches the same folded witness \(z\)**:
   - \(A z = (c^\top\otimes G_{n_A})\hat t\). (Eq. (19).)

6. **Smallness/range constraints (coefficient bounds)**:
   - Coefficients of \(\hat w,\hat t,\hat z\) lie in the intended digit ranges (bounded by the gadget decomposition), and the protocol’s range-check machinery in §4.3 ultimately enforces these via the \(H_0\) constraint.

Figure 3 shows this “conceptual protocol”, ending with the prover *sending* \((\hat w,\hat t,\hat z)\), but the paper immediately notes:

- in the **final scheme**, the prover does not send these in the clear; instead it proves knowledge of them.

### Step C (paper §4.3): Prove the unstructured linear relation + coefficient smallness via ring switching + sumcheck over \(F_{q^k}\)

This is the main verifier-efficiency trick: transform ring relations into field relations (via evaluation at a random \(\alpha\)), then apply sumcheck over \(F_{q^k}\).

#### C.1. The generic linear relation they want to prove

They define:

- \(R^{\mathrm{lin}}_{q,d,n,\mu,b}\): given public \(M\in R_q^{n\times \mu}\), \(y\in R_q^n\), prove knowledge of \(z\in R_q^\mu\) such that \(Mz=y\) and \(\|z\|_\infty \le b-1\).

Eq. (20) is an instance of this relation (with \(z=(\hat w,\hat t,\hat z)\)).

#### C.2. Ring switching (Figure 4): lift equality in \(R_q\) to an identity in \(Z_q[X]\), then evaluate at random \(\alpha\)

Because \(R_q = Z_q[X]/(X^d+1)\), the ring equation \(Mz=y\) holds iff there exists a “slack” polynomial vector \(r\) such that (paper §4.3, “Ring switching.”):

- \(Mz = y + (X^d+1)\cdot r\) over \(Z_q[X]\).

##### C.2.1 Critical “hidden detail”: \(r\) is gadget-decomposed, and the protocol commits to the digits

The paper explicitly notes that **both \(z\) and \(r\)** are prover witnesses, and while the verifier ultimately needs a check involving \(r\), the prover *does not* want to commit to a large-\(q\) object and then range-check it “as-is”.

So, for “notation and implementation simplicity”, Hachi performs a base-\(b\) gadget decomposition of the quotient witness \(r\) (paper §4.3):

- \(r = \sum_u b^u \cdot r_u\),
- the prover commits to \((z, r_1, \dots, r_{\log_b(q)})\) instead of \((z,r)\),
- and the prover proves \(\|r_u\|_\infty \le b-1\) for every digit-vector \(r_u\).

The paper then says it **omits the subscript \(u\)** from this point on, and that to incorporate this in the sumcheck view, “we can modify the multilinear extension correspondingly.”

Concretely, the “real” linear constraint is:

\[
Mz = y + (X^d+1)\cdot \sum_u b^u r_u \quad\text{over } Z_q[X],
\]

and the “real” smallness constraints are:

\[
\|z\|_\infty \le b-1
\quad\text{and}\quad
\|r_u\|_\infty \le b-1 \ \forall u.
\]

Protocol idea (Figure 4):

- **P → V**: commit to \((z,r)\): \(t := \mathrm{Com}(z,r)\). (In the “real” protocol: commit to \((z,(r_u)_u)\) as above; the paper keeps writing \((z,r)\) after omitting the digit index.)
- **V → P**: sample \(\alpha \leftarrow F_{q^k}\).
- Reduce to checking the *field* equations:
  - \(M(\alpha)z(\alpha) = y(\alpha) + (\alpha^d+1)r(\alpha)\) over \(F_{q^k}\). (Figure 4.)

Soundness of this “evaluate at random \(\alpha\)” step is formalized as \(2d\)-special soundness (Lemma 9), reflecting degree \(\le 2d-1\).

They also need to enforce that the witness coefficients are genuinely in \(Z_q\) and small (not arbitrary in \(F_{q^k}\)), and they fold those checks into the next “sumcheck view”.

#### C.3. Represent the constraints as multilinear polynomials and batch them (Eqs. (21)–(23), Figure 5)

This is where the **full constraint system** (including the range check) is spelled out in the paper.

##### C.3.1 The witness polynomial \(\tilde w\) (Eq. (21))

The prover’s committed witness for sumcheck is a multilinear polynomial \(\tilde w\) that encodes coefficient tables.

**Important:** because of §4.3’s hidden gadget decomposition \(r=\sum_u b^u r_u\), the *real* committed witness should be thought of as encoding coefficient tables of:

- the vector of polynomials \(z\), and
- all digit-vectors \(r_u\),

not just a single undigitized \(r\). The paper immediately omits the digit index \(u\) and keeps writing \((z,r)\); this is why Eq. (21) below only has one \(r\).

The paper defines a multilinear polynomial \(e_w\) (same role as \(\tilde w\) in our prose) as:

- \(e_w(u,\ell) = z_{u,\ell}\) if \(u \le \mu\),
- \(e_w(u,\ell) = r_{u-\mu,\ell}\) if \(\mu < u \le \mu+n\). (Eq. (21).)

Here \(u\) and \(\ell\) are treated as binary strings indexing \([\,\mu+n\,]\) and \([\,d\,]\).

To “incorporate the gadget decomposition” in the exact same shape, one natural flattened encoding is:

- let \(\delta := \lceil \log_b(q)\rceil\) be the number of digits,
- treat the committed table as indexed by \([\,\mu + n\cdot \delta\,]\times[\,d\,]\),
- keep \(e_w(u,\ell)=z_{u,\ell}\) for \(u\in[\mu]\),
- and set \(e_w(\mu + u'\cdot n + i,\ell) := (r_{u'})_{i,\ell}\) for digit index \(u'\in[\delta]\) and row index \(i\in[n]\).

With this flattening, the “range check” constraints apply to **every** coordinate of \(z\) and **every** coordinate of every digit vector \(r_{u'}\).

##### C.3.1.1 Digit-aware \(e_{M_\alpha}\): the fully expanded constraint coefficient function

The paper’s simplified definition (after Eq. (21)) encodes the linear constraint at \(\alpha\) by defining a public function \(e_{M_\alpha}(i,u)\) that (informally) gives the coefficient multiplying the polynomial \(w_u(\alpha)\) inside row \(i\).

Once you flatten the digit vectors \((r_{u'})_{u'\in[\delta]}\) into the witness index \(u\in[\,\mu+n\delta\,]\), the *literal* digit-aware version is:

- Let \(u\in[\,\mu+n\delta\,]\).
- Define the “decoded” witness coordinate \(W_u(\alpha)\) by:
  - for \(1 \le u \le \mu\): \(W_u(\alpha) := z_u(\alpha)\),
  - for \(\mu + (u'-1)n + i\) with \(u'\in[\delta]\) and \(i\in[n]\): \(W_{\mu + (u'-1)n + i}(\alpha) := (r_{u'})_i(\alpha)\).

Then define:

\[
e^{\text{dig}}_{M_\alpha}(i,u) :=
\begin{cases}
M_{i,u}(\alpha) & \text{if } 1 \le u \le \mu, \\
-\,b^{u'-1}\cdot(\alpha^d+1) & \text{if } u = \mu + (u'-1)n + i \text{ for some } u'\in[\delta], \\
0 & \text{otherwise.}
\end{cases}
\]

With this explicit \(e^{\text{dig}}_{M_\alpha}\), the digitized ring-switch check for each row \(i\in[n]\) is exactly:

\[
\sum_{u=1}^{\mu+n\delta} e^{\text{dig}}_{M_\alpha}(i,u)\cdot W_u(\alpha) \;=\; y_i(\alpha).
\]

This is just the statement:

\[
\sum_{j=1}^{\mu} M_{i,j}(\alpha)\,z_j(\alpha)\;-\;(\alpha^d+1)\sum_{u'=1}^{\delta} b^{u'-1}\,(r_{u'})_i(\alpha) \;=\; y_i(\alpha),
\]

which is equivalent to the “real” lifted identity \(Mz = y + (X^d+1)\sum_{u'}b^{u'-1}r_{u'}\) after evaluation at \(X=\alpha\).

##### C.3.2 The two *literal* constraints before batching (paper right before Eq. (22)/(23))

For a fixed ring-switch challenge \(\alpha\in F_{q^k}\), the verifier wants to enforce:

1. **Linear constraints (ring switching at \(\alpha\))**: for each row \(i\in[n]\),

   \[
   \sum_{u=1}^{\mu+n} e_{M_\alpha}(i,u)\cdot \sum_{\ell} e_w(u,\ell)\cdot e_\alpha(\ell)\;=\;y_i(\alpha),
   \]

   where \(e_\alpha(\ell)=\alpha^\ell\) and \(e_{M_\alpha}(i,u)\) is the public multilinear encoding of \(M(\alpha)\) plus the extra \(-(\alpha^d+1)\) term on the \(r\)-coordinates. (This is the first bullet list item under Figure 5 in the paper.)

   With the hidden gadget decomposition \(r=\sum_u b^u r_u\), the same constraint is enforced except the right-hand side becomes \(y_i(\alpha) + (\alpha^d+1)\sum_u b^u r_{i,u}(\alpha)\). In the “\(e_{M_\alpha}\cdot e_w\)” encoding, this is handled by:

   - expanding \(e_w\) to include all digit-vectors \((r_u)_u\) (as described in C.3.1), and
   - modifying the \(-(\alpha^d+1)\) part of \(e_{M_\alpha}\) to include the appropriate digit weights \(-b^u(\alpha^d+1)\) on the digit blocks.

2. **Smallness / range constraints (this is the “range check”)**: for *every* coordinate \((u,\ell)\),

   \[
   P_b\big(e_w(u,\ell)\big) = 0
   \quad\text{where}\quad
   P_b(T) := \prod_{t=-(b-1)}^{b-1}(T-t).
   \]

   The paper writes this explicitly as the vanishing product:
   \(e_w(u,\ell)\cdot(e_w(u,\ell)-1)\cdot(e_w(u,\ell)+1)\cdots(e_w(u,\ell)-b+1)\cdot(e_w(u,\ell)+b-1)=0\).

This is **not optional**: it is exactly how Hachi enforces that the coefficients the prover is claiming for \(z\) (and the \(r\)-side witness they commit) are small integers (embedded into \(F_{q^k}\)), i.e. a range/membership proof via a root-check polynomial.

##### C.3.3 Batching with equality polynomials: the exact \(H_\alpha\) and \(H_0\) (Eqs. (22)–(23))

Let the multilinear equality polynomial be:

- \(e_{eq}(t,i) = \prod_j (t_j i_j + (1-t_j)(1-i_j))\).

Then the paper defines:

- **Linear constraint batch** \(H_\alpha(t)\) (Eq. (22)):

  \[
  H_\alpha(t) :=
  \sum_{i\in[n]} e_{eq}(t,i)\cdot
  \Big(\sum_{u,\ell} e_{M_\alpha}(i,u)\cdot e_w(u,\ell)\cdot e_\alpha(\ell) - y_i(\alpha)\Big).
  \]

- **Smallness/range batch** \(H_0(t)\) (Eq. (23)):

  \[
  H_0(t) :=
  \sum_{u,\ell} e_{eq}\big(t,(u,\ell)\big)\cdot P_b\big(e_w(u,\ell)\big).
  \]

The goal is to prove both are **identically zero polynomials**, which they reduce (Figure 5) to random-point checks:

- **V → P**: send random points \(\tau_0,\tau_1\),
- prove \(H_0(\tau_0)=0\) and \(H_\alpha(\tau_1)=0\).

#### C.4. Use sumcheck to prove the remaining batched sums (Figure 6)

After fixing \(\tau_0,\tau_1\), they rewrite \(H_0(\tau_0)\) and \(H_\alpha(\tau_1)\) as sums over \((u,\ell)\) of polynomials \(F_{0,\tau_0}\) and \(F_{\alpha,\tau_1}\), and then apply **sumcheck** over \(F_{q^k}\) (discussion right after Eq. (23)).

Figure 6 gives the “single-round view” of sumcheck: prover sends univariate \(g_i\), verifier sends random challenge scalars \(a_i\), and in the end the verifier reduces everything to checking an evaluation of the witness polynomial \(\tilde w\) at one final random point.

Crucially, the sumcheck ends in:

- a **final evaluation claim** of the committed \(\tilde w\) at a random point \(r^\*\),
- plus the requirement to prove this evaluation is consistent with the commitment \(t\).

That “prove the evaluation claim for \(\tilde w\)” is exactly where recursion happens: you invoke the PCS again on the smaller committed object.

##### C.4.1 Full message flow (Figure 7), spelled out

Figure 7 in the paper is the full composition of Figures 4, 5, and 6. Written as an explicit transcript skeleton:

- **P → V**: \(t := \mathrm{Com}(z,r)\). (In the “real” protocol: \(t := \mathrm{Com}(z,(r_u)_u)\) with \(r=\sum_u b^u r_u\).)
- **V → P**: sample challenges:
  - \(\alpha \leftarrow F_{q^k}\),
  - \(\tau_0 \leftarrow F_{q^k}^{\log(\mu)+\log(d)}\),
  - \(\tau_1 \leftarrow F_{q^k}^{\log(n)}\).
- **P ↔ V (sumcheck)**: run sumcheck to prove \(H_0(\tau_0)=0\) and \(H_\alpha(\tau_1)=0\):
  - in each sumcheck round, **P → V** sends a univariate polynomial \(g_i(X_i)\),
  - and **V → P** responds with a random scalar challenge \(a_i \leftarrow F_{q^k}\). (Figure 6.)
- **P → V (final opening)**: open the commitment \(t\) at the final point determined by \((a_1,\dots,a_\ell)\) to provide the needed value(s) of \(\tilde w\).
- **V (final checks)**:
  - evaluate the public multilinear extensions (notably \(e_{M_\alpha}\) / \(e_\alpha\)) at the same final point,
  - and check the final sumcheck identities (Figure 6’s last-round checks).

### Step D (paper §3 + §5): recursion shape and concrete proof-size accounting

Recursion shape (high-level):

- One invocation reduces the big opening proof into opening proofs for **smaller** committed objects (smaller \(\ell\), smaller coefficient domains after decomposition, and evaluation points living in \(F_{q^k}\)).
- Eventually, Hachi suggests switching to different sub-proofs when the witness is small (paper discusses both switching to LaBRADOR/JL and composing with Greyhound).

Concrete proof-size estimate for \(\ell=30\) (paper §5.2):

- Sumcheck for the first round: ~7.3KB.
- “Adaptation + Greyhound subproof” gives total ~55.1KB for an evaluation proof.

This is the explicit estimate in §5.2:

- first-round sumcheck: \(7.3\)KB,
- plus preparation/adaptation \(4.8\)KB,
- plus Greyhound evaluation subproof \(43\)KB,
- total \(7.3 + 4.8 + 43 = 55.1\)KB.

#### D.1 What the “adaptation to Greyhound” actually means (paper §4.5, §5.2)

When the witness becomes small enough, Hachi can stop running its own §4.3 ring-switch + sumcheck recursion and instead reduce the remaining claim into a **Greyhound-native** opening proof.

The key shape (paper §4.5 and the concrete instantiation in §5.2) is:

- you end up with an evaluation claim where the verifier has reduced everything to checking that a committed multilinear object \(\tilde w\) evaluates correctly at a random point;
- the prover groups / “packs” the needed coefficients into extension-field elements, and then uses the §3 embedding machinery (the \(\psi\) map and trace identity) to turn the remaining check into a ring statement supported by Greyhound.

In the concrete accounting, the additional prover communication for this handoff is bounded as (Equation (28) in the paper excerpted in §4.5):

- \((k-1)\cdot k \cdot \log q\) bits for the sent partial evaluations \((y_i)\), plus
- \(d'\cdot \log q\) bits for sending one ring element \(p\in R'_{q}\) (where \(d'\) is the Greyhound ring dimension, e.g. \(d'=64\) in §5.2).

This is why §5.2 reports the “adaptation overhead” as small (~0.3KB for the non-commitment part), and then adds the cost to **commit** to the new Greyhound witness element(s).

### Message flow summary (first / dominant round)

If you want a “wire-format mental model” aligned to Figures 3–6, the dominant first round contains:

- **Ring commitments**: the main commitment \(u\in R_q^{n_B}\) (Eq. (14)) and the auxiliary commitment \(v\in R_q^{n_D}\) (Eq. (16)).
- **Ring challenge**: short/sparse \(c\in C^{2^r}\) used to fold the opening witness (Eq. (18)–(19)).
- **Field challenge**: \(\alpha\in F_{q^k}\) for ring switching (Figure 4), and random points \(\tau_0,\tau_1\) (Figure 5).
- **Sumcheck transcript**: univariate polynomials \(g_i\) and challenge scalars \(a_i\) (Figure 6), ending in a point \(r^\*\) and an evaluation value \(\tilde w(r^\*)\).
- **Opening proof(s)**: recursive PCS openings that prove the claimed evaluations of the committed \(\tilde w\) values match the commitments.

## Implementation-oriented PCS flow spec (what we need before coding)

This section re-states the end-to-end PCS as a **software spec**: what the prover/verifier compute, what objects exist (and their “witness” roles), and what must be pinned down *before* we implement §4.3 “witness table embedding” or the ring-switch sumcheck instances.

### E.0 The key mental model: “stacked linear relation” → “ring switch” → “sumcheck” → “new opening claim”

The paper explains §4.3 in the simplified setting \(Mz=y\) for \(z\in R_q^\mu\). In the actual PCS opening proof, §4.2 produces exactly such a relation, but with:

- a **stacked witness vector** \(z\) that bundles multiple unknowns (notably \(\hat w,\hat t,\hat z\) from Eq. (20)), and
- a stacked statement \(y\) and matrix \(M\) derived from public matrices \(A,B,D\), the opening point \(x\) (via \(a,b\)), the claimed opening value, and the verifier challenge \(c\).

Once the prover and verifier agree on that concrete \((M,y)\), §4.3 proceeds as:

1. **Ring switching introduces a quotient witness** \(r\) such that:

   \[
   Mz = y + (X^d+1)\cdot r \quad \text{over } Z_q[X].
   \]

2. **Digitize the quotient witness** \(r = \sum_{u'=0}^{\delta-1} b^{u'} r_{u'}\) and range-check each digit block.
3. Encode the (digitized) coefficient tables of \((z,(r_{u'})_{u'})\) into one multilinear object \(\tilde w\) (paper Eq. (21), with the digit index omitted in the paper’s notation).
4. Prove the linear and range constraints by sumcheck, which ends in an evaluation claim \(\tilde w(r^\*)\) at a random point \(r^\*\).
5. **Recursion boundary**: the next PCS subproblem is “open the commitment to \(\tilde w\) at point \(r^\*\)”.

### E.1 What the prover’s “witness” is after the first interaction rounds (ring switching + sumcheck)

If by “after the first rounds” you mean “after the verifier samples \(\alpha,\tau_0,\tau_1\) and sumcheck begins”, then the prover’s relevant witnesses are:

- **The stacked linear-relation witness** \(z\) from §4.2 (Eq. (20)’s unknown vector; in the paper’s naming, this is \((\hat w,\hat t,\hat z)\) with the implicit recomposition \(z := J\hat z\)).
- **The ring-switch quotient witness digits** \((r_{u'})_{u'\in[\delta]}\) satisfying:

  \[
  Mz - y = (X^d+1)\sum_{u'} b^{u'} r_{u'}.
  \]

These are not sent directly. Instead they are **committed** and then accessed *only* through:

- evaluations of the committed multilinear object \(\tilde w\) during sumcheck, and finally
- one PCS opening of \(\tilde w\) at the final sumcheck point \(r^\*\).

So the short answer is:

> After ring switching starts, the “PCS witness” (for the next recursion layer) is the committed multilinear polynomial \(\tilde w\) that encodes the coefficient table of the stacked witness \(z\) and the digitized quotient witness \((r_{u'})_{u'}\). Sumcheck reduces everything to opening \(\tilde w\) at one random point \(r^\*\).

### E.2 What “witness table embedding” (Eq. (21)) really must encode in the full PCS (not the simplified \(Mz=y\) story)

Eq. (21) defines \(e_w(u,\ell)\) in the simplified \((z,r)\) notation. For implementation, the important points are:

- The “\(z\)” rows are not just “some vector”; in the PCS they are the **stacked unknowns** of Eq. (20), i.e. the prover’s hidden objects that tie together:
  - the main commitment \(u\),
  - the aux commitment \(v\),
  - the opening equation,
  - the fold identities (18)–(19),
  - and the redecomposition via \(J\).
- The “\(r\)” rows are not optional: they are the **quotient witness** for the lifted equation over \(Z_q[X]\), and in the real protocol are digitized as \((r_{u'})_{u'}\) and range-checked.

In other words, the witness-table encoder we build should take as input:

- the stacked witness vector (call it `linear_relation_witness` instead of paper’s `z`), and
- the digitized quotient witness blocks `quotient_digits` (instead of paper’s `r`),

and output a padded, evaluation-form multilinear object \(\tilde w\).

### E.3 Why the PoC’s “next witness” contains “extra quotient terms” (and why they don’t contradict the paper)

The PoC does not implement §4.3 for a single abstract relation \(Mz=y\). It already constructs (a variant of) Eq. (20)’s **stacked linear system**, which bundles multiple constraints.

Each lifted constraint row contributes its own quotient polynomial(s) when you rewrite it as:

\[
\text{(row LHS)} - \text{(row RHS)} = (X^d+1)\cdot r_i(X),
\]

so the PoC ends up with several quotient vectors (for several different stacked sub-constraints), and it concatenates them into the “next witness table” before padding to a power of two.

This is an implementation manifestation of the same principle the paper uses:

- §4.2 stacks many checks into one linear system (Eq. (20)),
- §4.3 introduces quotient witnesses for that system when lifted to \(Z_q[X]\),
- §4.3 then commits to a single coefficient-table object \(\tilde w\) representing *all* witness coordinates and *all* quotient-digit coordinates.

So: seeing “more quotient chunks” in a prototype is expected whenever you expand the simplified \(Mz=y\) exposition into the concrete PCS relation.

### E.4 Concrete “before we code” checklist (MVP up through §4.3)

To avoid implementing §4.3 machinery “in the dark”, we should pin down the following concrete spec items first.

#### E.4.1 Fix the exact stacked linear relation produced by §4.2

We need a concrete `LinearRelationInstance` spec with:

- **Public statement**:
  - the stacked matrix \(M\in R_q^{n\times \mu}\),
  - the stacked right-hand side \(y\in R_q^n\),
  - and the coefficient smallness parameters (base \(b\), digit length \(\delta\), and any redecomposition \(\tau\) where relevant).
- **Witness semantics**:
  - what each coordinate of the stacked witness vector means (e.g., slices corresponding to \(\hat w,\hat t,\hat z\)),
  - and which coordinates are subject to the range-check polynomial \(P_b\).

Without this, we cannot correctly define how many “\(z\) rows” Eq. (21) has (the paper’s \(\mu\)).

#### E.4.2 Fix the quotient witness structure introduced by ring switching

For the chosen \((M,y)\), ring switching requires:

- the quotient witness vector \(r\in (Z_q[X]_{<d})^n\) satisfying \(Mz-y=(X^d+1)r\),
- its digitization \(r=\sum_{u'} b^{u'} r_{u'}\),
- and the exact flattened indexing for the digit blocks (how we map \((u',i,\ell)\) into the witness-table’s row/col indices).

This determines the total number of rows in the coefficient table: \(\mu + n\cdot \delta\) (digit-aware), not just \(\mu+n\).

#### E.4.3 Fix the committed object for sumcheck: \(\tilde w\) layout + padding

We should document (and then implement) a precise `WitnessCoeffTableLayout`:

- `rows_z = mu`
- `rows_r_digits = n * delta`
- `cols = d`
- `total_entries = (rows_z + rows_r_digits) * cols`
- `padded_len = next_power_of_two(total_entries)`
- a deterministic row-major flattening \((row, col) -> idx\) to become “evaluations on the hypercube”.

This is the “witness table embedding” deliverable: it is the object whose commitment is opened at the end of sumcheck (Figure 6 / Figure 7).

#### E.4.4 Fix the public multilinear encodings needed by §4.3 constraints

To define \(H_\alpha\) and \(H_0\) (Eqs. (22)–(23)), we need:

- `AlphaPowers`: the table \(e_\alpha(\ell)=\alpha^\ell\),
- `LinearCoeffEncoding`: the digit-aware coefficient function \(e^{dig}_{M_\alpha}(i,u)\) (i.e., how \(M(\alpha)\) and \(-(\alpha^d+1)\cdot b^{u'}\) are encoded),
- and the equality polynomials \(e_{eq}\) used for batching.

#### E.4.5 Fix how sumcheck’s final oracle check becomes a PCS opening claim

Our sumcheck core deliberately stops at “here is the final point \(r^\*\)”; the ring-switch module must:

- compute the expected final value using the public parts, and
- reduce to a **single opening claim** of the committed \(\tilde w\) at \(r^\*\).

This is exactly where the PCS prover/verifier “open-check” logic plugs in (currently stubbed in this repo).

## Modulus switching / cross-prime sumcheck (Jolt-motivated extension; not in the Hachi paper)

This section sketches how to adapt the Hachi “ring switch \(\to\) sumcheck” pipeline to the setting where:

- **Commitments** are over a *small* prime field / ring modulus \(q\) (e.g. \(\approx 2^{32}\)), because that makes commitment-time arithmetic and NTT/CRT layouts fast.
- **Sumcheck / arithmetization** must run over a *large* prime field \(F_{q'}\) (e.g. 128-bit prime), because the application (e.g. Jolt) requires characteristic large enough to avoid wrap-around in \(u64\cdot u64\) accumulation.

This is *similar in spirit* to §3’s extension-field story, but **strictly harder**: there is no field embedding \(F_q \hookrightarrow F_{q'}\) that preserves addition/multiplication mod the prime, so we must explicitly control an **integer lift** via range checks.

### F.0 Target statement (“foreign-field opening”)

Let \(q\) be a small prime, \(q'\) a large prime, and let \(f\) be an \(\ell\)-variate multilinear with **small coefficients**, ideally bits:

\[
f \in F_q^{\le 1}[X_1,\dots,X_\ell],\quad f_i \in \{0,1\}\subset F_q.
\]

We commit to the coefficient table \((f_i)_{i\in\{0,1\}^\ell}\) using the Hachi/Greyhound-style commitment core over \(R_q\) (or over \(F_q\) as the \(d=1\) special case).

The opening claim we want to support is over the **large prime field**:

\[
\text{given } x\in F_{q'}^\ell,\ y\in F_{q'},\ \text{prove } f(x)=y\ \text{(interpreting the coefficients as small integers in }F_{q'}\text{).}
\]

Because the coefficients are in \(\{0,1\}\), there is a canonical injection \(\iota:\{0,1\}\to F_{q'}\) (map to the same integers). The only remaining job is to enforce that the committed coefficients are indeed in \(\{0,1\}\) (bitness) and that every algebraic check is performed with respect to this integer lift.

#### F.0.1 Important clarification: this is **not** “digit-decompose \(x,y\)”

It is tempting to think “we have an evaluation claim \(f(x)=y\) in the large field \(F_{q'}\), so we should decompose \(x\) and \(y\) into base-\(b\) digits and prove digit-wise subclaims.” That is **not** what the modulus-switching reduction does, and it generally does not work (polynomial evaluation does not decompose into independent digit evaluations).

Instead:

- The evaluation point \(x\in F_{q'}^\ell\) and claimed value \(y\in F_{q'}\) are already *native* to the field where Jolt runs sumcheck. There is typically no reason to decompose them.
- The lift issue arises because the **committed objects** live over the *small* modulus \(q\) (i.e. values are only defined modulo \(q\)), while the verifier wants to check equations **inside \(F_{q'}\)**.

So the purpose of \(\mathrm{lift}_q\) is to assign a **canonical integer meaning** to committed mod-\(q\) values (made unambiguous by range constraints like “bitness”), so those values can be interpreted inside \(F_{q'}\).

Warm-up (scalar version). For integers \(\tilde A,\tilde B\in Z\), the statement “\(A=B\) in \(Z_q\)” is exactly:

\[
\tilde A \equiv \tilde B \pmod q
\quad\Longleftrightarrow\quad
\exists s\in Z:\ \tilde A - \tilde B = q\cdot s.
\]

The \(q\cdot s\) term is the “modulus switching slack”. The ring/polynomial version in F.1 is the same idea applied coefficient-wise (and with an additional cyclotomic slack \((X^d+1)\cdot r\) for the ring quotient).

#### F.0.2 How this plugs into Hachi’s six core constraints (B.4.1) in the Jolt Stage-8 setting

Jolt Stage 8 asks the PCS to prove openings of the form “\(P(\mathbf r)=v\)” where both the point \(\mathbf r\) and value \(v\) live in the **Jolt field** \(F\) (in your target case, \(F=F_{q'}\)). See:

- `../jolt/jolt-core/src/poly/opening_proof.rs`: `pub type Opening<F> = (OpeningPoint<..., F>, F);`
- `../jolt/jolt-core/src/zkvm/prover.rs`: Stage 8 calls `PCS::prove(..., &opening_point.r, ...)` where `opening_point.r` is a vector of field challenges.

Hachi’s Step B constraints (the six items in B.4.1) are written as equalities over the **ring** \(R_q\). To use a Hachi-style PCS under Jolt’s interface, we keep the *same witness objects* \((\hat w,\hat t,\hat z,\dots)\) and the *same logical constraints*, but we do **not** expect the verifier to check them natively in \(R_q\) (and we cannot even form the “\(a,b\in R_q\) from the opening point” parts when \(\mathbf r\in F_{q'}^\ell\)).

Instead, the PCS opening proof checks these constraints **after ring switching**, i.e. after applying an evaluation map

\[
\mathrm{ev}_\alpha: R_q \to F_{q'}
\]

at a random \(\alpha\in F_{q'}\) (and including the modulus-switch slack \(q\cdot s\) to make “mod \(q\)” equalities meaningful in \(F_{q'}\)).

Concretely:

- Constraints **(1), (2), and the purely ring-linear parts** of (4),(5) are still “ring equations”, but they are verified in \(F_{q'}\) by checking \(\mathrm{ev}_\alpha(\text{LHS}-\text{RHS})=0\) (with quotient witnesses for \((X^d+1)\) and \(q\) as in F.1).
- Constraints **(3)–(5)** are the ones that *mention the evaluation point* (via the monomial vectors \(a,b\)) and therefore must be interpreted in the **field**:
  - compute \(a,b\) directly from the Stage-8 opening point \(\mathbf r\in F_{q'}^\ell\),
  - treat unknown ring elements like \(w_i\in R_q\) only through their field images \(w_i(\alpha):=\mathrm{ev}_\alpha(w_i)\in F_{q'}\),
  - and enforce the same algebraic equalities (Eq. (17)–(19)) in \(F_{q'}\).

So the “generalization to account for \(q'\)” is not “change a few constraints while keeping the rest in \(q\)”; it is:

> keep the constraint *structure*, but **move their verification domain** from \(R_q\) to \(F_{q'}\) via ring switching (plus a \(q\cdot s\) slack), because Stage 8’s statement lives in \(F_{q'}\).

#### F.0.3 Why you cannot keep the *entire* opening proof “purely in \(F_q/R_q\)” under Jolt’s interface

Under Jolt, the opening point \(\mathbf r\) is sampled as transcript challenges in the **same field as the sumchecks**, i.e. \(\mathbf r\in F_{q'}^m\) (see `OpeningPoint<..., F>` in `../jolt/jolt-core/src/poly/opening_proof.rs`).

Any PCS that plugs into Stage 8 must therefore convince the verifier of a statement that is *parameterized by* these \(F_{q'}\) elements:

\[
P(\mathbf r)=v\quad\text{for }\mathbf r,v\in F_{q'}.
\]

If a verifier refuses to do any \(F_{q'}\) arithmetic, it cannot even *evaluate the public weights* (equality polynomials) at \(\mathbf r\), nor can it check the final identity that defines “evaluation at \(\mathbf r\)”. The only way around this would be to represent every \(F_{q'}\) element (including \(\mathbf r\), \(v\), and all derived weights) as **non-native data** over \(F_q\) (e.g. base-\(b\) limbs) and then prove that these limbs satisfy the mod-\(q'\) arithmetic via additional quotient/carry constraints.

That “all-\(F_q\) verification” route is a different, much heavier design: it is essentially a SNARK for \(F_{q'}\) arithmetic implemented over \(F_q\), and it introduces digit/carry constraints for every multiplication/addition in the opening protocol. This is *not* what Hachi’s ring switching is optimizing for.

So the realistic design space is:

- commitments / witness representation over \(F_q\) / \(R_q\) for performance,
- but verifier-side checking (sumcheck challenges, equality weights, and final identities) in \(F_{q'}\).

#### F.0.4 Recursion / “next folding steps” in the cross-prime setting

In Hachi, the “recursion boundary” is: sumcheck reduces a large set of constraints to **one new opening claim** for a committed multilinear object (the \(\tilde w\) table). The next layer repeats the same pattern on a smaller witness.

In the cross-prime Stage-8 adaptation:

- the **commitment domain stays** \(F_q/R_q\) at every layer (you keep committing to digitized tables over the small modulus),
- the **opening points stay** in \(F_{q'}\) at every layer, because they are derived from sumcheck challenges / transcript in \(F_{q'}\),
- and the verifier continues to check constraints in \(F_{q'}\) via ring switching (evaluation at \(\alpha\in F_{q'}\) plus the \((X^d+1)\cdot r\) and \(q\cdot s\) quotient witnesses).

So “continuing in \(F_{q'}\)” does not mean “switch commitments to \(q'\)”; it means the *interactive checking algebra* (sumcheck, batching, evaluation weights) remains in the Jolt field where the statement is expressed.

#### F.0.5 One full “folding/recursion step” of the adapted PCS (clean, end-to-end)

This subsection gives a clean one-layer view: how we go from **one opening claim at point \(\mathbf r\in F_{q'}^m\)** to **a smaller opening claim at a new point \(\mathbf r^\*\in F_{q'}^{m'}\)**, and why that output is amenable to repeating the same procedure.

We describe this as a PCS protocol (what Stage 8 wants), independent of Jolt’s earlier stages.

##### Inputs (statement) and commitment domain

- **Commitment domain** (small): commitments are to coefficient tables over \(F_q\) (or ring elements over \(R_q\)) using Hachi’s §4.1 Ajtai-style structure.
- **Opening statement domain** (large): the opening point \(\mathbf r\) and claimed value \(v\) are in \(F_{q'}\) (Jolt’s `JoltField`).

So the opening statement is:

\[
\text{Given commitment } C \text{ to a multilinear table } P,\ \text{and public } \mathbf r\in F_{q'}^m,\ v\in F_{q'},\ \text{prove } P(\mathbf r)=v.
\]

Here \(P(\mathbf r)\) is defined by interpreting the committed coefficients as integers (via \(\mathrm{lift}_q\)) and reducing them into \(F_{q'}\); for bit/one-hot tables, this interpretation is canonical.

##### Step 1 (Hachi Step B “split-and-fold”): reduce evaluation to a structured linear relation witness

This step is unchanged in *spirit* from Hachi: we rewrite “\(P(\mathbf r)=v\)” into a small set of algebraic constraints involving intermediate witnesses (partial evaluations, decomposed digits, and folded combinations).

The key point in the two-field setting is typing:

- any time the original §4.2 constraints multiply “point-derived scalars” into witness quantities, those scalars are in \(F_{q'}\) and we interpret the witness quantities through their images in \(F_{q'}\) (via ring switching in Step 2), rather than trying to multiply \(F_{q'}\) scalars by \(R_q\) elements directly.

Operationally, the prover still constructs the same witness objects (\(\hat w,\hat t,\hat z\), folded \(z\), etc.) and commits to the same ring elements (\(u,v\), etc.). The verifier will not check the ring equations directly; it will check their *ring-switched* images in Step 2.

##### Step 2 (Hachi Step C generalized): ring switching + modulus switching to get field-native constraints

The verifier samples a random \(\alpha\leftarrow F_{q'}\) and defines the evaluation map:

\[
\mathrm{ev}_\alpha: R_q \to F_{q'}
\]

as “lift coefficients to integers (per chosen \(\mathrm{lift}_q\)), then evaluate the polynomial at \(X=\alpha\) in \(F_{q'}\)”.

Now take each ring equation from the Step B constraint set (the six constraints in B.4.1) and convert it into a field equation at \(\alpha\) by:

1. lifting it from \(R_q\) to \(Z[X]\) with a cyclotomic quotient witness \(r\) (as in the paper), and
2. adding a modulus quotient witness \(s\) so that “mod \(q\) equality” becomes an *integer* equality plus a \(q\cdot s\) slack (Section F.1).

After evaluating at \(X=\alpha\), every check becomes an identity in \(F_{q'}\) involving:

- public scalars derived from \(\mathbf r\) (equality weights / monomial vectors),
- the prover’s unknowns only through \(\mathrm{ev}_\alpha(\cdot)\),
- and the quotient witnesses \(r(\alpha), s(\alpha)\).

##### Step 3 (Hachi §4.3 sumcheck): batch all constraints and reduce to one oracle evaluation

Exactly as in Hachi, we encode the relevant coefficient tables into one committed multilinear object \(\tilde w\) (“witness table embedding”), except that in the cross-prime setting \(\tilde w\) must encode the extra modulus quotient witness \(s\) as well as \(r\).

Then we define batched constraint polynomials (the analogs of \(H_\alpha\) and \(H_0\)), and run sumcheck over **\(F_{q'}\)** (not over \(R_q\)). The sumcheck transcript produces a final random point:

\[
\mathbf r^\* \in F_{q'}^{m'}
\]

and reduces verification to a **single evaluation claim** of the committed witness-table multilinear:

\[
\tilde w(\mathbf r^\*) = v^\*\in F_{q'}.
\]

##### Step 4 (recursion boundary): output a smaller opening claim of the same type

This is the crucial “amenable to further folding” point:

- The new claim “\(\tilde w(\mathbf r^\*)=v^\*\)” has the **same shape** as the original opening claim, just on a different committed multilinear and a different point.
- The commitment to \(\tilde w\) is again over the **small modulus domain** (it is built from digitized tables over \(F_q/R_q\)).
- The opening point \(\mathbf r^\*\) is again in the **large field** \(F_{q'}\), because it is derived from the sumcheck challenges (and thus from the Jolt transcript).

Therefore, we can repeat the same 4-step pipeline on \(\tilde w\):

\[
P(\mathbf r)=v
\ \leadsto\
\tilde w(\mathbf r^\*)=v^\*
\ \leadsto\
\tilde w^{(2)}(\mathbf r^{\*\*})=v^{\*\*}
\ \leadsto\ \cdots
\]

and eventually hand off to a base PCS / small-instance prover once the witness table is small enough (exactly the same “stop recursion when small” design choice as Hachi’s §5 composition discussion).

#### F.0.6 Option B (for comparison): decompose the \(F_{q'}\) opening point/value and prove everything over \(F_q/R_q\)

This subsection works out the alternative you asked for: **avoid doing any verifier arithmetic in \(F_{q'}\)** by representing all \(F_{q'}\) elements (the opening point \(\mathbf r\) and claimed value \(v\), plus all derived weights) as *digits/limbs* over the small field.

This is not “ring switching”; it is **non-native (foreign-field) arithmetic**: we simulate mod-\(q'\) arithmetic inside a proof system whose native arithmetic is mod-\(q\).

##### Why this is a different interface than Jolt Stage 8

In Jolt, Stage 8’s opening statement is parameterized by an actual \(\mathbf r\in F_{q'}^m\) sampled from the transcript (see `OpeningPoint<..., F>`).

Option B instead treats \(\mathbf r\) and \(v\) as **public limb vectors over \(F_q\)**. To plug this into Jolt unchanged, you would need to either:

- change Jolt’s transcript/challenges to live in the small field (not compatible with the stated “char \(> u64\cdot u64\)” requirement), or
- keep \(\mathbf r\in F_{q'}\) as usual but additionally provide a limb decomposition of \(\mathbf r\) and then **prove inside the PCS** that those limbs reconstruct the same \(\mathbf r\) (which reintroduces \(F_{q'}\) operations unless you also non-natively model the transcript).

So Option B is best viewed as a “theoretical comparison point”, not a drop-in Stage-8 replacement.

##### Representation: limbs for \(F_{q'}\) elements

Fix a radix \(B=2^t\) (e.g. \(B=2^{16}\) so limbs fit comfortably under a 32-bit-ish \(q\)), and let

\[
L := \left\lceil \log_B(q') \right\rceil
\]

be the limb count (for a 128-bit prime and \(B=2^{16}\), \(L\approx 8\)).

We represent a field element \(a\in F_{q'}\) by an integer representative \(\tilde a\in[0,q')\) (canonical lift mod \(q'\)) and a limb decomposition:

\[
\tilde a = \sum_{j=0}^{L-1} a_j B^j,\quad a_j\in\{0,1,\dots,B-1\}.
\]

All limbs \(a_j\) are then encoded as small elements of \(F_q\) (and range-checked to lie in \([0,B)\)).

##### Non-native arithmetic constraints (core gadget)

To simulate \(F_{q'}\) arithmetic, every operation becomes an integer identity plus a \(q'\)-multiple slack:

- **Addition**: to enforce \(c \equiv a+b \pmod{q'}\), prove

  \[
  \tilde a + \tilde b - \tilde c = q'\cdot k
  \]

  for some integer \(k\) (with \(k\in\{0,1\}\) if \(\tilde a,\tilde b,\tilde c\in[0,q')\)).

- **Multiplication**: to enforce \(c \equiv a\cdot b \pmod{q'}\), prove

  \[
  \tilde a\cdot \tilde b - \tilde c = q'\cdot k
  \]

  for some integer \(k\) with \(0 \le k < q'\).

In practice, you do not materialize \(\tilde a\) as a single huge integer; you enforce these identities **in base \(B\) with carries**:

- introduce carry variables \(u_t\) so that the schoolbook convolution of limbs matches the target limbs,
- and range-check carries so that “equality mod \(q\)” implies “equality over integers” (no wrap-around) at the limb level.

Cost model: one non-native multiplication of two \(L\)-limb numbers costs

\[
\Theta(L^2)\ \text{small-field multiplications} \quad + \quad \Theta(L)\ \text{carry/range constraints}.
\]

##### How to prove an opening claim \(P(\mathbf r)=v\) using Option B

Let \(P\) be a multilinear polynomial committed over \(F_q\) (or \(R_q\)) whose coefficients are small integers (bits/one-hot is the cleanest case).

Public input for Option B:

- limb decompositions of each coordinate \(r_i\in F_{q'}\) (so \(\tilde r_i=\sum_j r_{i,j}B^j\)),
- limb decomposition of \(v\in F_{q'}\),
- and the modulus \(q'\) itself (public constant).

Prover witness:

- limb decompositions for all intermediate \(F_{q'}\) values needed to evaluate \(P\) at \(\mathbf r\),
- plus all carry/quotient witnesses for the modular reduction constraints above.

Then enforce, inside the proof system over \(F_q/R_q\):

1. **Range**: every limb is in \([0,B)\) (bitness is the special case \(B=2\)).
2. **Recomposition**: the limb vectors correspond to integers in \([0,q')\) (often implemented by providing a quotient \(t\) such that \(\sum_j a_jB^j = \tilde a + q' t\) and constraining \(\tilde a<q'\)).
3. **Evaluation correctness**: encode an arithmetic circuit for multilinear evaluation using only additions and multiplications in \(F_{q'}\), and replace every such gate by non-native constraints over limbs.
4. **Commitment consistency**: connect the evaluation circuit’s “inputs” (the coefficient table) to the committed PCS witness over \(F_q/R_q\).

At the end you obtain: a proof *over \(F_q/R_q\)* that the non-native evaluation equals the public \(v\).

##### Why Option B is usually much more expensive than Option A

Option A keeps all “point-derived” weights and batching algebra native in \(F_{q'}\), and only uses digitization/range-checks for the *mod-\(q\)* commitment-side witnesses and quotient slacks.

Option B must additionally:

- represent the opening point \(\mathbf r\) itself in limbs,
- compute equality-polynomial weights / batching coefficients in limbs,
- and simulate every multiplication/addition that the opening proof needs, paying the \(\Theta(L^2)\) overhead per multiplication.

So the difference is qualitative:

- **Option A**: \(F_{q'}\) arithmetic is done natively; small-field/ring is used for commitments and for “smallness” enforcement.
- **Option B**: \(F_{q'}\) arithmetic is *itself* proved non-natively over \(F_q\), which can swamp the benefits unless the circuit is extremely small.

### F.1 Why this needs “modulus switching” (the extra quotient for \(q\))

Hachi ring switching (paper §4.3) lifts the **cyclotomic quotient**:

\[
Mz = y\ \text{in } R_q \iff
Mz = y + (X^d+1)\cdot r\ \text{over } Z_q[X].
\]

If we now want to check identities by evaluating at a random \(\alpha\in F_{q'}\), we face a second issue:

- an element of \(Z_q\) is only defined **mod \(q\)**, so interpreting it as an integer in \([0,q)\) inside \(F_{q'}\) is a *choice* (a lift),
- and a polynomial identity that holds “mod \(q\)” need not hold for the chosen lift unless we also account for multiples of \(q\).

The standard fix is to introduce an **additional modulus quotient witness** (call it \(s\)) and lift the equation to integers:

\[
\mathrm{lift}_q(M)\,\mathrm{lift}_q(z) \;-\; \mathrm{lift}_q(y)
\;=\;
(X^d+1)\,r \;+\; q\cdot s
\quad\text{over } Z[X],
\]

where \(\mathrm{lift}_q(\cdot)\) chooses canonical representatives of coefficients, and where the witness polynomials \(r,s\) have degree \(<d\) and are range-checked / digit-decomposed so they correspond to genuine integers (not arbitrary \(F_{q'}\) values).

#### F.1.1 Two lift conventions (and how they change the \(q\cdot s\) term)

Fix an odd prime \(q\). A lift convention is a deterministic function:

\[
\mathrm{lift}_q: Z_q \to Z
\]

that picks one integer representative for each residue class mod \(q\). Two common choices:

1) **Nonnegative / canonical lift** (implementation-friendly):

\[
\mathrm{lift}_q^{\mathrm{can}}(a) \in \{0,1,\dots,q-1\}.
\]

2) **Centered / balanced lift** (often gives smaller absolute values):

\[
\mathrm{lift}_q^{\mathrm{ctr}}(a) \in \left\{-\tfrac{q-1}{2},\dots,\tfrac{q-1}{2}\right\}.
\]

These differ by a multiple of \(q\). In fact, for every \(a\in Z_q\) there exists a bit \(\epsilon(a)\in\{0,1\}\) such that:

\[
\mathrm{lift}_q^{\mathrm{can}}(a) = \mathrm{lift}_q^{\mathrm{ctr}}(a) + q\cdot \epsilon(a).
\]

Consequence: switching lift conventions does **not** change the *shape* of the modulus-switch identity; it only changes the required quotient witness \(s\).

Concretely, if a prover exhibits witnesses \((r,s)\) satisfying

\[
\mathrm{lift}_q^{\mathrm{can}}(M)\,\mathrm{lift}_q^{\mathrm{can}}(z) - \mathrm{lift}_q^{\mathrm{can}}(y)
=(X^d+1)r + q s,
\]

then the same underlying mod-\(q\) ring equation can also be written under the centered lift as

\[
\mathrm{lift}_q^{\mathrm{ctr}}(M)\,\mathrm{lift}_q^{\mathrm{ctr}}(z) - \mathrm{lift}_q^{\mathrm{ctr}}(y)
=(X^d+1)r + q s',
\]

for some (generally different) \(s'\). Intuitively, the “difference” between lifts is absorbed into the \(q\cdot s\) slack.

So: **either lift convention is viable**, but the coefficient bounds (and hence digitization depth) for \(s\) depend on the choice.

#### F.1.2 Where \(s\) comes from (coefficient-wise division by \(q\))

Once \(r\) accounts for the cyclotomic modulus, the remainder polynomial

\[
\Delta(X) := \mathrm{lift}_q(M)\,\mathrm{lift}_q(z) - \mathrm{lift}_q(y) - (X^d+1)r
\]

has degree \(<d\) and satisfies \(\Delta(X)\equiv 0\pmod q\) coefficient-wise (exactly the statement “the original equation holds in \(R_q\)”).
Therefore there exists \(s\) with degree \(<d\) such that \(\Delta(X)=q\cdot s(X)\) over \(Z[X]\).

The protocol must ensure \(s\) is a *genuine integer* polynomial (not a forged \(F_{q'}\) polynomial), via digitization + range checks.

#### F.1.3 Coefficient size heuristics under each lift (what it does to \(\delta_s\))

Write \(b\) for the digit base used in decomposition (e.g. \(b=16\)). Assume (as in Hachi §4.3) that:

- the unknown witness coordinates (digits of \(z\) and of the quotient witnesses) are bounded by \(\approx b\) in magnitude,
- the public matrix \(M\) is “random-looking” in \(Z_q\) (typical for Ajtai commitments), so its lifted coefficients have magnitude about:
  - \(\approx q/2\) under \(\mathrm{lift}_q^{\mathrm{ctr}}\),
  - \(\approx q\) under \(\mathrm{lift}_q^{\mathrm{can}}\) (worst-case), with mean \(\approx q/2\).

Then a very rough (worst-case) per-coefficient bound for the modulus slack is:

\[
|s[\cdot]| \lesssim \frac{|M\cdot z| + |y| + |(X^d+1)r|}{q}.
\]

Heuristically this scales like “(how many terms you sum) \(\times\) (digit bound) \(\times\) (ring-degree convolution factor)”, and **does not contain \(q\)** after dividing by \(q\).

Centered lift typically reduces absolute values by at most a constant factor (about 2), so it usually saves at most ~1 digit plane in \(\delta_s\).
Its bigger benefit is that it makes “reasonable” symmetric range-checks for signed digits feel more natural.

In either convention, \(s\) can have **negative coefficients** (because \(\Delta\) can be negative), so \(\delta_s\) must support signed integers (either via centered digit sets or an explicit sign representation).

Evaluating at \(\alpha\in F_{q'}\) yields a *field* equation:

\[
\mathrm{lift}_q(M)(\alpha)\,\mathrm{lift}_q(z)(\alpha)
=
\mathrm{lift}_q(y)(\alpha) + (\alpha^d+1)\,r(\alpha) + q\cdot s(\alpha)
\quad\text{in } F_{q'}.
\]

Now the verifier can run **sumcheck over \(F_{q'}\)** to enforce these equalities (plus the coefficient smallness constraints) without doing ring multiplication.

### F.2 What changes in the Hachi §4.3 “witness table” for sumcheck

In the paper’s simplified notation, §4.3 commits to a multilinear table encoding \((z,r)\) (with \(r\) digitized). With modulus switching, the committed table must encode **\((z,r,s)\)**:

- \(z\): the stacked linear-relation witness from §4.2 (digits / redecomposition as usual),
- \(r\): the cyclotomic quotient witness for \((X^d+1)\) (digitized as in the paper),
- \(s\): the modulus quotient witness for \(q\) (also digitized, with coefficient bounds).

So the “witness-table embedding” shape becomes:

- `rows_z = mu`
- `rows_r_digits = n * delta_r`
- `rows_s_digits = n * delta_s`
- `cols = d`

For a base \(b=2^{\texttt{LOG_BASIS}}\), one natural choice is:

\[
\delta_s \approx \lceil \log_b q\rceil.
\]

For example, if \(q\approx 2^{32}\) and \(b=16\), then \(\delta_s=8\) digit planes.

### F.3 Smallness constraints get easier in the “bit / one-hot” regime

If the committed coefficients are bits (as in the one-hot regime in `docs/ONE_HOT_COMMITMENT_COST_AND_GPU_PRG.md`), then the “range check” polynomial used inside the sumcheck constraints can be as small as:

\[
P_{\{0,1\}}(T) := T(T-1),
\]

instead of a degree-\((2b-1)\) root-check polynomial over \([-(b-1),\dots,(b-1)]\).

This significantly reduces the algebraic overhead of enforcing “integer lift is well-defined” for the *data*.
The remaining overhead comes from digitizing / range-checking the quotient witnesses \((r,s)\).

### F.4 Complexity headline (relative to vanilla Hachi §4.3)

- **Sumcheck rounds**: unchanged in form; still logarithmic in the witness-table size. (The field is now \(F_{q'}\) instead of \(F_{q^k}\).)
- **Extra witness size**: adds the \(s\) block, i.e. roughly an extra \(n\cdot d\cdot \delta_s\) coefficients in the committed table.
- **Prover time**: increases proportionally to the larger committed witness table and the extra constraints (roughly “one more quotient family” to enforce).
- **Verifier time**: remains polylog in witness size (sumcheck verifier work), plus the final PCS opening(s), but arithmetic is in the 128-bit prime field.

### F.5 What must be pinned down to make this fully rigorous (checklist)

To turn this into a proof, we must specify:

1. **Canonical lift convention** \(\mathrm{lift}_q\): centered vs \([0,q)\); this affects coefficient bounds for \(s\).
2. **Coefficient bounds** for \(r\) and \(s\) after lifting, and therefore the digitization depths \(\delta_r,\delta_s\) and range-check polynomials.
3. **Soundness lemma**: degree bound of the lifted difference polynomial (still \(O(d)\)) and resulting soundness error \(\le O(d)/q'\) when \(\alpha\leftarrow F_{q'}\) is uniform.
4. **Exact constraint polynomials** (the \(H_\alpha\)-style batching) updated to include the \(q\cdot s(\alpha)\) term.
5. **Integration point with the application sumcheck (e.g. Jolt)**: the PCS must support openings at points in \(F_{q'}^\ell\); the “foreign-field opening” layer above is intended to provide exactly that by running Hachi’s verifier-side checks natively over \(F_{q'}\).

