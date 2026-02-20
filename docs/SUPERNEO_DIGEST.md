# SuperNeo Digest for Hachi

This file captures the parts of `docs/superneo.pdf` that are most actionable for Hachi's algebra layer, in a compact AI-readable format.

## Canonical parameter tuples (from paper)

```yaml
superneo_parameter_tuples:
  - id: almost_goldilocks
    field_modulus_q: "2^64 - 2^32 - 31"
    prime: true
    strict_solinas_2x_minus_2y_plus_1: false
    cyclotomic:
      eta: 128
      phi: "X^64 + 1"
      degree_d: 64
    extension_field: "F_{q^2}"
    challenge_coeff_set: [-1, 0, 1, 2]
    expansion_factor_T: 128
    security_estimate: "~129-bit Module-SIS"

  - id: goldilocks
    field_modulus_q: "2^64 - 2^32 + 1"
    prime: true
    strict_solinas_2x_minus_2y_plus_1: true
    cyclotomic:
      eta: 81
      phi: "X^54 + X^27 + 1"
      degree_d: 54
    extension_field: "F_{q^2}"
    challenge_coeff_set: [-2, -1, 0, 1, 2]
    expansion_factor_T: 216
    security_estimate: "~129-bit Module-SIS"

  - id: mersenne61
    field_modulus_q: "2^61 - 1"
    prime: true
    strict_solinas_2x_minus_2y_plus_1: true
    cyclotomic:
      eta: 81
      phi: "X^54 + X^27 + 1"
      degree_d: 54
    extension_field: "F_{q^2}"
    challenge_coeff_set: [-2, -1, 0, 1, 2]
    expansion_factor_T: 216
    security_estimate: "~129-bit Module-SIS"
```

## Notation glossary (from `docs/superneo.pdf`)

This is a quick map from the paper’s symbols to “what it is”. The paper uses **both** \(K\) (extension field) and \(K\) (number of folded instances); below I disambiguate in words.

### Fields and rings

- **\(F\)**: base prime field \(F = \mathbb{F}_q\).
- **\(q\)**: the prime modulus defining \(F\).
- **\(K\) (field)**: the *smallest-degree extension field* of \(F\) such that \(1/|K| = \mathrm{negl}(\lambda)\). In all appendix parameter sets they take **\(K = \mathbb{F}_{q^2}\)**.
- **\(\eta\)**: cyclotomic index.
- **\(\Phi(X)\)**: the \(\eta\)-th cyclotomic polynomial over \(F\).
- **\(d\)**: ring degree, i.e. \(d = \deg \Phi = \varphi(\eta)\).
- **\(R_F\)**: cyclotomic ring \(R_F := F[X]/(\Phi(X))\).
- **\(R_K\)**: same ring over the extension field \(R_K := K[X]/(\Phi(X))\).

### Dimensions / sizes

- **\(m\)**: number of constraints / rows in CCS matrices (paper often sets \(m = n_F\) and assumes it’s a power of two for sum-check convenience).
- **\(t\)**: number of CCS matrices \(M_1,\dots,M_t\).
- **\(u\)**: degree bound of the CCS “constraint polynomial” \(f\) in \(t\) variables.
- **\(n_R\)**: ring-vector length (number of ring elements in a committed vector).
- **\(n_F\)**: field-vector length; tied by **\(n_F = d\cdot n_R\)** under coefficient embedding.
- **\(n_{R,\mathrm{in}}\), \(n_{F,\mathrm{in}}\)**: input lengths (public part of the witness vector), again related by \(n_{F,\mathrm{in}} = d\cdot n_{R,\mathrm{in}}\).

### Commitments and homomorphisms

- **\(\kappa\)**: Ajtai commitment “height”: the public matrix is \(A \in R_F^{\kappa \times n_R}\), and a commitment is \(c = A z \in R_F^\kappa\).
- **\(\mathrm{com}=(\mathrm{Setup},\mathrm{Commit})\)**: Ajtai commitment scheme over \(R_F\).
- **\(L\)**: the module-homomorphism “commitment map” \(L(z) := \mathrm{Commit}(pp,z)\).
- **\(\mathrm{Lin}\)**: the trivial projection that takes the “input part” of a ring vector (first \(n_{R,\mathrm{in}}\) coordinates).

### Norm bounds and decomposition

- **\(b\)**: the “small” \(\ell_\infty\) bound for committed openings (binding holds only for \(\|z\|_\infty < b\)).
- **\(k\) (decomposition depth)**: number of base-\(b\) digits in the decomposition reduction; also used as the number of accumulator claims carried between folds.
- **\(B\)**: the “large” bound after random linear combination, defined as **\(B = b^k\)**.
- **\(\mathrm{split}_b(\cdot)\)**: base-\(b\) digit decomposition used by \(\Pi_{\mathrm{DEC}}\).

### Folding instance counts (the other \(K\))

- **\(K\) (instance count)**: number of *fresh* CCS instances folded in one step (multi-folding fan-in). In the appendix they give ranges like “\(K\in[61]\)” meaning “up to 61 instances per fold”.
- **\(k\)**: number of accumulator evaluation claims carried along (same \(k\) as in \(B=b^k\)).
- **\(K+k\)**: total number of witness vectors/evaluation-claims that show up inside \(\Pi_{\mathrm{CCS}}\).

### Evaluation / sum-check symbols (per fold)

- **\(z \in F^{n_F}\)**: underlying field witness vector; under the SuperNeo coefficient embedding it corresponds to a ring vector in \(R_F^{n_R}\).
- **\(c \in R_F^\kappa\)**: commitment to \(z\) (via Ajtai).
- **\(r \in K^{\log m}\)**: random evaluation point (from sum-check).
- **\(y_j \in R_K\)**: lifted evaluation claims \(y_j = \bar M_j \tilde z(r)\) (one per matrix \(M_j\)).
- **\(\alpha \in K^{\log m}\), \(\gamma \in K\)**: random challenges used inside \(\Pi_{\mathrm{CCS}}\).
- **\(r'\)**: the final random point produced by sum-check.

### Challenge set / expansion

- **\(C \subseteq R_F\)**: “strong sampling set” of short ring elements used as folding challenges.
- **\(T\)**: expansion factor of \(C\) (upper bounds norm growth under multiplication by \(\rho\in C\)).
- **“challenge_coeff_set”** in the YAML above: concrete coefficient alphabet defining \(C\) (e.g. \([-2,-1,0,1,2]\)).

## Protocol overview (what gets folded, and what is sent)

This section summarizes **SuperNeo’s folding scheme** (paper §7.3–§7.5) at the level of:

- what relation is being accumulated,
- how one folding step is structured (Π\_CCS, Π\_RLC, Π\_DEC),
- and what the *proof payload* is per step (in terms of ring/field elements).

### Relations: CCS vs CE (accumulator relation)

SuperNeo folds a batch of **fresh CCS instances** into a running accumulator made of **CCS evaluation claims**.

- **CCS(b, L)** (paper Def. 12): instance is \((c, x)\) where \(c = L(z)\) is a commitment to witness vector \(z=[x,w]\in F^{n_F}\), with \(\|z\|_\infty < b\), and CCS constraints hold.
- **CE(b, L)** (paper Def. 13): instance is \((c, x, r, \{y_j\}_{j\in[t]})\) where \(r\in K^{\log m}\) is a sum-check point and each \(y_j \in R_K\) is a lifted evaluation claim \(y_j = \bar M_j \tilde z(r)\).

The important “shape” fact:

- A CE claim stores **\(t\) ring elements in \(R_K\)**, not \(t\) field elements. Each \(R_K\) element has degree \(d\), i.e. \(d\) coefficients in the extension field \(K\).

### One folding step: ΠDEC ∘ ΠRLC ∘ ΠCCS

Each fold step conceptually does:

1. **Π\_CCS** (strong interactive reduction): verify CCS + norm + prior evaluation claims via one sum-check over \(K\); output **\(K+k\)** new CE claims at a fresh point \(r'\).
2. **Π\_RLC** (weak interactive reduction): take a random linear combination of those \(K+k\) CE claims using short ring challenges \(\rho_i \in C\); output **one** CE claim but with larger norm bound \(B=b^k\).
3. **Π\_DEC** (reduction of knowledge): base-\(b\) decompose that one big-norm CE claim back into **\(k\)** small-norm CE claims (norm bound \(b\)), which become the next accumulator state.

Diagram:

```
fresh batch:   CCS(b,L)^K        accumulator: CE(b,L)^k
        \           |                 /
         \          |                /
          \      Π_CCS (sum-check)  /
           \        |              /
            -->   CE(b,L)^(K+k)  --
                      |
                   Π_RLC
                      |
                  CE(B,L)
                      |
                   Π_DEC
                      |
             next accumulator: CE(b,L)^k
```

### What the prover sends in each sub-protocol

Below is the communication “payload” you should have in mind (Fiat–Shamir makes verifier randomness transcript-derived, but sizes are the same).

#### Π_CCS (paper §7.3)

Prover payload:

- **Sum-check transcript over \(K\)** for \(\log m\) rounds (one univariate polynomial per round; degree is small in typical CCS, e.g. R1CS-like \(u=2\)).
- **Oracle answers after sum-check**: for every claim \(i\in[K+k]\) and every matrix \(j\in[t]\), send
  \[
  y'_{i,j} = \bar M_j \tilde z_i(r') \in R_K.
  \]

Rule of thumb: the \(y'_{i,j}\) payload dominates.

#### Π_RLC (paper §7.4)

This is just random linear combination using \(\rho_i\in C\):

\[
c := \sum_i \rho_i c_i,\quad y_j := \sum_i \rho_i y_{i,j},\quad z := \sum_i \rho_i z_i.
\]

Prover payload: essentially **nothing new** (the reduction is “algebraic bookkeeping”; in FS the \(\rho_i\) come from the transcript).

#### Π_DEC (paper §7.5)

Prover payload:

- Decompose \(z\) into digits \((z_1,\dots,z_k)=\mathrm{split}_b(z)\).
- Send **\(k\) new commitments** \(c_i = L(z_i)\).
- Send **\(k\cdot t\)** new lifted evaluations \(y_{i,j} = \bar M_j \tilde z_i(r)\in R_K\).

### Proof size accounting (symbolic, per fold step)

Let:

- commitment output be \(c\in R_F^\kappa\) (Ajtai: \(\kappa\) ring elements over \(R_F\)),
- each lifted evaluation be one ring element in \(R_K\) (degree \(d\) over \(K\)),
- \(\ell := \log m\) be the sum-check arity.

Then per fold step, prover sends approximately:

- **Π\_CCS**:
  - sum-check: \(O(\ell)\) field elements in \(K\),
  - evaluations: \((K+k)\cdot t\) ring elements in \(R_K\).
- **Π\_DEC**:
  - commitments: \(k\cdot \kappa\) ring elements in \(R_F\),
  - evaluations: \(k\cdot t\) ring elements in \(R_K\).

Total (dominant terms):

\[
\text{#}(R_K\text{-elements}) \approx (K+k)t + kt = (K+2k)t,
\]
\[
\text{#}(R_F\text{-elements}) \approx k\kappa,
\]
plus a small \(\tilde O(\log m)\) number of \(K\)-field elements from sum-check.

To convert to bytes for appendix parameter sets where \(K=\mathbb{F}_{q^2}\) and \(q\) is 61–64 bits:

- 1 \(F\)-element ≈ 8 bytes
- 1 \(K\)-element ≈ 16 bytes
- 1 \(R_F\)-element ≈ \(d\cdot 8\) bytes
- 1 \(R_K\)-element ≈ \(d\cdot 16\) bytes
- 1 commitment \(c\in R_F^\kappa\) ≈ \(\kappa\cdot d\cdot 8\) bytes

So a back-of-the-envelope fold proof byte size is:

\[
\approx (K+2k)t\cdot(d\cdot 16)\;+\;k\kappa\cdot(d\cdot 8)\;+\;\tilde O(\log m)\cdot 16.
\]

## Exact counts

- Total concrete field/cyclotomic tuples in SuperNeo appendix: **3**
- Unique base fields in those tuples: **3**
- Unique cyclotomic polynomials in those tuples: **2**
  - `X^64 + 1`
  - `X^54 + X^27 + 1`
- Strict Solinas-only tuples (`q = 2^x - 2^y + 1`): **2 / 3**
  - Goldilocks (`2^64 - 2^32 + 1`)
  - Mersenne61 (`2^61 - 1 = 2^61 - 2^1 + 1`)
- Non-Solinas tuple: **1 / 3**
  - Almost-Goldilocks (`2^64 - 2^32 - 31`)

## What SuperNeo suggests (for their purpose)

- Keep sum-check and norm-check over a small field (or small extension), not over ring arithmetic.
- Do not restrict to only power-of-two cyclotomics (`X^d + 1`); using broader cyclotomics can unlock better field compatibility.
- For Goldilocks/M61, the paper uses a trinomial cyclotomic (`X^54 + X^27 + 1`) instead of `X^d + 1`.
- They explicitly discuss why power-of-two cyclotomics are problematic for some small fields (full splitting / security issues in prior designs).

## Fit into Hachi design (easy vs harder)

- Easy fit now:
  - Almost-Goldilocks-like setup with `d = 64` and `X^64 + 1` is structurally close to Hachi's current ring form.
- Requires refactor:
  - Goldilocks and M61 with `X^54 + X^27 + 1` do not match the current hardcoded negacyclic `X^D + 1` shape.

## Minimal integration plan into current architecture

1. Add a cyclotomic profile abstraction (`CyclotomicProfile`) that defines modulus polynomial behavior.
2. Keep current profile as `Negacyclic<D>` (`X^D + 1`) for existing NTT/CRT path.
3. Add a `Trinomial54` profile (`X^54 + X^27 + 1`) for SuperNeo-style field/ring experiments.
4. Keep backend split:
   - current CRT+NTT backend for `Negacyclic<D>` only,
   - coefficient-domain backend first for `Trinomial54` (no forced NTT dependency).
5. Add explicit domain aliases for profile-bound rings so APIs remain clear (`CoeffDomain<Profile, F>`).

## Practical implication if we insist on Solinas-only fields

- Within SuperNeo's concrete options, we still have **2 viable Solinas choices**:
  - Goldilocks + `X^54 + X^27 + 1`
  - M61 + `X^54 + X^27 + 1`
- If we also insist on power-of-two cyclotomics only (`X^d + 1`), SuperNeo's concrete Solinas count drops to **0**.
