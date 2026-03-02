# Clean two-field opening spec for \(d>1\) (e.g. \(d=1024\))

This note is a **correctness-first** (not performance-first) specification of a Hachi-like PCS interface in the **genuine two-prime** setting:

- commitment / witness modulus: a small prime \(q\approx 2^{32}\) and ring \(R_q := \mathbb Z_q[X]/(X^d+1)\),
- verifier / transcript / sumcheck field: a large prime field \(F_{q'}\) with \(q'\approx 2^{128}\),
- packing degree \(d=2^\alpha>1\) (e.g. \(d=1024\), \(\alpha=10\)).

The main purpose is to make the typing and “what is lifted to integers vs what stays in \(F_{q'}\)” completely explicit, to avoid the common fallacies.

---

## 1. What we are trying to prove (the PCS statement)

### 1.1 Scalar semantics (what Jolt Stage 8 wants)

Let \(n\) be the number of Boolean variables in the scalar polynomial that Jolt wants to open.

In the packed regime:

- choose \(\ell\) so that \(2^\ell = N/d\), where \(N=2^n\) is the scalar coefficient-table size,
- write \(d=2^\alpha\), so \(n=\ell+\alpha\),
- split the Boolean index \(x\in\{0,1\}^n\) as \(x=(I,k)\) with \(I\in\{0,1\}^\ell\) and \(k\in\{0,1\}^\alpha\).

The committed scalar coefficient/evaluation table is:

\[
p_{I,k}\in\{0,1\}\subset \mathbb Z_q.
\]

Jolt provides an opening point and claimed value in \(F_{q'}\):

\[
\mathbf r=(\mathbf r_{\mathrm{out}},\mathbf r_{\mathrm{in}})\in F_{q'}^\ell\times F_{q'}^\alpha,\qquad
v\in F_{q'}.
\]

The semantic claim is:

\[
\boxed{
v \stackrel{?}{=} \mathrm{MLE}_{F_{q'}}(p)(\mathbf r)
:=
\sum_{I\in\{0,1\}^\ell}\sum_{k\in\{0,1\}^\alpha}
\iota(p_{I,k})\cdot \mathrm{eq}(\mathbf r_{\mathrm{out}},I)\cdot \mathrm{eq}(\mathbf r_{\mathrm{in}},k)
}
\]

where:

- \(\mathrm{eq}(t,i)=\prod_j (t_j i_j + (1-t_j)(1-i_j))\) is the standard multilinear equality polynomial, and
- \(\iota\) is the **canonical injection of bounded integers** into \(F_{q'}\) (defined below).

### 1.2 Important: this is not a ring evaluation claim

Nothing in the statement asks for evaluating a ring polynomial at a ring point. It is purely a scalar multilinear evaluation in \(F_{q'}\).

That matters because in the genuine two-prime setting, there is **no homomorphism** that lets you multiply arbitrary \(F_{q'}\) scalars into \(R_q\) elements while staying in \(R_q\).

---

## 2. Commitment object (what is bound by Module/Ring-SIS)

### 2.1 Coefficient embedding into ring elements

Define ring elements \(f_I\in R_q\) by coefficient embedding:

\[
f_I(X) := \sum_{k=0}^{d-1} p_{I,k}\,X^k \in R_q.
\]

This is **not** an extra claim — it is the definition of what “packing degree \(d\)” means.

### 2.2 Ajtai-style commitments over \(R_q\)

Fix public (PRG-seeded) Ajtai matrices over \(R_q\) and gadget decomposition base \(b\):

\[
A\in R_q^{n_A\times (\delta\cdot 2^m)},\quad
B\in R_q^{n_B\times (n_A\delta\cdot 2^r)}.
\]

Commitment construction follows the Hachi/Greyhound “block-decompose then inner/outer commit” pattern over \(R_q\).

What matters for this note:

- the commitment \(u\in R_q^{n_B}\) binds (computationally, under Ring/Module-SIS and shortness/range constraints) to a unique **small** witness,
- that witness determines the packed ring coefficient table \((f_I)_I\), and hence the underlying scalar table \((p_{I,k})_{I,k}\), because the coefficients are range restricted (bits).

---

## 3. The ONLY bridge between \(R_q\) and \(F_{q'}\): canonical integer lifting

There is no algebraic embedding \(F_{q'}\hookrightarrow R_q\). So we use **\(\mathbb Z\)** as the common language.

### 3.1 What gets lifted

Only **\(\mathbb Z_q\)-valued coefficients** get lifted.

Concretely, any ring element \(a\in R_q=\mathbb Z_q[X]/(X^d+1)\) has a coefficient representation
\(a(X)=\sum_{t=0}^{d-1} a_t X^t\) with \(a_t\in\mathbb Z_q\).

Choose a lift convention:

\[
\mathrm{lift}_q:\mathbb Z_q\to\mathbb Z
\]

For example (centered representatives):

\[
\mathrm{lift}_q(a_t)\in\left[-\tfrac{q-1}{2},\,\tfrac{q-1}{2}\right]\cap\mathbb Z.
\]

Then define the lifted integer polynomial:

\[
\mathrm{lift}_q(a)(X) := \sum_{t=0}^{d-1} \mathrm{lift}_q(a_t)\,X^t\in \mathbb Z[X]_{<d}.
\]

Apply \(\mathrm{lift}_q\) entrywise to vectors/matrices over \(R_q\).

### 3.2 Which objects must be range-restricted (so the lift is meaningful)

To make \(\mathrm{lift}_q\) a *canonical meaning* (and not an adversarial degree of freedom), every witness coordinate that will be interpreted in \(F_{q'}\) must be constrained to a bounded integer range.

In the Jolt regime we target:

- \(p_{I,k}\in\{0,1\}\) (bitness), so the lift is unambiguous.
- gadget digits \(\in[-b/2,b/2)\) (or \([0,b)\)), range-checked.
- quotient-witness digits for the cyclotomic quotient and modulus quotient, range-checked.

### 3.3 Injecting lifted integers into \(F_{q'}\)

Once an integer \(z\in\mathbb Z\) is uniquely determined (by range constraints), interpret it in \(F_{q'}\) via:

\[
\iota(z) := z \bmod q' \in F_{q'}.
\]

This is the injection used in the semantic evaluation claim (Section 1.1).

---

## 4. What must be proven (the combined relation)

The opening proof must establish **one witness** that simultaneously satisfies:

### 4.1 Commitment consistency in \(R_q\) (mod \(q\), mod \(X^d+1\))

These are purely ring equations, e.g. “this decomposed witness opens the Ajtai commitment \(u\)”, “these gadget digits recompose correctly”, etc.

Write these collectively as:

\[
E_j = 0 \quad \text{in } R_q \quad \text{for } j\in[J],
\]

for some constant \(J\) (depending on how you package the commitment equations).

### 4.2 The scalar evaluation equation in \(F_{q'}\)

This is the actual Stage-8 claim:

\[
v = \sum_{I,k} \iota(p_{I,k})\,\mathrm{eq}(\mathbf r_{\mathrm{out}},I)\,\mathrm{eq}(\mathbf r_{\mathrm{in}},k).
\]

### 4.3 Range constraints (to pin down the lift)

All witness-table entries representing:

- bits \(p_{I,k}\),
- gadget digits,
- quotient digits,

must satisfy their declared bounds.

This is what prevents “changing representatives mod \(q\)” from changing the interpreted integer meaning in \(F_{q'}\).

---

## 5. How to verify ring equations in \(F_{q'}\): ring switching + modulus switching

This is where integer lifting is actually used.

### 5.1 From “\(E=0\) in \(R_q\)” to an integer ideal membership statement

Fix one ring equation \(E=0\) in \(R_q\). Expand \(E\) as a polynomial in \(\mathbb Z_q[X]\) reduced mod \(X^d+1\).

Lift coefficients to \(\mathbb Z\) to get \(\mathrm{lift}_q(E)\in\mathbb Z[X]_{<d}\).

The statement “\(E=0\) in \(R_q\)” means:

- \(E\equiv 0\pmod{X^d+1}\) and
- all coefficients are \(0\) mod \(q\).

Equivalently, there exist integer polynomials \(r,s\in\mathbb Z[X]_{<d}\) such that:

\[
\boxed{
\mathrm{lift}_q(E) = (X^d+1)\,r \;+\; q\,s
}
\quad\text{in }\mathbb Z[X].
\]

This is the **cyclotomic quotient** \((X^d+1)r\) plus the **modulus quotient** \(q\,s\).

Range constraints on digits of \(r,s\) ensure these are honest integer polynomials, not arbitrary \(F_{q'}\)-values.

### 5.2 From an integer identity to a checkable field identity

Sample \(\alpha\leftarrow F_{q'}\) (Fiat–Shamir).

Evaluate the integer-polynomial identity at \(X=\alpha\), interpreting integers in \(F_{q'}\) via \(\iota\):

\[
\boxed{
\iota(\mathrm{lift}_q(E))(\alpha)
\;=\;
(\alpha^d+1)\cdot r(\alpha) \;+\; q\cdot s(\alpha)
}
\quad\text{in }F_{q'}.
\]

Soundness: if \(\mathrm{lift}_q(E) - (X^d+1)r - q s\) is a nonzero polynomial of degree \(<2d\), then it has at most \(2d-1\) roots in a field, so:

\[
\Pr[\text{false ring equation passes at random }\alpha] \le \frac{2d-1}{|F_{q'}|}.
\]

---

## 6. One coherent verification strategy (IOP view)

This is the high-level structure that avoids the typing fallacies:

1. **Single witness table**: the prover commits (with the lattice PCS) to a witness-table multilinear \(\tilde w\) whose entries include:
   - the bit table \((p_{I,k})\),
   - all commitment-opening gadget digits,
   - all quotient digits for every ring equation’s \((r,s)\).

2. **Verifier challenges in \(F_{q'}\)**: sample \(\alpha\) for ring switching, plus the usual sumcheck challenges.

3. **One combined sumcheck over \(F_{q'}\)** that proves:
   - the ring-switched identities at \(\alpha\) for all ring equations (Section 5.2),
   - the scalar evaluation identity \(P(\mathbf r)=v\) (Section 1.1),
   - the range/bitness constraints (Section 4.3).

4. **Recursion boundary**: sumcheck reduces everything to opening \(\tilde w\) at a random point \(\mathbf r^*\in F_{q'}^{m'}\). That output has the same “Stage-8 shape”, so the PCS can recurse/handoff as in Hachi.

The key correctness property is that **every constraint family reads from the same committed \(\tilde w\)**. That is what makes the binding between:

- the ring commitment \(u\) (Module/Ring-SIS world), and
- the scalar evaluation \(P(\mathbf r)=v\) (Jolt world)

logically solid.

---

## 7. What you should and should not expect

### 7.1 What this DOES guarantee

If the range constraints are strong enough to make all relevant lifts canonical, then an accepting proof implies the prover “knows” a single bounded integer coefficient table \((p_{I,k})\) such that:

- it opens the lattice commitment \(u\) (in \(R_q\)), and
- it evaluates to \(v\) at \(\mathbf r\) (in \(F_{q'}\)).

This is the exact “binding between the commitment and the opening claim” you’re worried about.

### 7.2 What this DOES NOT magically provide

It does **not** port Hachi’s Step-B “ring-valued partial evaluations” story to the two-prime setting. That story relies on having evaluation-derived weights live inside the ring (via an extension-field embedding), which is absent here.

So, unless you add additional machinery (e.g. non-native arithmetic or a second PCS over \(F_{q'}\)), you should not expect the same square-root-style reductions to survive unchanged in the genuine two-field regime.

---

## 8. Making the ring-side constraints explicit (what are the \(E_j\)?)

This section spells out one concrete choice of “ring equations” whose satisfaction means:

- the lattice commitment \(u\) is consistent with a *single* packed ring coefficient table \((f_I)_I\), and
- that ring coefficient table’s **coefficients are bits** \((p_{I,k})\).

The exact commitment wiring (heights \(n_A,n_B\), decomposition base \(b\), etc.) can match Hachi/Greyhound; the key is that every equation is an equality in \(R_q\) and is **linear** in the witness variables.

### 8.1 Witness variables (explicit)

Fix \(\ell=m+r\) and index outer blocks by \(i\in\{0,1\}^r\), inner block positions by \(j\in\{0,1\}^m\), and coefficient slots by \(k\in\{0,1\}^\alpha\).

The prover’s witness includes:

- **bit coefficients**: \(p_{i\|j,k}\in\{0,1\}\subset \mathbb Z_q\).
- **packed ring elements** (definitionally from \(p\)): for each \((i,j)\),
  \[
  f_{i\|j}(X) := \sum_{k=0}^{d-1} p_{i\|j,k}\,X^k \in R_q.
  \]
  (You can treat \(f_{i\|j}\) as a derived value, but it is often convenient to name it.)
- **gadget digits** \(s_i\in R_q^{2^m\delta}\) such that \(f_i = G_{2^m}s_i\), where \(f_i^\top=(f_{i\|j})_{j\in\{0,1\}^m}\in R_q^{2^m}\).
- **inner commitments** \(t_i := A s_i \in R_q^{n_A}\) and their digitizations \(\hat t_i := G^{-1}_{n_A}(t_i)\in R_q^{n_A\delta}\).

Public values:

- Ajtai matrices \(A,B\) over \(R_q\) (usually PRG-defined),
- commitment \(u := B[\hat t_1;\dots;\hat t_{2^r}]\in R_q^{n_B}\).

### 8.2 Ring equations \(E_j=0\) (commitment consistency)

Define the following ring equalities in \(R_q\). These are the concrete \(E_j\) you can plug into Section 5:

1) **Packing/definition of ring elements from bits** (per \((i,j)\)):
\[
E^{\mathrm{pack}}_{i,j} := f_{i\|j}(X) - \sum_{k=0}^{d-1} p_{i\|j,k}\,X^k = 0 \quad\text{in }R_q.
\]

2) **Digit recomposition** (per block \(i\)):
\[
E^{\mathrm{recomp}}_{i} := f_i - G_{2^m}s_i = 0 \quad\text{in }R_q^{2^m}.
\]

3) **Inner commitment** (per block \(i\)):
\[
E^{\mathrm{inner}}_{i} := t_i - A s_i = 0 \quad\text{in }R_q^{n_A}.
\]

4) **Digitization of inner commitment** (per block \(i\)):
\[
E^{\mathrm{dig}}_{i} := t_i - G_{n_A}\hat t_i = 0 \quad\text{in }R_q^{n_A}.
\]

5) **Outer commitment consistency** (single vector equation):
\[
E^{\mathrm{outer}} := u - B[\hat t_1;\dots;\hat t_{2^r}] = 0 \quad\text{in }R_q^{n_B}.
\]

These are all linear over \(R_q\) in the witness variables \((p,s,\hat t)\) (and \(f,t\) if you keep them explicit).

### 8.3 Range constraints (what makes the lift canonical)

Alongside ring equations, enforce:

- **Bitness**: \(p_{i\|j,k}\cdot(p_{i\|j,k}-1)=0\) in the sumcheck field \(F_{q'}\) (after injecting the \(p\) entries).
- **Digit bounds** for every coordinate of \(s_i\) and \(\hat t_i\) (and for any quotient digits).

This is what prevents “changing representatives mod \(q\)” from changing the intended integers.

---

## 9. Row-by-row lifting: from \(E=0\) in \(R_q\) to a check in \(F_{q'}\)

This section answers your “what EXACTLY is lifted?” question with a literal example.

### 9.1 Example: outer commitment equation

Take the ring equation \(E^{\mathrm{outer}}=0\) (Section 8.2(5)):

\[
E^{\mathrm{outer}} := u - B\hat t = 0 \quad\text{in }R_q^{n_B}.
\]

Expand each coordinate \(E^{\mathrm{outer}}_j\in R_q\) in the coefficient basis:
\[
E^{\mathrm{outer}}_j(X)=\sum_{t=0}^{d-1} e_{j,t}\,X^t,\qquad e_{j,t}\in\mathbb Z_q.
\]

Now lift coefficient-wise:
\[
\mathrm{lift}_q(E^{\mathrm{outer}}_j)(X)=\sum_{t=0}^{d-1} \mathrm{lift}_q(e_{j,t})\,X^t\in \mathbb Z[X]_{<d}.
\]

The claim “\(E^{\mathrm{outer}}_j=0\) in \(R_q\)” is equivalent to existence of \(r_j,s_j\in \mathbb Z[X]_{<d}\) such that:
\[
\mathrm{lift}_q(E^{\mathrm{outer}}_j) = (X^d+1)\,r_j + q\,s_j\quad\text{in }\mathbb Z[X].
\]

This is the exact meaning of “add quotient witnesses for \(X^d+1\) and for \(q\)”.

### 9.2 Turning it into an \(F_{q'}\) check (ring switching)

Sample \(\alpha\leftarrow F_{q'}\) and evaluate:
\[
\iota(\mathrm{lift}_q(E^{\mathrm{outer}}_j))(\alpha)
\stackrel{?}{=} (\alpha^d+1)\,r_j(\alpha) + q\,s_j(\alpha)\quad\text{in }F_{q'}.
\]

If this holds for all coordinates \(j\) (and for every ring equation \(E_j\) you include), then with probability \(1-O(d)/|F_{q'}|\) the original ring equations were true (given range constraints on the witnesses so they correspond to honest integers).

---

## 10. Where “the two fields are linked” (and where they are not)

There is no map \(F_{q'}\\to R_q\). The linking happens only through:

1) **Shared witness data**: the same committed witness table \(\tilde w\) contains the \(p_{I,k}\) bits used in the scalar evaluation and the digit data used in the ring equations.

2) **Canonical lifting**: range checks ensure the \(p\) and digit entries have a unique integer meaning, and \(\iota\) injects that meaning into \(F_{q'}\).

3) **Ring switching checks**: ring equations over \(R_q\) are translated into integer polynomial identities and then evaluated at \(\alpha\in F_{q'}\).

If any of these three is missing, the binding between the commitment and the opening claim is not solid.

---

## 11. “But \(R_q\) equations contain products” — what is actually being lifted?

You are exactly right that the ring-side equalities \(E_j=0\) typically contain **products** (e.g. matrix–vector products like \(B\hat t\), which multiply ring elements).

The crucial point is:

> **We never assume \(\mathrm{lift}_q(\cdot)\) is a ring homomorphism.**  
> In particular, \(\mathrm{lift}_q(a\cdot b)\neq \mathrm{lift}_q(a)\cdot \mathrm{lift}_q(b)\) in general.

That “non-homomorphism gap” is **precisely** what the extra quotient witnesses \((X^d+1)r\) and \(q\,s\) are for.

### 11.1 One ring multiplication, made explicit

Let \(a,b\in R_q=\mathbb Z_q[X]/(X^d+1)\). Choose coefficient representatives and lift to integer polynomials
\(\tilde a:=\mathrm{lift}_q(a)\in\mathbb Z[X]_{<d}\), \(\tilde b:=\mathrm{lift}_q(b)\in\mathbb Z[X]_{<d}\).

Compute the integer product \(\tilde a\cdot \tilde b\in\mathbb Z[X]_{<2d-1}\).

Let \(c := a\cdot b\in R_q\) be the *ring* product, i.e. the product in \(\mathbb Z_q[X]\) reduced modulo both \(q\) and \(X^d+1\), and let \(\tilde c:=\mathrm{lift}_q(c)\in\mathbb Z[X]_{<d}\).

Then there exist integer polynomials \(r,s\in\mathbb Z[X]_{<d}\) such that:

\[
\boxed{
\tilde a\cdot \tilde b \;-\; \tilde c \;=\; (X^d+1)\,r \;+\; q\,s
}
\quad\text{in }\mathbb Z[X].
\]

Interpretation:

- \((X^d+1)\,r\) accounts for the **degree reduction** that happens in \(R_q\) (negacyclic reduction),
- \(q\,s\) accounts for the **coefficient reduction mod \(q\)** (because \(\tilde a,\tilde b\) are integers, while \(a,b\) multiply mod \(q\)).

This is the right way to think about products under lifting: the quotient witnesses absorb the difference between “integer arithmetic” and “ring arithmetic.”

### 11.2 Matrix–vector products are the same story

Every ring equation \(E_j=0\) is some expression built out of:

- additions/subtractions in \(R_q\), and
- multiplications of witness ring elements by **public** ring elements (matrix entries),
- plus gadget linear maps.

When you expand \(E_j(X)\) coefficient-wise in \(\mathbb Z_q\) and lift to \(\mathbb Z\), you get a well-defined integer polynomial \(\mathrm{lift}_q(E_j)\).

You do **not** need to lift intermediate products term-by-term. It is enough to assert existence of \(r_j,s_j\) such that:

\[
\mathrm{lift}_q(E_j) = (X^d+1)\,r_j + q\,s_j.
\]

This single ideal-membership statement already covers *all* products that occurred inside \(E_j\).

### 11.3 Why \(q\,s\) is non-optional in the two-field setting

If verification happened in characteristic \(q\) (i.e. inside \(\mathbb Z_q\) or \(F_q\)), then “\(\equiv 0\pmod q\)” would be the same as equality.

But we are checking in \(F_{q'}\) with a different characteristic. Without explicitly asserting “the integer polynomial is a multiple of \(q\)” via \(q\,s\), a prover could exploit the fact that equality mod \(q\) does not imply equality as integers, and hence does not imply equality in \(F_{q'}\) under any chosen lift.

---

## 12. “What if integers wrap mod \(q'\)?” — this is a *correctness* condition, not a probabilistic soundness term

In \(F_{q'}\), every integer is interpreted modulo \(q'\). So when we write checks like:

\[
\iota(\mathrm{lift}_q(E))(\alpha) \stackrel{?}{=} (\alpha^d+1)\,r(\alpha) + q\,s(\alpha),
\]

we are **really** checking an identity in \(\mathbb Z/q'\mathbb Z\).

If the lifted integers were allowed to be arbitrarily large, then:

- an integer identity could fail in \(\mathbb Z[X]\) but still hold modulo \(q'\),
- and the verifier would accept.

This is not the usual “degree/field-size” soundness error (root test). It is a **wraparound ambiguity**: you would be proving equality modulo \(q'\), not equality over \(\mathbb Z\).

### 12.1 The fix: explicit magnitude bounds via range checks

To ensure the \(F_{q'}\) check corresponds to the intended integer check, you need a bound \(B\) such that **every integer** that appears in the checked identity has magnitude \(<B\), and choose \(q'\) so that:

\[
\boxed{2B < q'.}
\]

Then the reduction \(\iota:\{-B,\dots,B\}\to F_{q'}\) is injective, and equality in \(F_{q'}\) implies equality over \(\mathbb Z\) for all quantities that the protocol ever interprets.

In practice, this is why range checks must cover **not only** the data bits \(p_{I,k}\) and gadget digits, but also the quotient-witness digits for \(r\) and \(s\): they are part of the integers being interpreted.

### 12.2 The remaining probabilistic soundness term (root test)

Once wraparound is excluded by bounds, the only remaining soundness error for ring switching is the standard root test:

- if the integer polynomial \(\Delta(X)=\mathrm{lift}_q(E)-(X^d+1)r-qs\) is nonzero of degree \(<2d\),
- then \(\Pr[\Delta(\alpha)=0]\le (2d-1)/|F_{q'}|\).

This is the **only** probabilistic soundness term from ring switching.

