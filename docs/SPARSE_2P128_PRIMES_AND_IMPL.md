# Sparse primes near \(2^{128}\) and fast field arithmetic (CPU + CUDA)

This note records a small set of **provably prime** 128-bit moduli of the form
\[
p = 2^{128} - 2^a \pm 2^b \pm 1
\]
that lie in the interval
\[
(2^{64}-1)^2 \le p < 2^{128},
\]
and documents the fastest modular reduction / multiplication strategies on:

- **CPU (64-bit)**: two 64-bit limbs with `u128` temporaries, pseudo‑Mersenne (“Solinas”) folding.
- **CUDA GPU (32-bit)**: four 32-bit limbs, PTX `mul.lo/mul.hi` + `add.cc/addc` + `sub.cc/subc`, with an **interleaved multiply+reduce** schedule.

All primality statements below were checked with Sage using `is_prime(proof=True)`.

---

## 1. Interval and why it caps \(a\)

\[
(2^{64}-1)^2 = 2^{128} - 2^{65} + 1.
\]
So if \(p = 2^{128} - d\) lies in this interval, then \(1 \le d \le 2^{65}-1\). In particular, for sparse offsets dominated by a power of two, this implies you only need to consider \(a \le 65\) when enumerating \(2^{128} - 2^a \pm \cdots\) inside the interval.

---

## 2. “Sparse” primes we found (provable)

### 2.1 Primes satisfying the residue filter

The residue filter used in the search was:
\[
p \equiv 5 \ (\mathrm{mod}\ 8)\quad \text{OR}\quad p \equiv 9 \ (\mathrm{mod}\ 16)\quad \text{OR}\quad p \equiv 17 \ (\mathrm{mod}\ 32).
\]

The exhaustive search over all
\[
p = 2^{128} - 2^a \pm 2^b \pm 1,\quad 128>a>b\ge 1,
\]
in the interval above yields **exactly 4 primes** satisfying that residue filter:

| ID | Sparse form | Hex | Residue class |
|---:|---|---|---|
| P13 | \(2^{128} - 2^{13} - 2^{4} + 1\) | `0xffffffffffffffffffffffffffffdff1` | \(17 \bmod 32\) |
| P37 | \(2^{128} - 2^{37} + 2^{3} + 1\) | `0xffffffffffffffffffffffe000000009` | \(9 \bmod 16\) |
| P52 | \(2^{128} - 2^{52} - 2^{3} + 1\) | `0xffffffffffffffffffeffffffffffff9` | \(9 \bmod 16\) |
| P54 | \(2^{128} - 2^{54} + 2^{4} + 1\) | `0xffffffffffffffffffc0000000000011` | \(17 \bmod 32\) |

For each modulus, define the pseudo‑Mersenne constant:
\[
c := 2^{128} - p \quad\Rightarrow\quad 2^{128} \equiv c \pmod p.
\]
For the four primes above:

- **P13**: \(c = 2^{13} + 2^{4} - 1\)
- **P37**: \(c = 2^{37} - 2^{3} - 1\)
- **P52**: \(c = 2^{52} + 2^{3} - 1\)
- **P54**: \(c = 2^{54} - 2^{4} - 1\)

### 2.2 Largest prime \(<2^{128}\) that is \(5 \bmod 8\)

Define:
\[
p_{5\bmod 8} := 2^{128} - 275.
\]

- Hex: `0xfffffffffffffffffffffffffffffeed`
- \(p_{5\bmod 8} \equiv 5 \ (\mathrm{mod}\ 8)\)
- `is_prime(proof=True)` is **True**
- Additionally, for all \(d \equiv 3 \ (\mathrm{mod}\ 8)\) with \(3 \le d < 275\), \(2^{128}-d\) is composite (so \(2^{128}-275\) is the **largest** such prime below \(2^{128}\)).

This modulus corresponds to \(c = 275 = 2^8 + 2^4 + 2^1 + 2^0\), which also makes reduction extremely cheap.

### 2.3 (Bonus) Other very sparse primes in the same interval (no residue filter)

There are also 3-term primes \(2^{128}-2^a\pm 1\) inside the interval, but they **do not** satisfy the residue filter above. The ones we observed/proved were:

| Sparse form | Hex |
|---|---|
| \(2^{128} - 2^{18} - 1\) | `0xfffffffffffffffffffffffffffbffff` |
| \(2^{128} - 2^{26} - 1\) | `0xfffffffffffffffffffffffffbffffff` |
| \(2^{128} - 2^{54} + 1\) | `0xffffffffffffffffffc0000000000001` |
| \(2^{128} - 2^{58} - 1\) | `0xfffffffffffffffffbffffffffffffff` |

### 2.4 Extended sweep for \(p \equiv (2k+1)\ (\mathrm{mod}\ 4k)\), \(k=2^t \le 512\)

We continued the same exhaustive search over
\[
p = 2^{128} - 2^a \pm 2^b \pm 1,\quad 128>a>b\ge 1,\quad (2^{64}-1)^2 \le p < 2^{128},
\]
and grouped prime hits by residue classes
\[
p \equiv 2k+1 \pmod{4k},\quad k\in\{2,4,8,16,32,64,128,256,512\}.
\]
All primality checks were done with Sage `is_prime(proof=True)`.

Summary of unique prime hits (deduplicated by numeric value \(p\)):

| \(k\) | Residue class | Count in sparse family | Closest hit to \(2^{128}\) |
|---:|---|---:|---|
| 2 | \(5 \bmod 8\) | 0 | none in this 4-term family (use \(2^{128}-275\) from §2.2 if needed) |
| 4 | \(9 \bmod 16\) | 2 | \(2^{128} - 2^{37} + 2^3 + 1\) |
| 8 | \(17 \bmod 32\) | 2 | \(2^{128} - 2^{13} - 2^4 + 1\) |
| 16 | \(33 \bmod 64\) | 1 | \(2^{128} - 2^7 - 2^5 + 1\) |
| 32 | \(65 \bmod 128\) | 4 | \(2^{128} - 2^{25} + 2^6 + 1\) |
| 64 | \(129 \bmod 256\) | 6 | \(2^{128} - 2^{24} - 2^7 + 1\) |
| 128 | \(257 \bmod 512\) | 2 | \(2^{128} - 2^{33} + 2^8 + 1\) |
| 256 | \(513 \bmod 1024\) | 5 | \(2^{128} - 2^{10} - 2^9 + 1\) |
| 512 | \(1025 \bmod 2048\) | 1 | \(2^{128} - 2^{43} + 2^{10} + 1\) |

New hits for the previously missing buckets \(k\ge 16\):

| \(k\) | Sparse form | Hex |
|---:|---|---|
| 16 | \(2^{128} - 2^7 - 2^5 + 1\) | `0xffffffffffffffffffffffffffffff61` |
| 32 | \(2^{128} - 2^{25} + 2^6 + 1\) | `0xfffffffffffffffffffffffffe000041` |
| 32 | \(2^{128} - 2^{43} + 2^6 + 1\) | `0xfffffffffffffffffffff80000000041` |
| 32 | \(2^{128} - 2^{55} + 2^6 + 1\) | `0xffffffffffffffffff80000000000041` |
| 32 | \(2^{128} - 2^{62} + 2^6 + 1\) | `0xffffffffffffffffc000000000000041` |
| 64 | \(2^{128} - 2^{24} - 2^7 + 1\) | `0xfffffffffffffffffffffffffeffff81` |
| 64 | \(2^{128} - 2^{31} + 2^7 + 1\) | `0xffffffffffffffffffffffff80000081` |
| 64 | \(2^{128} - 2^{45} - 2^7 + 1\) | `0xffffffffffffffffffffdfffffffff81` |
| 64 | \(2^{128} - 2^{57} - 2^7 + 1\) | `0xfffffffffffffffffdffffffffffff81` |
| 64 | \(2^{128} - 2^{61} + 2^7 + 1\) | `0xffffffffffffffffe000000000000081` |
| 64 | \(2^{128} - 2^{64} - 2^7 + 1\) | `0xfffffffffffffffeffffffffffffff81` |
| 128 | \(2^{128} - 2^{33} + 2^8 + 1\) | `0xfffffffffffffffffffffffe00000101` |
| 128 | \(2^{128} - 2^{53} - 2^8 + 1\) | `0xffffffffffffffffffdfffffffffff01` |
| 256 | \(2^{128} - 2^{10} - 2^9 + 1\) | `0xfffffffffffffffffffffffffffffa01` |
| 256 | \(2^{128} - 2^{21} + 2^9 + 1\) | `0xffffffffffffffffffffffffffe00201` |
| 256 | \(2^{128} - 2^{39} - 2^9 + 1\) | `0xffffffffffffffffffffff7ffffffe01` |
| 256 | \(2^{128} - 2^{47} + 2^9 + 1\) | `0xffffffffffffffffffff800000000201` |
| 256 | \(2^{128} - 2^{52} - 2^9 + 1\) | `0xffffffffffffffffffeffffffffffe01` |
| 512 | \(2^{128} - 2^{43} + 2^{10} + 1\) | `0xfffffffffffffffffffff80000000401` |

### 2.5 Focused analysis for \(d=512\), targeting \(k=64\)

This section specializes the LS18 invertibility criteria to the ring
\[
R_q = \mathbb{Z}_q[X]/(X^{512}+1)
\]
and to primes in the \(k=64\) bucket:
\[
q \equiv 129 \pmod{256}.
\]

From the sweep above, we have 6 sparse primes in this class.

#### 2.5.1 Splitting shape and NTT implications

If \(q \equiv 129 \pmod{256}\), LS18 Cor. 1.2 implies:
\[
X^{512}+1 \equiv \prod_{j=1}^{64} (X^{8}-r_j)\pmod q,
\]
so we get **64-way partial splitting** (factor degree \(512/64 = 8\)).

This is not full native NTT at size \(512\): that would require \(2\cdot 512 = 1024 \mid (q-1)\), i.e. \(v_2(q-1)\ge 10\), while this class gives \(v_2(q-1)=7\).

Still, compared to \(k=2\), this is a much deeper FFT-style split:

| Split \(k\) | FFT levels \(\log_2 k\) | Base factor degree \(512/k\) |
|---:|---:|---:|
| 2 | 1 | 256 |
| 32 | 5 | 16 |
| 64 | 6 | 8 |

So \(k=64\) significantly reduces base-case polynomial degree even without full native NTT.

#### 2.5.2 Why the plain \(\ell_\infty\) route fails, but \(\ell_2\) can work

LS18 Cor. 1.2 gives invertibility if either:
\[
0<\|y\|_\infty < \frac{1}{\sqrt{k}}q^{1/k}
\quad\text{or}\quad
0<\|y\|_2 < q^{1/k}.
\]

For \(k=64\) and \(q\approx 2^{128}\), \(q^{1/64}\approx 4\), so:

- \(\frac{1}{\sqrt{64}}q^{1/64}\approx 0.5\): too small for usual difference sets with coefficients in \(\{-2,-1,0,1,2\}\),
- but \(\|y\|_2 < 4\) is realistic for structured challenges.

Hence, for \(k=64\), the practical path is to enforce a small \(\ell_2\)-norm on the **difference blocks** \(y'_i\) (LS18 Lemma 3.2 style), rather than relying on a global \(\ell_\infty\) bound.

#### 2.5.3 A structured challenge family for \(d=512, k=64\)

Use LS18's decomposition into \(n/k=512/64=8\) interleaved blocks \(y'_i\), each of length \(64\).

Define family \(\mathcal{C}_{g,t}\):

1. choose exactly \(g\) active blocks out of 8,
2. in each active block choose exactly \(t\) indices among 64,
3. assign signs \(\pm1\) to the chosen entries.

Cardinality:
\[
|\mathcal{C}_{g,t}| = \binom{8}{g}\left(\binom{64}{t}2^t\right)^g.
\]

For a difference \(y=c-c'\):

- each block has \(\|y'_i\|_2 \le 2\sqrt{t}\),
- if \(t=3\), then \(2\sqrt{3}\approx 3.464 < 4 \approx q^{1/64}\).

So every non-zero block \(y'_i\) is invertible in \(\mathbb{Z}_q[X]/(X^{64}+1)\) (LS18 Cor. 1.2 \(\ell_2\)-branch), and by Lemma 3.2 this lifts to invertibility of \(y\) in \(\mathbb{Z}_q[X]/(X^{512}+1)\).

Concrete candidates:

| Family | \(\log_2 |\mathcal{C}_{g,t}|\) | \(\|c\|_1 = g t\) | \(\max_i \| (c-c')'_i\|_2\) |
|---|---:|---:|---:|
| \(\mathcal{C}_{7,3}\) | \(\approx 131.43\) | 21 | \(2\sqrt{3}\approx 3.464\) |
| \(\mathcal{C}_{8,3}\) | \(\approx 146.77\) | 24 | \(2\sqrt{3}\approx 3.464\) |

Both exceed \(2^{128}\) challenge space and satisfy the target per-block \(\ell_2\) margin for \(k=64\), \(q\approx 2^{128}\).

#### 2.5.4 Arithmetic strategy tradeoff (partial-splitting vs CRT/RNS)

Two practical multiplication paths:

1. **Partial splitting over \(q\)** (this section): do \(\log_2(64)=6\) split levels, then multiply degree-8 factors and recombine.
2. **CRT/RNS emulation**: represent mod \(q\) via several NTT-friendly machine-word primes.

For 128-bit \(q\), \(d=512\), rough dynamic-range estimates:

- full \(\times\) full product coefficient budget is about \(265\) bits \((\log_2 512 + 2\cdot 128)\): typically needs around 5 channels with \(\sim 60\)-bit primes.
- full \(\times\) small product (small operand bounded by \(\sim 2^{16}\)) is about \(153\) bits: around 3 channels at \(\sim 60\)-bit primes.

So CRT/RNS is fully viable, but channel count and conversion/reduction overhead must be paid each multiplication. Partial splitting cuts base degree a lot and can be competitive even without native full-size NTT.

#### 2.5.5 What still needs proof/benchmarking

- The challenge family above is an LS18-style design candidate; protocol-level soundness/Fiat-Shamir analysis should confirm that this structured \(\mathcal{C}\) is acceptable for the full argument system.
- Runtime crossover between:
  - \(k=32\) (cleaner bounds, degree-16 base factors),
  - \(k=64\) (more split, tighter margin),
  - CRT/RNS kernels
  must be measured on target hardware (CPU/GPU), since constants dominate.

### 2.6 Can \(k=128\) work? (design-space analysis)

Short answer: **yes, plausibly**, but the clean theorem-only route is much tighter, so practical sets at \(k=128\) likely need LS18-style ad-hoc reasoning (or empirical validation) rather than a direct corollary bound.

#### 2.6.1 What the clean bound gives at \(k=128\)

For \(d=512\), \(k=128\), the prime class is:
\[
q \equiv 257 \pmod{512},
\]
with examples from §2.4:

- `0xfffffffffffffffffffffffe00000101`
- `0xffffffffffffffffffdfffffffffff01`

LS18 Cor. 1.2 in this regime gives (for the relevant local ring scale) a threshold essentially
\[
\|u\|_2 < q^{1/128}\approx 2.
\]
This is very tight: many natural sparse-\(\{-1,0,1\}\) difference patterns exceed it.

Important: for this \(k=128\) prime class, the local 128-coefficient block ring also sits in the \(k=128\) congruence regime (not \(k=64\)), so using a \(q^{1/64}\approx 4\) bound here is not justified.

So, unlike \(k=64\), one should not expect a large practical challenge family to be certified by the bare \(\|u\|_2<q^{1/128}\) inequality alone.

#### 2.6.2 Core-idea path from LS18 that still helps

The useful structural tool is still LS18 Lemma 3.2:

- decompose \(y=c-c'\) into 4 interleaved blocks \(y'_0,\ldots,y'_3\) (since \(512/128=4\)),
- if at least one nonzero \(y'_i\) is invertible in the smaller ring, then \(y\) is invertible in \(R_q\).

This suggests designing challenge families where block-polynomials stay in a "likely invertible" region, even if not covered by the clean corollary inequality.

#### 2.6.3 Candidate families (for flexibility, incl. future rejection-sampling)

Below are practical families over coefficient basis length 512, indexed by residue classes mod 4 (each class length 128).

| Family | Construction sketch | \(\log_2 |\mathcal C|\) | \(T=\|c\|_1\) | Status |
|---|---|---:|---:|---|
| Baseline sparse ternary | exactly weight 19 over all 512, signs \(\pm 1\) | \(\approx 132.76\) | 19 | empirical only at \(k=128\) |
| Signed class-balanced | each of 4 classes: exactly 5 positions with signs \(\pm 1\) | \(\approx 131.92\) | 20 | empirical only at \(k=128\) |
| Signed class-unbalanced | class weights \((1,6,6,7)\), signs \(\pm 1\) | \(\approx 128.13\) | 20 | empirical only at \(k=128\) |
| Fixed-sign class-balanced | each of 4 classes: exactly 6 positions, fixed sign mask | \(\approx 129.35\) | 24 | ad-hoc-leaning, still not theorem-direct |

Monte Carlo checks we ran (Sage, over both \(k=128\) sparse primes above):

- baseline sparse ternary (weight 19): no failures observed in 50k pair samples on one prime; 2k additional on the second prime;
- the three class-structured families above: no failures observed in 5k (or 2k) pair samples per family/prime;
- no sampled pair produced:
  - a non-invertible full difference in \(Z_q[X]/(X^{512}+1)\), nor
  - a failure of the LS18-Lemma-3.2 "some block invertible" criterion.

These are **not proofs**; they are evidence.

#### 2.6.4 Why these options are useful for future ZK/rejection-sampling

If future rejection-sampling prefers lower challenge norm:

- `T=19` (baseline sparse ternary) is best among the above,
- `T=20` options (`5,5,5,5` signed or `1,6,6,7` signed) are close and still exceed \(2^{128}\),
- `T=24` fixed-sign has more margin for some combinatorial controls but higher norm budget.

Representative completeness impact via \((K+k)T(b-1)<b^k\):

| \(T\) | \(K_{\max}\) at \((b,k)=(8,5)\) | \(K_{\max}\) at \((4,7)\) | \(K_{\max}\) at \((2,13)\) |
|---:|---:|---:|---:|
| 19 | 241 | 280 | 418 |
| 20 | 229 | 266 | 396 |
| 24 | 190 | 220 | 328 |

So moving from 19 to 20 is usually mild; 24 is more noticeable.

#### 2.6.5 What I am unsure about / open technical risk

- I do **not** currently have a fully general LS18-style proof that these \(k=128\) families are subtractive in the strong deterministic sense used by classical extraction arguments.
- Evidence is strong empirically, but a production proof should either:
  1. build a dedicated ad-hoc local-invertibility lemma (like LS18 §3.3 style), or
  2. explicitly model a negligible "bad-difference" failure event in soundness accounting.

Given your stated flexibility goal for potential rejection sampling, keeping both `T=19` and `T=20` families as first-class options seems prudent.

### 2.7 Fully rigorous families we can prove today

This section gives constructions that are fully rigorous from LS18 factorization + basic ring algebra. They are less "nice" than the empirical families above, but they are provable.

#### 2.7.1 Single-round, one-shot \(>2^{128}\) (high-norm) family

Assume \(q\) is in the \(k=128\) class (\(q\equiv 257 \bmod 512\)), so:
\[
X^{512}+1 \equiv \prod_{j=1}^{128} f_j(X)\pmod q,\quad \deg f_j = 4.
\]

Define
\[
\mathcal C_{\deg<4,B}
:=
\left\{
c(X)=a_0+a_1X+a_2X^2+a_3X^3
\ :\ 
a_i\in[-B,B]\cap\mathbb Z
\right\}.
\]

Size:
\[
|\mathcal C_{\deg<4,B}|=(2B+1)^4.
\]
Choosing \(B=2^{31}\) already gives \((2^{32}+1)^4>2^{128}\).

**Proposition (rigorous):** \(\mathcal C_{\deg<4,B}\) is subtractive in \(R_q=\mathbb Z_q[X]/(X^{512}+1)\), i.e. every non-zero difference is invertible.

**Proof:** Let \(d(X)=c(X)-c'(X)\neq 0\) with \(c,c'\in \mathcal C_{\deg<4,B}\). Then \(\deg d<4\).
If \(d\) were non-invertible in \(R_q\), then \(\gcd(d,X^{512}+1)\neq 1\), so some irreducible factor \(f_j\) of \(X^{512}+1\) would divide \(d\). But \(\deg f_j=4>\deg d\), impossible. Hence \(d\) is invertible. \(\square\)

Tradeoff: fully rigorous, but large coefficient magnitude (\(\|c\|_\infty\sim 2^{31}\)).

#### 2.7.2 Low-norm rigorous option via multi-challenge composition

If protocol design can use \(r\) independent challenge draws (or one challenge vector of \(r\) components), there is a low-norm fully rigorous option.

For one draw, define \(\mathcal C_{\text{1hot}}\):

- split indices into 4 classes modulo 4 (each class length 128),
- in each class choose exactly one index,
- place a fixed public sign mask (\(\pm 1\)) at chosen indices.

Then:

- one-draw size: \(|\mathcal C_{\text{1hot}}|=128^4=2^{28}\),
- one-draw norm: \(\|c\|_1=4\).

For any two one-draw challenges \(c\neq c'\):

- each local block difference \(y'_i\) has at most 2 nonzero \(\pm1\) entries,
- so \(\|y'_i\|_2\le \sqrt2 < 2 \approx q^{1/128}\).

Hence every nonzero \(y'_i\) is invertible in \(\mathbb Z_q[X]/(X^{128}+1)\) by LS18 Cor. 1.2, and global invertibility follows from LS18 Lemma 3.2.

To exceed \(2^{128}\) overall challenge space, take \(r=5\) independent draws:
\[
|\mathcal C_{\text{1hot}}|^5 = 2^{140}.
\]

Tradeoff: fully rigorous and RS-friendly norms, but requires multi-challenge composition in the protocol.

#### 2.7.3 One-shot rigorous family with explicit root-gap certificate

Define
\[
\mathcal C_{\text{lin4},B}
:=
\Big\{
c(X)=\sum_{i=0}^{3}\left(a_iX^i+b_iX^{i+4}\right)
\ :\ a_i,b_i\in[-B,B]\cap\mathbb Z
\Big\}.
\]

Size:
\[
|\mathcal C_{\text{lin4},B}|=(2B+1)^8.
\]
Choosing \(B=2^{15}\) gives \((2^{16}+1)^8>2^{128}\).

For \(q\equiv 257\pmod{512}\), LS18 Cor. 1.2 gives
\[
X^{512}+1=\prod_{r\in\mathcal R_q}(X^4-r),
\]
where \(\mathcal R_q=\{r\in\mathbb Z_q^\*: r^{128}=-1\}\) and \(|\mathcal R_q|=128\).

Define the **root-gap condition** at radius \(A\):
\[
\forall r\in\mathcal R_q,\ \forall b\in\mathbb Z,\ 1\le |b|\le A:\quad
\left|(-rb)\bmod^{\pm} q\right|>A,
\]
where \(\bmod^{\pm} q\) denotes the centered representative in \([-(q-1)/2,(q-1)/2]\).

Set \(A:=2B\). Then:

**Proposition (rigorous, conditional on root-gap):**  
If root-gap holds at \(A=2B\), then \(\mathcal C_{\text{lin4},B}\) is subtractive in \(R_q=\mathbb Z_q[X]/(X^{512}+1)\).

**Proof:** Let \(d=c-c'\neq 0\) with \(c,c'\in\mathcal C_{\text{lin4},B}\). Write
\[
d(X)=\sum_{i=0}^{3}(u_iX^i+v_iX^{i+4}),\quad u_i,v_i\in[-2B,2B].
\]
Assume \(d\) is non-invertible. Then \(X^4-r\mid d\) for some \(r\in\mathcal R_q\). Reducing modulo \(X^4-r\):
\[
d(X)\equiv \sum_{i=0}^{3}(u_i+rv_i)X^i.
\]
Hence \(u_i+rv_i=0\) for all \(i\). If some \(v_i\neq 0\), then
\[
-u_i\equiv rv_i\pmod q,\quad |u_i|\le 2B,\ 1\le |v_i|\le 2B,
\]
which contradicts root-gap with \(A=2B\). So all \(v_i=0\), then all \(u_i=0\), thus \(d=0\), contradiction. Therefore \(d\) is invertible. \(\square\)

For our two \(k=128\) sparse primes (`0xfffffffffffffffffffffffe00000101`, `0xffffffffffffffffffdfffffffffff01`), we exhaustively checked root-gap at \(A=2^{16}\) (equivalently \(B=2^{15}\)) and it holds. Thus \(\mathcal C_{\text{lin4},2^{15}}\) is rigorously subtractive for both primes.

Reproducibility script:

```bash
python3 scripts/check_root_gap_k128.py --B 32768
```

Tradeoff: one-shot, rigorous, \(>2^{128}\) challenge space, but still high norm (\(\|c\|_1\le 8B=2^{18}\)).

### 2.8 A from-scratch barrier (why one-shot \(k=128\) is hard under corollary-only criteria)

This subsection gives a fully rigorous upper bound for a broad and common design pattern.

Consider product-structured class families
\[
\mathcal C = \mathcal A_0 \times \mathcal A_1 \times \mathcal A_2 \times \mathcal A_3,
\]
where each \(\mathcal A_i\subset \mathbb Z^{128}\) controls one residue class modulo 4.

Assume we try to certify invertibility using only the local LS18 Cor. 1.2 \(k=128\) bound
\[
\|u\|_2 < q^{1/128}\approx 2,
\]
for all nonzero within-class differences \(u\in \mathcal A_i-\mathcal A_i\).

Then for each \(i\), every pair \(a,b\in\mathcal A_i\) satisfies \(\|a-b\|_2<2\).
Fix \(a_0\in\mathcal A_i\). For any \(a\in\mathcal A_i\), \(\Delta:=a-a_0\in\mathbb Z^{128}\) has:
\[
\|\Delta\|_2^2 < 4.
\]
Since \(\Delta\) has integer coordinates, \(\Delta\) can have at most 3 nonzero entries, each in \(\{\pm1\}\). Therefore
\[
|\mathcal A_i|
\le
\sum_{t=0}^{3}\binom{128}{t}2^t
=
2{,}763{,}777
\approx 2^{21.40}.
\]

So any such product family satisfies
\[
|\mathcal C|
\le
\left(2{,}763{,}777\right)^4

\approx 2^{85.59}
\ll 2^{128}.
\]

**Conclusion (rigorous):** one-shot \(k=128\) challenge sets of this product form cannot reach \(>2^{128}\) size if the proof relies only on the local corollary threshold \(\|u\|_2<2\).

This explains why practical one-shot \(k=128\) designs need at least one of:

1. a stronger ad-hoc invertibility lemma than the clean corollary bound, or
2. empirical acceptance of tiny bad-difference probability in soundness accounting, or
3. multi-challenge composition (as in §2.7.2).

### 2.9 How LS18 Cor. 1.2 is derived (and where tightening can come from)

This is the exact derivation path in LS18:

1. Start from Theorem 1.1 (general cyclotomic statement with parameters \(m,z\)):
   - factorization condition \(p\equiv 1 \pmod z\) and \(\mathrm{ord}_m(p)=m/z\),
   - invertibility bounds
     \[
     \|y\|_\infty < \frac{1}{s_1(z)}\,p^{1/\varphi(z)}
     \quad\text{or}\quad
     \|y\|_2 < \frac{\sqrt{\varphi(m)}}{s_1(m)}\,p^{1/\varphi(z)}.
     \]
2. Specialize to \(X^n+1\) by setting \(m=2n\), \(z=2k\) (with \(n,k\) powers of two).
3. Use \(p\equiv 2k+1 \pmod{4k}\) to get \(p\equiv 1 \pmod z\), and use LS18 Lemma 2.4 to obtain \(\mathrm{ord}_m(p)=m/z\).
4. Plug in \(s_1(z)=\sqrt{k}\), \(s_1(m)=\sqrt{n}\), \(\varphi(z)=k\), \(\varphi(m)=n\), yielding Cor. 1.2:
   \[
   \|y\|_\infty < \frac{1}{\sqrt{k}}p^{1/k}
   \quad\text{or}\quad
   \|y\|_2 < p^{1/k}.
   \]

Where this comes from in LS18:

- Cor. 1.2 statement: `paper/ls18-short-invertible-elements.pdf`, lines 239–252.
- Corollary proof substitution \(m=2n,z=2k\): lines 797–807.
- Theorem 1.1 general statement: lines 205–224.
- Lift step (one invertible block is enough): Lemma 3.2, lines 727–741.

#### Tightening levers (conceptual)

There are three real levers:

1. **Better local criterion than global norm bounds**  
   Cor. 1.2 is norm-only and uniform. If your challenge differences live in a strict language (e.g. low degree in certain block coordinates), you can prove invertibility directly via factor-degree or root-exclusion arguments (as in §2.7.3), which can beat corollary-style norm thresholds.

2. **Exploit Lemma 3.2 asymmetry**  
   You do **not** need all blocks invertible; one nonzero invertible block suffices. This enables constructions where only one block is guaranteed "easy", with the others unconstrained.

3. **Composition over rounds**  
   If one-shot constraints are too strict, product challenge spaces across rounds preserve rigorous invertibility while keeping each round low norm (as in §2.7.2).

---

## 3. Fast modular reduction for \(p = 2^{128}-c\) (generic)

Let a 256-bit intermediate be written as:
\[
t = t_{\mathrm{lo}} + 2^{128} t_{\mathrm{hi}},\quad 0 \le t_{\mathrm{lo}},t_{\mathrm{hi}} < 2^{128}.
\]
Then:
\[
t \bmod p \equiv t_{\mathrm{lo}} + c \cdot t_{\mathrm{hi}} \pmod p.
\]

Because \(c\) is a **signed sum of a few powers of two**, the multiply \(c\cdot t_{\mathrm{hi}}\) is implemented as **shifts + adds/subs**:

- If \(c = 2^A + 2^B - 1\), then \(c h = (h\ll A) + (h\ll B) - h\)
- If \(c = 2^A - 2^B - 1\), then \(c h = (h\ll A) - (h\ll B) - h\)
- If \(c\) is small (e.g. 275), pre-decompose into powers of two.

### 3.1 Two-fold “Solinas” folding

In practice you fold twice:

1. \(x \leftarrow t_{\mathrm{lo}} + c\cdot t_{\mathrm{hi}}\)
2. Split \(x = x_{\mathrm{lo}} + 2^{128} x_{\mathrm{hi}}\), then \(y \leftarrow x_{\mathrm{lo}} + c\cdot x_{\mathrm{hi}}\)
3. Normalize: conditional subtract \(p\) (sometimes you subtract once; depending on how much headroom you allow, you may need a second subtract, but for these parameters a single subtract is typically enough if you keep tight bounds).

This avoids Montgomery/Barrett overhead and is typically optimal for these moduli.

---

## 4. CPU strategy (64-bit limbs)

### 4.1 Representation

Use two 64-bit limbs:
\[
x = x_0 + 2^{64}x_1,\quad x_0,x_1\in[0,2^{64}).
\]

### 4.2 Multiplication

Compute a 256-bit product via schoolbook:

- \(x_0y_0\), \(x_0y_1\), \(x_1y_0\), \(x_1y_1\) using `u128`
- accumulate into four 64-bit limbs (or keep as two `u128` chunks)

On x86_64, the fastest variant typically uses `mulx` + `adc`/`sbb` chains (via intrinsics or inline asm), but plain `u128` is often competitive and much simpler.

### 4.3 Reduction

Split the 256-bit product into `lo` and `hi` 128-bit halves, then do two folds with the chosen \(c\):

- **P37**: \(c = 2^{37} - 2^{3} - 1\) so \(c\cdot h = (h\ll 37) - (h\ll 3) - h\)
- **P52**: \(c = 2^{52} + 2^{3} - 1\) so \(c\cdot h = (h\ll 52) + (h\ll 3) - h\)
- **P13**: \(c = 2^{13} + 2^{4} - 1\) so \(c\cdot h = (h\ll 13) + (h\ll 4) - h\)
- **P54**: \(c = 2^{54} - 2^{4} - 1\) so \(c\cdot h = (h\ll 54) - (h\ll 4) - h\)
- **\(2^{128}-275\)**: \(c = 275 = 2^8+2^4+2^1+1\) so \(c\cdot h = (h\ll 8)+(h\ll 4)+(h\ll 1)+h\)

### 4.4 Constant-time normalization

After the final fold, do a constant-time conditional subtract of \(p\) using borrow flags; avoid data-dependent branches if you care about side channels.

---

## 5. CUDA strategy (32-bit limbs, PTX carry chains)

### 5.1 Representation

Use four 32-bit limbs (little-endian):
\[
x = x_0 + 2^{32}x_1 + 2^{64}x_2 + 2^{96}x_3.
\]
A product gives eight limbs \(t_0,\ldots,t_7\).

### 5.2 Key PTX instructions you want

- `mul.lo.u32`, `mul.hi.u32`: low/high halves of a 32×32→64 multiply.
- `add.cc.u32`, `addc.cc.u32`, `addc.u32`: add with carry propagation.
- `sub.cc.u32`, `subc.cc.u32`, `subc.u32`: subtract with borrow propagation.
- `shf.l.wrap.b32` / `shf.r.wrap.b32`: **funnel shifts**.

#### Funnel shift (what it does)

A funnel shift treats two 32-bit registers as a 64-bit concatenation for shifting across the boundary. Conceptually, for left shift by \(s\in[1,31]\):
\[
\texttt{shf\_l(hi, lo, s)} = (hi \ll s)\;|\;(lo \gg (32-s)).
\]
This is the building block for multi-limb shifts like “\(\ll 37\)” without doing multiple shifts + ORs manually.

### 5.3 Recommended CUDA modulus: P13

If your primary goal is **fast 32-bit (GPU) modular arithmetic**, the best pick among the 4-term primes above is:

\[
\boxed{p = \textbf{P13} = 2^{128} - 2^{13} - 2^{4} + 1 = 2^{128} - 8207}
\]
with
\[
c = 2^{128}-p = 8207 = 2^{13}+2^{4}-1,
\quad\Rightarrow\quad
2^{128} \equiv c \pmod p.
\]

This is “nicer” than the other 4-term primes for 32-bit code because:

- The only shifts needed in \(c\cdot hi\) are by **13** and **4** bits.
- There is **no** “\(+32\) limb offset” shift (e.g. \(37=32+5\), \(52=32+20\)), which tends to simplify instruction scheduling and reduce carry/borrow propagation overhead.

### 5.4 Interleaved multiply + reduce (P13)

For P13, the folding identity is:
\[
t = lo + 2^{128}hi
\quad\Rightarrow\quad
t \bmod p \equiv lo + c\cdot hi
= lo + (hi\ll 13) + (hi\ll 4) - hi
\pmod p.
\]

The key optimization on GPUs is to **interleave** multiplication and reduction:

- Compute a 4×4→8-limb product using a Comba schedule producing limbs \(t_0,\ldots,t_7\) in increasing order.
- Write \(t_0..t_3\) directly into a 192-bit accumulator `r[0..5]` (initialize `r[4]=r[5]=0`).
- As soon as each high limb \(t_{4+k}\) is produced, immediately “fold” it into `r` using only shifts and adds/subs with carry/borrow.

#### Per-word fold for P13

Let `w = t[4+k]` for \(k\in\{0,1,2,3\}\). Folding `w` corresponds to adding:
\[
(w\cdot 2^{32k})\cdot c
 =
(w\cdot 2^{32k})\cdot (2^{13}+2^{4}-1).
\]

In 32-bit limb arithmetic this is:

- **at limb `k`**: add \((w\ll 13) + (w\ll 4) - w\)
- **at limb `k+1`**: add \((w\gg 19) + (w\gg 28)\)

and then propagate carry/borrow using `add.cc/addc` and `sub.cc/subc`.

In PTX terms (conceptually), the fold is:

```
// r[0..5] is a 192-bit accumulator (6×u32 limbs), little-endian.
// fold high word w at position k (k=0..3)
sub   r[k]   -= w
add   r[k]   += (w << 13)
add   r[k+1] += (w >> 19)
add   r[k]   += (w << 4)
add   r[k+1] += (w >> 28)
// each add/sub is a carry/borrow chain across r[k..5]
```

After folding \(t_4..t_7\), you have a 192-bit value `r`. Do a **second fold** using the top 64 bits (`r[4]`,`r[5]`) exactly the same way (treat them as a new `hi2`), then perform a constant-time conditional subtract of \(p\).

### 5.5 Why P37 is still convenient (but not the cheapest)

For **P37**, the big shift is \(37 = 32 + 5\). That means “\(\ll 37\)” is just:

- a **+1 limb** offset, and
- a **5-bit** cross-limb shift.

Shifts by 5 tend to be slightly simpler to schedule than shifts by 20 (as in P52), though both are fine.

### 5.6 Interleaved multiply + reduce (P37)

For P37, \(c = 2^{37}-2^3-1\), so for the high half \(hi\):
\[
c\cdot hi = (hi\ll 37) - (hi\ll 3) - hi.
\]

Instead of computing all \(t_0..t_7\) then reducing, you can **interleave**:

- During Comba multiplication, as soon as a high limb \(t_{4+k}\) is produced, immediately fold it into a 192-bit accumulator representing
  \[
  lo + c\cdot hi.
  \]

Concretely, if `w = t[4+k]` (a 32-bit word at high index \(k\in\{0,1,2,3\}\)), fold it into a 6-limb accumulator `r[0..5]` by applying:

- subtract `w` at limb `k` (the \(-hi\) term)
- subtract `(w<<3)` at limb `k` and subtract `(w>>29)` at limb `k+1` (the \(-(hi<<3)\) term)
- add `(w<<5)` at limb `k+1` and add `(w>>27)` at limb `k+2` (the \(+(hi<<37)\) term)

All of these are just `add.cc/addc` and `sub.cc/subc` chains on `r[0..5]`.

After folding \(t_4..t_7\), you still have a 192-bit value `r`; do a **second fold** using the top 64 bits (`r[4], r[5]`) exactly the same way (treat them as the new “hi”).

Finally do one constant-time conditional subtract of \(p\).

### 5.7 Constant-time final subtract on GPU

Compute `r - p` with `sub.cc/subc` to get the borrow, then select `r` vs `r-p` with a mask or predicate select (`selp`) per limb.

---

## 6. Practical recommendation

If you want one modulus to optimize hard for CUDA:

- Prefer **P13** (`0xffffffffffffffffffffffffffffdff1`) because \(c = 2^{13}+2^{4}-1\) makes reduction extremely cheap on 32-bit limbs: it uses only shifts by 13 and 4, plus add/sub carry chains.

If you want the absolute closest prime under your original residue preference \(5 \bmod 8\):

- Use **\(2^{128}-275\)** (`0xfffffffffffffffffffffffffffffeed`), which is extremely reduction-friendly because \(c=275\) is tiny and very sparse in binary.

