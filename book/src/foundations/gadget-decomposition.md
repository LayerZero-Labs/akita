# Gadget decomposition

Gadget decomposition writes a field or ring coefficient as a short list of
small signed digits. Akita uses these digits in commitment and opening
witnesses.

## The gadget matrix

Let \(b = 2^\ell\) and let
\(\mathcal{D}_b = \{-b/2, \ldots, b/2 - 1\}\). For \(n\) coefficients and
\(\delta\) digits per coefficient, the gadget matrix is

\[
\mathbf{G}_{b,n} = I_n \otimes (1, b, \ldots, b^{\delta-1}).
\]

The corresponding decomposition operation writes each coefficient \(x\) as
digits
\(d_0, \ldots, d_{\delta-1} \in \mathcal{D}_b\) such that

\[
x = \sum_{j=0}^{\delta-1} d_j b^j \pmod q.
\]

The first digit is the least significant digit. Flat decomposition output is
digit major: all coefficient digits for level zero come first, followed by all
digits for level one, and so on.

## The representable interval

Balanced digits have an asymmetric range. Define

\[
S_\delta = 1 + b + \cdots + b^{\delta-1}.
\]

Then \(\delta\) digits represent every integer in

\[
[-M_\delta, T_\delta], \qquad
M_\delta = \frac{b}{2}S_\delta, \qquad
T_\delta = \left(\frac{b}{2} - 1\right)S_\delta.
\]

The negative side reaches farther because the digit set includes \(-b/2\) but
not \(b/2\). For example, three base 4 digits represent \([-42, 21]\), while
two base 8 digits represent \([-36, 27]\).

The exact range helper returns `None` for an endpoint when the mathematical
reach is larger than `u128`. The saturating helper instead returns a
conservative finite lower bound. Code that decides whether a value is accepted
uses the exact range.

## How Akita chooses the depth

Full field decomposition uses
\(\delta = \lceil F / \ell \rceil\), where \(F\) is the field bit width. If
\(\delta\ell = F\), the decomposition uses an asymmetric centering threshold
that matches the positive reach above. This avoids adding a digit only to make
the integer interval symmetric. For example, a 128 bit field with \(\ell = 3\)
uses 43 digits.

A bounded committed source follows a different rule. A declared signed width
\(B\) denotes
\([-2^{B-1}, 2^{B-1} - 1]\). Akita starts with
\(\lceil B / \ell \rceil\) digits, checks the exact positive reach
\(T_\delta\), and adds one digit if that reach is too small. This extra check
matters when \(\ell\) divides \(B\) exactly.

A nonnegative magnitude bounded by \(2^m\) needs signed width \(B = m + 1\).
For this reason, a source that may contain any `u64` value declares width 65,
not 64. When a caller has an exact infinity norm cap instead of a signed bit
width, Akita selects the smallest depth whose exact interval contains that
cap.

## Commitment and opening roles

The required depth depends on what is being decomposed.

| Role | Bound used to choose the depth |
| --- | --- |
| Root committed source | `log_commit_bound` |
| Recursive committed source | The recursive source bound selected by the schedule |
| Opening values and setup prefixes | `log_open_bound` or the full field width |
| Folded response | The exact response cap selected by the schedule |

The source bound and source class are separate parts of the contract. A width
of one does not select the `UnitOneHot` source class. The class describes the
source shape, while the width describes its allowed coefficient interval.

A committed source is accepted only when every coefficient satisfies both the
declared interval and the interval represented by the selected digit depth.
The accepted range is their intersection. Producers reject a value outside
that intersection instead of committing a truncated decomposition.

## Signed digit storage

The digit width determines the signed storage type.

| Basis exponent \(\ell\) | Digit interval | Storage |
| --- | --- | --- |
| 1 through 8 | \([-2^{\ell-1}, 2^{\ell-1}-1]\) | `i8` |
| 9 through 16 | \([-2^{\ell-1}, 2^{\ell-1}-1]\) | `i16` |

Here \(\ell\) is the base 2 logarithm of the basis. Thus \(\ell = 10\) means
base 1024 with digits in \([-512, 511]\), not base 10.

Production schedule validation keeps opening bases within the `i8` path and
inner bases within the `i16` path. The current schedule search considers
opening exponents from 3 through 6. It considers inner exponents up to 10 for
fields with 32 bits and up to 11 for fields with 64 or 128 bits.

The ring API can return field elements, `i8` digits, or `i16` digits. The
packed `i8` path uses AVX2 for supported canonical 32 bit field slices when the
coefficient count is a multiple of eight. Its scalar fallback has the same
digit major layout.

## Implementation map

- Gadget arithmetic and ring decomposition:
  `crates/akita-algebra/src/ring/cyclotomic/decomposition.rs`
- Exact ranges and depth selection:
  `crates/akita-types/src/sis/decomposition_digits.rs`
- Source acceptance contract:
  `crates/akita-types/src/sis/committed_source.rs`
- Signed digit storage:
  `crates/akita-types/src/signed_digit.rs`
- Schedule basis limits:
  `crates/akita-schedules/src/runtime.rs`
- Production basis search:
  `crates/akita-config/src/proof_optimized.rs`

The mathematical definitions come from paper section 2.2. Appendix B.1.3
describes the implementation choices behind the balanced decomposition.
