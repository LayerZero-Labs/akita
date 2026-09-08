# Terminal verification

The final fold ends the chain of recursive openings. Its input is the
preceding fold's witness-opening claim and a binding to that witness's inner
A images. The prover sends a small folded response and the verifier checks the
remaining relations directly.

The terminal reveals that response, the ring opening partials, and the inner
images. It does not send the complete original application table. These are
still witness-derived values, so this path provides no zero-knowledge
guarantee.

## The terminal input and response

Start with one polynomial in one source group, one response chunk, and a
base-field opening over $R=F[X]/(X^D+1)$. Block $b$ has digit vector
$\mathbf s_b$, with positions $p$
and inner digits $a$. The public inner matrix is
$\mathbf A\in R^{n_A\times Lk_{\mathrm{in}}}$, where $L$ is the number
of positions and $k_{\mathrm{in}}$ the inner digit count. Its images are

$$
\mathbf t_b=\mathbf A\mathbf s_b.
\tag{1}
$$

The predecessor binds the canonical coefficients of all $\mathbf t_b$
before its ring-switch and sumcheck challenges. The terminal response must
contain exactly those bytes. Thus the prover cannot choose a new collection
of inner images after seeing the opening point handed to the terminal.

Let $Q_p$ be the position weights derived from that opening point, and
$G_a^{\mathrm{in}}$ the inner digit recomposition weights. The honest
prover computes one ring partial per block,

$$
E_b=\sum_{p,a}Q_pG_a^{\mathrm{in}}s_{b,p,a},
\tag{2}
$$

and binds the partials before sampling the terminal sparse challenges
$c_b\in R$. Its folded response is

$$
z_{p,a}=\sum_b c_b s_{b,p,a}.
\tag{3}
$$

The terminal response contains $\mathbf z$, the $E_b$, and the
$\mathbf t_b$. For $m$ live blocks, there are $m$ opening rings,
$mn_A$ inner-image rings, and $Lk_{\mathrm{in}}$ response rings, all
with $D$ coefficients. The schedule fixes these counts, including the
convention for a partial final block.

The response uses recomposed ring values for $E_b$ and $\mathbf t_b$.
There is no need to commit their digits through another B or D matrix,
because the verifier reads them here.

## The two ring checks

The terminal physical relation contains

```text
consistency | A
```

The consistency check is

$$
\boxed{
\sum_b c_b E_b
=
\sum_{p,a}Q_pG_a^{\mathrm{in}}z_{p,a}.
}
\tag{4}
$$

Substituting (2) and (3) shows why it holds for an honest response. Both sides
are the same weighted sum of source digits. The A check is

$$
\boxed{
\mathbf A\mathbf z=\sum_b c_b\mathbf t_b.
}
\tag{5}
$$

For two blocks, the right side is
$c_0\mathbf t_0+c_1\mathbf t_1$. By (1), it equals
$\mathbf A(c_0\mathbf s_0+c_1\mathbf s_1)$, which is the left side by
(3). The same response $\mathbf z$ appears in both checks. Checking only
(4) would leave its connection to the inner commitment unverified; checking
only (5) would leave the opening partials unverified.

The verifier checks (4) and (5) in the native ring. There is no outer
commitment value, B block, D block, polynomial-modulus quotient sumcheck, or
Stage 1, Stage 2, or Stage 3 at this level.

## Recover the scalar opening

The ring checks connect the response to the opening partials, but they have
not yet compared those partials with the incoming scalar claim. Let $B_b$
be the opening point's block weights and $I_\ell$ its coefficient weights.
These weights come from the query. They are different from the random fold
challenges $c_b$.

Writing $E_{b,\ell}$ for coefficient $\ell$ of $E_b$, the remaining
base-field check is

$$
v=\sum_b B_b\sum_{\ell=0}^{D-1}I_\ell E_{b,\ell}.
\tag{6}
$$

For a two-block example, first compute the coefficient pairings
$e_0=\sum_\ell I_\ell E_{0,\ell}$ and
$e_1=\sum_\ell I_\ell E_{1,\ell}$. The verifier checks
$B_0e_0+B_1e_1=v$. This is the original table evaluation regrouped by
block, as derived in
[field to ring reduction](../proving/field-ring-reduction.md).

For openings over a proper extension $E/F$, the terminal first replays
[extension-opening reduction](../proving/extension-opening-reduction.md).
It obtains a transformed opening point, a terminal claim $h$, and a known
factor $\theta$. The direct trace check then has the form

$$
h=\theta\sum_b B_b\,\operatorname{TraceOpen}(E_b).
\tag{7}
$$

Here $\operatorname{TraceOpen}$ recovers the extension-valued coefficient
pairing from the canonical subfield representation at the transformed point.
It is the extension version of the inner sum in (6). Position multiplication
in (4) uses the corresponding canonical ring multipliers.

The verifier computes these pairings in compact subfield coordinates. The
inverse subfield map checks the complete canonical image, so malformed
coordinates cannot be silently projected into a valid value. The preparation
also supports the scheduled opening basis.

Extension-opening reduction has its own sumcheck. Therefore, eliminating the
terminal's ring-switch and range sumchecks does not mean that every terminal
proof is entirely sumcheck-free.

## Response bounds and canonical decoding

Equations (4) and (5) are linear. Their binding argument also needs the
response to be short. The schedule selects one of two mutually exclusive
admission routes:

- A Linf route carries a coefficient cap $B_z$. The verifier checks integer
  representatives with $\lVert\mathbf z\rVert_\infty\le B_z$.
- An L2 route carries no independent Linf cap. After checking the signed
  16 bit representation, canonical Golomb Rice encoding, and scheduled
  payload budget, the verifier checks the exact integer square sum

$$
\sum_{p,a,\ell}z_{p,a,\ell}^2\le B_{2,z}^2.
\tag{8}
$$

The right side is the schedule's squared-norm cap. This sum is not computed
modulo the base field.

Response coefficients use the schedule-selected Golomb Rice encoding. The
decoder checks the exact coordinate count, a bounded maximum quotient,
canonical zero padding in the final byte, and the absence of trailing bytes.
The maximum quotient is derived from $B_z$ on a Linf route and from the
signed 16 bit wire bound on an L2 route. Both routes enforce the scheduled
payload budget and reject values outside `[-32768,32767]` before ring
arithmetic. The signed 16 bit representation is not an independent
schedule-selected Linf security cap on an L2 route.

The prover may search over the bounded fold-response nonces to find a
response that satisfies these bounds. The verifier reconstructs the selected
challenge and checks the bounds itself. It does not trust the prover's
acceptance decision.

## Replay order

The verifier performs the terminal handoff in this order:

1. Check that the scheduled terminal geometry matches the remaining witness
   and that there is no deferred setup-prefix opening.
2. Absorb the incoming inner-image binding and check that the response's
   $\mathbf t_b$ coefficients match it exactly.
3. Prepare the incoming point and claim. Replay extension-opening reduction
   when required.
4. Absorb the $E_b$, consume the terminal fold-response nonce, and derive
   the sparse $c_b$. The L2 route also uses its scheduled challenge
   admission rule.
5. Absorb the remaining response bytes, decode $\mathbf z$, and check
   its bounds.
6. Check the ring relations and the scalar opening.

The predecessor's binding fixes $\mathbf t_b$ before the terminal query
is known, and step 4 fixes $E_b$ before $c_b$ is known. Both ordering
constraints matter to the reduction. Once these direct checks pass, there is
no further witness-opening claim.

## Sparse products and the A matrix

To evaluate the sums involving $c_b$, the verifier uses checked negacyclic
shifts. A challenge monomial $X^j$ shifts coefficients and changes the sign
of coefficients that wrap past $X^{D-1}$. It does not require a dense
quadratic ring product.

Challenge coefficients `1`, `-1`, `2`, and `-2` use addition,
subtraction, and doubling. Other admitted coefficients use exact field
scaling. These sparse products multiply the opening partials and inner images
on the left of (4) and right of (5).

The A relation multiplies the decoded response by the prepared public
matrix. The schedule audit selects an exact CRT and NTT capability before
proof replay. It uses the base prime profile when the signed accumulation
bound fits and adds the signed 16 bit tail only when required. A schedule
whose bound exceeds every supported exact profile is invalid.

Prepared matrix views are derived from the coefficient setup and are never
serialized. These execution choices compute the same relation (5).

## No-panic boundary and code map

The verifier validates payload lengths, ring dimensions, coordinate counts,
sparse support, NTT capability, and matrix ranges before the hot kernels index
prepared state. Malformed terminal bytes return `AkitaError` or
`SerializationError`.

- `crates/akita-verifier/src/protocol/core/suffix.rs` checks the predecessor
  binding and replays the terminal transcript.
- `crates/akita-verifier/src/protocol/core/terminal_direct.rs` checks the
  decoded response norm, ring relations, and opening trace.
- `crates/akita-verifier/src/protocol/core/terminal_ntt.rs` evaluates the
  exact A product.
- `crates/akita-types/src/golomb_rice.rs` implements the codec and its
  canonical-decoding tests.
- `crates/akita-types/src/proof/tail_segments.rs` defines the response
  layout and transcript segments.
- `crates/akita-types/src/field_reduction.rs` implements subfield recovery.
  `crates/akita-pcs/tests/transcript_hardening.rs` supplies end-to-end
  regression coverage for terminal binding and proof tampering.
