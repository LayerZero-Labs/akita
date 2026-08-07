# Structured E Term

This note spells out the verifier's structured `E` / `e_hat` contribution in the
relation-matrix evaluation.

## Definitions

For group `g`:

- `K_g` is the number of claims in the group.
- `B_g` is the total number of live blocks in the group.
- `d_A(g)` is the source / A ring dimension.
- `d_D(g)` is the D / opening role ring dimension.
- `d_0` is the common coefficient block size used by the verifier point split.
- `q_D = d_A(g) / d_D(g)` is the number of D-role subcolumns inside one
  A-sized source ring.
- `L_D = d_D(g) / d_0` is the number of common-block lanes inside one D ring.
- `delta_open,g` is the number of opening decomposition digits.
- `lambda_cons(g) = eq(tau_1, i_cons(g))` is the row-combination weight for
  group `g`'s consistency row.
- `c_{g,k,b}(alpha)` is the challenge ring for claim `k`, block `b`, evaluated
  with the A/source alpha powers.
- `G_j^open` is the opening digit gadget weight.
- `x_addr` is the high-address part of the stage-2 evaluation point, after the
  common low coefficient coordinates have been split off.

The low coefficient factor

```math
K_\alpha(x_{\mathrm{coeff}})
=
\sum_{c=0}^{d_0-1}
\alpha^c eq(x_{\mathrm{coeff}}, c)
```

is applied outside this term. The equation below is the high-address part.

## E Segment Address

Fix one group `g`. Let `C` be the witness chunk count. The current layout
requires `B_g` to be a multiple of `C`, so every chunk has the same number of
live blocks:

```math
N_g = \frac{B_g}{C}.
```

The first global block in chunk `c` is therefore:

```math
S_{g,c}=cN_g.
```

The local block index inside the chunk is `beta`, with:

```math
0 \le \beta < N_g.
```

The corresponding global block is:

```math
b = cN_g+\beta.
```

For each group `h`, define the per-chunk unit lengths:

```math
Z_h = P_h\delta_{\mathrm{wit},h}\delta_{\mathrm{fold},h}d_A(h),
```

```math
E_h = K_h N_h\delta_{\mathrm{open},h}d_A(h),
```

```math
T_h = K_h N_h n_{A,h}\delta_{\mathrm{commit},h}d_A(h).
```

Let:

```math
W_{\mathrm{chunk}} = \sum_h (Z_h+E_h+T_h)
```

be the physical width of one full chunk across all groups. Also define the
within-chunk E-start offset for group `g`:

```math
O_g =
\sum_{h \prec g}(Z_h+E_h+T_h)+Z_g,
```

where `h \prec g` means that group `h` appears before group `g` in the relation
group order. Then the physical coefficient offset where chunk `c` of group `g`
starts its `E` segment is the affine layout constant:

```math
\mathcal{E}_{g,c}=cW_{\mathrm{chunk}}+O_g.
```

Since every segment length above is a multiple of `d_0`, define the corresponding
lane-level constants:

```math
\overline{W}_{\mathrm{chunk}}=\frac{W_{\mathrm{chunk}}}{d_0},
\qquad
\overline{O}_g=\frac{O_g}{d_0}.
```

The physical start of the D-role ring segment for `(g,c,k,beta,s,j)` is:

```math
addr_E^{phys,g}(c,k,\beta,s,j)
=
\mathcal{E}_{g,c}
+
\Big(
\big((k N_g+\beta)q_D+s\big)\delta_{\mathrm{open},g}
+j
\Big)d_D(g),
```

where:

```math
q_D=\frac{d_A(g)}{d_D(g)}.
```

Because `d_0` divides both `\mathcal{E}_{g,c}` and `d_D(g)`, the corresponding
relation-lane address is:

```math
addr_E^{lane,g}(c,k,\beta,s,j,\ell)
=
c\overline{W}_{\mathrm{chunk}}+\overline{O}_g
+
\Big(
\big((k N_g+\beta)q_D+s\big)\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell,
```

where:

```math
L_D=\frac{d_D(g)}{d_0},
\qquad
0 \le \ell < L_D.
```

### Single-Chunk Simplification

If group `g` has only one witness chunk, then:

```math
C=1,
\qquad
N_g=B_g.
```

The lane address simplifies to:

```math
addr_E^{lane,g}(0,k,\beta,s,j,\ell)
=
\overline{O}_g
+
\Big(
\big((k B_g + \beta)q_D+s\big)\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell.
```

## Structured E Equation

For a fixed group `g`, the verifier's high-address structured E contribution is:

```math
S_E^g
=
\sum_{c=0}^{C-1}
\sum_{k=0}^{K_g-1}
\sum_{\beta=0}^{N_g-1}
\lambda_{\mathrm{cons}(g)}
c_{g,k,cN_g+\beta}(\alpha)
\sum_{s=0}^{q_D-1}
\sum_{j=0}^{\delta_{\mathrm{open},g}-1}
\sum_{\ell=0}^{L_D-1}
G^{open}_j
\alpha^{\ell d_0}
eq\!\left(
x_{\mathrm{addr}},
addr_E^{lane,g}(c,k,\beta,s,j,\ell)
\right).
```

Equivalently, substituting the lane address:

```math
S_E^g
=
\sum_{c=0}^{C-1}
\sum_{k=0}^{K_g-1}
\sum_{\beta=0}^{N_g-1}
\lambda_{\mathrm{cons}(g)}
c_{g,k,cN_g+\beta}(\alpha)
\sum_{s=0}^{q_D-1}
\sum_{j=0}^{\delta_{\mathrm{open},g}-1}
\sum_{\ell=0}^{L_D-1}
G^{open}_j
\alpha^{\ell d_0}
eq\!\left(
x_{\mathrm{addr}},
c\overline{W}_{\mathrm{chunk}}+\overline{O}_g
+
\Big(
\big((kN_g+\beta)q_D+s\big)
\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell
\right).
```

The full structured E contribution is then:

```math
S_E = \sum_g S_E^g.
```

## Hardened Polynomial Notation

To avoid overloading the chunk index `c` with the challenge value notation, write
the evaluated challenge as the polynomial/function:

```math
C(g,k,b,\alpha) = c_{g,k,b}(\alpha).
```

Also write the fixed opening gadget and alpha lane power as:

```math
G_{\mathrm{open}}(j)=G^{open}_j,
\qquad
A(\ell d_0)=\alpha^{\ell d_0}.
```

With this notation, and moving the challenge factor into the innermost product,
the fixed-group equation is:

```math
S_E^g
=
\sum_{c=0}^{C-1}
\sum_{k=0}^{K_g-1}
\sum_{\beta=0}^{N_g-1}
\lambda_{\mathrm{cons}(g)}
\sum_{s=0}^{q_D-1}
\sum_{j=0}^{\delta_{\mathrm{open},g}-1}
\sum_{\ell=0}^{L_D-1}
C(g,k,cN_g+\beta,\alpha)
G_{\mathrm{open}}(j)
A(\ell d_0)
eq\!\left(
x_{\mathrm{addr}},
c\overline{W}_{\mathrm{chunk}}+\overline{O}_g
+
\Big(
\big((kN_g+\beta)q_D+s\big)
\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell
\right).
```

## Interpretation

The term says: for every group, claim, and live block, multiply the A-native
challenge evaluation `c_{g,k,b}(alpha)` by the consistency-row weight. Then scan
the D-role representation of `e_hat` across:

1. D subcolumns `s` inside the A-sized source ring,
2. opening digits `j`,
3. common-block lanes `ell` inside the D ring.

The factor `alpha^{ell d_0}` accounts for which common-block lane of the D-role
ring is being evaluated. The missing low coefficient powers `alpha^c` are shared
by all relation terms and are applied later through `K_alpha(x_coeff)`.

## Multilinear Shifted-Equality Variation

The hardened equation above contains the shifted address equality

```math
eq\!\left(
x_{\mathrm{addr}},
c\overline{W}_{\mathrm{chunk}}+\overline{O}_g
+
\Big(
\big((kN_g+\beta)q_D+s\big)
\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell
\right).
```

To use this factor as a multilinear object over the local E coordinates, first
view it as a table indexed by `(c,k,beta,s,j,ell)`, then take the multilinear
extension with the equality kernel on those coordinates. Let
`z_E=(z_c,z_k,z_\beta,z_s,z_j,z_\ell)` be the local multilinear point. Define:

```math
\widetilde{Eq}_E^g(z_E; x_{\mathrm{addr}})
:=
\sum_{c=0}^{C-1}
\sum_{k=0}^{K_g-1}
\sum_{\beta=0}^{N_g-1}
\sum_{s=0}^{q_D-1}
\sum_{j=0}^{\delta_{\mathrm{open},g}-1}
\sum_{\ell=0}^{L_D-1}
eq\!\left(z_E,(c,k,\beta,s,j,\ell)\right)
eq\!\left(
x_{\mathrm{addr}},
c\overline{W}_{\mathrm{chunk}}+\overline{O}_g
+
\Big(
\big((kN_g+\beta)q_D+s\big)
\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell
\right).
```

Equivalently, if the local tuple point is also denoted by `x_E`, this is the
compact form:

```math
\widetilde{Eq}_E^g(x_E; x_{\mathrm{addr}})
=
\sum_{c,k,\beta,s,j,\ell}
eq\!\left(x_E,(c,k,\beta,s,j,\ell)\right)
eq\!\left(
x_{\mathrm{addr}},
c\overline{W}_{\mathrm{chunk}}+\overline{O}_g
+
\Big(
\big((kN_g+\beta)q_D+s\big)
\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell
\right),
```

where the summation ranges are the live ranges from the previous display. The
second `eq` factor is the original shifted address table value; the first `eq`
factor is what performs the multilinear extension over `c`, `k`, `beta`, `s`,
`j`, and `ell`.

The hardened fixed-group E contribution can then use this multilinear extension
in place of the raw shifted equality factor:

```math
S_E^g
=
\sum_{c=0}^{C-1}
\sum_{k=0}^{K_g-1}
\sum_{\beta=0}^{N_g-1}
\lambda_{\mathrm{cons}(g)}
\sum_{s=0}^{q_D-1}
\sum_{j=0}^{\delta_{\mathrm{open},g}-1}
\sum_{\ell=0}^{L_D-1}
C(g,k,cN_g+\beta,\alpha)
G_{\mathrm{open}}(j)
A(\ell d_0)
\widetilde{Eq}_E^g
\!\left((c,k,\beta,s,j,\ell); x_{\mathrm{addr}}\right).
```

On Boolean tuple inputs, the extension agrees with the original shifted equality:

```math
\widetilde{Eq}_E^g((c,k,\beta,s,j,\ell); x_{\mathrm{addr}})
=
eq\!\left(
x_{\mathrm{addr}},
c\overline{W}_{\mathrm{chunk}}+\overline{O}_g
+
\Big(
\big((kN_g+\beta)q_D+s\big)
\delta_{\mathrm{open},g}
+j
\Big)L_D
+\ell
\right).
```

## Compact Verifier Evaluation

At the end of the sum-check, the verifier must evaluate
`\widetilde{Eq}_E^g(z_E;x_{\mathrm{addr}})` at the random local point

```math
z_E=(z_c,z_k,z_\beta,z_s,z_j,z_\ell).
```

There is a two-state, logarithmic-time evaluation analogous to the standard
shifted-EQ carry algorithm, but it requires a binary-aligned E layout. The
required conditions are:

```math
\begin{aligned}
C&=2^{m_c}, & K_g&=2^{m_k}, & N_g&=2^{m_\beta},\\
q_D&=2^{m_s}, &
\delta_{\mathrm{open},g}&=2^{m_j}, &
L_D&=2^{m_\ell},\\
\overline W_{\mathrm{chunk}}&=2^w.
\end{aligned}
```

Write

```math
m_E=m_k+m_\beta+m_s+m_j+m_\ell
```

and require the complete E interval to remain inside the low `w` bits of its
chunk:

```math
m_c\le n_{\mathrm{addr}}-w,
\qquad
\overline O_g+2^{m_E}-1<2^w.
```

Under these conditions, the mixed-radix local offset is just bit
concatenation. For big-endian bit strings,

```math
y
=
\Big(
\big((kN_g+\beta)q_D+s\big)
\delta_{\mathrm{open},g}+j
\Big)L_D+\ell
=
(k\,\|\,\beta\,\|\,s\,\|\,j\,\|\,\ell)_2.
```

Consequently, if

```math
z_y=z_k\,\|\,z_\beta\,\|\,z_s\,\|\,z_j\,\|\,z_\ell,
```

then the local equality kernel factors as

```math
eq\!\left(z_E,(c,k,\beta,s,j,\ell)\right)
=
eq(z_c,c)\,eq(z_y,y).
```

Split the big-endian address point after its high `n_{\mathrm{addr}}-w`
coordinates:

```math
x_{\mathrm{addr}}=(x_{\mathrm{hi}},x_{\mathrm{lo}}),
\qquad |x_{\mathrm{lo}}|=w.
```

Because `\overline W_{\mathrm{chunk}}=2^w` and the low interval does not
overflow, the address bits also concatenate:

```math
\operatorname{bin}_{n_{\mathrm{addr}}}
\!\left(c2^w+\overline O_g+y\right)
=
\operatorname{bin}_{n_{\mathrm{addr}}-w}(c)
\,\|\,
\operatorname{bin}_{w}(\overline O_g+y).
```

The desired evaluation therefore separates into two compact factors:

```math
\boxed{
\widetilde{Eq}_E^g(z_E;x_{\mathrm{addr}})
=
\operatorname{PadEq}(x_{\mathrm{hi}},z_c)
\cdot
\operatorname{ShiftEq}_w
\!\left(x_{\mathrm{lo}},\overline O_g,z_y\right)
}
```

Here

```math
\operatorname{PadEq}(x_{\mathrm{hi}},z_c)
:=
\sum_{c\in\{0,1\}^{m_c}}
eq(z_c,c)
eq\!\left(x_{\mathrm{hi}},
\operatorname{bin}_{n_{\mathrm{addr}}-w}(c)\right).
```

If `x_{\mathrm{hi}}=x_{\mathrm{pad}}\,\|\,x_c`, where
`|x_c|=m_c`, this factor is evaluated directly as

```math
\operatorname{PadEq}(x_{\mathrm{hi}},z_c)
=
\left(\prod_{r\in x_{\mathrm{pad}}}(1-r)\right)
eq(x_c,z_c).
```

The second factor is exactly the shifted-EQ MLE:

```math
\operatorname{ShiftEq}_w(x_{\mathrm{lo}},\overline O_g,z_y)
=
\sum_{y\in\{0,1\}^{m_E}}
eq(z_y,y)
eq\!\left(
x_{\mathrm{lo}},
\operatorname{bin}_w(\overline O_g+y)
\right).
```

It is evaluated from least significant bit to most significant bit with only
two field states, indexed by the carry:

```text
dp[0] <- 1; dp[1] <- 0
for t = 0 .. w-1:
    next[0] <- 0; next[1] <- 0
    a <- bit(O_bar_g, t)
    rho <- x_lo[w-1-t]
    choices <- {0,1} if t < m_E, otherwise {0}

    for carry in {0,1}:
        for b in choices:
            total <- a + carry + b
            out_bit <- total mod 2
            next_carry <- floor(total / 2)
            addr_weight <- rho if out_bit = 1, otherwise 1-rho
            y_weight <- 1 if t >= m_E
                        else z_y[m_E-1-t] if b = 1
                        else 1-z_y[m_E-1-t]
            next[next_carry] += dp[carry] * addr_weight * y_weight

    dp <- next
return PadEq(x_hi, z_c) * (dp[0] + dp[1])
```

The interval condition makes the final low carry zero. The algorithm performs

```math
O(n_{\mathrm{addr}}+m_c+m_k+m_\beta+m_s+m_j+m_\ell)
```

field operations and uses `O(1)` field working memory beyond the input points.
It never materializes the E equality table.

### Why the alignment conditions matter

The logarithmic formula above is not valid for an arbitrary instance of the
hardened equation. In particular:

- if `N_g` or `\delta_{\mathrm{open},g}` is not a power of two, the tuple
  `(k,\beta,s,j,\ell)` is not the binary concatenation of its row-major local
  offset; and
- if `\overline W_{\mathrm{chunk}}` is not a power of two, `c` is multiplied by
  a genuine binary stride instead of being placed in a disjoint high-bit field.

The exact general evaluation can still be written as a carry transducer. Define
the affine coefficients

```math
\begin{aligned}
a_c&=\overline W_{\mathrm{chunk}}, &
a_k&=N_gq_D\delta_{\mathrm{open},g}L_D, &
a_\beta&=q_D\delta_{\mathrm{open},g}L_D,\\
a_s&=\delta_{\mathrm{open},g}L_D, &
a_j&=L_D, &
a_\ell&=1.
\end{aligned}
```

Initialize an integer carry state with `q_0=\overline O_g`. At address bit `t`,
branch on the current bit `b_{v,t}` of every live local coordinate and apply

```math
T_t=q_t+\sum_{v\in\{c,k,\beta,s,j,\ell\}}a_vb_{v,t},
\qquad
u_t=T_t\bmod 2,
\qquad
q_{t+1}=\left\lfloor\frac{T_t}{2}\right\rfloor.
```

Each transition is multiplied by the corresponding local multilinear weights
and by the address factor `(1-x_{\mathrm{addr},t})` or
`x_{\mathrm{addr},t}` selected by `u_t`. For a non-power-of-two live range, a
three-state comparison flag for that coordinate excludes bit strings outside
the range. After all address bits, accept only states with zero final carry and
with all six coordinates inside their live ranges. This DP is exact.

However, its carry set is no longer `{0,1}`. A constant multiplier such as
`c\overline W_{\mathrm{chunk}}` can keep a distinct reachable carry for each of
linearly many prefixes of `c` (and the unrestricted multiplier transducer has
up to `\overline W_{\mathrm{chunk}}` carry states). If `Q` carry values are
reachable, the direct six-axis recurrence costs

```math
O\!\left(n_{\mathrm{addr}}\,2^6\,Q\,3^6\right)
```

operations in the worst case; the `3^6` factor is only needed when all six
ranges require live-prefix comparison states. Thus the hardened equation by
itself does not imply a verifier cost logarithmic in all six range sizes. Such a
claim needs the binary-alignment conditions above, a binary-padded physical
layout that enforces them, or a different polynomial coordinate that uses one
flat contiguous E offset.
