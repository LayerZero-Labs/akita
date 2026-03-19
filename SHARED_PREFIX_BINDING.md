# Shared-Prefix Matrix Distribution: Hachi-Specific Security Argument

This note replaces the earlier generic argument.

The previous version was too broad. In particular, it claimed that any proof
that "reduces to SIS on one of {A, B, D}" automatically survives the
shared-prefix distribution. That is not true in general: if a proof genuinely
needs hardness of a mixed-role matrix that reuses overlapping columns from two
roles in the same row space, then the shared-prefix correlation can matter.

So the right question is not "is every SIS-style argument preserved?"  The
right question is:

> Do the actual Hachi and repo-specific Labrador proof paths ever need a
> mixed-role SIS instance, or can every reduction be rewritten so that it lands
> on a single role matrix whose marginal distribution is unchanged?

For the current repo, the answer is: the proof path can be kept single-role.
That is the real reason the shared-prefix construction is safe here.

## Distribution Change

The protocol uses three public matrices:

- `A`: inner commitment key
- `B`: outer commitment key
- `D`: opening / linear-garbage key

Old distribution:

$$
A[r,c] = \mathrm{XOF}(\text{seed}, "A", r, c), \quad
B[r,c] = \mathrm{XOF}(\text{seed}, "B", r, c), \quad
D[r,c] = \mathrm{XOF}(\text{seed}, "D", r, c).
$$

New shared-prefix distribution:

$$
M[r,c] = \mathrm{XOF}(\text{seed}, "shared", r, c),
$$

with

$$
A = M_{[n_A, m_A]}, \qquad
B = M_{[n_B, m_B]}, \qquad
D = M_{[n_D, m_D]}.
$$

Hence the three roles are correlated on their overlap, but each individual role
still has the same marginal distribution as before: a top-left prefix of a
uniform matrix is itself uniform.

In particular, `D` is **not** independently sampled from `A` and `B` under the
shared-prefix construction. It is another prefix view of the same backing
matrix, so it is correlated with them just as `A` and `B` are correlated with
each other.

That marginal fact is enough only if the concrete reductions target one role at
a time. The rest of this note checks exactly that.

## What The Earlier Note Got Wrong

The earlier note implicitly assumed the following principle:

> "If each of `A`, `B`, `D` is individually uniform, then any proof that
> mentions them is fine."

That is too strong.

The user's counterexample pattern is the right thing to worry about: a proof
that truly reduced to a same-row mixed matrix built from overlapping copies of
two roles could fail under shared-prefix correlation. So we must inspect the
actual Hachi proof, not argue generically.

In the Hachi paper, the delicate point is Section 4.1: the printed weak-binding
lemma is stated using a mixed matrix notation `[A | B]`. If that statement were
the only available reduction, the shared-prefix argument would indeed be on
shaky ground.

The key fix is that Hachi does not need that mixed-role lemma. A cleaner
single-role case split is enough for all downstream uses.

## Repaired Weak-Binding Lemma For Hachi Section 4.1

Recall the Section 4.1 two-tier commitment:

1. For each block, compute an inner witness `s_i`.
2. Compute `t_i = A s_i`.
3. Decompose `t_i` to `\hat t_i`.
4. Commit with `u = B \hat t`.

A weak opening is a tuple `(s_i, \hat t_i, c_i)_i` satisfying:

- `A s_i = G \hat t_i`
- `B \hat t = u`
- `c_i` is invertible
- `||c_i||_1 <= \bar\omega`
- `||c_i s_i||_\infty <= \bar\beta`
- `||\hat t||_\infty <= \bar\gamma`

Suppose we are given two weak openings
`(s_i, \hat t_i, c_i)_i` and `(s'_i, \hat t'_i, c'_i)_i` for the same
commitment `u`, and suppose `s_j != s'_j` for some `j`.

Then we can extract a short kernel vector for a single role matrix by a simple
case split.

### Case 1: the outer witnesses differ

If `\hat t != \hat t'`, define

$$
x_B := \hat t - \hat t'.
$$

Then `x_B != 0`, and

$$
B x_B = B\hat t - B\hat t' = u - u = 0.
$$

Also,

$$
||x_B||_\infty \le 2 \bar\gamma.
$$

So this is already a short nonzero MSIS solution for `B`.

### Case 2: the outer witnesses are equal

Now assume `\hat t = \hat t'`. Since the openings still differ, pick an index
`j` with `s_j != s'_j`. Define

$$
x_A := c'_j (c_j s_j) - c_j (c'_j s'_j)
    = c_j c'_j (s_j - s'_j).
$$

Because both `c_j` and `c'_j` are invertible and `s_j != s'_j`, we have
`x_A != 0`.

Moreover,

$$
A x_A
= c'_j c_j A s_j - c_j c'_j A s'_j
= c'_j c_j G \hat t_j - c_j c'_j G \hat t'_j
= 0,
$$

since `\hat t_j = \hat t'_j`.

The norm bound is

$$
||x_A||_\infty
\le ||c'_j (c_j s_j)||_\infty + ||c_j (c'_j s'_j)||_\infty
\le \bar\omega \bar\beta + \bar\omega \bar\beta
= 2 \bar\omega \bar\beta.
$$

So this is a short nonzero MSIS solution for `A`.

### Consequence

The Section 4.1 commitment does not require any mixed-role SIS instance.
Two distinct weak openings always yield:

- either a short kernel vector for `B`, or
- a short kernel vector for `A`.

This is stronger, and more useful for the shared-prefix setting, than the mixed
notation printed in Lemma 5 of the paper.

Because the shared-prefix change leaves the marginal distributions of `A` and
`B` unchanged, this repaired weak-binding lemma is fully compatible with the
new setup distribution.

## Hachi Appendix A Already Uses Single-Role Case Splits

Once Section 4.1 is repaired as above, the rest of the Hachi proof path fits
the shared-prefix distribution cleanly.

The crucial point is Appendix A, Lemma 6 (the coordinate-wise special
soundness argument for Figure 3). That lemma already performs the correct
single-role case split:

- if two accepting transcripts produce different `\hat t`, it outputs a short
  solution for `B`
- if they produce different `\hat w`, it outputs a short solution for `D`
- otherwise it fixes `\hat t` and `\hat w` and uses the `A`-equations only to
  extract a valid weak opening

So Hachi's stage-1 proof never needs a mixed SIS instance spanning multiple
roles. The proof is already organized role-by-role.

This matters because the shared-prefix optimization changes only the joint
distribution of `(A, B, D)`, not the marginal distribution of any one role.
Lemma 6 only ever lands on one role at a time, so the correlation is irrelevant
to that reduction.

For `D` specifically, the point is: it is not statistically separate, but it is
reduction-separate. Appendix A uses `D` only in the `\hat w` branch, where a
disagreement immediately yields a short kernel vector for `D` alone; it is not
combined with `A` or `B` into one SIS instance.

## Hachi Section 4.3 And Recursion

Section 4.3 (Figures 4, 5, and 6; Lemmas 7, 8, and 9) does not introduce any
new mixed-role SIS instance.

Those lemmas say: either

- extract the required witness, or
- break binding of the commitment scheme `Com`

But `Com` is exactly the two-tier Section 4.1 commitment discussed above. So
after replacing the printed mixed-role Lemma 5 with the repaired single-role
case split, the recursive argument also targets only `A` or `B` individually.

In other words:

1. Stage 4.2 uses `B` and `D` separately, then extracts a weak opening.
2. Stage 4.3 only needs binding of `Com`.
3. Binding of `Com` can be proved using only `A` or `B` individually.

Therefore the full Hachi proof path can be rewritten without ever requiring a
same-row mixed matrix built from multiple role views.

## Why The Dangerous Counterexample Does Not Apply Here

The user's concern was exactly right in principle:

> if some proof step reduced to a mixed matrix that put overlapping columns of
> two role matrices into the same kernel equation, the shared-prefix
> distribution could create trivial short solutions.

That concern is real. The point of this note is not to deny it.

The point is that the actual Hachi proof does not need such a step:

- Section 4.1 commitment binding can be restated as "break `A` or break `B`"
- Appendix A, Lemma 6 already says "break `B` or break `D` or extract"
- Section 4.3 only reuses commitment binding

So the dangerous mixed-role pattern is not part of the Hachi reduction once the
weak-binding lemma is stated in the correct case-split form.

## Mapping Back To The Current Code

The repo implementation mirrors that same separation by role.

### Hachi

The quadratic-equation / ring-switch layer organizes the public relation as a
stack of row blocks:

- `D * \hat w = v`
- `B * \hat t = u`
- one scalar evaluation row
- one scalar fold row
- `A * z = ...`

The implementation computes those blocks separately rather than as one
mixed-role kernel matrix:

- `compute_r_split_eq` handles `D` rows, `B` rows, and `A` rows in separate
  branches
- `generate_y` builds the corresponding right-hand side as
  `[v | u | y_eval | 0 | 0_{N_A}]`
- recursive `w` commitments are still the same two-tier `A/B` commitment, via
  `commit_w`

So the code follows exactly the proof structure described above: separate role
checks, not a mixed-role SIS reduction.

### Repo-specific Labrador

The Hachi paper only says that one can hand off to LaBRADOR; it does not give a
paper proof for the exact linear-only Labrador implemented in this repo. So for
Labrador we should be precise about what is actually established here.

In the repo's current linear-only Labrador implementation, the verifier checks
the commitment matrices role-by-role:

- `inner_opening_payload = B * inner_opening_digits`
- `linear_garbage_payload = D * linear_garbage_digits`
- `A z = sum_i c_i t_i`

After those checks, the remaining relations are on public transcript weights,
`amortized_phi`, the witness, and the `h_ij` values; they do not introduce a new
mixed-role kernel equation involving two setup matrices at once.

Likewise, the recursive constraint builders scalarize the next-level statement
into:

- `B`-row commitment constraints for `inner_opening_payload`
- `D`-row commitment constraints for `linear_garbage_payload`
- `A`-row amortized-opening constraints
- matrix-free linear-garbage and diagonal constraints

So the implemented Labrador verifier structure is also compatible with the same
single-role viewpoint: the first differing commitment-side object would break
`B` or `D`; once those are fixed, a disagreement in the amortized opening lands
on `A`.

So here too, `D` is correlated in distribution but isolated in the check path:
the verifier tests `linear_garbage_payload = D * linear_garbage_digits` as its own role-specific condition before
moving on to the `A`-based amortized-opening check.

This is enough to rule out the specific shared-prefix flaw discussed above.

## What Is Actually Established

The correct claim is the following.

### Established

For the current Hachi proof path, and for the current repo's linear-only
Labrador verifier structure, the shared-prefix distribution is compatible with
security because every matrix-based extraction step can be routed to a single
role matrix:

- `A` only
- `B` only
- or `D` only

Since each individual role retains the same uniform marginal distribution under
shared-prefix sampling, the underlying MSIS assumptions used by those
role-specific reductions are unchanged.

### Not established

This is **not** a generic theorem for arbitrary protocols using three correlated
matrices.

In particular, this note does **not** claim that a proof reducing to a genuine
mixed-role SIS instance would remain valid under the shared-prefix distribution.
If a future proof step really needs that kind of mixed matrix, the shared-prefix
optimization must be re-evaluated.

## Bottom Line

The earlier note was wrong because it tried to prove too much.

The repaired argument is narrower and concrete:

1. Each role matrix keeps the same marginal distribution under shared-prefix
   sampling.
2. Hachi's actual proof can be rewritten so that every SIS extraction lands on
   `A`, `B`, or `D` individually.
3. The current repo's linear-only Labrador checks are organized the same way.

That is the right reason the shared-prefix setup is safe here.

## Code Anchors

| Component | File | Relevant symbols |
| --- | --- | --- |
| Shared backing and role views | `src/protocol/commitment/utils/shared_public_matrix.rs` | `SharedPublicMatrix`, `SharedRoleLayout`, `RoleMatrixView` |
| Prefix-stable matrix derivation | `src/protocol/commitment/utils/matrix.rs` | `derive_public_matrix`, `ShakeXofRng` |
| Hachi shared setup derivation | `src/protocol/commitment/commit.rs` | `derive_shared_public_matrix`, `build_shared_ntt_slot` |
| Quadratic-equation row split | `src/protocol/quadratic_equation.rs` | `compute_r_split_eq`, `generate_y` |
| Recursive Hachi commitment | `src/protocol/ring_switch.rs` | `commit_w`, `WCommitmentConfig` |
| Labrador shared setup derivation | `src/protocol/labrador/setup.rs` | `LabradorSetupMatrices::new` |
| Labrador recursive constraints | `src/protocol/labrador/constraints.rs` | `build_outer_commitment_constraints`, `build_linear_garbage_commitment_constraints`, `build_amortized_opening_constraints` |
| Labrador verifier checks | `src/protocol/labrador/verifier.rs` | `verify_single_level`, `verify_tail_level` |
