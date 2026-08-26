use super::*;

#[cfg(feature = "parallel")]
use std::mem::{ManuallyDrop, MaybeUninit};

#[cfg(feature = "parallel")]
const DENSE_PARALLEL_PAIR_THRESHOLD: usize = 1 << 14;

#[cfg(feature = "parallel")]
fn uninitialized_fold_output<E>(len: usize) -> Vec<MaybeUninit<E>> {
    let mut output = Vec::with_capacity(len);
    output.resize_with(len, MaybeUninit::uninit);
    output
}

#[cfg(feature = "parallel")]
fn assume_initialized_fold_output<E>(output: Vec<MaybeUninit<E>>) -> Vec<E> {
    let mut output = ManuallyDrop::new(output);
    let ptr = output.as_mut_ptr().cast::<E>();
    let len = output.len();
    let capacity = output.capacity();
    // SAFETY: `MaybeUninit<E>` has the same layout as `E`. Every caller writes
    // all `len` entries before converting, and `ManuallyDrop` leaves ownership
    // of the allocation to the returned vector.
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

pub(crate) fn accumulate_dense_round<E: Field + Unreduced>(
    witness_evals: &[E],
    factor_evals: &[E],
    coeff: E,
) -> (E, E) {
    let _span = tracing::trace_span!(
        "dense_extension_reduction_accumulate_round",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    if coeff == E::zero() {
        return (E::zero(), E::zero());
    }

    // Sum the wide products in `E::Product` only when the field has proven
    // that delayed reduction is exact for these batch sizes; otherwise reduce
    // each product immediately so the coefficients stay byte-identical to
    // per-term `Mul` (the `SUM_IS_EXACT` contract).
    let (constant, quadratic) = if E::SUM_IS_EXACT {
        accumulate_dense_round_with::<E, DelayedDeg2<E>>(witness_evals, factor_evals)
    } else {
        accumulate_dense_round_with::<E, DirectDeg2<E>>(witness_evals, factor_evals)
    };
    (coeff * constant, coeff * quadratic)
}

fn accumulate_dense_round_with<E, A>(witness_evals: &[E], factor_evals: &[E]) -> (E, E)
where
    E: Field + Unreduced,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;

    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            return (0..half)
                .into_par_iter()
                .fold(A::zero, |mut acc, i| {
                    let w0 = witness_evals[2 * i];
                    let w1 = witness_evals[2 * i + 1];
                    let a0 = factor_evals[2 * i];
                    let a1 = factor_evals[2 * i + 1];

                    acc.add_constant_product(w0, a0);
                    acc.add_quadratic_product(w1 - w0, a1 - a0);
                    acc
                })
                .reduce(A::zero, A::merge)
                .finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..half {
        let w0 = witness_evals[2 * i];
        let w1 = witness_evals[2 * i + 1];
        let a0 = factor_evals[2 * i];
        let a1 = factor_evals[2 * i + 1];

        acc.add_constant_product(w0, a0);
        acc.add_quadratic_product(w1 - w0, a1 - a0);
    }
    acc.finish()
}

/// Fold a group's factor and first witness together while pre-computing the
/// next round. Later witnesses reuse the folded factor through
/// [`fused_fold_witness_and_accumulate`].
pub(in crate::protocol::extension_opening_reduction) fn fused_fold_group_head_and_accumulate<
    E: Unreduced + Fold,
>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) -> (E, E) {
    let _span = tracing::trace_span!(
        "fused_fold_group_head_and_accumulate",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    debug_assert!(witness_evals.len().is_power_of_two());
    debug_assert!(witness_evals.len() >= 4);

    if E::SUM_IS_EXACT {
        fused_fold_group_head_and_accumulate_with::<E, DelayedDeg2<E>>(
            witness_evals,
            factor_evals,
            r_round,
        )
    } else {
        fused_fold_group_head_and_accumulate_with::<E, DirectDeg2<E>>(
            witness_evals,
            factor_evals,
            r_round,
        )
    }
}

fn fused_fold_group_head_and_accumulate_with<E, A>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) -> (E, E)
where
    E: Field + Unreduced + Fold,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;
    let quarter = half / 2;
    let ctx = E::precompute(r_round);

    #[cfg(feature = "parallel")]
    {
        if quarter >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let mut folded_w = uninitialized_fold_output(half);
            let mut folded_f = uninitialized_fold_output(half);
            let acc = {
                let input_w: &[E] = witness_evals;
                let input_f: &[E] = factor_evals;
                folded_w
                    .par_chunks_mut(2)
                    .zip(folded_f.par_chunks_mut(2))
                    .enumerate()
                    .fold(A::zero, |mut acc, (i, (w_out, f_out))| {
                        let fw0 = E::fold_one(&ctx, input_w[4 * i], input_w[4 * i + 1]);
                        let fw1 = E::fold_one(&ctx, input_w[4 * i + 2], input_w[4 * i + 3]);
                        let fa0 = E::fold_one(&ctx, input_f[4 * i], input_f[4 * i + 1]);
                        let fa1 = E::fold_one(&ctx, input_f[4 * i + 2], input_f[4 * i + 3]);

                        acc.add_constant_product(fw0, fa0);
                        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);
                        w_out[0].write(fw0);
                        w_out[1].write(fw1);
                        f_out[0].write(fa0);
                        f_out[1].write(fa1);
                        acc
                    })
                    .reduce(A::zero, A::merge)
            };
            *witness_evals = assume_initialized_fold_output(folded_w);
            *factor_evals = assume_initialized_fold_output(folded_f);
            return acc.finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..quarter {
        let fw0 = E::fold_one(&ctx, witness_evals[4 * i], witness_evals[4 * i + 1]);
        let fw1 = E::fold_one(&ctx, witness_evals[4 * i + 2], witness_evals[4 * i + 3]);
        let fa0 = E::fold_one(&ctx, factor_evals[4 * i], factor_evals[4 * i + 1]);
        let fa1 = E::fold_one(&ctx, factor_evals[4 * i + 2], factor_evals[4 * i + 3]);

        acc.add_constant_product(fw0, fa0);
        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);
        witness_evals[2 * i] = fw0;
        witness_evals[2 * i + 1] = fw1;
        factor_evals[2 * i] = fa0;
        factor_evals[2 * i + 1] = fa1;
    }
    witness_evals.truncate(half);
    factor_evals.truncate(half);
    acc.finish()
}

/// Fold one witness by one variable and pre-compute the next round's
/// `(constant, quadratic)` accumulation against an already-folded group factor.
pub(in crate::protocol::extension_opening_reduction) fn fused_fold_witness_and_accumulate<
    E: Unreduced + Fold,
>(
    witness_evals: &mut Vec<E>,
    folded_factor: &[E],
    r_round: E,
) -> (E, E) {
    let _span = tracing::trace_span!(
        "fused_fold_witness_and_accumulate",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len() / 2, folded_factor.len());
    debug_assert!(witness_evals.len().is_power_of_two());
    debug_assert!(witness_evals.len() >= 4);

    // The witness fold itself (`E::fold_one`) is always exact; only the product
    // accumulation respects `SUM_IS_EXACT`, matching
    // `accumulate_dense_round`. The factor is folded once by the owning group
    // before this function is called for each member witness.
    if E::SUM_IS_EXACT {
        fused_fold_witness_and_accumulate_with::<E, DelayedDeg2<E>>(
            witness_evals,
            folded_factor,
            r_round,
        )
    } else {
        fused_fold_witness_and_accumulate_with::<E, DirectDeg2<E>>(
            witness_evals,
            folded_factor,
            r_round,
        )
    }
}

fn fused_fold_witness_and_accumulate_with<E, A>(
    witness_evals: &mut Vec<E>,
    folded_factor: &[E],
    r_round: E,
) -> (E, E)
where
    E: Field + Unreduced + Fold,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;
    let quarter = half / 2;
    let ctx = E::precompute(r_round);

    #[cfg(feature = "parallel")]
    {
        if quarter >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let mut folded_w = uninitialized_fold_output(half);

            let acc = {
                let input_w: &[E] = witness_evals;

                folded_w
                    .par_chunks_mut(2)
                    .enumerate()
                    .fold(A::zero, |mut acc, (i, w_out)| {
                        let fw0 = E::fold_one(&ctx, input_w[4 * i], input_w[4 * i + 1]);
                        let fw1 = E::fold_one(&ctx, input_w[4 * i + 2], input_w[4 * i + 3]);
                        let fa0 = folded_factor[2 * i];
                        let fa1 = folded_factor[2 * i + 1];

                        acc.add_constant_product(fw0, fa0);
                        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

                        w_out[0].write(fw0);
                        w_out[1].write(fw1);

                        acc
                    })
                    .reduce(A::zero, A::merge)
            };

            *witness_evals = assume_initialized_fold_output(folded_w);
            return acc.finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..quarter {
        let fw0 = E::fold_one(&ctx, witness_evals[4 * i], witness_evals[4 * i + 1]);
        let fw1 = E::fold_one(&ctx, witness_evals[4 * i + 2], witness_evals[4 * i + 3]);
        let fa0 = folded_factor[2 * i];
        let fa1 = folded_factor[2 * i + 1];

        acc.add_constant_product(fw0, fa0);
        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

        witness_evals[2 * i] = fw0;
        witness_evals[2 * i + 1] = fw1;
    }
    witness_evals.truncate(half);
    acc.finish()
}
