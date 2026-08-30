//! Linear-time coefficient functionals for reduced negacyclic relations.

use akita_error::AkitaError;
use jolt_field::{ExtField, Field};

/// Evaluate every reduced shift of one public multiplier at `alpha`.
///
/// For `A(X) = sum_k coefficients[k] X^k` in a ring of dimension `d`, the
/// returned entry `j` is
///
/// `kappa[j] = (A(X) X^j mod (X^d + 1))(alpha)`.
///
/// The implementation uses the signed-wrap recurrence and never divides by
/// `alpha^d + 1`, so roots of the polynomial modulus are valid evaluation
/// points.
///
/// # Errors
///
/// Returns an error unless the coefficient count is a nonzero power of two,
/// or if the output allocation cannot be reserved.
pub fn residue_kernel<F, E>(coefficients: &[F], alpha: E) -> Result<Vec<E>, AkitaError>
where
    F: Field,
    E: Field + ExtField<F>,
{
    residue_recurrence(coefficients.iter().copied().map(E::lift_base), alpha)
}

/// Evaluate every reduced shift of a sparse public multiplier at `alpha`.
///
/// This is the sparse-input counterpart of [`residue_kernel`]. It preserves
/// the same signed-wrap recurrence while storing only the supplied nonzero
/// `(position, coefficient)` pairs, avoiding a dimension-sized temporary input
/// vector. The returned kernel remains dense because every reduced shift is
/// consumed by the relation-weight compiler.
///
/// # Errors
///
/// Returns an error unless `dimension` is a nonzero power of two and every
/// supplied position is unique, in range, and paired with a nonzero weight.
pub fn sparse_residue_kernel<E>(
    dimension: usize,
    terms: impl IntoIterator<Item = (usize, E)>,
    alpha: E,
) -> Result<Vec<E>, AkitaError>
where
    E: Field,
{
    if dimension == 0 || !dimension.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "residue-kernel dimension must be a nonzero power of two".into(),
        ));
    }
    let mut terms = terms.into_iter().collect::<Vec<_>>();
    terms.sort_unstable_by_key(|(position, _)| *position);
    for (index, &(position, weight)) in terms.iter().enumerate() {
        if position >= dimension {
            return Err(AkitaError::InvalidInput(
                "sparse residue-kernel position is out of range".into(),
            ));
        }
        if weight.is_zero() {
            return Err(AkitaError::InvalidInput(
                "sparse residue-kernel weights must be nonzero".into(),
            ));
        }
        if index != 0 && terms[index - 1].0 == position {
            return Err(AkitaError::InvalidInput(
                "sparse residue-kernel positions must be unique".into(),
            ));
        }
    }

    let mut alpha_power = E::one();
    let mut current = E::zero();
    let mut term_index = 0usize;
    for position in 0..dimension {
        if terms
            .get(term_index)
            .is_some_and(|(term_position, _)| *term_position == position)
        {
            current += terms[term_index].1 * alpha_power;
            term_index += 1;
        }
        alpha_power *= alpha;
    }
    let modulus_evaluation = alpha_power + E::one();

    let mut kernel = Vec::new();
    kernel
        .try_reserve_exact(dimension)
        .map_err(|_| AkitaError::InvalidInput("residue-kernel allocation failed".into()))?;
    kernel.push(current);
    let mut reverse_term_index = terms.len();
    for wrap_position in (1..dimension).rev() {
        let wrap_weight =
            if reverse_term_index != 0 && terms[reverse_term_index - 1].0 == wrap_position {
                reverse_term_index -= 1;
                terms[reverse_term_index].1
            } else {
                E::zero()
            };
        current = alpha * current - modulus_evaluation * wrap_weight;
        kernel.push(current);
    }
    Ok(kernel)
}

/// Prepare terminal weights for one exact physical equality window.
///
/// `equality_weights[j]` must be the checked value `eq(point, start + j)` for
/// the physical native coefficient window being evaluated. The returned entry
/// `k` is the multilinear contraction of the `k`-th signed negacyclic shift:
///
/// `H[k] = sum_j equality_weights[j] * sign(k + j) * alpha^((k + j) mod d)`.
///
/// Windows with different physical starts must be prepared separately, even
/// when their native dimensions agree.
///
/// # Errors
///
/// Returns an error unless the equality-window length is a nonzero power of
/// two, or if the output allocation cannot be reserved.
pub fn terminal_residue_kernel<E>(equality_weights: &[E], alpha: E) -> Result<Vec<E>, AkitaError>
where
    E: Field,
{
    residue_recurrence(equality_weights.iter().copied(), alpha)
}

fn residue_recurrence<E, I>(weights: I, alpha: E) -> Result<Vec<E>, AkitaError>
where
    E: Field,
    I: Clone + DoubleEndedIterator<Item = E> + ExactSizeIterator,
{
    let dimension = weights.len();
    if dimension == 0 || !dimension.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "residue-kernel dimension must be a nonzero power of two".into(),
        ));
    }

    let mut alpha_power = E::one();
    let mut current = E::zero();
    for weight in weights.clone() {
        current += weight * alpha_power;
        alpha_power *= alpha;
    }
    let modulus_evaluation = alpha_power + E::one();

    let mut kernel = Vec::new();
    kernel
        .try_reserve_exact(dimension)
        .map_err(|_| AkitaError::InvalidInput("residue-kernel allocation failed".into()))?;
    kernel.push(current);
    for wrap_weight in weights.rev().take(dimension - 1) {
        current = alpha * current - modulus_evaluation * wrap_weight;
        kernel.push(current);
    }
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset_eq::OffsetEqWindow;
    use crate::ring::{eval_ring_at_pows, scalar_powers, CyclotomicRing};
    use jolt_field::{
        Ext2, ExtField, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
    };
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn quadratic_kernel<F, E>(coefficients: &[F], alpha: E) -> Vec<E>
    where
        F: Field,
        E: Field + ExtField<F>,
    {
        let dimension = coefficients.len();
        let powers = scalar_powers(alpha, dimension);
        (0..dimension)
            .map(|shift| {
                coefficients
                    .iter()
                    .enumerate()
                    .fold(E::zero(), |sum, (coefficient, &value)| {
                        let exponent = coefficient + shift;
                        let term = E::lift_base(value) * powers[exponent % dimension];
                        if exponent < dimension {
                            sum + term
                        } else {
                            sum - term
                        }
                    })
            })
            .collect()
    }

    fn check_product_identity<F, E, const D: usize>(seed: u64)
    where
        F: Field,
        E: Field + ExtField<F>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let multiplier = CyclotomicRing::<F, D>::random(&mut rng);
        let witness = CyclotomicRing::<F, D>::random(&mut rng);
        let alpha = E::random(&mut rng);
        let powers = scalar_powers(alpha, D);
        let kernel = residue_kernel(multiplier.coefficients(), alpha).expect("residue kernel");

        assert_eq!(
            kernel,
            quadratic_kernel(multiplier.coefficients(), alpha),
            "linear and quadratic kernels differ at dimension {D}"
        );
        let product_evaluation = eval_ring_at_pows(&(multiplier * witness), &powers);
        let kernel_evaluation = kernel
            .iter()
            .zip(witness.coefficients())
            .fold(E::zero(), |sum, (&weight, &coefficient)| {
                sum + weight.mul_base(coefficient)
            });
        assert_eq!(product_evaluation, kernel_evaluation);
    }

    macro_rules! check_admitted_dimensions {
        ($base:ty, $extension:ty, $seed:expr) => {{
            check_product_identity::<$base, $extension, 64>($seed);
            check_product_identity::<$base, $extension, 128>($seed + 1);
            check_product_identity::<$base, $extension, 256>($seed + 2);
            check_product_identity::<$base, $extension, 512>($seed + 3);
            check_product_identity::<$base, $extension, 1024>($seed + 4);
            check_product_identity::<$base, $extension, 2048>($seed + 5);
        }};
    }

    #[test]
    fn residue_recurrence_matches_quadratic_and_ring_product_oracles() {
        for seed in 1..=8 {
            check_product_identity::<Prime32Offset99, FpExt4<Prime32Offset99>, 1>(seed);
            check_product_identity::<Prime32Offset99, FpExt4<Prime32Offset99>, 2>(seed + 10);
            check_product_identity::<Prime32Offset99, FpExt4<Prime32Offset99>, 4>(seed + 20);
            check_product_identity::<Prime32Offset99, FpExt4<Prime32Offset99>, 8>(seed + 30);
            check_product_identity::<Prime64Offset59, Ext2<Prime64Offset59>, 16>(seed + 40);
            check_product_identity::<Prime64Offset59, Ext2<Prime64Offset59>, 32>(seed + 50);
            check_product_identity::<Prime128OffsetA7F7, Prime128OffsetA7F7, 64>(seed + 60);
            check_product_identity::<Prime128OffsetA7F7, Prime128OffsetA7F7, 128>(seed + 70);
        }

        // This is the complete commitment-dimension admission surface. Keep
        // it synchronized with `akita_types::SUPPORTED_COMMITMENT_RING_DIMS`.
        check_admitted_dimensions!(Prime32Offset99, FpExt4<Prime32Offset99>, 101);
        check_admitted_dimensions!(Prime64Offset59, Ext2<Prime64Offset59>, 201);
        check_admitted_dimensions!(Prime128OffsetA7F7, Prime128OffsetA7F7, 301);
    }

    #[test]
    fn fp64_terminal_kernel_uses_the_configured_extension_field() {
        type E = Ext2<Prime64Offset59>;
        let mut rng = StdRng::seed_from_u64(401);
        let weights = (0..64).map(|_| E::random(&mut rng)).collect::<Vec<_>>();
        let alpha = E::random(&mut rng);
        assert_eq!(
            terminal_residue_kernel(&weights, alpha).unwrap(),
            quadratic_kernel(&weights, alpha)
        );
    }

    #[test]
    fn terminal_recurrence_consumes_exact_offset_equality_windows() {
        type E = FpExt4<Prime32Offset99>;

        for seed in 1..=8 {
            let mut rng = StdRng::seed_from_u64(seed);
            let point = (0..8).map(|_| E::random(&mut rng)).collect::<Vec<_>>();
            let equality = OffsetEqWindow::new(&point).expect("equality window");
            let alpha = E::random(&mut rng);
            for (start, dimension) in [(0, 8), (3, 8), (64, 16), (127, 16)] {
                let mut weights = vec![E::zero(); dimension];
                equality
                    .fill_interval(start, &mut weights)
                    .expect("checked physical equality interval");
                assert_eq!(
                    terminal_residue_kernel(&weights, alpha).expect("terminal kernel"),
                    quadratic_kernel(&weights, alpha),
                    "terminal kernel differs at seed {seed}, start {start}, dimension {dimension}"
                );
            }

            let mut first = vec![E::zero(); 8];
            let mut second = vec![E::zero(); 8];
            equality.fill_interval(0, &mut first).unwrap();
            equality.fill_interval(3, &mut second).unwrap();
            assert_ne!(
                terminal_residue_kernel(&first, alpha).unwrap(),
                terminal_residue_kernel(&second, alpha).unwrap(),
                "different physical starts must not share one terminal kernel"
            );
        }
    }

    #[test]
    fn modulus_roots_do_not_require_division() {
        type F = Prime128OffsetA7F7;
        let alpha = jolt_field::fft::primitive_nth_root::<F>(4);
        let coefficients = [F::from_u64(37), F::from_u64(13)];
        let weights = [F::from_u64(41), F::from_u64(19)];

        assert_eq!(alpha.square() + F::one(), F::zero());
        assert_eq!(
            residue_kernel(&coefficients, alpha).unwrap(),
            quadratic_kernel(&coefficients, alpha)
        );
        assert_eq!(
            terminal_residue_kernel(&weights, alpha).unwrap(),
            quadratic_kernel(&weights, alpha)
        );
    }

    #[test]
    fn malformed_dimensions_are_rejected_before_output_allocation() {
        type F = Prime32Offset99;
        for coefficients in [&[][..], &[F::one(); 3][..]] {
            assert!(residue_kernel::<F, F>(coefficients, F::from_u64(7)).is_err());
            assert!(terminal_residue_kernel(coefficients, F::from_u64(7)).is_err());
        }
    }

    #[test]
    fn sparse_recurrence_matches_dense_recurrence() {
        type F = Prime128OffsetA7F7;
        let alpha = F::from_u64(7);
        let mut dense = vec![F::zero(); 64];
        let terms = [(0, F::from_u64(3)), (17, -F::one()), (63, F::from_u64(2))];
        for &(position, coefficient) in &terms {
            dense[position] = coefficient;
        }

        assert_eq!(
            sparse_residue_kernel(64, terms, alpha).unwrap(),
            residue_kernel::<F, F>(&dense, alpha).unwrap()
        );
    }

    #[test]
    fn sparse_recurrence_rejects_malformed_terms() {
        type F = Prime128OffsetA7F7;
        let alpha = F::from_u64(7);
        assert!(sparse_residue_kernel(3, [], alpha).is_err());
        assert!(sparse_residue_kernel(4, [(4, F::one())], alpha).is_err());
        assert!(sparse_residue_kernel(4, [(1, F::one()), (1, -F::one())], alpha).is_err());
        assert!(sparse_residue_kernel(4, [(1, F::zero())], alpha).is_err());
    }
}
