//! Coefficients of a small polynomial evaluated on an affine line.

use akita_field::FieldCore;

/// Maximum supported polynomial degree.
pub const MAX_AFFINE_POLYNOMIAL_DEGREE: usize = 4;

/// Compose a degree-at-most-four polynomial with `offset + X * slope`.
///
/// The input and output use ascending coefficient order. Missing input
/// coefficients are zero.
pub fn compose_polynomial_with_affine<E: FieldCore>(
    coefficients: &[E],
    offset: E,
    slope: E,
) -> [E; MAX_AFFINE_POLYNOMIAL_DEGREE + 1] {
    debug_assert!(coefficients.len() <= MAX_AFFINE_POLYNOMIAL_DEGREE + 1);
    let [constant, linear, quadratic, cubic, quartic] = match coefficients {
        [] => return [E::zero(); 5],
        [c0] => return [*c0, E::zero(), E::zero(), E::zero(), E::zero()],
        [c0, c1] => {
            return [
                *c0 + *c1 * offset,
                *c1 * slope,
                E::zero(),
                E::zero(),
                E::zero(),
            ];
        }
        [c0, c1, c2] => [*c0, *c1, *c2, E::zero(), E::zero()],
        [c0, c1, c2, c3] => [*c0, *c1, *c2, *c3, E::zero()],
        [c0, c1, c2, c3, c4] => [*c0, *c1, *c2, *c3, *c4],
        _ => unreachable!("polynomial degree was bounded by four"),
    };

    let two_quadratic = quadratic + quadratic;
    let three_cubic = cubic + cubic + cubic;
    let four_quartic = (quartic + quartic) + (quartic + quartic);
    let six_quartic = four_quartic + quartic + quartic;
    let value =
        constant + offset * (linear + offset * (quadratic + offset * (cubic + offset * quartic)));
    let first_derivative =
        linear + offset * (two_quadratic + offset * (three_cubic + offset * four_quartic));
    let second_divided_derivative = quadratic + offset * (three_cubic + offset * six_quartic);
    let third_divided_derivative = cubic + offset * four_quartic;
    let slope_squared = slope * slope;

    [
        value,
        slope * first_derivative,
        slope_squared * second_divided_derivative,
        slope_squared * slope * third_divided_derivative,
        slope_squared * slope_squared * quartic,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{FromPrimitiveInt, Prime128Offset275};

    type F = Prime128Offset275;

    #[test]
    fn composition_matches_direct_evaluation() {
        let coefficients = (1..=5).map(F::from_u64).collect::<Vec<_>>();
        let offset = F::from_u64(7);
        let slope = F::from_u64(11);
        let composed = compose_polynomial_with_affine(&coefficients, offset, slope);
        for point in 0..8 {
            let point = F::from_u64(point);
            let input = offset + point * slope;
            let expected = coefficients
                .iter()
                .rev()
                .fold(F::zero(), |value, coefficient| value * input + *coefficient);
            let actual = composed
                .iter()
                .rev()
                .fold(F::zero(), |value, coefficient| value * point + *coefficient);
            assert_eq!(actual, expected);
        }
    }
}
