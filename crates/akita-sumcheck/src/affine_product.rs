//! Coefficients of batched products of affine factors.

use akita_field::FieldCore;

/// Maximum supported product degree.
pub const MAX_AFFINE_PRODUCT_DEGREE: usize = 4;

/// Compute a weighted batch of products of affine factors.
///
/// Each parent owns `arity` consecutive lanes. Lane `i` represents
/// `left[i] + X * (right[i] - left[i])`. Supported arities are two and four.
pub fn batched_affine_product_coefficients<E: FieldCore>(
    left: &[E],
    right: &[E],
    arity: usize,
    parent_weights: &[E],
) -> [E; MAX_AFFINE_PRODUCT_DEGREE + 1] {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), arity * parent_weights.len());
    debug_assert!(matches!(arity, 2 | 4));

    let mut batched = [E::zero(); MAX_AFFINE_PRODUCT_DEGREE + 1];
    for (parent_index, &weight) in parent_weights.iter().enumerate() {
        let first_lane = parent_index * arity;
        let polynomial = match arity {
            2 => quadratic_affine_product(
                [left[first_lane], left[first_lane + 1]],
                [right[first_lane], right[first_lane + 1]],
            ),
            4 => quartic_affine_product(
                [
                    left[first_lane],
                    left[first_lane + 1],
                    left[first_lane + 2],
                    left[first_lane + 3],
                ],
                [
                    right[first_lane],
                    right[first_lane + 1],
                    right[first_lane + 2],
                    right[first_lane + 3],
                ],
            ),
            _ => unreachable!("validated affine-product arity"),
        };
        if parent_weights.len() == 1 && weight == E::one() {
            batched = polynomial;
        } else {
            for degree in 0..=arity {
                batched[degree] += weight * polynomial[degree];
            }
        }
    }
    batched
}

#[inline(always)]
fn quadratic_affine_product<E: FieldCore>(
    left: [E; 2],
    right: [E; 2],
) -> [E; MAX_AFFINE_PRODUCT_DEGREE + 1] {
    let first_slope = right[0] - left[0];
    let second_slope = right[1] - left[1];
    [
        left[0] * left[1],
        left[0] * second_slope + first_slope * left[1],
        first_slope * second_slope,
        E::zero(),
        E::zero(),
    ]
}

#[inline(always)]
fn quartic_affine_product<E: FieldCore>(
    left: [E; 4],
    right: [E; 4],
) -> [E; MAX_AFFINE_PRODUCT_DEGREE + 1] {
    let first = quadratic_affine_product([left[0], left[1]], [right[0], right[1]]);
    let second = quadratic_affine_product([left[2], left[3]], [right[2], right[3]]);
    [
        first[0] * second[0],
        first[0] * second[1] + first[1] * second[0],
        first[0] * second[2] + first[1] * second[1] + first[2] * second[0],
        first[1] * second[2] + first[2] * second[1],
        first[2] * second[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{FromPrimitiveInt, Prime128Offset275};

    type F = Prime128Offset275;

    #[test]
    fn optimized_coefficients_match_direct_multiplication() {
        for arity in [2, 4] {
            for parent_count in [1, 2] {
                let lane_count = arity * parent_count;
                let left = (0..lane_count)
                    .map(|lane| F::from_u64(lane as u64 + 3))
                    .collect::<Vec<_>>();
                let right = (0..lane_count)
                    .map(|lane| F::from_u64(2 * lane as u64 + 11))
                    .collect::<Vec<_>>();
                let weights = (0..parent_count)
                    .map(|parent| F::from_u64(parent as u64 + 17))
                    .collect::<Vec<_>>();
                let coefficients =
                    batched_affine_product_coefficients(&left, &right, arity, &weights);

                for point in 0..8 {
                    let point = F::from_u64(point);
                    let expected = (0..parent_count).fold(F::zero(), |sum, parent| {
                        let first_lane = parent * arity;
                        let product = (0..arity).fold(F::one(), |product, child| {
                            let lane = first_lane + child;
                            product * (left[lane] + point * (right[lane] - left[lane]))
                        });
                        sum + weights[parent] * product
                    });
                    let actual = coefficients
                        .iter()
                        .rev()
                        .fold(F::zero(), |value, coefficient| value * point + *coefficient);
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
