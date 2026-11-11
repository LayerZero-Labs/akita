#![cfg(feature = "labinius")]

use akita_algebra::{binary::BinaryField162, PlusTrinomial};

use akita_algebra::trinomial::signed::{signed_trinomial_product, signed_trinomial_remainder};
use akita_labinius_prover::lowered::binary_weighted_integer_convolution;

fn reference(weights: &[BinaryField162], rows: &[[i64; 162]]) -> [i128; 323] {
    let mut product = [0i128; 323];
    for (&weight, row) in weights.iter().zip(rows) {
        let words = weight.to_words();
        for i in 0..162 {
            let bit = (words[i / 64] >> (i % 64)) & 1;
            for (j, &coefficient) in row.iter().enumerate() {
                product[i + j] += i128::from(bit) * i128::from(coefficient);
            }
        }
    }
    product
}

#[test]
fn shared_binary_integer_convolution_matches_dense_signed_products() {
    for count in [0, 1, 3, 16, 64] {
        let weights = (0..count)
            .map(|row| {
                BinaryField162::from_words([
                    0x1234_5678_9abc_def0u64.rotate_left(row as u32),
                    0xfedcba98_76543210u64.rotate_right(row as u32),
                    (row as u64) & 3,
                ])
                .unwrap()
            })
            .collect::<Vec<_>>();
        let rows = (0..count)
            .map(|row| {
                std::array::from_fn(|coefficient| ((coefficient * 7 + row * 11) % 9) as i64 - 4)
            })
            .collect::<Vec<_>>();
        let actual = binary_weighted_integer_convolution(&weights, &rows).unwrap();
        assert_eq!(actual, reference(&weights, &rows));
        let compact = rows
            .iter()
            .map(|row| row.map(|value| i8::try_from(value).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            binary_weighted_integer_convolution(&weights, &compact).unwrap(),
            actual
        );
        let mut expected_remainder = [0i128; 162];
        for (&weight, row) in weights.iter().zip(&rows) {
            let words = weight.to_words();
            let lift: [i128; 162] =
                std::array::from_fn(|i| i128::from((words[i / 64] >> (i % 64)) & 1));
            let product = signed_trinomial_product::<162, PlusTrinomial>(&lift, row).unwrap();
            for (sum, value) in expected_remainder.iter_mut().zip(product) {
                *sum += value;
            }
        }
        assert_eq!(
            signed_trinomial_remainder::<162, PlusTrinomial>(&actual).unwrap(),
            expected_remainder
        );
    }
}

#[test]
fn shared_binary_integer_convolution_retains_wide_and_shape_checks() {
    let weight = BinaryField162::from_words([u64::MAX, u64::MAX, 3]).unwrap();
    let rows = [
        std::array::from_fn(|i| if i % 2 == 0 { i64::MIN } else { i64::MAX }),
        [i64::MAX; 162],
    ];
    assert_eq!(
        binary_weighted_integer_convolution(&[weight, weight], &rows).unwrap(),
        reference(&[weight, weight], &rows)
    );
    assert!(binary_weighted_integer_convolution(&[weight], &rows).is_err());
}
