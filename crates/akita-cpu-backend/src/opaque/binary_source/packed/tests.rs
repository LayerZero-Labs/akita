use super::*;
use akita_algebra::binary::BinaryField162;
use akita_algebra::ring::trinomial::signed::signed_trinomial_product;
use akita_algebra::MinusTrinomial;
use akita_challenges::{FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_types::binary_source::{
    BinaryCommitmentProfiles, BinaryIndexMap, BinaryTerminalClaim, PrimeGroupClaim,
    PrimePlaneClaim, PrimePlaneManifest,
};
use jolt_field::Prime128Offset275;

struct SeedDraw;
impl FoldDraw for SeedDraw {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        decode(&vec![17; len])
    }
}

fn expand(descriptor: &BinarySourceDescriptor, words: &[Vec<u128>]) -> Vec<[i128; 648]> {
    let index = descriptor.index_map();
    let rows = index.padded_carriers() / (4 * index.columns());
    let mut expanded = vec![[0; 648]; words.len() * index.columns() * rows];
    for (owner, words) in words.iter().enumerate() {
        for (word, value) in words.iter().enumerate() {
            for bit in 0..index.word_width() {
                let position = index.logical_to_ring(word, bit).unwrap();
                expanded[(owner * index.columns() + position.column) * rows + position.ring_row]
                    [position.coefficient] =
                    i128::from(position.sign) * ((value >> bit) & 1) as i128;
            }
        }
    }
    expanded
}

#[test]
fn packed_kernels_match_signed_reference_with_padding_and_owners() {
    type F = Prime128Offset275;
    for width in [1, 8, 32, 128] {
        let logical_len = 16 * (128 / width) - 3;
        let descriptor = BinarySourceDescriptor::new(
            BinaryIndexMap::new(width, logical_len, 16, 2).unwrap(),
            vec![[1; 32], [2; 32]],
            BinaryCommitmentProfiles {
                a: [3; 32],
                b: [4; 32],
            },
        )
        .unwrap();
        let mask = u128::MAX >> (128 - width);
        let words = (0..2)
            .map(|owner| {
                (0..logical_len)
                    .map(|i| {
                        ((i as u128).wrapping_mul(0x9137_afe9_1234_5678)
                            ^ ((owner + 1) as u128).rotate_left((i % 128) as u32))
                            & mask
                    })
                    .collect()
            })
            .collect::<Vec<Vec<u128>>>();
        let packed = pack_logical_words(&descriptor, &words).unwrap();
        let expanded = expand(&descriptor, &words);
        for owner in 0..2 {
            for carrier in 0..16 {
                let mut expected = 0u128;
                for slot in 0..128 / width {
                    if let Some(&word) = words[owner].get(carrier * (128 / width) + slot) {
                        expected |= word << (slot * width);
                    }
                }
                assert_eq!(
                    packed[owner * 16
                        + descriptor
                            .index_map()
                            .kernel_carrier_index(carrier)
                            .unwrap()],
                    expected
                );
            }
        }
        let matrix = (0..3)
            .map(|rank| {
                (0..2)
                    .map(|row| {
                        std::array::from_fn(|i| {
                            ((i * 997 + row * 123 + rank * 11) % 10001) as i128 - 5000
                        })
                    })
                    .collect()
            })
            .collect::<Vec<Vec<[i128; 648]>>>();
        let images = image_hint(&descriptor, 3, &matrix, &packed).unwrap();
        let mut reference_images = Vec::new();
        for column in expanded.chunks_exact(2) {
            for row in &matrix {
                let mut image = [0i128; 648];
                for (source, multiplier) in column.iter().zip(row) {
                    for (output, value) in image.iter_mut().zip(
                        signed_trinomial_product::<648, MinusTrinomial>(source, multiplier)
                            .unwrap(),
                    ) {
                        *output += value;
                    }
                }
                reference_images.push(image.map(|x| {
                    Q24Image::from_residue(x.rem_euclid(i128::from(BINARY_SOURCE_Q)) as u32)
                        .unwrap()
                }));
            }
        }
        assert_eq!(images, reference_images);
        let challenges = SeedDraw
            .draw_fold_challenges(
                &FoldChallengeFamily::new(
                    FoldChallengeDrawDomain::EvaluationTrace,
                    162,
                    FoldChallengeDistribution::BinaryRoot,
                    None,
                )
                .unwrap(),
                2,
                2,
            )
            .unwrap()
            .unwrap();
        let reference = crate::binary_root::BinaryRootRelation::build(
            &descriptor,
            &matrix,
            &expanded,
            &images,
            &challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
        .unwrap();
        let actual = probe_packed(
            &descriptor,
            &matrix,
            &packed,
            &images,
            &challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
        .unwrap()
        .unwrap();
        assert_eq!(actual.response, reference.response);
        assert_eq!(actual.embedded_challenges, reference.embedded_challenges);
        for (actual, reference) in actual.rows.iter().zip(&reference.rows) {
            assert_eq!(actual.remainder, reference.remainder);
            assert_eq!(actual.quotient, reference.quotient);
        }
        let remainders = crate::binary_root::BinaryRootRelation::remainders(
            &descriptor,
            &matrix,
            &actual.response,
            &images,
            &challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
        .unwrap();
        assert!(remainders
            .iter()
            .eq(reference.rows.iter().map(|row| &row.remainder)));
        let point = (0..4)
            .map(|i| {
                BinaryField162::from_words([13 + i, 0, 0])
                    .unwrap()
                    .to_bytes()
            })
            .collect::<Vec<_>>();
        let terminal = BinaryTerminalClaim::new(
            &descriptor,
            [2; 32],
            point[..3].to_vec(),
            point[3..].to_vec(),
            [0; 21],
        )
        .unwrap();
        assert_eq!(
            crate::binary_root::evaluation::packed_row_partials(&descriptor, &terminal, &packed)
                .unwrap(),
            crate::binary_root::binary_row_partials(&descriptor, &terminal, &expanded).unwrap()
        );
        let primes = PrimePlaneManifest::with_field_profile(
            &descriptor,
            Some(PrimeGroupClaim {
                point: (0..12).map(|i| ((i + 3) as u128).into()).collect(),
                evaluations: vec![PrimePlaneClaim {
                    polynomial: 1,
                    value: 0.into(),
                }],
            }),
            BinaryFieldProfile::P128,
        )
        .unwrap();
        assert_eq!(
            crate::binary_root::prime::packed_prime_plane_digits::<F>(
                &descriptor,
                &primes,
                &packed
            )
            .unwrap(),
            crate::binary_root::prime_plane_digits::<F>(&descriptor, &primes, &expanded).unwrap()
        );
    }
}

#[test]
fn compact_import_rejects_wrong_shapes_and_oversized_words() {
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(8, 3, 8, 2).unwrap(),
        vec![[1; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap();
    assert!(pack_logical_words(&descriptor, &[]).is_err());
    assert!(pack_logical_words(&descriptor, &[vec![1, 2]]).is_err());
    assert!(pack_logical_words(&descriptor, &[vec![1, 2, 256]]).is_err());
}
