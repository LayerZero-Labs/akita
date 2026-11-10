use super::super::*;
use jolt_field::{Fp32, Prime128Offset275, Prime64Offset59};

#[test]
fn permutation_matches_keccak_every_lane() {
    use rand::{rngs::StdRng, SeedableRng};
    if !super::available() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(44);
    for case in 0..67 {
        let mut state = std::array::from_fn(|_| {
            std::array::from_fn(|lane| match case {
                0 => 0,
                1 => u64::MAX,
                2 => [0, u64::MAX, 0x0123456789abcdef, 0xfedcba9876543210][lane],
                _ => rng.next_u64(),
            })
        });
        let mut expected: [[u64; 25]; 4] = std::array::from_fn(|lane| state.map(|word| word[lane]));
        for lane in &mut expected {
            keccak::Keccak::new().with_f1600(|permute| permute(lane));
        }
        // SAFETY: available() above checked AVX2 and OS support.
        unsafe { super::permute(&mut state) };
        for lane in 0..4 {
            assert_eq!(
                state.map(|word| word[lane]),
                expected[lane],
                "case {case}, lane {lane}"
            );
        }
    }
}

fn compare_streams<F: Field + CanonicalEncoding>() {
    for seed in [
        sample_akita_setup_seed(),
        [0; 32].into(),
        [31; 32].into(),
        [255; 32].into(),
    ] {
        for first_page in [0, 7, 8724] {
            for blocks in [0, 1, 3] {
                // SAFETY: the test entry point checked AVX2 and OS support.
                let readers = unsafe { super::page_streams::<F>(&seed, first_page, blocks) };
                for (lane, reader) in readers.into_iter().enumerate() {
                    let mut actual = SetupSeedPageXof {
                        reader,
                        buffer: [0; SETUP_XOF_RATE],
                        position: SETUP_XOF_RATE,
                    };
                    let mut expected = SetupSeedPageXof::new::<F>(&seed, first_page + lane);
                    // Cross precomputed blocks and then the independent scalar tail.
                    for length in [0, 1, 7, 16, 135, 136, 137, 4096] {
                        let mut a = vec![0; length];
                        let mut b = vec![0; length];
                        actual.fill_bytes(&mut a);
                        expected.fill_bytes(&mut b);
                        assert_eq!(a, b);
                        assert_eq!(actual.next_u32(), expected.next_u32());
                        assert_eq!(actual.next_u64(), expected.next_u64());
                        actual.try_fill_bytes(&mut a).unwrap();
                        expected.try_fill_bytes(&mut b).unwrap();
                        assert_eq!(a, b);
                    }
                }
            }
        }
    }
}

#[test]
fn four_streams_and_scalar_tails_match_p64_p128() {
    if super::available() {
        compare_streams::<Prime64Offset59>();
        compare_streams::<Prime128Offset275>();
        compare_streams::<Fp32<251>>();
    }
}

fn reject_at_tail<F: Field + CanonicalEncoding>(modulus: u128, width: usize) {
    let seed = sample_akita_setup_seed();
    // SAFETY: the test entry point checked AVX2 and OS support.
    let readers = unsafe { super::page_streams::<F>(&seed, 0, 1) };
    for (lane, mut reader) in readers.into_iter().enumerate() {
        // Force rejection at the first scalar-tail boundary. Consume different
        // tail lengths per lane; accepted values must keep their positions.
        let bytes = &mut reader.blocks.as_mut_slice()[0];
        let start = SETUP_XOF_RATE - 2 * width;
        bytes[start..start + width].copy_from_slice(&u128::MAX.to_le_bytes()[..width]);
        bytes[start + width..].copy_from_slice(&modulus.to_le_bytes()[..width]);
        let mut actual = SetupSeedPageXof {
            reader,
            buffer: [0; SETUP_XOF_RATE],
            position: SETUP_XOF_RATE,
        };
        let mut expected = SetupSeedPageXof::new::<F>(&seed, lane);
        actual.fill_bytes(&mut vec![0; start]);
        expected.fill_bytes(&mut [0; SETUP_XOF_RATE]);
        for _ in 0..20 + lane {
            assert_eq!(F::random(&mut actual), F::random(&mut expected));
        }
    }
}

#[test]
fn rejections_continue_each_lane_at_its_scalar_tail() {
    if super::available() {
        reject_at_tail::<Prime64Offset59>((1u128 << 64) - 59, 8);
        reject_at_tail::<Prime128Offset275>(u128::MAX - 274, 16);
    }
}
