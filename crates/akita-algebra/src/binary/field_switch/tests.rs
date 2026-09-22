use super::*;
use crate::binary::{BinaryField128 as H128, BinaryField192 as H192, PackedBinary162};

fn f(seed: u64) -> F {
    F([
        seed,
        seed.rotate_left(19),
        seed.rotate_right(7) & F::TOP_MASK,
    ])
}

// Independent pointwise definition, unlike the production doubling recurrence.
fn host_eq<H: SwitchField>(point: &[H], index: usize) -> H {
    point.iter().enumerate().fold(H::ONE, |acc, (axis, &r)| {
        acc * if index & (1 << axis) == 0 {
            H::ONE + r
        } else {
            r
        }
    })
}

fn binary_eq(point: &[F], index: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |acc, (axis, &r)| {
        acc * if index & (1 << axis) == 0 {
            F::ONE + r
        } else {
            r
        }
    })
}

fn profile_identities<H: SwitchField + std::fmt::Debug>(source: &[H::Source], point: &[H])
where
    H::Source: std::fmt::Debug,
{
    let mut eq = Vec::new();
    let partials = partial_evaluations::<H>(source, point, &mut eq).unwrap();
    let dense_host = source.iter().enumerate().fold(H::ZERO, |sum, (j, &value)| {
        sum + H::embed_source(value) * host_eq(point, j)
    });
    assert_eq!(partials.reconstruct(), dense_host);
    // Construct every partial by an independent dense bit matrix.
    for k in 0..H::ROWS {
        let mut expected = H::Source::default();
        for (j, &value) in source.iter().enumerate() {
            if (host_eq(point, j).coordinates()[k / 64] >> (k % 64)) & 1 == 1 {
                expected ^= value;
            }
        }
        assert_eq!(partials.values()[k], expected);
    }
    let batch: Vec<_> = (0..H::BATCH_BITS)
        .map(|i| f(0x79ab_487d_eb05_3026 ^ i as u64))
        .collect();
    let z: Vec<_> = (0..point.len())
        .map(|i| f(0x385e_6147_e07a_dac9 ^ (i * 71) as u64))
        .collect();
    let mut packed_coefficients = PackedBinary162::new();
    batched_weights(&eq, &batch, &mut packed_coefficients).unwrap();
    let coefficients = packed_coefficients.to_scalars();
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    ))]
    if source.len() >= 64 {
        let rows = row_weights::<H>(&batch).unwrap();
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("gfni")
            {
                // SAFETY: both features were detected and the full tile shape
                // was established by the source length.
                unsafe {
                    backend_partials_match::<H>(source, &eq, &partials, x86_256::partials::<H>);
                    backend_coefficients_match::<H>(
                        &eq,
                        &rows,
                        &coefficients,
                        x86_256::coefficients::<H>,
                    );
                }
            }
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vbmi")
                && std::arch::is_x86_feature_detected!("gfni")
            {
                // SAFETY: every required feature and the tile shape were checked.
                unsafe {
                    backend_partials_match::<H>(source, &eq, &partials, x86::partials::<H>);
                    backend_coefficients_match::<H>(
                        &eq,
                        &rows,
                        &coefficients,
                        x86::coefficients::<H>,
                    );
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON and the full tile shape were checked.
            unsafe {
                backend_partials_match::<H>(source, &eq, &partials, arm_partials::partials::<H>);
                backend_coefficients_match::<H>(&eq, &rows, &coefficients, arm::coefficients::<H>);
            }
        }
    }
    for (j, &coefficient) in coefficients.iter().enumerate() {
        let bits = host_eq(point, j).coordinates();
        let expected = (0..H::ROWS)
            .filter(|&k| (bits[k / 64] >> (k % 64)) & 1 == 1)
            .fold(F::ZERO, |sum, k| sum + binary_eq(&batch, k));
        assert_eq!(coefficient, expected);
    }
    let source_f: Vec<_> = source.iter().copied().map(embed_source::<H>).collect();
    let mut claim = partials.batch(&batch).unwrap();
    assert_eq!(Some(claim), F::dot_product(&source_f, &coefficients));
    let dense_weight = coefficients
        .iter()
        .enumerate()
        .fold(F::ZERO, |sum, (j, &value)| sum + value * binary_eq(&z, j));
    assert_eq!(transparent_weight(point, &z, &batch).unwrap(), dense_weight);
    // Exercise the complete new arithmetic path through #54's packed kernels.
    let mut left = PackedBinary162::from_scalars(&source_f);
    let mut right = PackedBinary162::from_scalars(&coefficients);
    for &r in &z {
        let [c0, c1, c2] = left.round_product(&right, claim).unwrap();
        claim = c0 + r * (c1 + r * c2);
        left.fold_in_place(r);
        right.fold_in_place(r);
    }
    assert_eq!(right.get(0), Some(dense_weight));
    assert_eq!(claim, left.get(0).unwrap() * dense_weight);

    // Select individual basis rows, including word boundaries and padding.
    for k in [0, 63, 64, 127, 128, 191, 192, 255] {
        if k >= (1 << H::BATCH_BITS) {
            continue;
        }
        let selector: Vec<_> = (0..H::BATCH_BITS)
            .map(|i| if (k >> i) & 1 == 0 { F::ZERO } else { F::ONE })
            .collect();
        let expected = (0..source.len())
            .filter(|&j| {
                k < H::ROWS && (host_eq(point, j).coordinates()[k / 64] >> (k % 64)) & 1 == 1
            })
            .fold(F::ZERO, |sum, j| sum + binary_eq(&z, j));
        assert_eq!(transparent_weight(point, &z, &selector).unwrap(), expected);
    }
    // A source/basis-row alteration is visible in both reconstruction and batching.
    if let Some(&nonzero) = source.iter().find(|&&x| x != H::Source::default()) {
        let mut changed = partials.values().to_vec();
        changed[0] ^= nonzero;
        let changed = SwitchPartials::<H>::try_from_values(changed).unwrap();
        assert_ne!(changed.reconstruct(), dense_host);
        assert_ne!(
            changed.batch(&batch).unwrap(),
            partials.batch(&batch).unwrap()
        );
    }
}

#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_endian = "little")
))]
unsafe fn backend_partials_match<H: SwitchField>(
    source: &[H::Source],
    weights: &[H],
    expected_partials: &SwitchPartials<H>,
    partial_kernel: unsafe fn(&[H::Source], &[H]) -> [u128; 192],
) where
    H::Source: std::fmt::Debug,
{
    // SAFETY: the caller checked all backend features and whole-tile lengths.
    let actual = unsafe { partial_kernel(source, weights) };
    for (row, &expected) in expected_partials.values()[..H::ROWS].iter().enumerate() {
        assert_eq!(actual[row], expected.into(), "partial row {row}");
    }
    assert!(actual[H::ROWS..].iter().all(|&value| value == 0));
}

#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_endian = "little")
))]
unsafe fn backend_coefficients_match<H: SwitchField>(
    weights: &[H],
    rows: &[F; 256],
    expected_coefficients: &[F],
    coefficient_kernel: unsafe fn(&[H], &[F; 256], [&mut [u64]; 3]),
) {
    let mut limbs = std::array::from_fn(|_| vec![0u64; weights.len()]);
    let [low, high, top] = &mut limbs;
    // SAFETY: the caller checked features and the output contains full limbs.
    unsafe { coefficient_kernel(weights, rows, [low, high, top]) };
    for (index, &expected) in expected_coefficients.iter().enumerate() {
        assert_eq!(
            F([limbs[0][index], limbs[1][index], limbs[2][index]]),
            expected,
            "coefficient {index}"
        );
    }
}

#[test]
fn f128_switch_matches_dense_and_packed_paths() {
    let point = [
        H128::from_words([0x1379, 0xc317_eb09_da67_f103]),
        H128::from_words([0xf104_aade_5381_cdef, 0x1234]),
        H128::from_words([0x174d, 0xe930_4567_89ab_cdef]),
    ];
    for bits in [0, 1, 3] {
        let source: Vec<_> = (0..1 << bits)
            .map(|j| (u128::MAX >> (j % 3)) ^ (j as u128 * 713))
            .collect();
        profile_identities::<H128>(&source, &point[..bits]);
    }
    profile_identities::<H128>(&[0; 8], &point);
}

#[test]
fn f64_f192_switch_matches_dense_and_packed_paths() {
    let point = [
        H192::from_words([0x1379, 0xc317_eb09_da67_f103, 1 << 63]),
        H192::from_words([0xf104_aade_5381_cdef, 0x1234, u64::MAX]),
        H192::from_words([0x174d, 0xe930_4567_89ab_cdef, 0x9158]),
    ];
    for bits in [0, 1, 3] {
        let source: Vec<_> = (0..1 << bits)
            .map(|j| (u64::MAX >> (j % 3)) ^ (j as u64 * 713))
            .collect();
        profile_identities::<H192>(&source, &point[..bits]);
    }
    profile_identities::<H192>(&[0; 8], &point);
}

#[test]
fn shapes_padding_and_order_are_explicit() {
    let mut scratch = vec![H192::ONE; 8];
    let point = [H192::from_words([3, 5, 7]); 3];
    assert!(partial_evaluations::<H192>(&[], &[], &mut scratch).is_err());
    assert!(partial_evaluations::<H192>(&[0; 7], &point, &mut scratch).is_err());
    assert_eq!(scratch, vec![H192::ONE; 8]);
    assert!(partial_evaluations::<H192>(
        &[0],
        &vec![H192::ONE; usize::BITS as usize],
        &mut scratch
    )
    .is_err());
    assert!(SwitchPartials::<H192>::try_from_values(vec![0; 192]).is_err());
    let mut bad_padding = vec![0; 256];
    bad_padding[192] = 1;
    assert!(SwitchPartials::<H192>::try_from_values(bad_padding).is_err());
    assert!(SwitchPartials::<H128>::try_from_values(vec![0; 129]).is_err());
    let partials = SwitchPartials::<H128>::try_from_values(vec![0; 128]).unwrap();
    assert!(partials.batch(&[F::ZERO; 8]).is_err());
    assert!(transparent_weight(&point, &[F::ZERO; 2], &[F::ZERO; 8]).is_err());
    assert!(transparent_weight(&point, &[F::ZERO; 3], &[F::ZERO; 7]).is_err());
    let mut out = PackedBinary162::from_scalars(&[F::ONE]);
    assert!(batched_weights::<H128>(&[], &[F::ZERO; 7], &mut out).is_err());
    assert!(batched_weights::<H128>(&[H128::ONE], &[F::ZERO; 8], &mut out).is_err());
    assert_eq!(out.to_scalars(), [F::ONE]);

    // Boolean points give exact index fixtures and expose reversed variables.
    let source = [10, 20, 30, 40, 50, 60, 70, 80];
    let p = [H192::ONE, H192::ZERO, H192::ZERO];
    let allocations = scratch.as_ptr();
    let partials = partial_evaluations::<H192>(&source, &p, &mut scratch).unwrap();
    assert_eq!(partials.reconstruct(), H192::embed_source(20));
    assert_eq!(scratch[1], H192::ONE);
    let swapped = partial_evaluations::<H192>(&source, &[p[2], p[1], p[0]], &mut scratch).unwrap();
    assert_eq!(swapped.reconstruct(), H192::embed_source(50));
    partial_evaluations::<H192>(&source[..2], &p[..1], &mut scratch).unwrap();
    partial_evaluations::<H192>(&source, &p, &mut scratch).unwrap();
    assert_eq!(scratch.as_ptr(), allocations);
    let embedded = embed_source::<H128>(1u128 << 127);
    assert_eq!(embedded.to_words(), [0, 1 << 63, 0]);
    assert_eq!(embed_source::<H192>(1 << 63).to_words(), [1 << 63, 0, 0]);
}

#[test]
fn full_tiles_match_dense_for_both_profiles() {
    let source128: Vec<_> = (0..128u128)
        .map(|j| (u128::MAX >> (j % 7)) ^ j.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    let source64: Vec<_> = source128.iter().map(|&j| j as u64).collect();
    let point128: Vec<_> = (0..7)
        .map(|j| H128::from_words([0x7192_834a_bcde_f678 ^ j, 0xd391_0385_7abc_def0 ^ (j * 519)]))
        .collect();
    let point192: Vec<_> = (0..7)
        .map(|j| {
            H192::from_words([
                0x7192_834a_bcde_f678 ^ j,
                0xd391_0385_7abc_def0 ^ (j * 519),
                0xac56_0123_cbe9_6384 ^ j,
            ])
        })
        .collect();
    profile_identities::<H128>(&source128[..64], &point128[..6]);
    profile_identities::<H192>(&source64[..64], &point192[..6]);
    profile_identities::<H128>(&source128, &point128);
    profile_identities::<H192>(&source64, &point192);
}
