use super::{
    aligned_i8_tile_width, fused_split_eq_quotients, mat_vec_mul_crt_ntt, mat_vec_mul_crt_ntt_many,
    mat_vec_mul_digits_i8_strided_with_params, mat_vec_mul_digits_i8_with_params,
    mat_vec_mul_i8_dense_with_params, mat_vec_mul_i8_strided_with_params,
    mat_vec_mul_i8_with_params, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8_dense_single_row,
    mat_vec_mul_ntt_raw_i8_strided, mat_vec_mul_ntt_single_i8_cyclic, mat_vec_mul_unchecked,
    precompute_dense_mat_ntt_with_params,
};
use crate::kernels::crt_ntt::{build_ntt_slot, select_crt_ntt_params, ProtocolCrtNttParams};
use akita_algebra::ntt::{
    tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES},
    PrimeWidth,
};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing};
use akita_field::{CanonicalField, FieldCore, Fp64, Prime128Offset275};
use akita_types::layout::FlatMatrix;

mod reduced_profiles;

fn centered_i32_ring<F: akita_field::CanonicalField, const D: usize>(
    coeffs: &[i32; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(coeffs[idx] as i64)))
}

fn cyclic_product<F: akita_field::FieldCore, const D: usize>(
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let mut out = CyclotomicRing::<F, D>::zero();
    for (i, &a) in lhs.coefficients().iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs.coefficients().iter().enumerate() {
            if !b.is_zero() {
                out.coefficients_mut()[(i + j) % D] += a * b;
            }
        }
    }
    out
}

fn mat_vec_mul_i8_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_with_params(ntt_mat, blocks, num_digits, log_basis, params)
}

fn mat_vec_mul_i8_dense_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_dense_with_params(ntt_mat, blocks, num_digits, log_basis, params)
}

fn mat_vec_mul_i8_strided_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_strided_with_params(
        ntt_mat, coeffs, num_blocks, block_len, num_digits, log_basis, params,
    )
}

fn mat_vec_mul_digits_i8_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params(ntt_mat, blocks, log_basis, params)
}

fn mat_vec_mul_digits_i8_strided_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_strided_with_params(
        ntt_mat, coeffs, num_blocks, block_len, log_basis, params,
    )
}

fn quotient_from_cyclic_and_negacyclic<
    F: akita_field::FieldCore + akita_field::HalvingField,
    const D: usize,
>(
    cyclic: &CyclotomicRing<F, D>,
    negacyclic: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc = cyclic.coefficients();
    let neg = negacyclic.coefficients();
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| (cyc[idx] - neg[idx]).half()))
}

fn schoolbook_digit_mat_vec<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    blocks: &[Vec<[i8; D]>],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    blocks
        .iter()
        .map(|block| {
            mat.iter()
                .map(|row| {
                    row.iter().zip(block.iter()).fold(
                        CyclotomicRing::<F, D>::zero(),
                        |mut acc, (lhs, digit)| {
                            let rhs = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                                F::from_i64(i64::from(digit[k]))
                            }));
                            acc += *lhs * rhs;
                            acc
                        },
                    )
                })
                .collect()
        })
        .collect()
}

#[test]
fn aligned_i8_tile_width_keeps_full_tiles_on_digit_boundaries() {
    assert_eq!(aligned_i8_tile_width(130, 512, 64), 128);
    assert_eq!(aligned_i8_tile_width(63, 512, 64), 64);
    assert_eq!(aligned_i8_tile_width(1024, 65, 64), 64);
    assert_eq!(aligned_i8_tile_width(1024, 48, 64), 48);
}

#[test]
fn predecomposed_digit_api_rejects_digits_outside_log_basis_range() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let row = CyclotomicRing::<F, D>::one();
    let flat = FlatMatrix::from_ring_slice(&[row]);
    let slot = build_ntt_slot(flat.ring_view::<D>(1, 1).expect("valid matrix"))
        .expect("Q32 dispatch should support this field and ring dimension");
    let bad_digits = vec![[4i8; D]];
    let blocks: Vec<&[[i8; D]]> = vec![bad_digits.as_slice()];

    assert!(matches!(
        mat_vec_mul_ntt_digits_i8::<F, D>(&slot, 1, 1, &blocks, 3),
        Err(akita_field::AkitaError::InvalidInput(_))
    ));
}

#[test]
fn raw_i8_strided_accepts_signed_unit_outside_binary_digit_range() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let num_rows = 2;
    let block_len = 3;
    let num_blocks = 2;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..num_rows)
        .map(|row| {
            (0..block_len)
                .map(|col| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                        F::from_u64((row + col + k + 1) as u64)
                    }))
                })
                .collect()
        })
        .collect();
    let flat_rows: Vec<_> = mat.iter().flatten().copied().collect();
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(num_rows, block_len)
            .expect("valid matrix view"),
    )
    .expect("Q32 dispatch should support this field and ring dimension");
    let coeffs: Vec<[i8; D]> = (0..block_len)
        .flat_map(|col| {
            (0..num_blocks).map(move |block| {
                if (col + block) % 2 == 0 {
                    [1i8; D]
                } else {
                    [-1i8; D]
                }
            })
        })
        .collect();
    let blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block| {
            (0..block_len)
                .map(|col| coeffs[col * num_blocks + block])
                .collect()
        })
        .collect();

    let got = mat_vec_mul_ntt_raw_i8_strided::<F, D>(
        &slot, num_rows, block_len, &coeffs, num_blocks, block_len,
    )
    .expect("raw signed-i8 strided mat-vec");
    let expected = schoolbook_digit_mat_vec(&mat, &blocks);

    assert_eq!(got, expected);
}

#[test]
fn fused_split_eq_quotients_uses_all_cyclic_role_rows() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let rows = 3;
    let cols = 5;
    let flat_rows: Vec<CyclotomicRing<F, D>> = (0..rows * cols)
        .map(|idx| {
            let coeffs = std::array::from_fn(|k| {
                let raw = (idx as i64 * 17 + k as i64 * 5) % 31;
                F::from_i64(raw - 15)
            });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(rows, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q32 dispatch should support this field and ring dimension");

    let w_hat: Vec<[i8; D]> = (0..cols)
        .map(|j| std::array::from_fn(|k| ((j + 2 * k) % 7) as i8 - 3))
        .collect();
    let t_hat: Vec<[i8; D]> = (0..cols)
        .map(|j| std::array::from_fn(|k| ((3 * j + k) % 5) as i8 - 2))
        .collect();
    let z_pre: Vec<[i32; D]> = (0..cols)
        .map(|j| std::array::from_fn(|k| ((j + k) % 3) as i32 - 1))
        .collect();

    let log_basis = 3;
    let expected_d = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, rows, cols, &w_hat, log_basis)
        .expect("expected D rows");
    let expected_b = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, rows, cols, &t_hat, log_basis)
        .expect("expected B rows");
    let (d_rows, b_rows, _a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, rows, rows, 1, &w_hat, &t_hat, &z_pre, 1)
            .expect("fused split-eq rows");

    assert_eq!(d_rows, expected_d);
    assert_eq!(b_rows, expected_b);
}

#[test]
fn fused_split_eq_q128_quotient_chunks_before_crt_wrap() {
    type F = Prime128Offset275;
    const D: usize = 32;
    let cols = 4;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let (_d_rows, _b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 0, 0, 1, &[], &[], &z_pre, 32_768)
            .expect("fused split-eq rows");

    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, j| {
        let z = centered_i32_ring(&z_pre[j]);
        let cyclic = cyclic_product(&row, &z);
        let negacyclic = row * z;
        acc += quotient_from_cyclic_and_negacyclic(&cyclic, &negacyclic);
        acc
    });

    assert_eq!(a_rows, vec![expected]);
}

#[test]
fn fused_split_eq_q128_quotient_falls_back_when_one_term_exceeds_crt() {
    type F = Prime128Offset275;
    const D: usize = 128;
    let cols = 1;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat = FlatMatrix::from_ring_slice(&[row]);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let (_d_rows, _b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 0, 0, 1, &[], &[], &z_pre, 32_768)
            .expect("fused split-eq rows");

    let z = centered_i32_ring(&z_pre[0]);
    let expected = quotient_from_cyclic_and_negacyclic(&cyclic_product(&row, &z), &(row * z));

    assert_eq!(a_rows, vec![expected]);
}

#[test]
fn fused_split_eq_uses_actual_centered_bound_when_hint_is_underreported() {
    type F = Prime128Offset275;
    const D: usize = 32;
    let cols = 4;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let (_d_rows, _b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 0, 0, 1, &[], &[], &z_pre, 1)
            .expect("fused split-eq rows");

    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, j| {
        let z = centered_i32_ring(&z_pre[j]);
        let cyclic = cyclic_product(&row, &z);
        let negacyclic = row * z;
        acc += quotient_from_cyclic_and_negacyclic(&cyclic, &negacyclic);
        acc
    });

    assert_eq!(a_rows, vec![expected]);
}

#[test]
fn fused_split_eq_q128_cyclic_i8_chunks_before_crt_wrap() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let w_hat = vec![[-32i8; D]; cols];

    let (d_rows, _b_rows, _a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, 1, 0, 0, &w_hat, &[], &[], 0)
            .expect("fused split-eq rows");

    let digit = CyclotomicRing::from_coefficients([F::from_i64(-32); D]);
    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += cyclic_product(&row, &digit);
        acc
    });

    assert_eq!(d_rows, vec![expected]);
}

#[test]
fn mat_vec_mul_ntt_i8_dense_single_row_chunks_q128() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let log_basis = 6;
    let num_digits = 1;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let digit_ring = CyclotomicRing::from_coefficients([F::from_i64(-32); D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let block = vec![digit_ring; cols];
    let block_slices: Vec<&[CyclotomicRing<F, D>]> = vec![block.as_slice()];

    let got =
        mat_vec_mul_ntt_i8_dense_single_row(&slot, cols, &block_slices, num_digits, log_basis)
            .expect("single-row dense mat-vec");

    let product = row * digit_ring;
    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += product;
        acc
    });

    assert_eq!(got, vec![expected]);
}

#[test]
fn q128_many_blocks_digits_chunk_instead_of_unsafe_block_parallel() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let num_blocks = 16;
    let log_basis = 6;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let half = F::from_canonical_u128_reduced(modulus / 2);
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|_| vec![CyclotomicRing::from_coefficients([half; D]); cols])
        .collect();
    let digit_blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block_idx| {
            (0..cols)
                .map(|col| {
                    let digit = if (block_idx + col) % 2 == 0 { -32 } else { 31 };
                    [digit; D]
                })
                .collect()
        })
        .collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q128(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let got = mat_vec_mul_digits_i8_with_params_for_log_basis::<F, i32, Q128_NUM_PRIMES, D>(
                &ntt_mat,
                &digit_block_slices,
                log_basis,
                &params,
            );
            let expected = schoolbook_digit_mat_vec(&mat, &digit_blocks);

            assert_eq!(got, expected);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn fused_split_eq_quotients_uses_role_local_packed_widths() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let n_d = 2;
    let n_b = 3;
    let n_a = 2;
    let d_width = 2;
    let b_width = 4;
    let a_width = 3;
    let total_len = (n_d * d_width).max(n_b * b_width).max(n_a * a_width);
    let flat_rows: Vec<CyclotomicRing<F, D>> = (0..total_len)
        .map(|idx| {
            let coeffs = std::array::from_fn(|k| {
                let raw = (idx as i64 * 19 + k as i64 * 7) % 37;
                F::from_i64(raw - 18)
            });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = build_ntt_slot(
        flat.ring_view::<D>(1, total_len)
            .expect("valid packed setup prefix"),
    )
    .expect("Q32 dispatch should support this field and ring dimension");

    let w_hat: Vec<[i8; D]> = (0..d_width)
        .map(|j| std::array::from_fn(|k| ((j + 2 * k) % 5) as i8 - 2))
        .collect();
    let t_hat: Vec<[i8; D]> = (0..b_width)
        .map(|j| std::array::from_fn(|k| ((2 * j + k) % 7) as i8 - 3))
        .collect();
    let z_pre: Vec<[i32; D]> = (0..a_width)
        .map(|j| std::array::from_fn(|k| ((3 * j + k) % 7) as i32 - 3))
        .collect();
    let z_rings: Vec<CyclotomicRing<F, D>> = z_pre
        .iter()
        .map(|row| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| F::from_i64(row[k] as i64)))
        })
        .collect();

    let log_basis = 3;
    let expected_d =
        mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, n_d, d_width, &w_hat, log_basis)
            .expect("expected D rows");
    let expected_b =
        mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, n_b, b_width, &t_hat, log_basis)
            .expect("expected B rows");
    let expected_a = (0..n_a)
        .map(|row_idx| {
            (0..a_width).fold(CyclotomicRing::<F, D>::zero(), |mut acc, col_idx| {
                let lhs = flat_rows[row_idx * a_width + col_idx];
                let z = z_rings[col_idx];
                let cyclic = cyclic_product(&lhs, &z);
                let negacyclic = lhs * z;
                acc += quotient_from_cyclic_and_negacyclic(&cyclic, &negacyclic);
                acc
            })
        })
        .collect::<Vec<_>>();
    let (d_rows, b_rows, a_rows) =
        fused_split_eq_quotients::<F, D>(&slot, n_d, n_b, n_a, &w_hat, &t_hat, &z_pre, 3)
            .expect("fused split-eq rows");

    assert_eq!(d_rows, expected_d);
    assert_eq!(b_rows, expected_b);
    assert_eq!(a_rows, expected_a);
}

#[test]
fn dense_mat_vec_matches_schoolbook_q32_d64() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..4)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        F::from_u64((i as u64 * 10_000 + j as u64 * 100 + k as u64 + 1) % 97)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();
    let vec: Vec<CyclotomicRing<F, D>> = (0..4)
        .map(|j| {
            let coeffs = std::array::from_fn(|k| F::from_u64((j as u64 * 50 + k as u64 + 3) % 89));
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();

    let schoolbook = mat_vec_mul_unchecked(&mat, &vec);
    let crt_ntt = mat_vec_mul_crt_ntt(&mat, &vec).expect("Q32 dispatch should succeed");
    assert_eq!(schoolbook, crt_ntt);
}

#[test]
fn dense_mat_vec_matches_schoolbook_q64_dispatch_for_large_d() {
    type F = Fp64<4294967197>;
    const D: usize = 128;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
        .map(|i| {
            (0..2)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        F::from_u64((i as u64 * 20_000 + j as u64 * 300 + k as u64 + 7) % 113)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();
    let vec: Vec<CyclotomicRing<F, D>> = (0..2)
        .map(|j| {
            let coeffs =
                std::array::from_fn(|k| F::from_u64((j as u64 * 70 + k as u64 + 11) % 101));
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();

    let schoolbook = mat_vec_mul_unchecked(&mat, &vec);
    let crt_ntt = mat_vec_mul_crt_ntt(&mat, &vec).expect("Q64 dispatch should succeed");
    assert_eq!(schoolbook, crt_ntt);
}

#[test]
fn dense_mat_vec_many_matches_individual_crt_ntt_q32_d64() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..4)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        F::from_u64((i as u64 * 10_000 + j as u64 * 100 + k as u64 + 1) % 97)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let vecs: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|seed| {
            (0..4)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        F::from_u64((seed as u64 * 700 + j as u64 * 50 + k as u64 + 3) % 89)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let expected: Vec<Vec<CyclotomicRing<F, D>>> = vecs
        .iter()
        .map(|v| mat_vec_mul_crt_ntt(&mat, v).expect("single CRT+NTT mat-vec should succeed"))
        .collect();

    let got =
        mat_vec_mul_crt_ntt_many(&mat, &vecs).expect("batched CRT+NTT mat-vec should succeed");
    assert_eq!(expected, got);
}

#[test]
fn mat_vec_mul_digits_i8_matches_num_digits_one_roundtrip() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = (i as i64 * 19 + j as i64 * 7 + k as i64) % 7;
                        F::from_i64(raw - 3)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = vec![
        (0..6)
            .map(|j| std::array::from_fn(|k| ((j + 2 * k) % 7) as i8 - 3))
            .collect(),
        (0..4)
            .map(|j| std::array::from_fn(|k| ((2 * j + k) % 7) as i8 - 3))
            .collect(),
        vec![],
    ];

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
        .iter()
        .map(|block| {
            block
                .iter()
                .map(|digit| {
                    let coeffs = std::array::from_fn(|k| F::from_i64(digit[k] as i64));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let via_roundtrip = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                1,
                log_basis,
                &params,
            );
            let direct = mat_vec_mul_digits_i8_with_params_for_log_basis(
                &ntt_mat,
                &digit_block_slices,
                log_basis,
                &params,
            );
            assert_eq!(via_roundtrip, direct);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_matches_direct_digits_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 5 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..16)
        .map(|block_idx| {
            (0..6)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i16 * 3 + digit_idx as i16 * 5 + k as i16) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
        .iter()
        .map(|block| {
            block
                .chunks(num_digits)
                .map(|digits_for_ring| {
                    let coeffs = std::array::from_fn(|k| {
                        let mut acc = 0i64;
                        let mut place = 1i64;
                        for digit in digits_for_ring {
                            acc += i64::from(digit[k]) * place;
                            place <<= log_basis;
                        }
                        F::from_i64(acc)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let via_roundtrip = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let direct = mat_vec_mul_digits_i8_with_params_for_log_basis(
                &ntt_mat,
                &digit_block_slices,
                log_basis,
                &params,
            );
            assert_eq!(via_roundtrip, direct);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_matches_direct_digits_on_multi_tile_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;
    let num_blocks = 4;
    let rings_per_block = 1_400;
    let digits_per_block = rings_per_block * num_digits;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..5)
        .map(|i| {
            (0..digits_per_block)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 5 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block_idx| {
            (0..digits_per_block)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i16 * 3 + digit_idx as i16 * 5 + k as i16) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
        .iter()
        .map(|block| {
            block
                .chunks(num_digits)
                .map(|digits_for_ring| {
                    let coeffs = std::array::from_fn(|k| {
                        let mut acc = 0i64;
                        let mut place = 1i64;
                        for digit in digits_for_ring {
                            acc += i64::from(digit[k]) * place;
                            place <<= log_basis;
                        }
                        F::from_i64(acc)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let via_roundtrip = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let dense = mat_vec_mul_i8_dense_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let direct = mat_vec_mul_digits_i8_with_params_for_log_basis(
                &ntt_mat,
                &digit_block_slices,
                log_basis,
                &params,
            );
            assert_eq!(via_roundtrip, direct);
            assert_eq!(dense, direct);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_dense_fast_path_matches_generic_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((13 * i as i64 + 7 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = (0..16)
        .map(|block_idx| {
            (0..2)
                .map(|ring_idx| {
                    let coeffs = std::array::from_fn(|k| {
                        let d0 = ((block_idx as i64 + 2 * ring_idx as i64 + k as i64) % 7) - 3;
                        let d1 = ((2 * block_idx as i64 + ring_idx as i64 + 3 * k as i64) % 7) - 3;
                        let d2 = ((3 * block_idx as i64 + ring_idx as i64 + 5 * k as i64) % 7) - 3;
                        F::from_i64(d0 + (d1 << log_basis) + (d2 << (2 * log_basis)))
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let generic = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let dense = mat_vec_mul_i8_dense_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            assert_eq!(dense, generic);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_dense_single_row_matches_generic_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = vec![(0..6)
        .map(|j| {
            let coeffs = std::array::from_fn(|k| {
                let raw = ((7 * j as i64 + k as i64) % 9) - 4;
                F::from_i64(raw)
            });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect()];

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = (0..16)
        .map(|block_idx| {
            (0..2)
                .map(|ring_idx| {
                    let coeffs = std::array::from_fn(|k| {
                        let d0 = ((block_idx as i64 + 2 * ring_idx as i64 + k as i64) % 7) - 3;
                        let d1 = ((2 * block_idx as i64 + ring_idx as i64 + 3 * k as i64) % 7) - 3;
                        let d2 = ((3 * block_idx as i64 + ring_idx as i64 + 5 * k as i64) % 7) - 3;
                        F::from_i64(d0 + (d1 << log_basis) + (d2 << (2 * log_basis)))
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let generic = mat_vec_mul_i8_dense_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let single = super::mat_vec_mul_i8_dense_single_row_with_params(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let generic_single: Vec<CyclotomicRing<F, D>> =
                generic.into_iter().map(|row| row[0]).collect();
            assert_eq!(single, generic_single);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_dense_three_row_matches_generic_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 9 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = (0..16)
        .map(|block_idx| {
            (0..2)
                .map(|ring_idx| {
                    let coeffs = std::array::from_fn(|k| {
                        let d0 = ((block_idx as i64 + 2 * ring_idx as i64 + k as i64) % 7) - 3;
                        let d1 = ((2 * block_idx as i64 + ring_idx as i64 + 3 * k as i64) % 7) - 3;
                        let d2 = ((3 * block_idx as i64 + ring_idx as i64 + 5 * k as i64) % 7) - 3;
                        F::from_i64(d0 + (d1 << log_basis) + (d2 << (2 * log_basis)))
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let generic = mat_vec_mul_i8_dense_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let triple = super::mat_vec_mul_i8_dense_three_row_fused_with_params(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            assert_eq!(triple, generic);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_digits_i8_three_row_matches_generic_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((17 * i as i64 + 9 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..16)
        .map(|block_idx| {
            (0..6)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i64 + 2 * digit_idx as i64 + k as i64) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();

    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let generic = mat_vec_mul_digits_i8_with_params_for_log_basis::<
                F,
                i32,
                Q32_NUM_PRIMES,
                D,
            >(&ntt_mat, &digit_block_slices, log_basis, &params);
            let fused = super::mat_vec_mul_digits_i8_three_row_block_parallel::<
                F,
                i32,
                Q32_NUM_PRIMES,
                D,
                true,
            >(
                &ntt_mat,
                &digit_block_slices,
                super::balanced_digit_abs_bound(log_basis),
                &params,
            );
            assert_eq!(fused, generic);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_digits_i8_strided_three_row_matches_block_path_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((13 * i as i64 + 5 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..16)
        .map(|block_idx| {
            (0..6)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((2 * block_idx as i64 + digit_idx as i64 + 3 * k as i64) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();
    let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();
    let strided_digits: Vec<[i8; D]> = (0..6)
        .flat_map(|col| digit_blocks.iter().map(move |block| block[col]))
        .collect();

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let block_path = mat_vec_mul_digits_i8_with_params_for_log_basis::<
                F,
                i32,
                Q32_NUM_PRIMES,
                D,
            >(&ntt_mat, &digit_block_slices, log_basis, &params);
            let strided_path = super::mat_vec_mul_digits_i8_strided_block_parallel(
                &ntt_mat,
                &strided_digits,
                digit_blocks.len(),
                digit_blocks[0].len(),
                super::balanced_digit_abs_bound(log_basis),
                &params,
            );
            assert_eq!(strided_path, block_path);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_strided_matches_block_path_on_block_parallel_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((19 * i as i64 + 11 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = (0..16)
        .map(|block_idx| {
            (0..2)
                .map(|ring_idx| {
                    let coeffs = std::array::from_fn(|k| {
                        let d0 = ((block_idx as i64 + ring_idx as i64 + k as i64) % 7) - 3;
                        let d1 = ((2 * block_idx as i64 + ring_idx as i64 + k as i64) % 7) - 3;
                        let d2 = ((3 * block_idx as i64 + ring_idx as i64 + k as i64) % 7) - 3;
                        F::from_i64(d0 + (d1 << log_basis) + (d2 << (2 * log_basis)))
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
        ring_blocks.iter().map(Vec::as_slice).collect();

    let mut strided_coeffs = Vec::with_capacity(ring_blocks.len() * ring_blocks[0].len());
    for col in 0..ring_blocks[0].len() {
        for block in &ring_blocks {
            strided_coeffs.push(block[col]);
        }
    }

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let block_path = mat_vec_mul_i8_with_params_for_log_basis(
                &ntt_mat,
                &ring_block_slices,
                num_digits,
                log_basis,
                &params,
            );
            let strided_path = mat_vec_mul_i8_strided_with_params_for_log_basis(
                &ntt_mat,
                &strided_coeffs,
                ring_blocks.len(),
                ring_blocks[0].len(),
                num_digits,
                log_basis,
                &params,
            );
            assert_eq!(block_path, strided_path);
        }
        _ => panic!("unexpected parameter family"),
    }
}

#[test]
fn mat_vec_mul_i8_strided_matches_direct_digits_on_multi_tile_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let log_basis = 3;
    let num_digits = 3;
    let num_blocks = 4;
    let rings_per_block = 1_400;
    let digits_per_block = rings_per_block * num_digits;

    let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..5)
        .map(|i| {
            (0..digits_per_block)
                .map(|j| {
                    let coeffs = std::array::from_fn(|k| {
                        let raw = ((19 * i as i64 + 11 * j as i64 + k as i64) % 9) - 4;
                        F::from_i64(raw)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let digit_blocks: Vec<Vec<[i8; D]>> = (0..num_blocks)
        .map(|block_idx| {
            (0..digits_per_block)
                .map(|digit_idx| {
                    std::array::from_fn(|k| {
                        (((block_idx as i16 * 5 + digit_idx as i16 + 3 * k as i16) % 7) - 3) as i8
                    })
                })
                .collect()
        })
        .collect();

    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
        .iter()
        .map(|block| {
            block
                .chunks(num_digits)
                .map(|digits_for_ring| {
                    let coeffs = std::array::from_fn(|k| {
                        let mut acc = 0i64;
                        let mut place = 1i64;
                        for digit in digits_for_ring {
                            acc += i64::from(digit[k]) * place;
                            place <<= log_basis;
                        }
                        F::from_i64(acc)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect();

    let mut strided_coeffs = Vec::with_capacity(num_blocks * rings_per_block);
    for col in 0..rings_per_block {
        for block in &ring_blocks {
            strided_coeffs.push(block[col]);
        }
    }

    let mut strided_digits = Vec::with_capacity(num_blocks * digits_per_block);
    for col in 0..digits_per_block {
        for block in &digit_blocks {
            strided_digits.push(block[col]);
        }
    }

    match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
        ProtocolCrtNttParams::Q32(params) => {
            let ntt_mat_vecs = precompute_dense_mat_ntt_with_params(&mat, &params);
            let ntt_mat: Vec<&[_]> = ntt_mat_vecs.iter().map(Vec::as_slice).collect();
            let via_roundtrip = mat_vec_mul_i8_strided_with_params_for_log_basis(
                &ntt_mat,
                &strided_coeffs,
                num_blocks,
                rings_per_block,
                num_digits,
                log_basis,
                &params,
            );
            let direct = mat_vec_mul_digits_i8_strided_with_params_for_log_basis(
                &ntt_mat,
                &strided_digits,
                num_blocks,
                digits_per_block,
                log_basis,
                &params,
            );
            assert_eq!(via_roundtrip, direct);
        }
        _ => panic!("unexpected parameter family"),
    }
}
