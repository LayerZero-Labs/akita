use super::*;

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
    let slot = prepare_both_transforms(
        flat.ring_view::<D>(rows, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q32 dispatch should support this field and ring dimension");

    let t_hat: Vec<[i8; D]> = (0..cols)
        .map(|j| std::array::from_fn(|k| ((3 * j + k) % 5) as i8 - 2))
        .collect();
    let z_pre: Vec<[i32; D]> = (0..cols)
        .map(|j| std::array::from_fn(|k| ((j + k) % 3) as i32 - 1))
        .collect();

    let log_basis = 3;
    let expected_b = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, rows, cols, &t_hat, log_basis)
        .expect("expected B rows");
    let fused = fused_split_eq_quotients::<F, D>(&slot, rows, 1, &t_hat, &z_pre, 1)
        .expect("fused split-eq rows");

    assert_eq!(fused.b_cyclic, expected_b);
}

#[test]
fn fused_split_eq_q128_quotient_chunks_before_crt_wrap() {
    type F = Prime128Offset275;
    const D: usize = 32;
    let cols = 4;
    let modulus = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let half = F::from_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = prepare_both_transforms(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let fused = fused_split_eq_quotients::<F, D>(&slot, 0, 1, &[], &z_pre, 32_768)
        .expect("fused split-eq rows");

    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, j| {
        let z = centered_i32_ring(&z_pre[j]);
        let cyclic = cyclic_product(&row, &z);
        let negacyclic = row * z;
        acc += quotient_from_cyclic_and_negacyclic(&cyclic, &negacyclic);
        acc
    });

    assert_eq!(fused.a_quotients, vec![expected]);
}

#[test]
fn fused_split_eq_q128_quotient_falls_back_when_one_term_exceeds_crt() {
    type F = Prime128Offset275;
    const D: usize = 128;
    let cols = 1;
    let modulus = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let half = F::from_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat = FlatMatrix::from_ring_slice(&[row]);
    let slot = prepare_both_transforms(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let tail = prepare_ntt_cache(
        flat.ring_view::<D>(1, cols)
            .expect("valid tail matrix view"),
        NttCacheMode::I16TailBothTransforms,
    )
    .expect("Q128 quotient tail");
    let z_pre = vec![[32_768i32; D]; cols];

    let fused = fused_split_eq_quotients::<F, D>(&slot, 0, 1, &[], &z_pre, 32_768)
        .expect("fused split-eq rows");

    let z = centered_i32_ring(&z_pre[0]);
    let expected = quotient_from_cyclic_and_negacyclic(&cyclic_product(&row, &z), &(row * z));

    assert_eq!(fused.a_quotients, vec![expected]);
    let tail_rows =
        centered_quotient_rows_with_i16_tail::<F, D>(&slot, &slot, &tail, 1, &z_pre, 32_768)
            .expect("base-plus-tail quotient");
    assert_eq!(tail_rows, vec![expected]);
}

#[test]
fn fused_split_eq_uses_actual_centered_bound_when_hint_is_underreported() {
    type F = Prime128Offset275;
    const D: usize = 32;
    let cols = 4;
    let modulus = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let half = F::from_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = prepare_both_transforms(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let z_pre = vec![[32_768i32; D]; cols];

    let fused =
        fused_split_eq_quotients::<F, D>(&slot, 0, 1, &[], &z_pre, 1).expect("fused split-eq rows");

    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, j| {
        let z = centered_i32_ring(&z_pre[j]);
        let cyclic = cyclic_product(&row, &z);
        let negacyclic = row * z;
        acc += quotient_from_cyclic_and_negacyclic(&cyclic, &negacyclic);
        acc
    });

    assert_eq!(fused.a_quotients, vec![expected]);
}

#[test]
fn fused_split_eq_q128_cyclic_i8_chunks_before_crt_wrap() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let cols = 2_050;
    let modulus = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let half = F::from_u128_reduced(modulus / 2);
    let row = CyclotomicRing::from_coefficients([half; D]);
    let flat_rows = vec![row; cols];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = prepare_both_transforms(
        flat.ring_view::<D>(1, cols)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let e_hat = vec![[-32i8; D]; cols];

    let d_rows = crate::kernels::linear::digit_relation_rows_cached_prover_bounds::<F, D>(
        &slot, &slot, 1, &e_hat, 6,
    )
    .expect("D relation rows")
    .cyclic;

    let digit = CyclotomicRing::from_coefficients([F::from_i64(-32); D]);
    let expected = (0..cols).fold(CyclotomicRing::<F, D>::zero(), |mut acc, _| {
        acc += cyclic_product(&row, &digit);
        acc
    });

    assert_eq!(d_rows, vec![expected]);
}

#[test]
fn fused_split_eq_quotients_uses_role_local_packed_widths() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let n_b = 3;
    let n_a = 2;
    let b_width = 4;
    let a_width = 3;
    let total_len = (n_b * b_width).max(n_a * a_width);
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
    let slot = prepare_both_transforms(
        flat.ring_view::<D>(1, total_len)
            .expect("valid packed setup prefix"),
    )
    .expect("Q32 dispatch should support this field and ring dimension");

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
    let fused = fused_split_eq_quotients::<F, D>(&slot, n_b, n_a, &t_hat, &z_pre, 3)
        .expect("fused split-eq rows");

    assert_eq!(fused.b_cyclic, expected_b);
    assert_eq!(fused.a_quotients, expected_a);
}

fn run_test_value(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Checks the cached, streamed, and (for Q128) base-plus-tail quotient routes
/// against schoolbook products when zero columns split the lazy-dot runs.
fn assert_fused_quotient_runs_match_schoolbook<F: Field + CanonicalEncoding, const D: usize>(
    check_i16_tail: bool,
) {
    let (n_b, n_a, b_width, a_width) = (3, 2, 23, 17);
    let z_max_abs = 4_000u32;
    let total_len = (n_b * b_width).max(n_a * a_width);
    let flat_rows: Vec<CyclotomicRing<F, D>> = (0..total_len)
        .map(|idx| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                F::from_i64(run_test_value((idx * D + k) as u64) as i64)
            }))
        })
        .collect();
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = prepare_both_transforms(
        flat.ring_view::<D>(1, total_len)
            .expect("valid packed setup prefix"),
    )
    .expect("protocol CRT dispatch should support this field and ring dimension");

    let t_hat: Vec<[i8; D]> = (0..b_width)
        .map(|j| {
            if j == 0 || j == 13 || j == 14 || j % 9 == 4 {
                [0; D]
            } else {
                std::array::from_fn(|k| {
                    (run_test_value((1 << 32) + (j * D + k) as u64) % 64) as i8 - 32
                })
            }
        })
        .collect();
    let z_pre: Vec<[i32; D]> = (0..a_width)
        .map(|j| {
            if j % 7 == 3 {
                [0; D]
            } else {
                std::array::from_fn(|k| {
                    let raw = run_test_value((2 << 32) + (j * D + k) as u64);
                    (raw % (2 * u64::from(z_max_abs) + 1)) as i32 - z_max_abs as i32
                })
            }
        })
        .collect();

    let expected_b = (0..n_b)
        .map(|row| {
            t_hat
                .iter()
                .enumerate()
                .fold(CyclotomicRing::<F, D>::zero(), |mut acc, (j, t)| {
                    let rhs = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                        F::from_i64(i64::from(t[k]))
                    }));
                    acc += cyclic_product(&flat_rows[row * b_width + j], &rhs);
                    acc
                })
        })
        .collect::<Vec<_>>();
    let expected_a = (0..n_a)
        .map(|row| {
            z_pre
                .iter()
                .enumerate()
                .fold(CyclotomicRing::<F, D>::zero(), |mut acc, (j, z)| {
                    let lhs = flat_rows[row * a_width + j];
                    let z = centered_i32_ring(z);
                    acc +=
                        quotient_from_cyclic_and_negacyclic(&cyclic_product(&lhs, &z), &(lhs * z));
                    acc
                })
        })
        .collect::<Vec<_>>();

    let fused = fused_split_eq_quotients::<F, D>(&slot, n_b, n_a, &t_hat, &z_pre, z_max_abs)
        .expect("cached fused rows");
    assert_eq!(fused.b_cyclic, expected_b);
    assert_eq!(fused.a_quotients, expected_a);

    let streamed = crate::kernels::linear::fused_split_eq_quotients_streamed_prover_bounds::<F, D>(
        &flat_rows, n_b, n_a, &t_hat, &z_pre, z_max_abs, 6,
    )
    .expect("streamed fused rows");
    assert_eq!(streamed.b_cyclic, expected_b);
    assert_eq!(streamed.a_quotients, expected_a);

    if check_i16_tail {
        let tail = prepare_ntt_cache(
            flat.ring_view::<D>(1, total_len)
                .expect("valid tail matrix view"),
            NttCacheMode::I16TailBothTransforms,
        )
        .expect("Q128 quotient tail");
        let tail_rows = centered_quotient_rows_with_i16_tail::<F, D>(
            &slot, &slot, &tail, n_a, &z_pre, z_max_abs,
        )
        .expect("base-plus-tail quotient rows");
        assert_eq!(tail_rows, expected_a);
    }
}

#[test]
fn fused_quotient_runs_match_schoolbook() {
    assert_fused_quotient_runs_match_schoolbook::<Fp64<4294967197>, 64>(false);
    assert_fused_quotient_runs_match_schoolbook::<Prime64Offset59, 64>(false);
    assert_fused_quotient_runs_match_schoolbook::<Prime128Offset275, 32>(false);
    assert_fused_quotient_runs_match_schoolbook::<Prime128Offset275, 64>(true);
}

#[test]
fn fused_quotient_chunked_runs_match_schoolbook() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let (b_width, a_width) = (2_050, 20);
    let modulus = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let row = CyclotomicRing::from_coefficients([F::from_u128_reduced(modulus / 2); D]);
    let flat_rows = vec![row; b_width];
    let flat = FlatMatrix::from_ring_slice(&flat_rows);
    let slot = prepare_both_transforms(
        flat.ring_view::<D>(1, b_width)
            .expect("valid ring matrix view"),
    )
    .expect("Q128 dispatch should support this field and ring dimension");
    let tail = prepare_ntt_cache(
        flat.ring_view::<D>(1, b_width)
            .expect("valid tail matrix view"),
        NttCacheMode::I16TailBothTransforms,
    )
    .expect("Q128 quotient tail");

    let is_zero_column = |j: usize| j % 11 == 5 || j % 97 < 3;
    let t_hat: Vec<[i8; D]> = (0..b_width)
        .map(|j| if is_zero_column(j) { [0; D] } else { [-32; D] })
        .collect();
    let z_pre: Vec<[i32; D]> = (0..a_width)
        .map(|j| {
            if is_zero_column(j) {
                [0; D]
            } else {
                [32_768; D]
            }
        })
        .collect();

    let digit = CyclotomicRing::from_coefficients([F::from_i64(-32); D]);
    let b_term = cyclic_product(&row, &digit);
    let expected_b = (0..b_width).filter(|&j| !is_zero_column(j)).fold(
        CyclotomicRing::<F, D>::zero(),
        |mut acc, _| {
            acc += b_term;
            acc
        },
    );
    let z = centered_i32_ring(&[32_768; D]);
    let a_term = quotient_from_cyclic_and_negacyclic(&cyclic_product(&row, &z), &(row * z));
    let expected_a = (0..a_width).filter(|&j| !is_zero_column(j)).fold(
        CyclotomicRing::<F, D>::zero(),
        |mut acc, _| {
            acc += a_term;
            acc
        },
    );

    let fused = fused_split_eq_quotients::<F, D>(&slot, 1, 1, &t_hat, &z_pre, 32_768)
        .expect("cached fused rows");
    assert_eq!(fused.b_cyclic, vec![expected_b]);
    assert_eq!(fused.a_quotients, vec![expected_a]);

    let streamed = crate::kernels::linear::fused_split_eq_quotients_streamed_prover_bounds::<F, D>(
        &flat_rows, 1, 1, &t_hat, &z_pre, 32_768, 6,
    )
    .expect("streamed fused rows");
    assert_eq!(streamed.b_cyclic, vec![expected_b]);
    assert_eq!(streamed.a_quotients, vec![expected_a]);

    let tail_rows =
        centered_quotient_rows_with_i16_tail::<F, D>(&slot, &slot, &tail, 1, &z_pre, 32_768)
            .expect("base-plus-tail quotient rows");
    assert_eq!(tail_rows, vec![expected_a]);
}

#[test]
fn nonzero_column_runs_split_at_zeros_and_batch_limit() {
    let zeros = [0usize, 3, 4, 11];
    let mut runs = Vec::new();
    crate::kernels::linear::common::for_each_nonzero_column_run(
        0..16,
        3,
        |column| zeros.contains(&column),
        |run| runs.push(run),
    );
    assert_eq!(runs, vec![1..3, 5..8, 8..11, 12..15, 15..16]);
}
