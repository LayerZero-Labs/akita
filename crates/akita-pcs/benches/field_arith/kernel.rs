use akita_field::packed::{HasPacking, PackedField, PackedValue};
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
#[cfg(target_arch = "x86_64")]
use akita_field::Fp32;
use akita_field::{
    CanonicalField, FieldCore, FpExt4, Prime128Offset275, Prime32Offset99, RandomSampling, Zero,
};
use akita_prover::kernels::sumcheck::SumcheckKernelPlan;
use akita_sumcheck::{
    compute_product_round_scalar, fold_and_compute_product_round_scalar, EvaluationTable,
};
use criterion::{black_box, BatchSize, Criterion, Throughput};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use super::cases::*;
use super::data::rand_u128;

pub(crate) fn bench_kernel_patterns(c: &mut Criterion) {
    bench_packed_sumcheck_mix(c);
    bench_fp32_ext4_sumcheck_fold(c);
    bench_fp32_ext4_tensor_factor_pair(c);
    #[cfg(target_arch = "x86_64")]
    bench_fp32_ext4_tensor_factor_materialization(c);
    bench_fp128_accumulator_pattern(c);
}

#[cfg(target_arch = "x86_64")]
fn bench_fp32_ext4_tensor_factor_materialization(c: &mut Criterion) {
    type F = Prime32Offset99;
    type E = FpExt4<F>;

    let inner_len = 1usize << 8;
    let outer_len = 1usize << 6;
    let table_len = 2 * inner_len * outer_len;
    let mut rng = StdRng::seed_from_u64(0xf032_fac7_5120);
    let witness_values = (0..table_len)
        .map(|_| E::random(&mut rng))
        .collect::<Vec<_>>();
    let witness = EvaluationTable::from_evaluations(&witness_values);
    let equality_inner_values = (0..inner_len)
        .map(|_| E::random(&mut rng))
        .collect::<Vec<_>>();
    let equality_inner = EvaluationTable::from_evaluations(&equality_inner_values);
    let equality_outer = (0..outer_len)
        .map(|_| E::random(&mut rng))
        .collect::<Vec<_>>();
    let zero_weights = std::array::from_fn(|_| E::random(&mut rng));
    let one_weights = std::array::from_fn(|_| E::random(&mut rng));
    let witness = witness.coefficient_slices::<4>();
    let equality_inner = equality_inner.coefficient_slices::<4>();

    let mut group = c.benchmark_group("field_arith/kernel/fp32_ext4_tensor_factor_materialization");
    group.throughput(Throughput::Elements(table_len as u64));

    if std::is_x86_feature_detected!("avx2") {
        group.bench_function("runtime_avx2", |b| {
            b.iter(|| unsafe {
                black_box(
                    akita_field::packed::runtime_x86::materialize_tensor_factor_and_compute_product_round_fp_ext4_fp32_avx2(
                        witness,
                        equality_inner,
                        black_box(&equality_outer),
                        zero_weights,
                        one_weights,
                    ),
                )
            })
        });
    }

    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512ifma")
    {
        group.bench_function("runtime_avx512_ifma", |b| {
            b.iter(|| unsafe {
                black_box(
                    akita_field::packed::runtime_x86::materialize_tensor_factor_and_compute_product_round_fp_ext4_fp32_avx512_ifma(
                        witness,
                        equality_inner,
                        black_box(&equality_outer),
                        zero_weights,
                        one_weights,
                    ),
                )
            })
        });
    }
    group.finish();
}

fn bench_fp32_ext4_tensor_factor_pair(c: &mut Criterion) {
    type E = FpExt4<Prime32Offset99>;
    type PE = <E as HasPacking>::Packing;

    let n = 1usize << 14;
    let mut rng = StdRng::seed_from_u64(0xf032_fac7);
    let suffix_tables: [Vec<E>; 4] =
        std::array::from_fn(|_| (0..n).map(|_| E::random(&mut rng)).collect());
    let coeff_zero: [Vec<E>; 4] =
        std::array::from_fn(|_| (0..n).map(|_| E::random(&mut rng)).collect());
    let coeff_one: [Vec<E>; 4] =
        std::array::from_fn(|_| (0..n).map(|_| E::random(&mut rng)).collect());
    let packed_suffix: [Vec<PE>; 4] = std::array::from_fn(|j| PE::pack_slice(&suffix_tables[j]));
    let packed_zero: [Vec<PE>; 4] = std::array::from_fn(|j| PE::pack_slice(&coeff_zero[j]));
    let packed_one: [Vec<PE>; 4] = std::array::from_fn(|j| PE::pack_slice(&coeff_one[j]));

    let mut scalar_out = vec![(E::zero(), E::zero()); n];
    let mut packed_out = vec![(PE::broadcast(E::zero()), PE::broadcast(E::zero())); n / PE::WIDTH];
    let mut group = c.benchmark_group("field_arith/kernel/fp32_ext4_tensor_factor_pair");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("scalar_delayed_four_term", |b| {
        b.iter(|| {
            for (i, output) in scalar_out.iter_mut().enumerate() {
                let mut zero = <E as HasUnreducedOps>::ProductAccum::zero();
                let mut one = <E as HasUnreducedOps>::ProductAccum::zero();
                for j in 0..4 {
                    let column = suffix_tables[j][i];
                    zero += coeff_zero[j][i].mul_to_product_accum(column);
                    one += coeff_one[j][i].mul_to_product_accum(column);
                }
                *output = (E::reduce_product_accum(zero), E::reduce_product_accum(one));
            }
            black_box(&scalar_out);
        })
    });

    group.bench_function(format!("packed_persistent_w{}", PE::WIDTH), |b| {
        b.iter(|| {
            for (i, output) in packed_out.iter_mut().enumerate() {
                let mut zero = PE::broadcast(E::zero());
                let mut one = PE::broadcast(E::zero());
                for j in 0..4 {
                    let column = packed_suffix[j][i];
                    zero = zero + packed_zero[j][i] * column;
                    one = one + packed_one[j][i] * column;
                }
                *output = (zero, one);
            }
            black_box(&packed_out);
        })
    });

    group.bench_function(format!("packed_repack_w{}", PE::WIDTH), |b| {
        b.iter(|| {
            for (packed_index, output) in packed_out.iter_mut().enumerate() {
                let first = packed_index * PE::WIDTH;
                let mut zero = PE::broadcast(E::zero());
                let mut one = PE::broadcast(E::zero());
                for j in 0..4 {
                    let column = PE::from_fn(|lane| suffix_tables[j][first + lane]);
                    let packed_coeff_zero = PE::from_fn(|lane| coeff_zero[j][first + lane]);
                    let packed_coeff_one = PE::from_fn(|lane| coeff_one[j][first + lane]);
                    zero = zero + packed_coeff_zero * column;
                    one = one + packed_coeff_one * column;
                }
                *output = (zero, one);
            }
            black_box(&packed_out);
        })
    });

    group.bench_function(
        format!("packed_persistent_suffix_repack_coeff_w{}", PE::WIDTH),
        |b| {
            b.iter(|| {
                for (packed_index, output) in packed_out.iter_mut().enumerate() {
                    let first = packed_index * PE::WIDTH;
                    let mut zero = PE::broadcast(E::zero());
                    let mut one = PE::broadcast(E::zero());
                    for j in 0..4 {
                        let column = packed_suffix[j][packed_index];
                        let packed_coeff_zero = PE::from_fn(|lane| coeff_zero[j][first + lane]);
                        let packed_coeff_one = PE::from_fn(|lane| coeff_one[j][first + lane]);
                        zero = zero + packed_coeff_zero * column;
                        one = one + packed_coeff_one * column;
                    }
                    *output = (zero, one);
                }
                black_box(&packed_out);
            })
        },
    );

    group.bench_function(
        format!("packed_broadcast_suffix_repack_coeff_w{}", PE::WIDTH),
        |b| {
            b.iter(|| {
                for (packed_index, output) in packed_out.iter_mut().enumerate() {
                    let first = packed_index * PE::WIDTH;
                    let suffix_index = packed_index / 4;
                    let mut zero = PE::broadcast(E::zero());
                    let mut one = PE::broadcast(E::zero());
                    for j in 0..4 {
                        let column = PE::broadcast(suffix_tables[j][suffix_index]);
                        let packed_coeff_zero = PE::from_fn(|lane| coeff_zero[j][first + lane]);
                        let packed_coeff_one = PE::from_fn(|lane| coeff_one[j][first + lane]);
                        zero = zero + packed_coeff_zero * column;
                        one = one + packed_coeff_one * column;
                    }
                    *output = (zero, one);
                }
                black_box(&packed_out);
            })
        },
    );

    group.bench_function(
        format!("packed_broadcast_suffix_stack_coeff_w{}", PE::WIDTH),
        |b| {
            b.iter(|| {
                for (packed_index, output) in packed_out.iter_mut().enumerate() {
                    let first = packed_index * PE::WIDTH;
                    let suffix_index = packed_index / 4;
                    let mut zero = PE::broadcast(E::zero());
                    let mut one = PE::broadcast(E::zero());
                    for j in 0..4 {
                        let column = PE::broadcast(suffix_tables[j][suffix_index]);
                        let packed_coeff_zero = PE::from_coeff_fn(|lane, coefficient| {
                            coeff_zero[j][first + lane].coeffs[coefficient]
                        });
                        let packed_coeff_one = PE::from_coeff_fn(|lane, coefficient| {
                            coeff_one[j][first + lane].coeffs[coefficient]
                        });
                        zero = zero + packed_coeff_zero * column;
                        one = one + packed_coeff_one * column;
                    }
                    *output = (zero, one);
                }
                black_box(&packed_out);
            })
        },
    );
    group.finish();
}

fn bench_fp32_ext4_sumcheck_fold(c: &mut Criterion) {
    type E = FpExt4<Prime32Offset99>;
    type PE = <E as HasPacking>::Packing;

    let half = 1usize << 14;
    let mut rng = StdRng::seed_from_u64(0xf032_f01d);
    let left = (0..half).map(|_| E::random(&mut rng)).collect::<Vec<_>>();
    let right = (0..half).map(|_| E::random(&mut rng)).collect::<Vec<_>>();
    let interleaved = left
        .iter()
        .zip(&right)
        .flat_map(|(&lhs, &rhs)| [lhs, rhs])
        .collect::<Vec<_>>();
    let challenge = E::random(&mut rng);
    let fold_ctx = E::precompute_fold(challenge);
    let packed_left = PE::pack_slice(&left);
    let packed_right = PE::pack_slice(&right);
    let packed_challenge = PE::broadcast(challenge);
    let runtime_plan = SumcheckKernelPlan::detect();
    let runtime_table = EvaluationTable::from_evaluation_fn(2 * half, |row| {
        if row < half {
            left[row]
        } else {
            right[row - half]
        }
    });
    let product_factor_values = (0..2 * half)
        .map(|_| E::random(&mut rng))
        .collect::<Vec<_>>();
    let product_factor = EvaluationTable::from_evaluations(&product_factor_values);

    let mut scalar_out = vec![E::zero(); half];
    let mut packed_out = vec![PE::broadcast(E::zero()); packed_left.len()];
    let mut group = c.benchmark_group("field_arith/kernel/fp32_ext4_sumcheck_fold");
    group.throughput(Throughput::Elements(half as u64));

    group.bench_function("scalar_optimized_fold", |b| {
        b.iter(|| {
            let lhs = black_box(&left);
            let rhs = black_box(&right);
            for (i, dst) in scalar_out.iter_mut().enumerate() {
                *dst = E::fold_one(&fold_ctx, lhs[i], rhs[i]);
            }
            black_box(&scalar_out);
        })
    });

    group.bench_function("scalar_scale", |b| {
        b.iter(|| {
            let input = black_box(&left);
            for (dst, &value) in scalar_out.iter_mut().zip(input) {
                *dst = value * challenge;
            }
            black_box(&scalar_out);
        })
    });

    group.bench_function("scalar_optimized_zero_left_fold", |b| {
        b.iter(|| {
            let input = black_box(&left);
            for (dst, &value) in scalar_out.iter_mut().zip(input) {
                *dst = E::fold_one(&fold_ctx, E::zero(), value);
            }
            black_box(&scalar_out);
        })
    });

    group.bench_function(format!("packed_fold_w{}", PE::WIDTH), |b| {
        b.iter(|| {
            let lhs = black_box(&packed_left);
            let rhs = black_box(&packed_right);
            for (i, dst) in packed_out.iter_mut().enumerate() {
                *dst = lhs[i] + packed_challenge * (rhs[i] - lhs[i]);
            }
            black_box(&packed_out);
        })
    });

    group.bench_function("runtime_evaluation_table", |b| {
        b.iter_batched(
            || runtime_table.clone(),
            |mut table| {
                runtime_plan.fold_first_variable_fp32(&mut table, challenge);
                black_box(table);
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("scalar_product_round_evaluation_tables", |b| {
        b.iter(|| {
            black_box(compute_product_round_scalar(
                black_box(&runtime_table),
                black_box(&product_factor),
            ))
        })
    });

    group.bench_function("runtime_product_round_evaluation_tables", |b| {
        b.iter(|| {
            black_box(
                runtime_plan.compute_product_round_fp32(
                    black_box(&runtime_table),
                    black_box(&product_factor),
                ),
            )
        })
    });

    group.bench_function("scalar_fused_fold_product_round_evaluation_tables", |b| {
        b.iter_batched(
            || (runtime_table.clone(), product_factor.clone()),
            |(mut witness, mut factor)| {
                black_box(fold_and_compute_product_round_scalar(
                    &mut witness,
                    &mut factor,
                    challenge,
                ));
                black_box((witness, factor));
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("runtime_fused_fold_product_round_evaluation_tables", |b| {
        b.iter_batched(
            || (runtime_table.clone(), product_factor.clone()),
            |(mut witness, mut factor)| {
                black_box(runtime_plan.fold_and_compute_product_round_fp32(
                    &mut witness,
                    &mut factor,
                    challenge,
                ));
                black_box((witness, factor));
            },
            BatchSize::SmallInput,
        )
    });

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        group.bench_function("runtime_avx2_evaluation_table", |b| {
            b.iter_batched(
                || runtime_table.clone(),
                |mut table| {
                    bench_fold_fp32_avx2(&mut table, challenge);
                    black_box(table);
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("runtime_avx2_product_round_evaluation_tables", |b| {
            b.iter(|| {
                black_box(bench_product_round_fp32_avx2(
                    black_box(&runtime_table),
                    black_box(&product_factor),
                ))
            })
        });
        group.bench_function(
            "runtime_avx2_fused_fold_product_round_evaluation_tables",
            |b| {
                b.iter_batched(
                    || (runtime_table.clone(), product_factor.clone()),
                    |(mut witness, mut factor)| {
                        black_box(bench_fused_product_round_fp32_avx2(
                            &mut witness,
                            &mut factor,
                            challenge,
                        ));
                        black_box((witness, factor));
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512ifma")
    {
        group.bench_function("runtime_avx512_ifma_evaluation_table", |b| {
            b.iter_batched(
                || runtime_table.clone(),
                |mut table| {
                    bench_fold_fp32_avx512_ifma(&mut table, challenge);
                    black_box(table);
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("runtime_avx512_ifma_product_round_evaluation_tables", |b| {
            b.iter(|| {
                black_box(bench_product_round_fp32_avx512_ifma(
                    black_box(&runtime_table),
                    black_box(&product_factor),
                ))
            })
        });
        group.bench_function(
            "runtime_avx512_ifma_fused_fold_product_round_evaluation_tables",
            |b| {
                b.iter_batched(
                    || (runtime_table.clone(), product_factor.clone()),
                    |(mut witness, mut factor)| {
                        black_box(bench_fused_product_round_fp32_avx512_ifma(
                            &mut witness,
                            &mut factor,
                            challenge,
                        ));
                        black_box((witness, factor));
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.throughput(Throughput::Elements((2 * half) as u64));
    group.bench_function(format!("pack_two_halves_w{}", PE::WIDTH), |b| {
        b.iter(|| {
            black_box((
                PE::pack_slice(black_box(&left)),
                PE::pack_slice(black_box(&right)),
            ))
        })
    });

    group.throughput(Throughput::Elements(half as u64));
    group.bench_function(format!("unpack_folded_w{}", PE::WIDTH), |b| {
        b.iter(|| black_box(PE::unpack_slice(black_box(&packed_out))))
    });

    group.bench_function(format!("repack_adjacent_fold_unpack_w{}", PE::WIDTH), |b| {
        b.iter(|| {
            let input = black_box(&interleaved);
            for (packed_index, output) in scalar_out.chunks_exact_mut(PE::WIDTH).enumerate() {
                let first_output = packed_index * PE::WIDTH;
                let lhs = PE::from_fn(|lane| input[2 * (first_output + lane)]);
                let rhs = PE::from_fn(|lane| input[2 * (first_output + lane) + 1]);
                let folded = lhs + packed_challenge * (rhs - lhs);
                for (lane, value) in output.iter_mut().enumerate() {
                    *value = folded.extract(lane);
                }
            }
            black_box(&scalar_out);
        })
    });
    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn bench_fold_fp32_avx2<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) {
    let half = table.len() / 2;
    let (left, right) = bench_coefficient_halves_mut(table);

    // SAFETY: the benchmark only calls this helper after detecting AVX2. Its
    // fixed power-of-two input gives every coefficient equal complete halves.
    unsafe { akita_field::packed::runtime_x86::fold_fp_ext4_fp32_avx2(left, right, challenge) };
    table.truncate(half);
}

#[cfg(target_arch = "x86_64")]
fn bench_fold_fp32_avx512_ifma<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) {
    let half = table.len() / 2;
    let (left, right) = bench_coefficient_halves_mut(table);

    // SAFETY: the benchmark only calls this helper after detecting AVX-512F,
    // AVX-512DQ, and AVX-512IFMA. Its fixed power-of-two input gives every
    // coefficient equal complete halves.
    unsafe {
        akita_field::packed::runtime_x86::fold_fp_ext4_fp32_avx512_ifma(left, right, challenge)
    };
    table.truncate(half);
}

#[cfg(target_arch = "x86_64")]
fn bench_coefficient_halves<const P: u32>(
    table: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
) -> ([&[Fp32<P>]; 4], [&[Fp32<P>]; 4]) {
    let half = table.len() / 2;
    (
        std::array::from_fn(|coefficient| &table.coefficient_slice(coefficient)[..half]),
        std::array::from_fn(|coefficient| &table.coefficient_slice(coefficient)[half..]),
    )
}

#[cfg(target_arch = "x86_64")]
fn bench_product_round_fp32_avx2<const P: u32>(
    witness: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    factor: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    let (witness_0, witness_1) = bench_coefficient_halves(witness);
    let (factor_0, factor_1) = bench_coefficient_halves(factor);
    // SAFETY: the benchmark only calls this helper after detecting AVX2. All
    // slices come from equal halves of same-length power-of-two tables.
    unsafe {
        akita_field::packed::runtime_x86::compute_product_round_fp_ext4_fp32_avx2(
            witness_0, witness_1, factor_0, factor_1,
        )
    }
}

#[cfg(target_arch = "x86_64")]
fn bench_product_round_fp32_avx512_ifma<const P: u32>(
    witness: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    factor: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    let (witness_0, witness_1) = bench_coefficient_halves(witness);
    let (factor_0, factor_1) = bench_coefficient_halves(factor);
    // SAFETY: the benchmark only calls this helper after detecting AVX-512F,
    // AVX-512DQ, and AVX-512IFMA. All slices come from equal halves of
    // same-length power-of-two tables.
    unsafe {
        akita_field::packed::runtime_x86::compute_product_round_fp_ext4_fp32_avx512_ifma(
            witness_0, witness_1, factor_0, factor_1,
        )
    }
}

#[cfg(target_arch = "x86_64")]
fn bench_fused_product_round_fp32_avx2<const P: u32>(
    witness: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    factor: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    let half = witness.len() / 2;
    let (witness_left, witness_right) = bench_coefficient_halves_mut(witness);
    let (factor_left, factor_right) = bench_coefficient_halves_mut(factor);
    // SAFETY: the benchmark only calls this helper after detecting AVX2. All
    // slices come from equal halves of same-length power-of-two tables.
    let round = unsafe {
        akita_field::packed::runtime_x86::fold_and_compute_product_round_fp_ext4_fp32_avx2(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    };
    witness.truncate(half);
    factor.truncate(half);
    round
}

#[cfg(target_arch = "x86_64")]
fn bench_fused_product_round_fp32_avx512_ifma<const P: u32>(
    witness: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    factor: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    let half = witness.len() / 2;
    let (witness_left, witness_right) = bench_coefficient_halves_mut(witness);
    let (factor_left, factor_right) = bench_coefficient_halves_mut(factor);
    // SAFETY: the benchmark only calls this helper after detecting AVX-512F,
    // AVX-512DQ, and AVX-512IFMA. All slices come from equal halves of
    // same-length power-of-two tables.
    let round = unsafe {
        akita_field::packed::runtime_x86::fold_and_compute_product_round_fp_ext4_fp32_avx512_ifma(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    };
    witness.truncate(half);
    factor.truncate(half);
    round
}

#[cfg(target_arch = "x86_64")]
fn bench_coefficient_halves_mut<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
) -> ([&mut [Fp32<P>]; 4], [&[Fp32<P>]; 4]) {
    let half = table.len() / 2;
    let [coefficient_0, coefficient_1, coefficient_2, coefficient_3] =
        table.coefficient_slices_mut::<4>();
    let (left_0, right_0) = coefficient_0.split_at_mut(half);
    let (left_1, right_1) = coefficient_1.split_at_mut(half);
    let (left_2, right_2) = coefficient_2.split_at_mut(half);
    let (left_3, right_3) = coefficient_3.split_at_mut(half);
    (
        [left_0, left_1, left_2, left_3],
        [right_0, right_1, right_2, right_3],
    )
}

fn bench_packed_sumcheck_mix(c: &mut Criterion) {
    let n = 4096u64;
    let mut rng = StdRng::seed_from_u64(0x5151_cafe);

    let mut group = c.benchmark_group("field_arith/kernel/packed_macc");
    group.throughput(Throughput::Elements(n));

    use akita_field::{Prime31Offset19, Prime32Offset99, Prime40Offset195, Prime64Offset59};

    sumcheck_bench::<Prime31Offset19, P31O19>(&mut group, PRIME31_OFFSET19, &mut rng, n);
    sumcheck_bench::<Mersenne31, PackedMersenne31>(&mut group, MERSENNE31, &mut rng, n);
    sumcheck_bench::<Prime32Offset99, P32O99>(&mut group, PRIME32_OFFSET99, &mut rng, n);
    sumcheck_bench::<Prime40Offset195, P40O195>(&mut group, PRIME40_OFFSET195, &mut rng, n);
    sumcheck_bench::<Prime64Offset59, P64O59>(&mut group, PRIME64_OFFSET59, &mut rng, n);
    sumcheck_bench::<F128, P128O275>(&mut group, PRIME128_OFFSET275, &mut rng, n);

    group.finish();
}

fn sumcheck_bench<F, PF>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    rng: &mut StdRng,
    n: u64,
) where
    F: FieldCore + RandomSampling + 'static,
    PF: PackedField<Scalar = F> + Copy + 'static,
{
    let eq: Vec<F> = (0..n).map(|_| F::random(rng)).collect();
    let poly: Vec<F> = (0..n).map(|_| F::random(rng)).collect();
    let eq_p = PF::pack_slice(&eq);
    let poly_p = PF::pack_slice(&poly);

    group.bench_function(format!("{label}_packed_macc"), |b| {
        b.iter(|| {
            let e = black_box(&eq_p);
            let p_v = black_box(&poly_p);
            let mut acc = PF::broadcast(F::zero());
            for i in 0..e.len() {
                acc = acc + e[i] * p_v[i];
            }
            black_box(acc)
        })
    });
}

fn bench_fp128_accumulator_pattern(c: &mut Criterion) {
    type F = Prime128Offset275;

    let mut rng = StdRng::seed_from_u64(0xacc0_1a70_0002);
    let inputs_a: Vec<F> = (0..256)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let inputs_b_u64: Vec<u64> = (0..256).map(|_| rng.next_u64()).collect();
    let inputs_b_f: Vec<F> = (0..256)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let mut group = c.benchmark_group("field_arith/kernel/fp128_accumulator");

    for &n in &[16, 64, 256] {
        group.bench_function(format!("eager_mul_u64_{n}"), |bench| {
            bench.iter(|| {
                let a_s = black_box(&inputs_a[..n]);
                let b_s = black_box(&inputs_b_u64[..n]);
                let mut acc = F::zero();
                for i in 0..n {
                    acc += a_s[i] * F::from_u64(b_s[i]);
                }
                black_box(acc)
            })
        });

        group.bench_function(format!("widening_accum_u64_{n}"), |bench| {
            bench.iter(|| {
                let a_s = black_box(&inputs_a[..n]);
                let b_s = black_box(&inputs_b_u64[..n]);
                let mut acc = [0u64; 5];
                for i in 0..n {
                    let wide = a_s[i].mul_wide_u64(b_s[i]);
                    let mut carry: u64 = 0;
                    for j in 0..3 {
                        let sum = acc[j] as u128 + wide[j] as u128 + carry as u128;
                        acc[j] = sum as u64;
                        carry = (sum >> 64) as u64;
                    }
                    for item in &mut acc[3..5] {
                        let sum = *item as u128 + carry as u128;
                        *item = sum as u64;
                        carry = (sum >> 64) as u64;
                    }
                }
                black_box(F::solinas_reduce(&acc))
            })
        });

        group.bench_function(format!("eager_mul_full_{n}"), |bench| {
            bench.iter(|| {
                let a_s = black_box(&inputs_a[..n]);
                let b_s = black_box(&inputs_b_f[..n]);
                let mut acc = F::zero();
                for i in 0..n {
                    acc += a_s[i] * b_s[i];
                }
                black_box(acc)
            })
        });

        group.bench_function(format!("widening_accum_full_{n}"), |bench| {
            bench.iter(|| {
                let a_s = black_box(&inputs_a[..n]);
                let b_s = black_box(&inputs_b_f[..n]);
                let mut acc = [0u64; 6];
                for i in 0..n {
                    let wide = a_s[i].mul_wide(b_s[i]);
                    let mut carry: u64 = 0;
                    for j in 0..4 {
                        let sum = acc[j] as u128 + wide[j] as u128 + carry as u128;
                        acc[j] = sum as u64;
                        carry = (sum >> 64) as u64;
                    }
                    for item in &mut acc[4..6] {
                        let sum = *item as u128 + carry as u128;
                        *item = sum as u64;
                        carry = (sum >> 64) as u64;
                    }
                }
                black_box(F::solinas_reduce(&acc))
            })
        });
    }

    group.finish();
}
