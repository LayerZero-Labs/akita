use akita_field::fields::Prime128OffsetA7F7;
use akita_field::CanonicalField;
use akita_metal::field::fp128::Fp128VectorOp;
use akita_metal::MetalBackend;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const P_A7F7: u128 = 0xffffffffffffffffffffffff00005809;
fn sample_inputs<F: CanonicalField>(len: usize) -> (Vec<F>, Vec<F>) {
    let mut lhs = Vec::with_capacity(len);
    let mut rhs = Vec::with_capacity(len);
    let mut state = 0x6a09_e667_f3bc_c908_bb67_ae85_84ca_a73bu128;

    for i in 0..len {
        state = state
            .wrapping_mul(0x94d0_49bb_1331_11eb_dbe6_d5d5_fe4c_ce2f)
            .wrapping_add(i as u128 + 0x9e37_79b9);
        lhs.push(F::from_canonical_u128_reduced(state));
        rhs.push(F::from_canonical_u128_reduced(
            state.rotate_left(41) ^ i as u128,
        ));
    }

    (lhs, rhs)
}

fn cpu_vector_op<F>(op: Fp128VectorOp, lhs: &[F], rhs: &[F], out: &mut [F])
where
    F: Copy + core::ops::Add<Output = F> + core::ops::Sub<Output = F> + core::ops::Mul<Output = F>,
{
    for ((out, &lhs), &rhs) in out.iter_mut().zip(lhs).zip(rhs) {
        *out = match op {
            Fp128VectorOp::Add => lhs + rhs,
            Fp128VectorOp::Sub => lhs - rhs,
            Fp128VectorOp::Mul => lhs * rhs,
        };
    }
}

fn op_name(op: Fp128VectorOp) -> &'static str {
    match op {
        Fp128VectorOp::Add => "add",
        Fp128VectorOp::Sub => "sub",
        Fp128VectorOp::Mul => "mul",
    }
}

fn bench_prime128_a7f7(c: &mut Criterion) {
    type F = Prime128OffsetA7F7;

    let mut group = c.benchmark_group("metal/fp128_vector/prime128_offset_a7f7");
    for len in [1 << 12, 1 << 16, 1 << 20] {
        let (lhs, rhs) = sample_inputs::<F>(len);
        let mut cpu_out = vec![F::zero(); len];
        group.throughput(Throughput::Elements(len as u64));

        for op in [Fp128VectorOp::Add, Fp128VectorOp::Sub, Fp128VectorOp::Mul] {
            group.bench_with_input(
                BenchmarkId::new(format!("cpu_{}", op_name(op)), len),
                &len,
                |bench, _| {
                    bench.iter(|| {
                        cpu_vector_op(
                            op,
                            black_box(&lhs),
                            black_box(&rhs),
                            black_box(&mut cpu_out),
                        );
                        black_box(cpu_out[0]);
                    });
                },
            );
        }

        let Ok(backend) = MetalBackend::new() else {
            eprintln!("skipping Metal benchmarks: no usable Metal backend");
            continue;
        };

        for op in [Fp128VectorOp::Add, Fp128VectorOp::Sub, Fp128VectorOp::Mul] {
            let mut buffers = backend.create_fp128_vector_buffers::<P_A7F7>(len).unwrap();
            backend
                .upload_fp128_vector_inputs(&mut buffers, &lhs, &rhs)
                .unwrap();
            backend.dispatch_fp128_vector(op, &buffers).unwrap();
            let mut metal_out = vec![F::zero(); len];
            backend
                .read_fp128_vector_output_into(&buffers, &mut metal_out)
                .unwrap();
            cpu_vector_op(op, &lhs, &rhs, &mut cpu_out);
            assert_eq!(metal_out, cpu_out);

            group.bench_with_input(
                BenchmarkId::new(format!("metal_{}_dispatch_only", op_name(op)), len),
                &len,
                |bench, _| {
                    bench.iter(|| {
                        backend
                            .dispatch_fp128_vector(black_box(op), black_box(&buffers))
                            .unwrap();
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("metal_{}_roundtrip_reuse_buffers", op_name(op)),
                    len,
                ),
                &len,
                |bench, _| {
                    bench.iter(|| {
                        backend
                            .upload_fp128_vector_inputs(
                                black_box(&mut buffers),
                                black_box(&lhs),
                                black_box(&rhs),
                            )
                            .unwrap();
                        backend
                            .dispatch_fp128_vector(black_box(op), black_box(&buffers))
                            .unwrap();
                        backend
                            .read_fp128_vector_output_into(
                                black_box(&buffers),
                                black_box(&mut metal_out),
                            )
                            .unwrap();
                        black_box(metal_out[0]);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(fp128_vector, bench_prime128_a7f7);
criterion_main!(fp128_vector);
