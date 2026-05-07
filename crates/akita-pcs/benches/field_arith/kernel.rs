use akita_field::{CanonicalField, FieldCore, PackedField, Prime128Offset275, RandomSampling};
use criterion::{black_box, Criterion, Throughput};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use super::cases::*;
use super::data::rand_u128;

pub(crate) fn bench_kernel_patterns(c: &mut Criterion) {
    bench_packed_sumcheck_mix(c);
    bench_fp128_accumulator_pattern(c);
}

fn bench_packed_sumcheck_mix(c: &mut Criterion) {
    let n = 4096u64;
    let mut rng = StdRng::seed_from_u64(0x5151_cafe);

    let mut group = c.benchmark_group("field_arith/kernel/packed_macc");
    group.throughput(Throughput::Elements(n));

    use akita_field::fields::pseudo_mersenne::*;

    sumcheck_bench::<Pow2Offset31Field, P31>(&mut group, FP32_31B, &mut rng, n);
    sumcheck_bench::<M31, PM31>(&mut group, FP32_M31, &mut rng, n);
    sumcheck_bench::<Pow2Offset32Field, P32>(&mut group, FP32_32B, &mut rng, n);
    sumcheck_bench::<Pow2Offset40Field, P40>(&mut group, FP64_40B, &mut rng, n);
    sumcheck_bench::<Pow2Offset64Field, P64>(&mut group, FP64_64B, &mut rng, n);
    sumcheck_bench::<F128, P128>(&mut group, FP128, &mut rng, n);

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
