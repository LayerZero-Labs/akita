//! One-thread arithmetic-only commitment columns; plans and matrix transforms are untimed.

use std::hint::black_box;
use std::time::Duration;

use akita_algebra::{
    Field, MinusTrinomial, TrinomialLimbAccumulator, TrinomialLimbDomain, TrinomialLimbSlots,
    TrinomialNttDomain, TrinomialRing, TrinomialWideLimbAccumulator, TrinomialWideLimbDomain,
    TrinomialWideLimbSlots,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::Prime128OffsetA7F7;
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

const D: usize = 648;
const M: usize = 4096;
const WORDS: usize = D.div_ceil(64);
const BITS: u64 = (M * D) as u64;
const MAX_ROWS: usize = 4;
const SEED: u64 = 0x215_197;

type Bits = [u64; WORDS];
type BaselineDomain = TrinomialNttDomain<Prime128OffsetA7F7, D, MinusTrinomial>;
type BaselineRing = TrinomialRing<Prime128OffsetA7F7, D, MinusTrinomial>;

struct LimbColumn {
    domain: TrinomialLimbDomain,
    matrix: Vec<TrinomialLimbSlots>,
    transformed: Vec<TrinomialLimbSlots>,
    source_scratch: TrinomialLimbSlots,
    accumulators: Vec<TrinomialLimbAccumulator>,
    sums: Vec<TrinomialLimbSlots>,
    outputs: [[u16; D]; MAX_ROWS],
}

impl LimbColumn {
    fn prepare(prime: u16, source: &[Bits], rng: &mut StdRng) -> Self {
        let domain = TrinomialLimbDomain::new(prime).expect("admitted benchmark prime");
        let matrix = (0..MAX_ROWS * M)
            .map(|_| {
                let coefficients = std::array::from_fn::<_, D, _>(|_| rng.gen_range(0..prime));
                let mut slots = domain.zero_slots();
                domain
                    .forward_canonical(&coefficients, &mut slots)
                    .expect("canonical benchmark matrix");
                slots
            })
            .collect();
        let transformed = source
            .iter()
            .map(|bits| {
                let mut slots = domain.zero_slots();
                domain
                    .forward_bits(bits, &mut slots)
                    .expect("valid benchmark bits");
                slots
            })
            .collect();
        let source_scratch = domain.zero_slots();
        let accumulators = (0..MAX_ROWS)
            .map(|_| TrinomialLimbAccumulator::new(&domain))
            .collect();
        let sums = vec![domain.zero_slots(); MAX_ROWS];
        Self {
            domain,
            matrix,
            transformed,
            source_scratch,
            accumulators,
            sums,
            outputs: [[0; D]; MAX_ROWS],
        }
    }

    fn commit(&mut self, source: &[Bits], rows: usize) {
        for (column, bits) in source.iter().enumerate() {
            self.domain
                .forward_bits(bits, &mut self.source_scratch)
                .expect("valid benchmark bits");
            for (accumulator, matrix_row) in self.accumulators[..rows]
                .iter_mut()
                .zip(self.matrix.chunks_exact(M))
            {
                accumulator
                    .add_product(&matrix_row[column], &self.source_scratch)
                    .expect("matching limb prime");
            }
        }
        for ((accumulator, sum), output) in self.accumulators[..rows]
            .iter_mut()
            .zip(&mut self.sums)
            .zip(&mut self.outputs)
        {
            accumulator.finish(sum).expect("matching limb prime");
            self.domain
                .inverse(sum, output)
                .expect("matching limb prime and output geometry");
        }
        black_box(&self.outputs);
    }
}

fn limb_benchmarks(criterion: &mut Criterion, source: &[Bits]) {
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let mut columns = TrinomialLimbDomain::ADMITTED_PRIMES
        .map(|prime| LimbColumn::prepare(prime, source, &mut rng));
    let mut group = criterion.benchmark_group("trinomial_limb/commit_column");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(BITS));
    for limbs in 1..=4 {
        for rows in [1, 2, 4] {
            let parameter = format!("k={limbs}/n_a={rows}");
            group.bench_with_input(
                BenchmarkId::new("limb", &parameter),
                &(limbs, rows),
                |b, &(limbs, rows)| {
                    b.iter(|| {
                        for column in &mut columns[..limbs] {
                            column.commit(black_box(source), rows);
                        }
                    });
                },
            );
        }
    }
    group.finish();

    let column = &mut columns[0];
    let mut group = criterion.benchmark_group("trinomial_limb/phases/k=1/n_a=1");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(BITS));
    group.bench_function("forward_bits", |b| {
        b.iter(|| {
            for (bits, slots) in black_box(source).iter().zip(&mut column.transformed) {
                column
                    .domain
                    .forward_bits(bits, slots)
                    .expect("valid benchmark bits");
            }
            black_box(&column.transformed);
        });
    });
    group.bench_function("accumulate", |b| {
        b.iter(|| {
            for (matrix, source) in black_box(&column.matrix[..M])
                .iter()
                .zip(black_box(&column.transformed))
            {
                column.accumulators[0]
                    .add_product(matrix, source)
                    .expect("matching limb prime");
            }
            column.accumulators[0]
                .finish(&mut column.sums[0])
                .expect("matching limb prime");
            black_box(&column.sums[0]);
        });
    });
    // Prepare one nonzero sum outside the inverse phase's timer, even when
    // Criterion's filter excludes the accumulate measurement above.
    for (matrix, source) in column.matrix[..M].iter().zip(&column.transformed) {
        column.accumulators[0]
            .add_product(matrix, source)
            .expect("matching limb prime");
    }
    column.accumulators[0]
        .finish(&mut column.sums[0])
        .expect("matching limb prime");
    group.bench_function("inverse", |b| {
        b.iter(|| {
            column
                .domain
                .inverse(black_box(&column.sums[0]), &mut column.outputs[0])
                .expect("matching limb prime and output geometry");
            black_box(&column.outputs[0]);
        });
    });
    group.finish();
}

fn baseline_benchmark(criterion: &mut Criterion, source: &[Bits]) {
    let mut rng = StdRng::seed_from_u64(SEED + 2);
    let domain = BaselineDomain::new().expect("fully splitting baseline field");
    let lut = domain.prepare_i8_lut(2).expect("table admits zero and one");
    let mut workspace = domain.workspace();
    let matrix = (0..M)
        .map(|_| {
            let coefficients = std::array::from_fn(|_| Prime128OffsetA7F7::random(&mut rng));
            let ring =
                BaselineRing::from_coefficients(coefficients).expect("valid benchmark ring degree");
            domain.forward_with_workspace(&ring, &mut workspace)
        })
        .collect::<Vec<_>>();
    let zero = domain.zero_ntt();
    let mut transformed = zero.clone();
    let mut accumulator = zero.clone();
    let mut group = criterion.benchmark_group("trinomial_limb/commit_column");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(BITS));
    group.bench_function("p128_a7f7/n_a=1", |b| {
        b.iter(|| {
            accumulator.clone_from(&zero);
            for (entry, bits) in black_box(&matrix).iter().zip(black_box(source)) {
                let input =
                    std::array::from_fn::<_, D, _>(|i| ((bits[i / 64] >> (i % 64)) & 1) as i8);
                domain
                    .forward_i8_with_lut_into_workspace(
                        &input,
                        &lut,
                        &mut transformed,
                        &mut workspace,
                    )
                    .expect("benchmark bits lie in lookup table range");
                accumulator.add_assign_pointwise_mul_packed(entry, &transformed);
            }
            black_box(domain.inverse_with_workspace(&accumulator, &mut workspace));
        });
    });
    group.finish();
}

struct WideColumn {
    domain: TrinomialWideLimbDomain,
    matrix: Vec<TrinomialWideLimbSlots>,
    transformed: Vec<TrinomialWideLimbSlots>,
    source_scratch: TrinomialWideLimbSlots,
    accumulators: Vec<TrinomialWideLimbAccumulator>,
    sums: Vec<TrinomialWideLimbSlots>,
    outputs: [[i32; D]; MAX_ROWS],
}

impl WideColumn {
    fn prepare(prime: u32, source: &[Bits], rng: &mut StdRng) -> Self {
        let domain = TrinomialWideLimbDomain::new(prime).expect("admitted benchmark prime");
        let half = (prime / 2) as i32;
        let matrix = (0..MAX_ROWS * M)
            .map(|_| {
                let coefficients = std::array::from_fn::<_, D, _>(|_| rng.gen_range(-half..=half));
                let mut slots = domain.zero_slots();
                domain
                    .forward_centered(&coefficients, &mut slots)
                    .expect("centered benchmark matrix");
                slots
            })
            .collect();
        let transformed = source
            .iter()
            .map(|bits| {
                let mut slots = domain.zero_slots();
                domain
                    .forward_bits(bits, &mut slots)
                    .expect("valid benchmark bits");
                slots
            })
            .collect();
        let source_scratch = domain.zero_slots();
        let accumulators = (0..MAX_ROWS)
            .map(|_| TrinomialWideLimbAccumulator::new(&domain))
            .collect();
        let sums = vec![domain.zero_slots(); MAX_ROWS];
        Self {
            domain,
            matrix,
            transformed,
            source_scratch,
            accumulators,
            sums,
            outputs: [[0; D]; MAX_ROWS],
        }
    }

    fn commit(&mut self, source: &[Bits], rows: usize) {
        for (column, bits) in source.iter().enumerate() {
            self.domain
                .forward_bits(bits, &mut self.source_scratch)
                .expect("valid benchmark bits");
            for (accumulator, matrix_row) in self.accumulators[..rows]
                .iter_mut()
                .zip(self.matrix.chunks_exact(M))
            {
                accumulator
                    .add_product(&matrix_row[column], &self.source_scratch)
                    .expect("matching wide limb prime");
            }
        }
        for ((accumulator, sum), output) in self.accumulators[..rows]
            .iter_mut()
            .zip(&mut self.sums)
            .zip(&mut self.outputs)
        {
            accumulator.finish(sum).expect("matching wide limb prime");
            self.domain
                .inverse_centered(sum, output)
                .expect("matching wide limb prime and output geometry");
        }
        black_box(&self.outputs);
    }
}

fn wide_benchmarks(criterion: &mut Criterion, source: &[Bits]) {
    let mut rng = StdRng::seed_from_u64(SEED + 3);
    let mut columns = TrinomialWideLimbDomain::ADMITTED_PRIMES
        .map(|prime| WideColumn::prepare(prime, source, &mut rng));
    let mut group = criterion.benchmark_group("trinomial_limb/wide/commit_column");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(BITS));
    for column in &mut columns {
        for rows in 1..=MAX_ROWS {
            if column.domain.prime() != 268_433_353 && rows != 3 {
                continue;
            }
            let parameter = format!("p={}/n_a={rows}", column.domain.prime());
            group.bench_with_input(BenchmarkId::new("wide", parameter), &rows, |b, &rows| {
                b.iter(|| column.commit(black_box(source), rows))
            });
        }
    }
    group.finish();

    let column = &mut columns[2];
    let mut group = criterion.benchmark_group("trinomial_limb/wide/phases/p=268433353");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(BITS));
    group.bench_function("forward_bits", |b| {
        b.iter(|| {
            for (bits, slots) in black_box(source).iter().zip(&mut column.transformed) {
                column
                    .domain
                    .forward_bits(bits, slots)
                    .expect("valid benchmark bits");
            }
            black_box(&column.transformed);
        });
    });
    for (name, variant) in [
        ("forward_one_table", (false, true, true)),
        ("forward_loop_gather", (true, false, true)),
        ("forward_generic_last", (true, true, false)),
    ] {
        group.bench_with_input(name, &variant, |b, &(fused, transpose, unroll)| {
            b.iter(|| {
                for (bits, slots) in black_box(source).iter().zip(&mut column.transformed) {
                    column
                        .domain
                        .forward_bits_variant(bits, slots, fused, transpose, unroll)
                        .expect("valid benchmark bits");
                }
                black_box(&column.transformed);
            });
        });
    }
    group.bench_function("forward_corrected", |b| {
        b.iter(|| {
            for (bits, slots) in black_box(source).iter().zip(&mut column.transformed) {
                column
                    .domain
                    .forward_bits_corrected(bits, slots)
                    .expect("valid benchmark bits");
            }
            black_box(&column.transformed);
        });
    });
    group.bench_function("accumulate", |b| {
        b.iter(|| {
            for (matrix, source) in black_box(&column.matrix[..M])
                .iter()
                .zip(black_box(&column.transformed))
            {
                column.accumulators[0]
                    .add_product(matrix, source)
                    .expect("matching wide limb prime");
            }
            column.accumulators[0]
                .finish(&mut column.sums[0])
                .expect("matching wide limb prime");
            black_box(&column.sums[0]);
        });
    });
    // Keep inverse input nonzero when a Criterion filter excludes accumulation.
    for (matrix, source) in column.matrix[..M].iter().zip(&column.transformed) {
        column.accumulators[0]
            .add_product(matrix, source)
            .expect("matching wide limb prime");
    }
    column.accumulators[0]
        .finish(&mut column.sums[0])
        .expect("matching wide limb prime");
    group.bench_function("inverse", |b| {
        b.iter(|| {
            column
                .domain
                .inverse_centered(black_box(&column.sums[0]), &mut column.outputs[0])
                .expect("matching wide limb prime and output geometry");
            black_box(&column.outputs[0]);
        });
    });
    let mut gathered = [0u8; 81];
    group.bench_function("bit_gather", |b| {
        b.iter(|| {
            for bits in black_box(source) {
                TrinomialWideLimbDomain::gather_bits(bits, &mut gathered)
                    .expect("valid benchmark bits");
                black_box(&gathered);
            }
        });
    });
    group.finish();
}

fn benchmarks(criterion: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let source = (0..M)
        .map(|_| {
            let mut bits = std::array::from_fn::<_, WORDS, _>(|_| rng.next_u64());
            bits[WORDS - 1] &= (1u64 << (D % 64)) - 1;
            bits
        })
        .collect::<Vec<_>>();
    eprintln!("trinomial_limb: one thread, m={M}, D={D}, committed bits={BITS}, seed={SEED}");
    limb_benchmarks(criterion, &source);
    baseline_benchmark(criterion, &source);
    wide_benchmarks(criterion, &source);
}

criterion_group!(trinomial_limb, benchmarks);
criterion_main!(trinomial_limb);
