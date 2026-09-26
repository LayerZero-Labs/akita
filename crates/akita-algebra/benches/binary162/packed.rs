use std::hint::black_box;

use akita_algebra::binary::{BinaryField162 as F, PackedBinary162};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};

// The existing AoS API can defer reduction for each of the two dot products.
// Reuse gathering scratch so this baseline does not pay for fresh allocations.
#[derive(Clone, Default)]
struct AosRoundScratch {
    left: [Vec<F>; 2],
    right: [Vec<F>; 2],
}

impl AosRoundScratch {
    fn round(&mut self, lhs: &[F], rhs: &[F], claim: F) -> [F; 3] {
        for (input, columns) in [(lhs, &mut self.left), (rhs, &mut self.right)] {
            for column in columns.iter_mut() {
                column.clear();
            }
            for pair in input.chunks(2) {
                let a = pair[0];
                let b = pair.get(1).copied().unwrap_or(F::ZERO);
                columns[0].push(a);
                columns[1].push(a + b);
            }
        }
        let c0 = F::dot_product(&self.left[0], &self.right[0]).unwrap();
        let c2 = F::dot_product(&self.left[1], &self.right[1]).unwrap();
        [c0, claim + c2, c2]
    }
}

fn aos_fold(values: &mut Vec<F>, r: F) {
    let next_len = values.len().div_ceil(2);
    for i in 0..next_len {
        let a = values[2 * i];
        let b = values.get(2 * i + 1).copied().unwrap_or(F::ZERO);
        values[i] = a + r * (a + b);
    }
    values.truncate(next_len);
}

fn packed_rounds(lhs: &mut PackedBinary162, rhs: &mut PackedBinary162, mut claim: F, r: F) -> F {
    let mut digest = F::ZERO;
    while lhs.len() > 1 {
        let [c0, c1, c2] = lhs.round_product(rhs, claim).unwrap();
        claim = black_box(c0 + r * (c1 + r * c2));
        digest += claim;
        lhs.fold_in_place(r);
        rhs.fold_in_place(r);
    }
    black_box((digest, lhs.get(0), rhs.get(0))).0
}

fn aos_rounds(
    lhs: &mut Vec<F>,
    rhs: &mut Vec<F>,
    scratch: &mut AosRoundScratch,
    mut claim: F,
    r: F,
) -> F {
    let mut digest = F::ZERO;
    while lhs.len() > 1 {
        let [c0, c1, c2] = scratch.round(lhs, rhs, claim);
        claim = black_box(c0 + r * (c1 + r * c2));
        digest += claim;
        aos_fold(lhs, r);
        aos_fold(rhs, r);
    }
    black_box((digest, lhs.first(), rhs.first())).0
}

pub(super) fn bench(c: &mut Criterion) {
    let r = F::from_words([0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210, 0x12345]).unwrap();
    let mut group = c.benchmark_group("binary162_packed");
    for len in [256, 4096, 65536, 262144] {
        let (lhs, rhs) = super::inputs(len);
        let claim = F::dot_product(&lhs, &rhs).unwrap();
        let packed_lhs = PackedBinary162::from_scalars(&lhs);
        let packed_rhs = PackedBinary162::from_scalars(&rhs);
        let mut scratch = AosRoundScratch::default();
        // Both correctness checks and scratch warmup are outside the timers.
        assert_eq!(
            packed_lhs.round_product(&packed_rhs, claim).unwrap(),
            scratch.round(&lhs, &rhs, claim)
        );
        assert_eq!(
            packed_rounds(&mut packed_lhs.clone(), &mut packed_rhs.clone(), claim, r),
            aos_rounds(&mut lhs.clone(), &mut rhs.clone(), &mut scratch, claim, r),
        );
        scratch.round(&lhs, &rhs, claim);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("message_packed", len), &len, |b, _| {
            b.iter(|| black_box(packed_lhs.round_product(black_box(&packed_rhs), black_box(claim))))
        });
        group.bench_with_input(
            BenchmarkId::new("message_aos_deferred", len),
            &len,
            |b, _| {
                b.iter(|| {
                    black_box(scratch.round(black_box(&lhs), black_box(&rhs), black_box(claim)))
                })
            },
        );
        group.bench_with_input(BenchmarkId::new("all_rounds_packed", len), &len, |b, _| {
            b.iter_batched_ref(
                || (packed_lhs.clone(), packed_rhs.clone()),
                |(a, b)| black_box(packed_rounds(a, b, black_box(claim), black_box(r))),
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(
            BenchmarkId::new("all_rounds_aos_deferred", len),
            &len,
            |b, _| {
                b.iter_batched_ref(
                    || (lhs.clone(), rhs.clone(), scratch.clone()),
                    |(a, b, scratch)| {
                        black_box(aos_rounds(a, b, scratch, black_box(claim), black_box(r)))
                    },
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("convert_and_all_rounds", len),
            &len,
            |b, _| {
                let (mut a, mut b_buffer) = (packed_lhs.clone(), packed_rhs.clone());
                b.iter(|| {
                    a.refill(black_box(&lhs));
                    b_buffer.refill(black_box(&rhs));
                    black_box(packed_rounds(
                        &mut a,
                        &mut b_buffer,
                        black_box(claim),
                        black_box(r),
                    ))
                })
            },
        );
        group.bench_with_input(BenchmarkId::new("convert_reused", len), &len, |b, _| {
            let mut buffer = packed_lhs.clone();
            b.iter(|| {
                buffer.refill(black_box(&lhs));
                black_box(&buffer);
            })
        });
        group.bench_with_input(BenchmarkId::new("convert_allocate", len), &len, |b, _| {
            b.iter(|| black_box(PackedBinary162::from_scalars(black_box(&lhs))))
        });
    }
    group.finish();
}
