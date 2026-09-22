use std::hint::black_box;

use akita_algebra::binary::{
    field_switch::{
        batched_weights, embed_source, partial_evaluations, transparent_weight, SwitchField,
    },
    BinaryField128 as H128, BinaryField162 as F, BinaryField192 as H192, PackedBinary162,
};
use criterion::{BatchSize, BenchmarkId, Criterion};

fn all_rounds(a: &mut PackedBinary162, b: &mut PackedBinary162, mut claim: F, r: F) -> F {
    while a.len() > 1 {
        let [c0, c1, c2] = a.round_product(b, claim).unwrap();
        claim = black_box(c0 + r * (c1 + r * c2));
        a.fold_in_place(r);
        b.fold_in_place(r);
    }
    black_box((claim, a.get(0), b.get(0))).0
}

fn profile<H: SwitchField>(c: &mut Criterion, name: &str, source: &[H::Source], point: &[H]) {
    let r = F::from_words([0xb415_ee83_b074_d978, 0x164f_ab70_9546_ce18, 0x2_abca_9567]).unwrap();
    let batch = vec![r; H::BATCH_BITS];
    let mut group = c.benchmark_group(format!("binary_field_switch/{name}"));
    for bits in [8, 12, 16] {
        let n = 1 << bits;
        let source = &source[..n];
        let point = &point[..bits];
        let z = vec![r; bits];
        let mut equality = Vec::new();
        let partials = partial_evaluations::<H>(source, point, &mut equality).unwrap();
        let mut weights = Vec::new();
        batched_weights(&equality, &batch, &mut weights).unwrap();
        let mut source_f: Vec<_> = source.iter().copied().map(embed_source::<H>).collect();
        let a = PackedBinary162::from_scalars(&source_f);
        let b = PackedBinary162::from_scalars(&weights);
        let claim = partials.batch(&batch).unwrap();
        assert_eq!(F::dot_product(&source_f, &weights), Some(claim));
        let mut folded_a = a.clone();
        let mut folded_b = b.clone();
        let terminal = all_rounds(&mut folded_a, &mut folded_b, claim, r);
        let terminal_weight = transparent_weight(point, &z, &batch).unwrap();
        assert_eq!(folded_b.get(0), Some(terminal_weight));
        assert_eq!(terminal, folded_a.get(0).unwrap() * terminal_weight);

        group.bench_with_input(BenchmarkId::new("partials_reused", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(
                    partial_evaluations::<H>(black_box(source), black_box(point), &mut equality)
                        .unwrap(),
                )
            })
        });
        group.bench_with_input(
            BenchmarkId::new("batch_weights_reused", n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    batched_weights(black_box(&equality), black_box(&batch), &mut weights).unwrap();
                    black_box(&weights);
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("all_rounds_prepared", n),
            &n,
            |bench, _| {
                bench.iter_batched_ref(
                    || (a.clone(), b.clone()),
                    |(a, b)| black_box(all_rounds(a, b, black_box(claim), black_box(r))),
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("switch_and_all_rounds_reused", n),
            &n,
            |bench, _| {
                let (mut a, mut b) = (a.clone(), b.clone());
                bench.iter(|| {
                    let partials = partial_evaluations::<H>(
                        black_box(source),
                        black_box(point),
                        &mut equality,
                    )
                    .unwrap();
                    batched_weights(&equality, black_box(&batch), &mut weights).unwrap();
                    let claim = partials.batch(&batch).unwrap();
                    source_f.clear();
                    source_f.extend(source.iter().copied().map(embed_source::<H>));
                    a.refill(&source_f);
                    b.refill(&weights);
                    black_box(all_rounds(&mut a, &mut b, claim, black_box(r)))
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("transparent_verifier", n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    black_box(
                        transparent_weight(black_box(point), black_box(&z), black_box(&batch))
                            .unwrap(),
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("host_reconstruction", n),
            &n,
            |bench, _| bench.iter(|| black_box(black_box(&partials).reconstruct())),
        );
    }
    group.finish();
}

pub(super) fn bench(c: &mut Criterion) {
    let mut state = 0x6284_dda7_391b_adefu64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let source64: Vec<_> = (0..1 << 16).map(|_| next()).collect();
    let source128: Vec<_> = (0..1 << 16)
        .map(|_| u128::from(next()) | (u128::from(next()) << 64))
        .collect();
    let point128: Vec<_> = (0..16)
        .map(|_| H128::from_words([next(), next()]))
        .collect();
    let point192: Vec<_> = (0..16)
        .map(|_| H192::from_words([next(), next(), next()]))
        .collect();
    profile::<H128>(c, "f128_f128", &source128, &point128);
    profile::<H192>(c, "f64_f192", &source64, &point192);
}
