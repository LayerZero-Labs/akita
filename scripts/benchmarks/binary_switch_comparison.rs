//! External F128 switch comparison. Built by bench-binary-switch-comparison.py.
use akita_algebra::binary::field_switch::{
    batched_weights, partial_evaluations, transparent_weight,
};
use akita_algebra::binary::{BinaryField128 as H, BinaryField162 as F, PackedBinary162};
use labinius::fields::crossfield::{self as reference, SwitchProver, Workspace};
use labinius::{B128, F162 as RF};
use std::{hint::black_box, time::Instant};

fn measure_pair(mut ours: impl FnMut(), mut reference: impl FnMut(), target_ms: u64) {
    fn calibrate(f: &mut impl FnMut(), target_ms: u64) -> usize {
        for _ in 0..3 {
            f();
        }
        let start = Instant::now();
        for _ in 0..5 {
            f();
        }
        ((target_ms as f64 * 0.001 / (start.elapsed().as_secs_f64() / 5.0)) as usize)
            .clamp(1, 100_000)
    }
    fn sample(f: &mut impl FnMut(), loops: usize) -> f64 {
        let start = Instant::now();
        for _ in 0..loops {
            f();
        }
        start.elapsed().as_secs_f64() * 1e6 / loops as f64
    }
    let a_loops = calibrate(&mut ours, target_ms);
    let b_loops = calibrate(&mut reference, target_ms);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..21 {
        if i % 2 == 0 {
            a.push(sample(&mut ours, a_loops));
            b.push(sample(&mut reference, b_loops));
        } else {
            b.push(sample(&mut reference, b_loops));
            a.push(sample(&mut ours, a_loops));
        }
    }
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    println!(
        "Akita {:.3} [{:.3},{:.3}]; LaBinius {:.3} [{:.3},{:.3}]; ratio={:.3}",
        a[10],
        a[2],
        a[18],
        b[10],
        b[2],
        b[18],
        a[10] / b[10]
    );
}
fn rf(x: F) -> RF {
    RF(x.to_words())
}
fn next(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn rounds(a: &mut PackedBinary162, b: &mut PackedBinary162, mut h: F, z: &[F]) -> F {
    for &r in z {
        let [c0, c1, c2] = a.round_product(b, h).unwrap();
        h = c0 + r * (c1 + r * c2);
        a.fold_in_place(r);
        b.fold_in_place(r);
    }
    black_box((h, a.get(0), b.get(0))).0
}
fn main() {
    assert!(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("gfni"));
    println!("F128/F128, one core, 21 samples; us median [p10,p90]. Initial batched claim/challenges given to both; warm buffers. Reference raw partial orientation retained during timing.");
    for bits in [8, 12, 16, 18] {
        let n = 1usize << bits;
        let mut rng = 0x6371_994f_bc81_678du64;
        let source: Vec<u128> = (0..n)
            .map(|_| u128::from(next(&mut rng)) | (u128::from(next(&mut rng)) << 64))
            .collect();
        let point: Vec<_> = (0..bits)
            .map(|_| H::from_words([next(&mut rng), next(&mut rng)]))
            .collect();
        let batch: Vec<_> = (0..7)
            .map(|_| {
                F::from_words([
                    next(&mut rng),
                    next(&mut rng),
                    next(&mut rng) & 0x3_ffff_ffff,
                ])
                .unwrap()
            })
            .collect();
        let z: Vec<_> = (0..bits)
            .map(|_| {
                F::from_words([
                    next(&mut rng),
                    next(&mut rng),
                    next(&mut rng) & 0x3_ffff_ffff,
                ])
                .unwrap()
            })
            .collect();
        // Half-folding on bit-reversed source matches adjacent folding. Host
        // point and folding challenges retain the same coordinate order.
        let ref_source: Vec<_> = (0..n)
            .map(|j| B128(source[j.reverse_bits() >> (usize::BITS - bits)]))
            .collect();
        let ref_point: Vec<_> = point
            .iter()
            .map(|x| {
                let [l, h] = x.to_words();
                B128(u128::from(l) | (u128::from(h) << 64))
            })
            .collect();
        let ref_batch_point: Vec<_> = batch.iter().rev().copied().map(rf).collect();
        let ref_batch = reference::eq_expand_f162(&ref_batch_point);
        let mut eq = Vec::new();
        let partials = partial_evaluations::<H>(&source, &point, &mut eq).unwrap();
        let claim = partials.batch(&batch).unwrap();
        let mut workspace = Workspace::new(bits as usize, 1);
        let ref_partials = workspace.partial_evals(0, &ref_source, &ref_point);
        // Compare the whole matrix, not merely the reconstructed evaluation.
        for k in 0..128 {
            let p = ref_partials
                .iter()
                .enumerate()
                .fold(0u128, |p, (q, v)| p | (((v.0 >> k) & 1) << q));
            assert_eq!(partials.values()[k], p);
        }
        assert_eq!(rf(claim), reference::slice_sum(&ref_partials, &ref_batch));
        let mut a = PackedBinary162::new();
        a.refill_binary_words(&source);
        let mut b = PackedBinary162::new();
        batched_weights(&eq, &batch, &mut b).unwrap();
        let mut prover =
            SwitchProver::from_workspace(workspace, &ref_source, &[RF::ONE], &ref_batch);
        let mut h = claim;
        for &r in &z {
            let [c0, c1, c2] = a.round_product(&b, h).unwrap();
            assert_eq!(prover.msg(), [rf(c0), rf(c2)]);
            h = c0 + r * (c1 + r * c2);
            a.fold_in_place(r);
            b.fold_in_place(r);
            prover.fold(rf(r));
        }
        assert_eq!(prover.final_eval(), rf(a.get(0).unwrap()));
        let rz: Vec<_> = z.iter().copied().map(rf).collect();
        assert_eq!(
            reference::transparent_coeff(&ref_point, &rz, &ref_batch),
            rf(transparent_weight(&point, &z, &batch).unwrap())
        );
        let mut ws = Some(prover.into_workspace());
        print!("N={n} partials ");
        measure_pair(
            || {
                black_box(
                    partial_evaluations::<H>(black_box(&source), black_box(&point), &mut eq)
                        .unwrap(),
                );
            },
            || {
                black_box(ws.as_mut().unwrap().partial_evals(
                    0,
                    black_box(&ref_source),
                    black_box(&ref_point),
                ));
            },
            30,
        );
        print!("N={n} complete ");
        measure_pair(
            || {
                black_box(
                    partial_evaluations::<H>(black_box(&source), black_box(&point), &mut eq)
                        .unwrap(),
                );
                batched_weights(&eq, black_box(&batch), &mut b).unwrap();
                a.refill_binary_words(black_box(&source));
                black_box(rounds(&mut a, &mut b, black_box(claim), black_box(&z)));
            },
            || {
                let mut workspace = ws.take().unwrap();
                black_box(workspace.partial_evals(
                    0,
                    black_box(&ref_source),
                    black_box(&ref_point),
                ));
                let expanded = reference::eq_expand_f162(black_box(&ref_batch_point));
                let mut prover = SwitchProver::from_workspace(
                    workspace,
                    black_box(&ref_source),
                    &[RF::ONE],
                    black_box(&expanded),
                );
                for &r in black_box(&rz) {
                    black_box(prover.msg());
                    prover.fold(r);
                }
                black_box(prover.final_eval());
                ws = Some(prover.into_workspace());
            },
            30,
        );
        print!("N={n} coefficient verifier ");
        measure_pair(
            || {
                black_box(
                    transparent_weight(black_box(&point), black_box(&z), black_box(&batch))
                        .unwrap(),
                );
            },
            || {
                let expanded = reference::eq_expand_f162(black_box(&ref_batch_point));
                black_box(reference::transparent_coeff(
                    black_box(&ref_point),
                    black_box(&rz),
                    black_box(&expanded),
                ));
            },
            20,
        );
    }
}
