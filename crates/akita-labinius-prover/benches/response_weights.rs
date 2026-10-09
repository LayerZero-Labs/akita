//! Transform-adjoint response weights at the first profile's full row length.

#![cfg(feature = "labinius")]

use akita_algebra::{binary::BinaryField162 as B, MinusTrinomial, TrinomialRing};
use akita_challenges::{BinaryChallengeSampler, FoldDraw};
use akita_error::AkitaError;
use akita_labinius_prover::{response_weights::coefficient_weights, PreparedCommitMatrix};
use akita_labinius_verifier::{
    lowered::{LoweredChallenges, LoweredPublic, LoweredRootLayout},
    AdmittedRootSetup, BinaryEvaluationClaim,
};
use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootProfile};
use akita_types::proof::AkitaSetupSeed;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use jolt_field::{Prime128OffsetA7F7 as F, Ring, Zero};
use std::{hint::black_box, time::Duration};

struct Draw;
impl FoldDraw for Draw {
    fn absorb_and_squeeze(&mut self, _payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok([0x53; 32])
    }
}

fn benchmarks(criterion: &mut Criterion) {
    let profile = LabiniusRootProfile::D648P128BoundedW46Delta16;
    let admitted = AdmittedRootSetup::<F, 648, MinusTrinomial>::derive(
        profile,
        22,
        8,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .expect("first profile admits");
    let setup = admitted.setup();
    let prepared = PreparedCommitMatrix::prepare(setup).expect("matrix prepares");
    let layout = LoweredRootLayout::new(setup, admitted.shape(), LabiniusDigitBase::Bits2)
        .expect("layout admits");
    let fold = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut Draw, b"weights", setup.columns())
        .expect("fold samples");
    let claim = BinaryEvaluationClaim {
        point: vec![B::ZERO; setup.num_vars()],
        value: B::ZERO,
    };
    let public = LoweredPublic::new(
        &layout,
        setup,
        &claim,
        &vec![B::ZERO; setup.columns()],
        &fold,
        &[],
        &[0; 161],
        &[0; 162],
        LoweredChallenges {
            alpha: F::from_u64(7),
            xi: F::from_u64(11),
            gamma: F::from_u64(13),
        },
    )
    .expect("public builds");
    let mut group = criterion.benchmark_group("response_weights/p128/minus/D648/m4096");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((setup.m() * 648) as u64));
    group.bench_function("adjoint_compact", |b| {
        b.iter(|| {
            black_box(
                coefficient_weights(&prepared, &layout, setup, &public).expect("weights build"),
            )
        })
    });
    group.finish();

    // Dense fixed ring operands isolate terminal arithmetic from setup scanning.
    let left =
        TrinomialRing::<F, 648, MinusTrinomial>::from_coefficients(std::array::from_fn(|t| {
            F::from_u64((t * 17 + 3) as u64)
        }))
        .expect("terminal left operand has degree D");
    let right =
        TrinomialRing::<F, 648, MinusTrinomial>::from_coefficients(std::array::from_fn(|t| {
            F::from_u64((t * 31 + 11) as u64)
        }))
        .expect("terminal right operand has degree D");
    let alpha = F::from_u64(7);
    let evaluate = |product: &TrinomialRing<F, 648, MinusTrinomial>| {
        product
            .coefficients()
            .iter()
            .rev()
            .fold(F::zero(), |sum, &coefficient| sum * alpha + coefficient)
    };
    let domain = prepared.domain();
    let mut workspace = domain.workspace();
    assert_eq!(
        evaluate(
            &left
                .schoolbook_mul(&right)
                .expect("schoolbook product reduces")
        ),
        evaluate(&domain.multiply_with_workspace(&left, &right, &mut workspace)),
    );
    let mut terminal = criterion.benchmark_group("terminal_product");
    terminal.sample_size(10);
    terminal.warm_up_time(Duration::from_secs(1));
    terminal.measurement_time(Duration::from_secs(2));
    terminal.bench_function("schoolbook", |b| {
        b.iter(|| {
            black_box(evaluate(
                &black_box(&left)
                    .schoolbook_mul(black_box(&right))
                    .expect("schoolbook product reduces"),
            ))
        })
    });
    terminal.bench_function("transform", |b| {
        b.iter(|| {
            black_box(evaluate(&domain.multiply_with_workspace(
                black_box(&left),
                black_box(&right),
                &mut workspace,
            )))
        })
    });
    terminal.finish();
}
criterion_group!(response_weights, benchmarks);
criterion_main!(response_weights);
