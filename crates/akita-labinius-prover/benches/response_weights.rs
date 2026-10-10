//! Shift-recurrence response weights at the shipped profile's full row length.

#![cfg(feature = "labinius")]

use akita_algebra::MinusTrinomial;
use akita_challenges::{BinaryChallengeSampler, FoldDraw};
use akita_error::AkitaError;
use akita_labinius_verifier::{
    lowered::{coefficient_weights, LoweredChallenges, LoweredPublic, LoweredRootLayout},
    AdmittedRootSetup,
};
use akita_params::sis::labinius::LabiniusRootProfile;
use akita_types::proof::AkitaSetupSeed;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use jolt_field::{CanonicalEncoding, Ext2, ExtField, Field, Prime128Offset275, Prime64Offset59};
use std::{hint::black_box, time::Duration};

struct Draw;
impl FoldDraw for Draw {
    fn absorb_and_squeeze(&mut self, _payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok([0x53; 32])
    }
}

/// Response weights over the base field `F` and the challenge field `E`. The
/// matrix comes from the `Prime128Offset275` stream for both pairs.
fn pair<F: Field + CanonicalEncoding, E: ExtField<F>>(criterion: &mut Criterion, name: &str) {
    let profile = LabiniusRootProfile::D648Q25BoundedW46;
    let admitted = AdmittedRootSetup::<648, MinusTrinomial>::derive::<Prime128Offset275>(
        profile,
        22,
        8,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .expect("shipped profile admits");
    let setup = admitted.setup();
    let layout =
        LoweredRootLayout::new::<F, E, _, _>(setup, admitted.shape()).expect("layout admits");
    let fold = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut Draw, b"weights", setup.columns())
        .expect("fold samples");
    // The commitment rows alone: the matrix pass dominates, and neither the
    // parity row nor the prime row scans the matrix.
    let public = LoweredPublic::new(
        &layout,
        setup,
        &fold,
        &vec![0; layout.encoding().a_carry_len()],
        LoweredChallenges {
            alpha: E::from_u64(7),
            gamma: E::from_u64(13),
        },
        None,
        None,
    )
    .expect("public builds");
    let mut group = criterion.benchmark_group(format!("response_weights/{name}/q25/minus/D648"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((setup.m() * 648) as u64));
    group.bench_function("shift_recurrence", |b| {
        b.iter(|| black_box(coefficient_weights(&layout, &public, setup).expect("weights build")))
    });
    group.finish();
}

fn benchmarks(criterion: &mut Criterion) {
    pair::<Prime128Offset275, Prime128Offset275>(criterion, "p128");
    pair::<Prime64Offset59, Ext2<Prime64Offset59>>(criterion, "p64x2");
}

criterion_group!(response_weights, benchmarks);
criterion_main!(response_weights);
