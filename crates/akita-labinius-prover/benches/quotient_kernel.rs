//! Honest root A-quotient timings at the first profile's full matrix row length.

#![cfg(feature = "labinius")]

use std::hint::black_box;
use std::time::Duration;

use akita_algebra::{
    binary::BinaryField128, embed_scalar, MinusTrinomial, PlusTrinomial, TrinomialNttDomain,
    TrinomialRing,
};
use akita_challenges::{BinaryChallengeSampler, FoldDraw};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear_prepared,
    lowered::a_relation_quotients as reference,
    quotient_kernel::{a_relation_quotients as kernel, PreparedQuotientMatrix},
    PreparedCommitMatrix,
};
use akita_labinius_verifier::{
    endpoint::fold_integer, source::challenge_scalar, AdmittedRootSetup,
};
use akita_params::sis::labinius::{LabiniusRootProfile, LabiniusRootShape};
use akita_types::proof::AkitaSetupSeed;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use jolt_field::Prime128OffsetA7F7 as F;
use rand::{rngs::StdRng, RngCore, SeedableRng};

const D: usize = 648;
const M: usize = 4096;
const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128BoundedW46Delta16;

struct Draw;
impl FoldDraw for Draw {
    fn absorb_and_squeeze(&mut self, _payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok([0x53; 32])
    }
}

fn benchmarks(criterion: &mut Criterion) {
    // Every admitted fold width is a power of two. Search from one column up;
    // N=2^(14+log_C) fixes scalars_per_column=16384 and ring row length=4096.
    let log_columns = (0..=8)
        .find(|&log| {
            LabiniusRootShape::derive(PROFILE, 14 + log, log, 128)
                .is_ok_and(|shape| shape.rank_a() == 1)
        })
        .expect("first root profile admits a full-length rank-one row");
    let admitted = AdmittedRootSetup::<F, D, MinusTrinomial>::derive(
        PROFILE,
        14 + log_columns,
        log_columns,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .expect("selected root shape is admitted");
    let setup = admitted.setup();
    assert_eq!(setup.m(), M);
    assert_eq!(setup.n_a(), 1);
    let mut rng = StdRng::seed_from_u64(0x215_197);
    let source = (0..setup.source_len())
        .map(|_| u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
        .collect::<Vec<_>>();
    let commit = PreparedCommitMatrix::prepare(setup).expect("root matrix prepares");
    let quotient = PreparedQuotientMatrix::prepare(setup).expect("conjugate matrix prepares");
    let commitment = commit_binary_clear_prepared::<BinaryField128, F, D, MinusTrinomial>(
        &commit, setup, &source,
    )
    .expect("honest source commits");
    let challenges = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut Draw, b"akita/labinius/root-fold/v1", setup.columns())
        .expect("root challenges sample");
    let response = fold_integer::<BinaryField128>(
        &source,
        setup.scalar_rows(),
        setup.columns(),
        &challenges,
        setup.profile(),
    )
    .expect("honest source folds");
    assert!(response
        .iter()
        .flatten()
        .all(|&value| value >= setup.lower() && value <= setup.upper()));

    #[cfg(feature = "parallel")]
    let single_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-thread benchmark pool builds");
    let call_kernel = || {
        kernel(
            &commit,
            &quotient,
            setup,
            &commitment,
            &challenges,
            &response,
        )
        .expect("honest A relation holds")
    };
    #[cfg(feature = "parallel")]
    let expected = single_thread.install(call_kernel);
    #[cfg(not(feature = "parallel"))]
    let expected = call_kernel();
    assert_eq!(
        expected,
        reference(setup, &commitment, &challenges, &response)
            .expect("honest reference A relation holds")
    );
    eprintln!(
        "quotient_kernel: P128/Minus/D={D}/n_a=1/m={M}/columns={}, log_cells={}, log_columns={log_columns}, scalar_rows={}, quotient_prepared_payload={} bytes, commit_prepared_payload={} bytes; column_side measures both-domain challenge/image transforms, products, conjugate inverse, and slot/coefficient accumulation per column",
        setup.columns(), 14 + log_columns, setup.scalar_rows(),
        quotient.prepared_bytes().expect("prepared payload size fits usize"), commit.prepared_bytes(),
    );
    let parameter = format!("p128/minus/D={D}/n_a=1/m={M}/columns={}", setup.columns());
    let mut group = criterion.benchmark_group(format!("quotient_kernel/{parameter}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((M + setup.columns()) as u64));
    group.bench_function("reference", |b| {
        b.iter(|| {
            black_box(
                reference(
                    black_box(setup),
                    black_box(&commitment),
                    black_box(&challenges),
                    black_box(&response),
                )
                .expect("honest reference A relation holds"),
            )
        });
    });
    group.bench_function("kernel", |b| {
        b.iter(|| {
            #[cfg(feature = "parallel")]
            let value = single_thread.install(call_kernel);
            #[cfg(not(feature = "parallel"))]
            let value = call_kernel();
            black_box(value)
        });
    });
    group.bench_function("prepare", |b| {
        b.iter(|| {
            black_box(
                PreparedQuotientMatrix::prepare(black_box(setup))
                    .expect("conjugate matrix prepares"),
            )
        });
    });
    #[cfg(feature = "parallel")]
    group.bench_function("kernel_parallel", |b| {
        b.iter(|| black_box(call_kernel()));
    });
    group.finish();

    // Isolate the work that scales with image column count, without remaking a
    // 256-column source or timing matrix-vector transforms a second time.
    let domain = commit.domain();
    let conjugate = TrinomialNttDomain::<F, D, PlusTrinomial>::new()
        .expect("P128 splits the conjugate modulus");
    let mut workspace = domain.workspace();
    let mut conjugate_workspace = conjugate.workspace();
    let zero = domain.zero_ntt();
    let conjugate_zero = conjugate.zero_ntt();
    let mut accumulator = *zero.slots();
    let mut conjugate_accumulator = *conjugate_zero.slots();
    let mut component = criterion.benchmark_group(format!("quotient_columns/{parameter}"));
    component.sample_size(20);
    component.warm_up_time(Duration::from_secs(1));
    component.measurement_time(Duration::from_secs(2));
    component.throughput(Throughput::Elements(setup.columns() as u64));
    let mut run_columns = || {
        accumulator = *zero.slots();
        conjugate_accumulator = *conjugate_zero.slots();
        for (challenge, image) in black_box(&challenges)
            .iter()
            .zip(black_box(&commitment.images))
        {
            let challenge = embed_scalar::<F, 162, D, MinusTrinomial>(
                &challenge_scalar(challenge).expect("sampled challenge is scalar"),
            )
            .expect("root challenge embeds");
            let challenge_conjugate = TrinomialRing::from_coefficients(*challenge.coefficients())
                .expect("conjugate challenge has degree D");
            let image_conjugate = TrinomialRing::from_coefficients(*image.coefficients())
                .expect("conjugate image has degree D");
            let c = domain.forward_with_workspace(&challenge, &mut workspace);
            let t = domain.forward_with_workspace(image, &mut workspace);
            let product = c.pointwise_mul(&t);
            for (sum, &value) in accumulator.iter_mut().zip(product.slots()) {
                *sum -= value;
            }
            let c =
                conjugate.forward_with_workspace(&challenge_conjugate, &mut conjugate_workspace);
            let t = conjugate.forward_with_workspace(&image_conjugate, &mut conjugate_workspace);
            let product = c.pointwise_mul(&t);
            let coefficients = conjugate.inverse_with_workspace(&product, &mut conjugate_workspace);
            for (sum, &value) in conjugate_accumulator
                .iter_mut()
                .zip(coefficients.coefficients())
            {
                *sum -= value;
            }
        }
        black_box((&accumulator, &conjugate_accumulator));
    };
    component.bench_function("column_side", |b| {
        b.iter(|| {
            #[cfg(feature = "parallel")]
            single_thread.install(&mut run_columns);
            #[cfg(not(feature = "parallel"))]
            run_columns();
        });
    });
    component.finish();
}

criterion_group!(quotient_kernel, benchmarks);
criterion_main!(quotient_kernel);
