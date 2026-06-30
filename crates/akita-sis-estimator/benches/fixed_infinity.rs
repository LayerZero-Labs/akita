use akita_sis_estimator::{
    cost_infinity, scalar_sis_from_ring, AkitaModulusFamily, EstimateConfig, OptimizerConfig,
    ReductionCostModel, ShapeModel,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[derive(Clone, Copy)]
struct FixedInfinityCase {
    label: &'static str,
    family: AkitaModulusFamily,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
    beta: u32,
    zeta: u32,
}

const CASES: &[FixedInfinityCase] = &[
    FixedInfinityCase {
        label: "q32_d32_r1_w2_linf2_beta63_zeta0",
        family: AkitaModulusFamily::Q32,
        d: 32,
        rank: 1,
        width: 2,
        coeff_linf_bound: 2,
        beta: 63,
        zeta: 0,
    },
    FixedInfinityCase {
        label: "q32_d32_r1_w2_linf15_beta40_zeta0",
        family: AkitaModulusFamily::Q32,
        d: 32,
        rank: 1,
        width: 2,
        coeff_linf_bound: 15,
        beta: 40,
        zeta: 0,
    },
    FixedInfinityCase {
        label: "q32_d32_r5_w10_linf15_beta50_zeta0",
        family: AkitaModulusFamily::Q32,
        d: 32,
        rank: 5,
        width: 10,
        coeff_linf_bound: 15,
        beta: 50,
        zeta: 0,
    },
    FixedInfinityCase {
        label: "q64_d32_r1_w2_linf15_beta63_zeta0",
        family: AkitaModulusFamily::Q64,
        d: 32,
        rank: 1,
        width: 2,
        coeff_linf_bound: 15,
        beta: 63,
        zeta: 0,
    },
    FixedInfinityCase {
        label: "q128_d32_r1_w2_linf15_beta63_zeta0",
        family: AkitaModulusFamily::Q128,
        d: 32,
        rank: 1,
        width: 2,
        coeff_linf_bound: 15,
        beta: 63,
        zeta: 0,
    },
];

fn bench_fixed_infinity(c: &mut Criterion) {
    let mut group = c.benchmark_group("sis_fixed_infinity");
    for case in CASES {
        let params = scalar_sis_from_ring(
            case.family,
            case.d,
            case.rank,
            case.width,
            case.coeff_linf_bound,
        )
        .unwrap();
        let config = EstimateConfig {
            red_cost_model: ReductionCostModel::default(),
            red_shape_model: ShapeModel::Lgsa,
            optimizer: OptimizerConfig::Fixed {
                beta: case.beta,
                zeta: case.zeta,
            },
            ..EstimateConfig::default()
        };

        group.bench_function(BenchmarkId::new("cost_infinity", case.label), |bench| {
            bench.iter(|| {
                black_box(
                    cost_infinity(
                        black_box(case.beta),
                        black_box(&params),
                        black_box(case.zeta),
                        black_box(&config),
                    )
                    .unwrap(),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(fixed_infinity, bench_fixed_infinity);
criterion_main!(fixed_infinity);
