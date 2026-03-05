#![allow(missing_docs)]

use criterion::{black_box, criterion_group, Criterion};
use hachi_pcs::algebra::fields::Fp32;
use hachi_pcs::algebra::ring::CyclotomicRing;
use hachi_pcs::protocol::labrador::comkey::LabradorComKeySeed;
use hachi_pcs::protocol::labrador::fold::prove_level;
use hachi_pcs::protocol::labrador::select_config;
use hachi_pcs::protocol::labrador::setup::LabradorSetup;
use hachi_pcs::protocol::labrador::verifier::verify;
use hachi_pcs::protocol::labrador::{
    LabradorConstraint, LabradorConstraintTerm, LabradorProof, LabradorReductionConfig,
    LabradorStatement, LabradorWitness,
};
use hachi_pcs::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::{FieldCore, FromSmallInt, Transcript};
use std::time::Duration;

type F = Fp32<4294967197>;
const D: usize = 64;

fn mk_ring(c: i64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
        if i == 0 {
            F::from_i64(c)
        } else {
            F::zero()
        }
    }))
}

struct BenchInstance {
    witness: LabradorWitness<F, D>,
    statement: LabradorStatement<F, D>,
    config: LabradorReductionConfig,
    setup: LabradorSetup<F, D>,
    comkey_seed: LabradorComKeySeed,
}

fn build_instance(num_constraints: usize) -> BenchInstance {
    let row_len = 1 << 10;
    let num_rows = 16;

    let rows: Vec<Vec<CyclotomicRing<F, D>>> = (0..num_rows)
        .map(|r| {
            (0..row_len)
                .map(|i| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                        let mix = (r as u64)
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(i as u64 * 31 + j as u64);
                        F::from_i64(((mix % 7) as i64) - 3)
                    }))
                })
                .collect()
        })
        .collect();
    let witness = LabradorWitness::new(rows);

    let constraints: Vec<LabradorConstraint<F, D>> = (0..num_constraints)
        .map(|c_idx| {
            let coeffs: Vec<Vec<CyclotomicRing<F, D>>> = (0..num_rows)
                .map(|r| {
                    (0..row_len)
                        .map(|j| {
                            let v = ((c_idx + r + j) % 5) as i64 - 2;
                            mk_ring(v)
                        })
                        .collect()
                })
                .collect();
            let mut target = CyclotomicRing::<F, D>::zero();
            for (r, coeff_row) in coeffs.iter().enumerate() {
                for (j, coeff) in coeff_row.iter().enumerate() {
                    target += *coeff * witness.rows()[r][j];
                }
            }
            let terms = coeffs
                .into_iter()
                .enumerate()
                .map(|(row_idx, row_coeffs)| LabradorConstraintTerm::new(row_idx, 0, row_coeffs))
                .collect();
            LabradorConstraint::new(terms, target)
        })
        .collect();

    let statement = LabradorStatement {
        u1: Vec::new(),
        u2: Vec::new(),
        challenges: Vec::new(),
        constraints,
        beta_sq: 1 << 60,
        hash: [0u8; 16],
    };

    let config = select_config::<F, D>(&witness).expect("select_config failed");

    let comkey_seed: LabradorComKeySeed = [42u8; 32];
    let r = witness.rows().len();
    let max_len = witness
        .rows()
        .iter()
        .map(|row| row.len())
        .max()
        .unwrap_or(0);
    let setup = LabradorSetup::new(&config, r, max_len, &comkey_seed);

    BenchInstance {
        witness,
        statement,
        config,
        setup,
        comkey_seed,
    }
}

fn report_sizes(inst: &BenchInstance) {
    let witness_rings: usize = inst.witness.rows().iter().map(|r| r.len()).sum();
    let witness_bytes = witness_rings * std::mem::size_of::<CyclotomicRing<F, D>>();
    let witness_norm = inst.witness.norm();

    let mut prover_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
    let fold = prove_level(
        &inst.witness,
        &inst.statement,
        &inst.config,
        &inst.setup,
        0,
        &mut prover_transcript,
    )
    .unwrap();

    let proof = LabradorProof {
        levels: vec![fold.level_proof],
        final_opening_witness: fold.next_witness.clone(),
    };
    let proof_bytes = proof.size();

    let out_rings: usize = fold.next_witness.rows().iter().map(|r| r.len()).sum();
    let out_bytes = out_rings * std::mem::size_of::<CyclotomicRing<F, D>>();
    let out_norm = fold.next_witness.norm();

    eprintln!("=== Labrador single-fold report ===");
    eprintln!(
        "  constraints         : {}",
        inst.statement.constraints.len()
    );
    eprintln!(
        "  witness rows x len  : {} x {}",
        inst.witness.rows().len(),
        inst.witness.rows()[0].len()
    );
    eprintln!("  witness size        : {witness_bytes} bytes");
    eprintln!("  witness ||s||²      : {witness_norm}");
    eprintln!(
        "  output rows x lens  : {} rows, {out_rings} total ring elems",
        fold.next_witness.rows().len(),
    );
    eprintln!("  output witness size : {out_bytes} bytes");
    eprintln!("  output ||s'||²     : {out_norm}");
    eprintln!("  proof size          : {proof_bytes} bytes");
    eprintln!("===================================");
}

fn bench_labrador_single_fold(c: &mut Criterion) {
    let num_constraints = 1 << 4;
    let inst = build_instance(num_constraints);

    report_sizes(&inst);

    let mut group = c.benchmark_group(format!("labrador/fold_1x/{num_constraints}c"));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("prove", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
            black_box(
                prove_level(
                    black_box(&inst.witness),
                    black_box(&inst.statement),
                    black_box(&inst.config),
                    black_box(&inst.setup),
                    0,
                    &mut transcript,
                )
                .unwrap(),
            )
        })
    });

    let mut prover_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
    let fold = prove_level(
        &inst.witness,
        &inst.statement,
        &inst.config,
        &inst.setup,
        0,
        &mut prover_transcript,
    )
    .unwrap();
    let proof = LabradorProof {
        levels: vec![fold.level_proof],
        final_opening_witness: fold.next_witness,
    };

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
            black_box(
                verify(
                    black_box(&inst.statement),
                    black_box(&proof),
                    black_box(&inst.comkey_seed),
                    &mut transcript,
                )
                .unwrap(),
            )
        })
    });

    group.finish();
}

fn bench_labrador_two_level_fold(c: &mut Criterion) {
    let num_constraints = 1 << 4;
    let inst = build_instance(num_constraints);

    let mut prover_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
    let fold1 = prove_level(
        &inst.witness,
        &inst.statement,
        &inst.config,
        &inst.setup,
        0,
        &mut prover_transcript,
    )
    .unwrap();

    let r2 = fold1.next_witness.rows().len();
    let max_len2 = fold1
        .next_witness
        .rows()
        .iter()
        .map(|row| row.len())
        .max()
        .unwrap_or(0);
    let config2 = select_config::<F, D>(&fold1.next_witness).expect("select_config level 2");
    let setup2 = LabradorSetup::new(&config2, r2, max_len2, &inst.comkey_seed);

    let fold2 = prove_level(
        &fold1.next_witness,
        &fold1.statement,
        &config2,
        &setup2,
        1,
        &mut prover_transcript,
    )
    .unwrap();

    let proof = LabradorProof {
        levels: vec![fold1.level_proof.clone(), fold2.level_proof],
        final_opening_witness: fold2.next_witness.clone(),
    };

    {
        let w1_rings: usize = inst.witness.rows().iter().map(|r| r.len()).sum();
        let w1_bytes = w1_rings * std::mem::size_of::<CyclotomicRing<F, D>>();
        let w2_rings: usize = fold1.next_witness.rows().iter().map(|r| r.len()).sum();
        let w2_bytes = w2_rings * std::mem::size_of::<CyclotomicRing<F, D>>();
        let final_rings: usize = fold2.next_witness.rows().iter().map(|r| r.len()).sum();
        let final_bytes = final_rings * std::mem::size_of::<CyclotomicRing<F, D>>();

        eprintln!("=== Labrador two-level fold report ===");
        eprintln!(
            "  constraints          : {}",
            inst.statement.constraints.len()
        );
        eprintln!(
            "  L0 witness rows x len: {} x {}",
            inst.witness.rows().len(),
            inst.witness.rows()[0].len()
        );
        eprintln!("  L0 witness size      : {w1_bytes} bytes");
        let c0 = &inst.config;
        eprintln!(
            "  L0 config            : f={} b={} fu={} bu={} kappa={} kappa1={}",
            c0.f, c0.b, c0.fu, c0.bu, c0.kappa, c0.kappa1
        );
        eprintln!(
            "  L1 witness rows      : {} rows, {w2_rings} total ring elems",
            fold1.next_witness.rows().len(),
        );
        eprintln!("  L1 witness size      : {w2_bytes} bytes");
        eprintln!(
            "  L1 config            : f={} b={} fu={} bu={} kappa={} kappa1={}",
            config2.f, config2.b, config2.fu, config2.bu, config2.kappa, config2.kappa1
        );
        eprintln!(
            "  final witness        : {} rows, {final_rings} total ring elems",
            fold2.next_witness.rows().len(),
        );
        eprintln!("  final witness size   : {final_bytes} bytes");
        eprintln!("  total proof size     : {} bytes", proof.size());
        eprintln!("======================================");
    }

    let mut group = c.benchmark_group(format!("labrador/fold_2x/{num_constraints}c"));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("prove", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
            let f1 = prove_level(
                black_box(&inst.witness),
                black_box(&inst.statement),
                black_box(&inst.config),
                black_box(&inst.setup),
                0,
                &mut transcript,
            )
            .unwrap();
            let r = f1.next_witness.rows().len();
            let ml = f1
                .next_witness
                .rows()
                .iter()
                .map(|row| row.len())
                .max()
                .unwrap_or(0);
            let cfg2 = select_config::<F, D>(&f1.next_witness).unwrap();
            let s2 = LabradorSetup::new(&cfg2, r, ml, &inst.comkey_seed);
            black_box(
                prove_level(
                    black_box(&f1.next_witness),
                    black_box(&f1.statement),
                    black_box(&cfg2),
                    black_box(&s2),
                    1,
                    &mut transcript,
                )
                .unwrap(),
            )
        })
    });

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
            black_box(
                verify(
                    black_box(&inst.statement),
                    black_box(&proof),
                    black_box(&inst.comkey_seed),
                    &mut transcript,
                )
                .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(
    labrador_benches,
    bench_labrador_single_fold,
    bench_labrador_two_level_fold,
);

fn main() {
    #[cfg(feature = "parallel")]
    {
        let num_threads = if std::env::var("HACHI_PARALLEL")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            eprintln!("HACHI_PARALLEL=0: running single-threaded");
            1
        } else {
            0
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .stack_size(64 * 1024 * 1024)
            .build_global()
            .ok();
    }

    labrador_benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
