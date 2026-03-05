#![allow(missing_docs)]

use criterion::{black_box, criterion_group, Criterion};
use hachi_pcs::algebra::fields::Fp32;
use hachi_pcs::algebra::ring::CyclotomicRing;
use hachi_pcs::protocol::labrador::comkey::LabradorComKeySeed;
use hachi_pcs::protocol::labrador::fold::prove_level;
use hachi_pcs::protocol::labrador::select_config;
use hachi_pcs::protocol::labrador::setup::LabradorSetup;
use hachi_pcs::protocol::labrador::types::{
    LabradorConstraint, LabradorProof, LabradorReductionConfig, LabradorStatement, LabradorWitness,
};
use hachi_pcs::protocol::labrador::verifier::verify;
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
    let num_rows = 8;

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
            LabradorConstraint {
                coefficients: coeffs,
                target: vec![target],
            }
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
    eprintln!("  witness size        : {} bytes", witness_bytes);
    eprintln!("  witness ||s||²      : {}", witness_norm);
    eprintln!(
        "  output rows x lens  : {} rows, {} total ring elems",
        fold.next_witness.rows().len(),
        out_rings,
    );
    eprintln!("  output witness size : {} bytes", out_bytes);
    eprintln!("  output ||s'||²     : {}", out_norm);
    eprintln!("  proof size          : {} bytes", proof_bytes);
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

criterion_group!(labrador_benches, bench_labrador_single_fold,);

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
