#![allow(missing_docs)]

use criterion::{black_box, criterion_group, Criterion};
use hachi_pcs::algebra::fields::Fp32;
use hachi_pcs::algebra::ring::CyclotomicRing;
use hachi_pcs::protocol::greyhound::{
    greyhound_eval, greyhound_reduce, greyhound_verify_stage1, GreyhoundEvalProof,
};
use hachi_pcs::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use hachi_pcs::protocol::labrador::fold::prove_level;
use hachi_pcs::protocol::labrador::setup::LabradorSetup;
use hachi_pcs::protocol::labrador::transcript::{
    absorb_greyhound_eval_claim, absorb_greyhound_eval_context, absorb_greyhound_u2,
    sample_greyhound_fold_challenge, GreyhoundEvalTranscriptContext,
};
use hachi_pcs::protocol::labrador::verifier::verify;
use hachi_pcs::protocol::labrador::{
    plan_fold, LabradorFoldPlan, LabradorProof, LabradorReductionConfig, LabradorStatement,
    LabradorWitness,
};
use hachi_pcs::protocol::transcript::labels::{DOMAIN_GREYHOUND_EVAL, DOMAIN_LABRADOR_PROTOCOL};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::{FieldCore, FromSmallInt, Transcript};
use std::fs;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

type F = Fp32<4294967197>;
const D: usize = 64;
const GREYHOUND_POLY_VARS: usize = 17;

struct BenchInstance {
    witness: LabradorWitness<F, D>,
    statement: LabradorStatement<F, D>,
    config: LabradorReductionConfig,
    plan: LabradorFoldPlan,
    setup: LabradorSetup<F, D>,
    comkey_seed: LabradorComKeySeed,
    greyhound_proof: GreyhoundEvalProof<F, D>,
    poly_vars: usize,
    coeff_count: usize,
    ring_witness_len: usize,
    eval_point_len: usize,
}

fn sample_coefficients(num_coeffs: usize) -> Vec<F> {
    (0..num_coeffs)
        .map(|i| {
            // Keep only the constant coefficient of each packed ring element non-zero so the
            // Greyhound claim remains a scalar field evaluation, matching what the current
            // Rust reduction can verify.
            if i % D == 0 {
                let block = (i / D) as u64;
                let mix = block
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                F::from_i64(((mix % 7) as i64) - 3)
            } else {
                F::zero()
            }
        })
        .collect()
}

fn pack_coefficients_to_ring(coeffs: &[F]) -> Vec<CyclotomicRing<F, D>> {
    coeffs
        .chunks(D)
        .map(|chunk| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
                chunk.get(i).copied().unwrap_or_else(F::zero)
            }))
        })
        .collect()
}

fn choose_dimensions(num_ring_elements: usize) -> (usize, usize, usize) {
    let n = num_ring_elements.max(1).next_power_of_two();
    let k_total = n.trailing_zeros() as usize;
    let inner_vars = k_total / 2;
    let outer_vars = k_total - inner_vars;
    (1usize << inner_vars, 1usize << outer_vars, inner_vars)
}

fn multilinear_lagrange_basis(output: &mut [F], point: &[F]) {
    output.fill(F::zero());
    output[0] = F::one();
    let mut width = 1usize;
    for &coord in point {
        for idx in (0..width).rev() {
            let v = output[idx];
            output[idx + width] = v * coord;
            output[idx] = v - output[idx + width];
        }
        width <<= 1;
    }
}

fn reshape_columns(
    ring_witness: &[CyclotomicRing<F, D>],
    m_rows: usize,
    n_cols: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..n_cols)
        .map(|col| {
            (0..m_rows)
                .map(|row| {
                    let idx = col * m_rows + row;
                    ring_witness
                        .get(idx)
                        .copied()
                        .unwrap_or_else(CyclotomicRing::<F, D>::zero)
                })
                .collect()
        })
        .collect()
}

fn partial_evaluate_columns(
    columns: &[Vec<CyclotomicRing<F, D>>],
    inner_basis: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    columns
        .iter()
        .map(|col| {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (elem, &basis) in col.iter().zip(inner_basis.iter()) {
                acc += elem.scale(&basis);
            }
            acc
        })
        .collect()
}

fn mat_vec_mul(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    mat.iter()
        .map(|row| {
            row.iter()
                .zip(vec.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (a, b)| {
                    acc + (*a * *b)
                })
        })
        .collect()
}

fn derive_real_greyhound_instance(
    poly_vars: usize,
) -> (
    GreyhoundEvalProof<F, D>,
    LabradorWitness<F, D>,
    LabradorStatement<F, D>,
    LabradorComKeySeed,
    usize,
    usize,
) {
    let coeff_count = 1usize << poly_vars;
    let coeffs = sample_coefficients(coeff_count);
    let comkey_seed: LabradorComKeySeed = [42u8; 32];

    let (eval_point, eval_value, ring_witness_len) = {
        let ring_witness = pack_coefficients_to_ring(&coeffs);
        let (m_rows, n_cols, inner_vars) = choose_dimensions(ring_witness.len());
        let outer_vars = n_cols.trailing_zeros() as usize;
        let eval_point: Vec<F> = (0..(inner_vars + outer_vars))
            .map(|i| F::from_i64((i as i64 % 29) + 2))
            .collect();

        let inner_point = &eval_point[eval_point.len() - inner_vars..];
        let mut inner_basis = vec![F::zero(); 1usize << inner_vars];
        multilinear_lagrange_basis(&mut inner_basis, inner_point);
        let matrix = reshape_columns(&ring_witness, m_rows, n_cols);
        let partial_evals = partial_evaluate_columns(&matrix, &inner_basis);

        let mut outer_basis = vec![F::zero(); 1usize << outer_vars];
        multilinear_lagrange_basis(&mut outer_basis, &eval_point[..outer_vars]);
        let mut eval_ring = CyclotomicRing::<F, D>::zero();
        for (v, basis) in partial_evals.iter().zip(outer_basis.iter()) {
            eval_ring += v.scale(basis);
        }

        (eval_point, eval_ring.coefficients()[0], ring_witness.len())
    };

    let mut gh_transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
    let (proof, witness, _statement) = greyhound_eval(
        &coeffs,
        &eval_point,
        eval_value,
        &[],
        &comkey_seed,
        &mut gh_transcript,
    )
    .expect("greyhound_eval failed");

    let t_hat = &witness.rows()[2];
    let u1 = if proof.config.kappa1 > 0 {
        let b_mat = derive_extendable_comkey_matrix::<F, D>(
            proof.config.kappa1,
            t_hat.len(),
            &comkey_seed,
            b"labrador/comkey/B",
        );
        mat_vec_mul(&b_mat, t_hat)
    } else {
        t_hat.clone()
    };

    let z_norm_sq = witness.rows()[0]
        .iter()
        .chain(witness.rows()[1].iter())
        .map(|ring| ring.coeff_norm_sq())
        .fold(0u128, |acc, v| acc.saturating_add(v));

    let mut gh_verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
    greyhound_verify_stage1(
        &proof,
        &u1,
        &eval_point,
        eval_value,
        &witness,
        z_norm_sq,
        &comkey_seed,
        &mut gh_verify_transcript,
    )
    .expect("greyhound_verify_stage1 failed");

    let mut transcript_replay = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
    absorb_greyhound_eval_context(
        &mut transcript_replay,
        &GreyhoundEvalTranscriptContext {
            m_rows: proof.m_rows,
            n_cols: proof.n_cols,
            inner_vars: proof.inner_vars,
            eval_point_len: eval_point.len(),
        },
    )
    .expect("failed to absorb Greyhound context");
    absorb_greyhound_eval_claim(&mut transcript_replay, &eval_point, &eval_value);
    absorb_greyhound_u2(&mut transcript_replay, &proof.u2);
    let fold_challenges: Vec<F> = (0..proof.n_cols)
        .map(|_| sample_greyhound_fold_challenge(&mut transcript_replay))
        .collect();

    let mut statement = greyhound_reduce(
        &proof,
        &u1,
        &eval_point,
        eval_value,
        &fold_challenges,
        &comkey_seed,
    )
    .expect("greyhound_reduce failed");
    statement.beta_sq = witness.norm();

    (
        proof,
        witness,
        statement,
        comkey_seed,
        coeff_count,
        ring_witness_len,
    )
}

fn build_instance(poly_vars: usize) -> BenchInstance {
    let (greyhound_proof, witness, statement, comkey_seed, coeff_count, ring_witness_len) =
        derive_real_greyhound_instance(poly_vars);
    let plan = plan_fold::<F, D>(&witness, false).expect("plan_fold for Labrador fold");
    let config = plan.config;
    let rr: usize = plan.nu.iter().sum();
    let setup = LabradorSetup::new(&config, rr, plan.nn, &comkey_seed);
    let eval_point_len =
        greyhound_proof.inner_vars + greyhound_proof.n_cols.trailing_zeros() as usize;

    BenchInstance {
        witness,
        statement,
        config,
        plan,
        setup,
        comkey_seed,
        greyhound_proof,
        poly_vars,
        coeff_count,
        ring_witness_len,
        eval_point_len,
    }
}

fn row_lengths_string(rows: &[Vec<CyclotomicRing<F, D>>]) -> String {
    let lengths: Vec<String> = rows.iter().map(|row| row.len().to_string()).collect();
    format!("[{}]", lengths.join(", "))
}

fn report_sizes(inst: &BenchInstance) {
    let witness_rings: usize = inst.witness.rows().iter().map(|r| r.len()).sum();
    let witness_bytes = witness_rings * std::mem::size_of::<CyclotomicRing<F, D>>();
    let witness_norm = inst.witness.norm();
    let rr: usize = inst.plan.nu.iter().sum();

    let mut prover_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
    let fold = prove_level(
        &inst.witness,
        &inst.statement,
        &inst.config,
        &inst.plan,
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
    eprintln!("  polynomial vars      : {}", inst.poly_vars);
    eprintln!("  coefficient count    : {}", inst.coeff_count);
    eprintln!("  packed ring elems    : {}", inst.ring_witness_len);
    eprintln!(
        "  greyhound dims       : m_rows={} n_cols={} inner_vars={} eval_point_len={}",
        inst.greyhound_proof.m_rows,
        inst.greyhound_proof.n_cols,
        inst.greyhound_proof.inner_vars,
        inst.eval_point_len
    );
    eprintln!(
        "  constraints         : {}",
        inst.statement.constraints.len()
    );
    eprintln!(
        "  witness row lens    : {}",
        row_lengths_string(inst.witness.rows())
    );
    let gh = &inst.greyhound_proof.config;
    eprintln!(
        "  greyhound config    : f={} b={} fu={} bu={} kappa={} kappa1={}",
        gh.f, gh.b, gh.fu, gh.bu, gh.kappa, gh.kappa1
    );
    eprintln!(
        "  labrador L0 config  : f={} b={} fu={} bu={} kappa={} kappa1={}",
        inst.config.f,
        inst.config.b,
        inst.config.fu,
        inst.config.bu,
        inst.config.kappa,
        inst.config.kappa1
    );
    eprintln!(
        "  labrador L0 reshape : nn={} rr={} nu={:?}",
        inst.plan.nn, rr, &inst.plan.nu
    );
    eprintln!("  witness size        : {witness_bytes} bytes");
    eprintln!("  witness ||s||²      : {witness_norm}");
    eprintln!(
        "  output row lens     : {}",
        row_lengths_string(fold.next_witness.rows())
    );
    eprintln!("  output ring elems   : {out_rings}");
    eprintln!("  output witness size : {out_bytes} bytes");
    eprintln!("  output ||s'||²     : {out_norm}");
    eprintln!("  proof size          : {proof_bytes} bytes");
    eprintln!("===================================");
}

fn profile_single_fold() {
    let trace_dir = "profile_traces";
    fs::create_dir_all(trace_dir).expect("failed to create trace directory");

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX_EPOCH")
        .as_secs();
    let trace_file = format!("{trace_dir}/labrador_fold_{GREYHOUND_POLY_VARS}v_{timestamp}.json");
    let (chrome_layer, guard) = ChromeLayerBuilder::new()
        .include_args(true)
        .file(&trace_file)
        .build();

    tracing_subscriber::registry().with(chrome_layer).init();

    let inst = build_instance(GREYHOUND_POLY_VARS);
    eprintln!("Perfetto trace: {trace_file}");
    eprintln!(
        "Profiling labrador single fold: poly_vars={} constraints={} rows={}",
        inst.poly_vars,
        inst.statement.constraints.len(),
        inst.witness.rows().len()
    );
    eprintln!("Set HACHI_PARALLEL=0 to capture a single-threaded trace.");

    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
    let t0 = Instant::now();
    let fold = prove_level(
        &inst.witness,
        &inst.statement,
        &inst.config,
        &inst.plan,
        &inst.setup,
        0,
        &mut transcript,
    )
    .unwrap();

    eprintln!("prove_level wall time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!(
        "output row lens: {}",
        row_lengths_string(fold.next_witness.rows())
    );
    eprintln!("output ||s'||²: {}", fold.next_witness.norm());
    drop(fold);
    drop(guard);
    eprintln!("Done. Open the trace in https://ui.perfetto.dev/.");
}

fn bench_labrador_single_fold(c: &mut Criterion) {
    let inst = build_instance(GREYHOUND_POLY_VARS);

    report_sizes(&inst);

    let mut group = c.benchmark_group(format!("labrador/fold_1x/greyhound_{GREYHOUND_POLY_VARS}v"));
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
                    black_box(&inst.plan),
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
        &inst.plan,
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

#[allow(dead_code)]
fn bench_labrador_two_level_fold(c: &mut Criterion) {
    let inst = build_instance(GREYHOUND_POLY_VARS);

    let mut prover_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
    let fold1 = prove_level(
        &inst.witness,
        &inst.statement,
        &inst.config,
        &inst.plan,
        &inst.setup,
        0,
        &mut prover_transcript,
    )
    .unwrap();

    let plan2 = plan_fold::<F, D>(&fold1.next_witness, false).expect("plan_fold level 2");
    let config2 = plan2.config;
    let rr2: usize = plan2.nu.iter().sum();
    let setup2 = LabradorSetup::new(&config2, rr2, plan2.nn, &inst.comkey_seed);

    let fold2 = prove_level(
        &fold1.next_witness,
        &fold1.statement,
        &config2,
        &plan2,
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
        let rr0: usize = inst.plan.nu.iter().sum();

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
        eprintln!("  L0 reshape           : nn={} rr={}", inst.plan.nn, rr0);
        eprintln!(
            "  L1 witness rows      : {} rows, {w2_rings} total ring elems",
            fold1.next_witness.rows().len(),
        );
        eprintln!("  L1 witness size      : {w2_bytes} bytes");
        eprintln!(
            "  L1 config            : f={} b={} fu={} bu={} kappa={} kappa1={}",
            config2.f, config2.b, config2.fu, config2.bu, config2.kappa, config2.kappa1
        );
        eprintln!("  L1 reshape           : nn={} rr={}", plan2.nn, rr2);
        eprintln!(
            "  final witness        : {} rows, {final_rings} total ring elems",
            fold2.next_witness.rows().len(),
        );
        eprintln!("  final witness size   : {final_bytes} bytes");
        eprintln!("  total proof size     : {} bytes", proof.size());
        eprintln!("======================================");
    }

    let mut group = c.benchmark_group(format!("labrador/fold_2x/greyhound_{GREYHOUND_POLY_VARS}v"));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("prove", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
            let f1 = prove_level(
                black_box(&inst.witness),
                black_box(&inst.statement),
                black_box(&inst.config),
                black_box(&inst.plan),
                black_box(&inst.setup),
                0,
                &mut transcript,
            )
            .unwrap();
            let p2 = plan_fold::<F, D>(&f1.next_witness, false).unwrap();
            let c2 = p2.config;
            let r2: usize = p2.nu.iter().sum();
            let s2 = LabradorSetup::new(&c2, r2, p2.nn, &inst.comkey_seed);
            black_box(
                prove_level(
                    black_box(&f1.next_witness),
                    black_box(&f1.statement),
                    black_box(&c2),
                    black_box(&p2),
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
// criterion_group!(labrador_benches, bench_labrador_single_fold,);

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

    if std::env::var("HACHI_TRACE_LABRADOR")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
    {
        profile_single_fold();
        return;
    }

    labrador_benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
