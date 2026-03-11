#![allow(missing_docs)]

use criterion::{black_box, criterion_group, Criterion};
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
use hachi_pcs::protocol::labrador::{plan_fold, LabradorProof, LabradorStatement, LabradorWitness};
use hachi_pcs::protocol::transcript::labels::{DOMAIN_GREYHOUND_EVAL, DOMAIN_LABRADOR_PROTOCOL};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::{FieldCore, FromSmallInt, Transcript};
use std::time::{Duration, Instant};

const TARGET_WITNESS_RING_ELEMS: usize = 600;

macro_rules! define_bench {
    (
        mod_name: $mod:ident,
        field_type: $ftype:ty,
        ring_dim: $d:expr,
        poly_vars: $pv:expr,
        label: $label:expr
    ) => {
        #[allow(unreachable_pub)]
        mod $mod {
            use super::*;

            type F = $ftype;
            const D: usize = $d;
            const POLY_VARS: usize = $pv;
            const RING_BYTES: usize = std::mem::size_of::<CyclotomicRing<F, D>>();
            const TARGET_WITNESS_BYTES: usize = TARGET_WITNESS_RING_ELEMS * RING_BYTES;

            fn sample_coefficients(num_coeffs: usize) -> Vec<F> {
                (0..num_coeffs)
                    .map(|i| {
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

            fn witness_bytes(w: &LabradorWitness<F, D>) -> usize {
                w.rows().iter().map(|r| r.len()).sum::<usize>() * RING_BYTES
            }

            fn witness_ring_elems(w: &LabradorWitness<F, D>) -> usize {
                w.rows().iter().map(|r| r.len()).sum::<usize>()
            }

            fn row_lengths_string(rows: &[Vec<CyclotomicRing<F, D>>]) -> String {
                let lengths: Vec<String> =
                    rows.iter().map(|row| row.len().to_string()).collect();
                format!("[{}]", lengths.join(", "))
            }

            fn row_norms_string(rows: &[Vec<CyclotomicRing<F, D>>]) -> String {
                let norms: Vec<String> = rows
                    .iter()
                    .map(|row| {
                        let norm_sq: u128 = row.iter().map(|r| r.coeff_norm_sq()).sum();
                        format!("{norm_sq}")
                    })
                    .collect();
                format!("[{}]", norms.join(", "))
            }

            fn derive_greyhound_instance(
                poly_vars: usize,
            ) -> (
                GreyhoundEvalProof<F, D>,
                LabradorWitness<F, D>,
                LabradorStatement<F, D>,
                LabradorComKeySeed,
            ) {
                let coeff_count = 1usize << poly_vars;
                let coeffs = sample_coefficients(coeff_count);
                let comkey_seed: LabradorComKeySeed = [42u8; 32];

                let (eval_point, eval_target) = {
                    let ring_witness = pack_coefficients_to_ring(&coeffs);
                    let (m_rows, n_cols, inner_vars) =
                        choose_dimensions(ring_witness.len());
                    let outer_vars = n_cols.trailing_zeros() as usize;
                    let eval_point: Vec<F> = (0..(inner_vars + outer_vars))
                        .map(|i| F::from_i64((i as i64 % 29) + 2))
                        .collect();

                    let inner_point = &eval_point[..inner_vars];
                    let mut inner_basis = vec![F::zero(); 1usize << inner_vars];
                    multilinear_lagrange_basis(&mut inner_basis, inner_point);
                    let matrix = reshape_columns(&ring_witness, m_rows, n_cols);
                    let partial_evals =
                        partial_evaluate_columns(&matrix, &inner_basis);

                    let mut outer_basis = vec![F::zero(); 1usize << outer_vars];
                    multilinear_lagrange_basis(
                        &mut outer_basis,
                        &eval_point[inner_vars..],
                    );
                    let mut eval_ring = CyclotomicRing::<F, D>::zero();
                    for (v, basis) in partial_evals.iter().zip(outer_basis.iter()) {
                        eval_ring += v.scale(basis);
                    }

                    (eval_point, eval_ring)
                };

                let mut gh_transcript =
                    Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
                let (proof, witness, _statement, _fold_ch) = greyhound_eval(
                    &coeffs,
                    &eval_point,
                    eval_target,
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

                let mut gh_verify_transcript =
                    Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
                greyhound_verify_stage1(
                    &proof,
                    &u1,
                    &eval_point,
                    eval_target,
                    &witness,
                    z_norm_sq,
                    &comkey_seed,
                    &mut gh_verify_transcript,
                )
                .expect("greyhound_verify_stage1 failed");

                let mut transcript_replay =
                    Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
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
                absorb_greyhound_eval_claim(
                    &mut transcript_replay,
                    &eval_point,
                    &eval_target,
                );
                absorb_greyhound_u2(&mut transcript_replay, &proof.u2);
                let fold_challenges: Vec<F> = (0..proof.n_cols)
                    .map(|_| {
                        sample_greyhound_fold_challenge(&mut transcript_replay)
                    })
                    .collect();

                let mut statement = greyhound_reduce(
                    &proof,
                    &u1,
                    &eval_point,
                    eval_target,
                    &fold_challenges,
                    &comkey_seed,
                )
                .expect("greyhound_reduce failed");
                statement.beta_sq = witness.norm();

                (proof, witness, statement, comkey_seed)
            }

            fn recursive_prove_until_target(
                witness: LabradorWitness<F, D>,
                statement: &LabradorStatement<F, D>,
                comkey_seed: &LabradorComKeySeed,
                transcript: &mut Blake2bTranscript<F>,
                verbose: bool,
            ) -> LabradorProof<F, D> {
                let max_levels = 16usize;
                let mut levels = Vec::new();
                let mut cur_witness = witness;
                let mut cur_statement = statement.clone();
                let mut level_idx = 0usize;

                if verbose {
                    eprintln!(
                        "  L{level_idx} input : {} rows, {} ring elems, {} bytes, ||s||²={}",
                        cur_witness.rows().len(),
                        witness_ring_elems(&cur_witness),
                        witness_bytes(&cur_witness),
                        cur_witness.norm(),
                    );
                    eprintln!(
                        "             row lens : {}",
                        row_lengths_string(cur_witness.rows())
                    );
                    eprintln!(
                        "             row norms: {}",
                        row_norms_string(cur_witness.rows())
                    );
                }

                while level_idx < max_levels {
                    let wb = witness_bytes(&cur_witness);
                    if wb <= TARGET_WITNESS_BYTES {
                        if verbose {
                            eprintln!(
                                "  ==> witness is {} bytes <= {} target, stopping.",
                                wb, TARGET_WITNESS_BYTES
                            );
                        }
                        break;
                    }
                    if cur_witness.rows().len() <= 1 {
                        if verbose {
                            eprintln!(
                                "  ==> single row witness, cannot fold further."
                            );
                        }
                        break;
                    }

                    let is_tail = wb <= 2 * TARGET_WITNESS_BYTES;
                    let plan = plan_fold::<F, D>(&cur_witness, is_tail)
                        .expect("plan_fold failed");
                    let cfg = plan.config;
                    let rr: usize = plan.nu.iter().sum();
                    let setup =
                        LabradorSetup::new(&cfg, rr, plan.nn, comkey_seed);

                    let t0 = Instant::now();
                    let fold = prove_level(
                        &cur_witness,
                        &cur_statement,
                        &cfg,
                        &plan,
                        &setup,
                        level_idx,
                        transcript,
                    )
                    .expect("prove_level failed");
                    let elapsed = t0.elapsed();

                    let out_elems = witness_ring_elems(&fold.next_witness);
                    let out_bytes = witness_bytes(&fold.next_witness);
                    let out_norm = fold.next_witness.norm();

                    if verbose {
                        let m = cfg.fu * rr * cfg.kappa
                            + cfg.fu * (rr * rr + rr) / 2;
                        eprintln!();
                        eprintln!(
                            "  L{level_idx} config: f={} b={} fu={} bu={} kappa={} kappa1={} tail={}",
                            cfg.f, cfg.b, cfg.fu, cfg.bu, cfg.kappa, cfg.kappa1,
                            cfg.tail
                        );
                        eprintln!(
                            "  L{level_idx} reshape: nn={} rr={} nu={:?}",
                            plan.nn, rr, &plan.nu
                        );
                        eprintln!("  L{level_idx} m (t_hat): {m}");
                        eprintln!(
                            "  L{level_idx} output: {} rows, {} ring elems, {} bytes ({:.1} KB), ||s||²={}",
                            fold.next_witness.rows().len(),
                            out_elems,
                            out_bytes,
                            out_bytes as f64 / 1024.0,
                            out_norm,
                        );
                        eprintln!(
                            "             row lens : {}",
                            row_lengths_string(fold.next_witness.rows())
                        );
                        eprintln!(
                            "             row norms: {}",
                            row_norms_string(fold.next_witness.rows())
                        );
                        eprintln!(
                            "  L{level_idx} proof : {} bytes, time={:.3}s",
                            fold.level_proof.size(),
                            elapsed.as_secs_f64()
                        );
                    }

                    if verbose {
                        for (ci, cnst) in fold.statement.constraints.iter().enumerate() {
                            let mut lhs = CyclotomicRing::<F, D>::zero();
                            for term in &cnst.terms {
                                let row = &fold.next_witness.rows()[term.row];
                                for (j, coeff) in term.coefficients.iter().enumerate() {
                                    lhs += *coeff * row[term.offset + j];
                                }
                            }
                            if lhs != cnst.target {
                                eprintln!(
                                    "  L{level_idx} CONSTRAINT {ci} FAILED after fold"
                                );
                            }
                        }
                    }

                    levels.push(fold.level_proof);
                    cur_statement = fold.statement;
                    cur_witness = fold.next_witness;
                    level_idx += 1;
                }

                LabradorProof {
                    levels,
                    final_opening_witness: cur_witness,
                }
            }

            pub fn report_recursive(poly_vars: usize) {
                let (gh_proof, witness, statement, comkey_seed) =
                    derive_greyhound_instance(poly_vars);

                eprintln!(
                    "=== Labrador recursive prover report ({}, D={D}, poly_vars={poly_vars}) ===",
                    $label
                );
                eprintln!(
                    "  greyhound dims  : m_rows={} n_cols={} inner_vars={}",
                    gh_proof.m_rows, gh_proof.n_cols, gh_proof.inner_vars
                );
                let ghc = &gh_proof.config;
                eprintln!(
                    "  greyhound config: f={} b={} fu={} bu={} kappa={} kappa1={}",
                    ghc.f, ghc.b, ghc.fu, ghc.bu, ghc.kappa, ghc.kappa1
                );
                eprintln!(
                    "  constraints     : {}",
                    statement.constraints.len()
                );
                eprintln!(
                    "  target witness  : {} KB",
                    TARGET_WITNESS_BYTES / 1024
                );
                eprintln!(
                    "  ring elem size  : {} bytes",
                    RING_BYTES
                );
                eprintln!();

                let mut transcript =
                    Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
                let t0 = Instant::now();
                let proof = recursive_prove_until_target(
                    witness.clone(),
                    &statement,
                    &comkey_seed,
                    &mut transcript,
                    true,
                );
                let total_prove_time = t0.elapsed();

                let final_elems =
                    witness_ring_elems(&proof.final_opening_witness);
                let final_bytes =
                    witness_bytes(&proof.final_opening_witness);
                let proof_bytes = proof.size();
                let levels_bytes: usize =
                    proof.levels.iter().map(|l| l.size()).sum();

                eprintln!();
                eprintln!("  --- summary ---");
                eprintln!(
                    "  fold levels          : {}",
                    proof.levels.len()
                );
                eprintln!(
                    "  final witness        : {final_elems} ring elems, {final_bytes} bytes ({:.1} KB)",
                    final_bytes as f64 / 1024.0
                );
                eprintln!(
                    "  total proof payload  : {levels_bytes} bytes ({:.1} KB)",
                    levels_bytes as f64 / 1024.0
                );
                eprintln!(
                    "  total proof + witness: {proof_bytes} bytes ({:.1} KB)",
                    proof_bytes as f64 / 1024.0
                );
                eprintln!(
                    "  total prove time     : {:.3}s",
                    total_prove_time.as_secs_f64()
                );

                eprintln!();
                eprintln!("  --- verifying ---");
                let mut verify_transcript =
                    Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
                let t0 = Instant::now();
                verify(
                    &statement,
                    &proof,
                    &comkey_seed,
                    &mut verify_transcript,
                )
                .expect("verification failed");
                let verify_time = t0.elapsed();
                eprintln!(
                    "  verification         : OK ({:.3}s)",
                    verify_time.as_secs_f64()
                );
                eprintln!(
                    "===================================================",
                );
            }

            pub fn bench(c: &mut Criterion) {
                let (_gh_proof, witness, statement, comkey_seed) =
                    derive_greyhound_instance(POLY_VARS);

                report_recursive(POLY_VARS);

                let mut group = c.benchmark_group(format!(
                    "labrador/recursive/{}_{}v",
                    $label, POLY_VARS
                ));
                group.sample_size(10);
                group.measurement_time(Duration::from_secs(60));

                group.bench_function("prove", |b| {
                    b.iter(|| {
                        let mut transcript =
                            Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
                        black_box(recursive_prove_until_target(
                            black_box(witness.clone()),
                            black_box(&statement),
                            black_box(&comkey_seed),
                            &mut transcript,
                            false,
                        ))
                    })
                });

                let mut prover_transcript =
                    Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
                let proof = recursive_prove_until_target(
                    witness.clone(),
                    &statement,
                    &comkey_seed,
                    &mut prover_transcript,
                    false,
                );

                group.bench_function("verify", |b| {
                    b.iter(|| {
                        let mut transcript =
                            Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
                        black_box(
                            verify(
                                black_box(&statement),
                                black_box(&proof),
                                black_box(&comkey_seed),
                                &mut transcript,
                            )
                            .unwrap(),
                        )
                    })
                });

                group.finish();
            }
        }
    };
}

define_bench! {
    mod_name: fp32_17v,
    field_type: hachi_pcs::algebra::fields::Fp32<4294967197>,
    ring_dim: 64,
    poly_vars: 17,
    label: "fp32"
}

define_bench! {
    mod_name: fp128_17v,
    field_type: hachi_pcs::algebra::Fp128<0xfffffffffffffffffffffffffffffeed>,
    ring_dim: 64,
    poly_vars: 17,
    label: "fp128"
}

fn bench_fp32(c: &mut Criterion) {
    fp32_17v::bench(c);
}

fn bench_fp128(c: &mut Criterion) {
    fp128_17v::bench(c);
}

criterion_group!(labrador_recursive_benches, bench_fp32, bench_fp128);

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

    labrador_recursive_benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
