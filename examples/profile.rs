#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::error::HachiError;
use hachi_pcs::primitives::serialization::Compress;
use hachi_pcs::protocol::commitment::{
    DecompositionParams, Fp128CommitmentConfig, HachiCommitmentLayout,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::proof::HachiProof;
use hachi_pcs::protocol::sumcheck::norm_sumcheck::choose_round_kernel;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{BasisMode, CommitmentScheme, FromSmallInt, HachiSerialize, Transcript};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;

const D: usize = Fp128CommitmentConfig::D;

#[derive(Clone, Copy, Debug)]
struct ProfileCfg;
impl CommitmentConfig for ProfileCfg {
    const D: usize = D;
    const N_A: usize = Fp128CommitmentConfig::N_A;
    const N_B: usize = Fp128CommitmentConfig::N_B;
    const N_D: usize = Fp128CommitmentConfig::N_D;
    const CHALLENGE_WEIGHT: usize = Fp128CommitmentConfig::CHALLENGE_WEIGHT;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 4,
            log_commit_bound: 128,
            log_open_bound: None,
        }
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let m = env::var("HACHI_M_VARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let r = env::var("HACHI_R_VARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        HachiCommitmentLayout::new::<Self>(m, r, &Self::decomposition())
    }
}

type Scheme = HachiCommitmentScheme<D, ProfileCfg>;

fn run_prove(
    label: &str,
    setup: &<Scheme as CommitmentScheme<F, D>>::ProverSetup,
    poly: &DensePoly<F, D>,
    pt: &[F],
    opening: F,
    layout: &HachiCommitmentLayout,
) {
    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme as CommitmentScheme<F, D>>::commit(poly, setup, layout).unwrap();
    eprintln!("[{label}] commit: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme as CommitmentScheme<F, D>>::prove(
        setup,
        poly,
        pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
        layout,
    )
    .unwrap();
    eprintln!("[{label}] prove: {:.3}s", t0.elapsed().as_secs_f64());
    print_proof_summary(label, &proof);

    let t0 = Instant::now();
    let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
    <Scheme as CommitmentScheme<F, D>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        pt,
        &opening,
        &commitment,
        BasisMode::Lagrange,
        layout,
    )
    .unwrap();
    eprintln!("[{label}] verify: {:.3}s", t0.elapsed().as_secs_f64());
}

fn print_proof_summary(label: &str, proof: &HachiProof<F, D>) {
    eprintln!(
        "[{label}]   levels: {}, proof size: {} bytes",
        proof.levels.len(),
        proof.size()
    );
    for (i, lp) in proof.levels.iter().enumerate() {
        let w_comm_size = lp.w_commitment.serialized_size(Compress::No);
        let sc_size = lp.sumcheck_proof.serialized_size(Compress::No);
        eprintln!(
            "[{label}]   L{i}: w_commitment={} ring elems ({} bytes), sumcheck={} bytes",
            lp.w_commitment.u.len(),
            w_comm_size,
            sc_size,
        );
    }
    eprintln!(
        "[{label}]   final_w: {} elems, {} bits/elem, packed {} bytes",
        proof.final_w.num_elems,
        proof.final_w.bits_per_elem,
        proof.final_w.serialized_size(Compress::No),
    );
}

fn run_prove_onehot(
    label: &str,
    setup: &<Scheme as CommitmentScheme<F, D>>::ProverSetup,
    onehot_poly: &OneHotPoly<F, D>,
    _dense_poly: &DensePoly<F, D>,
    pt: &[F],
    opening: F,
    layout: &HachiCommitmentLayout,
) {
    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme as CommitmentScheme<F, D>>::commit(onehot_poly, setup, layout).unwrap();
    eprintln!(
        "[{label}] onehot commit: {:.3}s",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme as CommitmentScheme<F, D>>::prove(
        setup,
        onehot_poly,
        pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
        layout,
    )
    .unwrap();
    eprintln!("[{label}] prove: {:.3}s", t0.elapsed().as_secs_f64());
    print_proof_summary(label, &proof);

    let t0 = Instant::now();
    let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
    <Scheme as CommitmentScheme<F, D>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        pt,
        &opening,
        &commitment,
        BasisMode::Lagrange,
        layout,
    )
    .unwrap();
    eprintln!("[{label}] verify: {:.3}s", t0.elapsed().as_secs_f64());
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    let trace_dir = "profile_traces";
    fs::create_dir_all(trace_dir).ok();

    let log_basis: u32 = 3;
    let b = 1u32 << log_basis;

    let nv = {
        let alpha = D.trailing_zeros() as usize;
        let layout = ProfileCfg::commitment_layout(0).expect("layout");
        layout.m_vars + layout.r_vars + alpha
    };

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let trace_file = format!("{trace_dir}/hachi_nv{nv}_b{b}_{timestamp}.json");

    let (chrome_layer, _guard) = ChromeLayerBuilder::new()
        .include_args(true)
        .file(&trace_file)
        .build();

    tracing_subscriber::registry().with(chrome_layer).init();

    eprintln!("Perfetto trace will be written to: {trace_file}");
    eprintln!("Open at https://ui.perfetto.dev/");
    eprintln!("num_vars = {nv}, b = {b}");

    let len = 1usize << nv;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
    let pt: Vec<F> = (0..nv).map(|i| F::from_u64((i + 2) as u64)).collect();

    let opening = {
        let mut weights = vec![F::from_u64(0); len];
        weights[0] = F::from_u64(1);
        for (k, &x) in pt.iter().enumerate() {
            let half = 1usize << k;
            for i in (0..half).rev() {
                weights[i + half] = weights[i] * x;
                weights[i] = weights[i] - weights[i + half];
            }
        }
        evals
            .iter()
            .zip(weights.iter())
            .fold(F::from_u64(0), |a, (&e, &w)| a + e * w)
    };

    let t0 = Instant::now();
    let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(nv);
    eprintln!("setup: {:.3}s", t0.elapsed().as_secs_f64());

    let ab_mode = env::var("HACHI_AB_TEST").unwrap_or_default();

    let layout = ProfileCfg::commitment_layout(0).expect("layout");

    if ab_mode == "1" {
        eprintln!("\n=== A/B TEST: running both kernels ===\n");

        env::set_var("HACHI_NORM_KERNEL", "affine_coeff");
        eprintln!("--- kernel: affine_coeff ---");
        run_prove("affine", &setup, &poly, &pt, opening, &layout);

        env::set_var("HACHI_NORM_KERNEL", "point_eval");
        eprintln!("\n--- kernel: point_eval ---");
        run_prove("point", &setup, &poly, &pt, opening, &layout);

        env::remove_var("HACHI_NORM_KERNEL");
    } else {
        eprintln!(
            "kernel: {:?} (set HACHI_AB_TEST=1 to compare both)",
            choose_round_kernel(b as usize)
        );
        run_prove("default", &setup, &poly, &pt, opening, &layout);
    }

    eprintln!("\n--- one-hot commit path ---");
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;
    let num_chunks = total_ring; // K == D, so each ring element is one chunk

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..num_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();
    let onehot_poly =
        OneHotPoly::<F, D>::new(onehot_k, indices.clone(), layout.r_vars, layout.m_vars).unwrap();

    let dense_evals: Vec<F> = {
        let mut evals = vec![F::from_u64(0); num_chunks * onehot_k];
        for (ci, opt_idx) in indices.iter().enumerate() {
            if let Some(idx) = opt_idx {
                evals[ci * onehot_k + idx] = F::from_u64(1);
            }
        }
        evals
    };
    let dense_poly_oh = DensePoly::<F, D>::from_field_evals(nv, &dense_evals).unwrap();
    let opening_oh = {
        let mut weights = vec![F::from_u64(0); len];
        weights[0] = F::from_u64(1);
        for (k, &x) in pt.iter().enumerate() {
            let half = 1usize << k;
            for i in (0..half).rev() {
                weights[i + half] = weights[i] * x;
                weights[i] = weights[i] - weights[i + half];
            }
        }
        dense_evals
            .iter()
            .zip(weights.iter())
            .fold(F::from_u64(0), |a, (&e, &w)| a + e * w)
    };
    run_prove_onehot(
        "onehot",
        &setup,
        &onehot_poly,
        &dense_poly_oh,
        &pt,
        opening_oh,
        &layout,
    );

    eprintln!("\nDone. Trace saved to {trace_file}");
}
