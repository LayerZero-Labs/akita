#![allow(missing_docs)]

use hachi_pcs::algebra::poly::multilinear_eval;
use hachi_pcs::algebra::Fp128;
use hachi_pcs::primitives::serialization::Compress;
use hachi_pcs::protocol::commitment::{
    Fp128FullCommitmentConfig, Fp128LogBasisCommitmentConfig, Fp128OneHotCommitmentConfig,
    HachiCommitmentLayout,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::proof::HachiProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentScheme, FromSmallInt, HachiSerialize, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;

fn run_prove<const D: usize, Cfg: CommitmentConfig, P: hachi_pcs::HachiPolyOps<F, D>>(
    label: &str,
    setup: &<HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::ProverSetup,
    poly: &P,
    pt: &[F],
    opening: F,
    layout: &HachiCommitmentLayout,
) {
    type Scheme<const D: usize, Cfg> = HachiCommitmentScheme<D, Cfg>;

    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentScheme<F, D>>::commit(poly, setup, layout).unwrap();
    eprintln!("[{label}] commit: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
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
    print_proof_summary::<D>(label, &proof);

    let t0 = Instant::now();
    let verifier_setup = <Scheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
    match <Scheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        pt,
        &opening,
        &commitment,
        BasisMode::Lagrange,
        layout,
    ) {
        Ok(()) => eprintln!("[{label}] verify: {:.3}s OK", t0.elapsed().as_secs_f64()),
        Err(e) => eprintln!(
            "[{label}] verify: {:.3}s FAILED ({e})",
            t0.elapsed().as_secs_f64()
        ),
    }
}

fn print_proof_summary<const D: usize>(label: &str, proof: &HachiProof<F, D>) {
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

fn print_layout(layout: &HachiCommitmentLayout) {
    eprintln!(
        "  layout: m_vars={}, r_vars={}, num_blocks={}, block_len={}, \
         delta_commit={}, delta_open={}, delta_fold={}, log_basis={}",
        layout.m_vars,
        layout.r_vars,
        layout.num_blocks,
        layout.block_len,
        layout.num_digits_commit,
        layout.num_digits_open,
        layout.num_digits_fold,
        layout.log_basis,
    );
}

fn run_dense<const D: usize, Cfg: CommitmentConfig>(nv: usize, layout: &HachiCommitmentLayout) {
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
    let evals: Vec<F> = if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        (0..len)
            .map(|_| F::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    };
    let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = multilinear_eval(&evals, &pt).unwrap();

    let t0 = Instant::now();
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);
    eprintln!("  setup: {:.3}s", t0.elapsed().as_secs_f64());

    run_prove::<D, Cfg, _>("dense", &setup, &poly, &pt, opening, layout);
}

fn run_onehot<const D: usize, Cfg: CommitmentConfig>(nv: usize, layout: &HachiCommitmentLayout) {
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;

    let indices: Vec<Option<usize>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();
    let onehot_poly =
        OneHotPoly::<F, D>::new(onehot_k, indices.clone(), layout.r_vars, layout.m_vars).unwrap();

    let onehot_evals: Vec<F> = {
        let mut evals = vec![F::from_u64(0); total_ring * onehot_k];
        for (ci, opt_idx) in indices.iter().enumerate() {
            if let Some(idx) = opt_idx {
                evals[ci * onehot_k + idx] = F::from_u64(1);
            }
        }
        evals
    };
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = multilinear_eval(&onehot_evals, &pt).unwrap();

    let t0 = Instant::now();
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);
    eprintln!("  setup: {:.3}s", t0.elapsed().as_secs_f64());

    run_prove::<D, Cfg, _>("onehot", &setup, &onehot_poly, &pt, opening, layout);
}

fn main() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    let trace_dir = "profile_traces";
    fs::create_dir_all(trace_dir).ok();

    let nv: usize = env::var("HACHI_NUM_VARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);

    let mode = env::var("HACHI_MODE").unwrap_or_else(|_| "full".to_string());

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let trace_file = format!("{trace_dir}/hachi_nv{nv}_{mode}_{timestamp}.json");

    let (chrome_layer, _guard) = ChromeLayerBuilder::new()
        .include_args(true)
        .file(&trace_file)
        .build();

    tracing_subscriber::registry().with(chrome_layer).init();

    eprintln!("Perfetto trace: {trace_file}");
    eprintln!("num_vars={nv}, mode={mode}");
    eprintln!();

    match mode.as_str() {
        "full" => {
            type Cfg = Fp128FullCommitmentConfig;
            let layout = resolve_layout::<Cfg>(nv);
            eprintln!("=== full (dense, log_commit_bound=128) ===");
            print_layout(&layout);
            run_dense::<{ Fp128FullCommitmentConfig::D }, Cfg>(nv, &layout);
        }
        "onehot" => {
            type Cfg = Fp128OneHotCommitmentConfig;
            let layout = resolve_layout::<Cfg>(nv);
            eprintln!("=== onehot (log_commit_bound=1) ===");
            print_layout(&layout);
            run_onehot::<{ Fp128OneHotCommitmentConfig::D }, Cfg>(nv, &layout);
        }
        "logbasis" => {
            type Cfg = Fp128LogBasisCommitmentConfig;
            let layout = resolve_layout::<Cfg>(nv);
            eprintln!("=== logbasis (dense, log_commit_bound=3) ===");
            print_layout(&layout);
            run_dense::<{ Fp128LogBasisCommitmentConfig::D }, Cfg>(nv, &layout);
        }
        "all" => {
            {
                type Cfg = Fp128FullCommitmentConfig;
                let layout = resolve_layout::<Cfg>(nv);
                eprintln!("=== full (dense, log_commit_bound=128) ===");
                print_layout(&layout);
                run_dense::<{ Fp128FullCommitmentConfig::D }, Cfg>(nv, &layout);
                eprintln!();
            }
            {
                type Cfg = Fp128OneHotCommitmentConfig;
                let layout = resolve_layout::<Cfg>(nv);
                eprintln!("=== onehot (log_commit_bound=1) ===");
                print_layout(&layout);
                run_onehot::<{ Fp128OneHotCommitmentConfig::D }, Cfg>(nv, &layout);
                eprintln!();
            }
            {
                type Cfg = Fp128LogBasisCommitmentConfig;
                let layout = resolve_layout::<Cfg>(nv);
                eprintln!("=== logbasis (dense, log_commit_bound=3) ===");
                print_layout(&layout);
                run_dense::<{ Fp128LogBasisCommitmentConfig::D }, Cfg>(nv, &layout);
            }
        }
        other => {
            eprintln!("Unknown HACHI_MODE={other}. Use: full, onehot, logbasis, all");
            std::process::exit(1);
        }
    }

    eprintln!("\nDone. Trace saved to {trace_file}");
}

fn resolve_layout<Cfg: CommitmentConfig>(nv: usize) -> HachiCommitmentLayout {
    Cfg::commitment_layout(nv).expect("layout")
}
