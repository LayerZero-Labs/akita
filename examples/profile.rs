#![allow(missing_docs)]

use hachi_pcs::algebra::poly::multilinear_eval;
use hachi_pcs::algebra::Fp128;
use hachi_pcs::primitives::serialization::Compress;
use hachi_pcs::protocol::commitment::{
    Fp128BoundedCommitmentConfig, Fp128FullCommitmentConfig, Fp128LogBasisCommitmentConfig,
    Fp128OneHotCommitmentConfig, HachiCommitmentLayout,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::proof::HachiProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentScheme, FromSmallInt, HachiPolyOps, HachiSerialize,
    Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;

fn run_prove<const D: usize, Cfg: CommitmentConfig, P: HachiPolyOps<F, D>>(
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
    tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "commit");

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
    tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "prove");
    print_proof_summary(label, &proof);

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
        Ok(()) => tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "verify OK"),
        Err(e) => {
            tracing::error!(label, elapsed_s = t0.elapsed().as_secs_f64(), error = %e, "verify FAILED")
        }
    }
}

fn print_proof_summary(label: &str, proof: &HachiProof<F>) {
    tracing::info!(
        label,
        levels = proof.levels.len(),
        proof_size_bytes = proof.size(),
        "proof summary"
    );
    for (i, lp) in proof.levels.iter().enumerate() {
        let w_comm_size = lp.w_commitment.serialized_size(Compress::No);
        let sc_size = lp.sumcheck_proof.serialized_size(Compress::No);
        tracing::debug!(
            label,
            level = i,
            w_commitment_elems = lp.w_commitment.count(),
            D = lp.w_commit_d(),
            w_commitment_bytes = w_comm_size,
            sumcheck_bytes = sc_size,
            "level detail"
        );
    }
    if let Some(final_w) = proof.final_w() {
        tracing::debug!(
            label,
            num_elems = final_w.num_elems,
            bits_per_elem = final_w.bits_per_elem,
            packed_bytes = final_w.serialized_size(Compress::No),
            "final_w"
        );
    } else {
        tracing::debug!(label, "final_w: Labrador tail");
    }
}

fn print_layout(layout: &HachiCommitmentLayout) {
    tracing::debug!(
        m_vars = layout.m_vars,
        r_vars = layout.r_vars,
        num_blocks = layout.num_blocks,
        block_len = layout.block_len,
        delta_commit = layout.num_digits_commit,
        delta_open = layout.num_digits_open,
        delta_fold = layout.num_digits_fold,
        log_basis = layout.log_basis,
        "layout"
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
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "setup");

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
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "setup");

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

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_span_events(FmtSpan::CLOSE)
        .compact()
        .with_target(false);

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(chrome_layer)
        .init();

    tracing::info!(trace_file = %trace_file, "Perfetto trace");
    tracing::info!(num_vars = nv, mode = %mode, "profile config");

    match mode.as_str() {
        "full" => {
            type Cfg = Fp128FullCommitmentConfig;
            let layout = resolve_layout::<Cfg>(nv);
            tracing::info!("=== full (dense, log_commit_bound=128) ===");
            print_layout(&layout);
            run_dense::<{ Fp128FullCommitmentConfig::D }, Cfg>(nv, &layout);
        }
        "onehot" => {
            type Cfg = Fp128OneHotCommitmentConfig;
            let layout = resolve_layout::<Cfg>(nv);
            tracing::info!("=== onehot (log_commit_bound=1) ===");
            print_layout(&layout);
            run_onehot::<{ Fp128OneHotCommitmentConfig::D }, Cfg>(nv, &layout);
        }
        "logbasis" => {
            type Cfg = Fp128LogBasisCommitmentConfig;
            let layout = resolve_layout::<Cfg>(nv);
            tracing::info!("=== logbasis (dense, log_commit_bound=3) ===");
            print_layout(&layout);
            run_dense::<{ Fp128LogBasisCommitmentConfig::D }, Cfg>(nv, &layout);
        }
        "all" => {
            {
                type Cfg = Fp128FullCommitmentConfig;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== full (dense, log_commit_bound=128) ===");
                print_layout(&layout);
                run_dense::<{ Fp128FullCommitmentConfig::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128OneHotCommitmentConfig;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== onehot (log_commit_bound=1) ===");
                print_layout(&layout);
                run_onehot::<{ Fp128OneHotCommitmentConfig::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128LogBasisCommitmentConfig;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== logbasis (dense, log_commit_bound=3) ===");
                print_layout(&layout);
                run_dense::<{ Fp128LogBasisCommitmentConfig::D }, Cfg>(nv, &layout);
            }
        }
        "compare_onehot" => {
            {
                type Cfg = Fp128BoundedCommitmentConfig<1, 3, 3>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [A] onehot, basis=3 everywhere ===");
                print_layout(&layout);
                run_onehot::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<1, 2, 2>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [B] onehot, basis=2 everywhere ===");
                print_layout(&layout);
                run_onehot::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<1, 2, 3>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [C] onehot, L0 basis=2, w-levels basis=3 ===");
                print_layout(&layout);
                run_onehot::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<1, 2, 4>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [D] onehot, L0 basis=2, w-levels basis=4 ===");
                print_layout(&layout);
                run_onehot::<{ Cfg::D }, Cfg>(nv, &layout);
            }
        }
        "compare_logbasis" => {
            {
                type Cfg = Fp128BoundedCommitmentConfig<3, 3, 3>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [A] logbasis coeffs, basis=3 everywhere ===");
                print_layout(&layout);
                run_dense::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<3, 2, 2>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [B] logbasis coeffs, basis=2 everywhere ===");
                print_layout(&layout);
                run_dense::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<3, 2, 3>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [C] logbasis coeffs, L0 basis=2, w-levels basis=3 ===");
                print_layout(&layout);
                run_dense::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<3, 2, 4>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [D] logbasis coeffs, L0 basis=2, w-levels basis=4 ===");
                print_layout(&layout);
                run_dense::<{ Cfg::D }, Cfg>(nv, &layout);
            }
        }
        "compare_basis" => {
            {
                type Cfg = Fp128BoundedCommitmentConfig<128, 3, 3>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [A] baseline: log_basis=3 everywhere ===");
                print_layout(&layout);
                run_dense::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<128, 2, 2>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [B] log_basis=2 everywhere ===");
                print_layout(&layout);
                run_dense::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<128, 2, 3>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [C] L0 basis=2, w-levels basis=3 ===");
                print_layout(&layout);
                run_dense::<{ Cfg::D }, Cfg>(nv, &layout);
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<128, 2, 4>;
                let layout = resolve_layout::<Cfg>(nv);
                tracing::info!("=== [D] L0 basis=2, w-levels basis=4 ===");
                print_layout(&layout);
                run_dense::<{ Cfg::D }, Cfg>(nv, &layout);
            }
        }
        other => {
            tracing::error!(
                mode = other,
                "Unknown HACHI_MODE. Use: full, onehot, logbasis, compare_basis, all"
            );
            std::process::exit(1);
        }
    }

    tracing::info!(trace_file = %trace_file, "Done. Trace saved");
}

fn resolve_layout<Cfg: CommitmentConfig>(nv: usize) -> HachiCommitmentLayout {
    Cfg::commitment_layout(nv).expect("layout")
}
