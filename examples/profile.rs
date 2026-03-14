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
use hachi_pcs::protocol::proof::{
    FlatLabradorLevelProof, FlatLabradorWitness, HachiLevelProof, HachiProof, HachiProofTail,
    LabradorTail,
};
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
    let top_levels_len_size = std::mem::size_of::<u32>();
    let top_tail_tag_size = std::mem::size_of::<u8>();
    let hachi_levels_total: usize = proof
        .levels
        .iter()
        .map(|level| level.serialized_size(Compress::No))
        .sum();
    let tail_total = match &proof.tail {
        HachiProofTail::Direct(final_w) => final_w.serialized_size(Compress::No),
        HachiProofTail::Labrador(tail) => tail.serialized_size(Compress::No),
    };
    let accounted_total = top_levels_len_size + top_tail_tag_size + hachi_levels_total + tail_total;

    tracing::info!(
        label,
        levels = proof.levels.len(),
        proof_size_bytes = proof.size(),
        accounted_bytes = accounted_total,
        hachi_fold_bytes = hachi_levels_total,
        tail_bytes = tail_total,
        "proof summary"
    );
    debug_assert_eq!(accounted_total, proof.size());
    eprintln!(
        "[{label}]   proof framing: levels_len={} bytes, tail_tag={} byte",
        top_levels_len_size, top_tail_tag_size
    );

    for (i, lp) in proof.levels.iter().enumerate() {
        print_hachi_level_breakdown(label, i, lp);
    }
    match &proof.tail {
        HachiProofTail::Direct(final_w) => {
            eprintln!(
                "[{label}]   final_w: total={} bytes, elems={}, bits/elem={}",
                final_w.serialized_size(Compress::No),
                final_w.num_elems,
                final_w.bits_per_elem,
            );
        }
        HachiProofTail::Labrador(tail) => {
            print_labrador_tail_breakdown(label, tail);
        }
    }
}

fn print_hachi_level_breakdown(label: &str, level_idx: usize, level: &HachiLevelProof<F>) -> usize {
    let y_ring_size = level.y_ring.serialized_size(Compress::No);
    let v_size = level.v.serialized_size(Compress::No);
    let sumcheck_size = level.sumcheck_proof.serialized_size(Compress::No);
    let w_commitment_size = level.w_commitment.serialized_size(Compress::No);
    let w_eval_size = level.w_eval.serialized_size(Compress::No);
    let total = level.serialized_size(Compress::No);

    eprintln!("[{label}]   hachi_fold L{level_idx}: total={total} bytes");
    eprintln!(
        "[{label}]     y_ring={} bytes ({} ring elems, D={})",
        y_ring_size,
        level.y_ring.count(),
        level.y_ring.ring_dim(),
    );
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        level.v.count(),
        level.v.ring_dim(),
    );
    eprintln!("[{label}]     sumcheck={} bytes", sumcheck_size);
    eprintln!(
        "[{label}]     w_commitment={} bytes ({} ring elems, D={})",
        w_commitment_size,
        level.w_commitment.count(),
        level.w_commit_d(),
    );
    eprintln!("[{label}]     w_eval={} bytes", w_eval_size);

    debug_assert_eq!(
        total,
        y_ring_size + v_size + sumcheck_size + w_commitment_size + w_eval_size
    );
    total
}

fn print_labrador_tail_breakdown(label: &str, tail: &LabradorTail<F>) -> usize {
    let labrador_proof_size = tail.labrador_proof.serialized_size(Compress::No);
    let v_size = tail.v.serialized_size(Compress::No);
    let y_ring_size = tail.y_ring.serialized_size(Compress::No);
    let beta_sq_size = tail.beta_sq.serialized_size(Compress::No);
    let total = tail.serialized_size(Compress::No);

    eprintln!("[{label}]   final_w: Labrador tail");
    eprintln!("[{label}]   labrador_tail: total={total} bytes");
    eprintln!("[{label}]     labrador_proof={} bytes", labrador_proof_size);
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        tail.v.count(),
        tail.v.ring_dim(),
    );
    eprintln!(
        "[{label}]     y_ring={} bytes ({} ring elems, D={})",
        y_ring_size,
        tail.y_ring.count(),
        tail.y_ring.ring_dim(),
    );
    eprintln!("[{label}]     beta_sq={} bytes", beta_sq_size);
    debug_assert_eq!(
        total,
        labrador_proof_size + v_size + y_ring_size + beta_sq_size
    );

    let labrador_levels_len_size = std::mem::size_of::<u32>();
    let labrador_levels_total: usize = tail
        .labrador_proof
        .levels
        .iter()
        .map(|level| level.serialized_size(Compress::No))
        .sum();
    let final_opening_witness_size = tail
        .labrador_proof
        .final_opening_witness
        .serialized_size(Compress::No);
    let labrador_accounted =
        labrador_levels_len_size + labrador_levels_total + final_opening_witness_size;
    eprintln!(
        "[{label}]   labrador_fold: levels={}, total={} bytes, levels_len={} bytes, final_opening_witness={} bytes",
        tail.labrador_proof.levels.len(),
        labrador_proof_size,
        labrador_levels_len_size,
        final_opening_witness_size,
    );
    debug_assert_eq!(labrador_proof_size, labrador_accounted);

    for (i, level) in tail.labrador_proof.levels.iter().enumerate() {
        print_labrador_level_breakdown(label, i, level);
    }
    print_labrador_final_witness_breakdown(label, &tail.labrador_proof.final_opening_witness);

    total
}

fn print_labrador_level_breakdown(
    label: &str,
    level_idx: usize,
    level: &FlatLabradorLevelProof<F>,
) -> usize {
    let tail_flag_size = std::mem::size_of::<u8>();
    let input_row_lengths_size = level.input_row_lengths.serialized_size(Compress::No);
    let config_size = level.config.serialized_size(Compress::No);
    let nn_size = level.nn.serialized_size(Compress::No);
    let nu_size = level.nu.serialized_size(Compress::No);
    let u1_size = level.u1.serialized_size(Compress::No);
    let u2_size = level.u2.serialized_size(Compress::No);
    let jl_projection_size = level.jl_projection.len() * std::mem::size_of::<i64>();
    let jl_nonce_size = level.jl_nonce.serialized_size(Compress::No);
    let bb_size = level.bb.serialized_size(Compress::No);
    let norm_sq_size = level.norm_sq.serialized_size(Compress::No);
    let total = level.serialized_size(Compress::No);

    eprintln!(
        "[{label}]     labrador_fold L{level_idx}: total={total} bytes, tail={}",
        level.tail
    );
    eprintln!(
        "[{label}]       params: input_row_lengths={:?}, nn={}, rr={}, nu={:?}, f={}, b={}, fu={}, bu={}, kappa={}, kappa1={}",
        level.input_row_lengths,
        level.nn,
        level.nu.iter().sum::<usize>(),
        level.nu,
        level.config.f,
        level.config.b,
        level.config.fu,
        level.config.bu,
        level.config.kappa,
        level.config.kappa1,
    );
    eprintln!(
        "[{label}]       framing: tail_flag={}, input_row_lengths={}, config={}, nn={}, nu={}, norm_sq={}",
        tail_flag_size, input_row_lengths_size, config_size, nn_size, nu_size, norm_sq_size,
    );
    eprintln!(
        "[{label}]       msg u1={} bytes ({} ring elems, D={})",
        u1_size,
        level.u1.count(),
        level.u1.ring_dim(),
    );
    eprintln!(
        "[{label}]       msg u2={} bytes ({} ring elems, D={})",
        u2_size,
        level.u2.count(),
        level.u2.ring_dim(),
    );
    eprintln!(
        "[{label}]       msg jl_projection={} bytes, jl_nonce={} bytes",
        jl_projection_size, jl_nonce_size
    );
    eprintln!(
        "[{label}]       msg bb={} bytes ({} ring elems, D={})",
        bb_size,
        level.bb.count(),
        level.bb.ring_dim(),
    );

    debug_assert_eq!(
        total,
        tail_flag_size
            + input_row_lengths_size
            + config_size
            + nn_size
            + nu_size
            + u1_size
            + u2_size
            + jl_projection_size
            + jl_nonce_size
            + bb_size
            + norm_sq_size
    );
    total
}

fn print_labrador_final_witness_breakdown(label: &str, witness: &FlatLabradorWitness<F>) -> usize {
    let rows_len_size = std::mem::size_of::<u32>();
    let rows_total: usize = witness
        .rows
        .iter()
        .map(|row| row.serialized_size(Compress::No))
        .sum();
    let total = witness.serialized_size(Compress::No);

    eprintln!(
        "[{label}]     final_opening_witness: total={} bytes, rows_len={} bytes",
        total, rows_len_size
    );
    for (row_idx, row) in witness.rows.iter().enumerate() {
        eprintln!(
            "[{label}]       row{row_idx}={} bytes ({} ring elems, D={})",
            row.serialized_size(Compress::No),
            row.count(),
            row.ring_dim(),
        );
    }
    debug_assert_eq!(total, rows_len_size + rows_total);
    total
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

    if cfg!(debug_assertions) && env::var("HACHI_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        eprintln!("examples/profile must be run with --release for meaningful timings.");
        eprintln!("Re-run with: cargo run --release --example profile");
        eprintln!("Set HACHI_ALLOW_DEBUG_PROFILE=1 to override this guard.");
        std::process::exit(2);
    }

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
