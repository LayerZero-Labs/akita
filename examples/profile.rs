#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::primitives::serialization::Compress;
use hachi_pcs::protocol::commitment::{
    hachi_batched_root_layout, Fp128BoundedCommitmentConfig, Fp128D64BoundedCommitmentConfig,
    Fp128FullCommitmentConfig, Fp128LogBasisCommitmentConfig, Fp128OneHotCommitmentConfig,
    HachiCommitmentLayout,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
};
use hachi_pcs::protocol::proof::{
    HachiBatchedProof, HachiBatchedRootProof, HachiLevelProof, HachiProof,
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
use tracing_subscriber::EnvFilter;

type F = Fp128<0xffffffffffffffffffffffffffffe941>;
const ONEHOT_K: usize = 256;

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| value != "0")
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn opening_from_poly<const D: usize, P: HachiPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &HachiCommitmentLayout,
    basis: BasisMode,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point =
        ring_opening_point_from_field(reduced_point, layout.r_vars, layout.m_vars, basis)
            .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

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
    let hachi_levels_total: usize = proof
        .levels
        .iter()
        .map(|level| level.serialized_size(Compress::No))
        .sum();
    let tail_total = proof.tail.direct.serialized_size(Compress::No);
    let accounted_total = top_levels_len_size + hachi_levels_total + tail_total;

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
    eprintln!("[{label}]   proof framing: levels_len={top_levels_len_size} bytes");

    for (i, lp) in proof.levels.iter().enumerate() {
        print_hachi_level_breakdown(label, i, lp);
    }
    let final_w = &proof.tail.direct;
    eprintln!(
        "[{label}]   final_w: total={} bytes, elems={}, bits/elem={}",
        final_w.serialized_size(Compress::No),
        final_w.num_elems,
        final_w.bits_per_elem,
    );
}

fn ring_elem_count(coeff_len: usize, d: usize) -> usize {
    coeff_len / d
}

fn print_hachi_level_breakdown(label: &str, level_idx: usize, level: &HachiLevelProof<F>) -> usize {
    let level_d = level.level_d();
    let y_ring_size = level.y_ring.serialized_size(Compress::No);
    let v_size = level.v.serialized_size(Compress::No);
    let total = level.serialized_size(Compress::No);

    eprintln!("[{label}]   hachi_fold L{level_idx}: total={total} bytes");
    eprintln!(
        "[{label}]     y_ring={} bytes ({} ring elems, D={})",
        y_ring_size, 1, level_d,
    );
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        ring_elem_count(level.v.coeff_len(), level_d),
        level_d,
    );
    let stage1 = &level.stage1;
    let stage2 = &level.stage2;
    let stage1_sumcheck_size = stage1.sumcheck.serialized_size(Compress::No);
    let stage1_s_claim_size = stage1.s_claim.serialized_size(Compress::No);
    let stage2_sumcheck_size = stage2.sumcheck.serialized_size(Compress::No);
    let next_w_commitment_size = stage2.next_w_commitment.serialized_size(Compress::No);
    let next_w_eval_size = stage2.next_w_eval.serialized_size(Compress::No);
    eprintln!("[{label}]     stage1_sumcheck={stage1_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_s_claim={stage1_s_claim_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     next_w_commitment={next_w_commitment_size} bytes ({} coeffs)",
        stage2.next_w_commitment.coeff_len(),
    );
    eprintln!("[{label}]     next_w_eval={next_w_eval_size} bytes");
    debug_assert_eq!(
        total,
        y_ring_size
            + v_size
            + stage1_sumcheck_size
            + stage1_s_claim_size
            + stage2_sumcheck_size
            + next_w_commitment_size
            + next_w_eval_size
    );
    total
}

fn print_batched_root_breakdown<const D: usize>(
    label: &str,
    root: &HachiBatchedRootProof<F>,
) -> usize {
    let y_rings_size = root.y_rings.serialized_size(Compress::No);
    let v_size = root.v.serialized_size(Compress::No);
    let total = root.serialized_size(Compress::No);
    let stage1 = &root.stage1;
    let stage2 = &root.stage2;
    let stage1_sumcheck_size = stage1.sumcheck.serialized_size(Compress::No);
    let stage1_s_claim_size = stage1.s_claim.serialized_size(Compress::No);
    let stage2_sumcheck_size = stage2.sumcheck.serialized_size(Compress::No);
    let next_w_commitment_size = stage2.next_w_commitment.serialized_size(Compress::No);
    let next_w_eval_size = stage2.next_w_eval.serialized_size(Compress::No);

    eprintln!("[{label}]   batched_root: total={total} bytes");
    eprintln!(
        "[{label}]     y_rings={} bytes ({} ring elems, D={})",
        y_rings_size,
        ring_elem_count(root.y_rings.coeff_len(), D),
        D,
    );
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        ring_elem_count(root.v.coeff_len(), D),
        D,
    );
    eprintln!("[{label}]     stage1_sumcheck={stage1_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_s_claim={stage1_s_claim_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     next_w_commitment={next_w_commitment_size} bytes ({} coeffs)",
        stage2.next_w_commitment.coeff_len(),
    );
    eprintln!("[{label}]     next_w_eval={next_w_eval_size} bytes");
    debug_assert_eq!(
        total,
        y_rings_size
            + v_size
            + stage1_sumcheck_size
            + stage1_s_claim_size
            + stage2_sumcheck_size
            + next_w_commitment_size
            + next_w_eval_size
    );
    total
}

fn print_batched_proof_summary<const D: usize>(label: &str, proof: &HachiBatchedProof<F>) {
    let root_total = proof.root.serialized_size(Compress::No);
    let recursive_levels_total: usize = proof
        .levels
        .iter()
        .map(|level| level.serialized_size(Compress::No))
        .sum();
    let hachi_levels_total = root_total + recursive_levels_total;
    let tail_total = proof.tail.direct.serialized_size(Compress::No);
    let accounted_total = hachi_levels_total + tail_total;

    tracing::info!(
        label,
        levels = proof.levels.len() + 1,
        proof_size_bytes = proof.size(),
        accounted_bytes = accounted_total,
        hachi_fold_bytes = hachi_levels_total,
        tail_bytes = tail_total,
        "proof summary"
    );
    debug_assert_eq!(accounted_total, proof.size());
    print_batched_root_breakdown::<D>(label, &proof.root);
    for (i, lp) in proof.levels.iter().enumerate() {
        print_hachi_level_breakdown(label, i + 1, lp);
    }
    let final_w = &proof.tail.direct;
    eprintln!(
        "[{label}]   final_w: total={} bytes, elems={}, bits/elem={}",
        final_w.serialized_size(Compress::No),
        final_w.num_elems,
        final_w.bits_per_elem,
    );
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
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let (poly, opening) = {
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
        let opening = opening_from_poly(&poly, &pt, layout, BasisMode::Lagrange);
        (poly, opening)
    };

    let t0 = Instant::now();
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv, 1);
    tracing::info!(
        label = "dense",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "setup"
    );

    run_prove::<D, Cfg, _>("dense", &setup, &poly, &pt, opening, layout);
}

fn run_onehot<const D: usize, Cfg: CommitmentConfig>(nv: usize, layout: &HachiCommitmentLayout) {
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(D)
        .expect("total field size overflow");
    let onehot_k = ONEHOT_K;
    let total_chunks = total_field / onehot_k;
    assert_eq!(
        total_chunks * onehot_k,
        total_field,
        "onehot K must divide total field size"
    );

    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    let onehot_poly =
        OneHotPoly::<F, D, u8>::new(onehot_k, indices, layout.r_vars, layout.m_vars).unwrap();
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = opening_from_poly(&onehot_poly, &pt, layout, BasisMode::Lagrange);

    let t0 = Instant::now();
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv, 1);
    tracing::info!(
        label = "onehot",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "setup"
    );

    run_prove::<D, Cfg, _>("onehot", &setup, &onehot_poly, &pt, opening, layout);
}

fn run_batched_onehot<const D: usize, Cfg: CommitmentConfig>(
    nv: usize,
    num_polys: usize,
    layout: &HachiCommitmentLayout,
) {
    type Scheme<const D: usize, Cfg> = HachiCommitmentScheme<D, Cfg>;

    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(D)
        .expect("total field size overflow");
    let onehot_k = ONEHOT_K;
    let total_chunks = total_field / onehot_k;
    assert_eq!(
        total_chunks * onehot_k,
        total_field,
        "onehot K must divide total field size"
    );

    let polys: Vec<OneHotPoly<F, D, u8>> = (0..num_polys)
        .map(|poly_idx| {
            let mut rng = StdRng::seed_from_u64(0xbeef_cafe ^ ((poly_idx as u64 + 1) << 32));
            let indices: Vec<Option<u8>> = (0..total_chunks)
                .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
                .collect();
            OneHotPoly::<F, D, u8>::new(onehot_k, indices, layout.r_vars, layout.m_vars).unwrap()
        })
        .collect();
    let mut point_rng = StdRng::seed_from_u64(0xfeed_face);
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(point_rng.gen::<u128>()))
        .collect();
    let openings: Vec<F> = polys
        .iter()
        .map(|poly| opening_from_poly(poly, &pt, layout, BasisMode::Lagrange))
        .collect();
    let poly_refs: Vec<&OneHotPoly<F, D, u8>> = polys.iter().collect();
    let poly_groups = [&poly_refs[..]];
    let opening_groups = [&openings[..]];

    let t0 = Instant::now();
    let setup = <Scheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv, num_polys);
    tracing::info!(
        label = "onehot",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "setup"
    );

    let t0 = Instant::now();
    let (commitments, hints) =
        <Scheme<D, Cfg> as CommitmentScheme<F, D>>::batched_commit(&poly_groups, &setup, layout)
            .unwrap();
    tracing::info!(
        label = "onehot",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "commit"
    );

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentScheme<F, D>>::batched_prove(
        &setup,
        &poly_groups,
        &pt,
        hints,
        &mut prover_transcript,
        &commitments,
        BasisMode::Lagrange,
        layout,
    )
    .unwrap();
    tracing::info!(
        label = "onehot",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "prove"
    );
    print_batched_proof_summary::<D>("onehot", &proof);

    let t0 = Instant::now();
    let verifier_setup = <Scheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
    match <Scheme<D, Cfg> as CommitmentScheme<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        &pt,
        &opening_groups,
        &commitments,
        BasisMode::Lagrange,
        layout,
    ) {
        Ok(()) => tracing::info!(
            label = "onehot",
            elapsed_s = t0.elapsed().as_secs_f64(),
            "verify OK"
        ),
        Err(e) => {
            tracing::error!(label = "onehot", elapsed_s = t0.elapsed().as_secs_f64(), error = %e, "verify FAILED")
        }
    }
}

fn run_dense_mode<const D: usize, Cfg: CommitmentConfig>(title: &str, nv: usize) {
    let layout = resolve_layout::<Cfg>(nv);
    tracing::info!("{}", title);
    print_layout(&layout);
    run_dense::<D, Cfg>(nv, &layout);
}

fn run_onehot_mode<const D: usize, Cfg: CommitmentConfig>(
    title: &str,
    nv: usize,
    num_polys: usize,
) {
    let layout = if num_polys == 1 {
        resolve_layout::<Cfg>(nv)
    } else {
        hachi_batched_root_layout::<Cfg, D>(nv, num_polys).expect("layout")
    };
    tracing::info!("{}", title);
    print_layout(&layout);
    if num_polys == 1 {
        run_onehot::<D, Cfg>(nv, &layout);
    } else {
        run_batched_onehot::<D, Cfg>(nv, num_polys, &layout);
    }
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

    let nv: usize = env::var("HACHI_NUM_VARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let num_polys = env_usize("HACHI_NUM_POLYS", 1);

    let mode = env::var("HACHI_MODE").unwrap_or_else(|_| "full".to_string());
    let enable_trace = env_flag("HACHI_PROFILE_TRACE", true);
    let enable_ansi = env_flag("HACHI_PROFILE_ANSI", true);
    let span_events = if env_flag("HACHI_PROFILE_SPAN_CLOSES", true) {
        FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };
    let log_filter =
        EnvFilter::try_new(env::var("HACHI_PROFILE_LOG").unwrap_or_else(|_| "trace".to_string()))
            .unwrap_or_else(|_| EnvFilter::new("trace"));

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let trace_file = if num_polys == 1 {
        format!("profile_traces/hachi_nv{nv}_{mode}_{timestamp}.json")
    } else {
        format!("profile_traces/hachi_nv{nv}_np{num_polys}_{mode}_{timestamp}.json")
    };

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(enable_ansi)
        .with_span_events(span_events)
        .compact()
        .with_target(false);
    let _chrome_guard = if enable_trace {
        fs::create_dir_all("profile_traces").ok();
        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .include_args(true)
            .file(&trace_file)
            .build();
        tracing_subscriber::registry()
            .with(log_filter)
            .with(fmt_layer)
            .with(chrome_layer)
            .init();
        tracing::info!(trace_file = %trace_file, "Perfetto trace");
        Some(guard)
    } else {
        tracing_subscriber::registry()
            .with(log_filter)
            .with(fmt_layer)
            .init();
        tracing::info!("Perfetto trace disabled");
        None
    };
    tracing::info!(num_vars = nv, num_polys, mode = %mode, "profile config");

    match mode.as_str() {
        "full" => {
            type Cfg = Fp128FullCommitmentConfig;
            run_dense_mode::<{ Fp128FullCommitmentConfig::D }, Cfg>(
                "=== full (D=128 dense, log_commit_bound=128) ===",
                nv,
            );
        }
        "onehot" => {
            type Cfg = Fp128OneHotCommitmentConfig;
            let title = if num_polys == 1 {
                "=== onehot (D=64, 1-of-256, log_commit_bound=1) ===".to_string()
            } else {
                format!(
                    "=== onehot batched (D=64, 1-of-256, log_commit_bound=1, same-point batch={num_polys}) ==="
                )
            };
            run_onehot_mode::<{ Fp128OneHotCommitmentConfig::D }, Cfg>(&title, nv, num_polys);
        }
        "logbasis" => {
            type Cfg = Fp128LogBasisCommitmentConfig;
            run_dense_mode::<{ Fp128LogBasisCommitmentConfig::D }, Cfg>(
                "=== logbasis (D=128 dense, log_commit_bound=3) ===",
                nv,
            );
        }
        "all" => {
            {
                type Cfg = Fp128FullCommitmentConfig;
                run_dense_mode::<{ Fp128FullCommitmentConfig::D }, Cfg>(
                    "=== full (D=128 dense, log_commit_bound=128) ===",
                    nv,
                );
            }
            {
                type Cfg = Fp128OneHotCommitmentConfig;
                run_onehot_mode::<{ Fp128OneHotCommitmentConfig::D }, Cfg>(
                    "=== onehot (D=64, 1-of-256, log_commit_bound=1) ===",
                    nv,
                    1,
                );
            }
            {
                type Cfg = Fp128LogBasisCommitmentConfig;
                run_dense_mode::<{ Fp128LogBasisCommitmentConfig::D }, Cfg>(
                    "=== logbasis (D=128 dense, log_commit_bound=3) ===",
                    nv,
                );
            }
        }
        "compare_onehot" => {
            {
                type Cfg = Fp128D64BoundedCommitmentConfig<1, 3, 3>;
                run_onehot_mode::<{ Cfg::D }, Cfg>(
                    "=== [A] onehot (D=64, 1-of-256), basis=3 everywhere ===",
                    nv,
                    1,
                );
            }
            {
                type Cfg = Fp128D64BoundedCommitmentConfig<1, 2, 2>;
                run_onehot_mode::<{ Cfg::D }, Cfg>(
                    "=== [B] onehot (D=64, 1-of-256), basis=2 everywhere ===",
                    nv,
                    1,
                );
            }
            {
                type Cfg = Fp128D64BoundedCommitmentConfig<1, 2, 3>;
                run_onehot_mode::<{ Cfg::D }, Cfg>(
                    "=== [C] onehot (D=64, 1-of-256), L0 basis=2, w-levels basis=3 ===",
                    nv,
                    1,
                );
            }
            {
                type Cfg = Fp128D64BoundedCommitmentConfig<1, 2, 4>;
                run_onehot_mode::<{ Cfg::D }, Cfg>(
                    "=== [D] onehot (D=64, 1-of-256), L0 basis=2, w-levels basis=4 ===",
                    nv,
                    1,
                );
            }
        }
        "compare_logbasis" => {
            {
                type Cfg = Fp128BoundedCommitmentConfig<3, 3, 3>;
                run_dense_mode::<{ Cfg::D }, Cfg>(
                    "=== [A] logbasis coeffs (D=128), basis=3 everywhere ===",
                    nv,
                );
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<3, 2, 2>;
                run_dense_mode::<{ Cfg::D }, Cfg>(
                    "=== [B] logbasis coeffs (D=128), basis=2 everywhere ===",
                    nv,
                );
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<3, 2, 3>;
                run_dense_mode::<{ Cfg::D }, Cfg>(
                    "=== [C] logbasis coeffs (D=128), L0 basis=2, w-levels basis=3 ===",
                    nv,
                );
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<3, 2, 4>;
                run_dense_mode::<{ Cfg::D }, Cfg>(
                    "=== [D] logbasis coeffs (D=128), L0 basis=2, w-levels basis=4 ===",
                    nv,
                );
            }
        }
        "compare_basis" => {
            {
                type Cfg = Fp128BoundedCommitmentConfig<128, 3, 3>;
                run_dense_mode::<{ Cfg::D }, Cfg>(
                    "=== [A] baseline (D=128): log_basis=3 everywhere ===",
                    nv,
                );
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<128, 2, 2>;
                run_dense_mode::<{ Cfg::D }, Cfg>(
                    "=== [B] baseline (D=128): log_basis=2 everywhere ===",
                    nv,
                );
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<128, 2, 3>;
                run_dense_mode::<{ Cfg::D }, Cfg>(
                    "=== [C] baseline (D=128): L0 basis=2, w-levels basis=3 ===",
                    nv,
                );
            }
            {
                type Cfg = Fp128BoundedCommitmentConfig<128, 2, 4>;
                run_dense_mode::<{ Cfg::D }, Cfg>(
                    "=== [D] baseline (D=128): L0 basis=2, w-levels basis=4 ===",
                    nv,
                );
            }
        }
        other => {
            tracing::error!(
                mode = other,
                "Unknown HACHI_MODE. Use: full, onehot, logbasis, all, compare_onehot, compare_logbasis, compare_basis"
            );
            std::process::exit(1);
        }
    }

    if enable_trace {
        tracing::info!(trace_file = %trace_file, "Done. Trace saved");
    } else {
        tracing::info!("Done");
    }
}

fn resolve_layout<Cfg: CommitmentConfig>(nv: usize) -> HachiCommitmentLayout {
    Cfg::commitment_layout(nv).expect("layout")
}
