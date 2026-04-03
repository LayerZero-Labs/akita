#![allow(missing_docs)]

use hachi_pcs::primitives::serialization::Compress;
use hachi_pcs::protocol::commitment::{
    hachi_batched_root_layout, presets::fp128, CommitmentConfig, HachiCommitmentLayout,
    HachiSchedulePlan,
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
use hachi_pcs::{
    BasisMode, BlockOrder, CanonicalField, CommitmentScheme, DenseMultilinear,
    DynamicCommitmentScheme, DynamicHachiCommitmentScheme, DynamicRootConfigFamily, FromSmallInt,
    HachiPolyOps, HachiSerialize, MultilinearPolynomial, OneHotMultilinear, Transcript,
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

type F = fp128::Field;
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
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        basis,
        BlockOrder::RowMajor,
    )
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

fn run_prove<const D: usize, Cfg: CommitmentConfig<Field = F>, P: HachiPolyOps<F, D>>(
    label: &str,
    setup: &<HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::ProverSetup,
    poly: &P,
    pt: &[F],
    opening: F,
    plan: Option<&HachiSchedulePlan>,
) {
    type Scheme<const D: usize, Cfg> = HachiCommitmentScheme<D, Cfg>;

    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentScheme<F, D>>::commit(std::slice::from_ref(poly), setup)
            .unwrap();
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
    )
    .unwrap();
    tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "prove");
    print_proof_summary(label, &proof, plan);

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
    ) {
        Ok(()) => tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "verify OK"),
        Err(e) => {
            tracing::error!(label, elapsed_s = t0.elapsed().as_secs_f64(), error = %e, "verify FAILED")
        }
    }
}

fn emit_planned_schedule_summary(label: &str, plan: &HachiSchedulePlan) {
    tracing::info!(
        label,
        levels = plan.num_fold_levels(),
        exact_proof_bytes = plan.exact_proof_bytes,
        no_wrapper_bytes = plan.no_wrapper_bytes,
        "planned schedule"
    );

    for level in plan.fold_levels() {
        let next_w_len = level.next_inputs.current_w_len;
        tracing::info!(
            label,
            level = level.inputs.level,
            d = level.params.d,
            n_a = level.params.n_a,
            n_b = level.params.n_b,
            n_d = level.params.n_d,
            challenge_l1_mass = level.params.challenge_l1_mass,
            log_basis = level.params.log_basis,
            m_vars = level.layout.m_vars,
            r_vars = level.layout.r_vars,
            num_blocks = level.layout.num_blocks,
            block_len = level.layout.block_len,
            delta_commit = level.layout.num_digits_commit,
            delta_open = level.layout.num_digits_open,
            delta_fold = level.layout.num_digits_fold,
            current_w_len = level.inputs.current_w_len,
            next_w_ring = next_w_len / level.params.d,
            next_w_len,
            level_bytes = level.level_bytes,
            "planned fold level"
        );
    }

    let terminal = plan.terminal_state();
    tracing::info!(
        label,
        final_w_len = terminal.current_w_len,
        final_log_basis = terminal.log_basis,
        "planned terminal state"
    );
}

fn print_proof_summary(label: &str, proof: &HachiProof<F>, plan: Option<&HachiSchedulePlan>) {
    let hachi_levels_total: usize = proof
        .fold_levels()
        .map(|level| level.serialized_size(Compress::No))
        .sum();
    let tail_total = proof.final_w().serialized_size(Compress::No);
    let accounted_total = hachi_levels_total + tail_total;

    tracing::info!(
        label,
        levels = proof.num_fold_levels(),
        proof_size_bytes = proof.size(),
        accounted_bytes = accounted_total,
        hachi_fold_bytes = hachi_levels_total,
        tail_bytes = tail_total,
        "proof summary"
    );
    debug_assert_eq!(accounted_total, proof.size());

    if let Some(plan) = plan {
        debug_assert_eq!(
            proof.size(),
            plan.exact_proof_bytes,
            "runtime proof bytes should match the planned proof size"
        );
        emit_planned_schedule_summary(label, plan);
    }

    for (i, lp) in proof.fold_levels().enumerate() {
        print_hachi_level_breakdown(label, i, lp);
    }
    let final_w = proof.final_w();
    tracing::info!(
        label,
        tail_bytes = final_w.serialized_size(Compress::No),
        final_w_num_elems = final_w.num_elems,
        final_w_bits_per_elem = final_w.bits_per_elem,
        "proof tail summary"
    );
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
    let stage1_sumcheck_size = stage1
        .stages
        .iter()
        .map(|stage| stage.sumcheck.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_interstage_claims_size = stage1
        .stages
        .iter()
        .flat_map(|stage| stage.child_claims.iter())
        .map(|claim| claim.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_s_claim_size = stage1.s_claim.serialized_size(Compress::No);
    let stage2_sumcheck_size = stage2.sumcheck.serialized_size(Compress::No);
    let next_w_commitment_size = stage2.next_w_commitment.serialized_size(Compress::No);
    let next_w_eval_size = stage2.next_w_eval.serialized_size(Compress::No);
    tracing::info!(
        label,
        level = level_idx,
        d = level_d,
        total_bytes = total,
        y_ring_bytes = y_ring_size,
        v_bytes = v_size,
        stage1_sumcheck_bytes = stage1_sumcheck_size,
        stage1_interstage_claims_bytes = stage1_interstage_claims_size,
        stage1_s_claim_bytes = stage1_s_claim_size,
        stage2_sumcheck_bytes = stage2_sumcheck_size,
        next_w_commitment_bytes = next_w_commitment_size,
        next_w_eval_bytes = next_w_eval_size,
        "proof fold level"
    );
    eprintln!("[{label}]     stage1_sumcheck={stage1_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_interstage_claims={stage1_interstage_claims_size} bytes");
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
            + stage1_interstage_claims_size
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
    let stage1_sumcheck_size = stage1
        .stages
        .iter()
        .map(|stage| stage.sumcheck.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_interstage_claims_size = stage1
        .stages
        .iter()
        .flat_map(|stage| stage.child_claims.iter())
        .map(|claim| claim.serialized_size(Compress::No))
        .sum::<usize>();
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
    eprintln!("[{label}]     stage1_interstage_claims={stage1_interstage_claims_size} bytes");
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
            + stage1_interstage_claims_size
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
        .fold_levels()
        .map(|level| level.serialized_size(Compress::No))
        .sum();
    let hachi_levels_total = root_total + recursive_levels_total;
    let tail_total = proof.final_w().serialized_size(Compress::No);
    let accounted_total = hachi_levels_total + tail_total;

    tracing::info!(
        label,
        levels = proof.num_fold_levels() + 1,
        proof_size_bytes = proof.size(),
        accounted_bytes = accounted_total,
        hachi_fold_bytes = hachi_levels_total,
        tail_bytes = tail_total,
        "proof summary"
    );
    debug_assert_eq!(accounted_total, proof.size());
    print_batched_root_breakdown::<D>(label, &proof.root);
    for (i, lp) in proof.fold_levels().enumerate() {
        print_hachi_level_breakdown(label, i + 1, lp);
    }
    let final_w = proof.final_w();
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

fn run_dense<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    nv: usize,
    layout: &HachiCommitmentLayout,
    plan: Option<&HachiSchedulePlan>,
) {
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

    run_prove::<D, Cfg, _>("dense", &setup, &poly, &pt, opening, plan);
}

fn run_onehot<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    nv: usize,
    layout: &HachiCommitmentLayout,
    plan: Option<&HachiSchedulePlan>,
) {
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

    run_prove::<D, Cfg, _>("onehot", &setup, &onehot_poly, &pt, opening, plan);
}

fn run_batched_onehot<const D: usize, Cfg: CommitmentConfig<Field = F>>(
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
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentScheme<F, D>>::commit(&poly_refs, &setup).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];
    tracing::info!(
        label = "onehot",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "commit"
    );

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentScheme<F, D>>::batched_prove(
        &setup,
        &[&poly_groups[..]],
        &[&pt[..]],
        vec![hints],
        &mut prover_transcript,
        &[&commitments[..]],
        BasisMode::Lagrange,
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
        &[&pt[..]],
        &[&opening_groups[..]],
        &[&commitments[..]],
        BasisMode::Lagrange,
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

fn dynamic_singleton_plan<Family: DynamicRootConfigFamily<F>>(
    nv: usize,
    root_d: usize,
) -> Option<HachiSchedulePlan> {
    match root_d {
        32 => Family::Cfg32::schedule_plan(nv).expect("dynamic singleton schedule plan"),
        64 => Family::Cfg64::schedule_plan(nv).expect("dynamic singleton schedule plan"),
        128 => Family::Cfg128::schedule_plan(nv).expect("dynamic singleton schedule plan"),
        _ => unreachable!("dynamic schemes only select D in 32/64/128"),
    }
}

fn run_dynamic_dense_mode<Family>(label: &str, title: &str, nv: usize)
where
    Family: DynamicRootConfigFamily<F>,
{
    type Scheme<Family> = DynamicHachiCommitmentScheme<Family>;

    tracing::info!("{}", title);

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let decomp = Family::Cfg128::decomposition();
    let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
    let evals: Vec<F> = if decomp.log_commit_bound >= 128 {
        (0..(1usize << nv))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        (0..(1usize << nv))
            .map(|_| F::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    };
    let dense = DenseMultilinear::from_field_evals(nv, &evals).unwrap();
    let poly: MultilinearPolynomial<F> = dense.clone().into();

    let t0 = Instant::now();
    let setup = <Scheme<Family> as DynamicCommitmentScheme<F>>::setup_prover(nv, 1);
    tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "setup");

    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme<Family> as DynamicCommitmentScheme<F>>::commit(std::slice::from_ref(&poly), &setup)
            .unwrap();
    let root_d = commitment.root_ring_dim();
    tracing::info!(
        label,
        root_d,
        elapsed_s = t0.elapsed().as_secs_f64(),
        "commit"
    );

    let opening = match root_d {
        32 => {
            let layout = hachi_batched_root_layout::<Family::Cfg32, 32>(nv, 1).unwrap();
            let typed = dense.to_typed::<32>().unwrap();
            opening_from_poly(&typed, &pt, &layout, BasisMode::Lagrange)
        }
        64 => {
            let layout = hachi_batched_root_layout::<Family::Cfg64, 64>(nv, 1).unwrap();
            let typed = dense.to_typed::<64>().unwrap();
            opening_from_poly(&typed, &pt, &layout, BasisMode::Lagrange)
        }
        128 => {
            let layout = hachi_batched_root_layout::<Family::Cfg128, 128>(nv, 1).unwrap();
            let typed = dense.to_typed::<128>().unwrap();
            opening_from_poly(&typed, &pt, &layout, BasisMode::Lagrange)
        }
        _ => unreachable!("dynamic schemes only select D in 32/64/128"),
    };
    let plan = dynamic_singleton_plan::<Family>(nv, root_d);

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme<Family> as DynamicCommitmentScheme<F>>::prove(
        &setup,
        &poly,
        &pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
    )
    .unwrap();
    tracing::info!(
        label,
        root_d,
        elapsed_s = t0.elapsed().as_secs_f64(),
        "prove"
    );
    print_proof_summary(label, &proof, plan.as_ref());

    let t0 = Instant::now();
    let verifier_setup = <Scheme<Family> as DynamicCommitmentScheme<F>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
    match <Scheme<Family> as DynamicCommitmentScheme<F>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        &pt,
        &opening,
        &commitment,
        BasisMode::Lagrange,
    ) {
        Ok(()) => tracing::info!(
            label,
            root_d,
            elapsed_s = t0.elapsed().as_secs_f64(),
            "verify OK"
        ),
        Err(e) => tracing::error!(
            label,
            root_d,
            elapsed_s = t0.elapsed().as_secs_f64(),
            error = %e,
            "verify FAILED"
        ),
    }
}

fn run_dynamic_onehot_mode<Family>(label: &str, title: &str, nv: usize, num_polys: usize)
where
    Family: DynamicRootConfigFamily<F>,
{
    type Scheme<Family> = DynamicHachiCommitmentScheme<Family>;

    tracing::info!("{}", title);

    let onehots: Vec<OneHotMultilinear> = (0..num_polys)
        .map(|poly_idx| {
            let mut rng = StdRng::seed_from_u64(0xbeef_cafe ^ ((poly_idx as u64 + 1) << 32));
            let total_chunks = (1usize << nv) / ONEHOT_K;
            let indices: Vec<Option<usize>> = (0..total_chunks)
                .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                .collect();
            OneHotMultilinear::new(nv, ONEHOT_K, indices).unwrap()
        })
        .collect();
    let polys: Vec<MultilinearPolynomial<F>> = onehots
        .iter()
        .cloned()
        .map(MultilinearPolynomial::from)
        .collect();
    let mut point_rng = StdRng::seed_from_u64(0xfeed_face);
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(point_rng.gen::<u128>()))
        .collect();

    let t0 = Instant::now();
    let setup = <Scheme<Family> as DynamicCommitmentScheme<F>>::setup_prover(nv, num_polys);
    tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "setup");

    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme<Family> as DynamicCommitmentScheme<F>>::commit(&polys, &setup).unwrap();
    let root_d = commitment.root_ring_dim();
    tracing::info!(
        label,
        root_d,
        elapsed_s = t0.elapsed().as_secs_f64(),
        "commit"
    );

    let openings: Vec<F> = match root_d {
        32 => {
            let layout = hachi_batched_root_layout::<Family::Cfg32, 32>(nv, num_polys).unwrap();
            onehots
                .iter()
                .map(|onehot| {
                    let typed = onehot.to_typed::<F, 32>(layout).unwrap();
                    opening_from_poly(&typed, &pt, &layout, BasisMode::Lagrange)
                })
                .collect()
        }
        64 => {
            let layout = hachi_batched_root_layout::<Family::Cfg64, 64>(nv, num_polys).unwrap();
            onehots
                .iter()
                .map(|onehot| {
                    let typed = onehot.to_typed::<F, 64>(layout).unwrap();
                    opening_from_poly(&typed, &pt, &layout, BasisMode::Lagrange)
                })
                .collect()
        }
        128 => {
            let layout = hachi_batched_root_layout::<Family::Cfg128, 128>(nv, num_polys).unwrap();
            onehots
                .iter()
                .map(|onehot| {
                    let typed = onehot.to_typed::<F, 128>(layout).unwrap();
                    opening_from_poly(&typed, &pt, &layout, BasisMode::Lagrange)
                })
                .collect()
        }
        _ => unreachable!("dynamic schemes only select D in 32/64/128"),
    };

    let verifier_setup = <Scheme<Family> as DynamicCommitmentScheme<F>>::setup_verifier(&setup);
    if num_polys == 1 {
        let plan = dynamic_singleton_plan::<Family>(nv, root_d);
        let poly = &polys[0];
        let opening = openings[0];

        let t0 = Instant::now();
        let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
        let proof = <Scheme<Family> as DynamicCommitmentScheme<F>>::prove(
            &setup,
            poly,
            &pt,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
        )
        .unwrap();
        tracing::info!(
            label,
            root_d,
            elapsed_s = t0.elapsed().as_secs_f64(),
            "prove"
        );
        print_proof_summary(label, &proof, plan.as_ref());

        let t0 = Instant::now();
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
        match <Scheme<Family> as DynamicCommitmentScheme<F>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &pt,
            &opening,
            &commitment,
            BasisMode::Lagrange,
        ) {
            Ok(()) => tracing::info!(
                label,
                root_d,
                elapsed_s = t0.elapsed().as_secs_f64(),
                "verify OK"
            ),
            Err(e) => tracing::error!(
                label,
                root_d,
                elapsed_s = t0.elapsed().as_secs_f64(),
                error = %e,
                "verify FAILED"
            ),
        }
    } else {
        let poly_refs: Vec<&[MultilinearPolynomial<F>]> = vec![polys.as_slice()];
        let point_refs: Vec<&[&[MultilinearPolynomial<F>]]> = vec![poly_refs.as_slice()];
        let commitments = [commitment];
        let commitment_refs: Vec<&[<Scheme<Family> as DynamicCommitmentScheme<F>>::Commitment]> =
            vec![commitments.as_slice()];
        let opening_groups = [&openings[..]];

        let t0 = Instant::now();
        let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
        let proof = <Scheme<Family> as DynamicCommitmentScheme<F>>::batched_prove(
            &setup,
            &point_refs,
            &[&pt[..]],
            vec![vec![hint]],
            &mut prover_transcript,
            &commitment_refs,
            BasisMode::Lagrange,
        )
        .unwrap();
        tracing::info!(
            label,
            root_d,
            elapsed_s = t0.elapsed().as_secs_f64(),
            "prove"
        );
        match root_d {
            32 => print_batched_proof_summary::<32>(label, &proof),
            64 => print_batched_proof_summary::<64>(label, &proof),
            128 => print_batched_proof_summary::<128>(label, &proof),
            _ => unreachable!("dynamic schemes only select D in 32/64/128"),
        }

        let t0 = Instant::now();
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
        match <Scheme<Family> as DynamicCommitmentScheme<F>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &[&pt[..]],
            &[&opening_groups[..]],
            &commitment_refs,
            BasisMode::Lagrange,
        ) {
            Ok(()) => tracing::info!(
                label,
                root_d,
                elapsed_s = t0.elapsed().as_secs_f64(),
                "verify OK"
            ),
            Err(e) => tracing::error!(
                label,
                root_d,
                elapsed_s = t0.elapsed().as_secs_f64(),
                error = %e,
                "verify FAILED"
            ),
        }
    }
}

fn run_dense_mode<const D: usize, Cfg: CommitmentConfig<Field = F>>(title: &str, nv: usize) {
    let layout = resolve_layout::<Cfg>(nv);
    let plan = Cfg::schedule_plan(nv).expect("schedule plan");
    tracing::info!("{}", title);
    print_layout(&layout);
    run_dense::<D, Cfg>(nv, &layout, plan.as_ref());
}

fn run_onehot_mode<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    title: &str,
    nv: usize,
    num_polys: usize,
) {
    tracing::info!("{}", title);
    if num_polys == 1 {
        let layout = resolve_layout::<Cfg>(nv);
        let plan = Cfg::schedule_plan(nv).expect("schedule plan");
        print_layout(&layout);
        run_onehot::<D, Cfg>(nv, &layout, plan.as_ref());
    } else {
        let layout = hachi_batched_root_layout::<Cfg, D>(nv, num_polys).expect("layout");
        print_layout(&layout);
        run_batched_onehot::<D, Cfg>(nv, num_polys, &layout);
    }
}

type ProfileModeRunner = fn(usize, usize);

struct ProfileMode {
    name: &'static str,
    run: ProfileModeRunner,
}

const PROFILE_MODES: &[ProfileMode] = &[
    ProfileMode {
        name: "full",
        run: run_profile_full,
    },
    ProfileMode {
        name: "onehot",
        run: run_profile_onehot,
    },
    ProfileMode {
        name: "full_d128",
        run: run_profile_full_d128,
    },
    ProfileMode {
        name: "onehot_d64",
        run: run_profile_onehot_d64,
    },
    ProfileMode {
        name: "full_d32",
        run: run_profile_full_d32,
    },
    ProfileMode {
        name: "onehot_d32",
        run: run_profile_onehot_d32,
    },
    ProfileMode {
        name: "logbasis",
        run: run_profile_logbasis,
    },
    ProfileMode {
        name: "compare_onehot",
        run: run_profile_compare_onehot,
    },
    ProfileMode {
        name: "compare_logbasis",
        run: run_profile_compare_logbasis,
    },
    ProfileMode {
        name: "compare_basis",
        run: run_profile_compare_basis,
    },
];

const ALL_PROFILE_MODE_NAMES: &[&str] = &[
    "full",
    "onehot",
    "full_d128",
    "onehot_d64",
    "full_d32",
    "onehot_d32",
    "logbasis",
];

fn assert_singleton_mode(mode: &str, num_polys: usize) {
    assert_eq!(
        num_polys, 1,
        "{mode} currently profiles only singleton commitments"
    );
}

fn fixed_onehot_title(d: usize, num_polys: usize) -> String {
    if num_polys == 1 {
        format!("=== onehot_d{d} (q=2^128-275, D={d}, 1-of-256, log_commit_bound=1) ===")
    } else {
        format!(
            "=== onehot_d{d} batched (q=2^128-275, D={d}, 1-of-256, log_commit_bound=1, same-point batch={num_polys}) ==="
        )
    }
}

fn run_profile_full(nv: usize, num_polys: usize) {
    assert_singleton_mode("full", num_polys);
    run_dynamic_dense_mode::<fp128::FullFamily>(
        "dense",
        "=== full (q=2^128-275, runtime-selected root D, dense) ===",
        nv,
    );
}

fn run_profile_onehot(nv: usize, num_polys: usize) {
    let title = if num_polys == 1 {
        "=== onehot (q=2^128-275, runtime-selected root D, 1-of-256) ===".to_string()
    } else {
        format!(
            "=== onehot batched (q=2^128-275, runtime-selected root D, 1-of-256, same-point batch={num_polys}) ==="
        )
    };
    run_dynamic_onehot_mode::<fp128::OneHotFamily>("onehot", &title, nv, num_polys);
}

fn run_profile_full_d128(nv: usize, num_polys: usize) {
    type Cfg = fp128::D128Full;
    assert_singleton_mode("full_d128", num_polys);
    run_dense_mode::<{ Cfg::D }, Cfg>(
        "=== full_d128 (q=2^128-275, D=128 dense, log_commit_bound=128) ===",
        nv,
    );
}

fn run_profile_onehot_d64(nv: usize, num_polys: usize) {
    type Cfg = fp128::D64OneHot;
    let title = fixed_onehot_title(64, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>(&title, nv, num_polys);
}

fn run_profile_full_d32(nv: usize, num_polys: usize) {
    type Cfg = fp128::D32Full;
    assert_singleton_mode("full_d32", num_polys);
    run_dense_mode::<{ Cfg::D }, Cfg>(
        "=== full_d32 (q=2^128-275, D=32 dense, log_commit_bound=128) ===",
        nv,
    );
}

fn run_profile_onehot_d32(nv: usize, num_polys: usize) {
    type Cfg = fp128::D32OneHot;
    let title = fixed_onehot_title(32, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>(&title, nv, num_polys);
}

fn run_profile_logbasis(nv: usize, num_polys: usize) {
    type Cfg = fp128::LogBasis;
    assert_singleton_mode("logbasis", num_polys);
    run_dense_mode::<{ Cfg::D }, Cfg>(
        "=== logbasis (q=2^128-275, D=128 dense, log_commit_bound=3) ===",
        nv,
    );
}

fn run_profile_compare_onehot(nv: usize, num_polys: usize) {
    assert_singleton_mode("compare_onehot", num_polys);

    type A = fp128::D64StaticBounded<1, 3, 3>;
    type B = fp128::D64StaticBounded<1, 2, 2>;
    type C = fp128::D64StaticBounded<1, 2, 3>;
    type D = fp128::D64StaticBounded<1, 2, 4>;

    run_onehot_mode::<{ A::D }, A>(
        "=== [A] onehot (D=64, 1-of-256), basis=3 everywhere ===",
        nv,
        1,
    );
    run_onehot_mode::<{ B::D }, B>(
        "=== [B] onehot (D=64, 1-of-256), basis=2 everywhere ===",
        nv,
        1,
    );
    run_onehot_mode::<{ C::D }, C>(
        "=== [C] onehot (D=64, 1-of-256), L0 basis=2, w-levels basis=3 ===",
        nv,
        1,
    );
    run_onehot_mode::<{ D::D }, D>(
        "=== [D] onehot (D=64, 1-of-256), L0 basis=2, w-levels basis=4 ===",
        nv,
        1,
    );
}

fn run_profile_compare_logbasis(nv: usize, num_polys: usize) {
    assert_singleton_mode("compare_logbasis", num_polys);

    type A = fp128::StaticBounded<3, 3, 3>;
    type B = fp128::StaticBounded<3, 2, 2>;
    type C = fp128::StaticBounded<3, 2, 3>;
    type D = fp128::StaticBounded<3, 2, 4>;

    run_dense_mode::<{ A::D }, A>(
        "=== [A] logbasis coeffs (D=128), basis=3 everywhere ===",
        nv,
    );
    run_dense_mode::<{ B::D }, B>(
        "=== [B] logbasis coeffs (D=128), basis=2 everywhere ===",
        nv,
    );
    run_dense_mode::<{ C::D }, C>(
        "=== [C] logbasis coeffs (D=128), L0 basis=2, w-levels basis=3 ===",
        nv,
    );
    run_dense_mode::<{ D::D }, D>(
        "=== [D] logbasis coeffs (D=128), L0 basis=2, w-levels basis=4 ===",
        nv,
    );
}

fn run_profile_compare_basis(nv: usize, num_polys: usize) {
    assert_singleton_mode("compare_basis", num_polys);

    type A = fp128::StaticBounded<128, 3, 3>;
    type B = fp128::StaticBounded<128, 2, 2>;
    type C = fp128::StaticBounded<128, 2, 3>;
    type D = fp128::StaticBounded<128, 2, 4>;

    run_dense_mode::<{ A::D }, A>("=== [A] baseline (D=128): log_basis=3 everywhere ===", nv);
    run_dense_mode::<{ B::D }, B>("=== [B] baseline (D=128): log_basis=2 everywhere ===", nv);
    run_dense_mode::<{ C::D }, C>(
        "=== [C] baseline (D=128): L0 basis=2, w-levels basis=3 ===",
        nv,
    );
    run_dense_mode::<{ D::D }, D>(
        "=== [D] baseline (D=128): L0 basis=2, w-levels basis=4 ===",
        nv,
    );
}

fn run_profile_mode(mode: &str, nv: usize, num_polys: usize) {
    let profile_mode = PROFILE_MODES
        .iter()
        .find(|entry| entry.name == mode)
        .unwrap_or_else(|| {
            let mut known_modes = PROFILE_MODES
                .iter()
                .map(|entry| entry.name)
                .collect::<Vec<_>>();
            known_modes.push("all");
            tracing::error!(
                mode,
                known_modes = %known_modes.join(", "),
                "Unknown HACHI_MODE"
            );
            std::process::exit(1);
        });
    (profile_mode.run)(nv, num_polys);
}

fn run_all_profile_modes(nv: usize) {
    for mode in ALL_PROFILE_MODE_NAMES {
        run_profile_mode(mode, nv, 1);
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

    if mode == "all" {
        run_all_profile_modes(nv);
    } else {
        run_profile_mode(&mode, nv, num_polys);
    }

    if enable_trace {
        tracing::info!(trace_file = %trace_file, "Done. Trace saved");
    } else {
        tracing::info!("Done");
    }
}

fn resolve_layout<Cfg: CommitmentConfig<Field = F>>(nv: usize) -> HachiCommitmentLayout {
    Cfg::commitment_layout(nv).expect("layout")
}
