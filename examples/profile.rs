#![allow(missing_docs)]

use akita_serialization::Compress;
use akita_transcript::Blake2bTranscript;
use hachi_pcs::protocol::commitment::{
    hachi_batched_root_layout, HachiRootBatchSummary, HachiScheduleLookupKey, HachiSchedulePlan,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::config::proof_optimized::fp128;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
};
use hachi_pcs::protocol::params::LevelParams;
use hachi_pcs::protocol::proof::{
    DirectWitnessProof, HachiBatchedProof, HachiBatchedRootProof, HachiCommitmentHint,
    HachiLevelProof,
};
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::protocol::Step;
use hachi_pcs::{
    BasisMode, BlockOrder, CanonicalField, CommitmentProver, CommitmentVerifier, CommittedOpenings,
    CommittedPolynomials, FieldCore, FromSmallInt, HachiPolyOps, HachiSerialize,
    PseudoMersenneField, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::io::BufWriter;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

type F = fp128::Field;
const ONEHOT_K: usize = 256;

/// Short label for the active Fp128 prime, derived from `MODULUS_OFFSET`
/// so that `examples/profile.rs` cannot drift away from the real prime
/// when `fp128::Field` is retargeted (e.g. switching between
/// `Prime128Offset2355` and `Prime128OffsetA7F7`).
fn fp128_prime_label() -> String {
    match <F as PseudoMersenneField>::MODULUS_OFFSET {
        2355 => "q=2^128-2355".to_string(),
        // Prime128OffsetA7F7: p = 2^128 - 2^32 + 22537 = 2^128 - 0xFFFFA7F7.
        0xFFFFA7F7 => "q=2^128-2^32+22537".to_string(),
        offset => format!("q=2^128-{offset:#x}"),
    }
}

fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
}

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
    layout: &LevelParams,
    basis: BasisMode,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.m_vars + layout.r_vars;
    assert!(
        point.len() <= target_num_vars,
        "opening point length {} exceeds target root arity {}",
        point.len(),
        target_num_vars
    );
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, F::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
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
    setup: &<HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::ProverSetup,
    poly: &P,
    pt: &[F],
    opening: F,
    plan: Option<&HachiSchedulePlan>,
) where
    HachiCommitmentScheme<D, Cfg>: CommitmentProver<
        F,
        D,
        BatchedProof = HachiBatchedProof<F>,
        CommitHint = HachiCommitmentHint<F, D>,
    >,
{
    type Scheme<const D: usize, Cfg> = HachiCommitmentScheme<D, Cfg>;

    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentProver<F, D>>::commit(std::slice::from_ref(poly), setup)
            .unwrap();
    tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "commit");

    let poly_refs: [&P; 1] = [poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        setup,
        vec![(
            pt,
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    tracing::info!(label, elapsed_s = t0.elapsed().as_secs_f64(), "prove");
    print_batched_proof_summary::<D>(label, &proof);
    if let Some(plan) = plan {
        debug_assert_eq!(
            proof.size(),
            plan.exact_proof_bytes,
            "runtime proof bytes should match the planned proof size"
        );
        emit_planned_schedule_summary(label, plan);
    }

    let t0 = Instant::now();
    let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
    match <Scheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            pt,
            vec![CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            }],
        )],
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
            d = level.lp.ring_dimension,
            n_a = level.lp.a_key.row_len(),
            n_b = level.lp.b_key.row_len(),
            n_d = level.lp.d_key.row_len(),
            challenge_l1_mass = level.lp.challenge_l1_mass(),
            log_basis = level.lp.log_basis,
            m_vars = level.lp.m_vars,
            r_vars = level.lp.r_vars,
            num_blocks = level.lp.num_blocks,
            block_len = level.lp.block_len,
            delta_commit = level.lp.num_digits_commit,
            delta_open = level.lp.num_digits_open,
            delta_fold = level.lp.num_digits_fold,
            current_w_len = level.inputs.current_w_len,
            next_w_ring = next_w_len / level.lp.ring_dimension,
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
    let Some(fold) = root.as_fold() else {
        let total = root.serialized_size(Compress::No);
        eprintln!("[{label}]   batched_root: total={total} bytes (root-direct)");
        tracing::info!(
            label,
            level = 0usize,
            d = D,
            total_bytes = total,
            y_ring_bytes = 0usize,
            v_bytes = 0usize,
            stage1_sumcheck_bytes = 0usize,
            stage1_interstage_claims_bytes = 0usize,
            stage1_s_claim_bytes = 0usize,
            stage2_sumcheck_bytes = 0usize,
            next_w_commitment_bytes = 0usize,
            next_w_eval_bytes = 0usize,
            root_variant = "direct",
            "proof fold level"
        );
        return total;
    };
    let y_rings_size = fold.y_rings.serialized_size(Compress::No);
    let v_size = fold.v.serialized_size(Compress::No);
    let total = fold.serialized_size(Compress::No);
    let stage1 = &fold.stage1;
    let stage2 = &fold.stage2;
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
        level = 0usize,
        d = D,
        total_bytes = total,
        y_ring_bytes = y_rings_size,
        v_bytes = v_size,
        stage1_sumcheck_bytes = stage1_sumcheck_size,
        stage1_interstage_claims_bytes = stage1_interstage_claims_size,
        stage1_s_claim_bytes = stage1_s_claim_size,
        stage2_sumcheck_bytes = stage2_sumcheck_size,
        next_w_commitment_bytes = next_w_commitment_size,
        next_w_eval_bytes = next_w_eval_size,
        root_variant = "fold",
        "proof fold level"
    );
    eprintln!("[{label}]   batched_root: total={total} bytes");
    eprintln!(
        "[{label}]     y_rings={} bytes ({} ring elems, D={})",
        y_rings_size,
        ring_elem_count(fold.y_rings.coeff_len(), D),
        D,
    );
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        ring_elem_count(fold.v.coeff_len(), D),
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
    let tail_total = proof.final_witness().serialized_size(Compress::No);
    let accounted_total = hachi_levels_total + tail_total;
    let framing_total = proof.size() - accounted_total;

    tracing::info!(
        label,
        levels = proof.num_fold_levels() + 1,
        proof_size_bytes = proof.size(),
        accounted_bytes = accounted_total,
        hachi_fold_bytes = hachi_levels_total,
        tail_bytes = tail_total,
        proof_framing_bytes = framing_total,
        "proof summary"
    );
    debug_assert_eq!(accounted_total, proof.size());
    print_batched_root_breakdown::<D>(label, &proof.root);
    for (i, lp) in proof.fold_levels().enumerate() {
        print_hachi_level_breakdown(label, i + 1, lp);
    }
    emit_observed_tail_summary(label, proof.final_witness());
}

fn emit_observed_tail_summary(label: &str, final_w: &DirectWitnessProof<F>) {
    let tail_bytes = final_w.serialized_size(Compress::No);
    let num_elems = final_w.num_elems();
    if let Some(packed) = final_w.as_packed_digits() {
        tracing::info!(
            label,
            tail_bytes,
            final_w_num_elems = num_elems,
            final_w_bits_per_elem = packed.bits_per_elem,
            final_w_encoding = "packed_digits",
            "proof tail summary"
        );
        eprintln!(
            "[{label}]   final_w: total={tail_bytes} bytes, elems={num_elems}, bits/elem={}",
            packed.bits_per_elem,
        );
    } else {
        tracing::info!(
            label,
            tail_bytes,
            final_w_num_elems = num_elems,
            final_w_encoding = "field_elements",
            "proof tail summary"
        );
        eprintln!(
            "[{label}]   final_w: total={tail_bytes} bytes, elems={num_elems}, bits/elem=field"
        );
    }
}

fn print_layout(layout: &LevelParams) {
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
    layout: &LevelParams,
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
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
    tracing::info!(
        label = "dense",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "setup"
    );

    run_prove::<D, Cfg, _>("dense", &setup, &poly, &pt, opening, plan);
}

fn run_onehot<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    nv: usize,
    layout: &LevelParams,
    plan: Option<&HachiSchedulePlan>,
) {
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(D)
        .expect("total field size overflow");
    let onehot_k = onehot_k_for_num_vars(nv);
    let total_chunks = total_field / onehot_k;
    assert_eq!(
        total_chunks * onehot_k,
        total_field,
        "onehot K must divide total field size"
    );

    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    let onehot_poly = OneHotPoly::<F, D, u8>::new(onehot_k, indices).unwrap();
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = opening_from_poly(&onehot_poly, &pt, layout, BasisMode::Lagrange);

    let t0 = Instant::now();
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
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
    layout: &LevelParams,
) {
    type Scheme<const D: usize, Cfg> = HachiCommitmentScheme<D, Cfg>;

    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(D)
        .expect("total field size overflow");
    let onehot_k = onehot_k_for_num_vars(nv);
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
            OneHotPoly::<F, D, u8>::new(onehot_k, indices).unwrap()
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
    let opening_groups = [&openings[..]];

    let t0 = Instant::now();
    let setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, num_polys, 1);
    tracing::info!(
        label = "onehot",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "setup"
    );

    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentProver<F, D>>::commit(&poly_refs, &setup).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];
    tracing::info!(
        label = "onehot",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "commit"
    );

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &pt[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    tracing::info!(
        label = "onehot",
        elapsed_s = t0.elapsed().as_secs_f64(),
        "prove"
    );
    print_batched_proof_summary::<D>("onehot", &proof);
    let batch_summary =
        HachiRootBatchSummary::new(num_polys, 1, 1).expect("same-point batch summary");
    let schedule =
        Cfg::get_params_for_prove(nv, nv, num_polys, batch_summary).expect("batched schedule");
    if let Some(Step::Fold(root_step)) = schedule.steps.first() {
        tracing::info!(
            label = "onehot",
            root_bytes = root_step.level_bytes,
            observed_total_bytes = proof.size(),
            "batched planner root-fold summary"
        );
    } else if let Some(Step::Direct(root_direct)) = schedule.steps.first() {
        tracing::info!(
            label = "onehot",
            root_bytes = root_direct.direct_bytes,
            observed_total_bytes = proof.size(),
            "batched planner direct-root estimate"
        );
    }

    let t0 = Instant::now();
    let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
    match <Scheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &pt[..],
            vec![CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            }],
        )],
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

fn best_full_d(nv: usize) -> usize {
    let key = HachiScheduleLookupKey::singleton(nv, nv, 1);
    let d32_bytes = fp128::D32Full::schedule_plan(key)
        .ok()
        .flatten()
        .map(|p| p.exact_proof_bytes);
    let d128_bytes = fp128::D128Full::schedule_plan(key)
        .ok()
        .flatten()
        .map(|p| p.exact_proof_bytes);
    match (d32_bytes, d128_bytes) {
        (Some(b32), Some(b128)) if b32 <= b128 => 32,
        (None, Some(_)) => 128,
        _ => 32,
    }
}

fn best_onehot_d(nv: usize) -> usize {
    let key = HachiScheduleLookupKey::singleton(nv, nv, 1);
    let d32_bytes = fp128::D32OneHot::schedule_plan(key)
        .ok()
        .flatten()
        .map(|p| p.exact_proof_bytes);
    let d64_bytes = fp128::D64OneHot::schedule_plan(key)
        .ok()
        .flatten()
        .map(|p| p.exact_proof_bytes);
    match (d32_bytes, d64_bytes) {
        (Some(b32), Some(b64)) if b32 <= b64 => 32,
        (None, Some(_)) => 64,
        _ => 32,
    }
}

fn run_dense_mode<const D: usize, Cfg: CommitmentConfig<Field = F>>(title: &str, nv: usize) {
    let layout = resolve_layout::<Cfg>(nv);
    let plan =
        Cfg::schedule_plan(HachiScheduleLookupKey::singleton(nv, nv, 1)).expect("schedule plan");
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
        let required_vars = layout.m_vars + layout.r_vars + D.trailing_zeros() as usize;
        if required_vars > nv {
            tracing::info!(
                required_vars,
                "skipping fixed onehot mode because the typed root layout exceeds the public polynomial arity"
            );
            return;
        }
        let plan = Cfg::schedule_plan(HachiScheduleLookupKey::singleton(nv, nv, 1))
            .expect("schedule plan");
        print_layout(&layout);
        run_onehot::<D, Cfg>(nv, &layout, plan.as_ref());
    } else {
        let layout = hachi_batched_root_layout::<Cfg>(nv, num_polys).expect("layout");
        let required_vars = layout.m_vars + layout.r_vars + D.trailing_zeros() as usize;
        if required_vars > nv {
            tracing::info!(
                required_vars,
                "skipping fixed batched onehot mode because the typed root layout exceeds the public polynomial arity"
            );
            return;
        }
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
];

const ALL_PROFILE_MODE_NAMES: &[&str] = &[
    "full",
    "onehot",
    "full_d128",
    "onehot_d64",
    "full_d32",
    "onehot_d32",
];

fn assert_singleton_mode(mode: &str, num_polys: usize) {
    assert_eq!(
        num_polys, 1,
        "{mode} currently profiles only singleton commitments"
    );
}

fn fixed_onehot_title(d: usize, nv: usize, num_polys: usize) -> String {
    let onehot_k = onehot_k_for_num_vars(nv);
    let prime = fp128_prime_label();
    if num_polys == 1 {
        format!("=== onehot_d{d} ({prime}, D={d}, 1-of-{onehot_k}, log_commit_bound=1) ===")
    } else {
        format!(
            "=== onehot_d{d} batched ({prime}, D={d}, 1-of-{onehot_k}, log_commit_bound=1, same-point batch={num_polys}) ==="
        )
    }
}

fn run_profile_full(nv: usize, num_polys: usize) {
    assert_singleton_mode("full", num_polys);
    let d = best_full_d(nv);
    let prime = fp128_prime_label();
    let title = format!("=== full ({prime}, D={d}, dense) ===");
    match d {
        32 => run_dense_mode::<32, fp128::D32Full>(&title, nv),
        128 => run_dense_mode::<128, fp128::D128Full>(&title, nv),
        _ => unreachable!(),
    }
}

fn run_profile_onehot(nv: usize, num_polys: usize) {
    let onehot_k = onehot_k_for_num_vars(nv);
    let d = best_onehot_d(nv);
    let prime = fp128_prime_label();
    let title = if num_polys == 1 {
        format!("=== onehot ({prime}, D={d}, 1-of-{onehot_k}) ===")
    } else {
        format!(
            "=== onehot batched ({prime}, D={d}, 1-of-{onehot_k}, same-point batch={num_polys}) ==="
        )
    };
    match d {
        32 => run_onehot_mode::<32, fp128::D32OneHot>(&title, nv, num_polys),
        64 => run_onehot_mode::<64, fp128::D64OneHot>(&title, nv, num_polys),
        _ => unreachable!(),
    }
}

fn run_profile_full_d128(nv: usize, num_polys: usize) {
    type Cfg = fp128::D128Full;
    assert_singleton_mode("full_d128", num_polys);
    let prime = fp128_prime_label();
    run_dense_mode::<{ Cfg::D }, Cfg>(
        &format!("=== full_d128 ({prime}, D=128 dense, log_commit_bound=128) ==="),
        nv,
    );
}

fn run_profile_onehot_d64(nv: usize, num_polys: usize) {
    type Cfg = fp128::D64OneHot;
    let title = fixed_onehot_title(64, nv, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>(&title, nv, num_polys);
}

fn run_profile_full_d32(nv: usize, num_polys: usize) {
    type Cfg = fp128::D32Full;
    assert_singleton_mode("full_d32", num_polys);
    let prime = fp128_prime_label();
    run_dense_mode::<{ Cfg::D }, Cfg>(
        &format!("=== full_d32 ({prime}, D=32 dense, log_commit_bound=128) ==="),
        nv,
    );
}

fn run_profile_onehot_d32(nv: usize, num_polys: usize) {
    type Cfg = fp128::D32OneHot;
    let title = fixed_onehot_title(32, nv, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>(&title, nv, num_polys);
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
        let file = fs::File::create(&trace_file).expect("Failed to create trace file");
        let buffered = BufWriter::with_capacity(4 * 1024 * 1024, file);
        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .include_args(true)
            .writer(buffered)
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
    tracing::info!(
        "fp128 protocol prime active: modulus_offset = 0x{:x}, probe(2^128 + 1) = 0x{:x}",
        <F as PseudoMersenneField>::MODULUS_OFFSET,
        F::solinas_reduce(&[1u64, 0, 1]).to_canonical_u128(),
    );

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

fn resolve_layout<Cfg: CommitmentConfig<Field = F>>(nv: usize) -> LevelParams {
    Cfg::commitment_layout(nv).expect("layout")
}
