//! Split-vs-fused-vs-package proof frontier for representative dense shapes.
//!
//! Emits, for each representative `fp128_dense` generated row, the exact
//! serialized-proof totals of:
//!
//! 1. the shipping split schedule (two sumchecks per fold level),
//! 2. the same geometry re-priced with the fused range-relation check
//!    ([`akita_types::fused_level_proof_bytes`]: one sumcheck per level with
//!    `b + 1` stored coefficients per round), and
//! 3. the four-level thin-tail fused package: the shipping root, two
//!    synthesized coefficient-L-infinity EvaluationTrace tail folds at basis
//!    `b = 8`, and a terminal response, all sized against the audited SIS
//!    tables and selected by exhaustive search over the tail knobs.
//!
//! Every total uses the same audited pricing primitives as the planner walker
//! (`nonterminal_level_payload_bytes` and its fused counterpart, the
//! extension-opening reduction formula, the grind nonce, and
//! `terminal_response_planner_bytes`), so the package row settles the
//! total-proof-bytes question for the fused package by direct accounting.
//!
//! The synthesized package rows are byte-exact for their stated geometry; the
//! tail geometry itself is chosen by this example's search, not by the full
//! schedule DP, and the package terminal reuses the shipping terminal's
//! Golomb-Rice bytes-per-coordinate rate (stated in the output). Norm-route
//! caveat: fused re-pricing of a selective-L2 level carries the norm payload
//! unchanged; those levels are marked in the per-level table.
//!
//! ```bash
//! cargo run --release -p akita-planner --features catalog-check --example fused_frontier
//! ```

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_error::AkitaError;
use akita_schedules::planner_support::{nonterminal_level_payload_bytes, planned_next_witness_len};
use akita_schedules::{
    expanded_schedule_fused_proof_payload_bytes, expanded_schedule_proof_payload_bytes,
    fused_nonterminal_level_payload_bytes, schedule_from_entry, PlannerPolicy,
};
use akita_types::sis::{
    decomposed_s_block_ring_count, num_digits_inner, num_digits_open,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, SisMatrixRole, SisTableKey,
};
use akita_types::{
    extension_opening_reduction_level_bytes, fused_level_chain_rounds, padded_boolean_opening_vars,
    split_level_chain_rounds, sumcheck_rounds, terminal_response_planner_bytes, BlockGeometry,
    ChunkedWitnessCfg, CommitmentPayloadMode, CommitmentSliceCount, CommitmentSliceGeometry,
    CommittedGroupParams, CommittedSourceEncoding, DecompositionParams, DigitRangePlan,
    FoldSchedule, GadgetDigits, GroupCommitPhaseParams, GroupOpenPhaseParams, GroupOpeningPlan,
    InnerCommitMatrixParams, InnerCommitSecurityRoute, OpenCommitMatrixParams, OpeningMethod,
    OuterCommitMatrixParams, PolynomialGroupLayout, RoleParams, TailSegmentGroupLayout,
    TailSegmentLayout, TerminalResponseShape, FOLD_GRIND_NONCE_BYTES,
};

/// Representative dense final-group sizes; 28 and 30 bracket a 2^29 campaign
/// shape (the checked-in catalog has no 29-variable row).
const REPRESENTATIVE_NUM_VARS: &[usize] = &[24, 28, 30];

/// Synthesized-tail search domain (all coefficient-L-infinity, EvaluationTrace,
/// opening basis 8 so the fused check applies at degree 9).
const TAIL_RING_DIMS: &[usize] = &[64, 256, 512];
const TAIL_FOLD_DIGITS: &[usize] = &[2, 3];
const TAIL_TARGET_BLOCKS: &[usize] = &[32, 128, 512, 2048];
const TERMINAL_RING_DIMS: &[usize] = &[64, 256];
const TERMINAL_POSITIONS: &[usize] = &[256, 512, 1024, 2048, 4096];
const TERMINAL_FOLD_LOG_BASIS: &[u32] = &[3, 5];
const TERMINAL_FOLD_DIGITS: &[usize] = &[2, 3];

const OUTER_RING_DIM: usize = 64;
const OPEN_RING_DIM: usize = 64;
const TAIL_LOG_BASIS: u32 = 3;

fn challenge_elem_bytes(policy: &PlannerPolicy) -> Result<usize, AkitaError> {
    Ok((policy.challenge_field_bits()? as usize).div_ceil(8))
}

fn sis_key(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    ring_dimension: usize,
    coeff_linf_bound: u128,
) -> SisTableKey {
    SisTableKey {
        policy: policy.sis_security_policy,
        table_digest: policy.sis_table_digest,
        modulus_profile: policy.sis_modulus_profile,
        role,
        ring_dimension: ring_dimension as u32,
        coeff_linf_bound,
    }
}

fn no_layout(what: &str) -> AkitaError {
    AkitaError::InvalidSetup(format!("no audited layout for synthesized {what}"))
}

/// Round-chain and round-message-stream summary of one non-terminal level.
struct LevelRounds {
    split_chain: usize,
    fused_chain: usize,
    split_stream_bytes: usize,
    fused_stream_bytes: usize,
    l2_route: bool,
}

fn level_rounds(
    params: &CommittedGroupParams,
    output_witness_len: usize,
    elem_bytes: usize,
) -> Result<LevelRounds, AkitaError> {
    let rounds = sumcheck_rounds(params.d_a(), output_witness_len);
    let b = 1usize << params.open().digits.log_basis;
    let route = params.inner().matrix.security_route();
    let plan = DigitRangePlan::new(b)?;
    let (stages, norm) = plan.proof_shapes_for_route(rounds, route)?;
    let stage_stream: usize = stages
        .iter()
        .map(|stage| stage.sumcheck_proof.0 * stage.sumcheck_proof.1 * elem_bytes)
        .sum();
    let norm_stream: usize = norm
        .as_ref()
        .map_or(0, |shape| shape.sumcheck.iter().sum::<usize>() * elem_bytes);
    Ok(LevelRounds {
        split_chain: split_level_chain_rounds(rounds, b, route)?,
        fused_chain: fused_level_chain_rounds(rounds, b, route)?,
        split_stream_bytes: stage_stream + norm_stream + rounds * 3 * elem_bytes,
        fused_stream_bytes: rounds * (b + 1) * elem_bytes + norm_stream,
        l2_route: matches!(route, InnerCommitSecurityRoute::L2 { .. }),
    })
}

fn schedule_rounds(
    schedule: &FoldSchedule,
    elem_bytes: usize,
) -> Result<(LevelRounds, usize), AkitaError> {
    let mut total = LevelRounds {
        split_chain: 0,
        fused_chain: 0,
        split_stream_bytes: 0,
        fused_stream_bytes: 0,
        l2_route: false,
    };
    let mut l2_levels = 0usize;
    let root = std::iter::once((&schedule.root.params, schedule.root.output_witness_len));
    let tails = schedule
        .recursive_folds
        .iter()
        .map(|fold| (&fold.params, fold.output_witness_len));
    for (params, output_witness_len) in root.chain(tails) {
        let level = level_rounds(params, output_witness_len, elem_bytes)?;
        total.split_chain += level.split_chain;
        total.fused_chain += level.fused_chain;
        total.split_stream_bytes += level.split_stream_bytes;
        total.fused_stream_bytes += level.fused_stream_bytes;
        l2_levels += usize::from(level.l2_route);
    }
    Ok((total, l2_levels))
}

/// One synthesized coefficient-L-infinity EvaluationTrace tail fold at basis 8.
///
/// Mirrors the generated-row expansion for the scalar Linf path: exact block
/// geometry from the incoming witness length, digit depths from the policy
/// decomposition, collision buckets and minimum secure ranks from the audited
/// SIS tables.
fn synthesize_tail_fold(
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    input_witness_len: usize,
    d_a: usize,
    num_digits_fold: usize,
    target_blocks: usize,
) -> Result<(CommittedGroupParams, usize), AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    let n_live = input_witness_len.div_ceil(d_a);
    let m = n_live.div_ceil(target_blocks).next_power_of_two();
    let b_blocks = n_live.div_ceil(m);

    let witness_decomp = DecompositionParams {
        log_basis: TAIL_LOG_BASIS,
        log_commit_bound: field_bits,
        log_open_bound: Some(field_bits),
    };
    let outer_decomp = DecompositionParams {
        log_basis: TAIL_LOG_BASIS,
        ..policy.decomposition
    };
    let open_decomp = DecompositionParams {
        log_basis: TAIL_LOG_BASIS,
        ..policy.decomposition
    };
    let ndi = num_digits_inner(witness_decomp, false);
    let ndo = num_digits_open(outer_decomp);
    let ndov = num_digits_open(open_decomp);
    let inner_width = decomposed_s_block_ring_count(m, ndi).ok_or_else(|| no_layout("A"))?;

    let fold_challenge_config = ring_challenge_config(d_a)?;
    let a_bucket = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        d_a,
        TAIL_LOG_BASIS,
        &fold_challenge_config,
        num_digits_fold,
    )
    .ok_or_else(|| no_layout("A"))?;
    let inner = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key(policy, SisMatrixRole::Inner, d_a, a_bucket),
        inner_width,
    )?;

    let b_bucket = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Outer,
        OUTER_RING_DIM,
        TAIL_LOG_BASIS,
    )
    .ok_or_else(|| no_layout("B"))?;
    let outer_width = CommitmentSliceGeometry::try_new(
        CommitmentSliceCount::ONE,
        b_blocks,
        1,
        inner.output_rank(),
        ndo,
        d_a,
        OUTER_RING_DIM,
    )?
    .physical_input_width();
    let outer = OuterCommitMatrixParams::try_new_with_min_rank(
        sis_key(policy, SisMatrixRole::Outer, OUTER_RING_DIM, b_bucket),
        outer_width,
    )?;

    let d_bucket = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Open,
        OPEN_RING_DIM,
        TAIL_LOG_BASIS,
    )
    .ok_or_else(|| no_layout("D"))?;
    let d_width = akita_types::opening_d_segment_width(
        OpeningMethod::EvaluationTrace,
        policy.claim_ext_degree,
        d_a,
        OPEN_RING_DIM,
        ndov,
        b_blocks,
        1,
    )?;
    let open = OpenCommitMatrixParams::try_new_with_min_rank(
        sis_key(policy, SisMatrixRole::Open, OPEN_RING_DIM, d_bucket),
        d_width,
    )?;

    let source_encoding = CommittedSourceEncoding::for_producer(
        OpeningMethod::EvaluationTrace,
        policy.claim_ext_degree,
        d_a,
        input_witness_len.trailing_zeros() as usize,
        false,
    );
    let params = CommittedGroupParams::try_new(
        vec![GroupOpenPhaseParams {
            profile: GroupCommitPhaseParams {
                version: GroupCommitPhaseParams::VERSION,
                group: PolynomialGroupLayout::singleton(padded_boolean_opening_vars(
                    input_witness_len,
                )?),
                blocks: BlockGeometry::new(n_live, m, b_blocks),
                outer_slice_count: CommitmentSliceCount::ONE,
                inner: RoleParams::new(GadgetDigits::new(TAIL_LOG_BASIS, ndi), inner),
                outer: RoleParams::new(GadgetDigits::new(TAIL_LOG_BASIS, ndo), outer),
            },
            opening: GroupOpeningPlan {
                opening_method: OpeningMethod::EvaluationTrace,
                fold_challenge_config,
                log_basis_open: TAIL_LOG_BASIS,
                num_digits_open: ndov,
                num_digits_fold,
            },
            setup_natural_len: None,
        }],
        open,
        CommitmentPayloadMode::Compressed,
        source_encoding,
        ChunkedWitnessCfg::default_non_chunked(),
    )?;
    let output_witness_len =
        planned_next_witness_len(field_bits, policy.claim_ext_degree, &params, 1, 1)?
            .ok_or_else(|| no_layout("compression source"))?;
    Ok((params, output_witness_len))
}

/// One synthesized terminal response shape plus its planner byte estimate.
///
/// The Golomb-Rice payload budget reuses the shipping terminal's exact
/// bytes-per-z-coordinate rate (`z_rate_milli` is that rate in millibytes per
/// coordinate) and rice split, so the package terminal is priced at the same
/// per-coordinate encoding as the shipping wire while its `e`/`t` segments
/// scale exactly with the synthesized block count and audited rank.
#[allow(clippy::too_many_arguments)]
fn synthesize_terminal(
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    input_witness_len: usize,
    d: usize,
    m: usize,
    fold_log_basis: u32,
    fold_digit_count: usize,
    z_rate_milli: usize,
    z_rice_low_bits: u32,
) -> Result<(TerminalResponseShape, usize), AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    let n_live = input_witness_len.div_ceil(d);
    let b_blocks = n_live.div_ceil(m);
    // A terminal input is already one balanced digit per coefficient.
    let inner_width = decomposed_s_block_ring_count(m, 1).ok_or_else(|| no_layout("terminal A"))?;
    let fold_challenge_config = ring_challenge_config(d)?;
    let a_bucket = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        d,
        fold_log_basis,
        &fold_challenge_config,
        fold_digit_count,
    )
    .ok_or_else(|| no_layout("terminal A"))?;
    let inner = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key(policy, SisMatrixRole::Inner, d, a_bucket),
        inner_width,
    )?;
    let z_coords = inner_width
        .checked_mul(d)
        .ok_or_else(|| no_layout("terminal z"))?;
    let e_field_elems = b_blocks
        .checked_mul(d)
        .ok_or_else(|| no_layout("terminal e"))?;
    let t_field_elems = b_blocks
        .checked_mul(inner.output_rank())
        .and_then(|value| value.checked_mul(d))
        .ok_or_else(|| no_layout("terminal t"))?;
    let z_payload_bytes = z_coords
        .checked_mul(z_rate_milli)
        .ok_or_else(|| no_layout("terminal z budget"))?
        .div_ceil(1000);
    let logical_num_elems = z_coords + e_field_elems + t_field_elems;
    let shape = TerminalResponseShape {
        layout: TailSegmentLayout {
            ring_dimension: d,
            groups: vec![TailSegmentGroupLayout {
                z_coords,
                e_field_elems,
                t_field_elems,
                z_linf_cap: None,
                z_rice_low_bits,
                z_payload_bytes,
            }],
            logical_num_elems,
        },
    };
    let bytes = terminal_response_planner_bytes(field_bits, &shape, None);
    Ok((shape, bytes))
}

struct PackageRow {
    split_total: usize,
    fused_total: usize,
    rounds: LevelRounds,
    tail_knobs: [(usize, usize, usize); 2],
    terminal_knobs: (usize, usize, u32, usize),
    terminal_input: usize,
    terminal_bytes: usize,
    level_lens: [usize; 3],
    candidates_searched: usize,
    candidates_admissible: usize,
}

/// Exhaustive search over the synthesized four-level package
/// (shipping root + two basis-8 Linf tails + terminal), minimizing the fused
/// total. Candidates whose SIS lookups or layout constructions fail are
/// skipped.
fn best_package(
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>
        + Copy,
    schedule: &FoldSchedule,
    z_rate_milli: usize,
    z_rice_low_bits: u32,
) -> Result<Option<PackageRow>, AkitaError> {
    let elem_bytes = challenge_elem_bytes(policy)?;
    let root_params = &schedule.root.params;
    let root_input = schedule.root.input_witness_len;
    let root_output = schedule.root.output_witness_len;
    let mut best: Option<PackageRow> = None;
    let mut candidates_searched = 0usize;
    let mut candidates_admissible = 0usize;

    for &d1 in TAIL_RING_DIMS {
        for &ndf1 in TAIL_FOLD_DIGITS {
            for &blocks1 in TAIL_TARGET_BLOCKS {
                let Ok((tail1, out1)) = synthesize_tail_fold(
                    policy,
                    ring_challenge_config,
                    root_output,
                    d1,
                    ndf1,
                    blocks1,
                ) else {
                    continue;
                };
                for &d2 in TAIL_RING_DIMS {
                    for &ndf2 in TAIL_FOLD_DIGITS {
                        for &blocks2 in TAIL_TARGET_BLOCKS {
                            let Ok((tail2, out2)) = synthesize_tail_fold(
                                policy,
                                ring_challenge_config,
                                out1,
                                d2,
                                ndf2,
                                blocks2,
                            ) else {
                                continue;
                            };
                            for &dt in TERMINAL_RING_DIMS {
                                for &mt in TERMINAL_POSITIONS {
                                    for &lbt in TERMINAL_FOLD_LOG_BASIS {
                                        for &fdct in TERMINAL_FOLD_DIGITS {
                                            candidates_searched += 1;
                                            let Ok((_, terminal_bytes)) = synthesize_terminal(
                                                policy,
                                                ring_challenge_config,
                                                out2,
                                                dt,
                                                mt,
                                                lbt,
                                                fdct,
                                                z_rate_milli,
                                                z_rice_low_bits,
                                            ) else {
                                                continue;
                                            };
                                            let Ok(row) = price_package(
                                                policy,
                                                elem_bytes,
                                                root_params,
                                                root_input,
                                                root_output,
                                                &tail1,
                                                out1,
                                                &tail2,
                                                out2,
                                                terminal_bytes,
                                            ) else {
                                                continue;
                                            };
                                            candidates_admissible += 1;
                                            let row = PackageRow {
                                                tail_knobs: [
                                                    (d1, ndf1, blocks1),
                                                    (d2, ndf2, blocks2),
                                                ],
                                                terminal_knobs: (dt, mt, lbt, fdct),
                                                terminal_input: out2,
                                                terminal_bytes,
                                                level_lens: [root_output, out1, out2],
                                                ..row
                                            };
                                            if best
                                                .as_ref()
                                                .is_none_or(|b| row.fused_total < b.fused_total)
                                            {
                                                best = Some(row);
                                            }
                                            if let Some(best) = best.as_mut() {
                                                best.candidates_searched = candidates_searched;
                                                best.candidates_admissible = candidates_admissible;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
fn price_package(
    policy: &PlannerPolicy,
    elem_bytes: usize,
    root_params: &CommittedGroupParams,
    root_input: usize,
    root_output: usize,
    tail1: &CommittedGroupParams,
    out1: usize,
    tail2: &CommittedGroupParams,
    out2: usize,
    terminal_bytes: usize,
) -> Result<PackageRow, AkitaError> {
    let levels: [(
        &CommittedGroupParams,
        Option<&CommittedGroupParams>,
        usize,
        usize,
    ); 3] = [
        (root_params, Some(tail1), root_input, root_output),
        (tail1, Some(tail2), root_output, out1),
        (tail2, None, out1, out2),
    ];
    let mut split_total = 0usize;
    let mut fused_total = 0usize;
    let mut rounds = LevelRounds {
        split_chain: 0,
        fused_chain: 0,
        split_stream_bytes: 0,
        fused_stream_bytes: 0,
        l2_route: false,
    };
    for (params, successor, input_len, output_len) in levels {
        let (split_direct, split_stage3) =
            nonterminal_level_payload_bytes(policy, params, successor, input_len, output_len)?;
        let (fused_direct, fused_stage3) = fused_nonterminal_level_payload_bytes(
            policy, params, successor, input_len, output_len,
        )?;
        split_total += split_direct + split_stage3;
        fused_total += fused_direct + fused_stage3;
        let level = level_rounds(params, output_len, elem_bytes)?;
        rounds.split_chain += level.split_chain;
        rounds.fused_chain += level.fused_chain;
        rounds.split_stream_bytes += level.split_stream_bytes;
        rounds.fused_stream_bytes += level.fused_stream_bytes;
    }
    let terminal_eor = extension_opening_reduction_level_bytes(
        policy.challenge_field_bits()?,
        policy.claim_ext_degree,
        PolynomialGroupLayout::singleton(padded_boolean_opening_vars(out2)?),
    )?;
    let terminal_common = FOLD_GRIND_NONCE_BYTES + terminal_eor + terminal_bytes;
    Ok(PackageRow {
        split_total: split_total + terminal_common,
        fused_total: fused_total + terminal_common,
        rounds,
        tail_knobs: [(0, 0, 0); 2],
        terminal_knobs: (0, 0, 0, 0),
        terminal_input: out2,
        terminal_bytes,
        level_lens: [root_output, out1, out2],
        candidates_searched: 0,
        candidates_admissible: 0,
    })
}

fn run() -> Result<(), AkitaError> {
    type Cfg = fp128::Dense;
    let policy = policy_of::<Cfg>();
    let elem_bytes = challenge_elem_bytes(&policy)?;
    let table = Cfg::schedule_catalog().ok_or_else(|| {
        AkitaError::UnsupportedSchedule(
            "fp128_dense generated table is not enabled; build with --features catalog-check"
                .to_string(),
        )
    })?;

    println!(
        "fused-check frontier (fp128_dense, exact planner accounting, 16 B challenge elements)"
    );
    println!();

    for &num_vars in REPRESENTATIVE_NUM_VARS {
        let final_group = PolynomialGroupLayout::new(num_vars, 1);
        let Some(entry) = table.entries.iter().find(|entry| {
            entry.final_group == final_group && entry.root.precommitted_groups.is_empty()
        }) else {
            println!("== 2^{num_vars}: no scalar generated row; skipped ==");
            continue;
        };
        let key = entry.to_runtime_lookup_key();
        let schedule = schedule_from_entry(entry, &key, &policy, Cfg::ring_challenge_config)?;
        let split_total = expanded_schedule_proof_payload_bytes(&key, &schedule, &policy)?;
        let fused_total = expanded_schedule_fused_proof_payload_bytes(&key, &schedule, &policy)?;
        let (rounds, l2_levels) = schedule_rounds(&schedule, elem_bytes)?;
        let nonterminal_levels = 1 + schedule.recursive_folds.len();

        let shipping_terminal = &schedule.terminal.response_shape.layout.groups[0];
        let z_rate_milli =
            (shipping_terminal.z_payload_bytes * 1000).div_ceil(shipping_terminal.z_coords.max(1));
        let package = best_package(
            &policy,
            Cfg::ring_challenge_config,
            &schedule,
            z_rate_milli,
            shipping_terminal.z_rice_low_bits,
        )?;

        println!("== 2^{num_vars} (scalar dense row) ==");
        println!(
            "  shipping split   : {nonterminal_levels} nonterminal levels + terminal, \
             chain {} rounds, round stream {} B, TOTAL {} B{}",
            rounds.split_chain,
            rounds.split_stream_bytes,
            split_total,
            if l2_levels > 0 {
                format!(" ({l2_levels} selective-L2 levels)")
            } else {
                String::new()
            }
        );
        println!(
            "  fused same-geom  : chain {} rounds, round stream {} B, TOTAL {} B (delta {:+} B; \
             levels above basis 8 fused at their own basis; L2 norm payloads carried unchanged)",
            rounds.fused_chain,
            rounds.fused_stream_bytes,
            fused_total,
            fused_total as i64 - split_total as i64,
        );
        match package {
            Some(row) => {
                println!(
                    "  package split    : 3 nonterminal levels + terminal, chain {} rounds, \
                     round stream {} B, TOTAL {} B",
                    row.rounds.split_chain, row.rounds.split_stream_bytes, row.split_total,
                );
                println!(
                    "  PACKAGE (fused 4-level, all-Linf, b=8): chain {} rounds, round stream {} B, \
                     TOTAL {} B (delta vs shipping split {:+} B)",
                    row.rounds.fused_chain,
                    row.rounds.fused_stream_bytes,
                    row.fused_total,
                    row.fused_total as i64 - split_total as i64,
                );
                println!(
                    "    geometry: witness {} -> {} -> {} -> terminal ({} B response, input {}), \
                     tails (d, fold digits, target blocks) = {:?}, terminal (d, M, fold lb, fold digits) = {:?}, \
                     terminal z rate {} mB/coord",
                    row.level_lens[0],
                    row.level_lens[1],
                    row.level_lens[2],
                    row.terminal_bytes,
                    row.terminal_input,
                    row.tail_knobs,
                    row.terminal_knobs,
                    z_rate_milli,
                );
                println!(
                    "    search: {} of {} grid candidates admissible under the audited SIS tables",
                    row.candidates_admissible, row.candidates_searched,
                );
                let verdict = if row.fused_total <= split_total {
                    "CLEARS"
                } else {
                    "DOES NOT CLEAR"
                };
                println!(
                    "    OPEN-7 verdict: the fused package total {} the shipping wire total \
                     ({} B vs {} B).",
                    verdict, row.fused_total, split_total,
                );
            }
            None => println!("  package          : no admissible synthesized 4-level geometry"),
        }
        println!();
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("fused_frontier failed: {error}");
        std::process::exit(1);
    }
}
