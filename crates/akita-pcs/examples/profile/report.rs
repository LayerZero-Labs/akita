use akita_field::{CanonicalField, FieldCore};
use akita_prover::{FoldGrindObservation, PreparedCrtNttProfile};
use akita_serialization::{AkitaSerialize, Compress};
use akita_types::{
    golomb_rice::{golomb_rice_low_bits_sweep_payload_bytes, rice_low_bits_for_cap},
    layout::proof_size::field_bytes,
    schedule_terminal_direct_witness_shape, tail_segment_multiplicities_from_layout,
    z_fold_decoded_from_segment, z_fold_encoding_stats_from_segment, AkitaBatchedProof,
    AkitaBatchedRootProof, AkitaLevelProof, CleartextWitnessProof, CleartextWitnessShape,
    LevelParams, Schedule, SetupSumcheckProof, Step, TerminalLevelProof, ZFoldEncodingStats,
};

const TAIL_Z_LENGTH_PREFIX_BYTES: usize = 8;

pub(crate) fn report_timing(label: &str, phase: &str, elapsed_s: f64) {
    tracing::info!(label, elapsed_s, "{phase}");
    eprintln!("[{label}] {phase}: {elapsed_s:.6}s");
}

/// Structured tail witness report for profile bench / CI (`scripts/profile_bench_report.py`).
pub(crate) fn emit_proof_tail_report<FF, L>(
    label: &str,
    proof: &AkitaBatchedProof<FF, L>,
    schedule: &Schedule,
    field_bits: u32,
) where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    L: FieldCore,
{
    if proof.is_root_direct() {
        tracing::info!(
            label,
            tail_bytes = 0u32,
            final_w_encoding = "none",
            final_w_policy = "root_direct",
            "proof tail summary"
        );
        eprintln!("[{label}]   final_w: none (root-direct zero-fold; no cleartext tail witness)");
        return;
    }

    let final_w = proof.final_witness();
    let tail_bytes = final_w.serialized_size(Compress::No);
    let num_elems = final_w.num_elems();

    if let Some(segment) = final_w.as_segment_typed() {
        let field_sz = field_bytes(FF::modulus_bits());
        let ring_dim = segment.layout.ring_dimension;
        let z_golomb_bytes = segment.z_payload.len();
        let z_field_elems = segment.layout.z_coords;
        let z_ring_elems = z_field_elems / ring_dim.max(1);
        let z_wire_bytes = TAIL_Z_LENGTH_PREFIX_BYTES.saturating_add(z_golomb_bytes);
        let e_field_elems = segment.e_fields.coeff_len();
        let t_field_elems = segment.t_fields.coeff_len();
        let r_field_elems = segment.r_fields.coeff_len();
        let e_ring_elems = e_field_elems / ring_dim.max(1);
        let t_ring_elems = t_field_elems / ring_dim.max(1);
        let r_ring_elems = r_field_elems / ring_dim.max(1);
        let e_bytes = e_field_elems.saturating_mul(field_sz);
        let t_bytes = t_field_elems.saturating_mul(field_sz);
        let r_bytes = r_field_elems.saturating_mul(field_sz);
        let z_budget_bytes = schedule_terminal_direct_witness_shape(schedule)
            .ok()
            .and_then(|shape| match shape {
                CleartextWitnessShape::SegmentTyped(scheduled) => Some(scheduled.z_payload_bytes),
                _ => None,
            })
            .unwrap_or(0);
        let z_slack_bytes = z_budget_bytes.saturating_sub(z_golomb_bytes);
        let z_stats = segment_typed_z_fold_stats(segment, schedule, field_bits).ok();
        let z_witness_linf_cap = z_stats.as_ref().map(|s| s.witness_linf_cap).unwrap_or(0);
        let z_rice_low_bits_wire = z_stats.as_ref().map(|s| s.rice_low_bits_wire).unwrap_or(0);
        let z_rice_low_bits_cap = z_stats.as_ref().map(|s| s.rice_low_bits_cap).unwrap_or(0);
        let z_stats_coords = z_stats.as_ref().map(|s| s.coord_count).unwrap_or(0);
        let z_bits_per_coord_golomb = z_stats
            .as_ref()
            .map(|s| s.bits_per_coord_at_wire)
            .unwrap_or(0.0);
        let z_bits_per_coord_packed = z_stats
            .as_ref()
            .map(|s| s.bits_per_coord_packed_digits)
            .unwrap_or(0.0);
        let z_packed_hypothetical_bytes = z_stats
            .as_ref()
            .map(|s| s.total_bits_packed_digits.div_ceil(8))
            .unwrap_or(0);
        let z_golomb_savings_bytes = z_packed_hypothetical_bytes.saturating_sub(z_golomb_bytes);

        tracing::info!(
            label,
            tail_bytes,
            final_w_num_elems = num_elems,
            final_w_encoding = "segment_typed",
            final_w_policy = "non_zk_default",
            tail_log_basis = segment.layout.log_basis,
            tail_z_prefix_bytes = TAIL_Z_LENGTH_PREFIX_BYTES,
            tail_z_golomb_bytes = z_golomb_bytes,
            tail_z_bytes = z_wire_bytes,
            tail_z_field_elems = z_field_elems,
            tail_z_ring_elems = z_ring_elems,
            tail_z_budget_bytes = z_budget_bytes,
            tail_z_slack_bytes = z_slack_bytes,
            tail_e_field_elems = e_field_elems,
            tail_e_ring_elems = e_ring_elems,
            tail_t_field_elems = t_field_elems,
            tail_t_ring_elems = t_ring_elems,
            tail_r_field_elems = r_field_elems,
            tail_r_ring_elems = r_ring_elems,
            tail_e_bytes = e_bytes,
            tail_t_bytes = t_bytes,
            tail_r_bytes = r_bytes,
            z_witness_linf_cap,
            z_rice_low_bits_wire,
            z_rice_low_bits_cap,
            z_coords = z_stats_coords,
            z_bits_per_coord_golomb,
            z_bits_per_coord_packed,
            z_packed_hypothetical_bytes,
            z_golomb_savings_bytes,
            "proof tail summary"
        );

        let golomb_line = z_stats
            .map(|stats| {
                format!(
                    " Golomb z: witness_linf_cap={} wire_low_bits={} cap_low_bits={} sample_low_bits={} ring_elems={z_ring_elems} field_coeffs={} \
                     {:.2} bits/coord@wire vs {:.2}@sample vs packed {:.2} bits/field_coeff \
                     (hypothetical packed z={} B, savings={} B); \
                     planner z budget={z_budget_bytes} B (slack {z_slack_bytes} B); \
                     dist max={} median={} p90={} p99={}",
                    stats.witness_linf_cap,
                    stats.rice_low_bits_wire,
                    stats.rice_low_bits_cap,
                    stats.rice_low_bits_sample,
                    stats.coord_count,
                    stats.bits_per_coord_at_wire,
                    stats.bits_per_coord_at_sample,
                    stats.bits_per_coord_packed_digits,
                    stats.total_bits_packed_digits.div_ceil(8),
                    stats
                        .total_bits_packed_digits
                        .div_ceil(8)
                        .saturating_sub(z_golomb_bytes),
                    stats.observed_max_abs,
                    stats.median_abs,
                    stats.p90_abs,
                    stats.p99_abs,
                )
            })
            .unwrap_or_default();

        eprintln!(
            "[{label}]   final_w: encoding=segment_typed (non-zk default), total={tail_bytes} bytes, \
             logical_elems={num_elems}, log_basis={}{}",
            segment.layout.log_basis,
            golomb_line,
        );
        eprintln!(
            "[{label}]     z: {z_wire_bytes} B (len_prefix={TAIL_Z_LENGTH_PREFIX_BYTES} + golomb={z_golomb_bytes}), \
             field_coeffs={z_field_elems}, ring_elems={z_ring_elems}",
        );
        eprintln!(
            "[{label}]     e: {e_bytes} B, field_coeffs={e_field_elems}, ring_elems={e_ring_elems}",
        );
        eprintln!(
            "[{label}]     t: {t_bytes} B, field_coeffs={t_field_elems}, ring_elems={t_ring_elems}",
        );
        eprintln!(
            "[{label}]     r: {r_bytes} B, field_coeffs={r_field_elems}, ring_elems={r_ring_elems}",
        );
        if std::env::var("AKITA_Z_GOLOMB_SWEEP").ok().as_deref() == Some("1") {
            emit_z_golomb_k_sweep(label, segment, schedule, field_bits, z_golomb_bytes);
        }
        return;
    }

    tracing::info!(
        label,
        tail_bytes,
        final_w_num_elems = num_elems,
        final_w_encoding = "field_elements",
        final_w_policy = "root_direct_witness",
        "proof tail summary"
    );
    eprintln!(
        "[{label}]   final_w: encoding=field_elements (root-direct witness), \
         total={tail_bytes} bytes, field_elems={num_elems}",
    );
}

fn segment_typed_z_fold_stats<FF: FieldCore>(
    witness: &akita_types::SegmentTypedWitness<FF>,
    schedule: &Schedule,
    field_bits: u32,
) -> Result<ZFoldEncodingStats, akita_field::AkitaError> {
    let terminal_fold_level = schedule.num_fold_levels().saturating_sub(1);
    let terminal_scheduled = schedule.get_execution_schedule(terminal_fold_level)?;
    let lp = &terminal_scheduled.params;
    let Ok((_num_w_vectors, num_t_vectors, _num_public_rows)) =
        tail_segment_multiplicities_from_layout(lp, &witness.layout)
    else {
        return Err(akita_field::AkitaError::InvalidSetup(
            "tail segment multiplicities".to_string(),
        ));
    };
    z_fold_encoding_stats_from_segment(witness, lp, num_t_vectors, field_bits)
}

fn emit_z_golomb_k_sweep<FF: FieldCore>(
    label: &str,
    witness: &akita_types::SegmentTypedWitness<FF>,
    schedule: &Schedule,
    field_bits: u32,
    actual_z_payload_bytes: usize,
) {
    let terminal_fold_level = schedule.num_fold_levels().saturating_sub(1);
    let Ok(terminal_scheduled) = schedule.get_execution_schedule(terminal_fold_level) else {
        return;
    };
    let lp = &terminal_scheduled.params;
    let Ok((_num_w_vectors, num_t_vectors, _num_public_rows)) =
        tail_segment_multiplicities_from_layout(lp, &witness.layout)
    else {
        return;
    };
    let Ok(z_values) = z_fold_decoded_from_segment(witness, lp, num_t_vectors) else {
        return;
    };
    let Ok(stats) = z_fold_encoding_stats_from_segment(witness, lp, num_t_vectors, field_bits)
    else {
        return;
    };
    let low_bits_hi = stats
        .rice_low_bits_cap
        .saturating_add(4)
        .max(stats.rice_low_bits_sample);
    let Ok(sweep) =
        golomb_rice_low_bits_sweep_payload_bytes(&z_values, stats.zigzag_w, low_bits_hi)
    else {
        return;
    };
    let low_bits_observed = rice_low_bits_for_cap(u128::from(stats.observed_max_abs));
    eprintln!(
        "[{label}]   z_golomb_low_bits_sweep (coords={}):",
        z_values.len()
    );
    for &(rice_low_bits, bytes) in &sweep {
        let marker = if rice_low_bits == stats.rice_low_bits_wire {
            "  <-- wire low bits"
        } else if rice_low_bits == stats.rice_low_bits_cap {
            "  <-- cap low bits (planner reference)"
        } else if rice_low_bits == stats.rice_low_bits_sample {
            "  <-- sample-optimal on this witness"
        } else if rice_low_bits == low_bits_observed {
            "  <-- low bits from observed max only (NOT sound)"
        } else {
            ""
        };
        let delta = bytes as i64 - actual_z_payload_bytes as i64;
        eprintln!(
            "[{label}]     low_bits={rice_low_bits:2}: payload={bytes:6} B ({:.2} bits/coord, delta_vs_actual={delta:+}){marker}",
            (bytes.saturating_mul(8)) as f64 / z_values.len().max(1) as f64,
        );
    }
    if let Some((rice_low_bits, bytes)) = sweep.iter().min_by_key(|(_, b)| *b) {
        let save_vs_beta = actual_z_payload_bytes.saturating_sub(*bytes);
        eprintln!(
            "[{label}]   z_golomb_sweep_summary: best low_bits={rice_low_bits} -> {bytes} B \
             (vs actual {actual_z_payload_bytes} B at wire_low_bits={}, delta {save_vs_beta} B; \
             wire low bits must be >= best for honest encodes)",
            stats.rice_low_bits_wire,
        );
    }
}

/// Surface the prepared-setup memory footprint: the plain shared setup vector
/// (`Vec<F>`) versus the NTT slot cache built from it. The cache stores both
/// negacyclic and cyclic CRT-NTT forms, so it is several times larger than the
/// vector; reporting both makes that expansion visible in the bench report.
pub(crate) fn report_setup_sizes(
    label: &str,
    setup_ring_elements: usize,
    setup_vector_bytes: usize,
    setup_ntt_cache_bytes: usize,
) {
    tracing::info!(
        label,
        setup_ring_elements,
        setup_vector_bytes,
        setup_ntt_cache_bytes,
        "setup sizes"
    );
    eprintln!(
        "[{label}] setup sizes: ring_elems={setup_ring_elements}, vector={setup_vector_bytes} bytes, ntt_cache={setup_ntt_cache_bytes} bytes"
    );
}

pub(crate) fn report_crt_profile(label: &str, profile: PreparedCrtNttProfile) {
    tracing::info!(
        label,
        crt_profile = profile.profile_id,
        crt_num_primes = profile.num_primes,
        crt_limb_bits = profile.limb_bits,
        max_i8_log_basis = profile.max_i8_log_basis,
        balanced_digit_safe_width = profile.balanced_digit_safe_width,
        raw_i8_safe_width = profile.raw_i8_safe_width,
        "CRT NTT profile"
    );
    eprintln!(
        "[{label}] CRT NTT profile: profile={}, K={}, limb_bits={}, max_i8_log_basis={}, balanced_digit_safe_width={}, raw_i8_safe_width={}",
        profile.profile_id,
        profile.num_primes,
        profile.limb_bits,
        profile.max_i8_log_basis,
        profile.balanced_digit_safe_width,
        profile.raw_i8_safe_width
    );
}

pub(crate) fn emit_runtime_schedule_summary(
    label: &str,
    schedule: &Schedule,
    root_num_claims: usize,
    field_bits: u32,
) {
    let levels = schedule
        .steps
        .iter()
        .filter(|step| matches!(step, Step::Fold(_)))
        .count();
    tracing::info!(
        label,
        levels,
        total_proof_bytes = schedule.total_bytes,
        "runtime schedule"
    );

    for (level_idx, level) in schedule
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::Fold(level) => Some(level),
            Step::Direct(_) => None,
        })
        .enumerate()
    {
        let lp = &level.params;
        let num_claims = if level_idx == 0 { root_num_claims } else { 1 };
        tracing::info!(
            label,
            level = level_idx,
            d = lp.ring_dimension,
            n_a = lp.a_key.row_len(),
            n_b = lp.b_key.row_len(),
            n_d = lp.d_key.row_len(),
            challenge_l1_mass = lp.challenge_l1_mass(),
            log_basis = lp.log_basis,
            m_vars = lp.m_vars,
            r_vars = lp.r_vars,
            num_blocks = lp.num_blocks,
            block_len = lp.block_len,
            delta_commit = lp.num_digits_commit,
            delta_open = lp.num_digits_open,
            delta_fold = lp.num_digits_fold(num_claims, field_bits).unwrap_or(0),
            current_w_len = level.current_w_len,
            next_w_ring = level.next_w_len / lp.ring_dimension,
            next_w_len = level.next_w_len,
            level_bytes = level.level_bytes,
            "planned fold level"
        );
    }

    if let Some(Step::Direct(terminal)) = schedule.steps.last() {
        tracing::info!(
            label,
            final_w_len = terminal.current_w_len,
            final_log_basis = terminal.log_basis(field_bits),
            "planned terminal state"
        );
    }
}

fn ring_elem_count(coeff_len: usize, d: usize) -> usize {
    coeff_len / d
}

fn extension_opening_reduction_sizes<L: FieldCore + AkitaSerialize>(
    reduction: Option<&akita_types::ExtensionOpeningReductionProof<L>>,
) -> (usize, usize) {
    reduction.map_or((0, 0), |reduction| {
        let partials = reduction
            .partials
            .iter()
            .map(|value| value.serialized_size(Compress::No))
            .sum();
        let sumcheck = reduction.sumcheck.serialized_size(Compress::No);
        (partials, sumcheck)
    })
}

fn stage3_sumcheck_size<L: FieldCore + AkitaSerialize>(
    proof: Option<&SetupSumcheckProof<L>>,
) -> usize {
    proof.map_or(0, |proof| {
        proof.claim.serialized_size(Compress::No)
            + proof.next_w_eval.serialized_size(Compress::No)
            + proof.sumcheck.serialized_size(Compress::No)
    })
}

/// Total serialized bytes of the recursive-mode stage-3 setup-product
/// sumcheck payloads across every non-terminal fold level (the folded root and
/// each intermediate step). This is the proof-size overhead that
/// `SetupContributionMode::Recursive` adds on top of the direct-mode payload
/// priced by `akita_types::level_proof_bytes`; terminal levels carry no
/// stage-3 proof and contribute zero.
pub(crate) fn observed_stage3_setup_product_bytes<FF, L>(proof: &AkitaBatchedProof<FF, L>) -> usize
where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    let root_bytes = proof.root.as_fold().map_or(0, |fold| {
        stage3_sumcheck_size(fold.stage3_sumcheck_proof.as_ref())
    });
    let step_bytes: usize = proof
        .steps
        .iter()
        .map(|step| match step {
            AkitaLevelProof::Intermediate { .. } => {
                stage3_sumcheck_size(step.stage3_sumcheck_proof())
            }
            AkitaLevelProof::Terminal { .. } => 0,
        })
        .sum();
    root_bytes + step_bytes
}

fn fold_grind_nonce_wire_bytes() -> usize {
    0u32.serialized_size(Compress::No)
}

fn take_fold_grind_observation<'a>(
    grind_observations: &'a [FoldGrindObservation],
    obs_idx: &mut usize,
    grind_nonce: u32,
) -> &'a FoldGrindObservation {
    let obs = grind_observations.get(*obs_idx).unwrap_or_else(|| {
        panic!(
            "missing fold grind observation for level obs_idx={obs_idx} grind_nonce={grind_nonce}"
        )
    });
    assert_eq!(
        obs.grind_nonce, grind_nonce,
        "fold grind observation nonce mismatch at obs_idx={obs_idx}"
    );
    *obs_idx += 1;
    obs
}

fn collect_fold_grind_nonces<FF, L>(proof: &AkitaBatchedProof<FF, L>) -> Vec<u32>
where
    FF: FieldCore,
    L: FieldCore,
{
    if proof.is_root_direct() {
        return Vec::new();
    }

    let mut nonces = Vec::with_capacity(1 + proof.steps.len());
    if let Ok(root_nonce) = proof.root.fold_grind_nonce() {
        nonces.push(root_nonce);
    }
    for step in &proof.steps {
        nonces.push(step.fold_grind_nonce());
    }
    nonces
}

fn emit_fold_grind_summary(label: &str, grind_observations: &[FoldGrindObservation]) {
    if grind_observations.is_empty() {
        tracing::info!(label, grind_levels = 0u32, "fold grind summary");
        eprintln!("[{label}] fold grind: no tail-bound fold levels");
        return;
    }

    let max_nonce = grind_observations
        .iter()
        .map(|obs| obs.grind_nonce)
        .max()
        .expect("non-empty observation list");
    let attempts_sum: u64 = grind_observations
        .iter()
        .map(|obs| u64::from(obs.grind_probe_count))
        .sum();
    let nonces_csv = grind_observations
        .iter()
        .map(|obs| obs.grind_nonce.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let probe_counts_csv = grind_observations
        .iter()
        .map(|obs| obs.grind_probe_count.to_string())
        .collect::<Vec<_>>()
        .join(",");

    tracing::info!(
        label,
        grind_levels = grind_observations.len(),
        grind_nonce_max = max_nonce,
        grind_attempts_sum = attempts_sum,
        grind_nonces = nonces_csv.as_str(),
        grind_probe_counts = probe_counts_csv.as_str(),
        "fold grind summary"
    );
    eprintln!(
        "[{label}] fold grind: levels={}, attempts_sum={}, max_nonce={}, nonces=[{nonces_csv}], probe_counts=[{probe_counts_csv}]",
        grind_observations.len(),
        attempts_sum,
        max_nonce,
    );
}

fn print_akita_level_breakdown<FF, L, const D: usize>(
    label: &str,
    level_idx: usize,
    level: &AkitaLevelProof<FF, L>,
    grind_observations: &[FoldGrindObservation],
    obs_idx: &mut usize,
) -> usize
where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(level.extension_opening_reduction());
    let v_size = level.v().serialized_size(Compress::No);
    let total = level.serialized_size(Compress::No);
    let stage2_intermediate = level
        .stage2()
        .as_intermediate()
        .expect("Akita level proof must carry intermediate stage-2 proof");

    eprintln!("[{label}]   akita_fold L{level_idx}: total={total} bytes");
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        ring_elem_count(level.v().coeff_len(), D),
        D,
    );
    let stage1 = level.stage1();
    let stage1_sumcheck_size = stage1
        .stages
        .iter()
        .map(|stage| stage.sumcheck_proof.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_interstage_claims_size = stage1
        .stages
        .iter()
        .flat_map(|stage| stage.child_claims.iter())
        .map(|claim| claim.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_s_claim_size = stage1.s_claim.serialized_size(Compress::No);
    let stage2_sumcheck_size = stage2_intermediate
        .sumcheck_proof
        .serialized_size(Compress::No);
    let stage3_sumcheck_size = stage3_sumcheck_size(level.stage3_sumcheck_proof());
    let next_w_commitment_size = stage2_intermediate
        .next_w_commitment
        .serialized_size(Compress::No);
    let next_w_eval_size = stage2_intermediate
        .next_w_eval()
        .serialized_size(Compress::No);
    let fold_grind_nonce_size = fold_grind_nonce_wire_bytes();
    let grind_nonce = level.fold_grind_nonce();
    let grind_observation = take_fold_grind_observation(grind_observations, obs_idx, grind_nonce);
    let grind_attempts = grind_observation.grind_probe_count;

    tracing::info!(
        label,
        level = level_idx,
        d = D,
        total_bytes = total,
        extension_opening_partials_bytes = extension_opening_partials_size,
        extension_opening_sumcheck_bytes = extension_opening_sumcheck_size,
        v_bytes = v_size,
        fold_grind_nonce_bytes = fold_grind_nonce_size,
        grind_nonce,
        grind_attempts,
        stage1_sumcheck_bytes = stage1_sumcheck_size,
        stage1_interstage_claims_bytes = stage1_interstage_claims_size,
        stage1_s_claim_bytes = stage1_s_claim_size,
        stage2_sumcheck_bytes = stage2_sumcheck_size,
        stage3_sumcheck_bytes = stage3_sumcheck_size,
        next_w_commitment_bytes = next_w_commitment_size,
        next_w_eval_bytes = next_w_eval_size,
        "proof fold level"
    );
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!("[{label}]     fold_grind_nonce={fold_grind_nonce_size} bytes");
    eprintln!("[{label}]     stage1_sumcheck={stage1_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_interstage_claims={stage1_interstage_claims_size} bytes");
    eprintln!("[{label}]     stage1_s_claim={stage1_s_claim_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!("[{label}]     stage3_sumcheck={stage3_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     next_w_commitment={next_w_commitment_size} bytes ({} coeffs)",
        stage2_intermediate.next_w_commitment.coeff_len(),
    );
    eprintln!("[{label}]     next_w_eval={next_w_eval_size} bytes");
    assert_eq!(
        total,
        extension_opening_partials_size
            + extension_opening_sumcheck_size
            + v_size
            + fold_grind_nonce_size
            + stage1_sumcheck_size
            + stage1_interstage_claims_size
            + stage1_s_claim_size
            + stage2_sumcheck_size
            + stage3_sumcheck_size
            + next_w_commitment_size
            + next_w_eval_size
    );
    total
}

trait TerminalProofView<FF: FieldCore, L: FieldCore>: AkitaSerialize {
    fn extension_opening_reduction(
        &self,
    ) -> Option<&akita_types::ExtensionOpeningReductionProof<L>>;
    fn stage2(&self) -> &akita_types::AkitaStage2Proof<FF, L>;
    fn final_witness(&self) -> &CleartextWitnessProof<FF>;
    fn fold_grind_nonce_value(&self) -> u32;
}

impl<FF: FieldCore + CanonicalField + AkitaSerialize, L: FieldCore + AkitaSerialize>
    TerminalProofView<FF, L> for TerminalLevelProof<FF, L>
{
    fn extension_opening_reduction(
        &self,
    ) -> Option<&akita_types::ExtensionOpeningReductionProof<L>> {
        self.extension_opening_reduction.as_ref()
    }

    fn stage2(&self) -> &akita_types::AkitaStage2Proof<FF, L> {
        &self.stage2
    }

    fn final_witness(&self) -> &CleartextWitnessProof<FF> {
        self.final_witness()
    }

    fn fold_grind_nonce_value(&self) -> u32 {
        self.fold_grind_nonce
    }
}

impl<FF: FieldCore + CanonicalField + AkitaSerialize, L: FieldCore + AkitaSerialize>
    TerminalProofView<FF, L> for AkitaLevelProof<FF, L>
{
    fn extension_opening_reduction(
        &self,
    ) -> Option<&akita_types::ExtensionOpeningReductionProof<L>> {
        self.extension_opening_reduction()
    }

    fn stage2(&self) -> &akita_types::AkitaStage2Proof<FF, L> {
        self.stage2()
    }

    fn final_witness(&self) -> &CleartextWitnessProof<FF> {
        self.stage2()
            .final_witness()
            .expect("terminal Akita level proof must carry final witness")
    }

    fn fold_grind_nonce_value(&self) -> u32 {
        self.fold_grind_nonce()
    }
}

fn print_terminal_level_breakdown<FF, L, P, const D: usize>(
    label: &str,
    level_idx: usize,
    level: &P,
    root_variant: &'static str,
    grind_observations: &[FoldGrindObservation],
    obs_idx: &mut usize,
) -> usize
where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
    P: TerminalProofView<FF, L>,
{
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(level.extension_opening_reduction());
    let stage2_sumcheck_size = { level.stage2().sumcheck().serialized_size(Compress::No) };
    let final_witness_size = level.final_witness().serialized_size(Compress::No);
    let fold_grind_nonce_size = fold_grind_nonce_wire_bytes();
    let grind_nonce = level.fold_grind_nonce_value();
    let grind_observation = take_fold_grind_observation(grind_observations, obs_idx, grind_nonce);
    let grind_attempts = grind_observation.grind_probe_count;
    let full = level.serialized_size(Compress::No);
    // `total_bytes` excludes `final_witness` to mirror the planner's
    // `terminal_level_proof_bytes`. `final_witness` is reported separately as
    // the proof tail (`tail_bytes`) and accounted for in `accounted_bytes`.
    let total = full - final_witness_size;

    // Only the fields structurally present in `TerminalLevelProof` are
    // emitted: optional extension-opening reduction, the
    // stage-2 sumcheck, and `final_witness`. The intermediate-level
    // fields (`v`, `stage1_*`, `stage3_sumcheck`, `next_w_*`) are absent at
    // terminal and therefore omitted from the tracing payload; downstream
    // parsers default missing keys to zero.
    tracing::info!(
        label,
        level = level_idx,
        d = D,
        total_bytes = total,
        extension_opening_partials_bytes = extension_opening_partials_size,
        extension_opening_sumcheck_bytes = extension_opening_sumcheck_size,
        fold_grind_nonce_bytes = fold_grind_nonce_size,
        grind_nonce,
        grind_attempts,
        stage2_sumcheck_bytes = stage2_sumcheck_size,
        final_witness_bytes = final_witness_size,
        root_variant = root_variant,
        "proof fold level"
    );

    let header = if level_idx == 0 {
        "batched_root (terminal)".to_string()
    } else {
        format!("akita_fold L{level_idx} (terminal)")
    };
    eprintln!(
        "[{label}]   {header}: total={total} bytes (excl. final_witness={final_witness_size})"
    );
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!("[{label}]     fold_grind_nonce={fold_grind_nonce_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!("[{label}]     final_witness={final_witness_size} bytes (absorbed via transcript)");
    assert_eq!(
        full,
        extension_opening_partials_size
            + extension_opening_sumcheck_size
            + fold_grind_nonce_size
            + stage2_sumcheck_size
            + final_witness_size
    );
    total
}

fn print_batched_root_breakdown<FF, L, const D: usize>(
    label: &str,
    root: &AkitaBatchedRootProof<FF, L>,
    grind_observations: &[FoldGrindObservation],
    obs_idx: &mut usize,
) -> usize
where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    if let Some(terminal) = root.as_terminal_root() {
        return print_terminal_level_breakdown::<FF, L, _, D>(
            label,
            0,
            terminal,
            "terminal",
            grind_observations,
            obs_idx,
        );
    }
    let Some(fold) = root.as_fold() else {
        let total = root.serialized_size(Compress::No);
        eprintln!("[{label}]   batched_root: total={total} bytes (root-direct)");
        // Root-direct is a single bare witness payload with no folded
        // substructure. Only `total_bytes` and `root_variant` are
        // structurally meaningful here; all per-component fields are
        // omitted (parsers default missing keys to zero).
        tracing::info!(
            label,
            level = 0usize,
            d = D,
            total_bytes = total,
            root_variant = "direct",
            "proof fold level"
        );
        return total;
    };
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(fold.extension_opening_reduction.as_ref());
    let v_size = fold.v.serialized_size(Compress::No);
    let total = fold.serialized_size(Compress::No);
    let stage1 = &fold.stage1;
    let stage2_intermediate = fold
        .stage2
        .as_intermediate()
        .expect("fold root proof must carry intermediate stage-2 proof");
    let stage1_sumcheck_size = stage1
        .stages
        .iter()
        .map(|stage| stage.sumcheck_proof.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_interstage_claims_size = stage1
        .stages
        .iter()
        .flat_map(|stage| stage.child_claims.iter())
        .map(|claim| claim.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_s_claim_size = stage1.s_claim.serialized_size(Compress::No);
    let stage2_sumcheck_size = stage2_intermediate
        .sumcheck_proof
        .serialized_size(Compress::No);
    let stage3_sumcheck_size = stage3_sumcheck_size(fold.stage3_sumcheck_proof.as_ref());
    let next_w_commitment_size = stage2_intermediate
        .next_w_commitment
        .serialized_size(Compress::No);
    let next_w_eval_size = stage2_intermediate
        .next_w_eval()
        .serialized_size(Compress::No);
    let fold_grind_nonce_size = fold_grind_nonce_wire_bytes();
    let grind_nonce = fold.fold_grind_nonce;
    let grind_observation = take_fold_grind_observation(grind_observations, obs_idx, grind_nonce);
    let grind_attempts = grind_observation.grind_probe_count;

    tracing::info!(
        label,
        level = 0usize,
        d = D,
        total_bytes = total,
        extension_opening_partials_bytes = extension_opening_partials_size,
        extension_opening_sumcheck_bytes = extension_opening_sumcheck_size,
        v_bytes = v_size,
        fold_grind_nonce_bytes = fold_grind_nonce_size,
        grind_nonce,
        grind_attempts,
        stage1_sumcheck_bytes = stage1_sumcheck_size,
        stage1_interstage_claims_bytes = stage1_interstage_claims_size,
        stage1_s_claim_bytes = stage1_s_claim_size,
        stage2_sumcheck_bytes = stage2_sumcheck_size,
        stage3_sumcheck_bytes = stage3_sumcheck_size,
        next_w_commitment_bytes = next_w_commitment_size,
        next_w_eval_bytes = next_w_eval_size,
        root_variant = "fold",
        "proof fold level"
    );
    eprintln!("[{label}]   batched_root: total={total} bytes");
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        ring_elem_count(fold.v.coeff_len(), D),
        D,
    );
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!("[{label}]     fold_grind_nonce={fold_grind_nonce_size} bytes");
    eprintln!("[{label}]     stage1_sumcheck={stage1_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_interstage_claims={stage1_interstage_claims_size} bytes");
    eprintln!("[{label}]     stage1_s_claim={stage1_s_claim_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!("[{label}]     stage3_sumcheck={stage3_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     next_w_commitment={next_w_commitment_size} bytes ({} coeffs)",
        stage2_intermediate.next_w_commitment.coeff_len(),
    );
    eprintln!("[{label}]     next_w_eval={next_w_eval_size} bytes");
    assert_eq!(
        total,
        extension_opening_partials_size
            + extension_opening_sumcheck_size
            + v_size
            + fold_grind_nonce_size
            + stage1_sumcheck_size
            + stage1_interstage_claims_size
            + stage1_s_claim_size
            + stage2_sumcheck_size
            + stage3_sumcheck_size
            + next_w_commitment_size
            + next_w_eval_size
    );
    total
}

pub(crate) fn print_batched_proof_summary<FF, L, const D: usize>(
    label: &str,
    proof: &AkitaBatchedProof<FF, L>,
    grind_observations: &[FoldGrindObservation],
) where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    let root_total = proof.root.serialized_size(Compress::No);
    let recursive_steps_total: usize = proof
        .steps
        .iter()
        .map(|step| step.serialized_size(Compress::No))
        .sum();
    let tail_total = if proof.is_root_direct() {
        0
    } else {
        proof.final_witness().serialized_size(Compress::No)
    };
    // The terminal step's serialized size includes `final_witness`, which is
    // already accounted for in `tail_total`. Subtract it so the Akita-fold
    // line item only counts the per-level non-witness bytes.
    let akita_levels_total = root_total + recursive_steps_total - tail_total;
    let accounted_total = akita_levels_total + tail_total;
    let framing_total = proof
        .size()
        .checked_sub(accounted_total)
        .unwrap_or_else(|| {
            panic!(
                "[{label}] proof accounting exceeded total: accounted={accounted_total}, total={}",
                proof.size()
            )
        });
    // Total fold levels = 1 root + every entry in `proof.steps` (which
    // already includes the terminal step in the multi-fold case).
    // `num_fold_levels()` counts intermediate-only steps and would
    // undercount the terminal step.
    let fold_levels = if proof.is_root_direct() {
        0
    } else {
        1 + proof.steps.len()
    };

    tracing::info!(
        label,
        levels = fold_levels,
        proof_size_bytes = proof.size(),
        accounted_bytes = accounted_total,
        akita_fold_bytes = akita_levels_total,
        tail_bytes = tail_total,
        proof_framing_bytes = framing_total,
        "proof summary"
    );
    eprintln!(
        "[{label}] proof: total={} bytes, akita_fold={} bytes, tail={} bytes, framing={} bytes, levels={}",
        proof.size(),
        akita_levels_total,
        tail_total,
        framing_total,
        fold_levels,
    );
    assert_eq!(
        accounted_total,
        proof.size(),
        "[{label}] proof accounting must exactly match serialized proof size"
    );
    let mut obs_idx = 0usize;
    print_batched_root_breakdown::<FF, L, D>(label, &proof.root, grind_observations, &mut obs_idx);
    for (i, step) in proof.steps.iter().enumerate() {
        let level_idx = i + 1;
        match step {
            AkitaLevelProof::Intermediate { .. } => {
                print_akita_level_breakdown::<FF, L, D>(
                    label,
                    level_idx,
                    step,
                    grind_observations,
                    &mut obs_idx,
                );
            }
            AkitaLevelProof::Terminal { .. } => {
                print_terminal_level_breakdown::<FF, L, _, D>(
                    label,
                    level_idx,
                    step,
                    "fold",
                    grind_observations,
                    &mut obs_idx,
                );
            }
        }
    }
    if !proof.is_root_direct() {
        assert_eq!(
            obs_idx,
            grind_observations.len(),
            "[{label}] fold grind observation count must match folded proof levels"
        );
        let proof_nonces = collect_fold_grind_nonces(proof);
        assert_eq!(
            proof_nonces,
            grind_observations
                .iter()
                .map(|obs| obs.grind_nonce)
                .collect::<Vec<_>>(),
            "[{label}] fold grind observation nonces must match proof wire nonces"
        );
    }
    emit_fold_grind_summary(label, grind_observations);
}

pub(crate) fn print_layout(layout: &LevelParams, num_claims: usize, field_bits: u32) {
    tracing::debug!(
        m_vars = layout.m_vars,
        r_vars = layout.r_vars,
        num_blocks = layout.num_blocks,
        block_len = layout.block_len,
        delta_commit = layout.num_digits_commit,
        delta_open = layout.num_digits_open,
        delta_fold = layout.num_digits_fold(num_claims, field_bits).unwrap_or(0),
        log_basis = layout.log_basis,
        "layout"
    );
}
