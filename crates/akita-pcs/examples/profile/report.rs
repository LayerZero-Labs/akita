use akita_field::{CanonicalField, FieldCore};
use akita_prover::PreparedCrtNttProfile;
use akita_serialization::{AkitaSerialize, Compress};
use akita_types::{
    golomb_rice::{
        analyze_z_fold_golomb_encoding, golomb_rice_low_bits_sweep_payload_bytes,
        golomb_rice_zigzag_width, rice_low_bits_for_cap,
    },
    layout::proof_size::field_bytes,
    sis::num_digits_for_bound,
    AkitaBatchedProof, CommittedGroupParams, FoldLevelProof, FoldSchedule, SetupSumcheckProof,
    TerminalLevelProof, ZFoldEncodingStats,
};

pub(crate) fn report_timing(label: &str, phase: &str, elapsed_s: f64) {
    tracing::info!(label, elapsed_s, "{phase}");
    eprintln!("[{label}] {phase}: {elapsed_s:.6}s");
}

/// Structured tail witness report for profile bench / CI (`scripts/profile_bench_report.py`).
pub(crate) fn emit_proof_tail_report<FF, E>(
    label: &str,
    proof: &AkitaBatchedProof<FF, E>,
    schedule: &FoldSchedule,
    field_bits: u32,
) where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    E: FieldCore,
{
    let final_w = proof.terminal_response();
    let tail_bytes = final_w.serialized_size(Compress::No);
    let num_elems = final_w.num_elems();

    {
        let segment = final_w;
        let field_sz = field_bytes(FF::modulus_bits());
        let ring_dim = segment.layout.ring_dimension;
        let z_golomb_bytes = segment.z_payloads.iter().map(Vec::len).sum::<usize>();
        let z_field_elems = segment.layout.z_coords();
        let z_ring_elems = z_field_elems / ring_dim.max(1);
        let e_field_elems = segment.e_fields.coeff_len();
        let t_field_elems = segment.t_fields.coeff_len();
        let e_ring_elems = e_field_elems / ring_dim.max(1);
        let t_ring_elems = t_field_elems / ring_dim.max(1);
        let e_bytes = e_field_elems.saturating_mul(field_sz);
        let t_bytes = t_field_elems.saturating_mul(field_sz);
        let z_wire_bytes = tail_bytes.saturating_sub(e_bytes.saturating_add(t_bytes));
        let z_prefix_bytes = z_wire_bytes.saturating_sub(z_golomb_bytes);
        let z_budget_bytes = schedule
            .terminal
            .params
            .response_shape
            .layout
            .z_payload_bytes();
        let z_slack_bytes = z_budget_bytes.saturating_sub(z_golomb_bytes);
        let z_stats = terminal_response_z_fold_stats(segment, schedule, field_bits).ok();
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
            final_w_encoding = "terminal_response",
            final_w_policy = "non_zk_default",
            tail_log_basis_inner = schedule.terminal.params.witness.log_basis_inner,
            tail_z_prefix_bytes = z_prefix_bytes,
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
            tail_e_bytes = e_bytes,
            tail_t_bytes = t_bytes,
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
            "[{label}]   final_w: encoding=terminal_response (non-zk default), total={tail_bytes} bytes, \
             logical_elems={num_elems}, inner_log_basis={}{}",
            schedule.terminal.params.witness.log_basis_inner,
            golomb_line,
        );
        eprintln!(
            "[{label}]     z: {z_wire_bytes} B (len_prefix={z_prefix_bytes} + golomb={z_golomb_bytes}), \
             field_coeffs={z_field_elems}, ring_elems={z_ring_elems}",
        );
        eprintln!(
            "[{label}]     e: {e_bytes} B, field_coeffs={e_field_elems}, ring_elems={e_ring_elems}",
        );
        eprintln!(
            "[{label}]     t: {t_bytes} B, field_coeffs={t_field_elems}, ring_elems={t_ring_elems}",
        );
        assert_eq!(tail_bytes, z_wire_bytes + e_bytes + t_bytes);
        if std::env::var("AKITA_Z_GOLOMB_SWEEP").ok().as_deref() == Some("1") {
            emit_z_golomb_k_sweep(label, segment, schedule, field_bits, z_golomb_bytes);
        }
    }
}

fn terminal_response_z_fold_stats<FF: FieldCore>(
    witness: &akita_types::TerminalResponse<FF>,
    schedule: &FoldSchedule,
    field_bits: u32,
) -> Result<ZFoldEncodingStats, akita_field::AkitaError> {
    let params = &schedule.terminal.params.witness;
    let sparse = &schedule.terminal.params.sparse_challenge_config;
    let group = witness
        .layout
        .groups
        .first()
        .ok_or(akita_field::AkitaError::InvalidProof)?;
    let admission_cap = params.response_linf_policy(sparse)?.admission_cap;
    let z_values = akita_types::decode_terminal_z_golomb_payload(
        witness
            .z_payloads
            .first()
            .ok_or(akita_field::AkitaError::InvalidProof)?,
        group.z_coords,
        admission_cap,
        Some(group.z_payload_bytes),
    )?;
    let log_cap = u128::BITS - admission_cap.leading_zeros();
    let hypothetical_digits =
        num_digits_for_bound(log_cap, field_bits, params.log_basis_inner).max(1);
    analyze_z_fold_golomb_encoding(
        &z_values,
        admission_cap,
        golomb_rice_zigzag_width(admission_cap),
        hypothetical_digits,
        params.log_basis_inner,
        witness.z_payloads.first().map_or(0, Vec::len),
    )
}

fn emit_z_golomb_k_sweep<FF: FieldCore>(
    label: &str,
    witness: &akita_types::TerminalResponse<FF>,
    schedule: &FoldSchedule,
    field_bits: u32,
    actual_z_payload_bytes: usize,
) {
    let params = &schedule.terminal.params.witness;
    let sparse = &schedule.terminal.params.sparse_challenge_config;
    let Some(group) = witness.layout.groups.first() else {
        return;
    };
    let Ok(response_policy) = params.response_linf_policy(sparse) else {
        return;
    };
    let Ok(z_values) = akita_types::decode_terminal_z_golomb_payload(
        witness
            .z_payloads
            .first()
            .map(Vec::as_slice)
            .unwrap_or_default(),
        group.z_coords,
        response_policy.admission_cap,
        Some(group.z_payload_bytes),
    ) else {
        return;
    };
    let Ok(stats) = terminal_response_z_fold_stats(witness, schedule, field_bits) else {
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

pub(crate) fn report_verifier_ntt_cache_size(label: &str, verifier_ntt_cache_bytes: usize) {
    tracing::info!(label, verifier_ntt_cache_bytes, "verifier NTT cache size");
    eprintln!("[{label}] verifier NTT cache: ntt_cache={verifier_ntt_cache_bytes} bytes");
}

pub(crate) fn report_crt_profile(label: &str, profile: PreparedCrtNttProfile) {
    tracing::info!(
        label,
        crt_profile = profile.profile_id,
        crt_num_primes = profile.num_primes,
        crt_prime_modulus_bits = profile.prime_modulus_bits,
        crt_limb_bits = profile.limb_bits,
        max_i8_log_basis = profile.max_i8_log_basis,
        balanced_digit_safe_width = profile.balanced_digit_safe_width,
        raw_i8_safe_width = profile.raw_i8_safe_width,
        "CRT NTT profile"
    );
    eprintln!(
        "[{label}] CRT NTT profile: profile={}, primes={}, prime_modulus_bits={}, signed_storage_bits={}, max_i8_log_basis={}, balanced_digit_safe_width={}, raw_i8_safe_width={}",
        profile.profile_id,
        profile.num_primes,
        profile.prime_modulus_bits,
        profile.limb_bits,
        profile.max_i8_log_basis,
        profile.balanced_digit_safe_width,
        profile.raw_i8_safe_width
    );
}

pub(crate) fn emit_runtime_schedule_summary(
    label: &str,
    schedule: &FoldSchedule,
    root_num_claims: usize,
    field_bits: u32,
) {
    let levels = schedule.num_fold_levels();
    let setup_envelope_ring_elements = akita_types::setup_matrix_envelope_for_schedule(schedule)
        .map(|envelope| envelope.max_setup_len)
        .unwrap_or(0);
    let setup_envelope_field_elements = setup_envelope_ring_elements
        .saturating_mul(schedule.root.params.final_group.commitment.d_a());
    let setup_envelope_bytes =
        setup_envelope_field_elements.saturating_mul(field_bits.div_ceil(8) as usize);
    let selected_offload_edges = schedule
        .recursive_folds
        .iter()
        .filter(|fold| fold.params.incoming_setup_prefix.is_some())
        .count();
    tracing::info!(
        label,
        levels,
        selected_offload_edges,
        setup_envelope_ring_elements,
        setup_envelope_field_elements,
        setup_envelope_bytes,
        "runtime schedule"
    );

    let root_current_w_groups = root_current_w_groups(schedule, root_num_claims);
    let nonterminal = std::iter::once((
        0usize,
        &schedule.root.params.final_group.commitment,
        schedule.root.input_witness_len,
        schedule.root.output_witness_len,
        root_current_w_groups,
    ))
    .chain(
        schedule
            .recursive_folds
            .iter()
            .enumerate()
            .map(|(index, level)| {
                (
                    index + 1,
                    &level.params.witness,
                    level.input_witness_len,
                    level.output_witness_len,
                    format!("folded={}", level.input_witness_len),
                )
            }),
    );
    for (level_idx, lp, input_witness_len, output_witness_len, current_w_groups) in nonterminal {
        let role_dims = lp.role_dims();
        let num_claims = if level_idx == 0 { root_num_claims } else { 1 };
        let current_w_len = current_w_groups;
        let next_w_len = output_witness_len;
        let setup_prefix = schedule
            .recursive_folds
            .get(level_idx)
            .and_then(|fold| fold.params.incoming_setup_prefix.as_ref());
        let setup_prefix_natural_field_elements =
            setup_prefix.map_or(0, |prefix| prefix.natural_len);
        let setup_prefix_padded_field_elements =
            setup_prefix.map_or(0, |prefix| prefix.n_prefix().unwrap_or(0));
        tracing::info!(
            label,
            level = level_idx,
            d = lp.d_a(),
            d_a = role_dims.d_a(),
            d_b = role_dims.d_b(),
            d_d = role_dims.d_d(),
            n_a = lp.inner_commit_matrix.output_rank(),
            n_b = lp.outer_commit_matrix.output_rank(),
            n_d = lp.open_commit_matrix.output_rank(),
            challenge_l1_mass = lp.challenge_l1_mass(),
            log_basis_inner = lp.log_basis_inner,
            log_basis_outer = lp.log_basis_outer,
            log_basis_open = lp.log_basis_open,
            position_index_bits = lp.position_index_bits(),
            block_index_bits = lp.block_index_bits(),
            num_live_ring_elements_per_claim = lp.num_live_ring_elements_per_claim,
            num_live_blocks = lp.num_live_blocks,
            block_index_domain_size = lp.block_index_domain_size().unwrap_or(0),
            num_positions_per_block = lp.num_positions_per_block,
            num_digits_inner = lp.num_digits_inner,
            num_digits_outer = lp.num_digits_outer,
            num_digits_open = lp.num_digits_open,
            delta_fold = lp.num_digits_fold(num_claims, field_bits).unwrap_or(0),
            input_witness_len,
            output_witness_len,
            current_w_len,
            next_w_len,
            setup_prefix_natural_field_elements,
            setup_prefix_padded_field_elements,
            "planned fold level"
        );
    }

    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        if let Some(prefix) = &fold.params.incoming_setup_prefix {
            tracing::info!(
                label,
                successor_level = index + 1,
                setup_prefix_natural_field_elements = prefix.natural_len,
                setup_prefix_padded_field_elements = prefix.n_prefix().unwrap_or(0),
                "planned recursive setup edge"
            );
        }
    }

    tracing::info!(
        label,
        terminal_response_len = schedule.terminal.input_witness_len,
        final_inner_log_basis = schedule.terminal.params.witness.log_basis_inner,
        final_inner_ring_dimension = schedule.terminal.params.witness.d_a(),
        "planned terminal state"
    );
}

fn group_field_elements(num_vars: usize, num_polynomials: usize) -> usize {
    1usize
        .checked_shl(num_vars as u32)
        .and_then(|len| len.checked_mul(num_polynomials))
        .unwrap_or(0)
}

fn root_current_w_groups(schedule: &FoldSchedule, root_num_claims: usize) -> String {
    let mut groups = schedule
        .root
        .params
        .precommitted_groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let layout = group.descriptor.group;
            format!(
                "pre{index}={}",
                group_field_elements(layout.num_vars(), layout.num_polynomials())
            )
        })
        .collect::<Vec<_>>();
    let precommitted_polys = schedule
        .root
        .params
        .precommitted_groups
        .iter()
        .map(|group| group.descriptor.group.num_polynomials())
        .sum::<usize>();
    let final_polys = root_num_claims.saturating_sub(precommitted_polys);
    groups.push(format!(
        "final={}",
        schedule.root.input_witness_len.saturating_mul(final_polys)
    ));
    groups.join(";")
}

fn ring_elem_count(coeff_len: usize, d: usize) -> usize {
    coeff_len / d
}

fn extension_opening_reduction_sizes<E: FieldCore + AkitaSerialize>(
    reduction: Option<&akita_types::ExtensionOpeningReductionProof<E>>,
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

fn stage3_sumcheck_size<E: FieldCore + AkitaSerialize>(
    proof: Option<&SetupSumcheckProof<E>>,
) -> usize {
    proof.map_or(0, |proof| {
        proof.claim.serialized_size(Compress::No)
            + proof.setup_prefix_eval.serialized_size(Compress::No)
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
pub(crate) fn observed_stage3_setup_product_bytes<FF, E>(proof: &AkitaBatchedProof<FF, E>) -> usize
where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    E: FieldCore + AkitaSerialize,
{
    let root_bytes = stage3_sumcheck_size(proof.root.stage3_sumcheck_proof.as_ref());
    let step_bytes: usize = proof
        .recursive_folds
        .iter()
        .map(|step| stage3_sumcheck_size(step.stage3_sumcheck_proof.as_ref()))
        .sum();
    root_bytes + step_bytes
}

fn fold_grind_nonce_wire_bytes() -> usize {
    0u32.serialized_size(Compress::No)
}

fn print_akita_level_breakdown<FF, E, const D: usize>(
    label: &str,
    level_idx: usize,
    level: &FoldLevelProof<FF, E>,
) -> usize
where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    E: FieldCore + AkitaSerialize,
{
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(level.extension_opening_reduction.as_ref());
    let v_size = level.v.serialized_size(Compress::No);
    let total = level.serialized_size(Compress::No);
    let stage2_intermediate = &level.stage2;

    eprintln!("[{label}]   akita_fold L{level_idx}: total={total} bytes");
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        ring_elem_count(level.v.coeff_len(), D),
        D,
    );
    let stage1 = &level.stage1;
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
    let stage1_range_image_evaluation_size =
        stage1.range_image_evaluation.serialized_size(Compress::No);
    let stage2_sumcheck_size = stage2_intermediate
        .sumcheck_proof
        .serialized_size(Compress::No);
    let stage3_sumcheck_size = stage3_sumcheck_size(level.stage3_sumcheck_proof.as_ref());
    let next_w_commitment = stage2_intermediate.next_witness_binding.outer_commitment();
    let next_w_commitment_size = next_w_commitment
        .map(|commitment| commitment.serialized_size(Compress::No))
        .unwrap_or(0);
    let next_w_commitment_coeffs = next_w_commitment.map_or(0, akita_types::RingVec::coeff_len);
    let next_w_eval_size = stage2_intermediate
        .next_w_eval()
        .serialized_size(Compress::No);
    let fold_grind_nonce_size = fold_grind_nonce_wire_bytes();
    let grind_nonce = level.fold_grind_nonce;

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
        stage1_sumcheck_bytes = stage1_sumcheck_size,
        stage1_interstage_claims_bytes = stage1_interstage_claims_size,
        stage1_range_image_evaluation_bytes = stage1_range_image_evaluation_size,
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
    eprintln!(
        "[{label}]     stage1_range_image_evaluation={stage1_range_image_evaluation_size} bytes"
    );
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!("[{label}]     stage3_sumcheck={stage3_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     next_w_commitment={next_w_commitment_size} bytes ({} coeffs)",
        next_w_commitment_coeffs,
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
            + stage1_range_image_evaluation_size
            + stage2_sumcheck_size
            + stage3_sumcheck_size
            + next_w_commitment_size
            + next_w_eval_size
    );
    total
}

fn print_terminal_level_breakdown<FF, E, const D: usize>(
    label: &str,
    level_idx: usize,
    level: &TerminalLevelProof<FF, E>,
    root_variant: &'static str,
) -> usize
where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    E: FieldCore + AkitaSerialize,
{
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(level.extension_opening_reduction.as_ref());
    let terminal_response_size = level.terminal_response().serialized_size(Compress::No);
    let fold_grind_nonce_size = fold_grind_nonce_wire_bytes();
    let grind_nonce = level.fold_grind_nonce;
    let full = level.serialized_size(Compress::No);
    // `total_bytes` excludes the terminal response to mirror the planner's
    // `terminal_level_proof_bytes`. The response is reported separately as
    // the proof tail (`tail_bytes`) and accounted for in `accounted_bytes`.
    let total = full - terminal_response_size;

    // Only the fields structurally present in `TerminalLevelProof` are
    // emitted: optional extension-opening reduction and the terminal response.
    // The intermediate-level
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
        terminal_response_bytes = terminal_response_size,
        root_variant = root_variant,
        "proof fold level"
    );

    let header = if level_idx == 0 {
        "batched_root (terminal)".to_string()
    } else {
        format!("akita_fold L{level_idx} (terminal)")
    };
    eprintln!(
        "[{label}]   {header}: total={total} bytes (excl. terminal_response={terminal_response_size})"
    );
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!("[{label}]     fold_grind_nonce={fold_grind_nonce_size} bytes");
    eprintln!(
        "[{label}]     terminal_response={terminal_response_size} bytes (absorbed via transcript)"
    );
    assert_eq!(
        full,
        extension_opening_partials_size
            + extension_opening_sumcheck_size
            + fold_grind_nonce_size
            + terminal_response_size
    );
    total
}

pub(crate) fn print_batched_proof_summary<FF, E, const D: usize>(
    label: &str,
    proof: &AkitaBatchedProof<FF, E>,
) where
    FF: FieldCore + CanonicalField + AkitaSerialize,
    E: FieldCore + AkitaSerialize,
{
    let root_total = proof.root.serialized_size(Compress::No);
    let recursive_steps_total: usize = proof
        .recursive_folds
        .iter()
        .map(|step| step.serialized_size(Compress::No))
        .sum::<usize>()
        + proof.terminal.serialized_size(Compress::No);
    let tail_total = proof.terminal_response().serialized_size(Compress::No);
    // The terminal step's serialized size includes `terminal_response`, which is
    // already accounted for in `tail_total`. Subtract it so the Akita-fold
    // line item only counts the per-level non-witness bytes.
    let akita_levels_total = root_total + recursive_steps_total - tail_total;
    let accounted_total = akita_levels_total + tail_total;
    let fold_levels = proof.num_fold_levels();

    tracing::info!(
        label,
        levels = fold_levels,
        proof_size_bytes = proof.size(),
        accounted_bytes = accounted_total,
        akita_fold_bytes = akita_levels_total,
        tail_bytes = tail_total,
        "proof summary"
    );
    eprintln!(
        "[{label}] proof: total={} bytes, akita_fold={} bytes, tail={} bytes, levels={}",
        proof.size(),
        akita_levels_total,
        tail_total,
        fold_levels,
    );
    assert_eq!(
        accounted_total,
        proof.size(),
        "[{label}] proof accounting must exactly match serialized proof size"
    );
    print_akita_level_breakdown::<FF, E, D>(label, 0, &proof.root);
    for (i, step) in proof.recursive_folds.iter().enumerate() {
        print_akita_level_breakdown::<FF, E, D>(label, i + 1, step);
    }
    print_terminal_level_breakdown::<FF, E, D>(
        label,
        proof.num_fold_levels() - 1,
        &proof.terminal,
        "fold",
    );
}

pub(crate) fn print_layout(layout: &CommittedGroupParams, num_claims: usize, field_bits: u32) {
    tracing::debug!(
        position_index_bits = layout.position_index_bits(),
        block_index_bits = layout.block_index_bits(),
        num_live_ring_elements_per_claim = layout.num_live_ring_elements_per_claim,
        num_live_blocks = layout.num_live_blocks,
        block_index_domain_size = layout.block_index_domain_size().unwrap_or(0),
        num_positions_per_block = layout.num_positions_per_block,
        num_digits_inner = layout.num_digits_inner,
        num_digits_outer = layout.num_digits_outer,
        num_digits_open = layout.num_digits_open,
        delta_fold = layout.num_digits_fold(num_claims, field_bits).unwrap_or(0),
        log_basis_inner = layout.log_basis_inner,
        log_basis_outer = layout.log_basis_outer,
        log_basis_open = layout.log_basis_open,
        "layout"
    );
}
