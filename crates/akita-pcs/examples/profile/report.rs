use akita_field::FieldCore;
use akita_serialization::{AkitaSerialize, Compress};
use akita_types::{
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaLevelProof, AkitaProofStep, AkitaSchedulePlan,
    DirectWitnessProof, LevelParams, Schedule, Step, TerminalLevelProof,
};

pub(crate) fn report_timing(label: &str, phase: &str, elapsed_s: f64) {
    tracing::info!(label, elapsed_s, "{phase}");
    eprintln!("[{label}] {phase}: {elapsed_s:.6}s");
}

pub(crate) fn emit_planned_schedule_summary(
    label: &str,
    plan: &AkitaSchedulePlan,
    root_num_claims: usize,
    field_bits: u32,
) {
    tracing::info!(
        label,
        levels = plan.num_fold_levels(),
        exact_proof_bytes = plan.exact_proof_bytes,
        no_wrapper_bytes = plan.no_wrapper_bytes,
        "planned schedule"
    );

    for level in plan.fold_levels() {
        let next_w_len = level.next_inputs.current_w_len;
        let num_claims = if level.inputs.level == 0 {
            root_num_claims
        } else {
            1
        };
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
            delta_fold = level.lp.num_digits_fold(num_claims, field_bits),
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
            delta_fold = lp.num_digits_fold(num_claims, field_bits),
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
        #[cfg(not(feature = "zk"))]
        let sumcheck = reduction.sumcheck.serialized_size(Compress::No);
        #[cfg(feature = "zk")]
        let sumcheck = reduction
            .sumcheck_proof_masked
            .serialized_size(Compress::No);
        (partials, sumcheck)
    })
}

fn print_akita_level_breakdown<FF, L, const D: usize>(
    label: &str,
    level_idx: usize,
    level: &AkitaLevelProof<FF, L>,
) -> usize
where
    FF: FieldCore + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    let y_rings_size = level.y_ring.serialized_size(Compress::No);
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(level.extension_opening_reduction.as_ref());
    let v_size = level.v.serialized_size(Compress::No);
    let total = level.serialized_size(Compress::No);

    eprintln!("[{label}]   akita_fold L{level_idx}: total={total} bytes");
    eprintln!(
        "[{label}]     y_rings={} bytes ({} ring elems, D={})",
        y_rings_size,
        ring_elem_count(level.y_ring.coeff_len(), D),
        D,
    );
    eprintln!(
        "[{label}]     v={} bytes ({} ring elems, D={})",
        v_size,
        ring_elem_count(level.v.coeff_len(), D),
        D,
    );
    let stage1 = &level.stage1;
    let stage2 = &level.stage2;
    let stage1_sumcheck_size = stage1
        .stages
        .iter()
        .map(|stage| {
            #[cfg(not(feature = "zk"))]
            {
                stage.sumcheck_proof.serialized_size(Compress::No)
            }
            #[cfg(feature = "zk")]
            {
                stage.sumcheck_proof_masked.serialized_size(Compress::No)
            }
        })
        .sum::<usize>();
    let stage1_interstage_claims_size = stage1
        .stages
        .iter()
        .flat_map(|stage| stage.child_claims.iter())
        .map(|claim| claim.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_s_claim_size = stage1.s_claim.serialized_size(Compress::No);
    #[cfg(not(feature = "zk"))]
    let stage2_sumcheck_size = stage2.sumcheck_proof.serialized_size(Compress::No);
    #[cfg(feature = "zk")]
    let stage2_sumcheck_size = stage2.sumcheck_proof_masked.serialized_size(Compress::No);
    let next_w_commitment_size = stage2.next_w_commitment.serialized_size(Compress::No);
    let next_w_eval_size = stage2.next_w_eval().serialized_size(Compress::No);
    tracing::info!(
        label,
        level = level_idx,
        d = D,
        total_bytes = total,
        y_ring_bytes = y_rings_size,
        extension_opening_partials_bytes = extension_opening_partials_size,
        extension_opening_sumcheck_bytes = extension_opening_sumcheck_size,
        v_bytes = v_size,
        stage1_sumcheck_bytes = stage1_sumcheck_size,
        stage1_interstage_claims_bytes = stage1_interstage_claims_size,
        stage1_s_claim_bytes = stage1_s_claim_size,
        stage2_sumcheck_bytes = stage2_sumcheck_size,
        next_w_commitment_bytes = next_w_commitment_size,
        next_w_eval_bytes = next_w_eval_size,
        "proof fold level"
    );
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_sumcheck={stage1_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_interstage_claims={stage1_interstage_claims_size} bytes");
    eprintln!("[{label}]     stage1_s_claim={stage1_s_claim_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     next_w_commitment={next_w_commitment_size} bytes ({} coeffs)",
        stage2.next_w_commitment.coeff_len(),
    );
    eprintln!("[{label}]     next_w_eval={next_w_eval_size} bytes");
    assert_eq!(
        total,
        y_rings_size
            + extension_opening_partials_size
            + extension_opening_sumcheck_size
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

fn print_terminal_level_breakdown<FF, L, const D: usize>(
    label: &str,
    level_idx: usize,
    level: &TerminalLevelProof<FF, L>,
    root_variant: &'static str,
) -> usize
where
    FF: FieldCore + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    let y_rings_size = level.y_rings.serialized_size(Compress::No);
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(level.extension_opening_reduction.as_ref());
    let stage2_sumcheck_size = {
        #[cfg(not(feature = "zk"))]
        {
            level.stage2_sumcheck.serialized_size(Compress::No)
        }
        #[cfg(feature = "zk")]
        {
            level
                .stage2_sumcheck_proof_masked
                .serialized_size(Compress::No)
        }
    };
    let final_witness_size = level.final_witness.serialized_size(Compress::No);
    let full = level.serialized_size(Compress::No);
    // `total_bytes` excludes `final_witness` to mirror the planner's
    // `terminal_level_proof_bytes`. `final_witness` is reported separately as
    // the proof tail (`tail_bytes`) and accounted for in `accounted_bytes`.
    let total = full - final_witness_size;

    // Only the fields structurally present in `TerminalLevelProof` are
    // emitted: `y_rings`, optional extension-opening reduction, the
    // stage-2 sumcheck, and `final_witness`. The intermediate-level
    // fields (`v`, `stage1_*`, `next_w_*`) are absent at terminal and
    // therefore omitted from the tracing payload; downstream parsers
    // default missing keys to zero.
    tracing::info!(
        label,
        level = level_idx,
        d = D,
        total_bytes = total,
        y_ring_bytes = y_rings_size,
        extension_opening_partials_bytes = extension_opening_partials_size,
        extension_opening_sumcheck_bytes = extension_opening_sumcheck_size,
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
    eprintln!(
        "[{label}]     y_rings={} bytes ({} ring elems, D={})",
        y_rings_size,
        ring_elem_count(level.y_rings.coeff_len(), D),
        D,
    );
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!("[{label}]     final_witness={final_witness_size} bytes (absorbed via transcript)");
    assert_eq!(
        full,
        y_rings_size
            + extension_opening_partials_size
            + extension_opening_sumcheck_size
            + stage2_sumcheck_size
            + final_witness_size
    );
    total
}

fn print_batched_root_breakdown<FF, L, const D: usize>(
    label: &str,
    root: &AkitaBatchedRootProof<FF, L>,
) -> usize
where
    FF: FieldCore + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    if let Some(terminal) = root.as_terminal_root() {
        return print_terminal_level_breakdown::<FF, L, D>(label, 0, terminal, "terminal");
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
    let y_rings_size = fold.y_rings.serialized_size(Compress::No);
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(fold.extension_opening_reduction.as_ref());
    let v_size = fold.v.serialized_size(Compress::No);
    let total = fold.serialized_size(Compress::No);
    let stage1 = &fold.stage1;
    let stage2 = &fold.stage2;
    let stage1_sumcheck_size = stage1
        .stages
        .iter()
        .map(|stage| {
            #[cfg(not(feature = "zk"))]
            {
                stage.sumcheck_proof.serialized_size(Compress::No)
            }
            #[cfg(feature = "zk")]
            {
                stage.sumcheck_proof_masked.serialized_size(Compress::No)
            }
        })
        .sum::<usize>();
    let stage1_interstage_claims_size = stage1
        .stages
        .iter()
        .flat_map(|stage| stage.child_claims.iter())
        .map(|claim| claim.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_s_claim_size = stage1.s_claim.serialized_size(Compress::No);
    #[cfg(not(feature = "zk"))]
    let stage2_sumcheck_size = stage2.sumcheck_proof.serialized_size(Compress::No);
    #[cfg(feature = "zk")]
    let stage2_sumcheck_size = stage2.sumcheck_proof_masked.serialized_size(Compress::No);
    let next_w_commitment_size = stage2.next_w_commitment.serialized_size(Compress::No);
    let next_w_eval_size = stage2.next_w_eval().serialized_size(Compress::No);

    tracing::info!(
        label,
        level = 0usize,
        d = D,
        total_bytes = total,
        y_ring_bytes = y_rings_size,
        extension_opening_partials_bytes = extension_opening_partials_size,
        extension_opening_sumcheck_bytes = extension_opening_sumcheck_size,
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
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_sumcheck={stage1_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_interstage_claims={stage1_interstage_claims_size} bytes");
    eprintln!("[{label}]     stage1_s_claim={stage1_s_claim_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     next_w_commitment={next_w_commitment_size} bytes ({} coeffs)",
        stage2.next_w_commitment.coeff_len(),
    );
    eprintln!("[{label}]     next_w_eval={next_w_eval_size} bytes");
    assert_eq!(
        total,
        y_rings_size
            + extension_opening_partials_size
            + extension_opening_sumcheck_size
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

pub(crate) fn print_batched_proof_summary<FF, L, const D: usize>(
    label: &str,
    proof: &AkitaBatchedProof<FF, L>,
) where
    FF: FieldCore + AkitaSerialize,
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
    print_batched_root_breakdown::<FF, L, D>(label, &proof.root);
    for (i, step) in proof.steps.iter().enumerate() {
        let level_idx = i + 1;
        match step {
            AkitaProofStep::Intermediate(lp) => {
                print_akita_level_breakdown::<FF, L, D>(label, level_idx, lp);
            }
            AkitaProofStep::Terminal(lp) => {
                print_terminal_level_breakdown::<FF, L, D>(label, level_idx, lp, "fold");
            }
        }
    }
    if !proof.is_root_direct() {
        emit_observed_tail_summary(label, proof.final_witness());
    }
}

fn emit_observed_tail_summary<FF: FieldCore + AkitaSerialize>(
    label: &str,
    final_w: &DirectWitnessProof<FF>,
) {
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

pub(crate) fn print_layout(layout: &LevelParams, num_claims: usize, field_bits: u32) {
    tracing::debug!(
        m_vars = layout.m_vars,
        r_vars = layout.r_vars,
        num_blocks = layout.num_blocks,
        block_len = layout.block_len,
        delta_commit = layout.num_digits_commit,
        delta_open = layout.num_digits_open,
        delta_fold = layout.num_digits_fold(num_claims, field_bits),
        log_basis = layout.log_basis,
        "layout"
    );
}
