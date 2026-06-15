use akita_field::FieldCore;
use akita_prover::PreparedCrtNttProfile;
use akita_serialization::{AkitaSerialize, Compress};
use akita_types::{
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaLevelProof, CleartextWitnessProof, LevelParams,
    Schedule, SetupSumcheckProof, Step, TerminalLevelProof,
};

pub(crate) fn report_timing(label: &str, phase: &str, elapsed_s: f64) {
    tracing::info!(label, elapsed_s, "{phase}");
    eprintln!("[{label}] {phase}: {elapsed_s:.6}s");
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
        #[cfg(not(feature = "zk"))]
        let sumcheck = reduction.sumcheck.serialized_size(Compress::No);
        #[cfg(feature = "zk")]
        let sumcheck = reduction
            .sumcheck_proof_masked
            .serialized_size(Compress::No);
        (partials, sumcheck)
    })
}

fn stage3_sumcheck_size<L: FieldCore + AkitaSerialize>(
    proof: Option<&SetupSumcheckProof<L>>,
) -> usize {
    proof.map_or(0, |proof| {
        proof.claim.serialized_size(Compress::No) + proof.sumcheck.serialized_size(Compress::No)
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
    FF: FieldCore + AkitaSerialize,
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

/// Prover-side attempts for one fold level under the current sequential
/// `0..MAX` search (`nonce + 1`).
fn fold_grind_attempts_from_nonce(nonce: u32) -> u32 {
    // Under `sequential_min`, probe count equals `nonce + 1`. Under
    // `transcript_shuffle` this is diagnostic only (true probe count needs
    // prover tracing).
    nonce.saturating_add(1)
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

fn emit_fold_grind_summary(label: &str, nonces: &[u32]) {
    if nonces.is_empty() {
        tracing::info!(label, grind_levels = 0u32, "fold grind summary");
        eprintln!("[{label}] fold grind: no tail-bound fold levels");
        return;
    }

    let max_nonce = *nonces.iter().max().expect("non-empty nonce list");
    let attempts_sum: u64 = nonces
        .iter()
        .map(|nonce| u64::from(fold_grind_attempts_from_nonce(*nonce)))
        .sum();
    let nonces_csv = nonces
        .iter()
        .map(|nonce| nonce.to_string())
        .collect::<Vec<_>>()
        .join(",");

    tracing::info!(
        label,
        grind_levels = nonces.len(),
        grind_nonce_max = max_nonce,
        grind_attempts_sum = attempts_sum,
        grind_nonces = nonces_csv.as_str(),
        "fold grind summary"
    );
    eprintln!(
        "[{label}] fold grind: levels={}, attempts_sum={}, max_nonce={}, nonces=[{nonces_csv}]",
        nonces.len(),
        attempts_sum,
        max_nonce,
    );
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
    let stage2_sumcheck_size = stage2_intermediate
        .sumcheck_proof
        .serialized_size(Compress::No);
    #[cfg(feature = "zk")]
    let stage2_sumcheck_size = stage2_intermediate
        .sumcheck_proof_masked
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
    let grind_attempts = fold_grind_attempts_from_nonce(grind_nonce);

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

impl<FF: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> TerminalProofView<FF, L>
    for TerminalLevelProof<FF, L>
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

impl<FF: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> TerminalProofView<FF, L>
    for AkitaLevelProof<FF, L>
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
) -> usize
where
    FF: FieldCore + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
    P: TerminalProofView<FF, L>,
{
    let (extension_opening_partials_size, extension_opening_sumcheck_size) =
        extension_opening_reduction_sizes(level.extension_opening_reduction());
    let stage2_sumcheck_size = {
        #[cfg(not(feature = "zk"))]
        {
            level.stage2().sumcheck().serialized_size(Compress::No)
        }
        #[cfg(feature = "zk")]
        {
            level
                .stage2()
                .sumcheck_masked()
                .serialized_size(Compress::No)
        }
    };
    let final_witness_size = level.final_witness().serialized_size(Compress::No);
    let fold_grind_nonce_size = fold_grind_nonce_wire_bytes();
    let grind_nonce = level.fold_grind_nonce_value();
    let grind_attempts = fold_grind_attempts_from_nonce(grind_nonce);
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
) -> usize
where
    FF: FieldCore + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    if let Some(terminal) = root.as_terminal_root() {
        return print_terminal_level_breakdown::<FF, L, _, D>(label, 0, terminal, "terminal");
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
    let stage2_sumcheck_size = stage2_intermediate
        .sumcheck_proof
        .serialized_size(Compress::No);
    #[cfg(feature = "zk")]
    let stage2_sumcheck_size = stage2_intermediate
        .sumcheck_proof_masked
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
    let grind_attempts = fold_grind_attempts_from_nonce(grind_nonce);

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
            AkitaLevelProof::Intermediate { .. } => {
                print_akita_level_breakdown::<FF, L, D>(label, level_idx, step);
            }
            AkitaLevelProof::Terminal { .. } => {
                print_terminal_level_breakdown::<FF, L, _, D>(label, level_idx, step, "fold");
            }
        }
    }
    if !proof.is_root_direct() {
        emit_observed_tail_summary(label, proof.final_witness());
    }
    emit_fold_grind_summary(label, &collect_fold_grind_nonces(proof));
}

fn emit_observed_tail_summary<FF: FieldCore + AkitaSerialize>(
    label: &str,
    final_w: &CleartextWitnessProof<FF>,
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
        delta_fold = layout.num_digits_fold(num_claims, field_bits).unwrap_or(0),
        log_basis = layout.log_basis,
        "layout"
    );
}
