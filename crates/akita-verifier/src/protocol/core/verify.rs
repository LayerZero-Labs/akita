use super::suffix::{verify_suffix, SuffixVerifierState};
use super::*;
// Top-level batched verifier orchestration once a schedule is selected.

use crate::proof::direct::verify_zero_fold_openings_with_opening_batch;
use crate::protocol::validate_log_basis;
use akita_algebra::CyclotomicRing;
use akita_config::{
    bind_transcript_instance_descriptor, effective_batched_schedule, CommitmentConfig,
};
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::dispatch_ring_dim_result;
use akita_types::{
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaLevelProof, AkitaSetupSeed, AkitaVerifierSetup,
    BasisMode, CleartextWitnessProof, Commitment, FpExtEncoding, LevelParams, OpeningClaims,
    OpeningClaimsLayout, RingVec, RingView, Schedule, SetupContributionMode, Step,
    ValidatedScheduleContext, GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED,
    GROUPED_ROOT_TIERED_UNSUPPORTED,
};
use std::array::from_fn;

pub(crate) fn field_evals_to_rings<F, const D: usize>(
    evals: &[F],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    if D == 0 || !D.is_power_of_two() || !evals.len().is_power_of_two() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(evals
        .chunks(D)
        .map(|chunk| {
            CyclotomicRing::from_coefficients(from_fn(|idx| {
                chunk.get(idx).copied().unwrap_or_else(F::zero)
            }))
        })
        .collect())
}

fn check_batched_proof_step_shape<F, E>(proof: &AkitaBatchedProof<F, E>) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    match &proof.root {
        AkitaBatchedRootProof::Fold(_) => {
            let Some((last, rest)) = proof.steps.split_last() else {
                return Err(AkitaError::InvalidProof);
            };
            if !matches!(last, AkitaLevelProof::Terminal { .. })
                || rest
                    .iter()
                    .any(|step| !matches!(step, AkitaLevelProof::Intermediate { .. }))
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        AkitaBatchedRootProof::Terminal(_) | AkitaBatchedRootProof::ZeroFold { .. } => {
            if !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
        }
    }
    Ok(())
}

fn reject_unsupported_grouped_root<Cfg>(
    opening_batch: &OpeningClaimsLayout,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
{
    if opening_batch.num_groups() <= 1 {
        return Ok(());
    }
    if Cfg::TIERED_COMMITMENT {
        return Err(AkitaError::InvalidSetup(
            GROUPED_ROOT_TIERED_UNSUPPORTED.to_string(),
        ));
    }
    if setup_contribution_mode == SetupContributionMode::Recursive {
        return Err(AkitaError::InvalidSetup(
            GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED.to_string(),
        ));
    }
    // Unsupported grouped claims: `InvalidProof` (unit variant). See `GROUPED_ROOT_UNSUPPORTED`.
    Err(AkitaError::InvalidProof)
}

struct DirectRecommitGeometry {
    log_basis: u32,
    num_blocks: usize,
    block_len: usize,
    num_digits_commit: usize,
    num_digits_open: usize,
    a_row_len: usize,
    a_col_len: usize,
    b_row_len: usize,
    max_setup_len: usize,
}

impl DirectRecommitGeometry {
    fn from_level(params: &LevelParams, setup_seed: &AkitaSetupSeed) -> Self {
        Self {
            log_basis: params.log_basis,
            num_blocks: params.num_blocks,
            block_len: params.block_len,
            num_digits_commit: params.num_digits_commit,
            num_digits_open: params.num_digits_open,
            a_row_len: params.a_key.row_len(),
            a_col_len: params.a_key.col_len(),
            b_row_len: params.b_key.row_len(),
            max_setup_len: setup_seed.max_setup_len,
        }
    }
}

fn validate_root_direct_recommitment_shape<F>(
    witnesses: &[CleartextWitnessProof<F>],
    geom: &DirectRecommitGeometry,
    opening_batch: &OpeningClaimsLayout,
    ring_d: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_log_basis(geom.log_basis)?;
    if geom.num_blocks == 0 || geom.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "direct witness layout requires non-zero block geometry".to_string(),
        ));
    }
    if geom.num_digits_commit == 0 || geom.num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "direct witness layout requires non-zero digit depths".to_string(),
        ));
    }
    let expected_witness_len = 1usize
        .checked_shl(
            u32::try_from(opening_batch.max_num_vars()).map_err(|_| AkitaError::InvalidProof)?,
        )
        .ok_or(AkitaError::InvalidProof)?;
    let direct_capacity = geom
        .num_blocks
        .checked_mul(geom.block_len)
        .ok_or_else(|| AkitaError::InvalidSetup("direct witness capacity overflow".to_string()))?;
    if expected_witness_len.div_ceil(ring_d) > direct_capacity {
        return Err(AkitaError::InvalidSetup(
            "direct witness exceeds selected verifier layout".to_string(),
        ));
    }
    if opening_batch.num_total_polynomials() != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }

    let a_required_cols = geom
        .block_len
        .checked_mul(geom.num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("direct A width overflow".to_string()))?;
    let a_required = geom
        .a_row_len
        .checked_mul(a_required_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("direct A footprint overflow".to_string()))?;
    let per_witness_outer_cols = geom
        .num_blocks
        .checked_mul(geom.a_row_len)
        .and_then(|cols| cols.checked_mul(geom.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("direct B width overflow".to_string()))?;
    let b_required_cols = witnesses
        .len()
        .checked_mul(per_witness_outer_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("direct B width overflow".to_string()))?;
    let b_required = geom
        .b_row_len
        .checked_mul(b_required_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("direct B footprint overflow".to_string()))?;
    if a_required.max(b_required) > geom.max_setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for direct witness layout".to_string(),
        ));
    }
    for witness in witnesses {
        let witness_len = witness
            .as_field_elements()
            .ok_or(AkitaError::InvalidProof)?
            .coeff_len();
        if witness_len != expected_witness_len {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}

pub(crate) fn mat_vec_mul_i8_plain<F, const D: usize>(
    matrix_rows: &[&[CyclotomicRing<F, D>]],
    digits: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
{
    matrix_rows
        .iter()
        .map(|row| {
            row.iter().zip(digits.iter()).fold(
                CyclotomicRing::<F, D>::zero(),
                |acc, (entry, digit)| {
                    let digit_ring = CyclotomicRing::from_coefficients(from_fn(|idx| {
                        F::from_i64(digit[idx] as i64)
                    }));
                    acc + (*entry * digit_ring)
                },
            )
        })
        .collect()
}

fn decompose_rows_i8<F, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]>
where
    F: FieldCore + CanonicalField,
{
    let mut out = vec![[0i8; D]; rows.len() * num_digits];
    for (dst_chunk, row) in out.chunks_mut(num_digits).zip(rows.iter()) {
        row.balanced_decompose_pow2_i8_into(dst_chunk, log_basis);
    }
    out
}

fn direct_decomposed_inner_rows<F, const D: usize>(
    witness_rings: &[CyclotomicRing<F, D>],
    setup: &AkitaVerifierSetup<F>,
    geom: &DirectRecommitGeometry,
) -> Result<Vec<[i8; D]>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let a_matrix = setup
        .expanded
        .shared_matrix()
        .ring_view::<D>(geom.a_row_len, geom.a_col_len)?;
    let a_rows: Vec<_> = a_matrix.rows().collect();
    let out_capacity = geom
        .num_blocks
        .checked_mul(geom.a_row_len)
        .and_then(|len| len.checked_mul(geom.num_digits_open))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("direct witness row capacity overflow".to_string())
        })?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..geom.num_blocks {
        let start = block_idx.checked_mul(geom.block_len).ok_or_else(|| {
            AkitaError::InvalidSetup("direct witness block offset overflow".to_string())
        })?;
        let block = if start < witness_rings.len() {
            let end = start
                .checked_add(geom.block_len)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("direct witness block end overflow".to_string())
                })?
                .min(witness_rings.len());
            &witness_rings[start..end]
        } else {
            &[]
        };
        let block_digits = decompose_rows_i8(block, geom.num_digits_commit, geom.log_basis);
        let t_rows = mat_vec_mul_i8_plain::<F, D>(&a_rows, &block_digits);
        out.extend(decompose_rows_i8(
            &t_rows,
            geom.num_digits_open,
            geom.log_basis,
        ));
    }

    Ok(out)
}

fn recommit_direct_witness_group<F, const D: usize>(
    group_witnesses: &[CleartextWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    geom: &DirectRecommitGeometry,
    tiered: bool,
) -> Result<RingCommitment<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if tiered {
        return Err(AkitaError::InvalidSetup(
            "root-direct recommitment does not support tiered commitment \
             (f_key must be absent on the root-direct path)"
                .to_string(),
        ));
    }

    let mut outer_input = Vec::new();
    for witness in group_witnesses {
        let field_witness = witness
            .as_field_elements()
            .ok_or(AkitaError::InvalidProof)?
            .coeffs();
        let witness_rings = field_evals_to_rings::<F, D>(field_witness)?;
        outer_input.extend(direct_decomposed_inner_rows(&witness_rings, setup, geom)?);
    }

    let b_matrix = setup
        .expanded
        .shared_matrix()
        .ring_view::<D>(geom.b_row_len, outer_input.len())?;
    let b_rows: Vec<_> = b_matrix.rows().collect();
    let u = mat_vec_mul_i8_plain::<F, D>(&b_rows, &outer_input);
    Ok(RingCommitment { u })
}

/// Recompute root-direct commitments from direct witnesses and compare them to
/// the proof commitments.
///
/// # Errors
///
/// Returns an error if the direct witness shape does not match the batch shape,
/// if witness reconstruction fails, or if any recomputed commitment differs
/// from the proof commitment.
pub(crate) fn verify_root_direct_commitments_with_params<F, E>(
    witnesses: &[CleartextWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    E: FieldCore,
{
    let commitment = claims
        .single_group_commitment()
        .copied()
        .ok_or(AkitaError::InvalidProof)?;
    let opening_batch = claims.layout();
    // Validate the flat commitment shape against the schedule-derived ring
    // dimension before interpreting it. `RingView::new` enforces the
    // multiple-of-`ring_dim` invariant; no panic on malformed lengths.
    let ring_dim = params.role_dims().d_b();
    let commitment_view = RingView::new(commitment.rows().coeffs(), ring_dim)?;
    let geom = DirectRecommitGeometry::from_level(params, setup.expanded.seed());
    let recomputed_matches = dispatch_ring_dim_result!(ring_dim, |D| {
        validate_root_direct_recommitment_shape::<F>(witnesses, &geom, &opening_batch, ring_dim)?;
        let recomputed =
            recommit_direct_witness_group::<F, D>(witnesses, setup, &geom, params.f_key.is_some())?;
        // Compare recomputed `u` to the proof commitment as flat coefficients
        // under the same ring dimension (byte/coefficient parity with absorb).
        let recomputed_vec = RingVec::from_ring_elems(&recomputed.u);
        Ok(recomputed_vec.coeffs() == commitment_view.coeffs())
    })?;
    if !recomputed_matches {
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}

fn validate_schedule_onehot_chunk_size<Cfg: CommitmentConfig>(
    schedule: &Schedule,
) -> Result<(), AkitaError> {
    let expected = Cfg::onehot_chunk_size();
    if Cfg::decomposition().log_commit_bound != 1 || expected <= 1 {
        return Ok(());
    }
    let root_params = match schedule.steps.first() {
        Some(akita_types::Step::Fold(root)) => Some(&root.params),
        Some(akita_types::Step::Direct(root)) => root.params.as_ref(),
        None => None,
    }
    .ok_or(AkitaError::InvalidProof)?;
    if root_params.onehot_chunk_size != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// Verify a batched proof under config `Cfg`.
///
/// This is the verifier crate's top-level orchestration entrypoint. It owns
/// public claim normalization, schedule selection (from `Cfg`), the root-direct
/// rewrite, and transcript instance-descriptor binding before handing off to
/// `verify` for root-direct and folded-root replay.
///
/// The root-direct branch recomputes commitments with the same root commitment
/// layout the prover used at commit time (`Cfg::get_params_for_batched_commitment`
/// for the same opening_batch); a mismatching layout would cause root-direct
/// commitment recomputation to reject a correctly produced proof.
///
/// # Errors
///
/// Returns an error if public claims are malformed, schedule/layout policy
/// rejects the proof shape, root-direct commitment recomputation rejects, or
/// proof replay fails.
pub fn batched_verify<Cfg, T>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    transcript: &mut T,
    claims: OpeningClaims<'_, Cfg::ExtField, &Commitment<Cfg::Field>>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field>,
{
    // Reject malformed step shapes that the downstream `fold_levels()` filter
    // would silently skip past.
    check_batched_proof_step_shape(proof)?;

    claims
        .validate(setup.expanded.seed())
        .map_err(|_| AkitaError::InvalidProof)?;
    let opening_batch = claims.layout();
    reject_unsupported_grouped_root::<Cfg>(&opening_batch, setup_contribution_mode)?;
    let schedule = effective_batched_schedule::<Cfg>(&opening_batch, claims.point())
        .map_err(|_| AkitaError::InvalidProof)?;
    let schedule_ctx =
        ValidatedScheduleContext::new(&schedule, setup.expanded.seed().gen_ring_dim)?;
    validate_schedule_onehot_chunk_size::<Cfg>(schedule_ctx.schedule())?;

    // The transcript instance descriptor binds the setup-wide root ring
    // dimension (`gen_ring_dim`), which is byte-identical to the const `Cfg::D`
    // the prover binds for uniform-D presets. Dispatch on the runtime value so
    // the verifier entry stays D-free; the descriptor bytes are unchanged.
    dispatch_ring_dim_result!(setup.expanded.seed().gen_ring_dim, |D| {
        bind_transcript_instance_descriptor::<Cfg::Field, T, D, Cfg>(
            &setup.expanded,
            &opening_batch,
            schedule_ctx.schedule(),
            basis,
            transcript,
        )
    })?;

    verify::<Cfg, T>(
        proof,
        setup,
        transcript,
        claims,
        &schedule_ctx,
        basis,
        setup_contribution_mode,
    )
}

/// Verify a prepared batched proof once the schedule and transcript descriptor
/// are fixed.
///
/// This mirrors the prover's `prove` orchestration: the batched wrapper owns
/// public input preparation, while this function owns the root-direct vs
/// folded-root proof dispatch.
///
/// # Errors
///
/// Returns an error if the proof root shape does not match the selected
/// schedule, root-direct commitment recomputation rejects, or folded proof
/// replay rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn verify<Cfg, T>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    transcript: &mut T,
    claims: OpeningClaims<'_, Cfg::ExtField, &Commitment<Cfg::Field>>,
    schedule_ctx: &ValidatedScheduleContext,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field>,
{
    let schedule = schedule_ctx.schedule();
    match &proof.root {
        AkitaBatchedRootProof::ZeroFold { witnesses, .. } => {
            let Some(Step::Direct(direct)) = schedule.steps.first() else {
                return Err(AkitaError::InvalidProof);
            };
            let params = direct.params.as_ref().ok_or(AkitaError::InvalidProof)?;
            verify_zero_fold_openings_with_opening_batch::<Cfg::Field, Cfg::ExtField, _>(
                witnesses, &claims, basis,
            )?;
            verify_root_direct_commitments_with_params::<Cfg::Field, Cfg::ExtField>(
                witnesses, setup, &claims, params,
            )?;
        }
        AkitaBatchedRootProof::Fold(_) | AkitaBatchedRootProof::Terminal(_) => {
            verify_folded_batched_proof::<Cfg::Field, Cfg::ExtField, T>(
                proof,
                setup,
                transcript,
                claims,
                basis,
                schedule,
                setup_contribution_mode,
            )?;
        }
    }

    Ok(())
}

/// Verify the folded-root branch of a batched opening proof.
///
/// The caller owns config-backed schedule selection and passes the derived
/// root verifier layout plus the first suffix-level params. This function
/// owns the fold-root proof-shape checks, root opening preparation, root
/// transcript replay, and suffix handoff.
///
/// # Errors
///
/// Returns an error if the proof is not a folded-root proof, the schedule does
/// not match the proof shape, the root proof rejects, or a suffix level rejects.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_folded_batched_proof<F, E, T>(
    proof: &AkitaBatchedProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: OpeningClaims<'_, E, &Commitment<F>>,
    basis: BasisMode,
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidProof);
    };
    let root_lp = &root_step.params;
    let total_fold_levels = schedule_num_fold_levels(schedule);
    let terminal_direct = schedule
        .steps
        .last()
        .and_then(|step| match step {
            Step::Direct(direct) => Some(direct),
            Step::Fold(_) => None,
        })
        .ok_or(AkitaError::InvalidProof)?;

    match &proof.root {
        AkitaBatchedRootProof::ZeroFold { .. } => Err(AkitaError::InvalidProof),
        AkitaBatchedRootProof::Terminal(terminal) => {
            // 1-fold case: the root itself is the terminal fold. No suffix follows.
            if total_fold_levels != 1 {
                return Err(AkitaError::InvalidProof);
            }
            let final_witness = terminal
                .stage2
                .final_witness()
                .ok_or(AkitaError::InvalidProof)?;
            if !terminal_direct
                .witness_shape
                .admits_realized(&final_witness.shape())
            {
                return Err(AkitaError::InvalidProof);
            }
            verify_root::<F, E, T>(
                &proof.root,
                setup,
                transcript,
                &claims,
                basis,
                root_lp,
                setup_contribution_mode,
                None,
                root_step.next_w_len,
            )?;
            Ok(())
        }
        AkitaBatchedRootProof::Fold(fold_root) => {
            let expected_recursive_levels = total_fold_levels
                .checked_sub(1)
                .ok_or(AkitaError::InvalidProof)?;
            if proof.steps.len() != expected_recursive_levels {
                return Err(AkitaError::InvalidProof);
            }

            let terminal_step = proof
                .steps
                .last()
                .and_then(|step| match step {
                    AkitaLevelProof::Terminal { .. } => Some(step),
                    AkitaLevelProof::Intermediate { .. } => None,
                })
                .ok_or(AkitaError::InvalidProof)?;
            if !terminal_direct.witness_shape.admits_realized(
                &terminal_step
                    .stage2()
                    .final_witness()
                    .ok_or(AkitaError::InvalidProof)?
                    .shape(),
            ) {
                return Err(AkitaError::InvalidProof);
            }

            let first_recursive_params =
                scheduled_next_level_params(schedule, 1).map_err(|_| AkitaError::InvalidProof)?;
            let root_stage2 = fold_root
                .stage2
                .as_intermediate()
                .ok_or(AkitaError::InvalidProof)?;
            let root_challenges = verify_root::<F, E, T>(
                &proof.root,
                setup,
                transcript,
                &claims,
                basis,
                root_lp,
                setup_contribution_mode,
                Some(&first_recursive_params),
                root_step.next_w_len,
            )?;

            let first_level_d = first_recursive_params.role_dims().d_b();
            if !root_stage2.next_w_commitment.can_decode_vec(first_level_d) {
                return Err(AkitaError::InvalidProof);
            }
            let root_next_opening = proof
                .root
                .fold_stage3_sumcheck_proof(setup_contribution_mode)?
                .map_or_else(|| root_stage2.next_w_eval(), |proof| proof.next_w_eval);

            let current_state = SuffixVerifierState {
                opening_point: root_challenges,
                opening: root_next_opening,
                commitment: &root_stage2.next_w_commitment,
                basis: BasisMode::Lagrange,
                w_len: root_step.next_w_len,
            };
            verify_suffix::<F, E, T>(
                &proof.steps,
                setup,
                transcript,
                schedule,
                current_state,
                setup_contribution_mode,
            )?;
            Ok(())
        }
    }?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp32;
    use akita_types::{AjtaiKeyParams, RingVec, SisModulusFamily, DEFAULT_SIS_SECURITY_BITS};

    type F = Fp32<251>;
    const D: usize = 32;

    fn stage1_config() -> SparseChallengeConfig {
        SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![1],
        }
    }

    fn opening_batch(num_vars: usize) -> OpeningClaimsLayout {
        OpeningClaimsLayout::new(num_vars, 1).expect("valid opening batch summary")
    }

    fn setup_seed(max_setup_len: usize) -> AkitaSetupSeed {
        AkitaSetupSeed {
            max_num_vars: 6,
            max_num_batched_polys: 1,
            gen_ring_dim: D,
            max_setup_len,
            public_matrix_seed: [0u8; 32],
        }
    }

    #[test]
    fn root_direct_recommitment_rejects_undersized_setup() {
        let params =
            LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config())
                .with_decomp(1, 0, 2, 1, 0)
                .expect("valid direct layout");
        let setup_seed = setup_seed(3);
        let witnesses = vec![CleartextWitnessProof::FieldElements(RingVec::from_coeffs(
            vec![F::zero(); 64],
        ))];
        let err = validate_root_direct_recommitment_shape::<F>(
            &witnesses,
            &DirectRecommitGeometry::from_level(&params, &setup_seed),
            &opening_batch(6),
            D,
        )
        .expect_err("A layout needs four setup entries but setup has three");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn root_direct_recommitment_rejects_wrong_witness_dimension() {
        let mut params =
            LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config())
                .with_decomp(1, 0, 2, 1, 0)
                .expect("valid direct layout");
        params.b_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q32,
            1,
            128,
            0,
            D,
        );
        let setup_seed = setup_seed(128);
        let witnesses = vec![CleartextWitnessProof::FieldElements(RingVec::from_coeffs(
            vec![F::zero(); 32],
        ))];
        let err = validate_root_direct_recommitment_shape::<F>(
            &witnesses,
            &DirectRecommitGeometry::from_level(&params, &setup_seed),
            &opening_batch(6),
            D,
        )
        .expect_err("num_vars=6 requires 64 direct witness elements");
        assert!(matches!(err, AkitaError::InvalidProof));
    }

    /// The D-free commitment read path validates the flat coefficient length
    /// against the schedule-derived ring dimension via `RingView::new` and
    /// returns an error (never panics) when the length is not a multiple of the
    /// ring dimension. This is the no-panic gate the verifier relies on before
    /// interpreting any ring-shaped commitment.
    #[test]
    fn flat_commitment_length_not_multiple_of_ring_dim_rejects() {
        // 33 coefficients is not a multiple of D = 32.
        let commitment = RingVec::from_coeffs(vec![F::zero(); D + 1]);
        let err = RingView::new(commitment.coeffs(), D)
            .expect_err("commitment length must be a multiple of the ring dimension");
        assert!(matches!(err, AkitaError::InvalidProof));

        // A well-formed buffer (2 * D) is accepted and yields the expected ring count.
        let well_formed = vec![F::zero(); 2 * D];
        let ok = RingView::new(&well_formed, D).expect("valid flat commitment");
        assert_eq!(ok.num_rings(), 2);
    }

    #[test]
    fn reject_unsupported_grouped_root_rejects_generic_multi_group() {
        use akita_config::proof_optimized::fp128;

        let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 2]).expect("grouped batch");
        let err = reject_unsupported_grouped_root::<fp128::D64OneHot>(
            &batch,
            SetupContributionMode::Direct,
        )
        .expect_err("multi-group verify must reject before schedule lookup");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
