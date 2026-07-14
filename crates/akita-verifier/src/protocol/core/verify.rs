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
#[cfg(test)]
use akita_types::OpeningClaimsLayout;
use akita_types::{
    dispatch_for_field, validate_schedule_ring_dims, AkitaBatchedProof, AkitaBatchedRootProof,
    AkitaLevelProof, AkitaSetupSeed, AkitaVerifierSetup, BasisMode, CleartextWitnessProof,
    Commitment, FpExtEncoding, LevelParams, LevelParamsLike, OpeningClaims, RingCommitment,
    RingVec, RingView, Schedule, SetupContributionMode, Step,
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
    // Borrow flat coeffs as fixed-width ring rows: tail slots in the final row
    // are zero-filled when `evals.len()` is not a multiple of `D`. Root-direct
    // verify validates witness length before calling here; padding keeps this
    // helper aligned with the runtime-ring view where `D` is a local packing
    // width, not a protocol-wide invariant on flat vector length.
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

fn validate_direct_group_shape<F>(
    witnesses: &[CleartextWitnessProof<F>],
    params: &impl LevelParamsLike,
    setup_seed: &AkitaSetupSeed,
    num_vars: usize,
    ring_d: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_log_basis(params.log_basis())?;
    if params.live_fold_count() == 0 || params.fold_position_count() == 0 {
        return Err(AkitaError::InvalidSetup(
            "direct witness layout requires non-zero block geometry".to_string(),
        ));
    }
    if params.num_digits_commit() == 0 || params.num_digits_open() == 0 {
        return Err(AkitaError::InvalidSetup(
            "direct witness layout requires non-zero digit depths".to_string(),
        ));
    }
    let expected_witness_len = 1usize
        .checked_shl(u32::try_from(num_vars).map_err(|_| AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    let direct_capacity = params
        .live_fold_count()
        .checked_mul(params.fold_position_count())
        .ok_or_else(|| AkitaError::InvalidSetup("direct witness capacity overflow".to_string()))?;
    if expected_witness_len.div_ceil(ring_d) > direct_capacity {
        return Err(AkitaError::InvalidSetup(
            "direct witness exceeds selected verifier layout".to_string(),
        ));
    }
    let a_row_len = params.a_rows_len();
    let b_row_len = params.b_rows_len();
    let a_required_cols = params
        .fold_position_count()
        .checked_mul(params.num_digits_commit())
        .ok_or_else(|| AkitaError::InvalidSetup("direct A width overflow".to_string()))?;
    let a_required = a_row_len
        .checked_mul(a_required_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("direct A footprint overflow".to_string()))?;
    let per_witness_outer_cols = params
        .live_fold_count()
        .checked_mul(a_row_len)
        .and_then(|cols| cols.checked_mul(params.num_digits_open()))
        .ok_or_else(|| AkitaError::InvalidSetup("direct B width overflow".to_string()))?;
    let b_required_cols = witnesses
        .len()
        .checked_mul(per_witness_outer_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("direct B width overflow".to_string()))?;
    let b_required = b_row_len
        .checked_mul(b_required_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("direct B footprint overflow".to_string()))?;
    if a_required.max(b_required) > setup_seed.max_setup_len {
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

#[cfg(test)]
fn validate_root_direct_recommitment_shape<F>(
    witnesses: &[CleartextWitnessProof<F>],
    params: &LevelParams,
    setup_seed: &AkitaSetupSeed,
    opening_batch: &OpeningClaimsLayout,
    ring_d: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if opening_batch.num_total_polynomials() != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }
    validate_direct_group_shape::<F>(
        witnesses,
        params,
        setup_seed,
        opening_batch.max_num_vars(),
        ring_d,
    )
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
    params: &impl LevelParamsLike,
) -> Result<Vec<[i8; D]>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let a_row_len = params.a_rows_len();
    let a_matrix = setup
        .expanded
        .shared_matrix()
        .ring_view::<D>(a_row_len, params.a_col_len())?;
    let a_rows: Vec<_> = a_matrix.rows().collect();
    let out_capacity = params
        .live_fold_count()
        .checked_mul(a_row_len)
        .and_then(|len| len.checked_mul(params.num_digits_open()))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("direct witness row capacity overflow".to_string())
        })?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..params.live_fold_count() {
        let start = block_idx
            .checked_mul(params.fold_position_count())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("direct witness block offset overflow".to_string())
            })?;
        let block = if start < witness_rings.len() {
            let end = start
                .checked_add(params.fold_position_count())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("direct witness block end overflow".to_string())
                })?
                .min(witness_rings.len());
            &witness_rings[start..end]
        } else {
            &[]
        };
        let block_digits = decompose_rows_i8(block, params.num_digits_commit(), params.log_basis());
        let t_rows = mat_vec_mul_i8_plain::<F, D>(&a_rows, &block_digits);
        out.extend(decompose_rows_i8(
            &t_rows,
            params.num_digits_open(),
            params.log_basis(),
        ));
    }

    Ok(out)
}

fn recommit_direct_witness_group<F, const D: usize>(
    group_witnesses: &[CleartextWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    params: &impl LevelParamsLike,
) -> Result<RingCommitment<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let mut outer_input = Vec::new();
    for witness in group_witnesses {
        let field_witness = witness
            .as_field_elements()
            .ok_or(AkitaError::InvalidProof)?
            .coeffs();
        let witness_rings = field_evals_to_rings::<F, D>(field_witness)?;
        outer_input.extend(direct_decomposed_inner_rows(&witness_rings, setup, params)?);
    }

    let b_matrix = setup
        .expanded
        .shared_matrix()
        .ring_view::<D>(params.b_rows_len(), outer_input.len())?;
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
    let opening_batch = claims.layout().map_err(|_| AkitaError::InvalidProof)?;
    params.validate_root_opening_batch(&opening_batch)?;
    let ring_dim = params.role_dims().d_b();

    if opening_batch.num_total_polynomials() != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }

    let final_group = opening_batch.root_final_group_index()?;
    let setup_seed = setup.expanded.seed();
    for group_index in 0..opening_batch.num_groups() {
        let range = opening_batch.root_group_claim_range(group_index)?;
        if range.end > witnesses.len() {
            return Err(AkitaError::InvalidProof);
        }

        let commitment = claims.group_commitment(group_index).copied()?;
        let expected_rows = params.root_group_commitment_rows(&opening_batch, group_index)?;
        let commitment_view = RingView::new(commitment.rows().coeffs(), ring_dim)?;
        if commitment_view.num_rings() != expected_rows {
            return Err(AkitaError::InvalidProof);
        }

        let group_layout = opening_batch.group_layout(group_index)?;
        let group_witnesses = &witnesses[range.clone()];
        let recomputed_matches = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Outer),
            F,
            ring_dim,
            |D| {
                let recomputed = if group_index == final_group {
                    validate_direct_group_shape::<F>(
                        group_witnesses,
                        params,
                        setup_seed,
                        group_layout.num_vars(),
                        ring_dim,
                    )?;
                    recommit_direct_witness_group::<F, D>(group_witnesses, setup, params)?
                } else {
                    let group_params = params
                        .precommitted_groups
                        .get(group_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    validate_direct_group_shape::<F>(
                        group_witnesses,
                        group_params,
                        setup_seed,
                        group_layout.num_vars(),
                        ring_dim,
                    )?;
                    recommit_direct_witness_group::<F, D>(group_witnesses, setup, group_params)?
                };
                let recomputed_vec = RingVec::from_ring_elems(&recomputed.u);
                Ok(recomputed_vec.coeffs() == commitment_view.coeffs())
            }
        )?;
        if !recomputed_matches {
            return Err(AkitaError::InvalidProof);
        }
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
    let opening_batch = claims.layout().map_err(|_| AkitaError::InvalidProof)?;
    let schedule = effective_batched_schedule::<Cfg>(&opening_batch, claims.point())
        .map_err(|_| AkitaError::InvalidProof)?;
    validate_schedule_ring_dims(&schedule, setup.expanded.seed())?;
    schedule
        .reject_multi_group_multi_chunk("batched verify")
        .map_err(|_| AkitaError::InvalidProof)?;
    validate_schedule_onehot_chunk_size::<Cfg>(&schedule)?;

    // The transcript instance descriptor binds the setup-wide root ring
    // dimension (`gen_ring_dim`), which is byte-identical to the const `Cfg::D`
    // the prover binds for uniform-D presets. Dispatch on the runtime value so
    // the verifier entry stays D-free; the descriptor bytes are unchanged.
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Envelope,
        Cfg::Field,
        setup.expanded.seed().gen_ring_dim,
        |D| {
            bind_transcript_instance_descriptor::<Cfg::Field, T, D, Cfg>(
                &setup.expanded,
                &opening_batch,
                &schedule,
                basis,
                transcript,
            )
        }
    )?;

    verify::<Cfg, T>(
        proof,
        setup,
        transcript,
        claims,
        &schedule,
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
    schedule: &Schedule,
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
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidProof);
    };
    let root_lp = &root_step.params;
    let total_fold_levels = schedule.num_fold_levels();
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

    fn fold_challenge_config() -> SparseChallengeConfig {
        SparseChallengeConfig::pm1_only(1)
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
        let params = LevelParams::params_only(
            SisModulusFamily::Q32,
            D,
            2,
            1,
            1,
            1,
            fold_challenge_config(),
        )
        .with_decomp(2, 2, 2, 1)
        .expect("valid direct layout");
        let setup_seed = setup_seed(3);
        let witnesses = vec![CleartextWitnessProof::FieldElements(RingVec::from_coeffs(
            vec![F::zero(); 64],
        ))];
        let err = validate_root_direct_recommitment_shape::<F>(
            &witnesses,
            &params,
            &setup_seed,
            &opening_batch(6),
            D,
        )
        .expect_err("A layout needs four setup entries but setup has three");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn root_direct_recommitment_rejects_wrong_witness_dimension() {
        let mut params = LevelParams::params_only(
            SisModulusFamily::Q32,
            D,
            2,
            1,
            1,
            1,
            fold_challenge_config(),
        )
        .with_decomp(2, 2, 2, 1)
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
            &params,
            &setup_seed,
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
}
