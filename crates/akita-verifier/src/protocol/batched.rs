//! Top-level batched verifier orchestration once a schedule is selected.

use super::{validate_level_dispatch, validate_log_basis};
use crate::proof::claims::{prepare_verifier_claims, PreparedVerifierClaims};
use crate::proof::direct::verify_zero_fold_openings_with_incidence;
use crate::protocol::levels::verify_folded_batched_proof;
use akita_algebra::CyclotomicRing;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::{
    folded_root_supports_opening_shape, root_direct_schedule, root_tensor_projection_enabled,
    schedule_root_fold_step, AkitaBatchedProof, AkitaBatchedRootProof, AkitaProofStep,
    AkitaSetupSeed, AkitaVerifierSetup, BasisMode, ClaimIncidenceSummary, CleartextWitnessProof,
    LevelParams, RingCommitment, RingSubfieldEncoding, Schedule, SetupContributionMode, Step,
    VerifierClaims,
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

fn check_batched_proof_step_shape<F, L>(proof: &AkitaBatchedProof<F, L>) -> Result<(), AkitaError>
where
    F: FieldCore,
    L: FieldCore,
{
    match &proof.root {
        AkitaBatchedRootProof::Fold(_) => {
            let Some((last, rest)) = proof.steps.split_last() else {
                return Err(AkitaError::InvalidProof);
            };
            if !matches!(last, AkitaProofStep::Terminal(_))
                || rest
                    .iter()
                    .any(|step| !matches!(step, AkitaProofStep::Intermediate(_)))
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

fn effective_batched_schedule<Cfg, const D: usize>(
    incidence_summary: &ClaimIncidenceSummary,
    opening_points: &[&[Cfg::ClaimField]],
) -> Result<Schedule, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field> + ExtField<Cfg::ClaimField>,
{
    let num_vars = incidence_summary.num_vars();
    let mut schedule = Cfg::get_params_for_prove(incidence_summary)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, D>(
            opening_points,
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, D>(
            num_vars,
        ) {
            let commit_params = Cfg::get_params_for_batched_commitment(incidence_summary)?;
            schedule = root_direct_schedule(num_vars, commit_params)?;
        }
    }

    Ok(schedule)
}

fn validate_root_direct_recommitment_shape<F, const D: usize>(
    witnesses: &[CleartextWitnessProof<F>],
    setup_seed: &AkitaSetupSeed,
    incidence_summary: &ClaimIncidenceSummary,
    params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_level_dispatch::<D>(params)?;
    validate_log_basis(params.log_basis)?;
    if params.num_blocks == 0 || params.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "direct witness layout requires non-zero block geometry".to_string(),
        ));
    }
    if params.num_digits_commit == 0 || params.num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "direct witness layout requires non-zero digit depths".to_string(),
        ));
    }
    let expected_witness_len = 1usize
        .checked_shl(
            u32::try_from(incidence_summary.num_vars()).map_err(|_| AkitaError::InvalidProof)?,
        )
        .ok_or(AkitaError::InvalidProof)?;
    let direct_capacity = params
        .num_blocks
        .checked_mul(params.block_len)
        .ok_or_else(|| AkitaError::InvalidSetup("direct witness capacity overflow".to_string()))?;
    if expected_witness_len.div_ceil(D) > direct_capacity {
        return Err(AkitaError::InvalidSetup(
            "direct witness exceeds selected verifier layout".to_string(),
        ));
    }
    let total_group_polys = incidence_summary
        .num_polys_per_point()
        .iter()
        .try_fold(0usize, |acc, &count| {
            acc.checked_add(count).ok_or(AkitaError::InvalidProof)
        })?;
    if total_group_polys != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }

    let a_required_cols = params
        .block_len
        .checked_mul(params.num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("direct A width overflow".to_string()))?;
    let a_required = params
        .a_key
        .row_len()
        .checked_mul(a_required_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("direct A footprint overflow".to_string()))?;
    let per_witness_outer_cols = params
        .num_blocks
        .checked_mul(params.a_key.row_len())
        .and_then(|cols| cols.checked_mul(params.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("direct B width overflow".to_string()))?;
    let mut claim_offset = 0usize;
    for &point_size in incidence_summary.num_polys_per_point() {
        let b_required_cols = point_size
            .checked_mul(per_witness_outer_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("direct B width overflow".to_string()))?;
        let b_required = params
            .b_key
            .row_len()
            .checked_mul(b_required_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("direct B footprint overflow".to_string()))?;
        if a_required.max(b_required) > setup_seed.max_setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for direct witness layout".to_string(),
            ));
        }
        let group_end = claim_offset
            .checked_add(point_size)
            .ok_or(AkitaError::InvalidProof)?;
        for witness in &witnesses[claim_offset..group_end] {
            let witness_len = witness
                .as_field_elements()
                .ok_or(AkitaError::InvalidProof)?
                .coeff_len();
            if witness_len != expected_witness_len {
                return Err(AkitaError::InvalidProof);
            }
        }
        claim_offset = group_end;
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

#[cfg(feature = "zk")]
pub(crate) fn zk_b_blinding_rows<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    params: &LevelParams,
    blinding_digits: &[i8],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if D == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let row_len = params.b_key.row_len();
    let row_width = akita_types::zk::blinding_column_count::<F>(row_len, D, params.log_basis);
    let expected_digits = row_width.checked_mul(D).ok_or(AkitaError::InvalidProof)?;
    if blinding_digits.len() != expected_digits {
        return Err(AkitaError::InvalidProof);
    }
    let digits = blinding_digits
        .chunks_exact(D)
        .map(|chunk| {
            let mut plane = [0i8; D];
            plane.copy_from_slice(chunk);
            plane
        })
        .collect::<Vec<_>>();
    let b_zk_view = setup
        .expanded
        .zk_b_matrix()
        .ring_view::<D>(row_len, row_width)?;
    let b_zk_rows: Vec<_> = b_zk_view.rows().collect();
    Ok(mat_vec_mul_i8_plain::<F, D>(&b_zk_rows, &digits))
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

pub(crate) fn direct_decomposed_inner_rows<F, const D: usize>(
    witness_rings: &[CyclotomicRing<F, D>],
    setup: &AkitaVerifierSetup<F>,
    params: &LevelParams,
) -> Result<Vec<[i8; D]>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let a_matrix = setup
        .expanded
        .shared_matrix()
        .ring_view::<D>(params.a_key.row_len(), params.a_key.col_len())?;
    let a_rows: Vec<_> = a_matrix.rows().collect();
    let out_capacity = params
        .num_blocks
        .checked_mul(params.a_key.row_len())
        .and_then(|len| len.checked_mul(params.num_digits_open))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("direct witness row capacity overflow".to_string())
        })?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..params.num_blocks {
        let start = block_idx.checked_mul(params.block_len).ok_or_else(|| {
            AkitaError::InvalidSetup("direct witness block offset overflow".to_string())
        })?;
        let block = if start < witness_rings.len() {
            let end = start
                .checked_add(params.block_len)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("direct witness block end overflow".to_string())
                })?
                .min(witness_rings.len());
            &witness_rings[start..end]
        } else {
            &[]
        };
        let block_digits = decompose_rows_i8(block, params.num_digits_commit, params.log_basis);
        let t_rows = mat_vec_mul_i8_plain::<F, D>(&a_rows, &block_digits);
        out.extend(decompose_rows_i8(
            &t_rows,
            params.num_digits_open,
            params.log_basis,
        ));
    }

    Ok(out)
}

fn recommit_direct_witness_group<F, const D: usize>(
    group_witnesses: &[CleartextWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    params: &LevelParams,
    #[cfg(feature = "zk")] blinding_digits: &[i8],
) -> Result<RingCommitment<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    // Root-direct commitments are single-tier only: the sent commitment is the
    // plain `B·t̂`. Tiering is never planned on the root-direct (small-instance)
    // path.
    if params.f_key.is_some() {
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
        outer_input.extend(direct_decomposed_inner_rows(&witness_rings, setup, params)?);
    }

    let b_matrix = setup
        .expanded
        .shared_matrix()
        .ring_view::<D>(params.b_key.row_len(), outer_input.len())?;
    let b_rows: Vec<_> = b_matrix.rows().collect();
    let u = mat_vec_mul_i8_plain::<F, D>(&b_rows, &outer_input);
    #[cfg(feature = "zk")]
    {
        let mut u = u;
        let blinding_rows = zk_b_blinding_rows::<F, D>(setup, params, blinding_digits)?;
        for (row, blinding) in u.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
        Ok(RingCommitment { u })
    }
    #[cfg(not(feature = "zk"))]
    {
        Ok(RingCommitment { u })
    }
}

/// Recompute root-direct commitments from direct witnesses and compare them to
/// the proof commitments.
///
/// # Errors
///
/// Returns an error if the direct witness shape does not match the batch shape,
/// if witness reconstruction fails, or if any recomputed commitment differs
/// from the proof commitment.
pub(crate) fn verify_root_direct_commitments_with_params<F, const D: usize>(
    witnesses: &[CleartextWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    flat_commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    params: &LevelParams,
    #[cfg(feature = "zk")] b_blinding_digits: &[Vec<i8>],
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
{
    if flat_commitments.len() != incidence_summary.num_points() {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(feature = "zk")]
    if b_blinding_digits.len() != flat_commitments.len() {
        return Err(AkitaError::InvalidProof);
    }
    validate_root_direct_recommitment_shape::<F, D>(
        witnesses,
        setup.expanded.seed(),
        incidence_summary,
        params,
    )?;

    let mut claim_offset = 0usize;
    for (group_idx, &group_size) in incidence_summary.num_polys_per_point().iter().enumerate() {
        #[cfg(not(feature = "zk"))]
        let _ = group_idx;
        let group_witnesses = &witnesses[claim_offset..claim_offset + group_size];
        let commitment = recommit_direct_witness_group::<F, D>(
            group_witnesses,
            setup,
            params,
            #[cfg(feature = "zk")]
            &b_blinding_digits[group_idx],
        )?;
        if commitment != flat_commitments[group_idx] {
            return Err(AkitaError::InvalidProof);
        }
        claim_offset += group_size;
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
/// rewrite, transcript instance-descriptor binding, root-direct and folded-root
/// dispatch, and recursive verifier replay.
///
/// The root-direct branch recomputes commitments with the same root commitment
/// layout the prover used at commit time (`Cfg::get_params_for_batched_commitment`
/// for the same incidence); a mismatching layout would cause root-direct
/// commitment recomputation to reject a correctly produced proof.
///
/// # Errors
///
/// Returns an error if public claims are malformed, schedule/layout policy
/// rejects the proof shape, root-direct commitment recomputation rejects, or
/// proof replay fails.
pub fn verify_batched<'a, Cfg, T, const D: usize>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ChallengeField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    transcript: &mut T,
    claims: VerifierClaims<'a, Cfg::ClaimField, RingCommitment<Cfg::Field, D>>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>
        + ExtField<Cfg::ClaimField>
        + FrobeniusExtField<Cfg::Field>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field>,
{
    // Reject malformed step shapes that the downstream `fold_levels()` filter
    // would silently skip past.
    check_batched_proof_step_shape(proof)?;

    let prepared_claims = prepare_verifier_claims(&setup.expanded, &claims)?;
    let schedule = effective_batched_schedule::<Cfg, D>(
        &prepared_claims.incidence_summary,
        &prepared_claims.opening_points,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    validate_schedule_onehot_chunk_size::<Cfg>(&schedule)?;

    bind_transcript_instance_descriptor::<Cfg::Field, T, D, Cfg>(
        &setup.expanded,
        &prepared_claims.incidence_summary,
        &schedule,
        basis,
        transcript,
    )?;

    let PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        incidence_summary,
    } = prepared_claims;

    match &proof.root {
        AkitaBatchedRootProof::ZeroFold { witnesses, .. } => {
            #[cfg(feature = "zk")]
            if !proof.zk_hiding.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            let Some(Step::Direct(direct)) = schedule.steps.first() else {
                return Err(AkitaError::InvalidProof);
            };
            let params = direct.params.as_ref().ok_or(AkitaError::InvalidProof)?;
            verify_zero_fold_openings_with_incidence(
                witnesses,
                &opening_points,
                &openings,
                &incidence_summary,
                basis,
            )?;
            #[cfg(feature = "zk")]
            let direct_commitment_payload = proof
                .root
                .direct_b_blinding_digits()
                .ok_or(AkitaError::InvalidProof)?;
            verify_root_direct_commitments_with_params::<Cfg::Field, D>(
                witnesses,
                setup,
                &commitments,
                &incidence_summary,
                params,
                #[cfg(feature = "zk")]
                direct_commitment_payload,
            )?;
        }
        AkitaBatchedRootProof::Fold(_) | AkitaBatchedRootProof::Terminal(_) => {
            verify_folded_batched_proof::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, T, D>(
                proof,
                setup,
                transcript,
                &opening_points,
                &openings,
                &commitments,
                &incidence_summary,
                basis,
                &schedule,
                setup_contribution_mode,
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp32;
    use akita_types::{AjtaiKeyParams, FlatRingVec, SisModulusFamily};

    type F = Fp32<251>;
    const D: usize = 32;

    fn stage1_config() -> SparseChallengeConfig {
        SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![1],
        }
    }

    fn incidence_summary(num_vars: usize) -> ClaimIncidenceSummary {
        ClaimIncidenceSummary::same_point(num_vars, 1).expect("valid incidence summary")
    }

    #[test]
    fn root_direct_recommitment_rejects_undersized_setup() {
        let params =
            LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config())
                .with_decomp(1, 0, 2, 1, 0)
                .expect("valid direct layout");
        let setup_seed = AkitaSetupSeed {
            max_num_vars: 6,
            max_num_batched_polys: 1,
            max_num_points: 1,
            gen_ring_dim: D,
            max_setup_len: 3,
            #[cfg(feature = "zk")]
            max_zk_b_len: 1,
            #[cfg(feature = "zk")]
            max_zk_d_len: 1,
            public_matrix_seed: [0u8; 32],
        };
        let witnesses = vec![CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(vec![F::zero(); 64]),
        )];
        let err = validate_root_direct_recommitment_shape::<F, D>(
            &witnesses,
            &setup_seed,
            &incidence_summary(6),
            &params,
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
        params.b_key = AjtaiKeyParams::new_unchecked(SisModulusFamily::Q32, 1, 128, 0, D);
        let setup_seed = AkitaSetupSeed {
            max_num_vars: 6,
            max_num_batched_polys: 1,
            max_num_points: 1,
            gen_ring_dim: D,
            max_setup_len: 128,
            #[cfg(feature = "zk")]
            max_zk_b_len: 1,
            #[cfg(feature = "zk")]
            max_zk_d_len: 1,
            public_matrix_seed: [0u8; 32],
        };
        let witnesses = vec![CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(vec![F::zero(); 32]),
        )];
        let err = validate_root_direct_recommitment_shape::<F, D>(
            &witnesses,
            &setup_seed,
            &incidence_summary(6),
            &params,
        )
        .expect_err("num_vars=6 requires 64 direct witness elements");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
