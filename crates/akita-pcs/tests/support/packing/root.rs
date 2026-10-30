//! Rebuild one root with the fixture's reduced-width packing geometry.
use super::geometry::rebuild_group_output_matrices;
use akita_challenges::FoldChallengeFamily;
use akita_error::AkitaError;
use akita_params::{FoldParams, ScheduleLookupKey};
use akita_schedules::PlannerPolicy;

pub(super) fn build_root(
    template: &FoldParams,
    key: &ScheduleLookupKey,
    challenge_subring_dimension: usize,
    policy: &PlannerPolicy,
) -> Result<FoldParams, AkitaError> {
    let mut fold = template.clone();
    let root = &mut fold.params;
    let d_a = root.inner().matrix.ring_dimension();
    akita_params::SubringCoefficientPackingGeometry::try_new(
        policy.claim_ext_degree,
        d_a,
        challenge_subring_dimension,
    )?;
    root.own_group_mut().opening.opening_method =
        akita_params::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        };
    root.source_encoding = akita_params::CommittedSourceEncoding::CanonicalCoefficientTable;
    root.own_group_mut().opening.fold_challenge_config =
        FoldChallengeFamily::production_for_ring_dim(challenge_subring_dimension).ok_or_else(
            || {
                AkitaError::InvalidSetup(
                    "root packing subring is not in the production challenge ladder".into(),
                )
            },
        )?;
    let root_open_dimension = 128usize.min(
        policy
            .claim_ext_degree
            .checked_mul(challenge_subring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("packing width overflow".into()))?,
    );
    let root_open_bound = akita_params::sis::rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_params::SisMatrixRole::Open,
        root_open_dimension,
        root.open().digits.log_basis,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("root packing test has no audited D bound".into()))?;
    let mut root_open_key = root.open_matrix.sis_table_key();
    root_open_key.ring_dimension = root_open_dimension
        .try_into()
        .map_err(|_| AkitaError::InvalidSetup("root packing D dimension exceeds u32".into()))?;
    root_open_key.coeff_linf_bound = root_open_bound;
    root.open_matrix = akita_params::OpenCommitMatrixParams::try_new_with_min_rank(
        root_open_key,
        root.open_matrix.input_width(),
    )?;
    let required_a_bound = akita_params::sis::rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        d_a,
        root.open().digits.log_basis,
        &root.fold_challenge_config(),
        root.num_digits_fold(),
        root.witness_chunk.num_chunks,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup("root packing challenge family has no audited A bound".into())
    })?;
    let current_a = root.inner().matrix;
    let mut current_key = current_a.sis_table_key().ok_or_else(|| {
        AkitaError::InvalidSetup("root packing requires a L-infinity A matrix".into())
    })?;
    current_key.coeff_linf_bound = required_a_bound;
    root.own_group_mut().profile.inner.matrix =
        akita_params::InnerCommitMatrixParams::try_new_with_min_rank(
            current_key,
            current_a.input_width(),
        )?;
    rebuild_group_output_matrices(
        root,
        key.final_group.num_polynomials(),
        policy.claim_ext_degree,
    )?;

    let opening_batch = key.opening_layout()?;
    let root_output_witness_len = root.output_witness_len_for_field_bits(
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
        &opening_batch,
    )?;
    fold.output_witness_len = root_output_witness_len;

    Ok(fold)
}
