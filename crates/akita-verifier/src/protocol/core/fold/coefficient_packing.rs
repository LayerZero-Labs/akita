//! Coefficient-packing fold verifier prefix.

use super::{FoldClaimMaterial, PreparedFoldOpeningPoint};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::{
    append_claim_values_to_transcript, BasisMode, Commitment, CommittedGroupParams, FpExtEncoding,
    OpeningClaims, OpeningClaimsLayout, PreparedSubringCoefficientPackingPoint,
    SubringCoefficientPackingGeometry,
};

fn prepare_group<E: FieldCore>(
    point: &[E],
    basis: BasisMode,
    source_num_vars: usize,
    group_params: &(impl akita_types::LevelParamsLike + ?Sized),
    extension_degree: usize,
) -> Result<PreparedSubringCoefficientPackingPoint<E>, AkitaError> {
    let akita_types::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension,
    } = group_params.opening_method()
    else {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing prefix received an EvaluationTrace group".into(),
        ));
    };
    let d_a = group_params.inner_commit_matrix_params().ring_dimension();
    let geometry = SubringCoefficientPackingGeometry::try_new(
        extension_degree,
        d_a,
        challenge_subring_dimension,
    )?;
    PreparedSubringCoefficientPackingPoint::new(
        geometry,
        basis,
        group_params.num_live_ring_elements_per_claim(),
        group_params.num_positions_per_block(),
        source_num_vars,
        point,
    )
}

fn prepare_prefix_points<F: FieldCore, E: ExtField<F>, C>(
    claims: &OpeningClaims<'_, E, C>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    level_params: &CommittedGroupParams,
) -> Result<Vec<PreparedFoldOpeningPoint<F, E>>, AkitaError> {
    if openings.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let mut prepared_points = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let layout = opening_batch.group_layout(group_index)?;
        let group_params = level_params.group_params_geometry(opening_batch, group_index)?;
        prepared_points.push(PreparedFoldOpeningPoint::SubringCoefficientPacking(
            prepare_group::<E>(
                claims.group_point(group_index)?,
                basis,
                layout.num_vars(),
                &group_params,
                E::EXT_DEGREE,
            )?,
        ));
    }
    Ok(prepared_points)
}

/// Prepare a root packing prefix directly from the authenticated public claims.
pub(in crate::protocol::core) fn verify_coefficient_packing_root_prefix<F, E>(
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
) -> Result<FoldClaimMaterial<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize,
{
    let prepared_points =
        prepare_prefix_points::<F, E, _>(claims, openings, opening_batch, basis, root_lp)?;
    Ok(FoldClaimMaterial {
        prepared_points,
        openings: openings.to_vec(),
        reduction_final_claims: None,
        reduction_factors: None,
    })
}

/// Prepare a recursive packing prefix without extension-opening reduction.
pub(in crate::protocol::core) fn verify_coefficient_packing_suffix_prefix<F, E, T>(
    claims: &OpeningClaims<'_, E>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    lp: &CommittedGroupParams,
    transcript: &mut T,
) -> Result<FoldClaimMaterial<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let prepared_points =
        prepare_prefix_points::<F, E, _>(claims, openings, opening_batch, basis, lp)?;
    let mut point_refs = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let point = claims.group_point(group_index)?;
        point_refs.push(point);
    }
    super::single_field::absorb_protocol_opening_points::<F, E, T>(&point_refs, transcript);
    append_claim_values_to_transcript::<F, E, T>(openings, transcript);
    Ok(FoldClaimMaterial {
        prepared_points,
        openings: openings.to_vec(),
        reduction_final_claims: None,
        reduction_factors: None,
    })
}
