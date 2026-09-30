use super::terminal::verify_terminal_suffix;
use super::{verify_fold, NextWitnessPlan, PreparedFoldReplay, SetupPrefixOpening};
use crate::prepared_cache::TerminalNttCache;
use crate::stages::opening_claims::{
    finalize_claims, prepare_single_field_suffix_groups, verify_coefficient_packing_suffix_prefix,
    verify_extension_claim_suffix_prefix, FoldClaimMaterial, PreparedFoldOpeningPoint,
};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::GrindingReplay;
use akita_types::{
    AkitaVerifierSetup, BasisMode, CommittedGroupParams, FoldParams, FoldSchedule, FpExtEncoding,
    OpeningClaims, OpeningClaimsLayout, PolynomialGroupClaims, RelationWitnessGeometry, RingVec,
    SetupContributionMode, TerminalFoldParams,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};

pub(super) struct SuffixVerifierState<F: Field, E: Field> {
    pub opening_point: Vec<E>,
    pub opening: E,
    pub witness: RingVec<F>,
    pub basis: BasisMode,
    pub witness_len: usize,
    pub setup_prefix_opening: Option<SetupPrefixOpening<E>>,
}

fn suffix_commitment_payloads<F, E>(
    setup: &AkitaVerifierSetup<F>,
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    witness_commitment: &RingVec<F>,
) -> Result<Vec<RingVec<F>>, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    let mut group_payloads = Vec::with_capacity(opening_batch.num_groups());
    if let Some(setup_prefix) = lp.setup_prefix() {
        let setup_prefix_id = setup_prefix.slot_id().ok_or_else(|| {
            AkitaError::InvalidSetup("selected setup-prefix group has no slot identity".to_string())
        })?;
        let slot = setup.prefix_slots().get(&setup_prefix_id).ok_or_else(|| {
            AkitaError::InvalidSetup(
                "planned setup-prefix slot is missing from verifier setup".to_string(),
            )
        })?;
        let mut coeffs = Vec::new();
        for row in &slot.commitment.rows {
            coeffs.extend_from_slice(row.coeffs());
        }
        group_payloads.push(RingVec::from_coeffs(coeffs));
    }
    group_payloads.push(RingVec::from_coeffs(witness_commitment.coeffs().to_vec()));
    if group_payloads.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidProof);
    }

    let relation_geometry =
        RelationWitnessGeometry::for_level(lp, opening_batch, <E as ExtField<F>>::DEGREE)?;
    let relation_layout = relation_geometry.rhs_layout();
    let mut ordered = Vec::with_capacity(group_payloads.len());
    for (relation_group_index, group_index) in
        opening_batch.root_group_order()?.into_iter().enumerate()
    {
        let payload = group_payloads
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if payload.coeff_len()
            != relation_layout
                .group_payload_geometry(relation_group_index)?
                .transmitted_coefficients()
        {
            return Err(AkitaError::InvalidProof);
        }
        ordered.push(payload.clone());
    }
    Ok(ordered)
}

#[allow(clippy::too_many_arguments)]
fn prepare_fold_replay<'a, F, E>(
    setup: &'a AkitaVerifierSetup<F>,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    current_state: &SuffixVerifierState<F, E>,
    lp: &'a CommittedGroupParams,
    output_witness_len: usize,
    next_params: Option<&'a FoldParams>,
    terminal: &TerminalFoldParams,
) -> Result<PreparedFoldReplay<'a, F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    let payload_geometry = lp.outer_payload_geometry()?;
    if current_state.witness.coeff_len() != payload_geometry.transmitted_coefficients() {
        return Err(AkitaError::InvalidProof);
    }
    akita_transcript::public_fields_verifier(
        grinding.state_mut(),
        akita_types::FoldSite::WitnessCommitment {
            level,
            ring_dimension: payload_geometry.transcript_ring_dimension(),
        }
        .id()?,
        current_state.witness.coeffs(),
    )?;
    let recursive_num_vars = lp.recursive_opening_num_vars()?;
    if current_state.opening_point.len() > recursive_num_vars {
        return Err(AkitaError::InvalidProof);
    }
    let block_claims = match (&current_state.setup_prefix_opening, lp.setup_prefix()) {
        (Some((setup_prefix_point, setup_prefix_eval)), Some(_)) => {
            OpeningClaims::from_groups(vec![
                PolynomialGroupClaims::new(
                    setup_prefix_point.clone(),
                    vec![*setup_prefix_eval],
                    (),
                )?,
                PolynomialGroupClaims::new(
                    current_state.opening_point.clone(),
                    vec![current_state.opening],
                    (),
                )?,
            ])?
        }
        (None, None) => OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            current_state.opening_point.clone(),
            vec![current_state.opening],
            (),
        )?])?,
        _ => return Err(AkitaError::InvalidProof),
    };
    let opening_batch = block_claims.layout()?;
    let openings = block_claims.flat_evaluations();
    let group_points = (0..opening_batch.num_groups())
        .map(|group_index| block_claims.group_point(group_index))
        .collect::<Result<Vec<_>, _>>()?;
    let material = if matches!(
        lp.opening_method(),
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ) {
        verify_coefficient_packing_suffix_prefix::<F, E>(
            &block_claims,
            &openings,
            &opening_batch,
            current_state.basis,
            lp,
            grinding,
            level,
        )?
    } else if const { <E as ExtField<F>>::DEGREE == 1 } {
        let prepared =
            prepare_single_field_suffix_groups::<F, E>(&block_claims, lp, &opening_batch)?;
        for (group_index, point) in group_points.iter().enumerate() {
            akita_transcript::public_extensions::<F, E, _>(
                grinding.state_mut(),
                akita_types::FoldSite::GroupPoint {
                    level,
                    group: group_index,
                }
                .id()?,
                point,
            )?;
        }
        akita_transcript::public_extensions::<F, E, _>(
            grinding.state_mut(),
            akita_types::FoldSite::Openings { level }.id()?,
            &openings,
        )?;
        FoldClaimMaterial {
            prepared_points: prepared
                .into_iter()
                .map(PreparedFoldOpeningPoint::EvaluationTrace)
                .collect(),
            openings: openings.clone(),
            reduction_final_claims: None,
            reduction_factors: None,
        }
    } else {
        verify_extension_claim_suffix_prefix::<F, E>(
            &group_points,
            &openings,
            &opening_batch,
            current_state.basis,
            lp,
            grinding,
            level,
        )?
    };
    let relation_geometry = RelationWitnessGeometry::for_level(lp, &opening_batch, E::DEGREE)?;
    let opening_geometry = relation_geometry.rhs_layout().opening_payload_geometry()?;
    let opening_payload = akita_transcript::receive_field_group::<F>(
        grinding.state_mut(),
        akita_types::FoldSite::OpeningPayload {
            level,
            ring_dimension: opening_geometry.transcript_ring_dimension(),
        }
        .id()?,
        opening_geometry.transmitted_coefficients(),
    )
    .map(RingVec::from_coeffs)?;
    let prefix = finalize_claims::<F, E>(&opening_batch, material, grinding, level)?;
    let commitment_payloads =
        suffix_commitment_payloads::<F, E>(setup, lp, &opening_batch, &current_state.witness)?;
    let (next_witness, next_witness_ring_dim, next_opening_source_len, stage3) =
        if let Some(next) = next_params {
            let ring_dim = next.params.d_a();
            let coefficient_count = next
                .params
                .outer_payload_geometry()?
                .transmitted_coefficients();
            let committed_len =
                akita_types::witness_commitment_domain_len(output_witness_len, ring_dim)?;
            (
                NextWitnessPlan::OuterPayload { coefficient_count },
                ring_dim,
                committed_len / ring_dim,
                matches!(
                    next.predecessor_setup_contribution_mode(),
                    SetupContributionMode::Recursive
                )
                .then_some(&next.params),
            )
        } else {
            let ring_dim = terminal.d_a();
            let coefficient_count = terminal
                .response_shape
                .layout
                .groups
                .first()
                .ok_or(AkitaError::InvalidProof)?
                .t_field_elems;
            let committed_len =
                akita_types::witness_commitment_domain_len(output_witness_len, ring_dim)?;
            (
                NextWitnessPlan::TerminalT { coefficient_count },
                ring_dim,
                committed_len / ring_dim,
                None,
            )
        };
    let challenge_field_bits = F::MODULUS_BITS
        .checked_mul(
            u32::try_from(E::DEGREE)
                .map_err(|_| AkitaError::InvalidSetup("extension degree overflow".into()))?,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("challenge field width overflow".into()))?;
    let level_layout = akita_types::nonterminal_level_layout(
        F::MODULUS_BITS,
        challenge_field_bits,
        lp,
        lp.relation_address_geometry(
            &opening_batch,
            E::DEGREE,
            next_witness_ring_dim,
            output_witness_len,
        )?,
        next_params.map(|next| &next.params),
    )?;
    Ok(PreparedFoldReplay {
        lp,
        level,
        opening_payload,
        opening_shape: opening_batch,
        commitment_payloads,
        prefix,
        w_len: output_witness_len,
        level_layout,
        next_witness,
        next_witness_ring_dim,
        next_opening_source_len,
        stage3,
        evaluation_trace_basis: current_state.basis,
    })
}

pub(super) fn verify_suffix<F, E>(
    setup: &AkitaVerifierSetup<F>,
    terminal_ntt: &TerminalNttCache,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    schedule: &FoldSchedule,
    mut current_state: SuffixVerifierState<F, E>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    for (offset, step) in schedule.recursive_folds.iter().enumerate() {
        let level = u32::try_from(offset + 1).map_err(|_| AkitaError::InvalidProof)?;
        if current_state.witness_len != step.input_witness_len {
            return Err(AkitaError::InvalidProof);
        }
        let next = schedule.recursive_folds.get(offset + 1);
        let prepared = prepare_fold_replay::<F, E>(
            setup,
            grinding,
            level,
            &current_state,
            &step.params,
            step.output_witness_len,
            next,
            &schedule.terminal,
        )?;
        let output = verify_fold(setup, grinding, prepared)?;
        current_state = SuffixVerifierState {
            opening_point: output.challenges,
            opening: output.opening,
            witness: output.next_witness,
            basis: BasisMode::Lagrange,
            witness_len: step.output_witness_len,
            setup_prefix_opening: output.setup_prefix_opening,
        };
    }
    verify_terminal_suffix(
        terminal_ntt,
        grinding,
        u32::try_from(schedule.recursive_folds.len() + 1).map_err(|_| AkitaError::InvalidProof)?,
        &current_state,
        &schedule.terminal,
    )
}
