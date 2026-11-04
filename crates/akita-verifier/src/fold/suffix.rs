use super::terminal::verify_terminal_suffix;
use super::{
    receive_opening_payload, verify_fold, NextWitness, NextWitnessPlan, PreparedFoldReplay,
    SetupPrefixOpening,
};
use crate::claims::ClaimReplay;
use crate::prepared_cache::TerminalNttCache;
use crate::stages::opening_claims::{
    finalize_claims, prepare_single_field_suffix_groups, verify_coefficient_packing_suffix_prefix,
    verify_extension_claim_suffix_prefix, FoldClaimMaterial, PreparedFoldOpeningPoint,
};
use akita_error::AkitaError;
use akita_params::{
    BasisMode, CommittedGroupParams, FoldParams, FoldSchedule, OpeningClaimsLayout,
    RelationWitnessGeometry, SetupContributionMode, TerminalFoldParams,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::ChallengeCodec;
use akita_types::{
    AkitaVerifierSetup, FpExtEncoding, OpeningClaims, PolynomialGroupClaims, RingVec,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};

pub(super) struct SuffixVerifierState<F: Field, E: Field, V = E> {
    pub opening_point: Vec<E>,
    pub opening: V,
    pub witness: NextWitness<F>,
    pub basis: BasisMode,
    pub witness_len: usize,
    pub setup_prefix_opening: Option<SetupPrefixOpening<E>>,
}

impl<F: Field, E: Field, V> SuffixVerifierState<F, E, V> {
    /// The continuation the root fold hands the first recursive fold.
    pub(super) fn from_root(root: super::FoldVerifyOutput<F, E, V>, witness_len: usize) -> Self {
        Self {
            opening_point: root.challenges,
            opening: root.opening,
            witness: root.next_witness,
            basis: BasisMode::Lagrange,
            witness_len,
            setup_prefix_opening: root.setup_prefix_opening,
        }
    }
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
fn prepare_fold_replay<'a, F, E, G: ClaimReplay<F, E>>(
    setup: &'a AkitaVerifierSetup<F>,
    grinding: &mut G,
    level: u32,
    current_state: &SuffixVerifierState<F, E, G::Claim>,
    lp: &'a CommittedGroupParams,
    output_witness_len: usize,
    next_params: Option<&'a FoldParams>,
    terminal: &TerminalFoldParams,
) -> Result<PreparedFoldReplay<'a, F, E, G::Claim>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>
        + ChallengeCodec,
{
    let NextWitness::OuterPayload(witness) = &current_state.witness else {
        return Err(AkitaError::Internal(
            "recursive fold scheduled after a terminal t witness".into(),
        ));
    };
    let payload_geometry = lp.outer_payload_geometry()?;
    if witness.coeff_len() != payload_geometry.transmitted_coefficients() {
        return Err(AkitaError::InvalidProof);
    }
    // The outer payload was absorbed when it was received as the next witness.
    let recursive_num_vars = lp.recursive_opening_num_vars()?;
    if current_state.opening_point.len() > recursive_num_vars {
        return Err(AkitaError::InvalidProof);
    }
    // The witness evaluation is a replay claim, held beside the block's points:
    // `block_claims` carries a zero in its place and is read only for
    // geometry and points, except on the clear-only extension-field path.
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
                    vec![E::zero()],
                    (),
                )?,
            ])?
        }
        (None, None) => OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            current_state.opening_point.clone(),
            vec![E::zero()],
            (),
        )?])?,
        _ => return Err(AkitaError::InvalidProof),
    };
    let opening_batch = block_claims.layout()?;
    // Group order: the setup-prefix evaluation, then the witness evaluation.
    let prefix_evaluation = current_state
        .setup_prefix_opening
        .as_ref()
        .map(|(_, evaluation)| evaluation);
    let witness_evaluation = std::slice::from_ref(&current_state.opening);
    let openings: Vec<G::Claim> = prefix_evaluation
        .map(|evaluation| G::constant(*evaluation))
        .into_iter()
        .chain(std::iter::once(current_state.opening.clone()))
        .collect();
    let group_points = (0..opening_batch.num_groups())
        .map(|group_index| block_claims.group_point(group_index))
        .collect::<Result<Vec<_>, _>>()?;
    let material = if matches!(
        lp.opening_method(),
        akita_params::OpeningMethod::SubringCoefficientPacking { .. }
    ) {
        let material = verify_coefficient_packing_suffix_prefix::<F, E, _>(
            &block_claims,
            openings,
            &opening_batch,
            current_state.basis,
            lp,
            grinding,
            level,
        )?;
        grinding.bind_openings(level, prefix_evaluation, witness_evaluation)?;
        material
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
        grinding.bind_openings(level, prefix_evaluation, witness_evaluation)?;
        FoldClaimMaterial {
            prepared_points: prepared
                .into_iter()
                .map(PreparedFoldOpeningPoint::EvaluationTrace)
                .collect(),
            openings,
            reduction_final_claims: None,
            reduction_factors: None,
        }
    } else {
        let material = verify_extension_claim_suffix_prefix::<F, E, _>(
            &group_points,
            &G::clear_values(&openings)?,
            &opening_batch,
            current_state.basis,
            lp,
            grinding,
            level,
        )?;
        FoldClaimMaterial {
            prepared_points: material.prepared_points,
            openings: material
                .openings
                .into_iter()
                .map(G::clear_path)
                .collect::<Result<_, _>>()?,
            reduction_final_claims: material.reduction_final_claims,
            reduction_factors: material.reduction_factors,
        }
    };
    let relation_geometry = RelationWitnessGeometry::for_level(lp, &opening_batch, E::DEGREE)?;
    let opening_geometry = relation_geometry.rhs_layout().opening_payload_geometry()?;
    let opening_payload = receive_opening_payload::<F, E, _>(grinding, level, &opening_geometry)?;
    let prefix = finalize_claims::<F, E, _>(&opening_batch, material, grinding, level)?;
    let commitment_payloads =
        suffix_commitment_payloads::<F, E>(setup, lp, &opening_batch, witness)?;
    let (next_witness, next_witness_ring_dim, next_opening_source_len, stage3) =
        if let Some(next) = next_params {
            let ring_dim = next.params.d_a();
            let coefficient_count = next
                .params
                .outer_payload_geometry()?
                .transmitted_coefficients();
            let committed_len =
                akita_params::witness_commitment_domain_len(output_witness_len, ring_dim)?;
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
                akita_params::witness_commitment_domain_len(output_witness_len, ring_dim)?;
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
    let level_layout = akita_params::nonterminal_level_layout(
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

/// Replays recursive fold `offset + 1` from the predecessor's continuation and
/// returns its own.
pub(super) fn verify_suffix_level<F, E, G: ClaimReplay<F, E>>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut G,
    schedule: &FoldSchedule,
    offset: usize,
    current_state: SuffixVerifierState<F, E, G::Claim>,
    setup_policy: &mut crate::SetupPolicy<'_, '_, E>,
) -> Result<SuffixVerifierState<F, E, G::Claim>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>
        + ChallengeCodec,
{
    let step = schedule
        .recursive_folds
        .get(offset)
        .ok_or(AkitaError::InvalidProof)?;
    let level = u32::try_from(offset + 1).map_err(|_| AkitaError::InvalidProof)?;
    if current_state.witness_len != step.input_witness_len {
        return Err(AkitaError::InvalidProof);
    }
    let next = schedule.recursive_folds.get(offset + 1);
    let prepared = prepare_fold_replay::<F, E, _>(
        setup,
        grinding,
        level,
        &current_state,
        &step.params,
        step.output_witness_len,
        next,
        &schedule.terminal,
    )?;
    let output = verify_fold(setup, grinding, prepared, setup_policy)?;
    Ok(SuffixVerifierState {
        opening_point: output.challenges,
        opening: output.opening,
        witness: output.next_witness,
        basis: BasisMode::Lagrange,
        witness_len: step.output_witness_len,
        setup_prefix_opening: output.setup_prefix_opening,
    })
}

pub(super) fn verify_suffix<F, E>(
    setup: &AkitaVerifierSetup<F>,
    terminal_ntt: &TerminalNttCache,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    schedule: &FoldSchedule,
    mut current_state: SuffixVerifierState<F, E>,
    setup_policy: &mut crate::SetupPolicy<'_, '_, E>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>
        + ChallengeCodec,
{
    for offset in 0..schedule.recursive_folds.len() {
        current_state = verify_suffix_level(
            setup,
            grinding,
            schedule,
            offset,
            current_state,
            setup_policy,
        )?;
    }
    verify_terminal_suffix(
        terminal_ntt,
        grinding,
        u32::try_from(schedule.recursive_folds.len() + 1).map_err(|_| AkitaError::InvalidProof)?,
        &current_state,
        &schedule.terminal,
    )
}
