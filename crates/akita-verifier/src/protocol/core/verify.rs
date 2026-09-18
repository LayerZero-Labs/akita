use super::root_fold::verify_root_native;
use super::suffix::{verify_suffix_native, NativeSuffixVerifierState};
// Top-level batched verifier orchestration once a schedule is selected.

use akita_config::{
    ensure_verifier_schedule_fits_setup, transcript_instance_descriptor, CommitmentConfig,
    TrustedScheduleCatalog,
};
use akita_error::AkitaError;
use akita_serialization::{AkitaSerialize, Valid};
use jolt_field::{CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};

use akita_types::{
    validate_schedule_ring_dims, AkitaVerifierSetup, BasisMode, CommittedGroupBatchProfile,
    FpExtEncoding, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
};

/// Verify one authoritative native Spongefish argument under config `Cfg`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn batched_verify<Cfg>(
    proof: &[u8],
    setup: &AkitaVerifierSetup<Cfg::Field>,
    schedules: &TrustedScheduleCatalog<Cfg>,
    session: &[u8],
    statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field:
        Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + ExtField<Cfg::Field> + Ring + AkitaSerialize + Valid,
{
    let selection = statement.selection();
    let claims = statement.into_claims();
    claims
        .validate(setup.expanded().descriptor())
        .map_err(|_| AkitaError::InvalidProof)?;
    let opening_batch = claims
        .committed_layout()
        .map_err(|_| AkitaError::InvalidProof)?;
    let (final_group, precommitteds) = claims
        .groups()
        .split_last()
        .ok_or(AkitaError::InvalidProof)?;
    let final_descriptor = *final_group.commitment().profile();
    if final_descriptor.group.num_vars() != final_group.num_vars()
        || final_descriptor.group.num_polynomials() != final_group.num_evaluations()
        || precommitteds.iter().any(|group| {
            let descriptor = group.commitment().profile();
            descriptor.group.num_vars() != group.num_vars()
                || descriptor.group.num_polynomials() != group.num_evaluations()
        })
    {
        return Err(AkitaError::InvalidProof);
    }
    for group in claims.groups() {
        let committed = group.commitment();
        let descriptor = committed.profile();
        descriptor
            .validate_frozen_precommit(Cfg::decomposition().field_bits())
            .map_err(|_| AkitaError::InvalidProof)?;
        let source_coefficients = descriptor
            .outer_slice_count
            .complete_source_coefficients(
                descriptor.outer.matrix.output_rank(),
                descriptor.outer.matrix.ring_dimension(),
            )
            .map_err(|_| AkitaError::InvalidProof)?;
        let plan = akita_types::CompressionChainPlan::for_complete_source(
            descriptor.outer.matrix.sis_table_key().modulus_profile,
            source_coefficients,
        )?;
        if committed.commitment().rows().coeff_len() != plan.terminal_coefficients() {
            return Err(AkitaError::InvalidProof);
        }
    }
    let batch_profile = CommittedGroupBatchProfile {
        final_group: final_descriptor,
        precommitteds: precommitteds
            .iter()
            .map(|group| *group.commitment().profile())
            .collect(),
    };
    batch_profile
        .validate(Cfg::decomposition().field_bits())
        .map_err(|_| AkitaError::InvalidProof)?;
    let resolved = schedules.resolve_selection(selection)?;
    resolved
        .validate_opening_layout(&opening_batch)
        .map_err(|_| AkitaError::InvalidProof)?;
    if resolved.profiles() != &batch_profile {
        return Err(AkitaError::InvalidProof);
    }
    let schedule = resolved.schedule();
    let root_params = &schedule.root_fold().params;
    let expected_final_descriptor =
        akita_types::GroupCommitPhaseParams::try_from_params(final_descriptor.group, root_params)
            .map_err(|_| AkitaError::InvalidProof)?;
    if final_descriptor != expected_final_descriptor
        || root_params.precommitted_groups().len() != precommitteds.len()
        || root_params
            .precommitted_groups()
            .iter()
            .zip(precommitteds)
            .any(|(params, claims_group)| params.profile != *claims_group.commitment().profile())
    {
        return Err(AkitaError::InvalidProof);
    }
    validate_schedule_ring_dims(schedule)?;
    ensure_verifier_schedule_fits_setup(setup.expanded().as_ref(), schedule, &opening_batch)?;
    schedule
        .validate_nonterminal_opening_execution(Cfg::EXT_DEGREE)
        .map_err(|_| AkitaError::InvalidProof)?;
    super::terminal_ntt::warm_for_schedule(setup, schedule)?;
    let (grinding_plan, descriptor_bytes) = transcript_instance_descriptor::<Cfg::Field, Cfg>(
        setup.expanded(),
        &opening_batch,
        selection,
        schedule,
        basis,
    )?;
    let state = akita_transcript::new_native_verifier(session, &descriptor_bytes, proof)
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut grinding = akita_types::NativeVerifierGrinding::new(state, &grinding_plan);
    let raw_groups = claims
        .groups()
        .iter()
        .map(|group| {
            PolynomialGroupClaims::new(
                group.point().to_vec(),
                group.evaluations().to_vec(),
                group.commitment().commitment(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| AkitaError::InvalidProof)?;
    let raw_claims =
        OpeningClaims::from_groups(raw_groups).map_err(|_| AkitaError::InvalidProof)?;
    let root = verify_root_native::<Cfg::Field, Cfg::ExtField>(
        setup,
        &mut grinding,
        &raw_claims,
        &opening_batch,
        basis,
        root_params,
        schedule.recursive_folds.first(),
        &schedule.terminal,
    )
    .map_err(|error| AkitaError::InvalidInput(format!("native root replay failed: {error:?}")))?;
    verify_suffix_native::<Cfg::Field, Cfg::ExtField>(
        setup,
        &mut grinding,
        schedule,
        NativeSuffixVerifierState {
            opening_point: root.challenges,
            opening: root.opening,
            witness: root.next_witness,
            basis: BasisMode::Lagrange,
            witness_len: schedule.root_fold().output_witness_len,
            setup_prefix_opening: root.setup_prefix_opening,
        },
    )
    .map_err(|error| AkitaError::InvalidInput(format!("native suffix replay failed: {error:?}")))?;
    grinding.finish().map(|_accepted| ()).map_err(|error| {
        AkitaError::InvalidInput(format!("native proof completion failed: {error:?}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{RingVec, RingView};
    use jolt_field::{Fp32, Zero};

    type F = Fp32<251>;
    const D: usize = 32;

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
