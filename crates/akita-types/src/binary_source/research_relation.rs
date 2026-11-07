//! Assemble the direct research relation from the same public compiler on both sides.

use super::{BinaryAOperand, OrdinaryNativeRootClaim, ResearchBatchPlan, ResearchBinaryRelation};
use crate::native_root_relation::NativeRootWeights;
use crate::{AkitaSetupDescriptor, OpeningClaims};
use akita_challenges::Challenges;
use akita_error::AkitaError;
use akita_params::GroupCommitPhaseParams;
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::sync::Arc;

/// Bind payload slices to their exact local row ranges; no row is rebated or omitted.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn research_batch_relation<F, E, C>(
    plan: &ResearchBatchPlan,
    setup: &AkitaSetupDescriptor,
    claims: &OpeningClaims<'_, E, C>,
    profiles: &[GroupCommitPhaseParams],
    challenges: &[Challenges],
    d_payload: &[F],
    matrices: &[Arc<Vec<Vec<[i128; 648]>>>],
    alpha: E,
    batches: &[E],
    payload: impl for<'a> Fn(&'a C) -> &'a [F],
) -> Result<(NativeRootWeights<E>, Vec<(usize, E)>), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    if claims.num_groups() != 8
        || profiles.len() != 8
        || challenges.len() != 4
        || matrices.len() != 4
        || plan.d_ranges().last().map(|range| range.end) != Some(d_payload.len())
    {
        return Err(AkitaError::InvalidProof);
    }
    let commitments = claims
        .groups()
        .iter()
        .map(|group| {
            payload(group.commitment())
                .iter()
                .copied()
                .map(E::lift_base)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let d = d_payload
        .iter()
        .copied()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    let mut binary = Vec::with_capacity(4);
    for (index, range) in plan.d_ranges().iter().enumerate() {
        binary.push(ResearchBinaryRelation {
            challenges: challenges.get(index).ok_or(AkitaError::InvalidProof)?,
            commitment: commitments.get(index).ok_or(AkitaError::InvalidProof)?,
            d_payload: d.get(range.clone()).ok_or(AkitaError::InvalidProof)?,
            matrix: BinaryAOperand::new::<F>(matrices.get(index).ok_or(AkitaError::InvalidProof)?),
        });
    }
    let mut ordinary = Vec::with_capacity(8);
    let passes = if plan.mode() == super::ResearchTerminalMode::E64TwoPointS11a {
        2
    } else {
        1
    };
    for pass in 0..passes {
        for (index, group) in claims.groups().iter().enumerate().skip(4) {
            let crate::opening_claims::GroupClaimKind::ResearchOrdinary { width, mode, .. } =
                group.kind()
            else {
                return Err(AkitaError::InvalidProof);
            };
            let terminals = group.research_terminals().ok_or(AkitaError::InvalidProof)?;
            if *mode != plan.mode() || terminals.mode != plan.mode() {
                return Err(AkitaError::InvalidProof);
            }
            let terminal = terminals
                .validate::<F>()?
                .get(pass)
                .ok_or(AkitaError::InvalidProof)?;
            ordinary.push(OrdinaryNativeRootClaim {
                profile: profiles.get(index).ok_or(AkitaError::InvalidProof)?,
                width: *width,
                point: &terminal.point,
                evaluations: &terminal.evaluations,
                payload: commitments.get(index).ok_or(AkitaError::InvalidProof)?,
            });
        }
    }
    plan.relation(setup, &binary, &ordinary, alpha, batches)
}
