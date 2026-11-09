//! Shared transcript atoms of an explicitly selected binary root.

use super::{BinaryCommittedGroupManifest, BinaryWordClaim};
use crate::GrindingReplay;
use akita_challenges::{
    Challenges, FoldChallengeDistribution, FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw,
};
use akita_error::{checked, AkitaError};
use akita_params::{binary_setup::BINARY_SETUP_P64, GrindingSite};
use akita_transcript::{
    ext_challenges, ChallengeCodec, ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Extension seam: bind the complete order-zero binary and mixed statement.
/// <https://github.com/LayerZero-Labs/akita/issues/285>
///
/// Bind the source contract and caller's logical word claim before the field
/// switch produces its internal carrier terminal.
#[allow(clippy::type_complexity)]
pub fn binary_root_statement<F, E, S>(
    state: &mut S,
    manifest: &BinaryCommittedGroupManifest,
    claim: Option<&BinaryWordClaim>,
    prime: &super::PrimePlaneManifest,
    binary_commitment: &[F],
    ordinary: Option<(usize, &[F], &[E], &[E])>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    manifest.validate()?;
    manifest
        .field_profile()
        .check_field_pair(akita_params::field_modulus::<F>()?, E::DEGREE)?;
    if binary_commitment.len() != manifest.payload_coefficients()? {
        return Err(AkitaError::InvalidInput(
            "binary root payload shape mismatch".into(),
        ));
    }
    if let Some(claim) = claim {
        claim.validate(manifest.descriptor())?;
    }
    prime.validate(manifest.descriptor())?;
    if !prime.planes().is_empty() && prime.field_profile() != manifest.field_profile() {
        return Err(AkitaError::InvalidInput(
            "prime plane profile mismatch".into(),
        ));
    }
    let mut messages = vec![(3, manifest.canonical_bytes())];
    if let Some(claim) = claim {
        messages.push((4, claim.canonical_bytes()));
    }
    if !prime.planes().is_empty() {
        messages.push((5, prime.canonical_bytes()));
    }
    for (stage, bytes) in messages {
        akita_transcript::public_bytes(
            state,
            ProtocolSiteId {
                family: SITE_FAMILY_ROOT_STATEMENT,
                stage,
                ..ProtocolSiteId::default()
            },
            &bytes,
        )?;
    }
    // Both drivers embed base payloads in the selected challenge field.
    // P128 retains its degree-one encoding; P64 binds a zero second coordinate.
    let lift = |payload: &[F]| -> Result<Vec<E>, AkitaError> {
        let mut lifted = Vec::new();
        lifted
            .try_reserve_exact(payload.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        lifted.extend(payload.iter().copied().map(E::lift_base));
        Ok(lifted)
    };
    akita_transcript::public_extensions::<F, E, _>(
        state,
        crate::FoldSite::RootCommitment {
            group: 0,
            ring_dimension: manifest.b_ring_dimension(),
        }
        .id()?,
        &lift(binary_commitment)?,
    )?;
    if let Some((dimension, payload, point, evaluations)) = ordinary {
        akita_transcript::public_extensions::<F, E, _>(
            state,
            crate::FoldSite::RootCommitment {
                group: 1,
                ring_dimension: dimension,
            }
            .id()?,
            &lift(payload)?,
        )?;
        akita_transcript::public_extensions::<F, E, _>(
            state,
            crate::FoldSite::RootPoint { group: 1 }.id()?,
            point,
        )?;
        akita_transcript::public_extensions::<F, E, _>(
            state,
            crate::FoldSite::Openings { level: 0 }.id()?,
            evaluations,
        )?;
    }
    Ok(())
}

/// Bind the second ordinary record after the legacy first-point statement.
/// The new domain is emitted only for the explicit two-point shape.
pub fn ordinary_second_point_statement<F, E, S>(
    state: &mut S,
    second: crate::opening_claims::OrdinaryPointEvaluation<'_, E>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    akita_params::binary_setup::BinaryFieldProfile::P64
        .check_field_pair(akita_params::field_modulus::<F>()?, E::DEGREE)?;
    let mut bytes = b"AkitaOrdinaryPoints/e64".to_vec();
    bytes.push(1);
    bytes.extend_from_slice(&2u64.to_le_bytes());
    for values in [second.point, second.evaluations] {
        bytes.extend_from_slice(
            &u64::try_from(values.len())
                .map_err(|_| AkitaError::InvalidProof)?
                .to_le_bytes(),
        );
        let additional = checked::product([16, values.len()]).ok_or(AkitaError::InvalidProof)?;
        bytes
            .try_reserve_exact(additional)
            .map_err(|_| AkitaError::InvalidProof)?;
        for value in values {
            for coordinate in 0..2 {
                let residue = value
                    .base_coefficient(coordinate)
                    .to_u128_checked()
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or(AkitaError::InvalidProof)?;
                bytes.extend_from_slice(&residue.to_le_bytes());
            }
        }
    }
    akita_transcript::public_bytes(
        state,
        ProtocolSiteId {
            family: SITE_FAMILY_ROOT_STATEMENT,
            stage: 8,
            ..ProtocolSiteId::default()
        },
        &bytes,
    )
}

/// Elements per Tau1 challenge message. `jolt-transcript` caps one challenge
/// message at 16 MiB, and a mixed root draws millions of coefficients.
const BATCH_CHALLENGE_CHUNK: usize = 1 << 18;

/// Run the root's Tau1 proof of work, then draw its `count` independent batch
/// coefficients as consecutive challenge messages of at most
/// [`BATCH_CHALLENGE_CHUNK`] elements at that site.
pub fn binary_root_batch_challenges<E, G>(
    grinding: &mut G,
    count: usize,
) -> Result<Vec<E>, AkitaError>
where
    E: ChallengeCodec,
    G: GrindingReplay,
{
    let site = GrindingSite::Tau1Point { level: 0 };
    grinding.grind_query(site)?;
    let mut challenges = Vec::new();
    let result = challenges
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidInput("batch challenge allocation failed".into()))
        .and_then(|()| {
            let mut remaining = count;
            loop {
                let chunk = remaining.min(BATCH_CHALLENGE_CHUNK);
                challenges.extend(ext_challenges::<E, _>(
                    grinding.state_mut(),
                    site.site_id(0),
                    chunk,
                )?);
                remaining = remaining.saturating_sub(chunk);
                if remaining == 0 {
                    return Ok(());
                }
            }
        });
    if result.is_err() {
        grinding.poison();
    }
    result.map(|()| challenges)
}

/// P64 order-seven relation coordinates, including one independent E64 weight
/// per natural residual coefficient. P128 retains its separate frozen draw.
/// Both base coordinates are sampled canonically with at most 256 attempts.
/// The distinct public domain tag and diagnostic site identify this draw.
pub fn binary_root_p64_batch_challenges<F, E, G>(
    grinding: &mut G,
    count: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    if akita_params::field_modulus::<F>()? != u128::from(BINARY_SETUP_P64) || E::DEGREE != 2 {
        return Err(AkitaError::InvalidInput(
            "binary p64 batching requires the paired quadratic field".into(),
        ));
    }
    grinding.grind_query(GrindingSite::Tau1Point { level: 0 })?;
    let result = (|| {
        let channel = grinding.state_mut();
        let site = ProtocolSiteId {
            family: SITE_FAMILY_ROOT_STATEMENT,
            stage: 7,
            ..ProtocolSiteId::default()
        };
        akita_transcript::public_bytes(channel, site, b"binary p64 independent coefficient batch")?;
        p64_batch_from_blocks::<F, E>(count, || {
            channel.context(akita_transcript::ProtocolContextRecord::new(
                site.to_bytes(),
                akita_transcript::ProtocolMessageKind::Challenge as u32,
                0,
                0,
                32,
            ));
            super::challenge_block(channel)
        })
    })();
    if result.is_err() {
        grinding.poison();
    }
    result
}

fn p64_batch_from_blocks<F: Field, E: ExtField<F>>(
    count: usize,
    mut block: impl FnMut() -> Result<[u8; 32], AkitaError>,
) -> Result<Vec<E>, AkitaError> {
    checked::product([count, 2]).ok_or(AkitaError::InvalidProof)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut bytes = [0; 32];
    let mut offset = 32;
    for _ in 0..count {
        let mut coordinates = [F::zero(); 2];
        for coordinate in &mut coordinates {
            let mut accepted = None;
            for _ in 0..256 {
                if offset == 32 {
                    bytes = block()?;
                    offset = 0;
                }
                let candidate = u64::from_le_bytes(
                    bytes
                        .get(offset..offset + 8)
                        .ok_or(AkitaError::InvalidProof)?
                        .try_into()
                        .map_err(|_| AkitaError::InvalidProof)?,
                );
                offset += 8;
                if candidate < BINARY_SETUP_P64 {
                    accepted = Some(F::from_u64(candidate));
                    break;
                }
            }
            *coordinate = accepted.ok_or_else(|| {
                AkitaError::InvalidInput("binary p64 coordinate rejection exhausted".into())
            })?;
        }
        result.push(E::from_base_fn(|index| {
            if index == 0 {
                coordinates[0]
            } else {
                coordinates[1]
            }
        }));
    }
    Ok(result)
}

/// The fold challenge family of the binary root's single fold.
pub(crate) fn binary_root_family() -> Result<FoldChallengeFamily, AkitaError> {
    FoldChallengeFamily::new(
        FoldChallengeDrawDomain::EvaluationTrace,
        162,
        FoldChallengeDistribution::BinaryRoot,
        None,
    )
}

/// Draw the one root response family on the current transcript or its nonce preview.
pub fn binary_root_challenges(
    draw: &mut impl FoldDraw,
    columns: usize,
    polynomials: usize,
) -> Result<Challenges, AkitaError> {
    draw.draw_fold_challenges(&binary_root_family()?, columns, polynomials)?
        .ok_or_else(|| AkitaError::Internal("binary root family has no rejection".into()))
}

#[cfg(test)]
mod p64_tests {
    use super::*;
    use jolt_field::{Ext2, Prime64Offset59, Ring};
    type F = Prime64Offset59;
    type E = Ext2<F>;

    #[test]
    fn binary_source_p64_batch_rejects_noncanonical_candidates_and_keeps_both_coordinates() {
        let mut calls = 0;
        let values = p64_batch_from_blocks::<F, E>(2, || {
            calls += 1;
            Ok(if calls == 1 {
                [BINARY_SETUP_P64, u64::MAX, BINARY_SETUP_P64 - 1, 3]
            } else {
                [4, 5, 6, 7]
            }
            .map(u64::to_le_bytes)
            .concat()
            .try_into()
            .unwrap())
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(
            values,
            vec![
                E::from_base_fn(|i| F::from_u64([BINARY_SETUP_P64 - 1, 3][i])),
                E::from_base_fn(|i| F::from_u64([4, 5][i]))
            ]
        );
    }

    #[test]
    fn binary_source_p64_batch_caps_each_coordinate_at_256_attempts() {
        for coordinate in 0..2 {
            let mut calls = 0;
            let result = p64_batch_from_blocks::<F, E>(1, || {
                calls += 1;
                let mut words = [u64::MAX; 4];
                if calls == 1 && coordinate == 1 {
                    words[0] = 1;
                }
                Ok(words.map(u64::to_le_bytes).concat().try_into().unwrap())
            });
            assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
            assert_eq!(calls, (256usize + coordinate).div_ceil(4));
        }
        // The 256th attempt is accepted, rather than rejecting one attempt early.
        let mut calls = 0;
        let values = p64_batch_from_blocks::<F, E>(1, || {
            calls += 1;
            let words = match calls {
                64 => [u64::MAX, u64::MAX, u64::MAX, 7],
                65 => [9; 4],
                _ => [u64::MAX; 4],
            };
            Ok(words.map(u64::to_le_bytes).concat().try_into().unwrap())
        })
        .unwrap();
        assert_eq!(values, vec![E::from_base_fn(|i| F::from_u64([7, 9][i]))]);
    }
}

/// Bind the entire ordered grouped statement before any root challenge.
/// The legacy statement atom and its message order remain independent.
pub fn grouped_root_statement<F, E, S, C>(
    state: &mut S,
    binding: &super::GroupedTypedPrefix,
    claims: &crate::OpeningClaims<'_, E, C>,
    payloads: &[&[F]],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    use crate::opening_claims::GroupClaimKind;
    akita_params::binary_setup::BinaryFieldProfile::P64.check_field_pair(
        akita_params::field_modulus::<F>()?, E::DEGREE,
    )?;
    let plan = binding.plan();
    let signed = plan.ordinary().ok_or(AkitaError::InvalidProof)?;
    if claims.num_groups() != 8 || payloads.len() != 8 || plan.binary().len() != 4
        || signed.plans().len() != 4
    {
        return Err(AkitaError::InvalidProof);
    }
    let site = |group, stage, round| ProtocolSiteId {
        family: SITE_FAMILY_ROOT_STATEMENT, group, stage, round, ..ProtocolSiteId::default()
    };
    let mut header = b"AkitaGroupedRoot/p64/v1".to_vec();
    header.extend_from_slice(&8u64.to_le_bytes());
    header.extend_from_slice(&4u64.to_le_bytes());
    header.extend_from_slice(binding.identity());
    akita_transcript::public_bytes(state, site(0, 9, 0), &header)?;
    for (index, (group, payload)) in claims.groups().iter().zip(payloads).enumerate() {
        let group_index = u32::try_from(index).map_err(|_| AkitaError::InvalidProof)?;
        if let Some(local) = plan.binary().get(index) {
            let GroupClaimKind::Binary { manifest, claim, prime } = group.kind() else {
                return Err(AkitaError::InvalidProof);
            };
            if manifest.as_ref() != &local.source().manifest
                || prime != &local.source().prime || claim.is_some() != local.source().word
                || payload.len() != manifest.payload_coefficients()?
            {
                return Err(AkitaError::InvalidProof);
            }
            akita_transcript::public_bytes(
                state, site(group_index, 3, 0), &manifest.canonical_bytes(),
            )?;
            if let Some(claim) = claim {
                claim.validate(manifest.descriptor())?;
                akita_transcript::public_bytes(
                    state, site(group_index, 4, 0), &claim.canonical_bytes(),
                )?;
            }
            akita_transcript::public_bytes(
                state, site(group_index, 5, 0), &prime.canonical_bytes(),
            )?;
        } else {
            let source = signed.plans().get(index - 4).ok_or(AkitaError::InvalidProof)?;
            if group.kind() != &GroupClaimKind::Ordinary || group.num_points() != 2
                || payload.len()
                    != source.profile().outer_compression_plan()?.terminal_coefficients()
            {
                return Err(AkitaError::InvalidProof);
            }
            let mut contract = super::BinaryCommittedGroupManifest::carrier_profile_identity(
                source.profile(),
            )?.to_vec();
            contract.extend_from_slice(&source.source_width().ok_or(AkitaError::InvalidProof)?
                .bits().to_le_bytes());
            contract.extend_from_slice(&2u64.to_le_bytes());
            akita_transcript::public_bytes(state, site(group_index, 10, 0), &contract)?;
        }
        let mut lifted = Vec::new();
        lifted.try_reserve_exact(payload.len()).map_err(|_| AkitaError::InvalidProof)?;
        lifted.extend(payload.iter().copied().map(E::lift_base));
        akita_transcript::public_extensions::<F, E, _>(state, site(group_index, 11, 0), &lifted)?;
        if index >= 4 {
            for (point, reading) in group.point_evaluations().enumerate() {
                let point = u32::try_from(point).map_err(|_| AkitaError::InvalidProof)?;
                akita_transcript::public_extensions::<F, E, _>(
                    state, site(group_index, 12, point), reading.point,
                )?;
                akita_transcript::public_extensions::<F, E, _>(
                    state, site(group_index, 13, point), reading.evaluations,
                )?;
            }
        }
    }
    Ok(())
}
