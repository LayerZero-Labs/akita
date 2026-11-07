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
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
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
    let phase = tracing::info_span!("binary_p64_tau1_grind").entered();
    grinding.grind_query(GrindingSite::Tau1Point { level: 0 })?;
    drop(phase);
    let _phase = tracing::info_span!("binary_p64_batch_fill", count).entered();
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
    // Buffer accepted coordinates, not transcript state. Indexed extension
    // writes each output exactly once, in draw order, on disjoint worker ranges.
    // This boundary is unrelated to the fixed 32-byte transcript messages;
    // unused bytes and coordinate rejection continue across every buffer fill.
    const COORDINATE_BUFFER: usize = 1 << 14;
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(count.min(COORDINATE_BUFFER))
        .map_err(|_| AkitaError::InvalidProof)?;
    for index in 0..count {
        let mut coordinates = [0u64; 2];
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
                    accepted = Some(candidate);
                    break;
                }
            }
            *coordinate = accepted.ok_or_else(|| {
                AkitaError::InvalidInput("binary p64 coordinate rejection exhausted".into())
            })?;
        }
        pending.push(coordinates);
        if pending.len() == COORDINATE_BUFFER || index + 1 == count {
            let convert = |coordinates: &[u64; 2]| {
                let [first, second] = *coordinates;
                E::from_base_fn(|index| F::from_u64(if index == 0 { first } else { second }))
            };
            #[cfg(feature = "parallel")]
            result.par_extend(pending.par_iter().map(convert));
            #[cfg(not(feature = "parallel"))]
            result.extend(pending.iter().map(convert));
            pending.clear();
        }
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
        // Three accepted words per block make a conversion-buffer boundary
        // land inside a transcript block, with a rejection immediately after it.
        for count in [0, 1, 16_383, 16_384, 16_385, 32_769] {
            let mut calls = 0u64;
            let values = p64_batch_from_blocks::<F, E>(count, || {
                let start = 4 * calls;
                calls += 1;
                Ok([u64::MAX, start, start + 1, start + 2]
                    .map(u64::to_le_bytes)
                    .concat()
                    .try_into()
                    .unwrap())
            })
            .unwrap();
            let expected = (0..count)
                .map(|element| {
                    E::from_base_fn(|coordinate| {
                        let index = u64::try_from(2 * element + coordinate).unwrap();
                        F::from_u64(4 * (index / 3) + index % 3)
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(values, expected);
            assert_eq!(calls, u64::try_from(2 * count).unwrap().div_ceil(3));
        }
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
