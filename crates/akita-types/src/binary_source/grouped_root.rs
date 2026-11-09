//! Ordered capacity geometry for a staged four-group p64 session.

use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::ordinary_source::OrdinarySourceWidth;
use akita_params::GroupCommitPhaseParams;
use akita_serialization::AkitaSerialize;
use std::collections::BTreeSet;

use super::{
    BinaryCommittedGroupManifest, BinaryRootWitnessPlan, OrdinaryNativeRootBatch,
    PrimePlaneManifest,
};
use crate::root_repacking::RootCoefficientRepacking;

/// Caller-selected source contract. Construction revalidates every field.
#[derive(Clone, Debug)]
pub struct GroupedBinarySource {
    pub manifest: BinaryCommittedGroupManifest,
    pub carrier: GroupCommitPhaseParams,
    pub setup_group: u64,
    pub word: bool,
    pub prime: PrimePlaneManifest,
}

/// One unchanged local compiler result and its translation to the common table.
#[derive(Clone, Debug)]
pub struct GroupedBinaryPlan {
    source: GroupedBinarySource,
    local: BinaryRootWitnessPlan,
    first_segment: usize,
    coefficient_offset: usize,
}

impl GroupedBinaryPlan {
    pub fn source(&self) -> &GroupedBinarySource {
        &self.source
    }

    pub fn local(&self) -> &BinaryRootWitnessPlan {
        &self.local
    }

    pub fn first_segment(&self) -> usize {
        self.first_segment
    }

    pub fn coefficient_offset(&self) -> usize {
        self.coefficient_offset
    }
}

/// Ordered geometry consumed by the explicitly selected grouped opening drivers.
#[derive(Clone, Debug)]
pub struct GroupedNativeRootPlan {
    binary: Vec<GroupedBinaryPlan>,
    ordinary: Option<OrdinaryNativeRootBatch>,
    repacking: RootCoefficientRepacking,
    identity: Vec<u8>,
}

impl GroupedNativeRootPlan {
    /// Preserve source order, including each local quotient, then signed preimages.
    /// Prime manifests retain point order and both coordinates even when zero.
    pub fn new(
        binary: &[GroupedBinarySource],
        ordinary: &[(GroupCommitPhaseParams, OrdinarySourceWidth)],
        destination_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if binary.is_empty() {
            return Err(AkitaError::InvalidInput("empty grouped binary root".into()));
        }
        let mut owners = BTreeSet::new();
        let mut setup_groups = BTreeSet::new();
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(binary.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        let mut segments = Vec::new();
        let mut identity = b"akita/grouped-root/p64/capacity/v1".to_vec();
        binary
            .len()
            .serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidProof)?;
        for source in binary {
            let manifest = &source.manifest;
            manifest.validate_carrier(&source.carrier)?;
            source.carrier.validate_frozen_precommit(64)?;
            source.prime.validate(manifest.descriptor())?;
            if manifest.field_profile() != BinaryFieldProfile::P64
                || source.prime.field_profile() != BinaryFieldProfile::P64
                || source.setup_group != manifest.setup_group()
                || !setup_groups.insert(source.setup_group)
                || manifest
                    .descriptor()
                    .polynomials()
                    .iter()
                    .any(|owner| !owners.insert(*owner))
            {
                return Err(AkitaError::InvalidInput(
                    "grouped source field, setup group or owner mismatch".into(),
                ));
            }
            // This existing validator requires the actual D width's table entry.
            manifest.d_label(source.word, source.prime.planes().len())?;
            let local = BinaryRootWitnessPlan::for_profile(
                manifest.descriptor(),
                manifest.a_rank(),
                destination_ring_dimension,
                BinaryFieldProfile::P64,
                source.word,
                source.prime.planes().len(),
            )?;
            let first_segment = segments.len();
            segments
                .try_reserve_exact(local.repacking().segments().len())
                .map_err(|_| AkitaError::InvalidProof)?;
            segments.extend_from_slice(local.repacking().segments());
            manifest
                .canonical_bytes()
                .serialize_compressed(&mut identity)
                .map_err(|_| AkitaError::InvalidProof)?;
            identity.push(u8::from(source.word));
            source
                .prime
                .canonical_bytes()
                .serialize_compressed(&mut identity)
                .map_err(|_| AkitaError::InvalidProof)?;
            plans.push(GroupedBinaryPlan {
                source: source.clone(),
                local,
                first_segment,
                coefficient_offset: 0,
            });
        }
        ordinary
            .len()
            .serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidProof)?;
        let ordinary_batch = if ordinary.is_empty() {
            None
        } else {
            Some(OrdinaryNativeRootBatch::new(ordinary, segments.len())?)
        };
        if let Some(batch) = &ordinary_batch {
            for plan in batch.plans() {
                if plan.profile().inner.matrix.sis_modulus_profile().modulus()
                    != BinaryFieldProfile::P64.modulus()
                {
                    return Err(AkitaError::InvalidInput(
                        "grouped ordinary field mismatch".into(),
                    ));
                }
                segments
                    .try_reserve_exact(plan.segments().len())
                    .map_err(|_| AkitaError::InvalidProof)?;
                segments.extend_from_slice(plan.segments());
                identity.extend(BinaryCommittedGroupManifest::carrier_profile_identity(
                    plan.profile(),
                )?);
                identity.extend_from_slice(
                    &plan
                        .source_width()
                        .ok_or(AkitaError::InvalidProof)?
                        .bits()
                        .to_le_bytes(),
                );
            }
        }
        let repacking = RootCoefficientRepacking::new(segments, destination_ring_dimension)?;
        for plan in &mut plans {
            plan.coefficient_offset = repacking.segment_range(plan.first_segment)?.start;
        }
        for value in [destination_ring_dimension, repacking.domain_len()] {
            identity.extend_from_slice(
                &u64::try_from(value)
                    .map_err(|_| AkitaError::InvalidProof)?
                    .to_le_bytes(),
            );
        }
        Ok(Self {
            binary: plans,
            ordinary: ordinary_batch,
            repacking,
            identity,
        })
    }

    /// Ordered slices of the single concatenated root D message.
    pub fn d_payload_ranges(&self) -> Result<Vec<std::ops::Range<usize>>, AkitaError> {
        let mut ranges = Vec::new();
        let mut end = 0;
        for binary in &self.binary {
            let source = binary.source();
            let label = source.manifest.d_label(source.word, source.prime.planes().len())?;
            let len = checked::product([label.rank as usize, source.manifest.b_ring_dimension()])
                .ok_or(AkitaError::InvalidProof)?;
            let range = checked::range(end, len).ok_or(AkitaError::InvalidProof)?;
            end = range.end;
            ranges.push(range);
        }
        Ok(ranges)
    }

    pub fn binary(&self) -> &[GroupedBinaryPlan] {
        &self.binary
    }

    pub fn ordinary(&self) -> Option<&OrdinaryNativeRootBatch> {
        self.ordinary.as_ref()
    }

    pub fn repacking(&self) -> &RootCoefficientRepacking {
        &self.repacking
    }

    pub fn identity(&self) -> &[u8] {
        &self.identity
    }

    /// Compare a caller's complete layout; no omitted or reordered segment is accepted.
    pub fn validate_layout(&self, supplied: &RootCoefficientRepacking) -> Result<(), AkitaError> {
        if supplied != &self.repacking {
            return Err(AkitaError::InvalidInput(
                "grouped root layout mismatch".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "grouped_root_tests.rs"]
mod tests;

impl GroupedNativeRootPlan {
    /// Reconstruct the plan from ordered public claims and trusted source contracts.
    pub fn from_claims<E: Clone, C>(
        claims: &crate::OpeningClaims<'_, E, C>,
        profiles: &akita_params::CommittedGroupBatchProfile,
        widths: &[OrdinarySourceWidth; 4],
        destination: usize,
    ) -> Result<Self, AkitaError> {
        use crate::opening_claims::GroupClaimKind;
        if claims.num_groups() != 8 || profiles.precommitteds.len() != 7 {
            return Err(AkitaError::InvalidProof);
        }
        let mut binary = Vec::new();
        for (index, group) in claims.groups().iter().take(4).enumerate() {
            let GroupClaimKind::Binary { manifest, claim, prime } = group.kind() else {
                return Err(AkitaError::InvalidProof);
            };
            if manifest.setup_group() != index as u64 || prime.claims().len() != 2 {
                return Err(AkitaError::InvalidProof);
            }
            binary.push(GroupedBinarySource {
                manifest: manifest.as_ref().clone(),
                carrier: *profiles.precommitteds.get(index).ok_or(AkitaError::InvalidProof)?,
                setup_group: index as u64, word: claim.is_some(), prime: prime.clone(),
            });
        }
        for group in claims.groups().iter().skip(4) {
            if group.kind() != &GroupClaimKind::Ordinary || group.num_points() != 2 {
                return Err(AkitaError::InvalidProof);
            }
        }
        let signed = profiles.precommitteds.iter().skip(4)
            .chain(std::iter::once(&profiles.final_group)).copied()
            .zip(widths.iter().copied()).collect::<Vec<_>>();
        Self::new(&binary, &signed, destination)
    }
}
