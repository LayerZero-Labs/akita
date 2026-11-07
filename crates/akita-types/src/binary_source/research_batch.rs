//! Explicit, unaudited direct-setup four-plus-four research root.

use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::GroupCommitPhaseParams;
use jolt_field::Field;

use super::{
    BinaryAOperand, BinaryRootPublic, BinaryRootWitnessPlan, OrdinaryNativeRootBatch,
    OrdinaryNativeRootClaim, PrimeClaims, ResearchTerminalMode,
};
use crate::native_root_relation::NativeRootWeights;
use crate::opening_claims::GroupClaimKind;
use crate::root_repacking::RootCoefficientRepacking;
use crate::AkitaSetupDescriptor;

/// Complete local compilers and their aligned positions in one outgoing table.
#[derive(Clone, Debug)]
pub struct ResearchBatchPlan {
    binary: Vec<BinaryRootPublic>,
    ordinary: OrdinaryNativeRootBatch,
    ordinary_repacking: RootCoefficientRepacking,
    offsets: Vec<usize>,
    d_ranges: Vec<std::ops::Range<usize>>,
    fold_coordinates: Vec<usize>,
    physical_cells: usize,
    domain_len: usize,
    batch_count: usize,
    mode: ResearchTerminalMode,
}

/// Every public input of one binary local relation, after the all-D barrier.
pub struct ResearchBinaryRelation<'a, E: Field> {
    pub challenges: &'a Challenges,
    pub commitment: &'a [E],
    pub d_payload: &'a [E],
    pub matrix: BinaryAOperand<'a, E>,
}

impl ResearchBatchPlan {
    /// Fix the exact roster before allocating witnesses or drawing randomness.
    pub fn new(
        kinds: &[GroupClaimKind],
        profiles: &[GroupCommitPhaseParams],
        setup: &AkitaSetupDescriptor,
        destination_ring: usize,
    ) -> Result<Self, AkitaError> {
        if kinds.len() != 8 || profiles.len() != 8 {
            return Err(AkitaError::InvalidInput(
                "research root requires four plus four".into(),
            ));
        }
        let mut mode = None;
        let mut owners = std::collections::BTreeSet::new();
        let mut d_ranges = Vec::with_capacity(4);
        let mut d_end = 0;
        let mut fold_coordinates = Vec::with_capacity(4);
        let mut binary = Vec::with_capacity(4);
        let mut ordinary_sources = Vec::with_capacity(4);
        for (index, (kind, profile)) in kinds.iter().zip(profiles).enumerate() {
            match kind {
                GroupClaimKind::Binary {
                    manifest,
                    claim,
                    prime,
                } if index < 4 => {
                    manifest.validate_carrier(profile)?;
                    prime.validate(manifest.descriptor())?;
                    let selected = match prime.claim() {
                        Some(PrimeClaims::Base(_)) if prime.planes().len() == 1 => {
                            ResearchTerminalMode::BaseS11a
                        }
                        Some(PrimeClaims::Extension(points))
                            if points.len() == 2 && prime.planes().len() == 4 =>
                        {
                            ResearchTerminalMode::E64TwoPointS11a
                        }
                        _ => {
                            return Err(AkitaError::InvalidInput(
                                "research prime readings".into(),
                            ));
                        }
                    };
                    if mode.is_some_and(|mode| mode != selected) {
                        return Err(AkitaError::InvalidInput(
                            "research terminal mode mismatch".into(),
                        ));
                    }
                    mode = Some(selected);
                    if manifest.field_profile() != BinaryFieldProfile::P64
                        || manifest.setup_group()
                            != u64::try_from(index).map_err(|_| AkitaError::InvalidProof)?
                        || manifest.descriptor().polynomials().len() != 1
                        || (index != 0 && claim.is_some())
                        || prime.field_profile() != BinaryFieldProfile::P64
                    {
                        return Err(AkitaError::InvalidInput("research binary roster".into()));
                    }
                    if let Some(claim) = claim {
                        claim.validate(manifest.descriptor())?;
                    }
                    let d_len = checked::product([
                        usize::try_from(
                            manifest
                                .d_label(claim.is_some(), prime.planes().len())?
                                .rank,
                        )
                        .map_err(|_| AkitaError::InvalidProof)?,
                        manifest.b_ring_dimension(),
                    ])
                    .ok_or(AkitaError::InvalidProof)?;
                    let range = checked::range(d_end, d_len).ok_or(AkitaError::InvalidProof)?;
                    d_end = range.end;
                    d_ranges.push(range);
                    fold_coordinates.push(manifest.descriptor().index_map().columns());
                    for owner in manifest.descriptor().polynomials() {
                        if !owners.insert(*owner) {
                            return Err(AkitaError::InvalidInput(
                                "duplicate research owner".into(),
                            ));
                        }
                    }
                    let local = BinaryRootWitnessPlan::for_profile(
                        manifest.descriptor(),
                        manifest.a_rank(),
                        destination_ring,
                        BinaryFieldProfile::P64,
                        claim.is_some(),
                        prime.planes().len(),
                    )?;
                    binary.push(BinaryRootPublic::new(
                        manifest,
                        claim.is_some(),
                        prime,
                        setup,
                        local.repacking().domain_len(),
                        destination_ring,
                        None,
                    )?);
                }
                GroupClaimKind::ResearchOrdinary {
                    owners: ids,
                    width,
                    mode: selected,
                } if index >= 4 => {
                    let ordinary_index = index - 4;
                    let count = *[3, 3, 5, 1]
                        .get(ordinary_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    let bits = *[2, 9, 18, 33]
                        .get(ordinary_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    if mode != Some(*selected)
                        || ids.len() != count
                        || profile.group.num_polynomials() != count
                        || width.bits() != bits
                        || profile.inner.matrix.sis_modulus_profile()
                            != akita_params::SisModulusProfileId::Q64Offset59
                    {
                        return Err(AkitaError::InvalidInput("research ordinary roster".into()));
                    }
                    for owner in ids {
                        if !owners.insert(*owner) {
                            return Err(AkitaError::InvalidInput(
                                "duplicate research owner".into(),
                            ));
                        }
                    }
                    ordinary_sources.push((*profile, *width));
                }
                GroupClaimKind::Ordinary
                | GroupClaimKind::Binary { .. }
                | GroupClaimKind::ResearchOrdinary { .. } => {
                    return Err(AkitaError::InvalidInput("research group order".into()));
                }
            }
        }
        let ordinary = OrdinaryNativeRootBatch::new(&ordinary_sources, 0)?;
        let ordinary_repacking = RootCoefficientRepacking::new(
            ordinary
                .plans()
                .iter()
                .flat_map(|plan| plan.segments().iter().copied())
                .collect(),
            destination_ring,
        )?;
        let mut offsets = Vec::with_capacity(5);
        let mut physical_cells = 0;
        for len in binary
            .iter()
            .map(|local| local.plan().repacking().domain_len())
            .chain([ordinary_repacking.domain_len()])
        {
            let offset = checked::align_up(physical_cells, len).ok_or(AkitaError::InvalidProof)?;
            offsets.push(offset);
            physical_cells = checked::sum([offset, len]).ok_or(AkitaError::InvalidProof)?;
        }
        let domain_len = checked::ceil_log2(physical_cells)
            .and_then(checked::pow2)
            .ok_or(AkitaError::InvalidProof)?;
        let negative_count = ordinary.plans().iter().try_fold(0, |total, plan| {
            checked::sum([
                total,
                plan.negative_binary_addresses(&ordinary_repacking)?.len(),
            ])
            .ok_or(AkitaError::InvalidProof)
        })?;
        let mode = mode.ok_or(AkitaError::InvalidProof)?;
        let extra_readings = if mode == ResearchTerminalMode::E64TwoPointS11a {
            checked::sum(
                ordinary
                    .plans()
                    .iter()
                    .map(|plan| plan.profile().group.num_polynomials()),
            )
            .ok_or(AkitaError::InvalidProof)?
        } else {
            0
        };
        let batch_count = checked::sum(binary.iter().map(BinaryRootPublic::batch_count).chain([
            ordinary.linear_batch_count(),
            extra_readings,
            negative_count,
        ]))
        .ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            binary,
            ordinary,
            ordinary_repacking,
            offsets,
            d_ranges,
            fold_coordinates,
            physical_cells,
            domain_len,
            batch_count,
            mode,
        })
    }

    /// Bind the sole permitted word plane after replaying its field switch.
    pub fn bind_word_terminal(
        &mut self,
        terminal: Option<&super::BinaryTerminalClaim>,
    ) -> Result<(), AkitaError> {
        self.binary
            .first_mut()
            .ok_or(AkitaError::InvalidProof)?
            .bind_terminal(terminal)
    }

    pub fn d_ranges(&self) -> &[std::ops::Range<usize>] {
        &self.d_ranges
    }
    pub fn fold_coordinates(&self) -> &[usize] {
        &self.fold_coordinates
    }
    pub fn binary(&self) -> &[BinaryRootPublic] {
        &self.binary
    }
    pub fn ordinary(&self) -> &OrdinaryNativeRootBatch {
        &self.ordinary
    }
    pub fn ordinary_repacking(&self) -> &RootCoefficientRepacking {
        &self.ordinary_repacking
    }
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }
    pub fn physical_cells(&self) -> usize {
        self.physical_cells
    }
    pub fn domain_len(&self) -> usize {
        self.domain_len
    }
    pub fn mode(&self) -> ResearchTerminalMode {
        self.mode
    }
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// One independent coefficient per original row, including every K equation.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn relation<E: Field>(
        &self,
        setup: &AkitaSetupDescriptor,
        binary: &[ResearchBinaryRelation<'_, E>],
        ordinary: &[OrdinaryNativeRootClaim<'_, E>],
        alpha: E,
        batches: &[E],
    ) -> Result<(NativeRootWeights<E>, Vec<(usize, E)>), AkitaError> {
        let ordinary_count = if self.mode == ResearchTerminalMode::E64TwoPointS11a {
            8
        } else {
            4
        };
        if binary.len() != 4
            || ordinary.len() != ordinary_count
            || batches.len() != self.batch_count
        {
            return Err(AkitaError::InvalidProof);
        }
        let mut components = Vec::with_capacity(5);
        let mut cursor = 0;
        for ((local, input), &offset) in self.binary.iter().zip(binary).zip(&self.offsets) {
            let range =
                checked::range(cursor, local.batch_count()).ok_or(AkitaError::InvalidProof)?;
            cursor = range.end;
            let (weights, negative) = local.relation(
                input.challenges,
                alpha,
                batches.get(range).ok_or(AkitaError::InvalidProof)?,
                &[],
                &[],
                input.commitment,
                input.d_payload,
                None,
                Some(input.matrix),
            )?;
            if !negative.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            components.push((offset, weights));
        }
        let range = checked::range(cursor, self.ordinary.linear_batch_count())
            .ok_or(AkitaError::InvalidProof)?;
        cursor = range.end;
        let ordinary_offset = *self.offsets.get(4).ok_or(AkitaError::InvalidProof)?;
        let mut ordinary_weights = vec![self.ordinary.linear_weights(
            &self.ordinary_repacking,
            setup,
            ordinary.get(..4).ok_or(AkitaError::InvalidProof)?,
            alpha,
            batches.get(range).ok_or(AkitaError::InvalidProof)?,
        )?];
        for (local, reading) in self.ordinary.plans().iter().zip(ordinary.iter().skip(4)) {
            if reading.profile != local.profile() || Some(reading.width) != local.source_width() {
                return Err(AkitaError::InvalidProof);
            }
            let range = checked::range(cursor, local.profile().group.num_polynomials())
                .ok_or(AkitaError::InvalidProof)?;
            cursor = range.end;
            ordinary_weights.push(local.evaluation_weights(
                &self.ordinary_repacking,
                reading.point,
                reading.evaluations,
                batches.get(range).ok_or(AkitaError::InvalidProof)?,
            )?);
        }
        components.push((
            ordinary_offset,
            NativeRootWeights::merge(
                self.ordinary_repacking.domain_len(),
                ordinary_weights,
                E::zero(),
            )?,
        ));
        let mut negative = Vec::new();
        for plan in self.ordinary.plans() {
            for address in plan.negative_binary_addresses(&self.ordinary_repacking)? {
                negative
                    .try_reserve(1)
                    .map_err(|_| AkitaError::InvalidProof)?;
                negative.push((
                    checked::sum([ordinary_offset, address]).ok_or(AkitaError::InvalidProof)?,
                    *batches.get(cursor).ok_or(AkitaError::InvalidProof)?,
                ));
                cursor = checked::sum([cursor, 1]).ok_or(AkitaError::InvalidProof)?;
            }
        }
        if cursor != batches.len() {
            return Err(AkitaError::InvalidProof);
        }
        Ok((
            NativeRootWeights::concatenate(self.domain_len, components)?,
            negative,
        ))
    }
}

/// All handles belong to the same admitted proof session and frozen roster.
pub struct ResearchBatchWitnessRequest<H> {
    pub kinds: Vec<GroupClaimKind>,
    pub plan: ResearchBatchPlan,
    pub handles: Vec<H>,
}

/// One joint candidate: challenges are drawn for all four groups before probing.
pub struct ResearchBatchResponse {
    pub challenges: Vec<Challenges>,
}
