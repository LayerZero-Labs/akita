//! Canonical placement and binding of independently defined local setup tables.

use akita_error::{checked, AkitaError};
use akita_params::SetupPrefixSlotId;
use akita_serialization::{AkitaSerialize, Valid};
use jolt_field::Field;
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;

use super::{BinaryCommittedGroupManifest, BinaryTypedPrefix};
use crate::root_repacking::RootCoefficientRepacking;

const MAX_BLOCKS: usize = 64;
// Transient verifier prefix contexts admit at most 2^18 bytes. Reject the
// complete encoded context before any local setup scan, not during installation.
const MAX_IDENTITY_BYTES: usize = 1 << 18;
const DOMAIN: &[u8] = b"akita/binary-source/multigroup/composite-prefix/v1";

/// Canonical root binding shared by setup preparation and public weight compilation.
/// Each list maps one occurrence's local segment order to the global table;
/// together the lists must cover every segment exactly once, including K.
/// `prime_owners` binds each group's increasing selected-owner list (empty for
/// ordinary or word-only groups). Statement admission checks each list against
/// its authenticated descriptor and selected plane count.
pub fn composite_prefix_root_identity(
    root: &RootCoefficientRepacking,
    group_segments: &[&[usize]],
    prime_owners: &[&[usize]],
) -> Result<Vec<u8>, AkitaError> {
    if root.segments().len() > 448
        || root.domain_len() as u128 > (1u128 << 32)
        || group_segments.is_empty()
        || group_segments.len() > MAX_BLOCKS
        || prime_owners.len() != group_segments.len()
        || checked::sum(prime_owners.iter().map(|owners| owners.len()))
            .is_none_or(|count| count > 4096)
        || checked::sum(group_segments.iter().map(|map| map.len())) != Some(root.segments().len())
    {
        return Err(AkitaError::InvalidInput(
            "composite root layout exceeds caps".into(),
        ));
    }
    // The statement constructor additionally checks these owner selections
    // against each authenticated descriptor and its selected plane count.
    for owners in prime_owners {
        if owners.iter().any(|&owner| owner >= 4096)
            || owners.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(AkitaError::InvalidInput(
                "noncanonical composite prime owner subset".into(),
            ));
        }
    }
    let mut seen = [false; 448];
    for map in group_segments {
        if map.is_empty() {
            return Err(AkitaError::InvalidInput(
                "empty composite root segment map".into(),
            ));
        }
        for &index in *map {
            let present = seen
                .get_mut(index)
                .filter(|_| index < root.segments().len())
                .ok_or_else(|| AkitaError::InvalidInput("composite root segment index".into()))?;
            if *present {
                return Err(AkitaError::InvalidInput(
                    "duplicate composite root segment".into(),
                ));
            }
            *present = true;
        }
    }
    let mut identity = b"akita/binary-source/multigroup/root-address-map/v1".to_vec();
    append_word(&mut identity, root.destination_ring_dimension())?;
    append_word(&mut identity, root.domain_len())?;
    append_word(&mut identity, root.segments().len())?;
    for (index, segment) in root.segments().iter().enumerate() {
        let geometry = segment.geometry;
        let kind = match geometry.polynomial().kind() {
            akita_params::RelationPolynomialKind::Negacyclic => 0,
            #[cfg(feature = "binary")]
            akita_params::RelationPolynomialKind::Trinomial(akita_params::TrinomialSign::Plus) => 1,
            #[cfg(feature = "binary")]
            akita_params::RelationPolynomialKind::Trinomial(akita_params::TrinomialSign::Minus) => {
                2
            }
        };
        let range = root.segment_range(index)?;
        for value in [
            kind,
            geometry.polynomial_modulus_dimension(),
            geometry.padded_coefficient_count(),
            geometry.coordinate_plane_count(),
            segment.elements,
            segment.digits,
            range.start,
            range.end,
        ] {
            append_word(&mut identity, value)?;
        }
    }
    append_word(&mut identity, group_segments.len())?;
    for (map, owners) in group_segments.iter().zip(prime_owners) {
        append_word(&mut identity, map.len())?;
        for &index in *map {
            append_word(&mut identity, index)?;
        }
        append_word(&mut identity, owners.len())?;
        for &owner in *owners {
            append_word(&mut identity, owner)?;
        }
    }
    Ok(identity)
}

/// Canonical occurrence kind. Binary blocks precede opaque ordinary blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositePrefixKind {
    Binary,
    Ordinary,
}

/// An authenticated local table definition, independent of its global placement.
/// Ordinary setup role construction is deliberately outside this layout module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompositePrefixBlock {
    pub kind: CompositePrefixKind,
    /// Global occurrence index, including preceding binary occurrences.
    pub index: usize,
    pub setup_identity: [u8; 32],
    pub local_identity: Vec<u8>,
    pub domain_len: usize,
}

impl CompositePrefixBlock {
    /// Bind the original seed and manifest without changing the local labels or identity.
    pub fn binary(
        index: usize,
        seed: &[u8; 32],
        manifest: &BinaryCommittedGroupManifest,
        local: &BinaryTypedPrefix,
        local_slot: &SetupPrefixSlotId,
    ) -> Result<Self, AkitaError> {
        local.validate_prefix(local.domain_len)?;
        local_slot
            .check()
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        if local_slot.n_prefix()? != local.domain_len {
            return Err(AkitaError::InvalidInput(
                "local prefix slot does not match its padded domain".into(),
            ));
        }
        let local_identity = local.identity(manifest, local_slot)?;
        let mut hash = Shake256::default();
        hash.update(b"akita/binary-source/multigroup/local-setup/v1");
        hash.update(seed);
        hash.update(&manifest.canonical_bytes());
        let mut setup_identity = [0; 32];
        hash.finalize_xof().read(&mut setup_identity);
        Ok(Self {
            kind: CompositePrefixKind::Binary,
            index,
            setup_identity,
            local_identity,
            domain_len: local.domain_len,
        })
    }
}

/// One disjoint, aligned local domain inside the composed table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompositePrefixPlacement {
    pub block: CompositePrefixBlock,
    pub offset: usize,
}

/// The single source of truth for placements, selector dimensions and successor slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompositePrefixLayout {
    placements: Vec<CompositePrefixPlacement>,
    domain_len: usize,
    slot: SetupPrefixSlotId,
    identity: Vec<u8>,
}

impl CompositePrefixLayout {
    /// Place local padded domains in occurrence order, checking the complete plan
    /// before copying identities or allocating the composed coefficient table.
    /// `root_layout_identity` must canonically bind the full root address map,
    /// including every translated segment and the destination ring dimension.
    pub fn new(
        blocks: &[CompositePrefixBlock],
        root_layout_identity: &[u8],
        slot: &SetupPrefixSlotId,
    ) -> Result<Self, AkitaError> {
        if root_layout_identity.is_empty()
            || blocks.is_empty()
            || blocks.len() > MAX_BLOCKS
            || blocks.first().map(|b| b.kind) != Some(CompositePrefixKind::Binary)
        {
            return Err(AkitaError::InvalidInput(
                "composite prefix requires one to 64 blocks starting with binary".into(),
            ));
        }
        slot.check()
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        let mut end = 0usize;
        let mut ordinary = false;
        let mut identity_bytes = root_layout_identity.len();
        for (index, block) in blocks.iter().enumerate() {
            ordinary |= block.kind == CompositePrefixKind::Ordinary;
            identity_bytes = checked::sum([identity_bytes, block.local_identity.len()])
                .ok_or_else(|| {
                    AkitaError::InvalidInput("prefix identity length overflow".into())
                })?;
            if block.index != index
                || (ordinary && block.kind == CompositePrefixKind::Binary)
                || !block.domain_len.is_power_of_two()
                || block.domain_len as u128 > (1u128 << 32)
                || block.local_identity.is_empty()
                || identity_bytes > MAX_IDENTITY_BYTES
            {
                return Err(AkitaError::InvalidInput(
                    "noncanonical composite prefix block or identity cap exceeded".into(),
                ));
            }
            end = checked::align_up(end, block.domain_len)
                .and_then(|offset| offset.checked_add(block.domain_len))
                .ok_or_else(|| AkitaError::InvalidInput("prefix placement overflow".into()))?;
        }
        let domain_len = checked::ceil_log2(end)
            .and_then(checked::pow2)
            .filter(|&len| len as u128 <= (1u128 << 32))
            .ok_or_else(|| {
                AkitaError::InvalidInput("composite prefix domain exceeds cap".into())
            })?;
        if slot.n_prefix()? != domain_len {
            return Err(AkitaError::InvalidInput(
                "composite prefix slot does not match its exact padded domain".into(),
            ));
        }
        let mut identity = DOMAIN.to_vec();
        append_blob(&mut identity, root_layout_identity)?;
        append_word(&mut identity, blocks.len())?;
        let mut placements = Vec::with_capacity(blocks.len());
        let mut end = 0;
        for block in blocks {
            let offset = checked::align_up(end, block.domain_len)
                .ok_or_else(|| AkitaError::InvalidInput("prefix placement overflow".into()))?;
            identity.push(match block.kind {
                CompositePrefixKind::Binary => 0,
                CompositePrefixKind::Ordinary => 1,
            });
            append_word(&mut identity, block.index)?;
            identity.extend_from_slice(&block.setup_identity);
            append_blob(&mut identity, &block.local_identity)?;
            append_word(&mut identity, offset)?;
            append_word(&mut identity, block.domain_len)?;
            end = offset
                .checked_add(block.domain_len)
                .ok_or_else(|| AkitaError::InvalidInput("prefix placement overflow".into()))?;
            placements.push(CompositePrefixPlacement {
                block: block.clone(),
                offset,
            });
        }
        append_word(&mut identity, domain_len)?;
        slot.serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidInput("invalid composite prefix slot".into()))?;
        if identity.len() > MAX_IDENTITY_BYTES {
            return Err(AkitaError::InvalidInput(
                "composite prefix context exceeds verifier cap".into(),
            ));
        }
        Ok(Self {
            placements,
            domain_len,
            slot: slot.clone(),
            identity,
        })
    }

    pub fn placements(&self) -> &[CompositePrefixPlacement] {
        &self.placements
    }

    pub fn domain_len(&self) -> usize {
        self.domain_len
    }

    pub fn slot(&self) -> &SetupPrefixSlotId {
        &self.slot
    }

    /// New-version composite binding; each local identity remains unchanged inside it.
    pub fn identity(&self) -> &[u8] {
        &self.identity
    }

    /// High-bit equality selector, including leading zero coordinates.
    /// Coordinates are in low-bit-first table address order.
    pub fn selector<E: Field>(&self, index: usize, point: &[E]) -> Result<E, AkitaError> {
        if point.len() != self.domain_len.trailing_zeros() as usize {
            return Err(AkitaError::InvalidInput(
                "composite prefix point dimension".into(),
            ));
        }
        let placement = self.placements.get(index).ok_or_else(|| {
            AkitaError::InvalidInput("composite prefix block index out of range".into())
        })?;
        let bits = placement.block.domain_len.trailing_zeros() as usize;
        let address = placement.offset / placement.block.domain_len;
        Ok(point[bits..]
            .iter()
            .enumerate()
            .fold(E::one(), |factor, (bit, &z)| {
                factor
                    * if (address >> bit) & 1 == 1 {
                        z
                    } else {
                        E::one() - z
                    }
            }))
    }

    /// Sum local weights at their low coordinates with no additional group multiplier.
    pub fn evaluate<E: Field>(
        &self,
        point: &[E],
        local_evaluations: &[E],
    ) -> Result<E, AkitaError> {
        if local_evaluations.len() != self.placements.len() {
            return Err(AkitaError::InvalidInput(
                "composite prefix weight count".into(),
            ));
        }
        local_evaluations
            .iter()
            .enumerate()
            .try_fold(E::zero(), |sum, (index, &value)| {
                Ok(sum + self.selector(index, point)? * value)
            })
    }

    /// Construct the one table. The trusted caller supplies a cell budget before allocation.
    pub fn assemble<F: Field>(
        &self,
        local_tables: &[&[F]],
        max_cells: usize,
    ) -> Result<Vec<F>, AkitaError> {
        if self.domain_len > max_cells
            || local_tables.len() != self.placements.len()
            || local_tables
                .iter()
                .zip(&self.placements)
                .any(|(table, placement)| table.len() != placement.block.domain_len)
        {
            return Err(AkitaError::InvalidInput(
                "composite prefix table count, domain or trusted budget mismatch".into(),
            ));
        }
        checked::product([self.domain_len, std::mem::size_of::<F>()])
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                AkitaError::InvalidInput("composite prefix allocation overflow".into())
            })?;
        let mut table = Vec::new();
        table
            .try_reserve_exact(self.domain_len)
            .map_err(|_| AkitaError::InvalidInput("composite prefix allocation failed".into()))?;
        table.resize(self.domain_len, F::zero());
        for (local, placement) in local_tables.iter().zip(&self.placements) {
            let range = checked::range(placement.offset, placement.block.domain_len)
                .ok_or_else(|| AkitaError::InvalidInput("prefix placement overflow".into()))?;
            table
                .get_mut(range)
                .ok_or(AkitaError::InvalidProof)?
                .copy_from_slice(local);
        }
        Ok(table)
    }
}

fn append_word(bytes: &mut Vec<u8>, value: usize) -> Result<(), AkitaError> {
    bytes.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| AkitaError::InvalidInput("prefix identity word overflow".into()))?
            .to_le_bytes(),
    );
    Ok(())
}

fn append_blob(bytes: &mut Vec<u8>, blob: &[u8]) -> Result<(), AkitaError> {
    append_word(bytes, blob.len())?;
    bytes.extend_from_slice(blob);
    Ok(())
}

#[cfg(test)]
#[path = "composite_prefix_tests.rs"]
pub(crate) mod tests;
