//! Bounded setup-prefix preprocessing behind the ordinary CPU backend API.
use super::CachedSetupPrefix;
use crate::commitment::{PortableCommitmentHandle, SetupPrefixProverRegistry, SetupPrefixSlot};
use crate::CpuBackend;
use akita_error::AkitaError;
use akita_params::{GroupCommitPhaseParams, RingRelationMode, SetupPrefixSlotId};
use akita_serialization::Valid;
use akita_types::{
    derive_public_matrix_range, AkitaExpandedSetup, AkitaSetupDescriptor, RingVec,
    SetupPrefixPublicCommitment,
};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::{
    cmp::Reverse,
    ops::Range,
    sync::{Arc, OnceLock},
};

mod products;
use products::products;

#[derive(Clone, Copy)]
pub(super) enum PrefixSource<'a, F: Field> {
    Expanded(&'a AkitaExpandedSetup<F>),
    Seed(&'a AkitaSetupDescriptor),
}

impl<F: Field + CanonicalEncoding> PrefixSource<'_, F> {
    fn descriptor(&self) -> &AkitaSetupDescriptor {
        match self {
            Self::Expanded(e) => e.descriptor(),
            Self::Seed(d) => d,
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Expanded(e) => e.shared_matrix().num_field_elements(),
            Self::Seed(d) => d.num_field_elements,
        }
    }

    fn read(&self, range: Range<usize>) -> Result<Vec<F>, AkitaError> {
        if range.start > range.end || range.end > self.capacity() {
            return Err(AkitaError::InvalidSetup(
                "setup range exceeds capacity".into(),
            ));
        }
        match self {
            Self::Expanded(e) => e
                .shared_matrix()
                .as_field_slice()
                .get(range)
                .map(<[F]>::to_vec)
                .ok_or_else(|| AkitaError::InvalidSetup("missing setup range".into())),
            Self::Seed(d) => {
                let mut fields = vec![F::zero(); range.len()];
                derive_public_matrix_range(range.start, &mut fields, &d.setup_seed)?;
                Ok(fields)
            }
        }
    }
}

impl<F: Field + CanonicalEncoding + 'static, E> CpuBackend<F, E> {
    /// Generate portable prefix commitments directly from the public seed.
    /// Only bounded source/matrix tiles, shared inner rows and compression setup
    /// are materialized. The returned artifacts have the ordinary export format.
    pub fn export_setup_prefixes_from_seed(
        descriptor: &AkitaSetupDescriptor,
        ids: &[SetupPrefixSlotId],
    ) -> Result<SetupPrefixProverRegistry<F>, AkitaError>
    where
        F: Valid + Unreduced + WithCommitAccumulator,
    {
        descriptor
            .check()
            .map_err(|e| AkitaError::InvalidSetup(e.to_string()))?;
        let source = PrefixSource::<F>::Seed(descriptor);
        let mut fields = 1;
        for id in ids {
            crate::setup::setup_prefix::validate_setup_prefix_commitment(source.capacity(), id)?;
            fields = fields.max(
                id.commitment_profile
                    .outer_compression_plan()?
                    .max_setup_field_elements()?,
            );
        }
        if fields > source.capacity() {
            return Err(AkitaError::InvalidSetup(
                "setup range exceeds capacity".into(),
            ));
        }
        let expanded = AkitaExpandedSetup::derive(AkitaSetupDescriptor {
            max_num_vars: 1,
            max_num_batched_polys: 1,
            num_field_elements: fields,
            setup_seed: descriptor.setup_seed.clone(),
        })?;
        Self::new(Arc::new(expanded))?.export_prefixes_from(source, ids)
    }

    pub(super) fn export_prefixes_from(
        &self,
        source: PrefixSource<'_, F>,
        ids: &[SetupPrefixSlotId],
    ) -> Result<SetupPrefixProverRegistry<F>, AkitaError>
    where
        F: Valid + Unreduced + WithCommitAccumulator,
    {
        for id in ids {
            crate::setup::setup_prefix::validate_setup_prefix_commitment(source.capacity(), id)?;
        }
        let mut ordered = ids.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|id| {
            let p = id.commitment_profile;
            (
                p.inner.matrix.ring_dimension(),
                p.inner.digits.num_digits,
                p.inner.digits.log_basis,
                p.blocks.positions_per_block,
                Reverse(p.group.num_vars()),
                Reverse(p.inner.matrix.output_rank()),
            )
        });
        let mut artifacts = SetupPrefixProverRegistry::new(source.descriptor().setup_seed.clone());
        for id in ordered {
            let mut artifact = self.prefix_material_from(source, id)?.artifact.clone();
            artifact.commitment.rows = artifact
                .commitment
                .rows
                .into_iter()
                .map(RingVec::into_compact)
                .collect();
            artifacts.insert(artifact)?;
        }
        artifacts.mark_backend_validated();
        Ok(artifacts)
    }

    pub(super) fn prefix_material_from(
        &self,
        source: PrefixSource<'_, F>,
        id: &SetupPrefixSlotId,
    ) -> Result<Arc<CachedSetupPrefix<F>>, AkitaError>
    where
        F: Valid + Unreduced + WithCommitAccumulator,
    {
        crate::setup::setup_prefix::validate_setup_prefix_commitment(source.capacity(), id)?;
        self.setup_prefix_cache.get_or_build(id, || {
            let cached = self.setup_prefix_cache.snapshot()?;
            let artifact = match cached
                .iter()
                .find(|c| c.artifact.id.commitment_profile == id.commitment_profile)
            {
                Some(c) => SetupPrefixSlot {
                    id: id.clone(),
                    commitment: c.artifact.commitment.clone(),
                    hint: c.artifact.hint.clone(),
                },
                None => self.commit_streamed_prefix(source, id, &cached)?,
            };
            Ok(CachedSetupPrefix {
                artifact,
                source: OnceLock::new(),
            })
        })
    }

    fn commit_streamed_prefix(
        &self,
        source: PrefixSource<'_, F>,
        id: &SetupPrefixSlotId,
        cached: &[Arc<CachedSetupPrefix<F>>],
    ) -> Result<SetupPrefixSlot<F>, AkitaError>
    where
        F: Valid + Unreduced + WithCommitAccumulator,
    {
        let p = id.commitment_profile;
        let a = p.inner.matrix;
        let shape = Shape::inner(p);
        let count =
            akita_error::checked::product([p.blocks.live_blocks, shape.rows, shape.ring])
                .ok_or_else(|| AkitaError::InvalidSetup("prefix inner extent overflow".into()))?;
        let mut inner = vec![F::zero(); count];
        let mut reused = vec![0; count / shape.ring];
        for (index, output) in inner.chunks_exact_mut(shape.ring).enumerate() {
            if let Some((positions, values)) =
                contribution(cached, shape, index / shape.rows, index % shape.rows)
            {
                output.copy_from_slice(values);
                *reused.get_mut(index).ok_or(AkitaError::InvalidProof)? = positions;
            }
        }
        akita_params::dispatch_for_field!(
            akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
            F,
            shape.ring,
            |D| {
                products::<F, D>(source, shape, &mut inner, &reused, |block, positions| {
                    let begin = (block * shape.positions + positions.start) * D;
                    source.read(begin..begin + positions.len() * D)
                })
            }
        )?;
        let b = p.outer.matrix;
        let geometry = p.derive_slice_geometry()?;
        let ranges = geometry.block_ranges();
        let outer = Shape {
            positions: b.input_width() / p.outer.digits.num_digits,
            digits: p.outer.digits.num_digits,
            basis: p.outer.digits.log_basis,
            rows: b.output_rank(),
            ring: b.ring_dimension(),
        };
        let count = akita_error::checked::product([ranges.len(), outer.rows, outer.ring])
            .ok_or_else(|| AkitaError::InvalidSetup("prefix outer extent overflow".into()))?;
        let mut image = vec![F::zero(); count];
        akita_params::dispatch_for_field!(
            akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Outer),
            F,
            outer.ring,
            |D| {
                products::<F, D>(source, outer, &mut image, &[], |slice, positions| {
                    let blocks = ranges.get(slice).ok_or(AkitaError::InvalidProof)?;
                    let start = blocks.start * shape.rows * shape.ring + positions.start * D;
                    let end = blocks.end * shape.rows * shape.ring;
                    let mut fields = vec![F::zero(); positions.len() * D];
                    if start < end {
                        let live = fields.len().min(end - start);
                        fields
                            .get_mut(..live)
                            .ok_or(AkitaError::InvalidProof)?
                            .copy_from_slice(
                                inner
                                    .get(start..start + live)
                                    .ok_or(AkitaError::InvalidProof)?,
                            );
                    }
                    Ok(fields)
                })
            }
        )?;
        let plan = p.outer_compression_plan()?;
        let (payload, compression) = self
            .compress_relation_image(
                &plan,
                RingRelationMode::QuotientLift,
                RingVec::from_coeffs_with_ring_dim(image, outer.ring)?,
            )?
            .into_parts();
        let mut hint = PortableCommitmentHandle::new_with_outer_compression(
            a.ring_dimension(),
            vec![RingVec::from_coeffs_with_ring_dim(
                inner,
                a.ring_dimension(),
            )?],
            compression.witness(),
            compression.quotients().ok_or(AkitaError::InvalidProof)?,
        )?;
        hint.share_inner_rows(&self.setup_prefix_inner_rows)?;
        Ok(SetupPrefixSlot {
            id: id.clone(),
            commitment: SetupPrefixPublicCommitment {
                rows: vec![payload],
            },
            hint,
        })
    }
}

#[derive(Clone, Copy)]
struct Shape {
    positions: usize,
    digits: usize,
    basis: u32,
    ring: usize,
    rows: usize,
}

impl Shape {
    fn inner(p: GroupCommitPhaseParams) -> Self {
        Self {
            positions: p.blocks.positions_per_block,
            digits: p.inner.digits.num_digits,
            basis: p.inner.digits.log_basis,
            ring: p.inner.matrix.ring_dimension(),
            rows: p.inner.matrix.output_rank(),
        }
    }

    fn cached_ring(self, old: Self, block: usize, row: usize, old_prefix: usize) -> Option<usize> {
        if old.ring != self.ring
            || old.digits != self.digits
            || old.basis != self.basis
            || old.positions > self.positions
            || !self.positions.is_multiple_of(old.positions)
        {
            return None;
        }
        // Changing width changes the row stride. Match absolute setup offsets,
        // including nonzero output rows, and take every width-ratio-th block.
        let ratio = self.positions / old.positions;
        let old_row = row.checked_mul(ratio)?;
        let start = akita_error::checked::product([block, self.positions, self.ring])?;
        let width = old.positions.checked_mul(old.ring)?;
        if old_row >= old.rows || start.checked_add(width)? > old_prefix {
            return None;
        }
        akita_error::checked::mul_add(start / width, old.rows, old_row)
    }
}

fn contribution<F: Field>(
    cached: &[Arc<CachedSetupPrefix<F>>],
    shape: Shape,
    block: usize,
    row: usize,
) -> Option<(usize, &[F])> {
    cached
        .iter()
        .filter_map(|cached| {
            let artifact = &cached.artifact;
            let old = Shape::inner(artifact.id.commitment_profile);
            let index = shape.cached_ring(old, block, row, artifact.id.n_prefix().ok()?)?;
            Some((old.positions, artifact.hint.inner_ring(0, index)?))
        })
        .max_by_key(|(positions, _)| *positions)
}

#[cfg(test)]
mod tests;
