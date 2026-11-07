//! Explicit preprocessing for the binary root's typed setup prefix: A, then B/D.

use crate::{CommitmentHandle, CpuBackend};
use akita_error::{checked, AkitaError};
use akita_params::SetupPrefixSlotId;
use akita_prover::{PreparedSetupPrefix, SetupPrefixProverRegistry};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryTypedPrefix, CompositePrefixBlock, CompositePrefixKind,
    CompositePrefixLayout,
};
use akita_types::{
    AkitaVerifierSetup, FpExtEncoding, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
};
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, Ring, Unreduced,
    WithCommitAccumulator,
};

/// Authenticated local definitions supplied at trusted preprocessing time.
/// Each binary definition retains its original seed and typed label namespace.
/// Ordinary tables are opaque here; their role-specific authentication belongs
/// to their trusted preparer, rather than to the proof parser.
pub enum TrustedCompositePrefixDefinition<'a, F: Field> {
    Binary {
        seed: &'a [u8; 32],
        manifest: &'a BinaryCommittedGroupManifest,
        typed: &'a BinaryTypedPrefix,
        local_slot: &'a SetupPrefixSlotId,
    },
    Ordinary {
        block: &'a CompositePrefixBlock,
        coefficients: &'a [F],
    },
}

/// Explicit cost of constructing the one composed setup commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompositePrefixPreparationCost {
    pub table_len: usize,
    pub local_scans: usize,
}

/// Backend-created trusted commitment, bound to its complete roster identity.
/// Private fields prevent a proof-supplied slot from acquiring this authority.
pub struct PreparedCompositePrefix<F: Field, H> {
    prepared: PreparedSetupPrefix<F, H>,
    identity: Vec<u8>,
    successor_seed: [u8; 32],
    pub cost: CompositePrefixPreparationCost,
}

impl<F: Field, H> PreparedCompositePrefix<F, H> {
    pub fn prepared(&self) -> &PreparedSetupPrefix<F, H> {
        &self.prepared
    }

    pub fn identity(&self) -> &[u8] {
        &self.identity
    }
}

impl<F, E> CpuBackend<F, E>
where
    F: Field
        + CanonicalEncoding
        + AkitaSerialize
        + Valid
        + Ring
        + Unreduced
        + WithCommitAccumulator
        + 'static,
    F::Wide: From<F> + AdditiveGroup,
    E: ExtField<F>
        + FpExtEncoding<F>
        + MulBaseUnreduced<F>
        + Unreduced
        + Fold
        + AkitaSerialize
        + 'static,
{
    /// Derive each authenticated local setup table and commit the assembled
    /// coefficient table exactly once. No local commitment payloads are combined.
    /// The explicit trusted budget is checked before setup scans or allocations.
    pub fn prepare_composite_prefix(
        &self,
        layout: &CompositePrefixLayout,
        definitions: &[TrustedCompositePrefixDefinition<'_, F>],
        max_cells: usize,
    ) -> Result<PreparedCompositePrefix<F, CommitmentHandle<F, E>>, AkitaError> {
        if definitions.len() != layout.placements().len() || layout.domain_len() > max_cells {
            return Err(AkitaError::InvalidInput(
                "composite prefix definitions or trusted budget mismatch".into(),
            ));
        }
        let modulus = akita_params::field_modulus::<F>()?;
        let profile = &layout.slot().commitment_profile;
        if profile.inner.matrix.sis_modulus_profile().modulus() != modulus
            || profile.outer.matrix.sis_modulus_profile().modulus() != modulus
        {
            return Err(AkitaError::InvalidInput(
                "composite prefix successor field mismatch".into(),
            ));
        }
        checked::product([layout.domain_len(), std::mem::size_of::<F>()])
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                AkitaError::InvalidInput("composite prefix allocation overflow".into())
            })?;
        // Validate every definition before scanning even the first local setup.
        for (index, (definition, placement)) in
            definitions.iter().zip(layout.placements()).enumerate()
        {
            let expected = match definition {
                TrustedCompositePrefixDefinition::Binary {
                    seed,
                    manifest,
                    typed,
                    local_slot,
                } => {
                    if modulus != manifest.field_profile().modulus()
                        || E::DEGREE != manifest.field_profile().extension_degree()
                    {
                        return Err(AkitaError::InvalidInput(
                            "composite prefix field profile mismatch".into(),
                        ));
                    }
                    CompositePrefixBlock::binary(index, seed, manifest, typed, local_slot)?
                }
                TrustedCompositePrefixDefinition::Ordinary {
                    block,
                    coefficients,
                } => {
                    if block.kind != CompositePrefixKind::Ordinary
                        || block.index != index
                        || coefficients.len() != block.domain_len
                    {
                        return Err(AkitaError::InvalidInput(
                            "opaque ordinary prefix definition mismatch".into(),
                        ));
                    }
                    (*block).clone()
                }
            };
            if expected != placement.block {
                return Err(AkitaError::InvalidInput(
                    "composite prefix local setup identity mismatch".into(),
                ));
            }
        }
        let prepared = self.prepared()?;
        crate::setup::setup_prefix::validate_setup_prefix_commitment(
            prepared.expanded.shared_matrix().num_field_elements(),
            layout.slot(),
        )?;
        let successor_seed = prepared.expanded.descriptor().setup_seed.seed;
        let local_tables = definitions
            .iter()
            .map(|definition| match definition {
                TrustedCompositePrefixDefinition::Binary {
                    seed,
                    manifest,
                    typed,
                    ..
                } => typed.setup_coefficients(seed, manifest, typed.domain_len),
                TrustedCompositePrefixDefinition::Ordinary { coefficients, .. } => {
                    Ok(coefficients.to_vec())
                }
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let tables = local_tables.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let coefficients = layout.assemble(&tables, max_cells)?;
        let slot = self.prepare_setup_prefix_from_coefficients(layout.slot(), coefficients)?;
        Ok(PreparedCompositePrefix {
            prepared: slot,
            identity: layout.identity().to_vec(),
            successor_seed,
            cost: CompositePrefixPreparationCost {
                table_len: layout.domain_len(),
                local_scans: definitions.len(),
            },
        })
    }

    /// Install the one trusted composed slot and its exact identity for online
    /// verification. This API accepts only a preprocessing result, never a
    /// commitment read from the prover's proof stream.
    pub fn prepare_composite_verifier_setup(
        &self,
        ordinary: &AkitaVerifierSetup<F>,
        layout: &CompositePrefixLayout,
        trusted: &PreparedCompositePrefix<F, CommitmentHandle<F, E>>,
    ) -> Result<AkitaVerifierSetup<F>, AkitaError> {
        let prepared = self.prepared()?;
        if trusted.identity != layout.identity()
            || trusted.prepared.public.id != *layout.slot()
            || trusted.successor_seed != prepared.expanded.descriptor().setup_seed.seed
            || ordinary.expanded().descriptor().setup_seed
                != prepared.expanded.descriptor().setup_seed
        {
            return Err(AkitaError::InvalidInput(
                "trusted composite prefix context mismatch".into(),
            ));
        }
        let mut registry =
            SetupPrefixVerifierRegistry::new(ordinary.expanded().descriptor().setup_seed.clone());
        registry.insert(trusted.prepared.public.clone())?;
        for (id, slot) in ordinary.prefix_slots().iter() {
            if id != layout.slot() {
                registry.insert(slot.clone())?;
            }
        }
        let mut verifier = AkitaVerifierSetup::from_parts(ordinary.expanded().clone(), registry)?;
        for (id, _) in ordinary.prefix_slots().iter() {
            if id != layout.slot() {
                if let Some(context) = ordinary.prefix_context(id) {
                    verifier = verifier.with_prefix_context(id.clone(), context.to_vec())?;
                }
            }
        }
        verifier.with_prefix_context(layout.slot().clone(), layout.identity().to_vec())
    }

    /// Prepare a scoped registry whose first successor prefix commits the typed prefix.
    ///
    /// The same slot identifier can continue to name the ordinary stream in
    /// other registries. This operation never inserts typed material into the
    /// backend's ordinary prefix cache. `remaining` lists the later schedule
    /// prefixes; repeating `first` there would make the two streams ambiguous.
    pub fn prepare_binary_typed_prefixes(
        &self,
        manifest: &BinaryCommittedGroupManifest,
        typed: &BinaryTypedPrefix,
        first: &SetupPrefixSlotId,
        remaining: &[SetupPrefixSlotId],
    ) -> Result<SetupPrefixProverRegistry<F, CommitmentHandle<F, E>>, AkitaError> {
        if akita_params::field_modulus::<F>()? != manifest.field_profile().modulus()
            || E::DEGREE != manifest.field_profile().extension_degree()
        {
            return Err(AkitaError::InvalidInput(
                "typed prefix field profile mismatch".into(),
            ));
        }
        // Check the manifest/layout binding before deriving the setup table.
        typed.identity(manifest, first)?;
        if remaining.contains(first) {
            return Err(AkitaError::InvalidInput(
                "typed prefix slot is reused by a later ordinary level".into(),
            ));
        }
        let prepared = self.prepared()?;
        let coefficients = typed.setup_coefficients(
            &prepared.expanded.descriptor().setup_seed.seed,
            manifest,
            first.n_prefix()?,
        )?;
        let mut registry = SetupPrefixProverRegistry::default();
        registry.insert(self.prepare_setup_prefix_from_coefficients(first, coefficients)?)?;
        for id in remaining {
            // Several ordinary levels can share one immutable prefix slot.
            if registry.get(id).is_none() {
                registry.insert(self.prepare_setup_prefix(id)?)?;
            }
        }
        Ok(registry)
    }

    /// Authenticate a supplied typed prefix commitment once and install a verifier view.
    ///
    /// The expected prefix is independently regenerated from the manifest,
    /// typed layout and setup seed. A different A, B or D, or a prefix from the
    /// ordinary stream, rejects here. Online verification only consumes the
    /// resulting public commitment; its expanded setup continues to contain
    /// the ordinary stream exclusively. The installed transient context binds
    /// the prefix to this exact manifest and layout.
    /// Reprepare this view after loading a persisted ordinary setup.
    pub fn prepare_binary_typed_verifier_setup(
        &self,
        ordinary: &AkitaVerifierSetup<F>,
        manifest: &BinaryCommittedGroupManifest,
        typed: &BinaryTypedPrefix,
        first: &SetupPrefixSlotId,
        supplied: &SetupPrefixVerifierSlot<F>,
    ) -> Result<AkitaVerifierSetup<F>, AkitaError> {
        let prepared = self.prepared()?;
        if ordinary.expanded().descriptor().setup_seed != prepared.expanded.descriptor().setup_seed
        {
            return Err(AkitaError::InvalidInput(
                "typed prefix and verifier setup have different seeds".into(),
            ));
        }
        let expected = self.prepare_binary_typed_prefixes(manifest, typed, first, &[])?;
        let expected = &expected
            .get(first)
            .ok_or_else(|| AkitaError::Internal("prepared typed prefix is missing".into()))?
            .public;
        supplied
            .check()
            .map_err(|_| AkitaError::InvalidInput("invalid typed prefix commitment".into()))?;
        if supplied.id != expected.id
            || supplied.commitment.rows.len() != expected.commitment.rows.len()
            || supplied
                .commitment
                .rows
                .iter()
                .zip(&expected.commitment.rows)
                .any(|(actual, expected)| {
                    actual.coeffs() != expected.coeffs()
                        || (actual.ring_dim() != 0 && actual.ring_dim() != expected.ring_dim())
                })
        {
            return Err(AkitaError::InvalidInput(
                "typed prefix differs from its deterministic setup stream".into(),
            ));
        }
        let mut registry =
            SetupPrefixVerifierRegistry::new(ordinary.expanded().descriptor().setup_seed.clone());
        registry.insert(expected.clone())?;
        for (id, slot) in ordinary.prefix_slots().iter() {
            if id != first {
                registry.insert(slot.clone())?;
            }
        }
        let mut verifier = AkitaVerifierSetup::from_parts(ordinary.expanded().clone(), registry)?;
        for (id, _) in ordinary.prefix_slots().iter() {
            if id != first {
                if let Some(context) = ordinary.prefix_context(id) {
                    verifier = verifier.with_prefix_context(id.clone(), context.to_vec())?;
                }
            }
        }
        verifier.with_prefix_context(first.clone(), typed.identity(manifest, first)?)
    }
}

#[cfg(test)]
#[path = "setup_prefix/composite_tests.rs"]
mod composite_tests;
