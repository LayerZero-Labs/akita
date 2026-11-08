//! Explicit preprocessing for the binary root's typed setup prefix: A, then B/D.

use crate::{CommitmentHandle, CpuBackend};
use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;
use akita_prover::SetupPrefixProverRegistry;
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryTypedPrefix, GroupedTypedPrefix,
};
use akita_types::{
    AkitaVerifierSetup, FpExtEncoding, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
};
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, Ring, Unreduced,
    WithCommitAccumulator,
};

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
        self.prepare_typed_prefix_registry(first, remaining, coefficients)
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
        Self::authenticate_typed_prefix(ordinary, first, supplied, expected, || {
            typed.identity(manifest, first)
        })
    }

    /// Prepare one registry for the capacity-bound ordered grouped stream.
    pub fn prepare_grouped_typed_prefixes(
        &self,
        typed: &GroupedTypedPrefix,
        remaining: &[SetupPrefixSlotId],
    ) -> Result<SetupPrefixProverRegistry<F, CommitmentHandle<F, E>>, AkitaError> {
        let first = typed.first();
        if E::DEGREE != 2
            || self.prepared()?.expanded.descriptor() != typed.setup()
            || remaining.contains(first)
        {
            return Err(AkitaError::InvalidInput(
                "grouped prefix setup mismatch".into(),
            ));
        }
        let coefficients = typed.setup_coefficients()?;
        self.prepare_typed_prefix_registry(first, remaining, coefficients)
    }

    /// Authenticate the complete grouped stream before installing its one context.
    pub fn prepare_grouped_typed_verifier_setup(
        &self,
        ordinary: &AkitaVerifierSetup<F>,
        typed: &GroupedTypedPrefix,
        supplied: &SetupPrefixVerifierSlot<F>,
    ) -> Result<AkitaVerifierSetup<F>, AkitaError> {
        if ordinary.expanded().descriptor() != typed.setup() {
            return Err(AkitaError::InvalidInput(
                "grouped verifier setup mismatch".into(),
            ));
        }
        let expected = self.prepare_grouped_typed_prefixes(typed, &[])?;
        let expected = &expected
            .get(typed.first())
            .ok_or(AkitaError::InvalidProof)?
            .public;
        let verifier =
            Self::authenticate_typed_prefix(ordinary, typed.first(), supplied, expected, || {
                Ok(typed.identity().to_vec())
            })?;
        typed.validate_verifier_setup(&verifier)?;
        Ok(verifier)
    }

    fn prepare_typed_prefix_registry(
        &self,
        first: &SetupPrefixSlotId,
        remaining: &[SetupPrefixSlotId],
        coefficients: Vec<F>,
    ) -> Result<SetupPrefixProverRegistry<F, CommitmentHandle<F, E>>, AkitaError> {
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

    fn authenticate_typed_prefix(
        ordinary: &AkitaVerifierSetup<F>,
        first: &SetupPrefixSlotId,
        supplied: &SetupPrefixVerifierSlot<F>,
        expected: &SetupPrefixVerifierSlot<F>,
        identity: impl FnOnce() -> Result<Vec<u8>, AkitaError>,
    ) -> Result<AkitaVerifierSetup<F>, AkitaError> {
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
        verifier.with_prefix_context(first.clone(), identity()?)
    }
}
