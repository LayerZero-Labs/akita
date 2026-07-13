//! Key type for runtime ring-dimension NTT prepared-setup caches.

use akita_field::{AkitaError, CanonicalField, FieldCore};

use crate::dispatch::{protocol_dispatch_tier, slot_dim_supported_for_tier, ProtocolDispatchSlot};
use crate::proof::AkitaExpandedSetup;
use std::collections::BTreeMap;

/// Identifies one full-envelope NTT cache entry at a concrete ring degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NttCacheKey {
    /// Ring dimension `D` for the cached transform family.
    ring_d: usize,
    /// Number of ring elements in the cached matrix view at `ring_d`.
    num_ring_elements: usize,
}

impl NttCacheKey {
    #[must_use]
    pub fn ring_d(self) -> usize {
        self.ring_d
    }

    #[must_use]
    pub fn num_ring_elements(self) -> usize {
        self.num_ring_elements
    }

    /// Build the full-envelope cache key for `ring_d` on `expanded`.
    ///
    /// # Errors
    ///
    /// Returns an error when `ring_d` does not divide the setup envelope or the
    /// matrix view length cannot be computed.
    pub fn from_envelope<F: FieldCore>(
        expanded: &AkitaExpandedSetup<F>,
        ring_d: usize,
    ) -> Result<Self, AkitaError> {
        let num_ring_elements = expanded
            .shared_matrix()
            .total_ring_elements_at_dyn(ring_d)?;
        Ok(Self {
            ring_d,
            num_ring_elements,
        })
    }
}

/// Checked, canonical NTT-prefix preparation plan.
///
/// Slots are sorted by ring dimension and coalesced to one maximum prefix per
/// dimension. Logical roles and compression-map identities never enter cache keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedNttPlan {
    slots: Vec<NttCacheKey>,
}

impl PreparedNttPlan {
    /// Compile the current base setup requirement: the full generated envelope.
    pub fn base_envelope<F: FieldCore + CanonicalField>(
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        Self::envelope_at(expanded, expanded.seed().gen_ring_dim)
    }

    /// Compile one full-envelope slot at an explicitly selected envelope degree.
    pub fn envelope_at<F: FieldCore + CanonicalField>(
        expanded: &AkitaExpandedSetup<F>,
        ring_d: usize,
    ) -> Result<Self, AkitaError> {
        let tier = protocol_dispatch_tier::<F>();
        if !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Envelope, ring_d) {
            return Err(AkitaError::InvalidSetup(format!(
                "setup envelope ring dimension {ring_d} is outside field dispatch policy"
            )));
        }
        Ok(Self {
            slots: vec![NttCacheKey::from_envelope(expanded, ring_d)?],
        })
    }

    /// Compile the base envelope plus explicit compression prefix requirements.
    ///
    /// The compression catalog replay supplies `(ring_d, prefix_ring_elements)`
    /// pairs. Requirements are checked against the field's compression dispatch
    /// policy and the expanded matrix, then coalesced by maximum prefix at each `d`.
    pub fn with_compression_requirements<F: FieldCore + CanonicalField>(
        expanded: &AkitaExpandedSetup<F>,
        requirements: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self, AkitaError> {
        let base = Self::base_envelope(expanded)?;
        let mut by_d = BTreeMap::new();
        for key in base.slots {
            by_d.insert(key.ring_d, key.num_ring_elements);
        }
        let tier = protocol_dispatch_tier::<F>();
        for (ring_d, prefix) in requirements {
            if prefix == 0 {
                return Err(AkitaError::InvalidSetup(
                    "prepared NTT prefix must be nonzero".into(),
                ));
            }
            if !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Compression, ring_d) {
                return Err(AkitaError::InvalidSetup(format!(
                    "compression NTT ring dimension {ring_d} is outside field dispatch policy"
                )));
            }
            let available = expanded
                .shared_matrix()
                .total_ring_elements_at_dyn(ring_d)?;
            if prefix > available {
                return Err(AkitaError::InvalidSetup(format!(
                    "prepared NTT prefix {prefix} exceeds available {available} at ring_d={ring_d}"
                )));
            }
            by_d.entry(ring_d)
                .and_modify(|current| *current = (*current).max(prefix))
                .or_insert(prefix);
        }
        Ok(Self {
            slots: by_d
                .into_iter()
                .map(|(ring_d, num_ring_elements)| NttCacheKey {
                    ring_d,
                    num_ring_elements,
                })
                .collect(),
        })
    }

    /// Canonical planned slots in increasing ring-dimension order.
    pub fn slots(&self) -> &[NttCacheKey] {
        &self.slots
    }

    /// Resolve a requested prefix to its containing canonical planned slot.
    pub fn resolve(
        &self,
        ring_d: usize,
        required_prefix: usize,
    ) -> Result<NttCacheKey, AkitaError> {
        let key = self
            .slots
            .binary_search_by_key(&ring_d, |key| key.ring_d)
            .ok()
            .and_then(|index| self.slots.get(index))
            .copied()
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "prepared NTT plan has no slot for ring_d={ring_d}"
                ))
            })?;
        if required_prefix > key.num_ring_elements {
            return Err(AkitaError::InvalidSetup(format!(
                "prepared NTT prefix {required_prefix} exceeds planned {} at ring_d={ring_d}",
                key.num_ring_elements
            )));
        }
        Ok(key)
    }

    /// Resolve a planned containing slot, or derive one checked exact-prefix fallback key.
    ///
    /// The boolean is `true` for a planned hit. Backends use `false` to retain
    /// the existing diagnostic lazy-build path without exposing unchecked keys.
    /// The caller must already have validated the operation's role/compression
    /// dispatch slot; this fallback additionally checks the NTT kernel lattice.
    pub fn resolve_with_fallback<F: FieldCore + CanonicalField>(
        &self,
        expanded: &AkitaExpandedSetup<F>,
        ring_d: usize,
        required_prefix: usize,
    ) -> Result<(NttCacheKey, bool), AkitaError> {
        if let Ok(key) = self.resolve(ring_d, required_prefix) {
            return Ok((key, true));
        }
        if required_prefix == 0 {
            return Err(AkitaError::InvalidInput(
                "an unplanned zero-width NTT request has no cache slot".into(),
            ));
        }
        let tier = protocol_dispatch_tier::<F>();
        if !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Ntt, ring_d) {
            return Err(AkitaError::InvalidSetup(format!(
                "fallback NTT ring dimension {ring_d} is outside field dispatch policy"
            )));
        }
        let available = expanded
            .shared_matrix()
            .total_ring_elements_at_dyn(ring_d)?;
        if required_prefix > available {
            return Err(AkitaError::InvalidSetup(format!(
                "fallback NTT prefix {required_prefix} exceeds available {available} at ring_d={ring_d}"
            )));
        }
        Ok((
            NttCacheKey {
                ring_d,
                num_ring_elements: required_prefix,
            },
            false,
        ))
    }
}
