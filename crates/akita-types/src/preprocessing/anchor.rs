//! Compiled trust anchors: the `(bundle_id, set name)` pairs a binary trusts.

use super::identity::BundleId;
use akita_error::AkitaError;

/// One trusted parameter set of one release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorEntry {
    bundle_id: BundleId,
    set_name: &'static str,
}

impl AnchorEntry {
    /// Trust the set `set_name` of the release whose `bundle_id` is
    /// `bundle_id_hex` (64 lowercase hex digits).
    ///
    /// # Panics
    ///
    /// Panics on malformed hex; anchors are `const` items, where this is a
    /// compile error.
    #[must_use]
    pub const fn new(bundle_id_hex: &'static str, set_name: &'static str) -> Self {
        Self {
            bundle_id: BundleId::from_bytes(super::encoding::digest_from_hex(bundle_id_hex)),
            set_name,
        }
    }

    /// The anchored release.
    #[must_use]
    pub const fn bundle_id(&self) -> BundleId {
        self.bundle_id
    }

    /// The anchored set.
    #[must_use]
    pub const fn set_name(&self) -> &'static str {
        self.set_name
    }
}

/// The trust anchor compiled into an integration.
///
/// An anchor is declared as a compile-time constant from a `'static` list and
/// has no deserializer and no `FromStr`: an anchor read from storage voids
/// every guarantee a loader gives. `scripts/check-trust-anchors.sh` rejects
/// construction outside a `const` or `static` item in production code.
///
/// ```compile_fail
/// use akita_types::preprocessing::TrustAnchor;
/// let anchor: TrustAnchor = "00".parse().unwrap();
/// ```
///
/// ```compile_fail
/// use akita_types::preprocessing::TrustAnchor;
/// let anchor = akita_serialization::AkitaDeserialize::deserialize_compressed(&[][..], &());
/// let _: TrustAnchor = anchor.unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TrustAnchor {
    entries: &'static [AnchorEntry],
}

impl TrustAnchor {
    /// An anchor over a compiled list of pairs.
    #[must_use]
    pub const fn new(entries: &'static [AnchorEntry]) -> Self {
        Self { entries }
    }

    /// Every anchored pair.
    #[must_use]
    pub const fn entries(&self) -> &'static [AnchorEntry] {
        self.entries
    }

    /// Whether `(bundle_id, set_name)` is anchored.
    #[must_use]
    pub fn admits(&self, bundle_id: &BundleId, set_name: &str) -> bool {
        self.entries
            .iter()
            .any(|entry| &entry.bundle_id == bundle_id && entry.set_name == set_name)
    }

    /// Require `(bundle_id, set_name)` to be anchored.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the pair is not anchored.
    pub fn require(&self, bundle_id: &BundleId, set_name: &str) -> Result<(), AkitaError> {
        if self.admits(bundle_id, set_name) {
            Ok(())
        } else {
            Err(AkitaError::InvalidSetup(format!(
                "preprocessing set {set_name:?} of release {bundle_id} is not in the trust anchor"
            )))
        }
    }
}
