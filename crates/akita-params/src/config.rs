//! Shared commitment-config data shapes.

use akita_error::AkitaError;

/// Parameters controlling the gadget decomposition depth (called delta in the paper).
///
/// The gadget base is `b = 2^log_basis`. Each ring coefficient is decomposed
/// into balanced digits in `[-b/2, b/2)`. The exact depth comes from
/// [`crate::sis::num_digits_for_bound`]. A bounded signed width can need one
/// more digit than `ceil(log_commit_bound / log_basis)` because the balanced
/// interval reaches one value farther on the negative side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecompositionParams {
    /// Base-2 logarithm of the gadget base (e.g. 3 for base-8 digits in [-4, 3]).
    pub log_basis: u32,

    /// Bit-width of the largest committed source coefficient the commitment
    /// decomposition must represent.
    ///
    /// The centered representation maps each coefficient `c in [0, q)` to the
    /// signed value in `(-q/2, q/2]`. A value of `k` means the signed magnitude
    /// fits in `k` bits, i.e. lies in `[-2^(k-1), 2^(k-1) - 1]`.
    ///
    /// This is the **declared source bound**: how *wide* a committed coefficient
    /// may be. It is independent of the committed-source *class*
    /// ([`crate::sis::CommittedSourceClass`]), which says what *shape* the source
    /// has. Neither may be inferred from the other — in particular
    /// `log_commit_bound == 1` is **not** a test for "is this one-hot": a
    /// balanced-digit source may declare any bound from 1 to the field width, and
    /// a unit one-hot source is one because its class says so.
    ///
    /// The bound has two consumers:
    ///
    /// 1. **The A-role digit depth**, via [`crate::sis::num_digits_inner_for_bound`],
    ///    which fixes the A input width, the SIS rank it demands, and the
    ///    next-level witness length.
    /// 2. **The bounded source-moment model** in the planner, which charges a
    ///    bounded source's final digit plane only the range the bound leaves.
    ///    This is why the bound has to be *enforced* at commit rather than merely
    ///    documented: a coefficient past it inflates the level-1 witness beyond
    ///    the L2 response caps frozen into the recursion suffix.
    ///
    /// A commitment is binding and complete only for sources inside
    /// [`crate::sis::CommittedSourceContract::accepted_bounds`] — this bound
    /// intersected with what the selected digit depth can represent — *and* of the
    /// declared class. Producers must reject anything else rather than committing
    /// a value the schedule was not priced for.
    pub log_commit_bound: u32,

    /// Bit-width of the largest coefficient that the opening decomposition
    /// must represent. When `None`, this defaults to `log_commit_bound`.
    ///
    /// Opening witnesses (`t̂` / `ŵ`) and setup prefixes carry genuine field
    /// elements, so this is the true field width whenever the committed source
    /// is bounded below it. A bounded source therefore always sets this
    /// explicitly; see [`Self::validate`].
    pub log_open_bound: Option<u32>,
}

impl DecompositionParams {
    /// Effective field-element bit-width used for opening witnesses.
    ///
    /// This is deliberately the *open* width, not the committed source bound: a
    /// bounded source shrinks [`Self::log_commit_bound`] while its openings stay
    /// full-width.
    pub fn field_bits(self) -> u32 {
        self.log_open_bound.unwrap_or(self.log_commit_bound)
    }

    /// Whether the committed source is bounded strictly below the field width.
    #[must_use]
    pub fn has_bounded_committed_source(self) -> bool {
        self.log_commit_bound < self.field_bits()
    }

    /// Reject a decomposition whose bounds cannot describe a committed source.
    ///
    /// `log_basis` must be usable and the source bound must be a nonzero bit
    /// width that does not exceed the field width. A source bounded below the
    /// field width names that width through [`Self::log_open_bound`]; leaving it
    /// `None` collapses [`Self::field_bits`] onto the source bound, which is the
    /// full-field endpoint rather than a bounded source.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when any of those conditions fails.
    pub fn validate(self) -> Result<(), AkitaError> {
        if self.log_basis == 0 || self.log_basis >= 128 {
            return Err(AkitaError::InvalidSetup(format!(
                "decomposition log_basis {} is outside 1..128",
                self.log_basis
            )));
        }
        let field_bits = self.field_bits();
        if field_bits == 0 || field_bits > 128 {
            return Err(AkitaError::InvalidSetup(format!(
                "decomposition field width {field_bits} is outside 1..=128"
            )));
        }
        if self.log_commit_bound == 0 || self.log_commit_bound > field_bits {
            return Err(AkitaError::InvalidSetup(format!(
                "committed source bound {} is outside 1..={field_bits}",
                self.log_commit_bound
            )));
        }
        Ok(())
    }
}

/// Verifier strategy for the public setup contribution in the ring-switch row
/// evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupContributionMode {
    /// Evaluate the setup contribution directly from the expanded setup matrix.
    Direct,
    /// Use the recursive setup-contribution path.
    Recursive,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(log_commit_bound: u32, log_open_bound: Option<u32>) -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound,
            log_open_bound,
        }
    }

    #[test]
    fn every_source_bound_from_one_hot_to_full_field_is_accepted() {
        // Unit one-hot endpoint, an interior bounded source, and the full-field
        // endpoint are all valid points of the same parameter.
        for log_commit_bound in [1u32, 32, 64, 96, 128] {
            params(log_commit_bound, Some(128))
                .validate()
                .expect("bounded source within the field width");
        }
        params(128, None)
            .validate()
            .expect("full-field source without an explicit open bound");
    }

    #[test]
    fn bounded_source_is_reported_only_below_the_field_width() {
        assert!(params(64, Some(128)).has_bounded_committed_source());
        assert!(params(1, Some(128)).has_bounded_committed_source());
        assert!(!params(128, Some(128)).has_bounded_committed_source());
        assert!(!params(128, None).has_bounded_committed_source());
    }

    #[test]
    fn field_bits_follows_the_open_bound_not_the_source_bound() {
        assert_eq!(params(64, Some(128)).field_bits(), 128);
        assert_eq!(params(64, None).field_bits(), 64);
    }

    #[test]
    fn degenerate_bounds_are_rejected() {
        assert!(params(0, Some(128)).validate().is_err());
        assert!(params(129, Some(128)).validate().is_err());
        // A source bound above the declared field width cannot be represented.
        assert!(params(128, Some(64)).validate().is_err());
        assert!(params(64, Some(0)).validate().is_err());
        assert!(DecompositionParams {
            log_basis: 0,
            log_commit_bound: 64,
            log_open_bound: Some(128),
        }
        .validate()
        .is_err());
    }
}
