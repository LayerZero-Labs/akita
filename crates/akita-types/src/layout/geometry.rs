//! Leaf geometry components shared by every commitment group.
//!
//! These two types are the
//! single home for arithmetic that is currently spelled out at each site that
//! needs it. They own no policy and no matrix identity, so they are `Copy`,
//! const-constructible, and safe to embed in static commit-phase profiles.
//!
//! Nothing here changes a byte. The block triple `(N, M, B)` is already
//! contiguous in every descriptor encoder that writes it, so an atomic encoder
//! for this type is byte-neutral wherever it replaces three field writes.

use akita_error::AkitaError;

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::signed_digit::SignedDigitKernel;
use crate::sis::compute_num_digits_field_width;

/// Exact block geometry of one commitment group.
///
/// Field names match the generated mirror, so the runtime and the static tables
/// use one vocabulary. In protocol notation: `N` live source ring elements per
/// claim, split into blocks of `M` positions, giving `B = ceil(N / M)` live
/// blocks.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct BlockGeometry {
    /// Live source ring elements per claim (`N`).
    pub live_ring_elements_per_claim: usize,
    /// Positions per block (`M`), a power of two.
    pub positions_per_block: usize,
    /// Live blocks (`B = ceil(N / M)`).
    pub live_blocks: usize,
}

impl BlockGeometry {
    /// Assemble a block triple without checking it.
    ///
    /// `const` so offline generators and fixed test fixtures can construct it
    /// position. Call [`Self::validate`] on any triple that did not come from a
    /// checked-in table.
    #[must_use]
    pub const fn new(
        live_ring_elements_per_claim: usize,
        positions_per_block: usize,
        live_blocks: usize,
    ) -> Self {
        Self {
            live_ring_elements_per_claim,
            positions_per_block,
            live_blocks,
        }
    }

    /// `N` and `M` are nonzero, `M` is a power of two, `B == ceil(N / M)`, and
    /// the Boolean block-index domain does not overflow `usize`.
    pub fn validate(&self) -> Result<(), AkitaError> {
        if self.live_ring_elements_per_claim == 0
            || self.positions_per_block == 0
            || !self.positions_per_block.is_power_of_two()
            || self.live_blocks == 0
        {
            return Err(AkitaError::InvalidSetup(
                "invalid digit-innermost block geometry".to_string(),
            ));
        }
        let expected = self
            .live_ring_elements_per_claim
            .div_ceil(self.positions_per_block);
        if self.live_blocks != expected {
            return Err(AkitaError::InvalidSetup(format!(
                "num_live_blocks={} does not equal ceil(num_live_ring_elements_per_claim={} / num_positions_per_block={})={expected}",
                self.live_blocks, self.live_ring_elements_per_claim, self.positions_per_block,
            )));
        }
        self.block_index_domain_size()?;
        Ok(())
    }

    /// Boolean coordinates addressing one block's positions, for a bare `M`.
    ///
    /// Several call sites hold `M` and `B` without an `N` to build a whole
    /// triple. They use these associated functions so the formula still has one
    /// definition, while keeping their own error wording.
    #[inline]
    #[must_use]
    pub const fn position_index_bits_for(positions_per_block: usize) -> usize {
        positions_per_block.trailing_zeros() as usize
    }

    /// Padded Boolean block-index domain size for a bare `B`.
    ///
    /// `None` when the padded domain overflows `usize`.
    #[inline]
    #[must_use]
    pub fn checked_block_index_domain_size_for(live_blocks: usize) -> Option<usize> {
        live_blocks.checked_next_power_of_two()
    }

    /// Boolean coordinates addressing the padded block domain, for a bare `B`.
    ///
    /// `None` when the padded domain overflows `usize`.
    #[inline]
    #[must_use]
    pub fn checked_block_index_bits_for(live_blocks: usize) -> Option<usize> {
        Self::checked_block_index_domain_size_for(live_blocks)
            .map(|capacity| capacity.trailing_zeros() as usize)
    }

    /// Number of Boolean coordinates in one block-position slice.
    #[inline]
    #[must_use]
    pub fn position_index_bits(&self) -> usize {
        Self::position_index_bits_for(self.positions_per_block)
    }

    /// Number of Boolean coordinates in the block-index domain.
    ///
    /// Returns `0` when the domain size overflows, matching the historical
    /// accessor. Use [`Self::block_index_domain_size`] where the overflow must
    /// be an error rather than a saturating answer.
    #[inline]
    #[must_use]
    pub fn block_index_bits(&self) -> usize {
        Self::checked_block_index_bits_for(self.live_blocks).unwrap_or(0)
    }

    /// Boolean block-index domain size (`next_power_of_two(B)`).
    pub fn block_index_domain_size(&self) -> Result<usize, AkitaError> {
        Self::checked_block_index_domain_size_for(self.live_blocks).ok_or_else(|| {
            AkitaError::InvalidSetup("block-index domain size overflows usize".to_string())
        })
    }

    /// Atomic descriptor encoding, in declared field order.
    ///
    /// Byte-neutral against the historical `N, M, B` field writes, which are
    /// contiguous in every encoder that carries this triple.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.live_ring_elements_per_claim);
        push_usize(bytes, self.positions_per_block);
        push_usize(bytes, self.live_blocks);
    }
}

/// One gadget decomposition: a basis and an exact depth.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct GadgetDigits {
    /// Log2 of the decomposition basis.
    pub log_basis: u32,
    /// Exact number of digits retained at this basis.
    pub num_digits: usize,
}

impl GadgetDigits {
    /// Assemble a basis and depth without checking them.
    ///
    /// `const` for the same reason as [`BlockGeometry::new`].
    #[must_use]
    pub const fn new(log_basis: u32, num_digits: usize) -> Self {
        Self {
            log_basis,
            num_digits,
        }
    }

    /// A [`SignedDigitKernel`] exists for `log_basis`, and `num_digits` lies in
    /// `(0, compute_num_digits_field_width(field_bits, log_basis)]`.
    ///
    /// This is an **upper bound**, not an exact depth. Since bounded committed
    /// dense sources landed, the A role's exact depth is
    /// `ceil(log_commit_bound / log_basis_inner)`, which is level-aware and
    /// strictly below the field-width bound for a bounded source. That exact
    /// check lives in `akita_schedules::audit_committed_params` and cannot be
    /// expressed here, because this type does not know its level.
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError> {
        if SignedDigitKernel::for_log_basis(self.log_basis).is_none()
            || self.num_digits == 0
            || self.num_digits > compute_num_digits_field_width(field_bits, self.log_basis)
        {
            return Err(AkitaError::InvalidSetup(
                "commitment group inner basis or digit depth exceeds the supported field decomposition"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Atomic descriptor encoding, in declared field order.
    ///
    /// Byte-neutral only where `(log_basis, num_digits)` is already contiguous.
    /// Encoders that interleave role fields must write those fields individually.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_u32(bytes, self.log_basis);
        push_usize(bytes, self.num_digits);
    }
}

/// The A role: source digits and the inner commitment matrix.
pub type InnerRoleParams = RoleParams<crate::InnerCommitMatrixParams>;

/// The B role: `t_hat` digits and the outer commitment matrix.
pub type OuterRoleParams = RoleParams<crate::OuterCommitMatrixParams>;

/// The D role: opening digits and the shared open commitment matrix.
pub type OpenRoleParams = RoleParams<crate::OpenCommitMatrixParams>;

pub(crate) mod sealed_matrix {
    /// Prevents downstream crates from implementing [`super::MatrixDescriptorBytes`].
    pub trait Sealed {}
}

/// A matrix identity that can write itself into a canonical descriptor.
///
/// Sealed and implemented only for the three commitment-matrix types, so
/// [`RoleParams`] can encode atomically without duplicating the two-line body
/// once per role. It is `pub` only because it appears as a bound on a `pub`
/// impl; it is not an extension point.
pub trait MatrixDescriptorBytes: sealed_matrix::Sealed {
    /// Append this matrix identity to a canonical descriptor.
    fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>);
}

/// A gadget decomposition and the matrix that consumes it.
///
/// Generic over the **matrix type**, not over a role marker, because the A
/// role's matrix is `InnerCommitMatrixParams` and carries a security route,
/// while B and D share `LinfCommitMatrix<R>`. Parameterising by the matrix keeps
/// all three expressible without an `Option`-shaped tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RoleParams<M> {
    /// Basis and exact depth for this role.
    pub digits: GadgetDigits,
    /// Audited matrix identity that consumes those digits.
    pub matrix: M,
}

impl<M> RoleParams<M> {
    /// Pair a decomposition with the matrix that consumes it.
    ///
    /// `const` so the generated tables can build a role in `static` position.
    #[must_use]
    pub const fn new(digits: GadgetDigits, matrix: M) -> Self {
        Self { digits, matrix }
    }
}

impl<M: MatrixDescriptorBytes> RoleParams<M> {
    /// Atomic descriptor encoding: `basis, depth, matrix`.
    ///
    /// Byte-neutral wherever the containing encoder already wrote those three in
    /// that order, which the commit-phase profile does for both of its roles.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.digits.append_descriptor_bytes(bytes);
        self.matrix.append_descriptor_bytes(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_geometry_accepts_the_exact_ceiling_split() {
        let geometry = BlockGeometry::new(96, 32, 3);
        geometry.validate().expect("exact split");
        assert_eq!(geometry.position_index_bits(), 5);
        // B = 3 pads to a 4-element Boolean domain.
        assert_eq!(geometry.block_index_bits(), 2);
        assert_eq!(geometry.block_index_domain_size().expect("domain"), 4);
    }

    #[test]
    fn block_geometry_rejects_a_partial_or_misshapen_split() {
        // B must be the exact ceiling, not merely large enough.
        assert!(BlockGeometry::new(96, 32, 4).validate().is_err());
        assert!(BlockGeometry::new(96, 32, 2).validate().is_err());
        // M must be a power of two.
        assert!(BlockGeometry::new(96, 24, 4).validate().is_err());
        // Nothing may be zero.
        assert!(BlockGeometry::new(0, 32, 1).validate().is_err());
        assert!(BlockGeometry::new(96, 0, 1).validate().is_err());
        assert!(BlockGeometry::new(96, 32, 0).validate().is_err());
    }

    #[test]
    fn block_geometry_rounds_a_ragged_tail_up() {
        let geometry = BlockGeometry::new(97, 32, 4);
        geometry.validate().expect("ragged tail rounds up");
        assert_eq!(geometry.block_index_bits(), 2);
    }

    #[test]
    fn block_index_bits_saturates_where_the_domain_size_errors() {
        // The historical accessor answers 0 rather than failing; the checked
        // accessor is the one that reports overflow.
        let geometry = BlockGeometry::new(usize::MAX, 1, usize::MAX);
        assert_eq!(geometry.block_index_bits(), 0);
        assert!(geometry.block_index_domain_size().is_err());
    }

    #[test]
    fn atomic_geometry_encoding_matches_three_field_writes() {
        let geometry = BlockGeometry::new(96, 32, 3);
        let mut atomic = Vec::new();
        geometry.append_descriptor_bytes(&mut atomic);
        let mut fields = Vec::new();
        push_usize(&mut fields, 96);
        push_usize(&mut fields, 32);
        push_usize(&mut fields, 3);
        assert_eq!(
            atomic, fields,
            "atomic geometry encoder must be byte-neutral"
        );
    }

    #[test]
    fn gadget_digits_bound_the_depth_by_the_field_width() {
        let width = compute_num_digits_field_width(128, 8);
        GadgetDigits::new(8, width)
            .validate(128)
            .expect("depth at the bound");
        assert!(GadgetDigits::new(8, width + 1).validate(128).is_err());
        assert!(GadgetDigits::new(8, 0).validate(128).is_err());
    }

    #[test]
    fn gadget_digits_reject_a_basis_with_no_kernel() {
        assert!(GadgetDigits::new(crate::MAX_I16_LOG_BASIS + 1, 1)
            .validate(128)
            .is_err());
    }
}
