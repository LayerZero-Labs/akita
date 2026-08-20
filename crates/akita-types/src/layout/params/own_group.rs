//! Accessors for a fold's own new group.
//!
//! Step 7 made `groups` the only place a fold's groups are stored, so what were
//! flat fields on [`CommittedGroupParams`] are now reads of the last entry.
//! They live in their own module because `params.rs` is at its line cap.

use super::precommitted::GroupOpenPhaseParams;
use super::CommittedGroupParams;
use crate::OpeningMethod;
use akita_challenges::SparseChallengeConfig;

impl CommittedGroupParams {
    /// This fold's own new group: the last entry in `groups`.
    ///
    /// Every constructor pushes it last, so this is a construction invariant.
    #[inline]
    #[must_use]
    pub fn own_group(&self) -> &GroupOpenPhaseParams {
        self.groups
            .last()
            .expect("a fold always owns its own new group")
    }

    /// Mutable view of this fold's own new group.
    #[inline]
    pub fn own_group_mut(&mut self) -> &mut GroupOpenPhaseParams {
        self.groups
            .last_mut()
            .expect("a fold always owns its own new group")
    }

    /// A/source role of this fold's own new group.
    #[inline]
    #[must_use]
    pub fn inner(&self) -> &crate::InnerRoleParams {
        &self.own_group().profile.inner
    }

    /// B/outer role of this fold's own new group.
    #[inline]
    #[must_use]
    pub fn outer(&self) -> &crate::OuterRoleParams {
        &self.own_group().profile.outer
    }

    /// Outer slice count of this fold's own new group.
    #[inline]
    #[must_use]
    pub fn outer_slice_count(&self) -> crate::CommitmentSliceCount {
        self.own_group().profile.outer_slice_count
    }

    /// Opening procedure of this fold's own new group.
    #[inline]
    #[must_use]
    pub fn opening_method(&self) -> OpeningMethod {
        self.own_group().opening.opening_method
    }

    /// Fold-challenge family of this fold's own new group.
    #[inline]
    #[must_use]
    pub fn fold_challenge_config(&self) -> SparseChallengeConfig {
        self.own_group().opening.fold_challenge_config
    }

    /// Polynomial layout of this fold's own new group.
    #[inline]
    #[must_use]
    pub fn group(&self) -> crate::PolynomialGroupLayout {
        self.own_group().profile.group
    }

    /// Shared opening role: the fold's D matrix with this group's own digits.
    #[inline]
    #[must_use]
    pub fn open(&self) -> crate::OpenRoleParams {
        crate::RoleParams::new(
            crate::GadgetDigits::new(
                self.own_group().opening.log_basis_open,
                self.own_group().opening.num_digits_open,
            ),
            self.open_matrix,
        )
    }
}
