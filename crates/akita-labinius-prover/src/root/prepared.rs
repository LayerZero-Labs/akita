use akita_algebra::{SmoothFftField, TrinomialModulus};
use akita_error::AkitaError;
use akita_labinius_verifier::profile::BinaryClearSetup;

use crate::commit_kernel::PreparedCommitMatrix;

/// Own-ring matrix transform cache for commitments and root reduction weights.
///
/// This cache is bound to the matrix digest and independent of the source,
/// the host field, and the digit base. Tag-1 carry construction also uses it.
pub struct PreparedRootMatrices<F, const D: usize, M: TrinomialModulus> {
    commit: PreparedCommitMatrix<F, D, M>,
}

impl<F: SmoothFftField, const D: usize, M: TrinomialModulus> PreparedRootMatrices<F, D, M> {
    /// Transform every matrix entry once in its own ring.
    pub fn prepare(setup: &BinaryClearSetup<F, D, M>) -> Result<Self, AkitaError> {
        Ok(Self {
            commit: PreparedCommitMatrix::prepare(setup)?,
        })
    }

    /// Reject a setup whose matrix differs from the cached one.
    pub(crate) fn check_setup(&self, setup: &BinaryClearSetup<F, D, M>) -> Result<(), AkitaError> {
        self.commit.check_setup(setup)
    }

    /// Borrow the cache used for commitments, response weights, and carries.
    pub fn commit(&self) -> &PreparedCommitMatrix<F, D, M> {
        &self.commit
    }
}
