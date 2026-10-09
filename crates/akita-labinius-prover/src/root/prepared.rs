use akita_algebra::SmoothFftField;
use akita_error::AkitaError;
use akita_labinius_verifier::profile::BinaryClearSetup;

use crate::{
    commit_kernel::PreparedCommitMatrix,
    quotient_kernel::{ConjugateModulus, PreparedQuotientMatrix},
};

/// Transform caches of one setup matrix for committing and root proving.
///
/// The matrix is transformed once in its own ring, for commitments and the
/// A-relation remainder, and once in the conjugate ring, for the A-relation
/// quotient. Both caches are bound to the matrix digest and are independent of
/// the source, the host field and the digit base.
pub struct PreparedRootMatrices<F, const D: usize, M: ConjugateModulus> {
    commit: PreparedCommitMatrix<F, D, M>,
    quotient: PreparedQuotientMatrix<F, D, M>,
}

impl<F: SmoothFftField, const D: usize, M: ConjugateModulus> PreparedRootMatrices<F, D, M> {
    /// Transform every matrix entry once in each ring.
    pub fn prepare(setup: &BinaryClearSetup<F, D, M>) -> Result<Self, AkitaError> {
        Ok(Self {
            commit: PreparedCommitMatrix::prepare(setup)?,
            quotient: PreparedQuotientMatrix::prepare(setup)?,
        })
    }

    /// Reject a setup whose matrix differs from the one either cache came from.
    pub(crate) fn check_setup(&self, setup: &BinaryClearSetup<F, D, M>) -> Result<(), AkitaError> {
        self.commit.check_setup(setup)?;
        self.quotient.check_setup(setup)
    }

    /// Borrow the cache that `commit_binary_clear_prepared` consumes.
    pub fn commit(&self) -> &PreparedCommitMatrix<F, D, M> {
        &self.commit
    }

    /// Borrow the conjugate-ring cache of the A-relation quotient.
    pub fn quotient(&self) -> &PreparedQuotientMatrix<F, D, M> {
        &self.quotient
    }
}
