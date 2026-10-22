//! The residual witness a zero-knowledge opening hands the Σ-protocol at its
//! cut (`specs/whiteout-akita.md` §4.6, §4.8.1).

use akita_error::AkitaError;
use akita_params::CommittedGroupParams;
use jolt_algebra::{CanonicalEncoding, Field};
use zeroize::Zeroizing;

/// The cut successor's committed witness, its inner digits and its hiding
/// randomness, read out of the backend's handles.
///
/// The orders are the residual commitment's column semantics (jolt-crypto's
/// `TwoLevelLayout`), for one outer slice:
///
/// - `w_k`: w_k's digits in Akita's flat order, ring after ring; block `i`
///   is rings `i·M..(i+1)·M` for `M` positions per block;
/// - `t_hat`: the outer input digits of `t = A·G⁻¹(w_k)`, ordered
///   `[block][A row][digit][coefficient]` (d_A = d_B, so one subcolumn);
/// - `r_a`: the B hiding randomness, which follows `t_hat` in `[B | B′]`'s
///   input.
pub struct ResidualWitness {
    pub w_k: Zeroizing<Vec<i8>>,
    pub t_hat: Zeroizing<Vec<i8>>,
    pub r_a: Zeroizing<Vec<i8>>,
}

/// Exports the cut successor's [`ResidualWitness`].
pub trait ResidualExportKernel<F, E>: super::ProverHandleFamily<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    /// The residual witness of `witness`, committed with `material` under
    /// `params`, the cut successor's parameters.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] if the handles do not belong to one
    /// one-slice commitment under `params`.
    fn export_residual(
        &self,
        witness: &Self::WitnessHandle,
        material: &Self::CommitmentMaterialHandle,
        params: &CommittedGroupParams,
    ) -> Result<ResidualWitness, AkitaError>;
}
