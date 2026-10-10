use akita_challenges::BinaryChallengeProfile;
use akita_error::{checked, AkitaError};

use super::{
    labinius_min_secure_rank, LabiniusCommitmentLift, LabiniusFoldResponse, LabiniusRootProfile,
};

#[path = "root_encoding.rs"]
mod root_encoding;
pub use root_encoding::{LabiniusRootEncoding, LabiniusSignedRange};

/// Provisional policy floor for the reduced-matrix derivation hybrid bias.
/// This is not a derived requirement or a claim about composed security.
pub const LABINIUS_MIN_DERIVATION_BIAS_BITS: u32 = 64;

/// Fully derived and SIS-admitted geometry for a closed binary-root profile.
///
/// The shape depends on the commitment prime only. A proof prime is admitted
/// against it by [`Self::derive_encoding`], which is the only source of the
/// clear-integer ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabiniusRootShape {
    profile: LabiniusRootProfile,
    response: LabiniusFoldResponse,
    num_cells: usize,
    fold_width: usize,
    scalars_per_column: usize,
    ring_elements_per_column: usize,
    packing_degree: usize,
    scalar_degree: usize,
    commitment_degree: usize,
    rank_a: u32,
    parity_residual_bound: u128,
    honest_quotient_bound: u128,
    honest_carry_bound: u128,
    honest_a_carry_bound: u128,
}

impl LabiniusRootShape {
    /// Derive the one shared prover/verifier/planner admission decision.
    ///
    /// The response interval, its digit count and the extracted bound `eta_A`
    /// come from [`LabiniusFoldResponse::derive`]; the rank is the smallest one
    /// whose certified cell dominates `eta_A` and the matrix width. A geometry
    /// whose response needs more digits than a certified cell covers has no
    /// such cell and is rejected here.
    pub fn derive(
        profile: LabiniusRootProfile,
        log_num_cells: u32,
        log_fold_width: u32,
        lambda_fold: u32,
    ) -> Result<Self, AkitaError> {
        let geometry_overflow =
            || AkitaError::InvalidSetup("LaBinius root geometry size overflow".into());
        let challenge = profile.challenge_profile()?;
        let degree = profile.ring_degree();
        let num_cells =
            checked::pow2(usize::try_from(log_num_cells).map_err(|_| geometry_overflow())?)
                .ok_or_else(geometry_overflow)?;
        let fold_width =
            checked::pow2(usize::try_from(log_fold_width).map_err(|_| geometry_overflow())?)
                .ok_or_else(geometry_overflow)?;
        let scalar_degree = challenge.scalar_ring().degree();
        let commitment_degree =
            usize::try_from(degree.degree()).map_err(|_| geometry_overflow())?;
        let packing_degree = checked::exact_div(commitment_degree, scalar_degree)
            .filter(|&k| k > 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("LaBinius root scalar/commitment degree mismatch".into())
            })?;
        let scalars_per_column = checked::exact_div(num_cells, fold_width);
        let ring_elements_per_column = scalars_per_column
            .and_then(|m| checked::exact_div(m, packing_degree))
            .filter(|&m| m > 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("LaBinius root M must be a positive multiple of k".into())
            })?;
        let scalars_per_column = scalars_per_column.ok_or_else(geometry_overflow)?;
        let columns = u64::try_from(fold_width).map_err(|_| geometry_overflow())?;
        if !challenge.meets_budget(columns, lambda_fold) {
            return Err(AkitaError::InvalidSetup(
                "LaBinius root challenge profile does not meet fold budget".into(),
            ));
        }
        let response =
            LabiniusFoldResponse::derive(&challenge, fold_width, ring_elements_per_column, degree)?;
        let width = u64::try_from(ring_elements_per_column).map_err(|_| geometry_overflow())?;
        let rank_a = labinius_min_secure_rank(
            profile.commitment_modulus(),
            degree,
            response.eta_a(),
            width,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("LaBinius root has no certified SIS cell".into())
        })?;
        let parity_overflow = || AkitaError::InvalidSetup("LaBinius parity bound overflow".into());
        let parity_residual_bound = u128::try_from(scalar_degree)
            .map_err(|_| parity_overflow())?
            .checked_mul(u128::try_from(scalars_per_column).map_err(|_| parity_overflow())?)
            .and_then(|dm| dm.checked_mul(response_bound(&response)))
            .and_then(|v| {
                u128::from(columns)
                    .checked_mul(u128::from(challenge.coefficient_l1_bound()))?
                    .checked_add(v)
            })
            .ok_or_else(parity_overflow)?;
        let honest_quotient_bound = parity_residual_bound
            .checked_mul(2)
            .ok_or_else(parity_overflow)?;
        let honest_carry_bound = parity_residual_bound
            .checked_mul(3)
            .ok_or_else(parity_overflow)?
            / 2;
        let mut shape = Self {
            profile,
            response,
            num_cells,
            fold_width,
            scalars_per_column,
            ring_elements_per_column,
            packing_degree,
            scalar_degree,
            commitment_degree,
            rank_a,
            parity_residual_bound,
            honest_quotient_bound,
            honest_carry_bound,
            honest_a_carry_bound: 0,
        };
        shape.honest_a_carry_bound = shape.commitment_lift(&challenge).honest_carry_bound()?;
        Ok(shape)
    }

    /// The commitment relation of this shape, for its lift to a proof prime.
    fn commitment_lift<'a>(
        &self,
        challenge: &'a BinaryChallengeProfile,
    ) -> LabiniusCommitmentLift<'a> {
        LabiniusCommitmentLift {
            challenge,
            fold_columns: self.fold_width,
            matrix_width: self.ring_elements_per_column,
            ring_degree: self.profile.ring_degree(),
            commitment_modulus: self.profile.commitment_modulus().modulus(),
            response_bound: response_bound(&self.response),
        }
    }

    /// Admit the prime of the field-element stream the matrix is derived from.
    ///
    /// Each of the `n_A * m * D` matrix coefficients is a uniform element below
    /// `stream_prime` reduced modulo `q`. The check requires
    /// `n_A * m * D * q * 2^t <= 4 * stream_prime` at the policy floor
    /// `t = LABINIUS_MIN_DERIVATION_BIAS_BITS`. It bounds derivation bias only.
    pub fn check_derivation_bias(&self, stream_prime: u128) -> Result<(), AkitaError> {
        let overflow = || AkitaError::InvalidSetup("LaBinius derivation bias overflow".into());
        let rank = usize::try_from(self.rank_a).map_err(|_| overflow())?;
        let coefficients =
            checked::product([rank, self.ring_elements_per_column, self.commitment_degree])
                .and_then(|count| u128::try_from(count).ok())
                .and_then(|count| {
                    count.checked_mul(u128::from(self.profile.commitment_modulus().modulus()))
                })
                .ok_or_else(overflow)?;
        // `4 * stream_prime / 2^t` without leaving `u128`.
        let limit = LABINIUS_MIN_DERIVATION_BIAS_BITS
            .checked_sub(2)
            .and_then(|shift| stream_prime.checked_shr(shift))
            .ok_or_else(overflow)?;
        if coefficients > limit {
            return Err(AkitaError::InvalidSetup(
                "LaBinius reduced derivation bias is below policy floor".into(),
            ));
        }
        Ok(())
    }

    /// Selected closed profile.
    pub const fn profile(&self) -> LabiniusRootProfile {
        self.profile
    }
    /// Fold-response admission: accepted interval, digit count and `eta_A`.
    pub const fn response(&self) -> &LabiniusFoldResponse {
        &self.response
    }
    /// Source scalar cells `N`.
    pub const fn num_cells(&self) -> usize {
        self.num_cells
    }
    /// Columns/folding challenges `C`.
    pub const fn fold_width(&self) -> usize {
        self.fold_width
    }
    /// Scalar entries per column `M`.
    pub const fn scalars_per_column(&self) -> usize {
        self.scalars_per_column
    }
    /// Ring entries per column `m`.
    pub const fn ring_elements_per_column(&self) -> usize {
        self.ring_elements_per_column
    }
    /// Scalar components per ring element `k`.
    pub const fn packing_degree(&self) -> usize {
        self.packing_degree
    }
    /// Scalar polynomial degree `d`.
    pub const fn scalar_degree(&self) -> usize {
        self.scalar_degree
    }
    /// Commitment polynomial degree `D`.
    pub const fn commitment_degree(&self) -> usize {
        self.commitment_degree
    }
    /// Minimum certified cryptographic module rank `n_A`.
    pub const fn rank_a(&self) -> u32 {
        self.rank_a
    }
    /// Direct parity residual coefficient bound `H`.
    pub const fn parity_residual_bound(&self) -> u128 {
        self.parity_residual_bound
    }
    /// Honest direct parity quotient magnitude `B_Q = 2 * H`.
    pub const fn honest_quotient_bound(&self) -> u128 {
        self.honest_quotient_bound
    }
    /// Honest direct parity carry magnitude `B_K = floor(3 * H / 2)`.
    pub const fn honest_carry_bound(&self) -> u128 {
        self.honest_carry_bound
    }
    /// Honest commitment-row carry magnitude `B_KA`.
    pub const fn honest_a_carry_bound(&self) -> u128 {
        self.honest_a_carry_bound
    }
}

/// Enforced response coefficient magnitude: the larger absolute endpoint.
fn response_bound(response: &LabiniusFoldResponse) -> u128 {
    let (lower, upper) = response.interval();
    lower.unsigned_abs().max(upper.unsigned_abs())
}

#[cfg(test)]
#[path = "parity_tests.rs"]
mod parity_tests;
#[cfg(test)]
#[path = "root_shape_tests.rs"]
mod tests;
