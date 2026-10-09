use akita_challenges::BinaryChallengeProfile;
use akita_error::{checked, AkitaError};

use super::{
    checked_source_comparison_class_bound, labinius_min_secure_rank,
    labinius_small_modulus_min_secure_rank, LabiniusCoefficientPrime, LabiniusCommitmentModulus,
    LabiniusRingDegree, LabiniusRootProfile, LabiniusSourceComparisonId, SourceOccurrenceBound,
};

#[path = "foreign_modulus_lift.rs"]
mod foreign_modulus_lift;
pub use foreign_modulus_lift::LABINIUS_MIN_DERIVATION_BIAS_BITS;
use foreign_modulus_lift::{derive_foreign_modulus_lift, ForeignModulusLift};

#[path = "root_encoding.rs"]
mod root_encoding;
pub use root_encoding::{LabiniusDigitBase, LabiniusRootEncoding, LabiniusSignedDigitRange};

/// Fully derived and SIS-admitted geometry for a closed binary-root profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabiniusRootShape {
    profile: LabiniusRootProfile,
    coefficient_prime: LabiniusCoefficientPrime,
    response_interval: (i128, i128),
    num_cells: usize,
    fold_width: usize,
    scalars_per_column: usize,
    ring_elements_per_column: usize,
    packing_degree: usize,
    scalar_degree: usize,
    commitment_degree: usize,
    eta_a: u128,
    rank_a: u32,
    image_len: usize,
    parity_residual_bound: u128,
    honest_quotient_bound: u128,
    honest_carry_bound: u128,
    foreign_modulus_lift: Option<ForeignModulusLift>,
}

impl LabiniusRootShape {
    /// Derive the one shared prover/verifier/planner admission decision.
    pub fn derive(
        profile: LabiniusRootProfile,
        log_num_cells: u32,
        log_fold_width: u32,
        lambda_fold: u32,
    ) -> Result<Self, AkitaError> {
        derive_shape(
            profile,
            profile.coefficient_prime(),
            profile.commitment_modulus().small_modulus(),
            profile.ring_degree(),
            &profile.challenge_profile()?,
            profile.response_interval(),
            log_num_cells,
            log_fold_width,
            lambda_fold,
        )
    }

    /// Check the actual accepted quotient/carry envelopes, including range rounding.
    pub fn check_parity_no_wrap(
        &self,
        enforced_quotient_bound: u128,
        enforced_carry_bound: u128,
    ) -> Result<(), AkitaError> {
        if enforced_quotient_bound < self.honest_quotient_bound {
            return Err(AkitaError::InvalidSetup(
                "LaBinius parity quotient envelope is below the honest bound".into(),
            ));
        }
        if enforced_carry_bound < self.honest_carry_bound {
            return Err(AkitaError::InvalidSetup(
                "LaBinius parity carry envelope is below the honest bound".into(),
            ));
        }
        let total = parity_no_wrap_total(
            self.parity_residual_bound,
            enforced_quotient_bound,
            enforced_carry_bound,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("LaBinius parity no-wrap sum overflow".into()))?;
        if total >= self.coefficient_prime.modulus() {
            return Err(AkitaError::InvalidSetup(
                "LaBinius parity envelopes do not satisfy H + 3*B_Q + 2*B_K < P".into(),
            ));
        }
        Ok(())
    }

    /// Commitment modulus, independent of the opening coefficient prime.
    pub const fn commitment_modulus(&self) -> LabiniusCommitmentModulus {
        self.profile.commitment_modulus()
    }

    /// Honest unreduced A-row residual bound `H_A` for a foreign-modulus lift.
    pub fn a_residual_bound(&self) -> Option<u128> {
        self.foreign_modulus_lift
            .as_ref()
            .map(|lift| lift.residual_bound)
    }

    /// Honest A-row carry magnitude `B_KA`, when the primes differ.
    pub fn honest_a_carry_bound(&self) -> Option<u128> {
        self.foreign_modulus_lift
            .as_ref()
            .map(|lift| lift.honest_carry_bound)
    }

    /// A-row carry integers `n_A * D`, or zero when the primes coincide.
    pub fn a_carry_len(&self) -> usize {
        self.foreign_modulus_lift
            .as_ref()
            .map_or(0, |lift| lift.carry_len)
    }

    /// Largest `t` satisfying `n_A * m * D * q0 * 2^t <= 4P`.
    /// This is a bound on derivation bias, not total composed security.
    pub fn derivation_bias_bits(&self) -> Option<u32> {
        self.foreign_modulus_lift
            .as_ref()
            .map(|lift| lift.bias_bits)
    }

    /// Extracted A-row difference bound `B_nu` for the enforced carry bits.
    /// Rejects a range that omits either honest carry endpoint.
    pub fn a_carry_difference_bound(&self, enforced_bits: u32) -> Result<Option<u128>, AkitaError> {
        self.foreign_modulus_lift
            .as_ref()
            .map(|lift| lift.difference_bound(enforced_bits))
            .transpose()
    }

    /// Enforce honest carry endpoint inclusion and extraction no-wrap for the
    /// actual digit-base-rounded bit width. Shared-prime profiles need no lift.
    pub fn check_a_carry_no_wrap(&self, enforced_bits: u32) -> Result<(), AkitaError> {
        if let Some(lift) = &self.foreign_modulus_lift {
            lift.check_no_wrap(self.coefficient_prime.modulus(), enforced_bits)?;
        }
        Ok(())
    }

    /// Selected closed profile.
    pub const fn profile(&self) -> LabiniusRootProfile {
        self.profile
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
    /// Certified source-comparison collision bound `eta_A`.
    pub const fn eta_a(&self) -> u128 {
        self.eta_a
    }
    /// Minimum certified cryptographic module rank `n_A`.
    pub const fn rank_a(&self) -> u32 {
        self.rank_a
    }
    /// Image coefficient count `n_A * C * D`.
    pub const fn image_len(&self) -> usize {
        self.image_len
    }
    /// Direct parity residual coefficient bound `H`.
    pub const fn parity_residual_bound(&self) -> u128 {
        self.parity_residual_bound
    }
    /// Honest direct parity quotient magnitude `B_Q`.
    pub const fn honest_quotient_bound(&self) -> u128 {
        self.honest_quotient_bound
    }
    /// Honest direct parity carry magnitude `B_K`.
    pub const fn honest_carry_bound(&self) -> u128 {
        self.honest_carry_bound
    }
}

// Canonical domain derivation. Explicit fields also let tests exercise boundaries
// that the sole current closed profile cannot reach (not a public plugin seam).
#[allow(clippy::too_many_arguments)]
fn derive_shape(
    profile: LabiniusRootProfile,
    prime: LabiniusCoefficientPrime,
    small_modulus: Option<u32>,
    degree: LabiniusRingDegree,
    challenge: &BinaryChallengeProfile,
    interval: (i128, i128),
    log_num_cells: u32,
    log_fold_width: u32,
    lambda_fold: u32,
) -> Result<LabiniusRootShape, AkitaError> {
    let geometry_overflow =
        || AkitaError::InvalidSetup("LaBinius root geometry size overflow".into());
    let num_cells = checked::pow2(usize::try_from(log_num_cells).map_err(|_| geometry_overflow())?)
        .ok_or_else(geometry_overflow)?;
    let fold_width =
        checked::pow2(usize::try_from(log_fold_width).map_err(|_| geometry_overflow())?)
            .ok_or_else(geometry_overflow)?;
    let scalar_degree = challenge.scalar_ring().degree();
    let commitment_degree = usize::try_from(degree.degree()).map_err(|_| geometry_overflow())?;
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
    let (lower, upper) = interval;
    let diameter = upper
        .checked_sub(lower)
        .and_then(|delta| u128::try_from(delta).ok())
        .ok_or_else(|| {
            AkitaError::InvalidSetup("LaBinius response interval diameter overflow".into())
        })?;
    let gamma = u128::from(challenge.multiplication_linf_operator_bound());
    let honest_response = u128::from(columns).checked_mul(gamma).ok_or_else(|| {
        AkitaError::InvalidSetup("LaBinius honest response bound overflow".into())
    })?;
    let symmetric_cap = u128::try_from(upper)
        .ok()
        .zip(lower.checked_neg().and_then(|v| u128::try_from(v).ok()))
        .map(|(u, l)| u.min(l))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("LaBinius response interval must contain zero".into())
        })?;
    if honest_response > symmetric_cap {
        return Err(AkitaError::InvalidSetup(
            "LaBinius root honest response exceeds accepted interval".into(),
        ));
    }
    let occurrence = SourceOccurrenceBound::binary_extracted(gamma, diameter).ok_or_else(|| {
        AkitaError::InvalidSetup("LaBinius source occurrence bound overflow".into())
    })?;
    let eta_a = checked_source_comparison_class_bound(
        LabiniusSourceComparisonId {
            coefficient_prime: prime,
            ring_degree: degree,
            matrix_view_digest: [0; 32],
        },
        &[occurrence],
    )?;
    let width = u64::try_from(ring_elements_per_column).map_err(|_| geometry_overflow())?;
    let rank_a = if small_modulus.is_some() {
        labinius_small_modulus_min_secure_rank(profile.commitment_modulus(), degree, eta_a, width)
    } else {
        labinius_min_secure_rank(prime, degree, eta_a, width)
    }
    .ok_or_else(|| AkitaError::InvalidSetup("LaBinius root has no certified SIS cell".into()))?;
    let rank = usize::try_from(rank_a).map_err(|_| geometry_overflow())?;
    let image_len =
        checked::product([rank, fold_width, commitment_degree]).ok_or_else(geometry_overflow)?;
    let bv = lower.unsigned_abs().max(upper.unsigned_abs());
    let parity_overflow = || AkitaError::InvalidSetup("LaBinius parity bound overflow".into());
    let parity_residual_bound = u128::try_from(scalar_degree)
        .map_err(|_| parity_overflow())?
        .checked_mul(u128::try_from(scalars_per_column).map_err(|_| parity_overflow())?)
        .and_then(|dm| dm.checked_mul(bv))
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
    let mut shape = LabiniusRootShape {
        profile,
        coefficient_prime: prime,
        response_interval: interval,
        num_cells,
        fold_width,
        scalars_per_column,
        ring_elements_per_column,
        packing_degree,
        scalar_degree,
        commitment_degree,
        eta_a,
        rank_a,
        image_len,
        parity_residual_bound,
        honest_quotient_bound,
        honest_carry_bound,
        foreign_modulus_lift: None,
    };
    if let Some(q) = small_modulus {
        shape.foreign_modulus_lift = Some(derive_foreign_modulus_lift(&shape, q, challenge)?);
    }
    Ok(shape)
}

// One shared accepted-envelope formula for admission and exposed range totals.
fn parity_no_wrap_total(h: u128, q: u128, k: u128) -> Option<u128> {
    h.checked_add(q.checked_mul(3)?)?
        .checked_add(k.checked_mul(2)?)
}

#[cfg(test)]
#[path = "parity_tests.rs"]
mod parity_tests;
#[cfg(test)]
#[path = "root_shape_tests.rs"]
mod tests;
