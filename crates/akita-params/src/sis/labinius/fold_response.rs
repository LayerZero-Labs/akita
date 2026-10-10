//! Fold-response admission for the binary fold, sized by Akita's fold cap.
//!
//! One fold sends `z = sum_{i < C} c_i * s_i`. Each `s_i` is a binary column
//! of `m` commitment-ring elements of degree `D`, and each `c_i` is a signed
//! scalar-ring challenge of weight at most `w`. This module derives no tail
//! bound of its own. It states that fold in the inputs of Akita's cap:
//!
//! - challenge family `SparseChallengeConfig::pm1_only(w)`, so
//!   `max ||c||_2^2 = ||c||_1 = w` and `||c||_inf = 1`;
//! - `C` folded blocks of one claim;
//! - `m * D` response coefficients in the union bound;
//! - source norms `||s||_inf = Gamma / w` and `||s||_1 = (Gamma / w) * D`,
//!   where `Gamma` is the profile's multiplication bound: one signed challenge
//!   monomial maps a ring element with coefficients of magnitude at most one
//!   to coefficients of magnitude at most `Gamma / w = 2`. Scalar components
//!   that are binary in the power basis reach only magnitude one, so the cap
//!   is conservative for them.
//!
//! [`fold_witness_linf_cap`] then returns `min(beta_inf, t*)` with
//! `beta_inf = C * Gamma` and `t*^2 = 2 * C * w * (Gamma / w)^2 * ln_term`.
//! [`num_digits_for_linf_cap`] selects the balanced digit depth, whose exact
//! digit range is the accepted interval, and
//! [`role_a_collision_inf_norm_for_response_difference`] prices two accepted
//! responses under `Gamma`, giving `eta_A = 4 * Gamma * Delta`.
//!
//! Akita's `BalancedSignedDigitFoldPolicy` composes the same three calls, but
//! its query validation admits only the power-of-two production ring degrees,
//! so they are called here directly.
//!
//! Soundness uses only the enforced digit interval. The tail threshold is a
//! completeness bound for the honest prover, and it models the signs of the
//! challenge coefficients as independent and uniform given their supports.
//!
//! The response search is Akita's fold-response run, a bounded rejection
//! search with no proof-of-work target; its nonce width is
//! [`FOLD_RESPONSE_NONCE_BITS`](crate::FOLD_RESPONSE_NONCE_BITS).

use akita_challenges::{BinaryChallengeProfile, SparseChallengeConfig};
use akita_error::{checked, AkitaError};

use super::LabiniusRingDegree;
use crate::sis::{
    balanced_digit_interval_diameter, checked_balanced_digit_representable_bounds,
    fold_witness_linf_cap, num_digits_for_linf_cap,
    role_a_collision_inf_norm_for_response_difference, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN,
    FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM,
};

/// Bits per balanced digit of the response and image encodings: base 16.
pub const LABINIUS_BALANCED_LOG_BASIS: u32 = 4;

/// Admission data for one binary fold response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabiniusFoldResponse {
    honest_cap: u128,
    tail_threshold: u128,
    digit_count: usize,
    interval: (i128, i128),
    diameter: u128,
    eta_a: u128,
    abort_probability_bound: (u32, u32),
}

impl LabiniusFoldResponse {
    /// Derive the response admission for `fold_columns` binary columns of
    /// `matrix_width` ring elements each.
    pub fn derive(
        challenge: &BinaryChallengeProfile,
        fold_columns: usize,
        matrix_width: usize,
        ring_degree: LabiniusRingDegree,
    ) -> Result<Self, AkitaError> {
        let overflow = || AkitaError::InvalidSetup("LaBinius fold response bound overflow".into());
        let degree = usize::try_from(ring_degree.degree()).map_err(|_| overflow())?;
        if checked::exact_div(degree, challenge.scalar_ring().degree()).is_none() {
            return Err(AkitaError::InvalidSetup(
                "LaBinius fold response scalar/commitment degree mismatch".into(),
            ));
        }
        let sparse = SparseChallengeConfig::pm1_only(challenge.weight_cap());
        let norms = FoldChallengeNorms::new(&sparse);
        let gamma = u128::from(challenge.multiplication_linf_operator_bound());
        let monomial_gain = gamma
            .checked_div(norms.l1_norm)
            .filter(|gain| gain.checked_mul(norms.l1_norm) == Some(gamma))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("LaBinius fold response needs a nonzero challenge".into())
            })?;
        let source_l1 = u128::try_from(degree)
            .ok()
            .and_then(|degree| monomial_gain.checked_mul(degree))
            .ok_or_else(overflow)?;
        let response_coefficients =
            checked::product([matrix_width, degree]).ok_or_else(overflow)?;
        let (honest_cap, tail_threshold) = fold_witness_linf_cap(
            fold_columns,
            1,
            norms,
            FoldWitnessNorms::new(monomial_gain, source_l1),
            &FoldWitnessLinfCapConfig::for_fold_coeffs(&sparse, response_coefficients)?,
        )?;
        let digit_count =
            num_digits_for_linf_cap(honest_cap, u128::BITS, LABINIUS_BALANCED_LOG_BASIS);
        let (negative_reach, positive_reach) =
            checked_balanced_digit_representable_bounds(LABINIUS_BALANCED_LOG_BASIS, digit_count);
        // A cap at the `u128` field width falls back to a depth that may not
        // reach it; the accepted interval must contain every honest response.
        let positive_reach = positive_reach
            .filter(|&reach| reach >= honest_cap)
            .ok_or_else(overflow)?;
        let interval = negative_reach
            .and_then(|reach| i128::try_from(reach).ok())
            .zip(i128::try_from(positive_reach).ok())
            .map(|(negative, positive)| (-negative, positive))
            .ok_or_else(overflow)?;
        let diameter = balanced_digit_interval_diameter(LABINIUS_BALANCED_LOG_BASIS, digit_count);
        let eta_a = role_a_collision_inf_norm_for_response_difference(gamma, diameter)
            .filter(|_| diameter != u128::MAX)
            .ok_or_else(overflow)?;
        // Below the tail threshold the cap is the deterministic `beta_inf`,
        // which no honest response exceeds.
        let abort_numerator = if honest_cap < tail_threshold {
            0
        } else {
            FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN - FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM
        };
        Ok(Self {
            honest_cap,
            tail_threshold,
            digit_count,
            interval,
            diameter,
            eta_a,
            abort_probability_bound: (abort_numerator, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN),
        })
    }

    /// Akita's honest fold cap `min(beta_inf, t*)`.
    pub const fn honest_cap(&self) -> u128 {
        self.honest_cap
    }

    /// Akita's tail threshold `t*` for this fold.
    pub const fn tail_threshold(&self) -> u128 {
        self.tail_threshold
    }

    /// Balanced base-16 digits of one response coefficient.
    pub const fn digit_count(&self) -> usize {
        self.digit_count
    }

    /// Inclusive accepted coefficient interval: the exact balanced digit range.
    pub const fn interval(&self) -> (i128, i128) {
        self.interval
    }

    /// Diameter `Delta` of the accepted interval.
    pub const fn diameter(&self) -> u128 {
        self.diameter
    }

    /// Extracted commitment-relation bound `eta_A = 4 * Gamma * Delta`.
    pub const fn eta_a(&self) -> u128 {
        self.eta_a
    }

    /// Upper bound `(numerator, denominator)` on the probability that one
    /// honest attempt leaves the accepted interval.
    pub const fn abort_probability_bound(&self) -> (u32, u32) {
        self.abort_probability_bound
    }
}

#[cfg(test)]
#[path = "fold_response_tests.rs"]
mod tests;
