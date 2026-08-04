//! Ring fold challenge configuration for [`crate::SparseChallenge`].
//!
//! Witness-fold challenges are fixed-weight sparse polynomials: `count_pm1`
//! coefficients with magnitude 1 and `count_pm2` with magnitude 2, each with
//! random sign. When `count_pm2 == 0` every non-zero coefficient is ±1; when
//! `count_pm2 > 0` some coefficients are ±2 (production D=64).
//!
//! The actual sampler lives in [`crate::sampler`]; this file is policy-only.

/// Minimum min-entropy (bits) for every ring fold sparse-challenge transcript draw.
///
/// Flat folds sample one such draw per logical block. Tensor folds sample
/// independent fold-high and fold-low factor vectors; each factor is one draw
/// and is reused across many logical blocks
/// (`c_{p,q} = fold_high_p · fold_low_q`). Soundness therefore requires
/// **each draw** to clear this floor, not merely the product
/// `fold_high ⊗ fold_low` summed to 128 bits (a 64+64 split would pass a sum
/// rule but leave each factor brute-forceable).
pub const MIN_FOLD_CHALLENGE_ENTROPY_BITS: u32 = 128;

/// Production D=64 signed sparse ±1 count (LaBRADOR-aligned).
pub const D64_PRODUCTION_PM1_COUNT: usize = 31;
/// Production D=64 signed sparse ±2 count (LaBRADOR-aligned).
pub const D64_PRODUCTION_PM2_COUNT: usize = 10;

/// Ring degrees with a production fold-challenge ladder entry.
macro_rules! production_fold_challenge_ring_dims {
    ($($dim:literal),+ $(,)?) => {
        pub const PRODUCTION_FOLD_CHALLENGE_RING_DIMS: &[usize] = &[$($dim),+];

        macro_rules! __dispatch_fold_challenge_ring_dim {
            ($self:expr, $d:expr, $required_bits:expr) => {
                match $d {
                    $( $dim => $self.validate_min_entropy::<$dim>($required_bits), )+
                    _ => Err("unsupported ring dimension for fold-challenge entropy audit"),
                }
            };
        }
    };
}

production_fold_challenge_ring_dims!(64, 128, 256, 512, 1024, 2048);

const PRODUCTION_FOLD_CHALLENGE_LADDER: &[(usize, usize, usize)] = &[
    (64, D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT),
    (128, 31, 0),
    (256, 23, 0),
    (512, 19, 0),
    (1024, 16, 0),
    (2048, 14, 0),
];

/// Fixed-weight sparse ring fold challenge family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SparseChallengeConfig {
    /// Number of non-zero coefficients with magnitude 1 (random sign).
    pub count_pm1: usize,
    /// Number of non-zero coefficients with magnitude 2 (random sign).
    pub count_pm2: usize,
}

impl SparseChallengeConfig {
    /// ±1-only sparse family with Hamming weight `count_pm1`.
    #[inline]
    #[must_use]
    pub const fn pm1_only(count_pm1: usize) -> Self {
        Self {
            count_pm1,
            count_pm2: 0,
        }
    }

    /// Production ladder entry for ring degree `ring_d`, if defined.
    #[inline]
    #[must_use]
    pub fn production_for_ring_dim(ring_d: usize) -> Option<Self> {
        PRODUCTION_FOLD_CHALLENGE_LADDER
            .iter()
            .find(|(d, _, _)| *d == ring_d)
            .map(|(_, pm1, pm2)| Self {
                count_pm1: *pm1,
                count_pm2: *pm2,
            })
    }

    /// Whether this config matches the production ladder at `ring_d`.
    #[inline]
    #[must_use]
    pub fn matches_production_ladder(&self, ring_d: usize) -> bool {
        Self::production_for_ring_dim(ring_d).as_ref() == Some(self)
    }

    /// Total Hamming weight.
    #[inline]
    #[must_use]
    pub fn weight(&self) -> usize {
        self.count_pm1.saturating_add(self.count_pm2)
    }

    /// Worst-case `L1` norm of the sampled coefficients.
    #[inline]
    #[must_use]
    pub fn l1_norm(&self) -> usize {
        self.count_pm1
            .saturating_add(2usize.saturating_mul(self.count_pm2))
    }

    /// Worst-case squared ℓ₂ norm `max ‖c‖_2²` over the challenge family.
    #[inline]
    #[must_use]
    pub fn challenge_l2_sq_max(&self) -> u128 {
        (self.count_pm1 as u128).saturating_add(4u128.saturating_mul(self.count_pm2 as u128))
    }

    /// Worst-case number of non-zero coefficients in one sampled challenge.
    #[inline]
    #[must_use]
    pub fn nonzero_count_max(&self) -> usize {
        self.weight()
    }

    /// Worst-case `L_infinity` norm of the sampled coefficients.
    #[inline]
    #[must_use]
    pub fn infinity_norm(&self) -> u32 {
        if self.count_pm2 > 0 {
            2
        } else {
            1
        }
    }

    /// `log2` of the number of distinct challenges this family can emit for ring
    /// degree `D` — the (raw) min-entropy of a single sampled challenge.
    pub fn log2_support_bits<const D: usize>(&self) -> f64 {
        fn log2_binom(n: usize, k: usize) -> f64 {
            if k > n {
                return f64::NEG_INFINITY;
            }
            (1..=k)
                .map(|i| ((n - k + i) as f64 / i as f64).log2())
                .sum()
        }
        let w = self.weight();
        if w > D {
            return f64::NEG_INFINITY;
        }
        log2_binom(D, w) + log2_binom(w, self.count_pm1) + w as f64
    }

    /// Reject challenge families whose single-draw support is below
    /// `required_bits` of min-entropy for ring degree `D`.
    pub fn validate_min_entropy<const D: usize>(
        &self,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        if self.log2_support_bits::<D>() < f64::from(required_bits) {
            return Err("sparse challenge family has insufficient min-entropy for security floor");
        }
        Ok(())
    }

    /// Runtime ring-dimension dispatch for [`Self::validate_min_entropy`].
    pub fn validate_min_entropy_for_ring_dim(
        &self,
        ring_dim: usize,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        __dispatch_fold_challenge_ring_dim!(self, ring_dim, required_bits)
    }

    /// Structural invariants plus the 128-bit entropy floor at `ring_d`.
    pub fn validate_for_ring_dim(&self, ring_d: usize) -> Result<(), &'static str> {
        self.validate_dyn(ring_d)?;
        self.validate_min_entropy_for_ring_dim(ring_d, MIN_FOLD_CHALLENGE_ENTROPY_BITS)
    }

    /// Canonical byte encoding used for transcript domain separation.
    #[inline]
    pub fn domain_separator_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 16);
        out.push(0);
        out.extend_from_slice(&(self.count_pm1 as u64).to_le_bytes());
        out.extend_from_slice(&(self.count_pm2 as u64).to_le_bytes());
        out
    }

    /// Validate basic invariants for a given ring degree `D`.
    pub fn validate<const D: usize>(&self) -> Result<(), &'static str> {
        self.validate_dyn(D)
    }

    /// Runtime ring-dimension form of [`Self::validate`].
    pub fn validate_dyn(&self, ring_d: usize) -> Result<(), &'static str> {
        if self
            .count_pm1
            .checked_add(self.count_pm2)
            .is_none_or(|w| w > ring_d)
        {
            return Err("count_pm1 + count_pm2 must be <= ring degree D");
        }
        Ok(())
    }
}

#[cfg(test)]
mod entropy_tests {
    use super::*;

    #[test]
    fn production_ladder_matches_proof_optimized_dims() {
        for &d in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            let cfg = SparseChallengeConfig::production_for_ring_dim(d).expect("ladder entry");
            assert!(cfg.validate_for_ring_dim(d).is_ok(), "d={d}");
        }
    }

    #[test]
    fn tiny_shell_is_rejected_at_128_bits() {
        let tiny = SparseChallengeConfig::pm1_only(2);
        assert!(tiny.log2_support_bits::<32>() < 128.0);
        assert!(tiny.validate_min_entropy::<32>(128).is_err());
    }

    #[test]
    fn production_shell_clears_128_bits() {
        let shell = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        assert!(shell.log2_support_bits::<64>() >= 128.0);
        assert!(shell.validate_for_ring_dim(64).is_ok());
    }

    #[test]
    fn tensor_floor_is_per_draw_not_product_budget() {
        let weak = SparseChallengeConfig::pm1_only(1);
        let per_draw = weak.log2_support_bits::<4>();
        assert!(per_draw < 128.0);
        assert!(weak.validate_min_entropy::<4>(128).is_err());
    }

    #[test]
    fn log2_support_matches_small_closed_form() {
        let cfg = SparseChallengeConfig::pm1_only(1);
        assert!((cfg.log2_support_bits::<4>() - 3.0).abs() < 1e-9);
        let uni = SparseChallengeConfig::pm1_only(2);
        assert!((uni.log2_support_bits::<4>() - 24.0_f64.log2()).abs() < 1e-9);
    }

    #[test]
    fn challenge_l2_sq_max_matches_spec_table() {
        let shell = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        assert_eq!(shell.l1_norm(), 51);
        assert_eq!(shell.challenge_l2_sq_max(), 71);
        assert_eq!(shell.nonzero_count_max(), 41);

        let uni128 = SparseChallengeConfig::pm1_only(31);
        assert_eq!(uni128.challenge_l2_sq_max(), 31);
        assert_eq!(uni128.nonzero_count_max(), 31);

        let uni256 = SparseChallengeConfig::pm1_only(23);
        assert_eq!(uni256.challenge_l2_sq_max(), 23);
        assert_eq!(uni256.nonzero_count_max(), 23);

        for (d, pm1, pm2) in PRODUCTION_FOLD_CHALLENGE_LADDER {
            if *d >= 512 {
                let cfg = SparseChallengeConfig {
                    count_pm1: *pm1,
                    count_pm2: *pm2,
                };
                assert!(cfg.validate_for_ring_dim(*d).is_ok());
            }
        }
    }
}
