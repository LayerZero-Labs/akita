use std::sync::LazyLock;

use akita_error::AkitaError;
use akita_transcript::FOLD_CHALLENGE_SEED_LEN;
use num_bigint::BigUint;

use super::combinatorics::BinomialCdfRow;

static CYCLOTOMIC_243_COUNTS: LazyLock<BinomialCdfRow> =
    LazyLock::new(|| BinomialCdfRow::new(BinaryScalarRing::Cyclotomic243.degree()));
static CYCLOTOMIC_729_COUNTS: LazyLock<BinomialCdfRow> =
    LazyLock::new(|| BinomialCdfRow::new(BinaryScalarRing::Cyclotomic729.degree()));
const FOLD_ROOT_ENTROPY_BITS: u32 = (FOLD_CHALLENGE_SEED_LEN * 8) as u32;

fn required_budget(fold_width: u64, lambda_fold: u32) -> Option<BigUint> {
    if fold_width == 0 || lambda_fold > FOLD_ROOT_ENTROPY_BITS {
        return None;
    }
    let required = BigUint::from(fold_width) << lambda_fold;
    (required <= (BigUint::from(1u8) << FOLD_ROOT_ENTROPY_BITS)).then_some(required)
}

/// Supported inert binary scalar rings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryScalarRing {
    /// `Z[x]/(x^162 + x^81 + 1)`, whose binary reduction is `F_(2^162)`.
    Cyclotomic243,
    /// `Z[z]/(z^486 + z^243 + 1)`, whose binary reduction is `F_(2^486)`.
    Cyclotomic729,
}

impl BinaryScalarRing {
    /// Power-basis degree.
    #[inline]
    #[must_use]
    pub const fn degree(self) -> usize {
        match self {
            Self::Cyclotomic243 => 162,
            Self::Cyclotomic729 => 486,
        }
    }

    const fn identity(self) -> u8 {
        match self {
            Self::Cyclotomic243 => 0,
            Self::Cyclotomic729 => 1,
        }
    }

    pub(super) fn counts(self) -> &'static BinomialCdfRow {
        match self {
            Self::Cyclotomic243 => &CYCLOTOMIC_243_COUNTS,
            Self::Cyclotomic729 => &CYCLOTOMIC_729_COUNTS,
        }
    }
}

/// Exact support family used for a binary fold challenge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryChallengeFamily {
    /// Every support has exactly the configured weight.
    FixedWeight,
    /// Every support from weight zero through the configured cap is admitted.
    BoundedWeight,
}

impl BinaryChallengeFamily {
    const fn identity(self) -> u8 {
        match self {
            Self::FixedWeight => 0,
            Self::BoundedWeight => 1,
        }
    }
}

/// Fixed deterministic map from a canonical support to one sign pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinarySignRule {
    /// SHAKE256 under the profile identity and `akita/labinius/binary-sign/v1`.
    Shake256V1,
}

impl BinarySignRule {
    const fn identity(self) -> u8 {
        match self {
            Self::Shake256V1 => 0,
        }
    }
}

/// Complete identity and exact bounds for one binary challenge family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryChallengeProfile {
    scalar_ring: BinaryScalarRing,
    family: BinaryChallengeFamily,
    weight_cap: usize,
    sign_rule: BinarySignRule,
    cardinality: BigUint,
    identity: Vec<u8>,
}

impl BinaryChallengeProfile {
    /// Construct the paper-compatible exact-weight reference family.
    pub fn fixed_weight(scalar_ring: BinaryScalarRing, weight: usize) -> Result<Self, AkitaError> {
        Self::new(scalar_ring, BinaryChallengeFamily::FixedWeight, weight)
    }

    /// Construct the uniform support ball containing every weight up to `cap`.
    pub fn bounded_weight(scalar_ring: BinaryScalarRing, cap: usize) -> Result<Self, AkitaError> {
        Self::new(scalar_ring, BinaryChallengeFamily::BoundedWeight, cap)
    }

    fn new(
        scalar_ring: BinaryScalarRing,
        family: BinaryChallengeFamily,
        weight_cap: usize,
    ) -> Result<Self, AkitaError> {
        if weight_cap > scalar_ring.degree() {
            return Err(AkitaError::InvalidInput(format!(
                "binary challenge weight {weight_cap} exceeds scalar degree {}",
                scalar_ring.degree()
            )));
        }
        let counts = scalar_ring.counts();
        let cardinality = match family {
            BinaryChallengeFamily::FixedWeight => counts.binomial(weight_cap),
            BinaryChallengeFamily::BoundedWeight => counts.ball(weight_cap).clone(),
        };
        let sign_rule = BinarySignRule::Shake256V1;
        let identity = encode_identity(scalar_ring, family, weight_cap, sign_rule, &cardinality);
        Ok(Self {
            scalar_ring,
            family,
            weight_cap,
            sign_rule,
            cardinality,
            identity,
        })
    }

    /// Select the smallest weight/cap meeting `|C| >= fold_width * 2^lambda_fold`.
    ///
    /// Fixed-weight search stops at half the scalar degree, where the binomial
    /// shell is maximal. Bounded-weight search includes the full support ball.
    /// The requested budget must also fit the entropy of one fold-root draw.
    pub fn minimum_for_budget(
        scalar_ring: BinaryScalarRing,
        family: BinaryChallengeFamily,
        fold_width: u64,
        lambda_fold: u32,
    ) -> Result<Self, AkitaError> {
        if fold_width == 0 {
            return Err(AkitaError::InvalidInput(
                "binary challenge fold width must be nonzero".into(),
            ));
        }
        let required = required_budget(fold_width, lambda_fold).ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!(
                "binary challenge budget exceeds the {FOLD_ROOT_ENTROPY_BITS}-bit fold-root capacity"
            ))
        })?;
        let degree = scalar_ring.degree();
        if lambda_fold as usize > degree || (lambda_fold as usize == degree && fold_width > 1) {
            return Err(AkitaError::UnsupportedSchedule(format!(
                "binary challenge budget exceeds the degree-{degree} residue capacity"
            )));
        }
        let maximum_weight = match family {
            BinaryChallengeFamily::FixedWeight => degree / 2,
            BinaryChallengeFamily::BoundedWeight => degree,
        };
        let counts = scalar_ring.counts();
        let selected = (0..=maximum_weight).find(|&weight| {
            let cardinality = match family {
                BinaryChallengeFamily::FixedWeight => counts.binomial(weight),
                BinaryChallengeFamily::BoundedWeight => counts.ball(weight).clone(),
            };
            cardinality >= required
        });
        match selected {
            Some(weight) => Self::new(scalar_ring, family, weight),
            None => Err(AkitaError::UnsupportedSchedule(format!(
                "binary {:?} challenge family at degree {degree} cannot meet fold width {fold_width} with lambda_fold={lambda_fold}",
                family
            ))),
        }
    }

    /// Scalar-ring identity.
    #[inline]
    #[must_use]
    pub const fn scalar_ring(&self) -> BinaryScalarRing {
        self.scalar_ring
    }

    /// Support-family identity.
    #[inline]
    #[must_use]
    pub const fn family(&self) -> BinaryChallengeFamily {
        self.family
    }

    /// Exact weight for a shell, or maximum weight for a support ball.
    #[inline]
    #[must_use]
    pub const fn weight_cap(&self) -> usize {
        self.weight_cap
    }

    /// Deterministic sign-map identity.
    #[inline]
    #[must_use]
    pub const fn sign_rule(&self) -> BinarySignRule {
        self.sign_rule
    }

    /// Exact family cardinality. Deterministic signs add no factor.
    #[inline]
    #[must_use]
    pub fn cardinality(&self) -> &BigUint {
        &self.cardinality
    }

    /// Exact comparison against `fold_width * 2^lambda_fold`, capped by the
    /// entropy of the fold root from which the family is sampled.
    #[must_use]
    pub fn meets_budget(&self, fold_width: u64, lambda_fold: u32) -> bool {
        let degree = self.scalar_ring.degree();
        if lambda_fold as usize > degree || (lambda_fold as usize == degree && fold_width > 1) {
            return false;
        }
        required_budget(fold_width, lambda_fold)
            .is_some_and(|required| self.cardinality >= required)
    }

    /// Maximum coefficient infinity norm over the family.
    #[inline]
    #[must_use]
    pub const fn coefficient_linf_bound(&self) -> u8 {
        if self.weight_cap == 0 {
            0
        } else {
            1
        }
    }

    /// Maximum coefficient L1 norm over the family.
    #[inline]
    #[must_use]
    pub const fn coefficient_l1_bound(&self) -> u64 {
        self.weight_cap as u64
    }

    /// Maximum squared coefficient L2 norm over the family.
    #[inline]
    #[must_use]
    pub const fn coefficient_l2_squared_bound(&self) -> u64 {
        self.weight_cap as u64
    }

    /// Certified coefficient-infinity multiplication operator bound in the
    /// trinomial power basis: `Gamma_inf <= 2 * ||c||_1`. The factor two is
    /// necessary because multiplication by the half-degree monomial maps a
    /// coefficient pair `[a, b]` to `[-b, a-b]`.
    #[inline]
    #[must_use]
    pub const fn multiplication_linf_operator_bound(&self) -> u64 {
        2 * self.weight_cap as u64
    }

    /// Canonical profile bytes used by transcript and sign domain separation.
    #[must_use]
    pub fn identity_bytes(&self) -> &[u8] {
        &self.identity
    }
}

fn encode_identity(
    scalar_ring: BinaryScalarRing,
    family: BinaryChallengeFamily,
    weight_cap: usize,
    sign_rule: BinarySignRule,
    cardinality: &BigUint,
) -> Vec<u8> {
    let cardinality_bytes = cardinality.to_bytes_le();
    let mut identity = Vec::with_capacity(23 + cardinality_bytes.len());
    identity.push(1);
    identity.push(scalar_ring.identity());
    identity.push(family.identity());
    identity.push(sign_rule.identity());
    identity.push(0); // canonical support encoding: little-endian bitset v1
    identity.extend_from_slice(&(scalar_ring.degree() as u32).to_le_bytes());
    identity.extend_from_slice(&(weight_cap as u32).to_le_bytes());
    identity.extend_from_slice(&(2 * weight_cap as u64).to_le_bytes());
    identity.extend_from_slice(&(cardinality_bytes.len() as u16).to_le_bytes());
    identity.extend_from_slice(&cardinality_bytes);
    identity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_degree_162_budget_boundaries_match_the_specification() {
        let expected = [
            (128, Some(41), 41),
            (136, Some(47), 46),
            (144, Some(54), 53),
            (152, Some(63), 61),
            (156, Some(71), 67),
            (158, Some(81), 71),
            (160, None, 77),
        ];
        for (bits, fixed, bounded) in expected {
            let selected_fixed = BinaryChallengeProfile::minimum_for_budget(
                BinaryScalarRing::Cyclotomic243,
                BinaryChallengeFamily::FixedWeight,
                1,
                bits,
            );
            assert_eq!(
                selected_fixed.ok().map(|profile| profile.weight_cap()),
                fixed
            );
            assert_eq!(
                BinaryChallengeProfile::minimum_for_budget(
                    BinaryScalarRing::Cyclotomic243,
                    BinaryChallengeFamily::BoundedWeight,
                    1,
                    bits,
                )
                .unwrap()
                .weight_cap(),
                bounded
            );
        }
    }

    #[test]
    fn exact_fold_width_is_part_of_the_budget() {
        let profile = BinaryChallengeProfile::minimum_for_budget(
            BinaryScalarRing::Cyclotomic243,
            BinaryChallengeFamily::BoundedWeight,
            256,
            128,
        )
        .unwrap();
        assert_eq!(profile.weight_cap(), 46);
        assert!(profile.meets_budget(256, 128));
        assert!(
            !BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 45)
                .unwrap()
                .meets_budget(256, 128)
        );
    }

    #[test]
    fn degree_486_counts_remain_exact_beyond_256_bits() {
        let profile =
            BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic729, 243).unwrap();
        assert!(profile.cardinality().bits() > 256);
        assert_eq!(profile.cardinality().bits(), 482);
    }

    #[test]
    fn fold_root_entropy_caps_degree_486_budgets() {
        let ring = BinaryScalarRing::Cyclotomic729;
        let family = BinaryChallengeFamily::BoundedWeight;
        let profile = BinaryChallengeProfile::bounded_weight(ring, ring.degree()).unwrap();

        assert!(!profile.meets_budget(1, 300));
        assert!(BinaryChallengeProfile::minimum_for_budget(ring, family, 1, 300).is_err());
        assert!(profile.meets_budget(1, 256));
        assert!(profile.meets_budget(2, 255));
        assert!(profile.meets_budget(3, 254));
        assert!(!profile.meets_budget(2, 256));
        assert!(!profile.meets_budget(3, 255));
        assert!(BinaryChallengeProfile::minimum_for_budget(ring, family, 1, 256).is_ok());
        assert!(BinaryChallengeProfile::minimum_for_budget(ring, family, 2, 256).is_err());
        assert!(BinaryChallengeProfile::minimum_for_budget(ring, family, 3, 254).is_ok());
        assert!(BinaryChallengeProfile::minimum_for_budget(ring, family, 3, 255).is_err());
    }

    #[test]
    fn degree_486_fixed_weight_boundaries_match_the_specification() {
        for (bits, weight) in [(128, 23), (136, 25), (144, 27), (152, 29), (160, 31)] {
            let profile = BinaryChallengeProfile::minimum_for_budget(
                BinaryScalarRing::Cyclotomic729,
                BinaryChallengeFamily::FixedWeight,
                1,
                bits,
            )
            .unwrap();
            assert_eq!(profile.weight_cap(), weight);
        }
    }

    #[test]
    fn oversized_budget_exponents_are_rejected_before_big_integer_shifts() {
        let profile =
            BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 162).unwrap();
        assert!(!profile.meets_budget(1, u32::MAX));
        assert!(BinaryChallengeProfile::minimum_for_budget(
            BinaryScalarRing::Cyclotomic243,
            BinaryChallengeFamily::BoundedWeight,
            1,
            u32::MAX,
        )
        .is_err());
        assert!(profile.meets_budget(1, 162));
        assert!(!profile.meets_budget(2, 162));
    }

    #[test]
    fn norm_bounds_include_the_zero_family_truthfully() {
        let zero =
            BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 0).unwrap();
        assert_eq!(zero.cardinality(), &BigUint::from(1u8));
        assert_eq!(zero.coefficient_linf_bound(), 0);
        assert_eq!(zero.coefficient_l1_bound(), 0);
        assert_eq!(zero.coefficient_l2_squared_bound(), 0);
        assert_eq!(zero.multiplication_linf_operator_bound(), 0);

        let nonzero =
            BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 46).unwrap();
        assert_eq!(nonzero.coefficient_linf_bound(), 1);
        assert_eq!(nonzero.coefficient_l1_bound(), 46);
        assert_eq!(nonzero.coefficient_l2_squared_bound(), 46);
        assert_eq!(nonzero.multiplication_linf_operator_bound(), 92);
    }

    fn reduce_monomial(degree: usize, exponent: usize) -> Vec<(usize, i64)> {
        let half = degree / 2;
        if exponent < degree {
            vec![(exponent, 1)]
        } else if exponent < degree + half {
            vec![(exponent - degree, -1), (exponent - half, -1)]
        } else {
            vec![(exponent - degree - half, 1)]
        }
    }

    #[test]
    fn trinomial_wraparound_has_the_certified_factor_two() {
        for ring in [
            BinaryScalarRing::Cyclotomic243,
            BinaryScalarRing::Cyclotomic729,
        ] {
            let degree = ring.degree();
            let half = degree / 2;
            assert_eq!(reduce_monomial(degree, degree - 1), vec![(degree - 1, 1)]);
            assert_eq!(reduce_monomial(degree, degree), vec![(0, -1), (half, -1)]);
            assert_eq!(
                reduce_monomial(degree, degree + half - 1),
                vec![(half - 1, -1), (degree - 1, -1)]
            );
            assert_eq!(reduce_monomial(degree, degree + half), vec![(0, 1)]);
            assert_eq!(reduce_monomial(degree, 2 * degree - 2), vec![(half - 2, 1)]);
        }
    }
}
