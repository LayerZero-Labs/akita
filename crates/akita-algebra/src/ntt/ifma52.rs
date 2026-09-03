//! Canonical 50-bit NTT arithmetic for AVX-512IFMA hosts.

use akita_error::AkitaError;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

/// The three largest primes below `2^50` with `2^14 | (p - 1)`.
pub const IFMA52_PRIMES: [u64; 3] = [
    1_125_899_906_826_241,
    1_125_899_906_629_633,
    1_125_899_905_744_897,
];

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const RADIX: u64 = 1 << 52;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const MASK: u64 = RADIX - 1;

/// One canonical-residue prime and its IFMA Barrett constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ifma52Prime {
    /// Prime modulus.
    pub modulus: u64,
    pub(crate) barrett: u64,
}

impl Ifma52Prime {
    /// Validate and prepare one IFMA52 prime.
    pub fn new(modulus: u64) -> Result<Self, AkitaError> {
        if modulus >= (1 << 50) || modulus & 1 == 0 {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 modulus must be odd and below 2^50".into(),
            ));
        }
        Ok(Self {
            modulus,
            barrett: ((1u128 << 100) / u128::from(modulus)) as u64,
        })
    }

    #[inline]
    pub(crate) fn add(self, lhs: u64, rhs: u64) -> u64 {
        let sum = lhs + rhs;
        if sum >= self.modulus {
            sum - self.modulus
        } else {
            sum
        }
    }

    #[inline]
    pub(crate) fn sub(self, lhs: u64, rhs: u64) -> u64 {
        if lhs >= rhs {
            lhs - rhs
        } else {
            lhs + self.modulus - rhs
        }
    }

    #[inline]
    pub(crate) fn mul(self, lhs: u64, rhs: u64) -> u64 {
        ((u128::from(lhs) * u128::from(rhs)) % u128::from(self.modulus)) as u64
    }

    #[inline]
    pub(crate) fn canonical_i16(self, value: i16) -> u64 {
        if value >= 0 {
            value as u64
        } else {
            self.modulus - u64::from(value.unsigned_abs())
        }
    }

    fn pow(self, mut base: u64, mut exponent: u64) -> u64 {
        let mut result = 1;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = self.mul(result, base);
            }
            base = self.mul(base, base);
            exponent >>= 1;
        }
        result
    }

    fn inverse(self, value: u64) -> u64 {
        self.pow(value, self.modulus - 2)
    }

    pub(crate) fn precondition(self, value: u64) -> u64 {
        ((u128::from(value) << 52) / u128::from(self.modulus)) as u64
    }
}

/// Negacyclic NTT tables for one 50-bit prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ifma52Twiddles<const D: usize> {
    pub(crate) psi: [u64; D],
    pub(crate) psi_precon: [u64; D],
    pub(crate) forward: [u64; D],
    pub(crate) forward_precon: [u64; D],
    pub(crate) inverse: [u64; D],
    pub(crate) inverse_precon: [u64; D],
    pub(crate) forward_small: [[u64; 8]; 3],
    pub(crate) forward_small_precon: [[u64; 8]; 3],
    pub(crate) inverse_small: [[u64; 8]; 3],
    pub(crate) inverse_small_precon: [[u64; 8]; 3],
    pub(crate) inverse_scale: [u64; D],
    pub(crate) inverse_scale_precon: [u64; D],
}

impl<const D: usize> Ifma52Twiddles<D> {
    /// Compute tables for a supported power-of-two ring degree.
    pub fn compute(prime: Ifma52Prime) -> Result<Self, AkitaError> {
        if D < 64 || !D.is_power_of_two() || !(prime.modulus - 1).is_multiple_of(2 * D as u64) {
            return Err(AkitaError::InvalidSetup(format!(
                "IFMA52 prime does not support ring degree {D}"
            )));
        }
        let exponent = (prime.modulus - 1) / (2 * D as u64);
        let psi = (2u64..)
            .map(|candidate| prime.pow(candidate, exponent))
            .find(|&candidate| prime.pow(candidate, D as u64) == prime.modulus - 1)
            .ok_or_else(|| AkitaError::InvalidSetup("IFMA52 root search failed".into()))?;
        let psi_inverse = prime.inverse(psi);
        let omega = prime.mul(psi, psi);
        let omega_inverse = prime.inverse(omega);

        let mut psi_powers = [0; D];
        let mut psi_inverse_powers = [0; D];
        let (mut current, mut current_inverse) = (1, 1);
        for index in 0..D {
            psi_powers[index] = current;
            psi_inverse_powers[index] = current_inverse;
            current = prime.mul(current, psi);
            current_inverse = prime.mul(current_inverse, psi_inverse);
        }

        let mut forward = [0; D];
        let mut inverse = [0; D];
        let mut len = 1;
        while len < D {
            let stage_exponent = (D / (2 * len)) as u64;
            let forward_step = prime.pow(omega, stage_exponent);
            let inverse_step = prime.pow(omega_inverse, stage_exponent);
            let base = len - 1;
            let (mut wf, mut wi) = (1, 1);
            for index in 0..len {
                forward[base + index] = wf;
                inverse[base + index] = wi;
                wf = prime.mul(wf, forward_step);
                wi = prime.mul(wi, inverse_step);
            }
            len *= 2;
        }

        let inverse_d = prime.inverse(D as u64);
        let inverse_scale = psi_inverse_powers.map(|value| prime.mul(inverse_d, value));
        let lane_twiddles =
            |table: &[u64; D], len: usize| std::array::from_fn(|lane| table[len - 1 + lane % len]);
        let forward_small = [
            lane_twiddles(&forward, 4),
            lane_twiddles(&forward, 2),
            lane_twiddles(&forward, 1),
        ];
        let inverse_small = [
            lane_twiddles(&inverse, 1),
            lane_twiddles(&inverse, 2),
            lane_twiddles(&inverse, 4),
        ];
        Ok(Self {
            psi: psi_powers,
            psi_precon: psi_powers.map(|value| prime.precondition(value)),
            forward,
            forward_precon: forward.map(|value| prime.precondition(value)),
            inverse,
            inverse_precon: inverse.map(|value| prime.precondition(value)),
            forward_small,
            forward_small_precon: forward_small
                .map(|stage| stage.map(|value| prime.precondition(value))),
            inverse_small,
            inverse_small_precon: inverse_small
                .map(|stage| stage.map(|value| prime.precondition(value))),
            inverse_scale,
            inverse_scale_precon: inverse_scale.map(|value| prime.precondition(value)),
        })
    }
}

/// Whether this process can execute the AVX-512IFMA kernels.
#[must_use]
pub fn ifma52_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512ifma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Whether IFMA52 is available and SIMD has not been globally disabled.
#[must_use]
pub fn ifma52_enabled() -> bool {
    ifma52_available() && std::env::var("AKITA_SCALAR_NTT").ok().as_deref() != Some("1")
}

#[inline(always)]
pub(crate) fn forward<const D: usize>(
    values: &mut [u64; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
    use_ifma: bool,
) {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = use_ifma;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_ifma {
        // SAFETY: runtime feature detection covers every enabled instruction.
        unsafe { x86::forward(values, prime, twiddles) };
        return;
    }
    scalar_forward(values, prime, twiddles);
}

#[inline(always)]
pub(crate) fn inverse<const D: usize>(
    values: &mut [u64; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
    use_ifma: bool,
) {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = use_ifma;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_ifma {
        // SAFETY: runtime feature detection covers every enabled instruction.
        unsafe { x86::inverse(values, prime, twiddles) };
        return;
    }
    scalar_inverse(values, prime, twiddles);
}

#[inline(always)]
pub(crate) fn pointwise_dot_accumulate<const D: usize>(
    accumulator: &mut [u64; D],
    lhs: &[[u64; D]],
    rhs: &[[u64; D]],
    prime: Ifma52Prime,
    use_ifma: bool,
) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = use_ifma;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_ifma {
        // SAFETY: runtime feature detection covers every enabled instruction.
        unsafe { x86::pointwise_dot_accumulate(accumulator, lhs, rhs, prime) };
        return;
    }
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        for ((accumulator, &lhs), &rhs) in accumulator.iter_mut().zip(lhs).zip(rhs) {
            *accumulator = prime.add(*accumulator, prime.mul(lhs, rhs));
        }
    }
}

fn scalar_forward<const D: usize>(
    values: &mut [u64; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
) {
    for (value, &psi) in values.iter_mut().zip(&twiddles.psi) {
        *value = prime.mul(*value, psi);
    }
    let mut len = D / 2;
    while len != 0 {
        let base = len - 1;
        for start in (0..D).step_by(2 * len) {
            for index in 0..len {
                let x = values[start + index];
                let y = values[start + index + len];
                values[start + index] = prime.add(x, y);
                values[start + index + len] =
                    prime.mul(prime.sub(x, y), twiddles.forward[base + index]);
            }
        }
        len /= 2;
    }
}

fn scalar_inverse<const D: usize>(
    values: &mut [u64; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
) {
    let mut len = 1;
    while len < D {
        let base = len - 1;
        for start in (0..D).step_by(2 * len) {
            for index in 0..len {
                let x = values[start + index];
                let y = prime.mul(values[start + index + len], twiddles.inverse[base + index]);
                values[start + index] = prime.add(x, y);
                values[start + index + len] = prime.sub(x, y);
            }
        }
        len *= 2;
    }
    for (value, &scale) in values.iter_mut().zip(&twiddles.inverse_scale) {
        *value = prime.mul(*value, scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<const D: usize>() {
        for &modulus in &IFMA52_PRIMES {
            let prime = Ifma52Prime::new(modulus).expect("prime");
            let twiddles = Ifma52Twiddles::<D>::compute(prime).expect("twiddles");
            let original =
                std::array::from_fn(|index| (index as u64 * 1_000_003 + 17) % prime.modulus);
            let mut transformed = original;
            forward(&mut transformed, prime, &twiddles, ifma52_available());
            inverse(&mut transformed, prime, &twiddles, ifma52_available());
            assert_eq!(transformed, original, "modulus={modulus}, D={D}");
        }
    }

    #[test]
    fn all_profile_primes_round_trip_supported_degrees() {
        round_trip::<64>();
        round_trip::<128>();
        round_trip::<256>();
        round_trip::<512>();
        round_trip::<1024>();
        round_trip::<2048>();
    }

    #[test]
    #[ignore = "requires AVX-512F/DQ/IFMA hardware or emulation"]
    fn ifma52_hardware_round_trip_does_not_fallback() {
        assert!(ifma52_available(), "AVX-512F/DQ/IFMA is unavailable");
        assert!(ifma52_enabled(), "IFMA52 dispatch was not selected");
        round_trip::<64>();
        round_trip::<1024>();
        round_trip::<2048>();
    }
}
