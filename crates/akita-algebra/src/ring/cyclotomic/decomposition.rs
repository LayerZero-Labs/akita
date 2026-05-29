use super::*;

/// Compute the centering threshold for balanced decomposition.
///
/// When `levels * log_basis == field_bits`, uses asymmetric centering (T_k).
/// Otherwise falls back to symmetric centering (q/2).
pub fn decompose_centering_threshold(levels: usize, log_basis: u32, q: u128) -> u128 {
    let half_q = q / 2;
    let field_bits = 128u32 - q.saturating_sub(1).leading_zeros();
    let total_decomp_bits = (levels as u32).saturating_mul(log_basis);
    if total_decomp_bits == field_bits {
        let b: u128 = 1u128 << log_basis;
        let b_k_minus_1 = if total_decomp_bits >= 128 {
            u128::MAX
        } else {
            (1u128 << total_decomp_bits) - 1
        };
        let t_k = (b / 2 - 1) * (b_k_minus_1 / (b - 1));
        t_k.min(half_q)
    } else {
        half_q
    }
}

/// Center a canonical field element for balanced decomposition.
///
/// Returns `(centered_value, Option<first_digit>)`. When the magnitude
/// exceeds `i128::MAX`, the first balanced digit is pre-extracted in `u128`
/// arithmetic and returned separately; `centered_value` is then the remaining
/// quotient after removing that digit.
#[inline]
pub(crate) fn center_for_decomposition(
    canonical: u128,
    q: u128,
    threshold: u128,
    log_basis: u32,
) -> (i128, Option<i128>) {
    if canonical <= threshold {
        return (canonical as i128, None);
    }
    let diff = q - canonical;
    if diff <= i128::MAX as u128 {
        return (-(diff as i128), None);
    }
    let b_u = 1u128 << log_basis;
    let mask_u = b_u - 1;
    let half_b_u = b_u >> 1;
    let r = canonical.wrapping_sub(q) & mask_u;
    let balanced = if r >= half_b_u {
        r as i128 - b_u as i128
    } else {
        r as i128
    };
    let diff_adj = if balanced >= 0 {
        diff + balanced as u128
    } else {
        diff - ((-balanced) as u128)
    };
    debug_assert!(diff_adj & mask_u == 0);
    let c_prime = -((diff_adj >> log_basis) as i128);
    (c_prime, Some(balanced))
}

#[inline(always)]
/// Peel one balanced base-`2^log_basis` digit from a canonical value.
pub fn peel_first_balanced_digit(
    canonical: u128,
    q: u128,
    threshold: u128,
    mask: i128,
    half_b: i128,
    b: i128,
    log_basis: u32,
) -> (i128, i128) {
    let (c, first_digit) = center_for_decomposition(canonical, q, threshold, log_basis);
    if let Some(d0) = first_digit {
        (c, d0)
    } else {
        let d = c & mask;
        let balanced = if d >= half_b { d - b } else { d };
        ((c - balanced) >> log_basis, balanced)
    }
}

#[inline(always)]
fn balanced_digit_to_field<F: CanonicalField>(digit: i128, q: u128) -> F {
    if digit >= 0 {
        F::from_canonical_u128_reduced(digit as u128)
    } else {
        F::from_canonical_u128_reduced(q - ((-digit) as u128))
    }
}

/// Precomputed parameters for balanced power-of-two `i8` decomposition.
#[derive(Clone, Copy, Debug)]
pub struct BalancedDecomposePow2I8Params {
    levels: usize,
    log_basis: u32,
    q: u128,
    threshold: u128,
    half_b: i128,
    b: i128,
    mask: i128,
    overflow_possible: bool,
}

impl BalancedDecomposePow2I8Params {
    /// Build decomposition parameters for `levels` digits in base `2^log_basis`.
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is outside `1..=6`, or if the requested digit
    /// budget exceeds the supported field-width guard.
    pub fn new(levels: usize, log_basis: u32, q: u128) -> Self {
        assert!(
            log_basis > 0 && log_basis <= 6,
            "log_basis must be in 1..=6 for i8 output"
        );
        assert!(
            (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
            "levels * log_basis must be <= 128 + log_basis"
        );

        let half_b = 1i128 << (log_basis - 1);
        let b = half_b << 1;
        let threshold = decompose_centering_threshold(levels, log_basis, q);
        let overflow_possible = q.saturating_sub(threshold) > i128::MAX as u128;
        Self {
            levels,
            log_basis,
            q,
            threshold,
            half_b,
            b,
            mask: b - 1,
            overflow_possible,
        }
    }
}

impl<F: CanonicalField, const D: usize> CyclotomicRing<F, D> {
    /// Balanced decomposition writing directly into a pre-allocated output slice.
    ///
    /// `out` must have length exactly `levels`. Each element receives one digit plane.
    ///
    /// # Panics
    ///
    /// Panics if `log_basis == 0`, `log_basis >= 128`, or `out.len() * log_basis > 128 + log_basis`.
    pub fn balanced_decompose_pow2_into(&self, out: &mut [Self], log_basis: u32) {
        let levels = out.len();
        assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
        assert!(
            (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
            "levels * log_basis must be <= 128 + log_basis"
        );

        let half_b = 1i128 << (log_basis - 1);
        let b = half_b << 1;
        let mask = b - 1;
        let q = (-F::one()).to_canonical_u128() + 1;
        let threshold = decompose_centering_threshold(levels, log_basis, q);
        let overflow_possible = q.saturating_sub(threshold) > i128::MAX as u128;

        for plane in out.iter_mut() {
            *plane = Self::zero();
        }

        if overflow_possible {
            let (first_plane, remaining) = out
                .split_first_mut()
                .expect("balanced_decompose_pow2_into requires at least one plane");
            for i in 0..D {
                let canonical = self.coeffs[i].to_canonical_u128();
                let (mut c, d0) =
                    peel_first_balanced_digit(canonical, q, threshold, mask, half_b, b, log_basis);
                first_plane.coeffs[i] = balanced_digit_to_field::<F>(d0, q);

                for plane in remaining.iter_mut() {
                    let d = c & mask;
                    let balanced = if d >= half_b { d - b } else { d };
                    c = (c - balanced) >> log_basis;
                    plane.coeffs[i] = balanced_digit_to_field::<F>(balanced, q);
                }
            }
        } else {
            for i in 0..D {
                let canonical = self.coeffs[i].to_canonical_u128();
                let mut c: i128 = if canonical > threshold {
                    -((q - canonical) as i128)
                } else {
                    canonical as i128
                };

                for plane in out.iter_mut() {
                    let d = c & mask;
                    let balanced = if d >= half_b { d - b } else { d };
                    c = (c - balanced) >> log_basis;
                    plane.coeffs[i] = balanced_digit_to_field::<F>(balanced, q);
                }
            }
        }
    }

    /// Squared Euclidean norm of centered integer coefficients.
    ///
    /// Coefficients are centered into `(-q/2, q/2]` and accumulated as
    /// `sum_i c_i^2`, using saturating arithmetic.
    #[inline]
    pub fn coeff_norm_sq(&self) -> u128
    where
        F: CanonicalField,
    {
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        self.coeffs.iter().fold(0u128, |acc, &coeff| {
            let canonical = coeff.to_canonical_u128();
            let centered: i128 = if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            };
            let abs = centered.unsigned_abs();
            acc.saturating_add(abs.saturating_mul(abs))
        })
    }

    /// Functional gadget recomposition (`G * digits`) for base `2^log_basis`.
    ///
    /// Coefficients from each part are interpreted as one digit plane and
    /// recombined back into canonical integers (then reduced into the field).
    ///
    /// # Panics
    ///
    /// Panics if `log_basis == 0`, `log_basis >= 128`, or `parts.len() * log_basis > 128`.
    pub fn gadget_recompose_pow2(parts: &[Self], log_basis: u32) -> Self {
        if parts.is_empty() {
            return Self::zero();
        }

        assert!(
            log_basis > 0 && log_basis <= 128,
            "invalid log_basis: {log_basis}"
        );

        if parts.len() == 1 {
            return parts[0];
        }

        let b = F::from_canonical_u128_reduced(1u128 << log_basis);
        let coeffs = from_fn(|i| {
            let mut acc = F::zero();
            let mut power = F::one();
            for part in parts.iter() {
                acc += part.coeffs[i] * power;
                power *= b;
            }
            acc
        });
        Self { coeffs }
    }

    /// Recompose from i8 digit planes (output of `balanced_decompose_pow2_i8`).
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is zero or >= 128.
    pub fn gadget_recompose_pow2_i8(digits: &[[i8; D]], log_basis: u32) -> Self
    where
        F: CanonicalField,
    {
        if digits.is_empty() {
            return Self::zero();
        }
        assert!(
            log_basis > 0 && log_basis <= 128,
            "invalid log_basis: {log_basis}"
        );

        if digits.len() == 1 {
            let coeffs = from_fn(|i| F::from_i64(digits[0][i] as i64));
            return Self { coeffs };
        }

        let b = F::from_canonical_u128_reduced(1u128 << log_basis);
        let coeffs = from_fn(|i| {
            let mut acc = F::zero();
            let mut power = F::one();
            for plane in digits {
                acc += F::from_i64(plane[i] as i64) * power;
                power *= b;
            }
            acc
        });
        Self { coeffs }
    }

    /// Balanced (centered) base-`2^log_basis` gadget decomposition: `G^{-1}`.
    ///
    /// Each coefficient `c` (centered into `(-q/2, q/2]`) is decomposed into
    /// `levels` balanced digits `d_k ∈ [-b/2, b/2)` satisfying
    /// `c ≡ Σ_k d_k · b^k  (mod q)`.
    ///
    /// Negative digits are stored as their field representation (`q + d`).
    ///
    /// # Panics
    ///
    /// Panics if `log_basis == 0`, `log_basis >= 128`, or `levels * log_basis > 128`.
    pub fn balanced_decompose_pow2(&self, levels: usize, log_basis: u32) -> Vec<Self> {
        assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
        assert!(
            (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
            "levels * log_basis must be <= 128 + log_basis"
        );
        let mut digit_planes = vec![Self::zero(); levels];
        self.balanced_decompose_pow2_into(&mut digit_planes, log_basis);
        digit_planes
    }

    /// Balanced gadget decomposition into native `i8` digits.
    ///
    /// Same semantics as [`balanced_decompose_pow2`](Self::balanced_decompose_pow2)
    /// but stores each digit as `i8` instead of a field element, avoiding
    /// the cost of `F::from_canonical_u128_reduced`.
    ///
    /// Requires `log_basis <= 6` so digits fit in `[-32, 31]` (i8 range).
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is 0 or > 6, or if `levels * log_basis > 128 + log_basis`.
    #[inline]
    pub fn balanced_decompose_pow2_i8_into(&self, out: &mut [[i8; D]], log_basis: u32)
    where
        F: CanonicalField,
    {
        let levels = out.len();
        assert!(
            log_basis > 0 && log_basis <= 6,
            "log_basis must be in 1..=6 for i8 output"
        );
        assert!(
            (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
            "levels * log_basis must be <= 128 + log_basis"
        );

        let q = (-F::one()).to_canonical_u128() + 1;
        self.balanced_decompose_pow2_i8_into_with_modulus(out, log_basis, q);
    }

    /// Internal variant of [`balanced_decompose_pow2_i8_into`](Self::balanced_decompose_pow2_i8_into)
    /// that reuses a caller-supplied field modulus.
    #[inline]
    pub fn balanced_decompose_pow2_i8_into_with_modulus(
        &self,
        out: &mut [[i8; D]],
        log_basis: u32,
        q: u128,
    ) where
        F: CanonicalField,
    {
        let params = BalancedDecomposePow2I8Params::new(out.len(), log_basis, q);
        self.balanced_decompose_pow2_i8_into_with_params(out, &params);
    }

    #[inline]
    /// Decompose using caller-supplied precomputed decomposition parameters.
    pub fn balanced_decompose_pow2_i8_into_with_params(
        &self,
        out: &mut [[i8; D]],
        params: &BalancedDecomposePow2I8Params,
    ) where
        F: CanonicalField,
    {
        debug_assert_eq!(out.len(), params.levels);
        if params.overflow_possible {
            self.balanced_decompose_pow2_i8_overflow(out, params);
        } else {
            self.balanced_decompose_pow2_i8_fast(out, params);
        }
    }

    /// Fast path: no i128 overflow possible (threshold >= q - i128::MAX).
    #[inline]
    fn balanced_decompose_pow2_i8_fast(
        &self,
        out: &mut [[i8; D]],
        params: &BalancedDecomposePow2I8Params,
    ) where
        F: CanonicalField,
    {
        let bulk_end = D - (D % 3);

        for base in (0..bulk_end).step_by(3) {
            let canonical0 = self.coeffs[base].to_canonical_u128();
            let canonical1 = self.coeffs[base + 1].to_canonical_u128();
            let canonical2 = self.coeffs[base + 2].to_canonical_u128();

            let mut c0: i128 = if canonical0 > params.threshold {
                -((params.q - canonical0) as i128)
            } else {
                canonical0 as i128
            };
            let mut c1: i128 = if canonical1 > params.threshold {
                -((params.q - canonical1) as i128)
            } else {
                canonical1 as i128
            };
            let mut c2: i128 = if canonical2 > params.threshold {
                -((params.q - canonical2) as i128)
            } else {
                canonical2 as i128
            };

            for plane in out.iter_mut() {
                let d0 = c0 & params.mask;
                let balanced0 = if d0 >= params.half_b {
                    d0 - params.b
                } else {
                    d0
                };
                c0 = (c0 - balanced0) >> params.log_basis;
                plane[base] = balanced0 as i8;

                let d1 = c1 & params.mask;
                let balanced1 = if d1 >= params.half_b {
                    d1 - params.b
                } else {
                    d1
                };
                c1 = (c1 - balanced1) >> params.log_basis;
                plane[base + 1] = balanced1 as i8;

                let d2 = c2 & params.mask;
                let balanced2 = if d2 >= params.half_b {
                    d2 - params.b
                } else {
                    d2
                };
                c2 = (c2 - balanced2) >> params.log_basis;
                plane[base + 2] = balanced2 as i8;
            }
        }

        for i in bulk_end..D {
            let canonical = self.coeffs[i].to_canonical_u128();
            let mut c: i128 = if canonical > params.threshold {
                -((params.q - canonical) as i128)
            } else {
                canonical as i128
            };

            for plane in out.iter_mut() {
                let d = c & params.mask;
                let balanced = if d >= params.half_b { d - params.b } else { d };
                c = (c - balanced) >> params.log_basis;
                plane[i] = balanced as i8;
            }
        }
    }

    /// Overflow-aware path: peels the first digit per coefficient, then keeps
    /// the remaining digits in the same 3-at-a-time register loop.
    fn balanced_decompose_pow2_i8_overflow(
        &self,
        out: &mut [[i8; D]],
        params: &BalancedDecomposePow2I8Params,
    ) where
        F: CanonicalField,
    {
        let (first_plane, remaining) = out
            .split_first_mut()
            .expect("balanced_decompose_pow2_i8_overflow requires at least one plane");
        let bulk_end = D - (D % 3);

        for base in (0..bulk_end).step_by(3) {
            let canonical0 = self.coeffs[base].to_canonical_u128();
            let canonical1 = self.coeffs[base + 1].to_canonical_u128();
            let canonical2 = self.coeffs[base + 2].to_canonical_u128();

            let (mut c0, d0) = peel_first_balanced_digit(
                canonical0,
                params.q,
                params.threshold,
                params.mask,
                params.half_b,
                params.b,
                params.log_basis,
            );
            let (mut c1, d1) = peel_first_balanced_digit(
                canonical1,
                params.q,
                params.threshold,
                params.mask,
                params.half_b,
                params.b,
                params.log_basis,
            );
            let (mut c2, d2) = peel_first_balanced_digit(
                canonical2,
                params.q,
                params.threshold,
                params.mask,
                params.half_b,
                params.b,
                params.log_basis,
            );

            first_plane[base] = d0 as i8;
            first_plane[base + 1] = d1 as i8;
            first_plane[base + 2] = d2 as i8;

            for plane in remaining.iter_mut() {
                let d0 = c0 & params.mask;
                let balanced0 = if d0 >= params.half_b {
                    d0 - params.b
                } else {
                    d0
                };
                c0 = (c0 - balanced0) >> params.log_basis;
                plane[base] = balanced0 as i8;

                let d1 = c1 & params.mask;
                let balanced1 = if d1 >= params.half_b {
                    d1 - params.b
                } else {
                    d1
                };
                c1 = (c1 - balanced1) >> params.log_basis;
                plane[base + 1] = balanced1 as i8;

                let d2 = c2 & params.mask;
                let balanced2 = if d2 >= params.half_b {
                    d2 - params.b
                } else {
                    d2
                };
                c2 = (c2 - balanced2) >> params.log_basis;
                plane[base + 2] = balanced2 as i8;
            }
        }

        for i in bulk_end..D {
            let canonical = self.coeffs[i].to_canonical_u128();
            let (mut c, d0) = peel_first_balanced_digit(
                canonical,
                params.q,
                params.threshold,
                params.mask,
                params.half_b,
                params.b,
                params.log_basis,
            );
            first_plane[i] = d0 as i8;
            for plane in remaining.iter_mut() {
                let d = c & params.mask;
                let balanced = if d >= params.half_b { d - params.b } else { d };
                c = (c - balanced) >> params.log_basis;
                plane[i] = balanced as i8;
            }
        }
    }

    /// Allocating variant of [`balanced_decompose_pow2_i8_into`](Self::balanced_decompose_pow2_i8_into).
    pub fn balanced_decompose_pow2_i8(&self, levels: usize, log_basis: u32) -> Vec<[i8; D]>
    where
        F: CanonicalField,
    {
        let mut digit_planes: Vec<[i8; D]> = vec![[0i8; D]; levels];
        self.balanced_decompose_pow2_i8_into(&mut digit_planes, log_basis);
        digit_planes
    }

    /// Balanced decomposition where the last digit carries the remainder.
    ///
    /// The first `levels-1` digits are balanced in `[-b/2, b/2)`, while the
    /// final digit is the remaining (possibly larger) centered value.
    ///
    /// # Panics
    ///
    /// Panics if `levels` is zero, `log_basis` is zero or >= 128, or
    /// `(levels - 1) * log_basis >= 128`.
    pub fn balanced_decompose_pow2_with_carry_into(&self, out: &mut [Self], log_basis: u32)
    where
        F: CanonicalField,
    {
        let levels = out.len();
        assert!(levels > 0, "levels must be positive");
        assert!(
            log_basis > 0 && log_basis <= 128,
            "invalid log_basis: {log_basis}"
        );
        assert!(
            ((levels - 1) as u32).saturating_mul(log_basis) < 128,
            "(levels-1) * log_basis must be < 128"
        );

        // When levels==1 every coefficient takes the carry path and b/half_b
        // are unused, so skip the shift that would overflow at log_basis==128.
        let (b, half_b) = if levels == 1 {
            (0i128, 0i128)
        } else {
            let b = 1i128 << log_basis;
            (b, b / 2)
        };
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;

        for i in 0..D {
            let canonical = self.coeffs[i].to_canonical_u128();
            let mut c: i128 = if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            };

            for (plane_idx, plane) in out.iter_mut().enumerate() {
                let balanced = if plane_idx + 1 == levels {
                    c
                } else {
                    let d = c.rem_euclid(b);
                    let digit = if d >= half_b { d - b } else { d };
                    c = (c - digit) / b;
                    digit
                };

                plane.coeffs[i] = if balanced >= 0 {
                    F::from_canonical_u128_reduced(balanced as u128)
                } else {
                    F::from_canonical_u128_reduced(q - ((-balanced) as u128))
                };
            }
        }
    }

    /// Allocating variant of
    /// [`balanced_decompose_pow2_with_carry_into`](Self::balanced_decompose_pow2_with_carry_into).
    pub fn balanced_decompose_pow2_with_carry(&self, levels: usize, log_basis: u32) -> Vec<Self>
    where
        F: CanonicalField,
    {
        let mut out = vec![Self::zero(); levels];
        self.balanced_decompose_pow2_with_carry_into(&mut out, log_basis);
        out
    }
}
