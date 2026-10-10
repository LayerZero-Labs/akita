//! The base-16 alphabet polynomial `A(x) = prod_{a=0}^{15} (x - a)`: on a line
//! through its range image, and on class buckets through weighted moments.

use jolt_field::Field;

use super::{DIGIT_BITS, INNER_NODES};

/// Powers `1..=16` of one class value.
pub(super) const POWERS: usize = INNER_NODES - 1;

/// Constants of the two evaluation methods.
///
/// `A` is symmetric about `15/2`: the digits `7 - k` and `8 + k` share the
/// range image `k(k + 1)`, so `A` is the degree-8 polynomial in
/// `z = (x - 7)(x - 8)` with roots `0, 2, 6, 12, 20, 30, 42, 56`. The shift
/// `y = z - 16` centres the root pairs `(2, 30)` and `(12, 20)`. With
/// `u = y^2 - 106`,
/// ```text
/// A = (u^2 - 8100) * (u - (24y + 534)) * (u - (16y + 154)).
/// ```
#[derive(Clone, Copy, Debug)]
pub(super) struct Alphabet<F> {
    seven: F,
    eight: F,
    minus_16: F,
    minus_106: F,
    minus_8100: F,
    low: F,
    high: F,
    /// Coefficients of `A`, constant term first.
    coefficients: [F; INNER_NODES],
}

impl<F: Field> Alphabet<F> {
    pub(super) fn new() -> Self {
        let [seven, eight, low, high] = [7, 8, 154, 534].map(F::from_u64);
        let [minus_16, minus_106, minus_8100] = [16, 106, 8100].map(|c| -F::from_u64(c));
        let mut coefficients = [F::zero(); INNER_NODES];
        if let Some(constant) = coefficients.first_mut() {
            *constant = F::one();
        }
        for root in 0..1u64 << DIGIT_BITS {
            let root = F::from_u64(root);
            let mut below = F::zero();
            for coefficient in &mut coefficients {
                let current = *coefficient;
                *coefficient = below - root * current;
                below = current;
            }
        }
        Self {
            seven,
            eight,
            minus_16,
            minus_106,
            minus_8100,
            low,
            high,
            coefficients,
        }
    }

    #[inline(always)]
    fn image(&self, x: F) -> F {
        (x - self.seven).mul_add(x - self.eight, self.minus_16)
    }

    /// `(16v, 24v)`.
    #[inline(always)]
    fn multiples(value: F) -> (F, F) {
        let double = value + value;
        let quadruple = double + double;
        let octuple = quadruple + quadruple;
        let sixteen = octuple + octuple;
        (sixteen, sixteen + octuple)
    }

    /// Add `weight * A(left + t*(right - left))` at `t = 0..=16` to `sums`.
    ///
    /// Along the line, `y`, `24y + 534` and `16y + 154` are quadratic in `t`,
    /// so forward differences tabulate them with additions only; one node
    /// then costs two squarings and three multiplications. Each stage runs
    /// over all nodes before the next one starts, so the seventeen chains of
    /// dependent multiplications advance together.
    #[inline(always)]
    pub(super) fn add_line(&self, sums: &mut [F; INNER_NODES], left: F, right: F, weight: F) {
        let mut y = [F::zero(); INNER_NODES];
        let mut low = [F::zero(); INNER_NODES];
        let mut high = [F::zero(); INNER_NODES];
        let mut value = self.image(left);
        let mut step = self.image(right) - value;
        let delta = right - left;
        let second = delta.square();
        let second = second + second;
        let (mut low_value, mut high_value) = Self::multiples(value);
        low_value += self.low;
        high_value += self.high;
        let (mut low_step, mut high_step) = Self::multiples(step);
        let (low_second, high_second) = Self::multiples(second);
        for ((y, low), high) in y.iter_mut().zip(&mut low).zip(&mut high) {
            *y = value;
            *low = low_value;
            *high = high_value;
            value += step;
            step += second;
            low_value += low_step;
            low_step += low_second;
            high_value += high_step;
            high_step += high_second;
        }
        // u = y^2 - 106, in place.
        for y in &mut y {
            *y = y.square() + self.minus_106;
        }
        // (u - (24y + 534)) * (u - (16y + 154)), in place of the second factor.
        for ((low, &high), &u) in low.iter_mut().zip(&high).zip(&y) {
            *low = (u - high) * (u - *low);
        }
        for (u, &outer) in y.iter_mut().zip(&low) {
            *u = (u.square() + self.minus_8100) * outer;
        }
        for (sum, &value) in sums.iter_mut().zip(&y) {
            *sum = weight.mul_add(value, *sum);
        }
    }

    /// Coefficients in `t` of `sum_i e_i * A(base + t*(v_i - base))`, from the
    /// moments `H_j = sum_i e_i * v_i^j`, `j = 0..=16`.
    ///
    /// The sum is `sum_k t^k * T_k * M_k` with `T_k = A^(k)(base) / k!` and
    /// `M_k = sum_i e_i * (v_i - base)^k`. Both are Pascal transforms: `M`
    /// of the moments by `-base`, `T` of the coefficients of `A` by `base`.
    /// Every pass reads only values of the pass before it.
    pub(super) fn moment_coefficients(
        &self,
        base: F,
        moments: &[F; INNER_NODES],
    ) -> [F; INNER_NODES] {
        let minus_base = -base;
        let mut shifted = *moments;
        let mut taylor = self.coefficients;
        for pass in 0..POWERS {
            let previous = shifted;
            for (target, &below) in shifted
                .iter_mut()
                .skip(pass + 1)
                .zip(previous.iter().skip(pass))
            {
                *target = minus_base.mul_add(below, *target);
            }
            let first = POWERS - 1 - pass;
            let previous = taylor;
            for (target, &above) in taylor
                .iter_mut()
                .skip(first)
                .zip(previous.iter().skip(first + 1))
            {
                *target = base.mul_add(above, *target);
            }
        }
        for (coefficient, &moment) in taylor.iter_mut().zip(&shifted) {
            *coefficient *= moment;
        }
        taylor
    }
}

/// Add `weight * v^j` to `sums[j]`, `j = 0..=16`, from the powers `v^1..v^16`.
#[inline(always)]
pub(super) fn add_moments<F: Field>(sums: &mut [F], weight: F, powers: &[F]) {
    let Some((first, rest)) = sums.split_first_mut() else {
        return;
    };
    *first += weight;
    for (sum, &power) in rest.iter_mut().zip(powers) {
        *sum = weight.mul_add(power, *sum);
    }
}

/// Fill `row[j - 1] = value^j`. Each power is the product of two earlier
/// ones of about half its exponent, so the chain of dependent
/// multiplications has logarithmic depth.
pub(super) fn fill_powers<F: Field>(value: F, row: &mut [F]) {
    for exponent in 1..=row.len() {
        let (done, rest) = row.split_at_mut(exponent - 1);
        let Some(target) = rest.first_mut() else {
            return;
        };
        let half = exponent / 2;
        *target = match half
            .checked_sub(1)
            .and_then(|index| done.get(index))
            .zip(done.get(exponent - half - 1))
        {
            Some((&low, &high)) => low * high,
            None => value,
        };
    }
}

/// Values at `t = 0..=16` of the polynomial with these coefficients in `t`,
/// or in `1 - t` when `reversed`.
pub(super) fn node_values<F: Field>(
    coefficients: &[F; INNER_NODES],
    reversed: bool,
) -> [F; INNER_NODES] {
    let mut values = [F::zero(); INNER_NODES];
    for (node, value) in (0u64..).zip(&mut values) {
        let node = F::from_u64(node);
        let point = if reversed { F::one() - node } else { node };
        *value = coefficients
            .iter()
            .rev()
            .fold(F::zero(), |sum, &coefficient| {
                point.mul_add(sum, coefficient)
            });
    }
    values
}
