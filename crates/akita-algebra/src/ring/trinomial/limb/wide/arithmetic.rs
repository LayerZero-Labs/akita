//! Exact integer oracle and the bounds for lazy signed 32-bit butterflies.

use super::{TrinomialError, DEGREE, LANES};

#[derive(Clone, Copy, Debug)]
pub(super) struct Arithmetic {
    pub(super) prime: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct Twiddle {
    pub(super) value: i32,
    pub(super) quotient: i32,
}

// The SIMD deinterleaving load consumes value/quotient pairs without padding.
const _: [(); 8] = [(); std::mem::size_of::<Twiddle>()];

#[derive(Clone, Debug)]
pub(super) struct Split {
    pub(super) start: usize,
    pub(super) stride: usize,
    pub(super) r: [Twiddle; LANES],
    pub(super) r2: [Twiddle; LANES],
    pub(super) ri: [Twiddle; LANES],
    pub(super) ri2: [Twiddle; LANES],
}

impl Twiddle {
    pub(super) fn new(arithmetic: Arithmetic, value: i32) -> Self {
        let numerator = i64::from(value) * (1 << 31);
        let p = i64::from(arithmetic.prime);
        Self {
            value,
            quotient: ((numerator + p / 2).div_euclid(p)) as i32,
        }
    }
}

/// Scalar SQRDMULH, including its sole saturating pair MIN * MIN.
pub(super) fn sqrdmulh(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b) + (1 << 30)) >> 31).min(i64::from(i32::MAX)) as i32
}

impl Arithmetic {
    pub(super) fn center(self, value: i64) -> i32 {
        let p = i64::from(self.prime);
        let residue = value % p;
        if residue > p / 2 {
            (residue - p) as i32
        } else if residue < -p / 2 {
            (residue + p) as i32
        } else {
            residue as i32
        }
    }

    #[inline(always)]
    pub(super) fn mul(self, a: i32, w: Twiddle) -> i32 {
        // Let Q=2^31, w'=round(wQ/p), and t=floor(aw'/Q+1/2).
        // |w'-wQ/p| <= 1/2 and |t-aw'/Q| <= 1/2 imply
        // |aw-tp| <= p/2 + |a|p/2^32. Centered w has |w'|<=2^30:
        // SQRDMULH's sole saturation (MIN,MIN) is therefore impossible.
        // The exact result fits i32 for every i32 operand, so taking the low
        // 32 bits of both products then subtracting recovers that result.
        a.wrapping_mul(w.value)
            .wrapping_sub(sqrdmulh(a, w.quotient).wrapping_mul(self.prime as i32))
    }

    pub(super) fn pow(self, value: u32, mut exponent: u32) -> u32 {
        let p = u64::from(self.prime);
        let mut base = u64::from(value);
        let mut result = 1;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result * base % p;
            }
            base = base * base % p;
            exponent >>= 1;
        }
        result as u32
    }

    pub(super) fn dot(self, input: &[i32], weights: &[i32; LANES]) -> i32 {
        // Eight centered products have total magnitude <= 8*(p/2)^2 < 2^57.
        self.center(
            input
                .iter()
                .zip(weights)
                .map(|(&a, &b)| i64::from(a) * i64::from(b))
                .sum(),
        )
    }
}

pub(super) fn multiply_bound(prime: u32, operand: i64) -> i64 {
    let p = i64::from(prime);
    // Ceiling of p/2 + operand*p/2^32, with one common denominator.
    ((p << 31) + operand * p + (1i64 << 32) - 1) >> 32
}

pub(super) fn forward_bounds(prime: u32) -> [i64; 5] {
    let mut result = [0; 5];
    result[0] = i64::from(prime / 2);
    for level in 0..4 {
        let b = result[level];
        let c = multiply_bound(prime, b);
        result[level + 1] = (b + 2 * c).max(b + c + multiply_bound(prime, 2 * c));
    }
    result
}

pub(super) fn invert_evaluation(
    arithmetic: Arithmetic,
    evaluation: &[[i32; LANES]; LANES],
) -> Result<[[i32; LANES]; LANES], TrinomialError> {
    let p = u64::from(arithmetic.prime);
    let mut matrix = [[0u64; 2 * LANES]; LANES];
    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, &value) in evaluation[i].iter().enumerate() {
            row[j] = i64::from(value).rem_euclid(p as i64) as u64;
        }
        row[LANES + i] = 1;
    }
    for column in 0..LANES {
        let pivot = (column..LANES)
            .find(|&row| matrix[row][column] != 0)
            .ok_or(TrinomialError::LimbInput {
                reason: "singular wide limb interpolation matrix",
            })?;
        matrix.swap(column, pivot);
        let inverse =
            u64::from(arithmetic.pow(matrix[column][column] as u32, arithmetic.prime - 2));
        for value in &mut matrix[column] {
            *value = *value * inverse % p;
        }
        for row in 0..LANES {
            if row != column {
                let factor = matrix[row][column];
                let pivot_row = matrix[column];
                for (value, pivot) in matrix[row].iter_mut().zip(pivot_row) {
                    *value = (*value + p - factor * pivot % p) % p;
                }
            }
        }
    }
    let inverse_81 = u64::from(arithmetic.pow(81, arithmetic.prime - 2));
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| arithmetic.center((matrix[i][LANES + j] * inverse_81 % p) as i64))
    }))
}

pub(super) fn butterfly(
    arithmetic: Arithmetic,
    omega: Twiddle,
    inverse_omega: Twiddle,
    values: &mut [i32; DEGREE],
    split: &Split,
    inverse: bool,
) {
    for t in 0..split.stride {
        let offset = (split.start + t) * LANES;
        let step = split.stride * LANES;
        for lane in 0..LANES {
            let a = values[offset + lane];
            let mut b = values[offset + step + lane];
            let mut c = values[offset + 2 * step + lane];
            if !inverse {
                b = arithmetic.mul(b, split.r[lane]);
                c = arithmetic.mul(c, split.r2[lane]);
            }
            let difference = arithmetic.mul(b - c, if inverse { inverse_omega } else { omega });
            let o0 = a + b + c;
            let o1 = a - c + difference;
            let o2 = a - b - difference;
            if inverse {
                // Inputs centered: |o0| <= 3p/2; |difference|<=C(p), hence
                // |o1|,|o2|<=p+C(p). All intermediates fit signed i32.
                values[offset + lane] = arithmetic.center(i64::from(o0));
                values[offset + step + lane] =
                    arithmetic.center(i64::from(arithmetic.mul(o1, split.ri[lane])));
                values[offset + 2 * step + lane] =
                    arithmetic.center(i64::from(arithmetic.mul(o2, split.ri2[lane])));
            } else {
                values[offset + lane] = o0;
                values[offset + step + lane] = o1;
                values[offset + 2 * step + lane] = o2;
            }
        }
    }
}

pub(super) fn accumulate(sums: &mut [i64; DEGREE], lhs: &[i32; DEGREE], rhs: &[i32; DEGREE]) {
    for ((sum, &a), &b) in sums.iter_mut().zip(lhs).zip(rhs) {
        *sum += i64::from(a) * i64::from(b);
    }
}
