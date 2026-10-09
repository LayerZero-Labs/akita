//! Portable arithmetic and the scalar transform oracle.

use super::{TrinomialError, TrinomialLimbDomain, DEGREE, LANES};

#[derive(Clone, Copy, Debug)]
pub(super) struct Arithmetic {
    pub(super) prime: u16,
    pub(super) pinv: i16,
}

#[derive(Clone, Debug)]
pub(super) struct Split {
    pub(super) start: usize,
    pub(super) stride: usize,
    pub(super) r: [i16; LANES],
    pub(super) r2: [i16; LANES],
    pub(super) ri: [i16; LANES],
    pub(super) ri2: [i16; LANES],
}

impl Arithmetic {
    pub(super) fn new(prime: u16) -> Self {
        let mut inverse = 1u16;
        // Newton lifting doubles the correct bits on each iteration.
        for _ in 0..4 {
            inverse = inverse.wrapping_mul(2u16.wrapping_sub(prime.wrapping_mul(inverse)));
        }
        Self {
            prime,
            pinv: inverse as i16,
        }
    }

    #[inline(always)]
    pub(super) fn center(self, value: i32) -> i16 {
        let p = i32::from(self.prime);
        let half = p / 2;
        let value = if value > half { value - p } else { value };
        (if value < -half { value + p } else { value }) as i16
    }

    #[inline(always)]
    pub(super) fn add(self, a: i16, b: i16) -> i16 {
        self.center(i32::from(a) + i32::from(b))
    }

    #[inline(always)]
    pub(super) fn sub(self, a: i16, b: i16) -> i16 {
        self.center(i32::from(a) - i32::from(b))
    }

    #[inline(always)]
    pub(super) fn mul(self, a: i16, b: i16) -> i16 {
        let product = i32::from(a) * i32::from(b);
        let correction = (product as i16).wrapping_mul(self.pinv);
        // Centered operands give |q| <= p/2 + p^2/(4R) < p. Subtraction
        // fits i32: |product| + 32768*p < 732 million for admitted p.
        self.center((product - i32::from(correction) * i32::from(self.prime)) >> 16)
    }

    pub(super) fn reduce_wide(self, product: i64) -> i16 {
        let correction = (product as i16).wrapping_mul(self.pinv);
        let quotient = (product - i64::from(correction) * i64::from(self.prime)) >> 16;
        self.center((quotient % i64::from(self.prime)) as i32)
    }

    pub(super) fn encode(self, value: u16) -> i16 {
        self.center(((u32::from(value) << 16) % u32::from(self.prime)) as i32)
    }

    pub(super) fn decode(self, value: i16) -> u16 {
        let value = self.reduce_wide(i64::from(value));
        if value < 0 {
            (i32::from(value) + i32::from(self.prime)) as u16
        } else {
            value as u16
        }
    }

    pub(super) fn pow(self, value: u16, mut exponent: u32) -> u16 {
        let p = u32::from(self.prime);
        let mut base = u32::from(value);
        let mut result = 1;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result * base % p;
            }
            base = base * base % p;
            exponent >>= 1;
        }
        result as u16
    }

    pub(super) fn dot(self, input: &[i16], weights: &[i16; LANES]) -> i16 {
        input
            .iter()
            .zip(weights)
            .fold(0, |sum, (&a, &b)| self.add(sum, self.mul(a, b)))
    }
}

pub(super) fn invert_evaluation(
    arithmetic: &Arithmetic,
    evaluation: &[[i16; LANES]; LANES],
) -> Result<[[i16; LANES]; LANES], TrinomialError> {
    let p = u32::from(arithmetic.prime);
    let mut matrix = [[0u32; 2 * LANES]; LANES];
    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, value) in evaluation[i].iter().enumerate() {
            row[j] = u32::from(arithmetic.decode(*value));
        }
        row[LANES + i] = 1;
    }
    for column in 0..LANES {
        let pivot = (column..LANES)
            .find(|&row| matrix[row][column] != 0)
            .ok_or(TrinomialError::LimbInput {
                reason: "singular limb interpolation matrix",
            })?;
        matrix.swap(column, pivot);
        let inverse = u32::from(arithmetic.pow(matrix[column][column] as u16, p - 2));
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
    let inverse_81 = u32::from(arithmetic.pow(81, p - 2));
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| arithmetic.encode((matrix[i][LANES + j] * inverse_81 % p) as u16))
    }))
}

pub(super) fn butterfly(
    domain: &TrinomialLimbDomain,
    values: &mut [i16; DEGREE],
    split: &Split,
    inverse: bool,
) {
    let arithmetic = domain.arithmetic;
    let omega = if inverse {
        domain.inverse_omega
    } else {
        domain.omega
    };
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
            let difference = arithmetic.mul(arithmetic.sub(b, c), omega);
            let o0 = arithmetic.add(arithmetic.add(a, b), c);
            let mut o1 = arithmetic.add(arithmetic.sub(a, c), difference);
            let mut o2 = arithmetic.sub(arithmetic.sub(a, b), difference);
            if inverse {
                o1 = arithmetic.mul(o1, split.ri[lane]);
                o2 = arithmetic.mul(o2, split.ri2[lane]);
            }
            values[offset + lane] = o0;
            values[offset + step + lane] = o1;
            values[offset + 2 * step + lane] = o2;
        }
    }
}

pub(super) fn accumulate(sums: &mut [i32; DEGREE], lhs: &[i16; DEGREE], rhs: &[i16; DEGREE]) {
    for ((sum, &a), &b) in sums.iter_mut().zip(lhs).zip(rhs) {
        *sum += i32::from(a) * i32::from(b);
    }
}
