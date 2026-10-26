//! Exact integer oracle and the bounds for lazy signed 32-bit butterflies.

use super::{TrinomialError, TrinomialLimbSlots, DEGREE, LANES};

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

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Debug)]
pub(super) struct PackedTwiddles {
    pub(super) values: [i32; LANES],
    pub(super) quotients: [i32; LANES],
}

#[derive(Clone, Debug)]
pub(super) struct Split {
    #[cfg(target_arch = "x86_64")]
    pub(super) packed: [PackedTwiddles; 4],
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
#[cfg(test)]
pub(super) fn sqrdmulh(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b) + (1 << 30)) >> 31).min(i64::from(i32::MAX)) as i32
}

impl Arithmetic {
    #[inline(always)]
    pub(super) fn center(self, value: i64) -> i32 {
        let p = i64::from(self.prime);
        const PRIMES: [u32; 4] = super::TrinomialLimbDomain::ADMITTED_PRIMES;
        const RECIPROCALS: [i64; 4] = {
            let mut result = [0; 4];
            let mut index = 0;
            while index < 4 {
                let p = PRIMES[index] as u128;
                result[index] = (((1u128 << 64) + p / 2) / p) as i64;
                index += 1;
            }
            result
        };
        let reciprocal = match self.prime {
            prime if prime == PRIMES[0] => RECIPROCALS[0],
            prime if prime == PRIMES[1] => RECIPROCALS[1],
            prime if prime == PRIMES[2] => RECIPROCALS[2],
            prime if prime == PRIMES[3] => RECIPROCALS[3],
            _ => 0,
        };
        let residue = if reciprocal == 0 {
            value % p
        } else {
            // M=round(2^64/p), hence |M-2^64/p|<=1/2. For every i64 v
            // (including MIN), |v|<=2^63. Rounding v*M/2^64 to q therefore
            // gives |q-v/p|<=1/2+1/4=3/4, so |v-q*p|<=3p/4<p.
            // This covers the ENTIRE checkpoint interval h+128*(p-1)^2,
            // not just dot products. M<2^40 makes v*M+2^63 fit i128, and
            // |q|<2^39 fits i64. q*p can exceed i64 near its endpoints;
            // wrapping reconstructs the exact small signed residue modulo
            // 2^64. One correction below then yields the unique centered value.
            let quotient =
                ((i128::from(value) * i128::from(reciprocal) + (1i128 << 63)) >> 64) as i64;
            value.wrapping_sub(quotient.wrapping_mul(p))
        };
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
        let quotient = ((i64::from(a) * i64::from(w.quotient) + (1 << 30)) >> 31) as i32;
        a.wrapping_mul(w.value)
            .wrapping_sub(quotient.wrapping_mul(self.prime as i32))
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
                reason: "singular limb interpolation matrix",
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
    let step = split.stride * LANES;
    let start = split.start * LANES;
    let (a, rest) = values[start..start + 3 * step].split_at_mut(step);
    let (b, c) = rest.split_at_mut(step);
    let (a, _) = a.as_chunks_mut::<LANES>();
    let (b, _) = b.as_chunks_mut::<LANES>();
    let (c, _) = c.as_chunks_mut::<LANES>();
    // Separate value/quotient arrays give the vectorizer contiguous weights;
    // copy complete input rows and defer stores until all lanes are computed.
    // Wrapping operations below are exact: the forward B1..B4 and inverse I
    // bounds in limb.rs keep every mathematical sum/difference inside i32.
    if inverse {
        let r = split.ri.map(|w| w.value);
        let q = split.ri.map(|w| w.quotient);
        let r2 = split.ri2.map(|w| w.value);
        let q2 = split.ri2.map(|w| w.quotient);
        let p = arithmetic.prime as i32;
        let half = p / 2;
        // Inputs are centered, so |o0| <= 3h <= p+h. The other outputs
        // are multiplied first and obey C(I)<p (see limb.rs). Each needs
        // at most one correction; comparisons keep the lane loop branch-free.
        let center = |value: i32| {
            value
                .wrapping_sub(p & -i32::from(value > half))
                .wrapping_add(p & -i32::from(value < -half))
        };
        for ((a, b), c) in a.iter_mut().zip(b).zip(c) {
            let (av, bv, cv) = (*a, *b, *c);
            let (mut o0, mut o1, mut o2) = ([0; LANES], [0; LANES], [0; LANES]);
            for lane in 0..LANES {
                let difference = arithmetic.mul(bv[lane].wrapping_sub(cv[lane]), inverse_omega);
                o0[lane] = center(av[lane].wrapping_add(bv[lane]).wrapping_add(cv[lane]));
                o1[lane] = center(arithmetic.mul(
                    av[lane].wrapping_sub(cv[lane]).wrapping_add(difference),
                    Twiddle {
                        value: r[lane],
                        quotient: q[lane],
                    },
                ));
                o2[lane] = center(arithmetic.mul(
                    av[lane].wrapping_sub(bv[lane]).wrapping_sub(difference),
                    Twiddle {
                        value: r2[lane],
                        quotient: q2[lane],
                    },
                ));
            }
            *a = o0;
            *b = o1;
            *c = o2;
        }
    } else {
        let r = split.r.map(|w| w.value);
        let q = split.r.map(|w| w.quotient);
        let r2 = split.r2.map(|w| w.value);
        let q2 = split.r2.map(|w| w.quotient);
        for ((a, b), c) in a.iter_mut().zip(b).zip(c) {
            let (av, bv, cv) = (*a, *b, *c);
            let (mut o0, mut o1, mut o2) = ([0; LANES], [0; LANES], [0; LANES]);
            for lane in 0..LANES {
                let bv = arithmetic.mul(
                    bv[lane],
                    Twiddle {
                        value: r[lane],
                        quotient: q[lane],
                    },
                );
                let cv = arithmetic.mul(
                    cv[lane],
                    Twiddle {
                        value: r2[lane],
                        quotient: q2[lane],
                    },
                );
                let difference = arithmetic.mul(bv.wrapping_sub(cv), omega);
                o0[lane] = av[lane].wrapping_add(bv).wrapping_add(cv);
                o1[lane] = av[lane].wrapping_sub(cv).wrapping_add(difference);
                o2[lane] = av[lane].wrapping_sub(bv).wrapping_sub(difference);
            }
            *a = o0;
            *b = o1;
            *c = o2;
        }
    }
}

pub(super) fn accumulate(sums: &mut [i64; DEGREE], lhs: &[i32; DEGREE], rhs: &[i32; DEGREE]) {
    for ((sum, &a), &b) in sums.iter_mut().zip(lhs).zip(rhs) {
        *sum += i64::from(a) * i64::from(b);
    }
}

/// Retain four accumulator lanes across all products in this checkpoint slice.
pub(super) fn accumulate_batch(
    sums: &mut [i64; DEGREE],
    lhs: &[TrinomialLimbSlots],
    rhs: &[TrinomialLimbSlots],
) {
    for (chunk, sums) in sums.chunks_exact_mut(4).enumerate() {
        let offset = 4 * chunk;
        let mut lanes = [sums[0], sums[1], sums[2], sums[3]];
        for (a, b) in lhs.iter().zip(rhs) {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                *sum += i64::from(a.values[offset + lane]) * i64::from(b.values[offset + lane]);
            }
        }
        sums.copy_from_slice(&lanes);
    }
}
