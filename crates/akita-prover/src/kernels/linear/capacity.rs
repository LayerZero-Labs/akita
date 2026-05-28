use super::*;

#[cfg(test)]
pub(super) const BALANCED_DIGIT_RHS_MAX_ABS: u64 = 32;
#[cfg(test)]
pub(super) const I8_RHS_MAX_ABS: u64 = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
struct SmallNat {
    limbs: Vec<u32>,
}

impl SmallNat {
    fn one() -> Self {
        Self { limbs: vec![1] }
    }

    fn trim(&mut self) {
        while self.limbs.len() > 1 && self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn mul_u128(&mut self, rhs: u128) {
        if rhs == 0 {
            self.limbs.clear();
            self.limbs.push(0);
            return;
        }

        let mut rhs_limbs = Vec::new();
        let mut x = rhs;
        while x != 0 {
            rhs_limbs.push(x as u32);
            x >>= 32;
        }

        let mut out = vec![0u32; self.limbs.len() + rhs_limbs.len()];
        for (i, &lhs_limb) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &rhs_limb) in rhs_limbs.iter().enumerate() {
                let idx = i + j;
                let accum =
                    u128::from(out[idx]) + u128::from(lhs_limb) * u128::from(rhs_limb) + carry;
                out[idx] = accum as u32;
                carry = accum >> 32;
            }
            let mut idx = i + rhs_limbs.len();
            while carry != 0 {
                if idx == out.len() {
                    out.push(0);
                }
                let accum = u128::from(out[idx]) + carry;
                out[idx] = accum as u32;
                carry = accum >> 32;
                idx += 1;
            }
        }

        self.limbs = out;
        self.trim();
    }
}

impl Ord for SmallNat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            std::cmp::Ordering::Equal => self.limbs.iter().rev().cmp(other.limbs.iter().rev()),
            ordering => ordering,
        }
    }
}

impl PartialOrd for SmallNat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Conservative maximum number of products that may be accumulated in one CRT
/// accumulator before Garner reconstruction.
pub(super) fn max_safe_crt_accumulation_width<
    F: CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    params: &CrtNttParamSet<W, K, D>,
    rhs_abs_bound: u64,
) -> Option<usize> {
    if rhs_abs_bound == 0 {
        return Some(usize::MAX);
    }

    let modulus = (-F::one()).to_canonical_u128() + 1;
    let setup_abs_bound = modulus;
    if setup_abs_bound == 0 || D == 0 {
        return None;
    }

    let mut crt_product = SmallNat::one();
    for prime in &params.primes {
        crt_product.mul_u128(prime.p.to_i64() as u128);
    }

    if !crt_width_is_safe::<F, D>(&crt_product, 1, rhs_abs_bound) {
        return None;
    }

    let mut lo = 1usize;
    let mut hi = 2usize;
    while crt_width_is_safe::<F, D>(&crt_product, hi, rhs_abs_bound) {
        lo = hi;
        let Some(next) = hi.checked_mul(2) else {
            if crt_width_is_safe::<F, D>(&crt_product, usize::MAX, rhs_abs_bound) {
                return Some(usize::MAX);
            }
            hi = usize::MAX;
            break;
        };
        hi = next;
    }

    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if crt_width_is_safe::<F, D>(&crt_product, mid, rhs_abs_bound) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Some(lo)
}

fn crt_width_is_safe<F: CanonicalField, const D: usize>(
    crt_product: &SmallNat,
    width: usize,
    rhs_abs_bound: u64,
) -> bool {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let setup_abs_bound = modulus;

    let mut lhs = SmallNat::one();
    lhs.mul_u128(2);
    lhs.mul_u128(width as u128);
    lhs.mul_u128(D as u128);
    lhs.mul_u128(setup_abs_bound);
    lhs.mul_u128(u128::from(rhs_abs_bound));
    lhs < *crt_product
}

pub(super) fn safe_crt_chunk_width<
    F: CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    params: &CrtNttParamSet<W, K, D>,
    full_width: usize,
    rhs_abs_bound: u64,
) -> Option<usize> {
    if full_width == 0 {
        return Some(0);
    }
    max_safe_crt_accumulation_width::<F, W, K, D>(params, rhs_abs_bound)
        .map(|safe_width| safe_width.min(full_width))
        .filter(|&chunk_width| chunk_width > 0)
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use akita_algebra::ntt::tables::{q128_primes, Q128_NUM_PRIMES, Q32_PRIMES};
    use akita_field::{Fp64, Prime128Offset275};

    #[test]
    fn q128_digit_capacity_matches_expected_scale() {
        const D: usize = 64;
        let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        let width = max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
            &params,
            BALANCED_DIGIT_RHS_MAX_ABS,
        )
        .expect("one i8 term should fit");

        assert_eq!(width, 1023);
    }

    #[test]
    fn q128_balanced_digit_bound_recovers_chunk_width() {
        const D: usize = 64;
        let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        let balanced_width =
            max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
                &params,
                BALANCED_DIGIT_RHS_MAX_ABS,
            )
            .expect("one balanced digit term should fit");
        let full_i8_width =
            max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
                &params,
                I8_RHS_MAX_ABS,
            )
            .expect("one full i8 term should fit");

        assert_eq!(balanced_width, 4 * full_i8_width + 3);
    }

    #[test]
    fn q128_rejects_unsafe_single_centered_term() {
        const D: usize = 128;
        let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        let width = max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
            &params, 32_768,
        );

        assert_eq!(width, None);
    }

    #[test]
    fn q32_digit_capacity_is_not_artificially_small() {
        const D: usize = 64;
        let params = CrtNttParamSet::<i16, 6, D>::new(Q32_PRIMES);
        let width = max_safe_crt_accumulation_width::<Fp64<4294967197>, i16, 6, D>(
            &params,
            BALANCED_DIGIT_RHS_MAX_ABS,
        )
        .expect("Q32 i8 path should have headroom");

        assert_eq!(width, 245_207_459_281);
    }
}
