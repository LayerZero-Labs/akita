//! Exact unit-stride intervals via shared binary-addition contractions.
use super::{basis_bit_factor, basis_pair_factors, charge_work, checked_axis_offset, EqPairSeed};
use crate::Field;
use akita_error::AkitaError;
use std::collections::BTreeMap;

fn high_eval<F: Field, const MONOMIAL: bool>(
    point: &[F],
    bit: usize,
    index: usize,
    cache: &mut [BTreeMap<usize, F>],
    work: &mut usize,
) -> Result<F, AkitaError> {
    let Some(&challenge) = point.get(bit) else {
        return Ok(if index == 0 { F::one() } else { F::zero() });
    };
    let memo = cache
        .get(bit)
        .ok_or_else(|| AkitaError::Internal("interval high cache is absent".into()))?;
    // Ordered-map operations have logarithmic work, including cache hits.
    charge_work(
        work,
        1 + (usize::BITS - memo.len().leading_zeros()) as usize,
    )?;
    if let Some(value) = memo.get(&index) {
        return Ok(*value);
    }
    charge_work(work, 1)?;
    let high = high_eval::<F, MONOMIAL>(point, bit + 1, index >> 1, cache, work)?;
    let factor = basis_bit_factor::<F, MONOMIAL>(Some(challenge), index & 1)
        .ok_or_else(|| AkitaError::Internal("interval high bit factor is absent".into()))?;
    let value = high * factor;
    let memo = cache
        .get_mut(bit)
        .ok_or_else(|| AkitaError::Internal("interval high cache is absent".into()))?;
    // Charge the tree update and storage before inserting a computed suffix.
    charge_work(
        work,
        2 + (usize::BITS - memo.len().leading_zeros()) as usize,
    )?;
    memo.insert(index, value);
    Ok(value)
}

/// Sum the low `bits` free left coordinates and their shifted right coordinates,
/// retaining the carry into the right high coordinates. Cache by exact shift.
fn low_eval<F: Field, const STATES: usize>(
    bits: usize,
    shift: usize,
    factors: &[[F; 4]],
    cache: &mut [BTreeMap<usize, [F; STATES]>],
    work: &mut usize,
) -> Result<[F; STATES], AkitaError> {
    if bits == 0 {
        let mut value = [F::zero(); STATES];
        *value
            .first_mut()
            .ok_or_else(|| AkitaError::Internal("interval has no carry states".into()))? = F::one();
        return Ok(value);
    }
    let memo = cache
        .get(bits)
        .ok_or_else(|| AkitaError::Internal("interval low cache is absent".into()))?;
    // Charge ordered-map lookups even when the carry summary is cached.
    charge_work(
        work,
        1 + (usize::BITS - memo.len().leading_zeros()) as usize,
    )?;
    if let Some(value) = memo.get(&shift) {
        return Ok(*value);
    }
    charge_work(work, 2 * STATES)?;
    let half = 1usize << (bits - 1);
    let previous = low_eval::<F, STATES>(bits - 1, shift & (half - 1), factors, cache, work)?;
    let f = factors
        .get(bits - 1)
        .ok_or_else(|| AkitaError::Internal("interval low factor is absent".into()))?;
    let mut value = [F::zero(); STATES];
    if STATES == 2 {
        let (a, b) = (previous[0], previous[1]);
        let equal = f[0] + f[3];
        if shift & half == 0 {
            value[0] = a * equal + b * f[1];
            value[1] = b * f[2];
        } else {
            value[0] = a * f[1];
            value[1] = a * f[2] + b * equal;
        }
    } else {
        let shift_bit = usize::from(shift & half != 0);
        for (carry, weight) in previous.into_iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            for left_bit in 0..=1 {
                let sum = carry + shift_bit + (STATES - 1) * left_bit;
                let target = value.get_mut(sum >> 1).ok_or_else(|| {
                    AkitaError::Internal("interval carry exceeds its bound".into())
                })?;
                *target += weight * f[2 * left_bit + (sum & 1)];
            }
        }
    }
    let memo = cache
        .get_mut(bits)
        .ok_or_else(|| AkitaError::Internal("interval low cache is absent".into()))?;
    // Charge the tree update and storage before inserting a carry summary.
    charge_work(
        work,
        2 + (usize::BITS - memo.len().leading_zeros()) as usize,
    )?;
    memo.insert(shift, value);
    Ok(value)
}

pub(super) fn eval<
    F: Field,
    const LEFT_MONOMIAL: bool,
    const RIGHT_MONOMIAL: bool,
    const STATES: usize,
>(
    left: &[F],
    right: &[F],
    len: usize,
    seeds: &[EqPairSeed<F>],
    work: &mut usize,
) -> Result<F, AkitaError> {
    if len == 0 {
        return Ok(F::zero());
    }
    if STATES < 2 {
        return Err(AkitaError::Internal(
            "interval stride must be positive".into(),
        ));
    }
    let stride = STATES - 1;
    let left_domain = 1usize << left.len();
    let right_domain = 1usize << right.len();
    let bits = (usize::BITS - 1 - len.leading_zeros()) as usize;
    let mut factors = Vec::new();
    for bit in 0..bits.min(left.len()).min(right.len()) {
        let values = basis_pair_factors::<F, LEFT_MONOMIAL, RIGHT_MONOMIAL>(
            left.get(bit).copied(),
            right.get(bit).copied(),
        );
        let [Some(a), Some(b), Some(c), Some(d)] = values else {
            return Err(AkitaError::Internal(
                "interval pair bit factor is absent".into(),
            ));
        };
        factors.push([a, b, c, d]);
    }
    let mut low_cache = (0..=bits).map(|_| BTreeMap::new()).collect::<Vec<_>>();
    let mut left_cache = (0..left.len()).map(|_| BTreeMap::new()).collect::<Vec<_>>();
    let mut right_cache = (0..right.len())
        .map(|_| BTreeMap::new())
        .collect::<Vec<_>>();
    let mut out = F::zero();
    for seed in seeds {
        checked_axis_offset(seed.left_offset, 1, len - 1, "left")?;
        checked_axis_offset(seed.right_offset, stride, len - 1, "right")?;
        if seed.left_offset >= left_domain || seed.right_offset >= right_domain {
            continue;
        }
        let live = len
            .min(left_domain - seed.left_offset)
            .min((right_domain - 1 - seed.right_offset) / stride + 1);
        let mut cursor = 0usize;
        let mut value = F::zero();
        while cursor < live {
            charge_work(work, 1)?;
            // Previous endpoint checks cover these sums and all later carries.
            let l = seed.left_offset + cursor;
            let r = seed.right_offset + stride * cursor;
            let remaining_bits = (usize::BITS - 1 - (live - cursor).leading_zeros()) as usize;
            let block_bits = remaining_bits.min(l.trailing_zeros() as usize);
            let size = 1usize << block_bits;
            let shift = r & (size - 1);
            let carries = low_eval::<F, STATES>(block_bits, shift, &factors, &mut low_cache, work)?;
            let lh = high_eval::<F, LEFT_MONOMIAL>(
                left,
                block_bits,
                l >> block_bits,
                &mut left_cache,
                work,
            )?;
            let rh = r >> block_bits;
            let mut right_value = F::zero();
            for (carry, weight) in carries.into_iter().enumerate() {
                if weight.is_zero() {
                    continue;
                }
                let high_index = rh.checked_add(carry).ok_or_else(|| {
                    AkitaError::InvalidInput("interval high address overflow".into())
                })?;
                let high = high_eval::<F, RIGHT_MONOMIAL>(
                    right,
                    block_bits,
                    high_index,
                    &mut right_cache,
                    work,
                )?;
                right_value += weight * high;
            }
            value += lh * right_value;
            cursor += size;
        }
        out += seed.weight * value;
    }
    Ok(out)
}

pub(super) fn eval_scaled_right<F: Field, const LM: bool, const RM: bool, const STATES: usize>(
    left: &[F],
    right: &[F],
    stride: usize,
    len: usize,
    seeds: &[EqPairSeed<F>],
    work: &mut usize,
) -> Result<F, AkitaError> {
    if len == 0 {
        return Ok(F::zero());
    }
    let bits = stride.trailing_zeros() as usize;
    if stride == 0 || STATES < 2 || stride >> bits != STATES - 1 {
        return Err(AkitaError::InvalidInput(
            "interval stride does not match its carry bound".into(),
        ));
    }
    if bits == 0 {
        return eval::<F, LM, RM, STATES>(left, right, len, seeds, work);
    }
    let (low, high) = right
        .split_at_checked(bits)
        .ok_or_else(|| AkitaError::InvalidInput("interval stride exceeds right arity".into()))?;
    charge_work(
        work,
        seeds
            .len()
            .checked_mul(bits + 1)
            .ok_or_else(|| AkitaError::InvalidInput("interval seed factor work overflow".into()))?,
    )?;
    let mut reduced = Vec::new();
    reduced
        .try_reserve_exact(seeds.len())
        .map_err(|_| AkitaError::InvalidInput("interval seed allocation failed".into()))?;
    for seed in seeds {
        checked_axis_offset(seed.right_offset, stride, len - 1, "right")?;
        reduced.push(EqPairSeed {
            left_offset: seed.left_offset,
            right_offset: seed.right_offset >> bits,
            weight: seed.weight
                * super::basis_eval_at_index::<F, RM>(
                    low,
                    seed.right_offset & ((1usize << bits) - 1),
                ),
        });
    }
    eval::<F, LM, RM, STATES>(left, high, len, &reduced, work)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{One, Prime128OffsetA7F7 as F};

    #[test]
    fn suffix_cache_hits_and_misses_respect_the_work_limit() {
        let point = [F::one(); 4];
        let mut cache = (0..point.len())
            .map(|_| BTreeMap::new())
            .collect::<Vec<_>>();
        high_eval::<F, false>(&point, 0, 0, &mut cache, &mut 0).unwrap();
        for index in [0, 1] {
            let mut work = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS;
            assert!(high_eval::<F, false>(&point, 0, index, &mut cache, &mut work).is_err());
        }
    }

    fn check_low_cache_work_limit<const STATES: usize>() {
        let factors = [[F::one(); 4]; 4];
        let mut cache = (0..=2).map(|_| BTreeMap::new()).collect::<Vec<_>>();
        low_eval::<F, STATES>(2, 0, &factors, &mut cache, &mut 0).unwrap();
        let warmed = cache.clone();
        for shift in [0, 1] {
            let mut work = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS;
            assert!(low_eval::<F, STATES>(2, shift, &factors, &mut cache, &mut work).is_err());
            assert_eq!(cache, warmed);
        }

        // Leave enough work for the empty lookup and arithmetic, but not for
        // storing the result. The failed insertion must leave the cache empty.
        let mut cache = (0..=1).map(|_| BTreeMap::new()).collect::<Vec<_>>();
        let mut work = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS - (1 + 2 * STATES);
        assert!(low_eval::<F, STATES>(1, 0, &factors, &mut cache, &mut work).is_err());
        assert!(cache.iter().all(BTreeMap::is_empty));

        // A hit in a larger tree must be charged more than the same hit in a
        // singleton tree, despite performing no field arithmetic.
        let mut cache = (0..=4).map(|_| BTreeMap::new()).collect::<Vec<_>>();
        cache[4].insert(0, [F::one(); STATES]);
        let mut small_work = 0;
        low_eval::<F, STATES>(4, 0, &factors, &mut cache, &mut small_work).unwrap();
        for shift in 1..8 {
            cache[4].insert(shift, [F::one(); STATES]);
        }
        let mut large_work = 0;
        low_eval::<F, STATES>(4, 0, &factors, &mut cache, &mut large_work).unwrap();
        assert!(large_work > small_work);
    }

    #[test]
    fn low_cache_hits_misses_and_storage_respect_the_work_limit() {
        check_low_cache_work_limit::<2>();
        check_low_cache_work_limit::<4>();
        check_low_cache_work_limit::<6>();
        check_low_cache_work_limit::<8>();
    }
}
