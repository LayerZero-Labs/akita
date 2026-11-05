//! Exact unit-stride intervals via shared binary-addition contractions.
use super::unit_intervals_cache::Memo;
use super::{basis_bit_factor, basis_pair_factors, charge_work, checked_axis_offset, EqPairSeed};
use crate::Field;
use akita_error::AkitaError;
use std::collections::BTreeMap;

fn high_eval<F: Field, const MONOMIAL: bool>(
    point: &[F],
    bit: usize,
    index: usize,
    cache: &mut [Memo<F>],
    work: &mut usize,
) -> Result<F, AkitaError> {
    let Some(&challenge) = point.get(bit) else {
        return Ok(if index == 0 { F::one() } else { F::zero() });
    };
    if let Some(value) = cache
        .get(bit)
        .ok_or_else(|| AkitaError::Internal("interval high cache is absent".into()))?
        .get(index, work)?
    {
        return Ok(value);
    }
    charge_work(work, 1)?;
    let high = high_eval::<F, MONOMIAL>(point, bit + 1, index >> 1, cache, work)?;
    let factor = basis_bit_factor::<F, MONOMIAL>(Some(challenge), index & 1)
        .ok_or_else(|| AkitaError::Internal("interval high bit factor is absent".into()))?;
    let value = high * factor;
    cache
        .get_mut(bit)
        .ok_or_else(|| AkitaError::Internal("interval high cache is absent".into()))?
        .insert(index, value, work)?;
    Ok(value)
}

/// Sum the low `bits` free left coordinates and their shifted right coordinates,
/// retaining the carry into the right high coordinates. Cache by exact shift.
fn low_eval<F: Field>(
    bits: usize,
    shift: usize,
    factors: &[[F; 4]],
    cache: &mut [BTreeMap<usize, [F; 2]>],
    work: &mut usize,
) -> Result<[F; 2], AkitaError> {
    if bits == 0 {
        return Ok([F::one(), F::zero()]);
    }
    if let Some(value) = cache.get(bits).and_then(|m| m.get(&shift)) {
        return Ok(*value);
    }
    charge_work(work, 4)?;
    let half = 1usize << (bits - 1);
    let [a, b] = low_eval(bits - 1, shift & (half - 1), factors, cache, work)?;
    let f = factors
        .get(bits - 1)
        .ok_or_else(|| AkitaError::Internal("interval low factor is absent".into()))?;
    let equal = f[0] + f[3];
    let value = if shift & half == 0 {
        [a * equal + b * f[1], b * f[2]]
    } else {
        [a * f[1], a * f[2] + b * equal]
    };
    cache
        .get_mut(bits)
        .ok_or_else(|| AkitaError::Internal("interval low cache is absent".into()))?
        .insert(shift, value);
    Ok(value)
}

pub(super) fn eval<F: Field, const LEFT_MONOMIAL: bool, const RIGHT_MONOMIAL: bool>(
    left: &[F],
    right: &[F],
    len: usize,
    seeds: &[EqPairSeed<F>],
    work: &mut usize,
) -> Result<F, AkitaError> {
    if len == 0 {
        return Ok(F::zero());
    }
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
    let mut left_cache = (0..left.len()).map(|_| Memo::new()).collect::<Vec<_>>();
    let mut right_cache = (0..right.len()).map(|_| Memo::new()).collect::<Vec<_>>();
    let mut out = F::zero();
    for seed in seeds {
        checked_axis_offset(seed.left_offset, 1, len - 1, "left")?;
        checked_axis_offset(seed.right_offset, 1, len - 1, "right")?;
        if seed.left_offset >= left_domain || seed.right_offset >= right_domain {
            continue;
        }
        let live = len
            .min(left_domain - seed.left_offset)
            .min(right_domain - seed.right_offset);
        let mut cursor = 0usize;
        let mut value = F::zero();
        while cursor < live {
            charge_work(work, 1)?;
            // Previous endpoint checks cover these sums and all later carries.
            let l = seed.left_offset + cursor;
            let r = seed.right_offset + cursor;
            let remaining_bits = (usize::BITS - 1 - (live - cursor).leading_zeros()) as usize;
            let block_bits = remaining_bits.min(l.trailing_zeros() as usize);
            let size = 1usize << block_bits;
            let shift = r & (size - 1);
            let [a, b] = low_eval(block_bits, shift, &factors, &mut low_cache, work)?;
            let lh = high_eval::<F, LEFT_MONOMIAL>(
                left,
                block_bits,
                l >> block_bits,
                &mut left_cache,
                work,
            )?;
            let rh = r >> block_bits;
            let r0 = high_eval::<F, RIGHT_MONOMIAL>(right, block_bits, rh, &mut right_cache, work)?;
            let r1 = if b.is_zero() {
                F::zero()
            } else {
                high_eval::<F, RIGHT_MONOMIAL>(right, block_bits, rh + 1, &mut right_cache, work)?
            };
            value += lh * (a * r0 + b * r1);
            cursor += size;
        }
        out += seed.weight * value;
    }
    Ok(out)
}
