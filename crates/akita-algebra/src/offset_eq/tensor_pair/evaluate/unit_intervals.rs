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
    if let Some(value) = cache.get(bits).and_then(|m| m.get(&shift)) {
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
    cache
        .get_mut(bits)
        .ok_or_else(|| AkitaError::Internal("interval low cache is absent".into()))?
        .insert(shift, value);
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
    let mut left_cache = (0..left.len()).map(|_| Memo::new()).collect::<Vec<_>>();
    let mut right_cache = (0..right.len()).map(|_| Memo::new()).collect::<Vec<_>>();
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
