//! Stable byte-radix ordering of larger paired-carry batches.
use super::PairState;
use crate::Field;
use akita_error::AkitaError;

pub(super) fn sort<F: Field>(states: &mut Vec<PairState<F>>) -> Result<(), AkitaError> {
    if states.len() < 512 {
        states.sort_unstable_by_key(|(key, _)| *key);
        return Ok(());
    }
    let first = states.first().ok_or(AkitaError::InvalidProof)?.0;
    let mut varying = [0usize; 2];
    for &((l, r), _) in states.iter() {
        varying[0] |= l ^ first.0;
        varying[1] |= r ^ first.1;
    }
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(states.len())
        .map_err(|_| AkitaError::InvalidInput("tensor radix scratch allocation failed".into()))?;
    scratch.resize(states.len(), ((0, 0), F::zero()));
    for coordinate in [1, 0] {
        for shift in (0..usize::BITS).step_by(8) {
            if (varying[coordinate] >> shift) & 255 == 0 {
                continue;
            }
            let digit = |key: (usize, usize)| {
                let value = if coordinate == 0 { key.0 } else { key.1 };
                (value >> shift) & 255
            };
            let mut counts = [0usize; 256];
            for &(key, _) in states.iter() {
                counts[digit(key)] += 1;
            }
            let mut start = 0;
            for count in &mut counts {
                let len = *count;
                *count = start;
                start += len;
            }
            for &state in states.iter() {
                let cursor = &mut counts[digit(state.0)];
                *scratch.get_mut(*cursor).ok_or_else(|| {
                    AkitaError::Internal("tensor radix offset is out of range".into())
                })? = state;
                *cursor += 1;
            }
            std::mem::swap(states, &mut scratch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Prime128OffsetA7F7 as F, Ring, Zero};
    use std::collections::BTreeMap;
    #[test]
    fn radix_order_preserves_all_key_weights() {
        for count in [0, 1, 63, 64, 65, 257, 511, 512, 513, 4096] {
            let mut states = (0..count)
                .map(|i| {
                    let left = (i * 1777) % 317;
                    let right = (i * 7351) % 577;
                    let high = if i % 3 == 0 {
                        1usize << (usize::BITS - 1)
                    } else {
                        0
                    };
                    ((left | high, right | high), F::from_u64(i as u64 + 7))
                })
                .collect::<Vec<_>>();
            let aggregate = |states: &[PairState<F>]| {
                let mut map = BTreeMap::new();
                for &(key, weight) in states {
                    *map.entry(key).or_insert(F::zero()) += weight;
                }
                map
            };
            let expected = aggregate(&states);
            sort(&mut states).unwrap();
            assert!(states.windows(2).all(|w| w[0].0 <= w[1].0));
            assert_eq!(aggregate(&states), expected);
        }
    }
}
