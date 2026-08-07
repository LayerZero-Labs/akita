//! Monotone prefix and certified boundary search.

use crate::error::{EstimatorError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PrefixSearchResult {
    pub(super) max_value: u64,
    pub(super) next_value: Option<u64>,
    pub(super) hit_cap: bool,
}

pub(super) fn certified_boundary_from_hint<F>(
    cap: u64,
    hint: u64,
    mut predicate: F,
) -> Result<PrefixSearchResult>
where
    F: FnMut(u64) -> Result<bool>,
{
    if cap == 0 {
        return Err(EstimatorError::InvalidConfig {
            field: "search_cap",
            reason: "search cap must be positive".to_string(),
        });
    }
    let hint = hint.clamp(1, cap);
    if predicate(hint)? {
        if hint == cap {
            return Ok(PrefixSearchResult {
                max_value: cap,
                next_value: None,
                hit_cap: true,
            });
        }
        let next = hint + 1;
        if !predicate(next)? {
            return Ok(PrefixSearchResult {
                max_value: hint,
                next_value: Some(next),
                hit_cap: false,
            });
        }
        let mut low = next;
        let mut step = 2u64;
        loop {
            let high = low.saturating_add(step).min(cap);
            if predicate(high)? {
                low = high;
                if low == cap {
                    return Ok(PrefixSearchResult {
                        max_value: cap,
                        next_value: None,
                        hit_cap: true,
                    });
                }
                step = step.saturating_mul(2);
            } else {
                return binary_boundary(low, high, predicate);
            }
        }
    }

    if hint == 1 {
        return Ok(PrefixSearchResult {
            max_value: 0,
            next_value: Some(1),
            hit_cap: false,
        });
    }
    let previous = hint - 1;
    if predicate(previous)? {
        return Ok(PrefixSearchResult {
            max_value: previous,
            next_value: Some(hint),
            hit_cap: false,
        });
    }
    let mut high = previous;
    let mut step = 1u64;
    loop {
        let low = high.saturating_sub(step).max(1);
        if predicate(low)? {
            return binary_boundary(low, high, predicate);
        }
        if low == 1 {
            return Ok(PrefixSearchResult {
                max_value: 0,
                next_value: Some(1),
                hit_cap: false,
            });
        }
        high = low;
        step = step.saturating_mul(2);
    }
}

fn binary_boundary<F>(mut low: u64, mut high: u64, mut predicate: F) -> Result<PrefixSearchResult>
where
    F: FnMut(u64) -> Result<bool>,
{
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if predicate(mid)? {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(PrefixSearchResult {
        max_value: low,
        next_value: Some(high),
        hit_cap: false,
    })
}

pub(super) fn max_true_in_prefix<F>(
    start: u64,
    cap: u64,
    mut predicate: F,
) -> Result<PrefixSearchResult>
where
    F: FnMut(u64) -> Result<bool>,
{
    if start == 0 || cap < start {
        return Err(EstimatorError::InvalidConfig {
            field: "search range",
            reason: "must satisfy 0 < start <= cap".to_string(),
        });
    }
    if !predicate(start)? {
        return Ok(PrefixSearchResult {
            max_value: 0,
            next_value: Some(start),
            hit_cap: false,
        });
    }
    let mut low = start;
    let mut high = start.checked_mul(2).unwrap_or(cap).min(cap);
    if high == low && high < cap {
        high = high
            .checked_add(1)
            .ok_or_else(|| EstimatorError::InvalidConfig {
                field: "search_cap",
                reason: "search probe overflow".to_string(),
            })?;
    }
    while high < cap && predicate(high)? {
        low = high;
        high = high.checked_mul(2).unwrap_or(cap).min(cap);
    }
    if high == cap && predicate(cap)? {
        return Ok(PrefixSearchResult {
            max_value: cap,
            next_value: None,
            hit_cap: true,
        });
    }
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if predicate(mid)? {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(PrefixSearchResult {
        max_value: low,
        next_value: Some(high),
        hit_cap: false,
    })
}
