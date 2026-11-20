//! Bounded weighted recurrence grouping and singleton fallback.

use super::{charge_work, eval_multi_axis_families, EqPairTensorFamily, EqPairTensorWeights};
use crate::Field;
use akita_error::AkitaError;
use std::collections::{btree_map::Entry, BTreeMap};

pub(super) type RecurrenceBatches<'a, F> =
    BTreeMap<Vec<(usize, usize, usize, usize)>, Vec<Vec<&'a EqPairTensorFamily<F>>>>;

const MAX_WEIGHTED_BATCH_GROUPS: usize = 16;

/// Queue a matching recurrence, or evaluate an unmatched family once its
/// geometry has reached the group limit. Charge classification before work.
pub(super) fn queue_or_singleton<
    'a,
    F: Field,
    const LEFT_MONOMIAL: bool,
    const RIGHT_MONOMIAL: bool,
>(
    left_challenges: &[F],
    right_challenges: &[F],
    family: &'a EqPairTensorFamily<F>,
    recurrence_axes: &[usize],
    multi_axis_batches: &mut RecurrenceBatches<'a, F>,
    work: &mut usize,
) -> Result<Option<F>, AkitaError> {
    let recurrence_geometry = recurrence_axes
        .iter()
        .map(|&axis_index| {
            let axis = family.axes.get(axis_index).ok_or_else(|| {
                AkitaError::Internal("selected tensor recurrence axis is absent".into())
            })?;
            Ok((axis_index, axis.len, axis.left_stride, axis.right_stride))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    charge_work(work, recurrence_geometry.len())?;
    charge_work(
        work,
        1 + (usize::BITS - multi_axis_batches.len().leading_zeros()) as usize,
    )?;
    match multi_axis_batches.entry(recurrence_geometry) {
        Entry::Vacant(entry) => {
            charge_work(work, 1)?;
            entry.insert(vec![vec![family]]);
        }
        Entry::Occupied(mut entry) => {
            let mut matched = false;
            // Keep at most this many weight groups per geometry. Once
            // full, unmatched families use the singleton recurrence.
            for batch in entry.get_mut().iter_mut() {
                let first = batch.first().ok_or_else(|| {
                    AkitaError::Internal("tensor recurrence batch is empty".into())
                })?;
                let mut equal = true;
                for &index in recurrence_axes {
                    let axis = family.axes.get(index).ok_or_else(|| {
                        AkitaError::Internal("selected tensor recurrence axis is absent".into())
                    })?;
                    let terms = match &axis.weights {
                        EqPairTensorWeights::Unit => 0,
                        EqPairTensorWeights::Dense(weights) => weights.len(),
                        EqPairTensorWeights::BitProduct(factors) => factors.len(),
                    };
                    charge_work(work, 1)?;
                    charge_work(work, terms)?;
                    if first.axes.get(index) != Some(axis) {
                        equal = false;
                        break;
                    }
                }
                if equal {
                    charge_work(work, 1)?;
                    batch.push(family);
                    matched = true;
                    break;
                }
            }
            if !matched {
                if entry.get().len() < MAX_WEIGHTED_BATCH_GROUPS {
                    charge_work(work, 1)?;
                    entry.get_mut().push(vec![family]);
                } else {
                    return eval_multi_axis_families::<F, LEFT_MONOMIAL, RIGHT_MONOMIAL>(
                        left_challenges,
                        right_challenges,
                        &[family],
                        entry.key(),
                        work,
                    )
                    .map(Some);
                }
            }
        }
    }
    Ok(None)
}
