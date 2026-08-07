//! Parallel Stage 2 coefficient folds over canonical coefficient-first tables.

use super::{add_stage2_round_terms, reverse_power_of_two_index};
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{ExtField, FieldCore};
use akita_sumcheck::{DelayedProductSum, DirectProductSum, EvaluationTable, ProductSumAccumulator};

struct Stage2Rows<'a, F> {
    folded_0: Vec<&'a mut [F]>,
    folded_1: Vec<&'a mut [F]>,
    source_0: Vec<&'a mut [F]>,
    source_1: Vec<&'a mut [F]>,
    start: usize,
}

impl<'a, F> Stage2Rows<'a, F> {
    fn from_table<E: ExtField<F>>(table: &'a mut EvaluationTable<F, E>, block_len: usize) -> Self
    where
        F: FieldCore,
    {
        let coefficient_count = E::EXT_DEGREE;
        let mut folded_0 = Vec::with_capacity(coefficient_count);
        let mut folded_1 = Vec::with_capacity(coefficient_count);
        let mut source_0 = Vec::with_capacity(coefficient_count);
        let mut source_1 = Vec::with_capacity(coefficient_count);
        for coefficient in table.all_coefficient_slices_mut() {
            let (coefficient_folded_0, remaining) = coefficient.split_at_mut(block_len);
            let (coefficient_folded_1, remaining) = remaining.split_at_mut(block_len);
            let (coefficient_source_0, coefficient_source_1) = remaining.split_at_mut(block_len);
            folded_0.push(coefficient_folded_0);
            folded_1.push(coefficient_folded_1);
            source_0.push(coefficient_source_0);
            source_1.push(coefficient_source_1);
        }
        Self {
            folded_0,
            folded_1,
            source_0,
            source_1,
            start: 0,
        }
    }

    fn len(&self) -> usize {
        self.folded_0[0].len()
    }

    fn split(self, mid: usize) -> (Self, Self) {
        let (folded_0_left, folded_0_right) = split_slices(self.folded_0, mid);
        let (folded_1_left, folded_1_right) = split_slices(self.folded_1, mid);
        let (source_0_left, source_0_right) = split_slices(self.source_0, mid);
        let (source_1_left, source_1_right) = split_slices(self.source_1, mid);
        (
            Self {
                folded_0: folded_0_left,
                folded_1: folded_1_left,
                source_0: source_0_left,
                source_1: source_1_left,
                start: self.start,
            },
            Self {
                folded_0: folded_0_right,
                folded_1: folded_1_right,
                source_0: source_0_right,
                source_1: source_1_right,
                start: self.start + mid,
            },
        )
    }
}

fn split_slices<T>(slices: Vec<&mut [T]>, mid: usize) -> (Vec<&mut [T]>, Vec<&mut [T]>) {
    slices
        .into_iter()
        .map(|slice| slice.split_at_mut(mid))
        .unzip()
}

struct Stage2RoundFactors<'a, E> {
    live_lane_count: usize,
    next_pair_count: usize,
    next_alpha_factor: &'a [E],
    relation_lane_weights: &'a [E],
    first_equality: &'a [E],
    second_equality: &'a [E],
    lanes_use_binding_order: bool,
    include_norm_linear: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fold_and_compute_stage2_coefficient_round_parallel<F, E>(
    witness: &mut EvaluationTable<F, E>,
    live_lane_count: usize,
    old_coefficient_count: usize,
    next_alpha_factor: &[E],
    relation_lane_weights: &[E],
    first_equality: &[E],
    second_equality: &[E],
    challenge: E,
    include_norm_linear: bool,
) -> ([E; 3], [E; 3])
where
    F: FieldCore,
    E: ExtField<F> + HasOptimizedFold + HasUnreducedOps,
{
    assert!(old_coefficient_count.is_power_of_two() && old_coefficient_count >= 4);
    assert_eq!(witness.len(), live_lane_count * old_coefficient_count);
    assert!(relation_lane_weights.len() >= live_lane_count);
    assert!(first_equality.len().is_power_of_two());
    let next_coefficient_count = old_coefficient_count / 2;
    let next_pair_count = next_coefficient_count / 2;
    assert_eq!(next_alpha_factor.len(), next_coefficient_count);
    assert!(
        first_equality.len() * second_equality.len() >= live_lane_count * next_pair_count,
        "split equality table does not cover the live Stage 2 rows"
    );

    let rows = Stage2Rows::from_table(witness, live_lane_count * next_pair_count);
    let factors = Stage2RoundFactors {
        live_lane_count,
        next_pair_count,
        next_alpha_factor,
        relation_lane_weights,
        first_equality,
        second_equality,
        lanes_use_binding_order: live_lane_count == relation_lane_weights.len(),
        include_norm_linear,
    };
    let fold = E::precompute_fold(challenge);
    let target_tasks = rayon::current_num_threads().saturating_mul(4).max(1);
    let minimum_rows_per_task = rows.len().div_ceil(target_tasks).max(1_024);
    let result = if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        let (norm, relation) = fold_stage2_rows::<F, E, DelayedProductSum<E>>(
            rows,
            &factors,
            &fold,
            minimum_rows_per_task,
        );
        (
            norm.map(ProductSumAccumulator::finish),
            relation.map(ProductSumAccumulator::finish),
        )
    } else {
        let (norm, relation) = fold_stage2_rows::<F, E, DirectProductSum<E>>(
            rows,
            &factors,
            &fold,
            minimum_rows_per_task,
        );
        (
            norm.map(ProductSumAccumulator::finish),
            relation.map(ProductSumAccumulator::finish),
        )
    };
    witness.truncate(live_lane_count * next_coefficient_count);
    result
}

fn fold_stage2_rows<F, E, A>(
    rows: Stage2Rows<'_, F>,
    factors: &Stage2RoundFactors<'_, E>,
    fold: &E::FoldCtx,
    minimum_rows_per_task: usize,
) -> ([A; 3], [A; 3])
where
    F: FieldCore,
    E: ExtField<F> + HasOptimizedFold + HasUnreducedOps,
    A: ProductSumAccumulator<E>,
{
    if rows.len() <= minimum_rows_per_task {
        return fold_stage2_rows_sequential(rows, factors, fold);
    }
    let mid = rows.len() / 2;
    let (left_rows, right_rows) = rows.split(mid);
    let (left, right) = rayon::join(
        || fold_stage2_rows(left_rows, factors, fold, minimum_rows_per_task),
        || fold_stage2_rows(right_rows, factors, fold, minimum_rows_per_task),
    );
    (
        merge_product_sums(left.0, right.0),
        merge_product_sums(left.1, right.1),
    )
}

fn fold_stage2_rows_sequential<F, E, A>(
    mut rows: Stage2Rows<'_, F>,
    factors: &Stage2RoundFactors<'_, E>,
    fold: &E::FoldCtx,
) -> ([A; 3], [A; 3])
where
    F: FieldCore,
    E: ExtField<F> + HasOptimizedFold + HasUnreducedOps,
    A: ProductSumAccumulator<E>,
{
    let mut norm = std::array::from_fn(|_| A::zero());
    let mut relation = std::array::from_fn(|_| A::zero());
    for row in 0..rows.len() {
        let stored_index = rows.start + row;
        let stored_pair = stored_index / factors.live_lane_count;
        let stored_lane = stored_index % factors.live_lane_count;
        let witness_0 = read_extension::<F, E>(&rows.folded_0, row);
        let witness_1 = read_extension::<F, E>(&rows.folded_1, row);
        let folded_0 = E::fold_one(fold, witness_0, read_extension::<F, E>(&rows.source_0, row));
        let folded_1 = E::fold_one(fold, witness_1, read_extension::<F, E>(&rows.source_1, row));
        write_extension(&mut rows.folded_0, row, folded_0);
        write_extension(&mut rows.folded_1, row, folded_1);

        let logical_pair = reverse_power_of_two_index(stored_pair, factors.next_pair_count);
        let logical_lane = if factors.lanes_use_binding_order {
            reverse_power_of_two_index(stored_lane, factors.live_lane_count)
        } else {
            stored_lane
        };
        let equality_address = logical_lane * factors.next_pair_count + logical_pair;
        let equality = factors.first_equality
            [equality_address & (factors.first_equality.len() - 1)]
            * factors.second_equality[equality_address / factors.first_equality.len()];
        let alpha_0 = factors.next_alpha_factor[stored_pair];
        add_stage2_round_terms(
            &mut norm,
            &mut relation,
            folded_0,
            folded_1,
            equality,
            factors.relation_lane_weights[stored_lane],
            alpha_0,
            factors.next_alpha_factor[stored_pair + factors.next_pair_count] - alpha_0,
            factors.include_norm_linear,
        );
    }
    (norm, relation)
}

fn read_extension<F, E>(coefficients: &[&mut [F]], row: usize) -> E
where
    F: FieldCore,
    E: ExtField<F>,
{
    E::from_base_fn(|coefficient| coefficients[coefficient][row])
}

fn write_extension<F, E>(coefficients: &mut [&mut [F]], row: usize, value: E)
where
    F: FieldCore,
    E: ExtField<F>,
{
    for (coefficient, destination) in coefficients.iter_mut().enumerate() {
        destination[row] = value.base_coefficient(coefficient);
    }
}

fn merge_product_sums<E, A>(left: [A; 3], right: [A; 3]) -> [A; 3]
where
    E: FieldCore + HasUnreducedOps,
    A: ProductSumAccumulator<E>,
{
    let [left_0, left_1, left_2] = left;
    let [right_0, right_1, right_2] = right;
    [
        left_0.merge(right_0),
        left_1.merge(right_1),
        left_2.merge(right_2),
    ]
}
