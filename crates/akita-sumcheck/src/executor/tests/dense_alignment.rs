use super::*;

pub(super) fn round_sum(
    master_equality_point: &[F],
    q_table: &[F],
    suffix_offset: usize,
    coefficient: F,
    bound_prefix: &[F],
    current: F,
) -> F {
    let round = bound_prefix.len();
    let remaining = master_equality_point.len() - round - 1;
    (0..(1usize << remaining)).fold(F::zero(), |sum, assignment| {
        let mut point = bound_prefix.to_vec();
        point.push(current);
        point.extend((0..remaining).map(|bit| f(((assignment >> bit) & 1) as u64)));
        sum + coefficient
            * eq_eval(master_equality_point, &point)
            * dense_q_at(q_table, &point[suffix_offset..])
    })
}
