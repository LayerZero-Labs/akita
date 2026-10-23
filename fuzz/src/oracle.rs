//! Prover-independent opening oracles.
//!
//! These evaluate raw tables directly in little-endian variable order
//! (`point[0]` binds index bit 0) and share no code with the prover's
//! opening kernels, so a layout or fold-order bug cannot move both sides.

use jolt_field::{ExtField, Field};

/// `Σ_x eq(point, x) · table[x]` for a base-field table.
pub fn dense_lagrange<F: Field, E: ExtField<F>>(table: &[F], point: &[E]) -> E {
    fold(table, point, |low, high, x| low + (high - low) * x)
}

/// `Σ_x (∏_{i ∈ bits(x)} point[i]) · table[x]`.
pub fn dense_monomial<F: Field, E: ExtField<F>>(table: &[F], point: &[E]) -> E {
    fold(table, point, |low, high, x| low + high * x)
}

fn fold<F: Field, E: ExtField<F>>(table: &[F], point: &[E], combine: impl Fn(E, E, E) -> E) -> E {
    assert_eq!(
        table.len(),
        1usize << point.len(),
        "table length must be 2^|point|"
    );
    let mut layer: Vec<E> = table.iter().map(|&value| E::lift_base(value)).collect();
    for &x in point {
        let half = layer.len() / 2;
        for index in 0..half {
            layer[index] = combine(layer[2 * index], layer[2 * index + 1], x);
        }
        layer.truncate(half);
    }
    layer[0]
}

/// Weight of one Boolean index in the requested basis.
pub fn index_weight<E: Field>(point: &[E], index: usize, lagrange: bool) -> E {
    point.iter().enumerate().fold(E::one(), |acc, (bit, &x)| {
        let set = (index >> bit) & 1 == 1;
        match (lagrange, set) {
            (_, true) => acc * x,
            (true, false) => acc * (E::one() - x),
            (false, false) => acc,
        }
    })
}

/// One-hot opening: sum of the weights at every committed hot position.
pub fn onehot<E: Field>(
    indices: &[Option<u8>],
    chunk_size: usize,
    point: &[E],
    lagrange: bool,
) -> E {
    indices
        .iter()
        .enumerate()
        .filter_map(|(chunk, hot)| hot.map(|position| chunk * chunk_size + usize::from(position)))
        .fold(E::zero(), |acc, index| {
            acc + index_weight(point, index, lagrange)
        })
}
