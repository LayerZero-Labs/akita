//! Complete, value-independent carry programs. Generated data belong to the
//! trusted verifier image, never to the proof or an unchecked input stream.

use super::evaluate::basis_pair_factors;
use super::{EqPairTensorFamily, EqPairTensorWeights, MAX_COMPACT_STRIDE_TERMS};
use crate::Field;

#[cfg(not(target_arch = "riscv64"))]
mod compile;
#[cfg(not(target_arch = "riscv64"))]
pub use compile::emit_fixed_source;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightKind {
    Unit,
    BitProduct,
    Dense,
}

#[derive(Clone, Copy, Debug)]
struct Axis {
    len: u32,
    strides: [u32; 2],
    kind: WeightKind,
}

/// The normalized operands of the Lagrange/Lagrange recurrence. Live weights
/// are supplied by the family in the order and kinds recorded by these axes.
#[derive(Clone, Copy, Debug)]
struct Shape<'a> {
    /// Left and right point arities.
    bits: [u32; 2],
    /// Left and right base addresses.
    offsets: [u32; 2],
    /// Ordered, normalized axis geometry and weight-slot kinds.
    axes: &'a [Axis],
}

impl Shape<'_> {
    fn matches<F: Field>(&self, left: &[F], right: &[F], family: &EqPairTensorFamily<F>) -> bool {
        if left.len() != self.bits[0] as usize
            || right.len() != self.bits[1] as usize
            || family.left_offset != self.offsets[0] as usize
            || family.right_offset != self.offsets[1] as usize
            || family.axes.len() != self.axes.len()
        {
            return false;
        }
        self.axes.iter().zip(&family.axes).all(|(shape, axis)| {
            shape.len as usize == axis.len
                && shape.strides[0] as usize == axis.left_stride
                && shape.strides[1] as usize == axis.right_stride
                && match &axis.weights {
                    EqPairTensorWeights::Unit => shape.kind == WeightKind::Unit,
                    EqPairTensorWeights::BitProduct(factors) => {
                        shape.kind == WeightKind::BitProduct
                            && axis.len.is_power_of_two()
                            && factors.len() == axis.len.trailing_zeros() as usize
                    }
                    EqPairTensorWeights::Dense(weights) => {
                        shape.kind == WeightKind::Dense && weights.len() == axis.len
                    }
                }
        })
    }
}

// The emitter is the only constructor. Every destination has a first-write
// edge before its additive edges, so reused scratch never contributes a value.
#[derive(Debug)]
struct Graph<'a> {
    shape: Shape<'a>,
    layers: &'a [[u32; 3]],
    introductions: &'a [u16],
    edges: &'a [u32],
    peak: u16,
    residual: Option<u16>,
    seeds: u16,
    terminal: Option<u16>,
    work: u32,
}

mod compiled {
    #[allow(unused_imports)]
    use super::{Axis, Graph, Shape, WeightKind};
    include!(concat!(env!("OUT_DIR"), "/carry_graph.rs"));
}

#[cfg(test)]
thread_local! {
    static BYPASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn try_evaluate<F: Field, const LEFT_MONOMIAL: bool, const RIGHT_MONOMIAL: bool>(
    left: &[F],
    right: &[F],
    families: &[&EqPairTensorFamily<F>],
    work: &mut usize,
) -> Option<F> {
    #[cfg(test)]
    if BYPASS.with(std::cell::Cell::get) {
        return None;
    }
    if LEFT_MONOMIAL || RIGHT_MONOMIAL {
        return None;
    }
    let [family] = families else {
        return None;
    };
    let result = compiled::GRAPHS
        .iter()
        .find_map(|graph| graph.try_evaluate(left, right, family, work));
    #[cfg(test)]
    if result.is_some() {
        HITS.with(|hits| hits.set(hits.get() + 1));
    }
    result
}

impl Graph<'_> {
    fn try_evaluate<F: Field>(
        &self,
        left: &[F],
        right: &[F],
        family: &EqPairTensorFamily<F>,
        work: &mut usize,
    ) -> Option<F> {
        if !self.shape.matches(left, right, family)
            || work.checked_add(self.work as usize)? > MAX_COMPACT_STRIDE_TERMS
        {
            return None;
        }
        let (value, charged) = self.evaluate(left, right, family)?;
        *work += charged;
        Some(value)
    }

    #[inline(never)]
    fn evaluate<F: Field>(
        &self,
        left: &[F],
        right: &[F],
        family: &EqPairTensorFamily<F>,
    ) -> Option<(F, usize)> {
        let mut states = Vec::new();
        let mut next = Vec::new();
        states.try_reserve_exact(usize::from(self.peak)).ok()?;
        next.try_reserve_exact(usize::from(self.peak)).ok()?;
        states.resize(usize::from(self.peak), F::zero());
        next.resize(usize::from(self.peak), F::zero());
        for (coordinate, state) in states.iter_mut().take(usize::from(self.seeds)).enumerate() {
            let weight = match self.residual {
                Some(axis) => family.axes[usize::from(axis)].coordinate_weight(coordinate)?,
                None => F::one(),
            };
            *state = family.scalar * weight;
        }
        let mut state_width = usize::from(self.seeds);
        let mut charged = states[..state_width]
            .iter()
            .filter(|value| !value.is_zero())
            .count();
        let mut edge_start = 0;
        let mut introduction_start = 0;
        let mut choices = [F::zero(); 16];
        for (bit, &[edge_end, introduction_end, next_width]) in self.layers.iter().enumerate() {
            choices[0] = F::one();
            let mut width = 1;
            for &introduction in &self.introductions[introduction_start..introduction_end as usize]
            {
                charged += 2 * choices[..width]
                    .iter()
                    .filter(|value| !value.is_zero())
                    .count();
                let axis = usize::from(introduction >> 8);
                let coordinate_bit = usize::from(introduction & 255);
                let factors = family.axes[axis].bit_factors(coordinate_bit)?;
                for source in (0..width).rev() {
                    let weight = choices[source];
                    choices[2 * source] = weight * factors[0];
                    choices[2 * source + 1] = weight * factors[1];
                }
                width *= 2;
            }
            // Match the online zero-pruned work charge without rediscovering
            // support addresses. Charging structural zeros would change which
            // later families fit the shared work cap.
            charged += states[..state_width]
                .iter()
                .filter(|value| !value.is_zero())
                .count()
                * choices[..width]
                    .iter()
                    .filter(|value| !value.is_zero())
                    .count();
            let factors = basis_pair_factors::<F, false, false>(
                left.get(bit).copied(),
                right.get(bit).copied(),
            )
            .map(|factor| factor.unwrap_or_else(F::zero));
            for &edge in &self.edges[edge_start..edge_end as usize] {
                let source = (edge & 1023) as usize;
                let destination = ((edge >> 10) & 1023) as usize;
                let choice = ((edge >> 20) & 15) as usize;
                let parity = ((edge >> 24) & 3) as usize;
                let value = states[source] * choices[choice] * factors[parity];
                if edge & (1 << 26) != 0 {
                    next[destination] = value;
                } else {
                    next[destination] += value;
                }
            }
            std::mem::swap(&mut states, &mut next);
            state_width = next_width as usize;
            edge_start = edge_end as usize;
            introduction_start = introduction_end as usize;
        }
        Some((
            self.terminal
                .map_or_else(F::zero, |slot| states[usize::from(slot)]),
            charged,
        ))
    }
}
