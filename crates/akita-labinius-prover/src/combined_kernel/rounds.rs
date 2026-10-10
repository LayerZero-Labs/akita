//! Round sums of the combined kernel: class histograms on one or two packed
//! bytes, class moments on four packed bytes, and pairs evaluated directly.

use akita_error::{checked, AkitaError};
use jolt_field::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::alphabet::{add_moments, fill_powers, node_values, Alphabet, POWERS};
use super::{buffer, class_index, invalid, MomentRound, INNER_NODES};

/// Sums of one round: the alphabet term at the nodes `0..=16`, with one spare
/// slot for the extrapolated node, and the linear term.
#[derive(Clone, Copy, Debug)]
pub(super) struct Totals<F> {
    pub(super) alphabet: [F; INNER_NODES + 1],
    // L(0), L(1), and the coefficient of t^2.
    pub(super) linear: [F; 3],
}

impl<F: Field> Totals<F> {
    pub(super) fn zero() -> Self {
        Self {
            alphabet: [F::zero(); INNER_NODES + 1],
            linear: [F::zero(); 3],
        }
    }

    fn add(&mut self, other: &Self) {
        for (a, &b) in self.alphabet.iter_mut().zip(&other.alphabet) {
            *a += b;
        }
        self.add_linear(&other.linear);
    }

    fn add_linear(&mut self, linear: &[F; 3]) {
        for (a, &b) in self.linear.iter_mut().zip(linear) {
            *a += b;
        }
    }

    fn product(&mut self, w0: F, w1: F, k0: F, k1: F) {
        let products = [w0 * k0, w1 * k1, (w1 - w0) * (k1 - k0)];
        self.add_linear(&products);
    }

    /// Replace coefficients of the alphabet sum by its values at the nodes.
    fn coefficients_to_nodes(&mut self, reversed: bool) {
        let mut coefficients = [F::zero(); INNER_NODES];
        for (coefficient, &value) in coefficients.iter_mut().zip(&self.alphabet) {
            *coefficient = value;
        }
        let values = node_values(&coefficients, reversed);
        for (slot, value) in self.alphabet.iter_mut().zip(values) {
            *slot = value;
        }
    }
}

/// Tables read by every worker of one round.
pub(super) struct Tables<'a, F> {
    pub(super) packed: &'a [u8],
    pub(super) w: &'a [F],
    pub(super) kw: &'a [F],
    pub(super) digit_factor: &'a [F],
    pub(super) low: &'a [F],
    pub(super) high: &'a [F],
    pub(super) lut: &'a [F],
    /// Bits of one packed pair, or zero once the digits are lifted.
    pub(super) class_bits: usize,
}

impl<F: Field> Tables<'_, F> {
    fn weight_pair(&self, pair: usize) -> Option<(F, F)> {
        let start = checked::product([pair, 2])?;
        if self.digit_factor.is_empty() {
            let &[left, right] = self.kw.get(checked::range(start, 2)?)? else {
                return None;
            };
            Some((left, right))
        } else {
            let depth = self.digit_factor.len();
            let coefficient = *self.kw.get(start >> depth.trailing_zeros())?;
            let &[left, right] = self
                .digit_factor
                .get(checked::range(start & (depth - 1), 2)?)?
            else {
                return None;
            };
            Some((left * coefficient, right * coefficient))
        }
    }

    /// Values of the two halves of a packed class.
    fn class_values(&self, class: usize) -> Option<(F, F)> {
        let side = self.lut.len();
        let left = self.lut.get(class.checked_rem(side)?)?;
        Some((*left, *self.lut.get(class.checked_div(side)?)?))
    }

    /// Packed bytes of one equality block, and its first pair.
    fn block(&self, block: usize) -> Option<(&[u8], usize)> {
        let first = checked::product([block, self.low.len()])?;
        let bytes = checked::product([self.low.len(), self.class_bits / 8])?;
        let start = checked::product([block, bytes])?;
        Some((self.packed.get(checked::range(start, bytes)?)?, first))
    }
}

#[derive(Debug)]
pub(super) struct Workspace<F> {
    // Suffix equality weight and the two unweighted K endpoints per class.
    buckets: Vec<[F; 3]>,
    // Compact coefficients per (packed class, remaining digit-pair position).
    compact: Vec<F>,
    inner: Vec<F>,
    marked: Vec<u8>,
    touched: Vec<usize>,
    // Seventeen weighted moments per bucket class of the four-byte round.
    moments: Vec<F>,
    total: Totals<F>,
}

impl<F: Field> Workspace<F> {
    pub(super) fn new(classes: usize, compact: usize, moments: usize) -> Result<Self, AkitaError> {
        let mut touched = Vec::new();
        touched.try_reserve_exact(classes).map_err(|_| invalid())?;
        Ok(Self {
            buckets: buffer(classes, [F::zero(); 3])?,
            compact: buffer(compact, F::zero())?,
            inner: buffer(classes, F::zero())?,
            marked: buffer(classes, 0)?,
            touched,
            moments: buffer(moments, F::zero())?,
            total: Totals::zero(),
        })
    }

    /// Sum equality weights and weights per class of this worker's blocks,
    /// with additions only inside a block. `positions` is the number of
    /// digit-factor pairs summed in compact form, or zero for weights added
    /// per pair. The all-zero class contributes nothing and is skipped.
    fn histogram(
        &mut self,
        tables: &Tables<'_, F>,
        worker: usize,
        block_count: usize,
        classes: usize,
        positions: usize,
    ) -> Option<()> {
        self.buckets.get_mut(..classes)?.fill([F::zero(); 3]);
        self.compact
            .get_mut(..checked::product([classes, positions])?)?
            .fill(F::zero());
        let start = checked::product([worker, block_count])?;
        let bytes = tables.class_bits / 8;
        for (block, &outer) in tables.high.iter().enumerate().skip(start).take(block_count) {
            let (packed, first) = tables.block(block)?;
            for (inner, (code, &eq)) in packed.chunks_exact(bytes).zip(tables.low).enumerate() {
                let class = class_index(code, 0, tables.class_bits)?;
                if class == 0 {
                    continue;
                }
                let pair = checked::sum([first, inner])?;
                if positions > 0 {
                    let index = checked::mul_add(class, positions, pair & (positions - 1))?;
                    *self.compact.get_mut(index)? +=
                        *tables.kw.get(pair >> positions.trailing_zeros())?;
                } else {
                    let (k0, k1) = tables.weight_pair(pair)?;
                    let [_, sum0, sum1] = self.buckets.get_mut(class)?;
                    *sum0 += k0;
                    *sum1 += k1;
                }
                *self.inner.get_mut(class)? += eq;
                let mark = self.marked.get_mut(class)?;
                if *mark == 0 {
                    *mark = 1;
                    self.touched.push(class);
                }
            }
            // Outer equality is applied only to occupied class buckets,
            // once per inner block.
            for class in self.touched.drain(..) {
                let eq_sum = self.inner.get_mut(class)?;
                let [eq, _, _] = self.buckets.get_mut(class)?;
                *eq = eq_sum.mul_add(outer, *eq);
                *eq_sum = F::zero();
                *self.marked.get_mut(class)? = 0;
            }
        }
        Some(())
    }

    /// Sum the linear term of this worker's blocks and, per bucket class, the
    /// weighted powers of the other endpoint's value: a row of `powers`, or
    /// computed for the pair when the round has no table.
    fn moments(
        &mut self,
        tables: &Tables<'_, F>,
        worker: usize,
        block_count: usize,
        round: MomentRound,
        powers: &[F],
    ) -> Option<()> {
        self.total = Totals::zero();
        self.moments.fill(F::zero());
        let mut computed = [F::zero(); POWERS];
        let start = checked::product([worker, block_count])?;
        for (block, &outer) in tables.high.iter().enumerate().skip(start).take(block_count) {
            let (packed, first) = tables.block(block)?;
            for (inner, (code, &eq)) in packed.chunks_exact(4).zip(tables.low).enumerate() {
                let &[a, b, c, d] = code else {
                    return None;
                };
                let left = usize::from(u16::from_le_bytes([a, b]));
                let right = usize::from(u16::from_le_bytes([c, d]));
                if left == 0 && right == 0 {
                    continue;
                }
                let (k0, k1) = tables.weight_pair(checked::sum([first, inner])?)?;
                let (w0, w1) = (*tables.lut.get(left)?, *tables.lut.get(right)?);
                self.total.product(w0, w1, k0, k1);
                let (bucket, other, value) = if round.right {
                    (right, left, w0)
                } else {
                    (left, right, w1)
                };
                let row = if round.powers == 0 {
                    fill_powers(value, &mut computed);
                    computed.as_slice()
                } else {
                    powers.get(checked::range(checked::product([other, POWERS])?, POWERS)?)?
                };
                let sums = checked::range(checked::product([bucket, INNER_NODES])?, INNER_NODES)?;
                add_moments(self.moments.get_mut(sums)?, eq * outer, row);
            }
        }
        Some(())
    }

    /// Evaluate every pair of this worker's blocks on its line. A pair of
    /// zeros stays zero on the line, so both of its terms vanish.
    fn direct(
        &mut self,
        tables: &Tables<'_, F>,
        worker: usize,
        block_count: usize,
        alphabet: &Alphabet<F>,
    ) -> Option<()> {
        self.total = Totals::zero();
        let start = checked::product([worker, block_count])?;
        let block_width = checked::product([tables.low.len(), 2])?;
        for (block, &outer) in tables.high.iter().enumerate().skip(start).take(block_count) {
            let first = checked::product([block, tables.low.len()])?;
            let digits = if tables.class_bits == 0 {
                let offset = checked::product([block, block_width])?;
                tables.w.get(checked::range(offset, block_width)?)?
            } else {
                &[]
            };
            let mut sums = [F::zero(); INNER_NODES];
            for (inner, &eq) in tables.low.iter().enumerate() {
                let pair = checked::sum([first, inner])?;
                let (w0, w1) = if tables.class_bits == 0 {
                    let start = checked::product([inner, 2])?;
                    let &[left, right] = digits.get(checked::range(start, 2)?)? else {
                        return None;
                    };
                    (left, right)
                } else {
                    tables.class_values(class_index(tables.packed, pair, tables.class_bits)?)?
                };
                if w0 == F::zero() && w1 == F::zero() {
                    continue;
                }
                let (k0, k1) = tables.weight_pair(pair)?;
                self.total.product(w0, w1, k0, k1);
                alphabet.add_line(&mut sums, w0, w1, eq);
            }
            for (sum, inner) in self.total.alphabet.iter_mut().zip(sums) {
                *sum = outer.mul_add(inner, *sum);
            }
        }
        Some(())
    }
}

// Each worker mutates only its preallocated workspace. A pass that stops
// early reports it, and the round then has no sums.
fn for_each_worker<F: Field>(
    workers: &mut [Workspace<F>],
    pass: impl Fn(usize, &mut Workspace<F>) -> Option<()> + Sync + Send,
) -> Option<()> {
    #[cfg(feature = "parallel")]
    let complete = workers
        .par_iter_mut()
        .enumerate()
        .map(|(worker, workspace)| pass(worker, workspace).is_some())
        .reduce(|| true, |left, right| left && right);
    #[cfg(not(feature = "parallel"))]
    let complete = workers
        .iter_mut()
        .enumerate()
        .fold(true, |complete, (worker, workspace)| {
            pass(worker, workspace).is_some() && complete
        });
    complete.then_some(())
}

// Sum independent rows; the reduction only adds exact field elements.
fn sum_rows<F: Field>(
    count: usize,
    row: impl Fn(usize) -> Option<Totals<F>> + Sync + Send,
) -> Option<Totals<F>> {
    #[cfg(feature = "parallel")]
    let total = (0..count)
        .into_par_iter()
        .map(row)
        .try_reduce(Totals::zero, |mut sum, other| {
            sum.add(&other);
            Some(sum)
        });
    #[cfg(not(feature = "parallel"))]
    let total = (0..count).try_fold(Totals::zero(), |mut sum, index| {
        sum.add(&row(index)?);
        Some(sum)
    });
    total
}

// Powers `1..=16` of the first `classes` values; the capacity was reserved
// with the kernel.
fn fill_power_table<F: Field>(powers: &mut Vec<F>, values: &[F], classes: usize) -> Option<()> {
    powers.clear();
    powers.resize(checked::product([classes, POWERS])?, F::zero());
    #[cfg(feature = "parallel")]
    powers
        .par_chunks_mut(POWERS)
        .zip(values)
        .for_each(|(row, &value)| fill_powers(value, row));
    #[cfg(not(feature = "parallel"))]
    for (row, &value) in powers.chunks_mut(POWERS).zip(values) {
        fill_powers(value, row);
    }
    Some(())
}

/// A round on lifted digits, or on packed pairs when no class table fits.
pub(super) fn direct_round<F: Field>(
    tables: &Tables<'_, F>,
    workers: &mut [Workspace<F>],
    alphabet: &Alphabet<F>,
) -> Option<Totals<F>> {
    let blocks = checked::div_ceil(tables.high.len(), workers.len())?;
    for_each_worker(workers, |worker, workspace| {
        workspace.direct(tables, worker, blocks, alphabet)
    })?;
    let mut total = Totals::zero();
    for worker in workers.iter() {
        total.add(&worker.total);
    }
    Some(total)
}

/// A round whose pair is one or two packed bytes.
///
/// The workers that own a class table of this size build histograms. Each
/// class is then met once: rows follow the right half of the class, and the
/// left halves enter through their powers, so a class costs seventeen
/// multiplications and a row one pair of Pascal transforms.
pub(super) fn class_round<F: Field>(
    tables: &Tables<'_, F>,
    workers: &mut [Workspace<F>],
    powers: &mut Vec<F>,
    alphabet: &Alphabet<F>,
) -> Option<Totals<F>> {
    let classes = checked::pow2(tables.class_bits)?;
    let active = workers
        .iter()
        .take_while(|worker| worker.buckets.len() >= classes)
        .count();
    if active == 0 {
        return direct_round(tables, workers, alphabet);
    }
    let workers = workers.get_mut(..active)?;
    let pairs = tables.digit_factor.len() / 2;
    let compact = checked::product([classes, pairs])?;
    let positions = if workers.iter().all(|worker| worker.compact.len() >= compact) {
        pairs
    } else {
        0
    };
    let blocks = checked::div_ceil(tables.high.len(), active)?;
    for_each_worker(workers, |worker, workspace| {
        workspace.histogram(tables, worker, blocks, classes, positions)
    })?;
    let side = tables.lut.len();
    fill_power_table(powers, tables.lut, side)?;
    let (workers, powers) = (&*workers, &*powers);
    let mut total = sum_rows(side, |right| {
        let base = *tables.lut.get(right)?;
        let start = checked::product([right, side])?;
        let mut row = Totals::zero();
        let mut moments = [F::zero(); INNER_NODES];
        for (offset, (&left, powers)) in tables.lut.iter().zip(powers.chunks(POWERS)).enumerate() {
            let class = checked::sum([start, offset])?;
            let (mut eq, mut k0, mut k1) = (F::zero(), F::zero(), F::zero());
            for worker in workers {
                let &[e, a, b] = worker.buckets.get(class)?;
                eq += e;
                k0 += a;
                k1 += b;
            }
            if positions > 0 {
                let first = checked::product([class, positions])?;
                for (position, factor) in tables.digit_factor.chunks_exact(2).enumerate() {
                    let index = checked::sum([first, position])?;
                    let mut coefficient = F::zero();
                    for worker in workers {
                        coefficient += *worker.compact.get(index)?;
                    }
                    let &[f0, f1] = factor else {
                        return None;
                    };
                    k0 = coefficient.mul_add(f0, k0);
                    k1 = coefficient.mul_add(f1, k1);
                }
            }
            if eq == F::zero() && k0 == F::zero() && k1 == F::zero() {
                continue;
            }
            row.product(left, base, k0, k1);
            add_moments(&mut moments, eq, powers);
        }
        let coefficients = alphabet.moment_coefficients(base, &moments);
        for (slot, coefficient) in row.alphabet.iter_mut().zip(coefficients) {
            *slot = coefficient;
        }
        Some(row)
    })?;
    total.coefficients_to_nodes(true);
    Some(total)
}

/// The round whose pair is four packed bytes: two classes of two bytes.
///
/// Class pairs outnumber the pairs, so there is no histogram. One endpoint
/// indexes buckets of weighted moments, the other enters through its powers:
/// a pair costs seventeen multiplications, after fifteen more for the powers
/// when they are not tabulated, and a bucket one pair of Pascal transforms.
pub(super) fn moment_round<F: Field>(
    tables: &Tables<'_, F>,
    workers: &mut [Workspace<F>],
    powers: &mut Vec<F>,
    round: MomentRound,
    alphabet: &Alphabet<F>,
) -> Option<Totals<F>> {
    fill_power_table(powers, tables.lut, round.powers)?;
    let workers = workers.get_mut(..round.tables)?;
    let blocks = checked::div_ceil(tables.high.len(), round.tables)?;
    let powers = &*powers;
    for_each_worker(workers, |worker, workspace| {
        workspace.moments(tables, worker, blocks, round, powers)
    })?;
    let workers = &*workers;
    let mut total = sum_rows(round.buckets, |class| {
        let range = checked::range(checked::product([class, INNER_NODES])?, INNER_NODES)?;
        let mut moments = [F::zero(); INNER_NODES];
        for worker in workers {
            for (sum, &value) in moments.iter_mut().zip(worker.moments.get(range.clone())?) {
                *sum += value;
            }
        }
        let mut row = Totals::zero();
        if moments.iter().any(|&moment| moment != F::zero()) {
            let coefficients = alphabet.moment_coefficients(*tables.lut.get(class)?, &moments);
            for (slot, coefficient) in row.alphabet.iter_mut().zip(coefficients) {
                *slot = coefficient;
            }
        }
        Some(row)
    })?;
    for worker in workers {
        total.add_linear(&worker.total.linear);
    }
    total.coefficients_to_nodes(round.right);
    Some(total)
}
