//! Packed-digit kernel for the combined coefficient-field root sumcheck.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

use akita_algebra::SmoothFftField;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::channel::ClearChannel;
use akita_labinius_verifier::root_sumcheck::{
    alphabet_polynomial, bind_root_sumcheck_instance, combined_input_claim, combined_shape,
    RootSumcheckInstance,
};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::{
    prove_sumcheck, InfallibleSumcheck, SumcheckInstanceProver, SumcheckProverChannel,
};
use jolt_field::ExtField;
use jolt_poly::UnivariatePoly;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Combined alphabet-and-linear-relation prover with delayed digit lifting.
///
/// Zero-based class rounds are 0..4, 0..3 and 0..2 for one-, two- and
/// four-bit digits respectively, limited by `nu`. Buckets are used when the
/// class count is at most `N`; smaller inputs evaluate their packed pairs
/// directly in the remaining early rounds. After the last packed round,
/// the packed digits are lifted directly to `N / 2^r` field elements, where
/// `r = min(nu, 4/3/2)` respectively. Weights are supplied as a low-variable
/// digit factor of length `D=2^a` and a compact coefficient table of length
/// `C=N/D`. Only the digit factor folds during the first `a` rounds; its final
/// scalar is then applied to the compact table in place. No length-`N` weight
/// table is constructed. Equality uses two tensor factors.
///
/// When `a=r`, packed rounds accumulate compact coefficients per (class,
/// digit-pair position) with additions only, and apply digit factors once per
/// bucket. Other factorizations evaluate the two weight endpoints directly;
/// they retain the factors until round `a`, even after the digits are lifted.
///
/// Peak reserved payload, excluding Vec metadata, allocator rounding and the
/// interpolation library's `O(2^b)` scratch, is bounded by
/// ```text
/// (D + C + V + E + nu + 2*L + 21*P) * S + ceil(N*b/8)
/// + P * M * (4*S + size_of::<usize>() + 1) + P * B * S
/// ```
/// bytes.
/// Here `S = size_of::<F>()`, `V = N/2^r` (zero when `nu=0`),
/// `E = 2^ceil(max(nu-1,0)/2) + 2^floor(max(nu-1,0)/2)`,
/// `M` is the largest `2^(b*2^j) <= N` for `1 <= j <= r` (zero if absent),
/// and `L = 2^(b*2^(r-1))` for `nu>0` (one otherwise).
/// `B` is the largest `2^(b*2^j) * D/2^j` over those same `j` when `a=r`,
/// and zero otherwise. `P` is one without `parallel`; with it, `P` is the
/// lesser of the current Rayon pool size and initial outer equality length.
/// Workers own their buckets, so reductions only add exact field elements.
/// Replace `D` and `C` by the supplied factors' capacities for excess capacity.
/// At `nu=25`, `b=2`, `a=3`, `S=16` and `P=1`, the bound is 148,579,168 bytes
/// (about 141.7 MiB). At `P=16` it is 236,074,768 bytes (about 225.1 MiB).
/// All reservations happen in `new`; round computation and folding reuse them.
/// `SmoothFftField` already requires `Send + Sync` through `Field`.
#[derive(Debug)]
pub struct CombinedRootKernel<F: SmoothFftField> {
    base: LabiniusDigitBase,
    packed: Vec<u8>,
    w: Vec<F>,
    kw: Vec<F>,
    digit_factor: Vec<F>,
    factor_buckets: bool,
    tau: Vec<F>,
    eq_low: Vec<F>,
    eq_high: Vec<F>,
    prefix: F,
    beta: F,
    claim: F,
    degree: usize,
    next_round: usize,
    lift_round: usize,
    lut: Vec<F>,
    spare_lut: Vec<F>,
    workers: Vec<Workspace<F>>,
}

fn invalid() -> AkitaError {
    AkitaError::InvalidInput("invalid combined root sumcheck geometry".into())
}

// Allocation boundary for buffers reused by the infallible sumcheck trait.
fn buffer<T: Clone>(len: usize, value: T) -> Result<Vec<T>, AkitaError> {
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|_| invalid())?;
    output.resize(len, value);
    Ok(output)
}

impl<F: SmoothFftField> CombinedRootKernel<F> {
    /// Weights are the tensor of the low-variable digit factor and compact table.
    /// Both factors must have power-of-two length and their product must be `w.len()`.
    /// Statement validation and errors match the dense reference.
    /// A false linear claim is retained verbatim, without altering round values.
    pub fn new(
        base: LabiniusDigitBase,
        w: &[u8],
        mut digit_factor: Vec<F>,
        mut kw: Vec<F>,
        tau: &[F],
        beta: F,
        s: F,
    ) -> Result<Self, AkitaError> {
        let expected = checked::pow2(tau.len()).ok_or_else(invalid)?;
        if !w.len().is_power_of_two()
            || w.len() != expected
            || !digit_factor.len().is_power_of_two()
            || !kw.len().is_power_of_two()
            || checked::product([digit_factor.len(), kw.len()]) != Some(expected)
        {
            return Err(invalid());
        }
        let shape = combined_shape(tau.len(), base)?;
        let bits = base.bits() as usize;
        let alphabet = checked::pow2(bits).ok_or_else(invalid)?;
        if w.iter().any(|&digit| usize::from(digit) >= alphabet) {
            return Err(AkitaError::InvalidInput(
                "root digit leaves its alphabet".into(),
            ));
        }
        let lift_round = tau.len().min(match base {
            LabiniusDigitBase::Bits1 => 4,
            LabiniusDigitBase::Bits2 => 3,
            LabiniusDigitBase::Bits4 => 2,
        });
        let class_bits = checked::product([bits, checked::pow2(lift_round).ok_or_else(invalid)?])
            .ok_or_else(invalid)?;
        // Small inputs retain the same packed rounds but evaluate pairs
        // directly when a full class space would exceed the original table.
        let factor_buckets = digit_factor.len() == checked::pow2(lift_round).ok_or_else(invalid)?;
        let mut compact_classes = 0;
        let mut classes = 0;
        for round in 1..=lift_round {
            let bits = checked::product([bits, checked::pow2(round).ok_or_else(invalid)?])
                .ok_or_else(invalid)?;
            let count = checked::pow2(bits).ok_or_else(invalid)?;
            if count <= expected {
                classes = count;
                if factor_buckets {
                    compact_classes = compact_classes.max(
                        checked::product([count, digit_factor.len() >> round])
                            .ok_or_else(invalid)?,
                    );
                }
            }
        }
        let lut_capacity = if tau.is_empty() {
            1
        } else {
            checked::pow2(class_bits / 2).ok_or_else(invalid)?
        };
        let digits_per_byte = 8 / bits;
        let packed_len = checked::div_ceil(expected, digits_per_byte).ok_or_else(invalid)?;
        let mut packed = buffer(packed_len, 0u8)?;
        for (byte, digits) in packed.iter_mut().zip(w.chunks(digits_per_byte)) {
            for (index, &digit) in digits.iter().enumerate() {
                let shift = checked::product([index, bits]).ok_or_else(invalid)?;
                *byte |= digit << shift;
            }
        }
        let mut field_w = Vec::new();
        if !tau.is_empty() {
            let len = expected / checked::pow2(lift_round).ok_or_else(invalid)?;
            field_w.try_reserve_exact(len).map_err(|_| invalid())?;
        }
        let mut equality_point = Vec::new();
        equality_point
            .try_reserve_exact(tau.len())
            .map_err(|_| invalid())?;
        equality_point.extend_from_slice(tau);
        let suffix = tau.get(1..).unwrap_or_default();
        let low_bits = checked::div_ceil(suffix.len(), 2).ok_or_else(invalid)?;
        let (low, high) = suffix.split_at(low_bits);
        let eq_low = equality_table(low)?;
        let eq_high = equality_table(high)?;
        #[cfg(feature = "parallel")]
        let worker_count = rayon::current_num_threads().min(eq_high.len());
        #[cfg(not(feature = "parallel"))]
        let worker_count = 1;
        let mut workers = Vec::new();
        workers
            .try_reserve_exact(worker_count)
            .map_err(|_| invalid())?;
        for _ in 0..worker_count {
            workers.push(Workspace::new(classes, compact_classes)?);
        }
        let mut lut = Vec::new();
        lut.try_reserve_exact(lut_capacity).map_err(|_| invalid())?;
        if !tau.is_empty() {
            lut.extend((0..alphabet).map(|d| F::from_u64(d as u64)));
        }
        let mut spare_lut = Vec::new();
        spare_lut
            .try_reserve_exact(lut_capacity)
            .map_err(|_| invalid())?;
        if let &[scalar] = digit_factor.as_slice() {
            scale(&mut kw, scalar);
            digit_factor = Vec::new();
        }
        Ok(Self {
            base,
            packed,
            w: field_w,
            kw,
            digit_factor,
            factor_buckets,
            tau: equality_point,
            eq_low,
            eq_high,
            prefix: F::one(),
            beta,
            claim: combined_input_claim(beta, s),
            degree: shape.degree_bound(),
            next_round: 0,
            lift_round,
            lut,
            spare_lut,
            workers,
        })
    }

    /// Return `(W~(rho), K_W~(rho))` after every challenge has been bound.
    pub fn final_evaluations(&self) -> Option<(F, F)> {
        if self.next_round != self.tau.len() {
            return None;
        }
        let w = if self.tau.is_empty() {
            let mask = (1u8 << self.base.bits()) - 1;
            F::from_u64(u64::from(*self.packed.first()? & mask))
        } else {
            *self.w.first()?
        };
        Some((w, *self.kw.first()?))
    }
}

impl<F: SmoothFftField> SumcheckInstanceProver<F> for CombinedRootKernel<F> {
    fn num_rounds(&self) -> usize {
        self.tau.len()
    }
    fn degree_bound(&self) -> usize {
        self.degree
    }
    fn input_claim(&self) -> F {
        self.claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        let Some(&tau) = self.tau.get(self.next_round) else {
            return UnivariatePoly::zero();
        };
        let Some(nodes) = checked::sum([self.degree, 1]) else {
            return UnivariatePoly::zero();
        };
        let class_bits = checked::sum([self.next_round, 1])
            .and_then(checked::pow2)
            .and_then(|group| checked::product([group, self.base.bits() as usize]));
        let Some(blocks_per_worker) = checked::div_ceil(self.eq_high.len(), self.workers.len())
        else {
            return UnivariatePoly::zero();
        };
        let tables = Tables {
            base: self.base,
            packed: &self.packed,
            w: &self.w,
            kw: &self.kw,
            digit_factor: &self.digit_factor,
            factor_buckets: self.factor_buckets && !self.digit_factor.is_empty(),
            low: &self.eq_low,
            high: &self.eq_high,
            lut: &self.lut,
            class_bits: class_bits.unwrap_or(0),
            alphabet_nodes: self.degree,
            class_round: self.next_round < self.lift_round,
            buckets: class_bits.and_then(checked::pow2).is_some_and(|count| {
                self.workers
                    .first()
                    .is_some_and(|w| count <= w.buckets.len())
            }),
        };
        // Each partition mutates only its preallocated workspace. The final
        // reduction consists solely of associative, exact field additions.
        #[cfg(feature = "parallel")]
        self.workers
            .par_iter_mut()
            .enumerate()
            .for_each(|(worker, workspace)| {
                workspace.compute(&tables, worker, blocks_per_worker);
            });
        #[cfg(not(feature = "parallel"))]
        for (worker, workspace) in self.workers.iter_mut().enumerate() {
            workspace.compute(&tables, worker, blocks_per_worker);
        }
        let mut total = Totals::zero();
        for worker in &self.workers {
            total.add(&worker.total);
        }
        // The alphabet sum has degree 2^b. Extrapolate its one missing node by
        // forward differences, before multiplying by the current equality line.
        let mut differences = total.alphabet;
        let mut extra = F::zero();
        for length in (1..=self.degree).rev() {
            let Some(&last) = differences.get(length - 1) else {
                return UnivariatePoly::zero();
            };
            extra += last;
            for index in 0..length - 1 {
                let Some((&left, &right)) = differences.get(index).zip(differences.get(index + 1))
                else {
                    return UnivariatePoly::zero();
                };
                if let Some(target) = differences.get_mut(index) {
                    *target = right - left;
                }
            }
        }
        if let Some(last) = total.alphabet.get_mut(self.degree) {
            *last = extra;
        }
        let [l0, l1, quadratic] = total.linear;
        let mut linear = l0;
        let mut linear_step = l1 - l0;
        let linear_second = quadratic + quadratic;
        let mut equality = self.prefix * (F::one() - tau);
        let equality_step = self.prefix * (tau + tau - F::one());
        for value in total.alphabet.iter_mut().take(nodes) {
            *value = equality * *value + self.beta * linear;
            equality += equality_step;
            linear += linear_step;
            linear_step += linear_second;
        }
        let Some(values) = total.alphabet.get(..nodes) else {
            return UnivariatePoly::zero();
        };
        UnivariatePoly::from_evals(values)
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: F) {
        let Some(&tau) = self.tau.get(self.next_round) else {
            return;
        };
        if self.digit_factor.is_empty() {
            fold(&mut self.kw, challenge);
        } else {
            fold(&mut self.digit_factor, challenge);
            if let &[scalar] = self.digit_factor.as_slice() {
                scale(&mut self.kw, scalar);
                self.digit_factor = Vec::new();
            }
        }
        if self.next_round < self.lift_round {
            if checked::sum([self.next_round, 1]) == Some(self.lift_round) {
                let Some(bits) = checked::pow2(self.lift_round)
                    .and_then(|n| checked::product([n, self.base.bits() as usize]))
                else {
                    return;
                };
                let Some(len) = checked::pow2(self.tau.len() - self.lift_round) else {
                    return;
                };
                self.w.resize(len, F::zero());
                let lut = &self.lut;
                let packed = &self.packed;
                let bind = |(pair, output): (usize, &mut F)| {
                    let Some(class) = class_index(packed, pair, bits) else {
                        return;
                    };
                    let Some((&left, &right)) =
                        lut.get(class % lut.len()).zip(lut.get(class / lut.len()))
                    else {
                        return;
                    };
                    *output = left + challenge * (right - left);
                };
                #[cfg(feature = "parallel")]
                self.w.par_iter_mut().enumerate().for_each(bind);
                #[cfg(not(feature = "parallel"))]
                self.w.iter_mut().enumerate().for_each(bind);
                self.packed = Vec::new();
                self.lut = Vec::new();
                self.spare_lut = Vec::new();
            } else {
                self.spare_lut.clear();
                for &right in &self.lut {
                    for &left in &self.lut {
                        self.spare_lut.push(left + challenge * (right - left));
                    }
                }
                core::mem::swap(&mut self.lut, &mut self.spare_lut);
            }
        } else {
            fold(&mut self.w, challenge);
        }
        self.prefix *= (F::one() - tau) * (F::one() - challenge) + tau * challenge;
        if self.eq_low.len() > 1 {
            marginalize(&mut self.eq_low);
        } else if self.eq_high.len() > 1 {
            marginalize(&mut self.eq_high);
        }
        if let Some(next) = checked::sum([self.next_round, 1]) {
            self.next_round = next;
        }
    }
}

/// Bind the same instance identity as the reference and run the same engine.
pub fn prove_combined_rounds<F, C>(
    instance: &mut CombinedRootKernel<F>,
    channel: &mut C,
    invocation: u32,
) -> Result<(Vec<F>, F), AkitaError>
where
    F: SmoothFftField + ExtField<F>,
    C: SumcheckProverChannel<F> + ClearChannel,
{
    let num_vars = instance.num_rounds();
    let shape = combined_shape(num_vars, instance.base)?;
    bind_root_sumcheck_instance(
        channel,
        RootSumcheckInstance::Combined(instance.base),
        invocation,
        num_vars,
    )?;
    prove_sumcheck::<F, F, C, _>(
        &mut InfallibleSumcheck(instance),
        channel,
        shape,
        invocation,
    )
}

fn equality_table<F: SmoothFftField>(point: &[F]) -> Result<Vec<F>, AkitaError> {
    let capacity = checked::pow2(point.len()).ok_or_else(invalid)?;
    let mut table = Vec::new();
    table.try_reserve_exact(capacity).map_err(|_| invalid())?;
    table.push(F::one());
    for &coordinate in point {
        let len = table.len();
        for index in 0..len {
            let value = *table.get(index).ok_or_else(invalid)?;
            *table.get_mut(index).ok_or_else(invalid)? = value * (F::one() - coordinate);
            table.push(value * coordinate);
        }
    }
    Ok(table)
}

// Fold disjoint chunks in place first, then compact their live halves. No
// worker reads another worker's input, and compaction never overtakes unread
// chunks. This retains the caller's original allocation even with Rayon.
fn fold<F: SmoothFftField>(table: &mut Vec<F>, challenge: F) {
    if table.len() < 2 {
        return;
    }
    let half = table.len() / 2;
    #[cfg(feature = "parallel")]
    {
        let width = table.len().min(4096);
        table
            .par_chunks_mut(width)
            .for_each(|chunk| fold_chunk(chunk, challenge));
        let live = width / 2;
        for block in 1..table.len() / width {
            let Some(source) = checked::product([block, width]) else {
                return;
            };
            let Some(destination) = checked::product([block, live]) else {
                return;
            };
            let Some(range) = checked::range(source, live) else {
                return;
            };
            let Some(output) = checked::range(destination, live) else {
                return;
            };
            if table.get(range.clone()).is_none() || output.end > table.len() {
                return;
            }
            table.copy_within(range, destination);
        }
    }
    #[cfg(not(feature = "parallel"))]
    fold_chunk(table, challenge);
    table.truncate(half);
}

fn scale<F: SmoothFftField>(table: &mut [F], scalar: F) {
    #[cfg(feature = "parallel")]
    table.par_iter_mut().for_each(|value| *value *= scalar);
    #[cfg(not(feature = "parallel"))]
    table.iter_mut().for_each(|value| *value *= scalar);
}

fn fold_chunk<F: SmoothFftField>(table: &mut [F], challenge: F) {
    for index in 0..table.len() / 2 {
        let Some(start) = checked::product([index, 2]) else {
            return;
        };
        let Some(range) = checked::range(start, 2) else {
            return;
        };
        let Some(&[left, right]) = table.get(range) else {
            return;
        };
        if let Some(target) = table.get_mut(index) {
            *target = left + challenge * (right - left);
        }
    }
}

fn marginalize<F: SmoothFftField>(table: &mut Vec<F>) {
    let half = table.len() / 2;
    for index in 0..half {
        let Some(start) = checked::product([index, 2]) else {
            return;
        };
        let Some(range) = checked::range(start, 2) else {
            return;
        };
        let Some(&[left, right]) = table.get(range) else {
            return;
        };
        if let Some(target) = table.get_mut(index) {
            *target = left + right;
        }
    }
    table.truncate(half);
}

// Groups are byte-aligned once their class code spans more than eight bits.
fn class_index(packed: &[u8], pair: usize, bits: usize) -> Option<usize> {
    if bits <= 8 && bits != 0 {
        let per_byte = 8 / bits;
        let shift = checked::product([pair % per_byte, bits])?;
        let mask = checked::pow2(bits)?.checked_sub(1)?;
        Some((usize::from(*packed.get(pair / per_byte)?) >> shift) & mask)
    } else if bits == 16 {
        let start = checked::product([pair, 2])?;
        let &[low, high] = packed.get(checked::range(start, 2)?)? else {
            return None;
        };
        Some(usize::from(u16::from_le_bytes([low, high])))
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct Totals<F> {
    alphabet: [F; 18],
    // L(0), L(1), and the coefficient of t^2.
    linear: [F; 3],
}
impl<F: SmoothFftField> Totals<F> {
    fn zero() -> Self {
        Self {
            alphabet: [F::zero(); 18],
            linear: [F::zero(); 3],
        }
    }
    fn add(&mut self, other: &Self) {
        for (a, &b) in self.alphabet.iter_mut().zip(&other.alphabet) {
            *a += b;
        }
        for (a, &b) in self.linear.iter_mut().zip(&other.linear) {
            *a += b;
        }
    }
    fn product(&mut self, w0: F, w1: F, k0: F, k1: F) {
        let products = [w0 * k0, w1 * k1, (w1 - w0) * (k1 - k0)];
        for (sum, value) in self.linear.iter_mut().zip(products) {
            *sum += value;
        }
    }
}

struct Tables<'a, F> {
    base: LabiniusDigitBase,
    packed: &'a [u8],
    w: &'a [F],
    kw: &'a [F],
    digit_factor: &'a [F],
    factor_buckets: bool,
    low: &'a [F],
    high: &'a [F],
    lut: &'a [F],
    class_bits: usize,
    alphabet_nodes: usize,
    class_round: bool,
    buckets: bool,
}

impl<F: SmoothFftField> Tables<'_, F> {
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
}

#[derive(Debug)]
struct Workspace<F> {
    // Suffix equality weight and the two unweighted K endpoints per class.
    buckets: Vec<[F; 3]>,
    // Compact coefficients per (packed class, remaining digit-pair position).
    compact_buckets: Vec<F>,
    inner: Vec<F>,
    marked: Vec<u8>,
    touched: Vec<usize>,
    total: Totals<F>,
}
impl<F: SmoothFftField> Workspace<F> {
    fn new(classes: usize, compact_classes: usize) -> Result<Self, AkitaError> {
        let mut touched = Vec::new();
        touched.try_reserve_exact(classes).map_err(|_| invalid())?;
        Ok(Self {
            buckets: buffer(classes, [F::zero(); 3])?,
            compact_buckets: buffer(compact_classes, F::zero())?,
            inner: buffer(classes, F::zero())?,
            marked: buffer(classes, 0)?,
            touched,
            total: Totals::zero(),
        })
    }

    fn compute(&mut self, tables: &Tables<'_, F>, worker: usize, block_count: usize) {
        self.total = Totals::zero();
        let Some(start) = checked::product([worker, block_count]) else {
            return;
        };
        let Some(block_width) = checked::product([tables.low.len(), 2]) else {
            return;
        };
        if tables.class_round && tables.buckets {
            self.buckets.fill([F::zero(); 3]);
            self.compact_buckets.fill(F::zero());
        }
        for (block, &outer) in tables.high.iter().enumerate().skip(start).take(block_count) {
            if tables.class_round && tables.buckets {
                for (inner, &eq) in tables.low.iter().enumerate() {
                    let Some(pair) = checked::mul_add(block, tables.low.len(), inner) else {
                        return;
                    };
                    let Some(class) = class_index(tables.packed, pair, tables.class_bits) else {
                        return;
                    };
                    if tables.factor_buckets {
                        let pairs = tables.digit_factor.len() / 2;
                        let Some(index) = checked::mul_add(class, pairs, pair & (pairs - 1)) else {
                            return;
                        };
                        let Some(sum) = self.compact_buckets.get_mut(index) else {
                            return;
                        };
                        let Some(&coefficient) = tables.kw.get(pair >> pairs.trailing_zeros())
                        else {
                            return;
                        };
                        *sum += coefficient;
                    } else {
                        let Some((k0, k1)) = tables.weight_pair(pair) else {
                            return;
                        };
                        let Some([_, sum0, sum1]) = self.buckets.get_mut(class) else {
                            return;
                        };
                        *sum0 += k0;
                        *sum1 += k1;
                    }
                    let Some(eq_sum) = self.inner.get_mut(class) else {
                        return;
                    };
                    *eq_sum += eq;
                    let Some(mark) = self.marked.get_mut(class) else {
                        return;
                    };
                    if *mark == 0 {
                        *mark = 1;
                        self.touched.push(class);
                    }
                }
                // Outer equality is applied only to occupied class buckets,
                // once per inner block. The pair loop above uses additions only.
                for class in self.touched.drain(..) {
                    let Some(eq_sum) = self.inner.get_mut(class) else {
                        return;
                    };
                    let Some([eq, _, _]) = self.buckets.get_mut(class) else {
                        return;
                    };
                    *eq += *eq_sum * outer;
                    *eq_sum = F::zero();
                    if let Some(mark) = self.marked.get_mut(class) {
                        *mark = 0;
                    }
                }
            } else {
                let Some(offset) = checked::product([block, block_width]) else {
                    return;
                };
                let Some(range) = checked::range(offset, block_width) else {
                    return;
                };
                let digits = tables.w.get(range);
                let mut alphabet = [F::zero(); 18];
                for (inner, &eq) in tables.low.iter().enumerate() {
                    let (w0, w1) = if tables.class_round {
                        let Some(pair) = checked::mul_add(block, tables.low.len(), inner) else {
                            return;
                        };
                        let Some(class) = class_index(tables.packed, pair, tables.class_bits)
                        else {
                            return;
                        };
                        let Some((&left, &right)) = tables
                            .lut
                            .get(class % tables.lut.len())
                            .zip(tables.lut.get(class / tables.lut.len()))
                        else {
                            return;
                        };
                        (left, right)
                    } else {
                        let Some(start) = checked::product([inner, 2]) else {
                            return;
                        };
                        let Some(range) = checked::range(start, 2) else {
                            return;
                        };
                        let Some(&[left, right]) = digits.and_then(|d| d.get(range)) else {
                            return;
                        };
                        (left, right)
                    };
                    let Some(pair) = checked::mul_add(block, tables.low.len(), inner) else {
                        return;
                    };
                    let Some((k0, k1)) = tables.weight_pair(pair) else {
                        return;
                    };
                    self.total.product(w0, w1, k0, k1);
                    let delta = w1 - w0;
                    let mut value = w0;
                    for sum in alphabet.iter_mut().take(tables.alphabet_nodes) {
                        *sum += eq * alphabet_polynomial(tables.base, value);
                        value += delta;
                    }
                }
                for (sum, inner) in self.total.alphabet.iter_mut().zip(alphabet) {
                    *sum += outer * inner;
                }
            }
        }
        if tables.class_round && tables.buckets {
            let Some(classes) = checked::pow2(tables.class_bits) else {
                return;
            };
            for (class, &[eq, mut k0, mut k1]) in self.buckets.iter().take(classes).enumerate() {
                if tables.factor_buckets {
                    let pairs = tables.digit_factor.len() / 2;
                    let Some(start) = checked::product([class, pairs]) else {
                        return;
                    };
                    let Some(coefficients) = checked::range(start, pairs)
                        .and_then(|range| self.compact_buckets.get(range))
                    else {
                        return;
                    };
                    for (&coefficient, digit_pair) in
                        coefficients.iter().zip(tables.digit_factor.chunks_exact(2))
                    {
                        if coefficient == F::zero() {
                            continue;
                        }
                        let &[left, right] = digit_pair else {
                            return;
                        };
                        k0 += coefficient * left;
                        k1 += coefficient * right;
                    }
                }
                if eq == F::zero() && k0 == F::zero() && k1 == F::zero() {
                    continue;
                }
                let Some((&w0, &w1)) = tables
                    .lut
                    .get(class % tables.lut.len())
                    .zip(tables.lut.get(class / tables.lut.len()))
                else {
                    return;
                };
                self.total.product(w0, w1, k0, k1);
                let mut value = w0;
                let delta = w1 - w0;
                for sum in self.total.alphabet.iter_mut().take(tables.alphabet_nodes) {
                    *sum += eq * alphabet_polynomial(tables.base, value);
                    value += delta;
                }
            }
        }
    }
}
