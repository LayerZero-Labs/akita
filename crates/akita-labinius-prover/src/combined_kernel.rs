//! Packed-digit kernel for the combined challenge-field root sumcheck.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

use akita_error::{checked, AkitaError};
use akita_labinius_verifier::channel::ClearChannel;
use akita_labinius_verifier::root_sumcheck::{
    bind_root_sumcheck_instance, combined_input_claim, combined_shape, RootSumcheckInstance,
};
use akita_params::sis::labinius::LABINIUS_BALANCED_LOG_BASIS;
use akita_sumcheck::{
    prove_sumcheck, InfallibleSumcheck, SumcheckInstanceProver, SumcheckProverChannel,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use jolt_poly::UnivariatePoly;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod alphabet;
mod rounds;

use rounds::{Tables, Totals, Workspace};

/// Bits of one stored base-16 digit.
const DIGIT_BITS: usize = LABINIUS_BALANCED_LOG_BASIS as usize;
// Two digits share a byte, and the alphabet polynomial is evaluated through
// its base-16 range image.
const _: () = assert!(DIGIT_BITS == 4);
/// Rounds with a class table: a pair is one byte in round 0 and two bytes in
/// round 1.
const CLASS_ROUNDS: usize = 2;
/// Nodes `0..=16` that fix the alphabet sum of one round, of degree 16.
const INNER_NODES: usize = (1 << DIGIT_BITS) + 1;
/// Largest class bound whose powers the moment round tabulates. A larger
/// table leaves the cache, and computing a pair's powers is then cheaper
/// than reading them.
const POWER_TABLE_CLASSES: usize = 1 << 13;

/// Combined alphabet-and-linear-relation prover with delayed digit lifting.
///
/// Digits are base 16 (`b = 4`), packed two to a byte, and stay packed for
/// `r` rounds. Weights are supplied as a low-variable digit factor of length
/// `D = 2^a` and a compact coefficient table of length `C = N/D`. Only the
/// digit factor folds during the first `a` rounds; its final scalar is then
/// applied to the compact table in place. No length-`N` weight table is
/// constructed. Equality uses two tensor factors.
///
/// **Class rounds** `j = 0, 1` read a pair as one of `M_j = 2^(4*2^(j+1))`
/// classes. `T_j = min(P, N/2^(j+1) / M_j)` workers each own a class table
/// and build histograms with additions only; the round uses a table when
/// `T_j > 0`, so replicated classes never exceed the live pairs. Weights are
/// summed per (class, digit-factor pair) in compact form when
/// `T_j * M_j * D/2^(j+1) <= N/2^(j+1)`, and per class from the two weight
/// endpoints otherwise. The merged histogram is evaluated once: a class costs
/// seventeen multiplications by the powers of one half's value, and each of
/// the `sqrt(M_j)` values of the other half one pair of Pascal transforms.
/// Without a table, packed pairs are evaluated directly.
///
/// **The moment round** `j = 2` reads a pair as two 16-bit classes. Let `U`
/// and `Q` be the smaller and the larger of the two power-of-two bounds on
/// the classes that occur at the two endpoints. The round is used when
/// `8*U <= N/8`; then `r = 3`, and `T = min(P, N/(64*U))` workers each own
/// `17*U` moments, indexed by the class of the endpoint with the smaller
/// bound. The other endpoint enters through the powers of its value, which
/// are tabulated when `Q <= 2^13` and `16*Q <= N/4` and computed for each
/// pair otherwise. A pair costs seventeen multiplications after its powers,
/// and a bucket one pair of Pascal transforms. Otherwise `r = min(nu, 2)`.
/// The two shapes of the root reduction qualify: the last digit slot of
/// both tables is zero, so `U = 4096`; the response table has `Q = 4096`
/// and the image table `Q = 65536`.
///
/// **Later rounds** evaluate each pair of lifted digits on its line through
/// the range image of the alphabet polynomial, and skip pairs of zeros.
///
/// Peak reserved payload, excluding Vec metadata, allocator rounding and the
/// interpolation library's scratch, is bounded by
/// ```text
/// (D + C + V + E + nu + 2*L + G + 21*P) * S + ceil(N/2)
/// + sum_i (M(i) * (4*S + size_of::<usize>() + 1) + B(i) * S + H(i) * S)
/// ```
/// bytes. Here `S = size_of::<F>()`, `V = N/2^r` (zero when `nu = 0`),
/// `E = 2^ceil(max(nu-1,0)/2) + 2^floor(max(nu-1,0)/2)`,
/// `L = 2^(2^(r+1))` for `nu > 0` (one otherwise), and `G = 16 * max(256, Q)`
/// powers (`Q = 0` unless the moment round tabulates them, `G = 0` when
/// `nu = 0`). Worker `i`
/// of `P` owns `M(i)` classes, the largest `M_j` with `i < T_j`; `B(i)`
/// compact sums, the largest admitted `M_j * D/2^(j+1)` over those `j`; and
/// `H(i) = 17*U` moments when `i < T`. `P` is one without `parallel`; with
/// it, `P` is the lesser of the current Rayon pool size and the initial outer
/// equality length. Replace `D` and `C` by the supplied factors' capacities
/// for excess capacity.
///
/// With `S = 16` and `U = 4096`, the bound at `nu = 24`, `a = 2` and
/// `Q = 4096` is 119,243,536 bytes (about 113.7 MiB) for `P = 1` and
/// 223,450,816 bytes (about 213.1 MiB) for `P = 16`; at `nu = 22`, `a = 3`
/// and `Q = 65536` it is 29,082,416 bytes (about 27.7 MiB) and 115,726,048
/// bytes (about 110.4 MiB). All reservations happen in `new`; round
/// computation and folding reuse them. `Field` already requires
/// `Send + Sync`.
#[derive(Debug)]
pub struct CombinedRootKernel<F: Field> {
    packed: Vec<u8>,
    w: Vec<F>,
    kw: Vec<F>,
    digit_factor: Vec<F>,
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
    powers: Vec<F>,
    moment: Option<MomentRound>,
    workers: Vec<Workspace<F>>,
}

/// Geometry of the round that reads a pair as two 16-bit classes.
#[derive(Clone, Copy, Debug)]
struct MomentRound {
    /// Class bound of the endpoint that indexes the moment buckets.
    buckets: usize,
    /// Classes of the other endpoint whose powers are tabulated, or zero when
    /// each pair computes them.
    powers: usize,
    /// The buckets follow the right endpoint.
    right: bool,
    /// Workers that own a moment table.
    tables: usize,
}

impl MomentRound {
    // Each table serves at least `8 * buckets` pairs. Powers are tabulated
    // while the table stays within `POWER_TABLE_CLASSES` and takes no more
    // room than the lift this round postpones would have.
    fn admit(packed: &[u8], nu: usize, workers: usize) -> Option<Self> {
        let pairs = checked::pow2(nu.checked_sub(CLASS_ROUNDS + 1)?)?;
        let [left, right] = class_bounds(packed);
        let (buckets, powers, right) = if right < left {
            (right, left, true)
        } else {
            (left, right, false)
        };
        let tables = workers.min(pairs / checked::product([buckets, 8])?);
        let table = checked::product([powers, alphabet::POWERS])?;
        let tabulated = powers <= POWER_TABLE_CLASSES && table <= checked::product([pairs, 2])?;
        (tables > 0).then_some(Self {
            buckets,
            powers: if tabulated { powers } else { 0 },
            right,
            tables,
        })
    }
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

// Union of the bits of all digits: a digit leaves the alphabet exactly when
// the union has a bit above the digit width.
fn digit_union(digits: &[u8]) -> u8 {
    let union = |digits: &[u8]| digits.iter().fold(0, |all, &digit| all | digit);
    #[cfg(feature = "parallel")]
    let all = digits
        .par_chunks(1 << 16)
        .map(union)
        .reduce(|| 0, |left, right| left | right);
    #[cfg(not(feature = "parallel"))]
    let all = union(digits);
    all
}

// Two digits to a byte, low digit in the low half. A zero-variable table has
// one digit and no pair.
fn pack(digits: &[u8], packed: &mut [u8]) {
    let pack = |(packed, digits): (&mut [u8], &[u8])| {
        let pairs = digits.chunks_exact(2);
        if let (Some(byte), Some(&digit)) = (packed.last_mut(), pairs.remainder().first()) {
            *byte = digit;
        }
        for (byte, pair) in packed.iter_mut().zip(pairs) {
            if let &[low, high] = pair {
                *byte = low | (high << DIGIT_BITS);
            }
        }
    };
    #[cfg(feature = "parallel")]
    packed
        .par_chunks_mut(1 << 15)
        .zip(digits.par_chunks(1 << 16))
        .for_each(pack);
    #[cfg(not(feature = "parallel"))]
    pack((packed, digits));
}

// Power-of-two bounds on the 16-bit classes at the left and right endpoints
// of the pairs of round 2.
fn class_bounds(packed: &[u8]) -> [usize; 2] {
    let union = |packed: &[u8]| {
        packed
            .chunks_exact(4)
            .fold([0u16; 2], |[left, right], pair| match *pair {
                [a, b, c, d] => [
                    left | u16::from_le_bytes([a, b]),
                    right | u16::from_le_bytes([c, d]),
                ],
                _ => [left, right],
            })
    };
    #[cfg(feature = "parallel")]
    let all = packed
        .par_chunks(1 << 16)
        .map(union)
        .reduce(|| [0; 2], |[a, b], [c, d]| [a | c, b | d]);
    #[cfg(not(feature = "parallel"))]
    let all = union(packed);
    all.map(|classes| (usize::from(classes) + 1).next_power_of_two())
}

// Class table of worker `index`: its class count and its compact weight sums.
// Class counts grow and admitted workers shrink with the round, so the
// workers of a round form a prefix. Overflow in any sizing means no table.
fn class_capacity(nu: usize, factor_len: usize, workers: usize, index: usize) -> (usize, usize) {
    let mut capacity = (0, 0);
    for round in 1..=nu.min(CLASS_ROUNDS) {
        let admitted = checked::pow2(round).and_then(|group| {
            let classes = checked::product([DIGIT_BITS, group]).and_then(checked::pow2)?;
            let pairs = checked::pow2(nu.checked_sub(round)?)?;
            let tables = workers.min(pairs / classes);
            let compact = checked::product([classes, factor_len / group])?;
            let fits = checked::product([tables, compact])? <= pairs;
            (index < tables).then_some((classes, if fits { compact } else { 0 }))
        });
        let Some((classes, compact)) = admitted else {
            break;
        };
        capacity = (classes, capacity.1.max(compact));
    }
    capacity
}

impl<F: Field> CombinedRootKernel<F> {
    /// Weights are the tensor of the low-variable digit factor and compact table.
    /// Both factors must have power-of-two length and their product must be `w.len()`.
    /// Statement validation and errors match the dense reference.
    /// A false linear claim is retained verbatim, without altering round values.
    pub fn new(
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
        let shape = combined_shape(tau.len())?;
        if shape.degree_bound() != INNER_NODES {
            return Err(invalid());
        }
        if digit_union(w) >> DIGIT_BITS != 0 {
            return Err(AkitaError::InvalidInput(
                "root digit leaves its alphabet".into(),
            ));
        }
        let mut packed = buffer(checked::div_ceil(expected, 2).ok_or_else(invalid)?, 0u8)?;
        pack(w, &mut packed);
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
        let moment = MomentRound::admit(&packed, tau.len(), worker_count);
        let lift_round = if moment.is_some() {
            CLASS_ROUNDS + 1
        } else {
            tau.len().min(CLASS_ROUNDS)
        };
        let mut field_w = Vec::new();
        let (mut lut_capacity, mut power_capacity) = (1, 0);
        if !tau.is_empty() {
            let len = expected / checked::pow2(lift_round).ok_or_else(invalid)?;
            field_w.try_reserve_exact(len).map_err(|_| invalid())?;
            // The last table holds one value per half of a pair of the lift round.
            lut_capacity = checked::pow2(lift_round)
                .and_then(|digits| checked::product([digits, DIGIT_BITS / 2]))
                .and_then(checked::pow2)
                .ok_or_else(invalid)?;
            let classes = moment
                .map_or(0, |round| round.powers)
                .max(1 << (2 * DIGIT_BITS));
            power_capacity = checked::product([classes, alphabet::POWERS]).ok_or_else(invalid)?;
        }
        let mut workers = Vec::new();
        workers
            .try_reserve_exact(worker_count)
            .map_err(|_| invalid())?;
        for _ in 0..worker_count {
            workers.push(Workspace::new(0, 0, 0)?);
        }
        // The pool zeroes the workers' tables side by side.
        let factor_len = digit_factor.len();
        let allocate = |(index, workspace): (usize, &mut Workspace<F>)| {
            let (classes, compact) = class_capacity(tau.len(), factor_len, worker_count, index);
            let moments = match moment {
                Some(round) if index < round.tables => {
                    checked::product([round.buckets, INNER_NODES]).ok_or_else(invalid)?
                }
                _ => 0,
            };
            *workspace = Workspace::new(classes, compact, moments)?;
            Ok::<(), AkitaError>(())
        };
        #[cfg(feature = "parallel")]
        workers.par_iter_mut().enumerate().try_for_each(allocate)?;
        #[cfg(not(feature = "parallel"))]
        workers.iter_mut().enumerate().try_for_each(allocate)?;
        let mut lut = Vec::new();
        lut.try_reserve_exact(lut_capacity).map_err(|_| invalid())?;
        if !tau.is_empty() {
            lut.extend((0..1u64 << DIGIT_BITS).map(F::from_u64));
        }
        let mut spare_lut = Vec::new();
        spare_lut
            .try_reserve_exact(lut_capacity)
            .map_err(|_| invalid())?;
        let mut powers = Vec::new();
        powers
            .try_reserve_exact(power_capacity)
            .map_err(|_| invalid())?;
        if let &[scalar] = digit_factor.as_slice() {
            scale(&mut kw, scalar);
            digit_factor = Vec::new();
        }
        Ok(Self {
            packed,
            w: field_w,
            kw,
            digit_factor,
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
            powers,
            moment,
            workers,
        })
    }

    /// Return `(W~(rho), K_W~(rho))` after every challenge has been bound.
    pub fn final_evaluations(&self) -> Option<(F, F)> {
        if self.next_round != self.tau.len() {
            return None;
        }
        let w = if self.tau.is_empty() {
            let mask = (1u8 << DIGIT_BITS) - 1;
            F::from_u64(u64::from(*self.packed.first()? & mask))
        } else {
            *self.w.first()?
        };
        Some((w, *self.kw.first()?))
    }

    // Alphabet sums at the nodes `0..=16` and the linear sums of this round.
    fn round_totals(&mut self) -> Option<Totals<F>> {
        let round = self.next_round;
        let packed = round < self.lift_round;
        let class_bits = if packed {
            checked::sum([round, 1])
                .and_then(checked::pow2)
                .and_then(|group| checked::product([group, DIGIT_BITS]))?
        } else {
            0
        };
        let tables = Tables {
            packed: &self.packed,
            w: &self.w,
            kw: &self.kw,
            digit_factor: &self.digit_factor,
            low: &self.eq_low,
            high: &self.eq_high,
            lut: &self.lut,
            class_bits,
        };
        let alphabet = alphabet::Alphabet::new();
        if !packed {
            rounds::direct_round(&tables, &mut self.workers, &alphabet)
        } else if round < CLASS_ROUNDS {
            rounds::class_round(&tables, &mut self.workers, &mut self.powers, &alphabet)
        } else {
            let moment = self.moment?;
            rounds::moment_round(
                &tables,
                &mut self.workers,
                &mut self.powers,
                moment,
                &alphabet,
            )
        }
    }
}

impl<F: Field> SumcheckInstanceProver<F> for CombinedRootKernel<F> {
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
        let Some(mut total) = self.round_totals() else {
            return UnivariatePoly::zero();
        };
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
                let Some(bits) =
                    checked::pow2(self.lift_round).and_then(|n| checked::product([n, DIGIT_BITS]))
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
                self.powers = Vec::new();
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
///
/// The kernel runs in the challenge field `E`; the base field `F` fixes the
/// transcript encoding of its round messages.
pub fn prove_combined_rounds<F, E, C>(
    instance: &mut CombinedRootKernel<E>,
    channel: &mut C,
    invocation: u32,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: SumcheckProverChannel<E> + ClearChannel,
{
    let num_vars = instance.num_rounds();
    let shape = combined_shape(num_vars)?;
    bind_root_sumcheck_instance(
        channel,
        RootSumcheckInstance::Combined,
        invocation,
        num_vars,
    )?;
    prove_sumcheck::<F, E, C, _>(
        &mut InfallibleSumcheck(instance),
        channel,
        shape,
        invocation,
    )
}

fn equality_table<F: Field>(point: &[F]) -> Result<Vec<F>, AkitaError> {
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
fn fold<F: Field>(table: &mut Vec<F>, challenge: F) {
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

fn scale<F: Field>(table: &mut [F], scalar: F) {
    #[cfg(feature = "parallel")]
    table.par_iter_mut().for_each(|value| *value *= scalar);
    #[cfg(not(feature = "parallel"))]
    table.iter_mut().for_each(|value| *value *= scalar);
}

fn fold_chunk<F: Field>(table: &mut [F], challenge: F) {
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

fn marginalize<F: Field>(table: &mut Vec<F>) {
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

// Class code of one pair: its packed bytes, low byte first.
fn class_index(packed: &[u8], pair: usize, bits: usize) -> Option<usize> {
    let bytes = bits / 8;
    let start = checked::product([pair, bytes])?;
    let code = packed.get(checked::range(start, bytes)?)?;
    Some(
        code.iter()
            .rev()
            .fold(0, |class, &byte| (class << 8) | usize::from(byte)),
    )
}
