//! Compact-witness engine for the leading quotient-factored coefficient rounds.
//!
//! The first `min(coefficient_bits, MAX_PREFIX_ROUNDS)` coefficient rounds run
//! while the witness is still packed signed digits. One scan at construction
//! records everything those rounds need, and the witness is folded to field
//! elements only once, in the last prefix round.
//!
//! - The relation weight `alpha(c) * lane_weight(x)` and every structured
//!   linear weight are linear in the witness, so the scan contracts the lane
//!   coordinate immediately: `alpha_mass[c] = sum_x lane_weight(x) * w(c, x)`,
//!   and each linear source keeps the analogous mass per source lane. A
//!   coefficient round of the relation term is then a product sumcheck between
//!   these short vectors and the folded `alpha` and source tables.
//! - Each witness quad (four consecutive coefficients) is recorded by its digit
//!   class. After the first two challenges a quad's folded value depends only
//!   on its class. Class histograms weighted by the equality factor over the
//!   variables after the quad give the first two range-image messages (through
//!   the `{0, 1, Infinity}^2` grid) and the third one (through the even/odd
//!   quad split and the histogram of class differences).
//! - Each later prefix round reads the folded witness as a sum of class lookups
//!   into challenge-scaled copies of the two-round quad fold. The last prefix
//!   round writes the folded witness for the ordinary suffix path.

use super::*;
use crate::opaque::sumcheck::relation_range_image::evaluation_trace::PreparedLaneWeights;
use crate::opaque::sumcheck::two_round_prefix::Stage2PrefixCache;
use akita_algebra::eq_poly::EqPolynomial;

/// Maximum number of coefficient rounds served from the compact witness.
const MAX_PREFIX_ROUNDS: usize = 6;
const _: () = assert!(MAX_PREFIX_ROUNDS >= 2 && MAX_PREFIX_ROUNDS <= 7);

/// Minimum lanes per parallel scan task.
const MIN_SCAN_TASK_LANES: usize = 256;

/// Minimum folded-witness pairs per parallel lookup-round task.
const MIN_LOOKUP_TASK_PAIRS: usize = 1 << 12;

/// Range-image message of one prefix round.
pub(super) enum PrefixNormRound<E: Field> {
    /// Complete message, already scaled by the split-equality scalar.
    Polynomial(UnivariatePoly<E>),
    /// Inner message that the split-equality factor still multiplies.
    Terms(NormRoundTerms<E>),
}

/// Leading-round state built from one scan of the compact witness.
pub(super) struct CompactQuotientPrefix<E: Field> {
    b: usize,
    digit_bits: usize,
    /// Last coefficient round the engine serves; it materializes the witness.
    last_round: usize,
    /// Digit class of every witness quad, in witness order.
    classes: Vec<u16>,
    norm_cache: Stage2PrefixCache<E>,
    /// Round-2 equality weight of each quad class, for even and odd quads.
    even_histogram: Vec<E>,
    odd_histogram: Vec<E>,
    /// Round-2 equality weight of each odd-minus-even quad digit difference.
    delta_histogram: Vec<E>,
    alpha_mass: Vec<E>,
    source_mass: Vec<Vec<E>>,
    /// Folded value of every quad class after the first two challenges.
    quad_fold: Vec<E>,
    challenges: Vec<E>,
}

/// Radix of one digit of a quad difference: differences span `2b - 1` values.
#[inline]
fn delta_radix(b: usize) -> usize {
    2 * b - 1
}

#[inline(always)]
fn quad_class<const DIGIT_BITS: usize>(quad: [i8; 4]) -> u16 {
    // Adding `b / 2` to a digit in `[-b / 2, b / 2)` flips the top bit of its
    // low `DIGIT_BITS` bits; two shift-merge steps then pack the four bytes.
    let bytes = u32::from_le_bytes(quad.map(|digit| digit as u8));
    let biased =
        (bytes & (((1 << DIGIT_BITS) - 1) * 0x0101_0101)) ^ ((1 << (DIGIT_BITS - 1)) * 0x0101_0101);
    let pairs =
        (biased | (biased >> (8 - DIGIT_BITS))) & (((1 << (2 * DIGIT_BITS)) - 1) * 0x0001_0001);
    ((pairs | (pairs >> (16 - 2 * DIGIT_BITS))) & ((1 << (4 * DIGIT_BITS)) - 1)) as u16
}

/// Largest digit basis the engine serves.
const MAX_DIGIT_BASIS: usize = 8;

/// Signed-digit multiples of field elements, accumulated in wide lanes.
///
/// Slots are grouped into blocks that always receive a whole row of digits at
/// once. A block is reduced before any of its `i32` lanes can overflow.
struct WideMass<E: Field + Unreduced> {
    wide: Vec<E::Wide>,
    reduced: Vec<E>,
    pending: Vec<usize>,
    block_len: usize,
    max_pending: usize,
    /// Digits lie in `[-half, half)`.
    half: usize,
}

impl<E: Field + Unreduced> WideMass<E> {
    fn new(block_count: usize, block_len: usize, max_pending: usize, b: usize) -> Self {
        debug_assert!(b.is_power_of_two() && (2..=MAX_DIGIT_BASIS).contains(&b));
        Self {
            wide: vec![E::Wide::zero(); block_count * block_len],
            reduced: vec![E::zero(); block_count * block_len],
            pending: vec![0; block_count],
            block_len,
            max_pending,
            half: b / 2,
        }
    }

    /// Add `factor * digits[i]` to slot `i` of block `block`.
    #[inline]
    fn add_row(&mut self, block: usize, factor: E, digits: &[i8]) {
        if self.pending[block] == self.max_pending {
            self.flush(block);
        }
        self.pending[block] += 1;
        let start = block * self.block_len;
        let masses = &mut self.wide[start..start + self.block_len];
        if cfg!(target_arch = "aarch64") {
            // NEON multiplies `i32` lanes natively.
            for (mass, &digit) in masses.iter_mut().zip(digits) {
                *mass += factor.scale_wide(i32::from(digit));
            }
            return;
        }
        // Without a vector `i32` multiply (baseline x86-64 has none) every lane
        // would be scaled in scalar, so build the row's digit multiples of
        // `factor` by lane-wise additions and add one per digit.
        let half = self.half;
        let base = E::Wide::from(factor);
        let mut multiples = [E::Wide::zero(); MAX_DIGIT_BASIS];
        for offset in 1..=half {
            if half + offset < 2 * half {
                multiples[half + offset] = multiples[half + offset - 1] + base;
            }
            multiples[half - offset] = multiples[half + 1 - offset] - base;
        }
        for (mass, &digit) in masses.iter_mut().zip(digits) {
            debug_assert!((-(half as i16)..half as i16).contains(&i16::from(digit)));
            let index = (i16::from(digit) + half as i16) as usize & (MAX_DIGIT_BASIS - 1);
            *mass += multiples[index];
        }
    }

    fn flush(&mut self, block: usize) {
        let slots = block * self.block_len..(block + 1) * self.block_len;
        for (reduced, wide) in self.reduced[slots.clone()]
            .iter_mut()
            .zip(&mut self.wide[slots])
        {
            *reduced += E::reduce_wide(*wide);
            *wide = E::Wide::zero();
        }
        self.pending[block] = 0;
    }

    fn finish(mut self) -> Vec<E> {
        for block in 0..self.pending.len() {
            if self.pending[block] != 0 {
                self.flush(block);
            }
        }
        self.reduced
    }
}

struct ScanLayout<'a, E: Field> {
    witness: PackedSignedDigitView<'a>,
    lane_weights: &'a [E],
    linear_terms: &'a PreparedProverLinearTerms<E>,
    /// First source-mass block of each linear source.
    source_blocks: &'a [usize],
    source_block_count: usize,
    eq_low: &'a [E],
    eq_high: &'a [E],
    tau2: E,
    b: usize,
    coeff_count: usize,
    class_count: usize,
    /// Per-class base-`2b - 1` digit code; empty when round 2 is not served.
    delta_code: &'a [i32],
    delta_offset: i32,
    delta_count: usize,
    max_wide_adds: usize,
}

struct ScanTotals<E: Field> {
    alpha_mass: Vec<E>,
    source_mass: Vec<E>,
    even_histogram: Vec<E>,
    odd_histogram: Vec<E>,
    delta_histogram: Vec<E>,
}

fn add_assign_all<E: Field>(left: &mut [E], right: &[E]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left += *right;
    }
}

impl<E: Field> ScanTotals<E> {
    fn merge(mut self, other: Self) -> Self {
        add_assign_all(&mut self.alpha_mass, &other.alpha_mass);
        add_assign_all(&mut self.source_mass, &other.source_mass);
        add_assign_all(&mut self.even_histogram, &other.even_histogram);
        add_assign_all(&mut self.odd_histogram, &other.odd_histogram);
        add_assign_all(&mut self.delta_histogram, &other.delta_histogram);
        self
    }
}

/// Scan the lanes whose quad classes `classes` receives.
///
/// With at least eight coefficients per lane, the round-2 pair of a quad is the
/// adjacent quad in the same lane, so even and odd quads get separate
/// histograms. With four coefficients the pair crosses lanes; only the mixed
/// histogram for the first two rounds exists, and the engine stops at round 1.
fn scan_lanes<E: Field + Unreduced, const DIGIT_BITS: usize>(
    layout: &ScanLayout<'_, E>,
    first_lane: usize,
    classes: &mut [u16],
) -> ScanTotals<E> {
    let coeff_count = layout.coeff_count;
    let quads_per_lane = coeff_count / 4;
    let split_pairs = quads_per_lane >= 2;
    let low_mask = layout.eq_low.len() - 1;
    let low_bits = layout.eq_low.len().trailing_zeros();
    let one_minus_tau2 = E::one() - layout.tau2;

    let mut digits = vec![0i8; coeff_count];
    let mut alpha_mass = WideMass::<E>::new(1, coeff_count, layout.max_wide_adds, layout.b);
    let mut source_mass = WideMass::<E>::new(
        layout.source_block_count,
        coeff_count,
        layout.max_wide_adds,
        layout.b,
    );
    let mut even_histogram = vec![E::zero(); layout.class_count];
    let mut odd_histogram = if split_pairs {
        vec![E::zero(); layout.class_count]
    } else {
        Vec::new()
    };
    let mut delta_histogram = vec![E::zero(); layout.delta_count];

    for (local_lane, lane_classes) in classes.chunks_exact_mut(quads_per_lane).enumerate() {
        let lane = first_lane + local_lane;
        layout
            .witness
            .decode_range(lane * coeff_count, &mut digits)
            .expect("compact prefix lane is in bounds");

        alpha_mass.add_row(0, layout.lane_weights[lane], &digits);
        layout
            .linear_terms
            .for_each_source_term(lane, |factor, source_index, source_lane| {
                source_mass.add_row(
                    layout.source_blocks[source_index] + source_lane,
                    factor,
                    &digits,
                );
            });

        for (class, &quad) in lane_classes.iter_mut().zip(digits.as_chunks::<4>().0) {
            *class = quad_class::<DIGIT_BITS>(quad);
        }

        if split_pairs {
            let first_pair = lane * (quads_per_lane / 2);
            for (pair_offset, pair) in lane_classes.chunks_exact(2).enumerate() {
                let pair_index = first_pair + pair_offset;
                let weight =
                    layout.eq_low[pair_index & low_mask] * layout.eq_high[pair_index >> low_bits];
                let (even, odd) = (usize::from(pair[0]), usize::from(pair[1]));
                even_histogram[even] += weight;
                odd_histogram[odd] += weight;
                if !delta_histogram.is_empty() {
                    let delta =
                        layout.delta_code[odd] - layout.delta_code[even] + layout.delta_offset;
                    delta_histogram[delta as usize] += weight;
                }
            }
        } else {
            let pair_index = lane >> 1;
            let side = if lane & 1 == 0 {
                one_minus_tau2
            } else {
                layout.tau2
            };
            let weight = side
                * layout.eq_low[pair_index & low_mask]
                * layout.eq_high[pair_index >> low_bits];
            even_histogram[usize::from(lane_classes[0])] += weight;
        }
    }

    ScanTotals {
        alpha_mass: alpha_mass.finish(),
        source_mass: source_mass.finish(),
        even_histogram,
        odd_histogram,
        delta_histogram,
    }
}

/// Accumulate the range-image inner terms of pairs `pairs` of the folded
/// witness at round `2 + log2(G)`, optionally writing the folded pairs.
///
/// Folded entry `m` is `sum_{i < G} lookups[i][classes[m * G + i]]`, where
/// `lookups[i]` is the two-round quad fold scaled by the equality weight of
/// `i` over the challenges after the first two.
#[allow(clippy::too_many_arguments)]
fn lookup_pairs<E: Field + Unreduced, const G: usize, const SKIP_LINEAR: bool>(
    lookups: &[E],
    class_count: usize,
    classes: &[u16],
    eq_low: &[E],
    eq_high: &[E],
    pairs: std::ops::Range<usize>,
    mut out: Option<&mut [E]>,
) -> [E; 3] {
    let low_mask = eq_low.len() - 1;
    let low_bits = eq_low.len().trailing_zeros();
    let fold_entry = |entry: usize| {
        let entry_classes = &classes[entry * G..entry * G + G];
        let mut value = lookups[usize::from(entry_classes[0])];
        for (offset, &class) in entry_classes.iter().enumerate().skip(1) {
            value += lookups[offset * class_count + usize::from(class)];
        }
        value
    };

    let mut totals = [E::zero(); 3];
    let mut block_start = pairs.start;
    while block_start < pairs.end {
        let high = block_start >> low_bits;
        let block_end = ((high + 1) << low_bits).min(pairs.end);
        let mut inner = [ProductSum::<E>::zero(); 3];
        for pair in block_start..block_end {
            let w0 = fold_entry(2 * pair);
            let w1 = fold_entry(2 * pair + 1);
            if let Some(out) = out.as_deref_mut() {
                let local = 2 * (pair - pairs.start);
                out[local] = w0;
                out[local + 1] = w1;
            }
            let dw = w1 - w0;
            let e_in = eq_low[pair & low_mask];
            inner[0].add(e_in, w0.square() + w0);
            if !SKIP_LINEAR {
                inner[1].add(e_in, dw * (w0 + w0 + E::one()));
            }
            inner[2].add(e_in, dw.square());
        }
        let e_out = eq_high[high];
        for (total, inner) in totals.iter_mut().zip(inner) {
            *total += e_out * inner.finish();
        }
        block_start = block_end;
    }
    totals
}

impl<E: Field + Ring + Unreduced> CompactQuotientPrefix<E> {
    /// Scan the compact witness, or return `None` when its geometry or digit
    /// range is outside the engine's domain.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "CompactQuotientPrefix::new")]
    pub(super) fn new(
        witness: &PackedSignedDigits,
        lane_weights: &[E],
        linear_terms: &PreparedProverLinearTerms<E>,
        split_eq: &GruenSplitEq<E>,
        stage1_point: &[E],
        batching_coeff: E,
        b: usize,
        live_lane_count: usize,
        coefficient_bits: usize,
    ) -> Option<Self> {
        let digit_bits = match b {
            4 => 2,
            8 => 3,
            _ => return None,
        };
        if coefficient_bits < 2
            || stage1_point.len() < 3
            || !witness.bounds().fits_balanced_log_basis(digit_bits as u32)
            || matches!(linear_terms.lane_weights, PreparedLaneWeights::Dense(_))
            || lane_weights.len() < live_lane_count
            || witness.len() != live_lane_count << coefficient_bits
        {
            return None;
        }
        let (eq_low, eq_high) = split_eq.remaining_eq_tables_after(2)?;
        let last_round = coefficient_bits.min(MAX_PREFIX_ROUNDS) - 1;
        let coeff_count = 1usize << coefficient_bits;
        let quads_per_lane = coeff_count / 4;
        let class_count = 1usize << (4 * digit_bits);

        let source_blocks = linear_terms
            .sources
            .iter()
            .scan(0usize, |block, source| {
                let start = *block;
                *block += source.lane_count;
                Some(start)
            })
            .collect::<Vec<_>>();
        let source_block_count = linear_terms
            .sources
            .iter()
            .map(|source| source.lane_count)
            .sum();

        let serves_round2 = last_round >= 2;
        let radix = delta_radix(b);
        let delta_code = if serves_round2 {
            (0..class_count)
                .map(|class| {
                    (0..4).rev().fold(0i32, |code, digit| {
                        code * radix as i32 + ((class >> (digit_bits * digit)) & (b - 1)) as i32
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let delta_offset = ((b - 1) * (1 + radix + radix * radix + radix * radix * radix)) as i32;
        let layout = ScanLayout {
            witness: witness.view(),
            lane_weights,
            linear_terms,
            source_blocks: &source_blocks,
            source_block_count,
            eq_low,
            eq_high,
            tau2: stage1_point[2],
            b,
            coeff_count,
            class_count,
            delta_code: &delta_code,
            delta_offset,
            delta_count: if serves_round2 { radix.pow(4) } else { 0 },
            max_wide_adds: (i32::MAX as usize) / (usize::from(u16::MAX) * (b / 2)),
        };

        let task_lanes = live_lane_count
            .div_ceil(target_tasks())
            .max(MIN_SCAN_TASK_LANES);
        let mut classes = vec![0u16; live_lane_count * quads_per_lane];
        let scan = |(task, task_classes): (usize, &mut [u16])| match digit_bits {
            2 => scan_lanes::<E, 2>(&layout, task * task_lanes, task_classes),
            _ => scan_lanes::<E, 3>(&layout, task * task_lanes, task_classes),
        };
        #[cfg(feature = "parallel")]
        let totals = classes
            .par_chunks_mut(task_lanes * quads_per_lane)
            .enumerate()
            .map(scan)
            .reduce_with(ScanTotals::merge)?;
        #[cfg(not(feature = "parallel"))]
        let totals = classes
            .chunks_mut(task_lanes * quads_per_lane)
            .enumerate()
            .map(scan)
            .reduce(ScanTotals::merge)?;

        let tau2 = stage1_point[2];
        let mixed_histogram = if quads_per_lane >= 2 {
            totals
                .even_histogram
                .iter()
                .zip(&totals.odd_histogram)
                .map(|(&even, &odd)| even + tau2 * (odd - even))
                .collect()
        } else {
            totals.even_histogram.clone()
        };
        let norm_cache = Stage2PrefixCache::from_norm_histogram(
            &mixed_histogram,
            b,
            stage1_point[0],
            stage1_point[1],
            batching_coeff,
        );
        let (even_histogram, odd_histogram) = if serves_round2 {
            (totals.even_histogram, totals.odd_histogram)
        } else {
            (Vec::new(), Vec::new())
        };
        let source_mass = source_blocks
            .iter()
            .zip(&linear_terms.sources)
            .map(|(&block, source)| {
                totals.source_mass[block * coeff_count..(block + source.lane_count) * coeff_count]
                    .to_vec()
            })
            .collect();

        Some(Self {
            b,
            digit_bits,
            last_round,
            classes,
            norm_cache,
            even_histogram,
            odd_histogram,
            delta_histogram: totals.delta_histogram,
            alpha_mass: totals.alpha_mass,
            source_mass,
            quad_fold: Vec::new(),
            challenges: Vec::new(),
        })
    }

    #[inline]
    pub(super) fn challenges(&self) -> &[E] {
        &self.challenges
    }

    #[inline]
    pub(super) fn last_round(&self) -> usize {
        self.last_round
    }

    /// Relation and structured-linear message of the current round.
    pub(super) fn relation_coeffs(
        &self,
        alpha: &[E],
        linear_terms: &PreparedProverLinearTerms<E>,
    ) -> [E; 3] {
        let mut rel = [E::zero(); 3];
        let masses = std::iter::once((&self.alpha_mass, alpha)).chain(
            self.source_mass
                .iter()
                .zip(linear_terms.sources.iter().map(|source| &source.values[..])),
        );
        for (mass, weight) in masses {
            debug_assert_eq!(mass.len(), weight.len());
            for (mass, weight) in mass.chunks_exact(2).zip(weight.chunks_exact(2)) {
                accumulate_relation_coeffs(
                    &mut rel,
                    mass[0],
                    mass[1] - mass[0],
                    weight[0],
                    weight[1],
                );
            }
        }
        rel
    }

    /// Range-image message of a round that keeps the witness compact.
    pub(super) fn norm_round(
        &self,
        split_eq: &GruenSplitEq<E>,
        skip_linear: bool,
    ) -> PrefixNormRound<E> {
        match self.challenges.len() {
            0 => PrefixNormRound::Polynomial(self.norm_cache.round0_norm_poly()),
            1 => PrefixNormRound::Polynomial(self.norm_cache.round1_norm_poly(self.challenges[0])),
            2 => PrefixNormRound::Terms(self.round2_norm_terms()),
            round => {
                PrefixNormRound::Terms(self.lookup_round_terms(round, split_eq, skip_linear, None))
            }
        }
    }

    /// Materialize the witness folded by every challenge so far and return it
    /// with the range-image message of the last prefix round.
    pub(super) fn materialize(
        &self,
        witness: PackedSignedDigitView<'_>,
        split_eq: &GruenSplitEq<E>,
        skip_linear: bool,
    ) -> (Vec<E>, PrefixNormRound<E>) {
        debug_assert_eq!(self.challenges.len(), self.last_round);
        match self.last_round {
            1 => {
                let r0 = self.challenges[0];
                let half = (self.b / 2) as i16;
                let lut = CompactPairFoldLut::from_contiguous_range(-half, half - 1, r0);
                (
                    RelationRangeImageProver::<E>::materialize_compact_witness(witness, &lut),
                    PrefixNormRound::Polynomial(self.norm_cache.round1_norm_poly(r0)),
                )
            }
            2 => (
                cfg_iter!(self.classes)
                    .map(|&class| self.quad_fold[usize::from(class)])
                    .collect(),
                PrefixNormRound::Terms(self.round2_norm_terms()),
            ),
            round => {
                let mut folded = vec![E::zero(); self.classes.len() >> (round - 2)];
                let terms =
                    self.lookup_round_terms(round, split_eq, skip_linear, Some(&mut folded));
                (folded, PrefixNormRound::Terms(terms))
            }
        }
    }

    /// Round-2 range-image terms from the class histograms.
    ///
    /// The folded witness at a quad is linear in its digits, so the round-2
    /// slope of an even/odd quad pair is the quad fold of their digit
    /// difference.
    fn round2_norm_terms(&self) -> NormRoundTerms<E> {
        let mut at_zero = E::zero();
        let mut at_one = E::zero();
        for ((&fold, &even), &odd) in self
            .quad_fold
            .iter()
            .zip(&self.even_histogram)
            .zip(&self.odd_histogram)
        {
            let norm = fold * (fold + E::one());
            at_zero += even * norm;
            at_one += odd * norm;
        }

        let radix = delta_radix(self.b);
        let quad_weights = self.quad_weights();
        let digit_terms: [Vec<E>; 4] = std::array::from_fn(|digit| {
            (0..radix)
                .map(|value| quad_weights[digit] * E::from_i64(value as i64 - (self.b as i64 - 1)))
                .collect()
        });
        let mut at_infinity = E::zero();
        let mut weights = self.delta_histogram.chunks_exact(radix);
        for d3 in &digit_terms[3] {
            for d2 in &digit_terms[2] {
                let high = *d3 + *d2;
                for d1 in &digit_terms[1] {
                    let upper = high + *d1;
                    let row = weights.next().expect("difference histogram row");
                    for (&weight, d0) in row.iter().zip(&digit_terms[0]) {
                        if !weight.is_zero() {
                            let slope = upper + *d0;
                            at_infinity += weight * (slope * slope);
                        }
                    }
                }
            }
        }
        NormRoundTerms::Full([at_zero, at_one - at_zero - at_infinity, at_infinity])
    }

    /// Equality weights of the quad corners `[00, 10, 01, 11]` at `(r0, r1)`.
    fn quad_weights(&self) -> [E; 4] {
        let (r0, r1) = (self.challenges[0], self.challenges[1]);
        let (s0, s1) = (E::one() - r0, E::one() - r1);
        [s0 * s1, r0 * s1, s0 * r1, r0 * r1]
    }

    fn lookup_round_terms(
        &self,
        round: usize,
        split_eq: &GruenSplitEq<E>,
        skip_linear: bool,
        out: Option<&mut [E]>,
    ) -> NormRoundTerms<E> {
        let totals = match (round, skip_linear) {
            (3, false) => self.lookup_round::<2, false>(split_eq, out),
            (3, true) => self.lookup_round::<2, true>(split_eq, out),
            (4, false) => self.lookup_round::<4, false>(split_eq, out),
            (4, true) => self.lookup_round::<4, true>(split_eq, out),
            (5, false) => self.lookup_round::<8, false>(split_eq, out),
            (5, true) => self.lookup_round::<8, true>(split_eq, out),
            (6, false) => self.lookup_round::<16, false>(split_eq, out),
            (6, true) => self.lookup_round::<16, true>(split_eq, out),
            _ => unreachable!("compact prefix rounds are capped by MAX_PREFIX_ROUNDS"),
        };
        if skip_linear {
            NormRoundTerms::SkipLinear([totals[0], totals[2]])
        } else {
            NormRoundTerms::Full(totals)
        }
    }

    fn lookup_round<const G: usize, const SKIP_LINEAR: bool>(
        &self,
        split_eq: &GruenSplitEq<E>,
        out: Option<&mut [E]>,
    ) -> [E; 3] {
        let class_count = self.quad_fold.len();
        let group_weights = EqPolynomial::evals(&self.challenges[2..])
            .expect("compact prefix challenge count is bounded");
        debug_assert_eq!(group_weights.len(), G);
        let lookups = group_weights
            .iter()
            .flat_map(|&weight| self.quad_fold.iter().map(move |&fold| weight * fold))
            .collect::<Vec<_>>();
        let (eq_low, eq_high) = split_eq.remaining_eq_tables();
        let pair_count = self.classes.len() / (2 * G);
        let task_pairs = pair_count
            .div_ceil(target_tasks())
            .max(MIN_LOOKUP_TASK_PAIRS);
        let run = |task: usize, out: Option<&mut [E]>| {
            let start = task * task_pairs;
            lookup_pairs::<E, G, SKIP_LINEAR>(
                &lookups,
                class_count,
                &self.classes,
                eq_low,
                eq_high,
                start..(start + task_pairs).min(pair_count),
                out,
            )
        };
        let parts: Vec<[E; 3]> = match out {
            Some(out) => cfg_chunks_mut!(out, 2 * task_pairs)
                .enumerate()
                .map(|(task, chunk)| run(task, Some(chunk)))
                .collect(),
            None => cfg_into_iter!(0..pair_count.div_ceil(task_pairs))
                .map(|task| run(task, None))
                .collect(),
        };
        parts.into_iter().fold([E::zero(); 3], |mut sum, part| {
            for (sum, part) in sum.iter_mut().zip(part) {
                *sum += part;
            }
            sum
        })
    }
}

/// Parallel task count: a few tasks per worker for load balance.
#[inline]
fn target_tasks() -> usize {
    #[cfg(feature = "parallel")]
    {
        4 * rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

impl<E: Field + Ring + Unreduced + Fold> CompactQuotientPrefix<E> {
    /// Bind the relation masses and record `r`.
    pub(super) fn bind(&mut self, r: E) {
        fold_evals_in_place(&mut self.alpha_mass, r);
        for mass in &mut self.source_mass {
            if !mass.is_empty() {
                fold_evals_in_place(mass, r);
            }
        }
        self.challenges.push(r);
        if self.challenges.len() == 2 {
            let half = (self.b / 2) as i64;
            let quad_weights = self.quad_weights();
            let digit_terms: [Vec<E>; 4] = std::array::from_fn(|digit| {
                (0..self.b as i64)
                    .map(|value| quad_weights[digit] * E::from_i64(value - half))
                    .collect()
            });
            let mask = self.b - 1;
            self.quad_fold = (0..1usize << (4 * self.digit_bits))
                .map(|class| {
                    (0..4).fold(E::zero(), |sum, digit| {
                        sum + digit_terms[digit][(class >> (self.digit_bits * digit)) & mask]
                    })
                })
                .collect();
        }
    }
}

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    pub(super) fn compact_quotient_prefix(&self) -> Option<&CompactQuotientPrefix<E>> {
        match &self.relation_state {
            RelationRoundState::QuotientFactored {
                prefix: QuotientPrefixState::Compact(prefix),
                ..
            } => Some(prefix),
            _ => None,
        }
    }

    pub(super) fn norm_poly_from_prefix(&self, norm: PrefixNormRound<E>) -> UnivariatePoly<E> {
        match norm {
            PrefixNormRound::Polynomial(poly) => poly,
            PrefixNormRound::Terms(terms) => self.norm_poly_from_terms(terms),
        }
    }

    /// `(combined, range-image)` messages of a compact-prefix round.
    pub(super) fn compact_prefix_round_polys(
        &self,
        prefix: &CompactQuotientPrefix<E>,
        weights: &RelationWeightFactorization<E>,
    ) -> (UnivariatePoly<E>, UnivariatePoly<E>) {
        let norm = prefix.norm_round(&self.split_eq, self.can_skip_norm_linear_coeff());
        let norm_poly = self.norm_poly_from_prefix(norm);
        let relation = prefix.relation_coeffs(weights.common_alpha_factor(), &self.linear_terms);
        (
            self.combine_polys(&norm_poly, &coeffs_to_poly(relation)),
            norm_poly,
        )
    }
}

impl<E: Field + Ring + Unreduced + Fold> RelationRangeImageProver<E> {
    /// Bind `r` in a compact-prefix round. The round before the last prefix
    /// round also materializes the folded witness and caches the last prefix
    /// round's message.
    pub(super) fn ingest_compact_prefix_challenge(&mut self, r: E) {
        let RelationRoundState::QuotientFactored { weights, prefix } = &mut self.relation_state
        else {
            return;
        };
        let QuotientPrefixState::Compact(mut engine) =
            mem::replace(prefix, QuotientPrefixState::Disabled)
        else {
            return;
        };
        fold_evals_in_place(weights.components_mut().0, r);
        self.split_eq.bind(r);
        self.linear_terms.fold_coefficients(r);
        engine.bind(r);
        if self.rounds_completed + 1 < engine.last_round() {
            if let RelationRoundState::QuotientFactored { prefix, .. } = &mut self.relation_state {
                *prefix = QuotientPrefixState::Compact(engine);
            }
            return;
        }

        let WitnessState::CompactPrefix(witness) = &self.witness_state else {
            return;
        };
        let (folded, norm) = engine.materialize(
            witness.view(),
            &self.split_eq,
            self.can_skip_norm_linear_coeff(),
        );
        let relation = match &self.relation_state {
            RelationRoundState::QuotientFactored { weights, .. } => {
                engine.relation_coeffs(weights.common_alpha_factor(), &self.linear_terms)
            }
            RelationRoundState::ReducedDense { .. } | RelationRoundState::LaneProduct(_) => return,
        };
        self.witness_state = WitnessState::FoldedSuffix(folded);
        let norm_poly = self.norm_poly_from_prefix(norm);
        self.cached_round_poly = Some(self.combine_polys(&norm_poly, &coeffs_to_poly(relation)));
        self.prev_norm_poly = Some(norm_poly);
    }
}
