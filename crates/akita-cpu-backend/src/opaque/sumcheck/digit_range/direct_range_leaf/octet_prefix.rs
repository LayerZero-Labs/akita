//! Rounds 0 through 3 over octet classes of the compact table.
//!
//! Rounds 0, 1 and 2 bind the three low bits of the flat index, which address
//! an entry inside an aligned octet. Until round 3, a folded entry depends only
//! on its octet's class: the classes `k` of its eight range images `k(k+1)`,
//! packed first entry lowest. So round 2 sums over at most `2^(8 class_bits)`
//! classes, each weighted by the total round-2 equality weight of its live
//! octets, and rounds 0 and 1 need only the quad-class marginals of those
//! weights. Round 3 pairs adjacent octets. Its linear sum
//! `sum eq(pair) (Q(right) - Q(left))` separates over the classes of the two
//! octets, so the same scan also sums each pair's round-3 equality weight into
//! the classes of its even and odd octets; the round-2 weights are a `tau3`
//! blend of those two tables. The higher round-3 terms read, per class, the
//! folded value and the Taylor factors that depend on it alone, so each pair
//! pays only for the powers of its difference. The round-3 challenge
//! materializes the field table, fused with round 4.
//!
//! Class-zero octets contribute nothing to any of these rounds, since all their
//! folded values are zero, so their weight is not accumulated.

use super::*;
use crate::opaque::sumcheck::{parallel_tasks, sum_partials};

/// Rounds served from the compact table before it is materialized.
pub(super) const OCTET_PREFIX_ROUNDS: usize = 4;
/// Octet pairs per tile of the class-weight histogram.
const HISTOGRAM_TILE_PAIRS: usize = 1 << 11;
/// Octet classes per chunk of the parallel class-table merge.
const HISTOGRAM_MERGE_CLASSES: usize = 1 << 10;
/// Octet classes per round-2 accumulation chunk.
const ROUND2_CLASS_CHUNK: usize = 1 << 12;

/// Bits of a range-image class: one for basis 4, two for basis 8.
#[inline]
fn class_bits(basis: usize) -> usize {
    debug_assert!(matches!(basis, 4 | 8));
    basis / 4
}

/// Reads octet classes straight from the packed digit bytes.
///
/// An octet of `w`-bit digits fills exactly `w` bytes, first digit lowest, so
/// octet `o` is the little-endian word of bytes `w o .. w (o + 1)`. Widths up
/// to three bits look up each half of that word in a table of packed quad
/// classes; wider digits are classified one at a time.
struct OctetClassReader<'a> {
    encoded: &'a [u8],
    bit_width: usize,
    class_bits: usize,
    quad_classes: [u8; 1 << 12],
}

impl<'a> OctetClassReader<'a> {
    fn new(source: &'a PackedSignedDigits, class_bits: usize) -> Self {
        let bit_width = usize::from(source.bit_width());
        let mut quad_classes = [0u8; 1 << 12];
        if bit_width <= 3 {
            for (quad, slot) in quad_classes
                .iter_mut()
                .enumerate()
                .take(1 << (4 * bit_width))
            {
                *slot = packed_digit_classes(quad as u64, bit_width, class_bits, 4) as u8;
            }
        }
        Self {
            encoded: source.encoded_bytes(),
            bit_width,
            class_bits,
            quad_classes,
        }
    }

    /// Classes of `octets`; octets past the table have class zero.
    fn classes(&self, octets: Range<usize>) -> Vec<u16> {
        match self.bit_width {
            1 => self.classes_of_width::<1>(octets),
            2 => self.classes_of_width::<2>(octets),
            3 => self.classes_of_width::<3>(octets),
            4 => self.classes_of_width::<4>(octets),
            5 => self.classes_of_width::<5>(octets),
            6 => self.classes_of_width::<6>(octets),
            7 => self.classes_of_width::<7>(octets),
            8 => self.classes_of_width::<8>(octets),
            _ => unreachable!("packed signed digits are one to eight bits wide"),
        }
    }

    fn classes_of_width<const W: usize>(&self, octets: Range<usize>) -> Vec<u16> {
        let whole = (self.encoded.len() / W).min(octets.end);
        let mut classes = Vec::with_capacity(octets.len());
        classes.extend(
            self.encoded[W * octets.start.min(whole)..W * whole]
                .chunks_exact(W)
                .map(|bytes| self.word_class::<W>(little_endian_word(bytes))),
        );
        // The encoding stops at the last live digit, so the final octet may be
        // partial and later octets have no bytes at all.
        classes.extend((whole.max(octets.start)..octets.end).map(|octet| {
            let bytes = self.encoded.get(W * octet..).unwrap_or(&[]);
            self.word_class::<W>(little_endian_word(&bytes[..bytes.len().min(W)]))
        }));
        classes
    }

    #[inline(always)]
    fn word_class<const W: usize>(&self, word: u64) -> u16 {
        if W <= 3 {
            let quad_mask = (1u64 << (4 * W)) - 1;
            let low = self.quad_classes[(word & quad_mask) as usize];
            let high = self.quad_classes[((word >> (4 * W)) & quad_mask) as usize];
            u16::from(low) | u16::from(high) << (4 * self.class_bits)
        } else {
            packed_digit_classes(word, W, self.class_bits, 8)
        }
    }
}

/// Pack the classes of the first `count` `bit_width`-bit two's-complement
/// digits of `word`, first digit lowest. Digits `w` and `-1 - w` share a range
/// image, so the class of `w` is `w` for `w >= 0` and `!w` otherwise.
#[inline(always)]
fn packed_digit_classes(word: u64, bit_width: usize, class_bits: usize, count: usize) -> u16 {
    let unused_bits = 64 - bit_width;
    (0..count).fold(0, |packed, digit| {
        let value = (((word >> (digit * bit_width)) << unused_bits) as i64) >> unused_bits;
        let class = (value ^ (value >> 63)) as u16 & ((1 << class_bits) - 1);
        packed | class << (digit * class_bits)
    })
}

#[inline(always)]
fn little_endian_word(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .fold(0, |word, &byte| word << 8 | u64::from(byte))
}

/// Sum the round-3 equality weight `e_first[p & mask] * e_second[p >> bits]`
/// of every live octet pair `p` into the classes of its even (`[0]`) and odd
/// (`[1]`) octets.
///
/// A class table is megabytes wide, so each of a few tasks fills one table
/// over a contiguous run of tiles, and the tables are then summed per class.
fn octet_pair_class_weights<E: Field>(
    reader: &OctetClassReader<'_>,
    live_pairs: usize,
    e_first: &[E],
    e_second: &[E],
) -> Vec<[E; 2]> {
    let num_classes = 1usize << (8 * reader.class_bits);
    let first_mask = e_first.len() - 1;
    let first_bits = e_first.len().trailing_zeros();
    let tiles = live_pairs.div_ceil(HISTOGRAM_TILE_PAIRS);
    let tiles_per_task = tiles
        .div_ceil(parallel_tasks(HISTOGRAM_TASKS_PER_THREAD))
        .max(1);
    let mut tables: Vec<Vec<[E; 2]>> = cfg_into_iter!(0..tiles.div_ceil(tiles_per_task))
        .map(|task| {
            let mut weights = vec![[E::zero(); 2]; num_classes];
            for tile in task * tiles_per_task..((task + 1) * tiles_per_task).min(tiles) {
                let start = tile * HISTOGRAM_TILE_PAIRS;
                let end = (start + HISTOGRAM_TILE_PAIRS).min(live_pairs);
                let classes = reader.classes(2 * start..2 * end);
                for (pair, classes) in (start..end).zip(classes.chunks_exact(2)) {
                    let (even, odd) = (usize::from(classes[0]), usize::from(classes[1]));
                    if even | odd == 0 {
                        continue;
                    }
                    let weight = e_first[pair & first_mask] * e_second[pair >> first_bits];
                    if even != 0 {
                        weights[even][0] += weight;
                    }
                    if odd != 0 {
                        weights[odd][1] += weight;
                    }
                }
            }
            weights
        })
        .collect();
    let Some((totals, rest)) = tables.split_first_mut() else {
        return vec![[E::zero(); 2]; num_classes];
    };
    cfg_chunks_mut!(totals, HISTOGRAM_MERGE_CLASSES)
        .enumerate()
        .for_each(|(chunk, totals)| {
            let first = chunk * HISTOGRAM_MERGE_CLASSES;
            for table in rest.iter() {
                for (total, weight) in totals.iter_mut().zip(&table[first..]) {
                    total[0] += weight[0];
                    total[1] += weight[1];
                }
            }
        });
    tables.swap_remove(0)
}

/// Class-table tasks per worker: enough for load balance across uneven cores,
/// few enough that zeroing and merging the tables stays small beside the scan.
const HISTOGRAM_TASKS_PER_THREAD: usize = 2;

/// Quad-class weights for rounds 0 and 1.
///
/// Octet `o` holds quads `2o` and `2o + 1`, whose equality weights are
/// `(1 - tau2)` and `tau2` times the octet's.
fn quad_class_weights<E: Field>(octet_class_weights: &[E], class_bits: usize, tau2: E) -> Vec<E> {
    let num_quad_classes = 1usize << (4 * class_bits);
    let mut low = vec![E::zero(); num_quad_classes];
    let mut high = vec![E::zero(); num_quad_classes];
    for (high_weight, row) in high
        .iter_mut()
        .zip(octet_class_weights.chunks_exact(num_quad_classes))
    {
        for (low_weight, &weight) in low.iter_mut().zip(row) {
            *low_weight += weight;
            *high_weight += weight;
        }
    }
    low.into_iter()
        .zip(high)
        .map(|(low, high)| low + tau2 * (high - low))
        .collect()
}

/// Folded value of every quad class after rounds 0 and 1.
fn folded_quad_values<E: Field + Ring>(class_bits: usize, r0: E, r1: E) -> Vec<E> {
    let pair_bits = 2 * class_bits;
    let range_image = |class: usize| E::from_u64((class * (class + 1)) as u64);
    let pairs: Vec<E> = (0..1usize << pair_bits)
        .map(|pair| {
            let left = range_image(pair & ((1 << class_bits) - 1));
            left + r0 * (range_image(pair >> class_bits) - left)
        })
        .collect();
    (0..1usize << (2 * pair_bits))
        .map(|quad| {
            let left = pairs[quad & ((1 << pair_bits) - 1)];
            left + r1 * (pairs[quad >> pair_bits] - left)
        })
        .collect()
}

/// Round-3 terms of every octet class after round 2.
fn octet_class_terms<E: Field + Ring>(
    poly: &RangePoly,
    quad_values: &[E],
    r2: E,
) -> Vec<OctetClassTerms<E>> {
    let quad_mask = quad_values.len() - 1;
    let quad_bits = quad_values.len().trailing_zeros();
    cfg_into_iter!(0..quad_values.len() * quad_values.len())
        .map(|octet| {
            let left = quad_values[octet & quad_mask];
            let value = left + r2 * (quad_values[octet >> quad_bits] - left);
            poly.class_terms(value)
        })
        .collect()
}

/// Round-3 linear sum `sum eq(pair) (Q(right) - Q(left))`, gathered per octet
/// class from the pair-class weights.
fn octet_range_difference<E: Field + Ring + Unreduced>(
    terms: &[OctetClassTerms<E>],
    pair_class_weights: &[[E; 2]],
    range_poly: &RangePoly,
) -> E::Product {
    cfg_fold_reduce!(
        0..terms.len(),
        E::Product::zero,
        |mut sum, class| {
            let [even, odd] = pair_class_weights[class];
            if even != odd {
                sum += range_poly
                    .eval(terms[class].value)
                    .mul_unreduced(odd - even);
            }
            sum
        },
        |left, right| left + right
    )
}

impl<E: Field + Ring + Unreduced> OctetPrefix<E> {
    /// Build the octet-class weights and the round-0/1 cache before round 0 binds.
    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::ensure_initial_round_prefix"
    )]
    fn ensure_state(
        &mut self,
        split_eq: &GruenSplitEq<E>,
        basis: usize,
    ) -> (&PackedSignedDigits, &mut DirectRangePrefixState<E>) {
        let digits = &self.digits;
        let tau = &self.tau;
        let state = self.state.get_or_insert_with(|| {
            let class_bits = class_bits(basis);
            let (e_first, e_second) = split_eq
                .remaining_eq_tables_after(3)
                .expect("octet prefix has at least four rounds");
            let octet_pair_class_weights = octet_pair_class_weights(
                &OctetClassReader::new(digits, class_bits),
                digits.len().div_ceil(16),
                e_first,
                e_second,
            );
            let tau3 = tau[3];
            let octet_class_weights: Vec<E> = octet_pair_class_weights
                .iter()
                .map(|&[even, odd]| even + tau3 * (odd - even))
                .collect();
            let quad_class_weights = quad_class_weights(&octet_class_weights, class_bits, tau[2]);
            let cache = build_stage1_prefix_cache(&quad_class_weights, tau, basis)
                .expect("octet prefix has at least two rounds");
            DirectRangePrefixState {
                cache,
                octet_class_weights,
                octet_pair_class_weights,
                first_challenge: None,
                quad_values: Vec::new(),
                octet_terms: Vec::new(),
            }
        });
        (digits, state)
    }
}

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    pub(super) fn compute_octet_prefix_round(
        &self,
        prefix: &mut OctetPrefix<E>,
    ) -> OmittedConstantPoly<E> {
        let (source, state) = prefix.ensure_state(&self.split_eq, self.basis);
        match self.rounds_completed {
            0 => state.cache.reconstruct_round0_eq_poly(),
            1 => state.cache.reconstruct_round1_eq_poly(
                state
                    .first_challenge
                    .expect("round 1 requires the round-0 challenge"),
            ),
            2 => self.compute_octet_class_round(state),
            3 => {
                let terms = state.octet_terms.as_slice();
                let class_bits = class_bits(self.basis);
                let precomputation = &self.range_poly;
                let reader = OctetClassReader::new(source, class_bits);
                let mut sums = self.compute_round_live_prefix(source.len().div_ceil(16), |pairs| {
                    let start = pairs.start;
                    let classes = reader.classes(2 * start..2 * pairs.end);
                    move |pair, weight, sums| {
                        let local = 2 * (pair - start);
                        precomputation.accumulate_octet_pair_terms(
                            sums,
                            &terms[usize::from(classes[local])],
                            &terms[usize::from(classes[local + 1])],
                            weight,
                        );
                    }
                });
                sums[0] = octet_range_difference(
                    terms,
                    &state.octet_pair_class_weights,
                    &self.range_poly,
                );
                self.range_poly
                    .round_poly_from_sums(&sums, LinearSum::RangeDifference)
            }
            _ => unreachable!("octet prefix covers rounds 0 through 3"),
        }
    }

    /// Round 2: every octet is one pair, so sum each octet class's pair
    /// coefficients against its equality weight.
    #[tracing::instrument(skip_all, name = "LowBasisRangeCheckProver::compute_octet_class_round")]
    fn compute_octet_class_round(
        &self,
        prefix: &DirectRangePrefixState<E>,
    ) -> OmittedConstantPoly<E> {
        let quad_values = prefix.quad_values.as_slice();
        let quad_mask = quad_values.len() - 1;
        let quad_bits = quad_values.len().trailing_zeros();
        let precomputation = &self.range_poly;
        let chunk_accumulators: Vec<_> =
            cfg_chunks!(prefix.octet_class_weights, ROUND2_CLASS_CHUNK)
                .enumerate()
                .map(|(chunk, weights)| {
                    let mut accumulator = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
                    for (offset, &weight) in weights.iter().enumerate() {
                        if weight.is_zero() {
                            continue;
                        }
                        let class = chunk * ROUND2_CLASS_CHUNK + offset;
                        let left = quad_values[class & quad_mask];
                        precomputation.accumulate_entry_terms(
                            &mut accumulator,
                            left,
                            quad_values[class >> quad_bits] - left,
                            weight,
                        );
                    }
                    accumulator
                })
                .collect();
        precomputation.round_poly_from_sums(
            &sum_partials(E::Product::zero(), chunk_accumulators),
            LinearSum::Taylor,
        )
    }

    /// Bind an octet-prefix round. The round-3 challenge materializes the
    /// field table and, when another round follows, computes it in the same
    /// pass.
    pub(super) fn ingest_octet_prefix_challenge(
        &mut self,
        prefix: &mut OctetPrefix<E>,
        r: E,
    ) -> Option<Vec<E>> {
        let (source, state) = prefix.ensure_state(&self.split_eq, self.basis);
        self.split_eq.bind(r);
        let class_bits = class_bits(self.basis);
        match self.rounds_completed {
            0 => {
                state.first_challenge = Some(r);
            }
            1 => {
                let r0 = state
                    .first_challenge
                    .expect("round 1 requires the round-0 challenge");
                state.quad_values = folded_quad_values(class_bits, r0, r);
            }
            2 => {
                state.octet_terms = octet_class_terms(&self.range_poly, &state.quad_values, r);
            }
            3 => {
                let terms = state.octet_terms.as_slice();
                let next_live = source.len().div_ceil(8).div_ceil(2);
                let reader = OctetClassReader::new(source, class_bits);
                let folds_for_tile = |entries: Range<usize>| {
                    let start = entries.start;
                    let classes = reader.classes(2 * start..2 * entries.end);
                    move |entry: usize| {
                        let local = 2 * (entry - start);
                        let left = terms[usize::from(classes[local])].value;
                        left + r * (terms[usize::from(classes[local + 1])].value - left)
                    }
                };
                let (range_image, next_round_poly) = if self.rounds_completed + 1 < self.num_vars {
                    let (range_image, poly) =
                        self.fuse_live_prefix_and_compute_round(next_live, folds_for_tile);
                    (range_image, Some(poly))
                } else {
                    (
                        (0..next_live).map(folds_for_tile(0..next_live)).collect(),
                        None,
                    )
                };
                self.cached_round_poly = next_round_poly;
                return Some(range_image);
            }
            _ => unreachable!("octet prefix covers rounds 0 through 3"),
        }
        None
    }
}
