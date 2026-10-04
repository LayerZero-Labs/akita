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
use std::mem::size_of;

/// Rounds served from the compact table before it is materialized.
pub(super) const OCTET_PREFIX_ROUNDS: usize = 4;
/// Octet pairs per tile of the class-weight histogram.
const HISTOGRAM_TILE_PAIRS: usize = 1 << 11;
/// Octet classes per chunk of the parallel class-table merge.
const HISTOGRAM_MERGE_CLASSES: usize = 1 << 10;
/// Octet classes per round-2 accumulation chunk.
const ROUND2_CLASS_CHUNK: usize = 1 << 12;
/// Maximum aggregate payload of per-task class histograms.
const HISTOGRAM_BUFFER_BUDGET_BYTES: usize = 64 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HistogramTaskPlan {
    task_count: usize,
}

impl HistogramTaskPlan {
    fn tile_range(self, task: usize, tiles: usize) -> Range<usize> {
        let tiles_per_task = tiles / self.task_count;
        let extra_tiles = tiles % self.task_count;
        let start = task * tiles_per_task + task.min(extra_tiles);
        let task_tiles = tiles_per_task + usize::from(task < extra_tiles);
        start..start + task_tiles
    }
}

/// Choose enough histogram tasks for load balance without letting the number
/// of Rayon workers multiply the aggregate class-table allocation past the
/// fixed byte budget.
fn histogram_task_plan<E>(tiles: usize, num_classes: usize, workers: usize) -> HistogramTaskPlan {
    let table_bytes = num_classes * 2 * size_of::<E>();
    let max_tables = (HISTOGRAM_BUFFER_BUDGET_BYTES / table_bytes).max(1);
    let requested_tasks = workers.saturating_mul(HISTOGRAM_TASKS_PER_THREAD).max(1);
    let task_count = tiles.min(requested_tasks).min(max_tables);
    HistogramTaskPlan { task_count }
}

/// Bits of a range-image class: one for basis 4, two for basis 8.
#[inline]
fn class_bits(basis: usize) -> usize {
    debug_assert!(matches!(basis, 4 | 8));
    basis / 4
}

/// Width selected once at the fallible prefix boundary.
#[derive(Clone, Copy)]
pub(super) enum DigitWidth {
    W1,
    W2,
    W3,
    W4,
    W5,
    W6,
    W7,
    W8,
}

impl DigitWidth {
    fn new(width: u8) -> Result<Self, AkitaError> {
        match width {
            1 => Ok(Self::W1),
            2 => Ok(Self::W2),
            3 => Ok(Self::W3),
            4 => Ok(Self::W4),
            5 => Ok(Self::W5),
            6 => Ok(Self::W6),
            7 => Ok(Self::W7),
            8 => Ok(Self::W8),
            _ => Err(AkitaError::Internal(format!(
                "octet digit width must be one to eight; got {width}"
            ))),
        }
    }
}

/// Reads octet classes straight from the packed digit bytes.
///
/// An octet of `w`-bit digits fills exactly `w` bytes, first digit lowest, so
/// octet `o` is the little-endian word of bytes `w o .. w (o + 1)`. Widths up
/// to three bits look up each half of that word in a table of packed quad
/// classes; wider digits are classified one at a time.
struct OctetClassReader<'a> {
    encoded: &'a [u8],
    width: DigitWidth,
    class_bits: usize,
    quad_classes: [u8; 1 << 12],
}

impl<'a> OctetClassReader<'a> {
    fn new(source: &'a PackedSignedDigits, class_bits: usize, width: DigitWidth) -> Self {
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
            width,
            class_bits,
            quad_classes,
        }
    }

    /// Classes of `octets`; octets past the table have class zero.
    fn classes(&self, octets: Range<usize>) -> Vec<u16> {
        match self.width {
            DigitWidth::W1 => self.classes_of_width::<1>(octets),
            DigitWidth::W2 => self.classes_of_width::<2>(octets),
            DigitWidth::W3 => self.classes_of_width::<3>(octets),
            DigitWidth::W4 => self.classes_of_width::<4>(octets),
            DigitWidth::W5 => self.classes_of_width::<5>(octets),
            DigitWidth::W6 => self.classes_of_width::<6>(octets),
            DigitWidth::W7 => self.classes_of_width::<7>(octets),
            DigitWidth::W8 => self.classes_of_width::<8>(octets),
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
    let task_plan = histogram_task_plan::<E>(tiles, num_classes, parallel_tasks(1));
    let mut tables: Vec<Vec<[E; 2]>> = cfg_into_iter!(0..task_plan.task_count)
        .map(|task| {
            let mut weights = vec![[E::zero(); 2]; num_classes];
            for tile in task_plan.tile_range(task, tiles) {
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
    /// Build the initial histograms at the fallible prover boundary.
    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::ensure_initial_round_prefix"
    )]
    pub(super) fn new(
        digits: PackedSignedDigits,
        tau: &[E],
        split_eq: &GruenSplitEq<E>,
        basis: usize,
        poly: RangePoly,
    ) -> Result<Self, AkitaError> {
        let width = DigitWidth::new(digits.bit_width())?;
        let class_bits = class_bits(basis);
        let (e_first, e_second) = split_eq.remaining_eq_tables_after(3).ok_or_else(|| {
            AkitaError::Internal("octet prefix has fewer than four rounds".into())
        })?;
        let octet_pair_class_weights = octet_pair_class_weights(
            &OctetClassReader::new(&digits, class_bits, width),
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
        let cache =
            build_stage1_prefix_cache(&quad_class_weights, tau, RangePrefixBasis::new(poly)?)
                .ok_or_else(|| {
                    AkitaError::Internal(
                        "octet prefix interpolation grid or equality point is invalid".into(),
                    )
                })?;
        Ok(Self {
            digits,
            width,
            state: DirectRangePrefixState::Round0 {
                cache,
                weights: PrefixWeights {
                    octet_class_weights,
                    octet_pair_class_weights,
                },
            },
        })
    }
}

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    pub(super) fn compute_octet_prefix_round(
        &self,
        prefix: &OctetPrefix<E>,
    ) -> OmittedConstantPoly<E> {
        let source = &prefix.digits;
        match &prefix.state {
            DirectRangePrefixState::Round0 { cache, .. } => cache.reconstruct_round0_eq_poly(),
            DirectRangePrefixState::Round1 {
                cache,
                first_challenge,
                ..
            } => cache.reconstruct_round1_eq_poly(*first_challenge),
            DirectRangePrefixState::Round2 {
                quad_values,
                weights,
            } => self.compute_octet_class_round(quad_values, &weights.octet_class_weights),
            DirectRangePrefixState::Round3 {
                octet_terms,
                octet_pair_class_weights,
            } => {
                let terms = octet_terms.as_slice();
                let class_bits = class_bits(self.basis);
                let precomputation = &self.range_poly;
                let reader = OctetClassReader::new(source, class_bits, prefix.width);
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
                sums[0] = octet_range_difference(terms, octet_pair_class_weights, &self.range_poly);
                self.range_poly
                    .round_poly_from_sums(&sums, LinearSum::RangeDifference)
            }
        }
    }

    /// Round 2: every octet is one pair, so sum each octet class's pair
    /// coefficients against its equality weight.
    #[tracing::instrument(skip_all, name = "LowBasisRangeCheckProver::compute_octet_class_round")]
    fn compute_octet_class_round(
        &self,
        quad_values: &[E],
        octet_class_weights: &[E],
    ) -> OmittedConstantPoly<E> {
        let quad_mask = quad_values.len() - 1;
        let quad_bits = quad_values.len().trailing_zeros();
        let precomputation = &self.range_poly;
        let chunk_accumulators: Vec<_> = cfg_chunks!(octet_class_weights, ROUND2_CLASS_CHUNK)
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
        prefix: OctetPrefix<E>,
        r: E,
    ) -> LowBasisRangeImageStorage<E> {
        let OctetPrefix {
            digits: source,
            width,
            state,
        } = prefix;
        self.split_eq.bind(r);
        let class_bits = class_bits(self.basis);
        let state = match state {
            DirectRangePrefixState::Round0 { cache, weights } => DirectRangePrefixState::Round1 {
                cache,
                weights,
                first_challenge: r,
            },
            DirectRangePrefixState::Round1 {
                first_challenge,
                weights,
                ..
            } => DirectRangePrefixState::Round2 {
                quad_values: folded_quad_values(class_bits, first_challenge, r),
                weights,
            },
            DirectRangePrefixState::Round2 {
                quad_values,
                weights,
            } => DirectRangePrefixState::Round3 {
                octet_terms: octet_class_terms(&self.range_poly, &quad_values, r),
                octet_pair_class_weights: weights.octet_pair_class_weights,
            },
            DirectRangePrefixState::Round3 { octet_terms, .. } => {
                let terms = octet_terms.as_slice();
                let next_live = source.len().div_ceil(8).div_ceil(2);
                let reader = OctetClassReader::new(&source, class_bits, width);
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
                return LowBasisRangeImageStorage::Materialized(range_image);
            }
        };
        LowBasisRangeImageStorage::OctetPrefix(OctetPrefix {
            digits: source,
            width,
            state,
        })
    }
}

#[cfg(test)]
mod histogram_tests {
    use super::*;
    use jolt_field::{Prime128Offset275, Prime32Offset99};

    #[test]
    fn prefix_constructor_rejects_short_equality_point() {
        let tau = [Prime128Offset275::from_u64(7); 3];
        let split_eq = GruenSplitEq::new(&tau).unwrap();
        assert!(matches!(
            OctetPrefix::new(
                PackedSignedDigits::from_i8_digits_auto(vec![1; 8]),
                &tau, &split_eq, 4, RangePoly::Quadratic,
            ), Err(AkitaError::Internal(message))
                if message == "octet prefix has fewer than four rounds"
        ));
    }

    #[test]
    fn prefix_width_rejects_values_outside_packed_digit_widths() {
        for width in [0, 9, u8::MAX] {
            assert!(matches!(
                DigitWidth::new(width), Err(AkitaError::Internal(message))
                    if message == format!("octet digit width must be one to eight; got {width}")
            ));
        }
    }

    #[test]
    fn histogram_task_plan_caps_aggregate_tables_for_worker_counts() {
        let num_classes = 1 << 16;
        let tiles = 256;
        let table_bytes = num_classes * 2 * size_of::<Prime128Offset275>();
        let max_tables = (HISTOGRAM_BUFFER_BUDGET_BYTES / table_bytes).max(1);
        let workers_and_expected = [
            (1, 2.min(max_tables)),
            (16, 32.min(max_tables)),
            (64, 128.min(max_tables)),
        ];

        for (workers, expected_tasks) in workers_and_expected {
            let plan = histogram_task_plan::<Prime128Offset275>(tiles, num_classes, workers);
            assert_eq!(plan.task_count, expected_tasks, "workers={workers}");
            assert!(plan.task_count * table_bytes <= HISTOGRAM_BUFFER_BUDGET_BYTES);
        }

        let plan_16 = histogram_task_plan::<Prime128Offset275>(tiles, num_classes, 16);
        let plan_64 = histogram_task_plan::<Prime128Offset275>(tiles, num_classes, 64);
        assert_eq!(plan_16.task_count, plan_64.task_count);
    }

    fn assert_pair_class_weights_match_reference(digit_pattern: impl Fn(usize) -> i8) {
        type F = Prime32Offset99;

        let live_pairs = 3 * HISTOGRAM_TILE_PAIRS + 17;
        let digits: Vec<i8> = (0..live_pairs * 16).map(digit_pattern).collect();
        let packed = PackedSignedDigits::from_i8_digits_auto(digits);
        let reader =
            OctetClassReader::new(&packed, 1, DigitWidth::new(packed.bit_width()).unwrap());
        let e_first: Vec<F> = (0..32).map(|i| F::from_u64(i as u64 + 3)).collect();
        let e_second_len = live_pairs.div_ceil(e_first.len()).next_power_of_two();
        let e_second: Vec<F> = (0..e_second_len)
            .map(|i| F::from_u64(i as u64 + 11))
            .collect();
        let num_classes = 1 << 8;
        let actual = octet_pair_class_weights(&reader, live_pairs, &e_first, &e_second);

        let mut expected = vec![[F::zero(); 2]; num_classes];
        let classes = reader.classes(0..2 * live_pairs);
        for (pair, classes) in classes.chunks_exact(2).enumerate() {
            let even = usize::from(classes[0]);
            let odd = usize::from(classes[1]);
            if even | odd == 0 {
                continue;
            }
            let weight = e_first[pair & (e_first.len() - 1)]
                * e_second[pair >> e_first.len().trailing_zeros()];
            if even != 0 {
                expected[even][0] += weight;
            }
            if odd != 0 {
                expected[odd][1] += weight;
            }
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn octet_pair_class_weights_match_low_diversity_reference() {
        assert_pair_class_weights_match_reference(|_| 1);
    }

    #[test]
    fn octet_pair_class_weights_match_high_diversity_reference() {
        assert_pair_class_weights_match_reference(|digit| {
            let octet = digit / 8;
            ((octet >> (digit % 8)) & 1) as i8
        });
    }
}
