//! Rounds 0 through 3 over octet classes of the compact table.
//!
//! Rounds 0, 1 and 2 bind the three low bits of the flat index, which address
//! an entry inside an aligned octet. Until round 3, a folded entry depends only
//! on its octet's class: the classes `k` of its eight range images `k(k+1)`,
//! packed first entry lowest. So round 2 sums over at most `2^(8 class_bits)`
//! classes, each weighted by the total round-2 equality weight of its live
//! octets, and rounds 0 and 1 need only the quad-class marginals of those
//! weights. Round 3 pairs adjacent octets and reads their folded values from a
//! per-class table. Its challenge materializes the field table, fused with
//! round 4.
//!
//! Class-zero octets contribute nothing to any of these rounds, since all their
//! folded values are zero, so their weight is not accumulated.

use super::*;

/// Rounds served from the compact table before it is materialized.
pub(super) const OCTET_PREFIX_ROUNDS: usize = 4;
/// Octets per tile of the class-weight histogram.
const HISTOGRAM_TILE_OCTETS: usize = 1 << 12;
/// Octet classes per round-2 accumulation chunk.
const ROUND2_CLASS_CHUNK: usize = 1 << 12;

/// Bits of a range-image class: one for basis 4, two for basis 8.
#[inline]
fn class_bits(basis: usize) -> usize {
    debug_assert!(matches!(basis, 4 | 8));
    basis / 4
}

/// Pack eight range-image classes, first entry lowest.
#[inline(always)]
fn pack_octet_class(classes: [u8; 8], class_bits: usize) -> u16 {
    let x = u64::from_le_bytes(classes);
    let packed = if class_bits == 2 {
        let x = (x | (x >> 6)) & 0x000f_000f_000f_000f;
        let x = (x | (x >> 12)) & 0x0000_00ff_0000_00ff;
        (x | (x >> 24)) & 0xffff
    } else {
        let x = (x | (x >> 7)) & 0x0003_0003_0003_0003;
        let x = (x | (x >> 14)) & 0x0000_000f_0000_000f;
        (x | (x >> 28)) & 0xff
    };
    packed as u16
}

/// Classes of `octets` of the compact table; octets past the table have class
/// zero.
fn octet_classes<S: CompactRangeImageSource + ?Sized>(
    source: &S,
    octets: Range<usize>,
    class_bits: usize,
) -> Vec<u16> {
    let mut entry_classes = vec![0u8; 8 * octets.len()];
    source.range_image_classes(8 * octets.start, &mut entry_classes);
    entry_classes
        .chunks_exact(8)
        .map(|classes| {
            pack_octet_class(
                classes.try_into().expect("octet has eight entries"),
                class_bits,
            )
        })
        .collect()
}

/// Sum the round-2 equality weight `e_first[o & mask] * e_second[o >> bits]` of
/// every live octet `o` into its class.
fn octet_class_weights<E: Field, S: CompactRangeImageSource + ?Sized>(
    source: &S,
    e_first: &[E],
    e_second: &[E],
    class_bits: usize,
) -> Vec<E> {
    let num_classes = 1usize << (8 * class_bits);
    let live_octets = source.len().div_ceil(8);
    let first_mask = e_first.len() - 1;
    let first_bits = e_first.len().trailing_zeros();
    cfg_fold_reduce!(
        0..live_octets.div_ceil(HISTOGRAM_TILE_OCTETS),
        || vec![E::zero(); num_classes],
        |mut weights, tile| {
            let start = tile * HISTOGRAM_TILE_OCTETS;
            let end = (start + HISTOGRAM_TILE_OCTETS).min(live_octets);
            for (octet, class) in (start..end).zip(octet_classes(source, start..end, class_bits)) {
                if class != 0 {
                    weights[usize::from(class)] +=
                        e_first[octet & first_mask] * e_second[octet >> first_bits];
                }
            }
            weights
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        }
    )
}

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

/// Folded value of every octet class after round 2.
fn folded_octet_values<E: Field>(quad_values: &[E], r2: E) -> Vec<E> {
    let quad_mask = quad_values.len() - 1;
    let quad_bits = quad_values.len().trailing_zeros();
    (0..quad_values.len() * quad_values.len())
        .map(|octet| {
            let left = quad_values[octet & quad_mask];
            left + r2 * (quad_values[octet >> quad_bits] - left)
        })
        .collect()
}

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    #[inline]
    pub(super) fn using_octet_prefix(&self) -> bool {
        self.rounds_completed < OCTET_PREFIX_ROUNDS && self.prefix_tau.is_some()
    }

    fn compact_range_image(&self) -> &PackedSignedDigits {
        match &self.range_image {
            LowBasisRangeImageStorage::Compact(compact_range_image) => compact_range_image,
            LowBasisRangeImageStorage::Materialized(_) => {
                unreachable!("octet-prefix rounds read the compact table")
            }
        }
    }

    /// Build the octet-class weights and the round-0/1 cache before round 0
    /// binds.
    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::ensure_initial_round_prefix"
    )]
    fn ensure_initial_round_prefix(&mut self) {
        if self.initial_round_prefix.is_some() {
            return;
        }
        debug_assert_eq!(self.rounds_completed, 0);
        let tau0 = self
            .prefix_tau
            .as_deref()
            .expect("octet prefix requested without its equality point");
        let class_bits = class_bits(self.basis);
        let (e_first, e_second) = self
            .split_eq
            .remaining_eq_tables_after(2)
            .expect("octet prefix has at least three rounds");
        let octet_class_weights =
            octet_class_weights(self.compact_range_image(), e_first, e_second, class_bits);
        let quad_class_weights = quad_class_weights(&octet_class_weights, class_bits, tau0[2]);
        let cache = build_stage1_prefix_cache(&quad_class_weights, tau0, self.basis)
            .expect("octet prefix has at least two rounds");
        self.initial_round_prefix = Some(DirectRangePrefixState {
            cache,
            octet_class_weights,
            first_challenge: None,
            class_values: Vec::new(),
        });
    }

    fn octet_prefix(&self) -> &DirectRangePrefixState<E> {
        self.initial_round_prefix
            .as_ref()
            .expect("octet prefix is initialized before round 0 binds")
    }

    pub(super) fn compute_octet_prefix_round(&mut self) -> OmittedConstantPoly<E> {
        self.ensure_initial_round_prefix();
        let prefix = self.octet_prefix();
        match self.rounds_completed {
            0 => prefix.cache.reconstruct_round0_eq_poly(),
            1 => prefix.cache.reconstruct_round1_eq_poly(
                prefix
                    .first_challenge
                    .expect("round 1 requires the round-0 challenge"),
            ),
            2 => self.compute_octet_class_round(),
            3 => {
                let source = self.compact_range_image();
                let values = prefix.class_values.as_slice();
                let class_bits = class_bits(self.basis);
                let live_octets = source.len().div_ceil(8);
                self.compute_round_live_prefix(live_octets.div_ceil(2), |pairs| {
                    let start = pairs.start;
                    let classes = octet_classes(source, 2 * start..2 * pairs.end, class_bits);
                    move |pair| {
                        let local = 2 * (pair - start);
                        (
                            values[usize::from(classes[local])],
                            values[usize::from(classes[local + 1])],
                        )
                    }
                })
            }
            _ => unreachable!("octet prefix covers rounds 0 through 3"),
        }
    }

    /// Round 2: every octet is one pair, so sum each octet class's pair
    /// coefficients against its equality weight.
    #[tracing::instrument(skip_all, name = "LowBasisRangeCheckProver::compute_octet_class_round")]
    fn compute_octet_class_round(&self) -> OmittedConstantPoly<E> {
        let prefix = self.octet_prefix();
        let quad_values = prefix.class_values.as_slice();
        let quad_mask = quad_values.len() - 1;
        let quad_bits = quad_values.len().trailing_zeros();
        let precomputation = &self.polynomial_precomputation;
        let num_coeffs_q = precomputation.num_coefficients();
        let chunk_accumulators = cfg_chunks!(prefix.octet_class_weights, ROUND2_CLASS_CHUNK)
            .enumerate()
            .map(|(chunk, weights)| {
                let mut accumulator = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
                let mut entry = [E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
                for (offset, &weight) in weights.iter().enumerate() {
                    if weight.is_zero() {
                        continue;
                    }
                    let class = chunk * ROUND2_CLASS_CHUNK + offset;
                    let left = quad_values[class & quad_mask];
                    compute_entry_coefficients(
                        &mut entry,
                        precomputation,
                        left,
                        quad_values[class >> quad_bits] - left,
                    );
                    accumulate_dense_entry_coeffs(
                        &mut accumulator[..num_coeffs_q],
                        &entry[..num_coeffs_q],
                        weight,
                    );
                }
                accumulator
            })
            .collect();
        self.live_prefix_round_poly(chunk_accumulators)
    }

    /// Bind an octet-prefix round. The round-3 challenge materializes the
    /// field table and, when another round follows, computes it in the same
    /// pass.
    pub(super) fn ingest_octet_prefix_challenge(&mut self, r: E) {
        self.ensure_initial_round_prefix();
        self.split_eq.bind(r);
        let class_bits = class_bits(self.basis);
        match self.rounds_completed {
            0 => {
                let prefix = self.initial_round_prefix.as_mut().expect("octet prefix");
                prefix.first_challenge = Some(r);
            }
            1 => {
                let prefix = self.initial_round_prefix.as_mut().expect("octet prefix");
                let r0 = prefix
                    .first_challenge
                    .expect("round 1 requires the round-0 challenge");
                prefix.class_values = folded_quad_values(class_bits, r0, r);
            }
            2 => {
                let prefix = self.initial_round_prefix.as_mut().expect("octet prefix");
                prefix.class_values = folded_octet_values(&prefix.class_values, r);
            }
            3 => {
                let source = self.compact_range_image();
                let values = self.octet_prefix().class_values.as_slice();
                let next_live = source.len().div_ceil(8).div_ceil(2);
                let folds_for_tile = |entries: Range<usize>| {
                    let start = entries.start;
                    let classes = octet_classes(source, 2 * start..2 * entries.end, class_bits);
                    move |entry: usize| {
                        let local = 2 * (entry - start);
                        let left = values[usize::from(classes[local])];
                        left + r * (values[usize::from(classes[local + 1])] - left)
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
                self.range_image = LowBasisRangeImageStorage::Materialized(range_image);
                self.cached_round_poly = next_round_poly;
                self.initial_round_prefix = None;
            }
            _ => unreachable!("octet prefix covers rounds 0 through 3"),
        }
    }
}
