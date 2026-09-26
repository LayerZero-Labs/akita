//! Limb-split exact signed-i16 matrix products.
//!
//! A centered public entry `A` with `|A| < q/2` is written as
//! `A = sum_l 2^(b*l) * A_l` with balanced limbs `|A_l| <= 2^(b-1)`. Each limb
//! row accumulates exactly under two CRT primes, so one exact product over `R`
//! rows costs two transforms per column plus `2*L*R` pointwise dots, instead of
//! one transform and `R` dots per field-sized CRT prime. The field result is
//! `sum_l 2^(b*l) * y_l`, where `y_l` is the exact integer product of limb `l`.

use super::exact::{ifma52_cache_enabled, I32_TRANSFORM_DOTS, IFMA52_TRANSFORM_DOTS};
use super::*;
use jolt_field::{cfg_chunks, cfg_into_iter};

/// CRT primes per limb accumulation.
const LIMB_PRIMES: usize = 2;
/// Largest limb count considered by the planner.
const MAX_LIMBS: usize = 8;
/// Matrix columns split and transformed per parallel preparation task.
const PREPARE_TILE_COLUMNS: usize = 16;

/// Balanced base-`2^bits` split of centered field coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LimbSplit {
    count: usize,
    bits: u32,
}

impl LimbSplit {
    /// Smallest split whose limbs cover `modulus_bits` and whose accumulation
    /// fits `capacity` at this width and bound.
    ///
    /// Covering requires `bits * count >= modulus_bits + 1`: after `count - 1`
    /// balanced steps the remaining top value of a centered coefficient
    /// `|A| < 2^(modulus_bits - 1)` is then at most `2^(bits - 1)`.
    fn fitting(
        modulus_bits: u32,
        capacity: &CrtCapacity,
        ring_dimension: usize,
        width: usize,
        rhs_abs_bound: u64,
    ) -> Option<Self> {
        (2..=MAX_LIMBS).find_map(|count| {
            let bits = (modulus_bits + 1).div_ceil(count as u32);
            (bits < 127
                && capacity.supports_modulus(width, ring_dimension, 1u128 << bits, rhs_abs_bound))
            .then_some(Self { count, bits })
        })
    }

    /// Move the balanced low limb of every coefficient of `rest` into `low`,
    /// leaving the remaining high part in `rest`.
    ///
    /// After `count - 1` steps from a centered coefficient, `rest` holds the top
    /// limb.
    fn take_low_limb<const D: usize>(self, rest: &mut [i128; D], low: &mut [i128; D]) {
        let half = 1i128 << (self.bits - 1);
        let mask = (1i128 << self.bits) - 1;
        for (value, low) in rest.iter_mut().zip(low.iter_mut()) {
            let bits = *value & mask;
            let carry = i128::from(bits >= half);
            *low = bits - (carry << self.bits);
            *value = (*value >> self.bits) + carry;
        }
    }

    /// Limb `index` of every centered coefficient.
    fn limb<const D: usize>(self, mut rest: [i128; D], index: usize) -> [i128; D] {
        let mut low = [0; D];
        for _ in 0..index {
            self.take_low_limb(&mut rest, &mut low);
        }
        if index + 1 == self.count {
            return rest;
        }
        self.take_low_limb(&mut rest, &mut low);
        low
    }

    fn supports(
        self,
        width: usize,
        ring_dimension: usize,
        capacity: &CrtCapacity,
        bound: u64,
    ) -> bool {
        capacity.supports_modulus(width, ring_dimension, 1u128 << self.bits, bound)
    }
}

pub(super) enum LimbParams<const D: usize> {
    I32(Box<CrtNttParamSet<i32, LIMB_PRIMES, D>>),
    Ifma52(Box<Ifma52Params<LIMB_PRIMES, D>>),
}

pub(super) struct LimbPlan<const D: usize> {
    params: LimbParams<D>,
    candidate: LimbCandidate,
    width: usize,
    tier: ProtocolRingDispatchTierId,
}

impl<const D: usize> LimbPlan<D> {
    pub(super) fn cost(&self, rows: usize) -> usize {
        self.candidate.cost(rows)
    }
}

/// A limb split that fits, before its transform tables are built.
#[derive(Clone, Copy, Debug)]
pub(super) struct LimbCandidate {
    split: LimbSplit,
    ifma: bool,
}

impl LimbCandidate {
    /// Smallest limb split of the field whose accumulation fits two limb
    /// primes at this width and bound, or `None` when none does.
    pub(super) fn fitting<F: Field + CanonicalEncoding, const D: usize>(
        width: usize,
        rhs_abs_bound: u64,
    ) -> Result<Option<Self>, AkitaError> {
        let modulus_bits = u128::BITS - field_modulus::<F>()?.leading_zeros();
        let ifma = ifma52_cache_enabled::<D>();
        let capacity = if ifma {
            CrtCapacity::from_prime_moduli([IFMA52_PRIMES[0], IFMA52_PRIMES[1]].map(u128::from))
        } else if D <= Q32_MAX_RING_D {
            CrtCapacity::from_prime_moduli(Q32_PRIMES.map(|prime| prime.p as u128))
        } else {
            return Ok(None);
        };
        Ok(
            LimbSplit::fitting(modulus_bits, &capacity, D, width, rhs_abs_bound)
                .map(|split| Self { split, ifma }),
        )
    }

    /// Per-column cost over `rows` output rows, in half prime-row dots.
    pub(super) fn cost(self, rows: usize) -> usize {
        let transform = if self.ifma {
            IFMA52_TRANSFORM_DOTS
        } else {
            I32_TRANSFORM_DOTS
        };
        2 * LIMB_PRIMES * (transform + rows * self.split.count)
    }

    /// Build the limb primes' transform tables.
    pub(super) fn plan<F: Field + CanonicalEncoding, const D: usize>(
        self,
        width: usize,
    ) -> Result<LimbPlan<D>, AkitaError> {
        let params = if self.ifma {
            LimbParams::Ifma52(Box::new(Ifma52Params::new([
                IFMA52_PRIMES[0],
                IFMA52_PRIMES[1],
            ])?))
        } else {
            LimbParams::I32(Box::new(CrtNttParamSet::new(Q32_PRIMES)))
        };
        Ok(LimbPlan {
            params,
            candidate: self,
            width,
            tier: protocol_dispatch_tier::<F>(),
        })
    }
}

#[derive(Debug)]
enum LimbResidues<const D: usize> {
    I32 {
        neg: Vec<CyclotomicCrtNtt<i32, LIMB_PRIMES, D>>,
        params: CrtNttParamSet<i32, LIMB_PRIMES, D>,
    },
    Ifma52(Ifma52NttMatrix<LIMB_PRIMES, D>),
}

/// Prepared limb rows `(row * L + limb) * width + column` of an exact matrix.
#[derive(Debug)]
pub(super) struct PreparedLimbMatrix<const D: usize> {
    residues: LimbResidues<D>,
    split: LimbSplit,
    width: usize,
    tier: ProtocolRingDispatchTierId,
}

impl<const D: usize> PreparedLimbMatrix<D> {
    pub(super) fn prepare<F: Field + CanonicalEncoding>(
        matrix: RingMatrixView<'_, F, D>,
        plan: LimbPlan<D>,
    ) -> Self {
        let LimbPlan {
            params,
            candidate: LimbCandidate { split, .. },
            width,
            tier,
        } = plan;
        let entries = matrix.as_slice();
        let rows = entries.len() / width;
        let limb_entries = rows * split.count * width;
        let residues = match params {
            LimbParams::I32(params) => {
                let mut neg = vec![CyclotomicCrtNtt::zero(); limb_entries];
                for (row_entries, row_limbs) in entries
                    .chunks_exact(width)
                    .zip(neg.chunks_exact_mut(split.count * width))
                {
                    // Tile `t` owns columns `t * T..` of every limb row of this row.
                    let mut tiles: Vec<Vec<&mut [_]>> = (0..width.div_ceil(PREPARE_TILE_COLUMNS))
                        .map(|_| Vec::with_capacity(split.count))
                        .collect();
                    for limb_row in row_limbs.chunks_exact_mut(width) {
                        for (tile, columns) in tiles
                            .iter_mut()
                            .zip(limb_row.chunks_mut(PREPARE_TILE_COLUMNS))
                        {
                            tile.push(columns);
                        }
                    }
                    cfg_into_iter!(tiles)
                        .zip(cfg_chunks!(row_entries, PREPARE_TILE_COLUMNS))
                        .for_each(|(mut tile, columns)| {
                            let mut low = [0; D];
                            for (column, entry) in columns.iter().enumerate() {
                                let mut rest = entry.centered_coefficients_i128();
                                for (limb, limb_row) in tile.iter_mut().enumerate() {
                                    let value = if limb + 1 < split.count {
                                        split.take_low_limb(&mut rest, &mut low);
                                        &low
                                    } else {
                                        &rest
                                    };
                                    limb_row[column] = CyclotomicCrtNtt::from_centered_coefficients(
                                        value, &params,
                                    );
                                }
                            }
                        });
                }
                LimbResidues::I32 {
                    neg,
                    params: *params,
                }
            }
            LimbParams::Ifma52(params) => LimbResidues::Ifma52(Ifma52NttMatrix::prepare_centered(
                limb_entries,
                |index| {
                    let (limb_row, column) = (index / width, index % width);
                    let entry = &entries[(limb_row / split.count) * width + column];
                    split.limb(entry.centered_coefficients_i128(), limb_row % split.count)
                },
                &params,
            )),
        };
        Self {
            residues,
            split,
            width,
            tier,
        }
    }

    fn len(&self) -> usize {
        match &self.residues {
            LimbResidues::I32 { neg, .. } => neg.len(),
            LimbResidues::Ifma52(neg) => neg.len(),
        }
    }

    pub(super) fn validate(&self) -> Result<(), AkitaError> {
        let row_len = self.split.count * self.width;
        let tails = match &self.residues {
            LimbResidues::I32 { .. } => false,
            LimbResidues::Ifma52(neg) => {
                neg.has_tail(I16_TAIL_PRIME) || neg.has_tail(q128_primes()[0])
            }
        };
        if self.width == 0 || self.len() == 0 || !self.len().is_multiple_of(row_len) || tails {
            return Err(AkitaError::InvalidSetup(
                "prepared limb-split NTT cache is inconsistent".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn cache_bytes(&self) -> usize {
        match &self.residues {
            LimbResidues::I32 { neg, .. } => {
                neg.len() * D * LIMB_PRIMES * core::mem::size_of::<i32>()
            }
            LimbResidues::Ifma52(neg) => neg.cache_bytes(),
        }
    }

    pub(super) const fn uses_ifma52(&self) -> bool {
        matches!(self.residues, LimbResidues::Ifma52(_))
    }

    pub(super) const fn tier(&self) -> ProtocolRingDispatchTierId {
        self.tier
    }

    pub(super) fn mat_vec_i16<F: Field + CanonicalEncoding>(
        &self,
        log_basis: u32,
        num_rows: usize,
        rhs: &[[i16; D]],
    ) -> Result<Vec<akita_algebra::CyclotomicRing<F, D>>, AkitaError> {
        let rhs_abs_bound = validate_i16_rhs(log_basis, rhs)?;
        let capacity = match &self.residues {
            LimbResidues::I32 { params, .. } => params.crt_capacity(),
            LimbResidues::Ifma52(neg) => neg.crt_capacity(),
        };
        if rhs.len() != self.width || !self.split.supports(self.width, D, &capacity, rhs_abs_bound)
        {
            return Err(AkitaError::InvalidSetup(
                "signed-i16 matvec does not match the prepared limb-split shape".into(),
            ));
        }
        let limb_rows = num_rows
            .checked_mul(self.split.count)
            .filter(|rows| {
                rows.checked_mul(self.width)
                    .is_some_and(|len| len <= self.len())
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup("prepared limb-split NTT prefix is undersized".into())
            })?;
        let limb_products = match &self.residues {
            LimbResidues::I32 { neg, params } => {
                CyclotomicCrtNtt::mat_vec_i16::<F>(neg, limb_rows, self.width, rhs, params)?
            }
            LimbResidues::Ifma52(neg) => neg.mat_vec_i16::<F>(limb_rows, rhs)?,
        };
        let weights: Vec<F> = (1..self.split.count)
            .map(|limb| F::pow2(self.split.bits as usize * limb))
            .collect();
        Ok(limb_products
            .chunks_exact(self.split.count)
            .filter_map(<[_]>::split_first)
            .map(|(low, high)| {
                let mut row = *low;
                for (limb, weight) in high.iter().zip(&weights) {
                    limb.scale_accumulate_into(&mut row, *weight);
                }
                row
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::exact::ExactCachePlan;
    use super::*;
    use crate::FlatMatrix;
    use akita_algebra::CyclotomicRing;
    use jolt_field::{Prime128Offset275, Prime32Offset99, Prime64Offset59};

    fn extreme_matrix<F: Field + CanonicalEncoding, const D: usize>(
        len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let half = F::from_u128(field_modulus::<F>().expect("modulus") / 2);
        (0..len)
            .map(|entry| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                    let offset = F::from_u64((entry * 257 + coefficient * 17) as u64 % 97);
                    match (entry + coefficient) % 3 {
                        0 => half - offset,
                        1 => -(half - offset),
                        _ => offset - F::from_u64(48),
                    }
                }))
            })
            .collect()
    }

    fn alternating_rhs<const D: usize>(width: usize, bound: i16) -> Vec<[i16; D]> {
        (0..width)
            .map(|column| {
                std::array::from_fn(|coefficient| match (column + coefficient) % 3 {
                    0 => bound,
                    1 => -bound,
                    _ => (column * 31 + coefficient) as i16 % bound,
                })
            })
            .collect()
    }

    #[test]
    fn balanced_limbs_recompose_centered_extremes() {
        for (modulus_bits, count) in [(32, 2), (64, 3), (128, 2), (128, 5), (128, 8)] {
            let bits = (modulus_bits + 1u32).div_ceil(count as u32);
            let split = LimbSplit { count, bits };
            let max = i128::MAX >> (i128::BITS - modulus_bits);
            let values: [i128; 8] = [0, 1, -1, max, -max, max / 3, -max / 7, max >> 9];
            let limbs: Vec<[i128; 8]> = (0..count).map(|l| split.limb(values, l)).collect();
            for (index, value) in values.iter().enumerate() {
                let mut recomposed = 0i128;
                for limb in limbs.iter().rev() {
                    assert!(limb[index].unsigned_abs() <= 1u128 << (bits - 1));
                    // The exact sum fits i128; an intermediate Horner step may not.
                    recomposed = recomposed.wrapping_shl(bits).wrapping_add(limb[index]);
                }
                assert_eq!(recomposed, *value, "bits={bits} count={count}");
            }
        }
    }

    fn assert_limb_cache_matches_base<F: Field + CanonicalEncoding, const D: usize>(
        rows: usize,
        width: usize,
        log_basis: u32,
    ) -> bool {
        let bound = 1u64 << (log_basis - 1);
        let matrix = extreme_matrix::<F, D>(rows * width);
        let flat = FlatMatrix::from_ring_slice(&matrix);
        let view = || flat.ring_view::<D>(rows, width).expect("matrix view");
        let mode = NttCacheMode::ExactNegacyclic {
            width,
            rhs_abs_bound: bound,
        };
        let selected = prepare_ntt_cache(view(), mode).expect("selected cache");
        let base_plan =
            base_exact_cache_plan::<F, D>(select_crt_ntt_params::<F, D>().unwrap(), width, bound)
                .expect("base plan")
                .expect("base capacity");
        let base = prepare_exact_ntt_cache(view(), None, base_plan).expect("base cache");
        assert!(!base.uses_limb_split());
        let rhs = alternating_rhs::<D>(width, (bound - 1) as i16);
        for num_rows in 1..=rows {
            assert_eq!(
                selected
                    .mat_vec_i16::<F>(log_basis, num_rows, &rhs)
                    .unwrap(),
                base.mat_vec_i16::<F>(log_basis, num_rows, &rhs).unwrap(),
            );
        }
        selected.uses_limb_split()
    }

    #[test]
    fn limb_split_matches_base_products() {
        const D: usize = 64;
        let mut selected = Vec::new();
        for rows in [1, 2, 7] {
            for (width, log_basis) in [(3, 16), (48, 11)] {
                selected.push(assert_limb_cache_matches_base::<Prime128Offset275, D>(
                    rows, width, log_basis,
                ));
                selected.push(assert_limb_cache_matches_base::<Prime64Offset59, D>(
                    rows, width, log_basis,
                ));
                selected.push(assert_limb_cache_matches_base::<Prime32Offset99, D>(
                    rows, width, log_basis,
                ));
            }
        }
        assert!(selected.iter().any(|limbs| *limbs));
        assert!(selected.iter().any(|limbs| !*limbs));
    }

    #[test]
    fn limb_split_is_planned_when_the_base_capacity_is_exceeded() {
        const D: usize = 2048;
        type F = Prime64Offset59;
        let (width, bound) = (16_384, 1 << 15);
        let selected = || select_crt_ntt_params::<F, D>().expect("protocol parameters");
        assert!(base_exact_cache_plan::<F, D>(selected(), width, bound)
            .expect("base plan")
            .is_none());
        let plan = prover_exact_cache_plan::<F, D>(selected(), width, bound, width).expect("plan");
        assert!(matches!(plan, ExactCachePlan::Limbs(_)));
        // A partial row cannot be limb split, and the base cannot hold it.
        assert!(prover_exact_cache_plan::<F, D>(selected(), width, bound, width + 1).is_err());
    }

    #[test]
    fn limb_split_rejects_mismatched_shapes() {
        const D: usize = 64;
        let (rows, width) = (1, 512);
        let matrix = extreme_matrix::<Prime128Offset275, D>(rows * width);
        let flat = FlatMatrix::from_ring_slice(&matrix);
        let cache = prepare_ntt_cache(
            flat.ring_view::<D>(rows, width).expect("matrix view"),
            NttCacheMode::ExactNegacyclic {
                width,
                rhs_abs_bound: 1 << 10,
            },
        )
        .expect("cache");
        assert!(cache.uses_limb_split());
        let rhs = alternating_rhs::<D>(width, (1 << 10) - 1);
        assert!(cache.mat_vec_i16::<Prime128Offset275>(11, 1, &rhs).is_ok());
        assert!(cache.mat_vec_i16::<Prime128Offset275>(11, 2, &rhs).is_err());
        assert!(cache
            .mat_vec_i16::<Prime128Offset275>(11, 1, &rhs[1..])
            .is_err());
        // Two 65-bit IFMA limbs still hold a full i16 bound at this width.
        assert_eq!(
            cache.mat_vec_i16::<Prime128Offset275>(16, 1, &rhs).is_err(),
            !cache.uses_ifma52()
        );
        assert!(cache.mat_vec_i16::<Prime64Offset59>(11, 1, &rhs).is_err());
        let wide = alternating_rhs::<D>(width, 1 << 11);
        assert!(cache
            .mat_vec_i16::<Prime128Offset275>(11, 1, &wide)
            .is_err());
    }
}
