use super::limbs::{limb_plan, LimbPlan};
use super::*;

pub(super) fn ifma52_cache_enabled<const D: usize>() -> bool {
    ifma52_cache_enabled_for_ring_dimension(D)
}

pub(super) fn ifma52_cache_enabled_for_ring_dimension(ring_dimension: usize) -> bool {
    (64..=2048).contains(&ring_dimension) && ifma52_enabled()
}

fn ifma52_tail_requirement<F: Field + CanonicalEncoding, const K: usize, const D: usize>(
    moduli: [u64; K],
    tail_modulus: u128,
    width: usize,
    rhs_abs_bound: u64,
) -> Option<bool> {
    if !ifma52_cache_enabled::<D>() {
        return None;
    }
    let capacity = CrtCapacity::from_prime_moduli(moduli.map(u128::from));
    if capacity.supports::<F, D>(width, rhs_abs_bound) {
        return Some(false);
    }
    capacity
        .with_prime_modulus(tail_modulus)
        .supports::<F, D>(width, rhs_abs_bound)
        .then_some(true)
}

pub(super) enum ExactCachePlan<const D: usize> {
    Q32 {
        params: Box<CrtNttParamSet<i32, Q32_NUM_PRIMES, D>>,
        needs_tail: bool,
    },
    Q32Ifma52 {
        params: Box<Ifma52Params<1, D>>,
        needs_tail: bool,
    },
    Q64 {
        params: Box<CrtNttParamSet<i32, Q64_NUM_PRIMES, D>>,
        needs_tail: bool,
    },
    Q64Ifma52 {
        params: Box<Ifma52Params<2, D>>,
    },
    Q128 {
        params: Box<CrtNttParamSet<i32, Q128_NUM_PRIMES, D>>,
        needs_tail: bool,
    },
    Q128Ifma52 {
        params: Box<Ifma52Params<3, D>>,
        needs_tail: bool,
    },
    Limbs(LimbPlan<D>),
}

impl<const D: usize> ExactCachePlan<D> {
    pub(super) const fn needs_tail(&self) -> bool {
        match self {
            Self::Q32 { needs_tail, .. }
            | Self::Q32Ifma52 { needs_tail, .. }
            | Self::Q64 { needs_tail, .. }
            | Self::Q128 { needs_tail, .. }
            | Self::Q128Ifma52 { needs_tail, .. } => *needs_tail,
            Self::Q64Ifma52 { .. } | Self::Limbs(_) => false,
        }
    }
}

/// Plan an exact signed-i16 cache. With `limb_rows`, the number of rows the
/// prepared matrix holds, a limb split replaces the base plan when cheaper.
pub(super) fn exact_cache_plan<F: Field + CanonicalEncoding, const D: usize>(
    selected: ProtocolCrtNttParams<D>,
    width: usize,
    rhs_abs_bound: u64,
    limb_rows: Option<usize>,
) -> Result<ExactCachePlan<D>, AkitaError> {
    let base = base_exact_cache_plan::<F, D>(selected, width, rhs_abs_bound)?;
    if let Some(rows) = limb_rows {
        if let Some(plan) = limb_plan::<F, D>(&base, width, rhs_abs_bound, rows)? {
            return Ok(ExactCachePlan::Limbs(plan));
        }
    }
    Ok(base)
}

fn base_exact_cache_plan<F: Field + CanonicalEncoding, const D: usize>(
    selected: ProtocolCrtNttParams<D>,
    width: usize,
    rhs_abs_bound: u64,
) -> Result<ExactCachePlan<D>, AkitaError> {
    match selected {
        ProtocolCrtNttParams::Q32(params) => {
            if let Some(needs_tail) = ifma52_tail_requirement::<F, 1, D>(
                [IFMA52_PRIMES[0]],
                I16_TAIL_PRIME.p as u128,
                width,
                rhs_abs_bound,
            ) {
                let mut params = Ifma52Params::new([IFMA52_PRIMES[0]])?;
                if needs_tail {
                    params = params.with_tail(I16_TAIL_PRIME.p)?;
                }
                Ok(ExactCachePlan::Q32Ifma52 {
                    params: Box::new(params),
                    needs_tail,
                })
            } else {
                let needs_tail = required_profile_for_params::<F, _, Q32_NUM_PRIMES, D>(
                    &params,
                    width,
                    rhs_abs_bound,
                )?;
                Ok(ExactCachePlan::Q32 {
                    params: Box::new(params),
                    needs_tail,
                })
            }
        }
        ProtocolCrtNttParams::Q64(params) => {
            if matches!(
                ifma52_tail_requirement::<F, 2, D>(
                    [IFMA52_PRIMES[0], IFMA52_PRIMES[1]],
                    I16_TAIL_PRIME.p as u128,
                    width,
                    rhs_abs_bound,
                ),
                Some(false)
            ) {
                return Ok(ExactCachePlan::Q64Ifma52 {
                    params: Box::new(Ifma52Params::new([IFMA52_PRIMES[0], IFMA52_PRIMES[1]])?),
                });
            }
            let needs_tail = required_profile_for_params::<F, _, Q64_NUM_PRIMES, D>(
                &params,
                width,
                rhs_abs_bound,
            )?;
            Ok(ExactCachePlan::Q64 {
                params: Box::new(params),
                needs_tail,
            })
        }
        ProtocolCrtNttParams::Q128(params) => {
            let requirement = |tail_modulus: u128| {
                ifma52_tail_requirement::<F, 3, D>(
                    IFMA52_PRIMES,
                    tail_modulus,
                    width,
                    rhs_abs_bound,
                )
            };
            let i32_tail = q128_primes()[0];
            if let Some(needs_tail) = requirement(i32_tail.p as u128) {
                let mut params = Ifma52Params::new(IFMA52_PRIMES)?;
                // Prefer the 14-bit tail, whose transforms and products run
                // at twice the lane count, whenever its capacity suffices.
                if requirement(I16_TAIL_PRIME.p as u128) == Some(true) {
                    params = params.with_tail(I16_TAIL_PRIME.p)?;
                } else if needs_tail {
                    params = params.with_tail(i32_tail.p)?;
                }
                Ok(ExactCachePlan::Q128Ifma52 {
                    params: Box::new(params),
                    needs_tail,
                })
            } else {
                let needs_tail = required_profile_for_params::<F, _, Q128_NUM_PRIMES, D>(
                    &params,
                    width,
                    rhs_abs_bound,
                )?;
                Ok(ExactCachePlan::Q128 {
                    params: Box::new(params),
                    needs_tail,
                })
            }
        }
    }
}

/// Return whether an exact signed-coefficient request requires a CRT tail.
pub fn ntt_cache_requires_exactness_tail<F: Field + CanonicalEncoding, const D: usize>(
    width: usize,
    rhs_abs_bound: u64,
) -> Result<bool, AkitaError> {
    let mode = NttCacheMode::ExactNegacyclic {
        width,
        rhs_abs_bound,
    };
    validate_cache_mode(mode)?;
    Ok(
        exact_cache_plan::<F, D>(select_crt_ntt_params::<F, D>()?, width, rhs_abs_bound, None)?
            .needs_tail(),
    )
}

pub(super) fn prepare_exact_ntt_cache<F: Field + CanonicalEncoding, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    tail_prefix_len: Option<usize>,
    plan: ExactCachePlan<D>,
) -> Result<PreparedNttCache<D>, AkitaError> {
    macro_rules! homogeneous {
        ($params:expr, $variant:ident, $needs_tail:expr) => {{
            let params = *$params;
            let neg = cfg_iter!(matrix.as_slice())
                .map(|ring| CyclotomicCrtNtt::from_ring(ring, &params))
                .collect();
            let requested_tail_len = if $needs_tail {
                Some(tail_prefix_len.unwrap_or(matrix.as_slice().len()))
            } else {
                tail_prefix_len.filter(|&len| len > 0)
            };
            let tail = if let Some(tail_len) = requested_tail_len {
                if tail_len == 0 {
                    return Err(AkitaError::InvalidSetup(
                        "required i16-tail NTT prefix is empty".into(),
                    ));
                }
                let tail_params = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
                let tail_rings = matrix.as_slice().get(..tail_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("i16-tail NTT prefix exceeds the base matrix".into())
                })?;
                let negacyclic = cfg_iter!(tail_rings)
                    .map(|ring| CyclotomicCrtNtt::from_ring(ring, &tail_params))
                    .collect();
                Some(PreparedI16Tail {
                    negacyclic,
                    params: I16TailParams::new(params.clone(), tail_params),
                })
            } else {
                None
            };
            PreparedNttCacheRepr::$variant {
                neg: Some(neg),
                cyc: None,
                params,
                tail,
                exact: true,
            }
        }};
    }

    let prepared = match plan {
        ExactCachePlan::Q32 { params, needs_tail } => {
            homogeneous!(params, Q32, needs_tail)
        }
        ExactCachePlan::Q32Ifma52 { params, needs_tail } => {
            let (neg, tail) =
                prepare_ifma52_exact(matrix, tail_prefix_len, *params, I16_TAIL_PRIME, needs_tail)?;
            PreparedNttCacheRepr::Q32Ifma52 { neg, tail }
        }
        ExactCachePlan::Q64 { params, needs_tail } => {
            homogeneous!(params, Q64, needs_tail)
        }
        ExactCachePlan::Q64Ifma52 { params } => {
            if tail_prefix_len.is_some_and(|length| length > 0) {
                return Err(AkitaError::InvalidSetup(
                    "IFMA52 exact cache does not require an i16 tail".into(),
                ));
            }
            PreparedNttCacheRepr::Q64Ifma52 {
                neg: Ifma52NttMatrix::prepare(matrix.as_slice(), &params),
            }
        }
        ExactCachePlan::Q128 { params, needs_tail } => {
            homogeneous!(params, Q128, needs_tail)
        }
        ExactCachePlan::Q128Ifma52 { params, needs_tail } => {
            // A tail retained without a planned one is exactness-only: the
            // base primes suffice, so it takes the cheaper 14-bit prime.
            if params.has_tail::<i32>() {
                let tail_prime = q128_primes()[0];
                let (neg, tail) =
                    prepare_ifma52_exact(matrix, tail_prefix_len, *params, tail_prime, needs_tail)?;
                PreparedNttCacheRepr::Q128Ifma52 {
                    neg,
                    tail: tail.map(Q128Ifma52Tail::I32),
                }
            } else {
                let (neg, tail) = prepare_ifma52_exact(
                    matrix,
                    tail_prefix_len,
                    *params,
                    I16_TAIL_PRIME,
                    needs_tail,
                )?;
                PreparedNttCacheRepr::Q128Ifma52 {
                    neg,
                    tail: tail.map(Q128Ifma52Tail::I16),
                }
            }
        }
        ExactCachePlan::Limbs(plan) => {
            if tail_prefix_len.is_some() {
                return Err(AkitaError::InvalidSetup(
                    "limb-split exact cache does not take a tail prefix".into(),
                ));
            }
            PreparedNttCacheRepr::Limbs(PreparedLimbMatrix::prepare(matrix, plan))
        }
    };
    prepared.validate()?;
    Ok(PreparedNttCache(prepared))
}

fn prepare_ifma52_exact<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    matrix: RingMatrixView<'_, F, D>,
    tail_prefix_len: Option<usize>,
    mut params: Ifma52Params<K, D>,
    tail_prime: NttPrime<W>,
    needs_tail: bool,
) -> Result<(Ifma52NttMatrix<K, D>, Option<PreparedIfma52Tail<W, D>>), AkitaError> {
    // A verifier cache rebuild joins physical prefix lengths. Preserve a tail
    // installed by an earlier stronger request even when the current request's
    // exactness bound fits the IFMA base residues by themselves.
    let retain_tail = needs_tail || tail_prefix_len.is_some_and(|length| length > 0);
    if retain_tail && !params.has_tail::<W>() {
        params = params.with_tail(tail_prime.p)?;
    }
    let tail = retain_tail
        .then(|| prepare_ifma52_tail(matrix, tail_prefix_len, tail_prime))
        .transpose()?;
    Ok((Ifma52NttMatrix::prepare(matrix.as_slice(), &params), tail))
}

fn prepare_ifma52_tail<F: Field + CanonicalEncoding, W: PrimeWidth, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    tail_prefix_len: Option<usize>,
    prime: NttPrime<W>,
) -> Result<PreparedIfma52Tail<W, D>, AkitaError> {
    let tail_len = tail_prefix_len.unwrap_or(matrix.as_slice().len());
    if tail_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "required mixed IFMA52 tail prefix is empty".into(),
        ));
    }
    let tail_rings = matrix.as_slice().get(..tail_len).ok_or_else(|| {
        AkitaError::InvalidSetup("mixed IFMA52 tail prefix exceeds the base matrix".into())
    })?;
    let params = CrtNttParamSet::new([prime]);
    let negacyclic = cfg_iter!(tail_rings)
        .map(|ring| CyclotomicCrtNtt::from_ring(ring, &params))
        .collect();
    Ok(PreparedIfma52Tail { negacyclic, params })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatMatrix;
    use akita_algebra::CyclotomicRing;
    use jolt_field::Prime32Offset99;

    #[test]
    fn ifma_rebuild_retains_a_joined_tail_prefix() {
        const D: usize = 64;
        let flat =
            FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<Prime32Offset99, D>::zero(); 10]);
        let matrix = flat.ring_view::<D>(1, 10).expect("matrix view");
        let plan = ExactCachePlan::Q32Ifma52 {
            params: Box::new(Ifma52Params::new([IFMA52_PRIMES[0]]).expect("IFMA52 parameters")),
            needs_tail: false,
        };
        let cache = prepare_exact_ntt_cache(matrix, Some(4), plan).expect("retained tail");

        assert!(cache.uses_ifma52());
        assert!(cache.has_exactness_tail());
        assert_eq!(
            cache.cache_bytes(),
            10 * D * core::mem::size_of::<u64>() + 4 * D * core::mem::size_of::<i16>()
        );
    }

    #[test]
    fn q128_ifma_rebuild_retains_a_joined_tail_prefix_at_the_planned_width() {
        const D: usize = 64;
        let flat =
            FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<Prime128OffsetA7F7, D>::zero(); 10]);
        let base = || Ifma52Params::new(IFMA52_PRIMES).expect("IFMA52 parameters");
        let i32_tail = base().with_tail(q128_primes()[0].p).expect("i32 tail");
        for (params, needs_tail, tail_bytes) in [
            (base(), false, core::mem::size_of::<i16>()),
            (i32_tail, true, core::mem::size_of::<i32>()),
        ] {
            let matrix = flat.ring_view::<D>(1, 10).expect("matrix view");
            let plan = ExactCachePlan::Q128Ifma52 {
                params: Box::new(params),
                needs_tail,
            };
            let cache = prepare_exact_ntt_cache(matrix, Some(4), plan).expect("retained tail");

            assert!(cache.has_exactness_tail());
            assert_eq!(
                cache.cache_bytes(),
                10 * D * IFMA52_PRIMES.len() * core::mem::size_of::<u64>() + 4 * D * tail_bytes
            );
        }
    }

    #[test]
    fn q128_ifma_plan_prefers_the_i16_tail_within_its_capacity() {
        const D: usize = 1024;
        type F = Prime128OffsetA7F7;
        if !ifma52_cache_enabled::<D>() {
            return;
        }
        // Base plus the 14-bit tail holds 163.6 bits: width 1536 at D = 1024
        // and |rhs| <= 2^15.
        let tail = |width| {
            let params = select_crt_ntt_params::<F, D>().expect("protocol parameters");
            match exact_cache_plan::<F, D>(params, width, 1 << 15, None).expect("plan") {
                ExactCachePlan::Q128Ifma52 { params, needs_tail } => (
                    needs_tail,
                    params.has_tail::<i16>(),
                    params.has_tail::<i32>(),
                ),
                _ => panic!("expected the q128 IFMA52 plan"),
            }
        };
        assert_eq!(tail(1536), (true, true, false));
        assert_eq!(tail(1537), (true, false, true));
    }
}
