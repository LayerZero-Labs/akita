use super::*;

pub(super) fn ifma52_cache_enabled<const D: usize>() -> bool {
    (64..=512).contains(&D) && ifma52_enabled()
}

fn ifma52_tail_requirement<F: CanonicalField, const K: usize, const D: usize>(
    moduli: [u64; K],
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
        .with_prime_modulus(I16_TAIL_PRIME.p as u128)
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
}

impl<const D: usize> ExactCachePlan<D> {
    const fn needs_tail(&self) -> bool {
        match self {
            Self::Q32 { needs_tail, .. }
            | Self::Q32Ifma52 { needs_tail, .. }
            | Self::Q64 { needs_tail, .. }
            | Self::Q128 { needs_tail, .. }
            | Self::Q128Ifma52 { needs_tail, .. } => *needs_tail,
            Self::Q64Ifma52 { .. } => false,
        }
    }
}

pub(super) fn exact_cache_plan<F: CanonicalField, const D: usize>(
    selected: ProtocolCrtNttParams<D>,
    width: usize,
    rhs_abs_bound: u64,
) -> Result<ExactCachePlan<D>, AkitaError> {
    match selected {
        ProtocolCrtNttParams::Q32(params) => {
            if let Some(needs_tail) =
                ifma52_tail_requirement::<F, 1, D>([IFMA52_PRIMES[0]], width, rhs_abs_bound)
            {
                let mut params = Ifma52Params::new([IFMA52_PRIMES[0]])?;
                if needs_tail {
                    params = params.with_i16_tail(I16_TAIL_PRIME.p)?;
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
            if let Some(needs_tail) =
                ifma52_tail_requirement::<F, 3, D>(IFMA52_PRIMES, width, rhs_abs_bound)
            {
                let mut params = Ifma52Params::new(IFMA52_PRIMES)?;
                if needs_tail {
                    params = params.with_i16_tail(I16_TAIL_PRIME.p)?;
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

/// Return whether an exact signed-coefficient request requires the i16 tail.
pub fn ntt_cache_requires_i16_tail<F: CanonicalField, const D: usize>(
    width: usize,
    rhs_abs_bound: u64,
) -> Result<bool, AkitaError> {
    let mode = NttCacheMode::ExactNegacyclic {
        width,
        rhs_abs_bound,
    };
    validate_cache_mode(mode)?;
    Ok(
        exact_cache_plan::<F, D>(select_crt_ntt_params::<F, D>()?, width, rhs_abs_bound)?
            .needs_tail(),
    )
}

pub(super) fn prepare_exact_ntt_cache<F: FieldCore + CanonicalField, const D: usize>(
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
            let (neg, tail) = prepare_ifma52_exact(matrix, tail_prefix_len, *params, needs_tail)?;
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
            let (neg, tail) = prepare_ifma52_exact(matrix, tail_prefix_len, *params, needs_tail)?;
            PreparedNttCacheRepr::Q128Ifma52 { neg, tail }
        }
    };
    prepared.validate()?;
    Ok(PreparedNttCache(prepared))
}

fn prepare_ifma52_exact<F: FieldCore + CanonicalField, const K: usize, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    tail_prefix_len: Option<usize>,
    mut params: Ifma52Params<K, D>,
    needs_tail: bool,
) -> Result<(Ifma52NttMatrix<K, D>, Option<PreparedIfma52I16Tail<D>>), AkitaError> {
    // A verifier cache rebuild joins physical prefix lengths. Preserve a tail
    // installed by an earlier stronger request even when the current request's
    // exactness bound fits the IFMA base residues by themselves.
    let retain_tail = needs_tail || tail_prefix_len.is_some_and(|length| length > 0);
    if retain_tail && !params.has_i16_tail() {
        params = params.with_i16_tail(I16_TAIL_PRIME.p)?;
    }
    let tail = retain_tail
        .then(|| prepare_ifma52_i16_tail(matrix, tail_prefix_len))
        .transpose()?;
    Ok((Ifma52NttMatrix::prepare(matrix.as_slice(), &params), tail))
}

fn prepare_ifma52_i16_tail<F: FieldCore + CanonicalField, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    tail_prefix_len: Option<usize>,
) -> Result<PreparedIfma52I16Tail<D>, AkitaError> {
    let tail_len = tail_prefix_len.unwrap_or(matrix.as_slice().len());
    if tail_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "required mixed IFMA52 tail prefix is empty".into(),
        ));
    }
    let tail_rings = matrix.as_slice().get(..tail_len).ok_or_else(|| {
        AkitaError::InvalidSetup("mixed IFMA52 tail prefix exceeds the base matrix".into())
    })?;
    let params = CrtNttParamSet::new([I16_TAIL_PRIME]);
    let negacyclic = cfg_iter!(tail_rings)
        .map(|ring| CyclotomicCrtNtt::from_ring(ring, &params))
        .collect();
    Ok(PreparedIfma52I16Tail { negacyclic, params })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatMatrix;
    use akita_algebra::CyclotomicRing;
    use akita_field::Prime32Offset99;

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
        assert!(cache.has_i16_tail());
        assert_eq!(
            cache.cache_bytes(),
            10 * D * core::mem::size_of::<u64>() + 4 * D * core::mem::size_of::<i16>()
        );
    }
}
