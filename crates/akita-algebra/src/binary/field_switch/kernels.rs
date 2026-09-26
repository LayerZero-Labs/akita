//! Select bit-matrix kernels without exposing an instruction-set layout.

use std::sync::OnceLock;

use akita_error::AkitaError;

use super::{SwitchField, F};

const TILE: usize = 64;

#[derive(Clone, Copy)]
enum PartialBackend {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Gfni512,
    #[cfg(target_arch = "x86_64")]
    Gfni256,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
}

#[derive(Clone, Copy)]
enum CoefficientBackend {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Gfni512,
    #[cfg(target_arch = "x86_64")]
    Gfni256,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
}

// Partial construction and coefficient mapping may favor different backends.
// Keep the choice per stage, while both write the same canonical results.
struct StageBackends {
    partials: PartialBackend,
    coefficients: CoefficientBackend,
}

fn selected() -> &'static StageBackends {
    static SELECTED: OnceLock<StageBackends> = OnceLock::new();
    SELECTED.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
            && std::arch::is_x86_feature_detected!("gfni")
        {
            return StageBackends {
                partials: PartialBackend::Gfni512,
                coefficients: CoefficientBackend::Gfni512,
            };
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("gfni")
        {
            return StageBackends {
                partials: PartialBackend::Gfni256,
                coefficients: CoefficientBackend::Gfni256,
            };
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return StageBackends {
                partials: PartialBackend::Neon,
                coefficients: CoefficientBackend::Neon,
            };
        }
        StageBackends {
            partials: PartialBackend::Portable,
            coefficients: CoefficientBackend::Portable,
        }
    })
}

/// Compute canonical partial rows, including zero padding for F192.
pub(super) fn partials<H: SwitchField>(
    source: &[H::Source],
    weights: &[H],
) -> Result<Vec<H::Source>, AkitaError> {
    debug_assert_eq!(source.len(), weights.len());
    if source.len() >= TILE && source.len().is_multiple_of(TILE) {
        let bits: Option<[u128; 192]> = match selected().partials {
            PartialBackend::Portable => None,
            #[cfg(target_arch = "x86_64")]
            PartialBackend::Gfni512 => {
                // SAFETY: both slices contain whole tiles and runtime detection
                // established every feature required by the backend.
                Some(unsafe { super::x86::partials::<H>(source, weights) })
            }
            #[cfg(target_arch = "x86_64")]
            PartialBackend::Gfni256 => {
                // SAFETY: the same whole-tile shape and detected AVX2/GFNI
                // features establish this backend's preconditions.
                Some(unsafe { super::x86_256::partials::<H>(source, weights) })
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            PartialBackend::Neon => {
                // SAFETY: both slices contain complete tiles and runtime
                // detection established NEON support.
                Some(unsafe { super::arm_partials::partials::<H>(source, weights) })
            }
        };
        if let Some(bits) = bits {
            let mut values = vec![H::Source::default(); 1 << H::BATCH_BITS];
            for (dst, bits) in values.iter_mut().zip(bits) {
                *dst = H::Source::try_from(bits).map_err(|_| {
                    AkitaError::InvalidInput("field-switch source bits exceed profile width".into())
                })?;
            }
            return Ok(values);
        }
    }
    let mut values = vec![H::Source::default(); 1 << H::BATCH_BITS];
    for (&source, &weight) in source.iter().zip(weights) {
        for (word_index, mut word) in weight.coordinates().into_iter().enumerate() {
            while word != 0 {
                let row = word_index * 64 + word.trailing_zeros() as usize;
                values[row] ^= source;
                word &= word - 1;
            }
        }
    }
    Ok(values)
}

/// Write canonical packed F162 limbs for all host equality weights.
pub(super) fn coefficients<H: SwitchField>(
    weights: &[H],
    rows: &[F; 256],
    output: [&mut [u64]; 3],
) {
    debug_assert!(output.iter().all(|words| words.len() == weights.len()));
    if weights.len() >= TILE && weights.len().is_multiple_of(TILE) {
        match selected().coefficients {
            CoefficientBackend::Portable => {}
            #[cfg(target_arch = "x86_64")]
            CoefficientBackend::Gfni512 => {
                // SAFETY: the caller validated row weights; the output has
                // three exactly sized limbs and all features were detected.
                unsafe { super::x86::coefficients::<H>(weights, rows, output) };
                return;
            }
            #[cfg(target_arch = "x86_64")]
            CoefficientBackend::Gfni256 => {
                // SAFETY: three exactly sized limbs, canonical row weights,
                // whole tiles, and AVX2/GFNI were established above.
                unsafe { super::x86_256::coefficients::<H>(weights, rows, output) };
                return;
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            CoefficientBackend::Neon => {
                // SAFETY: the caller validated row weights, all output limbs
                // match the input length, and NEON was detected at runtime.
                unsafe { super::arm::coefficients::<H>(weights, rows, output) };
                return;
            }
        }
    }
    let mut lookup = [[F::ZERO; 16]; 48];
    for (chunk, table) in lookup[..H::ROWS / 4].iter_mut().enumerate() {
        for mask in 1usize..16 {
            let bit = mask.trailing_zeros() as usize;
            table[mask] = table[mask & (mask - 1)] + rows[4 * chunk + bit];
        }
    }
    let [low, high, top] = output;
    for (((lo, hi), top), &host) in low.iter_mut().zip(high).zip(top).zip(weights) {
        let mut value = F::ZERO;
        for (word_index, word) in host.coordinates()[..H::ROWS / 64].iter().enumerate() {
            for nibble in 0..16 {
                value += lookup[word_index * 16 + nibble][((word >> (4 * nibble)) & 15) as usize];
            }
        }
        [*lo, *hi, *top] = value.to_words();
    }
}
