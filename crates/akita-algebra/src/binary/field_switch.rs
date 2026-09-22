//! Binary-linear switching from F128/F128 or F64/F192 claims to F162.
//!
//! These are arithmetic building blocks, not a proof verifier. Point coordinate
//! zero controls the least-significant table-index bit. Callers must bind the
//! source, layout, host claim and partials before sampling batching challenges.

use akita_error::{checked, AkitaError};

use super::{BinaryField162 as F, PackedBinary162};
mod profile;
#[cfg(target_arch = "x86_64")]
mod x86;
pub use profile::SwitchField;

/// Inject the source's polynomial bits into F162's low coordinates.
///
/// This map is injective and F2-linear, but does not preserve multiplication.
pub fn embed_source<H: SwitchField>(value: H::Source) -> F {
    let bits: u128 = value.into();
    F([bits as u64, (bits >> 64) as u64, 0])
}

/// Ordered partial evaluations with canonical zero row padding.
///
/// F128 has 128 live rows. F192 has 192 live rows followed by 64 zero rows.
/// This storage has no proof encoding or transcript semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchPartials<H: SwitchField> {
    values: Vec<H::Source>,
}

impl<H: SwitchField> SwitchPartials<H> {
    /// Check the exact padded row count and zero padding before accepting values.
    pub fn try_from_values(values: Vec<H::Source>) -> Result<Self, AkitaError> {
        let expected = 1 << H::BATCH_BITS;
        if values.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: values.len(),
            });
        }
        if values[H::ROWS..].iter().any(|&x| x != H::Source::default()) {
            return Err(AkitaError::InvalidInput(
                "nonzero field-switch row padding".into(),
            ));
        }
        Ok(Self { values })
    }

    /// Return the ordered rows including their canonical padding.
    pub fn values(&self) -> &[H::Source] {
        &self.values
    }

    /// Reconstruct the host evaluation `sum_k beta_k * embed(partial_k)`.
    ///
    /// Equality with an authenticated host claim is the caller's obligation.
    pub fn reconstruct(&self) -> H {
        H::ONE.basis_products()[..H::ROWS]
            .iter()
            .zip(&self.values)
            .fold(H::ZERO, |sum, (&basis, &value)| {
                sum + basis * H::embed_source(value)
            })
    }

    /// Batch the embedded partials at the prescribed 7- or 8-coordinate point.
    pub fn batch(&self, point: &[F]) -> Result<F, AkitaError> {
        let weights = row_weights::<H>(point)?;
        Ok(self.values[..H::ROWS]
            .iter()
            .zip(weights)
            .fold(F::ZERO, |sum, (&value, weight)| {
                sum + embed_source::<H>(value) * weight
            }))
    }
}

/// Compute all partials and leave host equality weights in reusable scratch.
///
/// `source` must contain exactly `2^point.len()` entries, including explicit
/// caller-owned padding. Invalid shapes are rejected before resizing scratch.
/// The point's first coordinate controls the source index's low bit.
/// Work is O(N * host degree) binary selections; no bit matrix is materialized.
pub fn partial_evaluations<H: SwitchField>(
    source: &[H::Source],
    point: &[H],
    equality_scratch: &mut Vec<H>,
) -> Result<SwitchPartials<H>, AkitaError> {
    let expected = checked::pow2(point.len())
        .ok_or_else(|| AkitaError::InvalidInput("field-switch domain size overflow".into()))?;
    if source.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: source.len(),
        });
    }
    equality_scratch.resize(source.len(), H::ZERO);
    H::equality_weights(point, equality_scratch);
    #[cfg(target_arch = "x86_64")]
    if source.len() >= 64 && gfni_available() {
        // SAFETY: public shape validation gives equal power-of-two lengths;
        // at least 64 entries means whole tiles, and features were checked.
        let bits = unsafe { x86::partials::<H>(source, equality_scratch) };
        let mut values = vec![H::Source::default(); 1 << H::BATCH_BITS];
        for (dst, bits) in values.iter_mut().zip(bits) {
            *dst = H::Source::try_from(bits).map_err(|_| {
                AkitaError::InvalidInput("field-switch source bits exceed profile width".into())
            })?;
        }
        return Ok(SwitchPartials { values });
    }
    let mut values = vec![H::Source::default(); 1 << H::BATCH_BITS];
    for (&value, &weight) in source.iter().zip(equality_scratch.iter()) {
        for (word_index, mut word) in weight.coordinates().into_iter().enumerate() {
            while word != 0 {
                let row = word_index * 64 + word.trailing_zeros() as usize;
                values[row] ^= value;
                word &= word - 1;
            }
        }
    }
    Ok(SwitchPartials { values })
}

fn row_weights<H: SwitchField>(point: &[F]) -> Result<[F; 256], AkitaError> {
    if point.len() != H::BATCH_BITS {
        return Err(AkitaError::InvalidPointDimension {
            expected: H::BATCH_BITS,
            actual: point.len(),
        });
    }
    let mut weights = [F::ZERO; 256];
    weights[0] = F::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        for j in 0..width {
            let hi = weights[j] * r;
            weights[j + width] = hi;
            weights[j] += hi;
        }
    }
    Ok(weights)
}

/// Build F162 coefficients from host equality weights and a row-batching point.
///
/// `host_weights` is the scratch produced by `partial_evaluations` for the same
/// host point. The caller owns that correspondence. Output capacity is reused;
/// coefficients stay packed through every sumcheck round. GFNI handles full
/// tiles where available; the portable path uses a small nibble lookup.
pub fn batched_weights<H: SwitchField>(
    host_weights: &[H],
    batch_point: &[F],
    output: &mut PackedBinary162,
) -> Result<(), AkitaError> {
    if !host_weights.len().is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "field-switch weights need a nonempty power-of-two domain".into(),
        ));
    }
    let rows = row_weights::<H>(batch_point)?;
    let output = output.resize_words(host_weights.len());
    #[cfg(target_arch = "x86_64")]
    if host_weights.len() >= 64 && gfni_available() {
        // SAFETY: validated power-of-two length is a whole number of tiles;
        // output limbs match that length and the required features are present.
        unsafe { x86::coefficients::<H>(host_weights, &rows, output) };
        return Ok(());
    }
    let mut lookup = [[F::ZERO; 16]; 48];
    for (chunk, table) in lookup[..H::ROWS / 4].iter_mut().enumerate() {
        for mask in 1usize..16 {
            let bit = mask.trailing_zeros() as usize;
            table[mask] = table[mask & (mask - 1)] + rows[4 * chunk + bit];
        }
    }
    let [low, high, top] = output;
    for (((lo, hi), top), &host) in low.iter_mut().zip(high).zip(top).zip(host_weights) {
        let mut value = F::ZERO;
        for (word_index, word) in host.coordinates()[..H::ROWS / 64].iter().enumerate() {
            for nibble in 0..16 {
                value += lookup[word_index * 16 + nibble][((word >> (4 * nibble)) & 15) as usize];
            }
        }
        [*lo, *hi, *top] = value.to_words();
    }
    Ok(())
}

/// Evaluate the batched coefficient MLE without enumerating its source domain.
///
/// Computes in `host tensor_F2 F162`, using at most 192 coordinates regardless
/// of source length. The two source points must have equal dimension, and the
/// row-batching point must have exactly the profile's 7 or 8 coordinates.
/// This authenticates no claim by itself: the caller must enforce the terminal
/// product equation and open its source factor against the owning commitment.
pub fn transparent_weight<H: SwitchField>(
    host_point: &[H],
    binary_point: &[F],
    batch_point: &[F],
) -> Result<F, AkitaError> {
    if host_point.len() != binary_point.len() {
        return Err(AkitaError::InvalidPointDimension {
            expected: host_point.len(),
            actual: binary_point.len(),
        });
    }
    checked::pow2(host_point.len())
        .ok_or_else(|| AkitaError::InvalidInput("field-switch domain size overflow".into()))?;
    let rows = row_weights::<H>(batch_point)?;
    let mut tensor = [F::ZERO; 192];
    tensor[0] = F::ONE;
    for (&r, &z) in host_point.iter().zip(binary_point) {
        let images = r.basis_products();
        let mut next = [F::ZERO; 192];
        for (k, (&value, &image)) in tensor[..H::ROWS].iter().zip(&images).enumerate() {
            next[k] += value * (F::ONE + z);
            for (word_index, mut word) in image.coordinates().into_iter().enumerate() {
                while word != 0 {
                    next[word_index * 64 + word.trailing_zeros() as usize] += value;
                    word &= word - 1;
                }
            }
        }
        tensor = next;
    }
    Ok(tensor[..H::ROWS]
        .iter()
        .zip(rows)
        .fold(F::ZERO, |sum, (&value, row)| sum + value * row))
}

#[cfg(test)]
mod tests;

#[cfg(target_arch = "x86_64")]
fn gfni_available() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vbmi")
        && std::arch::is_x86_feature_detected!("gfni")
}
