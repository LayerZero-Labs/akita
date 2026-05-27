//! Compile-time ZK commitment masking helpers.

use crate::ZkBlindingSeed;
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use rand_core::{CryptoRng, RngCore};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;
use std::array::from_fn;

const ZK_BLINDING_SETUP_DOMAIN: &[u8] = b"akita/commitment/zk-blinding-setup";
const ZK_B_BLINDING_LABEL: &[u8] = b"B";
const ZK_D_BLINDING_LABEL: &[u8] = b"D";

/// Statistical security target used by the LHL hiding mask.
pub const LHL_STATISTICAL_SECURITY_BITS: usize = 128;

/// Number of fresh digit-ring planes needed for an output in
/// `R_q^{output_ring_len}` when compiled with the `zk` feature.
///
/// The digit-source LHL target is joint over the public hash seed and output,
/// `Delta((B, h_B(S)), (B, U))`.  For `kappa = output_ring_len`, each directly
/// sampled digit plane contributes `D * log_basis` bits, so the conservative
/// count is
/// `ceil((kappa * D * field_bits + 2 * lambda - 2) / (D * log_basis))`.
pub fn blinding_digit_plane_count_from_bits(
    output_ring_len: usize,
    ring_dimension: usize,
    log_basis: u32,
    field_bits: usize,
) -> usize {
    if output_ring_len == 0 {
        return 0;
    }
    let entropy_per_plane = ring_dimension.saturating_mul(log_basis as usize);
    if entropy_per_plane == 0 {
        return 0;
    }
    let lhl_slack = 2 * LHL_STATISTICAL_SECURITY_BITS - 2;
    output_ring_len
        .saturating_mul(ring_dimension)
        .saturating_mul(field_bits)
        .saturating_add(lhl_slack)
        .div_ceil(entropy_per_plane)
}

/// Number of fresh digit-ring planes needed for an output in
/// `R_q^{output_ring_len}`.
pub fn blinding_digit_plane_count<F: CanonicalField>(
    output_ring_len: usize,
    ring_dimension: usize,
    log_basis: u32,
) -> usize {
    blinding_digit_plane_count_from_bits(
        output_ring_len,
        ring_dimension,
        log_basis,
        F::modulus_bits() as usize,
    )
}

/// Number of B-matrix columns reserved for the fresh digit-source blinding.
pub fn blinding_column_count<F: CanonicalField>(
    output_ring_len: usize,
    ring_dimension: usize,
    log_basis: u32,
) -> usize {
    blinding_digit_plane_count::<F>(output_ring_len, ring_dimension, log_basis)
}

/// Number of B-matrix columns reserved for the fresh digit-source blinding when
/// only the field bit length is available.
pub fn blinding_column_count_from_bits(
    output_ring_len: usize,
    ring_dimension: usize,
    log_basis: u32,
    field_bits: usize,
) -> usize {
    blinding_digit_plane_count_from_bits(output_ring_len, ring_dimension, log_basis, field_bits)
}

/// Derive one B-blinding setup ring from the dedicated ZK setup seed.
///
/// B blinding is point-local: the same `(b_row, local)` coordinate is
/// domain-separated by `point_idx`.
pub fn derive_b_blinding_ring<F: FieldCore + RandomSampling, const D: usize>(
    seed: &ZkBlindingSeed,
    point_idx: usize,
    b_row: usize,
    local: usize,
) -> CyclotomicRing<F, D> {
    let mut rng = ZkBlindingXofRng::new_b(seed, point_idx, b_row, local);
    CyclotomicRing::random(&mut rng)
}

/// Derive one D-blinding setup ring from the dedicated ZK setup seed.
pub fn derive_d_blinding_ring<F: FieldCore + RandomSampling, const D: usize>(
    seed: &ZkBlindingSeed,
    d_row: usize,
    local: usize,
) -> CyclotomicRing<F, D> {
    let mut rng = ZkBlindingXofRng::new_d(seed, d_row, local);
    CyclotomicRing::random(&mut rng)
}

/// Negacyclic B-blinding row contribution for one point-local blinding vector.
pub fn b_blinding_negacyclic_rows<
    F: FieldCore + FromPrimitiveInt + RandomSampling,
    const D: usize,
>(
    seed: &ZkBlindingSeed,
    point_idx: usize,
    n_b: usize,
    blinding_digits: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    (0..n_b)
        .map(|b_row| {
            blinding_digits.iter().enumerate().fold(
                CyclotomicRing::<F, D>::zero(),
                |mut acc, (local, digits)| {
                    let setup_ring = derive_b_blinding_ring(seed, point_idx, b_row, local);
                    let digit_ring = i8_digit_ring(digits);
                    setup_ring.mul_accumulate_into(&digit_ring, &mut acc);
                    acc
                },
            )
        })
        .collect()
}

/// Negacyclic D-blinding row contribution.
pub fn d_blinding_negacyclic_rows<
    F: FieldCore + FromPrimitiveInt + RandomSampling,
    const D: usize,
>(
    seed: &ZkBlindingSeed,
    n_d: usize,
    blinding_digits: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    (0..n_d)
        .map(|d_row| {
            blinding_digits.iter().enumerate().fold(
                CyclotomicRing::<F, D>::zero(),
                |mut acc, (local, digits)| {
                    let setup_ring = derive_d_blinding_ring(seed, d_row, local);
                    let digit_ring = i8_digit_ring(digits);
                    setup_ring.mul_accumulate_into(&digit_ring, &mut acc);
                    acc
                },
            )
        })
        .collect()
}

/// Cyclic B-blinding row contribution for split-eq quotient construction.
pub fn b_blinding_cyclic_rows<F: FieldCore + FromPrimitiveInt + RandomSampling, const D: usize>(
    seed: &ZkBlindingSeed,
    point_idx: usize,
    n_b: usize,
    blinding_digits: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    (0..n_b)
        .map(|b_row| {
            blinding_digits.iter().enumerate().fold(
                CyclotomicRing::<F, D>::zero(),
                |mut acc, (local, digits)| {
                    let setup_ring = derive_b_blinding_ring(seed, point_idx, b_row, local);
                    let digit_ring = i8_digit_ring(digits);
                    cyclic_mul_accumulate_into(&setup_ring, &digit_ring, &mut acc);
                    acc
                },
            )
        })
        .collect()
}

/// Cyclic D-blinding row contribution for split-eq quotient construction.
pub fn d_blinding_cyclic_rows<F: FieldCore + FromPrimitiveInt + RandomSampling, const D: usize>(
    seed: &ZkBlindingSeed,
    n_d: usize,
    blinding_digits: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    (0..n_d)
        .map(|d_row| {
            blinding_digits.iter().enumerate().fold(
                CyclotomicRing::<F, D>::zero(),
                |mut acc, (local, digits)| {
                    let setup_ring = derive_d_blinding_ring(seed, d_row, local);
                    let digit_ring = i8_digit_ring(digits);
                    cyclic_mul_accumulate_into(&setup_ring, &digit_ring, &mut acc);
                    acc
                },
            )
        })
        .collect()
}

fn i8_digit_ring<F: FieldCore + FromPrimitiveInt, const D: usize>(
    digits: &[i8; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(from_fn(|idx| F::from_i64(digits[idx] as i64)))
}

fn cyclic_mul_accumulate_into<F: FieldCore, const D: usize>(
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
    dst: &mut CyclotomicRing<F, D>,
) {
    for (i, &a) in lhs.coefficients().iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs.coefficients().iter().enumerate() {
            if !b.is_zero() {
                dst.coeffs[(i + j) % D] += a * b;
            }
        }
    }
}

struct ZkBlindingXofRng {
    reader: Box<dyn XofReader>,
}

impl ZkBlindingXofRng {
    fn new_b(seed: &ZkBlindingSeed, point_idx: usize, row: usize, local: usize) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", ZK_BLINDING_SETUP_DOMAIN);
        absorb_len_prefixed(&mut xof, b"seed", seed);
        absorb_len_prefixed(&mut xof, b"role", ZK_B_BLINDING_LABEL);
        absorb_len_prefixed(&mut xof, b"point", &(point_idx as u64).to_le_bytes());
        absorb_len_prefixed(&mut xof, b"row", &(row as u64).to_le_bytes());
        absorb_len_prefixed(&mut xof, b"local", &(local as u64).to_le_bytes());
        Self {
            reader: Box::new(xof.finalize_xof()),
        }
    }

    fn new_d(seed: &ZkBlindingSeed, row: usize, local: usize) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", ZK_BLINDING_SETUP_DOMAIN);
        absorb_len_prefixed(&mut xof, b"seed", seed);
        absorb_len_prefixed(&mut xof, b"role", ZK_D_BLINDING_LABEL);
        absorb_len_prefixed(&mut xof, b"row", &(row as u64).to_le_bytes());
        absorb_len_prefixed(&mut xof, b"local", &(local as u64).to_le_bytes());
        Self {
            reader: Box::new(xof.finalize_xof()),
        }
    }
}

impl RngCore for ZkBlindingXofRng {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.reader.read(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for ZkBlindingXofRng {}

fn absorb_len_prefixed(xof: &mut Shake256, label: &[u8], data: &[u8]) {
    xof.update(&(label.len() as u64).to_le_bytes());
    xof.update(label);
    xof.update(&(data.len() as u64).to_le_bytes());
    xof.update(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp32;

    type F = Fp32<251>;
    const D: usize = 8;

    #[test]
    fn digit_plane_count_uses_direct_lhl_entropy() {
        // ceil((2 * 32 * 128 + 254) / (32 * 5)) = ceil(8446 / 160) = 53.
        assert_eq!(blinding_digit_plane_count_from_bits(2, 32, 5, 128), 53);
        // ceil((1 * 128 * 128 + 254) / (128 * 4)) = ceil(16638 / 512) = 33.
        assert_eq!(blinding_digit_plane_count_from_bits(1, 128, 4, 128), 33);
    }

    #[test]
    fn small_dimensions_can_need_many_digit_planes() {
        // ceil((3 * 8 * 8 + 254) / (8 * 2)) = ceil(446 / 16) = 28.
        assert_eq!(blinding_digit_plane_count_from_bits(3, 8, 2, 8), 28);
    }

    #[test]
    fn column_count_is_digit_plane_count() {
        assert_eq!(blinding_column_count_from_bits(3, 8, 2, 8), 28);
    }

    #[test]
    fn zero_output_needs_no_digit_planes() {
        assert_eq!(blinding_digit_plane_count_from_bits(0, 32, 4, 128), 0);
    }

    #[test]
    fn default_fp128_examples_match_spec() {
        assert_eq!(blinding_digit_plane_count_from_bits(1, 64, 5, 128), 27);
        assert_eq!(blinding_digit_plane_count_from_bits(1, 128, 5, 128), 26);
        assert_eq!(blinding_digit_plane_count_from_bits(1, 64, 4, 128), 33);
    }

    #[test]
    fn zero_output_needs_no_blinding_columns() {
        assert_eq!(blinding_column_count_from_bits(0, 32, 43, 128), 0);
    }

    #[test]
    fn zk_blinding_rings_are_role_and_point_separated() {
        let seed = [7u8; 32];
        let b0 = derive_b_blinding_ring::<F, D>(&seed, 0, 1, 2);
        assert_eq!(b0, derive_b_blinding_ring::<F, D>(&seed, 0, 1, 2));
        assert_ne!(b0, derive_b_blinding_ring::<F, D>(&seed, 1, 1, 2));
        assert_ne!(b0, derive_d_blinding_ring::<F, D>(&seed, 1, 2));
    }

    #[test]
    fn cyclic_and_negacyclic_helpers_differ_on_wraparound() {
        let seed = [9u8; 32];
        let mut digits = vec![[0i8; D]; 1];
        digits[0][D - 1] = 1;
        let neg = b_blinding_negacyclic_rows::<F, D>(&seed, 0, 1, &digits);
        let cyc = b_blinding_cyclic_rows::<F, D>(&seed, 0, 1, &digits);
        assert_ne!(neg, cyc);
    }
}
