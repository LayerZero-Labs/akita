//! Three-byte q24 slots and a product accumulator with one reduction per
//! `2^15` products.
//!
//! This module adds storage and accumulation kernels for the admitted q24
//! prime [`Q24`] only. It admits no new prime and changes no bound of the
//! existing limb domain, and no existing caller uses it.
//!
//! # Packing
//!
//! [`TrinomialQ24PackedSlots`] stores the 648 slots of a q24
//! [`TrinomialLimbSlots`] as consecutive three-byte offset words in the
//! [`Q24Image`] encoding. Packing centers every lazy slot first, so the packed
//! value is the canonical centered representative in
//! `[-(q-1)/2, (q-1)/2]`. Unpacking returns those centered values. Every
//! three-byte word decodes to an accepted representative `x` with
//! `IMAGE_MIN <= x <= IMAGE_MAX`, so `|x| < q` and the result is a valid lazy
//! slot.
//!
//! # Accumulation bound
//!
//! The accumulator decodes each packed slot to its canonical residue
//! `0 <= u < q` and adds unsigned products to one `u64` per slot. After a
//! reduction each sum is below `q`. After `k <= 2^15` further products a sum
//! is at most `q - 1 + k (q - 1)^2`, which at `k = 2^15` equals
//! `6_657_135_070_901_009_761 < 2^63`. The accumulator reduces every sum
//! modulo `q` exactly when the product count since the last reduction reaches
//! `2^15`, so no `u64` addition wraps. [`TrinomialQ24Accumulator::finish`]
//! returns the centered residue of each sum. That is the unique representative
//! in `[-(q-1)/2, (q-1)/2]` of the exact product sum, the same value
//! [`TrinomialLimbAccumulator::finish`](super::TrinomialLimbAccumulator::finish)
//! returns for the unpacked slots.
//!
//! # Kernels
//!
//! The kernel follows the domain's existing runtime backend choice. On x86_64
//! it uses AVX-512 IFMA (`vpmadd52luq`) when the domain uses SIMD and the CPU
//! reports AVX2, AVX-512F/DQ and AVX-512 IFMA. Both operands are below `2^24`,
//! so the low 52-bit product is the whole product, and the instruction adds it
//! to the full 64-bit lane. On AArch64 with NEON it uses `umlal` when the
//! domain uses NEON. Every other case, including `AKITA_SCALAR_NTT=1`, uses
//! the portable loop. All kernels give the same sums.

use super::super::image::{Q24Image, IMAGE_OFFSET, Q24};
use super::arithmetic::Arithmetic;
use super::backend::Backend;
use super::{TrinomialError, TrinomialLimbDomain, TrinomialLimbSlots, DEGREE};

#[cfg(target_arch = "x86_64")]
mod ifma;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
#[cfg(test)]
mod tests;

/// Bytes in one packed slot vector: three per slot.
pub const Q24_PACKED_SLOT_BYTES: usize = 3 * DEGREE;

/// Products accumulated between two reductions modulo q24.
pub const Q24_REDUCTION_TERMS: u32 = 1 << 15;

const Q: u64 = Q24 as u64;

const MAX_PRODUCT: u128 = (Q as u128 - 1) * (Q as u128 - 1);
// q - 1 + 2^15 (q - 1)^2 < 2^63: no u64 lane wraps between reductions.
const _: () = assert!(Q as u128 - 1 + Q24_REDUCTION_TERMS as u128 * MAX_PRODUCT < 1 << 63);
// Every three-byte offset word decodes inside (-q, q): the smallest word decodes to
// -IMAGE_OFFSET and the largest to 2^24 - 1 - IMAGE_OFFSET.
const _: () = assert!(IMAGE_OFFSET < Q24 as i32);
const _: () = assert!((1 << 24) - 1 - IMAGE_OFFSET < Q24 as i32);

/// The kernel for one accumulator, fixed at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Q24Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Ifma,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Neon,
}

#[cfg(target_arch = "x86_64")]
fn ifma_available() -> bool {
    crate::ntt::ifma52::ifma52_available() && std::is_x86_feature_detected!("avx2")
}

/// Derive the kernel from the domain's existing runtime backend choice.
fn q24_backend(backend: Backend) -> Q24Backend {
    match backend {
        Backend::Scalar => Q24Backend::Scalar,
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 | Backend::Avx512VbmiBatch => {
            if ifma_available() {
                Q24Backend::Ifma
            } else {
                Q24Backend::Scalar
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        Backend::Neon => Q24Backend::Neon,
    }
}

#[cfg(test)]
fn available_q24_backends() -> Vec<Q24Backend> {
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            all(target_arch = "aarch64", target_feature = "neon")
        )),
        expect(unused_mut)
    )]
    let mut backends = vec![Q24Backend::Scalar];
    #[cfg(target_arch = "x86_64")]
    if ifma_available() {
        backends.push(Q24Backend::Ifma);
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    backends.push(Q24Backend::Neon);
    backends
}

/// Canonical residue in `0..q` of one three-byte offset word.
#[inline(always)]
fn residue(bytes: [u8; 3]) -> u64 {
    let x = Q24Image::from_bytes(bytes).centered();
    // |x| < q, so one conditional addition gives the canonical residue.
    (if x < 0 { x + Q24 as i32 } else { x }) as u64
}

/// The 648 slots of a q24 [`TrinomialLimbSlots`] in three bytes each.
///
/// Slot `i` occupies bytes `3i..3i+3` as a [`Q24Image`] offset word. Packing
/// from slots stores canonical centered values; [`Self::from_bytes`] accepts
/// any word, each of which decodes to a valid lazy slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrinomialQ24PackedSlots {
    bytes: [u8; Q24_PACKED_SLOT_BYTES],
}

impl TrinomialQ24PackedSlots {
    /// Pack q24 slots, centering each lazy slot; reject any other prime.
    pub fn from_slots(slots: &TrinomialLimbSlots) -> Result<Self, TrinomialError> {
        if slots.prime != Q24 {
            return Err(TrinomialError::LimbInput {
                reason: "packed slots require the q24 prime",
            });
        }
        let arithmetic = Arithmetic { prime: Q24 };
        let mut bytes = [0; Q24_PACKED_SLOT_BYTES];
        for (&value, chunk) in slots.values.iter().zip(bytes.chunks_exact_mut(3)) {
            let centered = arithmetic.center(i64::from(value));
            let Ok(image) = Q24Image::from_centered(centered) else {
                return Err(TrinomialError::LimbInput {
                    reason: "centered q24 slot is outside the image interval",
                });
            };
            chunk.copy_from_slice(&image.bytes());
        }
        Ok(Self { bytes })
    }

    /// Wrap packed bytes. Every three-byte word is a valid packed slot.
    pub const fn from_bytes(bytes: [u8; Q24_PACKED_SLOT_BYTES]) -> Self {
        Self { bytes }
    }

    /// The packed bytes, three little-endian bytes per slot.
    pub const fn as_bytes(&self) -> &[u8; Q24_PACKED_SLOT_BYTES] {
        &self.bytes
    }

    /// Decode into q24 slots; reject an output tagged with another prime.
    ///
    /// The output holds the decoded representatives, which are canonical
    /// centered values whenever the bytes came from [`Self::from_slots`].
    pub fn to_slots(&self, out: &mut TrinomialLimbSlots) -> Result<(), TrinomialError> {
        if out.prime != Q24 {
            return Err(TrinomialError::LimbInput {
                reason: "packed slots require the q24 prime",
            });
        }
        for (value, chunk) in out.values.iter_mut().zip(self.bytes.chunks_exact(3)) {
            *value = Q24Image::from_bytes([chunk[0], chunk[1], chunk[2]]).centered();
        }
        Ok(())
    }
}

/// Product sums of packed q24 slots, reduced once per `2^15` products.
///
/// For any sequence of products, [`Self::finish`] writes the same values as
/// [`TrinomialLimbAccumulator::finish`](super::TrinomialLimbAccumulator::finish)
/// after the same products of the unpacked slots.
#[derive(Clone, Debug)]
pub struct TrinomialQ24Accumulator {
    backend: Q24Backend,
    sums: [u64; DEGREE],
    terms: u32,
}

impl TrinomialQ24Accumulator {
    /// Create an empty accumulator for a q24 domain; reject other primes.
    pub fn new(domain: &TrinomialLimbDomain) -> Result<Self, TrinomialError> {
        if domain.prime() != Q24 {
            return Err(TrinomialError::LimbInput {
                reason: "packed slots require the q24 prime",
            });
        }
        Ok(Self {
            backend: q24_backend(domain.backend),
            sums: [0; DEGREE],
            terms: 0,
        })
    }

    /// Add one slot-wise product.
    pub fn add_product(&mut self, lhs: &TrinomialQ24PackedSlots, rhs: &TrinomialQ24PackedSlots) {
        self.accumulate(core::slice::from_ref(lhs), core::slice::from_ref(rhs));
    }

    /// Add the products of equal-length batches.
    ///
    /// Unequal lengths are rejected and leave the accumulator unchanged.
    pub fn add_product_batch(
        &mut self,
        lhs: &[TrinomialQ24PackedSlots],
        rhs: &[TrinomialQ24PackedSlots],
    ) -> Result<(), TrinomialError> {
        if lhs.len() != rhs.len() {
            return Err(TrinomialError::LimbInput {
                reason: "product batch lengths do not match",
            });
        }
        self.accumulate(lhs, rhs);
        Ok(())
    }

    fn accumulate(
        &mut self,
        mut lhs: &[TrinomialQ24PackedSlots],
        mut rhs: &[TrinomialQ24PackedSlots],
    ) {
        while !lhs.is_empty() {
            // Stop each raw slice at the next reduction point.
            let count = lhs.len().min((Q24_REDUCTION_TERMS - self.terms) as usize);
            let (left, remaining_left) = lhs.split_at(count);
            let (right, remaining_right) = rhs.split_at(count);
            match self.backend {
                #[cfg(target_arch = "x86_64")]
                Q24Backend::Ifma => {
                    // SAFETY: This backend is selected only after detecting
                    // AVX2, AVX-512F/DQ and AVX-512 IFMA. The count keeps every
                    // sum below 2^63 (module bound).
                    unsafe { ifma::accumulate_batch(&mut self.sums, left, right) };
                }
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                Q24Backend::Neon => neon::accumulate_batch(&mut self.sums, left, right),
                Q24Backend::Scalar => accumulate_batch(&mut self.sums, left, right),
            }
            self.advance(count as u32);
            lhs = remaining_left;
            rhs = remaining_right;
        }
    }

    fn advance(&mut self, terms: u32) {
        self.terms += terms;
        if self.terms == Q24_REDUCTION_TERMS {
            for sum in &mut self.sums {
                *sum %= Q;
            }
            self.terms = 0;
        }
    }

    /// Write the centered sums and clear them; reject a non-q24 output.
    ///
    /// A rejected output leaves both the output and the sums unchanged.
    pub fn finish(&mut self, out: &mut TrinomialLimbSlots) -> Result<(), TrinomialError> {
        if out.prime != Q24 {
            return Err(TrinomialError::LimbInput {
                reason: "output prime does not match the accumulator",
            });
        }
        let half = Q / 2;
        for (value, &sum) in out.values.iter_mut().zip(&self.sums) {
            let residue = sum % Q;
            // residue < q < 2^24, so both branches fit i32.
            *value = if residue > half {
                residue as i32 - Q24 as i32
            } else {
                residue as i32
            };
        }
        self.sums.fill(0);
        self.terms = 0;
        Ok(())
    }
}

/// Portable kernel: the reference order for every SIMD kernel.
fn accumulate_batch(
    sums: &mut [u64; DEGREE],
    lhs: &[TrinomialQ24PackedSlots],
    rhs: &[TrinomialQ24PackedSlots],
) {
    for (a, b) in lhs.iter().zip(rhs) {
        let words = a.bytes.chunks_exact(3).zip(b.bytes.chunks_exact(3));
        for (sum, (a, b)) in sums.iter_mut().zip(words) {
            *sum += residue([a[0], a[1], a[2]]) * residue([b[0], b[1], b[2]]);
        }
    }
}
