//! Jolt RISC-V guest backend: the 64-point `i32` forward NTT and six-product
//! pointwise-dot inlines of `jolt-inlines-ntt` (the `ntt-inline` feature).
//!
//! Shared arithmetic selects this backend through [`selected`] and hands it
//! Akita's generic limb types. This module alone owns the width/degree proof,
//! the layout casts to the inlines' `[i32; 64]` arrays, the doubleword packing
//! of centered digits, zero padding of short dots, and the alignment contract
//! under which the inlines take their fast path. Inputs it cannot serve go
//! through the inlines' own portable fallbacks, which compute the same values.
//!
//! Off riscv64 the module compiles only for tests; `jolt-inlines-ntt` then
//! runs its portable fallbacks, so host tests pin the table layout, packing,
//! padding, and alignment handling.

use jolt_inlines_ntt::{forward_ntt64, pointwise_dot64, DEGREE, DOT_PRODUCTS};

use super::butterfly::NttTwiddles;
use super::prime::{MontCoeff, NttPrime, PrimeWidth, I32_LAZY_DOT_BATCH};
#[cfg(target_arch = "riscv64")]
use super::NttKernelPlan;

// Callers bound every lazy dot by `I32_LAZY_DOT_BATCH`; one inline call must
// cover a full batch.
const _: () = assert!(DOT_PRODUCTS == I32_LAZY_DOT_BATCH);

/// Whether `plan` selects this backend and it serves width `W` and degree `D`.
#[cfg(target_arch = "riscv64")]
#[inline(always)]
pub(crate) const fn selected<W: PrimeWidth, const D: usize>(plan: NttKernelPlan) -> bool {
    plan.uses_jolt_inline() && serves::<W, D>()
}

/// The single layout the inlines accept: `D = 64` residues of width `i32`.
/// `PrimeWidth` is sealed to `i16` (`R_LOG = 16`) and `i32` (`R_LOG = 32`).
#[inline(always)]
const fn serves<W: PrimeWidth, const D: usize>() -> bool {
    W::R_LOG == 32 && D == DEGREE
}

/// Doubleword-aligned storage, so the inlines' paired loads apply.
#[repr(C, align(8))]
struct Aligned([i32; DEGREE]);

/// Padding rows for dots shorter than [`DOT_PRODUCTS`]: zero products leave
/// the accumulator's value unchanged.
static ZERO_ROW: Aligned = Aligned([0; DEGREE]);

/// `limb` as the inlines' raw `[i32; 64]`.
///
/// # Panics
///
/// Panics unless [`serves`] holds; the check folds away per instantiation.
#[inline(always)]
fn residues<W: PrimeWidth, const D: usize>(limb: &[MontCoeff<W>; D]) -> &[i32; DEGREE] {
    assert!(
        serves::<W, D>(),
        "Jolt NTT inline serves only 64 i32 residues"
    );
    // SAFETY: `serves` pins `W = i32` (the sealed width with `R_LOG = 32`) and
    // `D = 64`; `MontCoeff` is `repr(transparent)`, so the array is 64
    // contiguous `i32`s with `i32` alignment.
    unsafe { &*limb.as_ptr().cast::<[i32; DEGREE]>() }
}

/// Mutable form of [`residues`].
#[inline(always)]
fn residues_mut<W: PrimeWidth, const D: usize>(limb: &mut [MontCoeff<W>; D]) -> &mut [i32; DEGREE] {
    assert!(
        serves::<W, D>(),
        "Jolt NTT inline serves only 64 i32 residues"
    );
    // SAFETY: as in `residues`; the exclusive borrow carries over.
    unsafe { &mut *limb.as_mut_ptr().cast::<[i32; DEGREE]>() }
}

/// A raw `[W; D]` table as `[i32; 64]`.
#[inline(always)]
fn raw_table<W: PrimeWidth, const D: usize>(table: &[W; D]) -> &[i32; DEGREE] {
    assert!(
        serves::<W, D>(),
        "Jolt NTT inline serves only 64 i32 residues"
    );
    // SAFETY: `serves` pins `W = i32` and `D = 64`.
    unsafe { &*table.as_ptr().cast::<[i32; DEGREE]>() }
}

/// `p` and `p^-1 mod 2^32` at the inlines' `i32` width.
#[inline(always)]
fn modulus<W: PrimeWidth>(prime: NttPrime<W>) -> (i32, i32) {
    (prime.p.to_i64() as i32, prime.pinv.to_i64() as i32)
}

/// Twisted forward negacyclic NTT of one limb, bit-reversed output, matching
/// [`super::butterfly::forward_ntt`].
///
/// # Panics
///
/// Panics unless `W = i32` and `D = 64`.
#[inline(always)]
pub(crate) fn forward_ntt<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let (p, pinv) = modulus(prime);
    forward_ntt64(
        residues_mut(a),
        residues(&tw.psi_pows),
        residues(&tw.fwd_twiddles),
        p,
        pinv,
    );
}

/// Centered `i16` digits prepared once for [`forward_ntt_centered_i16`]
/// across every CRT limb.
pub(crate) struct CenteredI16Digits<'a, const D: usize> {
    digits: &'a [i16; D],
    /// Sign-extended `i32` residues, two per little-endian doubleword. A Jolt
    /// guest expands each 32-bit store into a multi-row sequence, so aligned
    /// limbs take doubleword stores.
    words: [u64; DEGREE / 2],
}

impl<'a, const D: usize> CenteredI16Digits<'a, D> {
    /// # Panics
    ///
    /// Panics unless `D = 64`.
    #[inline(always)]
    pub(crate) fn new(digits: &'a [i16; D]) -> Self {
        assert!(D == DEGREE, "Jolt NTT inline serves only 64 coefficients");
        let residue = |value: i16| u64::from(i32::from(value) as u32);
        let words = std::array::from_fn(|index| {
            residue(digits[2 * index]) | (residue(digits[2 * index + 1]) << 32)
        });
        Self { digits, words }
    }
}

/// Overwrite `out` with the CRT+NTT image of centered digits for one prime.
///
/// The twist table `psi^i * R^2` folds conversion of the signed digits into
/// Montgomery form into the first product, so the digits enter as plain
/// sign-extended residues.
///
/// # Panics
///
/// Panics unless `W = i32` and `D = 64`.
#[inline(always)]
pub(crate) fn forward_ntt_centered_i16<W: PrimeWidth, const D: usize>(
    out: &mut [MontCoeff<W>; D],
    digits: &CenteredI16Digits<'_, D>,
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let state = residues_mut(out);
    let words = state.as_mut_ptr().cast::<u64>();
    // The packed words assume little-endian order; big-endian test hosts and
    // limbs off a doubleword boundary take the element-wise stores.
    if cfg!(target_endian = "little") && (words as usize).is_multiple_of(8) {
        // SAFETY: `state` is 64 `i32`s, 256 bytes, the size of `[u64; 32]`,
        // and the check above makes the address doubleword aligned.
        unsafe { words.cast::<[u64; DEGREE / 2]>().write(digits.words) };
    } else {
        for (slot, &value) in state.iter_mut().zip(digits.digits) {
            *slot = i32::from(value);
        }
    }
    let (p, pinv) = modulus(prime);
    forward_ntt64(
        state,
        raw_table(&tw.psi_pows_r2),
        residues(&tw.fwd_twiddles),
        p,
        pinv,
    );
}

/// `acc[i] += Σ_{j < count} lhs(j)[i] · rhs(j)[i] · 2^-32 mod p`, the lazy
/// Montgomery dot the SIMD backends' `pointwise_dot_acc_i32` computes.
///
/// Rows past `count` are padded with an aligned zero row. Rows and
/// accumulators off a doubleword boundary make the inline run its portable
/// loop, which computes the same values.
///
/// # Panics
///
/// Panics unless `W = i32` and `D = 64`.
#[inline(always)]
pub(crate) fn pointwise_dot_acc<'a, W, const D: usize>(
    acc: &mut [MontCoeff<W>; D],
    lhs: impl Fn(usize) -> &'a [MontCoeff<W>; D],
    rhs: impl Fn(usize) -> &'a [MontCoeff<W>; D],
    count: usize,
    prime: NttPrime<W>,
) where
    W: PrimeWidth,
{
    debug_assert!(count <= DOT_PRODUCTS);
    let (p, pinv) = modulus(prime);
    pointwise_dot64(
        residues_mut(acc),
        padded_rows(lhs, count),
        padded_rows(rhs, count),
        p,
        pinv,
    );
}

#[inline(always)]
fn padded_rows<'a, W: PrimeWidth, const D: usize>(
    row: impl Fn(usize) -> &'a [MontCoeff<W>; D],
    count: usize,
) -> [&'a [i32; DEGREE]; DOT_PRODUCTS] {
    std::array::from_fn(|index| {
        if index < count {
            residues(row(index))
        } else {
            &ZERO_ROW.0
        }
    })
}
