//! Persistent structure-of-arrays storage and sumcheck kernels.

use std::sync::OnceLock;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{product, BinaryField162 as F};

// Even-sized chunks preserve adjacent pairs and are large enough to retain
// the selected SIMD backend's amortized reduction costs.
#[cfg(feature = "parallel")]
const PARALLEL_CHUNK: usize = 16_384;

#[cfg(target_arch = "aarch64")]
pub(super) mod arm;
#[cfg(target_arch = "x86_64")]
pub(super) mod x86;
#[cfg(target_arch = "x86_64")]
pub(super) mod x86_512;

type RoundKernel = fn(&LimbView<&[u64]>, &LimbView<&[u64]>, F) -> [F; 3];
// Fold the first ceil(n/2) entries without changing the view length.
type FoldKernel = fn(&mut LimbView<&mut [u64]>, F);

/// Borrowed limbs with equal lengths, established once at construction.
struct LimbView<T> {
    words: [T; 3],
}

impl<T: AsRef<[u64]>> LimbView<T> {
    fn new(words: [T; 3]) -> Self {
        assert_eq!(words[0].as_ref().len(), words[1].as_ref().len());
        assert_eq!(words[0].as_ref().len(), words[2].as_ref().len());
        Self { words }
    }

    fn len(&self) -> usize {
        self.words[0].as_ref().len()
    }

    #[inline(always)]
    fn element(&self, index: usize) -> F {
        F(self.words.each_ref().map(|words| words.as_ref()[index]))
    }
}

impl<T: AsRef<[u64]> + AsMut<[u64]>> LimbView<T> {
    #[inline(always)]
    fn set_element(&mut self, index: usize, value: F) {
        for (dst, word) in self.words.iter_mut().zip(value.to_words()) {
            dst.as_mut()[index] = word;
        }
    }
}

/// An owning, reusable structure-of-arrays buffer of binary field elements.
///
/// Keeping each coefficient word contiguous lets packed carryless-multiply
/// kernels load the same word from multiple field elements without gathering.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedBinary162 {
    words: [Vec<u64>; 3],
}

impl PackedBinary162 {
    /// Construct an empty reusable buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy scalar field elements into structure-of-arrays storage.
    pub fn from_scalars(values: &[F]) -> Self {
        let mut packed = Self::new();
        packed.refill(values);
        packed
    }

    /// Replace the contents while retaining existing allocations when possible.
    pub fn refill(&mut self, values: &[F]) {
        for words in &mut self.words {
            words.clear();
            words.reserve(values.len());
        }
        for value in values {
            let words = value.to_words();
            for (dst, word) in self.words.iter_mut().zip(words) {
                dst.push(word);
            }
        }
    }

    /// Pack binary source words directly into reusable F162 storage.
    ///
    /// Each source word supplies at most 128 low polynomial coefficients.
    /// This preserves binary addition, not source-field multiplication.
    pub fn refill_binary_words<T: Copy + Into<u128>>(&mut self, values: &[T]) {
        let [low, high, top] = self.resize_words(values.len());
        for ((lo, hi), &value) in low.iter_mut().zip(high).zip(values) {
            let value: u128 = value.into();
            *lo = value as u64;
            *hi = (value >> 64) as u64;
        }
        top.fill(0);
    }

    // Internal writers must initialize all three words and keep word 2
    // canonical. Equal lengths and retained capacity are owned here.
    pub(super) fn resize_words(&mut self, len: usize) -> [&mut [u64]; 3] {
        for words in &mut self.words {
            words.resize(len, 0);
        }
        let [low, high, top] = &mut self.words;
        [low, high, top]
    }

    /// Return the number of stored field elements.
    pub fn len(&self) -> usize {
        self.words[0].len()
    }

    /// Return whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.words[0].is_empty()
    }

    /// Return one scalar element, or `None` when `index` is out of bounds.
    pub fn get(&self, index: usize) -> Option<F> {
        (index < self.len()).then(|| self.element(index))
    }

    /// Copy all packed elements back to scalar storage.
    pub fn to_scalars(&self) -> Vec<F> {
        (0..self.len()).map(|index| self.element(index)).collect()
    }

    /// Compute the degree-two sumcheck round polynomial in coefficient form.
    ///
    /// Adjacent entries `(a0, a1)` and `(b0, b1)` contribute
    /// `(a0 + X(a0 + a1)) (b0 + X(b0 + b1))`. A missing final odd entry is
    /// zero. The result is `None` when the buffer lengths differ or fewer than
    /// two evaluations remain, because those inputs do not define a next fold
    /// round. `current_claim` must be the caller's prover-state value
    /// `g(0) + g(1)`; this arithmetic kernel does not authenticate that hint.
    /// It computes only the constant and quadratic coefficients, accumulating
    /// their products before reduction, then derives the linear coefficient as
    /// `current_claim + quadratic` in characteristic two.
    pub fn round_product(&self, rhs: &Self, current_claim: F) -> Option<[F; 3]> {
        (self.len() == rhs.len() && self.len() >= 2).then(|| {
            #[cfg(feature = "parallel")]
            if self.len() > PARALLEL_CHUNK {
                let round = self.words[0]
                    .par_chunks(PARALLEL_CHUNK)
                    .zip(self.words[1].par_chunks(PARALLEL_CHUNK))
                    .zip(self.words[2].par_chunks(PARALLEL_CHUNK))
                    .zip(rhs.words[0].par_chunks(PARALLEL_CHUNK))
                    .zip(rhs.words[1].par_chunks(PARALLEL_CHUNK))
                    .zip(rhs.words[2].par_chunks(PARALLEL_CHUNK))
                    .map(|(((((a0, a1), a2), b0), b1), b2)| {
                        let lhs = LimbView::new([a0, a1, a2]);
                        let rhs = LimbView::new([b0, b1, b2]);
                        (kernels().round_product)(&lhs, &rhs, F::ZERO)
                    })
                    .reduce(
                        || [F::ZERO; 3],
                        |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                    );
                // The claim belongs to the whole round, not each chunk.
                return [round[0], current_claim + round[2], round[2]];
            }
            (kernels().round_product)(
                &LimbView::new(self.words.each_ref().map(|words| words.as_slice())),
                &LimbView::new(rhs.words.each_ref().map(|words| words.as_slice())),
                current_claim,
            )
        })
    }

    /// Fold adjacent evaluations at `r` and shrink to `ceil(len / 2)`.
    ///
    /// A missing final odd entry is zero. Buffers of length zero or one are
    /// already terminal and are left unchanged.
    pub fn fold_in_place(&mut self, r: F) {
        if self.len() > 1 {
            #[cfg(feature = "parallel")]
            if self.len() > PARALLEL_CHUNK {
                let old_len = self.len();
                let [low, high, top] = &mut self.words;
                low.par_chunks_mut(PARALLEL_CHUNK)
                    .zip(high.par_chunks_mut(PARALLEL_CHUNK))
                    .zip(top.par_chunks_mut(PARALLEL_CHUNK))
                    .for_each(|((low, high), top)| {
                        // Even chunk widths preserve pairs; a final singleton
                        // is nonterminal table input and must fold with zero.
                        (kernels().fold_in_place)(&mut LimbView::new([low, high, top]), r);
                    });
                // All chunks have finished reading before compaction. For
                // chunk k of even width C and live length L <= C/2, its
                // destination ends at k*C/2 + L <= (k+1)*C/2 <= (k+1)*C,
                // the next source's start. Ascending copies therefore cannot
                // overwrite any later live source; copy_within handles self-overlap.
                for source in (PARALLEL_CHUNK..old_len).step_by(PARALLEL_CHUNK) {
                    let live = (old_len - source).min(PARALLEL_CHUNK).div_ceil(2);
                    for words in &mut self.words {
                        words.copy_within(source..source + live, source / 2);
                    }
                }
                self.truncate(old_len.div_ceil(2));
                return;
            }
            let new_len = self.len().div_ceil(2);
            (kernels().fold_in_place)(
                &mut LimbView::new(self.words.each_mut().map(|words| words.as_mut_slice())),
                r,
            );
            self.truncate(new_len);
        }
    }

    #[inline(always)]
    fn element(&self, index: usize) -> F {
        F([
            self.words[0][index],
            self.words[1][index],
            self.words[2][index],
        ])
    }

    fn truncate(&mut self, len: usize) {
        for words in &mut self.words {
            words.truncate(len);
        }
    }
}

struct PackedKernels {
    round_product: RoundKernel,
    fold_in_place: FoldKernel,
}

fn kernels() -> &'static PackedKernels {
    static KERNELS: OnceLock<PackedKernels> = OnceLock::new();
    KERNELS.get_or_init(detect)
}

fn detect() -> PackedKernels {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes")
        && std::arch::is_aarch64_feature_detected!("pmull")
    {
        return PackedKernels {
            round_product: |a, b, current_claim| {
                // SAFETY: PMULL was detected. LimbView::new checks equal limb
                // lengths; the public round caller checks matching table lengths.
                // Backend loops bound offsets by view.len().
                unsafe { arm::round_product(a, b, current_claim) }
            },
            fold_in_place: |values, r| {
                // SAFETY: PMULL was detected. LimbView::new checks equal limb
                // lengths; the backend bounds every access by view.len().
                unsafe { arm::fold_in_place(values, r) }
            },
        };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("vpclmulqdq")
        {
            return PackedKernels {
                round_product: |a, b, claim| {
                    // SAFETY: required features were detected. LimbView::new checks
                    // equal limb lengths; the public round caller checks matching
                    // table lengths. Backend loops bound offsets by view.len().
                    unsafe {
                        if a.len() >= 8 {
                            x86_512::round_product_vec4(a, b, claim)
                        } else {
                            x86::round_product_vec2(a, b, claim)
                        }
                    }
                },
                fold_in_place: |values, r| {
                    // SAFETY: required features were detected. LimbView::new checks
                    // equal limb lengths; backend loops bound offsets by view.len().
                    unsafe {
                        if values.len() >= 8 {
                            x86_512::fold_in_place_vec4(values, r)
                        } else {
                            x86::fold_in_place_vec2(values, r)
                        }
                    }
                },
            };
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("vpclmulqdq")
        {
            return PackedKernels {
                round_product: |a, b, current_claim| {
                    // SAFETY: required features were detected. LimbView::new checks
                    // equal limb lengths; the public round caller checks matching
                    // table lengths. Backend loops bound offsets by view.len().
                    unsafe { x86::round_product_vec2(a, b, current_claim) }
                },
                fold_in_place: |values, r| {
                    // SAFETY: required features were detected. LimbView::new checks
                    // equal limb lengths; backend loops bound offsets by view.len().
                    unsafe { x86::fold_in_place_vec2(values, r) }
                },
            };
        }
        return PackedKernels {
            round_product: |a, b, current_claim| {
                // SAFETY: PCLMUL was detected. LimbView::new checks equal limb
                // lengths; the public round caller checks matching table lengths.
                // Backend loops bound offsets by view.len().
                unsafe { x86::round_product(a, b, current_claim) }
            },
            fold_in_place: |values, r| {
                // SAFETY: PCLMUL was detected. LimbView::new checks equal limb
                // lengths; the backend bounds every access by view.len().
                unsafe { x86::fold_in_place(values, r) }
            },
        };
    }
    PackedKernels {
        round_product: portable_round_product,
        fold_in_place: portable_fold_in_place,
    }
}

fn xor_product(sum: &mut [u64; 6], product: [u64; 6]) {
    for (sum, term) in sum.iter_mut().zip(product) {
        *sum ^= term;
    }
}

fn round_product_with(
    lhs: &LimbView<&[u64]>,
    rhs: &LimbView<&[u64]>,
    current_claim: F,
    product_fn: impl Fn(F, F) -> [u64; 6],
) -> [F; 3] {
    let mut sums = [[0; 6]; 2];
    let mut index = 0;
    while index < lhs.len() {
        let a0 = lhs.element(index);
        let b0 = rhs.element(index);
        xor_product(&mut sums[0], product_fn(a0, b0));
        if index + 1 < lhs.len() {
            let a1 = lhs.element(index + 1);
            let b1 = rhs.element(index + 1);
            xor_product(&mut sums[1], product_fn(a0 + a1, b0 + b1));
        } else {
            xor_product(&mut sums[1], product_fn(a0, b0));
        }
        index += 2;
    }
    let [constant, quadratic] = sums.map(product::reduce);
    [constant, current_claim + quadratic, quadratic]
}

fn fold_with(values: &mut LimbView<&mut [u64]>, r: F, multiply: impl Fn(F, F) -> F) {
    let old_len = values.len();
    let new_len = old_len.div_ceil(2);
    for output in 0..new_len {
        let input = 2 * output;
        let even = values.element(input);
        let odd = if input + 1 < old_len {
            values.element(input + 1)
        } else {
            F::ZERO
        };
        values.set_element(output, even + multiply(even + odd, r));
    }
}

fn portable_round_product(
    lhs: &LimbView<&[u64]>,
    rhs: &LimbView<&[u64]>,
    current_claim: F,
) -> [F; 3] {
    round_product_with(lhs, rhs, current_claim, product::portable_product)
}

fn portable_fold_in_place(values: &mut LimbView<&mut [u64]>, r: F) {
    fold_with(values, r, product::portable_multiply);
}

#[cfg(all(test, feature = "parallel"))]
mod parallel_tests;
#[cfg(test)]
mod tests;
