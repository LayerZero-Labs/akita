//! Persistent structure-of-arrays storage and sumcheck kernels.

use std::sync::OnceLock;

use super::{product, BinaryField162 as F};

#[cfg(target_arch = "aarch64")]
pub(super) mod arm;
#[cfg(target_arch = "x86_64")]
pub(super) mod x86;

type RoundKernel = fn(&PackedBinary162, &PackedBinary162, F) -> [F; 3];
type FoldKernel = fn(&mut PackedBinary162, F);

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
        (self.len() == rhs.len() && self.len() >= 2)
            .then(|| (kernels().round_product)(self, rhs, current_claim))
    }

    /// Fold adjacent evaluations at `r` and shrink to `ceil(len / 2)`.
    ///
    /// A missing final odd entry is zero. Buffers of length zero or one are
    /// already terminal and are left unchanged.
    pub fn fold_in_place(&mut self, r: F) {
        if self.len() > 1 {
            (kernels().fold_in_place)(self, r);
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

    #[inline(always)]
    fn set_element(&mut self, index: usize, value: F) {
        let words = value.to_words();
        for (dst, word) in self.words.iter_mut().zip(words) {
            dst[index] = word;
        }
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
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm::round_product(a, b, current_claim) }
            },
            fold_in_place: |values, r| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm::fold_in_place(values, r) }
            },
        };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("vpclmulqdq")
        {
            return PackedKernels {
                round_product: |a, b, current_claim| {
                    // SAFETY: all required vector features were detected.
                    unsafe { x86::round_product_vec2(a, b, current_claim) }
                },
                fold_in_place: |values, r| {
                    // SAFETY: all required vector features were detected.
                    unsafe { x86::fold_in_place_vec2(values, r) }
                },
            };
        }
        return PackedKernels {
            round_product: |a, b, current_claim| {
                // SAFETY: this closure is installed only after detecting PCLMUL.
                unsafe { x86::round_product(a, b, current_claim) }
            },
            fold_in_place: |values, r| {
                // SAFETY: this closure is installed only after detecting PCLMUL.
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

pub(super) fn round_product_with(
    lhs: &PackedBinary162,
    rhs: &PackedBinary162,
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

pub(super) fn fold_with(values: &mut PackedBinary162, r: F, multiply: impl Fn(F, F) -> F) {
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
    values.truncate(new_len);
}

pub(super) fn portable_round_product(
    lhs: &PackedBinary162,
    rhs: &PackedBinary162,
    current_claim: F,
) -> [F; 3] {
    round_product_with(lhs, rhs, current_claim, product::portable_product)
}

pub(super) fn portable_fold_in_place(values: &mut PackedBinary162, r: F) {
    fold_with(values, r, product::portable_multiply);
}

#[cfg(test)]
mod tests;
