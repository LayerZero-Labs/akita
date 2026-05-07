//! D-agnostic flat vector storage with typed ring-element views.
//!
//! [`FlatMatrix`] stores ring elements as raw field elements in a single
//! contiguous 1D vector, independent of any ring dimension. Each role
//! (A, B, D) interprets a prefix of this vector as its own matrix with
//! role-specific `(num_rows, num_cols)` dimensions.
//!
//! A [`RingMatrixView`] borrows a prefix of the flat data and interprets it
//! as a `rows × cols` matrix of `CyclotomicRing<F, D>` elements, enabling
//! the same underlying vector to serve multiple roles with different shapes.

use akita_algebra::CyclotomicRing;
use akita_field::FieldCore;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// Flat 1D vector of field elements, independent of ring dimension.
///
/// Stores `total_ring_elements * gen_ring_dim` contiguous field elements.
/// Each role matrix (A, B, D) views a prefix of this vector reshaped into
/// its own `(num_rows, num_cols)` dimensions via [`RingMatrixView`].
///
/// Any prefix of a uniformly random vector is uniformly random, so role matrices
/// derived from prefixes of the same flat vector are binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatMatrix<F: FieldCore> {
    data: Vec<F>,
    /// Ring dimension used when generating (D_max).
    gen_ring_dim: usize,
}

impl<F: FieldCore> FlatMatrix<F> {
    /// Total number of ring elements at the generation dimension.
    #[inline]
    pub fn total_ring_elements(&self) -> usize {
        if self.gen_ring_dim == 0 {
            0
        } else {
            self.data.len() / self.gen_ring_dim
        }
    }

    /// Ring dimension used during generation.
    #[inline]
    pub fn gen_ring_dim(&self) -> usize {
        self.gen_ring_dim
    }

    /// Total number of ring elements when viewed at dimension D.
    #[inline]
    pub fn total_ring_elements_at<const D: usize>(&self) -> usize {
        debug_assert!(D > 0 && self.gen_ring_dim.is_multiple_of(D));
        self.total_ring_elements() * (self.gen_ring_dim / D)
    }

    /// Build from pre-flattened field-element data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `gen_ring_dim`.
    pub fn from_flat_data(data: Vec<F>, gen_ring_dim: usize) -> Self {
        debug_assert!(
            gen_ring_dim > 0 && data.len().is_multiple_of(gen_ring_dim),
            "data length {} must be a positive multiple of gen_ring_dim={}",
            data.len(),
            gen_ring_dim,
        );
        Self { data, gen_ring_dim }
    }

    /// Build from a flat slice of ring elements.
    pub fn from_ring_slice<const D: usize>(elements: &[CyclotomicRing<F, D>]) -> Self {
        let mut data = Vec::with_capacity(elements.len() * D);
        for ring_elem in elements {
            data.extend_from_slice(&ring_elem.coeffs);
        }
        Self {
            data,
            gen_ring_dim: D,
        }
    }

    /// Create a typed matrix view at ring dimension D with the given shape.
    ///
    /// The view interprets the first `num_rows * num_cols` ring elements
    /// (at dimension D) as a `num_rows × num_cols` matrix.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not divide `gen_ring_dim` or if the requested
    /// shape exceeds the available data.
    pub fn ring_view<const D: usize>(
        &self,
        num_rows: usize,
        num_cols: usize,
    ) -> RingMatrixView<'_, F, D> {
        assert!(D > 0, "ring dimension must be positive");
        assert!(
            self.gen_ring_dim.is_multiple_of(D),
            "D={D} does not divide gen_ring_dim={}",
            self.gen_ring_dim
        );
        let total_at_d = self.total_ring_elements_at::<D>();
        let needed = num_rows * num_cols;
        assert!(
            needed <= total_at_d,
            "requested {num_rows}×{num_cols}={needed} ring elements at D={D}, \
             but only {total_at_d} available"
        );
        let field_len = needed * D;
        RingMatrixView {
            data: &self.data[..field_len],
            num_rows,
            num_cols,
        }
    }

    /// Create a multilinear-polynomial view of a shared setup matrix prefix.
    ///
    /// The resulting view interprets the selected `num_rows × num_cols` ring
    /// elements as `S(row, col, coeff)`, where `coeff` ranges over `0..D`.
    /// Non-power-of-two row and column dimensions are zero-padded by the
    /// multilinear evaluator.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::ring_view`], and if rows,
    /// columns, or `D` are zero.
    pub fn setup_polynomial_view<const D: usize>(
        &self,
        num_rows: usize,
        num_cols: usize,
    ) -> SetupMatrixPolynomialView<'_, F, D> {
        assert!(num_rows > 0, "setup polynomial view requires rows");
        assert!(num_cols > 0, "setup polynomial view requires columns");
        assert!(
            D.is_power_of_two(),
            "setup polynomial D must be power of two"
        );
        SetupMatrixPolynomialView {
            view: self.ring_view::<D>(num_rows, num_cols),
        }
    }
}

impl<F: FieldCore + Valid> Valid for FlatMatrix<F> {
    fn check(&self) -> Result<(), SerializationError> {
        for f in &self.data {
            f.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for FlatMatrix<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.total_ring_elements()
            .serialize_with_mode(&mut writer, compress)?;
        self.gen_ring_dim
            .serialize_with_mode(&mut writer, compress)?;
        for f in &self.data {
            f.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        2 * std::mem::size_of::<usize>()
            + self
                .data
                .iter()
                .map(|f| f.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for FlatMatrix<F> {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let total_ring = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let gen_ring_dim = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let total_fields = total_ring * gen_ring_dim;
        let mut data = Vec::with_capacity(total_fields);
        for _ in 0..total_fields {
            data.push(F::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self { data, gen_ring_dim };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Typed read-only view of a [`FlatMatrix`] prefix at a specific ring
/// dimension D, interpreted as a `num_rows × num_cols` matrix.
///
/// Provides zero-copy access to rows as `&[CyclotomicRing<F, D>]` by
/// transmuting the underlying `&[F]` slice (safe because `CyclotomicRing`
/// is `#[repr(transparent)]` over `[F; D]`).
#[derive(Debug, Clone, Copy)]
pub struct RingMatrixView<'a, F: FieldCore, const D: usize> {
    data: &'a [F],
    num_rows: usize,
    num_cols: usize,
}

impl<'a, F: FieldCore, const D: usize> RingMatrixView<'a, F, D> {
    /// Number of rows in the view.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of ring-element columns per row.
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Borrow a single row as a slice of ring elements (zero-copy).
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows`.
    #[inline]
    pub fn row(&self, row: usize) -> &'a [CyclotomicRing<F, D>] {
        assert!(row < self.num_rows, "row {row} out of bounds");
        let row_field_len = self.num_cols * D;
        let start = row * row_field_len;
        let field_slice = &self.data[start..start + row_field_len];
        // SAFETY: CyclotomicRing<F, D> is #[repr(transparent)] over [F; D],
        // so a contiguous &[F] of length num_cols*D has the same layout as
        // &[CyclotomicRing<F, D>] of length num_cols.
        unsafe {
            std::slice::from_raw_parts(
                field_slice.as_ptr() as *const CyclotomicRing<F, D>,
                self.num_cols,
            )
        }
    }
}

/// Read-only multilinear view of a shared setup matrix prefix.
///
/// This is the direct `S(row, col, coeff)` surface used by the setup-side
/// claim-reduction plan. It is intentionally a view over the existing shared
/// matrix storage, so introducing it does not change setup generation or proof
/// formats.
#[derive(Debug, Clone, Copy)]
pub struct SetupMatrixPolynomialView<'a, F: FieldCore, const D: usize> {
    view: RingMatrixView<'a, F, D>,
}

impl<'a, F: FieldCore, const D: usize> SetupMatrixPolynomialView<'a, F, D> {
    /// Number of setup matrix rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.view.num_rows()
    }

    /// Number of setup matrix ring-element columns.
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.view.num_cols()
    }

    /// Number of Boolean variables needed to index rows after zero-padding.
    #[inline]
    pub fn row_bits(&self) -> usize {
        padded_bits(self.num_rows())
    }

    /// Number of Boolean variables needed to index columns after zero-padding.
    #[inline]
    pub fn col_bits(&self) -> usize {
        padded_bits(self.num_cols())
    }

    /// Number of Boolean variables needed to index ring coefficients.
    #[inline]
    pub fn coeff_bits(&self) -> usize {
        D.trailing_zeros() as usize
    }

    /// Return `S(row, col, coeff)`, or zero when a padded index is requested.
    ///
    /// # Panics
    ///
    /// Panics if `coeff >= D`. Rows and columns outside the live prefix are
    /// treated as zero-padding.
    #[inline]
    pub fn coeff(&self, row: usize, col: usize, coeff: usize) -> F {
        assert!(coeff < D, "coefficient {coeff} out of bounds for D={D}");
        if row >= self.num_rows() || col >= self.num_cols() {
            return F::zero();
        }
        self.view.row(row)[col].coeffs[coeff]
    }

    /// Directly evaluate the multilinear extension of `S(row, col, coeff)`.
    ///
    /// Row and column dimensions are padded to powers of two. Coefficients use
    /// exactly `log2(D)` variables.
    ///
    /// # Errors
    ///
    /// Returns an error if any challenge slice has the wrong length.
    pub fn mle(
        &self,
        row_challenges: &[F],
        col_challenges: &[F],
        coeff_challenges: &[F],
    ) -> Result<F, akita_field::AkitaError> {
        if row_challenges.len() != self.row_bits() {
            return Err(akita_field::AkitaError::InvalidSize {
                expected: self.row_bits(),
                actual: row_challenges.len(),
            });
        }
        if col_challenges.len() != self.col_bits() {
            return Err(akita_field::AkitaError::InvalidSize {
                expected: self.col_bits(),
                actual: col_challenges.len(),
            });
        }
        if coeff_challenges.len() != self.coeff_bits() {
            return Err(akita_field::AkitaError::InvalidSize {
                expected: self.coeff_bits(),
                actual: coeff_challenges.len(),
            });
        }

        let row_len = 1usize << self.row_bits();
        let col_len = 1usize << self.col_bits();
        let mut acc = F::zero();
        for row in 0..row_len {
            let row_weight = eq_weight_at_index(row_challenges, row);
            if row_weight.is_zero() {
                continue;
            }
            for col in 0..col_len {
                let col_weight = eq_weight_at_index(col_challenges, col);
                if col_weight.is_zero() {
                    continue;
                }
                let rc_weight = row_weight * col_weight;
                for coeff in 0..D {
                    let coeff_weight = eq_weight_at_index(coeff_challenges, coeff);
                    if !coeff_weight.is_zero() {
                        acc += rc_weight * coeff_weight * self.coeff(row, col, coeff);
                    }
                }
            }
        }
        Ok(acc)
    }
}

#[inline]
fn padded_bits(len: usize) -> usize {
    len.next_power_of_two().trailing_zeros() as usize
}

#[inline]
fn eq_weight_at_index<F: FieldCore>(challenges: &[F], index: usize) -> F {
    challenges
        .iter()
        .enumerate()
        .fold(F::one(), |acc, (bit_idx, &challenge)| {
            if (index >> bit_idx) & 1 == 1 {
                acc * challenge
            } else {
                acc * (F::one() - challenge)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{fields::Prime128Offset275, RandomSampling};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Prime128Offset275;

    #[test]
    fn ring_view_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let rows = 3usize;
        let cols = 5usize;
        let elements: Vec<CyclotomicRing<F, 64>> = (0..rows * cols)
            .map(|_| CyclotomicRing::random(&mut rng))
            .collect();

        let flat = FlatMatrix::from_ring_slice(&elements);
        assert_eq!(flat.total_ring_elements(), rows * cols);
        assert_eq!(flat.gen_ring_dim(), 64);

        let view = flat.ring_view::<64>(rows, cols);
        assert_eq!(view.num_rows(), rows);
        assert_eq!(view.num_cols(), cols);

        for r in 0..rows {
            let view_row = view.row(r);
            for c in 0..cols {
                assert_eq!(view_row[c], elements[r * cols + c]);
            }
        }
    }

    #[test]
    fn ring_view_at_smaller_d() {
        let mut rng = StdRng::seed_from_u64(99);
        let total = 8usize;
        let elements: Vec<CyclotomicRing<F, 64>> = (0..total)
            .map(|_| CyclotomicRing::random(&mut rng))
            .collect();

        let flat = FlatMatrix::from_ring_slice(&elements);
        assert_eq!(flat.total_ring_elements_at::<32>(), total * 2);

        let view32 = flat.ring_view::<32>(2, total);
        assert_eq!(view32.num_rows(), 2);
        assert_eq!(view32.num_cols(), total);

        let view64 = flat.ring_view::<64>(2, 4);
        for r in 0..2 {
            let row64 = view64.row(r);
            let row32 = view32.row(r);
            for (j, orig_ring) in row64.iter().enumerate() {
                let lo = &row32[j * 2];
                let hi = &row32[j * 2 + 1];
                assert_eq!(&orig_ring.coeffs[..32], lo.coefficients());
                assert_eq!(&orig_ring.coeffs[32..], hi.coefficients());
            }
        }
    }

    #[test]
    fn different_role_views_from_same_flat() {
        let mut rng = StdRng::seed_from_u64(7);
        let total = 64usize;
        let elements: Vec<CyclotomicRing<F, 64>> = (0..total)
            .map(|_| CyclotomicRing::random(&mut rng))
            .collect();

        let flat = FlatMatrix::from_ring_slice(&elements);

        let view_a = flat.ring_view::<64>(2, 16);
        let view_b = flat.ring_view::<64>(4, 8);

        assert_eq!(view_a.row(0)[0], elements[0]);
        assert_eq!(view_a.row(1)[0], elements[16]);
        assert_eq!(view_b.row(0)[0], elements[0]);
        assert_eq!(view_b.row(1)[0], elements[8]);
    }

    #[test]
    fn setup_polynomial_view_selects_coefficients_and_padding() {
        const D: usize = 4;
        let rows = 3usize;
        let cols = 2usize;
        let elements: Vec<CyclotomicRing<F, D>> = (0..rows * cols)
            .map(|idx| {
                let coeffs = std::array::from_fn(|coeff| F::from_u64((10 * idx + coeff) as u64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let flat = FlatMatrix::from_ring_slice(&elements);
        let view = flat.setup_polynomial_view::<D>(rows, cols);

        assert_eq!(view.row_bits(), 2);
        assert_eq!(view.col_bits(), 1);
        assert_eq!(view.coeff_bits(), 2);
        assert_eq!(view.coeff(2, 1, 3), elements[2 * cols + 1].coeffs[3]);
        assert_eq!(view.coeff(3, 1, 3), F::zero());

        let selected = view
            .mle(&[F::zero(), F::one()], &[F::one()], &[F::one(), F::one()])
            .expect("selector MLE");
        assert_eq!(selected, elements[2 * cols + 1].coeffs[3]);

        let padded_row = view
            .mle(&[F::one(), F::one()], &[F::one()], &[F::one(), F::one()])
            .expect("padded selector MLE");
        assert_eq!(padded_row, F::zero());
    }

    #[test]
    fn setup_polynomial_view_mle_matches_manual_sum() {
        const D: usize = 4;
        let rows = 3usize;
        let cols = 3usize;
        let elements: Vec<CyclotomicRing<F, D>> = (0..rows * cols)
            .map(|idx| {
                let coeffs =
                    std::array::from_fn(|coeff| F::from_u64((100 + 7 * idx + coeff) as u64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let flat = FlatMatrix::from_ring_slice(&elements);
        let view = flat.setup_polynomial_view::<D>(rows, cols);
        let row_challenges = [F::from_u64(2), F::from_u64(5)];
        let col_challenges = [F::from_u64(7), F::from_u64(11)];
        let coeff_challenges = [F::from_u64(13), F::from_u64(17)];

        let mut expected = F::zero();
        for row in 0..(1usize << view.row_bits()) {
            let row_weight = eq_weight_at_index(&row_challenges, row);
            for col in 0..(1usize << view.col_bits()) {
                let col_weight = eq_weight_at_index(&col_challenges, col);
                for coeff in 0..D {
                    expected += row_weight
                        * col_weight
                        * eq_weight_at_index(&coeff_challenges, coeff)
                        * view.coeff(row, col, coeff);
                }
            }
        }

        let got = view
            .mle(&row_challenges, &col_challenges, &coeff_challenges)
            .expect("setup polynomial MLE");
        assert_eq!(got, expected);
    }

    #[test]
    fn setup_polynomial_claim_reduction_roundtrip() {
        use akita_sumcheck::{
            prove_sumcheck, verify_sumcheck, EqWeightedTableProver, EqWeightedTableVerifier,
            SumcheckInstanceProver,
        };
        use akita_transcript::{labels, Blake2bTranscript, Transcript};

        const D: usize = 4;
        let rows = 3usize;
        let cols = 2usize;
        let elements: Vec<CyclotomicRing<F, D>> = (0..rows * cols)
            .map(|idx| {
                let coeffs =
                    std::array::from_fn(|coeff| F::from_u64((200 + 5 * idx + coeff) as u64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let flat = FlatMatrix::from_ring_slice(&elements);
        let view = flat.setup_polynomial_view::<D>(rows, cols);

        let row_point = vec![F::from_u64(3), F::from_u64(5)];
        let col_point = vec![F::from_u64(7)];
        let coeff_point = vec![F::from_u64(11), F::from_u64(13)];
        let mut target_point = row_point.clone();
        target_point.extend_from_slice(&col_point);
        target_point.extend_from_slice(&coeff_point);

        let table_len = 1usize << target_point.len();
        let row_mask = (1usize << view.row_bits()) - 1;
        let col_mask = (1usize << view.col_bits()) - 1;
        let table: Vec<F> = (0..table_len)
            .map(|idx| {
                let row = idx & row_mask;
                let col = (idx >> view.row_bits()) & col_mask;
                let coeff = idx >> (view.row_bits() + view.col_bits());
                view.coeff(row, col, coeff)
            })
            .collect();
        let scale = F::from_u64(17);

        let mut prover =
            EqWeightedTableProver::new(table.clone(), &target_point, scale).expect("prover");
        let input_claim = prover.input_claim();
        assert_eq!(
            input_claim,
            scale
                * view
                    .mle(&row_point, &col_point, &coeff_point)
                    .expect("direct setup MLE")
        );

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"setup-claim-reduction");
        let (proof, prover_challenges, _) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .expect("prove setup claim");

        let verifier = EqWeightedTableVerifier::new(table, target_point, input_claim, scale)
            .expect("verifier");
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"setup-claim-reduction");
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut verifier_transcript, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .expect("verify setup claim");
        assert_eq!(verifier_challenges, prover_challenges);
    }
}
