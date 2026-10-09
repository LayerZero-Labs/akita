use akita_algebra::{fft::SmoothFftField, ring::TrinomialModulus};
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::{
    LabiniusDigitBase, LabiniusRootEncoding, LabiniusRootProfile, LabiniusRootShape,
};
use akita_types::{RelationPolynomial, TrinomialASetupView, TrinomialResponseLayout};

use crate::BinaryClearSetup;

/// Admitted geometry, using the canonical digit-innermost response layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoweredRootLayout {
    response: TrinomialResponseLayout,
    setup_view: TrinomialASetupView,
    encoding: LabiniusRootEncoding,
    k: usize,
    columns: usize,
    scalar_rows: usize,
    image_count: usize,
    image_len: usize,
    matrix_digest: [u8; 32],
    root_profile: LabiniusRootProfile,
    field_identity: (u32, u128),
}

impl LoweredRootLayout {
    /// Admit only a shape derived by the closed root-profile admission API.
    pub fn new<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
        setup: &BinaryClearSetup<F, D, M>,
        shape: &LabiniusRootShape,
        base: LabiniusDigitBase,
    ) -> Result<Self, AkitaError> {
        let invalid = || AkitaError::InvalidSetup("lowered setup and root shape disagree".into());
        let modulus = match F::MODULUS_BITS {
            64 => (1u128 << 64).checked_sub(F::OFFSET),
            128 => u128::MAX
                .checked_sub(F::OFFSET)
                .and_then(|v| v.checked_add(1)),
            _ => None,
        };
        let encoding = shape.derive_encoding(base)?;
        if modulus != Some(shape.profile().coefficient_prime().modulus())
            || D != shape.commitment_degree()
            || setup.k() != shape.packing_degree()
            || shape.scalar_degree() != 162
            || usize::try_from(shape.rank_a()).ok() != Some(setup.n_a())
            || setup.m() != shape.ring_elements_per_column()
            || setup.columns() != shape.fold_width()
            || setup.scalar_rows() != shape.scalars_per_column()
            || setup.source_len() != shape.num_cells()
            || setup.profile() != &shape.profile().challenge_profile()?
            || encoding.response().interval()
                != (i128::from(setup.lower()), i128::from(setup.upper()))
            || !encoding.response().digit_count().is_power_of_two()
            || !setup.m().is_power_of_two()
        {
            return Err(invalid());
        }
        let polynomial = match M::MIDDLE_COEFFICIENT {
            1 => RelationPolynomial::plus_trinomial(D)?,
            -1 => RelationPolynomial::minus_trinomial(D)?,
            _ => return Err(invalid()),
        };
        // The admitted encoding is the one source for both table sizes.
        let padded = encoding.padded_coefficient_len();
        let witness_len = encoding.response_table_len();
        let response = TrinomialResponseLayout::new(
            polynomial,
            padded,
            setup.m(),
            encoding.response().digit_count(),
            0,
            witness_len,
        )
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let setup_len = checked::product([setup.n_a(), setup.m(), D])
            .and_then(checked::ceil_log2)
            .and_then(checked::pow2)
            .ok_or_else(invalid)?;
        let setup_view =
            TrinomialASetupView::new(polynomial, setup.n_a(), setup.m(), 0, setup_len, response)
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let image_count = checked::product([setup.columns(), setup.n_a()]).ok_or_else(invalid)?;
        let image_len = checked::pow2(encoding.image_table_log_len()).ok_or_else(invalid)?;
        Ok(Self {
            response,
            setup_view,
            encoding,
            k: setup.k(),
            columns: setup.columns(),
            scalar_rows: setup.scalar_rows(),
            image_count,
            image_len,
            matrix_digest: *setup.matrix_view_digest(),
            root_profile: shape.profile(),
            field_identity: (F::MODULUS_BITS, F::OFFSET),
        })
    }
    pub fn response_layout(&self) -> TrinomialResponseLayout {
        self.response
    }
    pub fn setup_view(&self) -> TrinomialASetupView {
        self.setup_view
    }
    pub fn polynomial(&self) -> RelationPolynomial {
        self.response.polynomial()
    }
    pub fn encoding(&self) -> &LabiniusRootEncoding {
        &self.encoding
    }
    pub fn degree(&self) -> usize {
        self.polynomial().degree()
    }
    pub fn padded_coefficients(&self) -> usize {
        self.response.coefficient_layout().padded_len()
    }
    pub fn coefficient_stride(&self) -> usize {
        self.response.coefficient_stride()
    }
    pub fn m(&self) -> usize {
        self.response.polynomial_count()
    }
    pub fn n_a(&self) -> usize {
        self.setup_view.rows()
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn scalar_rows(&self) -> usize {
        self.scalar_rows
    }
    pub fn image_count(&self) -> usize {
        self.image_count
    }
    pub fn witness_len(&self) -> usize {
        self.response.domain_len()
    }
    pub fn witness_log_len(&self) -> usize {
        self.witness_len().trailing_zeros() as usize
    }
    pub fn image_len(&self) -> usize {
        self.image_len
    }
    pub fn image_log_len(&self) -> usize {
        self.image_len.trailing_zeros() as usize
    }
    /// Checked packed coefficient sign, independent of the ring-element index.
    pub fn sigma(&self, coefficient: usize) -> Result<i128, AkitaError> {
        if coefficient >= self.degree() {
            return Err(AkitaError::InvalidInput(
                "packed coefficient out of range".into(),
            ));
        }
        Ok(if self.k == 1 || (coefficient / self.k).is_multiple_of(2) {
            1
        } else {
            -1
        })
    }
    /// Offset of the unsigned packed coefficient, preserving the scalar interval.
    pub fn off(&self, coefficient: usize) -> Result<i128, AkitaError> {
        let off = i128::try_from(self.encoding.response().offset())
            .map_err(|_| AkitaError::InvalidSetup("response offset exceeds i128".into()))?;
        if self.sigma(coefficient)? == 1 {
            Ok(off)
        } else {
            off.checked_sub(1).ok_or(AkitaError::InvalidProof)
        }
    }
    /// Image coefficient address, including coefficient tails but excluding entry padding.
    pub fn image_address(&self, entry: usize, coefficient: usize) -> Result<usize, AkitaError> {
        if entry >= self.image_count || coefficient >= self.padded_coefficients() {
            return Err(AkitaError::InvalidInput(
                "image coordinate out of range".into(),
            ));
        }
        checked::mul_add(entry, self.padded_coefficients(), coefficient)
            .ok_or(AkitaError::InvalidProof)
    }
    pub(crate) fn validate_setup<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
        &self,
        setup: &BinaryClearSetup<F, D, M>,
    ) -> Result<(), AkitaError> {
        if self.degree() != D
            || self.m() != setup.m()
            || self.n_a() != setup.n_a()
            || self.columns() != setup.columns()
            || self.k() != setup.k()
            || &self.matrix_digest != setup.matrix_view_digest()
            || self.field_identity != (F::MODULUS_BITS, F::OFFSET)
            || self.encoding.response().interval()
                != (i128::from(setup.lower()), i128::from(setup.upper()))
            || setup.profile() != &self.root_profile.challenge_profile()?
        {
            return Err(AkitaError::InvalidSetup(
                "lowered layout belongs to a different setup".into(),
            ));
        }
        Ok(())
    }
}
