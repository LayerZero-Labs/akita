use akita_algebra::ring::TrinomialModulus;
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::{LabiniusRootEncoding, LabiniusRootProfile, LabiniusRootShape};
use akita_types::{RelationPolynomial, TrinomialASetupView, TrinomialResponseLayout};
use jolt_field::{CanonicalEncoding, ExtField, Field};

use crate::{grinding::field_runs, statement::RootOpeningMode, BinaryClearSetup};

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
    prime_len: usize,
    matrix_digest: [u8; 32],
    root_profile: LabiniusRootProfile,
}

impl LoweredRootLayout {
    /// Admit the field pair `(F, E)` for a shape derived by the closed
    /// root-profile admission API, and fix the table layout.
    ///
    /// This is where the proof fields enter. The characteristic `P` of the
    /// base field `F` must pass [`LabiniusRootShape::derive_encoding`]:
    /// fold-challenge differences are units modulo it, and neither the
    /// commitment rows nor the parity row can wrap. These are statements about
    /// `P`, never about `|E|`. The challenge field `E` is admitted exactly
    /// when every field-challenge site of the reduction has a proof-of-work
    /// target within Akita's cap
    /// ([`RootGrindingPlan`](crate::grinding::RootGrindingPlan)). Neither field
    /// needs a root of unity.
    pub fn new<F, E, const D: usize, M>(
        setup: &BinaryClearSetup<D, M>,
        shape: &LabiniusRootShape,
    ) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
        M: TrinomialModulus,
    {
        let invalid = || AkitaError::InvalidSetup("lowered setup and root shape disagree".into());
        let encoding = shape.derive_encoding(crate::admitted::field_characteristic::<F>()?)?;
        // Deriving the challenge-field sites is the admission of `E`. The
        // mode with both claims visits every site of the other two with a loss
        // factor at least as large. The statement binding derives the plan of
        // its own mode again, with the host field.
        field_runs::<F, E>(shape, &encoding, RootOpeningMode::Both, &mut Vec::new())?;
        if setup.commitment_modulus() != shape.profile().commitment_modulus()
            || D != shape.commitment_degree()
            || setup.k() != shape.packing_degree()
            || shape.scalar_degree() != 162
            || usize::try_from(shape.rank_a()).ok() != Some(setup.n_a())
            || setup.m() != shape.ring_elements_per_column()
            || setup.columns() != shape.fold_width()
            || setup.scalar_rows() != shape.scalars_per_column()
            || setup.source_len() != shape.num_cells()
            || setup.profile() != &shape.profile().challenge_profile()?
            || encoding.response_interval()
                != (i128::from(setup.lower()), i128::from(setup.upper()))
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
            encoding.response_digit_slots(),
            0,
            witness_len,
        )
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let setup_len = checked::product([setup.n_a(), setup.m(), D])
            .and_then(checked::ceil_log2)
            .and_then(checked::pow2)
            .ok_or_else(invalid)?;
        // Kept for its admission checks: the matrix and its padded domain stay
        // under the compact-stride work cap.
        let setup_view =
            TrinomialASetupView::new(polynomial, setup.n_a(), setup.m(), 0, setup_len, response)
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let image_count = checked::product([setup.columns(), setup.n_a()]).ok_or_else(invalid)?;
        let image_len = checked::pow2(encoding.image_table_log_len()).ok_or_else(invalid)?;
        // One padded ring element per column, with no padding columns.
        let prime_len = checked::product([padded, setup.columns()])
            .filter(|&len| checked::pow2(encoding.prime_table_log_len()) == Some(len))
            .ok_or_else(invalid)?;
        Ok(Self {
            response,
            setup_view,
            encoding,
            k: setup.k(),
            columns: setup.columns(),
            scalar_rows: setup.scalar_rows(),
            image_count,
            image_len,
            prime_len,
            matrix_digest: *setup.matrix_view_digest(),
            root_profile: shape.profile(),
        })
    }
    pub fn response_layout(&self) -> TrinomialResponseLayout {
        self.response
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
    /// Image entries `col * n_A + i`, before padding to a power of two.
    pub fn image_count(&self) -> usize {
        self.image_count
    }
    pub fn witness_len(&self) -> usize {
        self.response.domain_len()
    }
    pub fn witness_log_len(&self) -> usize {
        self.witness_len().trailing_zeros() as usize
    }
    /// Length of the image digit table: digit slot innermost, then padded
    /// coefficient, then image entry, padded to a power of two.
    pub fn image_len(&self) -> usize {
        self.image_len
    }
    pub fn image_log_len(&self) -> usize {
        self.image_len.trailing_zeros() as usize
    }
    /// Length of the prime left opening: padded coefficient innermost, then
    /// column. Both axes are powers of two, so the only padding entries are
    /// the coefficient tails.
    pub fn prime_len(&self) -> usize {
        self.prime_len
    }
    pub fn prime_log_len(&self) -> usize {
        self.prime_len.trailing_zeros() as usize
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
    /// Offset `o` with `packed coefficient + o` in `[0, upper - lower]`: `-lower`
    /// where the scalar coefficient is packed with sign `+1`, `upper` where it
    /// is negated.
    pub fn off(&self, coefficient: usize) -> Result<i128, AkitaError> {
        let (lower, upper) = self.encoding.response_interval();
        if self.sigma(coefficient)? == 1 {
            lower.checked_neg().ok_or(AkitaError::InvalidProof)
        } else {
            Ok(upper)
        }
    }
    pub(crate) fn validate_setup<const D: usize, M: TrinomialModulus>(
        &self,
        setup: &BinaryClearSetup<D, M>,
    ) -> Result<(), AkitaError> {
        if self.degree() != D
            || setup.commitment_modulus() != self.root_profile.commitment_modulus()
            || self.m() != setup.m()
            || self.n_a() != setup.n_a()
            || self.columns() != setup.columns()
            || self.k() != setup.k()
            || &self.matrix_digest != setup.matrix_view_digest()
            || self.encoding.response_interval()
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
