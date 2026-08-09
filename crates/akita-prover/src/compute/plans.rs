use akita_algebra::CyclotomicRing;
use akita_field::FieldCore;

/// Internal dense CPU commit representation.
pub(crate) enum DenseCommitInput<'a, F: FieldCore, const D: usize> {
    /// Balanced digit planes are already cached by the polynomial.
    CachedDigits {
        /// Per-block digit slices.
        digit_block_slices: Vec<&'a [[i8; D]]>,
        /// Logarithm of the gadget basis used to produce the cached digits.
        log_basis_inner: u32,
    },
    /// Ring coefficients need backend-side digit decomposition.
    CoeffBlocks {
        /// Per-block coefficient slices.
        block_slices: Vec<&'a [CyclotomicRing<F, D>]>,
        /// Number of balanced digits used for the A-side commit.
        num_digits_inner: usize,
        /// Logarithm of the gadget basis.
        log_basis_inner: u32,
    },
}

/// Full ring-switch relation operation input.
pub struct RingSwitchRelationRowsPlan<'a, const D: usize> {
    /// Number of D-side cyclic rows to produce.
    pub n_d: usize,
    /// Number of B-side cyclic rows to produce.
    pub n_b: usize,
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// Flat decomposed `e_hat` digits for the D-side relation rows.
    pub e_hat: &'a [[i8; D]],
    /// Flat decomposed inner-commitment digits for the B-side relation rows.
    pub t_hat: &'a [[i8; D]],
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
    /// Logarithm of the D/opening gadget basis used to produce `e_hat`.
    pub log_basis_open: u32,
    /// Logarithm of the B/outer gadget basis used to produce `t_hat`.
    pub log_basis_outer: u32,
}

/// Additional public-row quotient operation input.
pub struct RingSwitchQuotientRowsPlan<'a, const D: usize> {
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
}

/// Named ring-switch relation rows returned by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingSwitchRelationRows<F: FieldCore, const D: usize> {
    /// D-side cyclic rows.
    pub d_cyclic: Vec<CyclotomicRing<F, D>>,
    /// B-side cyclic rows.
    pub b_cyclic: Vec<CyclotomicRing<F, D>>,
    /// A-side quotient rows.
    pub a_quotients: Vec<CyclotomicRing<F, D>>,
}
