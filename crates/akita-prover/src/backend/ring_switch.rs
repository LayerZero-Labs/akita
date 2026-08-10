//! Source views for ring-switch relation and quotient kernels.

/// Borrowed source view for fused ring-switch relation rows.
#[derive(Debug, Clone, Copy)]
pub struct RingSwitchRelationView<'a, const D: usize> {
    /// Flat decomposed `e_hat` digits for D-side relation rows.
    pub e_hat: &'a [[i8; D]],
    /// Flat decomposed inner-commitment digits for B-side relation rows.
    pub t_hat: &'a [[i8; D]],
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
}

/// Borrowed source view for additional ring-switch quotient rows.
#[derive(Debug, Clone, Copy)]
pub struct RingSwitchQuotientView<'a, const D: usize> {
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
}
