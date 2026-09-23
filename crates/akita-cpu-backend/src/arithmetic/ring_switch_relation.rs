//! Source view for the ring-switch relation kernel.

/// Borrowed source view for fused ring-switch relation rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RingSwitchRelationView<'a, const D: usize> {
    /// Flat decomposed `e_hat` digits for D-side relation rows.
    pub e_hat: &'a [[i8; D]],
    /// Flat decomposed inner-commitment digits for B-side relation rows.
    pub t_hat: &'a [[i8; D]],
}
