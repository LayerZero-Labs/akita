//! Source-typed ring-switch relation and quotient kernels.
//!
//! PO-CUTOVER (Phase A, additive): kernels delegate to the existing
//! `RingSwitchComputeBackend` row methods so behavior and proof bytes are
//! identical.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};

use crate::compute::{
    CpuBackend, RingSwitchComputeBackend, RingSwitchQuotientKernel, RingSwitchQuotientPlan,
    RingSwitchQuotientRowsPlan, RingSwitchRelationKernel, RingSwitchRelationPlan,
    RingSwitchRelationRows, RingSwitchRelationRowsPlan,
};

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

impl<F, const D: usize> RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + HalvingField,
{
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: RingSwitchRelationView<'_, D>,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField,
    {
        self.ring_switch_relation_rows(
            prepared,
            RingSwitchRelationRowsPlan {
                n_d: plan.n_d,
                n_b: plan.n_b,
                n_a: plan.n_a,
                e_hat: source.e_hat,
                t_hat: source.t_hat,
                z_segment: source.z_segment,
                z_folded_centered_inf_norm: source.z_folded_centered_inf_norm,
                log_basis: plan.log_basis,
            },
        )
    }
}

impl<F, const D: usize> RingSwitchQuotientKernel<RingSwitchQuotientView<'_, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + HalvingField,
{
    fn quotient_rows(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: RingSwitchQuotientView<'_, D>,
        plan: RingSwitchQuotientPlan,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField,
    {
        self.ring_switch_quotient_rows(
            prepared,
            RingSwitchQuotientRowsPlan {
                n_a: plan.n_a,
                z_segment: source.z_segment,
                z_folded_centered_inf_norm: source.z_folded_centered_inf_norm,
            },
        )
    }
}
