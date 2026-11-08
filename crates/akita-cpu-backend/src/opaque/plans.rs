use akita_algebra::CyclotomicRing;
use jolt_algebra::Field;

/// Named ring-switch relation rows returned by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RingSwitchRelationRows<F: Field, const D: usize> {
    /// D-side negacyclic reduced rows used as the transcript-visible `v`.
    pub d_negacyclic: Vec<CyclotomicRing<F, D>>,
    /// D-side cyclic rows.
    pub d_cyclic: Vec<CyclotomicRing<F, D>>,
    /// B-side cyclic rows.
    pub b_cyclic: Vec<CyclotomicRing<F, D>>,
    /// A-side quotient rows.
    pub a_quotients: Vec<CyclotomicRing<F, D>>,
}
