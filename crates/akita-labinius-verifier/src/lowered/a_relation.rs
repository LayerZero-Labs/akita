//! Clear A-row auxiliary data, with optional foreign-modulus integer carries.

/// Canonical A-row witnesses: quotient rows followed by row-major carry coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ARelationAuxiliary<F> {
    pub quotients: Vec<Vec<F>>,
    pub carry: Vec<i128>,
}

/// Borrow the A-row witnesses for shared-prime or lifted relations.
///
/// Shared-prime callers may supply quotient rows directly. Lifted callers supply
/// `ARelationAuxiliary`; statement-derived length checks distinguish the cases.
pub trait ARelationRows<F> {
    fn quotients(&self) -> &[Vec<F>];
    fn quotients_mut(&mut self) -> &mut [Vec<F>];
    fn a_carry(&self) -> &[i128];
    fn a_carry_mut(&mut self) -> &mut [i128];
}

impl<F> ARelationRows<F> for ARelationAuxiliary<F> {
    fn quotients(&self) -> &[Vec<F>] {
        &self.quotients
    }
    fn quotients_mut(&mut self) -> &mut [Vec<F>] {
        &mut self.quotients
    }
    fn a_carry(&self) -> &[i128] {
        &self.carry
    }
    fn a_carry_mut(&mut self) -> &mut [i128] {
        &mut self.carry
    }
}

impl<F> ARelationRows<F> for Vec<Vec<F>> {
    fn quotients(&self) -> &[Vec<F>] {
        self
    }
    fn quotients_mut(&mut self) -> &mut [Vec<F>] {
        self
    }
    fn a_carry(&self) -> &[i128] {
        &[]
    }
    fn a_carry_mut(&mut self) -> &mut [i128] {
        &mut []
    }
}

impl<F> ARelationRows<F> for [Vec<F>] {
    fn quotients(&self) -> &[Vec<F>] {
        self
    }
    fn quotients_mut(&mut self) -> &mut [Vec<F>] {
        self
    }
    fn a_carry(&self) -> &[i128] {
        &[]
    }
    fn a_carry_mut(&mut self) -> &mut [i128] {
        &mut []
    }
}
