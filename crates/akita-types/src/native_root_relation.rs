//! Linear, role-native rows over a shared successor digit table.
//!
//! A root row is reduced in its own polynomial ring before evaluation at the
//! common ring-switch point. It does not change the ring of the successor
//! commitment. The caller fixes this plan and that commitment before choosing
//! any row functional (in particular before the projection seed).

use akita_algebra::ring::{
    shifted_remainder_evaluations, MinusTrinomial, PlusTrinomial, ResidueKernelPoint,
};
use akita_challenges::FoldChallengeFamily;
use akita_error::{checked, AkitaError};
use akita_params::{RelationPolynomial, RelationPolynomialKind, TrinomialSign};
use jolt_field::Field;

use crate::binary_source::{BinarySourceDescriptor, BINARY_SOURCE_Q};
use crate::root_repacking::RootCoefficientRepacking;
mod binary_commitment;
pub use binary_commitment::binary_commitment_group;

/// Source interpretation of one group. Feature enablement never selects it.
#[derive(Clone, Debug)]
pub enum NativeRootGroupKind {
    Ordinary(FoldChallengeFamily),
    Binary(BinarySourceDescriptor),
}

impl NativeRootGroupKind {
    pub fn challenge_family(&self) -> FoldChallengeFamily {
        match self {
            Self::Ordinary(family) => *family,
            Self::Binary(_) => FoldChallengeFamily::BinaryRoot,
        }
    }

    /// The binary A role uses q; ordinary roles use the proof field.
    pub fn commitment_modulus(&self) -> Option<u32> {
        match self {
            Self::Ordinary(_) => None,
            Self::Binary(_) => Some(BINARY_SOURCE_Q),
        }
    }
}

/// A public ring multiplier applied to one recomposed digit-table element.
/// All operands depending on the witness are addressed through `segment`.
#[derive(Clone, Debug)]
pub struct NativeRootTerm<E: Field> {
    pub segment: usize,
    pub element: usize,
    pub multiplier: Vec<E>,
}

/// One exact linear identity in its role-native quotient ring.
///
/// A binary commitment residual is not zero over the proof field merely
/// because it vanishes modulo q. The caller must include an authenticated
/// q-multiple term, or consume its projected value in the divisibility lane.
/// This type never silently treats a q-congruence as a field equality.
#[derive(Clone, Debug)]
pub struct NativeRootRow<E: Field> {
    pub polynomial: RelationPolynomial,
    pub terms: Vec<NativeRootTerm<E>>,
    pub target: Vec<E>,
}

/// Rows owned by one source group; B/D rows can have ordinary native rings.
#[derive(Clone, Debug)]
pub struct NativeRootGroup<E: Field> {
    pub kind: NativeRootGroupKind,
    pub rows: Vec<NativeRootRow<E>>,
}

/// Checked public linear rows, all on the same authenticated coefficient map.
#[derive(Clone, Debug)]
pub struct NativeRootRelation<E: Field> {
    repacking: RootCoefficientRepacking,
    groups: Vec<NativeRootGroup<E>>,
    row_count: usize,
}

impl<E: Field> NativeRootRelation<E> {
    pub fn new(
        repacking: RootCoefficientRepacking,
        groups: Vec<NativeRootGroup<E>>,
    ) -> Result<Self, AkitaError> {
        if groups.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "root relation has no groups".into(),
            ));
        }
        let row_count = checked::sum(groups.iter().map(|group| group.rows.len()))
            .ok_or_else(|| AkitaError::InvalidSetup("root row count overflow".into()))?;
        for group in &groups {
            if group.rows.is_empty() {
                return Err(AkitaError::InvalidSetup("root group has no rows".into()));
            }
            if let NativeRootGroupKind::Ordinary(family) = &group.kind {
                if family.signed_sparse().is_none() {
                    return Err(AkitaError::InvalidSetup(
                        "ordinary group has binary family".into(),
                    ));
                }
            }
            for row in &group.rows {
                let degree = row.polynomial.degree();
                if row.target.len() != degree || row.terms.is_empty() {
                    return Err(AkitaError::InvalidSetup("invalid native root row".into()));
                }
                if matches!(group.kind, NativeRootGroupKind::Ordinary(_))
                    && row.polynomial.kind() != RelationPolynomialKind::Negacyclic
                {
                    return Err(AkitaError::InvalidSetup(
                        "ordinary group has nonordinary row".into(),
                    ));
                }
                for term in &row.terms {
                    let segment = repacking
                        .segments()
                        .get(term.segment)
                        .ok_or(AkitaError::InvalidProof)?;
                    if segment.geometry.polynomial() != row.polynomial
                        || segment.geometry.coordinate_plane_count() != 1
                        || term.element >= segment.elements
                        || term.multiplier.len() != degree
                    {
                        return Err(AkitaError::InvalidSetup(
                            "native root term geometry mismatch".into(),
                        ));
                    }
                }
            }
        }
        Ok(Self {
            repacking,
            groups,
            row_count,
        })
    }

    pub fn repacking(&self) -> &RootCoefficientRepacking {
        &self.repacking
    }
    pub fn groups(&self) -> &[NativeRootGroup<E>] {
        &self.groups
    }
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Compile public weights at one shared ring-switch point and row batch.
    ///
    /// These sparse coefficients enter the ordinary Stage 2 witness oracle.
    /// The zero-padding equations are separate: `padding_weights` batches all
    /// virtual positions independently of the native ring identities.
    pub fn ring_switch(
        &self,
        alpha: E,
        row_weights: &[E],
        padding_weights: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        if row_weights.len() != self.row_count {
            return Err(AkitaError::InvalidSize {
                expected: self.row_count,
                actual: row_weights.len(),
            });
        }
        let mut terms = Vec::new();
        let mut claim = E::zero();
        for (row, &batch) in self
            .groups
            .iter()
            .flat_map(|group| &group.rows)
            .zip(row_weights)
        {
            claim += batch
                * row
                    .target
                    .iter()
                    .rev()
                    .fold(E::zero(), |acc, &c| acc * alpha + c);
            for term in &row.terms {
                let mut kernel = vec![E::zero(); row.polynomial.degree()];
                match row.polynomial.kind() {
                    RelationPolynomialKind::Negacyclic => {
                        ResidueKernelPoint::new(alpha, row.polynomial.degree())?
                            .field_kernel_into(&term.multiplier, &mut kernel)?;
                    }
                    RelationPolynomialKind::Trinomial(TrinomialSign::Minus) => {
                        shifted_remainder_evaluations::<E, MinusTrinomial>(
                            &term.multiplier,
                            alpha,
                            &mut kernel,
                        )
                        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
                    }
                    RelationPolynomialKind::Trinomial(TrinomialSign::Plus) => {
                        shifted_remainder_evaluations::<E, PlusTrinomial>(
                            &term.multiplier,
                            alpha,
                            &mut kernel,
                        )
                        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
                    }
                }
                for (coefficient, weight) in kernel.into_iter().enumerate() {
                    terms.extend(self.repacking.coefficient_weights(
                        term.segment,
                        term.element,
                        coefficient,
                        batch * weight,
                    )?);
                }
            }
        }
        let mut padding = padding_weights.iter();
        for index in 0..self.repacking.domain_len() {
            if self.repacking.is_padding_address(index)? {
                let weight = padding.next().copied().ok_or_else(|| {
                    AkitaError::InvalidInput("missing zero-padding weight".into())
                })?;
                terms.push((index, weight));
            }
        }
        if padding.next().is_some() {
            return Err(AkitaError::InvalidInput(
                "excess zero-padding weights".into(),
            ));
        }
        NativeRootWeights::new(self.repacking.domain_len(), terms, claim)
    }
}

/// Scalar relation after ring switching, in the successor's coefficient order.
#[derive(Clone, Debug)]
pub struct NativeRootWeights<E: Field> {
    table_len: usize,
    terms: Vec<(usize, E)>,
    claim: E,
}

impl<E: Field> NativeRootWeights<E> {
    pub fn new(table_len: usize, terms: Vec<(usize, E)>, claim: E) -> Result<Self, AkitaError> {
        if !table_len.is_power_of_two() || terms.iter().any(|(index, _)| *index >= table_len) {
            return Err(AkitaError::InvalidSetup(
                "invalid root weight addresses".into(),
            ));
        }
        Ok(Self {
            table_len,
            terms,
            claim,
        })
    }
    pub fn table_len(&self) -> usize {
        self.table_len
    }
    pub fn terms(&self) -> &[(usize, E)] {
        &self.terms
    }
    pub fn claim(&self) -> E {
        self.claim
    }

    /// Public MLE oracle for the ordinary fused Stage 2.
    pub fn evaluate(&self, point: &[E]) -> Result<E, AkitaError> {
        if point.len() != self.table_len.trailing_zeros() as usize {
            return Err(AkitaError::InvalidSize {
                expected: self.table_len.trailing_zeros() as usize,
                actual: point.len(),
            });
        }
        let equality = akita_algebra::offset_eq::OffsetEqWindow::new(point)?;
        Ok(self.terms.iter().fold(E::zero(), |acc, &(index, weight)| {
            acc + weight * equality.eval(index)
        }))
    }

    /// Check the scalar identity against a fixed table, for construction and
    /// arithmetic tests. Protocol verification uses authenticated MLE claims.
    pub fn check_table(&self, table: &[E]) -> Result<(), AkitaError> {
        if table.len() != self.table_len {
            return Err(AkitaError::InvalidSize {
                expected: self.table_len,
                actual: table.len(),
            });
        }
        let value = self
            .terms
            .iter()
            .try_fold(E::zero(), |acc, &(index, weight)| {
                Ok::<_, AkitaError>(
                    acc + weight * table.get(index).copied().ok_or(AkitaError::InvalidProof)?,
                )
            })?;
        if value != self.claim {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
