//! Borrowed evaluators for canonical relation-row families.
//!
//! A provider never stores a second copy of relation geometry. It borrows a
//! [`RelationRowFamily`] and resolves every coefficient span through its owning
//! [`RelationLayout`]. Compression matrices are logical views of the same setup
//! prefix: coefficient zero of every view is setup coefficient zero.

use akita_field::{AkitaError, FieldCore};

use super::{
    CoeffSpan, GadgetInput, RelationLayout, RelationRowFamily, RelationRowId, RelationRowInputs,
    RelationRowRhs, RelationSegmentId, RowSpan,
};

/// Compact geometry of one logical matrix view over the shared setup prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedSetupMatrixView {
    native_ring_dim: usize,
    rows: usize,
    ring_columns: usize,
    flat_row_coeffs: usize,
    flat_footprint: usize,
}

#[cfg(test)]
mod tests;
impl SharedSetupMatrixView {
    pub fn native_ring_dim(&self) -> usize {
        self.native_ring_dim
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn ring_columns(&self) -> usize {
        self.ring_columns
    }

    pub fn flat_row_coeffs(&self) -> usize {
        self.flat_row_coeffs
    }

    /// Number of field coefficients in this prefix view.
    pub fn flat_footprint(&self) -> usize {
        self.flat_footprint
    }
}

/// Ephemeral provider/evaluator for one canonical relation-row family.
///
/// Compression families represent `successor = K * input` over
/// `R = F[X]/(X^d + 1)`. At an evaluation point `alpha`, a matrix ring entry
/// contributes its coefficient `k_j` with weight
/// `row_weight * input_col(alpha) * alpha^j`. The negacyclic quotient and the
/// successor are witness terms, so neither changes the setup-prefix weight.
pub struct RelationFamilyProvider<'a> {
    layout: &'a RelationLayout,
    family: &'a RelationRowFamily,
}

impl<'a> RelationFamilyProvider<'a> {
    pub(super) fn new(
        layout: &'a RelationLayout,
        family: &'a RelationRowFamily,
    ) -> Result<Self, AkitaError> {
        let provider = Self { layout, family };
        let rows = provider.rows();
        let row_end = rows
            .start()
            .checked_add(rows.len())
            .ok_or_else(|| AkitaError::InvalidSetup("relation family row span overflow".into()))?;
        let d = provider.native_ring_dim();
        if rows.is_empty() || row_end > layout.row_plan().trace_row() || d == 0 {
            return Err(AkitaError::InvalidSetup(
                "relation family row span or native dimension is invalid".into(),
            ));
        }
        let row_unit = rows
            .len()
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidSetup("relation family row-unit overflow".into()))?;
        let expected_quotient = match family.id() {
            RelationRowId::Compression { source, map } => {
                RelationSegmentId::CompressionQuotient { source, map }
            }
            row => RelationSegmentId::BaseQuotient { row },
        };
        if family.quotient() != expected_quotient {
            return Err(AkitaError::InvalidSetup(
                "relation quotient identity disagrees with its row family".into(),
            ));
        }
        let quotient = provider.resolve_nonempty(family.quotient())?;
        if quotient.is_empty() || !quotient.len().is_multiple_of(row_unit) {
            return Err(AkitaError::InvalidSetup(
                "relation quotient span disagrees with its row family".into(),
            ));
        }
        match family.inputs() {
            RelationRowInputs::Consistency { z, e } => {
                if family.id() != RelationRowId::Consistency {
                    return Err(AkitaError::InvalidSetup(
                        "consistency inputs disagree with the family identity".into(),
                    ));
                }
                provider.require_group_segments(z, |group| RelationSegmentId::Z { group })?;
                provider.require_group_segments(e, |group| RelationSegmentId::E { group })?;
                for &id in z.iter().chain(e) {
                    let _ = provider.resolve_nonempty(id)?;
                }
                provider.require_rhs(RelationRowRhs::Zero)?;
            }
            RelationRowInputs::A { z } => {
                let group = match family.id() {
                    RelationRowId::A { group } => group,
                    _ => {
                        return Err(AkitaError::InvalidSetup(
                            "A inputs disagree with the family identity".into(),
                        ));
                    }
                };
                if *z != (RelationSegmentId::Z { group }) {
                    return Err(AkitaError::InvalidSetup(
                        "A input does not name its group Z segment".into(),
                    ));
                }
                let _ = provider.resolve_native(*z, d)?;
                provider.require_rhs(RelationRowRhs::Zero)?;
            }
            RelationRowInputs::B {
                t,
                compression_input,
            } => {
                let _ = provider.resolve_nonempty(*t)?;
                let expected = match (family.id(), compression_input) {
                    (super::RelationRowId::B { group }, None) => {
                        if *t != (RelationSegmentId::T { group }) {
                            return Err(AkitaError::InvalidSetup(
                                "B input does not name its group T segment".into(),
                            ));
                        }
                        RelationRowRhs::Commitment { group }
                    }
                    (RelationRowId::B { group }, Some(input)) => {
                        let source = match group {
                            super::RelationGroupId::Current => {
                                super::CompressionSourceId::CurrentOuter
                            }
                            super::RelationGroupId::Precommitted { index } => {
                                super::CompressionSourceId::PrecommittedOuter { index }
                            }
                        };
                        if *t != (RelationSegmentId::T { group })
                            || input.segment()
                                != (RelationSegmentId::CompressionInput { source, map: 0 })
                        {
                            return Err(AkitaError::InvalidSetup(
                                "augmented B inputs do not name their canonical segments".into(),
                            ));
                        }
                        let _ = provider.resolve_gadget_native(*input, d)?;
                        RelationRowRhs::Zero
                    }
                    _ => {
                        return Err(AkitaError::InvalidSetup(
                            "B relation inputs disagree with the family identity".into(),
                        ));
                    }
                };
                let _ = provider.resolve_native(*t, d)?;
                provider.require_rhs(expected)?;
            }
            RelationRowInputs::D {
                e,
                compression_input,
            } => {
                provider.require_group_segments(e, |group| RelationSegmentId::E { group })?;
                provider.resolve_all_native(e, d)?;
                let expected = if let Some(input) = compression_input {
                    if input.segment()
                        != (RelationSegmentId::CompressionInput {
                            source: super::CompressionSourceId::Opening,
                            map: 0,
                        })
                    {
                        return Err(AkitaError::InvalidSetup(
                            "augmented D input is not the first opening compression map".into(),
                        ));
                    }
                    let _ = provider.resolve_gadget_native(*input, d)?;
                    RelationRowRhs::Zero
                } else {
                    RelationRowRhs::Opening
                };
                if !matches!(family.id(), super::RelationRowId::D) {
                    return Err(AkitaError::InvalidSetup(
                        "D relation inputs disagree with the family identity".into(),
                    ));
                }
                provider.require_rhs(expected)?;
            }
            RelationRowInputs::Compression { input, successor } => {
                let (source, map) = match family.id() {
                    RelationRowId::Compression { source, map } => (source, map),
                    _ => {
                        return Err(AkitaError::InvalidSetup(
                            "compression inputs disagree with the family identity".into(),
                        ));
                    }
                };
                if *input != (RelationSegmentId::CompressionInput { source, map }) {
                    return Err(AkitaError::InvalidSetup(
                        "compression input identity disagrees with its row family".into(),
                    ));
                }
                let _ = provider.resolve_native(*input, d)?;
                let successor = successor
                    .map(|input| {
                        let next_map = map.checked_add(1).ok_or_else(|| {
                            AkitaError::InvalidSetup("compression successor map overflow".into())
                        })?;
                        if input.segment()
                            != (RelationSegmentId::CompressionInput {
                                source,
                                map: next_map,
                            })
                        {
                            return Err(AkitaError::InvalidSetup(
                                "compression successor identity is not the next map".into(),
                            ));
                        }
                        provider.resolve_gadget_nonempty(input)
                    })
                    .transpose()?;
                match (successor, provider.rhs()) {
                    (Some((span, _)), RelationRowRhs::Zero)
                        if span.len().is_multiple_of(row_unit) => {}
                    (None, RelationRowRhs::TerminalPayload { coeffs }) if coeffs == row_unit => {}
                    _ => {
                        return Err(AkitaError::InvalidSetup(
                            "compression successor and RHS semantics disagree".into(),
                        ));
                    }
                }
                let _ = provider.compression_setup_view()?;
            }
        }
        Ok(provider)
    }

    pub fn family(&self) -> &'a RelationRowFamily {
        self.family
    }

    pub fn rows(&self) -> RowSpan {
        self.family.rows()
    }

    pub fn native_ring_dim(&self) -> usize {
        self.family.native_ring_dim()
    }

    pub fn rhs(&self) -> RelationRowRhs {
        self.family.rhs()
    }

    pub fn quotient_span(&self) -> Result<CoeffSpan, AkitaError> {
        self.resolve(self.family.quotient())
    }

    pub fn compression_input_span(&self) -> Result<CoeffSpan, AkitaError> {
        match self.family.inputs() {
            RelationRowInputs::Compression { input, .. } => self.resolve(*input),
            _ => Err(AkitaError::InvalidInput(
                "relation family is not a compression provider".into(),
            )),
        }
    }

    pub fn compression_successor(&self) -> Result<Option<(CoeffSpan, u32)>, AkitaError> {
        match self.family.inputs() {
            RelationRowInputs::Compression { successor, .. } => successor
                .map(|input| self.resolve_gadget(input))
                .transpose(),
            _ => Err(AkitaError::InvalidInput(
                "relation family is not a compression provider".into(),
            )),
        }
    }

    /// Logical matrix geometry for a compression F/H row family.
    ///
    /// The matrix consumes one native ring per input column. Consequently the
    /// input span is exactly one flat setup row and the whole logical matrix
    /// occupies `rows * input_span.len()` field coefficients at prefix zero.
    pub fn compression_setup_view(&self) -> Result<SharedSetupMatrixView, AkitaError> {
        let input = self.compression_input_span()?;
        let d = self.native_ring_dim();
        if d == 0 || !input.len().is_multiple_of(d) {
            return Err(AkitaError::InvalidSetup(
                "compression input span disagrees with native ring dimension".into(),
            ));
        }
        let rows = self.rows().len();
        let footprint = rows.checked_mul(input.len()).ok_or_else(|| {
            AkitaError::InvalidSetup("compression setup footprint overflow".into())
        })?;
        Ok(SharedSetupMatrixView {
            native_ring_dim: d,
            rows,
            ring_columns: input.len() / d,
            flat_row_coeffs: input.len(),
            flat_footprint: footprint,
        })
    }

    /// Add this compression matrix's native-ring weights to a shared flat
    /// setup-prefix table.
    ///
    /// `input_column_evals` is a compact, strictly increasing list of nonzero
    /// `(ring_column, input_column(alpha))` values. `alpha_pows[j]` is
    /// `alpha^j` in the provider's native ring. Equal logical views naturally
    /// coalesce because every provider writes from prefix index zero.
    pub fn accumulate_compression_setup_weights<E: FieldCore>(
        &self,
        shared_prefix: &mut [E],
        row_weights: &[E],
        input_column_evals: &[(usize, E)],
        alpha_pows: &[E],
    ) -> Result<(), AkitaError> {
        let view = self.compression_setup_view()?;
        if row_weights.len() != view.rows {
            return Err(AkitaError::InvalidSize {
                expected: view.rows,
                actual: row_weights.len(),
            });
        }
        if alpha_pows.len() != view.native_ring_dim {
            return Err(AkitaError::InvalidSize {
                expected: view.native_ring_dim,
                actual: alpha_pows.len(),
            });
        }
        if shared_prefix.len() < view.flat_footprint {
            return Err(AkitaError::InvalidSize {
                expected: view.flat_footprint,
                actual: shared_prefix.len(),
            });
        }
        let mut previous = None;
        for &(column, value) in input_column_evals {
            if value.is_zero()
                || column >= view.ring_columns
                || previous.is_some_and(|prev| column <= prev)
            {
                return Err(AkitaError::InvalidInput(
                    "compression setup column support is not canonical".into(),
                ));
            }
            previous = Some(column);
        }
        for (row, &row_weight) in row_weights.iter().enumerate() {
            if row_weight.is_zero() {
                continue;
            }
            let row_start = row.checked_mul(view.flat_row_coeffs).ok_or_else(|| {
                AkitaError::InvalidSetup("compression setup row offset overflow".into())
            })?;
            for &(column, column_eval) in input_column_evals {
                let column_start = column
                    .checked_mul(view.native_ring_dim)
                    .and_then(|offset| row_start.checked_add(offset))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("compression setup column offset overflow".into())
                    })?;
                let scale = row_weight * column_eval;
                for (lane, &alpha_pow) in alpha_pows.iter().enumerate() {
                    let slot = shared_prefix.get_mut(column_start + lane).ok_or_else(|| {
                        AkitaError::InvalidSetup("compression setup view exceeds prefix".into())
                    })?;
                    *slot += scale * alpha_pow;
                }
            }
        }
        Ok(())
    }

    fn resolve(&self, id: RelationSegmentId) -> Result<CoeffSpan, AkitaError> {
        Ok(self.layout.segment(id)?.span())
    }

    fn resolve_nonempty(&self, id: RelationSegmentId) -> Result<CoeffSpan, AkitaError> {
        let span = self.resolve(id)?;
        if span.is_empty() || span.end()? > self.layout.total_coeffs() {
            return Err(AkitaError::InvalidSetup(
                "relation input span is empty or outside the logical arena".into(),
            ));
        }
        Ok(span)
    }

    fn resolve_native(&self, id: RelationSegmentId, d: usize) -> Result<CoeffSpan, AkitaError> {
        let span = self.resolve_nonempty(id)?;
        if !span.len().is_multiple_of(d) {
            return Err(AkitaError::InvalidSetup(
                "relation input span disagrees with native ring dimension".into(),
            ));
        }
        Ok(span)
    }

    fn resolve_all_native(&self, ids: &[RelationSegmentId], d: usize) -> Result<(), AkitaError> {
        if ids.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "relation input family is empty".into(),
            ));
        }
        for &id in ids {
            let _ = self.resolve_native(id, d)?;
        }
        Ok(())
    }

    fn require_group_segments(
        &self,
        ids: &[RelationSegmentId],
        expected: impl Fn(super::RelationGroupId) -> RelationSegmentId,
    ) -> Result<(), AkitaError> {
        if ids.len() != self.layout.row_plan().group_order().len()
            || ids
                .iter()
                .zip(self.layout.row_plan().group_order())
                .any(|(&id, &group)| id != expected(group))
        {
            return Err(AkitaError::InvalidSetup(
                "relation group input order disagrees with the row plan".into(),
            ));
        }
        Ok(())
    }

    fn require_rhs(&self, expected: RelationRowRhs) -> Result<(), AkitaError> {
        if self.rhs() != expected {
            return Err(AkitaError::InvalidSetup(
                "relation family RHS disagrees with its inputs".into(),
            ));
        }
        Ok(())
    }

    fn resolve_gadget(&self, input: GadgetInput) -> Result<(CoeffSpan, u32), AkitaError> {
        Ok((self.resolve(input.segment())?, input.log_basis()))
    }

    fn resolve_gadget_nonempty(&self, input: GadgetInput) -> Result<(CoeffSpan, u32), AkitaError> {
        if input.log_basis() == 0 || input.log_basis() >= 128 {
            return Err(AkitaError::InvalidSetup(
                "relation gadget basis must be non-zero".into(),
            ));
        }
        Ok((self.resolve_nonempty(input.segment())?, input.log_basis()))
    }

    fn resolve_gadget_native(
        &self,
        input: GadgetInput,
        d: usize,
    ) -> Result<(CoeffSpan, u32), AkitaError> {
        let (_, log_basis) = self.resolve_gadget_nonempty(input)?;
        Ok((self.resolve_native(input.segment(), d)?, log_basis))
    }
}
