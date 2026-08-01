//! Canonical materialization of B/D compression chains for one ring relation.

use crate::compute::compression::{
    execute_compression_chains, CompressionExecutionInput, CompressionExecutionOutput,
    CompressionExecutionReport,
};
use crate::compute::{CompressionComputeBackend, OperationCtx};
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{
    CompressionChainWitness, CompressionTerminalPayload, RelationRhsLayout, RingVec,
};

/// Semantic source of one compression chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CompressionSourceId {
    Outer { group_index: usize },
    Opening,
}

/// Persistent materialization for one source chain.
pub(crate) struct CompressionSourceWitness<F> {
    pub(crate) id: CompressionSourceId,
    pub(crate) witness: CompressionChainWitness,
    #[allow(dead_code)] // Read by the atomic compressed-RHS and wire cutover.
    pub(crate) terminal: CompressionTerminalPayload<F>,
    pub(crate) quotients: Vec<RingVec<F>>,
}

/// All source chains in canonical relation order: B groups, then D.
pub(crate) struct CompressionWitnessMaterialization<F> {
    sources: Vec<CompressionSourceWitness<F>>,
}

impl<F: FieldCore> CompressionWitnessMaterialization<F> {
    pub(crate) fn source(
        &self,
        id: CompressionSourceId,
    ) -> Result<&CompressionSourceWitness<F>, AkitaError> {
        self.sources
            .iter()
            .find(|source| source.id == id)
            .ok_or_else(|| AkitaError::InvalidSetup("compression source is missing".into()))
    }
}

fn into_source<F: FieldCore>(
    output: CompressionExecutionOutput<CompressionSourceId, F>,
) -> CompressionSourceWitness<F> {
    CompressionSourceWitness {
        id: output.id,
        witness: output.witness,
        terminal: output.terminal,
        quotients: output.quotients,
    }
}

/// Execute every B/D chain using plans owned by the canonical relation layout.
pub(crate) fn materialize_compression_witness<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    layout: &RelationRhsLayout,
    commitment_rows: &RingVec<F>,
    opening_rows: &RingVec<F>,
) -> Result<
    (
        CompressionWitnessMaterialization<F>,
        CompressionExecutionReport,
    ),
    AkitaError,
>
where
    F: FieldCore + CanonicalField + HalvingField,
    B: CompressionComputeBackend<F>,
{
    let mut inputs = Vec::with_capacity(layout.groups.len() + 1);
    let mut commitment_offset = 0usize;
    for relation_group_index in 0..layout.groups.len() {
        let (group_index, plan) = layout.group_compression_plan(relation_group_index)?;
        let end = commitment_offset
            .checked_add(plan.source_coefficients())
            .ok_or_else(|| AkitaError::InvalidSetup("compression source offset overflow".into()))?;
        let coefficients = commitment_rows
            .coeffs()
            .get(commitment_offset..end)
            .ok_or(AkitaError::InvalidProof)?
            .to_vec();
        inputs.push(CompressionExecutionInput {
            id: CompressionSourceId::Outer { group_index },
            plan: plan.clone(),
            coefficients,
        });
        commitment_offset = end;
    }
    if commitment_offset != commitment_rows.coeff_len() {
        return Err(AkitaError::InvalidSize {
            expected: commitment_offset,
            actual: commitment_rows.coeff_len(),
        });
    }

    let opening_plan = layout.opening_compression_plan()?;
    if opening_rows.coeff_len() != opening_plan.source_coefficients() {
        return Err(AkitaError::InvalidSize {
            expected: opening_plan.source_coefficients(),
            actual: opening_rows.coeff_len(),
        });
    }
    inputs.push(CompressionExecutionInput {
        id: CompressionSourceId::Opening,
        plan: opening_plan.clone(),
        coefficients: opening_rows.coeffs().to_vec(),
    });

    let (outputs, report) = execute_compression_chains(ctx, inputs)?;
    let sources = outputs.into_iter().map(into_source).collect::<Vec<_>>();
    if sources.len() != layout.groups.len() + 1 {
        return Err(AkitaError::InvalidSetup(
            "compression executor omitted a relation source".into(),
        ));
    }
    Ok((CompressionWitnessMaterialization { sources }, report))
}
