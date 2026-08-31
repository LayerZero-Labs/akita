//! Canonical materialization of B/D compression chains for one ring relation.

use crate::compute::compression::{
    execute_compression_chains, CompressionExecutionInput, CompressionExecutionOutput,
    CompressionExecutionReport, CompressionRelationOutput,
};
use crate::compute::{CompressionComputeBackend, OperationCtx};
use akita_error::AkitaError;
use akita_types::{
    AkitaCommitmentHint, CompressionChainPlan, CompressionChainWitness, CompressionTerminalPayload,
    RelationRhsLayout, RingRelationMode, RingVec,
};
use jolt_field::{CanonicalEncoding, Field};

/// Semantic source of one compression chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CompressionSourceId {
    Outer { group_index: usize },
    Opening,
}

/// Persistent materialization for one source chain.
pub(crate) struct CompressionSourceWitness<F: Field> {
    pub(crate) id: CompressionSourceId,
    pub(crate) witness: CompressionChainWitness,
    pub(crate) terminal: CompressionTerminalPayload<F>,
    relation: CompressionRelationOutput<F>,
}

/// All source chains in canonical relation order: B groups, then D.
pub(crate) struct CompressionWitnessMaterialization<F: Field> {
    sources: Vec<CompressionSourceWitness<F>>,
}

impl<F: Field> CompressionWitnessMaterialization<F> {
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

impl<F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize>
    CompressionSourceWitness<F>
{
    pub(crate) fn from_outer_hint(
        group_index: usize,
        plan: &CompressionChainPlan,
        hint: &AkitaCommitmentHint<F>,
        terminal_coefficients: Vec<F>,
        relation_mode: RingRelationMode,
    ) -> Result<Self, AkitaError> {
        let (witness, relation) = match relation_mode {
            RingRelationMode::QuotientLift => (
                hint.outer_compression_witness(plan)?,
                CompressionRelationOutput::QuotientLift {
                    quotients: hint.outer_compression_quotients(plan)?,
                },
            ),
            RingRelationMode::ReducedEvaluation => (
                hint.reduced_outer_compression_witness(plan)?,
                CompressionRelationOutput::ReducedEvaluation,
            ),
        };
        Ok(Self {
            id: CompressionSourceId::Outer { group_index },
            witness,
            terminal: CompressionTerminalPayload::new(plan.clone(), terminal_coefficients)?,
            relation,
        })
    }

    pub(crate) fn quotient(&self, map_index: usize) -> Result<&RingVec<F>, AkitaError> {
        match &self.relation {
            CompressionRelationOutput::QuotientLift { quotients } => {
                quotients.get(map_index).ok_or(AkitaError::InvalidProof)
            }
            CompressionRelationOutput::ReducedEvaluation => Err(AkitaError::InvalidProof),
        }
    }
}

fn into_source<F: Field>(
    output: CompressionExecutionOutput<CompressionSourceId, F>,
) -> CompressionSourceWitness<F> {
    CompressionSourceWitness {
        id: output.id,
        witness: output.witness,
        terminal: output.terminal,
        relation: output.relation,
    }
}

/// Execute every B/D chain using plans owned by the canonical relation layout.
pub(crate) fn materialize_compression_witness<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    layout: &RelationRhsLayout,
    mut outer_sources: Vec<CompressionSourceWitness<F>>,
    opening_rows: &RingVec<F>,
    relation_mode: RingRelationMode,
) -> Result<
    (
        CompressionWitnessMaterialization<F>,
        CompressionExecutionReport,
    ),
    AkitaError,
>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    B: CompressionComputeBackend<F>,
{
    if outer_sources.len() != layout.groups.len() {
        return Err(AkitaError::InvalidSetup(
            "retained outer compression source count disagrees with the relation layout".into(),
        ));
    }
    for (relation_group_index, source) in outer_sources.iter().enumerate() {
        let (group_index, plan) = layout.group_compression_plan(relation_group_index)?;
        let relation_matches = match (&source.relation, relation_mode) {
            (
                CompressionRelationOutput::QuotientLift { quotients },
                RingRelationMode::QuotientLift,
            ) => quotients.len() == plan.maps().len(),
            (CompressionRelationOutput::ReducedEvaluation, RingRelationMode::ReducedEvaluation) => {
                true
            }
            _ => false,
        };
        if source.id != (CompressionSourceId::Outer { group_index })
            || source.witness.plan() != plan
            || source.terminal.plan() != plan
            || !relation_matches
        {
            return Err(AkitaError::InvalidSetup(
                "retained outer compression source disagrees with the relation layout".into(),
            ));
        }
    }

    let opening_plan = layout.opening_compression_plan()?;
    if opening_rows.coeff_len() != opening_plan.source_coefficients() {
        return Err(AkitaError::InvalidSize {
            expected: opening_plan.source_coefficients(),
            actual: opening_rows.coeff_len(),
        });
    }
    let inputs = vec![CompressionExecutionInput {
        id: CompressionSourceId::Opening,
        plan: opening_plan.clone(),
        coefficients: opening_rows.coeffs().to_vec(),
        relation_mode,
    }];

    let (outputs, report) = execute_compression_chains(ctx, inputs)?;
    outer_sources.extend(outputs.into_iter().map(into_source));
    if outer_sources.len() != layout.groups.len() + 1 {
        return Err(AkitaError::InvalidSetup(
            "compression executor omitted a relation source".into(),
        ));
    }
    Ok((
        CompressionWitnessMaterialization {
            sources: outer_sources,
        },
        report,
    ))
}
