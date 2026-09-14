//! Canonical materialization of B/D compression chains for one ring relation.

use crate::commitment::PortableCompressionState;
use crate::compute::compression::{
    execute_compression_chains, CompressionExecutionInput, CompressionExecutionOutput,
    CompressionExecutionReport, CompressionRelationOutput,
};
use crate::compute::{CompressionComputeBackend, OperationCtx};
use akita_error::AkitaError;
use akita_types::{
    CompressionChainPlan, CompressionChainWitness, CompressionTerminalPayload, RelationRhsLayout,
    RingRelationMode, RingVec,
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
    material: PortableCompressionState<F>,
    pub(crate) terminal: CompressionTerminalPayload<F>,
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
    pub(crate) fn from_outer_state(
        group_index: usize,
        plan: &CompressionChainPlan,
        material: PortableCompressionState<F>,
        terminal_coefficients: Vec<F>,
        relation_mode: RingRelationMode,
    ) -> Result<Self, AkitaError> {
        material.validate(plan, relation_mode)?;
        Ok(Self {
            id: CompressionSourceId::Outer { group_index },
            material,
            terminal: CompressionTerminalPayload::new(plan.clone(), terminal_coefficients)?,
        })
    }
}

impl<F: Field> CompressionSourceWitness<F> {
    pub(crate) fn witness(&self) -> &CompressionChainWitness {
        self.material.witness()
    }

    pub(crate) fn quotient(&self, map_index: usize) -> Result<&RingVec<F>, AkitaError> {
        self.material
            .quotients()
            .and_then(|quotients| quotients.get(map_index))
            .ok_or(AkitaError::InvalidProof)
    }
}

fn into_source<F: Field>(
    output: CompressionExecutionOutput<CompressionSourceId, F>,
) -> Result<CompressionSourceWitness<F>, AkitaError> {
    let material = match output.relation {
        CompressionRelationOutput::QuotientLift { quotients } => {
            PortableCompressionState::quotient_lift(output.witness, quotients)?
        }
        CompressionRelationOutput::ReducedEvaluation => {
            PortableCompressionState::reduced_evaluation(output.witness)?
        }
    };
    Ok(CompressionSourceWitness {
        id: output.id,
        material,
        terminal: output.terminal,
    })
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
        if source.id != (CompressionSourceId::Outer { group_index })
            || source.material.validate(plan, relation_mode).is_err()
            || source.terminal.plan() != plan
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
    outer_sources.extend(
        outputs
            .into_iter()
            .map(into_source)
            .collect::<Result<Vec<_>, _>>()?,
    );
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
