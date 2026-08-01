//! Opt-in adapter from live protocol images to canonical compression execution.

use crate::compute::compression::{
    execute_compression_chains, CompressionExecutionInput, CompressionExecutionReport,
};
use crate::compute::{CompressionComputeBackend, OperationCtx};
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{field_modulus, CompressionChainPlan, SisModulusProfileId};

/// Origin of one live B/D image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompressionDiagnosticSourceKind {
    Outer { group_index: usize },
    Opening,
}

/// One live B/D image handed to the diagnostic adapter.
pub(crate) struct CompressionDiagnosticSource<'a, F> {
    pub(crate) kind: CompressionDiagnosticSourceKind,
    pub(crate) coefficients: &'a [F],
}

/// Compute real compressed commitments and discard all protocol-unbound output.
pub(crate) fn compute_shadow_compressed_commitments<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    profile: SisModulusProfileId,
    sources: &[CompressionDiagnosticSource<'_, F>],
) -> Result<CompressionExecutionReport, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    B: CompressionComputeBackend<F>,
{
    if !profile.matches_modulus(field_modulus::<F>()) {
        return Err(AkitaError::InvalidSetup(format!(
            "compression diagnostic profile {profile:?} does not match field modulus {}",
            field_modulus::<F>()
        )));
    }
    let inputs = sources
        .iter()
        .filter(|source| !source.coefficients.is_empty())
        .map(|source| {
            Ok(CompressionExecutionInput {
                id: source.kind,
                plan: CompressionChainPlan::for_complete_source(
                    profile,
                    source.coefficients.len(),
                )?,
                coefficients: source.coefficients.to_vec(),
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let (outputs, report) = execute_compression_chains(ctx, inputs)?;
    for batch in &report.batches {
        tracing::info!(
            map_index = batch.map_index,
            ring_dimension = batch.ring_dimension,
            input_width = batch.input_width,
            batch_size = batch.batch_size,
            input_bytes = batch.input_bytes,
            output_bytes = batch.output_bytes,
            packed_bytes = batch.packed_bytes,
            expanded_rhs_bytes = batch.expanded_rhs_bytes,
            digitization_micros = duration_micros(batch.digitization),
            kernel_including_prepare_micros = duration_micros(batch.kernel_including_prepare),
            "shadow compressed-commitment batch"
        );
    }
    for output in outputs {
        tracing::debug!(
            source = ?output.id,
            maps = output.witness.stages().len(),
            retained_packed_witness_bytes = output.witness.retained_bytes()?,
            terminal_bytes = output
                .terminal
                .coefficients()
                .len()
                .checked_mul(output.terminal.plan().field_bytes())
                .ok_or_else(|| AkitaError::InvalidSetup(
                    "compression terminal byte length overflow".into()
                ))?,
            "computed and discarded shadow compressed commitment"
        );
    }
    Ok(report)
}

fn duration_micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
