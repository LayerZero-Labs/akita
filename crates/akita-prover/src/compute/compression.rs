//! Reusable bounded-memory compressed-commitment execution.

use super::{CompressionComputeBackend, OperationCtx};
use akita_error::AkitaError;
use akita_types::{
    dispatch_for_field, field_modulus, CompressionChainPlan, CompressionChainWitness,
    CompressionTerminalPayload, PackedNegativeBinary, RingRelationMode, RingVec,
};
use jolt_field::{CanonicalEncoding, Field};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Maximum number of expanded right-hand sides in one compression kernel call.
pub(crate) const MAX_COMPRESSION_RHS_BATCH: usize = 8;

/// One source and checked chain handed to the compression executor.
pub(crate) struct CompressionExecutionInput<Id, F> {
    pub(crate) id: Id,
    pub(crate) plan: CompressionChainPlan,
    pub(crate) coefficients: Vec<F>,
    pub(crate) relation_mode: RingRelationMode,
}

/// Relation-owned data retained from one compression chain.
pub(crate) enum CompressionRelationOutput<F: Field> {
    /// Quotient-lift mode retains one quotient image per map.
    QuotientLift { quotients: Vec<RingVec<F>> },
    /// Reduced-evaluation mode retains no quotient image.
    ReducedEvaluation,
}

impl<F: Field> CompressionRelationOutput<F> {
    #[cfg(test)]
    pub(crate) fn quotient_lift(&self) -> Result<&[RingVec<F>], AkitaError> {
        match self {
            Self::QuotientLift { quotients } => Ok(quotients),
            Self::ReducedEvaluation => Err(AkitaError::InvalidSetup(
                "reduced compression output has no quotient rows".into(),
            )),
        }
    }

    pub(crate) fn into_quotient_lift(self) -> Result<Vec<RingVec<F>>, AkitaError> {
        match self {
            Self::QuotientLift { quotients } => Ok(quotients),
            Self::ReducedEvaluation => Err(AkitaError::InvalidSetup(
                "reduced compression output has no quotient rows".into(),
            )),
        }
    }
}

/// One source's persistent compression result.
pub(crate) struct CompressionExecutionOutput<Id, F: Field> {
    pub(crate) id: Id,
    pub(crate) witness: CompressionChainWitness,
    pub(crate) terminal: CompressionTerminalPayload<F>,
    pub(crate) relation: CompressionRelationOutput<F>,
}

/// Measurements for one bounded exact-shape kernel batch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CompressionBatchReport {
    pub(crate) map_index: usize,
    pub(crate) relation_mode: RingRelationMode,
    pub(crate) ring_dimension: usize,
    pub(crate) input_width: usize,
    pub(crate) batch_size: usize,
    pub(crate) input_bytes: usize,
    pub(crate) output_bytes: usize,
    pub(crate) packed_bytes: usize,
    pub(crate) expanded_rhs_bytes: usize,
    pub(crate) digitization: Duration,
    /// Includes cold preparation when the exact compression cache slot is absent.
    pub(crate) kernel_including_prepare: Duration,
}

/// Aggregate bounded-execution measurements.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CompressionExecutionReport {
    pub(crate) sources: usize,
    pub(crate) maps: usize,
    pub(crate) source_bytes: usize,
    pub(crate) terminal_bytes: usize,
    pub(crate) retained_packed_witness_bytes: usize,
    pub(crate) equivalent_i8_witness_bytes: usize,
    pub(crate) max_expanded_rhs_bytes: usize,
    pub(crate) max_current_image_bytes: usize,
    pub(crate) executor_peak_scratch_bytes: usize,
    pub(crate) cache_bytes_before: Option<usize>,
    pub(crate) cache_bytes_after: Option<usize>,
    pub(crate) digitization: Duration,
    pub(crate) kernel_including_prepare: Duration,
    pub(crate) elapsed: Duration,
    pub(crate) batches: Vec<CompressionBatchReport>,
    pub(crate) quotient_lift_batches: usize,
    pub(crate) reduced_evaluation_batches: usize,
    pub(crate) quotient_rows: usize,
}

struct WorkItem<Id, F> {
    id: Id,
    plan: CompressionChainPlan,
    coefficients: Vec<F>,
    relation_mode: RingRelationMode,
    stages: Vec<PackedNegativeBinary>,
    quotients: Vec<RingVec<F>>,
}

fn quotient_from_products<F: Field, const D: usize>(
    cyclic: &[akita_algebra::CyclotomicRing<F, D>],
    negacyclic: &[akita_algebra::CyclotomicRing<F, D>],
) -> Result<RingVec<F>, AkitaError> {
    if cyclic.len() != negacyclic.len() {
        return Err(AkitaError::InvalidSetup(
            "compression cyclic and negacyclic ranks disagree".into(),
        ));
    }
    let coefficients = cyclic
        .iter()
        .zip(negacyclic)
        .flat_map(|(cyclic, negacyclic)| {
            cyclic
                .coefficients()
                .iter()
                .zip(negacyclic.coefficients())
                .map(|(&cyclic, &negacyclic)| (cyclic - negacyclic).half())
        })
        .collect();
    RingVec::from_coeffs_with_ring_dim(coefficients, D)
}

fn checked_sum_bytes(
    lengths: impl IntoIterator<Item = usize>,
    field_bytes: usize,
    context: &str,
) -> Result<usize, AkitaError> {
    lengths.into_iter().try_fold(0usize, |total, length| {
        length
            .checked_mul(field_bytes)
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))
    })
}

fn execute_chunk<F, B, Id, const D: usize>(
    ctx: &OperationCtx<'_, F, B>,
    items: &mut [WorkItem<Id, F>],
    item_indices: &[usize],
    map_index: usize,
) -> Result<CompressionBatchReport, AkitaError>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    if item_indices.is_empty() || item_indices.len() > MAX_COMPRESSION_RHS_BATCH {
        return Err(AkitaError::InvalidSetup(
            "compression executor received an invalid bounded batch".into(),
        ));
    }
    let first_map = items
        .get(item_indices[0])
        .and_then(|item| item.plan.maps().get(map_index))
        .copied()
        .ok_or_else(|| AkitaError::InvalidSetup("compression map is absent".into()))?;
    let relation_mode = items[item_indices[0]].relation_mode;
    if first_map.ring_dimension() != D
        || item_indices.iter().any(|&item_index| {
            items
                .get(item_index)
                .and_then(|item| item.plan.maps().get(map_index))
                .is_none_or(|map| {
                    items[item_index].relation_mode != relation_mode
                        || map.modulus_profile() != first_map.modulus_profile()
                        || map.ring_dimension() != first_map.ring_dimension()
                        || map.input_width() != first_map.input_width()
                        || map.output_rank() != first_map.output_rank()
                })
        })
    {
        return Err(AkitaError::InvalidSetup(
            "compression execution batch contains mixed shapes".into(),
        ));
    }
    let field_bytes = items[item_indices[0]].plan.field_bytes();
    let input_bytes = checked_sum_bytes(
        item_indices
            .iter()
            .map(|&item_index| items[item_index].coefficients.len()),
        field_bytes,
        "compression batch input bytes",
    )?;

    let digitization_started = Instant::now();
    let packed = item_indices
        .iter()
        .map(|&item_index| {
            let map = items[item_index]
                .plan
                .maps()
                .get(map_index)
                .copied()
                .ok_or_else(|| AkitaError::InvalidSetup("compression map is absent".into()))?;
            PackedNegativeBinary::from_coefficients(map, &items[item_index].coefficients)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expanded = packed
        .iter()
        .map(PackedNegativeBinary::expand_rows::<D>)
        .collect::<Result<Vec<_>, _>>()?;
    let digitization = digitization_started.elapsed();
    let packed_bytes = packed.iter().try_fold(0usize, |total, digits| {
        total.checked_add(digits.bytes().len()).ok_or_else(|| {
            AkitaError::InvalidSetup("compression packed batch bytes overflow".into())
        })
    })?;
    let expanded_rhs_bytes = expanded.iter().try_fold(0usize, |total, rows| {
        rows.len()
            .checked_mul(D)
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression expanded RHS bytes overflow".into())
            })
    })?;
    let views = expanded.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let kernel_started = Instant::now();
    let (negacyclic_outputs, cyclic_outputs) = match relation_mode {
        RingRelationMode::QuotientLift => {
            let products = ctx
                .backend()
                .compression_rows_products(ctx.prepared(), &views)?;
            let (negacyclic, cyclic): (Vec<_>, Vec<_>) = products
                .into_iter()
                .map(|products| (products.negacyclic, products.cyclic))
                .unzip();
            (negacyclic, Some(cyclic))
        }
        RingRelationMode::ReducedEvaluation => (
            ctx.backend()
                .compression_negacyclic_rows(ctx.prepared(), &views)?,
            None,
        ),
    };
    let kernel_including_prepare = kernel_started.elapsed();
    if negacyclic_outputs.len() != item_indices.len()
        || cyclic_outputs
            .as_ref()
            .is_some_and(|outputs| outputs.len() != item_indices.len())
    {
        return Err(AkitaError::InvalidSetup(
            "compression backend returned the wrong batch length".into(),
        ));
    }
    for (batch_index, ((&item_index, packed_digits), _)) in
        item_indices.iter().zip(packed).zip(expanded).enumerate()
    {
        let negacyclic = negacyclic_outputs
            .get(batch_index)
            .ok_or(AkitaError::InvalidProof)?;
        if negacyclic.len() != first_map.output_rank() {
            return Err(AkitaError::InvalidSetup(
                "compression backend returned the wrong output rank".into(),
            ));
        }
        if let Some(cyclic_outputs) = &cyclic_outputs {
            let cyclic = cyclic_outputs
                .get(batch_index)
                .ok_or(AkitaError::InvalidProof)?;
            if cyclic.len() != first_map.output_rank() {
                return Err(AkitaError::InvalidSetup(
                    "compression backend returned the wrong output rank".into(),
                ));
            }
            items[item_index]
                .quotients
                .push(quotient_from_products(cyclic, negacyclic)?);
        }
        let negacyclic_image = RingVec::from_ring_elems(negacyclic).coeffs().to_vec();
        if negacyclic_image.len() != first_map.output_coefficients() {
            return Err(AkitaError::InvalidSetup(
                "compression backend returned the wrong image length".into(),
            ));
        }
        items[item_index].coefficients = negacyclic_image;
        items[item_index].stages.push(packed_digits);
    }
    let output_bytes = checked_sum_bytes(
        item_indices
            .iter()
            .map(|&item_index| items[item_index].coefficients.len()),
        field_bytes,
        "compression batch output bytes",
    )?;
    Ok(CompressionBatchReport {
        map_index,
        relation_mode,
        ring_dimension: D,
        input_width: first_map.input_width(),
        batch_size: item_indices.len(),
        input_bytes,
        output_bytes,
        packed_bytes,
        expanded_rhs_bytes,
        digitization,
        kernel_including_prepare,
    })
}

fn execute_stage<F, B, Id>(
    ctx: &OperationCtx<'_, F, B>,
    items: &mut [WorkItem<Id, F>],
    map_index: usize,
) -> Result<Vec<CompressionBatchReport>, AkitaError>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    let mut groups = BTreeMap::<(RingRelationMode, usize, usize, usize), Vec<usize>>::new();
    for (item_index, item) in items.iter().enumerate() {
        if let Some(map) = item.plan.maps().get(map_index) {
            groups
                .entry((
                    item.relation_mode,
                    map.ring_dimension(),
                    map.input_width(),
                    map.output_rank(),
                ))
                .or_default()
                .push(item_index);
        }
    }
    let mut reports = Vec::new();
    for ((_, ring_dimension, _, _), item_indices) in groups {
        for chunk in item_indices.chunks(MAX_COMPRESSION_RHS_BATCH) {
            let report = dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Compression,
                F,
                ring_dimension,
                |D| execute_chunk::<F, B, Id, D>(ctx, items, chunk, map_index)
            )?;
            reports.push(report);
        }
    }
    Ok(reports)
}

/// Execute checked compression chains with bounded expansion scratch.
pub(crate) fn execute_compression_chains<F, B, Id>(
    ctx: &OperationCtx<'_, F, B>,
    inputs: Vec<CompressionExecutionInput<Id, F>>,
) -> Result<
    (
        Vec<CompressionExecutionOutput<Id, F>>,
        CompressionExecutionReport,
    ),
    AkitaError,
>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    let started = Instant::now();
    let cache_bytes_before = ctx.backend().compression_cache_bytes(ctx.prepared());
    let mut items = Vec::with_capacity(inputs.len());
    for input in inputs {
        if !input
            .plan
            .modulus_profile()
            .matches_modulus(field_modulus::<F>()?)
        {
            return Err(AkitaError::InvalidSetup(
                "compression plan profile does not match the execution field".into(),
            ));
        }
        if input.coefficients.len() != input.plan.source_coefficients() {
            return Err(AkitaError::InvalidSize {
                expected: input.plan.source_coefficients(),
                actual: input.coefficients.len(),
            });
        }
        items.push(WorkItem {
            id: input.id,
            plan: input.plan,
            coefficients: input.coefficients,
            relation_mode: input.relation_mode,
            stages: Vec::new(),
            quotients: Vec::new(),
        });
    }
    let source_bytes = items.iter().try_fold(0usize, |total, item| {
        item.coefficients
            .len()
            .checked_mul(item.plan.field_bytes())
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression source byte total overflow".into())
            })
    })?;
    let map_count = items.iter().try_fold(0usize, |total, item| {
        total
            .checked_add(item.plan.maps().len())
            .ok_or_else(|| AkitaError::InvalidSetup("compression map count overflow".into()))
    })?;
    let max_maps = items
        .iter()
        .map(|item| item.plan.maps().len())
        .max()
        .unwrap_or(0);
    let mut report = CompressionExecutionReport {
        sources: items.len(),
        maps: map_count,
        source_bytes,
        cache_bytes_before,
        ..CompressionExecutionReport::default()
    };
    for map_index in 0..max_maps {
        let current_image_bytes = items.iter().try_fold(0usize, |total, item| {
            item.coefficients
                .len()
                .checked_mul(item.plan.field_bytes())
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("compression current image bytes overflow".into())
                })
        })?;
        report.max_current_image_bytes = report.max_current_image_bytes.max(current_image_bytes);
        for batch in execute_stage(ctx, &mut items, map_index)? {
            match batch.relation_mode {
                RingRelationMode::QuotientLift => report.quotient_lift_batches += 1,
                RingRelationMode::ReducedEvaluation => report.reduced_evaluation_batches += 1,
            }
            report.digitization += batch.digitization;
            report.kernel_including_prepare += batch.kernel_including_prepare;
            report.max_expanded_rhs_bytes =
                report.max_expanded_rhs_bytes.max(batch.expanded_rhs_bytes);
            report.batches.push(batch);
        }
    }
    report.executor_peak_scratch_bytes = report
        .max_expanded_rhs_bytes
        .checked_add(report.max_current_image_bytes)
        .ok_or_else(|| AkitaError::InvalidSetup("compression peak scratch overflow".into()))?;

    let mut outputs = Vec::with_capacity(items.len());
    for item in items {
        let witness = CompressionChainWitness::new(item.plan.clone(), item.stages)?;
        report.retained_packed_witness_bytes = report
            .retained_packed_witness_bytes
            .checked_add(witness.retained_bytes()?)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression retained witness total overflow".into())
            })?;
        report.equivalent_i8_witness_bytes = report
            .equivalent_i8_witness_bytes
            .checked_add(item.plan.unpacked_witness_bytes()?)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression i8 witness total overflow".into())
            })?;
        let terminal = CompressionTerminalPayload::new(item.plan, item.coefficients)?;
        report.terminal_bytes = report
            .terminal_bytes
            .checked_add(
                terminal
                    .coefficients()
                    .len()
                    .checked_mul(terminal.plan().field_bytes())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("compression terminal bytes overflow".into())
                    })?,
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression terminal byte total overflow".into())
            })?;
        let relation = match item.relation_mode {
            RingRelationMode::QuotientLift => {
                report.quotient_rows = report
                    .quotient_rows
                    .checked_add(item.quotients.len())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("compression quotient row count overflow".into())
                    })?;
                CompressionRelationOutput::QuotientLift {
                    quotients: item.quotients,
                }
            }
            RingRelationMode::ReducedEvaluation => {
                if !item.quotients.is_empty() {
                    return Err(AkitaError::InvalidProof);
                }
                CompressionRelationOutput::ReducedEvaluation
            }
        };
        outputs.push(CompressionExecutionOutput {
            id: item.id,
            witness,
            terminal,
            relation,
        });
    }
    report.cache_bytes_after = ctx.backend().compression_cache_bytes(ctx.prepared());
    report.elapsed = started.elapsed();
    Ok((outputs, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{ComputeBackendSetup, CpuBackend};
    use crate::AkitaProverSetup;
    use akita_types::{SetupMatrixCapacity, SisModulusProfileId};
    use jolt_field::{Prime128OffsetA7F7, Ring};

    type F = Prime128OffsetA7F7;

    fn prepared_context(
        max_source_coefficients: usize,
    ) -> (
        AkitaProverSetup<F>,
        <CpuBackend as ComputeBackendSetup<F>>::PreparedSetup,
    ) {
        let plan = CompressionChainPlan::for_complete_source(
            SisModulusProfileId::Q128OffsetA7F7,
            max_source_coefficients,
        )
        .unwrap();
        let max_flat_coefficients = plan
            .maps()
            .iter()
            .map(|map| map.input_width() * map.ring_dimension())
            .max()
            .unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: max_flat_coefficients,
            },
        )
        .unwrap();
        let prepared = CpuBackend::DEFAULT
            .prepare_expanded(setup.expanded.clone())
            .unwrap();
        (setup, prepared)
    }

    fn input(id: usize, coefficients: usize) -> CompressionExecutionInput<usize, F> {
        CompressionExecutionInput {
            id,
            plan: CompressionChainPlan::for_complete_source(
                SisModulusProfileId::Q128OffsetA7F7,
                coefficients,
            )
            .unwrap(),
            coefficients: (0..coefficients)
                .map(|index| F::from_u64(index as u64 * 17 + id as u64 + 1))
                .collect(),
            relation_mode: RingRelationMode::QuotientLift,
        }
    }

    #[test]
    fn sequential_and_batched_execution_match_and_preserve_identity() {
        let (setup, prepared) = prepared_context(64);
        let ctx =
            OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref()).unwrap();
        let (batched, report) =
            execute_compression_chains(&ctx, vec![input(0, 64), input(1, 64)]).unwrap();
        let (first, _) = execute_compression_chains(&ctx, vec![input(0, 64)]).unwrap();
        let (second, _) = execute_compression_chains(&ctx, vec![input(1, 64)]).unwrap();
        assert_eq!(batched[0].id, 0);
        assert_eq!(batched[1].id, 1);
        assert_eq!(
            batched[0].terminal.coefficients(),
            first[0].terminal.coefficients()
        );
        assert_eq!(
            batched[1].terminal.coefficients(),
            second[0].terminal.coefficients()
        );
        assert_eq!(
            batched[0].relation.quotient_lift().unwrap(),
            first[0].relation.quotient_lift().unwrap()
        );
        assert_eq!(
            batched[1].relation.quotient_lift().unwrap(),
            second[0].relation.quotient_lift().unwrap()
        );
        assert_eq!(
            batched[0].relation.quotient_lift().unwrap().len(),
            akita_types::COMPRESSION_MAP_COUNT
        );
        assert_eq!(report.batches.len(), 2);
        assert!(report.batches.iter().all(|batch| batch.batch_size == 2));
    }

    #[test]
    fn mixed_shapes_partition_and_rhs_expansion_is_bounded() {
        let (setup, prepared) = prepared_context(65);
        let ctx =
            OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref()).unwrap();
        let mut inputs = (0..MAX_COMPRESSION_RHS_BATCH + 3)
            .map(|id| input(id, 64))
            .collect::<Vec<_>>();
        inputs.push(input(99, 65));
        let (_, report) = execute_compression_chains(&ctx, inputs).unwrap();
        assert!(report
            .batches
            .iter()
            .all(|batch| batch.batch_size <= MAX_COMPRESSION_RHS_BATCH));
        assert!(report.batches.iter().any(|batch| batch.batch_size == 1));
        assert!(report.max_expanded_rhs_bytes <= 65 * 128 * MAX_COMPRESSION_RHS_BATCH);
        assert_eq!(
            report.equivalent_i8_witness_bytes,
            report.retained_packed_witness_bytes * 8
        );
    }

    #[test]
    fn reduced_execution_builds_no_quotients_or_paired_compression_cache() {
        let (setup, prepared) = prepared_context(64);
        let ctx =
            OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref()).unwrap();
        let mut reduced = input(0, 64);
        reduced.relation_mode = RingRelationMode::ReducedEvaluation;

        let (outputs, report) = execute_compression_chains(&ctx, vec![reduced]).unwrap();

        assert!(matches!(
            outputs[0].relation,
            CompressionRelationOutput::ReducedEvaluation
        ));
        assert_eq!(report.quotient_lift_batches, 0);
        assert_eq!(report.reduced_evaluation_batches, report.maps);
        assert_eq!(report.quotient_rows, 0);
        assert!(prepared.compression_ntt_cache_bytes() > 0);
        assert!(prepared.ntt_cache_bytes().unwrap() > 0);
    }

    /// Run with:
    /// `cargo test -p akita-prover --release --no-default-features \
    /// compression_execution_bench \
    /// -- --ignored --nocapture`
    #[test]
    #[ignore = "release-only compression execution evidence"]
    fn compression_execution_bench() {
        const SAMPLES: usize = 30;
        let (setup, prepared) = prepared_context(64);
        let ctx =
            OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref()).unwrap();
        let (_, cold) = execute_compression_chains(&ctx, vec![input(0, 64)]).unwrap();
        let mut digitization_ns = Vec::with_capacity(SAMPLES);
        let mut cached_kernel_ns = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let (_, report) =
                execute_compression_chains(&ctx, vec![input(sample + 1, 64)]).unwrap();
            digitization_ns.push(report.digitization.as_nanos());
            cached_kernel_ns.push(report.kernel_including_prepare.as_nanos());
        }
        digitization_ns.sort_unstable();
        cached_kernel_ns.sort_unstable();
        let median_digitization_ns = digitization_ns[SAMPLES / 2];
        let median_cached_kernel_ns = cached_kernel_ns[SAMPLES / 2];
        let inferred_cold_preparation_ns = cold
            .kernel_including_prepare
            .as_nanos()
            .saturating_sub(median_cached_kernel_ns);
        println!(
            "compression_execution_bench profile=q128 source_bytes={} samples={SAMPLES} \
             median_digitization_ns={median_digitization_ns} \
             cold_kernel_including_prepare_ns={} \
             median_cached_kernel_ns={median_cached_kernel_ns} \
             inferred_cold_preparation_ns={inferred_cold_preparation_ns} \
             terminal_bytes={} retained_packed_witness_bytes={} \
             equivalent_i8_witness_bytes={} max_expanded_rhs_bytes={} \
             max_current_image_bytes={} executor_peak_scratch_bytes={} \
             cache_bytes_before={:?} cache_bytes_after={:?}",
            cold.source_bytes,
            cold.kernel_including_prepare.as_nanos(),
            cold.terminal_bytes,
            cold.retained_packed_witness_bytes,
            cold.equivalent_i8_witness_bytes,
            cold.max_expanded_rhs_bytes,
            cold.max_current_image_bytes,
            cold.executor_peak_scratch_bytes,
            cold.cache_bytes_before,
            cold.cache_bytes_after,
        );
    }
}
