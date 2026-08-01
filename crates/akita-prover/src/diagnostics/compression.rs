//! Opt-in execution of compressed commitments without protocol effects.

use crate::compute::{DigitRowsComputeBackend, OperationCtx};
use crate::diagnostics::compression_plan::{plan_compression_diagnostic, CompressionDiagnosticMap};
use akita_algebra::balanced_decompose_coefficients_pow2_i8_into;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::sis::SisModulusProfileId;
use akita_types::{dispatch_for_field, field_modulus, RingVec};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Origin of one live B/D image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompressionDiagnosticSourceKind {
    Outer { group_index: usize },
    Opening,
}

/// One live B/D image handed to the diagnostic executor.
pub(crate) struct CompressionDiagnosticSource<'a, F> {
    pub(crate) kind: CompressionDiagnosticSourceKind,
    pub(crate) coefficients: &'a [F],
}

/// Aggregate facts emitted after shadow compression.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CompressionDiagnosticReport {
    pub(crate) sources: usize,
    pub(crate) maps: usize,
    pub(crate) batch_count: usize,
    pub(crate) source_bytes: usize,
    pub(crate) terminal_bytes: usize,
    pub(crate) cache_bytes_before: Option<usize>,
    pub(crate) cache_bytes_after: Option<usize>,
    pub(crate) elapsed: Duration,
}

struct WorkItem<F> {
    source: CompressionDiagnosticSourceKind,
    field_bytes: usize,
    maps: Vec<CompressionDiagnosticMap>,
    coefficients: Vec<F>,
}

fn negative_binary_digits<F: CanonicalField, const D: usize>(
    coefficients: &[F],
    input_width: usize,
) -> Result<Vec<[i8; D]>, AkitaError> {
    let field_bits = usize::try_from(F::modulus_bits())
        .map_err(|_| AkitaError::InvalidSetup("field bit length overflow".into()))?;
    let digit_coefficients = coefficients
        .len()
        .checked_mul(field_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("compression digit length overflow".into()))?;
    let capacity = input_width
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("compression digit capacity overflow".into()))?;
    if digit_coefficients > capacity {
        return Err(AkitaError::InvalidSetup(
            "compression map input width is undersized".into(),
        ));
    }
    let mut digits = vec![[0i8; D]; input_width];
    let q = (-F::one()).to_canonical_u128() + 1;
    let params =
        akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params::new(field_bits, 1, q);
    balanced_decompose_coefficients_pow2_i8_into(
        coefficients,
        &mut digits.as_flattened_mut()[..digit_coefficients],
        &params,
    );
    Ok(digits)
}

fn execute_group<F, B, const D: usize>(
    ctx: &OperationCtx<'_, F, B>,
    items: &mut [WorkItem<F>],
    item_indices: &[usize],
    map_index: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let first_map = items
        .get(*item_indices.first().ok_or_else(|| {
            AkitaError::InvalidSetup("compression execution group is empty".into())
        })?)
        .and_then(|item| item.maps.get(map_index))
        .copied()
        .ok_or_else(|| AkitaError::InvalidSetup("compression map is absent".into()))?;
    let input_bytes = item_indices.iter().try_fold(0usize, |total, &item_index| {
        let item = items
            .get(item_index)
            .ok_or_else(|| AkitaError::InvalidSetup("compression item index is invalid".into()))?;
        let bytes = item
            .coefficients
            .len()
            .checked_mul(item.field_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("compression input bytes overflow".into()))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("compression input byte total overflow".into()))
    })?;
    let digitization_started = Instant::now();
    let digit_vectors = item_indices
        .iter()
        .map(|&item_index| {
            let item = items.get(item_index).ok_or_else(|| {
                AkitaError::InvalidSetup("compression item index is invalid".into())
            })?;
            negative_binary_digits::<F, D>(&item.coefficients, first_map.input_width)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let digitization_elapsed = digitization_started.elapsed();
    let digit_views = digit_vectors.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let kernel_started = Instant::now();
    let outputs = ctx
        .backend()
        .compression_rows(ctx.prepared(), &digit_views)?;
    let kernel_elapsed = kernel_started.elapsed();
    if outputs.len() != item_indices.len() {
        return Err(AkitaError::InvalidSetup(
            "compression backend returned the wrong batch length".into(),
        ));
    }
    for (&item_index, rows) in item_indices.iter().zip(outputs) {
        let coefficients = RingVec::from_ring_elems(&rows).coeffs().to_vec();
        if coefficients.len() != first_map.ring_dimension {
            return Err(AkitaError::InvalidSetup(
                "compression backend returned the wrong image length".into(),
            ));
        }
        items[item_index].coefficients = coefficients;
    }
    let output_bytes = item_indices.iter().try_fold(0usize, |total, &item_index| {
        let item = items
            .get(item_index)
            .ok_or_else(|| AkitaError::InvalidSetup("compression item index is invalid".into()))?;
        let bytes = item
            .coefficients
            .len()
            .checked_mul(item.field_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("compression output bytes overflow".into()))?;
        total.checked_add(bytes).ok_or_else(|| {
            AkitaError::InvalidSetup("compression output byte total overflow".into())
        })
    })?;
    tracing::info!(
        map_index,
        ring_dimension = D,
        input_width = first_map.input_width,
        batch_size = item_indices.len(),
        input_bytes,
        output_bytes,
        digitization_micros = duration_micros(digitization_elapsed),
        kernel_micros = duration_micros(kernel_elapsed),
        "shadow compressed-commitment batch"
    );
    Ok(())
}

fn execute_stage<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    items: &mut [WorkItem<F>],
    map_index: usize,
) -> Result<usize, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let mut groups = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (item_index, item) in items.iter().enumerate() {
        if let Some(map) = item.maps.get(map_index) {
            groups
                .entry((map.ring_dimension, map.input_width))
                .or_default()
                .push(item_index);
        }
    }
    let batch_count = groups.len();
    for ((ring_dimension, _), item_indices) in groups {
        dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Compression,
            F,
            ring_dimension,
            |D| execute_group::<F, B, D>(ctx, items, &item_indices, map_index)
        )?;
    }
    Ok(batch_count)
}

/// Compute compressed commitments for live B/D images and retain only metrics.
pub(crate) fn compute_shadow_compressed_commitments<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    profile: SisModulusProfileId,
    sources: &[CompressionDiagnosticSource<'_, F>],
) -> Result<CompressionDiagnosticReport, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    if !profile.matches_modulus(field_modulus::<F>()) {
        return Err(AkitaError::InvalidSetup(format!(
            "compression diagnostic profile {profile:?} does not match field modulus {}",
            field_modulus::<F>()
        )));
    }
    let diagnostic_started = Instant::now();
    let cache_bytes_before = ctx.backend().compression_cache_bytes(ctx.prepared());
    let mut items = Vec::new();
    for source in sources {
        if source.coefficients.is_empty() {
            continue;
        }
        let plan = plan_compression_diagnostic(profile, source.coefficients.len())?;
        items.push(WorkItem {
            source: source.kind,
            field_bytes: plan.field_bytes,
            maps: plan.maps,
            coefficients: source.coefficients.to_vec(),
        });
    }
    let source_bytes = items.iter().try_fold(0usize, |total, item| {
        let bytes = item
            .coefficients
            .len()
            .checked_mul(item.field_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("compression source bytes overflow".into()))?;
        total.checked_add(bytes).ok_or_else(|| {
            AkitaError::InvalidSetup("compression source byte total overflow".into())
        })
    })?;
    let map_count = items.iter().map(|item| item.maps.len()).sum();
    let max_maps = items.iter().map(|item| item.maps.len()).max().unwrap_or(0);
    let mut batch_count = 0usize;
    for map_index in 0..max_maps {
        batch_count = batch_count
            .checked_add(execute_stage(ctx, &mut items, map_index)?)
            .ok_or_else(|| AkitaError::InvalidSetup("compression batch count overflow".into()))?;
    }
    let terminal_bytes = items.iter().try_fold(0usize, |total, item| {
        let bytes = item
            .coefficients
            .len()
            .checked_mul(item.field_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal byte length overflow".into()))?;
        tracing::debug!(
            source = ?item.source,
            maps = item.maps.len(),
            terminal_bytes = bytes,
            "computed shadow compressed commitment"
        );
        total
            .checked_add(bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal byte total overflow".into()))
    })?;
    let cache_bytes_after = ctx.backend().compression_cache_bytes(ctx.prepared());
    Ok(CompressionDiagnosticReport {
        sources: items.len(),
        maps: map_count,
        batch_count,
        source_bytes,
        terminal_bytes,
        cache_bytes_before,
        cache_bytes_after,
        elapsed: diagnostic_started.elapsed(),
    })
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{ComputeBackendSetup, CpuBackend};
    use crate::kernels::linear::mat_vec_mul_ntt_digits_i8;
    use crate::AkitaProverSetup;
    use akita_algebra::CyclotomicRing;
    use akita_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
    use akita_types::layout::FlatMatrix;
    use akita_types::prepare_compression_ntt_cache;
    use akita_types::SetupMatrixCapacity;
    use std::hint::black_box;

    fn assert_negative_binary_digits<F: FieldCore + CanonicalField, const D: usize>() {
        let values = [F::zero(), F::one(), F::from_u64(17), -F::from_u64(9)];
        let field_bits = F::modulus_bits() as usize;
        let input_width = values.len() * field_bits / D;
        let digits =
            negative_binary_digits::<F, D>(&values, input_width).expect("negative-binary digits");
        let mut reference = vec![[0i8; D]; input_width];
        let modulus = field_modulus::<F>();
        for bit in 0..field_bits {
            for (coefficient_index, coefficient) in values.iter().enumerate() {
                let canonical = coefficient.to_canonical_u128();
                let magnitude = if canonical == 0 {
                    0
                } else {
                    modulus - canonical
                };
                let linear = bit * values.len() + coefficient_index;
                reference[linear / D][linear % D] = -(((magnitude >> bit) & 1) as i8);
            }
        }
        assert_eq!(digits, reference);

        for (coefficient_index, expected) in values.iter().copied().enumerate() {
            let mut actual = F::zero();
            let mut power = F::one();
            for bit in 0..field_bits {
                actual += F::from_i8(
                    digits[(bit * values.len() + coefficient_index) / D]
                        [(bit * values.len() + coefficient_index) % D],
                ) * power;
                power += power;
            }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn negative_binary_digits_preserve_the_compression_layout() {
        assert_negative_binary_digits::<Prime128OffsetA7F7, 16>();
        assert_negative_binary_digits::<Prime64Offset59, 16>();
        assert_negative_binary_digits::<Prime32Offset99, 32>();
    }

    #[test]
    fn aggregate_report_distinguishes_logical_maps_from_equal_shape_batches() {
        type F = Prime64Offset59;
        const D: usize = 64;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: 256,
            },
        )
        .expect("diagnostic setup");
        let prepared = CpuBackend
            .prepare_expanded(setup.expanded.clone())
            .expect("prepared setup");
        let ctx = OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("operation context");
        let coefficients = vec![F::one(); 64];
        let sources = [
            CompressionDiagnosticSource {
                kind: CompressionDiagnosticSourceKind::Outer { group_index: 0 },
                coefficients: &coefficients,
            },
            CompressionDiagnosticSource {
                kind: CompressionDiagnosticSourceKind::Outer { group_index: 1 },
                coefficients: &coefficients,
            },
        ];

        let report =
            compute_shadow_compressed_commitments(&ctx, SisModulusProfileId::Q64Offset59, &sources)
                .expect("shadow compression");

        assert_eq!(report.sources, 2);
        assert_eq!(report.maps, 4);
        assert_eq!(report.batch_count, 2);
    }

    fn deterministic_matrix_row<F: FieldCore + CanonicalField, const D: usize>(
        column_count: usize,
        salt: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        (0..column_count)
            .map(|column| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_i64(((salt * 17 + column * 7 + coefficient * 3) % 19) as i64 - 9)
                }))
            })
            .collect()
    }

    fn schoolbook_rank_one_digit_mat_vec<F: FieldCore + CanonicalField, const D: usize>(
        matrix_row: &[CyclotomicRing<F, D>],
        digits: &[[i8; D]],
    ) -> CyclotomicRing<F, D> {
        matrix_row.iter().zip(digits.iter()).fold(
            CyclotomicRing::<F, D>::zero(),
            |mut acc, (lhs, digit)| {
                let rhs = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_i64(i64::from(digit[k]))
                }));
                acc += *lhs * rhs;
                acc
            },
        )
    }

    fn compress_map_stage_against_schoolbook<F: FieldCore + CanonicalField, const D: usize>(
        coefficients: &[F],
        map: CompressionDiagnosticMap,
        salt: usize,
    ) -> Vec<F> {
        assert_eq!(map.ring_dimension, D);
        let digits =
            negative_binary_digits::<F, D>(coefficients, map.input_width).expect("digitization");
        let matrix_row = deterministic_matrix_row::<F, D>(map.input_width, salt);
        let flat = FlatMatrix::from_ring_slice(&matrix_row);
        let slot = prepare_compression_ntt_cache(
            flat.ring_view::<D>(1, map.input_width)
                .expect("compression matrix view"),
            map.input_width,
        )
        .expect("exact-prefix compression cache");
        assert!(!slot.has_cyclic());
        let actual =
            mat_vec_mul_ntt_digits_i8::<F, D>(&slot, 1, map.input_width, &[digits.as_slice()], 1)
                .expect("kernel");
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].len(), 1);
        let expected = schoolbook_rank_one_digit_mat_vec(&matrix_row, &digits);
        assert_eq!(actual[0][0], expected);
        RingVec::from_ring_elems(&actual[0]).coeffs().to_vec()
    }

    fn run_ladder_against_schoolbook<F: FieldCore + CanonicalField>(
        profile: SisModulusProfileId,
        source_bytes: usize,
    ) {
        let field_bytes = (F::modulus_bits() as usize).div_ceil(8);
        let source_coefficients = source_bytes / field_bytes;
        let plan = plan_compression_diagnostic(profile, source_coefficients).expect("plan");
        let mut coefficients = (0..source_coefficients)
            .map(|index| F::from_u64((index as u64).wrapping_mul(0x9e37).wrapping_add(0x1234)))
            .collect::<Vec<_>>();
        for (map_index, map) in plan.maps.iter().copied().enumerate() {
            coefficients = match map.ring_dimension {
                8 => compress_map_stage_against_schoolbook::<F, 8>(&coefficients, map, map_index),
                16 => compress_map_stage_against_schoolbook::<F, 16>(&coefficients, map, map_index),
                32 => compress_map_stage_against_schoolbook::<F, 32>(&coefficients, map, map_index),
                64 => compress_map_stage_against_schoolbook::<F, 64>(&coefficients, map, map_index),
                128 => {
                    compress_map_stage_against_schoolbook::<F, 128>(&coefficients, map, map_index)
                }
                other => panic!("unexpected compression ring dimension {other}"),
            };
            assert_eq!(coefficients.len(), map.ring_dimension);
            assert_eq!(
                coefficients.len() * field_bytes,
                match map_index + 1 == plan.maps.len() {
                    true => 128,
                    false if plan.maps.len() == 2 => 256,
                    false if plan.maps.len() == 3 && map_index == 0 => 512,
                    false if plan.maps.len() == 3 && map_index == 1 => 256,
                    false => panic!("unexpected intermediate size"),
                }
            );
        }
        assert_eq!(coefficients.len() * field_bytes, 128);
    }

    #[test]
    fn one_kib_ladders_match_schoolbook_on_exact_prefix_cache() {
        run_ladder_against_schoolbook::<Prime128OffsetA7F7>(
            SisModulusProfileId::Q128OffsetA7F7,
            1024,
        );
        run_ladder_against_schoolbook::<Prime64Offset59>(SisModulusProfileId::Q64Offset59, 1024);
        run_ladder_against_schoolbook::<Prime32Offset99>(SisModulusProfileId::Q32Offset99, 1024);
    }

    #[test]
    fn sixteen_kib_ladders_match_schoolbook_on_exact_prefix_cache() {
        run_ladder_against_schoolbook::<Prime128OffsetA7F7>(
            SisModulusProfileId::Q128OffsetA7F7,
            16 * 1024,
        );
        run_ladder_against_schoolbook::<Prime64Offset59>(
            SisModulusProfileId::Q64Offset59,
            16 * 1024,
        );
        run_ladder_against_schoolbook::<Prime32Offset99>(
            SisModulusProfileId::Q32Offset99,
            16 * 1024,
        );
    }

    /// Run with:
    /// `cargo test -p akita-prover --release --features compression-diagnostics negative_binary_digitization_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "release-only compression digitization microbenchmark"]
    fn negative_binary_digitization_bench() {
        type F = Prime128OffsetA7F7;
        const D: usize = 16;
        const ITERATIONS: usize = 10_000;
        let values = (0..64)
            .map(|index| F::from_u64((index * 0x9e37 + 0x1234) as u64))
            .collect::<Vec<_>>();

        let started = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(
                negative_binary_digits::<F, D>(black_box(&values), 512).expect("digitization"),
            );
        }
        let elapsed = started.elapsed();
        println!(
            "negative-binary 1 KiB source: {} ns/iteration",
            elapsed.as_nanos() / ITERATIONS as u128
        );
    }
}
