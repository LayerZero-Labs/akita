use super::*;
use crate::compute::{ComputeBackendSetup, CpuBackend};
use crate::AkitaProverSetup;
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime128OffsetA7F7 as F;
use akita_types::sis::{sis_table_key_for_linf_bound, AjtaiKeyParams, DEFAULT_SIS_SECURITY_BITS};
use akita_types::{
    validate_compression_catalog, CompressionAlphabet, CompressionCatalogContext,
    CompressionChainSpec, CompressionMapSpec, LevelParams, OpeningClaimsLayout, PreparedNttPlan,
    SetupMatrixEnvelope, SisModulusFamily,
};

fn oracle<const D: usize>(
    setup: &[F],
    digits: &[i8],
    rows: usize,
    log_basis: u32,
) -> (Vec<F>, Vec<i8>) {
    let (typed, remainder) = digits.as_chunks::<D>();
    assert!(remainder.is_empty());
    let cols = typed.len();
    assert!(setup.len() >= rows * cols * D);
    let mut neg = vec![F::zero(); rows * D];
    let mut quotient = vec![[F::zero(); D]; rows];
    for row in 0..rows {
        for col in 0..cols {
            let matrix = &setup[(row * cols + col) * D..(row * cols + col + 1) * D];
            for (left, &matrix_coeff) in matrix.iter().enumerate() {
                for (right, &digit) in typed[col].iter().enumerate() {
                    let product = matrix_coeff * F::from_i64(i64::from(digit));
                    let degree = left + right;
                    if degree < D {
                        neg[row * D + degree] += product;
                    } else {
                        neg[row * D + degree - D] -= product;
                        quotient[row][degree - D] += product;
                    }
                }
            }
        }
    }
    let levels = akita_types::r_decomp_levels::<F>(log_basis);
    let params =
        BalancedDecomposePow2I8Params::new(levels, log_basis, (-F::one()).to_canonical_u128() + 1);
    let mut quotient_digits = Vec::new();
    let mut planes = vec![[0i8; D]; levels];
    for coefficients in quotient {
        planes.fill([0; D]);
        let ring = akita_algebra::CyclotomicRing::from_coefficients(coefficients);
        ring.balanced_decompose_pow2_i8_into_with_params(&mut planes, &params);
        quotient_digits.extend(planes.iter().flatten().copied());
    }
    (neg, quotient_digits)
}

fn key(d: usize, raw_bound: u128, cols: usize) -> AjtaiKeyParams {
    let table = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_BITS,
        SisModulusFamily::Q128,
        d as u32,
        raw_bound,
    )
    .expect("test SIS row");
    AjtaiKeyParams::try_new_with_min_rank(table, cols).expect("certified key")
}

fn chain(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    alphabets: &[CompressionAlphabet],
) -> CompressionChainSpec {
    let mut previous = source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = alphabets
        .iter()
        .copied()
        .enumerate()
        .map(|(map, alphabet)| {
            let d = if map == 0 { 64 } else { 32 };
            let depth = match alphabet {
                CompressionAlphabet::NegativeBinary => F::modulus_bits() as usize,
                CompressionAlphabet::OpeningBase { log_basis } => {
                    akita_types::sis::num_digits_for_bound(
                        F::modulus_bits(),
                        F::modulus_bits(),
                        log_basis,
                    )
                }
            };
            let bound = match alphabet {
                CompressionAlphabet::NegativeBinary => 1,
                CompressionAlphabet::OpeningBase { log_basis } => (1u128 << log_basis) - 1,
            };
            let map_key = key(d, bound, previous * depth / d);
            previous = map_key.row_len() * d;
            CompressionMapSpec::new(map_key, alphabet)
        })
        .collect();
    CompressionChainSpec::new(source, maps)
}

fn fixture() -> (
    LevelParams,
    akita_types::ValidatedCompressionCatalog,
    AkitaProverSetup<F>,
) {
    let mut lp = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .expect("level");
    lp.b_key = key(64, 63, 1);
    lp.d_key = key(64, 63, 1);
    let opening = OpeningClaimsLayout::new(2, 1).expect("opening");
    let catalog = validate_compression_catalog::<F>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            chain(
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[
                    CompressionAlphabet::NegativeBinary,
                    CompressionAlphabet::NegativeBinary,
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
            chain(
                CompressionSourceId::Opening,
                &lp.d_key,
                &[
                    CompressionAlphabet::OpeningBase { log_basis: 6 },
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
        ],
    )
    .expect("catalog");
    let projection = catalog.project_for_schedule().expect("projection");
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        8,
        1,
        64,
        SetupMatrixEnvelope {
            max_setup_len: projection.max_flat_setup_prefix_coeffs(),
        },
    )
    .expect("setup");
    (lp, catalog, setup)
}

#[test]
fn materializes_layer_major_inputs_quotients_and_terminal_payloads() {
    let (lp, catalog, setup) = fixture();
    let projection = catalog.project_for_schedule().expect("projection");
    let prepared = CpuBackend
        .prepare_setup(
            &setup,
            &PreparedNttPlan::with_compression_requirements(
                setup.expanded.as_ref(),
                projection.ntt_requirements().iter().copied(),
            )
            .expect("NTT plan"),
        )
        .expect("prepared");
    let ctx = OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref()).expect("ctx");
    let layout = catalog.co_generated_relation_layout().expect("layout");
    let current_len = lp.b_key.row_len() * 64;
    let opening_len = lp.d_key.row_len() * 64;
    let current = vec![-F::one(); current_len];
    let opening = vec![F::from_u64(3); opening_len];
    let result = materialize_compression_witness(
        &ctx,
        layout,
        &[
            (CompressionSourceId::CurrentOuter, current.as_slice()),
            (CompressionSourceId::Opening, opening.as_slice()),
        ],
        lp.log_basis,
    )
    .expect("materialized witness");

    let semantic_compression_len = layout
        .segments()
        .iter()
        .filter(|segment| {
            matches!(
                segment.id(),
                RelationSegmentId::CompressionInput { .. }
                    | RelationSegmentId::CompressionQuotient { .. }
            )
        })
        .map(|segment| segment.span().len())
        .sum::<usize>();
    assert_eq!(result.suffix_digits.len(), semantic_compression_len);
    assert_eq!(result.terminal_payloads.len(), 2);
    assert_eq!(
        result.terminal_payloads[0].0,
        CompressionSourceId::CurrentOuter
    );
    assert_eq!(result.terminal_payloads[1].0, CompressionSourceId::Opening);
    let mut physical_cursor = result.suffix_start;
    for segment in layout.segments().iter().filter(|segment| {
        matches!(
            segment.id(),
            RelationSegmentId::CompressionInput { .. }
                | RelationSegmentId::CompressionQuotient { .. }
        )
    }) {
        let physical = layout
            .physical_compression_segment_span(segment.id())
            .unwrap();
        assert_eq!(physical.start(), physical_cursor);
        physical_cursor += physical.len();
    }
    assert_eq!(
        physical_cursor,
        result.suffix_start + result.suffix_digits.len()
    );

    let mut predecessors = [
        (CompressionSourceId::CurrentOuter, current.clone()),
        (CompressionSourceId::Opening, opening.clone()),
    ];
    for family in layout
        .row_plan()
        .families()
        .iter()
        .filter(|family| matches!(family.id(), RelationRowId::Compression { .. }))
    {
        let (source, map) = compression_id(family).unwrap();
        let RelationRowInputs::Compression { input, .. } = family.inputs() else {
            unreachable!()
        };
        let input_span = layout.physical_compression_segment_span(*input).unwrap();
        let quotient_span = layout
            .physical_compression_segment_span(family.quotient())
            .unwrap();
        let local_input = input_span.start() - result.suffix_start
            ..input_span.start() - result.suffix_start + input_span.len();
        assert!(result.suffix_digits[local_input]
            .iter()
            .any(|&value| value != 0));
        assert_eq!(
            quotient_span.len(),
            family.rows().len()
                * family.native_ring_dim()
                * akita_types::r_decomp_levels::<F>(lp.log_basis)
        );
        assert!(quotient_span.range().end <= result.suffix_start + result.suffix_digits.len());
        let (expected_image, expected_quotient) = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Compression,
            F,
            family.native_ring_dim(),
            |D| {
                Ok::<_, AkitaError>(oracle::<D>(
                    setup.expanded.shared_matrix().as_field_slice(),
                    &result.suffix_digits[input_span.start() - result.suffix_start
                        ..input_span.start() - result.suffix_start + input_span.len()],
                    family.rows().len(),
                    lp.log_basis,
                ))
            }
        )
        .unwrap();
        let stored_quotient = &result.suffix_digits[quotient_span.start() - result.suffix_start
            ..quotient_span.start() - result.suffix_start + quotient_span.len()];
        assert_eq!(stored_quotient, expected_quotient);

        let predecessor = &predecessors
            .iter()
            .find(|(candidate, _)| *candidate == source)
            .unwrap()
            .1;
        let input_digits = &result.suffix_digits[input_span.start() - result.suffix_start
            ..input_span.start() - result.suffix_start + input_span.len()];
        let depth = input_digits.len() / predecessor.len();
        let log_basis = input_log_basis(layout, source, map, *input).unwrap();
        let mut recomposed = vec![F::zero(); predecessor.len()];
        let mut weight = F::one();
        let basis = F::from_u64(1u64 << log_basis);
        for digit in 0..depth {
            for coeff in 0..predecessor.len() {
                recomposed[coeff] +=
                    F::from_i64(i64::from(input_digits[digit * predecessor.len() + coeff]))
                        * weight;
            }
            weight *= basis;
        }
        assert_eq!(&recomposed, predecessor);

        match family.rhs() {
            RelationRowRhs::TerminalPayload { coeffs } => {
                let payload = &result
                    .terminal_payloads
                    .iter()
                    .find(|(candidate, _)| *candidate == source)
                    .unwrap()
                    .1;
                assert_eq!(payload.len(), coeffs);
                assert_eq!(payload, &expected_image);
            }
            RelationRowRhs::Zero => {
                let RelationRowInputs::Compression {
                    successor: Some(successor),
                    ..
                } = family.inputs()
                else {
                    panic!("nonterminal compression row requires a successor")
                };
                let successor_span = layout
                    .physical_compression_segment_span(successor.segment())
                    .unwrap();
                let successor_digits = &result.suffix_digits[successor_span.start()
                    - result.suffix_start
                    ..successor_span.start() - result.suffix_start + successor_span.len()];
                let output_len = family.rows().len() * family.native_ring_dim();
                let successor_depth = successor_digits.len() / output_len;
                let mut image = vec![F::zero(); output_len];
                let mut weight = F::one();
                let basis = F::from_u64(1u64 << successor.log_basis());
                for digit in 0..successor_depth {
                    for coeff in 0..output_len {
                        image[coeff] +=
                            F::from_i64(i64::from(successor_digits[digit * output_len + coeff]))
                                * weight;
                    }
                    weight *= basis;
                }
                predecessors
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == source)
                    .unwrap()
                    .1 = image;
                assert_eq!(
                    &predecessors
                        .iter()
                        .find(|(candidate, _)| *candidate == source)
                        .unwrap()
                        .1,
                    &expected_image
                );
            }
            _ => panic!("compression rows have only zero or terminal RHS"),
        }
    }
    for span in layout.physical_negative_binary_support().unwrap() {
        let local =
            span.start() - result.suffix_start..span.start() - result.suffix_start + span.len();
        assert!(result.suffix_digits[local]
            .iter()
            .all(|&value| value == 0 || value == -1));
    }
}

#[test]
fn rejects_missing_and_duplicate_source_images() {
    let (lp, catalog, setup) = fixture();
    let prepared = CpuBackend
        .prepare_setup(
            &setup,
            &PreparedNttPlan::with_compression_requirements(
                setup.expanded.as_ref(),
                catalog
                    .project_for_schedule()
                    .unwrap()
                    .ntt_requirements()
                    .iter()
                    .copied(),
            )
            .unwrap(),
        )
        .unwrap();
    let ctx = OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref()).unwrap();
    let layout = catalog.co_generated_relation_layout().unwrap();
    let current = vec![F::one(); lp.b_key.row_len() * 64];
    assert!(materialize_compression_witness(
        &ctx,
        layout,
        &[(CompressionSourceId::CurrentOuter, current.as_slice())],
        lp.log_basis,
    )
    .is_err());
    let short = &current[..current.len() - 1];
    let half = &current[..current.len() / 2];
    let opening = vec![F::one(); lp.d_key.row_len() * 64];
    assert!(materialize_compression_witness(
        &ctx,
        layout,
        &[
            (CompressionSourceId::CurrentOuter, short),
            (CompressionSourceId::Opening, opening.as_slice()),
        ],
        lp.log_basis,
    )
    .is_err());
    assert!(materialize_compression_witness(
        &ctx,
        layout,
        &[
            (CompressionSourceId::CurrentOuter, half),
            (CompressionSourceId::Opening, opening.as_slice()),
        ],
        lp.log_basis,
    )
    .is_err());
    assert!(materialize_compression_witness(
        &ctx,
        layout,
        &[
            (CompressionSourceId::CurrentOuter, current.as_slice()),
            (CompressionSourceId::Opening, opening.as_slice()),
            (
                CompressionSourceId::PrecommittedOuter { index: 0 },
                current.as_slice(),
            ),
        ],
        lp.log_basis,
    )
    .is_err());
    assert!(materialize_compression_witness(
        &ctx,
        layout,
        &[
            (CompressionSourceId::CurrentOuter, current.as_slice()),
            (CompressionSourceId::CurrentOuter, current.as_slice()),
        ],
        lp.log_basis,
    )
    .is_err());
}
