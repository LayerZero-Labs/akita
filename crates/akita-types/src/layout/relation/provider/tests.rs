use super::*;
use crate::layout::compression::{
    validate_compression_catalog, CompressionAlphabet, CompressionCatalogContext,
    CompressionChainSpec, CompressionMapSpec, CompressionSourceId,
};
use crate::schedule::PrecommittedGroupParams;
use crate::sis::{sis_table_key_for_linf_bound, AjtaiKeyParams, DEFAULT_SIS_SECURITY_BITS};
use crate::{
    LevelParams, OpeningClaimsLayout, PolynomialGroupLayout, PrecommittedLevelParams,
    RelationGroupId, RelationRowId, SisModulusFamily,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime128OffsetA7F7 as F;

mod mixed_consistency;

fn key(d: usize, bound: u128, columns: usize) -> AjtaiKeyParams {
    let table = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_BITS,
        SisModulusFamily::Q128,
        d as u32,
        bound,
    )
    .unwrap();
    AjtaiKeyParams::try_new_with_min_rank(table, columns).unwrap()
}

fn chain(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    dimensions: &[usize],
) -> CompressionChainSpec {
    let mut output_coeffs =
        source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = dimensions
        .iter()
        .copied()
        .map(|d| {
            let input_coeffs = output_coeffs * 128;
            let map_key = key(d, 1, input_coeffs / d);
            output_coeffs = map_key.row_len() * d;
            CompressionMapSpec::new(map_key, CompressionAlphabet::NegativeBinary)
        })
        .collect();
    CompressionChainSpec::new(source, 6, maps)
}

fn alphabet_chain(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    maps: &[(usize, CompressionAlphabet)],
) -> CompressionChainSpec {
    let mut output_coeffs =
        source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = maps
        .iter()
        .copied()
        .map(|(d, alphabet)| {
            let depth = crate::compression_digit_depth(alphabet, 128, 6).unwrap();
            let input_coeffs = output_coeffs * depth;
            let bound = match alphabet {
                CompressionAlphabet::NegativeBinary => 1,
                CompressionAlphabet::OpeningBase { log_basis } => (1u128 << log_basis) - 1,
            };
            let map_key = key(d, bound, input_coeffs / d);
            output_coeffs = map_key.row_len() * d;
            CompressionMapSpec::new(map_key, alphabet)
        })
        .collect();
    CompressionChainSpec::new(source, 6, maps)
}

fn mixed_layout() -> RelationLayout {
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
    .unwrap();
    lp.b_key = key(64, 63, 1);
    lp.d_key = key(64, 63, 1);
    let opening = OpeningClaimsLayout::new(2, 1).unwrap();
    validate_compression_catalog::<F>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            chain(CompressionSourceId::CurrentOuter, &lp.b_key, &[64, 32]),
            chain(CompressionSourceId::Opening, &lp.d_key, &[64, 32]),
        ],
    )
    .unwrap()
    .co_generated_relation_layout()
    .unwrap()
    .clone()
}

fn mixed_unequal_layout() -> RelationLayout {
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
    .unwrap();
    lp.b_key = key(64, 63, 1);
    lp.d_key = key(64, 63, 1);
    let opening = OpeningClaimsLayout::new(2, 1).unwrap();
    validate_compression_catalog::<F>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            chain(CompressionSourceId::CurrentOuter, &lp.b_key, &[64, 32]),
            alphabet_chain(
                CompressionSourceId::Opening,
                &lp.d_key,
                &[
                    (32, CompressionAlphabet::OpeningBase { log_basis: 6 }),
                    (64, CompressionAlphabet::NegativeBinary),
                ],
            ),
        ],
    )
    .unwrap()
    .co_generated_relation_layout()
    .unwrap()
    .clone()
}

fn base_layout() -> RelationLayout {
    let lp = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .unwrap();
    let opening = OpeningClaimsLayout::new(2, 1).unwrap();
    RelationLayout::from_authenticated_statement(
        &lp,
        &opening,
        crate::RelationMatrixRowLayout::WithDBlock,
        128,
    )
    .unwrap()
}

fn multi_group_layout() -> RelationLayout {
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
    .unwrap();
    let group = PolynomialGroupLayout::new(2, 1);
    lp.precommitted_groups.push(PrecommittedLevelParams {
        layout: PrecommittedGroupParams::from_params(group, &lp),
        a_key: lp.a_key.clone(),
        b_key: lp.b_key.clone(),
        num_blocks: 1,
        block_len: 1,
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold_one: 1,
    });
    let opening =
        OpeningClaimsLayout::from_root_groups(&[group], PolynomialGroupLayout::new(2, 1)).unwrap();
    RelationLayout::from_authenticated_statement(
        &lp,
        &opening,
        crate::RelationMatrixRowLayout::WithDBlock,
        128,
    )
    .unwrap()
}

fn powers(alpha: F, len: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(len);
    let mut power = F::one();
    for _ in 0..len {
        powers.push(power);
        power *= alpha;
    }
    powers
}

fn eval(coeffs: &[F], alpha_pows: &[F]) -> F {
    coeffs
        .iter()
        .zip(alpha_pows)
        .map(|(&coefficient, &power)| coefficient * power)
        .sum()
}

fn native_relation_oracle(
    setup: &[F],
    view: SharedSetupMatrixView,
    row: usize,
    column: usize,
    input: &[F],
    row_weight: F,
    alpha_pows: &[F],
) -> F {
    let d = view.native_ring_dim();
    let matrix_start = row * view.flat_row_coeffs() + column * d;
    let matrix = &setup[matrix_start..matrix_start + d];
    let mut cyclic = vec![F::zero(); 2 * d - 1];
    for (left, &matrix_coeff) in matrix.iter().enumerate() {
        for (right, &input_coeff) in input.iter().enumerate() {
            cyclic[left + right] += matrix_coeff * input_coeff;
        }
    }
    let mut negacyclic = cyclic[..d].to_vec();
    let mut quotient = vec![F::zero(); d];
    let high_len = cyclic.len() - d;
    quotient[..high_len].copy_from_slice(&cyclic[d..]);
    for (offset, &coefficient) in quotient[..high_len].iter().enumerate() {
        negacyclic[offset] -= coefficient;
    }
    let alpha = alpha_pows[1];
    let alpha_d_plus_one = alpha_pows[d - 1] * alpha + F::one();
    row_weight * (eval(&negacyclic, alpha_pows) + alpha_d_plus_one * eval(&quotient, alpha_pows))
}

#[test]
fn provider_resolves_mixed_native_views_and_terminal_rhs() {
    let layout = mixed_layout();
    let first = layout
        .family_provider(RelationRowId::Compression {
            source: CompressionSourceId::CurrentOuter,
            map: 0,
        })
        .unwrap();
    let terminal = layout
        .family_provider(RelationRowId::Compression {
            source: CompressionSourceId::CurrentOuter,
            map: 1,
        })
        .unwrap();
    assert_eq!(first.native_ring_dim(), 64);
    assert_eq!(terminal.native_ring_dim(), 32);
    assert!(first.compression_successor().unwrap().is_some());
    assert!(terminal.compression_successor().unwrap().is_none());
    assert_eq!(first.rhs(), RelationRowRhs::Zero);
    assert!(matches!(
        terminal.rhs(),
        RelationRowRhs::TerminalPayload { .. }
    ));
    for provider in [&first, &terminal] {
        let view = provider.compression_setup_view().unwrap();
        assert_eq!(
            view.flat_footprint(),
            view.rows() * provider.compression_input_span().unwrap().len()
        );
        assert_eq!(
            provider.quotient_span().unwrap().len() % provider.native_ring_dim(),
            0
        );
    }
}

#[test]
fn provider_validates_every_base_family_and_rejects_corruption() {
    let layout = base_layout();
    for family in layout.row_plan().families() {
        layout.family_provider(family.id()).unwrap();
    }

    let mut bad_rhs = layout.clone();
    bad_rhs.row_plan.families[0].rhs = RelationRowRhs::Opening;
    assert!(bad_rhs.family_provider(RelationRowId::Consistency).is_err());

    let mut empty_input = layout.clone();
    let z = match empty_input.row_plan.families[1].inputs {
        RelationRowInputs::A { z } => z,
        _ => unreachable!(),
    };
    empty_input
        .segments
        .iter_mut()
        .find(|segment| segment.id == z)
        .unwrap()
        .span
        .len = 0;
    assert!(empty_input
        .family_provider(empty_input.row_plan.families[1].id)
        .is_err());

    let mut bad_b = layout.clone();
    let b_index = bad_b
        .row_plan
        .families
        .iter()
        .position(|family| matches!(family.id, RelationRowId::B { .. }))
        .unwrap();
    bad_b.row_plan.families[b_index].rhs = RelationRowRhs::Zero;
    assert!(bad_b
        .family_provider(bad_b.row_plan.families[b_index].id)
        .is_err());

    let mut bad_d = layout.clone();
    let d_index = bad_d
        .row_plan
        .families
        .iter()
        .position(|family| matches!(family.id, RelationRowId::D))
        .unwrap();
    bad_d.row_plan.families[d_index].rhs = RelationRowRhs::Zero;
    assert!(bad_d.family_provider(RelationRowId::D).is_err());
}

#[test]
fn provider_rejects_row_quotient_successor_and_terminal_corruption() {
    let id = RelationRowId::Compression {
        source: CompressionSourceId::CurrentOuter,
        map: 0,
    };
    let mut bad_rows = mixed_layout();
    let index = bad_rows
        .row_plan
        .families
        .iter()
        .position(|family| family.id == id)
        .unwrap();
    bad_rows.row_plan.families[index].rows.start = bad_rows.row_plan.trace_row;
    assert!(bad_rows.family_provider(id).is_err());

    let mut bad_quotient = mixed_layout();
    let index = bad_quotient
        .row_plan
        .families
        .iter()
        .position(|family| family.id == id)
        .unwrap();
    let quotient = bad_quotient.row_plan.families[index].quotient;
    bad_quotient
        .segments
        .iter_mut()
        .find(|segment| segment.id == quotient)
        .unwrap()
        .span
        .len += 1;
    assert!(bad_quotient.family_provider(id).is_err());

    let mut bad_successor = mixed_layout();
    let index = bad_successor
        .row_plan
        .families
        .iter()
        .position(|family| family.id == id)
        .unwrap();
    let successor = match bad_successor.row_plan.families[index].inputs {
        RelationRowInputs::Compression {
            successor: Some(successor),
            ..
        } => successor.segment(),
        _ => unreachable!(),
    };
    bad_successor
        .segments
        .iter_mut()
        .find(|segment| segment.id == successor)
        .unwrap()
        .span
        .len += 1;
    assert!(bad_successor.family_provider(id).is_err());

    let terminal_id = RelationRowId::Compression {
        source: CompressionSourceId::CurrentOuter,
        map: 1,
    };
    let mut bad_terminal = mixed_layout();
    let index = bad_terminal
        .row_plan
        .families
        .iter()
        .position(|family| family.id == terminal_id)
        .unwrap();
    bad_terminal.row_plan.families[index].rhs = RelationRowRhs::TerminalPayload { coeffs: 1 };
    assert!(bad_terminal.family_provider(terminal_id).is_err());
}

#[test]
fn provider_rejects_typed_edge_identity_and_native_view_corruption() {
    let mut cross_group = multi_group_layout();
    let current_a = cross_group
        .row_plan
        .families
        .iter()
        .position(|family| {
            family.id
                == RelationRowId::A {
                    group: RelationGroupId::Current,
                }
        })
        .unwrap();
    cross_group.row_plan.families[current_a].inputs = RelationRowInputs::A {
        z: RelationSegmentId::Z {
            group: RelationGroupId::Precommitted { index: 0 },
        },
    };
    assert!(cross_group
        .family_provider(RelationRowId::A {
            group: RelationGroupId::Current,
        })
        .is_err());

    let mut wrong_order = multi_group_layout();
    let consistency = &mut wrong_order.row_plan.families[0];
    if let RelationRowInputs::Consistency { z, e } = &mut consistency.inputs {
        z.swap(0, 1);
        e.swap(0, 1);
    } else {
        unreachable!();
    }
    assert!(wrong_order
        .family_provider(RelationRowId::Consistency)
        .is_err());

    let mut wrong_variant = base_layout();
    let consistency_inputs = wrong_variant.row_plan.families[0].inputs.clone();
    let a_index = wrong_variant
        .row_plan
        .families
        .iter()
        .position(|family| matches!(family.id, RelationRowId::A { .. }))
        .unwrap();
    let a_id = wrong_variant.row_plan.families[a_index].id;
    wrong_variant.row_plan.families[a_index].inputs = consistency_inputs;
    assert!(wrong_variant.family_provider(a_id).is_err());

    let mut b_as_compression = mixed_layout();
    let b_index = b_as_compression
        .row_plan
        .families
        .iter()
        .position(|family| matches!(family.id, RelationRowId::B { .. }))
        .unwrap();
    let compression_inputs = b_as_compression
        .row_plan
        .families
        .iter()
        .find(|family| matches!(family.id, RelationRowId::Compression { .. }))
        .unwrap()
        .inputs
        .clone();
    let b_id = b_as_compression.row_plan.families[b_index].id;
    b_as_compression.row_plan.families[b_index].inputs = compression_inputs;
    assert!(b_as_compression.family_provider(b_id).is_err());

    let mut wrong_map = mixed_layout();
    let map0_id = RelationRowId::Compression {
        source: CompressionSourceId::CurrentOuter,
        map: 0,
    };
    let map0 = wrong_map
        .row_plan
        .families
        .iter()
        .position(|family| family.id == map0_id)
        .unwrap();
    if let RelationRowInputs::Compression { input, .. } =
        &mut wrong_map.row_plan.families[map0].inputs
    {
        *input = RelationSegmentId::CompressionInput {
            source: CompressionSourceId::CurrentOuter,
            map: 1,
        };
    } else {
        unreachable!();
    }
    assert!(wrong_map.family_provider(map0_id).is_err());

    let mut wrong_quotient = mixed_layout();
    let map0 = wrong_quotient
        .row_plan
        .families
        .iter()
        .position(|family| family.id == map0_id)
        .unwrap();
    wrong_quotient.row_plan.families[map0].quotient = RelationSegmentId::CompressionQuotient {
        source: CompressionSourceId::CurrentOuter,
        map: 1,
    };
    assert!(wrong_quotient.family_provider(map0_id).is_err());

    let mut bad_gadget = mixed_layout();
    let b = bad_gadget
        .row_plan
        .families
        .iter()
        .position(|family| {
            family.id
                == RelationRowId::B {
                    group: RelationGroupId::Current,
                }
        })
        .unwrap();
    if let RelationRowInputs::B {
        compression_input: Some(input),
        ..
    } = &mut bad_gadget.row_plan.families[b].inputs
    {
        input.log_basis = 128;
    } else {
        unreachable!();
    }
    assert!(bad_gadget
        .family_provider(RelationRowId::B {
            group: RelationGroupId::Current,
        })
        .is_err());

    let mut non_native = base_layout();
    let a_index = non_native
        .row_plan
        .families
        .iter()
        .position(|family| matches!(family.id, RelationRowId::A { .. }))
        .unwrap();
    let a_id = non_native.row_plan.families[a_index].id;
    let z = match non_native.row_plan.families[a_index].inputs {
        RelationRowInputs::A { z } => z,
        _ => unreachable!(),
    };
    non_native
        .segments
        .iter_mut()
        .find(|segment| segment.id == z)
        .unwrap()
        .span
        .len += 1;
    assert!(non_native.family_provider(a_id).is_err());

    let mut out_of_arena = base_layout();
    let a_index = out_of_arena
        .row_plan
        .families
        .iter()
        .position(|family| matches!(family.id, RelationRowId::A { .. }))
        .unwrap();
    let a_id = out_of_arena.row_plan.families[a_index].id;
    let quotient = out_of_arena.row_plan.families[a_index].quotient;
    let total = out_of_arena.total_coeffs;
    out_of_arena
        .segments
        .iter_mut()
        .find(|segment| segment.id == quotient)
        .unwrap()
        .span
        .start = total;
    assert!(out_of_arena.family_provider(a_id).is_err());
}

#[test]
fn sparse_native_weights_match_cyclic_relation_and_shared_prefix_coalescing() {
    let layout = mixed_layout();
    let current = layout
        .family_provider(RelationRowId::Compression {
            source: CompressionSourceId::CurrentOuter,
            map: 0,
        })
        .unwrap();
    let opening = layout
        .family_provider(RelationRowId::Compression {
            source: CompressionSourceId::Opening,
            map: 0,
        })
        .unwrap();
    let current_view = current.compression_setup_view().unwrap();
    let opening_view = opening.compression_setup_view().unwrap();
    assert_eq!(
        current_view, opening_view,
        "equal shapes share one prefix view"
    );

    let d = current_view.native_ring_dim();
    let alpha = F::from_u64(7);
    let alpha_pows = powers(alpha, d);
    let selected = [0usize, current_view.ring_columns() - 1];
    let input_rings = selected
        .iter()
        .map(|&column| {
            (0..d)
                .map(|lane| F::from_u64(2 + column as u64 + lane as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let column_evals = selected
        .iter()
        .copied()
        .zip(input_rings.iter().map(|ring| eval(ring, &alpha_pows)))
        .collect::<Vec<_>>();
    let row_weights = (0..current_view.rows())
        .map(|row| F::from_u64(row as u64 + 3))
        .collect::<Vec<_>>();
    let mut prefix = (0..current_view.flat_footprint())
        .map(|index| F::from_u64(index as u64 + 11))
        .collect::<Vec<_>>();
    let setup = prefix.clone();
    prefix.fill(F::zero());
    current
        .accumulate_compression_setup_weights(&mut prefix, &row_weights, &column_evals, &alpha_pows)
        .unwrap();
    opening
        .accumulate_compression_setup_weights(&mut prefix, &row_weights, &column_evals, &alpha_pows)
        .unwrap();
    let weighted_prefix: F = setup
        .iter()
        .zip(&prefix)
        .map(|(&coefficient, &weight)| coefficient * weight)
        .sum();

    // Dense oracle: multiply actual degree-d rings in F[X], split the
    // product into its negacyclic value and quotient, and evaluate
    // y(alpha) + (alpha^d + 1) q(alpha). This is exactly the setup-bearing
    // cyclic term of the native relation. Both equal-shape providers were
    // accumulated, hence the final factor two.
    let alpha_d_plus_one = alpha_pows[d - 1] * alpha + F::one();
    let mut oracle = F::zero();
    for (row, &row_weight) in row_weights.iter().enumerate() {
        let mut cyclic = vec![F::zero(); 2 * d - 1];
        for (&column, input) in selected.iter().zip(&input_rings) {
            let matrix_start = row * current_view.flat_row_coeffs() + column * d;
            let matrix = &setup[matrix_start..matrix_start + d];
            for (left, &matrix_coeff) in matrix.iter().enumerate() {
                for (right, &input_coeff) in input.iter().enumerate() {
                    cyclic[left + right] += matrix_coeff * input_coeff;
                }
            }
        }
        let mut negacyclic = cyclic[..d].to_vec();
        let mut quotient = vec![F::zero(); d];
        let high_len = cyclic.len() - d;
        quotient[..high_len].copy_from_slice(&cyclic[d..]);
        for (offset, &coefficient) in quotient[..high_len].iter().enumerate() {
            negacyclic[offset] -= coefficient;
        }
        oracle += row_weight
            * (eval(&negacyclic, &alpha_pows) + alpha_d_plus_one * eval(&quotient, &alpha_pows));
    }
    assert_eq!(weighted_prefix, oracle + oracle);
}

#[test]
fn mixed_dimensions_accumulate_into_one_flat_prefix_envelope() {
    let layout = mixed_unequal_layout();
    let provider64 = layout
        .family_provider(RelationRowId::Compression {
            source: CompressionSourceId::CurrentOuter,
            map: 0,
        })
        .unwrap();
    let provider32 = layout
        .family_provider(RelationRowId::Compression {
            source: CompressionSourceId::Opening,
            map: 0,
        })
        .unwrap();
    let view64 = provider64.compression_setup_view().unwrap();
    let view32 = provider32.compression_setup_view().unwrap();
    assert_eq!(view64.native_ring_dim(), 64);
    assert_eq!(view32.native_ring_dim(), 32);
    assert!(view64.flat_footprint() > view32.flat_footprint());
    let envelope = view64.flat_footprint().max(view32.flat_footprint());
    let setup = (0..envelope)
        .map(|index| F::from_u64(index as u64 + 19))
        .collect::<Vec<_>>();

    let alpha = F::from_u64(3);
    let pows64 = powers(alpha, 64);
    let pows32 = powers(alpha, 32);
    let row64 = view64.rows() - 1;
    let col64 = view64.ring_columns() - 1;
    let row32 = 0;
    let col32 = 0;
    let input64 = (0..64)
        .map(|lane| F::from_u64(lane as u64 + 2))
        .collect::<Vec<_>>();
    let input32 = (0..32)
        .map(|lane| F::from_u64(lane as u64 + 5))
        .collect::<Vec<_>>();
    let weight64 = F::from_u64(7);
    let weight32 = F::from_u64(11);
    let mut rows64 = vec![F::zero(); view64.rows()];
    rows64[row64] = weight64;
    let mut rows32 = vec![F::zero(); view32.rows()];
    rows32[row32] = weight32;
    let cols64 = [(col64, eval(&input64, &pows64))];
    let cols32 = [(col32, eval(&input32, &pows32))];

    let mut only64 = vec![F::zero(); envelope];
    provider64
        .accumulate_compression_setup_weights(&mut only64, &rows64, &cols64, &pows64)
        .unwrap();
    let mut combined = only64.clone();
    provider32
        .accumulate_compression_setup_weights(&mut combined, &rows32, &cols32, &pows32)
        .unwrap();
    let got: F = setup
        .iter()
        .zip(&combined)
        .map(|(&coefficient, &weight)| coefficient * weight)
        .sum();
    let expected =
        native_relation_oracle(&setup, view64, row64, col64, &input64, weight64, &pows64)
            + native_relation_oracle(&setup, view32, row32, col32, &input32, weight32, &pows32);
    assert_eq!(got, expected);
    assert_eq!(
        &combined[view32.flat_footprint()..],
        &only64[view32.flat_footprint()..]
    );
    assert!(combined[view32.flat_footprint()..]
        .iter()
        .any(|weight| !weight.is_zero()));
}

#[test]
fn d32_sparse_weights_match_native_cyclic_negacyclic_quotient_oracle() {
    let layout = mixed_layout();
    let provider = layout
        .family_provider(RelationRowId::Compression {
            source: CompressionSourceId::CurrentOuter,
            map: 1,
        })
        .unwrap();
    let view = provider.compression_setup_view().unwrap();
    assert_eq!(view.native_ring_dim(), 32);
    let d = view.native_ring_dim();
    let alpha = F::from_u64(5);
    let alpha_pows = powers(alpha, d);
    let column = view.ring_columns() - 1;
    let input = (0..d)
        .map(|lane| F::from_u64(lane as u64 + 2))
        .collect::<Vec<_>>();
    let column_evals = [(column, eval(&input, &alpha_pows))];
    let mut row_weights = vec![F::zero(); view.rows()];
    row_weights[view.rows() - 1] = F::from_u64(13);
    let mut setup = vec![F::zero(); view.flat_footprint()];
    let matrix_start = (view.rows() - 1) * view.flat_row_coeffs() + column * d;
    for (lane, coefficient) in setup[matrix_start..matrix_start + d].iter_mut().enumerate() {
        *coefficient = F::from_u64(lane as u64 + 17);
    }
    let mut weights = vec![F::zero(); view.flat_footprint()];
    provider
        .accumulate_compression_setup_weights(
            &mut weights,
            &row_weights,
            &column_evals,
            &alpha_pows,
        )
        .unwrap();
    let got: F = setup
        .iter()
        .zip(&weights)
        .map(|(&coefficient, &weight)| coefficient * weight)
        .sum();

    let matrix = &setup[matrix_start..matrix_start + d];
    let mut cyclic = vec![F::zero(); 2 * d - 1];
    for (left, &matrix_coeff) in matrix.iter().enumerate() {
        for (right, &input_coeff) in input.iter().enumerate() {
            cyclic[left + right] += matrix_coeff * input_coeff;
        }
    }
    let mut negacyclic = cyclic[..d].to_vec();
    let mut quotient = vec![F::zero(); d];
    let high_len = cyclic.len() - d;
    quotient[..high_len].copy_from_slice(&cyclic[d..]);
    for (offset, &coefficient) in quotient[..high_len].iter().enumerate() {
        negacyclic[offset] -= coefficient;
    }
    let alpha_d_plus_one = alpha_pows[d - 1] * alpha + F::one();
    let expected = row_weights[view.rows() - 1]
        * (eval(&negacyclic, &alpha_pows) + alpha_d_plus_one * eval(&quotient, &alpha_pows));
    assert_eq!(got, expected);
}
