use crate::{
    emit_witness_e_planes, emit_witness_r_planes, emit_witness_t_planes, emit_witness_z_planes,
    OpeningBatchWitnessGroup, OpeningBatchWitnessLayout, OpeningBlockLayout, SemanticGroupId,
    TraceWeightLayout, WitnessOwnershipUnit,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_field::Prime128OffsetA7F7;

const D: usize = 2;

fn oracle_unit_base(layout: &OpeningBatchWitnessLayout, target: &WitnessOwnershipUnit) -> usize {
    let mut cursor = 0usize;
    for &group_id in &layout.relation_group_order {
        let group = &layout.groups[group_id.0];
        let chunks = if layout.groups.len() == 1 {
            layout.num_machine_chunks()
        } else {
            1
        };
        let blocks = group.live_fold_count / chunks;
        for chunk in 0..chunks {
            if group_id == target.group && chunk == target.machine_chunk.0 {
                return cursor;
            }
            cursor += group.fold_position_count * group.depth_commit * group.depth_fold;
            cursor += group.num_claims * blocks * group.depth_open;
            cursor += group.num_claims * blocks * group.n_a * group.depth_open;
        }
    }
    panic!("oracle target unit is absent");
}

fn oracle_e(
    layout: &OpeningBatchWitnessLayout,
    unit: &WitnessOwnershipUnit,
    claim: usize,
    block: usize,
    digit: usize,
) -> usize {
    let group = &layout.groups[unit.group.0];
    oracle_unit_base(layout, unit)
        + group.fold_position_count * group.depth_commit * group.depth_fold
        + digit
        + group.depth_open * ((block - unit.global_block_base) + unit.blocks * claim)
}

fn oracle_t(
    layout: &OpeningBatchWitnessLayout,
    unit: &WitnessOwnershipUnit,
    claim: usize,
    block: usize,
    a_row: usize,
    digit: usize,
) -> usize {
    let group = &layout.groups[unit.group.0];
    oracle_unit_base(layout, unit)
        + group.fold_position_count * group.depth_commit * group.depth_fold
        + group.num_claims * unit.blocks * group.depth_open
        + digit
        + group.depth_open
            * (a_row + group.n_a * ((block - unit.global_block_base) + unit.blocks * claim))
}

fn oracle_z(
    layout: &OpeningBatchWitnessLayout,
    unit: &WitnessOwnershipUnit,
    position: usize,
    commit_digit: usize,
    fold_digit: usize,
) -> usize {
    let group = &layout.groups[unit.group.0];
    oracle_unit_base(layout, unit)
        + fold_digit
        + group.depth_fold * (commit_digit + group.depth_commit * position)
}

fn oracle_r(layout: &OpeningBatchWitnessLayout, row: usize, digit: usize) -> usize {
    layout.total_len() - layout.relation_rows * layout.quotient_depth
        + digit
        + layout.quotient_depth * row
}

fn group(id: usize, live_fold_count: usize) -> OpeningBatchWitnessGroup {
    OpeningBatchWitnessGroup {
        id: SemanticGroupId(id),
        num_claims: 2,
        live_fold_count,
        fold_position_count: 3,
        depth_open: 2,
        depth_commit: 2,
        depth_fold: 2,
        n_a: 2,
        e_setup_col_offset: 0,
    }
}

fn marker(index: usize) -> [i8; D] {
    [((index % 101) + 1) as i8, -(((index % 101) + 1) as i8)]
}

fn check_layout(layout: OpeningBatchWitnessLayout) {
    let mut emitted = vec![0i8; layout.total_len() * D];
    for descriptor in &layout.groups {
        let group_id = descriptor.id;
        let e_len = descriptor.num_claims * descriptor.live_fold_count * descriptor.depth_open;
        let e_source = (0..e_len).map(marker).collect::<Vec<_>>();
        let t_len = e_len * descriptor.n_a;
        let t_source = (0..t_len)
            .map(|index| marker(index + e_len))
            .collect::<Vec<_>>();
        emit_witness_e_planes(
            &mut emitted,
            &layout,
            group_id,
            &e_source,
            descriptor.live_fold_count,
        )
        .unwrap();
        emit_witness_t_planes(
            &mut emitted,
            &layout,
            group_id,
            &t_source,
            descriptor.live_fold_count,
        )
        .unwrap();
        for unit in layout.units_for_group(group_id).unwrap() {
            let z_len =
                descriptor.fold_position_count * descriptor.depth_commit * descriptor.depth_fold;
            let z_source = (0..z_len)
                .map(|index| marker(index + e_len + t_len + unit.machine_chunk.0))
                .collect::<Vec<_>>();
            emit_witness_z_planes(&mut emitted, &layout, unit, &z_source).unwrap();

            let claim = 1;
            let block = unit.global_block_base + unit.blocks - 1;
            let digit = 1;
            let e_index = layout.e_index(unit, claim, block, digit).unwrap();
            assert_eq!(e_index, oracle_e(&layout, unit, claim, block, digit));
            let e_source_index =
                (claim * descriptor.live_fold_count + block) * descriptor.depth_open + digit;
            assert_eq!(
                &emitted[e_index * D..(e_index + 1) * D],
                &e_source[e_source_index]
            );

            let a_row = 1;
            let t_index = layout.t_index(unit, claim, block, a_row, digit).unwrap();
            assert_eq!(t_index, oracle_t(&layout, unit, claim, block, a_row, digit));
            let t_source_index = (claim * descriptor.live_fold_count + block)
                * descriptor.n_a
                * descriptor.depth_open
                + a_row * descriptor.depth_open
                + digit;
            assert_eq!(
                &emitted[t_index * D..(t_index + 1) * D],
                &t_source[t_source_index]
            );

            let z_index = layout.z_index(unit, 2, 1, 1).unwrap();
            assert_eq!(z_index, oracle_z(&layout, unit, 2, 1, 1));
            let z_source_index = (2 * descriptor.depth_commit + 1) * descriptor.depth_fold + 1;
            assert_eq!(
                &emitted[z_index * D..(z_index + 1) * D],
                &z_source[z_source_index]
            );
        }
    }

    let r_source = (0..layout.relation_rows * layout.quotient_depth)
        .map(|index| marker(index + 79))
        .collect::<Vec<_>>();
    emit_witness_r_planes(&mut emitted, &layout, &r_source).unwrap();
    let r_index = layout.r_index(2, 1).unwrap();
    assert_eq!(r_index, oracle_r(&layout, 2, 1));
    assert_eq!(
        &emitted[r_index * D..(r_index + 1) * D],
        &r_source[2 * layout.quotient_depth + 1]
    );

    let group_id = layout.relation_group_order[0];
    let descriptor = layout.group(group_id).unwrap();
    let opening_layout =
        OpeningBlockLayout::new(8, layout.total_len().div_ceil(8)).expect("opening layout");
    let trace = TraceWeightLayout {
        ring_bits: 1,
        col_bits: opening_layout.opening_len().trailing_zeros() as usize,
        live_fold_count: descriptor.num_claims * descriptor.live_fold_count,
        num_digits_open: descriptor.depth_open,
        fold_bits: descriptor.live_fold_count.trailing_zeros() as usize,
        log_basis: 3,
        witness_layout: layout.clone(),
        opening_layout,
        group_id,
    };
    let logical_block = descriptor.live_fold_count + descriptor.live_fold_count - 1;
    let trace_index = trace.opening_digit_col_index(logical_block, 1).unwrap();
    let unit = layout
        .unit_for_block(group_id, descriptor.live_fold_count - 1)
        .unwrap();
    let physical_trace_index = oracle_e(&layout, unit, 1, descriptor.live_fold_count - 1, 1);
    assert_eq!(
        trace_index,
        opening_layout
            .opening_index_for_physical(physical_trace_index)
            .unwrap()
    );

    let mut dense_relation_columns = vec![0u8; opening_layout.opening_len()];
    for index in 0..layout.total_len() {
        let opening_index = opening_layout.opening_index_for_physical(index).unwrap();
        dense_relation_columns[opening_index] = (index % 251) as u8;
    }
    assert_eq!(
        dense_relation_columns[trace_index],
        (physical_trace_index % 251) as u8
    );
    for physical_index in [
        layout
            .e_index(unit, 1, descriptor.live_fold_count - 1, 1)
            .unwrap(),
        layout
            .t_index(unit, 1, descriptor.live_fold_count - 1, 1, 1)
            .unwrap(),
        layout.z_index(unit, 2, 1, 1).unwrap(),
        layout.r_index(2, 1).unwrap(),
    ] {
        let opening_index = opening_layout
            .opening_index_for_physical(physical_index)
            .unwrap();
        assert_eq!(
            dense_relation_columns[opening_index],
            (physical_index % 251) as u8
        );
    }
}

#[test]
fn canonical_witness_addresses_match_emitters_relation_columns_and_trace() {
    for units in [2, 8] {
        check_layout(
            OpeningBatchWitnessLayout::new(
                vec![group(0, 16)],
                vec![SemanticGroupId(0)],
                vec![SemanticGroupId(0)],
                units,
                3,
                2,
            )
            .unwrap(),
        );
    }
    check_layout(
        OpeningBatchWitnessLayout::new(
            vec![group(0, 4), group(1, 6)],
            vec![SemanticGroupId(0), SemanticGroupId(1)],
            vec![SemanticGroupId(1), SemanticGroupId(0)],
            1,
            3,
            2,
        )
        .unwrap(),
    );
}

#[test]
fn opening_block_layout_domain_identities_hold() {
    for live_fold_count in [1, 2, 8] {
        for fold_position_count in [3, 1184, 6660] {
            let layout = OpeningBlockLayout::new(live_fold_count, fold_position_count).unwrap();
            assert_eq!(
                layout.position_stride(),
                fold_position_count.next_power_of_two()
            );
            assert_eq!(
                (live_fold_count * fold_position_count).next_power_of_two(),
                layout.opening_len()
            );
            assert_eq!(
                layout
                    .physical_index(live_fold_count - 1, fold_position_count - 1)
                    .unwrap(),
                live_fold_count * fold_position_count - 1
            );
            assert_eq!(
                layout
                    .opening_index(live_fold_count - 1, fold_position_count - 1)
                    .unwrap(),
                (live_fold_count - 1) * layout.position_stride() + fold_position_count - 1
            );
        }
    }
}

#[test]
fn virtual_opening_factors_and_compact_address_does_not() {
    type F = Prime128OffsetA7F7;

    let layout = OpeningBlockLayout::new(2, 3).unwrap();
    let point = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
    let position_weights = EqPolynomial::evals(&point[..2]).unwrap();
    let block_weights = EqPolynomial::evals(&point[2..]).unwrap();
    let block = 1;
    let position = 0;
    let physical_index = layout.physical_index(block, position).unwrap();
    let opening_index = layout.opening_index(block, position).unwrap();
    let factored = position_weights[position] * block_weights[block];

    assert_eq!(eq_eval_at_index(&point, opening_index), factored);
    assert_ne!(eq_eval_at_index(&point, physical_index), factored);

    let coefficients = [
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
        F::from_u64(19),
        F::from_u64(23),
    ];
    let mut dense = vec![F::zero(); layout.opening_len()];
    for (physical_index, coefficient) in coefficients.iter().copied().enumerate() {
        dense[layout.opening_index_for_physical(physical_index).unwrap()] = coefficient;
    }
    let dense_eval = dense
        .iter()
        .enumerate()
        .fold(F::zero(), |sum, (index, coefficient)| {
            sum + *coefficient * eq_eval_at_index(&point, index)
        });
    let factored_eval = (0..layout.live_fold_count()).fold(F::zero(), |sum, block| {
        let inner = (0..layout.fold_position_count()).fold(F::zero(), |inner, position| {
            inner
                + coefficients[layout.physical_index(block, position).unwrap()]
                    * position_weights[position]
        });
        sum + block_weights[block] * inner
    });
    assert_eq!(dense_eval, factored_eval);
}
