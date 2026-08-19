use super::*;
use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr, Prime128OffsetA7F7};

type F = Fp32<251>;
type E = FpExt2<F, NegOneNr>;

#[test]
fn lifted_relation_claim_matches_base_for_constant_alpha() {
    const D: usize = 4;
    const N_A: usize = 1;
    let tau1 = [
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let alpha = F::from_u64(17);
    let v = [CyclotomicRing::from_coefficients([
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ])];
    let u = [CyclotomicRing::from_coefficients([
        F::from_u64(5),
        F::from_u64(6),
        F::from_u64(7),
        F::from_u64(8),
    ])];

    let base = relation_claim_from_rows::<F, D>(&tau1, alpha, N_A, &v, &u).unwrap();
    let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
    let lifted = relation_claim_from_rows_extension::<F, E, D>(
        &lifted_tau1,
        E::lift_base(alpha),
        N_A,
        &v,
        &u,
    )
    .unwrap();

    assert_eq!(lifted, E::lift_base(base));
}

#[test]
fn relation_claim_at_dims_matches_uniform_single_d() {
    const D: usize = 64;
    let dims = CommitmentRingDims::uniform(D);
    let tau1 = [
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
    ];
    let alpha = F::from_u64(13);
    let mut v_coeffs = [F::zero(); D];
    v_coeffs[..4].copy_from_slice(&[
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ]);
    let mut u_coeffs = [F::zero(); D];
    u_coeffs[..4].copy_from_slice(&[
        F::from_u64(5),
        F::from_u64(6),
        F::from_u64(7),
        F::from_u64(8),
    ]);
    let v = [CyclotomicRing::from_coefficients(v_coeffs)];
    let u = [CyclotomicRing::from_coefficients(u_coeffs)];
    let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
    const N_A: usize = 1;
    let layout =
        RelationRhsLayout::uniform(dims, 1, N_A, 1, CommitmentSliceCount::ONE, 1).expect("layout");
    let at_dims = relation_claim_from_layout_extension::<F, E>(
        &layout,
        &lifted_tau1,
        E::lift_base(alpha),
        &RingVec::from_ring_elems(&v),
        &RingVec::from_ring_elems(&u),
    )
    .unwrap();
    let monolithic = relation_claim_from_rows_extension::<F, E, D>(
        &lifted_tau1,
        E::lift_base(alpha),
        N_A,
        &v,
        &u,
    )
    .unwrap();
    assert_eq!(at_dims, monolithic);
}

#[test]
fn assemble_relation_rhs_matches_generate_rhs_for_uniform_dims() {
    const D: usize = 64;
    let dims = CommitmentRingDims::uniform(D);
    let mut v_coeffs = [F::zero(); D];
    v_coeffs[0] = F::from_u64(1);
    let v = [CyclotomicRing::from_coefficients(v_coeffs)];
    let mut u_coeffs = [F::zero(); D];
    u_coeffs[0] = F::from_u64(2);
    let u = [CyclotomicRing::from_coefficients(u_coeffs)];
    let layout =
        RelationRhsLayout::uniform(dims, 1, 2, 1, CommitmentSliceCount::ONE, 1).expect("layout");
    let typed = generate_relation_rhs::<F, D>(&v, &u, layout.n_d, 1, layout.groups[0].n_a).unwrap();
    let assembled = assemble_relation_rhs::<F>(
        &layout,
        &RingVec::from_ring_elems(&v),
        &RingVec::from_ring_elems(&u),
    )
    .unwrap();
    assert_eq!(
        assembled.coeffs(),
        RingVec::from_ring_elems(&typed).coeffs()
    );
}

#[test]
fn outer_row_families_are_bijective_slice_row_coordinates() {
    for slice_count in CommitmentSliceCount::ALL {
        let physical_rows = 3;
        let layout = RelationRhsLayout::uniform(
            CommitmentRingDims::uniform(64),
            1,
            2,
            physical_rows,
            slice_count,
            1,
        )
        .expect("layout");
        let outer = layout
            .row_families()
            .unwrap()
            .into_iter()
            .filter_map(|family| match family {
                RelationRowFamily::Outer {
                    group_index,
                    slice_index,
                    physical_row,
                    geometry,
                } => Some((
                    group_index,
                    slice_index,
                    physical_row,
                    geometry.physical_coefficient_width(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        let expected = (0..slice_count.get())
            .flat_map(|slice_index| {
                (0..physical_rows).map(move |physical_row| (0, slice_index, physical_row, 64))
            })
            .collect::<Vec<_>>();
        assert_eq!(outer, expected);
    }
}

#[test]
fn mixed_role_dims_relation_rhs_coeff_len_matches_per_segment_widths() {
    let dims = CommitmentRingDims {
        inner: 128,
        outer: 32,
        opening: 64,
    };
    let layout =
        RelationRhsLayout::uniform(dims, 2, 4, 4, CommitmentSliceCount::ONE, 1).expect("layout");
    let coeff_len = relation_rhs_coeff_len(&layout).expect("coeff len");
    let expected = 128 + 2 * 64 + 3 * 32 + 32 + 4 * 128;
    assert_eq!(coeff_len, expected);
    assert_eq!(relation_rhs_row_count(&layout), 1 + 2 + 3 + 1 + 4);
}

#[test]
fn group_local_a_b_dims_share_d_in_rhs_and_claim() {
    type G = Prime128OffsetA7F7;
    let final_dims = CommitmentRingDims {
        inner: 256,
        outer: 128,
        opening: 64,
    };
    let precommitted_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let layout = RelationRhsLayout {
        d_ring_dimension: 64,
        n_d: 1,
        groups: vec![
            RelationGroupRows {
                group_index: 0,
                role_dims: final_dims,
                opening_geometry: RelationRowGeometry::native(final_dims.d_a()).unwrap(),
                opening_method: OpeningMethod::EvaluationTrace,
                n_a: 1,
                physical_b_rows: 1,
                outer_slice_count: CommitmentSliceCount::ONE,
            },
            RelationGroupRows {
                group_index: 1,
                role_dims: precommitted_dims,
                opening_geometry: RelationRowGeometry::native(precommitted_dims.d_a()).unwrap(),
                opening_method: OpeningMethod::EvaluationTrace,
                n_a: 2,
                physical_b_rows: 2,
                outer_slice_count: CommitmentSliceCount::ONE,
            },
        ],
        compression: None,
    };
    assert_eq!(
        relation_rhs_coeff_len(&layout).expect("mixed group rhs length"),
        256 + 256 + 128 + 128 + 2 * 128 + 2 * 64 + 64
    );
    assert_eq!(
        layout
            .row_geometries()
            .expect("mixed quotient row dims")
            .into_iter()
            .map(RelationRowGeometry::physical_coefficient_width)
            .collect::<Vec<_>>(),
        vec![256, 256, 128, 128, 128, 128, 64, 64, 64]
    );

    let mut commitment_coeffs = vec![G::zero(); 128 + 2 * 64];
    commitment_coeffs[0] = G::from_u64(2);
    commitment_coeffs[128] = G::from_u64(3);
    commitment_coeffs[128 + 64] = G::from_u64(4);
    let commitment_rows = RingVec::from_coeffs(commitment_coeffs);
    let mut v_coeffs = vec![G::zero(); 64];
    v_coeffs[0] = G::from_u64(5);
    let v = RingVec::from_coeffs(v_coeffs);

    let rhs = assemble_relation_rhs(&layout, &v, &commitment_rows).expect("mixed group rhs");
    assert_eq!(
        rhs.coeff_len(),
        relation_rhs_coeff_len(&layout).expect("mixed group rhs length")
    );

    let tau1 = [
        G::from_u64(7),
        G::from_u64(11),
        G::from_u64(13),
        G::from_u64(19),
    ];
    let alpha = G::from_u64(17);
    let claim =
        relation_claim_from_layout_extension::<G, G>(&layout, &tau1, alpha, &v, &commitment_rows)
            .expect("mixed group claim");
    let expected = eq_eval_at_index(&tau1, 2) * G::from_u64(2)
        + eq_eval_at_index(&tau1, 6) * G::from_u64(3)
        + eq_eval_at_index(&tau1, 7) * G::from_u64(4)
        + eq_eval_at_index(&tau1, 8) * G::from_u64(5);
    assert_eq!(claim, expected);
}

#[test]
fn rows_allow_group_a_larger_than_final_group_a() {
    let layout = RelationRhsLayout {
        d_ring_dimension: 32,
        n_d: 1,
        groups: vec![
            RelationGroupRows {
                group_index: 0,
                role_dims: CommitmentRingDims {
                    inner: 64,
                    outer: 32,
                    opening: 32,
                },
                opening_geometry: RelationRowGeometry::native(64).unwrap(),
                opening_method: OpeningMethod::EvaluationTrace,
                n_a: 1,
                physical_b_rows: 1,
                outer_slice_count: CommitmentSliceCount::ONE,
            },
            RelationGroupRows {
                group_index: 1,
                role_dims: CommitmentRingDims {
                    inner: 128,
                    outer: 32,
                    opening: 32,
                },
                opening_geometry: RelationRowGeometry::native(128).unwrap(),
                opening_method: OpeningMethod::EvaluationTrace,
                n_a: 1,
                physical_b_rows: 1,
                outer_slice_count: CommitmentSliceCount::ONE,
            },
        ],
        compression: None,
    };
    assert_eq!(
        layout
            .row_geometries()
            .expect("native quotient row dims")
            .into_iter()
            .map(RelationRowGeometry::physical_coefficient_width)
            .collect::<Vec<_>>(),
        vec![64, 64, 32, 128, 128, 32, 32]
    );
}

#[test]
fn relation_row_weight_uses_requested_row() {
    // total_row_count = 4 → 2 row-index vars; eq table length 4.
    let tau1 = [F::from_u64(2), F::from_u64(3)];
    let weight = relation_row_weight(3, &tau1).unwrap();
    assert_eq!(weight, eq_eval_at_index(&tau1, 3));
    assert_ne!(weight, eq_eval_at_index(&tau1, 0));
}

#[test]
fn relation_row_weight_rejects_out_of_domain_index() {
    let tau1 = [F::from_u64(2), F::from_u64(3)];
    assert!(relation_row_weight(4, &tau1).is_err());
}

#[test]
fn fused_relation_claim_matches_full_logical_row_evaluation() {
    const D: usize = 64;
    let dims = CommitmentRingDims::uniform(D);
    let tau1 = [
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
    ];
    let alpha = F::from_u64(13);
    let mut v_coeffs = [F::zero(); D];
    v_coeffs[..4].copy_from_slice(&[
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ]);
    let mut u_coeffs = [F::zero(); D];
    u_coeffs[..4].copy_from_slice(&[
        F::from_u64(5),
        F::from_u64(6),
        F::from_u64(7),
        F::from_u64(8),
    ]);
    let v = [CyclotomicRing::from_coefficients(v_coeffs)];
    let u = [CyclotomicRing::from_coefficients(u_coeffs)];
    let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
    const N_A: usize = 1;
    let layout =
        RelationRhsLayout::uniform(dims, 1, N_A, 1, CommitmentSliceCount::ONE, 1).expect("layout");
    let evaluation_trace_row = relation_rhs_row_count(&layout);
    let trace_target = E::from_u64(19);
    let quotient_claim = relation_claim_from_layout_extension::<F, E>(
        &layout,
        &lifted_tau1,
        E::lift_base(alpha),
        &RingVec::from_ring_elems(&v),
        &RingVec::from_ring_elems(&u),
    )
    .unwrap();
    let weight = relation_row_weight(evaluation_trace_row, &lifted_tau1).unwrap();
    let fused = quotient_claim + weight * trace_target;

    let alpha_pows = scalar_powers(E::lift_base(alpha), D);
    let padded_domain = 1usize << lifted_tau1.len();
    let mut y_alpha = vec![E::zero(); padded_domain];
    let mut row_idx = 1usize + N_A;
    for ring in &u {
        y_alpha[row_idx] = eval_ring_at_pows_fast(ring, &alpha_pows);
        row_idx += 1;
    }
    for ring in &v {
        y_alpha[row_idx] = eval_ring_at_pows_fast(ring, &alpha_pows);
        row_idx += 1;
    }
    y_alpha[evaluation_trace_row] = trace_target;

    let mut independent = E::zero();
    for (row, value) in y_alpha.iter().enumerate() {
        independent += eq_eval_at_index(&lifted_tau1, row) * *value;
    }
    assert_eq!(fused, independent);
}
