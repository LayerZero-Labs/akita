#![allow(missing_docs)]

use akita_types::{
    derive_tensor_extension_opening_claim_from_partials, tensor_column_partials_from_base_evals,
    tensor_equality_factor_eval_at_point, tensor_equality_factor_evals,
    tensor_packed_witness_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, ExtensionOpeningTensorPartials,
};
use jolt_field::{Ext2, ExtField, Field, Prime64Offset59, Ring, Zero};

fn lifted_multilinear_eval<B, E>(evals: &[B], point: &[E]) -> E
where
    B: Field,
    E: ExtField<B>,
{
    let mut layer = evals.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    for &r in point {
        let one_minus_r = E::one() - r;
        let next_len = layer.len() / 2;
        for idx in 0..next_len {
            layer[idx] = layer[2 * idx] * one_minus_r + layer[2 * idx + 1] * r;
        }
        layer.truncate(next_len);
    }
    layer[0]
}

#[test]
fn tensor_partials_recompose_logical_extension_opening() {
    type B = Prime64Offset59;
    type E = Ext2<B>;

    let num_vars = 4;
    let base_evals = (0..(1usize << num_vars))
        .map(|idx| B::from_u64((17 * idx as u64 + 9) % 127))
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|idx| {
            E::from_base_slice(&[B::from_u64(idx as u64 + 3), B::from_u64(5 * idx as u64 + 2)])
        })
        .collect::<Vec<_>>();

    let column_partials =
        tensor_column_partials_from_base_evals::<B, E>(num_vars, &base_evals, &point).unwrap();
    let row_partials = tensor_row_partials_from_columns::<B, E>(&column_partials).unwrap();
    let partials = ExtensionOpeningTensorPartials {
        column_partials,
        row_partials,
    };
    assert_eq!(partials.column_partials.len(), <E as ExtField<B>>::DEGREE);
    assert_eq!(partials.row_partials.len(), <E as ExtField<B>>::DEGREE);

    let logical_claim = derive_tensor_extension_opening_claim_from_partials::<B, E>(
        &point,
        &partials.column_partials,
    )
    .unwrap();
    assert_eq!(logical_claim, lifted_multilinear_eval(&base_evals, &point));
}

#[test]
fn tensor_row_reduction_matches_dense_sumcheck_claim() {
    type B = Prime64Offset59;
    type E = Ext2<B>;

    let num_vars = 4;
    let base_evals = (0..(1usize << num_vars))
        .map(|idx| B::from_u64((23 * idx as u64 + 11) % 131))
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|idx| {
            E::from_base_slice(&[
                B::from_u64(3 * idx as u64 + 4),
                B::from_u64(7 * idx as u64 + 1),
            ])
        })
        .collect::<Vec<_>>();
    let eta = vec![E::from_base_slice(&[B::from_u64(19), B::from_u64(29)])];

    let packed_witness = tensor_packed_witness_evals::<B, E>(num_vars, &base_evals).unwrap();
    let column_partials =
        tensor_column_partials_from_base_evals::<B, E>(num_vars, &base_evals, &point).unwrap();
    let row_partials = tensor_row_partials_from_columns::<B, E>(&column_partials).unwrap();
    let partials = ExtensionOpeningTensorPartials {
        column_partials,
        row_partials,
    };
    let row_claim = tensor_reduction_claim_from_rows::<B, E>(&partials.row_partials, &eta).unwrap();
    let factor_evals = tensor_equality_factor_evals::<B, E>(&point[1..], &eta).unwrap();

    assert_eq!(packed_witness.len(), factor_evals.len());
    assert_eq!(
        packed_witness
            .iter()
            .zip(&factor_evals)
            .fold(E::zero(), |acc, (&w, &a)| acc + w * a),
        row_claim
    );

    let rho = vec![
        E::from_base_slice(&[B::from_u64(31), B::from_u64(37)]),
        E::from_base_slice(&[B::from_u64(41), B::from_u64(43)]),
        E::from_base_slice(&[B::from_u64(47), B::from_u64(53)]),
    ];
    assert_eq!(
        akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap(),
        tensor_equality_factor_eval_at_point::<B, E>(&point[1..], &eta, &rho).unwrap()
    );
}
