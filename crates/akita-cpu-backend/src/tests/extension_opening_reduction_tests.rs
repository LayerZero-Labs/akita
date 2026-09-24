#![allow(missing_docs)]

use akita_types::{
    derive_tensor_extension_opening_claim_from_partials, extension_opening_reduction_claim,
    tensor_column_partials_from_base_evals, tensor_equality_factor_eval_at_point,
    tensor_equality_factor_evals, tensor_packed_witness_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, ExtensionOpeningFactorTerm, ExtensionOpeningReductionFactor,
    ExtensionOpeningTensorPartials,
};
use jolt_field::{Ext2, ExtField, Field, One, Prime128Offset275, Prime64Offset59, Ring, Zero};

type F = Prime128Offset275;

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
        extension_opening_reduction_claim(&packed_witness, &factor_evals).unwrap(),
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

#[test]
fn singleton_factor_claim_matches_multilinear_opening() {
    let witness_evals: Vec<F> = (0..8).map(|i| F::from_u64((11 * i + 4) as u64)).collect();
    let opening_point = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    let factor = ExtensionOpeningReductionFactor::singleton(opening_point.clone()).unwrap();

    let claim = factor.claim_for_witness(&witness_evals).unwrap();
    let expected = akita_sumcheck::multilinear_eval(&witness_evals, &opening_point).unwrap();
    assert_eq!(claim, expected);

    let rho = vec![F::from_u64(2), F::from_u64(9), F::from_u64(6)];
    let factor_evals = factor.evals().unwrap();
    let folded_factor = akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap();
    assert_eq!(folded_factor, factor.evaluate(&rho).unwrap());
}

#[test]
fn row_factor_batches_multiple_opening_points() {
    let witness_evals: Vec<F> = (0..16).map(|i| F::from_u64((5 * i + 8) as u64)).collect();
    let point_a = vec![
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
        F::from_u64(5),
    ];
    let point_b = vec![
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
    ];
    let coeff_a = F::from_u64(19);
    let coeff_b = F::from_u64(23);
    let factor = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(point_a.clone(), coeff_a),
        ExtensionOpeningFactorTerm::new(point_b.clone(), coeff_b),
    ])
    .unwrap();

    assert_eq!(factor.num_vars(), 4);
    assert_eq!(factor.terms().len(), 2);
    let claim = factor.claim_for_witness(&witness_evals).unwrap();
    let expected = coeff_a * akita_sumcheck::multilinear_eval(&witness_evals, &point_a).unwrap()
        + coeff_b * akita_sumcheck::multilinear_eval(&witness_evals, &point_b).unwrap();
    assert_eq!(claim, expected);

    let rho = vec![
        F::from_u64(29),
        F::from_u64(31),
        F::from_u64(37),
        F::from_u64(41),
    ];
    let factor_evals = factor.evals().unwrap();
    assert_eq!(
        akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap(),
        factor.evaluate(&rho).unwrap()
    );
}

#[test]
fn factor_rejects_malformed_shapes() {
    let err = ExtensionOpeningReductionFactor::<F>::from_terms(Vec::new()).unwrap_err();
    assert!(matches!(err, akita_error::AkitaError::InvalidInput(_)));

    let err = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(vec![F::one(), F::zero()], F::one()),
        ExtensionOpeningFactorTerm::new(vec![F::one()], F::one()),
    ])
    .unwrap_err();
    assert!(matches!(err, akita_error::AkitaError::InvalidSize { .. }));
}
