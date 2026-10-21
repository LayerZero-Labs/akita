//! The stage-1 leaf and stage-2 oracle checks over jolt-zk's masked domains
//! (`specs/whiteout-akita.md` §4.4): on honest decoded values the residual
//! rows they push are satisfied, a shifted claim breaks one, and the shape
//! domain counts the lowering's products and auxiliary slots.

use super::{check_range_leaf, check_stage2_oracle, RangeLeaf, Stage2OracleWeights};
use akita_params::DigitRangePlan;
use jolt_algebra::{One, Prime128OffsetA7F7, Ring};
use jolt_zk::masking::{
    AffineRow, OperandShapeDomain, PadIndex, PadLayout, ResidualRelation, RowsDomain, ValuesDomain,
};

type F = Prime128OffsetA7F7;

fn pads(count: usize) -> Vec<F> {
    (0..count)
        .map(|index| F::from_u64(0x9e37_79b9_7f4a_7c15 ^ (index as u64 * 0x1000_0001)))
        .collect()
}

fn masked(value: F, pads: &[F], pad: usize) -> AffineRow<F> {
    AffineRow::masked(value + pads[pad], PadIndex(pad))
}

/// Runs the stage-1 leaf check over decoded `(range_image, claim)` in the
/// prover's value domain, then over the masked messages in the rows domain
/// with the auxiliary values the prover computed. Returns whether the
/// prover's own check passed and whether every row holds.
fn leaf_rows(leaf: &RangeLeaf<F>, range_image: F, claim: F) -> (bool, bool) {
    let mut values = ValuesDomain::default();
    let prover_check = check_range_leaf(&mut values, leaf, range_image, claim).is_ok();
    let layout = PadLayout::new(2, values.auxiliary.len()).unwrap();
    let pads = pads(layout.num_pads);
    let masked_auxiliary = values
        .auxiliary
        .iter()
        .zip(&pads[2..])
        .map(|(value, pad)| *value + *pad)
        .collect::<Vec<_>>();
    let mut relation = ResidualRelation::new(layout, 0);
    let mut rows = RowsDomain::new(&mut relation, &masked_auxiliary);
    check_range_leaf(
        &mut rows,
        leaf,
        masked(range_image, &pads, 0),
        masked(claim, &pads, 1),
    )
    .unwrap();
    rows.finish().unwrap();
    assert_eq!((relation.affine.len(), relation.products.len()), (0, 2));
    (
        prover_check,
        relation.first_violation(&pads, &[]).unwrap().is_none(),
    )
}

#[test]
fn range_leaf_rows_hold_on_honest_values_and_break_on_a_shifted_claim() {
    let plan = DigitRangePlan::new(8).unwrap();
    let coefficients = plan
        .batch_leaf_polynomials(&[F::one()], &plan.leaf_coeffs::<F>())
        .unwrap();
    let leaf = RangeLeaf::new(plan, &coefficients).unwrap();
    let range_image = F::from_u64(0x1234_5678_9abc_def1);
    let claim = plan.evaluate_leaf_polynomial(&coefficients, range_image);

    assert_eq!(leaf_rows(&leaf, range_image, claim), (true, true));
    assert_eq!(
        leaf_rows(&leaf, range_image, claim + F::one()),
        (false, false)
    );
    assert_eq!(
        leaf_rows(&leaf, range_image + F::one(), claim),
        (false, false)
    );

    let mut shape = OperandShapeDomain::new(2);
    check_range_leaf(&mut shape, &leaf, Some(PadIndex(0)), Some(PadIndex(1))).unwrap();
    let layout = shape.layout().unwrap();
    assert_eq!(layout.auxiliary, 2..3);
    assert_eq!(layout.operand_pads, [PadIndex(0), PadIndex(2)]);
}

fn oracle_rows(weights: &Stage2OracleWeights<F>, witness_eval: F, claim: F) -> bool {
    let layout = PadLayout::new(2, 0).unwrap();
    let pads = pads(layout.num_pads);
    let mut relation = ResidualRelation::new(layout, 0);
    let mut rows = RowsDomain::new(&mut relation, &[]);
    check_stage2_oracle(
        &mut rows,
        weights,
        masked(witness_eval, &pads, 0),
        masked(claim, &pads, 1),
    )
    .unwrap();
    rows.finish().unwrap();
    assert_eq!((relation.affine.len(), relation.products.len()), (0, 1));
    relation.first_violation(&pads, &[]).unwrap().is_none()
}

#[test]
fn stage2_oracle_rows_hold_on_honest_values_and_break_on_a_shifted_claim() {
    let weights = Stage2OracleWeights {
        binary: F::from_u64(0x0bad_cafe_f00d),
        linear: F::from_u64(0x00de_ad00_beef),
    };
    let witness_eval = F::from_u64(0x7777_1111_3333);
    // The oracle term by term: B·w(w+1) collects the virtual and binary
    // terms, R·w the linear ones.
    let claim =
        weights.binary * witness_eval * (witness_eval + F::one()) + weights.linear * witness_eval;
    assert!(oracle_rows(&weights, witness_eval, claim));
    assert!(!oracle_rows(&weights, witness_eval, claim + F::one()));

    let mut shape = OperandShapeDomain::new(2);
    check_stage2_oracle(&mut shape, &weights, Some(PadIndex(0)), Some(PadIndex(1))).unwrap();
    let layout = shape.layout().unwrap();
    assert!(layout.auxiliary.is_empty());
    assert_eq!(layout.operand_pads, [PadIndex(0)]);
}
