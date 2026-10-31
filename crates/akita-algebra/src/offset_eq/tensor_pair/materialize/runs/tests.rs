use super::*;
use crate::offset_eq::{eq_eval_at_index, EqPairTensorAxis, EqPairTensorFamily};
use jolt_field::{Fp64, One, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

type F = Fp64<4294967197>;

#[test]
fn runs_match_scalar_equality_across_carries_and_domain_end() {
    check_runs::<F>();
}

#[test]
fn fp128_runs_match_scalar_equality_across_carries_and_domain_end() {
    check_runs::<jolt_field::solinas::Prime128OffsetA7F7>();
}

fn check_runs<F: Field>() {
    let mut rng = StdRng::seed_from_u64(0x5445);
    for bits in [8, 10, 12, 20] {
        let random = (0..bits)
            .map(|_| F::from_u64(rng.gen_range(0..10000)))
            .collect::<Vec<_>>();
        let boolean = (0..bits)
            .map(|i| if i % 2 == 0 { F::zero() } else { F::one() })
            .collect::<Vec<_>>();
        for point in [random, boolean, vec![F::zero(); bits], vec![F::one(); bits]] {
            let eq = OffsetEqWindow::new(&point).unwrap();
            for stride in [1, 2] {
                for len in [16, 17, 31, 32, 33, 64] {
                    for start in [0, 1, eq.low_mask, (1 << bits) - 3] {
                        let family = EqPairTensorFamily::new(
                            0,
                            start,
                            F::from_u64(7),
                            vec![EqPairTensorAxis::unit(len, 1, stride)],
                        )
                        .unwrap();
                        let residual = vec![
                            (0, F::one()),
                            (3, F::from_u64(9)),
                            (eq.low_mask + 1, -F::one()),
                        ];
                        let view = DenseLeftTensorView {
                            family: &family,
                            destination_axis: 0,
                            residual: ResidualTerms::Table(residual.clone()),
                        };
                        let expected = (0..len)
                            .map(|i| {
                                F::from_u64(11)
                                    + family.scalar
                                        * residual
                                            .iter()
                                            .map(|&(offset, weight)| {
                                                weight
                                                    * eq_eval_at_index(
                                                        &point,
                                                        start + stride * i + offset,
                                                    )
                                            })
                                            .sum::<F>()
                            })
                            .collect::<Vec<_>>();
                        let current = (0..len)
                            .map(|i| {
                                F::from_u64(11)
                                    + view.evaluate_residual(&eq, start + stride * i).unwrap()
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(current, expected);
                        let mut actual = vec![F::from_u64(11); len];
                        assert!(try_accumulate(&eq, &[view], false, &mut actual).unwrap());
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn multiple_families_accumulate_and_mixed_admission_leaves_output_untouched() {
    let point = [F::from_u64(3); 12];
    let eq = OffsetEqWindow::new(&point).unwrap();
    let family =
        EqPairTensorFamily::new(0, 7, F::one(), vec![EqPairTensorAxis::unit(33, 1, 2)]).unwrap();
    let make_view = || DenseLeftTensorView {
        family: &family,
        destination_axis: 0,
        residual: ResidualTerms::Table(vec![(0, F::from_u64(4)), (15, F::one())]),
    };
    let first = make_view();
    let expected = (0..33)
        .map(|i| F::from_u64(2) * first.evaluate_residual(&eq, 7 + 2 * i).unwrap())
        .collect::<Vec<_>>();
    let mut output = vec![F::zero(); 33];
    assert!(try_accumulate(&eq, &[first, make_view()], false, &mut output).unwrap());
    assert_eq!(output, expected);
    let invalid =
        EqPairTensorFamily::new(0, 7, F::one(), vec![EqPairTensorAxis::unit(33, 1, 3)]).unwrap();
    let mut second = make_view();
    second.family = &invalid;
    let original = output.clone();
    assert!(!try_accumulate(&eq, &[make_view(), second], false, &mut output).unwrap());
    assert_eq!(output, original);
}

#[cfg(feature = "tensor-no-carry")]
#[test]
fn banked_no_carry_specialization_keeps_its_dispatch() {
    let eq = OffsetEqWindow::new(&[F::from_u64(3); 12]).unwrap();
    let family =
        EqPairTensorFamily::new(0, 0, F::one(), vec![EqPairTensorAxis::unit(16, 1, 1)]).unwrap();
    let view = DenseLeftTensorView {
        family: &family,
        destination_axis: 0,
        residual: ResidualTerms::Table(vec![(0, F::one()), (64, F::from_u64(5))]),
    };
    let separated = [super::super::separate::SeparatedResidual::prepare(&view, &eq).unwrap()];
    assert!(separated[0].is_some());
    let mut output = vec![F::zero(); 16];
    assert!(!try_accumulate(&eq, &[view], separated[0].is_some(), &mut output).unwrap());
    assert_eq!(output, vec![F::zero(); 16]);
}

#[test]
fn declined_or_overflowing_runs_do_not_write_the_destination() {
    let eq = OffsetEqWindow::new(&[F::from_u64(3); 10]).unwrap();
    for (start, stride, len, is_error) in [
        (0, 3, 32, false),
        (0, 1, 15, false),
        (usize::MAX - 2, 1, 32, true),
    ] {
        let family = EqPairTensorFamily::new(
            0,
            start,
            F::one(),
            vec![EqPairTensorAxis::unit(len, 1, stride)],
        )
        .unwrap();
        let view = DenseLeftTensorView {
            family: &family,
            destination_axis: 0,
            residual: ResidualTerms::Table(vec![(0, F::one())]),
        };
        let mut output = vec![F::from_u64(17); len];
        let result = try_accumulate(&eq, &[view], false, &mut output);
        if is_error {
            assert!(result.is_err());
        } else {
            assert!(!result.unwrap());
        }
        assert_eq!(output, vec![F::from_u64(17); len]);
    }
}
