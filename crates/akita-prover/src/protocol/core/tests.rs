use super::*;
use crate::RecursiveWitnessFlat;
use akita_config::{proof_optimized::fp128::D64OneHot, CommitmentConfig};
use akita_field::{Fp32, FpExt2, NegOneNr};
use akita_transcript::AkitaTranscript;
use akita_types::{OpeningClaimsLayout, PolynomialGroupLayout};

type F = Fp32<251>;
type E = FpExt2<F, NegOneNr>;

#[test]
fn recursive_extension_opening_reduction_pads_to_opening_cube() {
    let mut digits = vec![0; 3 * 64];
    digits[..6].copy_from_slice(&[1, -1, 2, 0, 3, -2]);
    let logical_w = RecursiveWitnessFlat::from_i8_digits(digits);
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
        E::new(F::from_u64(17), F::from_u64(19)),
        E::new(F::from_u64(23), F::from_u64(29)),
        E::new(F::from_u64(31), F::from_u64(37)),
        E::new(F::from_u64(41), F::from_u64(43)),
        E::new(F::from_u64(47), F::from_u64(53)),
    ];
    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    let groups = vec![ExtensionOpeningGroupInput {
        polynomials: vec![&logical_w],
        point: &point,
        ring_dimension: 64,
    }];
    let proved = prove_extension_opening_reduction::<F, E, _, RecursiveWitnessFlat, _>(
        &crate::compute::CpuBackend,
        None,
        &groups,
        true,
        &mut transcript,
        "recursive",
    )
    .expect("padded logical witnesses should reduce over the opening cube");

    assert_eq!(
        proved.reduction.proof.partials.len(),
        <E as ExtField<F>>::EXT_DEGREE
    );
    assert_eq!(proved.reduction.proof.num_rounds(), point.len() - 1);
}

#[test]
fn extension_opening_reduction_uses_one_sumcheck_for_all_groups() {
    let short_witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 64]);
    let mut long_digits = vec![0; 3 * 64];
    long_digits[..6].copy_from_slice(&[1, -1, 2, 0, 3, -2]);
    let long_witness = RecursiveWitnessFlat::from_i8_digits(long_digits);
    let short_point = (0..6)
        .map(|index| E::new(F::from_u64(index + 2), F::from_u64(index + 11)))
        .collect::<Vec<_>>();
    let long_point = (0..8)
        .map(|index| E::new(F::from_u64(index + 3), F::from_u64(index + 17)))
        .collect::<Vec<_>>();
    let groups = vec![
        ExtensionOpeningGroupInput {
            polynomials: vec![&short_witness],
            point: &short_point,
            ring_dimension: 64,
        },
        ExtensionOpeningGroupInput {
            polynomials: vec![&long_witness],
            point: &long_point,
            ring_dimension: 64,
        },
    ];
    let mut transcript = AkitaTranscript::<F>::new(b"test/grouped-extension-opening-reduction");

    let proved = prove_extension_opening_reduction::<F, E, _, RecursiveWitnessFlat, _>(
        &crate::compute::CpuBackend,
        None,
        &groups,
        true,
        &mut transcript,
        "recursive",
    )
    .expect("all groups should reduce through one sumcheck");

    assert_eq!(proved.protocol_points.len(), 2);
    assert_eq!(proved.reduction.final_factors.len(), 2);
    assert_eq!(proved.row_coefficients, vec![E::one(); 2]);
    assert_eq!(proved.reduction.proof.num_rounds(), long_point.len() - 1);
}

#[test]
fn proof_schedule_from_layout_includes_entire_batch() {
    let batch = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(32, 2),
    ])
    .expect("multi-group shape");
    assert_eq!(batch.num_groups(), 3);
    let schedule = D64OneHot::get_params_for_prove(&batch).expect("multi-group schedule");
    let root_params = schedule.root.params.final_group.commitment.clone();
    assert_eq!(root_params.precommitted_groups.len(), 2);
    for precommitted in &root_params.precommitted_groups {
        assert_eq!(precommitted.layout.group, PolynomialGroupLayout::new(16, 1));
    }
}
