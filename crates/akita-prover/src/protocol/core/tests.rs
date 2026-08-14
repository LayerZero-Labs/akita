use super::*;
use crate::RecursiveWitnessFlat;
use akita_config::{proof_optimized::fp128::OneHot, CommitmentConfig};
use akita_field::{Fp32, FpExt2, TwoNr};
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaScheduleLookupKey, OpeningClaimsLayout, PolynomialGroupLayout};

type F = Fp32<251>;
type E = FpExt2<F, TwoNr>;

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
    let logical_polys = [&logical_w];
    let logical_group = PreparedProverGroup::from_refs(&logical_polys).expect("logical group");

    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    let groups = vec![ExtensionOpeningGroupInput {
        group: &logical_group,
        point: &point,
        ring_dimension: 64,
    }];
    let proved = prove_extension_opening_reduction::<F, E, _, _, _>(
        &crate::compute::CpuBackend::DEFAULT,
        None,
        &groups,
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
fn extension_opening_reduction_shares_challenges_across_groups() {
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
    let polys = [&short_witness, &long_witness];
    let prepared_groups = [
        PreparedProverGroup::from_ref_vec(vec![polys[0]]).expect("short group"),
        PreparedProverGroup::from_ref_vec(vec![polys[1]]).expect("long group"),
    ];
    let groups = vec![
        ExtensionOpeningGroupInput {
            group: &prepared_groups[0],
            point: &short_point,
            ring_dimension: 64,
        },
        ExtensionOpeningGroupInput {
            group: &prepared_groups[1],
            point: &long_point,
            ring_dimension: 64,
        },
    ];
    let mut transcript = AkitaTranscript::<F>::new(b"test/grouped-extension-opening-reduction");

    let proved = prove_extension_opening_reduction::<F, E, _, _, _>(
        &crate::compute::CpuBackend::DEFAULT,
        None,
        &groups,
        &mut transcript,
        "recursive",
    )
    .expect("all groups should reduce through one shared challenge sequence");

    assert_eq!(proved.protocol_points.len(), 2);
    assert_eq!(proved.reduction.final_factors.len(), 2);
    assert_eq!(proved.reduction.proof.final_claims.len(), 2);
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
    let precommitted =
        OneHot::profile_without_precommitted_groups(PolynomialGroupLayout::new(16, 1))
            .expect("independent profile");
    let schedule = OneHot::select_schedule_for_key(&AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![precommitted, precommitted],
    })
    .expect("multi-group schedule")
    .into_schedule();
    let root_params = schedule.root.params.final_group.commitment.clone();
    assert_eq!(root_params.precommitted_groups.len(), 2);
    for precommitted in &root_params.precommitted_groups {
        assert_eq!(precommitted.layout.group, PolynomialGroupLayout::new(16, 1));
    }
}
