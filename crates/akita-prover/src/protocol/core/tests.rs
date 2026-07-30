use super::*;
use crate::RecursiveWitnessFlat;
use akita_config::{
    proof_optimized::fp128::D64OneHot, CommitmentConfig, PrecommittedCommitmentConfig,
};
use akita_field::{Fp32, FpExt2, NegOneNr};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupProfile, OpeningClaims, OpeningClaimsLayout,
    PolynomialGroupClaims, PolynomialGroupLayout,
};

type F = Fp32<251>;
type E = FpExt2<F, NegOneNr>;

#[test]
fn recursive_extension_opening_reduction_pads_to_opening_cube() {
    let logical_w = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2, 0, 3, -2]);
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
    ];
    let logical_polys = [&logical_w];

    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    let opening_batch = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        vec![E::zero()],
        (),
    )
    .expect("group claims")])
    .expect("opening batch");
    let proved = prove_extension_opening_reduction::<F, E, _, RecursiveWitnessFlat, _, 2>(
        &crate::compute::CpuBackend,
        None,
        &logical_polys,
        &opening_batch,
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
fn proof_schedule_from_layout_includes_entire_batch() {
    let batch = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(32, 2),
    ])
    .expect("multi-group shape");
    assert_eq!(batch.num_groups(), 3);
    let pre_layout = OpeningClaimsLayout::new(16, 1).expect("precommit layout");
    let pre_params =
        PrecommittedCommitmentConfig::<D64OneHot>::get_params_for_batched_commitment(&pre_layout)
            .expect("precommit params");
    let precommitted =
        CommittedGroupProfile::from_params(PolynomialGroupLayout::new(16, 1), &pre_params);
    let schedule = D64OneHot::runtime_schedule(AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        final_source: D64OneHot::group_source(),
        precommitteds: vec![precommitted, precommitted],
        precommitted_sources: vec![D64OneHot::group_source(), D64OneHot::group_source()],
    })
    .expect("multi-group schedule");
    let root_params = schedule.root.params.final_group.commitment.clone();
    assert_eq!(root_params.precommitted_groups.len(), 2);
    for precommitted in &root_params.precommitted_groups {
        assert_eq!(precommitted.layout.group, PolynomialGroupLayout::new(16, 1));
    }
}
