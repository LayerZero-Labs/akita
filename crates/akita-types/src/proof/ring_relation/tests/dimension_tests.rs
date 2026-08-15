use super::*;

#[test]
fn uniform_dimension_check_accepts_coefficient_packing() {
    const PACK_D: usize = 64;
    let config =
        SparseChallengeConfig::production_for_ring_dim(PACK_D).expect("packing challenge config");
    let mut params = CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q32Offset99,
        PACK_D,
        2,
        1,
        1,
        1,
        config,
    )
    .with_decomp(4, 8, 1, 2, 2)
    .expect("uniform packing params");
    params.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: PACK_D,
    };
    params.fold_challenge_config = config;
    let opening_batch = OpeningClaimsLayout::new(8, 1).expect("opening batch");
    let relation_geometry = crate::RelationWitnessGeometry::for_level(&params, &opening_batch, 1)
        .expect("packing relation geometry");
    let rhs_len = relation_rhs_coeff_len(relation_geometry.rhs_layout()).expect("rhs len");
    let opening = RingRelationGroupOpening::subring_coefficient_packing(
        SubringCoefficientPackingGeometry::try_new(1, PACK_D, PACK_D).expect("packing geometry"),
        packing_challenges(&params, 1),
    )
    .expect("packing opening");
    let instance = RingRelationInstance::<F>::new(
        vec![opening],
        1,
        opening_batch,
        vec![F::one()],
        RingVec::from_ring_elems::<PACK_D>(&[CyclotomicRing::one()]),
        RingVec::from_coeffs(vec![F::zero(); rhs_len]),
        RingVec::from_ring_elems::<PACK_D>(&[]),
        CommitmentRingDims::uniform(PACK_D),
    )
    .expect("uniform packing instance");

    instance
        .ensure_ring_dim::<PACK_D>()
        .expect("packing has no EvaluationTrace multiplier point to validate");

    let wrong_opening = RingRelationGroupOpening::subring_coefficient_packing(
        SubringCoefficientPackingGeometry::try_new(1, 128, PACK_D)
            .expect("wrong ambient packing geometry"),
        packing_challenges(&params, 1),
    )
    .expect("schedule-independent packing opening");
    let wrong_instance = RingRelationInstance::<F>::new(
        vec![wrong_opening],
        1,
        OpeningClaimsLayout::new(8, 1).expect("opening batch"),
        vec![F::one()],
        RingVec::from_ring_elems::<PACK_D>(&[CyclotomicRing::one()]),
        RingVec::from_coeffs(vec![F::zero(); rhs_len]),
        RingVec::from_ring_elems::<PACK_D>(&[]),
        CommitmentRingDims::uniform(PACK_D),
    )
    .expect("carrier construction is schedule-independent");
    assert!(wrong_instance.ensure_ring_dim::<PACK_D>().is_err());
}
