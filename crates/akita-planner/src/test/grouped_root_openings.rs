//! Five distinct dense producers must retain their root packing domain.
use akita_config::{policy_of, proof_optimized::fp128::DenseBounded, CommitmentConfig};
use akita_params::{
    CommitmentRingDims, GroupCommitPhaseParams, PolynomialGroupLayout, ScheduleLookupKey,
};

#[test]
fn five_distinct_dense_producers_keep_both_shared_opening_dimensions() {
    let policy = policy_of::<DenseBounded>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let contract = DenseBounded::committed_source_contract().unwrap();
    // Advice, bytecode, image and increment shapes from a committed-program root.
    let precommitteds = [21, 14, 29, 18, 30]
        .into_iter()
        .map(|arity| {
            let group = PolynomialGroupLayout::singleton(arity);
            let planned = crate::find_schedule(
                &ScheduleLookupKey::single(group),
                contract,
                &[],
                &policy,
                DenseBounded::ring_challenge_config,
            )
            .unwrap();
            GroupCommitPhaseParams::try_from_params(group, &planned.schedule.root.params).unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(precommitteds[2].inner.matrix.ring_dimension(), 1024);
    let key = ScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(34, 43),
        precommitteds,
    };
    for (opening, expected) in [(64, 1280), (128, 324)] {
        let dimensions = CommitmentRingDims {
            inner: 512,
            outer: 128,
            opening,
        };
        let products = super::packing_precommit_opening_products(
            &policy,
            dimensions,
            &key,
            &[contract; 5],
        )
        .unwrap();
        assert_eq!(products.len(), expected, "opening dimension {opening}");
    }
}
