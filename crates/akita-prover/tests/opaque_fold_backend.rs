#![allow(missing_docs)]

use akita_prover::backend::{AcceptedFoldMetadata, Stage1FinalClaims};
use jolt_field::{Prime128Offset275 as F, Ring};

#[test]
fn public_metadata_and_stage1_claims_are_externally_constructible() {
    let metadata = AcceptedFoldMetadata::try_new(64, 512, 8).unwrap();
    assert_eq!(metadata.ring_dimension(), 64);
    assert_eq!(metadata.response_coordinate_count(), 512);
    assert_eq!(metadata.num_chunks(), 8);
    assert!(AcceptedFoldMetadata::try_new(63, 512, 8).is_err());
    assert!(AcceptedFoldMetadata::try_new(64, 511, 8).is_err());
    assert!(AcceptedFoldMetadata::try_new(64, 512, 3).is_err());

    let point = vec![F::from_u64(3), F::from_u64(5)];
    let claims = Stage1FinalClaims::new(point.clone(), F::from_u64(7));
    assert_eq!(claims.point(), point);
    assert_eq!(claims.final_claim(), F::from_u64(7));
}
