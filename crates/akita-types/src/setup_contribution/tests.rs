use super::*;
use crate::{PreparedRelationAddress, SetupContributionPlan};
use akita_params::{CommitmentRingDims, CommittedGroupParams, OpeningClaimsLayout, WitnessLayout};
use jolt_field::{One, Prime128OffsetA7F7};

mod prepare;

type F = Prime128OffsetA7F7;
const TEST_D: usize = 64;

fn test_scalar(value: u128) -> F {
    F::from_u128_checked(value).expect("test scalar must be canonical")
}

#[test]
fn checked_slice_rejects_out_of_bounds_arguments() {
    assert!(matches!(
        checked_slice(&[1, 2], 1, 2, "test slice"),
        Err(AkitaError::InvalidInput(_))
    ));
}
