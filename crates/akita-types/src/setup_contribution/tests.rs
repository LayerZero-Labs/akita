use super::*;
use crate::{
    CommitmentRingDims, CommittedGroupParams, OpeningClaimsLayout, PreparedRelationAddress,
    SetupContributionPlan, WitnessLayout,
};
use jolt_field::{One, Prime128OffsetA7F7};

mod prepare;

type F = Prime128OffsetA7F7;
const TEST_D: usize = 64;

fn test_scalar(value: u128) -> F {
    F::from_u128_checked(value).expect("test scalar must be canonical")
}
