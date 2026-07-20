use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp32;
use akita_types::{
    OpeningClaimsLayout, RelationMatrixRowLayout, SetupContributionGroupInputs,
    SetupContributionPlan, SisModulusProfileId,
};

type F = Fp32<251>;
const D: usize = 32;

fn fold_challenge_config() -> SparseChallengeConfig {
    SparseChallengeConfig::pm1_only(1)
}

#[test]
fn ring_switch_prepare_rejects_invalid_log_basis() {
    let err = validate_log_basis(0).expect_err("invalid log_basis should be rejected");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn ring_switch_prepare_rejects_zero_num_live_blocks() {
    let lp = LevelParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    );
    let opening_batch = OpeningClaimsLayout::new(0, 1).expect("opening batch");
    let valid_lp = LevelParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    )
    .with_decomp(1, 1, 1, 1)
    .unwrap();
    let witness_layout = WitnessLayout::new(&valid_lp, &opening_batch, 1, 4, 1).unwrap();
    let setup_groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims: 1,
        depth_fold: 1,
        a_row_start: 1,
        b_row_start: 2,
    }];
    let err = match SetupContributionPlan::prepare::<F>(
        &lp,
        &opening_batch,
        RelationMatrixRowLayout::WithDBlock,
        vec![F::one(); 4].into(),
        &witness_layout,
        3,
        &setup_groups,
        &[],
        None,
        CommitmentRingDims::uniform(D),
    ) {
        Ok(_) => panic!("zero num_live_blocks should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}
