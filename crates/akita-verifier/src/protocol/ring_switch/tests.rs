use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp32;
use akita_types::{OpeningClaimsLayout, SetupContributionPlan, SisModulusProfileId};

type F = Fp32<251>;
const D: usize = 64;

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
    let lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    );
    let opening_batch = OpeningClaimsLayout::new(0, 1).expect("opening batch");
    let valid_lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    )
    .with_decomp(1, 1, 1, 1, 1)
    .unwrap();
    let witness_layout = WitnessLayout::new(&valid_lp, &opening_batch, 1, 4, 1).unwrap();
    let err = match SetupContributionPlan::prepare::<F>(
        &lp,
        &opening_batch,
        vec![F::one(); 4].into(),
        &witness_layout,
        &[],
        CommitmentRingDims::uniform(D),
        D,
    ) {
        Ok(_) => panic!("zero num_live_blocks should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}
