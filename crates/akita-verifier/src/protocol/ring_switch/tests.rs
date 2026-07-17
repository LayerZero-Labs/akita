use super::*;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::scalar_powers;
use akita_challenges::{SparseChallenge, SparseChallengeConfig, TensorChallenges};
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

#[test]
fn tensor_et_intervals_match_dense_oracle_across_residual_shards() {
    let lp = LevelParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        2,
        1,
        1,
        fold_challenge_config(),
    )
    .with_decomp(4, 25, 1, 3)
    .unwrap();
    let opening_batch = OpeningClaimsLayout::new(0, 2).unwrap();
    let witness_layout = WitnessLayout::new(&lp, &opening_batch, 2, 4, 2).unwrap();
    let units = witness_layout.units_for_group(0).unwrap();
    assert_eq!(
        units
            .iter()
            .map(|unit| unit.num_live_blocks())
            .collect::<Vec<_>>(),
        vec![4, 3]
    );

    let sparse = |position: u32, sign: i8| SparseChallenge {
        positions: vec![position],
        coeffs: vec![sign],
    };
    let tensor = TensorChallenges {
        fold_high: vec![sparse(0, 1), sparse(1, -1), sparse(2, 1), sparse(3, 1)],
        fold_low: vec![
            sparse(4, 1),
            sparse(5, 1),
            sparse(6, -1),
            sparse(7, 1),
            sparse(8, -1),
            sparse(9, 1),
            sparse(10, 1),
            sparse(11, -1),
        ],
        num_live_blocks_per_claim: 7,
        fold_low_len: 4,
        num_claims: 2,
    };
    let alpha_pows = scalar_powers(F::from_u64(5), D);
    let group = RelationMatrixGroupEvaluator {
        c_alphas: PreparedChallengeEvals::Tensor {
            challenges: tensor.clone(),
            alpha_pows: alpha_pows.clone(),
        },
        opening_a_evals: Vec::new(),
        group_id: 0,
        num_claims: 2,
        num_live_blocks: 7,
        depth_open: 3,
        depth_commit: 1,
        depth_fold: 1,
        log_basis: 2,
        n_a: 2,
        a_row_start: 1,
        b_row_start: 3,
    };
    let opening_source_len = witness_layout.total_len();
    let bits = opening_source_len.next_power_of_two().trailing_zeros() as usize;
    let x_challenges = (0..bits)
        .map(|index| F::from_u64(17 + index as u64))
        .collect::<Vec<_>>();
    let consistency_weight = F::from_u64(29);
    let a_row_weights = [F::from_u64(31), F::from_u64(37)];
    let gadget = [F::from_u64(1), F::from_u64(4), F::from_u64(16)];

    let got = evaluate_group_et_contributions::<F, F>(
        &group,
        &units,
        opening_source_len,
        &x_challenges,
        consistency_weight,
        &a_row_weights,
        &gadget,
    )
    .unwrap();

    let mut expected = (F::zero(), F::zero());
    for &unit in &units {
        for claim in 0..group.num_claims {
            for global_block in unit.global_block_range() {
                let logical = claim * group.num_live_blocks + global_block;
                let challenge = tensor
                    .eval_logical_at_pows::<F, F>(logical, &alpha_pows)
                    .unwrap();
                for (digit, &digit_weight) in gadget.iter().enumerate() {
                    let e_index = unit
                        .e_index(
                            group.num_claims,
                            group.depth_open,
                            claim,
                            global_block,
                            digit,
                        )
                        .unwrap();
                    expected.0 += eq_eval_at_index(&x_challenges, e_index)
                        * consistency_weight
                        * challenge
                        * digit_weight;
                    for (a_row, &row_weight) in a_row_weights.iter().enumerate() {
                        let t_index = unit
                            .t_index(
                                group.num_claims,
                                group.n_a,
                                group.depth_open,
                                claim,
                                global_block,
                                a_row,
                                digit,
                            )
                            .unwrap();
                        expected.1 += eq_eval_at_index(&x_challenges, t_index)
                            * row_weight
                            * challenge
                            * digit_weight;
                    }
                }
            }
        }
    }
    assert_eq!(got, expected);
}
