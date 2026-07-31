use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};

use crate::descriptor_bytes::{push_u128, push_usize};
use crate::sis::{FoldWitnessLinfCapConfig, FoldWitnessLinfCapPolicy};

pub(crate) fn append_sparse_challenge_descriptor_bytes(
    bytes: &mut Vec<u8>,
    config: &SparseChallengeConfig,
) {
    bytes.push(0);
    push_usize(bytes, config.count_pm1);
    push_usize(bytes, config.count_pm2);
}

fn append_fold_linf_policy_descriptor_bytes(bytes: &mut Vec<u8>, policy: FoldWitnessLinfCapPolicy) {
    bytes.push(match policy {
        FoldWitnessLinfCapPolicy::TailBoundWithGrind => 0,
        FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => 1,
        FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => 2,
    });
}

pub(crate) fn append_fold_linf_cap_config_descriptor_bytes(
    bytes: &mut Vec<u8>,
    config: &FoldWitnessLinfCapConfig,
) {
    append_fold_linf_policy_descriptor_bytes(bytes, config.policy);
    push_u128(bytes, config.challenge_l2_sq_max);
    push_u128(bytes, config.tensor_factor_l2_sq_max);
    push_u128(bytes, config.tensor_factor_nonzero_count_max);
    push_usize(bytes, config.tensor_fold_low_len);
    push_u128(bytes, config.num_fold_coeffs);
    push_u128(bytes, config.grind_target_accept_num);
    push_u128(bytes, config.grind_target_accept_den);
    push_u128(bytes, config.grind_union_ln);
}

pub(super) fn append_tensor_challenge_shape_descriptor_bytes(
    bytes: &mut Vec<u8>,
    shape: TensorChallengeShape,
) {
    match shape {
        TensorChallengeShape::Flat => bytes.push(0),
        TensorChallengeShape::Tensor { fold_low_len } => {
            bytes.push(1);
            push_usize(bytes, fold_low_len);
        }
    }
}
