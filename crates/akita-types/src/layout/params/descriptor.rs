use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};

use crate::descriptor_bytes::push_usize;
use crate::sis::FoldWitnessLinfCapPolicy;

pub(crate) fn append_sparse_challenge_descriptor_bytes(
    bytes: &mut Vec<u8>,
    config: &SparseChallengeConfig,
) {
    bytes.push(0);
    push_usize(bytes, config.count_pm1);
    push_usize(bytes, config.count_pm2);
}

pub(super) fn append_fold_linf_policy_descriptor_bytes(
    bytes: &mut Vec<u8>,
    policy: FoldWitnessLinfCapPolicy,
) {
    bytes.push(match policy {
        FoldWitnessLinfCapPolicy::TailBoundWithGrind => 0,
        FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => 1,
        FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => 2,
    });
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
