use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};

use crate::descriptor_bytes::push_usize;

pub(crate) fn append_sparse_challenge_descriptor_bytes(
    bytes: &mut Vec<u8>,
    config: &SparseChallengeConfig,
) {
    bytes.push(0);
    push_usize(bytes, config.count_pm1);
    push_usize(bytes, config.count_pm2);
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
