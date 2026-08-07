use akita_challenges::SparseChallengeConfig;

use crate::descriptor_bytes::push_usize;

pub(crate) fn append_sparse_challenge_descriptor_bytes(
    bytes: &mut Vec<u8>,
    config: &SparseChallengeConfig,
) {
    bytes.push(0);
    push_usize(bytes, config.count_pm1);
    push_usize(bytes, config.count_pm2);
}
