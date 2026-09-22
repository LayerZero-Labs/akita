use super::BinaryChallengeProfile;
use crate::sampler::shake256;

const SIGN_DOMAIN: &[u8] = b"akita/labinius/binary-sign/v1";

pub(super) fn coefficient_signs(
    profile: &BinaryChallengeProfile,
    support: &[u16],
    input: &mut Vec<u8>,
    output: &mut Vec<u8>,
) {
    input.clear();
    input.extend_from_slice(SIGN_DOMAIN);
    input.extend_from_slice(profile.identity_bytes());
    let support_offset = input.len();
    input.resize(
        support_offset + profile.scalar_ring().degree().div_ceil(8),
        0,
    );
    for &position in support {
        let position = usize::from(position);
        input[support_offset + position / 8] |= 1 << (position % 8);
    }
    output.resize(support.len(), 0);
    shake256(input, output);
}
