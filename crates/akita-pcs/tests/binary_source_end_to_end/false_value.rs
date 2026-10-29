//! Complete malicious openings with a false value bound before the frontend.
//!
//! Generated from commit 95a74e633 by changing the independent reference word
//! value's low bit before constructing the opening claims, removing only the
//! honest prover's `reconstructed_value != decoded_value` refusal, and running
//! the ordinary grouped prover to completion. All partials, sumcheck messages,
//! the terminal and the suffix are honest for that false statement's transcript.
//! The production prover guard was restored after generating these fixtures.
//! Recorded with `transcript-blake2b`, the acceptance-gate backend. Separate
//! fixtures cover both wire modes; no transcript policy is overridden.

use super::*;

pub(super) fn proof(width: usize) -> &'static [u8] {
    match (akita_params::DEV_PROTOCOL, width) {
        (false, 8) => include_bytes!("false_value/current-8.bin"),
        (false, 128) => include_bytes!("false_value/current-128.bin"),
        (true, 8) => include_bytes!("false_value/dev-8.bin"),
        (true, 128) => include_bytes!("false_value/dev-128.bin"),
        _ => panic!("unsupported false-value fixture"),
    }
}

/// Check the entire partial prefix independently of the field-switch kernels.
pub(super) fn assert_honest_partials(
    descriptor: &BinarySourceDescriptor,
    claim: &BinaryWordClaim,
    carriers: &[u128],
    proof: &[u8],
) {
    let point = claim
        .point()
        .iter()
        .map(|bytes| reference::host(u128::from_le_bytes(*bytes)))
        .collect::<Vec<_>>();
    assert_ne!(
        reference::encoding(reference::logical_value(descriptor, carriers, &point)),
        *claim.value(),
        "fixture must bind a false logical word value",
    );
    let intra = descriptor.index_map().claim_variable_split().intra_carrier;
    let mut partials = [0u128; 128];
    for (address, &carrier) in carriers.iter().enumerate() {
        let mut equality = BinaryField128::ONE;
        for (axis, &coordinate) in point[intra..].iter().enumerate() {
            equality = equality
                * if (address >> axis) & 1 == 0 {
                    BinaryField128::ONE + coordinate
                } else {
                    coordinate
                };
        }
        let bits = u128::from_le_bytes(reference::encoding(equality));
        for (bit, partial) in partials.iter_mut().enumerate() {
            if (bits >> bit) & 1 != 0 {
                *partial ^= carrier;
            }
        }
    }
    for (bit, (bytes, expected)) in proof[..128 * 16].chunks_exact(16).zip(partials).enumerate() {
        assert_eq!(bytes, expected.to_le_bytes(), "carrier partial {bit}");
    }
}
