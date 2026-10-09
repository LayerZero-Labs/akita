#![cfg(feature = "labinius")]

#[path = "root_reduction_support.rs"]
mod support;

use akita_algebra::binary::BinaryField192;
use jolt_field::{CanonicalBytes, Ring};
use support::{common::TestHost, Case, BASES, F, H};

// A backend-selected cryptographic digest of the complete byte string. Each
// byte is embedded as its canonical small integer, preserving length and order.
fn digest(bytes: &[u8]) -> String {
    let values: Vec<F> = bytes
        .iter()
        .map(|&byte| F::from_u64(u64::from(byte)))
        .collect();
    akita_transcript::field_digest(&values)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn check<T: TestHost>(expected: [(&str, &str); 3]) {
    for (base, (commitment_digest, proof_digest)) in BASES.into_iter().zip(expected) {
        let case = Case::<T>::new(base, 1);
        let commitment_bytes: Vec<u8> = case
            .commitment
            .images
            .iter()
            .flat_map(|image| image.coefficients().iter())
            .flat_map(|coefficient| {
                let mut bytes = [0u8; 16];
                coefficient.to_bytes_le(&mut bytes);
                bytes
            })
            .collect();
        let proof = case.prove().0;
        let actual = (digest(&commitment_bytes), digest(&proof));
        assert_eq!(actual.0, commitment_digest, "{base:?}");
        assert_eq!(actual.1, proof_digest, "{base:?}");
    }
}

// Commitment digests remain frozen at ba65e6c8a101def033aa555c5699118cfeefbb37.
// The v2 proof digests were captured after root_kernels matched both profiles,
// hosts, and all digit bases, at geometry (4, 1, 128).
#[test]
fn frozen_tag0_commitment_and_reduction_bytes() {
    #[cfg(feature = "transcript-blake2b")]
    let (host128, host192) = (
        [
            (
                "c382d2b13699fd3718724321f5c4a5facdd73257771a0efe04aaf22dadc3fbac",
                "9cfd4ac54e44f11a6069148220358a8e2299c63085c52b3293ed858870fbf8f9",
            ),
            (
                "c382d2b13699fd3718724321f5c4a5facdd73257771a0efe04aaf22dadc3fbac",
                "7eafcc45290888e413876c90116e4f77ca951f6244e978832240d42f48446527",
            ),
            (
                "c382d2b13699fd3718724321f5c4a5facdd73257771a0efe04aaf22dadc3fbac",
                "528687f0d5d784eea1ae1583696e4215072feb942f56b74129b6301dd0a93c73",
            ),
        ],
        [
            (
                "3df7927ec2e4529c6f481edec1d6140d2a0dfc973e083af2ced93fc84f8b478f",
                "02788171a56911163b3935f1b1eb1eef44f0882f1d00ff80423d5e26f013cf98",
            ),
            (
                "3df7927ec2e4529c6f481edec1d6140d2a0dfc973e083af2ced93fc84f8b478f",
                "e4a32c8f4340551830fe54b4098c6aea004f4d26e126ee89f51e4c6d13cde391",
            ),
            (
                "3df7927ec2e4529c6f481edec1d6140d2a0dfc973e083af2ced93fc84f8b478f",
                "0d23541ceb667330bd7beed0fa84b1caf37647f847f2a5c34eb70c2f0680b646",
            ),
        ],
    );
    #[cfg(feature = "transcript-keccak")]
    let (host128, host192) = (
        [
            (
                "52ce60a6b31a4655b93989f6579518b89f5c57d25c1f606a01f3994353ad63c1",
                "926b5d68e40019e03fa8c7e7fd3696c29beabe28716c4e31617ea4669b9ecc22",
            ),
            (
                "52ce60a6b31a4655b93989f6579518b89f5c57d25c1f606a01f3994353ad63c1",
                "0b891879ba0c5828e54f97459f162f0bac17c98ef2a38a7d56ab14e155f9bcc2",
            ),
            (
                "52ce60a6b31a4655b93989f6579518b89f5c57d25c1f606a01f3994353ad63c1",
                "9e6764f862fecab0e4764bd3cd71261e4c6be19a629c6f09e7106b5d2782ecb0",
            ),
        ],
        [
            (
                "d89a6458b47aa157f0a12d62e1148255c78b9801667b13ae7f3d9c04d0cccd98",
                "0d0b5a1463637d51fc6d3dcdd1d5fb563356267d16f09c03e6d3a71d58f62e4d",
            ),
            (
                "d89a6458b47aa157f0a12d62e1148255c78b9801667b13ae7f3d9c04d0cccd98",
                "886057eb5495e1f3985f050a4438815f8e42fa5c238f3f1622820230a60ef852",
            ),
            (
                "d89a6458b47aa157f0a12d62e1148255c78b9801667b13ae7f3d9c04d0cccd98",
                "a5b5051ede8dcf0501fcd6f46122b745d2a32b1b255d2ec14e379916977a0115",
            ),
        ],
    );
    check::<H>(host128);
    check::<BinaryField192>(host192);
}
