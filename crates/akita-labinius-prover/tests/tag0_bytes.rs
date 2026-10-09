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

// Captured at frozen base ba65e6c8a101def033aa555c5699118cfeefbb37,
// with (log_num_cells, log_fold_width, lambda_fold) = (4, 1, 128).
#[test]
fn frozen_tag0_commitment_and_reduction_bytes() {
    #[cfg(feature = "transcript-blake2b")]
    let (host128, host192) = (
        [
            (
                "c382d2b13699fd3718724321f5c4a5facdd73257771a0efe04aaf22dadc3fbac",
                "cd1a43605c593df2af7d6f632921dfb8620f54ef3989a660ff99ed74eed19993",
            ),
            (
                "c382d2b13699fd3718724321f5c4a5facdd73257771a0efe04aaf22dadc3fbac",
                "e2c2c456fb2199fed960f89c83199666f1cc8ed9d8f24600344c833ce66078be",
            ),
            (
                "c382d2b13699fd3718724321f5c4a5facdd73257771a0efe04aaf22dadc3fbac",
                "312b5148212fb9b57bd5b316b0b38466689dddc63544217d48abcd63315804f8",
            ),
        ],
        [
            (
                "3df7927ec2e4529c6f481edec1d6140d2a0dfc973e083af2ced93fc84f8b478f",
                "047b838c83a605d448e8aa6612628427a41e24843282241b008e432f2e5630bc",
            ),
            (
                "3df7927ec2e4529c6f481edec1d6140d2a0dfc973e083af2ced93fc84f8b478f",
                "53879c3f6ac5de93bb1883765560fe24e8fa517c61671a64fbef189ab1af5c64",
            ),
            (
                "3df7927ec2e4529c6f481edec1d6140d2a0dfc973e083af2ced93fc84f8b478f",
                "521af5921cba99b1ffc91c3c9b36de1ab6c73c50aae00d9a13a69070a0dc889f",
            ),
        ],
    );
    #[cfg(feature = "transcript-keccak")]
    let (host128, host192) = (
        [
            (
                "52ce60a6b31a4655b93989f6579518b89f5c57d25c1f606a01f3994353ad63c1",
                "7348d90eacc8298aabdd497f0fdbf12d52cb59cc2d2b63634f62c9bba04f6c7d",
            ),
            (
                "52ce60a6b31a4655b93989f6579518b89f5c57d25c1f606a01f3994353ad63c1",
                "610af648b88a23a5932b972e86bc9462bbf2edbf29b7473c39dfb9a486c49269",
            ),
            (
                "52ce60a6b31a4655b93989f6579518b89f5c57d25c1f606a01f3994353ad63c1",
                "43deefbcf2dc7456719e542f77f306949cf8ef3b6bea00736ef534b9dbf18386",
            ),
        ],
        [
            (
                "d89a6458b47aa157f0a12d62e1148255c78b9801667b13ae7f3d9c04d0cccd98",
                "2461cbddcb3f3af69a19578e445ed724f5d32174a3fcc659100c5d32d9229601",
            ),
            (
                "d89a6458b47aa157f0a12d62e1148255c78b9801667b13ae7f3d9c04d0cccd98",
                "8f2a786e74f59b0bce6bb32158243644f34401a2ebe19547600ea055313b0c97",
            ),
            (
                "d89a6458b47aa157f0a12d62e1148255c78b9801667b13ae7f3d9c04d0cccd98",
                "50442f4ba52496cf22afabb0b27a82cefb21453808b4f835bd2b29cc8c8b6b2d",
            ),
        ],
    );
    check::<H>(host128);
    check::<BinaryField192>(host192);
}
