use super::support::{evaluate, Fixture, Host};
use akita_algebra::binary::BinaryField128;
use akita_labinius_pcs::{config::Digits2, RootSetup};
use akita_types::proof::AkitaSetupSeed;
use std::time::Instant;

#[test]
#[ignore = "first admitted root geometry; run explicitly in release with parallel"]
fn full_first_profile_two_bit_digits() {
    let start = Instant::now();
    let root = RootSetup::derive(
        crate::common::PROFILE,
        22,
        8,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap();
    eprintln!(
        "root full root_setup_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let (images, digits) = Fixture::<Digits2>::catalogs(&root);
    eprintln!(
        "root full catalogs_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let fixture = Fixture::<Digits2>::from_catalogs(&root, images, digits);
    let prover = fixture.prover(root.clone());
    let verifier = fixture.verifier(root.clone());
    eprintln!(
        "root full setup_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    let source = BinaryField128::source(root.setup().source_len());
    let point = BinaryField128::point(root.setup().num_vars());
    let value = evaluate::<BinaryField128>(&source, &point);
    let start = Instant::now();
    let output = prover.commit::<BinaryField128>(&source).unwrap();
    eprintln!(
        "root full commit_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let proof = prover
        .open::<BinaryField128>(&source, &output, &point, value)
        .unwrap();
    eprintln!(
        "root full open_seconds={:.6} proof_bytes={}",
        start.elapsed().as_secs_f64(),
        proof.len()
    );
    let start = Instant::now();
    verifier
        .verify::<BinaryField128>(&output.committed_group, &point, value, &proof)
        .unwrap();
    eprintln!(
        "root full verify_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
}
