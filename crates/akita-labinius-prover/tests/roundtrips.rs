#![cfg(feature = "labinius")]

mod common;
use common::{data, setup, TestHost, TestPrime};

use akita_algebra::{
    binary::{BinaryField128, BinaryField192},
    MinusTrinomial, PlusTrinomial, Prime64Offset23703, TrinomialModulus,
};
use akita_labinius_prover::{commit_binary_clear, prove_binary_clear_bytes};
use akita_labinius_verifier::verify_binary_clear_bytes;
use jolt_field::Prime128OffsetA7F7;

fn roundtrip<H: TestHost, F: TestPrime, const D: usize, M: TrinomialModulus>(n_a: usize) {
    let (m, columns) = if D == 162 { (4, 4) } else { (2, 8) };
    roundtrip_geometry::<H, F, D, M>(n_a, m, columns);
}

fn roundtrip_geometry<H: TestHost, F: TestPrime, const D: usize, M: TrinomialModulus>(
    n_a: usize,
    m: usize,
    columns: usize,
) {
    let setup = setup::<F, D, M>(n_a, m, columns);
    let (source, point, claim) = data::<H>(setup.source_len(), setup.num_vars());
    let commitment = commit_binary_clear::<H, F, D, M>(&setup, &source).unwrap();
    let proof = prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim).unwrap();
    verify_binary_clear_bytes(&setup, &commitment, &point, claim, &proof).unwrap();
    let repeated = prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim).unwrap();
    assert_eq!(proof, repeated);
}

#[test]
fn both_hosts_primes_geometries_and_matrix_ranks() {
    for n_a in [1, 2] {
        roundtrip::<BinaryField128, Prime64Offset23703, 162, PlusTrinomial>(n_a);
        roundtrip::<BinaryField192, Prime64Offset23703, 162, PlusTrinomial>(n_a);
        roundtrip::<BinaryField128, Prime128OffsetA7F7, 162, PlusTrinomial>(n_a);
        roundtrip::<BinaryField192, Prime128OffsetA7F7, 162, PlusTrinomial>(n_a);
        roundtrip::<BinaryField128, Prime64Offset23703, 648, MinusTrinomial>(n_a);
        roundtrip::<BinaryField192, Prime64Offset23703, 648, MinusTrinomial>(n_a);
        roundtrip::<BinaryField128, Prime128OffsetA7F7, 648, MinusTrinomial>(n_a);
        roundtrip::<BinaryField192, Prime128OffsetA7F7, 648, MinusTrinomial>(n_a);
    }
    // The intermediate tower is admitted by the same public geometry contract.
    roundtrip::<BinaryField128, Prime64Offset23703, 324, MinusTrinomial>(1);
    roundtrip_geometry::<BinaryField128, Prime64Offset23703, 648, MinusTrinomial>(1, 2, 4);
    roundtrip_geometry::<BinaryField192, Prime128OffsetA7F7, 648, MinusTrinomial>(1, 2, 4);
}

#[test]
fn bounded_weight_46_d648_p128_roundtrips_for_both_hosts() {
    use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
    use akita_labinius_verifier::BinaryClearSetup;
    use akita_params::sis::labinius::{LabiniusCoefficientPrime, LabiniusRingDegree};

    fn bounded_roundtrip<H: TestHost>() {
        let ordinary = setup::<Prime128OffsetA7F7, 648, MinusTrinomial>(1, 2, 8);
        let bounded = BinaryClearSetup::new(
            ordinary.matrix().to_vec(),
            1,
            2,
            8,
            -1024,
            1024,
            128,
            BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 46).unwrap(),
            LabiniusCoefficientPrime::P128OffsetA7F7,
            LabiniusRingDegree::D648,
        )
        .unwrap();
        let (source, point, claim) = data::<H>(bounded.source_len(), bounded.num_vars());
        let commitment =
            commit_binary_clear::<H, Prime128OffsetA7F7, 648, MinusTrinomial>(&bounded, &source)
                .unwrap();
        let proof =
            prove_binary_clear_bytes(&bounded, &source, &commitment, &point, claim).unwrap();
        verify_binary_clear_bytes(&bounded, &commitment, &point, claim, &proof).unwrap();
    }
    bounded_roundtrip::<BinaryField128>();
    bounded_roundtrip::<BinaryField192>();
}
