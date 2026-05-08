//! Run a synthetic Ajtai opening proof for the current production tail shape.

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime128OffsetA7F7;
use akita_transcript::Blake2bTranscript;
use akita_zk::norm::{field_from_centered_i128, sample_ring_vec_box};
use akita_zk::protocols::{prove_compact_ajtai_opening, verify_compact_ajtai_opening};
use akita_zk::rejection::BoxRejectionParams;
use akita_zk::relations::AjtaiRelation;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

type F = Prime128OffsetA7F7;
type Tr = Blake2bTranscript<F>;

const D: usize = 32;
const TAIL_COEFFS: usize = 51_872;
const WITNESS_BITS: u32 = 5;
const WITNESS_BOUND: u128 = 1 << (WITNESS_BITS - 1);
const K: usize = 2;

fn sparse_ring(seed: usize) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = field_from_centered_i128(1).unwrap();
    coeffs[(seed * 7 + 3) % D] = field_from_centered_i128(2).unwrap();
    coeffs[(seed * 11 + 5) % D] = field_from_centered_i128(-1).unwrap();
    CyclotomicRing::from_coefficients(coeffs)
}

fn main() {
    assert_eq!(TAIL_COEFFS % D, 0);
    let ell = TAIL_COEFFS / D;
    let cfg = SparseChallengeConfig::BoundedL1Norm;
    let params = BoxRejectionParams::for_half_acceptance(ell, D, &cfg, WITNESS_BOUND).unwrap();

    let response_bits = 1 + u128::BITS - params.response_bound.leading_zeros();
    let packed_response_bytes = (TAIL_COEFFS * response_bits as usize).div_ceil(8);
    let announcement_bytes = K * D * 16;

    println!("tail_coeffs={TAIL_COEFFS}, D={D}, ell={ell}, k={K}");
    println!(
        "witness_bits={WITNESS_BITS}, witness_bound={}, challenge_l1={}",
        params.witness_bound, params.challenge_l1_bound
    );
    println!(
        "beta={}, gamma={}, response_bound={}, p_accept={:.9}",
        params.beta,
        params.gamma,
        params.response_bound,
        params.acceptance_probability()
    );
    println!(
        "current_full_field_bytes={}, compact_response_bits={}, compact_estimate_bytes={}",
        (K + ell) * D * 16,
        response_bits,
        announcement_bytes + packed_response_bytes
    );

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let witness = sample_ring_vec_box::<F, _, D>(&mut rng, ell, WITNESS_BOUND - 1).unwrap();
    let matrix = (0..K)
        .map(|row| (0..ell).map(|col| sparse_ring(row * ell + col)).collect())
        .collect::<Vec<Vec<_>>>();

    let start = Instant::now();
    let commitment = akita_zk::relations::matrix_vector_mul(&matrix, &witness).unwrap();
    let relation = AjtaiRelation::new(matrix, commitment).unwrap();
    println!("setup_time_ms={}", start.elapsed().as_millis());

    let start = Instant::now();
    let proof = prove_compact_ajtai_opening::<F, Tr, _, D>(
        &relation, &witness, &cfg, &params, &mut rng, 32,
    )
    .unwrap();
    let prove_ms = start.elapsed().as_millis();

    let start = Instant::now();
    let verifies =
        verify_compact_ajtai_opening::<F, Tr, D>(&relation, &cfg, &params, &proof).unwrap();
    let verify_ms = start.elapsed().as_millis();

    println!("verifies={verifies}, prove_time_ms={prove_ms}, verify_time_ms={verify_ms}");
    println!(
        "measured_compact_bytes={}, packed_response_bytes={}",
        proof.serialized_size(),
        proof.response.packed_byte_len()
    );
}
