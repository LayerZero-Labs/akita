//! Run a synthetic Ajtai opening proof for D=128 over the 32-bit field.

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime32Offset99;
use akita_transcript::Blake2bTranscript;
use akita_zk::compact::CompactRingVec;
use akita_zk::norm::{field_from_centered_i128, sample_ring_vec_box};
use akita_zk::protocols::{
    prove_compact_ajtai_opening, prove_compact_gaussian_heuristic_ajtai_opening,
    verify_compact_ajtai_opening, verify_compact_gaussian_heuristic_ajtai_opening,
};
use akita_zk::rejection::{BoxRejectionParams, GaussianRejectionParams};
use akita_zk::relations::AjtaiRelation;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

type F = Prime32Offset99;
type Tr = Blake2bTranscript<F>;

const D: usize = 128;
const TAIL_COEFFS: usize = 51_872;
const WITNESS_BITS: u32 = 5;
const WITNESS_BOUND: u128 = 1 << (WITNESS_BITS - 1);
const K: usize = 2;
const FIELD_BYTES: usize = 4;

fn sparse_ring(seed: usize) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = field_from_centered_i128(1).unwrap();
    coeffs[(seed * 7 + 3) % D] = field_from_centered_i128(2).unwrap();
    coeffs[(seed * 11 + 5) % D] = field_from_centered_i128(-1).unwrap();
    CyclotomicRing::from_coefficients(coeffs)
}

fn bits_for_response_bound(bound: u128) -> u32 {
    CompactRingVec::bits_for_bound(bound).expect("response bound must fit compact encoding")
}

fn compact_estimate_bytes(padded_coeffs: usize, response_bits: u32) -> usize {
    let packed_response_bytes = (padded_coeffs * response_bits as usize).div_ceil(8);
    let announcement_bytes = K * D * FIELD_BYTES;
    announcement_bytes + packed_response_bytes
}

fn main() {
    let ell = TAIL_COEFFS.div_ceil(D);
    let padded_coeffs = ell * D;
    let padding_coeffs = padded_coeffs - TAIL_COEFFS;
    let cfg = SparseChallengeConfig::Uniform {
        weight: 31,
        nonzero_coeffs: vec![-1, 1],
    };
    let params = BoxRejectionParams::for_half_acceptance(ell, D, &cfg, WITNESS_BOUND).unwrap();
    let gaussian_params =
        GaussianRejectionParams::for_l2_bound(ell, D, &cfg, WITNESS_BOUND, 16.0, 128, 128).unwrap();
    let response_bits = bits_for_response_bound(params.response_bound);
    let gaussian_response_bits = bits_for_response_bound(gaussian_params.response_bound);

    println!(
        "tail_coeffs={TAIL_COEFFS}, padded_coeffs={padded_coeffs}, padding_coeffs={padding_coeffs}, D={D}, ell={ell}, k={K}"
    );
    println!(
        "field_bytes=4, witness_bits={WITNESS_BITS}, witness_bound={}, challenge_l1={}",
        params.witness_bound, params.challenge_l1_bound
    );
    println!(
        "box_beta={}, box_gamma={}, box_response_bound={}, box_p_accept={:.9}",
        params.beta,
        params.gamma,
        params.response_bound,
        params.acceptance_probability()
    );
    println!(
        "box_compact_response_bits={}, box_compact_estimate_bytes={}",
        response_bits,
        compact_estimate_bytes(padded_coeffs, response_bits)
    );
    println!(
        "gaussian_beta={}, gaussian_sigma={:.3}, gaussian_response_bound={}, gaussian_estimated_p_accept={:.9}",
        gaussian_params.beta,
        gaussian_params.sigma,
        gaussian_params.response_bound,
        gaussian_params.estimated_acceptance_probability()
    );
    println!(
        "gaussian_compact_response_bits={}, gaussian_compact_estimate_bytes={}",
        gaussian_response_bits,
        compact_estimate_bytes(padded_coeffs, gaussian_response_bits)
    );
    println!("gaussian_width_sweep=width,bits,estimated_bytes,estimated_acceptance,response_bound");
    for width_factor in [
        32.0, 24.0, 20.0, 16.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0,
    ] {
        let sweep_params = GaussianRejectionParams::for_l2_bound(
            ell,
            D,
            &cfg,
            WITNESS_BOUND,
            width_factor,
            128,
            128,
        )
        .unwrap();
        let bits = bits_for_response_bound(sweep_params.response_bound);
        println!(
            "gaussian_sweep={width_factor:.1},{bits},{},{:.9},{}",
            compact_estimate_bytes(padded_coeffs, bits),
            sweep_params.estimated_acceptance_probability(),
            sweep_params.response_bound
        );
    }

    let mut rng = StdRng::seed_from_u64(0xd128_0032);
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
        "box_measured_compact_bytes={}, box_packed_response_bytes={}",
        proof.serialized_size(),
        proof.response.packed_byte_len()
    );

    for width_factor in [16.0, 8.0, 5.0, 2.5] {
        let gaussian_params = GaussianRejectionParams::for_l2_bound(
            ell,
            D,
            &cfg,
            WITNESS_BOUND,
            width_factor,
            128,
            128,
        )
        .unwrap();
        let start = Instant::now();
        let gaussian_proof = prove_compact_gaussian_heuristic_ajtai_opening::<F, Tr, _, D>(
            &relation,
            &witness,
            &cfg,
            &gaussian_params,
            &mut rng,
            20_000,
        )
        .unwrap();
        let gaussian_prove_ms = start.elapsed().as_millis();

        let start = Instant::now();
        let gaussian_verifies = verify_compact_gaussian_heuristic_ajtai_opening::<F, Tr, D>(
            &relation,
            &cfg,
            &gaussian_params,
            &gaussian_proof,
        )
        .unwrap();
        let gaussian_verify_ms = start.elapsed().as_millis();

        println!(
            "gaussian_width={width_factor:.1}, verifies={gaussian_verifies}, prove_time_ms={gaussian_prove_ms}, verify_time_ms={gaussian_verify_ms}, measured_compact_bytes={}, packed_response_bytes={}",
            gaussian_proof.serialized_size(),
            gaussian_proof.response.packed_byte_len()
        );
    }
}
