//! Side-by-side comparison of rejection-sampling policies on the Akita tail
//! Sigma proof.
//!
//! Compares three rejection rules at the same `(D, K, witness)` shape:
//!
//! - **Box**: exact uniform-box rejection. Zero-knowledge.
//! - **Gaussian (Lyubashevsky standard)**: discrete Gaussian + standard
//!   rejection rule from LNP/Lyubashevsky. Zero-knowledge.
//! - **Gärtner (single-step, public sign)**: discrete Gaussian + the
//!   `f_v / g_v` acceptance from Corollary 1 of Gärtner. **Non-ZK** in this
//!   form because the sign is sent in the clear; reported numbers are a
//!   ceiling on what a sign-hiding variant could achieve.
//!
//! ## Configurable knobs
//!
//! - `HACHI_ZK_TAIL_COEFFS`: total coefficient count (default `51872`,
//!   matching the production tail target on PR #70).
//! - `HACHI_ZK_WITNESS_BITS`: log basis bits (default `5`).
//! - `HACHI_ZK_RUN_PROOFS`: `1` to actually prove + verify selected
//!   width_factor values; `0` to skip and only print analytical numbers
//!   (default `1`).
//! - `HACHI_ZK_PROVE_ATTEMPTS`: max parallel prover attempts per width
//!   (default `20000`).
//!
//! `D = 128`, `K = 2`, and the 32-bit field are fixed in this binary. Other
//! shapes get their own siblings.

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime32Offset99;
use akita_transcript::Blake2bTranscript;
use akita_zk::compact::CompactRingVec;
use akita_zk::measurements::{
    prove_compact_public_sign_gaertner_ajtai_opening,
    verify_compact_public_sign_gaertner_ajtai_opening,
};
use akita_zk::norm::{field_from_centered_i128, sample_ring_vec_box};
use akita_zk::protocols::{
    prove_compact_ajtai_opening, prove_compact_gaussian_heuristic_ajtai_opening,
    verify_compact_ajtai_opening, verify_compact_gaussian_heuristic_ajtai_opening,
};
use akita_zk::rejection::{BoxRejectionParams, GaertnerRejectionParams, GaussianRejectionParams};
use akita_zk::relations::AjtaiRelation;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::time::Instant;

type F = Prime32Offset99;
type Tr = Blake2bTranscript<F>;

const D: usize = 128;
const K: usize = 2;
const FIELD_BYTES: usize = 4;
const DEFAULT_TAIL_COEFFS: usize = 51_872;
const DEFAULT_WITNESS_BITS: u32 = 5;
const DEFAULT_PROVE_ATTEMPTS: usize = 20_000;
const ZK_ERROR_BITS: u32 = 128;
const TAIL_ERROR_BITS: u32 = 128;
const RNG_SEED: u64 = 0x000a_11ce_b00b;

// Width factors for the analytical sweep.
const SWEEP_WIDTH_FACTORS: &[f64] = &[
    32.0, 24.0, 20.0, 16.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.2, 1.0,
];

// Width factors for the actual prove + verify pass.
const RUN_WIDTH_FACTORS: &[f64] = &[16.0, 8.0, 4.0, 2.5, 1.6, 1.0];

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str, default: bool) -> bool {
    match env::var(name).ok().as_deref() {
        Some("0") | Some("false") | Some("no") => false,
        Some(_) => true,
        None => default,
    }
}

fn witness_bound_from_bits(witness_bits: u32) -> Result<u128, String> {
    let shift = witness_bits
        .checked_sub(1)
        .ok_or_else(|| "must be at least 1".to_string())?;
    1u128
        .checked_shl(shift)
        .ok_or_else(|| "must be at most 128".to_string())
}

fn exit_invalid_env(name: &str, value: u32, reason: &str) -> ! {
    eprintln!("invalid {name}={value}: {reason}");
    std::process::exit(2);
}

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

fn compact_estimate_bytes(padded_coeffs: usize, response_bits: u32, sign_byte: bool) -> usize {
    let packed_response_bytes = (padded_coeffs * response_bits as usize).div_ceil(8);
    let announcement_bytes = K * D * FIELD_BYTES;
    announcement_bytes + packed_response_bytes + usize::from(sign_byte)
}

fn lyubashevsky_m(width_factor: f64, zk_error_bits: u32) -> f64 {
    let ln_2 = core::f64::consts::LN_2;
    let a_kappa = (2.0 * (zk_error_bits as f64 + 1.0) * ln_2).sqrt();
    (a_kappa / width_factor + 1.0 / (2.0 * width_factor * width_factor)).exp()
}

fn bliss_bimodal_m(width_factor: f64) -> f64 {
    (1.0 / (2.0 * width_factor * width_factor)).exp()
}

fn main() {
    let tail_coeffs = env_usize("HACHI_ZK_TAIL_COEFFS", DEFAULT_TAIL_COEFFS);
    let witness_bits = env_u32("HACHI_ZK_WITNESS_BITS", DEFAULT_WITNESS_BITS);
    let run_proofs = env_flag("HACHI_ZK_RUN_PROOFS", true);
    let prove_attempts = env_usize("HACHI_ZK_PROVE_ATTEMPTS", DEFAULT_PROVE_ATTEMPTS);

    let witness_bound = witness_bound_from_bits(witness_bits)
        .unwrap_or_else(|reason| exit_invalid_env("HACHI_ZK_WITNESS_BITS", witness_bits, &reason));
    if run_proofs && witness_bits > 64 {
        exit_invalid_env(
            "HACHI_ZK_WITNESS_BITS",
            witness_bits,
            "proof sampling supports values up to 64",
        );
    }
    let ell = tail_coeffs.div_ceil(D);
    let padded_coeffs = ell * D;
    let padding_coeffs = padded_coeffs - tail_coeffs;
    let cfg = SparseChallengeConfig::Uniform {
        weight: 31,
        nonzero_coeffs: vec![-1, 1],
    };

    println!("# akita-zk policy comparison harness");
    println!("# D={D}, K={K}, field=Prime32Offset99 (4 bytes)");
    println!(
        "shape: tail_coeffs={tail_coeffs}, padded_coeffs={padded_coeffs}, padding_coeffs={padding_coeffs}, ell={ell}"
    );
    println!(
        "witness: witness_bits={witness_bits}, witness_bound={witness_bound}, challenge_l1={}",
        cfg.l1_norm()
    );
    println!("zk_error_bits={ZK_ERROR_BITS}, tail_error_bits={TAIL_ERROR_BITS}");
    println!();

    println!("## Box rejection (ZK, exact uniform)");
    let box_params = BoxRejectionParams::for_half_acceptance(ell, D, &cfg, witness_bound).unwrap();
    let box_bits = bits_for_response_bound(box_params.response_bound);
    let box_bytes = compact_estimate_bytes(padded_coeffs, box_bits, false);
    println!(
        "box: beta={}, gamma={}, response_bound={}, p_accept={:.6}, response_bits={}, est_bytes={}",
        box_params.beta,
        box_params.gamma,
        box_params.response_bound,
        box_params.acceptance_probability(),
        box_bits,
        box_bytes
    );
    println!();

    // ---- Analytical sweep over width factors ----
    println!("## Analytical sweep over width_factor (alpha)");
    println!("policy=headers=alpha,sigma,response_bits,bytes,m,p_accept");
    for &wf in SWEEP_WIDTH_FACTORS {
        let g_params = GaussianRejectionParams::for_l2_bound(
            ell,
            D,
            &cfg,
            witness_bound,
            wf,
            ZK_ERROR_BITS,
            TAIL_ERROR_BITS,
        )
        .unwrap();
        let g_bits = bits_for_response_bound(g_params.response_bound);
        let g_bytes = compact_estimate_bytes(padded_coeffs, g_bits, false);
        let lyub_p = 1.0 / lyubashevsky_m(wf, ZK_ERROR_BITS);

        let gaertner_params = GaertnerRejectionParams::for_l2_bound(
            ell,
            D,
            &cfg,
            witness_bound,
            wf,
            ZK_ERROR_BITS,
            TAIL_ERROR_BITS,
        )
        .unwrap();
        let gaertner_bits = bits_for_response_bound(gaertner_params.response_bound);
        // Gärtner version sends one extra sign byte.
        let gaertner_bytes = compact_estimate_bytes(padded_coeffs, gaertner_bits, true);
        let gaertner_p = gaertner_params.estimated_acceptance_probability();
        let gaertner_m = gaertner_params.rejection_m;

        let bliss_p = 1.0 / bliss_bimodal_m(wf);

        println!(
            "sweep_gaussian={wf:.1},{:.0},{},{},{:.4},{:.9}",
            g_params.sigma,
            g_bits,
            g_bytes,
            lyubashevsky_m(wf, ZK_ERROR_BITS),
            lyub_p
        );
        println!(
            "sweep_bliss={wf:.1},{:.0},{},{},{:.4},{:.9}",
            gaertner_params.sigma,
            gaertner_bits,
            // BLISS would also need sign hiding; counted public-sign for parity.
            compact_estimate_bytes(padded_coeffs, gaertner_bits, true),
            bliss_bimodal_m(wf),
            bliss_p
        );
        println!(
            "sweep_gaertner={wf:.1},{:.0},{},{},{:.6},{:.9}",
            gaertner_params.sigma, gaertner_bits, gaertner_bytes, gaertner_m, gaertner_p,
        );
    }
    println!();

    if !run_proofs {
        println!("# HACHI_ZK_RUN_PROOFS=0; skipping prove/verify pass");
        return;
    }

    // ---- Real prove + verify pass ----
    println!("## Measured prove + verify (real Sigma execution)");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let witness = sample_ring_vec_box::<F, _, D>(&mut rng, ell, witness_bound - 1).unwrap();
    let matrix = (0..K)
        .map(|row| (0..ell).map(|col| sparse_ring(row * ell + col)).collect())
        .collect::<Vec<Vec<_>>>();

    let setup_start = Instant::now();
    let commitment = akita_zk::relations::matrix_vector_mul(&matrix, &witness).unwrap();
    let relation = AjtaiRelation::new(matrix, commitment).unwrap();
    println!("setup_time_ms={}", setup_start.elapsed().as_millis());

    // Box (only one width to run).
    {
        let box_params =
            BoxRejectionParams::for_half_acceptance(ell, D, &cfg, witness_bound).unwrap();
        let bits = bits_for_response_bound(box_params.response_bound);
        let est_bytes = compact_estimate_bytes(padded_coeffs, bits, false);
        let prove_start = Instant::now();
        let proof = prove_compact_ajtai_opening::<F, Tr, _, D>(
            &relation,
            &witness,
            &cfg,
            &box_params,
            &mut rng,
            64,
        )
        .unwrap();
        let prove_ms = prove_start.elapsed().as_millis();
        let verify_start = Instant::now();
        let ok =
            verify_compact_ajtai_opening::<F, Tr, D>(&relation, &cfg, &box_params, &proof).unwrap();
        let verify_ms = verify_start.elapsed().as_millis();
        println!(
            "run_box: verifies={ok}, prove_ms={prove_ms}, verify_ms={verify_ms}, response_bits={bits}, est_bytes={est_bytes}, measured_bytes={}",
            proof.serialized_size()
        );
    }

    for &wf in RUN_WIDTH_FACTORS {
        // Gaussian (Lyubashevsky standard).
        let g_params = GaussianRejectionParams::for_l2_bound(
            ell,
            D,
            &cfg,
            witness_bound,
            wf,
            ZK_ERROR_BITS,
            TAIL_ERROR_BITS,
        )
        .unwrap();
        let g_bits = bits_for_response_bound(g_params.response_bound);
        let g_est = compact_estimate_bytes(padded_coeffs, g_bits, false);
        let prove_start = Instant::now();
        let g_proof = prove_compact_gaussian_heuristic_ajtai_opening::<F, Tr, _, D>(
            &relation,
            &witness,
            &cfg,
            &g_params,
            &mut rng,
            prove_attempts,
        );
        let g_prove_ms = prove_start.elapsed().as_millis();
        match g_proof {
            Ok(proof) => {
                let verify_start = Instant::now();
                let ok = verify_compact_gaussian_heuristic_ajtai_opening::<F, Tr, D>(
                    &relation, &cfg, &g_params, &proof,
                )
                .unwrap();
                let verify_ms = verify_start.elapsed().as_millis();
                println!(
                    "run_gaussian: alpha={wf:.1}, verifies={ok}, prove_ms={g_prove_ms}, verify_ms={verify_ms}, response_bits={g_bits}, est_bytes={g_est}, measured_bytes={}, p_accept={:.6}",
                    proof.serialized_size(),
                    1.0 / lyubashevsky_m(wf, ZK_ERROR_BITS)
                );
            }
            Err(e) => {
                println!(
                    "run_gaussian: alpha={wf:.1}, FAILED prove_ms={g_prove_ms}, p_accept={:.6}, err={e}",
                    1.0 / lyubashevsky_m(wf, ZK_ERROR_BITS)
                );
            }
        }

        // Gärtner (public sign, non-ZK).
        let gaertner_params = GaertnerRejectionParams::for_l2_bound(
            ell,
            D,
            &cfg,
            witness_bound,
            wf,
            ZK_ERROR_BITS,
            TAIL_ERROR_BITS,
        )
        .unwrap();
        let gaertner_bits = bits_for_response_bound(gaertner_params.response_bound);
        let gaertner_est = compact_estimate_bytes(padded_coeffs, gaertner_bits, true);
        let prove_start = Instant::now();
        let gaertner_proof = prove_compact_public_sign_gaertner_ajtai_opening::<F, Tr, _, D>(
            &relation,
            &witness,
            &cfg,
            &gaertner_params,
            &mut rng,
            prove_attempts,
        );
        let gaertner_prove_ms = prove_start.elapsed().as_millis();
        match gaertner_proof {
            Ok(proof) => {
                let verify_start = Instant::now();
                let ok = verify_compact_public_sign_gaertner_ajtai_opening::<F, Tr, D>(
                    &relation,
                    &cfg,
                    &gaertner_params,
                    &proof,
                )
                .unwrap();
                let verify_ms = verify_start.elapsed().as_millis();
                println!(
                    "run_gaertner: alpha={wf:.1}, verifies={ok}, prove_ms={gaertner_prove_ms}, verify_ms={verify_ms}, response_bits={gaertner_bits}, est_bytes={gaertner_est}, measured_bytes={}, sign={}, p_accept={:.6}",
                    proof.serialized_size(),
                    proof.sign,
                    gaertner_params.estimated_acceptance_probability()
                );
            }
            Err(e) => {
                println!(
                    "run_gaertner: alpha={wf:.1}, FAILED prove_ms={gaertner_prove_ms}, p_accept={:.6}, err={e}",
                    gaertner_params.estimated_acceptance_probability()
                );
            }
        }
    }
}
