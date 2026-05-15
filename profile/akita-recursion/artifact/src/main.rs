//! Generate an Akita verifier-input blob to be consumed by the Jolt guest
//! program in `profile/akita-recursion/guest`.
//!
//! Mirrors `run_profile_onehot_d32` from `crates/akita-pcs/examples/profile.rs`:
//! single-poly OneHot polynomial commitment in `D=32` mode at the canonical
//! `q=2^128-2^32+22537` prime, opened at one random point. After running the
//! prover end-to-end we re-run the host verifier as a sanity check, then
//! serialize all verifier-side state into one contiguous blob via
//! [`akita_recursion_glue::AkitaJoltInputs`].
//!
//! Output paths are controlled via `AKITA_RECURSION_BLOB` (defaults to
//! `target/akita_recursion_inputs.bin`). Set `AKITA_NUM_VARS` (default 20)
//! to regenerate at a different polynomial arity. Stick with `D=32 OneHot`
//! so the guest's hard-coded monomorphization can read the blob.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::{CanonicalField, PseudoMersenneField};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{AkitaPolyOps, CommitmentProver, CommittedPolynomials, OneHotPoly};
use akita_recursion_glue::AkitaJoltInputs;
use akita_transcript::Blake2bTranscript;
use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BasisMode, BlockOrder,
    LevelParams,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

type F = fp128::Field;
const D: usize = 32;
type Cfg = fp128::D32OneHot;
const ONEHOT_K: usize = 256;

const TRANSCRIPT_DOMAIN: &[u8] = b"akita-recursion/onehot-d32";

fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
}

fn opening_from_poly<P: AkitaPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &LevelParams,
    basis: BasisMode,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.m_vars + layout.r_vars;
    assert!(
        point.len() <= target_num_vars,
        "opening point length {} exceeds target root arity {}",
        point.len(),
        target_num_vars
    );
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, F::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        basis,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

fn fp128_prime_label() -> String {
    match <F as PseudoMersenneField>::MODULUS_OFFSET {
        2355 => "q=2^128-2355".to_string(),
        0xFFFFA7F7 => "q=2^128-2^32+22537".to_string(),
        offset => format!("q=2^128-{offset:#x}"),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    if cfg!(debug_assertions) && env::var("AKITA_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        eprintln!("akita-recursion-artifact must be run with --release for sane runtimes.");
        eprintln!("Re-run with: cargo run --release -p akita-recursion-artifact");
        eprintln!("Set AKITA_ALLOW_DEBUG_PROFILE=1 to override this guard.");
        std::process::exit(2);
    }

    let log_filter =
        EnvFilter::try_new(env::var("AKITA_RECURSION_LOG").unwrap_or_else(|_| "info".to_string()))
            .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(log_filter)
        .with_target(false)
        .try_init();

    let nv: usize = env_usize("AKITA_NUM_VARS", 20);
    let onehot_k = onehot_k_for_num_vars(nv);
    let output_path = PathBuf::from(
        env::var("AKITA_RECURSION_BLOB")
            .unwrap_or_else(|_| "target/akita_recursion_inputs.bin".to_string()),
    );

    let prime = fp128_prime_label();
    tracing::info!(
        nv,
        d = D,
        onehot_k,
        prime = %prime,
        "generating Akita verifier-input artifact (single-poly OneHot, D=32)"
    );

    let layout: LevelParams = <Cfg as CommitmentConfig>::commitment_layout(nv).expect("layout");
    let alpha_bits = D.trailing_zeros() as usize;
    let required_vars = layout.m_vars + layout.r_vars + alpha_bits;
    // Both `main` (`required_vars <= nv`, layout fits in nv) and
    // `opening_from_poly` (`point.len() <= target_num_vars`, i.e.
    // `nv <= required_vars`) need to hold simultaneously, which means
    // they need to be equal. Catch the mismatch here with a clearer
    // message than the helper would emit.
    assert_eq!(
        required_vars, nv,
        "OneHot D={D} layout at nv={nv} expects exactly {required_vars} variables \
         (alpha_bits={alpha_bits} + m_vars={} + r_vars={}); pick an AKITA_NUM_VARS that matches the layout",
        layout.m_vars, layout.r_vars,
    );

    // The example reuses the deterministic seed from `examples/profile.rs`
    // for reproducibility.
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(D)
        .expect("total field size overflow");
    let total_chunks = total_field / onehot_k;
    assert_eq!(
        total_chunks * onehot_k,
        total_field,
        "OneHot K must divide total field size for nv={nv}"
    );

    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    let onehot_poly = OneHotPoly::<F, D, u8>::new(onehot_k, indices).unwrap();
    let opening_point: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = opening_from_poly(&onehot_poly, &opening_point, &layout, BasisMode::Lagrange);

    let t0 = Instant::now();
    let prover_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "prover setup complete"
    );

    let t0 = Instant::now();
    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&&onehot_poly),
        &prover_setup,
    )
    .unwrap();
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "commit complete");

    let poly_refs: [&OneHotPoly<F, D, u8>; 1] = [&onehot_poly];
    let openings = [opening];

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &prover_setup,
        vec![(
            &opening_point[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitment,
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched_prove");
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "prove complete");

    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&prover_setup);

    // Sanity check: the proof should verify with the same domain label.
    let t0 = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let opening_groups = [&openings[..]];
    <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        VerifierClaims {
            commitment: &commitment,
            points: vec![PointClaim::all(&opening_point[..], opening_groups[0])],
        },
        BasisMode::Lagrange,
    )
    .expect("host-side sanity verify");
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "host-side verify OK"
    );

    let proof_shape = proof.shape();
    let inputs: AkitaJoltInputs<F, D> = AkitaJoltInputs {
        transcript_domain: TRANSCRIPT_DOMAIN.to_vec(),
        num_vars: nv as u64,
        opening_point,
        opening,
        commitment,
        verifier_setup,
        proof_shape,
        proof,
    };

    let blob = inputs.write_to_bytes().expect("encode jolt inputs blob");

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    fs::write(&output_path, &blob).expect("write blob");

    let blob_kib = (blob.len() as f64) / 1024.0;
    let blob_mib = blob_kib / 1024.0;
    tracing::info!(
        nv,
        d = D,
        bytes = blob.len(),
        kib = blob_kib,
        mib = blob_mib,
        path = %output_path.display(),
        "wrote akita-recursion verifier-input blob"
    );
    eprintln!(
        "wrote {} bytes ({:.2} MiB) to {}",
        blob.len(),
        blob_mib,
        output_path.display()
    );

    // Round-trip immediately so a buggy encoding fails on the host instead of
    // inside the Jolt emulator.
    let decoded = AkitaJoltInputs::<F, D>::read_from_bytes(&blob)
        .expect("decode jolt inputs blob (round-trip)");
    let mut roundtrip_transcript = Blake2bTranscript::<F>::new(&decoded.transcript_domain);
    let openings_rt = [decoded.opening];
    let opening_groups_rt = [&openings_rt[..]];
    <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &decoded.proof,
        &decoded.verifier_setup,
        &mut roundtrip_transcript,
        VerifierClaims {
            commitment: &decoded.commitment,
            points: vec![PointClaim::all(&decoded.opening_point[..], opening_groups_rt[0])],
        },
        BasisMode::Lagrange,
    )
    .expect("decoded blob verify");
    tracing::info!("decoded-blob verify OK");
}
