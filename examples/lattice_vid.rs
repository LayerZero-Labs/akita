#![allow(missing_docs)]
//! Lattice VID protocol implementation following the Hachi-PCS DA proposal.
//!
//! The data matrix contains **field elements** (not ring elements).
//! RS encoding, column shares, and eq-folding all operate on field elements.
//! Ring packing is only used internally by the Hachi PCS commit / prove / verify.
//!
//! Layout (128 MB, K=256, N=1024):
//!   - 8 388 608 field elements → 32 768 rows × 256 columns
//!   - RS-encode each row via smooth-subgroup FFT → 1200 evaluation points
//!   - Assign first 1024 evaluation points to 1024 validators
//!   - Each validator receives a column of 32 768 field elements (512 KB)
//!
//! Run:
//!   cargo run --release --example lattice_vid

use hachi_pcs::algebra::fields::fft::{
    field_pow, primitive_root_of_unity, rs_extend_fft, SmoothDomain,
};
use hachi_pcs::algebra::Prime128Offset2355;
use hachi_pcs::primitives::serialization::{Compress, HachiSerialize};
use hachi_pcs::protocol::commitment::presets::fp128;
use hachi_pcs::protocol::commitment::CommitmentConfig;
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::DensePoly;
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::{
    BasisMode, BlockOrder, CanonicalField, CommitmentScheme, FieldCore, HachiPolyOps, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::env;
use std::time::Instant;

type F = Prime128Offset2355;
const D: usize = 32;
type Cfg = fp128::D32Full;

const DATA_MB: usize = 128;
const NUM_COLS: usize = 256;
const NUM_VALIDATORS: usize = 1024;
const RATE_FACTOR: usize = 4;

const SMOOTH_SUBGROUP_ORDER: usize = 14700;
const P_HEX: u128 = 0xfffffffffffffffffffffffffffff6cd;
const P_MINUS_1: u128 = P_HEX - 1;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the smallest divisor of `smooth_order / rate` that is >= `num_cols`.
fn smooth_fft_k(num_cols: usize, rate_factor: usize) -> usize {
    let max_k = SMOOTH_SUBGROUP_ORDER / rate_factor;
    assert!(
        num_cols <= max_k,
        "num_cols={num_cols} exceeds max FFT domain {max_k}"
    );
    let mut best = max_k;
    for d in 1..=max_k {
        if max_k % d == 0 && d >= num_cols && d < best {
            best = d;
        }
    }
    best
}

/// RS-encode one row of **field elements** using smooth-subgroup FFT.
/// Returns only the extension (padding zeros + coset evaluations).
fn rs_encode_row_field(
    row: &[F],
    fft_k: usize,
    domain_k: &SmoothDomain<F>,
    omega_n: F,
    blowup: usize,
) -> Vec<F> {
    let actual_k = row.len();
    let mut padded = row.to_vec();
    padded.resize(fft_k, F::zero());
    let coset_ext = rs_extend_fft(&padded, domain_k, omega_n, blowup);
    let mut extension = vec![F::zero(); fft_k - actual_k];
    extension.extend_from_slice(&coset_ext);
    extension
}

/// Fold a column of **field elements** with an eq-polynomial challenge vector.
///
/// Computes `Σ_r eq(r, challenge) · vals[r]` by iterative halving.
fn fold_field_column(vals: &[F], challenge: &[F]) -> F {
    debug_assert_eq!(vals.len(), 1 << challenge.len());
    let mut buf = vals.to_vec();
    for &r in challenge.iter() {
        let half = buf.len() / 2;
        for i in 0..half {
            buf[i] = buf[2 * i] + r * (buf[2 * i + 1] - buf[2 * i]);
        }
        buf.truncate(half);
    }
    buf[0]
}

/// Compute the expected PCS opening value for a given polynomial and point.
fn opening_from_poly(
    poly: &DensePoly<F, D>,
    point: &[F],
    layout: &hachi_pcs::protocol::commitment::HachiCommitmentLayout,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.m_vars + layout.r_vars;
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, F::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, BasisMode::Lagrange)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

// ---------------------------------------------------------------------------
// Main VID protocol
// ---------------------------------------------------------------------------

fn main() {
    if cfg!(debug_assertions) && env::var("HACHI_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        eprintln!("examples/lattice_vid must be run with --release for meaningful timings.");
        eprintln!("Re-run with: cargo run --release --example lattice_vid");
        eprintln!("Set HACHI_ALLOW_DEBUG_PROFILE=1 to override this guard.");
        std::process::exit(2);
    }

    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    // ─── Protocol parameters ───────────────────────────────────────────
    let data_bytes: usize = DATA_MB * 1024 * 1024;
    let field_elem_bytes = 16;
    let num_field_elems = data_bytes / field_elem_bytes;
    let num_vars = num_field_elems.trailing_zeros() as usize;
    let num_rows = num_field_elems / NUM_COLS;
    let num_row_vars = num_rows.trailing_zeros() as usize;

    let fft_k = smooth_fft_k(NUM_COLS, RATE_FACTOR);
    let fft_n = fft_k * RATE_FACTOR;
    let encoded_cols = fft_n;

    assert_eq!(num_field_elems, 1 << num_vars);
    assert!(num_rows.is_power_of_two());
    assert!(
        NUM_VALIDATORS <= encoded_cols,
        "need at most {encoded_cols} validators (FFT domain), got {NUM_VALIDATORS}"
    );

    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!("  LATTICE VID PROTOCOL (Hachi-PCS DA, common opening point)");
    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!("  Data size         : {DATA_MB} MB  ({num_field_elems} field elements)");
    eprintln!("  Ring degree D     : {D}  (internal to PCS only)");
    eprintln!("  num_vars          : {num_vars}  (= log₂ field elements)");
    eprintln!("  Matrix shape      : {num_rows} rows × {NUM_COLS} cols  (field elements)");
    eprintln!("  Row selector vars : {num_row_vars}  (ℓ_r = log₂ rows)");
    eprintln!(
        "  RS (smooth FFT)   : {RATE_FACTOR}×  (cols {NUM_COLS} → pad {fft_k} → FFT {fft_n})"
    );
    eprintln!("  Validators (N)    : {NUM_VALIDATORS}  (of {encoded_cols} FFT evaluation points)");
    #[cfg(feature = "parallel")]
    eprintln!(
        "  Parallelism       : rayon ({} threads)",
        rayon::current_num_threads()
    );
    #[cfg(not(feature = "parallel"))]
    eprintln!("  Parallelism       : none (sequential)");
    eprintln!("══════════════════════════════════════════════════════════════\n");

    let t_total = Instant::now();

    // ─── Phase 1 · Data generation ─────────────────────────────────────
    let t0 = Instant::now();
    let mut rng = StdRng::from_entropy();
    let field_data: Vec<F> = (0..num_field_elems)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    eprintln!(
        "[data-gen]       {:.3}s  — {} random field elements",
        t0.elapsed().as_secs_f64(),
        num_field_elems
    );

    // Pack into DensePoly for Hachi PCS (ring-packed internally).
    // The field_data array stays around as our field-element matrix.
    let t0 = Instant::now();
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &field_data).unwrap();
    eprintln!(
        "[poly-pack]      {:.3}s  — packed into {} ring elements for PCS (D={D})",
        t0.elapsed().as_secs_f64(),
        poly.coeffs.len()
    );

    // ─── Phase 2 · FFT setup + RS-encode every row (field elements) ────
    let t0 = Instant::now();
    let g = F::from_canonical_u128(2);
    let omega_n = primitive_root_of_unity(g, P_MINUS_1, fft_n);
    let omega_k = field_pow(omega_n, RATE_FACTOR as u64);
    let domain_k = SmoothDomain::new(omega_k, fft_k);
    eprintln!(
        "[fft-setup]      {:.3}s  — smooth FFT domain (fft_k={fft_k}, n={fft_n})",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();

    let encode_one_row = |row_idx: usize| -> Vec<F> {
        let row_start = row_idx * NUM_COLS;
        let row = &field_data[row_start..row_start + NUM_COLS];
        let extension = rs_encode_row_field(row, fft_k, &domain_k, omega_n, RATE_FACTOR);
        let mut full_row = Vec::with_capacity(encoded_cols);
        full_row.extend_from_slice(row);
        full_row.extend(extension);
        full_row.truncate(NUM_VALIDATORS);
        full_row
    };

    #[cfg(feature = "parallel")]
    let encoded_matrix: Vec<Vec<F>> = (0..num_rows).into_par_iter().map(encode_one_row).collect();
    #[cfg(not(feature = "parallel"))]
    let encoded_matrix: Vec<Vec<F>> = (0..num_rows).map(encode_one_row).collect();

    let rs_elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "[rs-encode]      {rs_elapsed:.3}s  — RS-encoded {num_rows} rows ({NUM_COLS} → {NUM_VALIDATORS} field elems/row)"
    );

    // ─── Phase 3 · Hachi commitment (ring-packed, internal) ────────────
    let t0 = Instant::now();
    let layout = Cfg::commitment_layout(num_vars).expect("commitment layout");
    let setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(num_vars, 1);
    eprintln!(
        "[hachi-setup]    {:.3}s  — prover setup",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .unwrap();
    let commit_bytes = commitment.serialized_size(Compress::No);
    eprintln!(
        "[hachi-commit]   {:.3}s  — commitment ({} bytes, {:.1} KB)",
        t0.elapsed().as_secs_f64(),
        commit_bytes,
        commit_bytes as f64 / 1024.0
    );

    // ─── Phase 4 · Fiat-Shamir challenge derivation ────────────────────
    let t0 = Instant::now();
    let mut challenge_transcript = Blake2bTranscript::<F>::new(b"lattice_vid");
    challenge_transcript.append_serde(b"vid/commitment", &commitment);

    let row_challenge: Vec<F> = (0..num_row_vars)
        .map(|_| challenge_transcript.challenge_scalar(b"vid/row_challenge"))
        .collect();

    let opening_point: Vec<F> = (0..num_vars)
        .map(|_| challenge_transcript.challenge_scalar(b"vid/opening_point"))
        .collect();
    eprintln!(
        "[challenge]      {:.3}s  — row challenge ({} elems) + opening point ({} elems)",
        t0.elapsed().as_secs_f64(),
        row_challenge.len(),
        opening_point.len()
    );

    // ─── Phase 5 · Single opening proof (ring-packed, internal) ────────
    let t0 = Instant::now();
    let expected_opening = opening_from_poly(&poly, &opening_point, &layout);
    eprintln!(
        "[eval-point]     {:.3}s  — evaluated polynomial at opening point",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"lattice_vid_pcs");
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
        &setup,
        &poly,
        &opening_point,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
    )
    .unwrap();
    let prove_elapsed = t0.elapsed().as_secs_f64();
    let proof_bytes = proof.serialized_size(Compress::No);
    eprintln!(
        "[hachi-prove]    {prove_elapsed:.3}s  — ONE proof for all {NUM_VALIDATORS} validators ({proof_bytes} B, {:.1} KB)",
        proof_bytes as f64 / 1024.0
    );

    // ─── Phase 6 · PCS verification (same proof for every validator) ───
    let t0 = Instant::now();
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"lattice_vid_pcs");
    match <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        &opening_point,
        &expected_opening,
        &commitment,
        BasisMode::Lagrange,
    ) {
        Ok(()) => eprintln!(
            "[hachi-verify]   {:.3}s  — PCS verification OK  (shared by all {NUM_VALIDATORS} validators)",
            t0.elapsed().as_secs_f64(),
        ),
        Err(e) => {
            eprintln!(
                "[hachi-verify]   {:.3}s  — PCS verification FAILED: {e}",
                t0.elapsed().as_secs_f64()
            );
            std::process::exit(1);
        }
    }

    // ─── Phase 7 · Eq-fold validator column shares (field elements) ────
    let t0 = Instant::now();

    let fold_validator = |v: usize| -> F {
        let column: Vec<F> = (0..num_rows).map(|r| encoded_matrix[r][v]).collect();
        fold_field_column(&column, &row_challenge)
    };

    #[cfg(feature = "parallel")]
    let folded_codeword: Vec<F> = (0..NUM_VALIDATORS)
        .into_par_iter()
        .map(fold_validator)
        .collect();
    #[cfg(not(feature = "parallel"))]
    let folded_codeword: Vec<F> = (0..NUM_VALIDATORS).map(fold_validator).collect();

    eprintln!(
        "[eq-fold]        {:.3}s  — folded column shares for all {NUM_VALIDATORS} validators",
        t0.elapsed().as_secs_f64(),
    );

    // ─── Phase 8 · RS codeword consistency check (field elements) ──────
    let t0 = Instant::now();

    let fold_original_col = |j: usize| -> F {
        let column: Vec<F> = (0..num_rows)
            .map(|r| field_data[r * NUM_COLS + j])
            .collect();
        fold_field_column(&column, &row_challenge)
    };

    #[cfg(feature = "parallel")]
    let folded_message_row: Vec<F> = (0..NUM_COLS)
        .into_par_iter()
        .map(fold_original_col)
        .collect();
    #[cfg(not(feature = "parallel"))]
    let folded_message_row: Vec<F> = (0..NUM_COLS).map(fold_original_col).collect();

    let folded_extension =
        rs_encode_row_field(&folded_message_row, fft_k, &domain_k, omega_n, RATE_FACTOR);
    let mut expected_codeword = Vec::with_capacity(encoded_cols);
    expected_codeword.extend_from_slice(&folded_message_row);
    expected_codeword.extend_from_slice(&folded_extension);

    let mut mismatches = 0usize;
    for i in 0..NUM_VALIDATORS {
        if folded_codeword[i] != expected_codeword[i] {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("  ✗ validator {i}: folded share ≠ expected codeword entry");
            }
        }
    }
    if mismatches == 0 {
        eprintln!(
            "[rs-check]       {:.3}s  — folded codeword consistency PASSED ({NUM_VALIDATORS} validators)",
            t0.elapsed().as_secs_f64(),
        );
    } else {
        eprintln!(
            "[rs-check]       {:.3}s  — FAILED: {mismatches}/{NUM_VALIDATORS} mismatches",
            t0.elapsed().as_secs_f64()
        );
        std::process::exit(1);
    }

    // ─── Summary ───────────────────────────────────────────────────────
    let share_bytes = num_rows * field_elem_bytes;
    let total_elapsed = t_total.elapsed().as_secs_f64();

    eprintln!();
    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!("  VID PROTOCOL SUMMARY");
    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!("  Data              : {DATA_MB} MB  ({num_field_elems} field elements, num_vars={num_vars})");
    eprintln!("  Matrix            : {num_rows} rows × {NUM_COLS} cols of field elements");
    eprintln!("  RS encoding       : {NUM_COLS} → {encoded_cols} (FFT), validators use first {NUM_VALIDATORS}");
    eprintln!("  Validators        : {NUM_VALIDATORS}");
    eprintln!(
        "  Column share size : {} bytes ({:.1} KB per validator)  [{num_rows} field elements]",
        share_bytes,
        share_bytes as f64 / 1024.0
    );
    eprintln!(
        "  Commitment size   : {} bytes ({:.1} KB)",
        commit_bytes,
        commit_bytes as f64 / 1024.0
    );
    eprintln!(
        "  Proof size        : {} bytes ({:.1} KB)  ← ONE proof for ALL validators",
        proof_bytes,
        proof_bytes as f64 / 1024.0
    );
    eprintln!(
        "  Total per-validator overhead : {:.1} KB  (share + amortised proof)",
        share_bytes as f64 / 1024.0 + proof_bytes as f64 / (1024.0 * NUM_VALIDATORS as f64)
    );
    eprintln!("  ─────────────────────────────────────────────────");
    eprintln!("  RS encoding       : {rs_elapsed:.3}s  (smooth FFT, field-element rows)");
    eprintln!("  Hachi prove       : {prove_elapsed:.3}s");
    eprintln!("  Total wall time   : {total_elapsed:.3}s");
    eprintln!("══════════════════════════════════════════════════════════════");
}
