#![allow(missing_docs)]

use hachi_pcs::algebra::ntt::tables::{q128_primes, Q128_NUM_PRIMES};
use hachi_pcs::algebra::fields::fft::{
    field_pow, primitive_root_of_unity, rs_extend_fft, SmoothDomain,
};
use hachi_pcs::algebra::{
    CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, Prime128Offset2355,
};
use hachi_pcs::primitives::serialization::{Compress, HachiSerialize, Valid};
use hachi_pcs::protocol::commitment::presets::fp128;
use hachi_pcs::protocol::commitment::CommitmentConfig;
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::DensePoly;
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::{
    BasisMode, BlockOrder, CanonicalField, CommitmentScheme, FieldCore, FieldSampling,
    FromSmallInt, HachiPolyOps, Invertible, PseudoMersenneField, Transcript,
};

use hachi_pcs::algebra::fields::wide::{HasUnreducedOps, HasWide};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::env;
use std::fmt::Debug;
use std::ops::MulAssign;
use std::time::Instant;

const D: usize = 32;
const FFT_NUM_COLS: usize = 256;
const LAGRANGE_NUM_COLS: usize = 256;
type CrtNtt = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_str(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn batch_inv<F: FieldCore + Invertible + MulAssign>(vals: &[F]) -> Vec<F> {
    let n = vals.len();
    if n == 0 {
        return vec![];
    }

    let mut prefix = Vec::with_capacity(n);
    let mut acc = F::one();
    for &v in vals {
        acc *= if v == F::zero() { F::one() } else { v };
        prefix.push(acc);
    }

    let mut inv_acc = prefix[n - 1].inv_or_zero();
    let mut result = vec![F::zero(); n];
    for i in (0..n).rev() {
        if vals[i] == F::zero() {
            result[i] = F::zero();
        } else {
            let prev = if i == 0 { F::one() } else { prefix[i - 1] };
            result[i] = inv_acc * prev;
            inv_acc *= vals[i];
        }
    }
    result
}

fn build_eval_matrix<F: FieldCore + FromSmallInt + Invertible + MulAssign + Send + Sync>(
    k: usize,
    extension_count: usize,
) -> Vec<Vec<F>> {
    let domain: Vec<F> = (0..k).map(|i| F::from_i64(i as i64)).collect();

    let mut denom_inv = Vec::with_capacity(k);
    for i in 0..k {
        let mut d = F::one();
        for j in 0..k {
            if j != i {
                d *= domain[i] - domain[j];
            }
        }
        denom_inv.push(d);
    }
    let denom_inv = batch_inv(&denom_inv);

    let build_row = |j: usize| -> Vec<F> {
        let x = F::from_i64((k + j) as i64);
        let mut full_prod = F::one();
        for d_m in &domain {
            full_prod *= x - *d_m;
        }
        let x_minus_d: Vec<F> = (0..k).map(|i| x - domain[i]).collect();
        let x_minus_d_inv = batch_inv(&x_minus_d);
        (0..k)
            .map(|i| full_prod * x_minus_d_inv[i] * denom_inv[i])
            .collect()
    };

    #[cfg(feature = "parallel")]
    {
        (0..extension_count)
            .into_par_iter()
            .map(build_row)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..extension_count).map(build_row).collect()
    }
}

fn build_eval_ntt_matrix<F: FieldCore + CanonicalField + Send + Sync>(
    eval_matrix: &[Vec<F>],
    params: &CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
) -> Vec<Vec<CrtNtt>> {
    let convert_row = |row: &Vec<F>| -> Vec<CrtNtt> {
        row.iter()
            .map(|s| CrtNtt::from_scalar_with_params(s, params))
            .collect()
    };

    #[cfg(feature = "parallel")]
    {
        eval_matrix.par_iter().map(convert_row).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        eval_matrix.iter().map(convert_row).collect()
    }
}

fn rs_encode_row_naive<F: FieldCore + Send + Sync>(
    row: &[CyclotomicRing<F, D>],
    eval_matrix: &[Vec<F>],
) -> Vec<CyclotomicRing<F, D>> {
    let k = row.len();

    let encode_point = |eval_row: &Vec<F>| -> CyclotomicRing<F, D> {
        let mut val = CyclotomicRing::<F, D>::zero();
        for i in 0..k {
            val += row[i].scale(&eval_row[i]);
        }
        val
    };

    #[cfg(feature = "parallel")]
    {
        eval_matrix.par_iter().map(encode_point).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        eval_matrix.iter().map(encode_point).collect()
    }
}

fn rs_encode_row_ntt<F: FieldCore + CanonicalField + Send + Sync>(
    row: &[CyclotomicRing<F, D>],
    eval_ntt: &[Vec<CrtNtt>],
    params: &CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
) -> Vec<CyclotomicRing<F, D>> {
    #[cfg(feature = "parallel")]
    let row_ntt: Vec<CrtNtt> = row
        .par_iter()
        .map(|r| CrtNtt::from_ring_with_params(r, params))
        .collect();
    #[cfg(not(feature = "parallel"))]
    let row_ntt: Vec<CrtNtt> = row
        .iter()
        .map(|r| CrtNtt::from_ring_with_params(r, params))
        .collect();

    let encode_point = |eval_row: &Vec<CrtNtt>| -> CyclotomicRing<F, D> {
        let mut acc = CrtNtt::zero();
        for (row_elem, coeff_ntt) in row_ntt.iter().zip(eval_row.iter()) {
            acc.add_assign_pointwise_mul_with_params(row_elem, coeff_ntt, params);
        }
        acc.to_ring_with_params(params)
    };

    #[cfg(feature = "parallel")]
    {
        eval_ntt.par_iter().map(encode_point).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        eval_ntt.iter().map(encode_point).collect()
    }
}

fn smooth_fft_k(smooth_subgroup_order: usize, num_cols: usize, rate_factor: usize) -> usize {
    let max_k = smooth_subgroup_order / rate_factor;
    assert!(
        num_cols <= max_k,
        "num_cols={num_cols} exceeds max FFT domain {max_k} \
         (smooth_order={smooth_subgroup_order} / rate={rate_factor})"
    );
    let mut best = max_k;
    for d in 1..=max_k {
        if max_k % d == 0 && d >= num_cols && d < best {
            best = d;
        }
    }
    best
}

fn rs_encode_row_fft<F: FieldCore + FromSmallInt + Invertible + Debug>(
    row: &[CyclotomicRing<F, D>],
    fft_k: usize,
    domain_k: &SmoothDomain<F>,
    omega_n: F,
    blowup: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let actual_k = row.len();
    let pad_count = fft_k - actual_k;
    let coset_extension_count = fft_k * (blowup - 1);
    let total_ext = pad_count + coset_extension_count;
    let mut extension = vec![CyclotomicRing::<F, D>::zero(); total_ext];

    for d in 0..D {
        let mut evals: Vec<F> = (0..actual_k).map(|i| row[i].coefficients()[d]).collect();
        evals.resize(fft_k, F::zero());
        let ext = rs_extend_fft(&evals, domain_k, omega_n, blowup);
        for (i, &val) in ext.iter().enumerate() {
            extension[pad_count + i].coefficients_mut()[d] = val;
        }
    }

    extension
}

fn opening_from_poly<F: FieldCore + CanonicalField>(
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

/// Field-specific parameters for the RS-encode example.
struct FieldParams {
    prime_label: &'static str,
    smooth_subgroup_order: usize,
    p_minus_1: u128,
    generator: u128,
}

const NTT_PARAMS: FieldParams = FieldParams {
    prime_label: "2^128 − 2355",
    smooth_subgroup_order: 14700,
    p_minus_1: 0xfffffffffffffffffffffffffffff6cc,
    generator: 2,
};

fn run_rs_encode<F, Cfg>(params: &FieldParams)
where
    F: FieldCore
        + CanonicalField
        + FromSmallInt
        + Invertible
        + FieldSampling
        + PseudoMersenneField
        + HasWide
        + HasUnreducedOps
        + Valid
        + Debug
        + MulAssign
        + Send
        + Sync
        + 'static,
    Cfg: CommitmentConfig<Field = F>,
{
    let rate_factor = env_usize("RS_RATE", 4);
    let rs_mode = env_str("RS_MODE", "lagrange");
    let data_mb = env_usize("RS_DATA_MB", 128);

    let use_fft = rs_mode == "fft";
    let use_ntt = rs_mode == "ntt";
    let default_cols = if use_fft {
        FFT_NUM_COLS
    } else {
        LAGRANGE_NUM_COLS
    };
    let num_cols = env_usize("RS_COLS", default_cols);

    let fft_k = if use_fft {
        smooth_fft_k(params.smooth_subgroup_order, num_cols, rate_factor)
    } else {
        0
    };

    let data_bytes: usize = data_mb * 1024 * 1024;
    let field_elem_bytes = 16;
    let num_field_elems = data_bytes / field_elem_bytes;
    let num_vars = (num_field_elems as f64).log2().floor() as usize;
    let num_field_elems = 1usize << num_vars;
    let actual_bytes = num_field_elems * field_elem_bytes;

    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!(
        "  RS-encode + Hachi PCS example  (p = {})",
        params.prime_label
    );
    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!(
        "  Data size         : {} MB ({} field elements)",
        actual_bytes / (1024 * 1024),
        num_field_elems
    );
    eprintln!("  num_vars          : {num_vars}");
    eprintln!("  Ring degree D     : {D}");
    eprintln!("  RS rate factor    : {rate_factor}x");
    eprintln!("  RS columns (k)    : {num_cols}");
    eprintln!(
        "  Smooth subgroup   : order {}",
        params.smooth_subgroup_order
    );
    eprintln!(
        "  RS backend        : {}",
        match rs_mode.as_str() {
            "fft" => "smooth FFT (mixed-radix, coset evaluation)",
            "ntt" => "Lagrange + CRT+NTT ring multiply",
            _ => "Lagrange + naive (schoolbook)",
        }
    );
    if use_fft {
        let n = fft_k * rate_factor;
        if fft_k == num_cols {
            eprintln!(
                "  FFT domain        : k={fft_k} → n={n}  (subgroup of order {})",
                params.smooth_subgroup_order
            );
        } else {
            eprintln!(
                "  FFT domain        : {num_cols} cols → pad to fft_k={fft_k} → n={n}  (subgroup of order {})",
                params.smooth_subgroup_order
            );
        }
    }
    #[cfg(feature = "parallel")]
    eprintln!(
        "  Parallelism       : rayon ({} threads)",
        rayon::current_num_threads()
    );
    #[cfg(not(feature = "parallel"))]
    eprintln!("  Parallelism       : none (sequential)");
    eprintln!("══════════════════════════════════════════════════════════════");

    let t_total = Instant::now();

    let t0 = Instant::now();
    let mut rng = StdRng::from_entropy();
    let evals: Vec<F> = (0..num_field_elems)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    eprintln!(
        "[data-gen]       {:.3}s  — generated {} random field elements",
        t0.elapsed().as_secs_f64(),
        num_field_elems
    );

    let t0 = Instant::now();
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
    let num_ring_elems = poly.coeffs.len();
    eprintln!(
        "[poly-pack]      {:.3}s  — packed into {} ring elements (D={D})",
        t0.elapsed().as_secs_f64(),
        num_ring_elems
    );

    let t0 = Instant::now();
    let num_rows = num_ring_elems.div_ceil(num_cols);
    let padded_total = num_rows * num_cols;
    let mut matrix: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(padded_total);
    matrix.extend_from_slice(&poly.coeffs);
    matrix.resize(padded_total, CyclotomicRing::<F, D>::zero());
    eprintln!(
        "[matrix-layout]  {:.3}s  — arranged into {num_rows} x {num_cols} matrix ({padded_total} ring elems, {} zero-padded)",
        t0.elapsed().as_secs_f64(),
        padded_total - num_ring_elems
    );

    let extension_count = if use_fft {
        fft_k * rate_factor - num_cols
    } else {
        num_cols * (rate_factor - 1)
    };
    let encoded_cols = num_cols + extension_count;

    let eval_matrix;
    let ntt_params;
    let eval_ntt_matrix;
    let fft_domain_k;
    let fft_omega_n;

    if use_fft {
        let t0 = Instant::now();
        let n = fft_k * rate_factor;
        let g =
            F::from_canonical_u128_checked(params.generator).expect("generator must be in field");
        let omega_n = primitive_root_of_unity(g, params.p_minus_1, n);
        let omega_k = field_pow(omega_n, rate_factor as u64);
        let domain_k = SmoothDomain::new(omega_k, fft_k);
        eprintln!(
            "[fft-setup]      {:.3}s  — smooth FFT domain initialized (fft_k={fft_k}, n={n}, ω order verified)",
            t0.elapsed().as_secs_f64()
        );
        fft_domain_k = Some(domain_k);
        fft_omega_n = Some(omega_n);
        eval_matrix = None;
        ntt_params = None;
        eval_ntt_matrix = None;
    } else {
        let t0 = Instant::now();
        let em = build_eval_matrix::<F>(num_cols, extension_count);
        eprintln!(
            "[eval-matrix]    {:.3}s  — built {}x{} Lagrange evaluation matrix",
            t0.elapsed().as_secs_f64(),
            extension_count,
            num_cols,
        );

        if use_ntt {
            let t0 = Instant::now();
            let primes = q128_primes();
            let crt_params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(primes);
            eprintln!(
                "[ntt-setup]      {:.3}s  — CRT+NTT parameters initialized",
                t0.elapsed().as_secs_f64()
            );
            let t0 = Instant::now();
            let enm = build_eval_ntt_matrix(&em, &crt_params);
            eprintln!(
                "[eval-ntt]       {:.3}s  — precomputed {}x{} eval scalars in NTT domain",
                t0.elapsed().as_secs_f64(),
                enm.len(),
                if enm.is_empty() { 0 } else { enm[0].len() }
            );
            ntt_params = Some(crt_params);
            eval_ntt_matrix = Some(enm);
        } else {
            ntt_params = None;
            eval_ntt_matrix = None;
        }

        eval_matrix = Some(em);
        fft_domain_k = None;
        fft_omega_n = None;
    }

    let t0 = Instant::now();

    let encode_one_row = |row_idx: usize| -> Vec<CyclotomicRing<F, D>> {
        let row_start = row_idx * num_cols;
        let row = &matrix[row_start..row_start + num_cols];

        let extension = if use_fft {
            rs_encode_row_fft(
                row,
                fft_k,
                fft_domain_k.as_ref().unwrap(),
                fft_omega_n.unwrap(),
                rate_factor,
            )
        } else if use_ntt {
            rs_encode_row_ntt(
                row,
                eval_ntt_matrix.as_ref().unwrap(),
                ntt_params.as_ref().unwrap(),
            )
        } else {
            rs_encode_row_naive(row, eval_matrix.as_ref().unwrap())
        };

        let mut full_row = Vec::with_capacity(encoded_cols);
        full_row.extend_from_slice(row);
        full_row.extend(extension);
        full_row
    };

    #[cfg(feature = "parallel")]
    let encoded_matrix: Vec<Vec<CyclotomicRing<F, D>>> =
        (0..num_rows).into_par_iter().map(encode_one_row).collect();
    #[cfg(not(feature = "parallel"))]
    let encoded_matrix: Vec<Vec<CyclotomicRing<F, D>>> =
        (0..num_rows).map(encode_one_row).collect();

    let rs_elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "[rs-encode]      {rs_elapsed:.3}s  — RS-encoded all {num_rows} rows ({num_cols} -> {encoded_cols} ring elems/row)"
    );

    assert_eq!(encoded_matrix.len(), num_rows);
    assert_eq!(encoded_matrix[0].len(), encoded_cols);

    let t0 = Instant::now();
    let layout = Cfg::commitment_layout(num_vars).expect("commitment layout");
    let pt: Vec<F> = (0..num_vars)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = opening_from_poly(&poly, &pt, &layout);
    eprintln!(
        "[eval-point]     {:.3}s  — evaluated polynomial at random point",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
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
    eprintln!(
        "[hachi-commit]   {:.3}s  — commitment computed",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"rs_encode");
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
        &setup,
        &poly,
        &pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
    )
    .unwrap();
    let prove_elapsed = t0.elapsed().as_secs_f64();
    let proof_bytes = proof.serialized_size(Compress::No);
    eprintln!("[hachi-prove]    {prove_elapsed:.3}s  — proof generated ({proof_bytes} bytes)");

    let t0 = Instant::now();
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"rs_encode");
    match <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        &pt,
        &opening,
        &commitment,
        BasisMode::Lagrange,
    ) {
        Ok(()) => eprintln!(
            "[hachi-verify]   {:.3}s  — verification OK",
            t0.elapsed().as_secs_f64()
        ),
        Err(e) => eprintln!(
            "[hachi-verify]   {:.3}s  — verification FAILED: {e}",
            t0.elapsed().as_secs_f64()
        ),
    }

    let total_elapsed = t_total.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!("  TIMING SUMMARY");
    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!("  RS encode ({num_rows} rows, {num_cols}->{encoded_cols}): {rs_elapsed:.3}s");
    eprintln!(
        "  RS backend: {}",
        match rs_mode.as_str() {
            "fft" => "smooth FFT",
            "ntt" => "Lagrange + CRT+NTT",
            _ => "Lagrange + naive",
        }
    );
    eprintln!("  Hachi prove:  {prove_elapsed:.3}s");
    eprintln!(
        "  Hachi proof size: {proof_bytes} bytes ({:.1} KB)",
        proof_bytes as f64 / 1024.0
    );
    eprintln!("  Total wall time:  {total_elapsed:.3}s");
    eprintln!("══════════════════════════════════════════════════════════════");
}

fn main() {
    if cfg!(debug_assertions) && env::var("HACHI_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        eprintln!("examples/rs_encode must be run with --release for meaningful timings.");
        eprintln!("Re-run with: cargo run --release --example rs_encode");
        eprintln!("Set HACHI_ALLOW_DEBUG_PROFILE=1 to override this guard.");
        std::process::exit(2);
    }

    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    let field_selector = env_str("HACHI_FIELD", "ntt");

    match field_selector.as_str() {
        "ntt" | "default" | "" => {
            run_rs_encode::<Prime128Offset2355, fp128::D32Full>(&NTT_PARAMS);
        }
        other => {
            eprintln!("Unknown HACHI_FIELD={other} (expected 'ntt')");
            std::process::exit(1);
        }
    }
}
