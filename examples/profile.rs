#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{
    DecompositionParams, HachiCommitmentLayout, ProductionFp128CommitmentConfig,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::DensePoly;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{BasisMode, CommitmentScheme, FromSmallInt, Transcript};
use std::time::Instant;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;

const D: usize = ProductionFp128CommitmentConfig::D;

#[derive(Clone, Copy, Debug)]
struct ProfileCfg;
impl CommitmentConfig for ProfileCfg {
    const D: usize = D;
    const N_A: usize = ProductionFp128CommitmentConfig::N_A;
    const N_B: usize = ProductionFp128CommitmentConfig::N_B;
    const N_D: usize = ProductionFp128CommitmentConfig::N_D;
    const CHALLENGE_WEIGHT: usize = ProductionFp128CommitmentConfig::CHALLENGE_WEIGHT;

    fn decomposition() -> DecompositionParams {
        ProductionFp128CommitmentConfig::decomposition()
    }

    fn commitment_layout(
        _max_num_vars: usize,
    ) -> Result<HachiCommitmentLayout, hachi_pcs::error::HachiError> {
        HachiCommitmentLayout::new::<Self>(6, 4, &Self::decomposition())
    }
}

type Scheme = HachiCommitmentScheme<D, ProfileCfg>;

fn main() {
    let trace_dir = "profile_traces";
    std::fs::create_dir_all(trace_dir).ok();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let trace_file = format!("{trace_dir}/hachi_{timestamp}.json");

    let (chrome_layer, _guard) = ChromeLayerBuilder::new()
        .include_args(true)
        .file(&trace_file)
        .build();

    tracing_subscriber::registry().with(chrome_layer).init();

    eprintln!("Perfetto trace will be written to: {trace_file}");
    eprintln!("Open at https://ui.perfetto.dev/");

    let nv = {
        let alpha = D.trailing_zeros() as usize;
        let layout = ProfileCfg::commitment_layout(0).expect("layout");
        layout.m_vars + layout.r_vars + alpha
    };
    eprintln!("num_vars = {nv}");

    let len = 1usize << nv;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
    let pt: Vec<F> = (0..nv).map(|i| F::from_u64((i + 2) as u64)).collect();

    let opening = {
        let mut weights = vec![F::from_u64(0); len];
        weights[0] = F::from_u64(1);
        for (k, &x) in pt.iter().enumerate() {
            let half = 1usize << k;
            for i in (0..half).rev() {
                weights[i + half] = weights[i] * x;
                weights[i] = weights[i] - weights[i + half];
            }
        }
        evals
            .iter()
            .zip(weights.iter())
            .fold(F::from_u64(0), |a, (&e, &w)| a + e * w)
    };

    let t0 = Instant::now();
    let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(nv);
    eprintln!("setup: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let (commitment, hint) = <Scheme as CommitmentScheme<F, D>>::commit(&poly, &setup).unwrap();
    eprintln!("commit: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"profile");
    let proof = <Scheme as CommitmentScheme<F, D>>::prove(
        &setup,
        &poly,
        &pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
    )
    .unwrap();
    eprintln!("prove: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"profile");
    <Scheme as CommitmentScheme<F, D>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        &pt,
        &opening,
        &commitment,
        BasisMode::Lagrange,
    )
    .unwrap();
    eprintln!("verify: {:.3}s", t0.elapsed().as_secs_f64());

    eprintln!("Done. Trace saved to {trace_file}");
}
