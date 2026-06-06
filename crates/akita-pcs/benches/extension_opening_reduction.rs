#![allow(missing_docs)]

use akita_config::proof_optimized::{fp32, fp64};
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{CanonicalBytes, CanonicalField, ExtField, TranscriptChallenge};
use akita_prover::protocol::extension_opening_reduction::{
    ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm, SparseExtensionOpeningWitness,
};
use akita_sumcheck::SumcheckInstanceProverExt;
use akita_transcript::{labels, sample_ext_challenge, AkitaTranscript, Transcript};
use akita_types::tensor_opening_split;
use criterion::measurement::WallTime;
use criterion::{criterion_group, BenchmarkGroup, Criterion, SamplingMode};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};

const DEFAULT_NUM_VARS: usize = 26;
const DEFAULT_NUM_POLYS: usize = 4;
const ONEHOT_K: usize = 256;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.nresamples(1001);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(10));
}

fn random_ext<F, E>(rng: &mut StdRng) -> E
where
    F: CanonicalField + CanonicalBytes + TranscriptChallenge,
    E: ExtField<F>,
{
    let coeffs = (0..E::EXT_DEGREE)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect::<Vec<_>>();
    E::from_base_slice(&coeffs)
}

fn onehot_sparse_tensor_witness<F, E>(
    num_vars: usize,
    num_polys: usize,
) -> SparseExtensionOpeningWitness<E>
where
    F: CanonicalField + CanonicalBytes + TranscriptChallenge,
    E: ExtField<F>,
{
    let (split_bits, width) = tensor_opening_split::<F, E>().unwrap();
    assert!(num_vars >= split_bits);
    assert!(ONEHOT_K >= width && ONEHOT_K.is_multiple_of(width));
    assert_eq!(
        1usize << num_vars,
        ((1usize << num_vars) / ONEHOT_K) * ONEHOT_K
    );

    let table_len = 1usize << (num_vars - split_bits);
    let total_chunks = (1usize << num_vars) / ONEHOT_K;
    let tails_per_chunk = ONEHOT_K / width;
    let mut rng = StdRng::seed_from_u64(0x6f70_656e_7265_6475);
    let coeffs = (0..num_polys)
        .map(|_| random_ext::<F, E>(&mut rng))
        .collect::<Vec<_>>();
    let basis = (0..width)
        .map(|head| {
            let mut coords = vec![F::zero(); width];
            coords[head] = F::one();
            E::from_base_slice(&coords)
        })
        .collect::<Vec<_>>();

    let mut entries = Vec::with_capacity(total_chunks * num_polys);
    let mut local = Vec::with_capacity(num_polys);
    for chunk_idx in 0..total_chunks {
        local.clear();
        for &coeff in &coeffs {
            let raw = rng.gen_range(0..ONEHOT_K);
            let head = raw % width;
            let local_tail = raw / width;
            let value = basis[head] * coeff;
            local.push((local_tail, value));
        }
        local.sort_unstable_by_key(|(local_tail, _)| *local_tail);
        for &(local_tail, value) in &local {
            let tail = chunk_idx * tails_per_chunk + local_tail;
            if let Some((last_tail, last_value)) = entries.last_mut() {
                if *last_tail == tail {
                    *last_value += value;
                    if *last_value == E::zero() {
                        entries.pop();
                    }
                    continue;
                }
            }
            if value != E::zero() {
                entries.push((tail, value));
            }
        }
    }
    SparseExtensionOpeningWitness::from_sorted_unique_entries(table_len, entries).unwrap()
}

fn sparse_tensor_term<F, E>(num_vars: usize, num_polys: usize) -> ExtensionOpeningReductionTerm<E>
where
    F: CanonicalField + CanonicalBytes + TranscriptChallenge,
    E: ExtField<F>,
{
    let (split_bits, _) = tensor_opening_split::<F, E>().unwrap();
    let tail_vars = num_vars - split_bits;
    let mut rng = StdRng::seed_from_u64(0x7465_6e73_6f72_7265);
    let tail_point = (0..tail_vars)
        .map(|_| random_ext::<F, E>(&mut rng))
        .collect::<Vec<_>>();
    let eta = (0..split_bits)
        .map(|_| random_ext::<F, E>(&mut rng))
        .collect::<Vec<_>>();
    let witness = onehot_sparse_tensor_witness::<F, E>(num_vars, num_polys);
    let lazy_rounds = tail_vars.min(
        akita_prover::protocol::extension_opening_reduction::SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS,
    );
    ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<F>(
        witness,
        tail_point,
        eta,
        E::one(),
        lazy_rounds,
    )
    .unwrap()
}

fn bench_case<F, E>(c: &mut Criterion, label: &str)
where
    F: CanonicalField + CanonicalBytes + TranscriptChallenge,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + akita_serialization::AkitaSerialize,
{
    let num_vars = env_usize("AKITA_EOR_NUM_VARS", DEFAULT_NUM_VARS);
    let num_polys = env_usize("AKITA_EOR_NUM_POLYS", DEFAULT_NUM_POLYS);
    let term = sparse_tensor_term::<F, E>(num_vars, num_polys);
    let terms = vec![term];
    let input_claim = ExtensionOpeningReductionProver::input_claim_from_terms(&terms).unwrap();

    let mut group = c.benchmark_group(format!(
        "extension_opening_reduction/{label}/onehot_nv{num_vars}_np{num_polys}"
    ));
    configure_group(&mut group);
    group.bench_function("prove_sumcheck", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut prover =
                    ExtensionOpeningReductionProver::new(terms.clone(), input_claim).unwrap();
                let mut transcript = <AkitaTranscript<F> as Transcript<F>>::new(b"bench/eor");
                let start = Instant::now();
                let proof = prover
                    .prove::<F, _, _>(&mut transcript, |transcript| {
                        sample_ext_challenge::<F, E, _>(
                            transcript,
                            labels::CHALLENGE_SUMCHECK_ROUND,
                        )
                    })
                    .unwrap();
                total += start.elapsed();
                black_box(proof);
            }
            total
        })
    });
    group.finish();
}

fn bench_extension_opening_reduction(c: &mut Criterion) {
    bench_case::<fp32::Field, fp32::ExtensionField>(c, "fp32_d64");
    bench_case::<fp64::Field, fp64::ExtensionField>(c, "fp64_d32");
}

criterion_group! {
    name = extension_opening_reduction;
    config = Criterion::default()
        .without_plots()
        .nresamples(1001);
    targets = bench_extension_opening_reduction
}

fn main() {
    extension_opening_reduction();
    Criterion::default()
        .without_plots()
        .configure_from_args()
        .final_summary();
}
