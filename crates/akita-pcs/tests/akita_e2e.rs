#![allow(missing_docs)]

use akita_prover::{ComputeBackendSetup, CpuBackend};

use akita_config::proof_optimized::fp128;
use akita_config::proof_optimized::{fp32, fp64};
use akita_config::test_support::akita_batched_root_layout;
use akita_config::CommitmentConfig;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
use akita_field::{
    CanonicalBytes, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, PseudoMersenneField, RandomSampling, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::DensePoly;
use akita_prover::OneHotPoly;
use akita_prover::ProverOpeningData;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{lagrange_weights, CommittedGroupParams, FpExtEncoding};
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaVerifierSetup, BasisMode, Commitment,
    OpeningClaims, PointVariableSelection, PolynomialGroupClaims,
};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
use std::sync::{Mutex, Once};
use std::time::Instant;

mod common;
use common::opening_from_poly;

type F = fp128::Field;
const ONEHOT_K: usize = 256;
const DENSE_TEST_NV: usize = 14;
const ONEHOT_TEST_NV: usize = 15;
const SAME_POINT_ONEHOT_BATCH_SIZE: usize = 4;

fn singleton_layout<Cfg: CommitmentConfig>(num_vars: usize) -> CommittedGroupParams {
    let opening_batch =
        akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
    Cfg::get_params_for_batched_commitment(&opening_batch).expect("singleton commitment layout")
}
const SMALL_FIELD_TEST_NV: usize = 8;
const STACK_SIZE: usize = 256 * 1024 * 1024;

static INIT_RAYON: Once = Once::new();
static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

fn random_point<FField: CanonicalField>(nv: usize) -> Vec<FField> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn random_claim_point<FField, E>(nv: usize) -> Vec<E>
where
    FField: CanonicalField,
    E: ExtField<FField>,
{
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| {
            let limbs = (0..E::EXT_DEGREE)
                .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

fn dense_lagrange_opening_from_evals<FField, E>(evals: &[FField], point: &[E]) -> E
where
    FField: FieldCore,
    E: ExtField<FField>,
{
    let weights = lagrange_weights(point).expect("valid opening point");
    evals
        .iter()
        .zip(weights.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight * E::lift_base(coeff)
        })
}

fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

fn prove_input<'a, FF: FieldCore + Clone, P, CommitF: FieldCore>(
    point: &'a [FF],
    polynomials: &'a [&'a P],
    commitment: &'a Commitment<CommitF>,
    hint: AkitaCommitmentHint<CommitF>,
) -> ProverOpeningData<'a, FF, P, CommitF> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).expect("full-point prover group"),
        vec![FF::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims =
        OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("valid prover claims");
    ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
        .expect("valid prover opening data")
}

fn verify_input<'a, FF: FieldCore, C>(
    point: &[FF],
    openings: &[FF],
    commitment: &'a C,
) -> OpeningClaims<'static, FF, &'a C> {
    OpeningClaims::from_groups(
        point.to_vec(),
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(point.len(), point.len()).expect("full-point group"),
            openings.to_vec(),
            commitment,
        )
        .expect("valid verifier claims group")],
    )
    .expect("valid verifier input")
}

type DenseFixture<FField, E, const D: usize> = (
    AkitaVerifierSetup<FField>,
    Commitment<FField>,
    AkitaBatchedProof<FField, E>,
    Vec<E>,
    E,
    CommittedGroupParams,
);

/// Count the total number of fold levels (including the batched root and the
/// terminal step) in a singleton-shaped batched proof, matching the planner's
/// `num_fold_levels` convention.
fn batched_total_fold_levels<FF: CanonicalField, E: FieldCore>(
    proof: &AkitaBatchedProof<FF, E>,
) -> usize {
    proof.num_fold_levels()
}

fn make_dense_fixture<FField, const D: usize, Cfg: CommitmentConfig<Field = FField>>(
    nv: usize,
    transcript_label: &'static [u8],
) -> DenseFixture<FField, Cfg::ExtField, D>
where
    FField: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + HasWide
        + RandomSampling
        + FromPrimitiveInt
        + 'static
        + HalvingField
        + PseudoMersenneField
        + Valid,
    Cfg::ExtField: FrobeniusExtField<FField> + HasUnreducedOps + HasOptimizedFold,
    <FField as HasWide>::Wide: From<FField> + ReduceTo<FField>,
    Cfg::ExtField: FpExtEncoding<FField> + AkitaSerialize,
{
    let layout = singleton_layout::<Cfg>(nv);

    let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
    let evals: Vec<FField> = (0..1usize << nv)
        .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let poly = DensePoly::<FField>::from_field_evals(nv, D, &evals).unwrap();
    let pt = random_claim_point::<FField, Cfg::ExtField>(nv);
    let expected_opening = dense_lagrange_opening_from_evals::<FField, Cfg::ExtField>(&evals, &pt);

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(nv);

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup =
        AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .unwrap();

    let poly_refs: [&DensePoly<FField>; 1] = [&poly];
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<FField>::new(transcript_label);
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    let [commitment] = commitments;
    (
        verifier_setup,
        commitment,
        proof,
        pt,
        expected_opening,
        layout,
    )
}

/// Remove any stale disk-persistence cache for `max_num_vars` so that a setup
/// written by a different `CommitmentConfig` doesn't get loaded by mistake.
#[cfg(feature = "disk-persistence")]
fn purge_setup_cache(max_num_vars: usize) {
    let cache_dir = std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|_| {
            std::env::var("HOME").map(|home| {
                let mut p = PathBuf::from(&home);
                if p.join("Library/Caches").exists() {
                    p.push("Library/Caches");
                } else {
                    p.push(".cache");
                }
                p
            })
        });
    if let Ok(mut path) = cache_dir {
        path.push("akita");
        if let Ok(entries) = std::fs::read_dir(&path) {
            let needle = format!("_nv{max_num_vars}.setup");
            let batch_needle = format!("_nv{max_num_vars}_batch");
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("akita_")
                            && (name.ends_with(&needle) || name.contains(&batch_needle))
                    })
                {
                    let _ = std::fs::remove_file(entry_path);
                }
            }
        }
    }
}

fn bump_flat_ring_vec<FField: FieldCore>(flat: &mut akita_types::RingVec<FField>) {
    let mut coeffs = flat.coeffs().to_vec();
    let first = coeffs
        .first_mut()
        .expect("tamper target must contain at least one coefficient");
    *first += FField::one();
    *flat = akita_types::RingVec::from_coeffs(coeffs);
}

fn mutate_terminal_e_hat_digit<FField: FieldCore>(
    witness: &mut akita_types::TerminalResponse<FField>,
) {
    bump_flat_ring_vec(&mut witness.e_fields);
}

fn terminal_witness_mut<FField: FieldCore, E: FieldCore>(
    proof: &mut AkitaBatchedProof<FField, E>,
) -> &mut akita_types::TerminalResponse<FField> {
    proof.terminal.terminal_response_mut()
}

fn assert_invalid_proof<T: core::fmt::Debug>(
    case: &str,
    result: Result<T, akita_field::AkitaError>,
) {
    match result {
        Err(akita_field::AkitaError::InvalidProof) => {}
        Err(akita_field::AkitaError::InvalidInput(msg)) if msg.contains("InvalidProof") => {}
        other => panic!("{case} must reject with InvalidProof, got {other:?}"),
    }
}

/// End-to-end chunked prove→verify: the multi-chunk preset stamps
/// `num_chunks = 8` on the two leading fold levels (NV=16 ⇒ 64 blocks each).
/// The single prover assembles the modified `[zᵢ|eᵢ|t̂ᵢ]…|r̂` relation and the
/// verifier evaluates the chunked row-MLE; the proof must verify.
#[test]
fn chunked_multi_chunk_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64DenseMultiChunk;
        const D: usize = Cfg::D;
        const NV: usize = 16;

        // Confirm the schedule actually activates chunking on the leading folds.
        let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(NV),
        ))
        .expect("multi-chunk schedule");
        let chunked_levels = usize::from(plan.root.params.witness_partition.num_chunks() > 1)
            + plan
                .recursive_folds
                .iter()
                .filter(|fold| fold.params.witness_partition.num_chunks() > 1)
                .count();
        assert!(
            chunked_levels >= 1,
            "multi-chunk preset must produce at least one chunked fold level"
        );

        let layout = singleton_layout::<Cfg>(NV);
        let mut rng = StdRng::seed_from_u64(0x6b1d_c0de);
        let evals: Vec<F> = (0..1usize << NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F>::from_field_evals(NV, D, &evals).unwrap();
        let pt = random_point::<F>(NV);
        let expected_opening = opening_from_poly::<D, _>(&poly, &pt, &layout);

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .unwrap();

        let poly_refs: [&DensePoly<F>; 1] = [&poly];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_chunked_e2e");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitment,
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let proof_bytes = proof.size();
        assert!(proof_bytes > 0, "chunked proof must be non-empty");
        assert_eq!(
            batched_total_fold_levels(&proof),
            plan.num_fold_levels(),
            "chunked proof level count must match the schedule"
        );

        let verifier_setup =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_chunked_e2e");
        let openings = [expected_opening];
        let verify_result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], &openings[..], &commitment),
            BasisMode::Lagrange,
        );
        assert!(
            verify_result.is_ok(),
            "chunked verification must pass: {:?}",
            verify_result.err()
        );

        tracing::info!(chunked_levels, proof_bytes, "chunked-d64/nv16 e2e");
    });
}

#[test]
fn dense_d64_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Dense;
        const D: usize = Cfg::D;

        let layout = singleton_layout::<Cfg>(DENSE_TEST_NV);

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..1usize << DENSE_TEST_NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let poly = DensePoly::<F>::from_field_evals(DENSE_TEST_NV, D, &evals).unwrap();
        let pt = random_point::<F>(DENSE_TEST_NV);
        let expected_opening = opening_from_poly::<D, _>(&poly, &pt, &layout);

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(DENSE_TEST_NV);

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(DENSE_TEST_NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .unwrap();

        let poly_refs: [&DensePoly<F>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e");
        let prove_start = Instant::now();
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();
        let prove_time = prove_start.elapsed();

        let proof_bytes = proof.size();
        assert!(proof_bytes > 0, "proof must be non-empty");
        let total_fold_levels = batched_total_fold_levels(&proof);
        assert!(total_fold_levels > 0, "proof must have at least one level");

        let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(DENSE_TEST_NV),
        ))
        .expect("schedule plan");
        assert_eq!(total_fold_levels, plan.num_fold_levels());

        let verifier_setup =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e");
        let verify_start = Instant::now();
        let verify_result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        let verify_time = verify_start.elapsed();

        assert!(
            verify_result.is_ok(),
            "verification must pass: {:?}",
            verify_result.err()
        );

        tracing::info!(
            prove_s = prove_time.as_secs_f64(),
            verify_s = verify_time.as_secs_f64(),
            proof_bytes,
            proof_kib = proof_bytes as f64 / 1024.0,
            levels = total_fold_levels,
            "dense-d64/nv{DENSE_TEST_NV} e2e"
        );
    });
}

/// Snap-regenerated `fp128_d64_dense` schedules must verify at production `nv` keys.
#[test]
fn dense_d64_snap_regen_prove_verify_nv24() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Dense;
        const D: usize = Cfg::D;
        const NV: usize = 24;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(NV, b"akita_e2e/snap-regen-nv24");

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/snap-regen-nv24");
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "snap-regen dense fp128_d64 nv={NV} must verify: {:?}",
            result.err()
        );
    });
}

#[test]
fn trace_internalization_rejects_tampered_root_fold_handle() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Dense;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(DENSE_TEST_NV, b"akita_e2e/root-trace-tamper");
        let mut malformed = proof.clone();
        bump_flat_ring_vec(&mut malformed.root.v);

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/root-trace-tamper");
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert_invalid_proof("tampered root fold handle", result);
    });
}

#[test]
fn trace_internalization_rejects_tampered_recursive_fold_handle() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;
        const NV: usize = 20;

        let opening_batch = akita_types::OpeningClaimsLayout::new(NV, 2).expect("opening_batch");
        let layout = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F>> = (0..2)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x3141_5926 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F>::new(ONEHOT_K, D, indices).unwrap()
            })
            .collect();
        let poly_refs: Vec<&OneHotPoly<F>> = polys.iter().collect();
        let point = random_point(NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly::<D, _>(poly, &point, &layout))
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(NV);

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 2).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack).unwrap();
        let commitments = [commitment];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/recursive-trace-tamper");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input(&point[..], &poly_refs[..], &commitments[0], hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut malformed = proof.clone();
        let recursive = malformed
            .recursive_folds
            .first_mut()
            .expect("fixture should include an intermediate recursive fold");
        bump_flat_ring_vec(&mut recursive.v);

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/recursive-trace-tamper");
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert_invalid_proof("tampered recursive fold handle", result);
    });
}

#[test]
fn trace_internalization_rejects_tampered_terminal_e_hat_digit() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Dense;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(DENSE_TEST_NV, b"akita_e2e/terminal-trace-tamper");
        let mut malformed = proof.clone();
        mutate_terminal_e_hat_digit(terminal_witness_mut(&mut malformed));

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/terminal-trace-tamper");
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert_invalid_proof("tampered terminal e_hat digit", result);
    });
}

#[test]
fn small_field_d64_dense_degenerate_roots_fail_fast() {
    for result in [
        fp32::D64Dense::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(SMALL_FIELD_TEST_NV),
        )),
        fp64::D64Dense::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(SMALL_FIELD_TEST_NV + 1),
        )),
    ] {
        assert!(matches!(
            result,
            Err(akita_field::AkitaError::UnsupportedSchedule(_))
        ));
    }
}

#[test]
fn dense_d64_tiny_roots_and_setup_capacities_are_rejected() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Dense;
        let nv = 4;
        let err = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        ))
        .expect_err("tiny roots must not produce a degenerate proof schedule");
        assert!(matches!(
            err,
            akita_field::AkitaError::UnsupportedSchedule(_)
        ));
        let setup_err = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1)
            .expect_err("tiny capacity must not produce a prover setup");
        assert!(
            matches!(setup_err, akita_field::AkitaError::InvalidSetup(_)),
            "setup capacity rejection should use the setup boundary: {setup_err:?}"
        );
    });
}

#[test]
fn dense_d64_adaptive_mixed_basis_roundtrip_and_serialization() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Dense;
        const D: usize = Cfg::D;

        let nv = DENSE_TEST_NV;
        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(nv, b"akita_e2e/adaptive-dense-mixed");

        let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        ))
        .expect("schedule plan");
        assert_eq!(batched_total_fold_levels(&proof), plan.num_fold_levels());

        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .expect("serialize adaptive proof");
        let mut cursor = std::io::Cursor::new(proof_bytes);
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut cursor, &proof.shape())
                .expect("deserialize adaptive proof");
        assert_eq!(decoded, proof);

        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/adaptive-dense-mixed");
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "adaptive mixed-basis verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn adaptive_onehot_direct_tail_uses_terminal_schedule_basis() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;

        let nv = ONEHOT_TEST_NV;
        let layout = singleton_layout::<Cfg>(nv);
        let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        let indices: Vec<Option<usize>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
            .collect();
        let onehot_poly = OneHotPoly::<F>::new(ONEHOT_K, D, indices).unwrap();
        let pt = random_point::<F>(nv);
        let expected_opening = opening_from_poly::<D, _>(&onehot_poly, &pt, &layout);

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        let (commitment, hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&onehot_poly),
            &stack,
        )
        .unwrap();

        let poly_refs: [&OneHotPoly<F>; 1] = [&onehot_poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/onehot-direct-tail");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut serialized = Vec::new();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize adaptive onehot proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut cursor, &proof.shape())
                .expect("deserialize adaptive onehot proof");

        let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        ))
        .expect("schedule plan");
        assert_eq!(batched_total_fold_levels(&proof), plan.num_fold_levels());
        assert_eq!(decoded.size(), proof.size());

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/onehot-direct-tail");
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "adaptive onehot direct-tail verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn batched_onehot_same_point_round_trip() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        // NV=20 is large enough to include a recursive suffix, while the
        // two-claim opening batch still misses singleton/4-batch generated tables
        // and routes through the planner DP fallback in `runtime_schedule`.
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;
        const NV: usize = 20;

        let nv = NV;
        let opening_batch = akita_types::OpeningClaimsLayout::new(nv, 2).expect("opening_batch");
        let layout = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(NV),
        ))
        .expect("runtime schedule");
        let fold_params = std::iter::once(&plan.root.params.final_group.commitment)
            .chain(plan.recursive_folds.iter().map(|step| &step.params.witness))
            .collect::<Vec<_>>();
        assert!(
            fold_params.iter().any(|params| {
                params.num_live_ring_elements_per_claim % params.num_positions_per_block != 0
                    && params.num_live_blocks
                        == params
                            .num_live_ring_elements_per_claim
                            .div_ceil(params.num_positions_per_block)
            }),
            "fixture must cross a production fold with an exact partial final row"
        );
        let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let mut rng_a = StdRng::seed_from_u64(0x1234_5678);
        let mut rng_b = StdRng::seed_from_u64(0x8765_4321);
        let indices_a: Vec<Option<usize>> = (0..total_chunks)
            .map(|_| Some(rng_a.gen_range(0..ONEHOT_K)))
            .collect();
        let indices_b: Vec<Option<usize>> = (0..total_chunks)
            .map(|_| Some(rng_b.gen_range(0..ONEHOT_K)))
            .collect();
        let poly_a = OneHotPoly::<F>::new(ONEHOT_K, D, indices_a).unwrap();
        let poly_b = OneHotPoly::<F>::new(ONEHOT_K, D, indices_b).unwrap();
        let poly_group = [&poly_a, &poly_b];
        let pt = random_point(nv);
        let openings = [
            opening_from_poly::<D, _>(&poly_a, &pt, &layout),
            opening_from_poly::<D, _>(&poly_b, &pt, &layout),
        ];

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 2).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        let commit_group = [poly_a.clone(), poly_b.clone()];
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &commit_group, &stack).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input(
                &pt[..],
                &poly_group[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize batched onehot proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(&mut cursor, &proof_shape)
            .expect("deserialize batched onehot proof");
        let terminal = decoded.terminal_response();
        assert_eq!(
            terminal.layout.groups.len(),
            1,
            "terminal consumer must retain one canonical scalar group"
        );
        terminal
            .terminal_transcript_parts()
            .expect("terminal witness must split into canonical transcript segments");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot");
        let opening_groups = [&openings[..]];
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "batched onehot verification must pass: {:?}",
            result.err()
        );

        assert!(!decoded.recursive_folds.is_empty());
        let mut truncated = decoded.clone();
        truncated.recursive_folds.remove(0);
        let mut truncated_transcript = AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot");
        let truncated_result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &truncated,
            &verifier_setup,
            &mut truncated_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            truncated_result.is_err(),
            "proof with a truncated scheduled recursive suffix must be rejected"
        );
    });
}

#[test]
fn batched_onehot_same_point_rejects_tampered_root_stage1_range_image_evaluation() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;

        let nv = ONEHOT_TEST_NV;
        let layout =
            akita_batched_root_layout::<Cfg>(nv, SAME_POINT_ONEHOT_BATCH_SIZE).expect("layout");
        let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F>> = (0..SAME_POINT_ONEHOT_BATCH_SIZE)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x8765_4321 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F>::new(ONEHOT_K, D, indices).unwrap()
            })
            .collect();
        let poly_group: Vec<&OneHotPoly<F>> = polys.iter().collect();
        let pt = random_point(nv);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly::<D, _>(poly, &pt, &layout))
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup =
            AkitaCommitmentScheme::<Cfg>::setup_prover(nv, SAME_POINT_ONEHOT_BATCH_SIZE).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input(
                &pt[..],
                &poly_group[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut malformed = proof.clone();
        malformed.root.stage1.range_image_evaluation += F::from_canonical_u128_reduced(1);

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let opening_groups = [&openings[..]];
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tampered batched root stage1 range_image_evaluation must be rejected"
        );
    });
}
