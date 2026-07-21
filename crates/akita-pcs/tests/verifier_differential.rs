//! Differential harness: pre-refactor `akita-verifier-legacy` vs. the live
//! `akita-verifier`, byte-for-byte.
//!
//! Phase-0 safety net for the akita-verifier refactor
//! (`specs/akita-verifier-refactor.md`). For every proof shape in the matrix it
//! runs the frozen legacy pipeline (`akita_verifier_legacy::batched_verify`)
//! and the live pipeline (`akita_verifier::batched_verify`) on *identical*
//! inputs and asserts they agree on:
//!
//!   1. the accept/reject verdict (and, on reject, the `AkitaError` variant), and
//!   2. the recorded public transcript event stream, byte-for-byte
//!      (`Preamble`/`Absorb`/`Squeeze` events, compared via canonical-byte
//!      digests). Transcript divergence is the failure mode a plain
//!      accept/reject check would miss.
//!
//! While the two crates are still identical this is a tautology; it gains teeth
//! as `akita-verifier` is rewritten in Phases 1-3. Deleted together with
//! `akita-verifier-legacy` at the end of Phase 3.
//!
//! Base: folded-only proofs (main @ #311/#312) — every proof is
//! `root + recursive_folds + terminal`; there is no ZeroFold/root-direct path.
//!
//! Requires the `logging-transcript` feature. Run:
//!   cargo test -p akita-pcs --features logging-transcript,parallel \
//!       --test verifier_differential

#![cfg(feature = "logging-transcript")]
#![allow(missing_docs)]

mod common;

use std::sync::Mutex;

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::CommitmentConfig;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
use akita_field::{
    AkitaError, CanonicalBytes, CanonicalField, ExtField, FieldCore, FrobeniusExtField,
    FromPrimitiveInt, HalvingField, PseudoMersenneField, RandomSampling, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend, DensePoly, UniformProverStack};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::{AkitaTranscript, LoggingTranscript, TranscriptEvent};
use akita_types::{
    lagrange_weights, AkitaBatchedProof, AkitaVerifierSetup, BasisMode, Commitment, FpExtEncoding,
    LevelParams, OpeningClaimsLayout, RingVec,
};

use common::{
    init_rayon_pool, make_onehot_poly, opening_from_poly, opening_from_poly_with_basis,
    prove_input, public_transcript_events, random_point, run_on_large_stack, verify_input,
    ONEHOT_D,
};

/// Serialize the heavy prove/verify cases: they share the global rayon pool and
/// each runs on a large dedicated stack.
static DIFFERENTIAL_LOCK: Mutex<()> = Mutex::new(());

type Fp128 = fp128::Field;

fn singleton_layout<Cfg: CommitmentConfig>(nv: usize) -> LevelParams {
    let batch = OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch");
    Cfg::get_params_for_batched_commitment(&batch).expect("singleton commitment layout")
}

// ---------------------------------------------------------------------------
// The comparison core.
// ---------------------------------------------------------------------------

/// Run `batched_verify` under a recording transcript and return the verdict
/// plus the recorded public event stream.
#[allow(clippy::too_many_arguments)]
fn run<Cfg, V>(
    verify: V,
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    commitment: &Commitment<Cfg::Field>,
    point: &[Cfg::ExtField],
    openings: &[Cfg::ExtField],
    basis: BasisMode,
    domain: &[u8],
) -> (Result<(), AkitaError>, Vec<TranscriptEvent>)
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge,
    V: FnOnce(
        &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
        &AkitaVerifierSetup<Cfg::Field>,
        &mut LoggingTranscript<AkitaTranscript<Cfg::Field>>,
        akita_types::OpeningClaims<'_, Cfg::ExtField, &Commitment<Cfg::Field>>,
        BasisMode,
    ) -> Result<(), AkitaError>,
{
    let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<Cfg::Field>::new(domain));
    let result = verify(
        proof,
        setup,
        &mut transcript,
        verify_input(point, openings, commitment),
        basis,
    );
    (result, public_transcript_events(transcript.events()))
}

/// Assert the legacy and live verifiers agree on a single input, then assert the
/// expected verdict.
#[allow(clippy::too_many_arguments)]
fn assert_parity<Cfg>(
    case: &str,
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    commitment: &Commitment<Cfg::Field>,
    point: &[Cfg::ExtField],
    openings: &[Cfg::ExtField],
    basis: BasisMode,
    domain: &[u8],
    expect_accept: bool,
) where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + PseudoMersenneField
        + HalvingField
        + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + FromPrimitiveInt
        + AkitaSerialize,
{
    let (live_result, live_events) = run::<Cfg, _>(
        akita_verifier::batched_verify::<Cfg, _>,
        proof,
        setup,
        commitment,
        point,
        openings,
        basis,
        domain,
    );
    let (legacy_result, legacy_events) = run::<Cfg, _>(
        akita_verifier_legacy::batched_verify::<Cfg, _>,
        proof,
        setup,
        commitment,
        point,
        openings,
        basis,
        domain,
    );

    match (&live_result, &legacy_result) {
        (Ok(()), Ok(())) => {}
        (Err(live), Err(legacy)) => assert!(
            std::mem::discriminant(live) == std::mem::discriminant(legacy),
            "{case}: legacy/live reject with different error variants: \
             live={live:?} legacy={legacy:?}",
        ),
        _ => panic!(
            "{case}: legacy/live disagree on accept/reject: \
             live={live_result:?} legacy={legacy_result:?}",
        ),
    }

    assert_eq!(
        live_events, legacy_events,
        "{case}: public transcript byte-log diverged between legacy and live verifier",
    );

    if expect_accept {
        assert!(
            live_result.is_ok(),
            "{case}: expected accept, got {:?}",
            live_result.err(),
        );
        // Guard against a degenerate empty recording silently passing the
        // byte-parity check: every accept run at least binds the instance
        // descriptor preamble, and folded proofs record many more events.
        assert!(
            !live_events.is_empty(),
            "{case}: recorded transcript is empty — the recording transcript \
             is not wired, so the byte-parity check has no teeth",
        );
    } else {
        assert!(
            live_result.is_err(),
            "{case}: expected reject, both accepted"
        );
    }
}

// ---------------------------------------------------------------------------
// Fixture generation (folded-only; every proof is root + recursive_folds + terminal).
// ---------------------------------------------------------------------------

type Fixture<F, E> = (
    AkitaVerifierSetup<F>,
    Commitment<F>,
    AkitaBatchedProof<F, E>,
    Vec<E>,
    E,
);

fn random_claim_point<F, E>(nv: usize) -> Vec<E>
where
    F: CanonicalField,
    E: ExtField<F>,
{
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| {
            let limbs = (0..E::EXT_DEGREE)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

fn dense_lagrange_opening<F, E>(evals: &[F], point: &[E]) -> E
where
    F: FieldCore,
    E: ExtField<F>,
{
    let weights = lagrange_weights(point).expect("valid opening point");
    evals
        .iter()
        .zip(weights.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight * E::lift_base(coeff)
        })
}

/// Dense Lagrange fixture generic over the field/config (covers fp32/fp64/fp128
/// and the `E != F` extension-field code path).
fn dense_fixture<F, const D: usize, Cfg>(
    nv: usize,
    domain: &'static [u8],
) -> Fixture<F, Cfg::ExtField>
where
    Cfg: CommitmentConfig<Field = F>,
    F: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + HasWide
        + RandomSampling
        + FromPrimitiveInt
        + 'static
        + HalvingField
        + PseudoMersenneField
        + Valid,
    Cfg::ExtField: FrobeniusExtField<F> + HasUnreducedOps + HasOptimizedFold,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    Cfg::ExtField: FpExtEncoding<F> + AkitaSerialize,
{
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
    let evals: Vec<F> = (0..1usize << nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(nv, D, &evals).unwrap();
    let point = random_claim_point::<F, Cfg::ExtField>(nv);
    let opening = dense_lagrange_opening::<F, Cfg::ExtField>(&evals, &point);

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
        .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).unwrap();
    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .unwrap();

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let mut prover_transcript = AkitaTranscript::<F>::new(domain);
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(&point[..], &poly_refs[..], &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    (verifier_setup, commitment, proof, point, opening)
}

/// fp128 dense fixture under an explicit basis (fp128 has `ExtField == Field`,
/// so this also lets us exercise `BasisMode::Monomial` with a correct opening).
fn fp128_dense_fixture(
    nv: usize,
    basis: BasisMode,
    domain: &'static [u8],
) -> Fixture<Fp128, Fp128> {
    type Cfg = fp128::D64Full;
    const D: usize = <fp128::D64Full as CommitmentConfig>::D;

    let layout = singleton_layout::<Cfg>(nv);
    let poly = common::make_dense_poly(nv, 0xdead_beef);
    let point = random_point(nv, 0xcafe_babe);
    let opening = opening_from_poly_with_basis::<D, _>(&poly, &point, &layout, basis);

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
        .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).unwrap();
    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .unwrap();

    let poly_refs: [&DensePoly<Fp128>; 1] = [&poly];
    let mut prover_transcript = AkitaTranscript::<Fp128>::new(domain);
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(&point[..], &poly_refs[..], &commitment, hint),
        &stack,
        &mut prover_transcript,
        basis,
    )
    .unwrap();

    (verifier_setup, commitment, proof, point, opening)
}

/// fp128 one-hot fixture (K=256 one-hot chunks).
fn fp128_onehot_fixture(nv: usize, domain: &'static [u8]) -> Fixture<Fp128, Fp128> {
    type Cfg = fp128::D64OneHot;

    let layout = singleton_layout::<Cfg>(nv);
    let poly = make_onehot_poly(&layout, 0x0123_4567);
    let point = random_point(nv, 0xcafe_babe);
    let opening = opening_from_poly::<ONEHOT_D, _>(&poly, &point, &layout);

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
        .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).unwrap();
    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .unwrap();

    let poly_refs = [&poly];
    let mut prover_transcript = AkitaTranscript::<Fp128>::new(domain);
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(&point[..], &poly_refs[..], &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    (verifier_setup, commitment, proof, point, opening)
}

// ---------------------------------------------------------------------------
// Accept-case shapes.
// ---------------------------------------------------------------------------

#[test]
fn fp128_dense_lagrange() {
    let _guard = DIFFERENTIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    init_rayon_pool();
    run_on_large_stack(|| {
        let (setup, commitment, proof, point, opening) =
            fp128_dense_fixture(14, BasisMode::Lagrange, b"diff/fp128/dense");
        assert!(
            proof.num_fold_levels() >= 2,
            "folded proof must have >= 2 levels"
        );
        assert_parity::<fp128::D64Full>(
            "fp128 dense (Lagrange)",
            &proof,
            &setup,
            &commitment,
            &point,
            &[opening],
            BasisMode::Lagrange,
            b"diff/fp128/dense",
            true,
        );
    });
}

#[test]
fn fp128_dense_monomial() {
    let _guard = DIFFERENTIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    init_rayon_pool();
    run_on_large_stack(|| {
        let (setup, commitment, proof, point, opening) =
            fp128_dense_fixture(14, BasisMode::Monomial, b"diff/fp128/monomial");
        assert_parity::<fp128::D64Full>(
            "fp128 dense (Monomial)",
            &proof,
            &setup,
            &commitment,
            &point,
            &[opening],
            BasisMode::Monomial,
            b"diff/fp128/monomial",
            true,
        );
    });
}

#[test]
fn fp128_onehot_multifold() {
    let _guard = DIFFERENTIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    init_rayon_pool();
    run_on_large_stack(|| {
        // nv=20 one-hot produces a non-empty recursive suffix, exercising the
        // suffix-fold replay path (not just root + terminal).
        let (setup, commitment, proof, point, opening) =
            fp128_onehot_fixture(20, b"diff/fp128/onehot");
        assert!(
            !proof.recursive_folds.is_empty(),
            "onehot nv=20 must produce a recursive suffix"
        );
        assert_parity::<fp128::D64OneHot>(
            "fp128 one-hot (multi-fold + suffix)",
            &proof,
            &setup,
            &commitment,
            &point,
            &[opening],
            BasisMode::Lagrange,
            b"diff/fp128/onehot",
            true,
        );
    });
}

// Extension-field (`E != F`) coverage — the code path the Phase-1 F->E generic
// rename touches — is exercised by the fp64 dense case below (degree-2
// extension). The spec matrix's Fp32 cell (degree-4 extension) is deferred: no
// fp32 *dense* schedule table ships in `schedules-default` (only one-hot), so a
// dense fp32 fixture cannot size its setup, and a fp32 one-hot fixture needs a
// bespoke extension-field one-hot opening. Follow-up: add fp32 one-hot coverage
// (or a generated fp32 dense schedule) to close the matrix.

#[test]
fn fp64_dense_extension_field() {
    let _guard = DIFFERENTIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    init_rayon_pool();
    run_on_large_stack(|| {
        let (setup, commitment, proof, point, opening) = dense_fixture::<
            fp64::Field,
            { <fp64::D64Full as CommitmentConfig>::D },
            fp64::D64Full,
        >(14, b"diff/fp64/dense");
        assert_parity::<fp64::D64Full>(
            "fp64 dense (extension field)",
            &proof,
            &setup,
            &commitment,
            &point,
            &[opening],
            BasisMode::Lagrange,
            b"diff/fp64/dense",
            true,
        );
    });
}

// ---------------------------------------------------------------------------
// Reject-case shapes: legacy and live must reject with the same error variant
// and identical transcript up to the point of divergence.
// ---------------------------------------------------------------------------

#[test]
fn reject_cases_agree() {
    let _guard = DIFFERENTIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    init_rayon_pool();
    run_on_large_stack(|| {
        let (setup, commitment, proof, point, opening) =
            fp128_dense_fixture(14, BasisMode::Lagrange, b"diff/fp128/reject");

        // Sanity: the untampered fixture is accepted by both.
        assert_parity::<fp128::D64Full>(
            "reject baseline (accept)",
            &proof,
            &setup,
            &commitment,
            &point,
            &[opening],
            BasisMode::Lagrange,
            b"diff/fp128/reject",
            true,
        );

        // Tampered opening value.
        assert_parity::<fp128::D64Full>(
            "reject: tampered opening value",
            &proof,
            &setup,
            &commitment,
            &point,
            &[opening + Fp128::one()],
            BasisMode::Lagrange,
            b"diff/fp128/reject",
            false,
        );

        // Tampered opening point.
        let mut bad_point = point.clone();
        bad_point[0] += Fp128::one();
        assert_parity::<fp128::D64Full>(
            "reject: tampered opening point",
            &proof,
            &setup,
            &commitment,
            &bad_point,
            &[opening],
            BasisMode::Lagrange,
            b"diff/fp128/reject",
            false,
        );

        // Tampered proof: bump a coefficient of the quotient-free terminal witness.
        let mut bad_proof = proof.clone();
        bump_terminal_e_field(&mut bad_proof);
        assert_parity::<fp128::D64Full>(
            "reject: tampered terminal witness",
            &bad_proof,
            &setup,
            &commitment,
            &point,
            &[opening],
            BasisMode::Lagrange,
            b"diff/fp128/reject",
            false,
        );
    });
}

/// Bump the first coefficient of the proof's terminal `e_fields` in place.
fn bump_terminal_e_field<F: FieldCore, E: FieldCore>(proof: &mut AkitaBatchedProof<F, E>) {
    let witness = proof.terminal.final_witness_mut();
    let mut coeffs = witness.e_fields.coeffs().to_vec();
    *coeffs.first_mut().expect("non-empty terminal e_fields") += F::one();
    witness.e_fields = RingVec::from_coeffs(coeffs);
}
