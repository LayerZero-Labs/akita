use crate::report::{emit_planned_schedule_summary, print_batched_proof_summary, report_timing};
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::fields::wide::HasWide;
use akita_field::{CanonicalBytes, CanonicalField, RandomSampling, TranscriptChallenge};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::kernels::crt_ntt::NttSlotCache;
use akita_prover::{AkitaPolyOps, CommitmentProver, CommittedPolynomials, DensePoly, OneHotPoly};
use akita_serialization::AkitaSerialize;
use akita_transcript::Blake2bTranscript;
use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaBatchedProof,
    AkitaCommitmentHint, AkitaRootBatchSummary, AkitaSchedulePlan, AkitaVerifierSetup, BasisMode,
    BlockOrder, LevelParams, RingCommitment, Step,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

type F = fp128::Field;

pub(crate) const ONEHOT_K: usize = 256;

pub(crate) fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
}

fn opening_from_poly<FF, const D: usize, P: AkitaPolyOps<FF, D>>(
    poly: &P,
    point: &[FF],
    layout: &LevelParams,
    basis: BasisMode,
) -> FF
where
    FF: CanonicalField,
{
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.m_vars + layout.r_vars;
    assert!(
        point.len() <= target_num_vars,
        "opening point length {} exceeds target root arity {}",
        point.len(),
        target_num_vars
    );
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, FF::zero());

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
    let v = reduce_inner_opening_to_ring_element::<FF, D>(inner_point, basis)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

fn run_prove<
    FF,
    const D: usize,
    Cfg: CommitmentConfig<Field = FF, ClaimField = FF, ChallengeField = FF>,
    P: AkitaPolyOps<FF, D, CommitCache = NttSlotCache<D>>,
>(
    label: &str,
    setup: &<AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FF, D>>::ProverSetup,
    poly: &P,
    pt: &[FF],
    opening: FF,
    plan: Option<&AkitaSchedulePlan>,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ClaimField = FF,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, FF>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = FF,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, FF>,
        >,
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + HasWide
        + AkitaSerialize
        + 'static,
{
    type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentProver<FF, D>>::commit(std::slice::from_ref(poly), setup)
            .unwrap();
    report_timing(label, "commit", t0.elapsed().as_secs_f64());

    let poly_refs: [&P; 1] = [poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<FF>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::batched_prove(
        setup,
        vec![(
            pt,
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    report_timing(label, "prove", t0.elapsed().as_secs_f64());
    print_batched_proof_summary::<FF, D>(label, &proof);
    if let Some(plan) = plan {
        debug_assert_eq!(
            proof.size(),
            plan.exact_proof_bytes,
            "runtime proof bytes should match the planned proof size"
        );
        emit_planned_schedule_summary(label, plan);
    }

    let t0 = Instant::now();
    let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::setup_verifier(setup);
    let mut verifier_transcript = Blake2bTranscript::<FF>::new(b"profile");
    match <Scheme<D, Cfg> as CommitmentVerifier<FF, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            pt,
            vec![CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            }],
        )],
        BasisMode::Lagrange,
    ) {
        Ok(()) => report_timing(label, "verify OK", t0.elapsed().as_secs_f64()),
        Err(e) => {
            let elapsed_s = t0.elapsed().as_secs_f64();
            tracing::error!(label, elapsed_s, error = %e, "verify FAILED");
            eprintln!("[{label}] verify FAILED: {elapsed_s:.6}s ({e})");
        }
    }
}

pub(crate) fn run_dense<
    const D: usize,
    Cfg: CommitmentConfig<Field = F, ClaimField = F, ChallengeField = F>,
>(
    nv: usize,
    layout: &LevelParams,
    plan: Option<&AkitaSchedulePlan>,
) {
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let pt: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let (poly, opening) = {
        let len = 1usize << nv;
        let decomp = Cfg::decomposition();
        let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
        let evals: Vec<F> = if decomp.log_commit_bound >= 128 {
            (0..len)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect()
        } else {
            (0..len)
                .map(|_| F::from_i64(rng.gen_range(-half_bound..half_bound)))
                .collect()
        };
        let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
        let opening = opening_from_poly(&poly, &pt, layout, BasisMode::Lagrange);
        (poly, opening)
    };

    let t0 = Instant::now();
    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
    report_timing("dense", "setup", t0.elapsed().as_secs_f64());

    run_prove::<F, D, Cfg, _>("dense", &setup, &poly, &pt, opening, plan);
}

pub(crate) fn run_onehot<
    FF,
    const D: usize,
    Cfg: CommitmentConfig<Field = FF, ClaimField = FF, ChallengeField = FF>,
>(
    label: &str,
    nv: usize,
    layout: &LevelParams,
    plan: Option<&AkitaSchedulePlan>,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + HasWide
        + AkitaSerialize
        + 'static,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ClaimField = FF,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, FF>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = FF,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, FF>,
        >,
{
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(D)
        .expect("total field size overflow");
    let onehot_k = onehot_k_for_num_vars(nv);
    let total_chunks = total_field / onehot_k;
    assert_eq!(
        total_chunks * onehot_k,
        total_field,
        "onehot K must divide total field size"
    );

    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    let onehot_poly = OneHotPoly::<FF, D, u8>::new(onehot_k, indices).unwrap();
    let pt: Vec<FF> = (0..nv)
        .map(|_| FF::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = opening_from_poly(&onehot_poly, &pt, layout, BasisMode::Lagrange);

    let t0 = Instant::now();
    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FF, D>>::setup_prover(nv, 1, 1);
    report_timing(label, "setup", t0.elapsed().as_secs_f64());

    run_prove::<FF, D, Cfg, _>(label, &setup, &onehot_poly, &pt, opening, plan);
}

pub(crate) fn run_batched_onehot<
    FF,
    const D: usize,
    Cfg: CommitmentConfig<Field = FF, ClaimField = FF, ChallengeField = FF>,
>(
    label: &str,
    nv: usize,
    num_polys: usize,
    layout: &LevelParams,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + HasWide
        + AkitaSerialize
        + 'static,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ClaimField = FF,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, FF>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = FF,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, FF>,
        >,
{
    type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(D)
        .expect("total field size overflow");
    let onehot_k = onehot_k_for_num_vars(nv);
    let total_chunks = total_field / onehot_k;
    assert_eq!(
        total_chunks * onehot_k,
        total_field,
        "onehot K must divide total field size"
    );

    let polys: Vec<OneHotPoly<FF, D, u8>> = (0..num_polys)
        .map(|poly_idx| {
            let mut rng = StdRng::seed_from_u64(0xbeef_cafe ^ ((poly_idx as u64 + 1) << 32));
            let indices: Vec<Option<u8>> = (0..total_chunks)
                .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
                .collect();
            OneHotPoly::<FF, D, u8>::new(onehot_k, indices).unwrap()
        })
        .collect();
    let mut point_rng = StdRng::seed_from_u64(0xfeed_face);
    let pt: Vec<FF> = (0..nv)
        .map(|_| FF::from_canonical_u128_reduced(point_rng.gen::<u128>()))
        .collect();
    let openings: Vec<FF> = polys
        .iter()
        .map(|poly| opening_from_poly(poly, &pt, layout, BasisMode::Lagrange))
        .collect();
    let poly_refs: Vec<&OneHotPoly<FF, D, u8>> = polys.iter().collect();
    let opening_groups = [&openings[..]];

    let t0 = Instant::now();
    let setup = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::setup_prover(nv, num_polys, 1);
    report_timing(label, "setup", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentProver<FF, D>>::commit(&poly_refs, &setup).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];
    report_timing(label, "commit", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<FF>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::batched_prove(
        &setup,
        vec![(
            &pt[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    report_timing(label, "prove", t0.elapsed().as_secs_f64());
    print_batched_proof_summary::<FF, D>(label, &proof);
    let batch_summary =
        AkitaRootBatchSummary::new(num_polys, 1, 1).expect("same-point batch summary");
    let schedule =
        Cfg::get_params_for_prove(nv, nv, num_polys, batch_summary).expect("batched schedule");
    if let Some(Step::Fold(root_step)) = schedule.steps.first() {
        tracing::info!(
            label,
            root_bytes = root_step.level_bytes,
            observed_total_bytes = proof.size(),
            "batched planner root-fold summary"
        );
    } else if let Some(Step::Direct(root_direct)) = schedule.steps.first() {
        tracing::info!(
            label,
            root_bytes = root_direct.direct_bytes,
            observed_total_bytes = proof.size(),
            "batched planner direct-root estimate"
        );
    }

    let t0 = Instant::now();
    let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<FF>::new(b"profile");
    match <Scheme<D, Cfg> as CommitmentVerifier<FF, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &pt[..],
            vec![CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            }],
        )],
        BasisMode::Lagrange,
    ) {
        Ok(()) => report_timing(label, "verify OK", t0.elapsed().as_secs_f64()),
        Err(e) => {
            let elapsed_s = t0.elapsed().as_secs_f64();
            tracing::error!(label, elapsed_s, error = %e, "verify FAILED");
            eprintln!("[{label}] verify FAILED: {elapsed_s:.6}s ({e})");
        }
    }
}
