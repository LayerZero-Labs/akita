use crate::report::{
    emit_planned_schedule_summary, emit_runtime_schedule_summary, print_batched_proof_summary,
    report_timing,
};
use akita_config::CommitmentConfig;
use akita_field::fields::wide::HasWide;
use akita_field::{
    CanonicalBytes, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    LiftBase, PseudoMersenneField, RandomSampling, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    AkitaPolyOps, AkitaProverSetup, CommitmentProver, CommittedPolynomials, DensePoly, OneHotIndex,
    OneHotPoly,
};
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::AkitaSerialize;
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    AkitaBatchedProof, AkitaCommitmentHint, AkitaSchedulePlan, AkitaVerifierSetup, BasisMode,
    BlockOrder, ClaimIncidenceSummary, LevelParams, RingCommitment, RingSubfieldEncoding, Step,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

pub(crate) const ONEHOT_K: usize = 256;

pub(crate) fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
}

fn assert_observed_proof_size<FF, L>(label: &str, proof: &AkitaBatchedProof<FF, L>)
where
    FF: FieldCore + AkitaSerialize,
    L: FieldCore + AkitaSerialize,
{
    let mut encoded = Vec::with_capacity(proof.size());
    proof
        .serialize_uncompressed(&mut encoded)
        .expect("profile proof serialization should succeed");
    assert_eq!(
        encoded.len(),
        proof.size(),
        "[{label}] proof.size() must match actual uncompressed serialization length"
    );
}

fn random_claim_point<FF, E>(nv: usize, rng: &mut StdRng) -> Vec<E>
where
    FF: CanonicalField,
    E: ExtField<FF>,
{
    (0..nv)
        .map(|_| {
            let limbs = (0..E::EXT_DEGREE)
                .map(|_| FF::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

fn degree_one_claim_point_to_base<FF, E>(point: &[E]) -> Option<Vec<FF>>
where
    FF: FieldCore,
    E: ExtField<FF>,
{
    (E::EXT_DEGREE == 1).then(|| {
        point
            .iter()
            .map(|coord| coord.to_base_vec()[0])
            .collect::<Vec<_>>()
    })
}

fn dense_lagrange_opening_from_evals<FF, E>(evals: &[FF], point: &[E]) -> E
where
    FF: FieldCore,
    E: ExtField<FF>,
{
    assert_eq!(evals.len(), 1usize << point.len());
    let mut layer = evals.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    for &r in point {
        let one_minus_r = E::one() - r;
        let next_len = layer.len() / 2;
        for i in 0..next_len {
            layer[i] = layer[2 * i] * one_minus_r + layer[2 * i + 1] * r;
        }
        layer.truncate(next_len);
    }
    layer[0]
}

fn onehot_lagrange_opening<FF, E, I, const D: usize>(poly: &OneHotPoly<FF, D, I>, point: &[E]) -> E
where
    FF: FieldCore,
    E: ExtField<FF>,
    I: OneHotIndex,
{
    let onehot_k = poly.onehot_k();
    assert!(onehot_k.is_power_of_two());
    assert_eq!(poly.indices().len() * onehot_k, 1usize << point.len());

    let low_vars = onehot_k.trailing_zeros() as usize;
    let low_weights = lagrange_weights(&point[..low_vars]).expect("valid low opening point");
    let high_weights = lagrange_weights(&point[low_vars..]).expect("valid high opening point");
    poly.indices()
        .iter()
        .enumerate()
        .filter_map(|(chunk_idx, hot_idx)| {
            hot_idx.map(|hot_idx| high_weights[chunk_idx] * low_weights[hot_idx.as_usize()])
        })
        .fold(E::zero(), |acc, weight| acc + weight)
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

fn run_prove<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>, P: AkitaPolyOps<FF, D>>(
    label: &str,
    setup: &<AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FF, D>>::ProverSetup,
    prepared: &<CpuBackend as ComputeBackendSetup<FF>>::PreparedSetup<D>,
    poly: &P,
    pt: &[Cfg::ClaimField],
    opening: Cfg::ClaimField,
    plan: Option<&AkitaSchedulePlan>,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ProverSetup = AkitaProverSetup<FF, D>,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
        >,
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + HasWide
        + AkitaSerialize
        + 'static,
    Cfg::ClaimField: RingSubfieldEncoding<FF> + AkitaSerialize,
    Cfg::ChallengeField: RingSubfieldEncoding<FF> + ExtField<Cfg::ClaimField> + AkitaSerialize,
{
    type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

    let t0 = Instant::now();
    let (commitment, hint) = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::commit(
        setup,
        &CpuBackend,
        prepared,
        std::slice::from_ref(poly),
    )
    .unwrap();
    report_timing(label, "commit", t0.elapsed().as_secs_f64());

    let poly_refs: [&P; 1] = [poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let t0 = Instant::now();
    let mut prover_transcript = AkitaTranscript::<FF>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::batched_prove(
        setup,
        &CpuBackend,
        prepared,
        vec![(
            pt,
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    report_timing(label, "prove", t0.elapsed().as_secs_f64());
    assert_observed_proof_size::<FF, Cfg::ChallengeField>(label, &proof);
    print_batched_proof_summary::<FF, Cfg::ChallengeField, D>(label, &proof);
    tracing::info!(
        label,
        claim_ext_degree = Cfg::CLAIM_EXT_DEGREE,
        challenge_ext_degree = Cfg::CHAL_EXT_DEGREE,
        "profile field roles"
    );
    eprintln!(
        "[{label}] field_roles: claim_ext_degree={}, challenge_ext_degree={}",
        Cfg::CLAIM_EXT_DEGREE,
        Cfg::CHAL_EXT_DEGREE,
    );
    if proof.is_root_direct() && Cfg::CLAIM_EXT_DEGREE > 1 {
        tracing::warn!(
            label,
            "extension opening used root-direct fallback; folded planner byte estimates do not apply"
        );
        eprintln!(
            "[{label}] extension opening fallback: root-direct proof for this unsupported shape; folded planner byte estimates do not apply"
        );
    }
    if let Some(plan) = plan {
        assert_eq!(
            proof.size(),
            plan.exact_proof_bytes,
            "runtime proof bytes should match the planned proof size"
        );
        emit_planned_schedule_summary(label, plan, 1, Cfg::decomposition().field_bits());
    } else {
        let incidence =
            ClaimIncidenceSummary::same_point(pt.len(), 1).expect("same-point incidence summary");
        let schedule = Cfg::get_params_for_prove(&incidence).expect("runtime schedule");
        assert_eq!(
            proof.size(),
            schedule.total_bytes,
            "runtime proof bytes should match the runtime schedule proof size"
        );
        emit_runtime_schedule_summary(label, &schedule, 1, Cfg::decomposition().field_bits());
    }

    let t0 = Instant::now();
    let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::setup_verifier(setup);
    let mut verifier_transcript = AkitaTranscript::<FF>::new(b"profile");
    match <Scheme<D, Cfg> as CommitmentVerifier<FF, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            pt,
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    ) {
        Ok(()) => report_timing(label, "verify OK", t0.elapsed().as_secs_f64()),
        Err(e) => {
            let elapsed_s = t0.elapsed().as_secs_f64();
            tracing::error!(label, elapsed_s, error = %e, "verify FAILED");
            eprintln!("[{label}] verify FAILED: {elapsed_s:.6}s ({e})");
            panic!("[{label}] profile verification failed: {e}");
        }
    }
}

pub(crate) fn run_dense_for<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    nv: usize,
    layout: &LevelParams,
    plan: Option<&AkitaSchedulePlan>,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + PseudoMersenneField
        + HasWide
        + AkitaSerialize
        + 'static,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ProverSetup = AkitaProverSetup<FF, D>,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
        >,
    Cfg::ClaimField: FrobeniusExtField<FF> + RingSubfieldEncoding<FF> + AkitaSerialize,
    Cfg::ChallengeField: RingSubfieldEncoding<FF> + ExtField<Cfg::ClaimField> + AkitaSerialize,
{
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let original_pt = random_claim_point::<FF, Cfg::ClaimField>(nv, &mut rng);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
    let evals: Vec<FF> = if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| FF::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        (0..len)
            .map(|_| FF::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    };
    let poly = DensePoly::<FF, D>::from_field_evals(nv, &evals).unwrap();
    let opening = if let Some(base_pt) =
        degree_one_claim_point_to_base::<FF, Cfg::ClaimField>(&original_pt)
    {
        Cfg::ClaimField::lift_base(opening_from_poly(
            &poly,
            &base_pt,
            layout,
            BasisMode::Lagrange,
        ))
    } else {
        dense_lagrange_opening_from_evals::<FF, Cfg::ClaimField>(&evals, &original_pt)
    };
    let t0 = Instant::now();
    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FF, D>>::setup_prover(
        poly.num_vars(),
        1,
        1,
    )
    .unwrap();
    let setup_expand_secs = t0.elapsed().as_secs_f64();
    let t_prepare = Instant::now();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    report_timing(label, "setup_expand", setup_expand_secs);
    report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
    report_timing(label, "setup", t0.elapsed().as_secs_f64());

    run_prove::<FF, D, Cfg, _>(label, &setup, &prepared, &poly, &original_pt, opening, plan);
}

pub(crate) fn run_onehot<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    nv: usize,
    layout: &LevelParams,
    plan: Option<&AkitaSchedulePlan>,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + PseudoMersenneField
        + HasWide
        + AkitaSerialize
        + 'static,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ProverSetup = AkitaProverSetup<FF, D>,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
        >,
    Cfg::ClaimField: FrobeniusExtField<FF> + RingSubfieldEncoding<FF> + AkitaSerialize,
    Cfg::ChallengeField: RingSubfieldEncoding<FF> + ExtField<Cfg::ClaimField> + AkitaSerialize,
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
    let pt = random_claim_point::<FF, Cfg::ClaimField>(nv, &mut rng);
    let opening = if let Some(base_pt) = degree_one_claim_point_to_base::<FF, Cfg::ClaimField>(&pt)
    {
        Cfg::ClaimField::lift_base(opening_from_poly(
            &onehot_poly,
            &base_pt,
            layout,
            BasisMode::Lagrange,
        ))
    } else {
        onehot_lagrange_opening::<FF, Cfg::ClaimField, u8, D>(&onehot_poly, &pt)
    };
    let t0 = Instant::now();
    let setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FF, D>>::setup_prover(nv, 1, 1).unwrap();
    let setup_expand_secs = t0.elapsed().as_secs_f64();
    let t_prepare = Instant::now();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    report_timing(label, "setup_expand", setup_expand_secs);
    report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
    report_timing(label, "setup", t0.elapsed().as_secs_f64());

    run_prove::<FF, D, Cfg, _>(label, &setup, &prepared, &onehot_poly, &pt, opening, plan);
}

pub(crate) fn run_batched_onehot<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    nv: usize,
    num_polys: usize,
    layout: &LevelParams,
    plan: Option<&AkitaSchedulePlan>,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + PseudoMersenneField
        + HasWide
        + AkitaSerialize
        + 'static,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ProverSetup = AkitaProverSetup<FF, D>,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
        >,
    Cfg::ClaimField: FrobeniusExtField<FF> + RingSubfieldEncoding<FF> + AkitaSerialize,
    Cfg::ChallengeField: RingSubfieldEncoding<FF> + ExtField<Cfg::ClaimField> + AkitaSerialize,
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
    let pt = random_claim_point::<FF, Cfg::ClaimField>(nv, &mut point_rng);
    let openings: Vec<Cfg::ClaimField> =
        if let Some(base_pt) = degree_one_claim_point_to_base::<FF, Cfg::ClaimField>(&pt) {
            polys
                .iter()
                .map(|poly| {
                    Cfg::ClaimField::lift_base(opening_from_poly(
                        poly,
                        &base_pt,
                        layout,
                        BasisMode::Lagrange,
                    ))
                })
                .collect()
        } else {
            polys
                .iter()
                .map(|poly| onehot_lagrange_opening::<FF, Cfg::ClaimField, u8, D>(poly, &pt))
                .collect()
        };
    let poly_refs: Vec<&OneHotPoly<FF, D, u8>> = polys.iter().collect();
    let opening_groups = [&openings[..]];

    let t0 = Instant::now();
    let setup =
        <Scheme<D, Cfg> as CommitmentProver<FF, D>>::setup_prover(nv, num_polys, 1).unwrap();
    let setup_expand_secs = t0.elapsed().as_secs_f64();
    let t_prepare = Instant::now();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    report_timing(label, "setup_expand", setup_expand_secs);
    report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
    report_timing(label, "setup", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let (commitment, hint) = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .unwrap();
    let commitments = [commitment];
    let hints = vec![hint];
    report_timing(label, "commit", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut prover_transcript = AkitaTranscript::<FF>::new(b"profile");
    let proof = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &pt[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    report_timing(label, "prove", t0.elapsed().as_secs_f64());
    assert_observed_proof_size::<FF, Cfg::ChallengeField>(label, &proof);
    print_batched_proof_summary::<FF, Cfg::ChallengeField, D>(label, &proof);
    let incidence =
        ClaimIncidenceSummary::same_point(nv, num_polys).expect("same-point incidence summary");
    let schedule = Cfg::get_params_for_prove(&incidence).expect("batched schedule");
    if let Some(plan) = plan {
        assert_eq!(
            proof.size(),
            plan.exact_proof_bytes,
            "runtime proof bytes should match the planned proof size"
        );
        emit_planned_schedule_summary(label, plan, num_polys, Cfg::decomposition().field_bits());
    } else {
        assert_eq!(
            proof.size(),
            schedule.total_bytes,
            "runtime proof bytes should match the runtime schedule proof size"
        );
        emit_runtime_schedule_summary(
            label,
            &schedule,
            num_polys,
            Cfg::decomposition().field_bits(),
        );
    }
    tracing::info!(
        label,
        claim_ext_degree = Cfg::CLAIM_EXT_DEGREE,
        challenge_ext_degree = Cfg::CHAL_EXT_DEGREE,
        "profile field roles"
    );
    eprintln!(
        "[{label}] field_roles: claim_ext_degree={}, challenge_ext_degree={}",
        Cfg::CLAIM_EXT_DEGREE,
        Cfg::CHAL_EXT_DEGREE,
    );
    if proof.is_root_direct() && Cfg::CLAIM_EXT_DEGREE > 1 {
        tracing::warn!(
            label,
            "extension opening used root-direct fallback; folded planner byte estimates do not apply"
        );
        eprintln!(
            "[{label}] extension opening fallback: root-direct proof for this unsupported shape; folded planner byte estimates do not apply"
        );
    }
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
    let mut verifier_transcript = AkitaTranscript::<FF>::new(b"profile");
    match <Scheme<D, Cfg> as CommitmentVerifier<FF, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &pt[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    ) {
        Ok(()) => report_timing(label, "verify OK", t0.elapsed().as_secs_f64()),
        Err(e) => {
            let elapsed_s = t0.elapsed().as_secs_f64();
            tracing::error!(label, elapsed_s, error = %e, "verify FAILED");
            eprintln!("[{label}] verify FAILED: {elapsed_s:.6}s ({e})");
            panic!("[{label}] batched profile verification failed: {e}");
        }
    }
}
