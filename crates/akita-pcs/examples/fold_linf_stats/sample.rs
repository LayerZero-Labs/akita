//! Prove+verify sampling harness for the `fold_linf_stats` example.

use akita_config::CommitmentConfig;
use akita_field::unreduced::HasWide;
use akita_field::{
    CanonicalBytes, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    LiftBase, PseudoMersenneField, RandomSampling, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::compute::{
    OpeningFoldKernel, RecursiveProveBackend, RootCommitBackend, RootCommitPoly, RootPolyShape,
    RootProvePoly,
};
use akita_prover::{
    AkitaProverSetup, CommitmentProver, DensePoly, FoldGrindObservation, FoldGrindObserverGuard,
    OneHotIndex, OneHotPoly, ProverOpeningData,
};
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::AkitaSerialize;
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    AkitaBatchedProof, AkitaCommitmentHint, AkitaVerifierSetup, BasisMode, BlockOrder,
    FpExtEncoding, LevelParams, OpeningClaims, PointVariableSelection, PolynomialGroupClaims,
    RingCommitment, Schedule, SetupContributionMode,
};
use akita_verifier::CommitmentVerifier;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const ONEHOT_K: usize = 256;

fn prover_claims<'a, E: FieldCore, P, CommitF: FieldCore, const D: usize>(
    point: &'a [E],
    polynomials: &'a [&'a P],
    commitment: &'a RingCommitment<CommitF, D>,
    hint: AkitaCommitmentHint<CommitF, D>,
) -> ProverOpeningData<'a, E, P, CommitF, D> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).expect("full-point prover group"),
        vec![E::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims =
        OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("valid prover claims");
    ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
        .expect("valid prover opening data")
}

fn verifier_claims<'a, E: FieldCore, C>(
    point: &[E],
    openings: &[E],
    commitment: &'a C,
) -> OpeningClaims<'static, E, &'a C> {
    OpeningClaims::from_groups(
        point.to_vec(),
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(point.len(), point.len()).expect("full-point group"),
            openings.to_vec(),
            commitment,
        )
        .expect("valid verifier claims group")],
    )
    .expect("valid verifier claims")
}

fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
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

fn opening_from_poly<'a, FF, const D: usize, P>(
    poly: &'a P,
    point: &[FF],
    layout: &LevelParams,
    basis: BasisMode,
) -> FF
where
    FF: CanonicalField,
    P: RootProvePoly<FF, D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, FF, D>,
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

    let opening = OpeningFoldKernel::<P::OpeningView<'a>, FF, D>::evaluate_and_fold(
        &CpuBackend,
        None,
        poly.opening_view().expect("opening view"),
        akita_prover::compute::OpeningFoldPlan::Base {
            eval_outer_scalars: &ring_opening_point.b,
            fold_scalars: &ring_opening_point.a,
            block_len: layout.block_len,
        },
    )
    .expect("evaluate_and_fold");
    let folded_ring = opening.eval;
    let packed_inner = reduce_inner_opening_to_ring_element::<FF, D>(inner_point, basis)
        .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}

fn run_fold_linf_prove_verify<FF, P, const D: usize, Cfg>(
    setup: &AkitaProverSetup<FF, D>,
    poly: &P,
    point: &[Cfg::ExtField],
    opening: Cfg::ExtField,
) -> Vec<FoldGrindObservation>
where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + PseudoMersenneField
        + HasWide
        + AkitaSerialize
        + 'static,
    P: RootCommitPoly<FF, D> + RootProvePoly<FF, D>,
    Cfg: CommitmentConfig<Field = FF>,
    Cfg::ExtField:
        ExtField<FF> + FieldCore + FrobeniusExtField<FF> + FpExtEncoding<FF> + AkitaSerialize,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ProverSetup = AkitaProverSetup<FF, D>,
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
        >,
    CpuBackend: ComputeBackendSetup<FF>
        + RootCommitBackend<FF, P, Cfg::ExtField, D>
        + RecursiveProveBackend<FF, P, Cfg::ExtField, D>,
{
    type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

    let prepared = CpuBackend.prepare_setup(setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    let (commitment, hint) = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::batched_commit(
        setup,
        std::slice::from_ref(poly),
        &stack,
    )
    .unwrap();

    let poly_refs = [poly];

    let commitments = [commitment];
    let openings = [opening];
    let setup_contribution_mode = SetupContributionMode::Direct;

    let mut prover_transcript = AkitaTranscript::<FF>::new(b"fold_linf_stats");
    let _grind_observer = FoldGrindObserverGuard::install();
    let proof = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::batched_prove(
        setup,
        prover_claims(point, &poly_refs, &commitments[0], hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        setup_contribution_mode,
    )
    .unwrap();
    let grind_observations = FoldGrindObserverGuard::take();

    let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::setup_verifier(setup);
    let mut verifier_transcript = AkitaTranscript::<FF>::new(b"fold_linf_stats");
    <Scheme<D, Cfg> as CommitmentVerifier<FF, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(point, &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        setup_contribution_mode,
    )
    .unwrap_or_else(|e| panic!("fold_linf_stats verify failed: {e}"));

    grind_observations
}

pub(crate) fn run_onehot_fold_linf_sample<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    nv: usize,
    layout: &LevelParams,
    _plan: &Schedule,
    seed: u64,
) -> Vec<FoldGrindObservation>
where
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
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
        >,
    Cfg::ExtField:
        ExtField<FF> + FieldCore + FrobeniusExtField<FF> + FpExtEncoding<FF> + AkitaSerialize,
{
    type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

    let mut rng = StdRng::seed_from_u64(seed);
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
    let pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut rng);
    let opening = if let Some(base_pt) = degree_one_claim_point_to_base::<FF, Cfg::ExtField>(&pt) {
        Cfg::ExtField::lift_base(opening_from_poly(
            &onehot_poly,
            &base_pt,
            layout,
            BasisMode::Lagrange,
        ))
    } else {
        onehot_lagrange_opening::<FF, Cfg::ExtField, u8, D>(&onehot_poly, &pt)
    };

    let setup = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::setup_prover(nv, 1).unwrap();
    run_fold_linf_prove_verify::<FF, OneHotPoly<FF, D, u8>, D, Cfg>(
        &setup,
        &onehot_poly,
        &pt,
        opening,
    )
}

pub(crate) fn run_dense_fold_linf_sample<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    nv: usize,
    layout: &LevelParams,
    _plan: &Schedule,
    seed: u64,
) -> Vec<FoldGrindObservation>
where
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
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
        >,
    Cfg::ExtField:
        ExtField<FF> + FieldCore + FrobeniusExtField<FF> + FpExtEncoding<FF> + AkitaSerialize,
{
    type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

    let mut rng = StdRng::seed_from_u64(seed);
    let original_pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut rng);
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
    let opening =
        if let Some(base_pt) = degree_one_claim_point_to_base::<FF, Cfg::ExtField>(&original_pt) {
            Cfg::ExtField::lift_base(opening_from_poly(
                &poly,
                &base_pt,
                layout,
                BasisMode::Lagrange,
            ))
        } else {
            dense_lagrange_opening_from_evals::<FF, Cfg::ExtField>(&evals, &original_pt)
        };

    let setup = <Scheme<D, Cfg> as CommitmentProver<FF, D>>::setup_prover(
        RootPolyShape::num_vars(&poly),
        1,
    )
    .unwrap();
    run_fold_linf_prove_verify::<FF, DensePoly<FF, D>, D, Cfg>(&setup, &poly, &original_pt, opening)
}
