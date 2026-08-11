#![cfg(any(feature = "schedules-default", feature = "profile-ci"))]

use super::*;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_prover::compute::RootOpeningSource;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_prover::{
    DensePoly, OneHotPoly, PreparedProverGroup, ProverOpeningData, SelectedProverOpeningData,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::AkitaTranscript;
use akita_types::CommittedGroupParams;
use akita_types::ExtensionOpeningReductionProof;
use akita_types::{lagrange_weights, RingVec};
use akita_types::{
    AkitaCommitmentHint, CommittedGroup, CommittedGroupBatchProfile, GroupBatchStatement,
    OpeningClaims, OpeningClaimsLayout, PolynomialGroupClaims,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
type Cfg = fp128::Dense;
type F = fp128::Field;
const D: usize = Cfg::D;
type Scheme = AkitaCommitmentScheme<Cfg>;

type OneHotF = fp128::Field;
type OneHotCfg = fp128::OneHot;
const ONEHOT_D: usize = OneHotCfg::D;
// `fp128::OneHot` uses K=256 one-hot chunks at its root ring dimension.
const BENCH_ONEHOT_K: usize = 256;
type OneHotScheme = AkitaCommitmentScheme<OneHotCfg>;
type HomogeneousSelectedProverData<'a, C, P> = SelectedProverOpeningData<
    'a,
    <C as CommitmentConfig>::ExtField,
    PreparedProverGroup<'a, P>,
    <C as CommitmentConfig>::Field,
>;
/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

mod batched;
mod dense_group;
mod layout;
mod onehot;
mod single;

fn selected_prover_data<'a, C, P>(
    claims: OpeningClaims<'a, C::ExtField, CommittedGroup<C::Field>>,
    hints: Vec<AkitaCommitmentHint<C::Field>>,
    polynomials: Vec<&'a [&'a P]>,
) -> Result<HomogeneousSelectedProverData<'a, C, P>, AkitaError>
where
    C: CommitmentConfig,
    P: akita_prover::RootPolyMeta<C::Field>,
{
    let profiles = batch_profiles::<C>(&claims)?;
    let selection = C::select_schedule_for_profiles(&profiles)?.selection();
    Ok((
        selection,
        ProverOpeningData::new(claims, hints, polynomials)?,
    ))
}

fn selected_statement<'a, C>(
    claims: OpeningClaims<'a, C::ExtField, &'a CommittedGroup<C::Field>>,
) -> Result<GroupBatchStatement<'a, C::ExtField, C::Field>, AkitaError>
where
    C: CommitmentConfig,
{
    let (final_group, precommitteds) = claims
        .groups()
        .split_last()
        .ok_or_else(|| AkitaError::InvalidInput("opening statement requires a group".into()))?;
    let profiles = CommittedGroupBatchProfile {
        final_group: *final_group.commitment().profile(),
        precommitteds: precommitteds
            .iter()
            .map(|group| *group.commitment().profile())
            .collect(),
    };
    let selection = C::select_schedule_for_profiles(&profiles)?.selection();
    GroupBatchStatement::new(selection, claims)
}

fn batch_profiles<C>(
    claims: &OpeningClaims<'_, C::ExtField, CommittedGroup<C::Field>>,
) -> Result<CommittedGroupBatchProfile, AkitaError>
where
    C: CommitmentConfig,
{
    let (final_group, precommitteds) = claims
        .groups()
        .split_last()
        .ok_or_else(|| AkitaError::InvalidInput("opening data requires a group".into()))?;
    Ok(CommittedGroupBatchProfile {
        final_group: *final_group.commitment().profile(),
        precommitteds: precommitteds
            .iter()
            .map(|group| *group.commitment().profile())
            .collect(),
    })
}

/// Batched recursion already consults the byte planner before folding
/// again. The runtime safety guard here only needs to catch tiny tails and
/// fixed points, not enforce the single-proof shrink-ratio heuristic.
fn should_stop_batched_folding(witness_len: usize, prev_w_len: usize) -> bool {
    witness_len <= MIN_W_LEN_FOR_FOLDING || witness_len >= prev_w_len
}

fn prover_claims<'a, P>(
    point: &'a [F],
    polynomials: &'a [&'a P],
    commitment: &'a CommittedGroup<F>,
    hint: AkitaCommitmentHint<F>,
) -> SelectedProverOpeningData<'a, F, PreparedProverGroup<'a, P>, F>
where
    P: akita_prover::RootPolyMeta<F>,
{
    let group = PolynomialGroupClaims::new(
        point.to_vec(),
        vec![F::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    selected_prover_data::<Cfg, _>(opening_claims, vec![hint], vec![polynomials])
        .expect("valid prover opening data")
}

fn verifier_claims<'a>(
    point: &[F],
    openings: &[F],
    commitment: &'a CommittedGroup<F>,
) -> GroupBatchStatement<'a, F, F> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        openings.to_vec(),
        commitment,
    )
    .expect("valid verifier claims group")])
    .expect("valid verifier claims");
    selected_statement::<Cfg>(claims).expect("valid verifier statement")
}

fn make_dense_poly(num_vars: usize) -> (DensePoly<F>, Vec<F>) {
    let len = 1usize << num_vars;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &evals).unwrap();
    (poly, evals)
}

fn singleton_layout<C: CommitmentConfig>(num_vars: usize) -> CommittedGroupParams {
    let opening_batch = OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
    C::get_params_for_batched_commitment(&opening_batch).expect("singleton commitment layout")
}

type VerifyFixture = (
    AkitaVerifierSetup<F>,
    CommittedGroup<F>,
    AkitaBatchedProof<F, F>,
    Vec<F>,
    F,
    CommittedGroupParams,
);

fn make_verify_fixture(num_vars: usize) -> VerifyFixture {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(num_vars);
    let full_num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;

    let (poly, evals) = make_dense_poly(full_num_vars);
    let setup = Scheme::setup_prover(full_num_vars, 1).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
    let (commitment, hint) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack).unwrap();

    let opening_point: Vec<F> = (0..full_num_vars)
        .map(|i| F::from_u64((i + 2) as u64))
        .collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof = Scheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&opening_point[..], &poly_refs[..], &commitments[0], hint),
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
        opening_point,
        opening,
        layout,
    )
}

fn debug_make_onehot_poly(
    num_vars: usize,
    ring_dimension: usize,
    seed: u64,
) -> OneHotPoly<OneHotF, u8> {
    let total_field = 1usize << num_vars;
    let total_chunks = total_field / BENCH_ONEHOT_K;

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..BENCH_ONEHOT_K) as u8))
        .collect();

    OneHotPoly::<OneHotF, u8>::new(BENCH_ONEHOT_K, ring_dimension, indices)
        .expect("debug onehot poly")
}
