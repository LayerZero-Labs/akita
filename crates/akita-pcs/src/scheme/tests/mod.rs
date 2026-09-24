use super::*;

fn workspace_scheme<C>() -> Result<AkitaCommitmentScheme<C>, AkitaError>
where
    C: CommitmentConfig,
    C::Field: Field + CanonicalEncoding + Unreduced + PseudoMersenne + Valid + AkitaSerialize,
    C::ExtField: FpExtEncoding<C::Field>,
    C::ExtField: ExtField<C::Field> + Ring + Unreduced + Fold + AkitaSerialize,
{
    Ok(AkitaCommitmentScheme::new(
        akita_config::test_support::workspace_schedule_catalog::<C>()?,
    ))
}
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_cpu_backend::evaluate_root_polynomial;
use akita_cpu_backend::CpuBackend;
use akita_cpu_backend::{DensePoly, OneHotPoly};
use akita_prover::CommitmentHandleMetadata;
use akita_prover::SelectedProverOpeningData;
use akita_serialization::AkitaSerialize;
use akita_types::lagrange_weights;
use akita_types::CommittedGroupParams;
use akita_types::{
    CommittedGroup, CommittedGroupBatchProfile, GroupBatchStatement, OpeningClaims,
    OpeningClaimsLayout, PolynomialGroupClaims,
};
use jolt_field::{One, Ring, Zero};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
type Cfg = fp128::Dense;
type F = fp128::Field;
const D: usize = 512;
type Scheme = AkitaCommitmentScheme<Cfg>;

type OneHotF = fp128::Field;
type OneHotCfg = fp128::OneHot;
const ONEHOT_D: usize = 256;
type OneHotScheme = AkitaCommitmentScheme<OneHotCfg>;

fn onehot_source_chunk_size<C: CommitmentConfig>() -> usize {
    akita_config::unit_onehot_source_chunk_size::<C>()
        .expect("one-hot fixture requires a unit-one-hot commitment config")
}

#[test]
fn scheme_owns_one_catalog_for_setup_and_row_resolution() {
    let workspace_catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace schedule catalog");
    let artifact = workspace_catalog
        .to_artifact_bytes()
        .expect("schedule artifact");
    let scheme = Scheme::from_schedule_artifact(&artifact).expect("artifact-backed scheme");

    assert_eq!(
        scheme.schedules().catalog_digest(),
        workspace_catalog.catalog_digest()
    );
    let key =
        akita_types::AkitaScheduleLookupKey::single(akita_types::PolynomialGroupLayout::new(14, 1));
    assert_eq!(
        scheme
            .schedules()
            .resolve_key(&key)
            .expect("artifact row")
            .selection(),
        workspace_catalog
            .resolve_key(&key)
            .expect("artifact row")
            .selection()
    );

    let expected_capacity =
        akita_config::SetupRequirements::from_catalog::<Cfg>(scheme.schedules(), 14, 1)
            .map(|requirements| requirements.matrix_capacity)
            .expect("catalog setup capacity");
    let setup = scheme.setup_prover(14, 1).expect("catalog-backed setup");
    assert!(
        setup.expanded.shared_matrix().num_field_elements() >= expected_capacity.num_field_elements,
        "setup must cover the exact catalog-derived matrix requirement"
    );
}

#[test]
fn scheme_rejects_a_catalog_bound_to_another_config() {
    let dense = akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("dense schedule catalog");
    let error = akita_config::TrustedScheduleCatalog::<OneHotCfg>::new(dense.catalog().clone())
        .expect_err("one-hot config must reject a dense catalog");
    assert!(error.to_string().contains("family"));
}

/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

mod batched;
mod coefficient_packing;
mod cross_mode;
mod dense_group;
#[cfg(feature = "response-model-diagnostics")]
mod diagnostics;
mod layout;
mod onehot;
mod single;

fn selected_prover_data<'a, C, S>(
    scheme: &AkitaCommitmentScheme<C>,
    claims: OpeningClaims<'a, C::ExtField, CommittedGroup<C::Field>>,
    prover_states: Vec<S>,
) -> Result<SelectedProverOpeningData<'a, C::ExtField, S, C::Field>, AkitaError>
where
    C: CommitmentConfig,
    S: CommitmentHandleMetadata,
{
    SelectedProverOpeningData::from_committed_claims::<C>(claims, prover_states, &scheme.schedules)
}

fn selected_statement<'a, C>(
    scheme: &AkitaCommitmentScheme<C>,
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
    let selection = scheme.schedules.resolve_profiles(&profiles)?.selection();
    GroupBatchStatement::new(selection, claims)
}

/// Batched recursion already consults the byte planner before folding
/// again. The runtime safety guard here only needs to catch tiny tails and
/// fixed points, not enforce the single-proof shrink-ratio heuristic.
fn should_stop_batched_folding(witness_len: usize, prev_w_len: usize) -> bool {
    witness_len <= MIN_W_LEN_FOR_FOLDING || witness_len >= prev_w_len
}

fn prover_claims<'a, S>(
    scheme: &Scheme,
    point: &'a [F],
    evaluations: &[F],
    commitment: &'a CommittedGroup<F>,
    private_handle: S,
) -> SelectedProverOpeningData<'a, F, S, F>
where
    S: CommitmentHandleMetadata,
{
    let group =
        PolynomialGroupClaims::new(point.to_vec(), evaluations.to_vec(), commitment.clone())
            .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    SelectedProverOpeningData::from_committed_claims::<Cfg>(
        opening_claims,
        vec![private_handle],
        scheme.schedules(),
    )
    .expect("valid prover opening data")
}

fn verifier_claims<'a>(
    scheme: &Scheme,
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
    selected_statement::<Cfg>(scheme, claims).expect("valid verifier statement")
}

fn make_dense_poly(num_vars: usize) -> (DensePoly<F>, Vec<F>) {
    let len = 1usize << num_vars;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, &evals).unwrap();
    (poly, evals)
}

fn singleton_layout<C: CommitmentConfig>(
    scheme: &AkitaCommitmentScheme<C>,
    num_vars: usize,
) -> CommittedGroupParams {
    catalog_root_layout(scheme, num_vars, 1)
}

fn catalog_root_layout<C: CommitmentConfig>(
    scheme: &AkitaCommitmentScheme<C>,
    num_vars: usize,
    num_polynomials: usize,
) -> CommittedGroupParams {
    let key = akita_types::AkitaScheduleLookupKey::single(akita_types::PolynomialGroupLayout::new(
        num_vars,
        num_polynomials,
    ));
    scheme
        .schedules
        .resolve_key(&key)
        .expect("catalog root layout")
        .schedule()
        .root
        .params
        .clone()
}

fn catalog_profile<C: CommitmentConfig>(
    scheme: &AkitaCommitmentScheme<C>,
    group: akita_types::PolynomialGroupLayout,
) -> akita_types::GroupCommitPhaseParams {
    scheme
        .schedules
        .resolve_key(&akita_types::AkitaScheduleLookupKey::single(group))
        .expect("catalog profile")
        .profiles()
        .final_group
}

type VerifyFixture = (
    Scheme,
    AkitaVerifierSetup<F>,
    CommittedGroup<F>,
    Vec<u8>,
    Vec<F>,
    F,
    CommittedGroupParams,
);

fn make_verify_fixture(num_vars: usize) -> VerifyFixture {
    let scheme = workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout(&scheme, num_vars);
    let full_num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;

    let (poly, evals) = make_dense_poly(full_num_vars);
    let setup = scheme.setup_prover(full_num_vars, 1).unwrap();
    let stack =
        CpuBackend::<Cfg>::new(setup.expanded.clone(), scheme.schedules()).expect("backend");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    let akita_cpu_backend::CommitOutput {
        committed_group: commitment,
        private_handle: prover_state,
    } = stack
        .commit(
            &stack.import_source(vec![poly.clone()]).expect("source"),
            akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();

    let opening_point: Vec<F> = (0..full_num_vars)
        .map(|i| F::from_u64((i + 2) as u64))
        .collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let commitments = [commitment];

    let proof = scheme
        .batched_prove(
            &setup,
            prover_claims(
                &scheme,
                &opening_point[..],
                &[opening],
                &commitments[0],
                prover_state,
            ),
            &stack,
            b"test/prove",
            BasisMode::Lagrange,
        )
        .unwrap();

    let [commitment] = commitments;
    (
        scheme,
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
    _ring_dimension: usize,
    seed: u64,
) -> OneHotPoly<OneHotF, u8> {
    let onehot_k = onehot_source_chunk_size::<OneHotCfg>();
    assert!(
        onehot_k <= usize::from(u8::MAX) + 1,
        "test u8 one-hot fixture cannot represent chunk size {onehot_k}"
    );
    let total_field = 1usize << num_vars;
    let total_chunks = total_field / onehot_k;

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();

    OneHotPoly::<OneHotF, u8>::new(onehot_k, indices).expect("debug onehot poly")
}

fn debug_random_point(nv: usize) -> Vec<OneHotF> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| OneHotF::from_u128_reduced(rng.r#gen::<u128>()))
        .collect()
}

fn opening_from_poly_at<const D_OPEN: usize>(
    poly: &OneHotPoly<OneHotF, u8>,
    point: &[OneHotF],
    num_positions_per_block: usize,
    num_live_blocks: usize,
) -> OneHotF
where
    OneHotPoly<OneHotF, u8>: akita_cpu_backend::RootPolynomialEvaluator<OneHotF, D_OPEN>,
{
    evaluate_root_polynomial::<OneHotF, _, D_OPEN>(
        poly,
        point,
        num_positions_per_block,
        num_live_blocks,
        BasisMode::Lagrange,
    )
    .expect("root polynomial opening")
}

fn opening_from_poly(
    poly: &OneHotPoly<OneHotF, u8>,
    point: &[OneHotF],
    ring_dimension: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
) -> OneHotF {
    akita_types::dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        OneHotF,
        ring_dimension,
        |D_OPEN| Ok(opening_from_poly_at::<D_OPEN>(
            poly,
            point,
            num_positions_per_block,
            num_live_blocks,
        ))
    )
    .expect("supported one-hot opening ring dimension")
}
