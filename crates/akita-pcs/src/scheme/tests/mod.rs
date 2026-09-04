use super::*;

fn workspace_scheme<C>() -> Result<AkitaCommitmentScheme<C>, AkitaError>
where
    C: CommitmentConfig,
    C::Field: Field + CanonicalEncoding + Unreduced + PseudoMersenne + Valid + AkitaSerialize,
    C::ExtField: FpExtEncoding<C::Field>,
    C::ExtField: ExtField<C::Field> + Ring + Unreduced + Fold + AkitaSerialize,
{
    AkitaCommitmentScheme::new(akita_config::test_support::workspace_schedule_catalog::<C>()?)
}
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource};
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_prover::{DensePoly, OneHotPoly, PreparedProverGroup, SelectedProverOpeningData};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::CommittedGroupParams;
use akita_types::DigitRangePlan;
use akita_types::ExtensionOpeningReductionProof;
use akita_types::{
    lagrange_weights, reduce_inner_opening_to_ring_element, ring_opening_point_from_field, RingVec,
};
use akita_types::{
    AkitaBatchedProofShape, LevelProofShape, NextWitnessBindingShape, TerminalLevelProofShape,
};
use akita_types::{
    AkitaCommitmentHint, CommittedGroup, CommittedGroupBatchProfile, GroupBatchStatement,
    OpeningClaims, OpeningClaimsLayout, PolynomialGroupClaims,
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
    let error = OneHotScheme::new(dense).expect_err("one-hot scheme must reject dense catalog");
    assert!(error.to_string().contains("family"));
}

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
mod coefficient_packing;
mod cross_mode;
mod dense_group;
mod layout;
mod onehot;
mod single;

fn selected_prover_data<'a, C, P>(
    scheme: &AkitaCommitmentScheme<C>,
    claims: OpeningClaims<'a, C::ExtField, CommittedGroup<C::Field>>,
    hints: Vec<AkitaCommitmentHint<C::Field>>,
    polynomials: Vec<&'a [&'a P]>,
) -> Result<HomogeneousSelectedProverData<'a, C, P>, AkitaError>
where
    C: CommitmentConfig,
    P: akita_prover::RootPolyMeta<C::Field>,
{
    SelectedProverOpeningData::from_committed_claims::<C>(
        claims,
        hints,
        polynomials,
        scheme.schedules.as_ref(),
    )
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

fn prover_claims<'a, P>(
    scheme: &Scheme,
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
    selected_prover_data::<Cfg, _>(scheme, opening_claims, vec![hint], vec![polynomials])
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
    AkitaBatchedProof<F, F>,
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
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
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

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof = scheme
        .batched_prove::<_, _, _>(
            &setup,
            prover_claims(
                &scheme,
                &opening_point[..],
                &poly_refs[..],
                &commitments[0],
                hint,
            ),
            &stack,
            &mut prover_transcript,
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

fn batched_shape_rounds(level_d: usize, output_witness_len: usize) -> usize {
    let num_ring_elems = output_witness_len.div_ceil(level_d);
    num_ring_elems.next_power_of_two().trailing_zeros() as usize + level_d.trailing_zeros() as usize
}

/// Derive the structural proof shape from the schedule. The terminal carries
/// only optional EOR and the clear terminal response; nonces are proof-level.
fn expected_same_point_batched_shape(
    scheme: &OneHotScheme,
    max_num_vars: usize,
    num_claims: usize,
    proof: &AkitaBatchedProof<OneHotF, OneHotF>,
) -> AkitaBatchedProofShape {
    let opening_batch =
        akita_types::OpeningClaimsLayout::new(max_num_vars, num_claims).expect("opening_batch");
    let key = akita_types::AkitaScheduleLookupKey::single(
        opening_batch
            .root_final_group_layout()
            .expect("batched root group layout"),
    );
    let schedule = scheme
        .schedules()
        .resolve_key(&key)
        .expect("batched root runtime plan")
        .schedule()
        .clone();
    let root_step = &schedule.root;
    let root_params = &root_step.params;
    let num_fold_levels = schedule.num_fold_levels();
    let root_rounds = batched_shape_rounds(root_params.d_a(), root_step.output_witness_len);

    assert!(
        num_fold_levels >= 2,
        "folded-only schedules have a root and terminal fold"
    );

    let root_successor = schedule.recursive_folds.first();
    let opening_payload_coeffs = |params: &akita_types::CommittedGroupParams| {
        params
            .opening_payload_geometry()
            .expect("opening payload geometry")
            .transmitted_coefficients()
    };
    let commitment_payload_coeffs = |params: &akita_types::CommittedGroupParams| {
        params
            .outer_payload_geometry()
            .expect("commitment payload geometry")
            .transmitted_coefficients()
    };
    let root_stage1 = DigitRangePlan::new(1usize << root_params.open().digits.log_basis)
        .expect("scheduled root range basis")
        .proof_shapes_for_route(root_rounds, root_params.inner().matrix.security_route())
        .expect("scheduled root Stage 1 shape");
    let root_shape = LevelProofShape {
        extension_opening_reduction: None,
        opening_payload_coeffs: opening_payload_coeffs(root_params),
        stage1_stages: root_stage1.0,
        stage1_norm: root_stage1.1,
        stage2_sumcheck_proof: vec![3; root_rounds],
        stage3_sumcheck: None,
        next_witness_binding: match root_successor {
            Some(successor) => {
                let next_level_params = &successor.params;
                NextWitnessBindingShape::OuterPayload {
                    coeffs: commitment_payload_coeffs(next_level_params),
                }
            }
            None => NextWitnessBindingShape::TerminalInnerState,
        },
    };
    // After Phase 1, the recursive suffix has `num_fold_levels - 1` steps in
    // total: `num_fold_levels - 2` intermediate steps followed by exactly one
    // terminal step. (We've already consumed the root.)
    let mut recursive_folds = Vec::with_capacity(schedule.recursive_folds.len());
    let mut input_witness_len = root_step.output_witness_len;
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        assert_eq!(step.input_witness_len, input_witness_len);
        let level_params = &step.params;
        let output_witness_len = step.output_witness_len;
        let rounds = batched_shape_rounds(level_params.d_a(), output_witness_len);
        let stage1 = DigitRangePlan::new(1usize << level_params.open().digits.log_basis)
            .expect("scheduled range basis")
            .proof_shapes_for_route(rounds, level_params.inner().matrix.security_route())
            .expect("scheduled Stage 1 shape");
        recursive_folds.push(LevelProofShape {
            extension_opening_reduction: None,
            opening_payload_coeffs: opening_payload_coeffs(level_params),
            stage1_stages: stage1.0,
            stage1_norm: stage1.1,
            stage2_sumcheck_proof: vec![3; rounds],
            stage3_sumcheck: None,
            next_witness_binding: match schedule.recursive_folds.get(index + 1) {
                Some(successor) => {
                    let next_level_params = &successor.params;
                    NextWitnessBindingShape::OuterPayload {
                        coeffs: commitment_payload_coeffs(next_level_params),
                    }
                }
                None => NextWitnessBindingShape::TerminalInnerState,
            },
        });
        input_witness_len = output_witness_len;
    }
    // Terminal fold step (always present in the multi-fold case); the
    // structural terminal field encodes its witness shape.
    assert_eq!(schedule.terminal.input_witness_len, input_witness_len);
    let terminal = TerminalLevelProofShape {
        extension_opening_reduction: None,
        terminal_response: schedule.terminal.response_shape.clone(),
    };
    AkitaBatchedProofShape {
        nonce_stream_bits: proof.nonce_stream.bit_len(),
        root: root_shape,
        recursive_folds,
        terminal,
    }
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
) -> OneHotF {
    let alpha_bits = D_OPEN.trailing_zeros() as usize;
    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        num_positions_per_block,
        num_live_blocks,
        BasisMode::Lagrange,
    )
    .expect("opening point shape should match layout");
    let opening = OpeningFoldKernel::<_, OneHotF, D_OPEN>::evaluate_and_fold(
        &CpuBackend::DEFAULT,
        None,
        poly.opening_view().expect("opening view"),
        OpeningFoldPlan::Base {
            live_block_weights: &ring_opening_point.live_block_weights,
            position_weights: &ring_opening_point.position_weights,
            num_positions_per_block,
        },
    )
    .expect("evaluate_and_fold");
    let folded_ring = opening.eval;
    let packed_inner =
        reduce_inner_opening_to_ring_element::<OneHotF, D_OPEN>(inner_point, BasisMode::Lagrange)
            .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
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
