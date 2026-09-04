use super::*;
use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{CommitInnerPlan, ComputeBackendSetup, DigitRowsComputeBackend, OperationCtx};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::{AkitaProverSetup, CpuBackend, DensePoly};
use akita_challenges::SparseChallengeConfig;
use akita_types::sis::{
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, SisMatrixRole, SisTableDigest,
    SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
};
use akita_types::{
    CommittedSourceEncoding, CompressionChainPlan, GroupCommitPhaseParams, InnerCommitMatrixParams,
    OpenCommitMatrixParams, OpeningMethod, OuterCommitMatrixParams, PolynomialGroupLayout, RingVec,
    SetupMatrixCapacity, SisModulusProfileId,
};
use jolt_field::Fp64;

type F = Fp64<4294967197>;
const D: usize = 64;

fn audited_commit_params(
    slice_count: akita_types::CommitmentSliceCount,
    positions_per_block: usize,
    live_ring_elements: usize,
    num_digits_inner: usize,
    num_digits_outer: usize,
    num_digits_open: usize,
) -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        3,
        1,
        1,
        1,
        SparseChallengeConfig::production_for_ring_dim(D)
            .expect("D=64 has a production challenge configuration"),
    );
    params.own_group_mut().profile.outer_slice_count = slice_count;
    params = params
        .with_decomp(
            positions_per_block,
            live_ring_elements,
            num_digits_inner,
            num_digits_outer,
            num_digits_open,
        )
        .expect("commitment fixture geometry");
    let source_len = live_ring_elements
        .checked_mul(D)
        .expect("commitment fixture source length");
    assert!(source_len.is_power_of_two());
    params.own_group_mut().profile.group =
        PolynomialGroupLayout::singleton(source_len.trailing_zeros() as usize);

    let a_bucket = rounded_up_role_a_inf_norm(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q32Offset99,
        D,
        params.open().digits.log_basis,
        &params.fold_challenge_config(),
        params.num_digits_fold(),
        params.witness_chunk.num_chunks,
    )
    .expect("audited fixture A bucket");
    params.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            policy: DEFAULT_SIS_SECURITY_POLICY,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q32Offset99,
            role: SisMatrixRole::Inner,
            ring_dimension: D as u32,
            coeff_linf_bound: a_bucket,
        },
        params.inner().matrix.input_width(),
    )
    .expect("audited fixture A matrix");
    params = params
        .with_decomp(
            positions_per_block,
            live_ring_elements,
            num_digits_inner,
            num_digits_outer,
            num_digits_open,
        )
        .expect("fixture geometry after A rank");

    let b_bucket = rounded_up_collision_inf_norm(
        DEFAULT_SIS_SECURITY_POLICY,
        SisModulusProfileId::Q32Offset99,
        SisMatrixRole::Outer,
        D,
        params.outer().digits.log_basis,
    )
    .expect("audited fixture B bucket");
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            policy: DEFAULT_SIS_SECURITY_POLICY,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q32Offset99,
            role: SisMatrixRole::Outer,
            ring_dimension: D as u32,
            coeff_linf_bound: b_bucket,
        },
        params.outer().matrix.input_width(),
    )
    .expect("audited fixture B matrix");

    let d_bucket = rounded_up_collision_inf_norm(
        DEFAULT_SIS_SECURITY_POLICY,
        SisModulusProfileId::Q32Offset99,
        SisMatrixRole::Open,
        D,
        params.open().digits.log_basis,
    )
    .expect("audited fixture D bucket");
    params.open_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            policy: DEFAULT_SIS_SECURITY_POLICY,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q32Offset99,
            role: SisMatrixRole::Open,
            ring_dimension: D as u32,
            coeff_linf_bound: d_bucket,
        },
        params.open().matrix.input_width(),
    )
    .expect("audited fixture D matrix");
    params
}

#[test]
fn commit_level_params_reject_log_basis_above_i8_range() {
    let expanded = AkitaProverSetup::<F>::generate_with_capacity(
        5,
        1,
        SetupMatrixCapacity {
            num_field_elements: D,
        },
    )
    .unwrap()
    .expanded;
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        9,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(2, 4, 2, 2, 2)
    .unwrap();

    assert!(matches!(
        validate_commit_level_params::<F>(&params, &expanded, 0, 1),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn commit_level_params_do_not_charge_unused_shared_d_footprint() {
    let mut params = audited_commit_params(akita_types::CommitmentSliceCount::ONE, 1, 1, 1, 1, 1);
    let d_key = params.open().matrix.sis_table_key();
    params.open_matrix = OpenCommitMatrixParams::new_unchecked(
        d_key.policy,
        d_key.table_digest,
        d_key.modulus_profile,
        8,
        8,
        d_key.coeff_linf_bound,
        D,
    );
    let commit_only_fields = akita_types::commit_only_setup_field_elements(
        &params.inner().matrix,
        &params.outer().matrix,
        params.outer_slice_count(),
    )
    .unwrap();
    let expanded = AkitaProverSetup::<F>::generate_with_capacity(
        5,
        1,
        SetupMatrixCapacity {
            num_field_elements: commit_only_fields,
        },
    )
    .unwrap()
    .expanded;

    validate_commit_level_params::<F>(&params, &expanded, 0, 1)
        .expect("standalone commitment only materializes A and B");
}

fn sliced_commit_params() -> CommittedGroupParams {
    audited_commit_params(akita_types::CommitmentSliceCount::FOUR, 2, 16, 1, 1, 1)
}

fn set_outer_width(params: &mut CommittedGroupParams, input_width: usize) {
    let key = params.outer().matrix.sis_table_key();
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        key.policy,
        key.table_digest,
        key.modulus_profile,
        params.outer().matrix.output_rank(),
        input_width,
        key.coeff_linf_bound,
        params.outer().matrix.ring_dimension(),
    );
}

#[test]
fn commitment_request_binds_slice_count_and_exact_b_width() {
    let params = sliced_commit_params();
    params
        .validate_commitment_request(0, 1)
        .expect("canonical sliced geometry");

    let mut wrong_slice_count = params.clone();
    wrong_slice_count.own_group_mut().profile.outer_slice_count =
        akita_types::CommitmentSliceCount::ONE;
    assert!(matches!(
        wrong_slice_count.validate_commitment_request(0, 1),
        Err(AkitaError::InvalidSetup(_))
    ));

    let mut wrong_width = params.clone();
    set_outer_width(&mut wrong_width, params.outer().matrix.input_width() + 1);
    assert!(matches!(
        wrong_width.validate_commitment_request(0, 1),
        Err(AkitaError::InvalidSetup(_))
    ));
    assert!(matches!(
        params.validate_commitment_request(2, 1),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn commitment_request_binds_polynomial_count_in_both_directions() {
    let one_polynomial = sliced_commit_params();
    assert!(matches!(
        one_polynomial.validate_commitment_request(0, 2),
        Err(AkitaError::InvalidSetup(_))
    ));

    let mut two_polynomials = one_polynomial.clone();
    two_polynomials.own_group_mut().profile.group = PolynomialGroupLayout::new(10, 2);
    let geometry = akita_types::CommitmentSliceGeometry::try_new(
        two_polynomials.outer_slice_count(),
        two_polynomials.blocks().live_blocks,
        2,
        two_polynomials.inner().matrix.output_rank(),
        two_polynomials.outer().digits.num_digits,
        two_polynomials.role_dims().d_a(),
        two_polynomials.role_dims().d_b(),
    )
    .unwrap();
    set_outer_width(&mut two_polynomials, geometry.physical_input_width());
    two_polynomials
        .validate_commitment_request(0, 2)
        .expect("two-polynomial B geometry");
    assert!(matches!(
        two_polynomials.validate_commitment_request(0, 1),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn commit_b_input_len_rejects_overflow() {
    assert_eq!(checked_commit_b_input_len(3, 5).expect("fits"), 15);
    assert!(matches!(
        checked_commit_b_input_len(usize::MAX, 2),
        Err(AkitaError::InvalidInput(_))
    ));
}

/// Inner digit depth that actually represents an `Fp32` coefficient at
/// `log_basis_inner = 3`.
///
/// The fixture used to declare a single base-8 digit, which cannot represent a
/// 32-bit field element at all: the commitment silently truncated, and the test
/// only passed because the production and reference paths truncated identically.
/// The commit path now rejects a source outside its scheduled digit envelope, so
/// the fixture states a depth consistent with the coefficients it commits.
fn slice_fixture_num_digits_inner() -> usize {
    akita_types::sis::compute_num_digits_field_width(32, 3)
}

fn commitment_params_for_slice_count(
    slice_count: akita_types::CommitmentSliceCount,
) -> CommittedGroupParams {
    audited_commit_params(slice_count, 2, 16, slice_fixture_num_digits_inner(), 1, 1)
}

fn commit_unsliced_reference(
    polys: &[DensePoly<F>],
    ctx: &OperationCtx<'_, F, CpuBackend>,
    params: &CommittedGroupParams,
) -> Result<(CommitmentWithHint<F>, CompressionChainPlan), AkitaError> {
    let backend = ctx.backend();
    let prepared = ctx.prepared();
    let plan = CommitInnerPlan::from_level(params);
    let inners = compute_inner_commitment::<F, DensePoly<F>, _, D>(backend, prepared, polys, plan)?;
    if inners.len() != polys.len() {
        return Err(AkitaError::InvalidSetup(
            "unsliced reference inner commitment count mismatch".into(),
        ));
    }
    let prepared_polynomials = inners
        .into_iter()
        .map(|inner| {
            validate_commit_inner_shape::<F, D>(&inner, params.blocks().live_blocks, plan.n_a)?;
            let blocks = (0..params.blocks().live_blocks)
                .map(|block| inner.block_rows::<D>(block, plan.n_a))
                .collect::<Result<Vec<_>, _>>()?;
            let digits = decompose_commit_blocks_into::<F, D, D>(
                &blocks,
                params.outer().digits.num_digits,
                params.outer().digits.log_basis,
            )?;
            Ok((inner.into_inner_rows(), digits))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let geometry = akita_types::CommitmentSliceGeometry::try_new(
        akita_types::CommitmentSliceCount::ONE,
        params.blocks().live_blocks,
        polys.len(),
        params.inner().matrix.output_rank(),
        params.outer().digits.num_digits,
        D,
        D,
    )?;

    // Independent pre-slicing B input: concatenate complete polynomial planes
    // directly. The shipping path reaches the same input through its slice
    // iterator, which is deliberately not used to build this reference.
    let mut reference_b_input = Vec::with_capacity(geometry.physical_input_width());
    for (_, digits) in &prepared_polynomials {
        reference_b_input.extend_from_slice(digits.typed_planes::<D>()?);
    }
    if reference_b_input.len() != params.outer().matrix.input_width() {
        return Err(AkitaError::InvalidSetup(
            "unsliced reference B input width mismatch".into(),
        ));
    }
    let production_b_inputs = outer_slice_inputs::<D>(
        &prepared_polynomials
            .iter()
            .map(|(_, digits)| digits)
            .collect::<Vec<_>>(),
        &geometry,
    )?;
    if production_b_inputs.as_slice() != [reference_b_input.as_slice()] {
        return Err(AkitaError::InvalidSetup(
            "S=1 sliced input differs from the unsliced B input".into(),
        ));
    }

    let n_b = params.outer().matrix.output_rank();
    let mut reference_b_batches = backend.digit_rows::<D>(
        prepared,
        n_b,
        &[reference_b_input.as_slice()],
        params.outer().digits.log_basis,
    )?;
    if reference_b_batches.len() != 1 {
        return Err(AkitaError::InvalidSetup(
            "single B input did not produce one row batch".into(),
        ));
    }
    let reference_b_image = reference_b_batches.pop().expect("length checked");
    let production_b_image = commit_outer_slices::<F, _, D>(
        backend,
        prepared,
        n_b,
        prepared_polynomials.iter().map(|(_, digits)| digits),
        &geometry,
        params.outer().digits.log_basis,
    )?;
    if production_b_image != reference_b_image {
        return Err(AkitaError::InvalidSetup(
            "S=1 sliced B image differs from the unsliced image".into(),
        ));
    }

    let source = RingVec::from_ring_elems(&reference_b_image);
    let compression_plan = CompressionChainPlan::for_complete_source(
        params.outer().matrix.sis_table_key().modulus_profile,
        source.coeff_len(),
    )?;
    let (mut outputs, _) = execute_compression_chains(
        ctx,
        vec![CompressionExecutionInput {
            id: (),
            plan: compression_plan.clone(),
            coefficients: source.into_coeffs(),
            relation_mode: akita_types::RingRelationMode::QuotientLift,
        }],
    )?;
    let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
    let terminal_ring_dim = output
        .witness
        .plan()
        .maps()
        .last()
        .ok_or(AkitaError::InvalidProof)?
        .ring_dimension();
    let payload = RingVec::from_coeffs_with_ring_dim(
        output.terminal.coefficients().to_vec(),
        terminal_ring_dim,
    )?;
    let inner_rows = prepared_polynomials
        .into_iter()
        .map(|(rows, _)| rows)
        .collect::<Vec<_>>();
    let quotients = output.relation.into_quotient_lift()?;
    let hint = AkitaCommitmentHint::new_with_outer_compression(
        D,
        inner_rows,
        &output.witness,
        &quotients,
    )?;
    Ok(((Commitment::new(payload), hint), compression_plan))
}

fn commit_fixture_with_profile(
    polys: &[DensePoly<F>],
    ctx: &OperationCtx<'_, F, CpuBackend>,
    profile: GroupCommitPhaseParams,
) -> Result<CommitmentWithHint<F>, AkitaError> {
    let (inner_rows, source) = compute_inner_outer_commitment(polys, ctx, profile)?;
    let CommitmentCompressionOutput {
        payload,
        witness,
        quotients,
    } = compute_commitment_compression(
        ctx,
        profile.outer.matrix.sis_table_key().modulus_profile,
        source,
    )?;
    let hint = AkitaCommitmentHint::new_with_outer_compression(
        profile.inner.matrix.ring_dimension(),
        inner_rows,
        &witness,
        &quotients,
    )?;
    Ok((Commitment::new(payload), hint))
}

#[test]
fn s1_matches_real_unsliced_commitment_pipeline() {
    const NUM_VARS: usize = 10;
    let params = commitment_params_for_slice_count(akita_types::CommitmentSliceCount::ONE);
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        NUM_VARS,
        1,
        SetupMatrixCapacity {
            num_field_elements: 2_000_000,
        },
    )
    .expect("deterministic setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let ctx = OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("commit context");
    let evals = (0..1usize << NUM_VARS)
        .map(|index| F::from_u64(index as u64 + 1))
        .collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(NUM_VARS, &evals).expect("dense polynomial");

    validate_commit_level_params::<F>(&params, setup.expanded.as_ref(), 0, 1)
        .expect("production S=1 geometry");
    let production = commit_fixture_with_profile(
        std::slice::from_ref(&poly),
        &ctx,
        GroupCommitPhaseParams::try_from_params(params.group(), &params).expect("S=1 profile"),
    )
    .expect("production S=1 commitment");
    let (reference, compression_plan) =
        commit_unsliced_reference(std::slice::from_ref(&poly), &ctx, &params)
            .expect("independent unsliced commitment");

    assert_eq!(production.0, reference.0, "terminal payload must match");
    assert_eq!(production.1.inner_rows(), reference.1.inner_rows());
    assert_eq!(
        production
            .1
            .outer_compression_witness(&compression_plan)
            .expect("production compression witness"),
        reference
            .1
            .outer_compression_witness(&compression_plan)
            .expect("reference compression witness")
    );
    assert_eq!(
        production
            .1
            .outer_compression_quotients(&compression_plan)
            .expect("production compression quotients"),
        reference
            .1
            .outer_compression_quotients(&compression_plan)
            .expect("reference compression quotients")
    );
    assert_eq!(production.1, reference.1, "complete hint must match");

    for slice_count in akita_types::CommitmentSliceCount::ALL {
        let sliced_params = commitment_params_for_slice_count(slice_count);
        validate_commit_level_params::<F>(&sliced_params, setup.expanded.as_ref(), 0, 1)
            .unwrap_or_else(|error| {
                panic!("real S={} geometry failed: {error}", slice_count.get())
            });
        let (commitment, hint) = commit_fixture_with_profile(
            std::slice::from_ref(&poly),
            &ctx,
            GroupCommitPhaseParams::try_from_params(sliced_params.group(), &sliced_params)
                .unwrap_or_else(|error| {
                    panic!("real S={} profile failed: {error}", slice_count.get())
                }),
        )
        .unwrap_or_else(|error| panic!("real S={} commitment failed: {error}", slice_count.get()));
        let source_coefficients = slice_count
            .complete_source_coefficients(
                sliced_params.outer().matrix.output_rank(),
                sliced_params.outer().matrix.ring_dimension(),
            )
            .expect("complete source coefficients");
        let plan = CompressionChainPlan::for_complete_source(
            sliced_params.outer().matrix.sis_table_key().modulus_profile,
            source_coefficients,
        )
        .expect("real compression plan");
        hint.validate_outer_compression(&plan)
            .expect("real sliced compression hint");
        assert!(!commitment.rows().coeffs().is_empty());
    }
}

#[test]
fn commitment_bytes_ignore_opening_method_and_profiles_reject_tensor_sources() {
    const NUM_VARS: usize = 10;
    let canonical = commitment_params_for_slice_count(akita_types::CommitmentSliceCount::ONE);
    let mut packing_plan = canonical.clone();
    packing_plan.own_group_mut().opening.opening_method =
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
    let group = PolynomialGroupLayout::new(NUM_VARS, 1);
    let profile = |params: &CommittedGroupParams| GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group,

        blocks: akita_types::BlockGeometry::new(
            params.blocks().live_ring_elements_per_claim,
            params.blocks().positions_per_block,
            params.blocks().live_blocks,
        ),

        outer_slice_count: params.outer_slice_count(),
        inner: akita_types::RoleParams::new(
            akita_types::GadgetDigits::new(
                params.inner().digits.log_basis,
                params.inner().digits.num_digits,
            ),
            params.inner().matrix,
        ),
        outer: akita_types::RoleParams::new(
            akita_types::GadgetDigits::new(
                params.outer().digits.log_basis,
                params.outer().digits.num_digits,
            ),
            params.outer().matrix,
        ),
    };
    assert_eq!(
        profile(&canonical),
        profile(&packing_plan),
        "opening policy must not enter commitment identity",
    );

    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        NUM_VARS,
        1,
        SetupMatrixCapacity {
            num_field_elements: 2_000_000,
        },
    )
    .unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let ctx = OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref()).unwrap();
    let evaluations = (0..1usize << NUM_VARS)
        .map(|index| F::from_u64((index * 17 + 9) as u64))
        .collect::<Vec<_>>();
    let polynomial = DensePoly::<F>::from_field_evals(NUM_VARS, &evaluations).unwrap();
    validate_commit_level_params::<F>(&canonical, setup.expanded.as_ref(), 0, 1).unwrap();
    let raw = commit_fixture_with_profile(
        std::slice::from_ref(&polynomial),
        &ctx,
        GroupCommitPhaseParams::try_from_params(canonical.group(), &canonical).unwrap(),
    )
    .unwrap();
    let raw_under_other_method = commit_fixture_with_profile(
        std::slice::from_ref(&polynomial),
        &ctx,
        GroupCommitPhaseParams::try_from_params(packing_plan.group(), &packing_plan).unwrap(),
    )
    .unwrap();
    assert_eq!(raw, raw_under_other_method);

    let mut tensor = canonical.clone();
    tensor.source_encoding = CommittedSourceEncoding::TensorSubfieldProjection {
        extension_degree: 2,
    };
    assert!(GroupCommitPhaseParams::try_from_params(group, &tensor).is_err());
}
