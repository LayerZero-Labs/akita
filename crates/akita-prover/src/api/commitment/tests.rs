use super::*;
use crate::compute::{ComputeBackendSetup, DigitRowsComputeBackend};
use crate::{AkitaProverSetup, CommitInnerWitness, CpuBackend, DensePoly};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp64;
use akita_types::{
    CommittedSourceEncoding, GroupCommitPhaseParams, OpenCommitMatrixParams, OpeningMethod,
    PolynomialGroupLayout, SetupMatrixCapacity, SisModulusProfileId,
};

type F = Fp64<4294967197>;
const D: usize = 64;

fn inner_witness(recomposed_blocks: usize, rows_per_block: usize) -> CommitInnerWitness<F> {
    CommitInnerWitness::from_rows(vec![
        vec![CyclotomicRing::<F, D>::zero(); rows_per_block];
        recomposed_blocks
    ])
}

#[test]
fn commit_inner_shape_accepts_expected_layout() {
    let inner = inner_witness(2, 3);
    validate_commit_inner_shape::<F, D>(&inner, 2, 3).expect("shape should match");
}

#[test]
fn commit_inner_shape_rejects_bad_block_count() {
    let inner = inner_witness(1, 3);
    assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3).is_err());
}

#[test]
fn commit_inner_shape_rejects_bad_row_count() {
    let inner = inner_witness(2, 2);
    assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3).is_err());
}

#[test]
fn commit_inner_shape_accepts_many_all_zero_blocks() {
    let num_live_blocks = 1024;
    let inner = inner_witness(num_live_blocks, 3);
    validate_commit_inner_shape::<F, D>(&inner, num_live_blocks, 3).expect("all-zero blocks");
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
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(1, 1, 1, 1, 1)
    .unwrap();
    let d_key = params.open_commit_matrix.sis_table_key();
    params.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
        d_key.policy,
        d_key.table_digest,
        d_key.modulus_profile,
        8,
        8,
        d_key.coeff_linf_bound,
        D,
    );
    let commit_only_fields = akita_types::commit_only_setup_field_elements(
        &params.inner.matrix,
        &params.outer.matrix,
        params.outer_slice_count,
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
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    );
    params.outer_slice_count = akita_types::CommitmentSliceCount::FOUR;
    params.with_decomp(2, 16, 1, 1, 1).unwrap()
}

fn set_outer_width(params: &mut CommittedGroupParams, input_width: usize) {
    let key = params.outer.matrix.sis_table_key();
    params.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        key.policy,
        key.table_digest,
        key.modulus_profile,
        params.outer.matrix.output_rank(),
        input_width,
        key.coeff_linf_bound,
        params.outer.matrix.ring_dimension(),
    );
}

#[test]
fn commitment_request_binds_slice_count_and_exact_b_width() {
    let params = sliced_commit_params();
    params
        .validate_commitment_request(0, 1)
        .expect("canonical sliced geometry");

    let mut wrong_slice_count = params.clone();
    wrong_slice_count.outer_slice_count = akita_types::CommitmentSliceCount::ONE;
    assert!(matches!(
        wrong_slice_count.validate_commitment_request(0, 1),
        Err(AkitaError::InvalidSetup(_))
    ));

    let mut wrong_width = params.clone();
    set_outer_width(&mut wrong_width, params.outer.matrix.input_width() + 1);
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
    let geometry = akita_types::CommitmentSliceGeometry::try_new(
        two_polynomials.outer_slice_count,
        two_polynomials.num_live_blocks,
        2,
        two_polynomials.inner.matrix.output_rank(),
        two_polynomials.outer.digits.num_digits,
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

#[test]
fn outer_slice_inputs_are_polynomial_major_and_zero_padded() {
    let first = akita_types::DigitBlocks::new(vec![10, 11, 12, 13, 14], vec![1; 5], 1)
        .expect("first digit blocks");
    let second = akita_types::DigitBlocks::new(vec![20, 21, 22, 23, 24], vec![1; 5], 1)
        .expect("second digit blocks");
    let geometry = akita_types::CommitmentSliceGeometry::try_new(
        akita_types::CommitmentSliceCount::TWO,
        5,
        2,
        1,
        1,
        1,
        1,
    )
    .expect("slice geometry");

    let inputs = outer_slice_inputs::<1>(&[&first, &second], &geometry).expect("slice inputs");
    assert_eq!(
        inputs,
        vec![
            vec![[10], [11], [0], [20], [21], [0]],
            vec![[12], [13], [14], [22], [23], [24]],
        ]
    );
}

#[test]
fn outer_slice_stream_reuses_one_physical_width_buffer() {
    let digits =
        akita_types::DigitBlocks::new((0..13).collect(), vec![1; 13], 1).expect("digit blocks");
    let geometry = akita_types::CommitmentSliceGeometry::try_new(
        akita_types::CommitmentSliceCount::FOUR,
        13,
        1,
        1,
        1,
        1,
        1,
    )
    .expect("slice geometry");
    let planes = digits.typed_planes::<1>().expect("typed planes");
    let mut addresses = Vec::new();

    for_each_outer_slice_input::<1>(std::iter::once(planes), &geometry, |input| {
        assert_eq!(input.len(), geometry.physical_input_width());
        addresses.push(input.as_ptr());
        Ok(())
    })
    .expect("stream slices");

    assert_eq!(addresses.len(), 4);
    assert!(addresses.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn sliced_b_images_match_independent_block_diagonal_oracle_for_all_counts() {
    const BLOCKS: usize = 9;
    const POLYS: usize = 2;
    const PER_BLOCK: usize = 2;
    const ROWS: usize = 3;

    let polynomial_digits = (0..POLYS)
        .map(|polynomial| {
            let digits = (0..BLOCKS * PER_BLOCK)
                .map(|index| (1 + polynomial * 31 + index) as i8)
                .collect::<Vec<_>>();
            akita_types::DigitBlocks::new(digits, vec![PER_BLOCK; BLOCKS], 1).unwrap()
        })
        .collect::<Vec<_>>();

    for slice_count in akita_types::CommitmentSliceCount::ALL {
        let geometry = akita_types::CommitmentSliceGeometry::try_new(
            slice_count,
            BLOCKS,
            POLYS,
            PER_BLOCK,
            1,
            1,
            1,
        )
        .unwrap();
        let matrix = (0..ROWS)
            .map(|row| {
                (0..geometry.physical_input_width())
                    .map(|column| 1 + (row as i64 + 1) * 17 + column as i64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let production_inputs =
            outer_slice_inputs::<1>(&polynomial_digits.iter().collect::<Vec<_>>(), &geometry)
                .unwrap();
        let production_image = production_inputs
            .iter()
            .flat_map(|input| {
                let matrix = &matrix;
                matrix.iter().map(move |row| {
                    row.iter()
                        .zip(input)
                        .map(|(&matrix_entry, digit)| matrix_entry * i64::from(digit[0]))
                        .sum::<i64>()
                })
            })
            .collect::<Vec<_>>();

        // Independent oracle: derive proportional boundaries and physical
        // columns directly, without calling slicing geometry helpers.
        let slices = slice_count.get();
        let max_blocks = BLOCKS.div_ceil(slices);
        let mut oracle_image = Vec::with_capacity(slices * ROWS);
        for slice_index in 0..slices {
            let start = BLOCKS * slice_index / slices;
            let end = BLOCKS * (slice_index + 1) / slices;
            for row in &matrix {
                let mut image = 0i64;
                for polynomial in 0..POLYS {
                    for global_block in start..end {
                        let local_block = global_block - start;
                        for offset in 0..PER_BLOCK {
                            let physical_column =
                                (polynomial * max_blocks + local_block) * PER_BLOCK + offset;
                            let digit = 1 + polynomial * 31 + global_block * PER_BLOCK + offset;
                            image += row[physical_column] * digit as i64;
                        }
                    }
                }
                oracle_image.push(image);
            }
        }
        assert_eq!(production_image, oracle_image);

        // The complete logical stack is the compression source. Compare a
        // second independent linear image so slice ordering cannot alias.
        let production_compressed = production_image
            .iter()
            .enumerate()
            .map(|(index, &value)| (index as i64 + 3) * value)
            .sum::<i64>();
        let oracle_compressed = oracle_image
            .iter()
            .enumerate()
            .map(|(index, &value)| (index as i64 + 3) * value)
            .sum::<i64>();
        assert_eq!(production_compressed, oracle_compressed);
    }
}

/// Inner digit depth that actually represents an `Fp32` coefficient at
/// `log_basis_inner = 2`.
///
/// The fixture used to declare a single base-4 digit, which cannot represent a
/// 32-bit field element at all: the commitment silently truncated, and the test
/// only passed because the production and reference paths truncated identically.
/// The commit path now rejects a source outside its scheduled digit envelope, so
/// the fixture states a depth consistent with the coefficients it commits.
fn slice_fixture_num_digits_inner() -> usize {
    akita_types::sis::compute_num_digits_field_width(32, 2)
}

/// Full-field balanced-digit contract matching the slice fixture's geometry.
///
/// `log_commit_bound == field_bits` is the unbounded endpoint, so the accepted
/// interval is representability alone and the fixture keeps committing arbitrary
/// field elements. The balanced-digit class imposes no structural requirement, so
/// the dense fixture source is admissible. Both restrictive paths — a bounded
/// declaration and the unit one-hot class — are covered by the `fp128` e2e tests,
/// which own real catalogs.
fn slice_fixture_contract() -> akita_types::sis::CommittedSourceContract {
    akita_types::sis::CommittedSourceContract::try_new(
        akita_types::sis::CommittedSourceClass::BalancedSignedDigit,
        akita_types::DecompositionParams {
            log_basis: 2,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        },
    )
    .expect("full-field slice fixture contract")
}

fn commitment_params_for_slice_count(
    slice_count: akita_types::CommitmentSliceCount,
) -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    );
    params.outer_slice_count = slice_count;
    params
        .with_decomp(2, 16, slice_fixture_num_digits_inner(), 1, 1)
        .expect("unsliced commitment geometry")
}

fn commit_unsliced_reference(
    polys: &[DensePoly<F>],
    ctx: &OperationCtx<'_, F, CpuBackend>,
    params: &CommittedGroupParams,
) -> Result<(CommitmentWithHint<F>, CompressionChainPlan), AkitaError> {
    let backend = ctx.backend();
    let prepared = ctx.prepared();
    let plan = CommitInnerPlan::from_level(params);
    let views = polys
        .iter()
        .map(RootCommitSource::<F, D>::commit_view)
        .collect::<Result<Vec<_>, _>>()?;
    let prepared_polynomials = prepare_inner_commit_group::<F, _, _, D, D>(
        backend,
        prepared,
        views,
        plan,
        params.num_live_blocks,
        params.outer.digits.num_digits,
        params.outer.digits.log_basis,
    )?;
    let geometry = akita_types::CommitmentSliceGeometry::try_new(
        akita_types::CommitmentSliceCount::ONE,
        params.num_live_blocks,
        polys.len(),
        params.inner.matrix.output_rank(),
        params.outer.digits.num_digits,
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
    if reference_b_input.len() != params.outer.matrix.input_width() {
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

    let n_b = params.outer.matrix.output_rank();
    let reference_b_image = backend.digit_rows::<D>(
        prepared,
        n_b,
        &reference_b_input,
        params.outer.digits.log_basis,
    )?;
    let production_b_image = commit_outer_slices::<F, _, D>(
        backend,
        prepared,
        n_b,
        prepared_polynomials.iter().map(|(_, digits)| digits),
        &geometry,
        params.outer.digits.log_basis,
    )?;
    if production_b_image != reference_b_image {
        return Err(AkitaError::InvalidSetup(
            "S=1 sliced B image differs from the unsliced image".into(),
        ));
    }

    let source = RingVec::from_ring_elems(&reference_b_image);
    let compression_plan = CompressionChainPlan::for_complete_source(
        params.outer.matrix.sis_table_key().modulus_profile,
        source.coeff_len(),
    )?;
    let (mut outputs, _) = execute_compression_chains(
        ctx,
        vec![CompressionExecutionInput {
            id: (),
            plan: compression_plan.clone(),
            coefficients: source.into_coeffs(),
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
    let hint = AkitaCommitmentHint::new_with_outer_compression(
        D,
        inner_rows,
        &output.witness,
        &output.quotients,
    )?;
    Ok(((Commitment::new(payload), hint), compression_plan))
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

    let production_geometry =
        validate_commit_level_params::<F>(&params, setup.expanded.as_ref(), 0, 1)
            .expect("production S=1 geometry");
    let production = commit_with_validated_geometry::<F, DensePoly<F>, CpuBackend>(
        std::slice::from_ref(&poly),
        &ctx,
        (&params).into(),
        &production_geometry,
        slice_fixture_contract(),
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
        let slice_geometry =
            validate_commit_level_params::<F>(&sliced_params, setup.expanded.as_ref(), 0, 1)
                .unwrap_or_else(|error| {
                    panic!("real S={} geometry failed: {error}", slice_count.get())
                });
        let (commitment, hint) = commit_with_validated_geometry::<F, DensePoly<F>, CpuBackend>(
            std::slice::from_ref(&poly),
            &ctx,
            (&sliced_params).into(),
            &slice_geometry,
            slice_fixture_contract(),
        )
        .unwrap_or_else(|error| panic!("real S={} commitment failed: {error}", slice_count.get()));
        let source_coefficients = slice_count
            .complete_source_coefficients(
                sliced_params.outer.matrix.output_rank(),
                sliced_params.outer.matrix.ring_dimension(),
            )
            .expect("complete source coefficients");
        let plan = CompressionChainPlan::for_complete_source(
            sliced_params.outer.matrix.sis_table_key().modulus_profile,
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
    packing_plan.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    let group = PolynomialGroupLayout::new(NUM_VARS, 1);
    let profile = |params: &CommittedGroupParams| GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group,
        blocks: akita_types::BlockGeometry::new(
            params.num_live_ring_elements_per_claim,
            params.num_positions_per_block,
            params.num_live_blocks,
        ),
        outer_slice_count: params.outer_slice_count,
        inner: akita_types::RoleParams::new(
            akita_types::GadgetDigits::new(
                params.inner.digits.log_basis,
                params.inner.digits.num_digits,
            ),
            params.inner.matrix,
        ),
        outer: akita_types::RoleParams::new(
            akita_types::GadgetDigits::new(
                params.outer.digits.log_basis,
                params.outer.digits.num_digits,
            ),
            params.outer.matrix,
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
    let slice_geometry =
        validate_commit_level_params::<F>(&canonical, setup.expanded.as_ref(), 0, 1).unwrap();
    let contract = akita_config::proof_optimized::fp64::Dense::committed_source_contract().unwrap();
    let raw = commit_with_validated_geometry::<F, DensePoly<F>, CpuBackend>(
        std::slice::from_ref(&polynomial),
        &ctx,
        (&canonical).into(),
        &slice_geometry,
        contract,
    )
    .unwrap();
    let raw_under_other_method = commit_with_validated_geometry::<F, DensePoly<F>, CpuBackend>(
        std::slice::from_ref(&polynomial),
        &ctx,
        (&packing_plan).into(),
        &slice_geometry,
        contract,
    )
    .unwrap();
    assert_eq!(raw, raw_under_other_method);

    let mut tensor = canonical.clone();
    tensor.source_encoding = CommittedSourceEncoding::TensorSubfieldProjection {
        extension_degree: 2,
    };
    assert!(GroupCommitPhaseParams::try_from_params(group, &tensor).is_err());
}
