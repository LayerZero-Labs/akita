use super::*;
use crate::commitment::{CommitmentExecutor, PortableStatePolicy};
use crate::opaque::owned::CommittedSource;
use crate::{AkitaProverSetup, DensePoly};
use akita_params::ordinary_source::OrdinarySourceWidth;
use akita_params::sis::{
    inner_coeff_linf_bounds, num_digits_inner, num_digits_open, SisMatrixRole, SisTableDigest,
    SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::{
    BlockGeometry, CommitmentSliceCount, GadgetDigits, GroupCommitPhaseParams,
    InnerCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout, RoleParams,
    SetupMatrixCapacity, SisModulusProfileId,
};
use akita_types::binary_source::{OrdinaryNativeRootBatch, OrdinaryNativeRootClaim};
use akita_types::root_repacking::RootCoefficientRepacking;
use jolt_algebra::{One, Prime128Offset275, Ring, Zero};
use std::collections::BTreeSet;
use std::sync::Arc;

type F = Prime128Offset275;
const D: usize = 64;
const N: usize = 128;

fn profile(bits: u32, slots: usize, slices: CommitmentSliceCount) -> GroupCommitPhaseParams {
    let width = OrdinarySourceWidth::new(bits).unwrap();
    let inner_digits = num_digits_inner(width.decomposition(), true);
    let outer_digits = num_digits_open(width.decomposition());
    let key = SisTableKey {
        policy: DEFAULT_SIS_SECURITY_POLICY,
        table_digest: SisTableDigest::CURRENT,
        modulus_profile: SisModulusProfileId::Q128Offset275,
        role: SisMatrixRole::Inner,
        ring_dimension: D as u32,
        coeff_linf_bound: inner_coeff_linf_bounds(SisModulusProfileId::Q128Offset275, D as u32)[0],
    };
    let a = InnerCommitMatrixParams::try_new_with_min_rank(key, inner_digits).unwrap();
    let geometry = akita_params::CommitmentSliceGeometry::try_new(
        slices,
        2,
        slots,
        a.output_rank(),
        outer_digits,
        D,
        D,
    )
    .unwrap();
    let b = OuterCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            role: SisMatrixRole::Outer,
            coeff_linf_bound: 7,
            ..key
        },
        geometry.physical_input_width(),
    )
    .unwrap();
    let profile = GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group: PolynomialGroupLayout::new(7, slots),
        blocks: BlockGeometry::new(2, 1, 2),
        outer_slice_count: slices,
        randomness_columns: 0,
        inner: RoleParams::new(GadgetDigits::new(3, inner_digits), a),
        outer: RoleParams::new(GadgetDigits::new(3, outer_digits), b),
    };
    width.validate_profile(&profile).unwrap();
    profile
}

struct Fixture {
    backend: CpuBackend<F, F>,
    setup: AkitaProverSetup<F>,
    handle: CommitmentHandle<F, F>,
    values: Vec<Vec<F>>,
    point: Vec<F>,
    evaluations: Vec<F>,
    payload: Vec<F>,
}

fn fixture(profile: GroupCommitPhaseParams, bits: u32) -> Fixture {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        7,
        profile.group.num_polynomials(),
        SetupMatrixCapacity {
            num_field_elements: 1 << 20,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
    let offset = 1i128 << (bits - 1);
    let values = (0..profile.group.num_polynomials())
        .map(|slot| {
            (0..N)
                .map(|i| {
                    F::from_i128(match (i + slot) % 4 {
                        0 => -offset,
                        1 => offset - 1,
                        2 => -1,
                        _ => 0,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let polynomials = values
        .iter()
        .map(|values| DensePoly::from_field_evals(7, values).unwrap())
        .collect();
    let source = backend.import_source(polynomials).unwrap();
    let prepared = backend.prepared().unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        prepared,
        &prepared.expanded,
        Vec::new(),
        PortableStatePolicy,
    )
    .unwrap();
    // Execute the real commitment pipeline. No verifier or receipt is mocked.
    let (public, retained) = executor
        .execute_full_via_outer_image(
            &backend,
            profile,
            &source.storage.commitment_sources(),
            None,
        )
        .unwrap();
    drop(executor);
    let handle = CommitmentHandle {
        owner: backend.owner_id(),
        committed: Arc::new(CommittedSource {
            source: source.storage,
            metadata: source.metadata,
            commitment_id: backend.owner().next_operation_id().unwrap(),
            parameters: profile,
            producer_contract: OrdinarySourceWidth::new(bits)
                .unwrap()
                .producer_contract()
                .unwrap(),
            public: public.commitment().clone(),
            retained: Some(retained),
        }),
    };
    let point = (0..7).map(|i| F::from_u64(17 + 3 * i)).collect::<Vec<_>>();
    // Dense folding is independent of the compiler's OffsetEqWindow.
    let evaluations = values
        .iter()
        .map(|values| {
            let mut table = values.clone();
            for &r in &point {
                table = table
                    .chunks_exact(2)
                    .map(|pair| pair[0] * (F::one() - r) + pair[1] * r)
                    .collect();
            }
            table[0]
        })
        .collect();
    let payload = public.rows().coeffs().to_vec();
    Fixture {
        backend,
        setup,
        handle,
        values,
        point,
        evaluations,
        payload,
    }
}

#[test]
fn signed_extrema_all_slots_and_all_commitment_rows_authenticate() {
    for (bits, slots) in [(2, 3), (9, 3), (18, 5), (33, 1), (34, 1), (64, 1)] {
        for slices in [CommitmentSliceCount::ONE, CommitmentSliceCount::TWO] {
            let profile = profile(bits, slots, slices);
            let plan = OrdinaryNativeRootPlan::new_signed(
                profile,
                0,
                OrdinarySourceWidth::new(bits).unwrap(),
            )
            .unwrap();
            let input = fixture(profile, bits);
            let natural = input
                .backend
                .ordinary_root_preimage(&input.handle, &plan)
                .unwrap();
            let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), 64).unwrap();
            let digits = repacking
                .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
                .unwrap();
            let table = digits
                .iter()
                .map(|&x| F::from_i64(i64::from(x)))
                .collect::<Vec<_>>();
            // Derive the complete membership set from the commitment maps and
            // source cube, independently of the compiler's segment iterator.
            let mut expected_addresses = BTreeSet::new();
            let compression = profile.outer_compression_plan().unwrap();
            for (stage, map) in compression.maps().iter().enumerate() {
                for element in 0..map.input_width() {
                    for coefficient in 0..map.ring_dimension() {
                        expected_addresses.insert(
                            repacking
                                .coefficient_address(2 + stage, element, 0, coefficient)
                                .unwrap(),
                        );
                    }
                }
            }
            for owner in 0..slots {
                for element in 0..N / D {
                    for bit in 0..bits as usize {
                        for coefficient in 0..D {
                            expected_addresses.insert(
                                repacking
                                    .coefficient_address(
                                        4,
                                        owner * (N / D) + element,
                                        bit,
                                        coefficient,
                                    )
                                    .unwrap(),
                            );
                        }
                    }
                }
            }
            let addresses = plan.negative_binary_addresses(&repacking).unwrap();
            assert_eq!(addresses.len(), expected_addresses.len());
            assert_eq!(
                addresses.iter().copied().collect::<BTreeSet<_>>(),
                expected_addresses
            );
            for address in addresses {
                assert!(table[address] == F::zero() || table[address] == -F::one());
            }
            if bits == 64 {
                assert_eq!(profile.inner.digits.num_digits, 22);
                assert_eq!(input.values[0][0], F::from_i128(-(1i128 << 63)));
                assert_eq!(input.values[0][1], F::from_i128((1i128 << 63) - 1));
                for (coefficient, expected) in [(0, F::zero()), (1, -F::one()), (3, -F::one())] {
                    let address = repacking
                        .coefficient_address(4, 0, 63, coefficient)
                        .unwrap();
                    assert_eq!(table[address], expected);
                }
            }
            for (owner, values) in input.values.iter().enumerate() {
                for (coefficient, &expected) in values.iter().enumerate() {
                    let element = owner * 2 + coefficient / D;
                    let value = (0..profile.inner.digits.num_digits)
                        .map(|digit| {
                            let address = repacking
                                .coefficient_address(0, element, digit, coefficient % D)
                                .unwrap();
                            table[address] * F::from_u128(1u128 << (3 * digit))
                        })
                        .sum::<F>();
                    assert_eq!(value, expected);
                }
            }
            // Independently executed A/B/F/H commitments must satisfy each
            // compiler row, not just a cancellation-prone combined equation.
            for row in 0..plan.linear_batch_count() {
                let mut batches = vec![F::zero(); plan.linear_batch_count()];
                batches[row] = F::one();
                plan.linear_weights(
                    &repacking,
                    input.setup.expanded.descriptor(),
                    &input.point,
                    &input.evaluations,
                    &input.payload,
                    F::from_u64(23),
                    &batches,
                )
                .unwrap()
                .check_table(&table)
                .unwrap();
            }
            let batches = (0..plan.linear_batch_count())
                .map(|i| F::from_u64(31 + i as u64))
                .collect::<Vec<_>>();
            let compile = |evaluations: &[F], payload: &[F]| {
                plan.linear_weights(
                    &repacking,
                    input.setup.expanded.descriptor(),
                    &input.point,
                    evaluations,
                    payload,
                    F::from_u64(23),
                    &batches,
                )
            };
            let weights = compile(&input.evaluations, &input.payload).unwrap();
            for (segment, shape) in plan.segments().iter().enumerate() {
                for digit in 0..shape.digits {
                    let address = repacking.coefficient_address(segment, 0, digit, 0).unwrap();
                    let mut changed = table.clone();
                    changed[address] += F::one();
                    assert!(
                        weights.check_table(&changed).is_err(),
                        "width {bits}, segment {segment}, digit {digit}"
                    );
                }
            }
            let mut wrong_values = input.evaluations.clone();
            for slot in 0..slots {
                wrong_values[slot] += F::one();
                assert!(compile(&wrong_values, &input.payload)
                    .unwrap()
                    .check_table(&table)
                    .is_err());
                wrong_values[slot] -= F::one();
            }
            if slots > 1 {
                wrong_values.swap(0, 1);
                assert!(compile(&wrong_values, &input.payload)
                    .unwrap()
                    .check_table(&table)
                    .is_err());
            }
            let mut wrong_payload = input.payload.clone();
            wrong_payload[0] += F::one();
            assert!(compile(&input.evaluations, &wrong_payload)
                .unwrap()
                .check_table(&table)
                .is_err());
            assert!(compile(&input.evaluations[..slots - 1], &input.payload).is_err());
        }
    }
}

#[test]
fn signed_width_rejects_values_beyond_both_endpoints_and_wrong_digit_depth() {
    for bits in [2, 9, 18, 33, 34, 64] {
        let profile = profile(bits, 1, CommitmentSliceCount::ONE);
        let plan =
            OrdinaryNativeRootPlan::new_signed(profile, 0, OrdinarySourceWidth::new(bits).unwrap())
                .unwrap();
        let offset = 1i128 << (bits - 1);
        for value in [-offset - 1, offset] {
            assert!(source_digits(&plan, &vec![F::from_i128(value); N]).is_err());
        }
        let mut wrong = profile;
        wrong.inner.digits.num_digits += 1;
        assert!(OrdinaryNativeRootPlan::new_signed(
            wrong,
            0,
            OrdinarySourceWidth::new(bits).unwrap()
        )
        .is_err());
        let input = fixture(profile, bits);
        // Width 18 and 19 deliberately share a digit depth. The producer
        // contract, not the gadget depth alone, prevents relabeling.
        if bits == 18 {
            let wrong = OrdinaryNativeRootPlan::new_signed(
                profile,
                0,
                OrdinarySourceWidth::new(19).unwrap(),
            )
            .unwrap();
            assert!(input
                .backend
                .ordinary_root_preimage(&input.handle, &wrong)
                .is_err());
        }
    }
}

#[test]
fn signed_offset_membership_rejects_non_boolean_linear_preserving_mutation() {
    let profile = profile(2, 3, CommitmentSliceCount::ONE);
    let width = OrdinarySourceWidth::new(2).unwrap();
    let plan = OrdinaryNativeRootPlan::new_signed(profile, 0, width).unwrap();
    let input = fixture(profile, 2);
    let natural = input
        .backend
        .ordinary_root_preimage(&input.handle, &plan)
        .unwrap();
    let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), D).unwrap();
    let digits = repacking
        .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap();
    assert!(digits.iter().all(|&digit| (-4..=3).contains(&digit)));
    let mut table = digits
        .iter()
        .map(|&digit| F::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    let addresses = plan.negative_binary_addresses(&repacking).unwrap();
    assert!(addresses
        .iter()
        .all(|&address| (table[address] * (table[address] + F::one())).is_zero()));
    assert_eq!(input.values[0][3], F::zero());
    let low = repacking.coefficient_address(4, 0, 0, 3).unwrap();
    let high = repacking.coefficient_address(4, 0, 1, 3).unwrap();
    assert_eq!((table[low], table[high]), (F::zero(), -F::one()));
    // Both (0, -1) and (-2, 0) recompose to -2 and stay in [-4, 3].
    // Only the negative-binary predicate can reject this changed witness.
    table[low] = F::from_i128(-2);
    table[high] = F::zero();
    for row in 0..plan.linear_batch_count() {
        let mut batches = vec![F::zero(); plan.linear_batch_count()];
        batches[row] = F::one();
        plan.linear_weights(
            &repacking,
            input.setup.expanded.descriptor(),
            &input.point,
            &input.evaluations,
            &input.payload,
            F::from_u64(23),
            &batches,
        )
        .unwrap()
        .check_table(&table)
        .unwrap();
    }
    assert!(addresses
        .iter()
        .any(|&address| !(table[address] * (table[address] + F::one())).is_zero()));
}

#[test]
fn ordered_batch_rejects_omitted_reordered_and_wrong_width_groups() {
    let sources = [(2, 3), (9, 3), (18, 5), (33, 1)].map(|(bits, slots)| {
        (
            profile(bits, slots, CommitmentSliceCount::ONE),
            OrdinarySourceWidth::new(bits).unwrap(),
        )
    });
    let batch = OrdinaryNativeRootBatch::new(&sources, 0).unwrap();
    let repacking = RootCoefficientRepacking::new(
        batch
            .plans()
            .iter()
            .flat_map(|plan| plan.segments().iter().copied())
            .collect(),
        64,
    )
    .unwrap();
    let inputs = sources
        .iter()
        .map(|&(profile, width)| fixture(profile, width.bits()))
        .collect::<Vec<_>>();
    let mut natural = Vec::new();
    for (input, plan) in inputs.iter().zip(batch.plans()) {
        natural.extend(
            input
                .backend
                .ordinary_root_preimage(&input.handle, plan)
                .unwrap(),
        );
    }
    let table = repacking
        .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap()
        .into_iter()
        .map(|x| F::from_i64(i64::from(x)))
        .collect::<Vec<_>>();
    let mut claims = inputs
        .iter()
        .zip(&sources)
        .map(|(input, (profile, width))| OrdinaryNativeRootClaim {
            profile,
            width: *width,
            point: &input.point,
            evaluations: &input.evaluations,
            payload: &input.payload,
        })
        .collect::<Vec<_>>();
    let batches = (0..batch.linear_batch_count())
        .map(|i| F::from_u64(41 + i as u64))
        .collect::<Vec<_>>();
    let compile = |claims: &[OrdinaryNativeRootClaim<'_, F>]| {
        batch.linear_weights(
            &repacking,
            inputs[0].setup.expanded.descriptor(),
            claims,
            F::from_u64(29),
            &batches,
        )
    };
    compile(&claims).unwrap().check_table(&table).unwrap();
    assert!(compile(&claims[..3]).is_err());
    claims.swap(0, 1);
    assert!(compile(&claims).is_err());
    claims.swap(0, 1);
    claims[2].width = OrdinarySourceWidth::new(19).unwrap();
    assert!(compile(&claims).is_err());
}

#[path = "binary_ordinary_legacy_weights.rs"]
mod legacy_weights;

#[test]
fn legacy_and_signed_preimages_accept_multi_block_slices() {
    // One B slice carries both live blocks. The legacy relation shares the
    // signed relation's canonical T order, so both emit the same segments.
    let profile = profile(3, 1, CommitmentSliceCount::ONE);
    assert!(
        profile
            .derive_slice_geometry()
            .unwrap()
            .max_blocks_per_slice()
            > 1
    );
    let plan = OrdinaryNativeRootPlan::new(profile, 0).unwrap();
    let input = fixture(profile, 3);
    let legacy = input
        .backend
        .ordinary_root_preimage(&input.handle, &plan)
        .unwrap();
    let signed =
        OrdinaryNativeRootPlan::new_signed(profile, 0, OrdinarySourceWidth::new(3).unwrap())
            .unwrap();
    let signed = input
        .backend
        .ordinary_root_preimage(&input.handle, &signed)
        .unwrap();
    assert_eq!(legacy.len(), 4);
    assert_eq!(legacy[..], signed[..4]);
}

#[test]
fn legacy_partial_block_preserves_base_relation_and_repacking_boundary() {
    let mut profile = profile(3, 1, CommitmentSliceCount::ONE);
    // Two live D64 rings in a four-position block: the base public constructor
    // accepts this profile, although its prover cannot repack the short source.
    // Constructor order is live rings, positions per block, live blocks.
    profile.blocks = BlockGeometry::new(2, 4, 1);
    profile.inner.matrix = InnerCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            policy: DEFAULT_SIS_SECURITY_POLICY,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q128Offset275,
            role: SisMatrixRole::Inner,
            ring_dimension: D as u32,
            coeff_linf_bound: inner_coeff_linf_bounds(SisModulusProfileId::Q128Offset275, D as u32)
                [0],
        },
        4,
    )
    .unwrap();
    let geometry = profile.derive_slice_geometry().unwrap();
    profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        profile.outer.matrix.sis_table_key(),
        geometry.physical_input_width(),
    )
    .unwrap();
    profile.validate_frozen_precommit(128).unwrap();
    let legacy = legacy_weights::OrdinaryNativeRootPlan::new(profile, 0).unwrap();
    let plan = OrdinaryNativeRootPlan::new(profile, 0).unwrap();
    assert!(
        OrdinaryNativeRootPlan::new_signed(profile, 0, OrdinarySourceWidth::new(3).unwrap())
            .is_err()
    );
    assert_eq!(plan.segments(), legacy.segments());
    assert_eq!(plan.linear_batch_count(), legacy.linear_batch_count());
    let input = fixture(profile, 3);
    let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), D).unwrap();
    assert_eq!(
        plan.negative_binary_addresses(&repacking).unwrap(),
        legacy.negative_binary_addresses(&repacking).unwrap()
    );
    let batches = vec![F::one(); plan.linear_batch_count()];
    let previous = legacy
        .linear_weights(
            &repacking,
            input.setup.expanded.descriptor(),
            &input.point,
            input.evaluations[0],
            &input.payload,
            F::from_u64(17),
            &batches,
        )
        .unwrap();
    let current = plan
        .linear_weights(
            &repacking,
            input.setup.expanded.descriptor(),
            &input.point,
            &input.evaluations,
            &input.payload,
            F::from_u64(17),
            &batches,
        )
        .unwrap();
    assert_eq!(
        previous.materialize().unwrap(),
        current.materialize().unwrap()
    );
    assert_eq!(previous.claim(), current.claim());
    let natural = input
        .backend
        .ordinary_root_preimage(&input.handle, &plan)
        .unwrap();
    // Base 6a294098 emits exactly the logical prefix, without block padding.
    assert_eq!(natural[0], [-4i8, 3, -1, 0].repeat(N / 4));
    assert!(matches!(
        repacking.repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>()),
        Err(AkitaError::InvalidSize {
            expected: 256,
            actual: 128,
        })
    ));
}

#[test]
fn one_digit_fixture_matches_frozen_pre_extension_relation_bytes() {
    use akita_serialization::AkitaSerialize;
    let profile = profile(3, 1, CommitmentSliceCount::TWO);
    let legacy = legacy_weights::OrdinaryNativeRootPlan::new(profile, 0).unwrap();
    let plan = OrdinaryNativeRootPlan::new(profile, 0).unwrap();
    let input = fixture(profile, 3);
    assert_eq!(plan.segments(), legacy.segments());
    assert_eq!(plan.linear_batch_count(), legacy.linear_batch_count());
    let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), 64).unwrap();
    assert_eq!(
        plan.negative_binary_addresses(&repacking).unwrap(),
        legacy.negative_binary_addresses(&repacking).unwrap()
    );
    let natural = input
        .backend
        .ordinary_root_preimage(&input.handle, &plan)
        .unwrap();
    // The original importer wrote raw small coefficients in this exact order.
    assert_eq!(natural[0], [-4i8, 3, -1, 0].repeat(N / 4));
    let batches = (0..plan.linear_batch_count())
        .map(|i| F::from_u64(19 + i as u64))
        .collect::<Vec<_>>();
    let previous = legacy
        .linear_weights(
            &repacking,
            input.setup.expanded.descriptor(),
            &input.point,
            input.evaluations[0],
            &input.payload,
            F::from_u64(17),
            &batches,
        )
        .unwrap();
    let current = plan
        .linear_weights(
            &repacking,
            input.setup.expanded.descriptor(),
            &input.point,
            &input.evaluations,
            &input.payload,
            F::from_u64(17),
            &batches,
        )
        .unwrap();
    // The current weights are structured, so compare the relation they
    // materialize rather than their explicit terms.
    let mut golden = Vec::new();
    let mut actual = Vec::new();
    previous.claim().serialize_compressed(&mut golden).unwrap();
    current.claim().serialize_compressed(&mut actual).unwrap();
    for weight in previous.materialize().unwrap() {
        weight.serialize_compressed(&mut golden).unwrap();
    }
    for weight in current.materialize().unwrap() {
        weight.serialize_compressed(&mut actual).unwrap();
    }
    assert_eq!(actual, golden);
}
