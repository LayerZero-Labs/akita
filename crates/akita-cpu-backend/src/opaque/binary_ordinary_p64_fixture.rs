//! The actual P64 ordinary commitment preimage used by the mixed root fixture.
use crate::commitment::{CommitmentExecutor, PortableStatePolicy};
use crate::opaque::owned::{CommitmentHandle, CommittedSource};
use crate::{CpuBackend, DensePoly};
use akita_algebra::eq_poly::EqPolynomial;
use akita_params::GroupCommitPhaseParams;
use akita_types::binary_source::OrdinaryNativeRootPlan;
use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as F, Ring, Zero};
use std::sync::Arc;
type E = Ext2<F>;

pub(crate) fn ordinary_preimage(
    profile: GroupCommitPhaseParams,
    plan: &OrdinaryNativeRootPlan,
) -> (Vec<Vec<i8>>, Vec<E>, Vec<E>, Vec<E>) {
    let inner = profile.inner.matrix;
    let outer = profile.outer.matrix;
    let setup_len = [
        1 << 16,
        inner.input_width() * inner.output_rank() * inner.ring_dimension(),
        outer.input_width() * outer.output_rank() * outer.ring_dimension(),
        profile
            .outer_compression_plan()
            .unwrap()
            .max_setup_field_elements()
            .unwrap(),
    ]
    .into_iter()
    .max()
    .unwrap()
    .next_power_of_two();
    let setup = akita_types::AkitaSetupDescriptor {
        max_num_vars: profile.group.num_vars(),
        max_num_batched_polys: profile.group.num_polynomials(),
        num_field_elements: setup_len,
        setup_seed: [41; 32].into(),
    };
    let matrix =
        akita_types::derive_public_matrix_prefix::<F>(setup.num_field_elements, &setup.setup_seed);
    let expanded = akita_types::AkitaExpandedSetup::from_verified_parts(setup, matrix).unwrap();
    let backend = CpuBackend::<F, E>::new(Arc::new(expanded)).unwrap();
    // Signed extrema include p-1 and each group's declared negative endpoint.
    let offset = plan.source_width().map_or(4, |width| width.offset()) as i128;
    let values = (0..profile.group.num_polynomials())
        .map(|slot| {
            (0..(1usize << profile.group.num_vars()))
                .map(|i| {
                    F::from_i128(if plan.source_width().is_none() {
                        (i % 8) as i128 - 4
                    } else {
                        match (i + slot) % 4 {
                            0 => -offset,
                            1 => offset - 1,
                            2 => -1,
                            _ => 0,
                        }
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let source = backend
        .import_source(
            values
                .iter()
                .map(|values| {
                    DensePoly::from_field_evals(profile.group.num_vars(), values).unwrap()
                })
                .collect(),
        )
        .unwrap();
    let prepared = backend.prepared().unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        prepared,
        &prepared.expanded,
        Vec::new(),
        PortableStatePolicy,
    )
    .unwrap();
    let (public, retained) = executor
        .execute_full_via_outer_image(
            &backend,
            profile,
            &source.storage.commitment_sources(),
            None,
        )
        .unwrap();
    drop(executor);
    let payload = public
        .rows()
        .coeffs()
        .iter()
        .copied()
        .map(E::lift_base)
        .collect();
    let handle = CommitmentHandle {
        owner: backend.owner_id(),
        committed: Arc::new(CommittedSource {
            source: source.storage,
            metadata: source.metadata,
            commitment_id: backend.owner().next_operation_id().unwrap(),
            parameters: profile,
            producer_contract: plan
                .source_width()
                .unwrap_or_else(|| {
                    akita_params::ordinary_source::OrdinarySourceWidth::new(3).unwrap()
                })
                .producer_contract(akita_params::SisModulusProfileId::Q64Offset59)
                .unwrap(),
            public: public.commitment().clone(),
            retained: Some(retained),
        }),
    };
    let preimage = backend.ordinary_root_preimage(&handle, plan).unwrap();
    assert!(preimage[0].contains(&-1));
    let point = (0..profile.group.num_vars())
        .map(|i| E::from_base_fn(|c| F::from_u64((17 + i * 3 + c * 43) as u64)))
        .collect::<Vec<_>>();
    let weights = EqPolynomial::evals(&point).unwrap();
    let evaluations = values
        .iter()
        .map(|values| {
            values
                .iter()
                .zip(&weights)
                .fold(E::zero(), |sum, (&v, &w)| sum + E::lift_base(v) * w)
        })
        .collect();
    (preimage, point, evaluations, payload)
}

#[test]
fn p64_four_signed_ordinary_groups_match_actual_commitments() {
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_params::ordinary_source::OrdinarySourceWidth;
    use akita_types::binary_source::{
        binary_ordinary_profile_fixture, OrdinaryNativeRootBatch, OrdinaryNativeRootClaim,
    };
    use akita_types::root_repacking::RootCoefficientRepacking;

    let sources = [(2, 3), (9, 3), (18, 5), (33, 1)].map(|(bits, slots)| {
        (
            binary_ordinary_profile_fixture(BinaryFieldProfile::P64, Some((bits, slots))),
            OrdinarySourceWidth::new(bits).unwrap(),
        )
    });
    let batch = OrdinaryNativeRootBatch::new(&sources, 0).unwrap();
    let inputs = sources
        .iter()
        .zip(batch.plans())
        .map(|(&(profile, _), plan)| ordinary_preimage(profile, plan))
        .collect::<Vec<_>>();
    let repacking = RootCoefficientRepacking::new(
        batch
            .plans()
            .iter()
            .flat_map(|plan| plan.segments().iter().cloned())
            .collect(),
        64,
    )
    .unwrap();
    let natural = inputs
        .iter()
        .flat_map(|(digits, _, _, _)| digits.iter().map(Vec::as_slice))
        .collect::<Vec<_>>();
    let table = repacking
        .repack(&natural)
        .unwrap()
        .into_iter()
        .map(|digit| E::lift_base(F::from_i128(i128::from(digit))))
        .collect::<Vec<_>>();
    let claims = sources
        .iter()
        .zip(&inputs)
        .map(
            |((profile, width), (_, point, evaluations, payload))| OrdinaryNativeRootClaim {
                profile,
                width: *width,
                point,
                evaluations,
                payload,
            },
        )
        .collect::<Vec<_>>();
    let setup = akita_types::AkitaSetupDescriptor {
        max_num_vars: 10,
        max_num_batched_polys: 5,
        num_field_elements: akita_types::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        setup_seed: [41; 32].into(),
    };
    let ext = |i: usize| E::from_base_fn(|c| F::from_u64((23 + i * 31 + c * 43) as u64));
    let batches = (0..batch.linear_batch_count()).map(ext).collect::<Vec<_>>();
    let weights = batch
        .linear_weights(&repacking, &setup, &claims, ext(400), &batches)
        .unwrap();
    weights.check_table(&table).unwrap();
    let mut first_segment = 0;
    let mut first_batch = 0;
    for (group, plan) in batch.plans().iter().enumerate() {
        // Check each identity separately against the actual A/B/F/H execution.
        for row in 0..plan.linear_batch_count() {
            let mut selected = vec![E::zero(); batch.linear_batch_count()];
            selected[first_batch + row] = E::one();
            batch
                .linear_weights(&repacking, &setup, &claims, ext(400), &selected)
                .unwrap()
                .check_table(&table)
                .unwrap();
        }
        first_batch += plan.linear_batch_count();
        for address in plan.negative_binary_addresses(&repacking).unwrap() {
            assert!(table[address] == E::zero() || table[address] == -E::one());
        }
        // Every source, image, compression and signed-offset digit bank matters.
        for (local, shape) in plan.segments().iter().enumerate() {
            for digit in 0..shape.digits {
                let address = repacking
                    .coefficient_address(first_segment + local, 0, digit, 0)
                    .unwrap();
                let mut changed = table.clone();
                changed[address] += E::one();
                assert!(
                    weights.check_table(&changed).is_err(),
                    "group {group}, bank {local}"
                );
            }
        }
        first_segment += plan.segments().len();
        for slot in 0..claims[group].evaluations.len() {
            let mut wrong = claims[group].evaluations.to_vec();
            wrong[slot] += E::one();
            let mut changed = claims
                .iter()
                .map(|claim| OrdinaryNativeRootClaim {
                    profile: claim.profile,
                    width: claim.width,
                    point: claim.point,
                    evaluations: claim.evaluations,
                    payload: claim.payload,
                })
                .collect::<Vec<_>>();
            changed[group].evaluations = &wrong;
            assert!(batch
                .linear_weights(&repacking, &setup, &changed, ext(400), &batches)
                .unwrap()
                .check_table(&table)
                .is_err());
        }
    }
}
