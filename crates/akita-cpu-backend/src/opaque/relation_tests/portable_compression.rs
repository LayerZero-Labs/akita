use super::*;
use crate::PortableCompressionState;
use akita_params::RelationRowFamily;
use jolt_field::CanonicalEncoding;

#[test]
fn portable_fragments_match_complete_cpu_witness_in_both_modes() {
    for mode in [
        RingRelationMode::QuotientLift,
        RingRelationMode::ReducedEvaluation,
    ] {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            REDUCED_D,
            2,
            1,
            2,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        params.own_group_mut().opening.num_digits_fold = 3;
        params.ring_relation_mode = mode;
        let (instance, rhs) = reduced_instance(&params);
        let outer_plan = rhs.group_compression_plan(0).unwrap().1;
        let opening_plan = rhs.opening_compression_plan().unwrap();
        let capacity = outer_plan
            .max_setup_field_elements()
            .unwrap()
            .max(opening_plan.max_setup_field_elements().unwrap())
            .max(active_setup_field_len(&params, instance.opening_batch()).unwrap());
        with_reduced_setup(capacity, |ctx, setup| {
            let backend = CpuBackend::<ReducedF, ReducedF>::new(setup.expanded.clone()).unwrap();
            let source = |plan: &akita_params::CompressionChainPlan| {
                RingVec::from_coeffs_with_ring_dim(
                    vec![ReducedF::from_i64(-1); plan.source_coefficients()],
                    REDUCED_D,
                )
                .unwrap()
            };
            let outer = backend
                .compress_relation_image(outer_plan, mode, source(outer_plan))
                .unwrap();
            let opening = backend
                .compress_relation_image(opening_plan, mode, source(opening_plan))
                .unwrap();
            let (compression, _) = materialize_compression_witness(
                ctx,
                &rhs,
                vec![CompressionSourceWitness::from_outer_state(
                    0,
                    outer_plan,
                    outer.state().clone(),
                    outer.compressed_payload().coeffs().to_vec(),
                    mode,
                )
                .unwrap()],
                &source(opening_plan),
                mode,
            )
            .unwrap();
            let h = compression
                .source_for_test(CompressionSourceId::Opening)
                .unwrap();
            assert_eq!(opening.state(), h.material_for_test());
            assert_eq!(
                opening.compressed_payload().coeffs(),
                h.terminal_for_test().coefficients()
            );
            let layout = instance.segment_layout(&params, None).unwrap();
            let fragment = backend
                .materialize_compression_witness_fragment(
                    &params,
                    &rhs,
                    &layout,
                    mode,
                    &[outer.state().clone()],
                    opening.state(),
                )
                .unwrap();
            // Supply the actual compressed RHS required by the ordinary quotient builder.
            let mut y = instance.rhs().coeffs().to_vec();
            let mut offset = 0;
            for family in rhs.row_families().unwrap() {
                let width = family.geometry().physical_coefficient_width();
                match family {
                    RelationRowFamily::CompressionF { map_index, .. }
                        if map_index + 1 == outer_plan.maps().len() =>
                    {
                        y[offset..offset + width]
                            .copy_from_slice(outer.compressed_payload().coeffs())
                    }
                    RelationRowFamily::CompressionH { map_index, .. }
                        if map_index + 1 == opening_plan.maps().len() =>
                    {
                        y[offset..offset + width]
                            .copy_from_slice(opening.compressed_payload().coeffs())
                    }
                    _ => {}
                }
                offset += width;
            }
            let mut gamma_ring = vec![ReducedF::zero(); REDUCED_D];
            gamma_ring[0] = instance.gamma()[0];
            let instance = RingRelationInstance::new(
                instance.group_openings().to_vec(),
                1,
                instance.opening_batch().clone(),
                instance.gamma().to_vec(),
                RingVec::from_coeffs_with_ring_dim(gamma_ring, REDUCED_D).unwrap(),
                RingVec::from_coeffs(y),
                params.role_dims(),
            )
            .unwrap();
            let d = match mode {
                RingRelationMode::QuotientLift => RelationDQuotientWitness::QuotientLift(
                    RingVec::from_coeffs_with_ring_dim(
                        vec![ReducedF::zero(); opening_plan.source_coefficients()],
                        REDUCED_D,
                    )
                    .unwrap(),
                ),
                RingRelationMode::ReducedEvaluation => RelationDQuotientWitness::ReducedEvaluation,
            };
            let full = cpu_recursive_witness_build(
                &instance,
                RingRelationWitness::from_groups(
                    vec![reduced_group_witness(&params, ctx)],
                    d,
                    Some(compression),
                ),
                ctx,
                ctx,
                &params,
                REDUCED_D,
                crate::opaque::OperationBinding::unbound(),
            )
            .unwrap()
            .to_i8_digits();
            let mut patched = vec![0; fragment.witness_len()];
            let mut expected_ranges = Vec::new();
            for layer in layout.compression_layers() {
                expected_ranges.extend(layer.f_spans().iter().map(|(_, span)| span.range()));
                expected_ranges.push(layer.h_span().range());
                if let Some(rows) = layer.f_quotient_rows() {
                    expected_ranges
                        .extend(rows.iter().map(|(_, row)| layout.r_rows()[*row].range()));
                    expected_ranges.push(layout.r_rows()[layer.h_quotient_row().unwrap()].range());
                }
            }
            expected_ranges.sort_by_key(|range| range.start);
            assert_eq!(fragment.patches().len(), expected_ranges.len());
            assert!(fragment
                .patches()
                .windows(2)
                .all(|pair| pair[0].offset() + pair[0].coefficients().len() <= pair[1].offset()));
            for (patch, range) in fragment.patches().iter().zip(&expected_ranges) {
                assert_eq!(patch.offset(), range.start);
                assert_eq!(patch.coefficients().len(), range.len());
                patched[range.clone()].copy_from_slice(patch.coefficients());
                assert_eq!(&patched[range.clone()], &full[range.clone()]);
            }
            let (len, patches) = fragment.clone().into_parts();
            assert_eq!(len, layout.live_coeff_len());
            assert_eq!(patches, fragment.patches());
            for patch in patches {
                let (offset, coefficients) = patch.clone().into_parts();
                assert_eq!(offset, patch.offset());
                assert_eq!(coefficients, patch.coefficients());
            }

            let fragment = |states: &[PortableCompressionState<ReducedF>],
                            h: &PortableCompressionState<ReducedF>,
                            lp: &CommittedGroupParams,
                            wl: &akita_params::WitnessLayout,
                            mode| {
                backend.materialize_compression_witness_fragment(lp, &rhs, wl, mode, states, h)
            };
            assert!(fragment(&[], opening.state(), &params, &layout, mode).is_err());
            assert!(fragment(
                &[outer.state().clone(), outer.state().clone()],
                opening.state(),
                &params,
                &layout,
                mode
            )
            .is_err());
            assert!(fragment(
                &[opening.state().clone()],
                outer.state(),
                &params,
                &layout,
                mode
            )
            .is_err());
            let other_mode = if mode == RingRelationMode::QuotientLift {
                RingRelationMode::ReducedEvaluation
            } else {
                RingRelationMode::QuotientLift
            };
            let wrong_mode = backend
                .compress_relation_image(opening_plan, other_mode, source(opening_plan))
                .unwrap();
            assert!(fragment(
                &[outer.state().clone()],
                wrong_mode.state(),
                &params,
                &layout,
                mode
            )
            .is_err());
            assert!(fragment(
                &[outer.state().clone()],
                opening.state(),
                &params,
                &layout,
                other_mode
            )
            .is_err());
            let mut changed_fold = params.clone();
            changed_fold.own_group_mut().opening.num_digits_fold += 1;
            let updated_layout = instance.segment_layout(&changed_fold, None).unwrap();
            assert!(updated_layout.tail_range().start > layout.tail_range().start);
            assert!(fragment(
                &[outer.state().clone()],
                opening.state(),
                &changed_fold,
                &layout,
                mode
            )
            .is_err());
            fragment(
                &[outer.state().clone()],
                opening.state(),
                &changed_fold,
                &updated_layout,
                mode,
            )
            .unwrap();
            let mut wrong_basis = params.clone();
            wrong_basis.own_group_mut().opening.log_basis_open = 0;
            assert!(fragment(
                &[outer.state().clone()],
                opening.state(),
                &wrong_basis,
                &layout,
                mode
            )
            .is_err());
            if mode == RingRelationMode::QuotientLift {
                wrong_basis.own_group_mut().opening.log_basis_open = 4;
                assert!(fragment(
                    &[outer.state().clone()],
                    opening.state(),
                    &wrong_basis,
                    &layout,
                    mode
                )
                .is_err());
            }
            if let Some(quotients) = opening.state().quotients() {
                assert!(PortableCompressionState::<ReducedF>::quotient_lift(
                    opening.state().witness().clone(),
                    vec![]
                )
                .is_err());
                let mut malformed = quotients.to_vec();
                malformed[0] = RingVec::from_coeffs(vec![ReducedF::zero()]);
                assert!(PortableCompressionState::<ReducedF>::quotient_lift(
                    opening.state().witness().clone(),
                    malformed
                )
                .is_err());
            }
        });
    }
}

#[test]
fn portable_fragment_validates_relation_order_for_grouped_q128_layout() {
    use akita_params::{
        PolynomialGroupLayout, RelationQuotientPlan, RelationWitnessGeometry, ScheduleLookupKey,
        WitnessLayout,
    };
    type F = Prime128OffsetA7F7;
    let catalog = akita_config::test_support::workspace_schedule_catalog::<
        akita_config::proof_optimized::fp128::OneHot,
    >()
    .unwrap();
    let precommitted = catalog
        .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
            16, 1,
        )))
        .unwrap()
        .profiles()
        .final_group;
    let resolved = catalog
        .resolve_key(&ScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![precommitted, precommitted],
        })
        .unwrap();
    let params = &resolved.schedule().root.params;
    let batch = params
        .opening_layout_for_final_group(params.group())
        .unwrap();
    let geometry = RelationWitnessGeometry::for_level(params, &batch, 2).unwrap();
    let rhs = geometry.rhs_layout();
    let layout = WitnessLayout::new(
        params,
        &batch,
        &geometry,
        params.witness_chunk.num_chunks,
        RelationQuotientPlan::for_field_bits(params, F::MODULUS_BITS).unwrap(),
    )
    .unwrap();
    let plans = (0..rhs.groups.len())
        .map(|index| rhs.group_compression_plan(index).unwrap().1)
        .chain(std::iter::once(rhs.opening_compression_plan().unwrap()))
        .collect::<Vec<_>>();
    let capacity = plans
        .iter()
        .map(|plan| plan.max_setup_field_elements().unwrap())
        .max()
        .unwrap();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: capacity,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::new(setup.expanded).unwrap();
    let mut states = plans
        .iter()
        .map(|plan| {
            backend
                .compress_relation_image(
                    plan,
                    params.ring_relation_mode,
                    RingVec::from_coeffs(vec![F::from_u64(7); plan.source_coefficients()]),
                )
                .unwrap()
                .into_parts()
                .1
        })
        .collect::<Vec<_>>();
    let opening = states.pop().unwrap();
    // Fragment construction needs neither prepared matrices nor a proof session.
    let cold = CpuBackend::<F, F>::for_arithmetic_tests();
    let fragment = cold
        .materialize_compression_witness_fragment(
            params,
            rhs,
            &layout,
            params.ring_relation_mode,
            &states,
            &opening,
        )
        .unwrap();
    assert_eq!(
        fragment.patches().len(),
        (rhs.groups.len() + 1) * plans[0].maps().len() * 2
    );
    assert_ne!(states[0].witness().plan(), states[1].witness().plan());
    states.swap(0, 1);
    assert!(cold
        .materialize_compression_witness_fragment(
            params,
            rhs,
            &layout,
            params.ring_relation_mode,
            &states,
            &opening
        )
        .is_err());
}
