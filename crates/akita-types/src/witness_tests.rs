use super::*;
use crate::{
    CommitmentPayloadMode, PolynomialGroupLayout, RingRelationMode, SisModulusProfileId,
    COMPRESSION_MAP_COUNT,
};

#[test]
fn default_is_single_chunk() {
    let cfg = ChunkedWitnessCfg::default();
    assert_eq!(cfg, ChunkedWitnessCfg::default_non_chunked());
    assert_eq!(cfg.num_chunks, 1);
    assert_eq!(cfg.num_activated_levels, 0);
    assert!(!cfg.uses_multi_chunk());
    cfg.validate().expect("default config is valid");
}

#[test]
fn d64_production_uses_multi_chunk() {
    let cfg = ChunkedWitnessCfg::d64_production();
    assert_eq!(cfg, MultiChunkProfileId::PRODUCTION.cfg());
    assert_eq!(cfg.num_chunks, 8);
    assert_eq!(cfg.num_activated_levels, 2);
    assert!(cfg.uses_multi_chunk());
    cfg.validate().expect("d64_production is valid");
}

#[test]
fn multi_chunk_profile_grid_roundtrip() {
    for (index, profile) in MultiChunkProfileId::ALL.into_iter().enumerate() {
        assert_eq!(profile.index(), index);
        assert_eq!(MultiChunkProfileId::from_index(index), profile);
        let cfg = ChunkedWitnessCfg::from_profile(profile);
        assert_eq!(cfg.profile_id(), Some(profile));
        cfg.validate().expect("grid profile is valid");
    }
}

fn test_layout(num_chunks: usize) -> (CommittedGroupParams, OpeningClaimsLayout, WitnessLayout) {
    let mut lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        32,
        2,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(4, 25, 1, 2, 2)
    .expect("test params");
    lp.own_group_mut().opening.num_digits_fold = 3;
    let opening_batch = OpeningClaimsLayout::new(0, 2).expect("opening batch");
    let relation_geometry =
        RelationWitnessGeometry::for_evaluation_trace_execution(&lp, &opening_batch)
            .expect("relation geometry");
    let layout = WitnessLayout::new(
        &lp,
        &opening_batch,
        &relation_geometry,
        num_chunks,
        RelationQuotientPlan::quotient_lift(2).unwrap(),
    )
    .expect("witness layout");
    (lp, opening_batch, layout)
}

#[test]
fn layout_indexing_matches_digit_innermost_semantics() {
    let (lp, opening_batch, layout) = test_layout(2);
    let unit = layout.unit(0, 1).expect("unit");
    let depth_fold = lp.num_digits_fold();
    assert_ne!(
        lp.inner().digits.num_digits,
        lp.outer().digits.num_digits,
        "fixture must distinguish witness and commitment depths"
    );
    assert_eq!(unit.global_block_range(), 3..7);
    let dims = lp.role_dims();
    assert_eq!(
        unit.e_coefficient_index(dims.d_d(), 2, 2, 1, 6, 0, 1, 0)
            .expect("e"),
        unit.e_range().start + 15 * dims.d_a()
    );
    assert_eq!(
        unit.t_coefficient_index(dims.d_a(), dims.d_b(), 2, 1, 2, 0, 5, 0, 0, 1, 0,)
            .expect("t"),
        unit.t_range().start + 5 * dims.d_a()
    );
    assert_eq!(
        unit.z_coefficient_index(dims.d_a(), 4, 1, depth_fold, 1, 0, 0, 0)
            .expect("z"),
        unit.z_range().start + depth_fold * dims.d_a()
    );
    assert_eq!(
        layout.r_coefficient_index(2, 1, 0, 0).expect("r"),
        layout.r_rows()[2].range().start
            + layout.r_rows()[2].geometry().physical_coefficient_width()
    );
    assert_eq!(opening_batch.num_total_polynomials(), 2);
}

#[test]
fn balanced_chunks_are_exact_and_contiguous() {
    let (_, _, layout) = test_layout(2);
    let mut units = layout.units_for_group(0).expect("units");
    let first = units.next().expect("first unit");
    let second = units.next().expect("second unit");
    assert!(units.next().is_none());
    assert_eq!(first.global_block_range(), 0..3);
    assert_eq!(second.global_block_range(), 3..7);
    assert_eq!(first.t_range().end, second.z_range().start);
    let support = layout.negative_binary_support_intervals();
    assert_eq!(support.len(), COMPRESSION_MAP_COUNT);
    assert_eq!(second.t_range().end, layout.tail_range().start);
    assert!(support[0].start < support[0].end);
    assert!(support[0].end < support[1].start);
    assert!(support[1].end <= layout.live_coeff_len());
    assert_eq!(layout.tail_range().end, layout.live_coeff_len());
    assert_eq!(layout.compression_layers().len(), COMPRESSION_MAP_COUNT);
    for (map_index, layer) in layout.compression_layers().iter().enumerate() {
        assert_eq!(layer.map_index(), map_index);
        assert_eq!(layer.f_spans().len(), 1);
        let (group_index, span) = &layer.f_spans()[0];
        assert_eq!(*group_index, 0);
        assert_eq!(span.range().len(), span.map().padded_digit_count());
        assert_eq!(support[map_index].start, span.range().start);
        assert_eq!(support[map_index].end, layer.h_span().range().end);
        assert_eq!(
            layout
                .f_compression_coefficient_index(0, map_index, 1, 2)
                .expect("F address"),
            span.range().start + span.map().ring_dimension() + 2
        );
        assert_eq!(
            layout
                .h_compression_coefficient_index(map_index, 1, 2)
                .expect("H address"),
            layer.h_span().range().start + layer.h_span().map().ring_dimension() + 2
        );
        let f_quotient_rows = layer
            .f_quotient_rows()
            .expect("quotient-lift compression rows");
        let f_quotient = &layout.r_rows()[f_quotient_rows[0].1];
        let h_quotient = &layout.r_rows()[layer.h_quotient_row().expect("quotient-lift H row")];
        assert_eq!(f_quotient_rows[0].0, 0);
        assert_eq!(f_quotient.range().start, layer.h_span().range().end);
        assert_eq!(h_quotient.range().start, f_quotient.range().end);
    }
    assert_eq!(layout.group_num_live_blocks(0).expect("fold count"), 7);
}

#[test]
fn relation_mode_is_the_authority_for_raw_and_compressed_quotient_ranges() {
    for payload_mode in [
        CommitmentPayloadMode::Raw,
        CommitmentPayloadMode::Compressed,
    ] {
        let (mut lifted_params, opening_batch, _) = test_layout(2);
        lifted_params.payload_mode = payload_mode;
        lifted_params.ring_relation_mode = RingRelationMode::QuotientLift;
        let relation_geometry =
            RelationWitnessGeometry::for_evaluation_trace_execution(&lifted_params, &opening_batch)
                .expect("relation geometry");
        let lifted = WitnessLayout::new(
            &lifted_params,
            &opening_batch,
            &relation_geometry,
            2,
            RelationQuotientPlan::quotient_lift(2).unwrap(),
        )
        .expect("quotient-lift layout");
        lifted
            .validate_internal_ranges()
            .expect("valid quotient-lift ranges");

        let mut reduced_params = lifted_params.clone();
        reduced_params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
        let reduced = WitnessLayout::new(
            &reduced_params,
            &opening_batch,
            &relation_geometry,
            2,
            RelationQuotientPlan::ReducedEvaluation,
        )
        .expect("reduced-evaluation layout without quotient metadata");
        reduced
            .validate_internal_ranges()
            .expect("valid reduced-evaluation ranges");
        let breakdown = QuotientCoefficientBreakdown::for_reduced_counterfactual(
            &reduced_params,
            PolynomialGroupLayout::new(0, 2),
            2,
            32,
        )
        .expect("canonical quotient-lift counterfactual");
        assert!(breakdown.ordinary > 0);
        assert_eq!(breakdown.compression > 0, payload_mode.is_compressed());
        assert!(QuotientCoefficientBreakdown::for_reduced_counterfactual(
            &lifted_params,
            PolynomialGroupLayout::new(0, 2),
            2,
            32,
        )
        .is_err());

        assert!(matches!(
            lifted.relation_quotient_layout(),
            RelationQuotientLayout::QuotientLift { .. }
        ));
        assert_eq!(lifted.quotient_depth(), Some(2));
        assert!(!lifted.r_rows().is_empty());
        assert!(matches!(
            reduced.relation_quotient_layout(),
            RelationQuotientLayout::ReducedEvaluation
        ));
        assert_eq!(reduced.quotient_depth(), None);
        assert!(reduced.r_rows().is_empty());
        assert!(reduced.r_coefficient_index(0, 0, 0, 0).is_err());
        assert!(RelationQuotientPlan::quotient_lift(0).is_err());
        assert!(WitnessLayout::new(
            &lifted_params,
            &opening_batch,
            &relation_geometry,
            2,
            RelationQuotientPlan::ReducedEvaluation,
        )
        .is_err());
        assert_eq!(lifted.units(), reduced.units());
        assert_eq!(lifted.tail_range().start, reduced.tail_range().start);
        assert!(reduced.live_coeff_len() < lifted.live_coeff_len());

        if payload_mode.is_compressed() {
            assert_eq!(reduced.compression_layers().len(), COMPRESSION_MAP_COUNT);
            for (lifted_layer, reduced_layer) in lifted
                .compression_layers()
                .iter()
                .zip(reduced.compression_layers())
            {
                assert_eq!(lifted_layer.map_index(), reduced_layer.map_index());
                assert_eq!(lifted_layer.f_spans().len(), reduced_layer.f_spans().len());
                assert_eq!(
                    lifted_layer.h_span().range().len(),
                    reduced_layer.h_span().range().len()
                );
                assert!(lifted_layer.f_quotient_rows().is_some());
                assert!(lifted_layer.h_quotient_row().is_some());
                assert!(reduced_layer.f_quotient_rows().is_none());
                assert!(reduced_layer.h_quotient_row().is_none());
            }
        } else {
            assert_eq!(reduced.tail_range().start, reduced.live_coeff_len());
            assert!(reduced.compression_layers().is_empty());
        }
    }
}

#[test]
fn layout_rejects_out_of_range_semantic_indices() {
    let (lp, _, layout) = test_layout(2);
    let unit = layout.unit(0, 0).expect("unit");
    let depth_fold = lp.num_digits_fold();
    let dims = lp.role_dims();
    assert!(unit
        .e_coefficient_index(dims.d_d(), 2, 2, 2, 0, 0, 0, 0)
        .is_err());
    assert!(unit
        .t_coefficient_index(dims.d_a(), dims.d_b(), 2, 1, 2, 0, 0, 1, 0, 0, 0)
        .is_err());
    assert!(unit
        .z_coefficient_index(dims.d_a(), 4, 1, depth_fold, 4, 0, 0, 0)
        .is_err());
    assert!(layout
        .r_coefficient_index(layout.r_rows().len(), 0, 0, 0)
        .is_err());
}

#[test]
fn layout_rejects_mismatched_shapes() {
    let (lp, _, layout) = test_layout(2);
    let unit = layout.unit(0, 0).expect("unit");
    let dims = lp.role_dims();
    assert!(unit
        .e_coefficient_index(dims.d_d(), 1, 2, 0, 0, 0, 0, 0)
        .is_err());
    assert!(unit
        .t_coefficient_index(dims.d_a(), dims.d_b(), 2, 2, 2, 0, 0, 0, 0, 0, 0,)
        .is_err());
    assert!(unit
        .z_coefficient_index(dims.d_a(), 1, 1, 1, 0, 0, 0, 0)
        .is_err());
}

#[test]
fn validate_rejects_invalid_configs() {
    assert!(ChunkedWitnessCfg {
        num_chunks: 0,
        num_activated_levels: 0,
    }
    .validate()
    .is_err());
    assert!(ChunkedWitnessCfg {
        num_chunks: 6,
        num_activated_levels: 2,
    }
    .validate()
    .is_err());
    assert!(ChunkedWitnessCfg {
        num_chunks: 1,
        num_activated_levels: 2,
    }
    .validate()
    .is_err());
    assert!(ChunkedWitnessCfg {
        num_chunks: 8,
        num_activated_levels: 0,
    }
    .validate()
    .is_err());
    assert!(ChunkedWitnessCfg {
        num_chunks: 128,
        num_activated_levels: 1,
    }
    .validate()
    .is_err());
    ChunkedWitnessCfg {
        num_chunks: MAX_WITNESS_CHUNKS,
        num_activated_levels: 1,
    }
    .validate()
    .expect("max chunk count is valid");
    for n in [2usize, 4, 8, 16] {
        ChunkedWitnessCfg {
            num_chunks: n,
            num_activated_levels: 1,
        }
        .validate()
        .expect("power-of-two chunk counts validate");
    }
}
