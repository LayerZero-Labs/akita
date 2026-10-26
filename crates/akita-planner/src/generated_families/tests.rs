use super::*;

#[test]
fn scalar_preplans_deduplicate_by_exact_producer_and_layout() {
    let key = PolynomialGroupLayout::new(14, 1);
    let preplans = GenerationPreplans::default();

    let schedules = std::thread::scope(|scope| {
        let handles = (0..3)
            .map(|_| {
                scope.spawn(|| {
                    preplans
                        .scalar::<fp128::OneHot>(key, ChunkedWitnessCfg::default_non_chunked())
                        .expect("one-hot plan")
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("preplan worker"))
            .collect::<Vec<_>>()
    });
    let first = schedules[0].clone();
    assert!(schedules.iter().all(|schedule| schedule == &first));
    assert_eq!(preplans.scalar.lock().expect("preplan cache").len(), 1);

    preplans
        .scalar::<fp32::OneHot>(key, ChunkedWitnessCfg::default_non_chunked())
        .expect("same layout under a distinct producer");
    assert_eq!(preplans.scalar.lock().expect("preplan cache").len(), 2);

    let family = ALL_GENERATED_FAMILIES
        .iter()
        .find(|family| family.family_name() == "fp128_onehot")
        .expect("known family");
    let mut spec = empty_emit_spec(
        family,
        (family.policy)().witness_chunk,
        std::path::PathBuf::new(),
    )
    .expect("shipped families declare a valid producer contract");
    spec.keys = vec![key];
    preplans.attach_to_spec(family, &mut spec);
    assert_eq!(
        spec.preplanned,
        vec![(ScheduleLookupKey::single(key), first)]
    );
}

#[test]
fn packaged_family_labels_match_their_geometry() {
    for family in ALL_GENERATED_FAMILIES {
        let name = family.family_name();
        let expected = if let Some((_, suffix)) = name.rsplit_once("_w") {
            let (chunks, levels) = suffix.split_once('r').expect("geometry suffix");
            ChunkedWitnessCfg {
                num_chunks: chunks.parse().unwrap(),
                num_activated_levels: levels.parse().unwrap(),
            }
        } else if name.ends_with("_multi_chunk") {
            MultiChunkProfileId::W8R2.cfg()
        } else {
            ChunkedWitnessCfg::default_non_chunked()
        };
        assert_eq!((family.policy)().witness_chunk, expected, "{name}");
    }
}

#[test]
fn generation_uses_requested_geometry_for_preplans_and_grouped_producers() {
    let family = ALL_GENERATED_FAMILIES
        .iter()
        .find(|family| family.family_name() == "fp128_onehot")
        .unwrap();
    let preplans = GenerationPreplans::default();
    let key = PolynomialGroupLayout::singleton(14);
    preplans
        .scalar::<fp128::OneHot>(key, ChunkedWitnessCfg::default_non_chunked())
        .unwrap();
    let chunking = MultiChunkProfileId::W2R2.cfg();
    let mut spec = empty_emit_spec(family, chunking, std::path::PathBuf::new()).unwrap();
    spec.keys = vec![key];
    preplans.attach_to_spec(family, &mut spec);
    assert!(
        spec.preplanned.is_empty(),
        "a preset preplan must not attach to another geometry"
    );

    let mut spec =
        emit_spec_for_family(family, &preplans, chunking, std::path::PathBuf::new()).unwrap();
    preplans.attach_to_spec(family, &mut spec);
    assert!(!spec.preplanned.is_empty());
    for (_, schedule) in &spec.preplanned {
        assert_eq!(schedule.root.params.witness_chunk, chunking);
    }
    let producer_group = PolynomialGroupLayout::singleton(16);
    let planned = preplans
        .scalar_for_family(family, producer_group, chunking)
        .unwrap();
    let producer =
        GroupCommitPhaseParams::try_from_params(producer_group, &planned.root.params).unwrap();
    let request = spec
        .grouped_requests
        .iter()
        .find(|request| request.key().final_group == PolynomialGroupLayout::new(34, 2))
        .unwrap();
    assert_eq!(
        request.key().precommitteds.as_slice(),
        &[producer, producer]
    );
}
