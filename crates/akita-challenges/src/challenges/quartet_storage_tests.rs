use super::*;
use jolt_field::Prime128OffsetA7F7 as F;

fn compact(blocks: usize, claims: usize) -> Challenges {
    let codes = include_bytes!("../../testdata/checked_hint_codes.bin")
        .chunks_exact(32)
        .take(blocks * claims)
        .map(|record| {
            std::array::from_fn(|i| {
                u64::from_le_bytes(record[i * 8..(i + 1) * 8].try_into().unwrap())
            })
        })
        .collect();
    Challenges::from_checked_quartets(codes, blocks, claims).unwrap()
}

#[test]
fn quartet_storage_preserves_values_without_rechecking_the_draw_threshold() {
    for count in [1, 4, 1024] {
        let compact = compact(1, count);
        assert_eq!(compact.len(), count);
        assert!(!compact.is_empty());
        assert_eq!(compact, compact.clone());
        assert!(compact.materialized.get().is_none());
        for stride in [1, 4] {
            let embedded = compact
                .embed_subring_positions(64, stride, 64 * stride)
                .unwrap();
            let powers = (0..64 * stride)
                .scan(F::from_u64(1), |power, _| {
                    let value = *power;
                    *power *= F::from_u64(37);
                    Some(value)
                })
                .collect::<Vec<_>>();
            let actual = embedded.evals_at_pows::<F, F>(&powers).unwrap();
            assert!(embedded.materialized.get().is_none());
            let expected = embedded
                .as_slice()
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, F>(&powers).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            let eager = Challenges::from_sparse(embedded.as_slice().to_vec(), 1, count).unwrap();
            assert_eq!(embedded, eager);
            assert_eq!(embedded.evals_at_pows::<F, F>(&powers).unwrap(), expected);
            assert!(embedded.evals_at_pows::<F, F>(&powers[..8]).is_err());
        }
    }
}

#[test]
fn quartet_embeddings_share_storage_and_preserve_claim_selection_and_errors() {
    let source = compact(2, 2);
    let once = source.embed_subring_positions(64, 4, 256).unwrap();
    let twice = once.embed_subring_positions(256, 2, 512).unwrap();
    let direct = source.embed_subring_positions(64, 8, 512).unwrap();
    let (ChallengeStorage::Quartets(a), ChallengeStorage::Quartets(b)) =
        (&source.storage, &twice.storage)
    else {
        panic!("quartet storage expected");
    };
    assert!(Arc::ptr_eq(a, b));
    assert_eq!(twice, direct);
    assert!(twice.materialized.get().is_none());
    let subset = twice.select_claims(&[1, 0, 1]).unwrap();
    assert_eq!(subset.len(), 6);
    assert_eq!(subset.num_claims(), 3);
    assert_eq!(subset.num_live_blocks_per_claim(), 2);
    assert_eq!(&subset.as_slice()[..2], &twice.as_slice()[2..4]);
    assert_eq!(&subset.as_slice()[2..4], &twice.as_slice()[..2]);
    assert_eq!(&subset.as_slice()[4..], &twice.as_slice()[2..4]);
    assert!(matches!(
        source.select_claims(&[2]),
        Err(AkitaError::InvalidSize {
            expected: 6,
            actual: 4
        })
    ));
    let error = source
        .eval_at_pows::<F, F>(4, &[F::from_u64(1); 64])
        .unwrap_err();
    assert!(error.to_string().contains("out of range for 4 challenges"));
}

#[test]
fn empty_sparse_and_quartet_batches_have_the_same_public_values() {
    let sparse = Challenges::from_sparse(Vec::new(), 0, 0).unwrap();
    let quartet = Challenges::from_checked_quartets(Vec::new(), 0, 0).unwrap();
    assert!(sparse.is_empty());
    assert!(quartet.is_empty());
    assert_eq!(sparse.len(), quartet.len());
    assert_eq!(sparse, quartet);
    assert_eq!(sparse.as_slice(), quartet.as_slice());
}
