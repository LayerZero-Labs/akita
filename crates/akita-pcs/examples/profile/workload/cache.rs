use akita_prover::PreparedNttCacheMetric;

pub(super) fn assert_profile_ntt_cache_did_not_grow(
    before: &[PreparedNttCacheMetric],
    after: &[PreparedNttCacheMetric],
) {
    assert!(
        after.len() <= before.len() && after.iter().all(|metric| before.contains(metric)),
        "commit/prove added to the prewarmed profile NTT cache: before={before:?}, after={after:?}"
    );
}
