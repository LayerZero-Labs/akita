//! Production distributed setup-offload NTT requirement regression.
use crate::opaque::{NttExecutionRequirements, NttOperationCluster};
use akita_config::proof_optimized::fp128;
use akita_config::RecursiveCommitmentConfig;
use akita_types::{
    centered_quotient_requires_i16_tail, AkitaScheduleLookupKey, InnerCommitMatrixParams,
    NttCacheKey, NttTransformDomain, PolynomialGroupLayout,
};

type W8R2Cfg = RecursiveCommitmentConfig<fp128::OneHotMultiChunk>;
fn w8r2_profiling_key(
    base_catalog: &akita_config::TrustedScheduleCatalog<fp128::OneHotMultiChunk>,
) -> AkitaScheduleLookupKey {
    let pre_group = PolynomialGroupLayout::new(16, 1);
    let precommitted = base_catalog
        .resolve_key(&AkitaScheduleLookupKey::single(pre_group))
        .expect("independent row")
        .profiles()
        .final_group;
    AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![precommitted, precommitted],
    }
}

#[test]
fn w8r2_ntt_requirements_match_distributed_a_tail_decisions() {
    let base_catalog =
        akita_config::test_support::workspace_schedule_catalog::<fp128::OneHotMultiChunk>()
            .expect("base W8R2 catalog");
    let catalog = akita_config::test_support::workspace_schedule_catalog::<W8R2Cfg>()
        .expect("recursive W8R2 catalog");
    let key = w8r2_profiling_key(&base_catalog);
    let schedule = catalog
        .resolve_key(&key)
        .expect("W8R2 schedule")
        .schedule()
        .clone();
    let first_recursive = &schedule.recursive_folds[0].params;
    assert_eq!(first_recursive.witness_chunk.num_chunks, 8);
    let prefix = first_recursive
        .setup_prefix()
        .expect("W8R2 first recursive fold must consume a setup prefix");
    let witness_a = &first_recursive.inner().matrix;
    let prefix_a = &prefix.profile.inner.matrix;
    assert_eq!(
        (
            witness_a.ring_dimension(),
            witness_a.output_rank(),
            witness_a.input_width(),
        ),
        (128, 3, 2_048),
    );
    assert_eq!(
        (
            prefix_a.ring_dimension(),
            prefix_a.output_rank(),
            prefix_a.input_width(),
        ),
        (128, 4, 2_048),
    );

    let witness_tail =
        NttCacheKey::from_matrix_shape(128, 3, 2_048, NttTransformDomain::I16TailBothTransforms)
            .expect("valid W8R2 witness tail key");
    let prefix_tail =
        NttCacheKey::from_matrix_shape(128, 4, 2_048, NttTransformDomain::I16TailBothTransforms)
            .expect("valid W8R2 prefix tail key");
    let requirements =
        NttExecutionRequirements::from_prove_schedule(&schedule).expect("NTT requirements");
    let has_tail = |expected| {
        requirements.entries().iter().any(|entry| {
            entry.fold_level == 1
                && entry.cluster == NttOperationCluster::RingSwitch
                && entry.key == expected
        })
    };
    let expects_tail =
        |matrix: &InnerCommitMatrixParams, log_basis: u32, num_digits_fold: usize| {
            let (negative, positive) =
                akita_types::sis::balanced_digit_representable_bounds(log_basis, num_digits_fold);
            let rhs_abs_bound = u64::try_from(
                negative
                    .max(positive)
                    .checked_mul(first_recursive.witness_chunk.num_chunks as u128)
                    .expect("W8 aggregated fold bound"),
            )
            .expect("W8 aggregated fold bound fits u64");
            centered_quotient_requires_i16_tail(
                matrix.sis_modulus_profile(),
                matrix.ring_dimension(),
                rhs_abs_bound,
            )
            .expect("W8 tail decision")
        };
    assert_eq!(
        has_tail(witness_tail),
        expects_tail(
            witness_a,
            first_recursive.open().digits.log_basis,
            first_recursive.num_digits_fold(),
        ),
        "recursive-witness prewarm must match its aggregated W8 bound"
    );
    assert_eq!(
        has_tail(prefix_tail),
        expects_tail(
            prefix_a,
            prefix.opening.log_basis_open,
            prefix.opening.num_digits_fold,
        ),
        "incoming-prefix prewarm must inherit the consuming W8 chunk count"
    );
}
