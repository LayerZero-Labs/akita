use super::*;
use crate::compute::{
    compile_commitment_request, BackendKindId, CommitInnerPlan, CommitmentRequestCapabilities,
    CommitmentSource, ComputeBackendSetup, DenseType, OneHotIndexWidth, OneHotType, PolynomialType,
    ShortNormType,
};
use crate::{AkitaProverSetup, DensePoly, OneHotPoly, RecursiveWitnessFlat};
use akita_types::{RingVec, SetupMatrixCapacity};
use jolt_field::{Prime128Offset275, Ring};

type F = Prime128Offset275;
const D: usize = 64;

struct CpuTestBackend;

fn plan() -> CommitInnerPlan {
    CommitInnerPlan {
        ring_dimension: D,
        num_live_blocks: 1,
        n_a: 2,
        num_positions_per_block: 8,
        num_digits_inner: 1,
        log_basis_inner: 1,
    }
}

fn prepared() -> (AkitaProverSetup<F>, CpuPreparedSetup<F>) {
    let plan = plan();
    let setup = AkitaProverSetup::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: plan.n_a * plan.num_positions_per_block * D,
        },
    )
    .unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    (setup, prepared)
}

fn capabilities(types: Vec<PolynomialType>) -> CommitmentRequestCapabilities {
    CommitmentRequestCapabilities::split::<()>(
        BackendKindId::of::<CpuTestBackend>("cpu-test").unwrap(),
        types,
    )
}

fn assert_rows_equal(left: &RingVec<F>, right: &RingVec<F>) {
    assert_eq!(left.ring_dim(), right.ring_dim());
    assert_eq!(left.coeffs(), right.coeffs());
}

#[test]
fn resolved_homogeneous_groups_cover_dense_and_all_onehot_widths() {
    let (_setup, prepared) = prepared();
    let backend = CpuBackend::DEFAULT;
    let dense = DensePoly::from_field_evals(
        9,
        (0usize..512)
            .map(|index| F::from_u64(u64::from(index.is_multiple_of(5))))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    macro_rules! onehot {
        ($index:ty) => {
            OneHotPoly::<F, $index>::new(
                64,
                (0usize..8)
                    .map(|chunk| {
                        (!chunk.is_multiple_of(3)).then_some(((chunk * 11) % 64) as $index)
                    })
                    .collect(),
            )
            .unwrap()
        };
    }
    let u8_poly = onehot!(u8);
    let u16_poly = onehot!(u16);
    let u32_poly = onehot!(u32);
    let usize_poly = onehot!(usize);
    macro_rules! assert_group {
        ($poly:expr, $kind:expr) => {{
            let source_refs: [&dyn CommitmentSource<F>; 1] = [$poly];
            let resolved =
                compile_commitment_request(&plan(), &source_refs, &capabilities(vec![$kind]))
                    .unwrap()
                    .materialize()
                    .unwrap();
            let actual = backend
                .commit_resolved_inner_host::<F, D>(&prepared, &resolved, plan())
                .unwrap();
            assert_eq!(actual.len(), 1);
            assert_eq!(actual[0].inner_rows.ring_dim(), D);
            assert_eq!(actual[0].inner_rows.count(), plan().n_a);
        }};
    }

    assert_group!(&dense, PolynomialType::Dense(DenseType::Coefficients));
    assert_group!(
        &u8_poly,
        PolynomialType::OneHot(OneHotType::new(64, OneHotIndexWidth::U8).unwrap())
    );
    assert_group!(
        &u16_poly,
        PolynomialType::OneHot(OneHotType::new(64, OneHotIndexWidth::U16).unwrap())
    );
    assert_group!(
        &u32_poly,
        PolynomialType::OneHot(OneHotType::new(64, OneHotIndexWidth::U32).unwrap())
    );
    assert_group!(
        &usize_poly,
        PolynomialType::OneHot(OneHotType::new(64, OneHotIndexWidth::Usize).unwrap())
    );
}

#[test]
fn resolved_packed_short_norm_commits_without_full_decode() {
    let (_setup, prepared) = prepared();
    let backend = CpuBackend::DEFAULT;
    let witness = RecursiveWitnessFlat::from_i8_digits(
        (0usize..512)
            .map(|index| match index % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect(),
    );
    let source_refs: [&dyn CommitmentSource<F>; 1] = [&witness];
    let resolved = compile_commitment_request(
        &plan(),
        &source_refs,
        &capabilities(vec![PolynomialType::ShortNorm(
            ShortNormType::new(2).unwrap(),
        )]),
    )
    .unwrap()
    .materialize()
    .unwrap();
    let actual = backend
        .commit_resolved_inner_host::<F, D>(&prepared, &resolved, plan())
        .unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].inner_rows.ring_dim(), D);
    assert_eq!(actual[0].inner_rows.count(), plan().n_a);
}

#[test]
fn resolved_dense_reuses_exact_plan_digit_cache() {
    let (_setup, prepared) = prepared();
    let backend = CpuBackend::DEFAULT;
    let dense = DensePoly::from_field_evals(
        9,
        (0usize..512)
            .map(|index| F::from_u64(u64::from(index.is_multiple_of(7))))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    // Populate the exact dense digit cache through its production consumer.
    // Request compilation should then prefer that zero-copy representation.
    let _ = dense.decompose_fold::<D>(
        &[],
        plan().num_positions_per_block,
        plan().num_digits_inner,
        plan().log_basis_inner,
    );
    let coefficient_refs: [&dyn CommitmentSource<F>; 1] = [&dense];
    let coefficient_resolved = compile_commitment_request(
        &plan(),
        &coefficient_refs,
        &capabilities(vec![PolynomialType::Dense(DenseType::Coefficients)]),
    )
    .unwrap()
    .materialize()
    .unwrap();
    let expected = backend
        .commit_resolved_inner_host::<F, D>(&prepared, &coefficient_resolved, plan())
        .unwrap();
    let source_refs: [&dyn CommitmentSource<F>; 1] = [&dense];
    let resolved = compile_commitment_request(
        &plan(),
        &source_refs,
        &capabilities(vec![
            PolynomialType::Dense(DenseType::PredecomposedDigits),
            PolynomialType::Dense(DenseType::Coefficients),
        ]),
    )
    .unwrap()
    .materialize()
    .unwrap();

    assert_eq!(
        resolved[0].selected_type(),
        Some(PolynomialType::Dense(DenseType::PredecomposedDigits))
    );
    let actual = backend
        .commit_resolved_inner_host::<F, D>(&prepared, &resolved, plan())
        .unwrap();
    assert_rows_equal(&actual[0].inner_rows, &expected[0].inner_rows);
}

#[test]
fn resolved_sources_reject_a_different_execution_plan() {
    let (_setup, prepared) = prepared();
    let dense = DensePoly::from_field_evals(9, vec![F::from_u64(0); 512]).unwrap();
    let source_refs: [&dyn CommitmentSource<F>; 1] = [&dense];
    let resolved = compile_commitment_request(
        &plan(),
        &source_refs,
        &capabilities(vec![PolynomialType::Dense(DenseType::Coefficients)]),
    )
    .unwrap()
    .materialize()
    .unwrap();
    let mut other_plan = plan();
    other_plan.n_a = 1;

    assert!(CpuBackend::DEFAULT
        .commit_resolved_inner_host::<F, D>(&prepared, &resolved, other_plan)
        .is_err());
}
