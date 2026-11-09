use super::*;
use crate::commitment::{
    compile_commitment_request, BackendKindId, CommitmentRequestCapabilities, CommitmentSource,
    DenseType, OneHotIndexWidth, OneHotType, PolynomialType, ShortNormType,
};
use crate::opaque::RecursiveWitnessFlat;
use crate::opaque::{CommitInnerPlan, ComputeBackendSetup};
use crate::{AkitaProverSetup, DensePoly, OneHotPoly};
use akita_error::AkitaError;
use akita_params::SetupMatrixCapacity;
use jolt_algebra::{CanonicalEncoding, Prime128Offset275, Ring, Zero};

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
    let prepared = CpuBackend::<F, F>::for_arithmetic_tests()
        .prepare_setup(&setup)
        .unwrap();
    (setup, prepared)
}

fn capabilities(types: Vec<PolynomialType>) -> CommitmentRequestCapabilities {
    CommitmentRequestCapabilities::split::<()>(
        BackendKindId::of::<CpuTestBackend>("cpu-test").unwrap(),
        types,
    )
}

#[test]
fn resolved_homogeneous_groups_cover_dense_and_all_onehot_widths() {
    let (_setup, prepared) = prepared();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
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
                    .materialize(None)
                    .unwrap();
            let actual = backend
                .commit_resolved_inner_host::<D>(&prepared, &resolved, plan())
                .unwrap();
            assert_eq!(actual.len(), 1);
            assert_eq!(actual[0].ring_dim(), D);
            assert_eq!(actual[0].count(), plan().n_a);
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
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
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
    .materialize(None)
    .unwrap();
    let actual = backend
        .commit_resolved_inner_host::<D>(&prepared, &resolved, plan())
        .unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].ring_dim(), D);
    assert_eq!(actual[0].count(), plan().n_a);
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
    .materialize(None)
    .unwrap();
    let mut other_plan = plan();
    other_plan.n_a = 1;

    assert!(CpuBackend::<F, F>::for_arithmetic_tests()
        .commit_resolved_inner_host::<D>(&prepared, &resolved, other_plan)
        .is_err());
}

struct SignedCommitSource<'a>(crate::commitment::SignedBlocks<'a>);

impl CommitmentSource<F> for SignedCommitSource<'_> {
    fn descriptor(&self) -> Result<crate::commitment::CommitSourceDescriptor, AkitaError> {
        let len = 1 << self.0.num_vars();
        crate::commitment::CommitSourceDescriptor::new(
            self.0.num_vars(),
            len,
            len,
            crate::commitment::CommitSourceClass::Dense,
            "signed-test",
        )
    }

    fn committed_centered_reach(&self, _: u128, _: u128) -> Result<(u128, u128), AkitaError> {
        Ok((32768, 32767))
    }

    fn available_polynomial_types(
        &self,
        _: &CommitInnerPlan,
    ) -> Result<crate::commitment::AvailablePolynomialTypes, AkitaError> {
        Ok(crate::commitment::AvailablePolynomialTypes::signed_blocks())
    }

    fn represent_as(
        &self,
        _: crate::commitment::PolynomialTypeSelection,
        _: &CommitInnerPlan,
    ) -> Result<crate::commitment::PolynomialRepresentation<'_, F>, AkitaError> {
        Ok(crate::commitment::PolynomialRepresentation::signed_blocks(
            self.0,
        ))
    }
}

struct SignedTestDecoder<'a>(&'a [i8]);
impl crate::commitment::SignedBlockDecoder for SignedTestDecoder<'_> {
    fn decode_i8(&self, start: usize, out: &mut [i8]) {
        out.copy_from_slice(&self.0[start..start + out.len()]);
    }
}

#[test]
fn signed_block_commit_matches_field_full_and_short_all_storage() {
    use crate::commitment::{AcceptedReach, SignedBlocks};
    let (_setup, prepared) = prepared();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let accepted = Some(AcceptedReach {
        modulus: (-F::from_u64(1)).to_u128_checked().unwrap() + 1,
        centering_threshold: akita_params::field_modulus::<F>().unwrap() / 2,
        negative_abs: None,
        positive: None,
    });
    for (num_digits, log_basis) in [(1, 3), (2, 3), (1, 9), (2, 9)] {
        let plan = CommitInnerPlan {
            num_digits_inner: num_digits,
            log_basis_inner: log_basis,
            num_positions_per_block: 4,
            num_live_blocks: 2,
            ..plan()
        };
        for len in [512, 509] {
            let values: Vec<i8> = (0..len).map(|i| (i % 7) as i8 - 3).collect();
            let wide: Vec<i16> = values
                .iter()
                .map(|&v| i16::from(v) * if log_basis >= 9 { 63 } else { 1 })
                .collect();
            let decoder = SignedTestDecoder(&values);
            let mut field_values = vec![F::zero(); 512];
            for (dst, &v) in field_values.iter_mut().zip(&values) {
                *dst = F::from_i8(v);
            }
            let dense = DensePoly::from_field_evals(9, field_values).unwrap();
            let dense_refs: [&dyn CommitmentSource<F>; 1] = [&dense];
            let expected = backend
                .commit_resolved_inner_host::<D>(
                    &prepared,
                    &compile_commitment_request(
                        &plan,
                        &dense_refs,
                        &capabilities(vec![PolynomialType::Dense(DenseType::Coefficients)]),
                    )
                    .unwrap()
                    .materialize(None)
                    .unwrap(),
                    plan,
                )
                .unwrap();
            let mut wide_fields = vec![F::zero(); 512];
            for (dst, &v) in wide_fields.iter_mut().zip(&wide) {
                *dst = F::from_i16(v);
            }
            let wide_dense = DensePoly::from_field_evals(9, wide_fields).unwrap();
            let wide_refs: [&dyn CommitmentSource<F>; 1] = [&wide_dense];
            let wide_expected = backend
                .commit_resolved_inner_host::<D>(
                    &prepared,
                    &compile_commitment_request(
                        &plan,
                        &wide_refs,
                        &capabilities(vec![PolynomialType::Dense(DenseType::Coefficients)]),
                    )
                    .unwrap()
                    .materialize(None)
                    .unwrap(),
                    plan,
                )
                .unwrap();
            for (blocks, expected) in [
                (SignedBlocks::i8(&values, 9).unwrap(), &expected),
                (SignedBlocks::i16(&wide, 9).unwrap(), &wide_expected),
                (
                    SignedBlocks::decoded_i8(&decoder, len, 9).unwrap(),
                    &expected,
                ),
            ] {
                let source = SignedCommitSource(blocks);
                let refs: [&dyn CommitmentSource<F>; 1] = [&source];
                let resolved = compile_commitment_request(
                    &plan,
                    &refs,
                    &capabilities(vec![PolynomialType::Dense(DenseType::SignedBlocks)]),
                )
                .unwrap()
                .materialize(accepted)
                .unwrap();
                let actual = backend
                    .commit_resolved_inner_host::<D>(&prepared, &resolved, plan)
                    .unwrap();
                assert_eq!(
                    &actual, expected,
                    "digits={num_digits}, basis={log_basis}, len={len}"
                );
            }
        }
    }
}

#[test]
fn signed_block_dispatch_requires_accepted_reach_and_rejects_outside_bound() {
    use crate::commitment::{AcceptedReach, SignedBlocks};
    let (_setup, prepared) = prepared();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let plan = CommitInnerPlan {
        log_basis_inner: 3,
        ..plan()
    };
    let values = vec![2i8; 512];
    let wide = vec![2i16; 512];
    let decoder = SignedTestDecoder(&values);
    for blocks in [
        SignedBlocks::i8(&values, 9).unwrap(),
        SignedBlocks::i16(&wide, 9).unwrap(),
        SignedBlocks::decoded_i8(&decoder, 512, 9).unwrap(),
    ] {
        let source = SignedCommitSource(blocks);
        let refs: [&dyn CommitmentSource<F>; 1] = [&source];
        for accepted in [
            None,
            Some(AcceptedReach {
                modulus: (-F::from_u64(1)).to_u128_checked().unwrap() + 1,
                centering_threshold: akita_params::field_modulus::<F>().unwrap() / 2,
                negative_abs: Some(1),
                positive: Some(1),
            }),
        ] {
            let resolved = compile_commitment_request(
                &plan,
                &refs,
                &capabilities(vec![PolynomialType::Dense(DenseType::SignedBlocks)]),
            )
            .unwrap()
            .materialize(accepted)
            .unwrap();
            let error = backend
                .commit_resolved_inner_host::<D>(&prepared, &resolved, plan)
                .unwrap_err();
            assert!(matches!(error, AkitaError::InvalidInput(_)));
        }
    }
}

#[test]
fn signed_block_full_width_base_two_commit_matches_dense() {
    use crate::commitment::{AcceptedReach, SignedBlocks};
    let modulus = (-F::from_u64(1)).to_u128_checked().unwrap() + 1;
    let num_digits = (128 - (modulus - 1).leading_zeros()) as usize;
    let plan = CommitInnerPlan {
        num_digits_inner: num_digits,
        ..plan()
    };
    let setup = AkitaProverSetup::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: plan.n_a * 8 * D * num_digits,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let values = vec![1i8; 512];
    let wide = vec![1i16; 512];
    let decoder = SignedTestDecoder(&values);
    let dense = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
    let bounds = AcceptedReach {
        modulus,
        centering_threshold: 0,
        negative_abs: Some(modulus - 1),
        positive: Some(0),
    };
    assert_eq!(
        dense.committed_centered_reach(modulus, 0).unwrap(),
        (modulus - 1, 0)
    );
    let refs: [&dyn CommitmentSource<F>; 1] = [&dense];
    let request = compile_commitment_request(
        &plan,
        &refs,
        &capabilities(vec![PolynomialType::Dense(DenseType::Coefficients)]),
    )
    .unwrap()
    .materialize(Some(bounds))
    .unwrap();
    let expected = backend
        .commit_resolved_inner_host::<D>(&prepared, &request, plan)
        .unwrap();
    for blocks in [
        SignedBlocks::i8(&values, 9).unwrap(),
        SignedBlocks::i16(&wide, 9).unwrap(),
        SignedBlocks::decoded_i8(&decoder, 512, 9).unwrap(),
    ] {
        let source = SignedCommitSource(blocks);
        let refs: [&dyn CommitmentSource<F>; 1] = [&source];
        let request = compile_commitment_request(
            &plan,
            &refs,
            &capabilities(vec![PolynomialType::Dense(DenseType::SignedBlocks)]),
        )
        .unwrap()
        .materialize(Some(bounds))
        .unwrap();
        assert_eq!(
            backend
                .commit_resolved_inner_host::<D>(&prepared, &request, plan)
                .unwrap(),
            expected
        );
    }
}
