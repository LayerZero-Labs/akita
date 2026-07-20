use super::*;
use akita_algebra::poly::multilinear_eval;
use akita_field::{Ext2, Fp32, FpExt4, FpExt8, Prime128OffsetA7F7, RandomSampling};
use rand::{rngs::StdRng, SeedableRng};

type TestField = Prime128OffsetA7F7;

fn term_for_cross_product(
    basis: BasisMode,
    source_ring_dimension: usize,
    coefficient: u64,
    physical_starts: [usize; 2],
) -> EvaluationTraceTerm<TestField> {
    EvaluationTraceTerm {
        coefficient: TestField::from_u64(coefficient),
        block_opening_point: vec![TestField::from_u64(17), TestField::from_u64(19)].into(),
        basis,
        group_block_count: 3,
        source_ring_dimension,
        opening_digit_weights: vec![TestField::one(), TestField::from_u64(3)].into(),
        inner_trace: (0..source_ring_dimension)
            .map(|index| TestField::from_u64(2 * index as u64 + coefficient))
            .collect::<Vec<_>>()
            .into(),
        segments: vec![
            EvaluationTraceSegment {
                physical_coefficient_start: physical_starts[0],
                global_block_start: 0,
                block_count: 2,
            },
            EvaluationTraceSegment {
                physical_coefficient_start: physical_starts[1],
                global_block_start: 2,
                block_count: 1,
            },
        ],
    }
}

fn materialize_term_oracle(term: &EvaluationTraceTerm<TestField>, table: &mut [TestField]) {
    let block_weights = basis_weights_prefix(
        &term.block_opening_point,
        term.basis,
        term.group_block_count,
    )
    .unwrap();
    let digit_inner = term.digit_inner_weights().unwrap();
    for segment in &term.segments {
        for local_block in 0..segment.block_count {
            let block = segment.global_block_start + local_block;
            let start = segment.physical_coefficient_start + local_block * digit_inner.len();
            for (offset, &weight) in digit_inner.iter().enumerate() {
                table[start + offset] += term.coefficient * block_weights[block] * weight;
            }
        }
    }
}

#[test]
fn terms_match_flat_oracle_across_bases_chunks_and_ring_dimensions() {
    const PHYSICAL_FIELD_LEN: usize = 256;
    let output_scale = TestField::from_u64(29);
    let point: Vec<TestField> = (0..8)
        .map(|index| TestField::from_u64(31 + 2 * index as u64))
        .collect();

    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        for source_ring_dimension in [2, 4, 8] {
            let terms = vec![
                term_for_cross_product(basis, source_ring_dimension, 5, [0, 128]),
                term_for_cross_product(basis, source_ring_dimension, 11, [64, 192]),
            ];
            let weights = EvaluationTraceWeights {
                terms: terms.clone(),
                physical_field_len: PHYSICAL_FIELD_LEN,
                num_vars: 8,
            };
            let mut expected = vec![TestField::zero(); PHYSICAL_FIELD_LEN];
            for term in &terms {
                materialize_term_oracle(term, &mut expected);
            }

            assert_eq!(
                weights.evaluate_at_point(&point).unwrap(),
                multilinear_eval(&expected, &point).unwrap()
            );
            for destination_ring_dimension in [2, 4, 8] {
                let materialized = weights
                    .materialize_prover_table::<TestField>(destination_ring_dimension, output_scale)
                    .unwrap()
                    .materialize_dense(
                        PHYSICAL_FIELD_LEN / destination_ring_dimension,
                        destination_ring_dimension,
                    );
                assert_eq!(
                    materialized,
                    expected
                        .iter()
                        .map(|&weight| output_scale * weight)
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

type BaseField = Fp32<251>;
type Extension2 = Ext2<BaseField>;
type Extension4 = FpExt4<BaseField>;
type Extension8 = FpExt8<BaseField>;

fn extension_trace_factorization_matches_per_block_rows<E, const D: usize>(seed: u64)
where
    E: FpExtEncoding<BaseField> + ExtField<BaseField> + FromPrimitiveInt + RandomSampling,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let inner_point_len = (D / E::EXT_DEGREE).trailing_zeros() as usize;
    for _ in 0..8 {
        let inner_point: Vec<E> = (0..inner_point_len).map(|_| E::random(&mut rng)).collect();
        let inner_weights = basis_weights(&inner_point, BasisMode::Lagrange).unwrap();
        let packed_inner = crate::embed_ring_subfield_vector::<BaseField, E, D>(
            &inner_weights,
            AkitaError::InvalidInput("test inner point does not embed".into()),
        )
        .unwrap();
        let block_point: Vec<E> = (0..2).map(|_| E::random(&mut rng)).collect();
        let block_weights = basis_weights(&block_point, BasisMode::Lagrange).unwrap();
        let block_rings =
            crate::block_rings_at_opening::<BaseField, E, D>(&block_point, block_weights.len())
                .unwrap();
        let inner_trace = trace_open_ring_row::<BaseField, E, D>(
            &CyclotomicRing::one(),
            &packed_inner,
            D.trailing_zeros() as usize,
        )
        .unwrap();
        for (&block_weight, block_ring) in block_weights.iter().zip(&block_rings) {
            let row = trace_open_ring_row::<BaseField, E, D>(
                block_ring,
                &packed_inner,
                D.trailing_zeros() as usize,
            )
            .unwrap();
            assert_eq!(
                row,
                inner_trace
                    .iter()
                    .map(|&inner| block_weight * inner)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn extension_inner_trace_factorization_is_exact() {
    extension_trace_factorization_matches_per_block_rows::<Extension2, 8>(0x2008);
    extension_trace_factorization_matches_per_block_rows::<Extension4, 8>(0x4008);
    extension_trace_factorization_matches_per_block_rows::<Extension8, 16>(0x8010);
}
