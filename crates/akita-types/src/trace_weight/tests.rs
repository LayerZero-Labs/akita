use super::{
    build_trace_weight_table_field_block_weights, build_trace_weight_table_field_terms,
    build_trace_weight_table_ring_block_weights, build_trace_weight_table_ring_terms,
    eval_trace_terms_closed, eval_trace_weight_at_point, trace_weight_mle_eval,
    TraceFieldBlockOpening, TraceOpeningAtPoint, TraceRingBlockOpening, TraceTerm,
    TraceWeightLayout,
};
use crate::{
    block_rings_at_opening, lagrange_weights, recover_ring_subfield_inner_product,
    reduce_inner_opening_to_ring_element, BasisMode,
};
use akita_algebra::CyclotomicRing;
use akita_field::{Prime128OffsetA7F7, RandomSampling};
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Prime128OffsetA7F7;
const D: usize = 128;
const LOG_BASIS: u32 = 3;

fn field_block_weights_layout() -> TraceWeightLayout {
    TraceWeightLayout {
        ring_bits: 7,
        col_bits: 2,
        opening_digit_offset: 0,
        num_blocks: 2,
        num_digits_open: 2,
        r_vars: 1,
        log_basis: LOG_BASIS,
    }
}

fn field_block_weights_layout_with_offset() -> TraceWeightLayout {
    TraceWeightLayout {
        ring_bits: 3,
        col_bits: 3,
        opening_digit_offset: 4,
        num_blocks: 2,
        num_digits_open: 2,
        r_vars: 1,
        log_basis: LOG_BASIS,
    }
}

fn random_point(rng: &mut StdRng, len: usize) -> Vec<F> {
    (0..len).map(|_| F::random(rng)).collect()
}

fn random_opening_points(rng: &mut StdRng, layout: &TraceWeightLayout) -> (Vec<F>, Vec<F>) {
    let inner_open = random_point(rng, layout.ring_bits);
    let b_open = random_point(rng, layout.r_vars);
    (inner_open, b_open)
}

fn random_folded_block(rng: &mut StdRng) -> CyclotomicRing<F, D> {
    let coeffs: Vec<F> = (0..D).map(|_| F::random(rng)).collect();
    CyclotomicRing::from_coefficients(
        coeffs
            .try_into()
            .unwrap_or_else(|_| panic!("D={D} coefficient vector length mismatch")),
    )
}

fn weighted_folded_block_sum(
    folded_blocks: &[CyclotomicRing<F, D>],
    block_weights: &[F],
) -> CyclotomicRing<F, D> {
    folded_blocks
        .iter()
        .zip(block_weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (block, weight)| {
            acc + block.scale(weight)
        })
}

fn trace_weight_witness_dot<E: akita_field::FieldCore>(witness: &[E], trace_weight: &[E]) -> E {
    witness
        .iter()
        .zip(trace_weight.iter())
        .fold(E::zero(), |acc, (w, weight)| acc + *w * *weight)
}

#[test]
fn closed_form_matches_dense_table_field_block_weights() {
    let layout = field_block_weights_layout();
    let mut rng = StdRng::seed_from_u64(0x7ACE_0001);

    for _ in 0..16 {
        let (inner_open, b_open) = random_opening_points(&mut rng, &layout);
        let inner_opening_ring =
            reduce_inner_opening_to_ring_element::<F, D>(&inner_open, BasisMode::Lagrange).unwrap();
        let block_weights = lagrange_weights(&b_open).unwrap();
        let table = build_trace_weight_table_field_block_weights::<F, F, D>(
            &layout,
            &block_weights,
            &inner_opening_ring,
        )
        .unwrap();

        let ring_point = random_point(&mut rng, layout.ring_bits);
        let col_point = random_point(&mut rng, layout.col_bits);
        let dense = trace_weight_mle_eval(&layout, &table, &col_point, &ring_point).unwrap();
        let term = TraceFieldBlockOpening {
            block_offset: 0,
            block_weights: block_weights.clone(),
            inner_opening_ring,
        };
        let closed = eval_trace_weight_at_point::<F, F, D, 1>(
            &layout,
            &ring_point,
            &col_point,
            TraceOpeningAtPoint::Field {
                terms: std::slice::from_ref(&term),
            },
        )
        .unwrap();
        assert_eq!(dense, closed);
    }
}

#[test]
fn closed_form_matches_dense_table_with_opening_digit_offset() {
    const D8: usize = 8;
    let layout = field_block_weights_layout_with_offset();
    let mut rng = StdRng::seed_from_u64(0x7ACE_0004);

    for _ in 0..16 {
        let (inner_open, b_open) = random_opening_points(&mut rng, &layout);
        let inner_opening_ring =
            reduce_inner_opening_to_ring_element::<F, D8>(&inner_open, BasisMode::Lagrange)
                .unwrap();
        let block_weights = lagrange_weights(&b_open).unwrap();
        let table = build_trace_weight_table_field_block_weights::<F, F, D8>(
            &layout,
            &block_weights,
            &inner_opening_ring,
        )
        .unwrap();

        let ring_point = random_point(&mut rng, layout.ring_bits);
        let col_point = random_point(&mut rng, layout.col_bits);
        let dense = trace_weight_mle_eval(&layout, &table, &col_point, &ring_point).unwrap();
        let term = TraceFieldBlockOpening {
            block_offset: 0,
            block_weights: block_weights.clone(),
            inner_opening_ring,
        };
        let closed = eval_trace_weight_at_point::<F, F, D8, 1>(
            &layout,
            &ring_point,
            &col_point,
            TraceOpeningAtPoint::Field {
                terms: std::slice::from_ref(&term),
            },
        )
        .unwrap();
        assert_eq!(dense, closed);
    }
}

#[test]
fn closed_form_matches_dense_table_multiple_field_terms() {
    let layout = TraceWeightLayout {
        ring_bits: 7,
        col_bits: 3,
        opening_digit_offset: 0,
        num_blocks: 4,
        num_digits_open: 2,
        r_vars: 2,
        log_basis: LOG_BASIS,
    };
    let mut rng = StdRng::seed_from_u64(0x7ACE_0005);

    for _ in 0..16 {
        let inner_open_0 = random_point(&mut rng, layout.ring_bits);
        let inner_open_1 = random_point(&mut rng, layout.ring_bits);
        let inner_ring_0 =
            reduce_inner_opening_to_ring_element::<F, D>(&inner_open_0, BasisMode::Lagrange)
                .unwrap();
        let inner_ring_1 =
            reduce_inner_opening_to_ring_element::<F, D>(&inner_open_1, BasisMode::Lagrange)
                .unwrap();
        let block_weights_0 = lagrange_weights(&random_point(&mut rng, 1)).unwrap();
        let block_weights_1 = lagrange_weights(&random_point(&mut rng, 1)).unwrap();
        let terms = vec![
            TraceFieldBlockOpening {
                block_offset: 0,
                block_weights: block_weights_0,
                inner_opening_ring: inner_ring_0,
            },
            TraceFieldBlockOpening {
                block_offset: 2,
                block_weights: block_weights_1,
                inner_opening_ring: inner_ring_1,
            },
        ];
        let table = build_trace_weight_table_field_terms::<F, F, D>(&layout, &terms).unwrap();

        let ring_point = random_point(&mut rng, layout.ring_bits);
        let col_point = random_point(&mut rng, layout.col_bits);
        let dense = trace_weight_mle_eval(&layout, &table, &col_point, &ring_point).unwrap();
        let closed = eval_trace_weight_at_point::<F, F, D, 1>(
            &layout,
            &ring_point,
            &col_point,
            TraceOpeningAtPoint::Field { terms: &terms },
        )
        .unwrap();
        assert_eq!(dense, closed);
    }
}

#[test]
fn witness_dot_matches_ring_subfield_inner_product_field_block_weights() {
    let layout = field_block_weights_layout();
    let mut rng = StdRng::seed_from_u64(0x7ACE_0002);

    for _ in 0..16 {
        let (inner_open, b_open) = random_opening_points(&mut rng, &layout);
        let inner_opening_ring =
            reduce_inner_opening_to_ring_element::<F, D>(&inner_open, BasisMode::Lagrange).unwrap();
        let block_weights = lagrange_weights(&b_open).unwrap();
        let table = build_trace_weight_table_field_block_weights::<F, F, D>(
            &layout,
            &block_weights,
            &inner_opening_ring,
        )
        .unwrap();

        let folded_blocks: Vec<CyclotomicRing<F, D>> = (0..layout.num_blocks)
            .map(|_| random_folded_block(&mut rng))
            .collect();
        let combined = weighted_folded_block_sum(&folded_blocks, &block_weights);
        let expected =
            recover_ring_subfield_inner_product::<F, F, D>(&combined, &inner_opening_ring).unwrap();

        let mut witness = vec![F::zero(); layout.table_len().unwrap()];
        for (block, folded) in folded_blocks.iter().enumerate() {
            let col = layout.opening_digit_col_index(block, 0);
            for ring_coord in 0..(1usize << layout.ring_bits) {
                let idx = layout.witness_index(col, ring_coord);
                witness[idx] = folded.coefficients()[ring_coord];
            }
        }

        let actual = trace_weight_witness_dot(&witness, &table);
        assert_eq!(actual, expected);
    }
}

mod ring_block_weights {
    use super::*;
    use crate::{basis_weights, embed_ring_subfield_vector};
    use akita_field::{AkitaError, Ext2, Fp32, FpExt4, FpExt8, LiftBase};
    use std::marker::PhantomData;

    type F2 = Fp32<251>;
    type E2 = Ext2<F2>;
    type F4 = Fp32<251>;
    type E4 = FpExt4<F4>;
    type F8 = Fp32<251>;
    type E8 = FpExt8<F8>;

    const LOG_BASIS: u32 = 3;

    fn ring_block_weights_layout<const D: usize>() -> TraceWeightLayout {
        TraceWeightLayout {
            ring_bits: D.trailing_zeros() as usize,
            col_bits: 2,
            opening_digit_offset: 0,
            num_blocks: 2,
            num_digits_open: 2,
            r_vars: 1,
            log_basis: LOG_BASIS,
        }
    }

    fn ring_block_weights_multi_term_layout<const D: usize>() -> TraceWeightLayout {
        TraceWeightLayout {
            ring_bits: D.trailing_zeros() as usize,
            col_bits: 3,
            opening_digit_offset: 0,
            num_blocks: 4,
            num_digits_open: 2,
            r_vars: 2,
            log_basis: LOG_BASIS,
        }
    }

    fn trace_inner_open_len<F, E, const D: usize>() -> usize
    where
        F: akita_field::FieldCore,
        E: akita_field::ExtField<F>,
    {
        (D / E::EXT_DEGREE).trailing_zeros() as usize
    }

    fn packed_inner_point<F, E, const D: usize>(trace_inner_open: &[E]) -> CyclotomicRing<F, D>
    where
        F: akita_field::FieldCore + akita_field::FromPrimitiveInt,
        E: akita_field::ExtField<F> + crate::FpExtEncoding<F> + akita_field::FieldCore,
    {
        let weights = basis_weights(trace_inner_open, BasisMode::Lagrange).unwrap();
        embed_ring_subfield_vector(
            &weights,
            AkitaError::InvalidInput("trace inner opening is not embeddable".to_string()),
        )
        .unwrap()
    }

    fn random_extension_point<E: akita_field::RandomSampling>(
        rng: &mut StdRng,
        len: usize,
    ) -> Vec<E> {
        (0..len).map(|_| E::random(rng)).collect()
    }

    fn random_opening_points<F, E, const D: usize>(
        rng: &mut StdRng,
        layout: &TraceWeightLayout,
    ) -> (Vec<E>, Vec<E>)
    where
        F: akita_field::FieldCore,
        E: akita_field::ExtField<F> + akita_field::RandomSampling,
    {
        let trace_inner_open = random_extension_point(rng, trace_inner_open_len::<F, E, D>());
        let b_open = random_extension_point(rng, layout.r_vars);
        (trace_inner_open, b_open)
    }

    fn random_folded_block<
        F: akita_field::FieldCore + akita_field::RandomSampling,
        const D: usize,
    >(
        rng: &mut StdRng,
    ) -> CyclotomicRing<F, D> {
        let coeffs: Vec<F> = (0..D).map(|_| F::random(rng)).collect();
        CyclotomicRing::from_coefficients(
            coeffs
                .try_into()
                .unwrap_or_else(|_| panic!("D={D} coefficient vector length mismatch")),
        )
    }

    fn run_closed_form_matches_dense_table<F, E, const D: usize, const K: usize>()
    where
        F: akita_field::FieldCore
            + akita_field::CanonicalField
            + akita_field::FromPrimitiveInt
            + akita_field::Invertible
            + akita_field::RandomSampling,
        E: crate::FpExtEncoding<F>
            + akita_field::ExtField<F>
            + akita_field::FieldCore
            + akita_field::FromPrimitiveInt
            + akita_field::RandomSampling,
    {
        let layout = ring_block_weights_layout::<D>();
        let mut rng = StdRng::seed_from_u64(0x7ACE_1000 + D as u64);

        for _ in 0..8 {
            let (trace_inner_open, b_open) = random_opening_points::<F, E, D>(&mut rng, &layout);
            let packed_inner = packed_inner_point::<F, E, D>(
                &trace_inner_open[..trace_inner_open_len::<F, E, D>()],
            );
            let block_rings = block_rings_at_opening::<F, E, D>(&b_open).unwrap();
            let table = build_trace_weight_table_ring_block_weights::<F, E, D>(
                &layout,
                &block_rings,
                &packed_inner,
            )
            .unwrap();

            let ring_point = random_extension_point(&mut rng, layout.ring_bits);
            let col_point = random_extension_point(&mut rng, layout.col_bits);
            let dense = trace_weight_mle_eval(&layout, &table, &col_point, &ring_point).unwrap();
            let term = TraceRingBlockOpening {
                block_offset: 0,
                block_rings: block_rings.clone(),
                packed_inner_point: packed_inner,
            };
            let closed = eval_trace_weight_at_point::<F, E, D, K>(
                &layout,
                &ring_point,
                &col_point,
                TraceOpeningAtPoint::Ring {
                    terms: std::slice::from_ref(&term),
                    _ext: PhantomData,
                },
            )
            .unwrap();
            assert_eq!(dense, closed);
        }
    }

    fn run_witness_dot_matches_ring_subfield_inner_product<F, E, const D: usize, const K: usize>()
    where
        F: akita_field::FieldCore
            + akita_field::CanonicalField
            + akita_field::FromPrimitiveInt
            + akita_field::Invertible
            + akita_field::RandomSampling,
        E: crate::FpExtEncoding<F>
            + akita_field::ExtField<F>
            + akita_field::FieldCore
            + akita_field::FromPrimitiveInt
            + akita_field::RandomSampling,
    {
        let layout = ring_block_weights_layout::<D>();
        let mut rng = StdRng::seed_from_u64(0x7ACE_2000 + D as u64);

        for _ in 0..8 {
            let (trace_inner_open, b_open) = random_opening_points::<F, E, D>(&mut rng, &layout);
            let packed_inner = packed_inner_point::<F, E, D>(
                &trace_inner_open[..trace_inner_open_len::<F, E, D>()],
            );
            let block_rings = block_rings_at_opening::<F, E, D>(&b_open).unwrap();
            let table = build_trace_weight_table_ring_block_weights::<F, E, D>(
                &layout,
                &block_rings,
                &packed_inner,
            )
            .unwrap();

            let folded_blocks: Vec<CyclotomicRing<F, D>> = (0..layout.num_blocks)
                .map(|_| random_folded_block::<F, D>(&mut rng))
                .collect();
            let combined = folded_blocks
                .iter()
                .zip(block_rings.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (block, weight)| {
                    acc + *block * *weight
                });
            let expected =
                recover_ring_subfield_inner_product::<F, E, D>(&combined, &packed_inner).unwrap();

            let mut witness = vec![E::zero(); layout.table_len().unwrap()];
            for (block, folded) in folded_blocks.iter().enumerate() {
                let col = layout.opening_digit_col_index(block, 0);
                for ring_coord in 0..(1usize << layout.ring_bits) {
                    let idx = layout.witness_index(col, ring_coord);
                    witness[idx] = E::lift_base(folded.coefficients()[ring_coord]);
                }
            }

            let actual = trace_weight_witness_dot(&witness, &table);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn multiple_ring_terms_match_dense_table_and_witness_dot_k4() {
        let layout = ring_block_weights_multi_term_layout::<8>();
        let mut rng = StdRng::seed_from_u64(0x7ACE_3004);

        for _ in 0..8 {
            let inner_0 = random_extension_point(&mut rng, trace_inner_open_len::<F4, E4, 8>());
            let inner_1 = random_extension_point(&mut rng, trace_inner_open_len::<F4, E4, 8>());
            let packed_0 = packed_inner_point::<F4, E4, 8>(&inner_0);
            let packed_1 = packed_inner_point::<F4, E4, 8>(&inner_1);
            let block_rings_0 =
                block_rings_at_opening::<F4, E4, 8>(&random_extension_point(&mut rng, 1)).unwrap();
            let block_rings_1 =
                block_rings_at_opening::<F4, E4, 8>(&random_extension_point(&mut rng, 1)).unwrap();
            let terms = vec![
                TraceRingBlockOpening {
                    block_offset: 0,
                    block_rings: block_rings_0,
                    packed_inner_point: packed_0,
                },
                TraceRingBlockOpening {
                    block_offset: 2,
                    block_rings: block_rings_1,
                    packed_inner_point: packed_1,
                },
            ];
            let table = build_trace_weight_table_ring_terms::<F4, E4, 8>(&layout, &terms).unwrap();

            let ring_point = random_extension_point(&mut rng, layout.ring_bits);
            let col_point = random_extension_point(&mut rng, layout.col_bits);
            let dense = trace_weight_mle_eval(&layout, &table, &col_point, &ring_point).unwrap();
            let closed = eval_trace_weight_at_point::<F4, E4, 8, 4>(
                &layout,
                &ring_point,
                &col_point,
                TraceOpeningAtPoint::Ring {
                    terms: &terms,
                    _ext: PhantomData,
                },
            )
            .unwrap();
            assert_eq!(dense, closed);

            let folded_blocks: Vec<CyclotomicRing<F4, 8>> = (0..layout.num_blocks)
                .map(|_| random_folded_block::<F4, 8>(&mut rng))
                .collect();
            let expected = terms.iter().fold(E4::zero(), |acc, term| {
                let combined = term.block_rings.iter().enumerate().fold(
                    CyclotomicRing::<F4, 8>::zero(),
                    |sum, (local_block, block_ring)| {
                        sum + folded_blocks[term.block_offset + local_block] * *block_ring
                    },
                );
                acc + recover_ring_subfield_inner_product::<F4, E4, 8>(
                    &combined,
                    &term.packed_inner_point,
                )
                .unwrap()
            });

            let mut witness = vec![E4::zero(); layout.table_len().unwrap()];
            for (block, folded) in folded_blocks.iter().enumerate() {
                let col = layout.opening_digit_col_index(block, 0);
                for ring_coord in 0..(1usize << layout.ring_bits) {
                    let idx = layout.witness_index(col, ring_coord);
                    witness[idx] = E4::lift_base(folded.coefficients()[ring_coord]);
                }
            }

            let actual = trace_weight_witness_dot(&witness, &table);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn closed_form_matches_dense_table_ring_block_weights_k2() {
        run_closed_form_matches_dense_table::<F2, E2, 4, 2>();
    }

    #[test]
    fn witness_dot_matches_ring_subfield_inner_product_ring_block_weights_k2() {
        run_witness_dot_matches_ring_subfield_inner_product::<F2, E2, 4, 2>();
    }

    #[test]
    fn closed_form_matches_dense_table_ring_block_weights_k4() {
        run_closed_form_matches_dense_table::<F4, E4, 8, 4>();
    }

    #[test]
    fn witness_dot_matches_ring_subfield_inner_product_ring_block_weights_k4() {
        run_witness_dot_matches_ring_subfield_inner_product::<F4, E4, 8, 4>();
    }

    #[test]
    fn closed_form_matches_dense_table_ring_block_weights_k8() {
        run_closed_form_matches_dense_table::<F8, E8, 16, 8>();
    }

    #[test]
    fn witness_dot_matches_ring_subfield_inner_product_ring_block_weights_k8() {
        run_witness_dot_matches_ring_subfield_inner_product::<F8, E8, 16, 8>();
    }
}

/// Tests for the short-data closed-form evaluator [`eval_trace_terms_closed`],
/// which reconstructs the same MLE the prover materializes but takes one `Tr_H`
/// per claim instead of folding every block.
mod closed_terms {
    use super::*;
    use crate::{basis_weights, embed_ring_subfield_vector, reduce_inner_opening_to_ring_element};
    use akita_field::{AkitaError, Ext2, Fp32, FpExt4, FpExt8};

    type Fk = Fp32<251>;
    type E2 = Ext2<Fk>;
    type E4 = FpExt4<Fk>;
    type E8 = FpExt8<Fk>;

    const LB: u32 = 3;

    fn ext_point<E: akita_field::RandomSampling>(rng: &mut StdRng, len: usize) -> Vec<E> {
        (0..len).map(|_| E::random(rng)).collect()
    }

    fn trace_inner_len<E, const D: usize>() -> usize
    where
        E: akita_field::ExtField<Fk>,
    {
        (D / E::EXT_DEGREE).trailing_zeros() as usize
    }

    fn packed_inner<E, const D: usize>(trace_inner_open: &[E]) -> CyclotomicRing<Fk, D>
    where
        E: akita_field::ExtField<Fk> + crate::FpExtEncoding<Fk> + akita_field::FieldCore,
    {
        let weights = basis_weights(trace_inner_open, BasisMode::Lagrange).unwrap();
        embed_ring_subfield_vector(
            &weights,
            AkitaError::InvalidInput("trace inner opening is not embeddable".to_string()),
        )
        .unwrap()
    }

    /// One random single-claim K>1 round: closed form must equal the dense MLE.
    fn run_single_ring<E, const D: usize>(seed: u64, layout: &TraceWeightLayout)
    where
        E: crate::FpExtEncoding<Fk>
            + akita_field::ExtField<Fk>
            + akita_field::FieldCore
            + akita_field::FromPrimitiveInt
            + akita_field::RandomSampling,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..8 {
            let trace_inner_open = ext_point::<E>(&mut rng, trace_inner_len::<E, D>());
            let inner = packed_inner::<E, D>(&trace_inner_open);
            let b_open = ext_point::<E>(&mut rng, layout.r_vars);
            let block_rings = block_rings_at_opening::<Fk, E, D>(&b_open).unwrap();
            let table = build_trace_weight_table_ring_block_weights::<Fk, E, D>(
                layout,
                &block_rings,
                &inner,
            )
            .unwrap();

            let ring_point = ext_point::<E>(&mut rng, layout.ring_bits);
            let col_point = ext_point::<E>(&mut rng, layout.col_bits);
            let dense = trace_weight_mle_eval(layout, &table, &col_point, &ring_point).unwrap();

            let term = TraceTerm {
                block_offset: 0,
                b_open: b_open.clone(),
                basis: BasisMode::Lagrange,
                packed_inner_point: inner,
                coefficient: E::one(),
            };
            let closed = eval_trace_terms_closed::<Fk, E, D>(
                layout,
                &ring_point,
                &col_point,
                std::slice::from_ref(&term),
            )
            .unwrap();
            assert_eq!(dense, closed);
        }
    }

    fn ring_layout<const D: usize>() -> TraceWeightLayout {
        TraceWeightLayout {
            ring_bits: D.trailing_zeros() as usize,
            col_bits: 2,
            opening_digit_offset: 0,
            num_blocks: 2,
            num_digits_open: 2,
            r_vars: 1,
            log_basis: LB,
        }
    }

    #[test]
    fn closed_terms_match_dense_k2() {
        run_single_ring::<E2, 4>(0x5EED_0002, &ring_layout::<4>());
    }

    #[test]
    fn closed_terms_match_dense_k4() {
        run_single_ring::<E4, 8>(0x5EED_0004, &ring_layout::<8>());
    }

    #[test]
    fn closed_terms_match_dense_k8() {
        run_single_ring::<E8, 16>(0x5EED_0008, &ring_layout::<16>());
    }

    #[test]
    fn closed_terms_match_dense_k1_single() {
        let layout = TraceWeightLayout {
            ring_bits: 3,
            col_bits: 3,
            opening_digit_offset: 4,
            num_blocks: 2,
            num_digits_open: 2,
            r_vars: 1,
            log_basis: LB,
        };
        const D8: usize = 8;
        let mut rng = StdRng::seed_from_u64(0x5EED_0001);
        for _ in 0..16 {
            let inner_open = random_point(&mut rng, layout.ring_bits);
            let inner =
                reduce_inner_opening_to_ring_element::<F, D8>(&inner_open, BasisMode::Lagrange)
                    .unwrap();
            let b_open = random_point(&mut rng, layout.r_vars);
            let block_weights = lagrange_weights(&b_open).unwrap();
            let table = build_trace_weight_table_field_block_weights::<F, F, D8>(
                &layout,
                &block_weights,
                &inner,
            )
            .unwrap();
            let ring_point = random_point(&mut rng, layout.ring_bits);
            let col_point = random_point(&mut rng, layout.col_bits);
            let dense = trace_weight_mle_eval(&layout, &table, &col_point, &ring_point).unwrap();

            let term = TraceTerm {
                block_offset: 0,
                b_open: b_open.clone(),
                basis: BasisMode::Lagrange,
                packed_inner_point: inner,
                coefficient: F::one(),
            };
            let closed = eval_trace_terms_closed::<F, F, D8>(
                &layout,
                &ring_point,
                &col_point,
                std::slice::from_ref(&term),
            )
            .unwrap();
            assert_eq!(dense, closed);
        }
    }

    #[test]
    fn closed_terms_match_dense_k1_multi_claim() {
        // Two claims tiled along the block axis (num_blocks = 2 claims * 2^1).
        let layout = TraceWeightLayout {
            ring_bits: 3,
            col_bits: 4,
            opening_digit_offset: 4,
            num_blocks: 4,
            num_digits_open: 2,
            r_vars: 2,
            log_basis: LB,
        };
        const D8: usize = 8;
        let mut rng = StdRng::seed_from_u64(0x5EED_1001);
        for _ in 0..16 {
            let mut terms = Vec::new();
            let mut dense_terms = Vec::new();
            for claim in 0..2usize {
                let inner_open = random_point(&mut rng, layout.ring_bits);
                let inner =
                    reduce_inner_opening_to_ring_element::<F, D8>(&inner_open, BasisMode::Lagrange)
                        .unwrap();
                let b_open = random_point(&mut rng, 1);
                let block_weights = lagrange_weights(&b_open).unwrap();
                dense_terms.push(TraceFieldBlockOpening {
                    block_offset: claim * 2,
                    block_weights,
                    inner_opening_ring: inner,
                });
                terms.push(TraceTerm {
                    block_offset: claim * 2,
                    b_open,
                    basis: BasisMode::Lagrange,
                    packed_inner_point: inner,
                    coefficient: F::one(),
                });
            }
            let table =
                build_trace_weight_table_field_terms::<F, F, D8>(&layout, &dense_terms).unwrap();
            let ring_point = random_point(&mut rng, layout.ring_bits);
            let col_point = random_point(&mut rng, layout.col_bits);
            let dense = trace_weight_mle_eval(&layout, &table, &col_point, &ring_point).unwrap();
            let closed =
                eval_trace_terms_closed::<F, F, D8>(&layout, &ring_point, &col_point, &terms)
                    .unwrap();
            assert_eq!(dense, closed);
        }
    }

    #[test]
    fn closed_terms_match_dense_k4_multi_claim() {
        // Two K=4 claims tiled along the block axis.
        let layout = TraceWeightLayout {
            ring_bits: 3,
            col_bits: 4,
            opening_digit_offset: 4,
            num_blocks: 4,
            num_digits_open: 2,
            r_vars: 2,
            log_basis: LB,
        };
        const D8: usize = 8;
        let mut rng = StdRng::seed_from_u64(0x5EED_4001);
        for _ in 0..8 {
            let mut terms = Vec::new();
            let mut dense_terms = Vec::new();
            for claim in 0..2usize {
                let trace_inner_open = ext_point::<E4>(&mut rng, trace_inner_len::<E4, D8>());
                let inner = packed_inner::<E4, D8>(&trace_inner_open);
                let b_open = ext_point::<E4>(&mut rng, 1);
                let block_rings = block_rings_at_opening::<Fk, E4, D8>(&b_open).unwrap();
                dense_terms.push(TraceRingBlockOpening {
                    block_offset: claim * 2,
                    block_rings,
                    packed_inner_point: inner,
                });
                terms.push(TraceTerm {
                    block_offset: claim * 2,
                    b_open,
                    basis: BasisMode::Lagrange,
                    packed_inner_point: inner,
                    coefficient: E4::one(),
                });
            }
            let table =
                build_trace_weight_table_ring_terms::<Fk, E4, D8>(&layout, &dense_terms).unwrap();
            let ring_point = ext_point::<E4>(&mut rng, layout.ring_bits);
            let col_point = ext_point::<E4>(&mut rng, layout.col_bits);
            let dense = trace_weight_mle_eval(&layout, &table, &col_point, &ring_point).unwrap();
            let closed =
                eval_trace_terms_closed::<Fk, E4, D8>(&layout, &ring_point, &col_point, &terms)
                    .unwrap();
            assert_eq!(dense, closed);
        }
    }

    #[test]
    fn closed_terms_coefficient_scales_linearly() {
        let layout = ring_layout::<8>();
        const D8: usize = 8;
        let mut rng = StdRng::seed_from_u64(0x5EED_C0EF);
        for _ in 0..8 {
            let trace_inner_open = ext_point::<E4>(&mut rng, trace_inner_len::<E4, D8>());
            let inner = packed_inner::<E4, D8>(&trace_inner_open);
            let b_open = ext_point::<E4>(&mut rng, layout.r_vars);
            let ring_point = ext_point::<E4>(&mut rng, layout.ring_bits);
            let col_point = ext_point::<E4>(&mut rng, layout.col_bits);
            let coeff = E4::random(&mut rng);

            let unit = TraceTerm {
                block_offset: 0,
                b_open: b_open.clone(),
                basis: BasisMode::Lagrange,
                packed_inner_point: inner,
                coefficient: E4::one(),
            };
            let scaled = TraceTerm {
                coefficient: coeff,
                ..unit.clone()
            };
            let base = eval_trace_terms_closed::<Fk, E4, D8>(
                &layout,
                &ring_point,
                &col_point,
                std::slice::from_ref(&unit),
            )
            .unwrap();
            let got = eval_trace_terms_closed::<Fk, E4, D8>(
                &layout,
                &ring_point,
                &col_point,
                std::slice::from_ref(&scaled),
            )
            .unwrap();
            assert_eq!(got, coeff * base);
        }
    }
}
