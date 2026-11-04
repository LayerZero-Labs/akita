use super::shapes::{CENSUS12, CENSUS20, SHAPES12, SHAPES20};
use super::{Axis, OwnedGraph, Shape, WeightKind};
use crate::offset_eq::tensor_pair::evaluate::eval_multi_axis_families;
use crate::offset_eq::{
    eq_eval_at_index, EqPairTensorAxis, EqPairTensorFamily, MAX_COMPACT_STRIDE_TERMS,
};
use crate::{Field, One, Ring, Zero};
use jolt_field::solinas::Prime128OffsetA7F7 as F;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Inputs {
    left: Vec<F>,
    right: Vec<F>,
    family: EqPairTensorFamily<F>,
}

impl Inputs {
    fn new(shape: Shape<'_>, value: &mut impl FnMut() -> F) -> Self {
        let axes = shape
            .axes
            .iter()
            .map(
                |&Axis {
                     len,
                     strides: [left, right],
                     kind,
                 }| match kind {
                    WeightKind::Unit => {
                        EqPairTensorAxis::unit(len as usize, left as usize, right as usize)
                    }
                    WeightKind::BitProduct => EqPairTensorAxis::bit_product(
                        left as usize,
                        right as usize,
                        (0..len.trailing_zeros())
                            .map(|_| [value(), value()])
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
                    WeightKind::Dense => EqPairTensorAxis::dense(
                        left as usize,
                        right as usize,
                        (0..len).map(|_| value()).collect::<Vec<_>>(),
                    ),
                },
            )
            .collect();
        Self {
            left: (0..shape.bits[0]).map(|_| value()).collect(),
            right: (0..shape.bits[1]).map(|_| value()).collect(),
            // These are normalized fixtures. Avoid value-dependent singleton
            // normalization so the all-zero case still exercises every slot.
            family: EqPairTensorFamily {
                left_offset: shape.offsets[0] as usize,
                right_offset: shape.offsets[1] as usize,
                scalar: value(),
                axes,
            },
        }
    }

    fn online(&self) -> (F, usize) {
        let geometry: Vec<_> = self
            .family
            .axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| {
                axis.supports_bit_recurrence() && (axis.left_stride != 0 || axis.right_stride != 0)
            })
            .map(|(index, axis)| (index, axis.len, axis.left_stride, axis.right_stride))
            .collect();
        let mut work = 0;
        super::super::BYPASS.with(|bypass| bypass.set(true));
        let value = eval_multi_axis_families::<F, false, false>(
            &self.left,
            &self.right,
            &[&self.family],
            &geometry,
            &mut work,
        )
        .unwrap();
        super::super::BYPASS.with(|bypass| bypass.set(false));
        (value, work)
    }

    fn check(&self, graph: &OwnedGraph<'_>) {
        let graph = graph.view();
        let mut charged = 0;
        let actual = graph.try_evaluate(&self.left, &self.right, &self.family, &mut charged);
        let (expected, work) = self.online();
        assert_eq!(actual, Some(expected));
        assert_eq!(
            charged, work,
            "the zero-pruned work budget is part of the contract"
        );
    }
}

#[test]
fn frozen_carry_graph_equivalence_all_keys() {
    let mut rng = StdRng::seed_from_u64(0x4341_5252_5947_5241);
    for (shapes, census) in [(SHAPES12, CENSUS12), (SHAPES20, CENSUS20)] {
        assert_eq!(shapes.len(), census.len());
        for (&shape, &(edges, states)) in shapes.iter().zip(census) {
            let graph = OwnedGraph::compile(shape).unwrap().unwrap();
            assert_eq!(graph.edges.len(), edges);
            assert_eq!(
                graph.seeds
                    + graph
                        .layers
                        .iter()
                        .map(|layer| layer[2] as usize)
                        .sum::<usize>(),
                states
            );
            for _ in 0..2048 {
                Inputs::new(shape, &mut || F::random(&mut rng)).check(&graph);
            }
            for constant in [F::zero(), F::one(), -F::one()] {
                let mut input = Inputs::new(shape, &mut || constant);
                input.check(&graph);
                if constant != F::one() {
                    input.family.scalar = F::one();
                    input.check(&graph);
                }
            }
            let mut counter = 0;
            Inputs::new(shape, &mut || {
                counter += 1;
                if counter % 2 == 0 {
                    F::zero()
                } else {
                    F::one()
                }
            })
            .check(&graph);
            Inputs::new(shape, &mut || {
                counter += 1;
                if counter % 2 == 0 {
                    -F::one()
                } else {
                    F::one()
                }
            })
            .check(&graph);
            for endpoint in [F::zero(), F::one()] {
                let mut input = Inputs::new(shape, &mut || F::random(&mut rng));
                input.left.fill(endpoint);
                input.right.fill(endpoint);
                input.check(&graph);
            }
        }
    }
}

#[test]
fn residual_coordinate_visits_match_online_budget() {
    let shape = Shape {
        bits: [0, 2],
        offsets: [0, 0],
        axes: &[Axis {
            len: 3,
            strides: [0, 1],
            kind: WeightKind::Dense,
        }],
    };
    let graph = OwnedGraph::compile(shape).unwrap().unwrap();
    let mut input = Inputs::new(shape, &mut || F::from_u64(2));
    input.check(&graph);
    let (_, online_work) = input.online();
    assert_eq!(graph.work, online_work);
    let mut work = MAX_COMPACT_STRIDE_TERMS - online_work;
    assert!(graph
        .view()
        .try_evaluate(&input.left, &input.right, &input.family, &mut work)
        .is_some());
    assert_eq!(work, MAX_COMPACT_STRIDE_TERMS);
    let mut work = MAX_COMPACT_STRIDE_TERMS - online_work + 1;
    let before = work;
    assert!(graph
        .view()
        .try_evaluate(&input.left, &input.right, &input.family, &mut work)
        .is_none());
    assert_eq!(work, before);

    input.family.axes[0] = EqPairTensorAxis::dense(0, 1, vec![F::zero(); 3]);
    input.check(&graph);
    assert_eq!(input.online().1, 3, "zero coordinates still cost a visit");
    input.family.scalar = F::zero();
    input.check(&graph);
    assert_eq!(input.online().1, 0, "a zero scalar skips all visits");
}

#[test]
fn carry_graph_misses_and_budget_preserve_fallback() {
    let shape = SHAPES12[0];
    let graph = OwnedGraph::compile(shape).unwrap().unwrap();
    let mut input = Inputs::new(shape, &mut F::one);
    let view = graph.view();
    let mut work = MAX_COMPACT_STRIDE_TERMS;
    assert_eq!(
        view.try_evaluate(&input.left, &input.right, &input.family, &mut work),
        None
    );
    assert_eq!(work, MAX_COMPACT_STRIDE_TERMS);
    input.family.right_offset += 1;
    assert_eq!(
        view.try_evaluate(&input.left, &input.right, &input.family, &mut 0),
        None
    );
    input.family.right_offset -= 1;
    input.right.push(F::one());
    assert_eq!(
        view.try_evaluate(&input.left, &input.right, &input.family, &mut 0),
        None
    );
    let collision = Shape {
        bits: [3, 3],
        offsets: [0, 0],
        axes: &[
            Axis {
                len: 2,
                strides: [1, 1],
                kind: WeightKind::BitProduct,
            },
            Axis {
                len: 2,
                strides: [1, 1],
                kind: WeightKind::BitProduct,
            },
        ],
    };
    assert!(OwnedGraph::compile(collision).unwrap().is_none());
    let oversized = Shape {
        bits: [20, 20],
        offsets: [0, 0],
        axes: &[Axis {
            len: 1025,
            strides: [1, 1],
            kind: WeightKind::Dense,
        }],
    };
    assert!(OwnedGraph::compile(oversized).unwrap().is_none());
}

#[test]
fn carry_graph_domain_filter_matches_dense_definition() {
    let shape = Shape {
        bits: [2, 3],
        offsets: [3, 2],
        axes: &[Axis {
            len: 4,
            strides: [1, 2],
            kind: WeightKind::BitProduct,
        }],
    };
    let graph = OwnedGraph::compile(shape).unwrap().unwrap();
    let mut rng = StdRng::seed_from_u64(0x444f_4d41_494e);
    let input = Inputs::new(shape, &mut || F::random(&mut rng));
    let mut expected = F::zero();
    for coordinate in 0..4 {
        let left = 3 + coordinate;
        let right = 2 + 2 * coordinate;
        if left < 4 && right < 8 {
            expected += input.family.scalar
                * input.family.axes[0].coordinate_weight(coordinate).unwrap()
                * eq_eval_at_index(&input.left, left)
                * eq_eval_at_index(&input.right, right);
        }
    }
    assert_eq!(
        graph
            .view()
            .evaluate(&input.left, &input.right, &input.family)
            .unwrap()
            .0,
        expected
    );
    input.check(&graph);
}

#[test]
fn generated_graph_is_used_by_public_dispatch() {
    if std::env::var("AKITA_REQUIRE_CARRY_GRAPH").as_deref() != Ok("1") {
        return;
    }
    assert!(!super::super::compiled::GRAPHS.is_empty());
    super::super::HITS.with(|hits| hits.set(0));
    let mut rng = StdRng::seed_from_u64(0x7075_626c_6963);
    let shapes = match std::env::var("AKITA_CARRY_EXPONENT").as_deref() {
        Ok("20") => SHAPES20,
        _ => SHAPES12,
    };
    for &shape in shapes {
        let input = Inputs::new(shape, &mut || F::random(&mut rng));
        let call = || {
            crate::offset_eq::eval_boolean_pair_tensor_families::<F, false, false>(
                &input.left,
                &input.right,
                std::slice::from_ref(&input.family),
            )
            .unwrap()
        };
        super::super::BYPASS.with(|bypass| bypass.set(true));
        let expected = call();
        super::super::BYPASS.with(|bypass| bypass.set(false));
        assert_eq!(call(), expected);
    }
    assert!(super::super::HITS.with(std::cell::Cell::get) > 0);
}
