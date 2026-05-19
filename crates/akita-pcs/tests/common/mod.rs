#![allow(dead_code)]

pub(super) use akita_config::proof_optimized::fp128;
pub(super) use akita_config::CommitmentConfig;
pub(super) use akita_field::{CanonicalField, FieldCore};
pub(super) use akita_prover::AkitaPolyOps;
pub(super) use akita_prover::DensePoly;
pub(super) use akita_prover::OneHotPoly;
pub(super) use akita_prover::{CommittedPolynomials, ProverClaims};
pub(super) use akita_types::LevelParams;
pub(super) use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BasisMode, BlockOrder,
};
pub(super) use akita_verifier::{CommittedOpenings, VerifierClaims};
pub(super) use rand::rngs::StdRng;
pub(super) use rand::{Rng, SeedableRng};
use std::sync::Once;

pub(super) type F = fp128::Field;
pub(super) const STACK_SIZE: usize = 256 * 1024 * 1024;

pub(super) type OneHotCfg = fp128::D64OneHot;
pub(super) const ONEHOT_D: usize = OneHotCfg::D;
pub(super) const ONEHOT_K: usize = ONEHOT_D;

pub(super) type DenseCfg = fp128::D128Full;
pub(super) const DENSE_D: usize = DenseCfg::D;

static INIT_RAYON: Once = Once::new();

pub(super) fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

pub(super) fn random_point(nv: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

pub(super) fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

pub(super) fn prove_input<'a, FF: FieldCore, P, C, H>(
    point: &'a [FF],
    polynomials: &'a [P],
    commitment: &'a C,
    hint: H,
) -> ProverClaims<'a, FF, P, C, H> {
    vec![(
        point,
        CommittedPolynomials {
            polynomials,
            commitment,
            hint,
        },
    )]
}

pub(super) fn verify_input<'a, FF: FieldCore, C>(
    point: &'a [FF],
    openings: &'a [FF],
    commitment: &'a C,
) -> VerifierClaims<'a, FF, C> {
    vec![(
        point,
        CommittedOpenings {
            openings,
            commitment,
        },
    )]
}

pub(super) fn prove_inputs_from_groups<'a, FF: FieldCore, P, C, H>(
    points: &[&'a [FF]],
    polynomials_by_point: &[&'a [P]],
    commitments: &'a [C],
    hints: Vec<H>,
) -> ProverClaims<'a, FF, P, C, H> {
    points
        .iter()
        .zip(polynomials_by_point.iter())
        .zip(commitments.iter())
        .zip(hints)
        .map(|(((point, polynomials), commitment), hint)| {
            (
                *point,
                CommittedPolynomials {
                    polynomials,
                    commitment,
                    hint,
                },
            )
        })
        .collect()
}

pub(super) fn verify_inputs_from_groups<'a, FF: FieldCore, C>(
    points: &[&'a [FF]],
    openings_by_point: &[&'a [FF]],
    commitments: &'a [C],
) -> VerifierClaims<'a, FF, C> {
    points
        .iter()
        .zip(openings_by_point.iter())
        .zip(commitments.iter())
        .map(|((point, openings), commitment)| {
            (
                *point,
                CommittedOpenings {
                    openings,
                    commitment,
                },
            )
        })
        .collect()
}

pub(super) fn opening_from_poly<const D: usize, P: AkitaPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &LevelParams,
) -> F {
    opening_from_poly_with_basis(poly, point, layout, BasisMode::Lagrange)
}

pub(super) fn opening_from_poly_with_basis<const D: usize, P: AkitaPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &LevelParams,
    basis_mode: BasisMode,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        basis_mode,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis_mode)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

pub(super) fn make_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<F, ONEHOT_D, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices).expect("onehot poly")
}

pub(super) fn make_dense_poly(nv: usize, seed: u64) -> DensePoly<F, DENSE_D> {
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<F> = (0..1usize << nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    DensePoly::<F, DENSE_D>::from_field_evals(nv, &evals).expect("dense poly")
}
