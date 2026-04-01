#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{
    hachi_batched_root_layout, Fp128FullCommitmentConfig, Fp128OneHotCommitmentConfig,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BlockOrder,
};
use hachi_pcs::protocol::proof::HachiBatchedProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::{CommitmentConfig, HachiCommitmentLayout};
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentScheme, HachiDeserialize, HachiError, HachiSerialize,
    Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::{Mutex, Once};

type F = Fp128<0xffffffffffffffffffffffffffffe941>;
const STACK_SIZE: usize = 256 * 1024 * 1024;

type OneHotCfg = Fp128OneHotCommitmentConfig;
const ONEHOT_D: usize = OneHotCfg::D;
const ONEHOT_K: usize = ONEHOT_D;

type DenseCfg = Fp128FullCommitmentConfig;
const DENSE_D: usize = DenseCfg::D;

type PointCommitments<const D: usize, Cfg> =
    Vec<Vec<<HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::Commitment>>;
type PointHints<const D: usize, Cfg> =
    Vec<Vec<<HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::CommitHint>>;

static INIT_RAYON: Once = Once::new();
static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

fn random_point(nv: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn opening_from_poly<const D: usize, P: HachiPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &HachiCommitmentLayout,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, BasisMode::Lagrange)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

fn make_onehot_poly(layout: &HachiCommitmentLayout, seed: u64) -> OneHotPoly<F, ONEHOT_D, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars)
        .expect("onehot poly")
}

fn make_dense_poly(nv: usize, seed: u64) -> DensePoly<F, DENSE_D> {
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<F> = (0..1usize << nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    DensePoly::<F, DENSE_D>::from_field_evals(nv, &evals).expect("dense poly")
}

fn build_group_slices<'a, T>(values: &'a [T], group_sizes: &[usize]) -> Vec<&'a [T]> {
    let mut groups = Vec::with_capacity(group_sizes.len());
    let mut offset = 0usize;
    for &group_size in group_sizes {
        groups.push(&values[offset..offset + group_size]);
        offset += group_size;
    }
    assert_eq!(offset, values.len());
    groups
}

fn commit_groups_by_point<const D: usize, Cfg: CommitmentConfig, P: HachiPolyOps<F, D>>(
    poly_groups_by_point: &[&[&[P]]],
    setup: &<HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::ProverSetup,
) -> Result<(PointCommitments<D, Cfg>, PointHints<D, Cfg>), HachiError>
where
    HachiCommitmentScheme<D, Cfg>: CommitmentScheme<F, D>,
{
    let mut commitments_by_point = Vec::with_capacity(poly_groups_by_point.len());
    let mut hints_by_point = Vec::with_capacity(poly_groups_by_point.len());
    for point_groups in poly_groups_by_point {
        let (commitments, hints): (Vec<_>, Vec<_>) = point_groups
            .iter()
            .map(|group| {
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(group, setup)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        commitments_by_point.push(commitments);
        hints_by_point.push(hints);
    }
    Ok((commitments_by_point, hints_by_point))
}

#[test]
fn multipoint_dense_round_trip_with_mixed_groups() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 10;
        let point_group_sizes = [vec![2], vec![1, 1], vec![2, 1]];
        let total_claims: usize = point_group_sizes.iter().flatten().sum();
        let layout = hachi_batched_root_layout::<DenseCfg, DENSE_D>(NV, total_claims).unwrap();

        let point_polys: Vec<Vec<DensePoly<F, DENSE_D>>> = point_group_sizes
            .iter()
            .enumerate()
            .map(|(point_idx, groups)| {
                (0..groups.iter().sum())
                    .map(|poly_idx| {
                        make_dense_poly(
                            NV,
                            0xd3e5_1000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let opening_points_owned: Vec<Vec<F>> = (0..point_group_sizes.len())
            .map(|point_idx| random_point(NV, 0xaaaa_1000 + point_idx as u64))
            .collect();
        let openings_by_point: Vec<Vec<F>> = point_polys
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(polys, point)| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly(poly, point, &layout))
                    .collect()
            })
            .collect();

        let poly_group_storage: Vec<Vec<&[DensePoly<F, DENSE_D>]>> = point_polys
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(polys, groups)| build_group_slices(polys, groups))
            .collect();
        let poly_groups_by_point: Vec<&[&[DensePoly<F, DENSE_D>]]> =
            poly_group_storage.iter().map(Vec::as_slice).collect();
        let opening_group_storage: Vec<Vec<&[F]>> = openings_by_point
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(openings, groups)| build_group_slices(openings, groups))
            .collect();
        let opening_groups_by_point: Vec<&[&[F]]> =
            opening_group_storage.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::setup_prover(
                NV,
                total_claims,
            );
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let (commitments_by_point, hints_by_point) =
            commit_groups_by_point::<DENSE_D, DenseCfg, _>(&poly_groups_by_point, &setup)
                .expect("multipoint grouped commit");
        for (point_idx, point_groups) in poly_groups_by_point.iter().enumerate() {
            let expected_commitments: Vec<_> = point_groups
                .iter()
                .map(|group| {
                    <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::commit(
                        group,
                        &setup,
                    )
                    .map(|(commitment, _)| commitment)
                })
                .collect::<Result<_, _>>()
                .expect("per-point grouped commit");
            assert_eq!(expected_commitments, commitments_by_point[point_idx]);
        }
        let commitment_slices: Vec<&[_]> = commitments_by_point.iter().map(Vec::as_slice).collect();

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let proof =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_prove(
                &setup,
                &poly_groups_by_point,
                &opening_points,
                hints_by_point,
                &mut prover_transcript,
                &commitment_slices,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = HachiBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense");
        let result = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_points,
            &opening_groups_by_point,
            &commitment_slices,
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "dense multipoint verification failed: {:?}",
            result.err()
        );
    });
}

#[test]
fn multipoint_onehot_round_trip_with_mixed_groups() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        let point_group_sizes = [vec![2], vec![1, 1], vec![2, 1]];
        let total_claims: usize = point_group_sizes.iter().flatten().sum();
        let layout = hachi_batched_root_layout::<OneHotCfg, ONEHOT_D>(NV, total_claims).unwrap();

        let point_polys: Vec<Vec<OneHotPoly<F, ONEHOT_D, u8>>> = point_group_sizes
            .iter()
            .enumerate()
            .map(|(point_idx, groups)| {
                (0..groups.iter().sum())
                    .map(|poly_idx| {
                        make_onehot_poly(
                            &layout,
                            0xa66e_2000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let opening_points_owned: Vec<Vec<F>> = (0..point_group_sizes.len())
            .map(|point_idx| random_point(NV, 0xf00d_2000 + point_idx as u64))
            .collect();
        let openings_by_point: Vec<Vec<F>> = point_polys
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(polys, point)| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly(poly, point, &layout))
                    .collect()
            })
            .collect();

        let poly_group_storage: Vec<Vec<&[OneHotPoly<F, ONEHOT_D, u8>]>> = point_polys
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(polys, groups)| build_group_slices(polys, groups))
            .collect();
        let poly_groups_by_point: Vec<&[&[OneHotPoly<F, ONEHOT_D, u8>]]> =
            poly_group_storage.iter().map(Vec::as_slice).collect();
        let opening_group_storage: Vec<Vec<&[F]>> = openings_by_point
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(openings, groups)| build_group_slices(openings, groups))
            .collect();
        let opening_groups_by_point: Vec<&[&[F]]> =
            opening_group_storage.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::setup_prover(
                NV,
                total_claims,
            );
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let (commitments_by_point, hints_by_point) =
            commit_groups_by_point::<ONEHOT_D, OneHotCfg, _>(&poly_groups_by_point, &setup)
                .expect("multipoint grouped commit");
        for (point_idx, point_groups) in poly_groups_by_point.iter().enumerate() {
            let expected_commitments: Vec<_> = point_groups
                .iter()
                .map(|group| {
                    <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
                            F,
                            ONEHOT_D,
                        >>::commit(group, &setup)
                        .map(|(commitment, _)| commitment)
                })
                .collect::<Result<_, _>>()
                .expect("per-point grouped commit");
            assert_eq!(expected_commitments, commitments_by_point[point_idx]);
        }
        let commitment_slices: Vec<&[_]> = commitments_by_point.iter().map(Vec::as_slice).collect();

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot");
        let proof =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::batched_prove(
                &setup,
                &poly_groups_by_point,
                &opening_points,
                hints_by_point,
                &mut prover_transcript,
                &commitment_slices,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = HachiBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot");
        let result = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_points,
            &opening_groups_by_point,
            &commitment_slices,
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "onehot multipoint verification failed: {:?}",
            result.err()
        );
    });
}

#[test]
fn multipoint_dense_verify_rejects_swapped_points() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 10;
        let point_group_sizes = [vec![2], vec![2]];
        let total_claims = 4usize;
        let layout = hachi_batched_root_layout::<DenseCfg, DENSE_D>(NV, total_claims).unwrap();

        let point_polys: Vec<Vec<DensePoly<F, DENSE_D>>> = point_group_sizes
            .iter()
            .enumerate()
            .map(|(point_idx, groups)| {
                (0..groups.iter().sum())
                    .map(|poly_idx| {
                        make_dense_poly(
                            NV,
                            0xd3e5_3000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let opening_points_owned: Vec<Vec<F>> =
            vec![random_point(NV, 0xaaaa_3000), random_point(NV, 0xaaaa_3001)];
        let openings_by_point: Vec<Vec<F>> = point_polys
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(polys, point)| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly(poly, point, &layout))
                    .collect()
            })
            .collect();

        let poly_group_storage: Vec<Vec<&[DensePoly<F, DENSE_D>]>> = point_polys
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(polys, groups)| build_group_slices(polys, groups))
            .collect();
        let poly_groups_by_point: Vec<&[&[DensePoly<F, DENSE_D>]]> =
            poly_group_storage.iter().map(Vec::as_slice).collect();
        let opening_group_storage: Vec<Vec<&[F]>> = openings_by_point
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(openings, groups)| build_group_slices(openings, groups))
            .collect();
        let opening_groups_by_point: Vec<&[&[F]]> =
            opening_group_storage.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::setup_prover(
                NV,
                total_claims,
            );
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let (commitments_by_point, hints_by_point) =
            commit_groups_by_point::<DENSE_D, DenseCfg, _>(&poly_groups_by_point, &setup)
                .expect("multipoint grouped commit");
        let commitment_slices: Vec<&[_]> = commitments_by_point.iter().map(Vec::as_slice).collect();

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
        let proof =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_prove(
                &setup,
                &poly_groups_by_point,
                &opening_points,
                hints_by_point,
                &mut prover_transcript,
                &commitment_slices,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let swapped_points = vec![opening_points[1], opening_points[0]];
        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/dense_wrong_point");
        let result = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &swapped_points,
            &opening_groups_by_point,
            &commitment_slices,
            BasisMode::Lagrange,
        );
        assert!(result.is_err(), "swapped opening points must be rejected");
    });
}

#[test]
fn multipoint_onehot_verify_rejects_wrong_group_nesting() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        let point_group_sizes = [vec![1, 2], vec![1, 1]];
        let total_claims: usize = point_group_sizes.iter().flatten().sum();
        let layout = hachi_batched_root_layout::<OneHotCfg, ONEHOT_D>(NV, total_claims).unwrap();

        let point_polys: Vec<Vec<OneHotPoly<F, ONEHOT_D, u8>>> = point_group_sizes
            .iter()
            .enumerate()
            .map(|(point_idx, groups)| {
                (0..groups.iter().sum())
                    .map(|poly_idx| {
                        make_onehot_poly(
                            &layout,
                            0xa66e_4000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let opening_points_owned: Vec<Vec<F>> =
            vec![random_point(NV, 0xf00d_4000), random_point(NV, 0xf00d_4001)];
        let openings_by_point: Vec<Vec<F>> = point_polys
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(polys, point)| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly(poly, point, &layout))
                    .collect()
            })
            .collect();

        let poly_group_storage: Vec<Vec<&[OneHotPoly<F, ONEHOT_D, u8>]>> = point_polys
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(polys, groups)| build_group_slices(polys, groups))
            .collect();
        let poly_groups_by_point: Vec<&[&[OneHotPoly<F, ONEHOT_D, u8>]]> =
            poly_group_storage.iter().map(Vec::as_slice).collect();
        let opening_group_storage: Vec<Vec<&[F]>> = openings_by_point
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(openings, groups)| build_group_slices(openings, groups))
            .collect();
        let _opening_groups_by_point: Vec<&[&[F]]> =
            opening_group_storage.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let setup =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::setup_prover(
                NV,
                total_claims,
            );
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let (commitments_by_point, hints_by_point) =
            commit_groups_by_point::<ONEHOT_D, OneHotCfg, _>(&poly_groups_by_point, &setup)
                .expect("multipoint grouped commit");
        let commitment_slices: Vec<&[_]> = commitments_by_point.iter().map(Vec::as_slice).collect();

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot_wrong_grouping");
        let proof =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::batched_prove(
                &setup,
                &poly_groups_by_point,
                &opening_points,
                hints_by_point,
                &mut prover_transcript,
                &commitment_slices,
                BasisMode::Lagrange,
            )
            .expect("multipoint batched prove");

        let wrong_group_sizes = [vec![2, 1], vec![1, 1]];
        let wrong_opening_group_storage: Vec<Vec<&[F]>> = openings_by_point
            .iter()
            .zip(wrong_group_sizes.iter())
            .map(|(openings, groups)| build_group_slices(openings, groups))
            .collect();
        let wrong_opening_groups_by_point: Vec<&[&[F]]> = wrong_opening_group_storage
            .iter()
            .map(Vec::as_slice)
            .collect();

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/onehot_wrong_grouping");
        let result = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_points,
            &wrong_opening_groups_by_point,
            &commitment_slices,
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "wrong verifier-side group nesting must be rejected"
        );
    });
}

#[test]
fn multipoint_batched_prove_rejects_capacity_overflow() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 10;
        let point_group_sizes = [vec![4], vec![1]];
        let total_claims: usize = point_group_sizes.iter().flatten().sum();

        let point_polys: Vec<Vec<DensePoly<F, DENSE_D>>> = point_group_sizes
            .iter()
            .enumerate()
            .map(|(point_idx, groups)| {
                (0..groups.iter().sum())
                    .map(|poly_idx| {
                        make_dense_poly(
                            NV,
                            0xd3e5_5000 + (point_idx as u64) * 100 + poly_idx as u64,
                        )
                    })
                    .collect()
            })
            .collect();
        let poly_group_storage: Vec<Vec<&[DensePoly<F, DENSE_D>]>> = point_polys
            .iter()
            .zip(point_group_sizes.iter())
            .map(|(polys, groups)| build_group_slices(polys, groups))
            .collect();
        let poly_groups_by_point: Vec<&[&[DensePoly<F, DENSE_D>]]> =
            poly_group_storage.iter().map(Vec::as_slice).collect();
        let opening_points_owned: Vec<Vec<F>> = (0..point_group_sizes.len())
            .map(|point_idx| random_point(NV, 0xaaaa_5000 + point_idx as u64))
            .collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

        let commit_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_prover(NV, total_claims);
        let prove_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_prover(NV, total_claims - 1);
        let (commitments_by_point, hints_by_point) =
            commit_groups_by_point::<DENSE_D, DenseCfg, _>(&poly_groups_by_point, &commit_setup)
                .expect("per-group commit should fit with matching setup");
        let commitment_slices: Vec<&[_]> = commitments_by_point.iter().map(Vec::as_slice).collect();
        let mut transcript =
            Blake2bTranscript::<F>::new(b"multipoint_batched_e2e/capacity-overflow");
        let result = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_prove(
            &prove_setup,
            &poly_groups_by_point,
            &opening_points,
            hints_by_point,
            &mut transcript,
            &commitment_slices,
            BasisMode::Lagrange,
        );
        assert!(result.is_err(), "capacity overflow must be rejected");
    });
}
