#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{hachi_batched_root_layout, presets::fp128};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BlockOrder,
};
use hachi_pcs::protocol::proof::HachiBatchedProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentScheme, HachiDeserialize, HachiSerialize, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::{Mutex, Once};

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;
const ONEHOT_K: usize = 256;
const TEST_NV: usize = 15;
const BATCH_SIZE: usize = 3;
const STACK_SIZE: usize = 256 * 1024 * 1024;

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

fn random_point(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xface_feed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

fn opening_from_poly<const D: usize, P: HachiPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &hachi_pcs::protocol::HachiCommitmentLayout,
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

#[test]
fn batched_onehot_round_trip_with_individual_commitments() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;

        let layout = hachi_batched_root_layout::<Cfg, D>(TEST_NV, BATCH_SIZE).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F, D>> = (0..BATCH_SIZE)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x700d_f00d_1234_0000 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F, D>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars).unwrap()
            })
            .collect();

        let pt = random_point(TEST_NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(
            TEST_NV, BATCH_SIZE,
        );
        let verifier_setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let mut commitments = Vec::with_capacity(BATCH_SIZE);
        let mut hints = Vec::with_capacity(BATCH_SIZE);
        for poly in &polys {
            let (commitment, hint) =
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                    std::slice::from_ref(poly),
                    &setup,
                )
                .expect("individual commit");
            commitments.push(commitment);
            hints.push(hint);
        }
        let poly_groups: Vec<&[OneHotPoly<F, D>]> =
            polys.iter().map(std::slice::from_ref).collect();
        let opening_groups: Vec<&[F]> = openings.iter().map(std::slice::from_ref).collect();
        assert_eq!(commitments.len(), BATCH_SIZE);
        assert_eq!(hints.len(), BATCH_SIZE);

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"hachi_e2e/batched-individual-commitments");
        let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_prove(
            &setup,
            &[&poly_groups[..]],
            &[&pt[..]],
            vec![hints],
            &mut prover_transcript,
            &[&commitments[..]],
            BasisMode::Lagrange,
        )
        .expect("batched prove with individual commitments");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize batched proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded = HachiBatchedProof::<F>::deserialize_compressed(&mut cursor, &proof_shape)
            .expect("deserialize batched proof");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"hachi_e2e/batched-individual-commitments");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &[&pt[..]],
            &[&opening_groups[..]],
            &[&commitments[..]],
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "batched verification with individual commitments must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn batched_onehot_round_trip_with_mixed_commitment_groups() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;

        let layout = hachi_batched_root_layout::<Cfg, D>(TEST_NV, BATCH_SIZE).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F, D>> = (0..BATCH_SIZE)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x6eed_f00d_1234_0000 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F, D>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars).unwrap()
            })
            .collect();

        let pt = random_point(TEST_NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(
            TEST_NV, BATCH_SIZE,
        );
        let verifier_setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let poly_groups = [&polys[..2], &polys[2..]];
        let opening_groups = [&openings[..2], &openings[2..]];

        let (group0_commitment, group0_hint) =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                poly_groups[0],
                &setup,
            )
            .expect("group 0 commit");
        let (group1_commitment, group1_hint) =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                poly_groups[1],
                &setup,
            )
            .expect("group 1 commit");
        let commitments = [group0_commitment, group1_commitment];
        let hints = vec![group0_hint, group1_hint];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"hachi_e2e/batched-mixed-commitment-groups");
        let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_prove(
            &setup,
            &[&poly_groups[..]],
            &[&pt[..]],
            vec![hints],
            &mut prover_transcript,
            &[&commitments[..]],
            BasisMode::Lagrange,
        )
        .expect("batched prove with mixed commitment groups");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize mixed batched proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded = HachiBatchedProof::<F>::deserialize_compressed(&mut cursor, &proof_shape)
            .expect("deserialize mixed batched proof");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"hachi_e2e/batched-mixed-commitment-groups");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &[&pt[..]],
            &[&opening_groups[..]],
            &[&commitments[..]],
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "batched verification with mixed commitment groups must pass: {:?}",
            result.err()
        );
    });
}
