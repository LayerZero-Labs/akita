#![allow(missing_docs)]

mod common;

use common::{
    init_rayon_pool, opening_from_poly, random_point, run_on_large_stack, BasisMode,
    CommitmentConfig, OneHotPoly, Rng, SeedableRng, StdRng, F,
};
use hachi_pcs::protocol::commitment::{hachi_batched_root_layout, presets::fp128};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::proof::HachiBatchedProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::{CommitmentScheme, HachiDeserialize, HachiSerialize, Transcript};
use std::sync::Mutex;

const ONEHOT_K: usize = 256;
const TEST_NV: usize = 15;
const BATCH_SIZE: usize = 3;

static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

fn fixed_random_point(nv: usize) -> Vec<F> {
    random_point(nv, 0xface_feed)
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

        let pt = fixed_random_point(TEST_NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(
            TEST_NV, BATCH_SIZE, 1,
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

        let pt = fixed_random_point(TEST_NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(
            TEST_NV, BATCH_SIZE, 1,
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
