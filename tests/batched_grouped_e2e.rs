//! End-to-end tests for **batched grouped** (detached) commitments.
//!
//! Polynomials in a batch are split into multiple commitment groups.  Each
//! group aggregates its own polynomials into a separate commitment, so
//! `batched_commit` returns one commitment per group.  The test exercises
//! `batched_commit` → `batched_prove` → serialize/deserialize →
//! `batched_verify`, and additionally cross-checks that committing each group
//! individually produces the same commitment as the multi-group call.
//!
//! This file intentionally keeps only a few representative grouped cases.
//! The broader batch-size and variable-count matrix is already exercised by
//! the aggregated, multipoint, and core batched E2E suites; the unique value
//! here is checking that multi-group commitment partitioning matches the
//! per-group individual commits.
//!
//! Retained partitions:
//!   2 → \[1,1\],  3 → \[2,1\],  7 → \[3,2,2\].

#![allow(missing_docs)]

mod common;

use common::*;
use hachi_pcs::protocol::commitment::hachi_batched_root_layout;
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::proof::HachiBatchedProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::{CommitmentScheme, HachiDeserialize, HachiSerialize, Transcript};

/// Return the group-size partition for a given total batch size.
fn group_partition(batch_size: usize) -> Vec<usize> {
    match batch_size {
        1 => vec![1],
        2 => vec![1, 1],
        3 => vec![2, 1],
        4 => vec![2, 2],
        7 => vec![3, 2, 2],
        12 => vec![4, 4, 4],
        16 => vec![5, 5, 3, 3],
        other => panic!("no partition defined for batch_size={other}"),
    }
}

/// One-hot polynomials are split into multiple commitment groups according to
/// [`group_partition`].  Each group produces its own commitment.  The test
/// also verifies that committing each group individually yields the same
/// commitment as the multi-group `batched_commit` result.
fn run_grouped_onehot(nv: usize, batch_size: usize) {
    init_rayon_pool();
    let group_sizes = group_partition(batch_size);
    run_on_large_stack(move || {
        let layout =
            hachi_batched_root_layout::<OneHotCfg, ONEHOT_D>(nv, batch_size).expect("layout");

        let polys: Vec<OneHotPoly<F, ONEHOT_D, u8>> = (0..batch_size)
            .map(|idx| make_onehot_poly(&layout, 0xbeef_0000 + (nv as u64) * 100 + idx as u64))
            .collect();

        let pt = random_point(nv, 0xd00d_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_prover(nv, batch_size, 1);
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let mut poly_groups: Vec<&[OneHotPoly<F, ONEHOT_D, u8>]> = Vec::new();
        let mut opening_groups: Vec<&[F]> = Vec::new();
        let mut offset = 0usize;
        for &gs in &group_sizes {
            poly_groups.push(&polys[offset..offset + gs]);
            opening_groups.push(&openings[offset..offset + gs]);
            offset += gs;
        }
        assert_eq!(offset, batch_size);

        let (commitments, hints): (Vec<_>, Vec<_>) = poly_groups
            .iter()
            .map(|group| {
                <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::commit(
                    group,
                    &setup,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("grouped commit")
            .into_iter()
            .unzip();

        assert_eq!(
            commitments.len(),
            group_sizes.len(),
            "number of commitments must equal number of groups"
        );

        for (group_idx, &gs) in group_sizes.iter().enumerate() {
            let start = group_sizes[..group_idx].iter().sum::<usize>();
            let (individual_commit, _) =
                <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::commit(
                    &polys[start..start + gs],
                    &setup,
                )
                .expect("individual group commit");
            assert_eq!(
                individual_commit, commitments[group_idx],
                "group {group_idx} commitment mismatch between individual and multi-group commit"
            );
        }

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"batched_grouped_e2e/onehot");
        let proof =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::batched_prove(
                &setup,
                &[&poly_groups[..]],
                &[&pt[..]],
                vec![hints],
                &mut prover_transcript,
                &[&commitments[..]],
                BasisMode::Lagrange,
            )
            .expect("batched prove");

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

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"batched_grouped_e2e/onehot");
        let result = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::batched_verify(
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
            "grouped onehot nv={nv} batch={batch_size} groups={group_sizes:?} \
             verification failed: {:?}",
            result.err()
        );
    });
}

/// Dense polynomials are split into multiple commitment groups according to
/// [`group_partition`].  Same cross-check as the one-hot variant.
fn run_grouped_dense(nv: usize, batch_size: usize) {
    init_rayon_pool();
    let group_sizes = group_partition(batch_size);
    run_on_large_stack(move || {
        let layout =
            hachi_batched_root_layout::<DenseCfg, DENSE_D>(nv, batch_size).expect("layout");

        let polys: Vec<DensePoly<F, DENSE_D>> = (0..batch_size)
            .map(|idx| make_dense_poly(nv, 0xd6e5_0000 + (nv as u64) * 100 + idx as u64))
            .collect();

        let pt = random_point(nv, 0xbbbb_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_prover(nv, batch_size, 1);
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let mut poly_groups: Vec<&[DensePoly<F, DENSE_D>]> = Vec::new();
        let mut opening_groups: Vec<&[F]> = Vec::new();
        let mut offset = 0usize;
        for &gs in &group_sizes {
            poly_groups.push(&polys[offset..offset + gs]);
            opening_groups.push(&openings[offset..offset + gs]);
            offset += gs;
        }
        assert_eq!(offset, batch_size);

        let (commitments, hints): (Vec<_>, Vec<_>) = poly_groups
            .iter()
            .map(|group| {
                <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::commit(
                    group, &setup,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("grouped commit")
            .into_iter()
            .unzip();

        assert_eq!(
            commitments.len(),
            group_sizes.len(),
            "number of commitments must equal number of groups"
        );

        for (group_idx, &gs) in group_sizes.iter().enumerate() {
            let start = group_sizes[..group_idx].iter().sum::<usize>();
            let (individual_commit, _) =
                <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::commit(
                    &polys[start..start + gs],
                    &setup,
                )
                .expect("individual group commit");
            assert_eq!(
                individual_commit, commitments[group_idx],
                "group {group_idx} commitment mismatch between individual and multi-group commit"
            );
        }

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"batched_grouped_e2e/dense");
        let proof =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_prove(
                &setup,
                &[&poly_groups[..]],
                &[&pt[..]],
                vec![hints],
                &mut prover_transcript,
                &[&commitments[..]],
                BasisMode::Lagrange,
            )
            .expect("batched prove");

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

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"batched_grouped_e2e/dense");
        let result =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_verify(
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
            "grouped dense nv={nv} batch={batch_size} groups={group_sizes:?} \
             verification failed: {:?}",
            result.err()
        );
    });
}

macro_rules! grouped_onehot_case {
    ($name:ident, $nv:expr, $batch_size:expr) => {
        #[test]
        fn $name() {
            run_grouped_onehot($nv, $batch_size);
        }
    };
}

macro_rules! grouped_dense_case {
    ($name:ident, $nv:expr, $batch_size:expr) => {
        #[test]
        fn $name() {
            run_grouped_dense($nv, $batch_size);
        }
    };
}

grouped_onehot_case!(grouped_onehot_nv10_batch2, 10, 2);
grouped_onehot_case!(grouped_onehot_nv20_batch7, 20, 7);
grouped_onehot_case!(grouped_onehot_nv25_batch3, 25, 3);

grouped_dense_case!(grouped_dense_nv10_batch2, 10, 2);
grouped_dense_case!(grouped_dense_nv15_batch3, 15, 3);
