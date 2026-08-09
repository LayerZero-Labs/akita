#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::CommitmentConfig;
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    inner_basis: u32,
    opening_basis: u32,
    positions: usize,
    blocks: usize,
    inner_digits: usize,
    n_a: usize,
    n_b: usize,
    n_d: usize,
    a_input_raw: usize,
    a_output_raw: usize,
    b_input_raw: usize,
    b_output_raw: usize,
    d_input_raw: usize,
    d_output_raw: usize,
    next_witness: usize,
}

fn snapshot<Cfg: CommitmentConfig>() -> Snapshot {
    let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(26),
    ))
    .expect("generated dense nv=26 schedule");
    let root = &schedule.root.params.final_group.commitment;
    Snapshot {
        inner_basis: root.log_basis_inner,
        opening_basis: root.log_basis_open,
        positions: root.num_positions_per_block,
        blocks: root.num_live_blocks,
        inner_digits: root.num_digits_inner,
        n_a: root.inner_commit_matrix.output_rank(),
        n_b: root.outer_commit_matrix.output_rank(),
        n_d: root.open_commit_matrix.output_rank(),
        a_input_raw: root.inner_commit_matrix.raw_input_dimension().unwrap(),
        a_output_raw: root.inner_commit_matrix.raw_output_dimension().unwrap(),
        b_input_raw: root.outer_commit_matrix.raw_input_dimension().unwrap(),
        b_output_raw: root.outer_commit_matrix.raw_output_dimension().unwrap(),
        d_input_raw: root.open_commit_matrix.raw_input_dimension().unwrap(),
        d_output_raw: root.open_commit_matrix.raw_output_dimension().unwrap(),
        next_witness: schedule.root.output_witness_len,
    }
}

#[test]
fn dense_nv26_proof_first_winners_keep_inner_basis_independent() {
    let fp32 = snapshot::<fp32::D128Dense>();
    assert_ne!(fp32.inner_basis, fp32.opening_basis);
    assert_eq!(
        fp32,
        Snapshot {
            inner_basis: 8,
            opening_basis: 3,
            positions: 2048,
            blocks: 256,
            inner_digits: 4,
            n_a: 13,
            n_b: 2,
            n_d: 2,
            a_input_raw: 1_048_576,
            a_output_raw: 1_664,
            b_input_raw: 4_685_824,
            b_output_raw: 256,
            d_input_raw: 360_448,
            d_output_raw: 256,
            next_witness: 11_385_728,
        }
    );

    let fp64 = snapshot::<fp64::D128Dense>();
    assert_ne!(fp64.inner_basis, fp64.opening_basis);
    assert_eq!(
        fp64,
        Snapshot {
            inner_basis: 6,
            opening_basis: 3,
            positions: 1024,
            blocks: 512,
            inner_digits: 11,
            n_a: 7,
            n_b: 1,
            n_d: 1,
            a_input_raw: 1_441_792,
            a_output_raw: 896,
            b_input_raw: 10_092_544,
            b_output_raw: 128,
            d_input_raw: 1_441_792,
            d_output_raw: 128,
            next_witness: 18_794_112,
        }
    );

    let fp128 = snapshot::<fp128::Dense>();
    assert_ne!(fp128.inner_basis, fp128.opening_basis);
    assert_eq!(
        fp128,
        Snapshot {
            inner_basis: 11,
            opening_basis: 3,
            positions: 2048,
            blocks: 512,
            inner_digits: 12,
            n_a: 9,
            n_b: 1,
            n_d: 1,
            a_input_raw: 1_572_864,
            a_output_raw: 576,
            b_input_raw: 12_681_216,
            b_output_raw: 64,
            d_input_raw: 1_409_024,
            d_output_raw: 64,
            next_witness: 25_155_904,
        }
    );
}
