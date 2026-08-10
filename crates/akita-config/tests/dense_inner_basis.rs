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
    outer_slices: usize,
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
        outer_slices: root.outer_slice_count.get(),
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
    let fp32 = snapshot::<fp32::Dense>();
    let fp64 = snapshot::<fp64::Dense>();
    let fp128 = snapshot::<fp128::Dense>();
    assert_ne!(fp32.inner_basis, fp32.opening_basis);
    assert_eq!(
        fp32,
        Snapshot {
            inner_basis: 5,
            opening_basis: 3,
            positions: 256,
            blocks: 512,
            outer_slices: 4,
            inner_digits: 7,
            n_a: 3,
            n_b: 1,
            n_d: 1,
            a_input_raw: 917_504,
            a_output_raw: 1_536,
            b_input_raw: 2_162_688,
            b_output_raw: 256,
            d_input_raw: 2_883_584,
            d_output_raw: 256,
            next_witness: 16_205_824,
        }
    );

    assert_ne!(fp64.inner_basis, fp64.opening_basis);
    assert_eq!(
        fp64,
        Snapshot {
            inner_basis: 10,
            opening_basis: 3,
            positions: 512,
            blocks: 256,
            outer_slices: 2,
            inner_digits: 7,
            n_a: 2,
            n_b: 1,
            n_d: 1,
            a_input_raw: 1_835_008,
            a_output_raw: 1_024,
            b_input_raw: 2_883_584,
            b_output_raw: 256,
            d_input_raw: 2_883_584,
            d_output_raw: 256,
            next_witness: 19_767_040,
        }
    );

    assert_ne!(fp128.inner_basis, fp128.opening_basis);
    assert_eq!(
        fp128,
        Snapshot {
            inner_basis: 8,
            opening_basis: 3,
            positions: 512,
            blocks: 512,
            outer_slices: 2,
            inner_digits: 16,
            n_a: 2,
            n_b: 1,
            n_d: 1,
            a_input_raw: 2_097_152,
            a_output_raw: 512,
            b_input_raw: 5_636_096,
            b_output_raw: 64,
            d_input_raw: 5_636_096,
            d_output_raw: 64,
            next_witness: 29_563_264,
        }
    );
}
