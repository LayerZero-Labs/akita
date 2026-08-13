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
    let schedule = Cfg::select_schedule_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(26),
    ))
    .expect("generated dense nv=26 schedule");
    let root = &schedule.schedule().root.params.final_group.commitment;
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
        next_witness: schedule.schedule().root.output_witness_len,
    }
}

#[test]
fn dense_nv26_proof_first_winners_keep_inner_basis_independent() {
    let fp32 = snapshot::<fp32::Dense>();
    assert_ne!(fp32.inner_basis, fp32.opening_basis);
    assert_eq!(
        fp32,
        Snapshot {
            inner_basis: 8,
            opening_basis: 3,
            positions: 512,
            blocks: 128,
            inner_digits: 4,
            n_a: 2,
            n_b: 1,
            n_d: 1,
            a_input_raw: 2_097_152,
            a_output_raw: 2_048,
            b_input_raw: 2_883_584,
            b_output_raw: 256,
            d_input_raw: 1_441_792,
            d_output_raw: 256,
            next_witness: 16_970_496,
        }
    );

    let fp64 = snapshot::<fp64::Dense>();
    assert_ne!(fp64.inner_basis, fp64.opening_basis);
    assert_eq!(
        fp64,
        Snapshot {
            inner_basis: 10,
            opening_basis: 3,
            positions: 512,
            blocks: 256,
            inner_digits: 7,
            n_a: 2,
            n_b: 1,
            n_d: 1,
            a_input_raw: 1_835_008,
            a_output_raw: 1_024,
            b_input_raw: 5_767_168,
            b_output_raw: 256,
            d_input_raw: 2_883_584,
            d_output_raw: 256,
            next_witness: 19_745_024,
        }
    );

    let fp128 = snapshot::<fp128::Dense>();
    assert_ne!(fp128.inner_basis, fp128.opening_basis);
    assert_eq!(
        fp128,
        Snapshot {
            inner_basis: 9,
            opening_basis: 3,
            positions: 1024,
            blocks: 256,
            inner_digits: 15,
            n_a: 2,
            n_b: 1,
            n_d: 1,
            a_input_raw: 3_932_160,
            a_output_raw: 512,
            b_input_raw: 5_636_096,
            b_output_raw: 64,
            d_input_raw: 2_818_048,
            d_output_raw: 64,
            next_witness: 32_108_224,
        }
    );
}
