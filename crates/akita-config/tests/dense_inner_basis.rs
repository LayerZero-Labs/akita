#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::CommitmentConfig;
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

fn catalog<Cfg: CommitmentConfig>() -> akita_config::TrustedScheduleCatalog {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = std::fs::read(path).expect("checked-in schedule artifact");
    akita_config::trusted_schedule_catalog_from_bytes::<Cfg>(&bytes).expect("trusted catalog")
}

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
    let catalog = catalog::<Cfg>();
    let schedule = catalog
        .resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(26),
        ))
        .expect("generated dense nv=26 schedule");
    let root = &schedule.schedule().root.params;
    Snapshot {
        inner_basis: root.inner().digits.log_basis,
        opening_basis: root.open().digits.log_basis,
        positions: root.blocks().positions_per_block,
        blocks: root.blocks().live_blocks,
        outer_slices: root.outer_slice_count().get(),
        inner_digits: root.inner().digits.num_digits,
        n_a: root.inner().matrix.output_rank(),
        n_b: root.outer().matrix.output_rank(),
        n_d: root.open().matrix.output_rank(),
        a_input_raw: root.inner().matrix.raw_input_dimension().unwrap(),
        a_output_raw: root.inner().matrix.raw_output_dimension().unwrap(),
        b_input_raw: root.outer().matrix.raw_input_dimension().unwrap(),
        b_output_raw: root.outer().matrix.raw_output_dimension().unwrap(),
        d_input_raw: root.open().matrix.raw_input_dimension().unwrap(),
        d_output_raw: root.open().matrix.raw_output_dimension().unwrap(),
        next_witness: schedule.schedule().root.output_witness_len,
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
            inner_basis: 8,
            opening_basis: 3,
            positions: 128,
            blocks: 256,
            outer_slices: 8,
            inner_digits: 4,
            n_a: 1,
            n_b: 1,
            n_d: 1,
            a_input_raw: 1_048_576,
            a_output_raw: 2_048,
            b_input_raw: 720_896,
            b_output_raw: 256,
            d_input_raw: 720_896,
            d_output_raw: 256,
            next_witness: 12_910_144,
        }
    );

    assert_ne!(fp64.inner_basis, fp64.opening_basis);
    assert_eq!(
        fp64,
        Snapshot {
            inner_basis: 7,
            opening_basis: 3,
            positions: 128,
            blocks: 512,
            outer_slices: 8,
            inner_digits: 10,
            n_a: 1,
            n_b: 1,
            n_d: 1,
            a_input_raw: 1_310_720,
            a_output_raw: 1_024,
            b_input_raw: 1_441_792,
            b_output_raw: 128,
            d_input_raw: 1_441_792,
            d_output_raw: 128,
            next_witness: 20_971_072,
        }
    );

    assert_ne!(fp128.inner_basis, fp128.opening_basis);
    assert_eq!(
        fp128,
        Snapshot {
            inner_basis: 16,
            opening_basis: 3,
            positions: 256,
            blocks: 256,
            outer_slices: 8,
            inner_digits: 8,
            n_a: 1,
            n_b: 1,
            n_d: 1,
            a_input_raw: 2_097_152,
            a_output_raw: 1_024,
            b_input_raw: 1_409_024,
            b_output_raw: 64,
            d_input_raw: 704_512,
            d_output_raw: 64,
            next_witness: 31_002_560,
        }
    );
}
