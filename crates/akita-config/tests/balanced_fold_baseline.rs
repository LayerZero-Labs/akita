//! Immutable pre-cutover baseline for every shipped balanced-digit catalog.
//!
//! Unlike the generator drift guard, these constants do not invoke the
//! current planner as an oracle. They fingerprint the accepted digit ranges,
//! expanded matrix geometry, exact terminal wire shape, and proof-byte estimate
//! of every committed row that was compared with the pre-cutover tables.

#![cfg(feature = "all-schedules")]
#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::{policy_of, CommitmentConfig};
use akita_schedules::{estimate_proof_bytes, schedule_from_entry, GeneratedScheduleTable};
use akita_types::sis::fold_witness_representable_linf_bounds;
use akita_types::{AkitaScheduleLookupKey, CommittedGroupParams, FoldSchedule};

struct Fingerprint(u64);

impl Fingerprint {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn usize(&mut self, value: usize) {
        self.bytes(&(value as u128).to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn u128(&mut self, value: u128) {
        self.bytes(&value.to_le_bytes());
    }
}

fn fingerprint_committed_group(fingerprint: &mut Fingerprint, params: &CommittedGroupParams) {
    for value in [
        params.log_basis_inner,
        params.log_basis_outer,
        params.log_basis_open,
    ] {
        fingerprint.u32(value);
    }
    for value in [
        params.num_live_ring_elements_per_claim,
        params.num_positions_per_block,
        params.num_live_blocks,
        params.num_digits_inner,
        params.num_digits_outer,
        params.num_digits_open,
        params.num_digits_fold,
    ] {
        fingerprint.usize(value);
    }
    let (negative_reach, positive_reach) =
        fold_witness_representable_linf_bounds(params.log_basis_open, params.num_digits_fold);
    fingerprint.u128(negative_reach);
    fingerprint.u128(positive_reach);
    for matrix in [
        (
            params.inner_commit_matrix.ring_dimension(),
            params.inner_commit_matrix.input_width(),
            params.inner_commit_matrix.output_rank(),
            params.inner_commit_matrix.coeff_linf_bound(),
        ),
        (
            params.outer_commit_matrix.ring_dimension(),
            params.outer_commit_matrix.input_width(),
            params.outer_commit_matrix.output_rank(),
            params.outer_commit_matrix.coeff_linf_bound(),
        ),
        (
            params.open_commit_matrix.ring_dimension(),
            params.open_commit_matrix.input_width(),
            params.open_commit_matrix.output_rank(),
            params.open_commit_matrix.coeff_linf_bound(),
        ),
    ] {
        fingerprint.usize(matrix.0);
        fingerprint.usize(matrix.1);
        fingerprint.usize(matrix.2);
        fingerprint.u128(matrix.3);
    }
}

fn fingerprint_schedule(fingerprint: &mut Fingerprint, schedule: &FoldSchedule) {
    fingerprint_committed_group(fingerprint, &schedule.root.params.final_group.commitment);
    fingerprint.usize(schedule.recursive_folds.len());
    for fold in &schedule.recursive_folds {
        fingerprint_committed_group(fingerprint, &fold.params.witness);
    }

    let terminal = &schedule.terminal.params;
    fingerprint.u32(terminal.witness.log_basis_inner);
    fingerprint.usize(terminal.witness.num_live_ring_elements_per_claim);
    fingerprint.usize(terminal.witness.num_positions_per_block);
    fingerprint.usize(terminal.witness.num_live_blocks);
    fingerprint.usize(terminal.witness.num_digits_inner);
    fingerprint.usize(terminal.witness.inner_commit_matrix.ring_dimension());
    fingerprint.usize(terminal.witness.inner_commit_matrix.input_width());
    fingerprint.usize(terminal.witness.inner_commit_matrix.output_rank());
    fingerprint.u128(terminal.witness.inner_commit_matrix.coeff_linf_bound());
    fingerprint.usize(terminal.response_shape.layout.ring_dimension);
    fingerprint.usize(terminal.response_shape.layout.logical_num_elems);
    fingerprint.usize(terminal.response_shape.layout.groups.len());
    for group in &terminal.response_shape.layout.groups {
        fingerprint.usize(group.z_coords);
        fingerprint.usize(group.e_field_elems);
        fingerprint.usize(group.t_field_elems);
        fingerprint.u128(group.z_admission_linf_cap);
        fingerprint.u32(group.z_rice_low_bits);
        fingerprint.usize(group.z_payload_bytes);
    }
}

fn catalog_baseline<Cfg: CommitmentConfig>(
    catalog: GeneratedScheduleTable,
) -> (usize, u64, usize, usize) {
    let policy = policy_of::<Cfg>();
    let mut fingerprint = Fingerprint::new();
    let mut first_proof_bytes = None;
    let mut last_proof_bytes = 0;
    for entry in catalog.entries {
        let key: AkitaScheduleLookupKey = entry.to_runtime_lookup_key();
        fingerprint.bytes(&key.canonical_descriptor_bytes());
        let schedule = schedule_from_entry(
            entry,
            &key,
            &policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        )
        .expect("frozen balanced row must expand");
        fingerprint_schedule(&mut fingerprint, &schedule);
        let proof_bytes = estimate_proof_bytes(
            entry,
            &key,
            &policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        )
        .expect("frozen balanced proof size");
        fingerprint.usize(proof_bytes);
        first_proof_bytes.get_or_insert(proof_bytes);
        last_proof_bytes = proof_bytes;
    }
    (
        catalog.entries.len(),
        fingerprint.0,
        first_proof_bytes.expect("balanced catalog must be nonempty"),
        last_proof_bytes,
    )
}

#[test]
fn every_shipped_balanced_family_matches_the_pre_cutover_baseline() {
    let actual = [
        (
            "fp128_d128_dense",
            catalog_baseline::<fp128::D128Dense>(akita_schedules::fp128_d128_dense_table()),
        ),
        (
            "fp128_d64_dense",
            catalog_baseline::<fp128::D64Dense>(akita_schedules::fp128_d64_dense_table()),
        ),
        (
            "fp128_d64_dense_multi_chunk",
            catalog_baseline::<fp128::D64DenseMultiChunk>(
                akita_schedules::fp128_d64_dense_multi_chunk_table(),
            ),
        ),
        (
            "fp64_d128_dense",
            catalog_baseline::<fp64::D128Dense>(akita_schedules::fp64_d128_dense_table()),
        ),
    ];
    let expected = [
        (
            "fp128_d128_dense",
            (110, 6_508_899_863_024_081_128, 96_228, 131_916),
        ),
        (
            "fp128_d64_dense",
            (111, 3_869_292_565_239_947_133, 82_552, 111_696),
        ),
        (
            "fp128_d64_dense_multi_chunk",
            (102, 2_538_829_786_020_842_209, 95_080, 114_044),
        ),
        (
            "fp64_d128_dense",
            (54, 6_958_799_319_525_602_414, 79_608, 100_676),
        ),
    ];
    assert_eq!(actual, expected);
}
