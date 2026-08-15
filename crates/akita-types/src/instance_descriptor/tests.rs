use super::*;
use crate::{
    CommittedGroupParams, FoldSchedule, InnerCommitMatrixParams, OpeningClaimsLayout,
    OpeningScheduleSelection, RootFinalGroupParams, RootFoldParams, RootFoldStep,
    ScheduleRowDigest, TerminalCommittedGroupParams, TerminalFoldParams, TerminalFoldStep,
    TerminalResponseShape, WitnessPartition,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime32Offset99;

// `pm1_only(3)` prices the fixtures' response cap 127 below A bucket 4095.
const TEST_TERMINAL_A_BUCKET: u128 = 4_095;

fn sample_schedule() -> FoldSchedule {
    let sparse = SparseChallengeConfig::pm1_only(3);
    let mut committed =
        CommittedGroupParams::params_only(SisModulusProfileId::Q32Offset99, 64, 3, 4, 3, 2, sparse)
            .with_decomp(4, 32, 2, 2, 2)
            .expect("sample committed params");
    let inner = committed.inner_commit_matrix;
    committed.inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        TEST_TERMINAL_A_BUCKET,
        inner.ring_dimension(),
    );
    let (terminal_witness, admission_cap) =
        TerminalCommittedGroupParams::try_from_expanded_group(committed.clone())
            .expect("terminal response bounds");
    let response_shape =
        TerminalResponseShape::derive(&terminal_witness, admission_cap).expect("terminal shape");
    FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    commitment: committed.clone(),
                },
                precommitted_groups: Vec::new(),
                open_commit_matrix: committed.open_commit_matrix,
                sparse_challenge_config: sparse,
                witness_partition: WitnessPartition::Single,
            },
            input_witness_len: 256,
            output_witness_len: 256,
        },
        recursive_folds: Vec::new(),
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: terminal_witness,
                sparse_challenge_config: sparse,
                response_shape,
            },
            input_witness_len: 256,
        },
    }
}

fn sample_selection() -> OpeningScheduleSelection {
    OpeningScheduleSelection {
        row_digest: ScheduleRowDigest::from_bytes([0x22; 32]),
    }
}

fn sample_descriptor() -> AkitaInstanceDescriptor {
    let opening_batch = OpeningClaimsLayout::new(5, 3).expect("valid opening batch");
    AkitaInstanceDescriptor::new(
        AlgebraSection::for_fields::<Prime32Offset99, Prime32Offset99>().expect("algebra"),
        SetupSection {
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 32,
                log_open_bound: Some(32),
            },
            sis_modulus_profile: SisModulusProfileId::Q32Offset99,
            compression_policy: COMPRESSION_POLICY,
            setup_seed_digest: [1; 32],
            protocol_features: ProtocolFeatureSet::current(),
            fold_linf: FoldLinfProtocolBinding::CURRENT,
        },
        PlanSection::from_schedule(sample_selection(), &sample_schedule()),
        CallSection::from_layout(&opening_batch, BasisMode::Lagrange).expect("call"),
    )
}

#[test]
fn schedule_selection_is_bound_into_the_v1_instance_descriptor() {
    let descriptor = sample_descriptor();
    let original = descriptor.canonical_bytes().expect("descriptor bytes");

    let mut changed_row = descriptor;
    changed_row.plan.schedule_selection.row_digest = ScheduleRowDigest::from_bytes([0x44; 32]);
    assert_ne!(
        original,
        changed_row
            .canonical_bytes()
            .expect("changed-row descriptor bytes")
    );
}

#[test]
fn rejects_removed_q16_sis_modulus_profile_tag() {
    let err = decode_sis_modulus_profile(std::io::Cursor::new([3u8]), Compress::No, Validate::Yes)
        .expect_err("historical Q16 tag 3 must be rejected");
    assert!(matches!(err, SerializationError::InvalidData(_)));
}

#[test]
fn setup_section_rejects_mismatched_zk_protocol_feature() {
    let mut descriptor = sample_descriptor();
    descriptor.setup.protocol_features.zk = true;
    assert!(matches!(
        descriptor.check(),
        Err(SerializationError::InvalidData(_))
    ));
}

#[test]
fn setup_section_rejects_unknown_compression_policy_tag() {
    let setup = sample_descriptor().setup;
    let mut bytes = Vec::new();
    setup
        .serialize_uncompressed(&mut bytes)
        .expect("serialize setup section");
    let policy_offset = decomposition_size(&setup.decomposition, Compress::No)
        + sis_modulus_profile_size(Compress::No);
    bytes[policy_offset] = u8::MAX;
    assert!(matches!(
        SetupSection::deserialize_uncompressed(&bytes[..], &()),
        Err(SerializationError::InvalidData(_))
    ));
}

#[test]
fn descriptor_roundtrip_preserves_typed_schedule_binding() {
    let descriptor = sample_descriptor();
    let bytes = descriptor.canonical_bytes().expect("serialize descriptor");
    let decoded = AkitaInstanceDescriptor::deserialize_uncompressed_exact(&bytes, &())
        .expect("deserialize descriptor");
    assert_eq!(decoded, descriptor);

    for suffix in [0, 0xa5] {
        let mut suffixed = bytes.clone();
        suffixed.push(suffix);
        assert!(AkitaInstanceDescriptor::deserialize_uncompressed_exact(&suffixed, &()).is_err());
    }
}

#[test]
fn call_section_rejects_oversized_group_count_before_allocation() {
    let mut bytes = Vec::new();
    u32::MAX
        .serialize_uncompressed(&mut bytes)
        .expect("serialize oversized count");

    assert!(matches!(
        CallSection::deserialize_uncompressed(&bytes[..], &()),
        Err(SerializationError::InvalidData(_))
    ));
}

#[test]
fn call_section_rejects_mismatched_arity_count_before_allocation() {
    let mut bytes = Vec::new();
    1u32.serialize_uncompressed(&mut bytes)
        .expect("serialize group count");
    u32::MAX
        .serialize_uncompressed(&mut bytes)
        .expect("serialize oversized arity count");

    assert!(matches!(
        CallSection::deserialize_uncompressed(&bytes[..], &()),
        Err(SerializationError::InvalidData(_))
    ));
}

#[test]
fn call_section_rejects_mismatched_polynomial_count_before_allocation() {
    let mut bytes = Vec::new();
    for value in [1u32, 1, 5, u32::MAX] {
        value
            .serialize_uncompressed(&mut bytes)
            .expect("serialize call section prefix");
    }

    assert!(matches!(
        CallSection::deserialize_uncompressed(&bytes[..], &()),
        Err(SerializationError::InvalidData(_))
    ));
}

#[test]
fn rejects_non_v1_descriptor_version() {
    let mut descriptor = sample_descriptor();
    descriptor.version = AKITA_INSTANCE_DESCRIPTOR_VERSION - 1;
    assert!(matches!(
        descriptor.check(),
        Err(SerializationError::InvalidData(_))
    ));

    let bytes = descriptor
        .canonical_bytes()
        .expect("serialize unsupported version");
    assert!(matches!(
        AkitaInstanceDescriptor::deserialize_uncompressed(&bytes[..], &()),
        Err(SerializationError::InvalidData(_))
    ));
}

#[test]
fn terminal_topology_changes_plan_binding() {
    let first = sample_schedule();
    let mut second = first.clone();
    second.terminal.input_witness_len += 1;
    assert_ne!(
        PlanSection::from_schedule(sample_selection(), &first),
        PlanSection::from_schedule(sample_selection(), &second)
    );
}

#[test]
fn terminal_sparse_sampler_changes_plan_binding() {
    let first = sample_schedule();
    let mut second = first.clone();
    second.terminal.params.sparse_challenge_config = SparseChallengeConfig::pm1_only(4);
    assert_ne!(
        PlanSection::from_schedule(sample_selection(), &first),
        PlanSection::from_schedule(sample_selection(), &second)
    );
}

#[test]
fn role_local_ring_dimension_changes_plan_binding() {
    let first = sample_schedule();
    let mut second = first.clone();
    let matrix = &second
        .root
        .params
        .final_group
        .commitment
        .inner_commit_matrix;
    second
        .root
        .params
        .final_group
        .commitment
        .inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
        matrix.security_policy(),
        matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        matrix.sis_modulus_profile(),
        matrix.output_rank(),
        matrix.input_width(),
        matrix.coeff_linf_bound().expect("L infinity test matrix"),
        matrix.ring_dimension() * 2,
    );

    assert_ne!(
        PlanSection::from_schedule(sample_selection(), &first),
        PlanSection::from_schedule(sample_selection(), &second)
    );
}
