use super::*;
use crate::{
    CommittedGroupParams, FoldParams, FoldSchedule, GrindingPlan, GrindingRun, GrindingSite,
    InnerCommitMatrixParams, OpeningClaimsLayout, OpeningScheduleSelection, ScheduleRowDigest,
    TerminalFoldParams, TerminalResponseShape,
};
use akita_challenges::SparseChallengeConfig;
use jolt_field::Prime32Offset99;

// `pm1_only(3)` prices the fixtures' response cap 127 below A bucket 4095.
const TEST_TERMINAL_A_BUCKET: u128 = 4_095;

fn sample_schedule() -> FoldSchedule {
    let sparse = SparseChallengeConfig::pm1_only(3);
    let mut committed =
        CommittedGroupParams::params_only(SisModulusProfileId::Q32Offset99, 64, 3, 4, 3, 2, sparse)
            .with_decomp(4, 32, 2, 2, 2)
            .expect("sample committed params");
    let inner = committed.inner().matrix;
    committed.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
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
        TerminalFoldParams::try_from_expanded_group(committed.clone())
            .expect("terminal response bounds");
    let response_shape =
        TerminalResponseShape::derive(&terminal_witness, admission_cap).expect("terminal shape");
    FoldSchedule {
        root: FoldParams {
            params: committed.clone(),
            input_witness_len: 256,
            output_witness_len: 256,
        },
        recursive_folds: Vec::new(),
        terminal: TerminalFoldParams {
            fold_challenge_config: sparse,
            response_shape,
            input_witness_len: 256,
            ..terminal_witness
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
    let grinding_plan = GrindingPlan::new(
        vec![
            GrindingRun::proof_of_work(GrindingSite::EvaluationBatch { level: 0 }, 1, 128)
                .expect("sample grinding run"),
        ],
        128,
    )
    .expect("sample grinding plan");
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
        },
        PlanSection::from_schedule(sample_selection(), &sample_schedule()),
        TranscriptGrindingBinding::for_plan(&grinding_plan).expect("grinding binding"),
        CallSection::from_layout(&opening_batch, BasisMode::Lagrange).expect("call"),
    )
}

#[test]
fn schedule_selection_is_bound_into_the_current_instance_descriptor() {
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

/// A deserialized descriptor cannot carry a committed-source bound the digit
/// math is unable to represent.
///
/// `SetupSection::check` delegates to `DecompositionParams::validate`, which is
/// the verifier-reachable enforcement of `1 <= log_commit_bound <= field_bits`
/// and a usable `log_basis`. This is the descriptor-layer half of that contract:
/// `validate` is unit-tested in isolation, but only this pins that a hostile
/// wire value actually reaches it and comes back as `InvalidData` rather than a
/// panic or a silently accepted schedule.
#[test]
fn setup_section_rejects_a_committed_source_bound_the_digits_cannot_represent() {
    let valid = sample_descriptor();
    valid.check().expect("the sample descriptor is well formed");
    let field_bits = valid.setup.decomposition.field_bits();

    let degenerate = [
        // A zero bound denotes an empty coefficient range.
        DecompositionParams {
            log_commit_bound: 0,
            ..valid.setup.decomposition
        },
        // A bound above the declared field width cannot be centered into it.
        DecompositionParams {
            log_commit_bound: field_bits + 1,
            ..valid.setup.decomposition
        },
        // A zero basis has no digit alphabet at all.
        DecompositionParams {
            log_basis: 0,
            ..valid.setup.decomposition
        },
        // A basis at or beyond the `u128` shift width would overflow the digit
        // math rather than merely be useless.
        DecompositionParams {
            log_basis: 128,
            ..valid.setup.decomposition
        },
        // An open bound below the commit bound collapses the field width under
        // the source it is supposed to contain.
        DecompositionParams {
            log_open_bound: Some(1),
            ..valid.setup.decomposition
        },
    ];

    for decomposition in degenerate {
        let mut descriptor = valid.clone();
        descriptor.setup.decomposition = decomposition;
        assert!(
            matches!(descriptor.check(), Err(SerializationError::InvalidData(_))),
            "descriptor must reject {decomposition:?}"
        );

        // The same rejection must happen on the deserialization path, not only
        // for a descriptor assembled in memory.
        let mut bytes = Vec::new();
        descriptor
            .setup
            .serialize_uncompressed(&mut bytes)
            .expect("serialize setup section");
        assert!(
            matches!(
                SetupSection::deserialize_uncompressed(&bytes[..], &()),
                Err(SerializationError::InvalidData(_))
            ),
            "deserialization must reject {decomposition:?}"
        );
    }

    // A bounded source strictly inside the field width stays valid: the check
    // rejects unrepresentable bounds, not bounded sources.
    let base_decomposition = valid.setup.decomposition;
    let mut bounded = valid;
    bounded.setup.decomposition = DecompositionParams {
        log_commit_bound: field_bits - 1,
        log_open_bound: Some(field_bits),
        ..base_decomposition
    };
    bounded
        .check()
        .expect("an interior committed-source bound is legal");
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
fn grinding_binding_has_the_exact_dedicated_descriptor_position() {
    let descriptor = sample_descriptor();
    let bytes = descriptor.canonical_bytes().expect("serialize descriptor");
    let offset = descriptor.version.serialized_size(Compress::No)
        + descriptor.algebra.serialized_size(Compress::No)
        + descriptor.setup.serialized_size(Compress::No)
        + descriptor.plan.serialized_size(Compress::No);
    let mut grinding_bytes = Vec::new();
    descriptor
        .grinding
        .serialize_uncompressed(&mut grinding_bytes)
        .expect("serialize grinding binding");
    assert_eq!(descriptor.version, 4);
    assert_eq!(
        &bytes[offset..offset + grinding_bytes.len()],
        grinding_bytes
    );

    let mut changed = descriptor.clone();
    changed.grinding.plan_digest[0] ^= 1;
    assert_ne!(
        bytes,
        changed.canonical_bytes().expect("changed grinding binding")
    );
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
fn rejects_noncurrent_descriptor_version() {
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
    second.terminal.fold_challenge_config = SparseChallengeConfig::pm1_only(4);
    assert_ne!(
        PlanSection::from_schedule(sample_selection(), &first),
        PlanSection::from_schedule(sample_selection(), &second)
    );
}

#[test]
fn role_local_ring_dimension_changes_plan_binding() {
    let first = sample_schedule();
    let mut second = first.clone();
    let matrix = &second.root.params.inner().matrix;
    second.root.params.own_group_mut().profile.inner.matrix =
        InnerCommitMatrixParams::new_unchecked(
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

#[test]
fn ring_relation_mode_changes_plan_and_transcript_preamble_binding() {
    let first = sample_schedule();
    let mut second = first.clone();
    second.root.params.ring_relation_mode = crate::RingRelationMode::ReducedEvaluation;
    let first_plan = PlanSection::from_schedule(sample_selection(), &first);
    let second_plan = PlanSection::from_schedule(sample_selection(), &second);
    assert_ne!(first_plan, second_plan);

    let mut first_descriptor = sample_descriptor();
    first_descriptor.plan = first_plan;
    let mut second_descriptor = sample_descriptor();
    second_descriptor.plan = second_plan;
    assert_ne!(
        first_descriptor.canonical_bytes().unwrap(),
        second_descriptor.canonical_bytes().unwrap(),
        "the mode must be bound before ring-switch alpha is sampled"
    );
}
