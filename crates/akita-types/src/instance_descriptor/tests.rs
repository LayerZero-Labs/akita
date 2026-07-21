use super::*;
use crate::{
    CommittedGroupParams, FoldSchedule, OpeningClaimsLayout, RootFinalChallenge,
    RootFinalGroupParams, RootFoldParams, RootFoldStep, RootSource, TerminalCommittedGroupParams,
    TerminalFoldParams, TerminalFoldStep, TerminalResponseShape, WitnessPartition,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime32Offset99;

fn sample_schedule() -> FoldSchedule {
    let sparse = SparseChallengeConfig::pm1_only(3);
    let committed =
        CommittedGroupParams::params_only(SisModulusProfileId::Q32Offset99, 64, 3, 2, 3, 2, sparse)
            .with_decomp(4, 32, 2, 2, 2)
            .expect("sample committed params");
    let (terminal_witness, honest_cap) =
        TerminalCommittedGroupParams::try_from_expanded_group(committed.clone())
            .expect("terminal response bounds");
    let response_shape =
        TerminalResponseShape::derive(&terminal_witness, honest_cap).expect("terminal shape");
    FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    source: RootSource::Dense {
                        coefficient_bits: 32,
                    },
                    challenge: RootFinalChallenge::Flat,
                    commitment: committed.clone(),
                },
                precommitted_groups: Vec::new(),
                open_commit_matrix: committed.open_commit_matrix.clone(),
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

fn sample_descriptor() -> AkitaInstanceDescriptor {
    let opening_batch = OpeningClaimsLayout::new(5, 3).expect("valid opening batch");
    AkitaInstanceDescriptor::new(
        AlgebraSection::for_fields::<Prime32Offset99, Prime32Offset99, 64>().expect("algebra"),
        SetupSection {
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 32,
                log_open_bound: Some(32),
            },
            sis_modulus_profile: SisModulusProfileId::Q32Offset99,
            setup_seed_digest: [1; 32],
            protocol_features: ProtocolFeatureSet::current(),
            fold_linf: FoldLinfProtocolBinding::CURRENT,
        },
        PlanSection::from_schedule(&sample_schedule()),
        CallSection::from_layout(&opening_batch, BasisMode::Lagrange).expect("call"),
    )
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
fn descriptor_roundtrip_preserves_typed_schedule_binding() {
    let descriptor = sample_descriptor();
    let bytes = descriptor.canonical_bytes().expect("serialize descriptor");
    let decoded = AkitaInstanceDescriptor::deserialize_uncompressed(&bytes[..], &())
        .expect("deserialize descriptor");
    assert_eq!(decoded, descriptor);
}

#[test]
fn rejects_pre_topology_descriptor_epoch() {
    let mut descriptor = sample_descriptor();
    descriptor.version = AKITA_INSTANCE_DESCRIPTOR_VERSION - 1;
    assert!(matches!(
        descriptor.check(),
        Err(SerializationError::InvalidData(_))
    ));

    let bytes = descriptor.canonical_bytes().expect("serialize old epoch");
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
        PlanSection::from_schedule(&first),
        PlanSection::from_schedule(&second)
    );
}

#[test]
fn terminal_sparse_sampler_changes_plan_binding() {
    let first = sample_schedule();
    let mut second = first.clone();
    second.terminal.params.sparse_challenge_config = SparseChallengeConfig::pm1_only(4);
    assert_ne!(
        PlanSection::from_schedule(&first),
        PlanSection::from_schedule(&second)
    );
}
