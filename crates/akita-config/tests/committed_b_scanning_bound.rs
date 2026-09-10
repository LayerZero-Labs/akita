//! Public admission must price B against the certified opening alphabet.

use akita_config::{
    policy_of,
    proof_optimized::{fp128, fp32},
    CommitmentConfig, TrustedScheduleCatalog, ValidatedScheduleCatalog,
};
use akita_schedules::{planner_support::planned_next_witness_len, ResolvedScheduleRow};
use akita_types::{
    sis::num_digits_open, AkitaScheduleLookupKey, BlockGeometry, CommittedGroupBatchProfile,
    DecompositionParams, FoldSchedule, GadgetDigits, OpenCommitMatrixParams,
    OuterCommitMatrixParams, PolynomialGroupLayout, ScheduleSisBound, ScheduleSisRole,
    TerminalResponseShape,
};

fn differing_basis_schedule<Cfg: CommitmentConfig>(
    num_vars: usize,
    b_bound: u128,
) -> (CommittedGroupBatchProfile, FoldSchedule) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = std::fs::read(path).expect("checked-in schedule artifact");
    let catalog = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(&bytes).unwrap();
    let row = catalog
        .resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(14),
        ))
        .unwrap();
    let mut schedule = row.schedule().clone();
    assert!(schedule.recursive_folds.is_empty());
    assert!(!schedule.root.params.has_preceding_groups());
    let policy = policy_of::<Cfg>();
    let root = &mut schedule.root.params;
    assert_eq!(root.open().digits.log_basis, 3);

    // Keep the scanner at base 8 while decomposing the honest B input at base 4.
    let own = root.own_group_mut();
    own.profile.group = PolynomialGroupLayout::singleton(num_vars);
    let live = (1usize << num_vars) / own.profile.inner.matrix.ring_dimension();
    let positions = own.profile.blocks.positions_per_block;
    own.profile.blocks = BlockGeometry::new(live, positions, live.div_ceil(positions));
    own.profile.outer.digits = GadgetDigits::new(
        2,
        num_digits_open(DecompositionParams {
            log_basis: 2,
            ..policy.decomposition
        }),
    );
    let width = own
        .profile
        .derive_slice_geometry()
        .unwrap()
        .physical_input_width();
    let mut key = own.profile.outer.matrix.sis_table_key();
    key.coeff_linf_bound = b_bound;
    own.profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(key, width).unwrap();

    let d_width = akita_types::opening_d_segment_width(
        root.opening_method(),
        policy.claim_ext_degree,
        root.d_a(),
        root.open_matrix.ring_dimension(),
        root.open().digits.num_digits,
        root.blocks().live_blocks,
        1,
    )
    .unwrap();
    root.open_matrix =
        OpenCommitMatrixParams::try_new_with_min_rank(root.open_matrix.sis_table_key(), d_width)
            .unwrap();
    schedule.root.input_witness_len = akita_types::root_input_witness_len(root);
    schedule.root.output_witness_len = planned_next_witness_len(
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
        root,
        1,
        root.witness_chunk.num_chunks,
    )
    .unwrap()
    .unwrap();

    let terminal = &mut schedule.terminal;
    terminal.input_witness_len = schedule.root.output_witness_len;
    let live = terminal.input_witness_len.div_ceil(terminal.d_a());
    let positions = terminal.blocks.positions_per_block;
    terminal.blocks = BlockGeometry::new(live, positions, live.div_ceil(positions));
    let cap = terminal.response_shape.layout.groups[0].z_linf_cap.unwrap();
    terminal.response_shape = TerminalResponseShape::derive(terminal, cap).unwrap();

    let profiles = CommittedGroupBatchProfile {
        final_group: root.own_group().profile,
        precommitteds: Vec::new(),
    };
    profiles
        .validate(policy.decomposition.field_bits())
        .unwrap();
    schedule
        .validate_nonterminal_opening_execution(policy.claim_ext_degree)
        .unwrap();
    (profiles, schedule)
}

#[test]
fn committed_b_rejects_the_honest_alphabet_bound() {
    let (profiles, schedule) = differing_basis_schedule::<fp128::OneHot>(14, 3);
    let policy = policy_of::<fp128::OneHot>();
    let error = ResolvedScheduleRow::try_new(profiles.clone(), schedule.clone(), &policy)
        .expect_err("base-8 scanning requires collision bound 7");
    assert!(error
        .to_string()
        .contains("declared coefficient bound 3 is below required bound 7"));
    let error = ValidatedScheduleCatalog::try_new(
        fp128::OneHot::schedule_family_name(),
        [(profiles, schedule)],
        &policy,
        fp128::OneHot::ring_challenge_config,
    )
    .expect_err("catalog admission must apply the same bound");
    assert!(error
        .to_string()
        .contains("declared coefficient bound 3 is below required bound 7"));
}

#[test]
fn committed_b_scanning_bound_changes_the_required_rank() {
    let (profiles, schedule) = differing_basis_schedule::<fp32::OneHot>(19, 3);
    let matrix = schedule.root.params.outer().matrix;
    assert_eq!(schedule.root.params.d_a(), 512);
    assert_eq!(matrix.ring_dimension(), 128);
    assert_eq!(matrix.input_width(), 4096);
    assert_eq!(matrix.output_rank(), 1);
    let mut scanner_key = matrix.sis_table_key();
    scanner_key.coeff_linf_bound = 7;
    let corrected = OuterCommitMatrixParams::try_new_with_min_rank(scanner_key, 4096).unwrap();
    assert_eq!(corrected.output_rank(), 2);

    let error = ValidatedScheduleCatalog::try_new(
        fp32::OneHot::schedule_family_name(),
        [(profiles, schedule)],
        &policy_of::<fp32::OneHot>(),
        fp32::OneHot::ring_challenge_config,
    )
    .expect_err("a rank-1 matrix priced for bound 3 must not certify bound 7");
    assert!(error
        .to_string()
        .contains("declared coefficient bound 3 is below required bound 7"));
}

fn round_trip_corrected_schedule<Cfg: CommitmentConfig>(num_vars: usize, rank: usize) -> Vec<u8> {
    let (profiles, schedule) = differing_basis_schedule::<Cfg>(num_vars, 7);
    let raw = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        [(profiles.clone(), schedule.clone())],
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("differing bases remain valid when B covers the scanning alphabet");
    let catalog = TrustedScheduleCatalog::<Cfg>::new(raw).unwrap();
    let bytes = catalog.to_artifact_bytes().unwrap();
    let loaded = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(&bytes).unwrap();
    let row = loaded.resolve_profiles(&profiles).unwrap();
    assert_eq!(row.schedule(), &schedule);
    let occurrences = row.schedule().sis_occurrences().unwrap();
    let b = occurrences
        .iter()
        .find(|entry| entry.role == ScheduleSisRole::Outer)
        .unwrap();
    assert_eq!(b.bound, ScheduleSisBound::Linf(7));
    assert_eq!(b.output_rank, rank);
    bytes
}

#[test]
fn committed_b_corrected_bound_survives_artifact_round_trip_and_reporting() {
    round_trip_corrected_schedule::<fp128::OneHot>(14, 1);
    round_trip_corrected_schedule::<fp32::OneHot>(19, 2);
}

#[test]
fn committed_b_artifact_decoder_reaudits_the_scanning_bound() {
    let bytes = round_trip_corrected_schedule::<fp128::OneHot>(14, 1);
    let artifact = String::from_utf8(bytes).unwrap();
    // In this single-row fixture the first bound 7 is the root B matrix.
    let underpriced = artifact.replacen("\"coeff_linf_bound\":7", "\"coeff_linf_bound\":3", 1);
    assert_ne!(artifact, underpriced);
    let error =
        TrustedScheduleCatalog::<fp128::OneHot>::from_artifact_bytes(underpriced.as_bytes())
            .expect_err("loading an artifact must recheck the certified alphabet");
    assert!(error
        .to_string()
        .contains("declared coefficient bound 3 is below required bound 7"));
}
