//! Public admission must price B against the certified opening alphabet.

use akita_config::{
    policy_of,
    proof_optimized::{fp128, fp32},
    CommitmentConfig, TrustedScheduleCatalog, ValidatedScheduleCatalog,
};
use akita_params::{
    sis::num_digits_open, BlockGeometry, CommittedGroupBatchProfile, DecompositionParams,
    FoldSchedule, GadgetDigits, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, PolynomialGroupLayout, ScheduleLookupKey, ScheduleSisBound,
    ScheduleSisRole, TerminalResponseShape,
};
use akita_schedules::{planner_support::planned_next_witness_len, ResolvedScheduleRow};

fn differing_basis_schedule<Cfg: CommitmentConfig>(
    num_vars: usize,
    b_bound: u128,
) -> (CommittedGroupBatchProfile, FoldSchedule) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = std::fs::read(path).expect("checked-in schedule artifact");
    let catalog = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(
        &bytes,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
    let row = catalog
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(14),
        ))
        .unwrap();
    let mut schedule = row.schedule().clone();
    assert!(schedule.recursive_folds.is_empty());
    assert!(!schedule.root.params.has_preceding_groups());
    let policy = policy_of::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let root = &mut schedule.root.params;
    assert_eq!(root.open().digits.log_basis, 3);

    // Keep the scanner at base 8 while decomposing the honest B input at base 4.
    let own = root.own_group_mut();
    own.profile.group = PolynomialGroupLayout::singleton(num_vars);
    own.profile.outer_slice_count = akita_params::CommitmentSliceCount::ONE;
    // Pin the rank-boundary geometry independently of the catalog optimum.
    // Both domains keep the existing base-eight opening alphabet.
    let (d_a, positions) = if Cfg::EXT_DEGREE == 1 {
        (128, 64)
    } else {
        (512, 16)
    };
    let mut inner_key = own.profile.inner.matrix.sis_table_key().unwrap();
    inner_key.ring_dimension = d_a as u32;
    own.profile.inner.matrix =
        InnerCommitMatrixParams::try_new_with_min_rank(inner_key, positions).unwrap();
    let live = (1usize << num_vars) / d_a;
    own.profile.blocks = BlockGeometry::new(live, positions, live.div_ceil(positions));
    own.profile.outer.digits = GadgetDigits::new(
        2,
        num_digits_open(DecompositionParams {
            log_basis: 2,
            ..policy.decomposition
        }),
    );
    let mut key = own.profile.outer.matrix.sis_table_key();
    key.coeff_linf_bound = b_bound;
    key.ring_dimension = if Cfg::EXT_DEGREE == 1 { 64 } else { 128 };
    let width = akita_params::CommitmentSliceGeometry::try_new(
        own.profile.outer_slice_count,
        own.profile.blocks.live_blocks,
        1,
        own.profile.inner.matrix.output_rank(),
        own.profile.outer.digits.num_digits,
        d_a,
        key.ring_dimension as usize,
    )
    .unwrap()
    .physical_input_width();
    own.profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(key, width).unwrap();

    let d_width = akita_params::opening_d_segment_width(
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
    schedule.root.input_witness_len = akita_params::root_input_witness_len(root);
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
    let terminal_d = if Cfg::EXT_DEGREE == 1 { 64 } else { 128 };
    let old_key = terminal.inner.matrix.sis_table_key().unwrap();
    let key = akita_params::sis::sis_table_key_for_linf_bound(
        old_key.policy,
        old_key.table_digest,
        old_key.modulus_profile,
        old_key.role,
        terminal_d as u32,
        old_key.coeff_linf_bound,
    )
    .unwrap();
    let positions = if Cfg::EXT_DEGREE == 1 { 256 } else { 128 };
    assert_eq!(terminal.inner.digits.num_digits, 1);
    terminal.inner.matrix = InnerCommitMatrixParams::try_new_with_min_rank(key, positions).unwrap();
    terminal.fold_challenge_config = Cfg::ring_challenge_config(terminal_d).unwrap();
    let live = terminal.input_witness_len.div_ceil(terminal_d);
    terminal.blocks = BlockGeometry::new(live, positions, live.div_ceil(positions));
    let cap = terminal.certified_response_linf_cap().unwrap();
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
    let policy = policy_of::<fp128::OneHot>(akita_params::ChunkedWitnessCfg::default_non_chunked());
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
        &policy_of::<fp32::OneHot>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
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
        &policy_of::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
        Cfg::ring_challenge_config,
    )
    .expect("differing bases remain valid when B covers the scanning alphabet");
    let catalog = TrustedScheduleCatalog::<Cfg>::new(
        raw,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
    let bytes = catalog.to_artifact_bytes().unwrap();
    let loaded = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(
        &bytes,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
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
    let error = TrustedScheduleCatalog::<fp128::OneHot>::from_artifact_bytes(
        underpriced.as_bytes(),
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("loading an artifact must recheck the certified alphabet");
    assert!(error
        .to_string()
        .contains("declared coefficient bound 3 is below required bound 7"));
}
