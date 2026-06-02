use super::*;

#[test]
fn recursive_w_commit_layout_rejects_unsupported_ring_dimension() {
    let params = LevelParams::params_only(
        akita_types::SisModulusFamily::Q128,
        42,
        3,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![1],
        },
    );
    let err = recursive_w_commit_layout_for_d::<Cfg>(42, &params, 64).unwrap_err();
    assert!(
        matches!(err, AkitaError::InvalidInput(message) if message.contains("unsupported ring dimension: 42"))
    );
}

#[test]
fn same_point_batched_root_preserves_opening_geometry() {
    for num_claims in [4usize, 6] {
        let incidence =
            akita_types::ClaimIncidenceSummary::same_point(20, num_claims).expect("incidence");
        let schedule = OneHotCfg::get_params_for_prove(&incidence).expect("same-point root plan");
        let Some(Step::Fold(root_step)) = schedule.steps.first() else {
            panic!("same-point schedule should start with a fold");
        };
        let root_inputs = AkitaScheduleInputs {
            num_vars: 20,
            level: 0,
            current_w_len: root_step.current_w_len,
        };
        let level_lp = &root_step.params;
        let root_lp = akita_types::root_level_params_for_layout_with_log_basis(
            OneHotCfg::sis_modulus_family(),
            OneHotCfg::D,
            OneHotCfg::decomposition(),
            OneHotCfg::stage1_challenge_config(OneHotCfg::D).unwrap(),
            OneHotCfg::ring_subfield_embedding_norm_bound(),
            OneHotCfg::onehot_chunk_size(),
            root_inputs,
            level_lp,
        )
        .unwrap();
        assert_eq!(root_lp.block_len, level_lp.block_len);
        assert_eq!(root_lp.num_blocks, level_lp.num_blocks);
        assert_eq!(root_lp.m_vars, level_lp.m_vars);
        assert_eq!(root_lp.r_vars, level_lp.r_vars);
    }
}

#[test]
fn batched_suffix_stop_guard_does_not_preempt_profitable_fold() {
    // These states came from the batched onehot nv=32 profile runs that
    // regressed after a generic shrink-ratio guard was briefly added to
    // the batched suffix. The runtime guard should not stop folding here.
    assert!(!should_stop_batched_folding(87_744, 140_672));
    assert!(!should_stop_batched_folding(129_216, 224_064));
}
