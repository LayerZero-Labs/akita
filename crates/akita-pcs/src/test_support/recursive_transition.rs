use super::*;

/// Build the recursive multi-group transition used to exercise mixed ring
/// dimensions with setup-contribution offloading:
///
/// - L0 final and precommitted groups use A=`Root::D`, B=`root_bd_ring_dim`,
///   with one shared D=`root_bd_ring_dim`;
/// - L1 uses A=`Mid::D`, B=D=`middle_bd_ring_dim` and consumes the setup prefix
///   emitted by L0;
/// - L2+ use uniform `Suffix::D`.
///
/// The setup prefix uses the suffix configuration's independent A dimension.
/// This is D64 for the requested `256/128/128 -> 128/64/64 -> 64` profile, so
/// producer Stage 3 D128, prefix A D64, and consumer A D128 remain distinct.
#[allow(clippy::too_many_arguments)]
pub fn recursive_ring_dimension_transition_schedule<Root, Mid, Suffix, ChunkCfg>(
    key: &AkitaScheduleLookupKey,
    root_bd_ring_dim: usize,
    middle_bd_ring_dim: usize,
) -> Result<FoldSchedule, AkitaError>
where
    Root: CommitmentConfig,
    Mid: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
    Suffix: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
    ChunkCfg: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
{
    cached_synthetic_schedule(
        SyntheticScheduleCacheKey {
            kind: SyntheticScheduleKind::RecursiveRingDimensionTransition,
            root: TypeId::of::<Root>(),
            middle: TypeId::of::<Mid>(),
            suffix: TypeId::of::<Suffix>(),
            num_vars: key.final_group.num_vars(),
            num_polynomials: key.num_polynomials()?,
            parameters: [
                root_bd_ring_dim,
                middle_bd_ring_dim,
                ChunkCfg::chunked_witness_cfg().num_chunks,
                ChunkCfg::chunked_witness_cfg().num_activated_levels,
            ],
            lookup_key: Some(key.clone()),
        },
        || {
            key.validate(Root::decomposition().field_bits())?;
            if key.precommitteds.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "recursive ring-dimension transition requires precommitted groups".into(),
                ));
            }
            let root_dims = CommitmentRingDims {
                inner: Root::D,
                outer: root_bd_ring_dim,
                opening: root_bd_ring_dim,
            };
            let middle_dims = CommitmentRingDims {
                inner: Mid::D,
                outer: middle_bd_ring_dim,
                opening: middle_bd_ring_dim,
            };
            root_dims.validate_role_projection()?;
            middle_dims.validate_role_projection()?;

            let field_bits = Root::decomposition().field_bits();
            let opening_layout = key.opening_layout()?;

            // Plan the exact multi-group root at Root::D, then rebuild B and
            // shared D from the requested final projection geometry.
            let root_policy = policy_of::<Root>().direct_only();
            let precommitted_honest_fold_policies =
                vec![Root::root_honest_fold_policy(); key.precommitteds.len()];
            let mut root = akita_planner::find_schedule(
                key,
                Root::root_honest_fold_policy(),
                &precommitted_honest_fold_policies,
                &root_policy,
                Root::ring_challenge_config,
                Root::fold_challenge_shape_at_level,
            )?
            .schedule
            .root;
            retarget_commitment_matrices(
                &mut root.params.final_group.commitment,
                key.final_group.num_polynomials(),
                root_bd_ring_dim,
                root_bd_ring_dim,
            )?;
            root.params.open_commit_matrix = root.params.final_group.commitment.open_commit_matrix;
            let root_out = outgoing_witness_field_len(
                field_bits,
                &root.params.final_group.commitment,
                &opening_layout,
            )?;
            root.output_witness_len = root_out;
            let root_lb = root.params.final_group.commitment.log_basis_open;

            // Plan L1 at A=Mid::D from the exact root boundary and then
            // rebuild its B/D matrices before attaching the setup prefix.
            let mut mid_policy = policy_of::<Mid>().direct_only();
            mid_policy.witness_chunk = ChunkCfg::chunked_witness_cfg();
            mid_policy.setup_prefix_inner_ring_dimension = Suffix::D;
            let mid = akita_planner::plan_optimal_suffix(
                &mid_policy,
                Mid::ring_challenge_config,
                Mid::fold_challenge_shape_at_level,
                key.final_group.num_vars(),
                akita_planner::SuffixPlanStart {
                    level: 1,
                    witness_len: root_out,
                    log_basis: root_lb,
                    payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
                },
            )?;
            let mut l1_step = planned_fold_step(mid.folds.first().ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "recursive ring-dimension transition produced no middle fold".into(),
                )
            })?);
            retarget_commitment_matrices(
                &mut l1_step.params.witness,
                1,
                middle_bd_ring_dim,
                middle_bd_ring_dim,
            )?;

            let root_commitment = &root.params.final_group.commitment;
            let setup_prefix_d = mid_policy.setup_prefix_inner_ring_dimension;
            let natural_prefix_len = active_setup_field_len(root_commitment, &opening_layout)?;
            let n_prefix = padded_setup_prefix_len(natural_prefix_len);
            let ring_challenge = Mid::ring_challenge_config(setup_prefix_d)?;
            let prefix_params = akita_planner::test_support::plan_setup_prefix_commitment(
                akita_planner::test_support::SetupPrefixPlanRequest {
                    policy: &mid_policy,
                    ring_challenge: &ring_challenge,
                    fold_shape: l1_step.params.witness.fold_challenge_shape,
                    log_basis_outer: l1_step.params.witness.log_basis_outer,
                    log_basis_open: l1_step.params.witness.log_basis_open,
                    prefix_field_elements: n_prefix,
                    num_chunks: l1_step.params.witness.witness_chunk.num_chunks,
                    outer_ring_dimension: middle_bd_ring_dim,
                },
            )?;
            let setup_prefix = setup_prefix_slot_id(natural_prefix_len, prefix_params);
            let prefix_d_width = setup_prefix
                .commitment_params
                .d_segment_width(middle_bd_ring_dim)?;
            let total_d_width = l1_step
                .params
                .witness
                .open_commit_matrix
                .input_width()
                .checked_add(prefix_d_width)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("mixed recursive L1 D width overflow".into())
                })?;
            l1_step.params.witness.open_commit_matrix =
                OpenCommitMatrixParams::try_new_with_min_rank(
                    l1_step.params.witness.open_commit_matrix.sis_table_key(),
                    total_d_width,
                )?;
            l1_step.params.witness.setup_prefix = Some(setup_prefix.clone());
            l1_step.params.incoming_setup_prefix = Some(setup_prefix);
            l1_step.params.open_commit_matrix = l1_step.params.witness.open_commit_matrix;
            let l1_layout = akita_planner::suffix_opening_layout(
                l1_step.input_witness_len,
                Some(natural_prefix_len),
            )?;
            let l1_out =
                outgoing_witness_field_len(field_bits, &l1_step.params.witness, &l1_layout)?;
            l1_step.output_witness_len = l1_out;
            let l1_lb = l1_step.params.witness.log_basis_open;

            let mut suffix_policy = policy_of::<Suffix>().direct_only();
            suffix_policy.witness_chunk = ChunkCfg::chunked_witness_cfg();
            let suffix = akita_planner::plan_optimal_suffix(
                &suffix_policy,
                Suffix::ring_challenge_config,
                Suffix::fold_challenge_shape_at_level,
                key.final_group.num_vars(),
                akita_planner::SuffixPlanStart {
                    level: 2,
                    witness_len: l1_out,
                    log_basis: l1_lb,
                    payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix
                        .after(l1_step.params.witness.payload_mode),
                },
            )?;
            finish_schedule(root, vec![l1_step], suffix, &opening_layout)
        },
    )
}

/// Test-only config for the recursive mixed-D transition benchmark.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct RecursiveRingDimensionTransitionConfig<
    Root,
    Mid,
    Suffix,
    ChunkCfg,
    const ROOT_BD_RING_DIM: usize,
    const L1_BD_RING_DIM: usize,
>(PhantomData<fn() -> (Root, Mid, Suffix, ChunkCfg)>);

impl<Root, Mid, Suffix, ChunkCfg, const ROOT_BD_RING_DIM: usize, const L1_BD_RING_DIM: usize> Clone
    for RecursiveRingDimensionTransitionConfig<
        Root,
        Mid,
        Suffix,
        ChunkCfg,
        ROOT_BD_RING_DIM,
        L1_BD_RING_DIM,
    >
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Root, Mid, Suffix, ChunkCfg, const ROOT_BD_RING_DIM: usize, const L1_BD_RING_DIM: usize> Copy
    for RecursiveRingDimensionTransitionConfig<
        Root,
        Mid,
        Suffix,
        ChunkCfg,
        ROOT_BD_RING_DIM,
        L1_BD_RING_DIM,
    >
{
}

impl<Root, Mid, Suffix, ChunkCfg, const ROOT_BD_RING_DIM: usize, const L1_BD_RING_DIM: usize>
    Default
    for RecursiveRingDimensionTransitionConfig<
        Root,
        Mid,
        Suffix,
        ChunkCfg,
        ROOT_BD_RING_DIM,
        L1_BD_RING_DIM,
    >
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Root, Mid, Suffix, ChunkCfg, const ROOT_BD_RING_DIM: usize, const L1_BD_RING_DIM: usize>
    CommitmentConfig
    for RecursiveRingDimensionTransitionConfig<
        Root,
        Mid,
        Suffix,
        ChunkCfg,
        ROOT_BD_RING_DIM,
        L1_BD_RING_DIM,
    >
where
    Root: CommitmentConfig,
    Mid: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
    Suffix: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
    ChunkCfg: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
{
    type Field = Root::Field;
    type ExtField = Root::ExtField;

    const D: usize = Root::D;

    fn decomposition() -> DecompositionParams {
        Root::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Root::ring_challenge_config(d)
            .or_else(|_| Mid::ring_challenge_config(d))
            .or_else(|_| Suffix::ring_challenge_config(d))
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        ChunkCfg::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Root::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Root::ring_subfield_embedding_norm_bound()
            .max(Mid::ring_subfield_embedding_norm_bound())
            .max(Suffix::ring_subfield_embedding_norm_bound())
    }

    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        // Two known fixtures share this synthetic config:
        // - CI e2e: 2×(14,1) pre + (24,1) final → setup_prover(24, 3)
        // - profile/bench: 2×(16,1) pre + (32,2) final → setup_prover(32, 4)
        let (pre_nv, final_nv, final_np) = match (max_num_vars, max_num_batched_polys) {
            (24, 3) => (14, 24, 1),
            (32, 4) => (16, 32, 2),
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "recursive mixed-D profile supports setup capacities (24,3) and (32,4) only"
                        .into(),
                ));
            }
        };
        let pre_group = PolynomialGroupLayout::new(pre_nv, 1);
        let descriptor = committed_group_profile::<Root>(&pre_group)?;
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(final_nv, final_np),
            precommitteds: vec![descriptor, descriptor],
        };
        let schedule = recursive_ring_dimension_transition_schedule::<Root, Mid, Suffix, ChunkCfg>(
            &key,
            ROOT_BD_RING_DIM,
            L1_BD_RING_DIM,
        )?;
        akita_types::setup_matrix_capacity_for_schedule(&schedule)
    }

    fn setup_prefix_inner_ring_dimension() -> usize {
        Suffix::D
    }

    fn basis_range() -> (u32, u32) {
        Root::basis_range()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Root::root_honest_fold_policy()
    }

    fn chunked_witness_cfg() -> ChunkedWitnessCfg {
        ChunkCfg::chunked_witness_cfg()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        if key.precommitteds.is_empty() {
            return per_matrix_ring_dims_root_schedule::<Root>(
                key.final_group.num_vars(),
                key.final_group.num_polynomials(),
                ROOT_BD_RING_DIM,
                ROOT_BD_RING_DIM,
            );
        }
        recursive_ring_dimension_transition_schedule::<Root, Mid, Suffix, ChunkCfg>(
            &key,
            ROOT_BD_RING_DIM,
            L1_BD_RING_DIM,
        )
    }

    fn select_schedule_for_profiles(
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        select_synthetic_schedule_row::<Self>(profiles, synthetic_schedule_key(profiles))
    }

    fn resolve_schedule_selection(
        selection: OpeningScheduleSelection,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        resolve_synthetic_schedule_row::<Self>(selection)
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        opening_batch.check()?;
        let precommitteds = opening_batch
            .root_precommitted_group_layouts()?
            .iter()
            .copied()
            .map(|group| {
                let schedule = Self::runtime_schedule(AkitaScheduleLookupKey::single(group))?;
                Ok(akita_types::CommittedGroupProfile::from_params(
                    group,
                    &schedule.root.params.final_group.commitment,
                ))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Self::runtime_schedule(AkitaScheduleLookupKey {
            final_group: opening_batch.root_final_group_layout()?,
            precommitteds,
        })
    }
}
