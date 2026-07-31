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
/// The setup prefix is planned at the producer's common relation dimension.
/// This is D128 for the requested `256/128/128 -> 128/64/64 -> 64` profile,
/// so the experiment exercises dynamic setup-prefix dispatch rather than the
/// shipped D64-only planner path.
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
            key.validate()?;
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
            root_dims.validate_a_carrier()?;
            middle_dims.validate_a_carrier()?;
            for descriptor in &key.precommitteds {
                if descriptor.inner_ring_dimension != Root::D
                    || descriptor.outer_ring_dimension != root_bd_ring_dim
                {
                    return Err(AkitaError::InvalidSetup(
                        "recursive transition precommit dimensions must match the root A/B band"
                            .into(),
                    ));
                }
            }

            let field_bits = Root::decomposition().field_bits();
            let opening_layout = key.opening_layout()?;

            // Plan the exact multi-group root at Root::D, then rebuild B and
            // shared D from the requested final carrier geometry.
            let root_policy = policy_of::<Root>().direct_only();
            let mut root = akita_planner::find_schedule(
                key,
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
            root.params.final_group.source =
                RootSource::from_commitment(&root.params.final_group.commitment);
            root.params.open_commit_matrix = root
                .params
                .final_group
                .commitment
                .open_commit_matrix
                .clone();
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
            let mid = akita_planner::plan_optimal_suffix(
                &mid_policy,
                Mid::ring_challenge_config,
                Mid::fold_challenge_shape_at_level,
                key.final_group.num_vars(),
                1,
                root_out,
                root_lb,
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

            // The producer's common relation dimension is the source ring for
            // the committed setup prefix and must equal the consuming A ring.
            let root_commitment = &root.params.final_group.commitment;
            let shared_root_d = root_commitment.role_dims().d_d();
            let mut setup_prefix_d = root_commitment.role_dims().common_relation_coeff_count();
            for group in &root_commitment.precommitted_groups {
                setup_prefix_d = setup_prefix_d
                    .min(group.role_dims(shared_root_d).common_relation_coeff_count());
            }
            if setup_prefix_d != Mid::D {
                return Err(AkitaError::InvalidSetup(format!(
                    "root setup projection D{setup_prefix_d} does not match L1 A dimension D{}",
                    Mid::D
                )));
            }
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
            let setup_prefix =
                setup_prefix_slot_id(setup_prefix_d, natural_prefix_len, prefix_params);
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
            l1_step.params.open_commit_matrix = l1_step.params.witness.open_commit_matrix.clone();
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
                2,
                l1_out,
                l1_lb,
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

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
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
        let pre_schedule = per_matrix_ring_dims_root_schedule::<Root>(
            pre_group.num_vars(),
            pre_group.num_polynomials(),
            ROOT_BD_RING_DIM,
            ROOT_BD_RING_DIM,
        )?;
        let descriptor = akita_types::PrecommittedGroupDescriptor::from_params(
            pre_group,
            &pre_schedule.root.params.final_group.commitment,
        );
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(final_nv, final_np),
            precommitteds: vec![descriptor, descriptor],
        };
        let schedule = recursive_ring_dimension_transition_schedule::<Root, Mid, Suffix, ChunkCfg>(
            &key,
            ROOT_BD_RING_DIM,
            L1_BD_RING_DIM,
        )?;
        akita_types::setup_matrix_envelope_for_schedule(&schedule, Root::D)
    }

    fn basis_range() -> (u32, u32) {
        Root::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Root::onehot_chunk_size()
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
                Ok(akita_types::PrecommittedGroupDescriptor::from_params(
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
