use super::*;
use akita_algebra::{
    offset_eq::{eval_affine_digit_intervals, AffineWeightProduct},
    ring::scalar_powers_with_stride,
};

impl<E: Field> SetupContributionPlan<E> {
    /// Contract one group's structured E/T/Z terms through its canonical
    /// relation-column tensors.
    pub fn evaluate_structured_group<F>(
        &self,
        group_id: usize,
        block_challenges: &[E],
        opening_a_evals: &[E],
        alpha: E,
    ) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let group_index = self
            .groups
            .iter()
            .position(|group| group.group_id == group_id)
            .ok_or(AkitaError::InvalidProof)?;
        let group = &self.groups[group_index];
        let uses_evaluation_trace_consistency =
            matches!(group.opening_method, crate::OpeningMethod::EvaluationTrace);
        let direct_weights = match &self.direct_scan_state {
            DirectScanState::Unprepared => None,
            DirectScanState::Lifted {
                alpha: prepared,
                groups,
            } if *prepared == alpha => groups.get(group_index),
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "structured relation requires matching lifted direct-scan state".into(),
                ));
            }
        };
        let block_claims = group
            .num_claims
            .checked_mul(group.num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("structured block count overflow".into()))?;
        if block_challenges.len() != block_claims
            || opening_a_evals.len() != group.num_positions_per_block
        {
            return Err(AkitaError::InvalidProof);
        }

        let opening_gadget = extension_gadget::<F, E>(group.depth_open, group.log_basis_open);
        let commitment_gadget = extension_gadget::<F, E>(group.depth_commit, group.log_basis_outer);
        let witness_gadget = extension_gadget::<F, E>(group.depth_witness, group.log_basis_inner);
        let (outer_subcolumns, _) =
            SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims)?;
        let opening_subcolumns = group.opening_subcolumns;
        let e_stride = checked::product([opening_subcolumns, group.depth_open])
            .ok_or_else(|| AkitaError::InvalidSetup("structured E stride overflow".into()))?;
        let t_stride = checked::product([group.n_a, group.depth_commit, outer_subcolumns])
            .ok_or_else(|| AkitaError::InvalidSetup("structured T stride overflow".into()))?;
        let e_len = block_claims
            .checked_mul(e_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("structured E width overflow".into()))?;
        let t_len = block_claims
            .checked_mul(t_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("structured T width overflow".into()))?;
        let z_cols = group
            .num_positions_per_block
            .checked_mul(group.depth_witness)
            .ok_or_else(|| AkitaError::InvalidSetup("structured Z width overflow".into()))?;

        if let Some(weights) = direct_weights {
            if weights.e.len() != e_len
                || weights.t.len() != t_len
                || weights.z.len() != z_cols
                || group.a_row_weights.len() != group.n_a
            {
                return Err(AkitaError::InvalidProof);
            }
            let projected_opening_gadget = (opening_subcolumns != 1)
                .then(|| {
                    scalar_powers_with_stride(alpha, group.role_dims.d_d(), opening_subcolumns)
                })
                .transpose()?
                .map(|scales| {
                    scales
                        .iter()
                        .flat_map(|&scale| opening_gadget.iter().map(move |&gadget| scale * gadget))
                        .collect::<Vec<_>>()
                });
            let projected_commitment_gadget = (outer_subcolumns != 1)
                .then(|| scalar_powers_with_stride(alpha, group.role_dims.d_b(), outer_subcolumns))
                .transpose()?
                .map(|scales| {
                    scales
                        .iter()
                        .flat_map(|&scale| {
                            commitment_gadget.iter().map(move |&gadget| scale * gadget)
                        })
                        .collect::<Vec<_>>()
                });
            let direct_opening_gadget = projected_opening_gadget
                .as_deref()
                .unwrap_or(&opening_gadget);
            let direct_commitment_gadget = projected_commitment_gadget
                .as_deref()
                .unwrap_or(&commitment_gadget);
            let direct_witness_gadget = &witness_gadget;
            let t_row_stride = checked::product([outer_subcolumns, group.depth_commit])
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("structured T row stride overflow".into())
                })?;
            if direct_opening_gadget.len() != e_stride
                || direct_commitment_gadget.len() != t_row_stride
            {
                return Err(AkitaError::InvalidProof);
            }
            // E and T share the block-claim index space, so they stay one
            // fused fold gated on their combined width, and Z keeps its own
            // gate. Splitting E from T would drop the multi-core reduce
            // whenever each side alone sits under the threshold but the pair
            // clears it.
            let fold_et = |acc: Result<E, AkitaError>, block_claim: usize| {
                let e_start = block_claim
                    .checked_mul(e_stride)
                    .ok_or(AkitaError::InvalidProof)?;
                let e_eq =
                    checked_slice(&weights.e, e_start, e_stride, "structured direct E slice")?;
                let e = e_eq
                    .iter()
                    .zip(direct_opening_gadget)
                    .fold(E::zero(), |sum, (&eq, &gadget)| sum + eq * gadget);

                let t_start = block_claim
                    .checked_mul(t_stride)
                    .ok_or(AkitaError::InvalidProof)?;
                let t_eq =
                    checked_slice(&weights.t, t_start, t_stride, "structured direct T slice")?;
                let t = t_eq
                    .chunks_exact(t_row_stride)
                    .zip(group.a_row_weights.iter())
                    .fold(E::zero(), |sum, (row, &row_weight)| {
                        sum + row_weight
                            * row
                                .iter()
                                .zip(direct_commitment_gadget)
                                .fold(E::zero(), |inner, (&eq, &gadget)| inner + eq * gadget)
                    });
                let block_challenge = *block_challenges
                    .get(block_claim)
                    .ok_or(AkitaError::InvalidProof)?;
                let consistency = if uses_evaluation_trace_consistency {
                    group.consistency_weight * e
                } else {
                    E::zero()
                };
                Ok(acc? + block_challenge * (consistency + t))
            };
            const PARALLEL_THRESHOLD: usize = 1 << 14;
            let et_work = e_len
                .checked_add(t_len)
                .ok_or_else(|| AkitaError::InvalidSetup("structured E/T width overflow".into()))?;
            let run_et = || -> Result<E, AkitaError> {
                if et_work >= PARALLEL_THRESHOLD {
                    cfg_fold_reduce!(
                        0..block_claims,
                        || Ok(E::zero()),
                        fold_et,
                        |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
                    )
                } else {
                    (0..block_claims).fold(Ok(E::zero()), fold_et)
                }
            };
            let fold_z = |acc: Result<E, AkitaError>, position: usize| {
                let start = position
                    .checked_mul(group.depth_witness)
                    .ok_or(AkitaError::InvalidProof)?;
                let eq = checked_slice(
                    &weights.z,
                    start,
                    group.depth_witness,
                    "structured direct Z slice",
                )?;
                let inner = eq
                    .iter()
                    .zip(direct_witness_gadget)
                    .fold(E::zero(), |sum, (&eq, &gadget)| sum + eq * gadget);
                Ok(acc?
                    + *opening_a_evals
                        .get(position)
                        .ok_or(AkitaError::InvalidProof)?
                        * inner)
            };
            let run_z = || -> Result<E, AkitaError> {
                if z_cols >= PARALLEL_THRESHOLD {
                    cfg_fold_reduce!(
                        0..group.num_positions_per_block,
                        || Ok(E::zero()),
                        fold_z,
                        |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
                    )
                } else {
                    (0..group.num_positions_per_block).fold(Ok(E::zero()), fold_z)
                }
            };
            // Running E/T against Z costs one `rayon::join`, and that cost
            // grows with the pool size, so it has to be repaid by both sides
            // at once. Requiring the smaller of the two to clear the same
            // threshold its own reduce uses keeps the fork on groups where
            // each side is independently worth a parallel reduce; a group
            // with one large and one small side runs them in sequence.
            let (et, z) = if et_work.min(z_cols) >= PARALLEL_THRESHOLD {
                let (et, z) = cfg_join!(run_et, run_z);
                (et?, z?)
            } else {
                (run_et()?, run_z()?)
            };
            // Z carries the group's consistency weight on every position;
            // applying it once after reduction drops one field multiplication
            // per position and leaves the contracted equation auditable.
            return Ok(if uses_evaluation_trace_consistency {
                et + group.consistency_weight * z
            } else {
                et
            });
        }

        let opening_scales = (opening_subcolumns != 1)
            .then(|| scalar_powers_with_stride(alpha, group.role_dims.d_d(), opening_subcolumns))
            .transpose()?;
        let outer_scales = (outer_subcolumns != 1)
            .then(|| scalar_powers_with_stride(alpha, group.role_dims.d_b(), outer_subcolumns))
            .transpose()?;
        let point = self.relation_address.point();
        let base_ring_dim = self
            .relation_address_geometry
            .relation_coefficient_block_len();
        let opening_low = opening_scales.as_deref().unwrap_or(&[]);
        let outer_low = outer_scales.as_deref().unwrap_or(&[]);
        let projected_digits = |gadget: &[E], ratio: usize| -> Result<Option<Vec<E>>, AkitaError> {
            if ratio == 1 {
                return Ok(None);
            }
            let lanes = scalar_powers_with_stride(alpha, base_ring_dim, ratio)?;
            Ok(Some(
                gadget
                    .iter()
                    .flat_map(|&digit| lanes.iter().map(move |&lane| digit * lane))
                    .collect(),
            ))
        };
        let projected_opening = projected_digits(&opening_gadget, group.d_relation_ratio)?;
        let opening_digits = projected_opening.as_deref().unwrap_or(&opening_gadget);
        let projected_commitment = projected_digits(&commitment_gadget, group.b_relation_ratio)?;
        let commitment_digits = projected_commitment
            .as_deref()
            .unwrap_or(&commitment_gadget);

        if group.num_claims == 0 || group.num_live_blocks == 0 {
            return Err(AkitaError::InvalidSetup(
                "structured role tensor families disagree".into(),
            ));
        }
        let active_unit_count = group.active_unit_ranges.len();
        if active_unit_count == 0 || group.num_physical_units == 0 {
            return Err(AkitaError::InvalidSetup(
                "structured tensor partition is empty".into(),
            ));
        }
        let family_count = group
            .num_claims
            .checked_mul(active_unit_count)
            .ok_or(AkitaError::InvalidProof)?;
        if group.d_tensors.len() != family_count
            || group.physical_b.relation_tensors.len()
                != usize::from(group.physical_b.logical_rows()? != 0) * family_count
            || group.a_tensors.len() != usize::from(group.n_a != 0) * group.num_physical_units
        {
            return Err(AkitaError::InvalidSetup(
                "structured tensor families disagree with compiled active and physical units"
                    .into(),
            ));
        }

        // `build_group_role_tensors` emits D/B families unit-major with claim
        // inside each unit. The evaluation fold itself is claim-major, so this
        // explicit conversion is the single ordering boundary between them.
        let fold_family = |acc: Result<E, AkitaError>, family_index: usize| {
            let claim = family_index / active_unit_count;
            let unit_index = family_index % active_unit_count;
            let tensor_index = unit_index
                .checked_mul(group.num_claims)
                .and_then(|index| index.checked_add(claim))
                .ok_or(AkitaError::InvalidProof)?;
            let d_tensor = group
                .d_tensors
                .get(tensor_index)
                .ok_or(AkitaError::InvalidProof)?;
            let b_tensor = if group.physical_b.logical_rows()? == 0 {
                None
            } else {
                Some(
                    group
                        .physical_b
                        .relation_tensors
                        .get(tensor_index)
                        .ok_or(AkitaError::InvalidProof)?,
                )
            };
            let unit = group
                .active_unit_ranges
                .get(unit_index)
                .ok_or(AkitaError::InvalidProof)?;
            let global_block_start = unit.global_block_start;
            let unit_blocks = unit.num_live_blocks;
            let setup_block = claim
                .checked_mul(group.num_live_blocks)
                .and_then(|block| block.checked_add(global_block_start))
                .ok_or(AkitaError::InvalidProof)?;
            let expected_d_offset = setup_block
                .checked_mul(e_stride)
                .ok_or(AkitaError::InvalidProof)?;
            debug_assert_eq!(d_tensor.left_offset, expected_d_offset);
            if let Some(b_tensor) = b_tensor {
                let expected_b_offset = setup_block
                    .checked_mul(t_stride)
                    .ok_or(AkitaError::InvalidProof)?;
                debug_assert_eq!(b_tensor.left_offset, expected_b_offset);
            }
            let claim_start = claim
                .checked_mul(group.num_live_blocks)
                .ok_or(AkitaError::InvalidProof)?;
            let claim_challenges = checked_slice(
                block_challenges,
                claim_start,
                group.num_live_blocks,
                "structured E block factors",
            )?;
            let e_outer_start = global_block_start
                .checked_mul(opening_subcolumns)
                .ok_or(AkitaError::InvalidProof)?;
            let e_live_len = unit_blocks
                .checked_mul(opening_subcolumns)
                .ok_or(AkitaError::InvalidProof)?;
            let mut contribution = if uses_evaluation_trace_consistency {
                group.consistency_weight
                    * eval_affine_digit_intervals(
                        point,
                        &[d_tensor.right_offset],
                        e_outer_start,
                        e_live_len,
                        opening_digits.len(),
                        1,
                        opening_digits,
                        claim_challenges,
                        opening_low,
                        &[],
                    )?
            } else {
                E::zero()
            };

            if let Some(b_tensor) = b_tensor {
                let t_high_weights =
                    AffineWeightProduct::new(claim_challenges, &group.a_row_weights)?;
                let t_outer_start = global_block_start
                    .checked_mul(group.n_a)
                    .and_then(|start| start.checked_mul(outer_subcolumns))
                    .ok_or(AkitaError::InvalidProof)?;
                let t_live_len = unit_blocks
                    .checked_mul(group.n_a)
                    .and_then(|len| len.checked_mul(outer_subcolumns))
                    .ok_or(AkitaError::InvalidProof)?;
                contribution += eval_affine_digit_intervals(
                    point,
                    &[b_tensor.right_offset],
                    t_outer_start,
                    t_live_len,
                    commitment_digits.len(),
                    1,
                    commitment_digits,
                    &t_high_weights,
                    outer_low,
                    &[],
                )?;
            }
            Ok(acc? + contribution)
        };
        let evaluation = if family_count == 1 {
            fold_family(Ok(E::zero()), 0)
        } else {
            cfg_fold_reduce!(
                0..family_count,
                || Ok(E::zero()),
                fold_family,
                |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
            )
        }?;

        let projection_lanes = (group.a_relation_ratio != 1)
            .then(|| scalar_powers_with_stride(alpha, base_ring_dim, group.a_relation_ratio))
            .transpose()?;
        let fold_digits = if let Some(lanes) = &projection_lanes {
            group
                .fold_gadget
                .iter()
                .flat_map(|&fold| lanes.iter().map(move |&lane| -(fold * lane)))
                .collect::<Vec<_>>()
        } else {
            group.fold_gadget.iter().map(|&fold| -fold).collect()
        };
        let a_base_offsets = group
            .a_tensors
            .iter()
            .map(|tensor| tensor.right_offset)
            .collect::<Vec<_>>();
        let z = if witness_gadget.len().is_power_of_two() {
            eval_affine_digit_intervals(
                point,
                &a_base_offsets,
                0,
                z_cols,
                fold_digits.len(),
                1,
                &fold_digits,
                opening_a_evals,
                &witness_gadget,
                &[],
            )?
        } else {
            let weighted_base_count =
                checked::product([opening_a_evals.len(), a_base_offsets.len()]).ok_or_else(
                    || AkitaError::InvalidSetup("structured weighted A base count overflow".into()),
                )?;
            let mut weighted_bases = Vec::new();
            let mut base_scales = Vec::new();
            weighted_bases
                .try_reserve_exact(weighted_base_count)
                .map_err(|_| {
                    AkitaError::InvalidSetup("structured weighted A bases are too large".into())
                })?;
            base_scales
                .try_reserve_exact(weighted_base_count)
                .map_err(|_| {
                    AkitaError::InvalidSetup("structured weighted A scales are too large".into())
                })?;
            for (position, &opening_a) in opening_a_evals.iter().enumerate() {
                let position_offset = position
                    .checked_mul(group.depth_witness)
                    .and_then(|offset| offset.checked_mul(fold_digits.len()))
                    .ok_or(AkitaError::InvalidProof)?;
                for &base in &a_base_offsets {
                    weighted_bases.push(
                        base.checked_add(position_offset)
                            .ok_or(AkitaError::InvalidProof)?,
                    );
                    base_scales.push(opening_a);
                }
            }
            eval_affine_digit_intervals(
                point,
                &weighted_bases,
                0,
                group.depth_witness,
                fold_digits.len(),
                1,
                &fold_digits,
                &witness_gadget,
                &[],
                &base_scales,
            )?
        };
        Ok(if uses_evaluation_trace_consistency {
            evaluation + group.consistency_weight * z
        } else {
            evaluation
        })
    }
}
