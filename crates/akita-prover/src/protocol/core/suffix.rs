use super::*;
#[cfg(not(feature = "zk"))]
use akita_types::schedule_terminal_direct_witness_shape;

/// Prover state carried between suffix fold levels.
pub struct SuffixProverState<F: FieldCore, L: FieldCore> {
    /// Current committed suffix witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical suffix witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Current suffix witness commitment.
    pub commitment: FlatRingVec<F>,
    /// D-erased suffix commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next suffix opening point.
    pub sumcheck_challenges: Vec<L>,
    /// Claimed logical opening of `logical_w` at `sumcheck_challenges`.
    pub opening: L,
    /// Transcript-visible masked handle for `opening`.
    #[cfg(feature = "zk")]
    pub opening_public: L,
    /// Proof-level ZK hiding material fixed at batched-prove startup.
    #[cfg(feature = "zk")]
    pub zk_hiding: ZkHidingProverState<F>,
}

impl<F: FieldCore, L: FieldCore> SuffixProverState<F, L> {
    /// Logical witness represented by the carried opening claim.
    #[inline]
    pub fn logical_w(&self) -> &RecursiveWitnessFlat {
        self.logical_w.as_ref().unwrap_or(&self.w)
    }
}

/// Drive the recursive fold suffix (after the root) under config `Cfg`.
///
/// The selected planner `schedule` is authoritative: it determines the fold
/// count, per-level `LevelParams`, successor params, and the terminal direct
/// witness basis. Earlier suffix levels run intermediate folds; the last
/// suffix level runs the terminal fold which ships the cleartext
/// `final_witness`.
///
/// # Errors
///
/// Returns an error if level proving fails, or an invalid-setup error when the
/// schedule's recursive suffix is empty (root-terminal proofs do not run this
/// helper).
#[allow(clippy::too_many_arguments)]
pub fn prove_suffix<Cfg, T, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    starting_state: SuffixProverState<Cfg::Field, Cfg::ExtField>,
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
) -> Result<RecursiveSuffixOutcome<Cfg::Field, Cfg::ExtField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<Cfg::Field>,
    T: Transcript<Cfg::Field>,
    B: ProverComputeBackend<Cfg::Field>,
{
    let planned_num_levels = schedule_num_fold_levels(schedule);
    if planned_num_levels < 2 {
        return Err(AkitaError::InvalidSetup(
            "prove_suffix expects a non-empty recursive suffix".to_string(),
        ));
    }
    let mut intermediate_levels = Vec::new();
    let mut current_state = starting_state;
    let mut level = 1usize;

    #[cfg(not(feature = "zk"))]
    let terminal_direct_witness_shape = schedule_terminal_direct_witness_shape(schedule)?;
    let terminal_result = loop {
        let scheduled = schedule.get_execution_schedule(level)?;
        scheduled.validate_current_w_len(current_state.w.len())?;
        let level_params = &scheduled.params;
        let level_d = level_params.ring_dimension;
        let is_terminal_level = scheduled.is_terminal;
        let m_row_layout = if is_terminal_level {
            MRowLayout::WithoutDBlock
        } else {
            MRowLayout::WithDBlock
        };
        let out = if level_d == D {
            let prepared_fold = prepare_fold_data::<Cfg::Field, Cfg::ExtField, T, B, D>(
                backend,
                prepared,
                transcript,
                current_state,
                level,
                level_params,
                m_row_layout,
            )?;
            prove_fold::<Cfg::Field, Cfg::ExtField, T, B, Cfg, D>(
                expanded,
                prefix_slots,
                backend,
                prepared,
                transcript,
                level,
                &scheduled,
                prepared_fold,
                setup_contribution_mode,
                is_terminal_level,
                #[cfg(not(feature = "zk"))]
                if is_terminal_level {
                    Some(terminal_direct_witness_shape)
                } else {
                    None
                },
            )
        } else {
            dispatch_ring_dim_result!(level_d, |D_LEVEL| {
                let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
                let level_prefix_slots = SetupPrefixProverRegistry::new();
                let prepared_fold =
                    prepare_fold_data::<Cfg::Field, Cfg::ExtField, T, B, { D_LEVEL }>(
                        backend,
                        &level_prepared,
                        transcript,
                        current_state,
                        level,
                        level_params,
                        m_row_layout,
                    )?;
                prove_fold::<Cfg::Field, Cfg::ExtField, T, B, Cfg, { D_LEVEL }>(
                    expanded,
                    &level_prefix_slots,
                    backend,
                    &level_prepared,
                    transcript,
                    level,
                    &scheduled,
                    prepared_fold,
                    setup_contribution_mode,
                    is_terminal_level,
                    #[cfg(not(feature = "zk"))]
                    if is_terminal_level {
                        Some(terminal_direct_witness_shape)
                    } else {
                        None
                    },
                )
            })
        }?;
        if is_terminal_level {
            break out.get_terminal()?;
        }

        let out = out.get_intermediate()?;
        intermediate_levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    };
    #[cfg(not(feature = "zk"))]
    let terminal = terminal_result;
    #[cfg(feature = "zk")]
    let (terminal, zk_hiding) = terminal_result;

    let mut steps = intermediate_levels;
    let final_w_len = terminal.final_witness().num_elems();
    steps.push(AkitaLevelProof::Terminal {
        extension_opening_reduction: terminal.extension_opening_reduction,
        stage2: terminal.stage2,
        final_w_len,
    });

    Ok(RecursiveSuffixOutcome {
        steps,
        #[cfg(feature = "zk")]
        zk_hiding,
        num_levels: planned_num_levels,
    })
}
/// Derive the fused-trace evaluation target and EOR tail scale, and fail-fast
/// check that the folded witness ties back to the carried claim.
///
/// On the degree-one (no-EOR) path the target is the recovered subfield inner
/// product, which must equal the carried `expected_opening`. On the EOR path it
/// is the reduction's `final_claim`, cross-checked against the recovered value
/// scaled by the transparent factor. This writes nothing to the transcript: the
/// verifier re-derives the same relation through the fused stage-2 term.
fn compute_trace_target<F, L, const D: usize>(
    reduction: &Option<ExtensionOpeningReduction<L>>,
    folded_rings: &[CyclotomicRing<F, D>],
    prepared_point: &PreparedOpeningPoint<F, L, D>,
    expected_opening: L,
) -> Result<(L, L), AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    L: ExtField<F> + FpExtEncoding<F>,
{
    let folded_ring = folded_rings.first().ok_or(AkitaError::InvalidProof)?;
    let internal_claim = recover_ring_subfield_inner_product::<F, L, D>(
        folded_ring,
        &prepared_point.packed_inner_point,
    )?;
    match reduction {
        Some(reduction) => {
            check_extension_opening_reduction_output(
                reduction.final_claim,
                internal_claim,
                reduction.final_factor,
            )?;
            Ok((reduction.final_claim, reduction.final_factor))
        }
        None => {
            if internal_claim != expected_opening {
                return Err(AkitaError::InvalidInput(
                    "recursive opening does not match carried claim".to_string(),
                ));
            }
            Ok((internal_claim, L::one()))
        }
    }
}

fn validate_recursive_opening_block_count<F, L, const D: usize>(
    prepared_point: &PreparedOpeningPoint<F, L, D>,
    level_params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    L: FieldCore,
{
    let actual = prepared_point.ring_opening_point.b.len();
    if actual != level_params.num_blocks {
        return Err(AkitaError::InvalidInput(format!(
            "recursive opening block count {actual} does not match scheduled num_blocks {}",
            level_params.num_blocks
        )));
    }
    Ok(())
}

/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// The caller owns schedule/config selection and passes the next-level
/// commitment params. This function owns recursive opening-point reduction,
/// witness folding, public recursive transcript absorbs, recursive
/// ring-relation construction, and the folded-level prover mechanics.
///
/// # Errors
///
/// Returns an error if the recursive opening point has the wrong dimension,
/// witness folding or ring-relation construction fails, or the folded
/// prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prepare_fold_data<F, L, T, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: SuffixProverState<F, L>,
    level: usize,
    level_params: &LevelParams,
    m_row_layout: MRowLayout,
) -> Result<PreparedFold<F, L, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: FpExtEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prepare_fold_data"
        );
    }

    let witness_view = current_state.w.view::<F, D>()?;
    let logical_w = current_state.logical_w.as_ref().unwrap_or(&current_state.w);
    let typed_hint = current_state.hint.to_typed::<D>()?;
    let opening_point = &current_state.sumcheck_challenges;
    #[cfg(feature = "zk")]
    let mut zk_hiding = current_state.zk_hiding;

    current_state
        .commitment
        .append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let (reduction, protocol_point) = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        (None, opening_point.to_vec())
    } else {
        let logical_view = logical_w.view::<F, D>()?;
        let logical_polys = [&logical_view];
        let opening_batch = OpeningBatch::same_point(opening_point.len(), 1)?;
        let proved = prove_extension_opening_reduction::<F, L, T, _, D>(
            &logical_polys,
            &opening_batch,
            opening_point,
            #[cfg(feature = "zk")]
            None,
            true,
            transcript,
            "recursive",
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?;
        match proved.openings.as_slice() {
            [tensor_opening] if *tensor_opening == current_state.opening => {}
            [_] => return Err(AkitaError::InvalidProof),
            [] => {
                return Err(AkitaError::InvalidInput(
                    "recursive EOR preparation produced no opening".to_string(),
                ));
            }
            _ => {
                return Err(AkitaError::InvalidInput(
                    "recursive EOR preparation produced extra openings".to_string(),
                ));
            }
        }
        (Some(proved.reduction), proved.protocol_point)
    };
    let prepared_point = prepare_opening_point::<F, L, D>(
        &protocol_point,
        BasisMode::Lagrange,
        level_params,
        alpha,
        BlockOrder::ColumnMajor,
    )?;
    validate_recursive_opening_block_count(&prepared_point, level_params)?;
    let recursive_polys = [&witness_view];

    let (folded_rings, e_folded_by_claim) = evaluate_claims_at_prepared_point(
        &recursive_polys,
        &prepared_point,
        level_params.block_len,
    )?;
    for pt in &prepared_point.padded_point {
        append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
    }

    let (trace_eval_target, trace_scale) = compute_trace_target::<F, L, D>(
        &reduction,
        &folded_rings,
        &prepared_point,
        current_state.opening,
    )?;
    #[cfg(feature = "zk")]
    let trace_eval_target_public = match &reduction {
        Some(reduction) => reduction.final_claim_public,
        None => current_state.opening_public,
    };
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let recursive_num_vars = level_params.recursive_opening_num_vars()?;
    let opening_batch = OpeningBatch::same_point(recursive_num_vars, 1)?;
    let recursive_commitment = RingCommitment {
        u: commitment_u.to_vec(),
    };
    let row_coefficient_rings = vec![CyclotomicRing::one(); opening_batch.num_claims()];
    let (instance, witness) = RingRelationProver::new::<F, D, _, _, _>(
        backend,
        prepared,
        prepared_point.ring_opening_point.clone(),
        prepared_point.ring_multiplier_point.clone(),
        &recursive_polys,
        e_folded_by_claim,
        opening_batch,
        level_params.clone(),
        vec![typed_hint],
        transcript,
        std::slice::from_ref(&recursive_commitment),
        row_coefficient_rings,
        m_row_layout,
    )?;
    Ok(PreparedFold {
        commitment: current_state.commitment,
        instance,
        witness,
        extension_opening_reduction: reduction.map(|reduction| reduction.proof),
        trace_eval_target,
        trace_scale,
        trace_prepared_point: Some(prepared_point),
        trace_claim_scales: None,
        #[cfg(feature = "zk")]
        trace_eval_target_public,
        #[cfg(feature = "zk")]
        zk_hiding,
        row_coefficients: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp32;
    use akita_types::{RingOpeningPoint, SisModulusFamily};

    type TestF = Fp32<251>;
    const D: usize = 4;

    fn level_params_with_four_blocks() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q32,
            D,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        )
        .with_decomp(1, 2, 1, 1, 0)
        .expect("synthetic level params")
    }

    #[test]
    fn recursive_opening_block_count_mismatch_is_rejected() {
        let level_params = level_params_with_four_blocks();
        assert_eq!(level_params.num_blocks, 4);
        let prepared_point: PreparedOpeningPoint<TestF, TestF, D> = PreparedOpeningPoint {
            padded_point: Vec::new(),
            ring_opening_point: RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one(), TestF::zero()],
            },
            ring_multiplier_point: RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one(), TestF::zero()],
            }),
            packed_inner_point: CyclotomicRing::<TestF, D>::zero(),
        };

        let err = validate_recursive_opening_block_count(&prepared_point, &level_params)
            .expect_err("mismatched recursive opening block count should reject");
        assert!(
            matches!(err, AkitaError::InvalidInput(message) if message.contains("scheduled num_blocks"))
        );
    }

    #[cfg(not(feature = "zk"))]
    #[test]
    fn non_zk_eor_mismatch_is_rejected() {
        let prepared_point: PreparedOpeningPoint<TestF, TestF, D> = PreparedOpeningPoint {
            padded_point: Vec::new(),
            ring_opening_point: RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one()],
            },
            ring_multiplier_point: RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one()],
            }),
            packed_inner_point: CyclotomicRing::<TestF, D>::zero(),
        };
        let folded_rings = [CyclotomicRing::<TestF, D>::zero()];
        let reduction = Some(ExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: Vec::new(),
                sumcheck: SumcheckProof {
                    round_polys: Vec::new(),
                },
            },
            final_claim: TestF::one(),
            final_factor: TestF::one(),
        });

        let err = compute_trace_target::<TestF, TestF, D>(
            &reduction,
            &folded_rings,
            &prepared_point,
            TestF::zero(),
        )
        .expect_err("non-zk EOR mismatch should reject");

        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
