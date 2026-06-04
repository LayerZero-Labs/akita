//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::{ClaimIncidenceSummary, CleartextWitnessShape, LevelParams, RingOpeningPoint};
use akita_field::{AkitaError, CanonicalField, FieldCore};

/// Public inputs that deterministically select one level's active Akita params.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleInputs {
    /// Root polynomial variable count.
    pub num_vars: usize,
    /// Fold level, where `0` is the original polynomial.
    pub level: usize,
    /// Current witness length in field elements before this level runs.
    pub current_w_len: usize,
}

/// Validate ring-switch opening-point routing against a level layout.
///
/// # Errors
///
/// Returns an error when there are no opening points, the claim-to-point table
/// has the wrong length, an opening point does not match `lp`, or a routed
/// point index is out of range.
pub fn validate_opening_points_for_claims<F: FieldCore>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    lp: &LevelParams,
    num_claims: usize,
) -> Result<(), AkitaError> {
    if opening_points.is_empty() {
        return Err(AkitaError::InvalidInput(
            "multipoint ring switch requires at least one opening point".to_string(),
        ));
    }
    if claim_to_point.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: claim_to_point.len(),
        });
    }
    for opening_point in opening_points {
        if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
            return Err(AkitaError::InvalidInput(
                "multipoint ring switch m-eval opening-point layout mismatch".to_string(),
            ));
        }
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= opening_points.len())
    {
        return Err(AkitaError::InvalidInput(
            "multipoint ring switch claim-to-point index out of range".to_string(),
        ));
    }
    Ok(())
}

/// Public runtime key that selects a concrete root schedule context.
///
/// This is intentionally narrower than a full schedule table entry: it records
/// only the public inputs that pick a root plan, not the resulting plan data.
///
/// Under the one-commitment-per-opening-point invariant, the number of
/// distinct point commitments equals the number of distinct opening points,
/// so the planner-facing projection records `num_points`. The generated
/// schedule table key still calls this field `num_commitment_groups` for ABI
/// stability; the translation happens in
/// `akita_planner::generated_schedule_lookup_key`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleLookupKey {
    /// Root polynomial arity.
    pub num_vars: usize,
    /// Number of distinct opening points (and therefore, distinct point
    /// commitments).
    pub num_points: usize,
    /// Number of commitment-side `t` protocol vectors.
    pub num_t_vectors: usize,
    /// Number of root relation `w` protocol vectors.
    pub num_w_vectors: usize,
    /// Number of distinct `z` protocol vectors.
    pub num_z_vectors: usize,
}

impl AkitaScheduleLookupKey {
    /// Singleton root-opening context.
    pub const fn singleton(num_vars: usize) -> Self {
        Self {
            num_vars,
            num_points: 1,
            num_t_vectors: 1,
            num_w_vectors: 1,
            num_z_vectors: 1,
        }
    }

    /// General root-opening context.
    pub const fn new(
        num_vars: usize,
        num_t_vectors: usize,
        num_w_vectors: usize,
        num_z_vectors: usize,
    ) -> Self {
        Self::new_with_points(
            num_vars,
            num_z_vectors,
            num_t_vectors,
            num_w_vectors,
            num_z_vectors,
        )
    }

    /// General root-opening context with an explicit opening-point count.
    pub const fn new_with_points(
        num_vars: usize,
        num_points: usize,
        num_t_vectors: usize,
        num_w_vectors: usize,
        num_z_vectors: usize,
    ) -> Self {
        Self {
            num_vars,
            num_points,
            num_t_vectors,
            num_w_vectors,
            num_z_vectors,
        }
    }

    /// Build a schedule lookup key from normalized opening incidence.
    ///
    /// Each opening point cites exactly one commitment, so the planner-facing
    /// projection carries only the per-point arities.
    ///
    /// # Errors
    ///
    /// Returns an error if the incidence routing tables are malformed.
    pub fn new_from_incidence(incidence: &ClaimIncidenceSummary) -> Result<Self, AkitaError> {
        let num_t_vectors = incidence.num_polynomials();
        if incidence.claim_to_point().len() != incidence.num_claims() {
            return Err(AkitaError::InvalidInput(
                "claim incidence summary lengths do not match aggregate counts".to_string(),
            ));
        }
        for &point_idx in incidence.claim_to_point() {
            if point_idx >= incidence.num_points() {
                return Err(AkitaError::InvalidInput(
                    "claim incidence summary contains out-of-range routing".to_string(),
                ));
            }
        }

        Ok(Self::new_with_points(
            incidence.num_vars(),
            incidence.num_points(),
            num_t_vectors,
            incidence.num_claims(),
            incidence.num_public_rows(),
        ))
    }
}

/// Number of gadget decomposition levels needed for `r` over field `F`.
pub fn r_decomp_levels<F: CanonicalField>(log_basis: u32) -> usize {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    crate::sis::compute_num_digits_full_field(field_bits, log_basis)
}

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `-1` in `Z_q` is `q - 1`.
pub fn detect_field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

/// Total ring elements in the recursive witness polynomial.
///
/// Components: `w_hat + t_hat + B-blinding + decomposed z_pre + decomposed r`.
pub fn w_ring_element_count<F: CanonicalField>(lp: &LevelParams) -> Result<usize, AkitaError> {
    w_ring_element_count_with_counts::<F>(lp, 1, 1, 1, 1)
}

/// Total ring elements in a recursive witness polynomial for explicit batch counts.
pub fn w_ring_element_count_with_counts<F: CanonicalField>(
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
) -> Result<usize, AkitaError> {
    w_ring_element_count_with_counts_for_layout::<F>(
        lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        crate::layout::MRowLayout::WithDBlock,
    )
}

/// Total ring elements in a recursive witness polynomial for an explicit
/// M-row layout. The terminal layout drops the D-block from the M-matrix,
/// which shrinks the per-row `r` quotients by `n_d * r_decomp_levels` ring
/// elements relative to the intermediate layout.
pub fn w_ring_element_count_with_counts_for_layout<F: CanonicalField>(
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
    layout: crate::layout::MRowLayout,
) -> Result<usize, AkitaError> {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    w_ring_element_count_with_counts_for_layout_bits(
        field_bits,
        lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        layout,
    )
}

/// Non-generic variant of [`w_ring_element_count_with_counts`] for callers
/// that already know the effective field bit width.
pub fn w_ring_element_count_with_counts_bits(
    field_bits: u32,
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
) -> Result<usize, AkitaError> {
    w_ring_element_count_with_counts_for_layout_bits(
        field_bits,
        lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        crate::layout::MRowLayout::WithDBlock,
    )
}

/// Non-generic variant of [`w_ring_element_count_with_counts_for_layout`] for
/// callers that already know the effective field bit width. The planner
/// search uses this to keep its API free of a base-field type parameter.
pub fn w_ring_element_count_with_counts_for_layout_bits(
    field_bits: u32,
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
    layout: crate::layout::MRowLayout,
) -> Result<usize, AkitaError> {
    let w_hat_count = num_w_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness W width overflow".to_string()))?;
    let t_hat_count = num_t_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.a_key.row_len()))
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".to_string()))?;
    let num_digits_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    let z_pre_count = num_public_rows
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".to_string()))?;
    // One public y-row per packaged public opening row.
    let r_rows = lp.m_row_count_for(num_points, num_public_rows, layout)?;
    let r_count = r_rows
        .checked_mul(crate::sis::compute_num_digits_full_field(
            field_bits,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("witness r-tail width overflow".to_string()))?;
    #[cfg(feature = "zk")]
    {
        // Terminal layout drops the D-block from the relation entirely, so
        // its per-row blinding is also unused. Intermediate layout keeps the
        // D-block blinding as before.
        let d_blinding_count = match layout {
            crate::layout::MRowLayout::WithDBlock => crate::zk::blinding_column_count_from_bits(
                lp.d_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
                field_bits as usize,
            ),
            crate::layout::MRowLayout::WithoutDBlock => 0,
        };
        let b_blinding_count = num_points
            .checked_mul(crate::zk::blinding_column_count_from_bits(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
                field_bits as usize,
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("ZK B-blinding width overflow".to_string()))?;
        w_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(b_blinding_count))
            .and_then(|n| n.checked_add(d_blinding_count))
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
    }
    #[cfg(not(feature = "zk"))]
    {
        w_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
    }
}

/// Parameters for one fold level in the computed schedule.
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Unified level parameters (ring dimension, Ajtai keys, block geometry,
    /// digit depths, challenge config).
    pub params: LevelParams,
    /// Witness length entering this level.
    pub current_w_len: usize,
    /// Witness length leaving this level.
    pub next_w_len: usize,
    /// Proof bytes for this level.
    pub level_bytes: usize,
}

/// Terminal direct-send step.
#[derive(Clone, Debug)]
pub struct DirectStep {
    /// Witness length entering the direct step.
    pub current_w_len: usize,
    /// Serialized terminal witness payload shape.
    pub witness_shape: CleartextWitnessShape,
    /// Direct witness bytes.
    pub direct_bytes: usize,
    /// Root commit layout for root-direct schedules (schedule starts
    /// with this `Direct`, `witness_shape = FieldElements`).
    ///
    /// `Some(_)` is the root commit layout — the verifier replays
    /// commitments against it and the transcript binds it through the
    /// per-proof effective-schedule digest (`PlanSection`). `None` is the
    /// *uncommittable* edge: a table-recorded large-`num_vars` entry
    /// whose singleton root layout exceeds the audited SIS floor. The
    /// schedule is intentionally usable for proof-size exploration and
    /// DP planning, but `get_params_for_batched_commitment` rejects it
    /// loudly and `setup_level_params_from_runtime_schedule` returns
    /// an empty list. Don't commit through such a schedule.
    ///
    /// Terminal-direct steps (`witness_shape = PackedDigits`, schedule
    /// is `[Fold, …, Fold, Direct]`) ship the cleartext witness without
    /// committing — the verifier absorbs the bytes into the transcript
    /// and re-evaluates the witness directly. They always carry
    /// `params = None`. The active `log_basis` lives on
    /// [`Self::witness_shape`]; `scheduled_next_level_params`
    /// synthesizes a [`LevelParams::log_basis_stub`] from it so the
    /// prover's terminal-fold path still receives a `LevelParams`-shaped
    /// successor (only `log_basis` is consulted there).
    pub params: Option<LevelParams>,
}

impl DirectStep {
    /// Active terminal log-basis for packed direct witnesses.
    pub fn log_basis(&self, field_bits: u32) -> u32 {
        match self.witness_shape {
            CleartextWitnessShape::PackedDigits((_, bits)) => bits,
            CleartextWitnessShape::FieldElements(_) => field_bits,
        }
    }
}

/// A single step in the schedule.
#[derive(Clone, Debug)]
pub enum Step {
    /// Fold through one recursive level.
    Fold(FoldStep),
    /// Send the terminal witness directly.
    Direct(DirectStep),
}

/// Complete schedule with step-by-step parameters.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Ordered proof schedule steps.
    pub steps: Vec<Step>,
    /// Exact total proof bytes for the schedule.
    pub total_bytes: usize,
}

impl Schedule {
    /// Iterate over the fold steps in execution order.
    pub fn fold_steps(&self) -> impl Iterator<Item = &FoldStep> + '_ {
        self.steps.iter().filter_map(|step| match step {
            Step::Fold(fold) => Some(fold),
            Step::Direct(_) => None,
        })
    }

    /// Number of fold levels before the terminal direct step.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_steps().count()
    }

    /// Witness length (field elements) entering the first step, or `None`
    /// when the schedule has no steps.
    pub fn initial_w_len(&self) -> Option<usize> {
        self.steps.first().map(|step| match step {
            Step::Fold(fold) => fold.current_w_len,
            Step::Direct(direct) => direct.current_w_len,
        })
    }

    /// Append the descriptor digest encoding for this effective schedule.
    ///
    /// Kept next to [`Schedule`] so protocol-affecting step field changes are
    /// reviewed with their Fiat-Shamir binding.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.steps.len());
        for step in &self.steps {
            match step {
                Step::Fold(fold) => {
                    bytes.push(0);
                    fold.params.append_descriptor_bytes(bytes);
                    push_usize(bytes, fold.current_w_len);
                    push_usize(bytes, fold.next_w_len);
                    push_usize(bytes, fold.level_bytes);
                }
                Step::Direct(direct) => {
                    bytes.push(1);
                    push_usize(bytes, direct.current_w_len);
                    append_direct_witness_shape_descriptor_bytes(bytes, &direct.witness_shape);
                    push_usize(bytes, direct.direct_bytes);
                    // Root-direct commit layout (`Some` for committable root
                    // entries, `None` for terminal-direct handoffs). Binding it
                    // here is what lets the transcript drop the redundant
                    // setup-level `level_params_digest`: the per-proof schedule
                    // digest now pins the root-direct commit params directly.
                    match &direct.params {
                        Some(params) => {
                            bytes.push(1);
                            params.append_descriptor_bytes(bytes);
                        }
                        None => bytes.push(0),
                    }
                }
            }
        }
        push_usize(bytes, self.total_bytes);
    }
}

fn append_direct_witness_shape_descriptor_bytes(
    bytes: &mut Vec<u8>,
    shape: &CleartextWitnessShape,
) {
    match shape {
        CleartextWitnessShape::PackedDigits((num_elems, bits_per_elem)) => {
            bytes.push(0);
            push_usize(bytes, *num_elems);
            push_u32(bytes, *bits_per_elem);
        }
        CleartextWitnessShape::FieldElements(coeff_len) => {
            bytes.push(1);
            push_usize(bytes, *coeff_len);
        }
    }
}

/// Witness length entering the root fold, in field elements.
pub fn root_current_w_len(lp: &LevelParams) -> usize {
    lp.num_blocks
        .checked_mul(lp.block_len)
        .and_then(|len| len.checked_mul(lp.ring_dimension))
        .unwrap_or(0)
}

/// Build the root-direct schedule for roots that do not admit a fold step.
///
/// `commit_params` carries the root commit layout that
/// `Cfg::get_params_for_batched_commitment` returns for this schedule shape.
///
/// # Errors
///
/// Returns an error if `num_vars` cannot be represented as a witness length.
pub fn root_direct_schedule(
    num_vars: usize,
    commit_params: LevelParams,
) -> Result<Schedule, AkitaError> {
    let current_w_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root-direct witness length overflow".to_string())
    })?;
    Ok(Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len,
            witness_shape: CleartextWitnessShape::FieldElements(current_w_len),
            direct_bytes: 0,
            // Root-direct: stores the root commit layout.
            params: Some(commit_params),
        })],
        total_bytes: 0,
    })
}

/// Scale a per-polynomial root layout to a batched root layout **without**
/// the SIS-floor audit on the scaled B/D keys.
///
/// Multiplies the outer (B) and prover (D) matrix widths by `num_claims`,
/// leaving ranks, buckets, and geometry unchanged. Use this only for
/// synthetic test fixtures (or other intermediate constructions) whose
/// `(family, ring_dimension)` is intentionally outside the audited
/// SIS-floor tables. Production-facing expansion goes through the strict,
/// `try_new`-audited path inside
/// `akita_planner::generated::GeneratedFoldStep::expand_to_level_params`.
///
/// # Errors
///
/// Returns an error only on `num_claims == 0` or arithmetic overflow
/// in the scaled widths; rank/width SIS-floor mismatches do **not**
/// surface here.
pub fn scale_batched_root_layout_unchecked(
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    if num_claims == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    let d = root_lp.ring_dimension;
    let b_col_len = root_lp
        .b_key
        .col_len()
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched outer width overflow".to_string()))?;
    let d_col_len = root_lp
        .d_key
        .col_len()
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched D width overflow".to_string()))?;
    let mut scaled = root_lp.clone();
    scaled.b_key = crate::AjtaiKeyParams::new_unchecked(
        scaled.b_key.sis_family(),
        scaled.b_key.row_len(),
        b_col_len,
        scaled.b_key.collision_inf(),
        d,
    );
    scaled.d_key = crate::AjtaiKeyParams::new_unchecked(
        scaled.d_key.sis_family(),
        scaled.d_key.row_len(),
        d_col_len,
        scaled.d_key.collision_inf(),
        d,
    );
    Ok(scaled)
}

/// Return the number of fold levels in a runtime schedule.
pub fn schedule_num_fold_levels(schedule: &Schedule) -> usize {
    schedule
        .steps
        .iter()
        .filter(|step| matches!(step, Step::Fold(_)))
        .count()
}

/// Return whether a runtime schedule uses the root-direct fast path.
pub fn schedule_is_root_direct(schedule: &Schedule) -> bool {
    matches!(schedule.steps.first(), Some(Step::Direct(_)))
}

/// Return the root fold step when a runtime schedule starts with one.
pub fn schedule_root_fold_step(schedule: &Schedule) -> Option<&FoldStep> {
    match schedule.steps.first() {
        Some(Step::Fold(step)) => Some(step),
        Some(Step::Direct(_)) | None => None,
    }
}

/// Return the terminal direct witness shape from a runtime schedule.
///
/// # Errors
///
/// Returns an error if the schedule does not end in a direct witness handoff.
pub fn schedule_terminal_direct_witness_shape(
    schedule: &Schedule,
) -> Result<&CleartextWitnessShape, AkitaError> {
    match schedule.steps.last() {
        Some(Step::Direct(step)) => Ok(&step.witness_shape),
        Some(Step::Fold(_)) => Err(AkitaError::InvalidSetup(
            "schedule must end in a terminal direct witness step".to_string(),
        )),
        None => Err(AkitaError::InvalidSetup(
            "schedule is missing terminal direct witness step".to_string(),
        )),
    }
}

/// Resolve one scheduled level's active Akita params.
///
/// `Fold` steps return the baked-in `params` set by the planner DP and
/// table materializer. A terminal `Direct(PackedDigits)` step has no
/// commitment of its own (the cleartext witness is absorbed into the
/// transcript directly), so it ships no `LevelParams`; this function
/// instead returns a [`LevelParams::log_basis_stub`] carrying only the
/// active `log_basis` read off `witness_shape`. The only caller that
/// actually consumes a field of the terminal-Direct successor is the
/// prover's terminal-fold path, which reads `log_basis`.
///
/// # Errors
///
/// Returns an error when `step_index` is outside the schedule or when a
/// recursive schedule transitions into a `Direct(FieldElements)` (only
/// the *first* step of a root-direct schedule may carry that shape).
pub fn scheduled_next_level_params(
    schedule: &Schedule,
    step_index: usize,
) -> Result<LevelParams, AkitaError> {
    match schedule.steps.get(step_index) {
        Some(Step::Fold(step)) => Ok(step.params.clone()),
        Some(Step::Direct(step)) => match step.witness_shape {
            CleartextWitnessShape::PackedDigits((_, log_basis)) => {
                Ok(LevelParams::log_basis_stub(log_basis))
            }
            CleartextWitnessShape::FieldElements(_) => Err(AkitaError::InvalidSetup(
                "recursive schedule cannot transition into a field-element direct step".to_string(),
            )),
        },
        None => Err(AkitaError::InvalidSetup(
            "schedule is missing successor step".to_string(),
        )),
    }
}

/// Resolve the current fold params and successor params for a scheduled fold.
///
/// This validates that the runtime witness length and log-basis agree with the
/// selected planner schedule before deriving the next level params.
///
/// # Errors
///
/// Returns an error if `level` is not a fold step or if the runtime state does
/// not match the scheduled fold.
pub fn scheduled_fold_execution(
    schedule: &Schedule,
    level: usize,
    inputs: AkitaScheduleInputs,
    current_log_basis: u32,
) -> Result<(LevelParams, LevelParams), AkitaError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    if step.current_w_len != inputs.current_w_len || step.params.log_basis != current_log_basis {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled recursive level {level} did not match runtime state: \
             expected_w_len={}, actual_w_len={}, expected_log_basis={}, actual_log_basis={}",
            step.current_w_len, inputs.current_w_len, step.params.log_basis, current_log_basis
        )));
    }
    let next_level_params = scheduled_next_level_params(schedule, level + 1)?;
    Ok((step.params.clone(), next_level_params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EXTENSION_OPENING_REDUCTION_DEGREE;
    use crate::{extension_opening_reduction_proof_bytes, root_extension_opening_partials};
    use akita_field::FieldCore;
    use akita_field::Prime128OffsetA7F7;
    use akita_serialization::{AkitaSerialize, Compress};
    #[cfg(not(feature = "zk"))]
    use akita_sumcheck::{CompressedUniPoly, SumcheckProof};
    #[cfg(feature = "zk")]
    use akita_sumcheck::{CompressedUniPoly, SumcheckProofMasked};

    use crate::ExtensionOpeningReductionProof;

    type F = Prime128OffsetA7F7;

    #[test]
    fn root_direct_schedule_uses_field_element_payload() {
        let dummy_commit_params = LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            64,
            3,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        );
        let schedule =
            root_direct_schedule(3, dummy_commit_params.clone()).expect("root-direct schedule");
        assert_eq!(schedule.total_bytes, 0);

        let [Step::Direct(step)] = schedule.steps.as_slice() else {
            panic!("root-direct schedule should contain one direct step");
        };
        assert_eq!(step.current_w_len, 8);
        assert_eq!(step.witness_shape, CleartextWitnessShape::FieldElements(8));
        assert_eq!(step.direct_bytes, 0);
        assert_eq!(step.params.as_ref(), Some(&dummy_commit_params));
    }

    #[cfg(not(feature = "zk"))]
    fn dummy_sumcheck<F: FieldCore>(rounds: usize, degree: usize) -> SumcheckProof<F> {
        SumcheckProof {
            round_polys: (0..rounds)
                .map(|_| CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); degree],
                })
                .collect(),
        }
    }

    #[cfg(feature = "zk")]
    fn dummy_sumcheck_proof_masked<F: FieldCore>(
        rounds: usize,
        degree: usize,
    ) -> SumcheckProofMasked<F> {
        let compressed_rounds = || {
            (0..rounds)
                .map(|_| CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); degree],
                })
                .collect()
        };
        SumcheckProofMasked {
            masked_round_polys: compressed_rounds(),
        }
    }

    #[test]
    fn planned_root_extension_reduction_bytes_match_payload() {
        let extension_width = 4;
        let num_claims = 3;
        let opening_vars = 12;
        let partials = root_extension_opening_partials(extension_width, num_claims);
        let reduction = ExtensionOpeningReductionProof {
            partials: vec![F::zero(); partials],
            #[cfg(not(feature = "zk"))]
            sumcheck: dummy_sumcheck(
                opening_vars - extension_width.trailing_zeros() as usize,
                EXTENSION_OPENING_REDUCTION_DEGREE,
            ),
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: dummy_sumcheck_proof_masked(
                opening_vars - extension_width.trailing_zeros() as usize,
                EXTENSION_OPENING_REDUCTION_DEGREE,
            ),
        };
        #[cfg(not(feature = "zk"))]
        let sumcheck_bytes = reduction.sumcheck.serialized_size(Compress::No);
        #[cfg(feature = "zk")]
        let sumcheck_bytes = reduction
            .sumcheck_proof_masked
            .serialized_size(Compress::No);

        assert_eq!(
            extension_opening_reduction_proof_bytes(128, partials, opening_vars, extension_width)
                .unwrap(),
            reduction
                .partials
                .iter()
                .map(|partial| partial.serialized_size(Compress::No))
                .sum::<usize>()
                + sumcheck_bytes,
            "planned root EOR bytes should match the headerless serialized payload"
        );
    }
}
