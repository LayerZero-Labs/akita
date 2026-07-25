//! Test-only mixed ring-dimension schedule builders.
//!
//! Relocated from `akita-config`: after the runtime-resolution move (#327),
//! `akita-planner` depends on `akita-config`, so `akita-config` can no longer
//! depend on the offline planner ([`akita_planner::plan_optimal_suffix`])
//! without a dependency cycle. These builders live here (a crate above both)
//! and are gated behind the `test-support` Cargo feature, which production
//! builds never enable.
//!
//! The non-planner layout helpers (`akita_batched_root_layout`,
//! `ring_plan_test_seed`) remain in [`akita_config::test_support`].

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_config::{policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_types::sis::{OpenCommitMatrixParams, OuterCommitMatrixParams, SisTableKey};
use akita_types::{
    intermediate_w_ring_element_count_with_counts_bits, AkitaScheduleInputs,
    AkitaScheduleLookupKey, CommittedGroupParams, DecompositionParams, FoldSchedule,
    OpeningClaimsLayout, PolynomialGroupLayout, RecursiveFoldParams, RecursiveFoldStep,
    SetupMatrixEnvelope, SisModulusProfileId, TerminalFoldParams, TerminalFoldStep,
    WitnessPartition,
};
use std::marker::PhantomData;

// -------------------------------------------------------------------------
// Mixed ring-dimension-per-level schedule (runtime ring cutover experiment).
//
// Builds a schedule whose root levels `[0, switch_at_fold)` run at the
// envelope preset's ring dimension and whose recursive/terminal levels run at
// the (smaller) suffix preset's ring dimension. This is the single source of
// truth for the `mixed_d_per_level` acceptance test and the mixed-D profiler
// mode; see `specs/runtime-ring-cutover.md`.
// -------------------------------------------------------------------------

/// Build a mixed ring-dimension-per-level [`FoldSchedule`].
///
/// Fold levels `[0, switch_at_fold)` keep the `EnvelopeCfg` schedule prefix
/// (its root, and any recursive levels before the switch), at the envelope's
/// ring dimension. From `switch_at_fold` onward the schedule is a
/// **proof-size-optimal** continuation planned by [`akita_planner::plan_optimal_suffix`]
/// at `SuffixCfg`'s ring dimension, starting from the envelope prefix's output
/// witness — not the envelope's own (differently sized) suffix repriced in
/// place. This is what lets a large-ring-dimension root fold hand off to a
/// properly planned small-ring-dimension tail instead of terminating early on
/// a huge cleartext witness.
///
/// Fold level 0 is the typed root, levels 1.. are recursive entries, and the
/// last level is the typed direct terminal fold.
///
/// # Errors
///
/// Returns an error when the requested batch is not a singleton, the switch is
/// at the root, the switch skips beyond the envelope suffix, or the planner
/// cannot terminate the suffix.
pub fn mixed_d_per_level_schedule<EnvelopeCfg, SuffixCfg>(
    num_vars: usize,
    num_polynomials: usize,
    switch_at_fold: usize,
) -> Result<FoldSchedule, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    if num_polynomials != 1 || switch_at_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "mixed-D fixture requires a singleton and a non-root switch".into(),
        ));
    }
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, num_polynomials));
    let envelope = EnvelopeCfg::runtime_schedule(key)?;
    let keep_recursive = switch_at_fold - 1;
    if keep_recursive > envelope.recursive_folds.len() {
        return Err(AkitaError::InvalidSetup(
            "mixed-D switch skips beyond the recursive suffix".into(),
        ));
    }
    // Envelope prefix kept at the envelope ring dimension: root + the recursive
    // folds before the switch point.
    let recursive_prefix = envelope.recursive_folds[..keep_recursive].to_vec();
    let (prefix_output_len, prefix_lb) = match recursive_prefix.last() {
        Some(last) => (last.output_witness_len, last.params.witness.log_basis_open),
        None => (
            envelope.root.output_witness_len,
            envelope.root.params.final_group.commitment.log_basis_open,
        ),
    };

    // Optimal small-ring-dimension continuation from the prefix output.
    let suffix = akita_planner::plan_optimal_suffix(
        &policy_of::<SuffixCfg>(),
        SuffixCfg::ring_challenge_config,
        SuffixCfg::fold_challenge_shape_at_level,
        num_vars,
        switch_at_fold,
        prefix_output_len,
        prefix_lb,
    )?;

    let mut recursive_folds = recursive_prefix;
    for fold in &suffix.folds {
        recursive_folds.push(RecursiveFoldStep {
            params: RecursiveFoldParams {
                open_commit_matrix: fold.params.open_commit_matrix.clone(),
                sparse_challenge_config: fold.params.fold_challenge_config,
                witness: fold.params.clone(),
                incoming_setup_prefix: None,
                witness_partition: WitnessPartition::Single,
            },
            input_witness_len: fold.input_witness_len,
            output_witness_len: fold.output_witness_len,
        });
    }

    let schedule = FoldSchedule {
        root: envelope.root,
        recursive_folds,
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: suffix.terminal.params,
                sparse_challenge_config: suffix.terminal.sparse_challenge_config,
                response_shape: suffix.terminal.response_shape,
            },
            input_witness_len: suffix.terminal.input_witness_len,
        },
    };
    schedule.validate_structure()?;
    let opening_batch = OpeningClaimsLayout::new(num_vars, 1)?;
    schedule
        .root
        .params
        .final_group
        .commitment
        .validate_opening_batch(&opening_batch)?;
    Ok(schedule)
}

/// Config adapter that opens through a mixed ring-dimension-per-level schedule:
/// levels `[0, SWITCH_AT_FOLD)` at `Env`'s ring dimension, later levels at
/// `Suffix`'s ring dimension. Delegates every policy hook to `Env` (so
/// `Env::D` sets the setup's `gen_ring_dim`) and overrides schedule resolution
/// to build the mixed schedule via [`mixed_d_per_level_schedule`].
#[derive(Debug)]
pub struct MixedDConfig<Env, Suffix, const SWITCH_AT_FOLD: usize>(
    PhantomData<fn() -> (Env, Suffix)>,
);

impl<Env, Suffix, const SWITCH_AT_FOLD: usize> Clone for MixedDConfig<Env, Suffix, SWITCH_AT_FOLD> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Env, Suffix, const SWITCH_AT_FOLD: usize> Copy for MixedDConfig<Env, Suffix, SWITCH_AT_FOLD> {}

impl<Env, Suffix, const SWITCH_AT_FOLD: usize> Default
    for MixedDConfig<Env, Suffix, SWITCH_AT_FOLD>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Env, Suffix, const SWITCH_AT_FOLD: usize> CommitmentConfig
    for MixedDConfig<Env, Suffix, SWITCH_AT_FOLD>
where
    Env: CommitmentConfig,
    Suffix: CommitmentConfig<Field = Env::Field, ExtField = Env::ExtField>,
{
    type Field = Env::Field;
    type ExtField = Env::ExtField;

    const D: usize = Env::D;

    fn decomposition() -> DecompositionParams {
        Env::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Env::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Env::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Env::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Env::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        Env::max_setup_matrix_size(max_num_vars, max_num_batched_polys)
    }

    fn basis_range() -> (u32, u32) {
        Env::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Env::onehot_chunk_size()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Env::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        mixed_d_per_level_schedule::<Env, Suffix>(
            key.final_group.num_vars(),
            key.final_group.num_polynomials(),
            SWITCH_AT_FOLD,
        )
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        mixed_d_per_level_schedule::<Env, Suffix>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            SWITCH_AT_FOLD,
        )
    }
}

// -------------------------------------------------------------------------
// Per-role (commitment-compression) mixed dims: A at the envelope ring
// dimension, B/D compressed to `compressed_d`. This is the within-level
// non-uniform axis (`d_d | d_b | d_a`), distinct from the per-level switch
// above. See specs/mixed-ring-dimension-per-level.md.
// -------------------------------------------------------------------------

/// Rebuild the root's outer (B) commit matrix at `outer_d` and open (D) commit
/// matrix at `open_d`, keeping the inner (A) matrix at the envelope dimension.
/// A role whose target equals the envelope dimension is left untouched, so
/// `(outer_d, open_d) = (128, 64)` compresses only the D role.
///
/// Halving a role's ring dimension doubles its matrix input width (same total
/// coefficients, half-size ring elements); the SIS output rank is re-derived
/// from the audited table at the new dimension. Mirrors the retarget recipe
/// from the (pre-schedule-merge) `mixed_role_e2e` fixture, adapted to the
/// current `FoldSchedule` API.
///
/// # Errors
///
/// Returns an error when `outer_d`/`open_d` do not divide the envelope ring
/// dimension, or when a compressed width falls outside the audited SIS table.
pub fn compressed_role_root_schedule<Env: CommitmentConfig>(
    num_vars: usize,
    num_polynomials: usize,
    outer_d: usize,
    open_d: usize,
) -> Result<FoldSchedule, AkitaError> {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, num_polynomials));
    let mut schedule = Env::runtime_schedule(key)?;
    let root = &mut schedule.root.params.final_group.commitment;
    let inner_d = root.d_a();
    for (label, target_d) in [("B", outer_d), ("D", open_d)] {
        if target_d == 0 || !inner_d.is_multiple_of(target_d) {
            return Err(AkitaError::InvalidSetup(format!(
                "compressed {label} dim {target_d} must divide the root inner dim {inner_d}"
            )));
        }
    }

    // Retarget only the roles that actually shrink (target == inner is a no-op).
    if outer_d != root.outer_commit_matrix.ring_dimension() {
        let column_scale = root.outer_commit_matrix.ring_dimension() / outer_d;
        let outer_key = SisTableKey {
            ring_dimension: outer_d as u32,
            ..root.outer_commit_matrix.sis_table_key()
        };
        root.outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
            outer_key,
            root.outer_commit_matrix.input_width() * column_scale,
        )?;
    }
    if open_d != root.open_commit_matrix.ring_dimension() {
        let column_scale = root.open_commit_matrix.ring_dimension() / open_d;
        let open_key = SisTableKey {
            ring_dimension: open_d as u32,
            ..root.open_commit_matrix.sis_table_key()
        };
        root.open_commit_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
            open_key,
            root.open_commit_matrix.input_width() * column_scale,
        )?;
    }

    // Recompute the root's outgoing witness length from the compressed root
    // params and stitch it into the (unchanged, uniform) successor level.
    let field_bits = Env::decomposition().field_bits();
    let next_inner_d = schedule.recursive_folds.first().map_or_else(
        || schedule.terminal.params.witness.d_a(),
        |next| next.params.witness.d_a(),
    );
    let next_w_len = intermediate_w_ring_element_count_with_counts_bits(
        field_bits,
        &schedule.root.params.final_group.commitment,
        num_polynomials,
        1,
    )?
    .checked_mul(next_inner_d)
    .ok_or_else(|| AkitaError::InvalidSetup("compressed-root witness length overflow".into()))?;
    schedule.root.output_witness_len = next_w_len;
    if let Some(next) = schedule.recursive_folds.first_mut() {
        next.input_witness_len = next_w_len;
    } else {
        schedule.terminal.input_witness_len = next_w_len;
    }

    schedule.validate_structure()?;
    let opening_batch = OpeningClaimsLayout::new(num_vars, num_polynomials)?;
    schedule
        .root
        .params
        .final_group
        .commitment
        .validate_opening_batch(&opening_batch)?;
    Ok(schedule)
}

/// Retarget one `Open` (D-role) commit matrix to `target_d`, widening its input
/// so the total flat coefficient count is preserved. No-op when already at
/// `target_d`.
fn retarget_open_matrix(
    open: &mut OpenCommitMatrixParams,
    target_d: usize,
) -> Result<(), AkitaError> {
    let current = open.ring_dimension();
    if current == target_d {
        return Ok(());
    }
    if target_d == 0 || !current.is_multiple_of(target_d) {
        return Err(AkitaError::InvalidSetup(format!(
            "compressed open dim {target_d} must divide current dim {current}"
        )));
    }
    let column_scale = current / target_d;
    let key = SisTableKey {
        ring_dimension: target_d as u32,
        ..open.sis_table_key()
    };
    *open = OpenCommitMatrixParams::try_new_with_min_rank(key, open.input_width() * column_scale)?;
    Ok(())
}

/// Retarget one `Outer` (B-role) commit matrix to `target_d`.
fn retarget_outer_matrix(
    outer: &mut OuterCommitMatrixParams,
    target_d: usize,
) -> Result<(), AkitaError> {
    let current = outer.ring_dimension();
    if current == target_d {
        return Ok(());
    }
    if target_d == 0 || !current.is_multiple_of(target_d) {
        return Err(AkitaError::InvalidSetup(format!(
            "compressed outer dim {target_d} must divide current dim {current}"
        )));
    }
    let column_scale = current / target_d;
    let key = SisTableKey {
        ring_dimension: target_d as u32,
        ..outer.sis_table_key()
    };
    *outer =
        OuterCommitMatrixParams::try_new_with_min_rank(key, outer.input_width() * column_scale)?;
    Ok(())
}

/// Field-element outgoing witness length for a level: outgoing ring-element
/// count (from the level's commitment) times the successor level's inner ring.
fn outgoing_witness_len(
    field_bits: u32,
    commitment: &CommittedGroupParams,
    num_polynomials: usize,
    next_inner_d: usize,
) -> Result<usize, AkitaError> {
    intermediate_w_ring_element_count_with_counts_bits(field_bits, commitment, num_polynomials, 1)?
        .checked_mul(next_inner_d)
        .ok_or_else(|| AkitaError::InvalidSetup("role-switch witness length overflow".into()))
}

/// Multi-level per-role role-switch schedule:
///
/// - L0 (root): `A = Env::D`, `B = Env::D`, `D = mid_d` (compress D only).
/// - L1: `A = Env::D`, `B = mid_d`, `D = mid_d` (compress B and D).
/// - L2+: uniform `mid_d` (the `Suffix` band).
///
/// Built on a `switch = 2` mixed base (`A = Env::D` for L0/L1, `mid_d` after),
/// then the per-role roles are compressed within L0/L1 and the outgoing witness
/// lengths re-stitched across L0 → L1 → L2.
///
/// # Errors
///
/// Returns an error when `mid_d` does not divide the envelope dimension, a
/// compressed width leaves the audited SIS table, or the schedule fails
/// structural validation.
pub fn role_switch_schedule<Env, Suffix>(
    num_vars: usize,
    num_polynomials: usize,
    mid_d: usize,
    root_open_d: usize,
) -> Result<FoldSchedule, AkitaError>
where
    Env: CommitmentConfig,
    Suffix: CommitmentConfig<Field = Env::Field, ExtField = Env::ExtField>,
{
    let mut schedule = mixed_d_per_level_schedule::<Env, Suffix>(num_vars, num_polynomials, 2)?;
    if schedule.recursive_folds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "role-switch schedule needs at least one recursive fold (L1)".into(),
        ));
    }
    let field_bits = Env::decomposition().field_bits();

    // L0 (root): compress the D (open) role to `root_open_d` (== `Env::D` leaves
    // the root fully uniform); A and B stay at `Env::D`.
    retarget_open_matrix(
        &mut schedule
            .root
            .params
            .final_group
            .commitment
            .open_commit_matrix,
        root_open_d,
    )?;

    // L1 (first recursive fold): compress B (outer) and D (open); A stays.
    {
        let l1 = &mut schedule.recursive_folds[0].params;
        retarget_outer_matrix(&mut l1.witness.outer_commit_matrix, mid_d)?;
        retarget_open_matrix(&mut l1.witness.open_commit_matrix, mid_d)?;
        // The fold-shared opening matrix mirrors the witness D role.
        retarget_open_matrix(&mut l1.open_commit_matrix, mid_d)?;
    }

    // Re-stitch outgoing witness lengths L0 -> L1 -> L2. The folded witness
    // lives in the *producing* level's inner ring (the successor re-groups it),
    // so the field length is `outgoing ring-element count × producing inner d`.
    let root_inner_d = schedule.root.params.final_group.commitment.d_a();
    let root_out = outgoing_witness_len(
        field_bits,
        &schedule.root.params.final_group.commitment,
        num_polynomials,
        root_inner_d,
    )?;
    schedule.root.output_witness_len = root_out;
    schedule.recursive_folds[0].input_witness_len = root_out;

    let l1_inner_d = schedule.recursive_folds[0].params.witness.d_a();
    let l1_out = outgoing_witness_len(
        field_bits,
        &schedule.recursive_folds[0].params.witness,
        num_polynomials,
        l1_inner_d,
    )?;
    schedule.recursive_folds[0].output_witness_len = l1_out;
    if let Some(next) = schedule.recursive_folds.get_mut(1) {
        next.input_witness_len = l1_out;
    } else {
        schedule.terminal.input_witness_len = l1_out;
    }

    schedule.validate_structure()?;
    let opening_batch = OpeningClaimsLayout::new(num_vars, num_polynomials)?;
    schedule
        .root
        .params
        .final_group
        .commitment
        .validate_opening_batch(&opening_batch)?;
    Ok(schedule)
}

/// Config adapter for [`role_switch_schedule`]: L0 `A=B=Env::D, D=ROOT_OPEN_D`,
/// L1 `A=Env::D, B=D=MID_D`, then uniform `MID_D`. `ROOT_OPEN_D == Env::D`
/// leaves the root fully uniform. Delegates policy to `Env`.
#[derive(Debug)]
pub struct RoleSwitchConfig<Env, Suffix, const MID_D: usize, const ROOT_OPEN_D: usize>(
    PhantomData<fn() -> (Env, Suffix)>,
);

impl<Env, Suffix, const MID_D: usize, const ROOT_OPEN_D: usize> Clone
    for RoleSwitchConfig<Env, Suffix, MID_D, ROOT_OPEN_D>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Env, Suffix, const MID_D: usize, const ROOT_OPEN_D: usize> Copy
    for RoleSwitchConfig<Env, Suffix, MID_D, ROOT_OPEN_D>
{
}

impl<Env, Suffix, const MID_D: usize, const ROOT_OPEN_D: usize> Default
    for RoleSwitchConfig<Env, Suffix, MID_D, ROOT_OPEN_D>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Env, Suffix, const MID_D: usize, const ROOT_OPEN_D: usize> CommitmentConfig
    for RoleSwitchConfig<Env, Suffix, MID_D, ROOT_OPEN_D>
where
    Env: CommitmentConfig,
    Suffix: CommitmentConfig<Field = Env::Field, ExtField = Env::ExtField>,
{
    type Field = Env::Field;
    type ExtField = Env::ExtField;

    const D: usize = Env::D;

    fn decomposition() -> DecompositionParams {
        Env::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Env::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Env::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Env::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Env::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        let mut max_setup_len = 1usize;
        for num_polys in 1..=max_num_batched_polys.max(1) {
            let schedule =
                role_switch_schedule::<Env, Suffix>(max_num_vars, num_polys, MID_D, ROOT_OPEN_D)?;
            akita_types::accumulate_matrix_envelope_for_level(
                &schedule.root.params.final_group.commitment,
                &mut max_setup_len,
            )?;
            for step in &schedule.recursive_folds {
                akita_types::accumulate_matrix_envelope_for_level(
                    &step.params.witness,
                    &mut max_setup_len,
                )?;
            }
            akita_types::accumulate_terminal_matrix_envelope(
                &schedule.terminal.params.witness,
                &mut max_setup_len,
            )?;
        }
        Ok(SetupMatrixEnvelope { max_setup_len })
    }

    fn basis_range() -> (u32, u32) {
        Env::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Env::onehot_chunk_size()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Env::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        role_switch_schedule::<Env, Suffix>(
            key.final_group.num_vars(),
            key.final_group.num_polynomials(),
            MID_D,
            ROOT_OPEN_D,
        )
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        role_switch_schedule::<Env, Suffix>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            MID_D,
            ROOT_OPEN_D,
        )
    }
}

/// Push one planned suffix fold onto `recursive_folds` (mirrors the
/// `mixed_d_per_level_schedule` conversion).
fn push_planned_fold(
    recursive_folds: &mut Vec<RecursiveFoldStep>,
    fold: &akita_planner::PlannedSuffixFold,
) {
    recursive_folds.push(RecursiveFoldStep {
        params: RecursiveFoldParams {
            open_commit_matrix: fold.params.open_commit_matrix.clone(),
            sparse_challenge_config: fold.params.fold_challenge_config,
            witness: fold.params.clone(),
            incoming_setup_prefix: None,
            witness_partition: WitnessPartition::Single,
        },
        input_witness_len: fold.input_witness_len,
        output_witness_len: fold.output_witness_len,
    });
}

/// Three-band descending role-switch schedule:
///
/// - L0 (root): `A = Root::D`, `B = D = root_bd` (compressed from `Root::D`).
/// - L1: `A = Mid::D`, `B = D = l1_bd` (compressed from `Mid::D`).
/// - L2+: uniform `Suffix::D`.
///
/// The A-band descends `Root::D → Mid::D → Suffix::D` (e.g. 256 → 128 → 64) via
/// two [`akita_planner::plan_optimal_suffix`] passes, and B/D are compressed
/// within L0 and L1. Requires a singleton batch.
///
/// # Errors
///
/// Returns an error when the batch is not a singleton, either planner pass
/// fails, a compressed width leaves the audited SIS table, or the schedule
/// fails structural validation.
pub fn three_band_role_switch_schedule<Root, Mid, Suffix>(
    num_vars: usize,
    num_polynomials: usize,
    root_bd: usize,
    l1_bd: usize,
) -> Result<FoldSchedule, AkitaError>
where
    Root: CommitmentConfig,
    Mid: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
    Suffix: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
{
    if num_polynomials != 1 {
        return Err(AkitaError::InvalidSetup(
            "three-band role-switch requires a singleton batch".into(),
        ));
    }
    let field_bits = Root::decomposition().field_bits();
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, num_polynomials));

    // L0: root from the `Root` envelope, then compress its B/D to `root_bd`.
    let mut root = Root::runtime_schedule(key)?.root;
    retarget_outer_matrix(
        &mut root.params.final_group.commitment.outer_commit_matrix,
        root_bd,
    )?;
    retarget_open_matrix(
        &mut root.params.final_group.commitment.open_commit_matrix,
        root_bd,
    )?;
    let root_inner_d = root.params.final_group.commitment.d_a();
    let root_out = outgoing_witness_len(
        field_bits,
        &root.params.final_group.commitment,
        num_polynomials,
        root_inner_d,
    )?;
    root.output_witness_len = root_out;
    let root_lb = root.params.final_group.commitment.log_basis_open;

    // Band 2 (`Mid`): plan an optimal `Mid`-dim continuation from the root
    // output and keep only its first fold as L1.
    let mid = akita_planner::plan_optimal_suffix(
        &policy_of::<Mid>(),
        Mid::ring_challenge_config,
        Mid::fold_challenge_shape_at_level,
        num_vars,
        1,
        root_out,
        root_lb,
    )?;
    let l1 = mid.folds.first().ok_or_else(|| {
        AkitaError::InvalidSetup("three-band role-switch: Mid band produced no fold".into())
    })?;

    let mut recursive_folds: Vec<RecursiveFoldStep> = Vec::new();
    push_planned_fold(&mut recursive_folds, l1);
    // Compress L1's B/D to `l1_bd`; A stays at `Mid::D`.
    {
        let l1_params = &mut recursive_folds[0].params;
        retarget_outer_matrix(&mut l1_params.witness.outer_commit_matrix, l1_bd)?;
        retarget_open_matrix(&mut l1_params.witness.open_commit_matrix, l1_bd)?;
        retarget_open_matrix(&mut l1_params.open_commit_matrix, l1_bd)?;
    }
    let l1_inner_d = recursive_folds[0].params.witness.d_a();
    let l1_out = outgoing_witness_len(
        field_bits,
        &recursive_folds[0].params.witness,
        num_polynomials,
        l1_inner_d,
    )?;
    recursive_folds[0].output_witness_len = l1_out;
    let l1_lb = recursive_folds[0].params.witness.log_basis_open;

    // Band 3 (`Suffix`): optimal small-ring continuation from L1's output.
    let suffix = akita_planner::plan_optimal_suffix(
        &policy_of::<Suffix>(),
        Suffix::ring_challenge_config,
        Suffix::fold_challenge_shape_at_level,
        num_vars,
        2,
        l1_out,
        l1_lb,
    )?;
    for fold in &suffix.folds {
        push_planned_fold(&mut recursive_folds, fold);
    }
    if let Some(next) = recursive_folds.get_mut(1) {
        next.input_witness_len = l1_out;
    }

    let schedule = FoldSchedule {
        root,
        recursive_folds,
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: suffix.terminal.params,
                sparse_challenge_config: suffix.terminal.sparse_challenge_config,
                response_shape: suffix.terminal.response_shape,
            },
            input_witness_len: suffix.terminal.input_witness_len,
        },
    };
    schedule.validate_structure()?;
    let opening_batch = OpeningClaimsLayout::new(num_vars, num_polynomials)?;
    schedule
        .root
        .params
        .final_group
        .commitment
        .validate_opening_batch(&opening_batch)?;
    Ok(schedule)
}

/// Config adapter for [`three_band_role_switch_schedule`]. `Root/Mid/Suffix`
/// set the A-band dims (e.g. 256/128/64); `ROOT_BD`/`L1_BD` compress B/D at
/// L0/L1. `Root::D` sets the setup generation ring dimension.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct ThreeBandRoleSwitchConfig<Root, Mid, Suffix, const ROOT_BD: usize, const L1_BD: usize>(
    PhantomData<fn() -> (Root, Mid, Suffix)>,
);

impl<Root, Mid, Suffix, const ROOT_BD: usize, const L1_BD: usize> Clone
    for ThreeBandRoleSwitchConfig<Root, Mid, Suffix, ROOT_BD, L1_BD>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Root, Mid, Suffix, const ROOT_BD: usize, const L1_BD: usize> Copy
    for ThreeBandRoleSwitchConfig<Root, Mid, Suffix, ROOT_BD, L1_BD>
{
}

impl<Root, Mid, Suffix, const ROOT_BD: usize, const L1_BD: usize> Default
    for ThreeBandRoleSwitchConfig<Root, Mid, Suffix, ROOT_BD, L1_BD>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Root, Mid, Suffix, const ROOT_BD: usize, const L1_BD: usize> CommitmentConfig
    for ThreeBandRoleSwitchConfig<Root, Mid, Suffix, ROOT_BD, L1_BD>
where
    Root: CommitmentConfig,
    Mid: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
    Suffix: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
{
    type Field = Root::Field;
    type ExtField = Root::ExtField;

    const D: usize = Root::D;

    fn decomposition() -> DecompositionParams {
        Root::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Root::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Root::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Root::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Root::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        let mut max_setup_len = 1usize;
        for num_polys in 1..=max_num_batched_polys.max(1) {
            let schedule = three_band_role_switch_schedule::<Root, Mid, Suffix>(
                max_num_vars,
                num_polys,
                ROOT_BD,
                L1_BD,
            )?;
            akita_types::accumulate_matrix_envelope_for_level(
                &schedule.root.params.final_group.commitment,
                &mut max_setup_len,
            )?;
            for step in &schedule.recursive_folds {
                akita_types::accumulate_matrix_envelope_for_level(
                    &step.params.witness,
                    &mut max_setup_len,
                )?;
            }
            akita_types::accumulate_terminal_matrix_envelope(
                &schedule.terminal.params.witness,
                &mut max_setup_len,
            )?;
        }
        Ok(SetupMatrixEnvelope { max_setup_len })
    }

    fn basis_range() -> (u32, u32) {
        Root::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Root::onehot_chunk_size()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Root::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        three_band_role_switch_schedule::<Root, Mid, Suffix>(
            key.final_group.num_vars(),
            key.final_group.num_polynomials(),
            ROOT_BD,
            L1_BD,
        )
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        three_band_role_switch_schedule::<Root, Mid, Suffix>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            ROOT_BD,
            L1_BD,
        )
    }
}

/// Config adapter opening through a per-role root (A = `Env::D`, B = `OUTER_D`,
/// D = `OPEN_D`). Delegates every policy hook to `Env`.
#[derive(Debug)]
pub struct CompressedRoleRootConfig<Env, const OUTER_D: usize, const OPEN_D: usize>(
    PhantomData<fn() -> Env>,
);

impl<Env, const OUTER_D: usize, const OPEN_D: usize> Clone
    for CompressedRoleRootConfig<Env, OUTER_D, OPEN_D>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Env, const OUTER_D: usize, const OPEN_D: usize> Copy
    for CompressedRoleRootConfig<Env, OUTER_D, OPEN_D>
{
}

impl<Env, const OUTER_D: usize, const OPEN_D: usize> Default
    for CompressedRoleRootConfig<Env, OUTER_D, OPEN_D>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Env, const OUTER_D: usize, const OPEN_D: usize> CommitmentConfig
    for CompressedRoleRootConfig<Env, OUTER_D, OPEN_D>
where
    Env: CommitmentConfig,
{
    type Field = Env::Field;
    type ExtField = Env::ExtField;

    const D: usize = Env::D;

    fn decomposition() -> DecompositionParams {
        Env::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Env::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Env::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Env::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Env::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        // Compressing B/D to a smaller ring widens those matrices, so the setup
        // envelope must be sized from the actual compressed schedule (not the
        // uniform envelope). Accumulate every level's A/B/D footprint.
        let mut max_setup_len = 1usize;
        for num_polys in 1..=max_num_batched_polys.max(1) {
            let schedule =
                compressed_role_root_schedule::<Env>(max_num_vars, num_polys, OUTER_D, OPEN_D)?;
            akita_types::accumulate_matrix_envelope_for_level(
                &schedule.root.params.final_group.commitment,
                &mut max_setup_len,
            )?;
            for step in &schedule.recursive_folds {
                akita_types::accumulate_matrix_envelope_for_level(
                    &step.params.witness,
                    &mut max_setup_len,
                )?;
            }
            akita_types::accumulate_terminal_matrix_envelope(
                &schedule.terminal.params.witness,
                &mut max_setup_len,
            )?;
        }
        Ok(SetupMatrixEnvelope { max_setup_len })
    }

    fn basis_range() -> (u32, u32) {
        Env::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Env::onehot_chunk_size()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Env::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        compressed_role_root_schedule::<Env>(
            key.final_group.num_vars(),
            key.final_group.num_polynomials(),
            OUTER_D,
            OPEN_D,
        )
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        compressed_role_root_schedule::<Env>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            OUTER_D,
            OPEN_D,
        )
    }
}
