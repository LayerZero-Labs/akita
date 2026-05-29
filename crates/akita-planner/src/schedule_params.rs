//! Schedule planner that finds the global minimum proof size.
//!
//! Public entry: [`find_schedule`], `<Cfg>`-generic. When `use_lookup` is
//! true the search consults `Cfg::schedule_table()` before running DP;
//! production callers pass `true`. The table-emitter binary passes
//! `false` to regenerate from scratch.

use std::collections::{BTreeMap, HashMap};

use akita_challenges::TensorChallengeShape;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::GeneratedScheduleTable;
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, root_extension_opening_partials,
    schedule_from_plan, w_ring_element_count_with_counts_for_layout_bits, AjtaiKeyParams,
    AkitaScheduleInputs, AkitaScheduleLookupKey, DirectStep, DirectWitnessShape, FoldStep,
    LevelParams, MRowLayout, Schedule, Step,
};

use akita_derive::{schedule_plan_from_table, PlanPolicy};

use crate::ajtai_params::{compute_all_ajtai_keys_params, WitnessType};
use crate::proof_size::level_proof_bytes;

// Suffix-DP depth cap. Schedules in our working parameter range never need
// more than this many recursive fold levels; deeper search only blows up
// memo state without changing emitted tables.
const MAX_RECURSION_DEPTH: usize = 12;

/// One recursive fold candidate: its level params plus the next-level
/// witness lengths under the Intermediate (`.1`) and Terminal (`.2`)
/// M-row layouts.
type CandidateLayout = (LevelParams, usize, usize);

/// Derive the recursive fold layout at `(current_w_len, log_basis)`,
/// or `None` if no split is SIS-feasible or the witness doesn't shrink.
///
/// Recursive-only: the root candidate path in `find_schedule` enumerates
/// `(num_blocks, log_basis)` explicitly, but here we pick the `(m, r)`
/// split that minimizes the actual next-level (Intermediate) witness
/// length, building one full candidate per split. We do NOT recurse into
/// the suffix DP per split — that would make every parent's `next_w_len`
/// a fresh memo key and blow up the DP state space.
fn derive_candidate_level_params<Cfg: CommitmentConfig>(
    current_w_len: usize,
    log_basis: u32,
) -> Result<Option<CandidateLayout>, AkitaError> {
    let Ok(stage1_config) = Cfg::stage1_challenge_config(Cfg::D) else {
        return Ok(None);
    };
    if !current_w_len.is_multiple_of(Cfg::D) {
        return Ok(None);
    }
    let num_ring_elems = current_w_len / Cfg::D;
    let total = num_ring_elems.next_power_of_two().max(1);
    let reduced_vars = total.trailing_zeros() as usize;

    let field_bits = Cfg::decomposition().field_bits();
    let num_digits_commit = WitnessType::S.decomposed_num_digits::<Cfg>(log_basis, false);
    let num_digits_open = WitnessType::T.decomposed_num_digits::<Cfg>(log_basis, false);
    let l1_mass = TensorChallengeShape::Flat.effective_l1_mass(&stage1_config);

    if reduced_vars <= 2 || reduced_vars >= 53 {
        return Err(AkitaError::InvalidSetup(format!(
            "recursive fold candidate reduced_vars={reduced_vars} is outside \
             the optimizable range [3, 52]"
        )));
    }

    let mut best: Option<CandidateLayout> = None;
    for r in 1..reduced_vars {
        let num_blocks = 1usize << r;
        let block_len = num_ring_elems.div_ceil(num_blocks);

        let Some((a_key, b_key, d_key)) =
            compute_all_ajtai_keys_params::<Cfg>(block_len, num_blocks, 1, log_basis, false)?
        else {
            continue;
        };

        let level_lp = LevelParams {
            ring_dimension: Cfg::D,
            log_basis,
            a_key,
            b_key,
            d_key,
            num_blocks,
            block_len,
            m_vars: reduced_vars - r,
            r_vars: r,
            stage1_config: stage1_config.clone(),
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_commit,
            num_digits_open,
            num_digits_fold: WitnessType::Z
                .decomposed_fold_num_digits::<Cfg>(log_basis, r, l1_mass, 1),
        };

        // Recursive folds carry a single witness opened at a single point
        // (all batching is absorbed into level 0), so the multiplicities
        // are all 1. Intermediate keeps the D-block (and zk D-blinding);
        // Terminal drops both.
        let next_w_len = w_ring_element_count_with_counts_for_layout_bits(
            field_bits,
            &level_lp,
            1,
            1,
            1,
            1,
            MRowLayout::Intermediate,
        )?
        .checked_mul(Cfg::D)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness length overflow".into()))?;
        let next_w_len_terminal = w_ring_element_count_with_counts_for_layout_bits(
            field_bits,
            &level_lp,
            1,
            1,
            1,
            1,
            MRowLayout::Terminal,
        )?
        .checked_mul(Cfg::D)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness length overflow".into()))?;

        if best.as_ref().is_none_or(|(_, c, _)| next_w_len < *c) {
            best = Some((level_lp, next_w_len, next_w_len_terminal));
        }
    }

    let Some((level_lp, next_w_len, next_w_len_terminal)) = best else {
        return Ok(None);
    };

    // Strict shrink check on the Intermediate shape (the witness arrives
    // in `log_basis`-bit digits and is re-emitted at the same basis, so it
    // reduces to "ring-element count strictly decreases", kept in bit form
    // to mirror the root's check and surface overflow). We deliberately do
    // NOT gate on the terminal shape: a candidate whose Intermediate shape
    // shrinks while its Terminal shape does not is still useful for
    // fold-then-...-then-direct chains.
    let next_bits = next_w_len
        .checked_mul(log_basis as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("next witness bit length overflow".into()))?;
    let current_bits = current_w_len
        .checked_mul(log_basis as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("current witness bit length overflow".into()))?;
    if next_bits >= current_bits {
        return Ok(None);
    }

    Ok(Some((level_lp, next_w_len, next_w_len_terminal)))
}

/// Unified proof-size formula for one fold level under either M-row
/// layout. `next_lp` is consulted only when `layout =
/// MRowLayout::Intermediate`; terminal callers may pass any
/// `&LevelParams` (typically `lp`) as a placeholder.
fn compute_level_proof_size<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
    num_public_outputs: usize,
    layout: MRowLayout,
) -> usize {
    let field_bits = Cfg::decomposition().field_bits();
    let challenge_bits = field_bits * Cfg::CHAL_EXT_DEGREE as u32;
    level_proof_bytes(
        field_bits,
        challenge_bits,
        lp,
        next_lp,
        next_w_len,
        num_public_outputs,
        layout,
    )
}

fn padded_boolean_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

fn extension_opening_reduction_level_bytes<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    fold_level: usize,
    current_w_len: usize,
) -> Result<usize, AkitaError> {
    let width = Cfg::CLAIM_EXT_DEGREE;
    if width <= 1 {
        return Ok(0);
    }
    let (partials, opening_vars) = if fold_level == 0 {
        (
            root_extension_opening_partials(width, key.num_w_vectors),
            key.num_vars,
        )
    } else {
        (width, padded_boolean_vars(current_w_len)?)
    };
    extension_opening_reduction_proof_bytes(
        Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
        partials,
        opening_vars,
        width,
    )
}

/// Result of the suffix DP at one state. Both shape options are reported
/// because the parent's proof-size formula depends on the child's first
/// step:
///
/// - `best_direct` — best schedule whose first step is a `Step::Direct`
///   (parent scores under `MRowLayout::Terminal`). `None` when infeasible.
/// - `best_fold_per_lb` — best `Step::Fold`-first schedule per first-fold
///   `log_basis`. Keyed by `log_basis` because the parent's intermediate
///   proof formula uses `next_lp.b_key.row_len`, which depends on the
///   first fold's layout; a single "best fold" min would hide that and
///   force a locally-cheap child that balloons the parent's proof. Empty
///   when no fold continuation is valid.
#[derive(Clone)]
struct SuffixResult {
    best_direct: Option<(usize, Vec<Step>)>,
    best_fold_per_lb: BTreeMap<u32, (usize, Vec<Step>)>,
}

impl SuffixResult {
    fn is_empty(&self) -> bool {
        self.best_direct.is_none() && self.best_fold_per_lb.is_empty()
    }
}

type ScheduleMemo = HashMap<(usize, usize, usize, u32), SuffixResult>;

/// Suffix DP for the optimal recursive schedule at
/// `(level, current_w_len, current_w_len_terminal, current_lb)`.
///
/// Two witness lengths are carried because the shape leaving a fold
/// depends on its successor: `current_w_len` is the `Intermediate` shape
/// (used if level `L` folds again) and `current_w_len_terminal` is the
/// `Terminal` shape (used if level `L` sends the witness directly — drops
/// the D-block and zk D-blinding, so it is `<= current_w_len`).
///
/// At each state: `best_direct` ships the witness directly (Terminal, no
/// SIS audit, always present); `best_fold` keeps one fold candidate per
/// `log_basis` (from [`derive_candidate_level_params`]) and scores both
/// child-shape options (`compute_level_proof_size` under Terminal if the
/// child is Direct, Intermediate if it folds), keeping the cheaper. One
/// candidate per `log_basis` is required for memo dedup — explicit
/// `(num_blocks, log_basis)` enumeration would give each parent fold a
/// fresh `next_w_len` memo key and blow up the state space.
fn derive_optimal_suffix_schedule<Cfg: CommitmentConfig>(
    memo: &mut ScheduleMemo,
    num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_w_len_terminal: usize,
    current_lb: u32,
    depth: usize,
) -> Result<SuffixResult, AkitaError> {
    let memo_key = (level, current_w_len, current_w_len_terminal, current_lb);
    if depth <= MAX_RECURSION_DEPTH {
        if let Some(cached) = memo.get(&memo_key) {
            return Ok(cached.clone());
        }
    }

    let best_direct = {
        let witness_shape = DirectWitnessShape::PackedDigits((current_w_len_terminal, current_lb));
        let direct_bytes = direct_witness_bytes(Cfg::decomposition().field_bits(), &witness_shape);
        let step = Step::Direct(DirectStep {
            current_w_len: current_w_len_terminal,
            witness_shape,
            direct_bytes,
            params: None,
        });
        Some((direct_bytes, vec![step]))
    };

    if depth > MAX_RECURSION_DEPTH {
        let result = SuffixResult {
            best_direct,
            best_fold_per_lb: BTreeMap::new(),
        };
        memo.insert(memo_key, result.clone());
        return Ok(result);
    }

    let mut best_fold_per_lb: BTreeMap<u32, (usize, Vec<Step>)> = BTreeMap::new();
    let (min_log_basis, max_log_basis) = Cfg::basis_range();
    for lb in min_log_basis..=max_log_basis {
        if lb < current_lb {
            continue;
        }
        let Some((lp, next_w_len, next_w_len_terminal)) =
            derive_candidate_level_params::<Cfg>(current_w_len, lb)?
        else {
            continue;
        };

        let child = derive_optimal_suffix_schedule::<Cfg>(
            memo,
            num_vars,
            level + 1,
            next_w_len,
            next_w_len_terminal,
            lb,
            depth + 1,
        )?;
        let Ok(eor_bytes) = extension_opening_reduction_level_bytes::<Cfg>(
            AkitaScheduleLookupKey::singleton(num_vars),
            level,
            current_w_len,
        ) else {
            continue;
        };

        // Best schedule with a Fold-at-this-level of `lb` as its first
        // step. Considers every grandchild-shape option (child=Direct
        // forces the terminal proof formula at this level; child=Fold
        // for each grandchild `first_lb` forces the intermediate proof
        // formula with the corresponding `next_lp`).
        let mut best_for_this_lb: Option<(usize, Vec<Step>)> = None;
        let try_update = |total: usize, steps: Vec<Step>, slot: &mut Option<(usize, Vec<Step>)>| {
            if slot.as_ref().map(|(c, _)| total < *c).unwrap_or(true) {
                *slot = Some((total, steps));
            }
        };

        // Branch A: child is a Direct at level+1. This level uses the
        // terminal proof formula.
        if let Some((child_cost, child_sched)) = child.best_direct.as_ref() {
            let level_proof_size = compute_level_proof_size::<Cfg>(
                &lp,
                &lp,
                next_w_len_terminal,
                1,
                MRowLayout::Terminal,
            ) + eor_bytes;
            let total = level_proof_size + child_cost;
            let mut steps = Vec::with_capacity(1 + child_sched.len());
            steps.push(Step::Fold(FoldStep {
                params: lp.clone(),
                current_w_len,
                next_w_len: next_w_len_terminal,
                level_bytes: level_proof_size,
            }));
            steps.extend(child_sched.iter().cloned());
            try_update(total, steps, &mut best_for_this_lb);
        }
        // Branch B: child is a Fold at level+1. Each child first_lb
        // gives a different `next_lp.b_key.row_len`, hence a different
        // intermediate proof at this level — so iterate them all
        // instead of pre-picking the child's local min.
        for (_child_first_lb, (child_cost, child_sched)) in child.best_fold_per_lb.iter() {
            let Step::Fold(child_fold_step) = &child_sched[0] else {
                unreachable!("best_fold_per_lb schedules start with Step::Fold");
            };
            let level_proof_size = compute_level_proof_size::<Cfg>(
                &lp,
                &child_fold_step.params,
                next_w_len,
                1,
                MRowLayout::Intermediate,
            ) + eor_bytes;
            let total = level_proof_size + child_cost;
            let mut steps = Vec::with_capacity(1 + child_sched.len());
            steps.push(Step::Fold(FoldStep {
                params: lp.clone(),
                current_w_len,
                next_w_len,
                level_bytes: level_proof_size,
            }));
            steps.extend(child_sched.iter().cloned());
            try_update(total, steps, &mut best_for_this_lb);
        }

        if let Some(entry) = best_for_this_lb {
            best_fold_per_lb.insert(lb, entry);
        }
    }

    let result = SuffixResult {
        best_direct,
        best_fold_per_lb,
    };
    memo.insert(memo_key, result.clone());
    Ok(result)
}

/// Consult the offline schedule tables for a pre-computed answer.
fn offline_schedule_for_key<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
) -> Result<Option<Schedule>, AkitaError> {
    // Materialization arithmetic is bit-width driven and uses
    // `root_decomp.field_bits()` everywhere, so the field marker is just
    // a phantom — pick a small canonical field that satisfies the bound.
    use akita_field::Prime128OffsetA7F7 as PhantomField;
    let plan = schedule_plan_from_table::<PhantomField, _>(
        key,
        table,
        PlanPolicy {
            sis_family: Cfg::sis_modulus_family(),
            ring_dimension: Cfg::D,
            root_decomp: Cfg::decomposition(),
            challenge_field_bits: Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
            recursive_public_rows: 1,
            extension_opening_width: Cfg::CLAIM_EXT_DEGREE,
            stage1_challenge_config: Cfg::stage1_challenge_config,
            ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
            fold_challenge_shape: Cfg::fold_challenge_shape_at_level,
        },
    )?;
    Ok(plan.map(|plan| schedule_from_plan(&plan)))
}

/// Root-direct commitment `LevelParams` with a symmetric `(m, r)` split.
///
/// Root-direct schedules ship the cleartext witness on the wire, so
/// they don't run the relation fold (D unused). They still build the
/// outer Ajtai commitment `u = Σ B · t_i = Σ B · A · s_i`, so both A
/// (witness commit) and B (outer commit) need real SIS-secure sizing.
fn compute_root_direct_level_params<Cfg: CommitmentConfig>(
    num_vars: usize,
    log_basis: u32,
) -> Result<Option<LevelParams>, AkitaError> {
    let d = Cfg::D;
    let alpha = (d as u32).trailing_zeros() as usize;
    let reduced_vars = num_vars.saturating_sub(alpha);
    // Symmetric split. `r_vars = log_num_blocks`, `m_vars = log_block_len`.
    let r_vars = reduced_vars / 2;
    let m_vars = reduced_vars - r_vars;
    let Some(num_blocks) = 1usize.checked_shl(r_vars as u32) else {
        return Ok(None);
    };
    let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
        return Ok(None);
    };
    // Only A (witness commit) and B (outer commit) are used at proof time.
    let Some((a_key, b_key, _d_key)) =
        compute_all_ajtai_keys_params::<Cfg>(block_len, num_blocks, 1, log_basis, true)?
    else {
        return Ok(None);
    };
    // D is unused at proof time for root-direct (no relation fold), so the
    // real key is dropped for a zero-width placeholder via `new_unchecked`
    // (bypasses the strict SIS audit; never flows through `try_new` here).
    let d_key = AjtaiKeyParams::new_unchecked(Cfg::sis_modulus_family(), 1, 0, 0, d);

    let witness_len = 1usize.checked_shl(num_vars as u32).unwrap_or(0);
    let stage1 = Cfg::stage1_challenge_config(d)?;
    let fold_shape = Cfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: witness_len,
    });
    Ok(Some(LevelParams {
        ring_dimension: d,
        log_basis,
        a_key,
        b_key,
        d_key,
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config: stage1.clone(),
        fold_challenge_shape: fold_shape,
        num_digits_commit: WitnessType::S.decomposed_num_digits::<Cfg>(log_basis, true),
        num_digits_open: WitnessType::T.decomposed_num_digits::<Cfg>(log_basis, true),
        // TODO: drop this fake fold-digit count once the ring-switch
        // validators are gated on schedule shape. Root-direct never
        // folds, but the verifier's ring-switch path currently rejects
        // any `LevelParams` with `num_digits_fold == 0`, so we compute
        // the canonical singleton value here just to keep that check
        // happy. The "real" value for root-direct is 0.
        num_digits_fold: WitnessType::Z.decomposed_fold_num_digits::<Cfg>(
            log_basis,
            r_vars,
            fold_shape.effective_l1_mass(&stage1),
            1,
        ),
    }))
}

/// Find the optimal schedule for a root schedule lookup key under `Cfg`.
///
/// When `use_lookup` is true the search consults `Cfg::schedule_table()`
/// as a fast path. Production callers pass `true`; the
/// `gen_schedule_tables` binary passes `false` so the output is a pure
/// function of `Cfg` itself.
///
/// # Errors
///
/// Returns an error if vector counts are invalid, if the witness length
/// overflows, or if generated-table materialization fails.
pub fn find_schedule<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    use_lookup: bool,
) -> Result<Schedule, AkitaError> {
    let t_vectors = key.num_t_vectors;
    let w_vectors = key.num_w_vectors;
    let z_vectors = key.num_z_vectors;
    let num_points = key.num_points;
    if num_points == 0 || t_vectors == 0 || w_vectors == 0 || z_vectors == 0 {
        return Err(AkitaError::InvalidSetup(
            "schedule key planner dimensions must be at least 1".into(),
        ));
    }
    if num_points > t_vectors || num_points > w_vectors {
        return Err(AkitaError::InvalidSetup(
            "schedule key opening-point count cannot exceed t or w vector counts".into(),
        ));
    }

    if use_lookup {
        if let Some(table) = Cfg::schedule_table() {
            if let Some(schedule) = offline_schedule_for_key::<Cfg>(key, table)? {
                tracing::debug!(
                    ?key,
                    total_bytes = schedule.total_bytes,
                    "schedule planner: served from offline tables"
                );
                return Ok(schedule);
            }
        }
    }

    let witness_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let root_witness_shape = DirectWitnessShape::FieldElements(witness_len);
    let mut best_cost =
        direct_witness_bytes(Cfg::decomposition().field_bits(), &root_witness_shape);
    let root_direct_commit_params =
        compute_root_direct_level_params::<Cfg>(key.num_vars, Cfg::decomposition().log_basis)?;
    let mut best_steps: Vec<Step> = vec![Step::Direct(DirectStep {
        current_w_len: witness_len,
        witness_shape: root_witness_shape,
        direct_bytes: best_cost,
        params: root_direct_commit_params,
    })];
    let mut memo = ScheduleMemo::new();

    let stage1 = Cfg::stage1_challenge_config(Cfg::D)?;
    let fold_shape = Cfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level: 0,
        current_w_len: witness_len,
    });
    let alpha = (Cfg::D as u32).trailing_zeros() as usize;
    let reduced_vars = key.num_vars.saturating_sub(alpha);

    if reduced_vars == 0 {
        return Ok(Schedule {
            steps: best_steps,
            total_bytes: best_cost,
        });
    }

    let min_r_vars: usize = if reduced_vars >= 3 { 1 } else { 0 };
    let max_r_vars: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);

    let field_bits = Cfg::decomposition().field_bits();

    let (min_log_basis, max_log_basis) = Cfg::basis_range();
    for candidate_log_basis in min_log_basis..=max_log_basis {
        let num_digits_commit =
            WitnessType::S.decomposed_num_digits::<Cfg>(candidate_log_basis, true);
        let num_digits_open =
            WitnessType::T.decomposed_num_digits::<Cfg>(candidate_log_basis, true);

        for r_vars in min_r_vars..=max_r_vars {
            let num_blocks: usize = 1usize << r_vars;
            let m_vars = reduced_vars - r_vars;

            let Some(block_len) = 1usize.checked_shl(m_vars as u32) else {
                continue;
            };

            let Some((a_key, b_key, d_key)) = compute_all_ajtai_keys_params::<Cfg>(
                block_len,
                num_blocks,
                t_vectors,
                candidate_log_basis,
                true,
            )?
            else {
                continue;
            };

            let num_digits_fold = WitnessType::Z.decomposed_fold_num_digits::<Cfg>(
                candidate_log_basis,
                r_vars,
                fold_shape.effective_l1_mass(&stage1),
                t_vectors,
            );

            let level_lp = LevelParams {
                ring_dimension: Cfg::D,
                log_basis: candidate_log_basis,
                a_key,
                b_key,
                d_key,
                num_blocks,
                block_len,
                m_vars,
                r_vars,
                stage1_config: stage1.clone(),
                fold_challenge_shape: fold_shape,
                num_digits_commit,
                num_digits_open,
                num_digits_fold,
            };

            let next_withness_len_impl = |layout| -> Result<usize, AkitaError> {
                let rings = w_ring_element_count_with_counts_for_layout_bits(
                    field_bits,
                    &level_lp,
                    key.num_points,
                    key.num_t_vectors,
                    key.num_w_vectors,
                    key.num_z_vectors,
                    layout,
                )?;
                rings.checked_mul(Cfg::D).ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness length overflow".into())
                })
            };
            let next_w_len = next_withness_len_impl(MRowLayout::Intermediate)?;
            let next_w_len_terminal = next_withness_len_impl(MRowLayout::Terminal)?;
            let initial_witness_len_bits = witness_len
                .checked_mul(field_bits as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root witness bit length overflow".into())
                })?;
            if next_w_len
                .checked_mul(candidate_log_basis as usize)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness bit length overflow".into())
                })?
                >= initial_witness_len_bits
            {
                continue;
            }

            let child = derive_optimal_suffix_schedule::<Cfg>(
                &mut memo,
                key.num_vars,
                1,
                next_w_len,
                next_w_len_terminal,
                candidate_log_basis,
                0,
            )?;
            if child.is_empty() {
                continue;
            }
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes::<Cfg>(key, 0, witness_len)
            else {
                continue;
            };

            // Branch A: suffix at level 1 is a Direct
            if let Some((child_cost, child_sched)) = child.best_direct.as_ref() {
                let root_proof_size = compute_level_proof_size::<Cfg>(
                    &level_lp,
                    &level_lp,
                    next_w_len_terminal,
                    z_vectors,
                    MRowLayout::Terminal,
                ) + eor_bytes;
                let total = root_proof_size + child_cost;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + child_sched.len());
                    steps.push(Step::Fold(FoldStep {
                        params: level_lp.clone(),
                        current_w_len: witness_len,
                        next_w_len: next_w_len_terminal,
                        level_bytes: root_proof_size,
                    }));
                    steps.extend(child_sched.iter().cloned());
                    best_steps = steps;
                }
            }
            // Branch B: suffix at level 1 is a Fold
            for (_child_first_lb, (child_cost, child_sched)) in child.best_fold_per_lb.iter() {
                let Step::Fold(child_fold_step) = &child_sched[0] else {
                    unreachable!("best_fold_per_lb schedules start with Step::Fold");
                };
                let root_proof_size = compute_level_proof_size::<Cfg>(
                    &level_lp,
                    &child_fold_step.params,
                    next_w_len,
                    z_vectors,
                    MRowLayout::Intermediate,
                ) + eor_bytes;
                let total = root_proof_size + child_cost;
                if total < best_cost {
                    best_cost = total;
                    let mut steps = Vec::with_capacity(1 + child_sched.len());
                    steps.push(Step::Fold(FoldStep {
                        params: level_lp.clone(),
                        current_w_len: witness_len,
                        next_w_len,
                        level_bytes: root_proof_size,
                    }));
                    steps.extend(child_sched.iter().cloned());
                    best_steps = steps;
                }
            }
        }
    }

    Ok(Schedule {
        steps: best_steps,
        total_bytes: best_cost,
    })
}
