//! Test-only utilities. Gated behind the `test-utils` Cargo feature so
//! production builds never link this module.
//!
//! [`PlannerCfg<Cfg>`][PlannerCfg] is a `CommitmentConfig` wrapper that adds
//! DP fallback on schedule-table miss. All `Cfg` trait hooks delegate to
//! the inner config; the three parameter accessors (`commitment_layout`,
//! `get_params_for_commitment`, `get_params_for_prove`) consult the inner
//! config's generated table first, and on miss fall through to
//! [`crate::find_schedule`] — the offline DP search restricted to `<Cfg>`.
//!
//! Use this from cross-crate test fixtures that exercise table-miss
//! incidences (multipoint, non-singleton `num_t_vectors`, presets with
//! `table = None`, full setup-matrix sizing iteration, etc.). Production
//! presets cover their expected workloads via generated tables and do not
//! need this wrapper.

use std::marker::PhantomData;

use akita_challenges::SparseChallengeConfig;
use akita_config::{
    matrix_envelope_for_schedule, worst_case_grouped_incidence_for_shape, CommitmentConfig,
};
use akita_field::AkitaError;
use akita_types::generated::GeneratedScheduleTable;
use akita_types::{
    schedule_from_plan, AkitaScheduleLookupKey, AkitaSchedulePlan, ClaimIncidenceSummary,
    DecompositionParams, LevelParams, Schedule, SisModulusFamily,
};

use crate::find_schedule;

/// `Cfg` wrapper that routes schedule-table misses through the planner DP.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlannerCfg<Cfg>(PhantomData<Cfg>);

#[allow(clippy::expl_impl_clone_on_copy)]
impl<Cfg> PlannerCfg<Cfg> {
    /// Construct the marker value. `PlannerCfg<Cfg>` carries no state; all
    /// trait methods are associated functions, so this is rarely needed.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Cfg: CommitmentConfig> CommitmentConfig for PlannerCfg<Cfg> {
    type Field = Cfg::Field;
    type ClaimField = Cfg::ClaimField;
    type ChallengeField = Cfg::ChallengeField;

    const D: usize = Cfg::D;

    fn decomposition() -> DecompositionParams {
        Cfg::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Cfg::stage1_challenge_config(d)
    }

    fn sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn schedule_table() -> Option<GeneratedScheduleTable> {
        Cfg::schedule_table()
    }

    fn schedule_plan(key: AkitaScheduleLookupKey) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        Cfg::schedule_plan(key)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        // The inner cfg's table-only sizing is a lower bound — it can miss
        // batched shapes that aren't in the generated table. We must iterate
        // every `(nv, polys, points)` shape ourselves and consult DP on
        // table miss to compute the true envelope.
        if max_num_batched_polys == 0 || max_num_points == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup matrix sizing requires nonzero poly/point counts".to_string(),
            ));
        }
        if max_num_points > max_num_batched_polys {
            return Err(AkitaError::InvalidSetup(format!(
                "max_num_points ({max_num_points}) cannot exceed max_num_batched_polys \
                 ({max_num_batched_polys})"
            )));
        }
        let mut max_setup_len: usize = 1;
        #[cfg(feature = "zk")]
        let mut max_zk_b_len: usize = 1;
        #[cfg(feature = "zk")]
        let mut max_zk_d_len: usize = 1;
        for num_vars in 1..=max_num_vars {
            for num_polys in 1..=max_num_batched_polys {
                let upper_pts = num_polys.min(max_num_points);
                for num_points in 1..=upper_pts {
                    // Mirror production sizing: skew excess claims into one
                    // group so the packed B width (`max_group_poly_count`) is
                    // sized for the worst-case runtime incidence, not an even
                    // split.
                    let incidence =
                        worst_case_grouped_incidence_for_shape(num_vars, num_polys, num_points)?;
                    let schedule = <Self as CommitmentConfig>::get_params_for_prove(&incidence)?;
                    let envelope = matrix_envelope_for_schedule::<Self>(&schedule, &incidence)?;
                    max_setup_len = max_setup_len.max(envelope.max_setup_len);
                    #[cfg(feature = "zk")]
                    {
                        max_zk_b_len = max_zk_b_len.max(envelope.max_zk_b_len);
                        max_zk_d_len = max_zk_d_len.max(envelope.max_zk_d_len);
                    }
                }
            }
        }
        Ok(akita_types::SetupMatrixEnvelope {
            max_setup_len,
            #[cfg(feature = "zk")]
            max_zk_b_len,
            #[cfg(feature = "zk")]
            max_zk_d_len,
        })
    }

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Cfg::ring_subfield_embedding_norm_bound()
    }

    // ---- DP-aware parameter accessors ----------------------------------

    fn get_params_for_prove(incidence: &ClaimIncidenceSummary) -> Result<Schedule, AkitaError> {
        let key = AkitaScheduleLookupKey::new_from_incidence(incidence)?;
        if let Some(plan) = Cfg::schedule_plan(key)? {
            return Ok(schedule_from_plan(&plan));
        }
        find_schedule::<Self>(key, true)
    }
}

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_claims` polynomials with `num_vars` variables.
///
/// First checks the pre-computed generated tables. When no table entry exists
/// (or the entry is root-direct), it falls back to the singleton-derived root
/// split. The returned layout has per-polynomial `B`/`D` widths; callers that
/// want the batched (scaled) root layout scale it themselves via
/// [`akita_types::scale_batched_root_layout`].
///
/// This helper is only useful for tests, benches, and the `profile` example
/// — they pre-size per-poly inputs (e.g. `OneHotPoly`) so the
/// `block_len`/`num_blocks` line up with what `Scheme::commit` will use under
/// the batched layout. Production callers always go through
/// `Cfg::get_params_for_batched_commitment(&incidence)` instead.
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn akita_batched_root_layout<Cfg>(
    num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = AkitaScheduleLookupKey::new(num_vars, num_claims, num_claims, 1);
    if let Some(plan) = Cfg::schedule_plan(lookup_key)? {
        if let Some(split) = akita_types::split_batched_root_params_from_schedule_plan(
            &plan,
            Cfg::decomposition().field_bits(),
        ) {
            tracing::info!(
                num_vars,
                num_claims,
                total_bytes = plan.exact_proof_bytes,
                root_m = split.log_block_len(),
                root_r = split.log_num_blocks(),
                root_lb = split.log_basis,
                "batched root split: read from pre-computed table"
            );
            return Ok(split);
        }
        tracing::info!(
            num_vars,
            num_claims,
            "batched root split: schedule is direct-only, falling back to config root layout"
        );
        return Cfg::get_params_for_batched_commitment(&ClaimIncidenceSummary::same_point(
            num_vars, 1,
        )?);
    }
    tracing::info!(
        num_vars,
        num_claims,
        "batched root split: generated table miss, using singleton-derived fallback"
    );
    Cfg::get_params_for_batched_commitment(&ClaimIncidenceSummary::same_point(num_vars, 1)?)
}
