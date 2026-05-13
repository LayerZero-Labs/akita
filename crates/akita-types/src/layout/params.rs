//! Unified per-level parameters for the Akita protocol.
//!
//! `LevelParams` merges ring dimension, matrix ranks, challenge config,
//! block geometry, and digit depths into a single struct that fully
//! describes one recursion level.

use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_field::AkitaError;

/// Shape-aware stage-1 SIS extraction accounting for one level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stage1SisExtractionReport {
    /// Honest logical challenge L1 mass used for fold witness bounds.
    pub honest_challenge_l1_mass: usize,
    /// Base sparse challenge coefficient L-infinity bound.
    pub base_challenge_linf: u32,
    /// Relative MSIS extraction degradation for the configured shape.
    pub extraction_relative_msis_degradation: u128,
    /// Shape-aware challenge coefficient bound used by A-role SIS sizing.
    pub extraction_linf: u32,
    /// Raw A-role collision bound before challenge extraction scaling.
    pub a_role_raw_collision: u32,
    /// Raw A-role collision multiplied by `extraction_linf`.
    pub a_role_extraction_collision: u32,
    /// Supported SIS collision bucket used for the A role.
    pub a_role_supported_collision_bucket: u32,
}

/// Parameters for a single Ajtai commitment matrix.
///
/// Each matrix in the protocol (A, B, D) is characterised by its row count
/// (security rank), column count (message width), and the worst-case L∞
/// collision bound used for SIS security sizing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AjtaiKeyParams {
    row_len: usize,
    col_len: usize,
    collision_inf: u32,
}

impl AjtaiKeyParams {
    fn sis_security_violation(
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Option<String> {
        if col_len > 0 && collision_inf > 0 && row_len > 0 {
            use crate::generated::sis_floor::min_rank_for_secure_width;
            if let Some(floor) =
                min_rank_for_secure_width(ring_dimension as u32, collision_inf, col_len as u64)
            {
                if row_len < floor {
                    return Some(format!(
                        "AjtaiKeyParams: row_len {row_len} < SIS floor {floor} \
                         (d={ring_dimension}, collision_inf={collision_inf}, col_len={col_len})"
                    ));
                }
            }
        }
        None
    }

    /// Create a new `AjtaiKeyParams` with SIS security enforcement.
    ///
    /// # Panics
    ///
    /// Panics if `row_len` is below the 128-bit SIS security floor for the
    /// given `(ring_dimension, collision_inf, col_len)` triple.
    pub fn new(row_len: usize, col_len: usize, collision_inf: u32, ring_dimension: usize) -> Self {
        if let Some(message) =
            Self::sis_security_violation(row_len, col_len, collision_inf, ring_dimension)
        {
            panic!("{message}");
        }
        Self {
            row_len,
            col_len,
            collision_inf,
        }
    }

    /// Create a new `AjtaiKeyParams`, returning an error on SIS violations.
    ///
    /// # Errors
    ///
    /// Returns an error if `row_len` is below the 128-bit SIS security floor
    /// for the given `(ring_dimension, collision_inf, col_len)` triple.
    pub fn try_new(
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if let Some(message) =
            Self::sis_security_violation(row_len, col_len, collision_inf, ring_dimension)
        {
            return Err(AkitaError::InvalidSetup(message));
        }
        Ok(Self {
            row_len,
            col_len,
            collision_inf,
        })
    }

    /// Create a new `AjtaiKeyParams` without enforcing SIS security.
    ///
    /// Logs a warning if `row_len` is below the SIS floor but does not
    /// panic. Use this for intermediate construction steps where ranks
    /// have not yet converged (e.g., batched scaling, iterative SIS
    /// fixed-point loops).
    pub fn new_unchecked(
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Self {
        if col_len > 0 && collision_inf > 0 && row_len > 0 {
            use crate::generated::sis_floor::min_rank_for_secure_width;
            if let Some(floor) =
                min_rank_for_secure_width(ring_dimension as u32, collision_inf, col_len as u64)
            {
                if row_len < floor {
                    tracing::warn!(
                        row_len,
                        floor,
                        ring_dimension,
                        collision_inf,
                        col_len,
                        "AjtaiKeyParams::new_unchecked: row_len below SIS floor"
                    );
                }
            }
        }
        Self {
            row_len,
            col_len,
            collision_inf,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// Number of columns.
    #[inline]
    pub fn col_len(&self) -> usize {
        self.col_len
    }

    /// Worst-case L∞ collision bound for SIS security sizing.
    #[inline]
    pub fn collision_inf(&self) -> u32 {
        self.collision_inf
    }
}

/// Per-commitment-group shape inside a multi-group batched Hachi commit.
///
/// The outer [`LevelParams`] carries the shared `(D, A)` matrices, ring
/// dimension, log_basis, and stage-1 challenge config across all groups.
/// Each `GroupSpec` carries the per-commitment-group `(m, r)` split,
/// `B`-matrix dimensions, and per-group digit decomposition depths.
///
/// The single-group case (today's batched Hachi commit) is the
/// `groups == None` shape on [`LevelParams`]; `groups == Some(vec)`
/// activates the multi-group path that the book §5.3 names a "split
/// commitment". The per-row machinery in `prepare_m_eval` and the stage-2
/// closing relation consume per-group sub-rows via these specs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupSpec {
    /// Block-select variable count for this group (`log₂ num_blocks_g`).
    pub m_vars: usize,
    /// Per-block variable count for this group.
    pub r_vars: usize,
    /// Number of committed blocks for this group (`2^r_vars_g`).
    pub num_blocks: usize,
    /// Ring elements per block for this group.
    pub block_len: usize,
    /// Per-group outer commitment matrix `B_g`.
    pub b_key: AjtaiKeyParams,
    /// Gadget decomposition depth for this group's commitment coefficients
    /// (`δ_commit,g`). For the `w`-group this is the recursive witness
    /// digit count; for the `S`-group at the L+1 join this is
    /// `⌈log₂ q / log₂ b⌉` (full-field, e.g. 65 at `b = 4`).
    pub num_digits_commit: usize,
    /// Gadget decomposition depth for this group's opening evaluations
    /// (`δ_open,g`).
    pub num_digits_open: usize,
    /// Gadget decomposition depth for this group's folded witness
    /// (`δ_fold,g` / `τ_g`).
    pub num_digits_fold: usize,
}

impl GroupSpec {
    /// Synthesize a `GroupSpec` describing the single-group case from an
    /// outer [`LevelParams`].
    ///
    /// This is the fallback used when `LevelParams::groups == None`: every
    /// commitment group inherits the outer LP's `(m, r, B, digit_count)`.
    /// The result is bit-equivalent to the existing single-LP code path.
    #[inline]
    pub fn from_outer(lp: &LevelParams) -> Self {
        Self {
            m_vars: lp.m_vars,
            r_vars: lp.r_vars,
            num_blocks: lp.num_blocks,
            block_len: lp.block_len,
            b_key: lp.b_key.clone(),
            num_digits_commit: lp.num_digits_commit,
            num_digits_open: lp.num_digits_open,
            num_digits_fold: lp.num_digits_fold,
        }
    }

    /// Lower this `GroupSpec` into a single-group [`LevelParams`] using the
    /// outer LP's shared `(D, A)`, ring dimension, log_basis, challenge
    /// config, and flags.
    ///
    /// The returned LP has `groups == None`, suitable for the existing
    /// single-group commit / stage-2 paths (e.g. when iterating per-group
    /// inside the multi-group commit kernel).
    #[inline]
    pub fn lower_into_outer(&self, outer: &LevelParams) -> LevelParams {
        LevelParams {
            ring_dimension: outer.ring_dimension,
            log_basis: outer.log_basis,
            a_key: outer.a_key.clone(),
            b_key: self.b_key.clone(),
            d_key: outer.d_key.clone(),
            num_blocks: self.num_blocks,
            block_len: self.block_len,
            m_vars: self.m_vars,
            r_vars: self.r_vars,
            stage1_config: outer.stage1_config.clone(),
            stage1_challenge_shape: outer.stage1_challenge_shape,
            use_setup_claim_reduction: outer.use_setup_claim_reduction,
            num_digits_commit: self.num_digits_commit,
            num_digits_open: self.num_digits_open,
            num_digits_fold: self.num_digits_fold,
            groups: None,
        }
    }
}

/// Unified per-level parameters for one Akita recursion level.
///
/// Combines ring dimension, Ajtai matrix descriptions, block geometry,
/// sparse-challenge configuration, and digit decomposition depths into a
/// single authoritative struct.
///
/// The optional `groups` field carries per-commitment-group shape for the
/// multi-group batched Hachi commit (book §5.3 "split commitment"). When
/// `None`, every commitment group shares the outer LP's `(m, r, B,
/// digit_count)` — the existing single-LP shape, bit-equivalent. When
/// `Some(vec)`, each commitment group carries its own [`GroupSpec`]; the
/// outer LP's `(D, A)`, ring dimension, log_basis, and challenge config
/// remain shared across groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelParams {
    /// Ring dimension (`d` in the protocol).
    pub ring_dimension: usize,
    /// Base-2 logarithm of the gadget decomposition base.
    pub log_basis: u32,
    /// Inner Ajtai matrix (A): `row_len = n_a`, `col_len = inner_width`.
    pub a_key: AjtaiKeyParams,
    /// Outer commitment matrix (B): `row_len = n_b`, `col_len = outer_width`.
    pub b_key: AjtaiKeyParams,
    /// Prover matrix (D): `row_len = n_d`, `col_len = d_matrix_width`.
    pub d_key: AjtaiKeyParams,
    /// Number of committed blocks (`2^r_vars`).
    pub num_blocks: usize,
    /// Number of ring elements per block. Equals `2^m_vars` at the root level
    /// but may differ at recursive levels (`ceil(num_ring / num_blocks)`).
    pub block_len: usize,
    /// Block-select variable count (log₂ `num_blocks`). Stored explicitly
    /// because `num_blocks.trailing_zeros()` suffices only when `num_blocks`
    /// is a power of two, which is always true by construction.
    pub m_vars: usize,
    /// Per-block variable count. Stored explicitly because at recursive
    /// levels `block_len` is not necessarily `2^r_vars`.
    pub r_vars: usize,
    /// Stage-1 sparse challenge family sampled at this level.
    pub stage1_config: SparseChallengeConfig,
    /// Stage-1 folding challenge transcript shape.
    pub stage1_challenge_shape: Stage1ChallengeShape,
    /// Gadget decomposition depth for commitment coefficients (δ_commit).
    pub num_digits_commit: usize,
    /// Gadget decomposition depth for opening evaluations (δ_open).
    pub num_digits_open: usize,
    /// Gadget decomposition depth for the folded witness (δ_fold / τ).
    pub num_digits_fold: usize,
    /// When true, run the setup-side claim-reduction sumcheck after stage 2
    /// instead of materializing the setup matrix during the closing oracle
    /// check. Mirrored from `CommitmentConfig::use_setup_claim_reduction`.
    pub use_setup_claim_reduction: bool,
    /// Optional per-commitment-group shape for multi-group batched Hachi.
    ///
    /// `None`: every commitment group inherits the outer LP's
    /// `(m, r, B, digit_count)` (today's single-LP shape).
    ///
    /// `Some(vec)`: per-commitment-group `(m_g, r_g, B_g, digit_count_g)`
    /// with shared outer `D, A`. The book §5.3's "split commitment" lives
    /// in this representation. Slices E and F use it for the joint
    /// recursive `(w, S)` open at level L+1.
    pub groups: Option<Vec<GroupSpec>>,
}

impl LevelParams {
    /// Build a params-only `LevelParams` with zeroed layout fields.
    ///
    /// Only ring dimension, matrix row counts, log_basis, and stage1_config
    /// are populated. Column counts, block geometry, and digit depths are
    /// zeroed. Call `with_layout` to fill them from a derived layout.
    pub fn params_only(
        ring_dimension: usize,
        log_basis: u32,
        n_a: usize,
        n_b: usize,
        n_d: usize,
        stage1_config: SparseChallengeConfig,
    ) -> Self {
        Self {
            ring_dimension,
            log_basis,
            a_key: AjtaiKeyParams {
                row_len: n_a,
                ..Default::default()
            },
            b_key: AjtaiKeyParams {
                row_len: n_b,
                ..Default::default()
            },
            d_key: AjtaiKeyParams {
                row_len: n_d,
                ..Default::default()
            },
            num_blocks: 0,
            block_len: 0,
            m_vars: 0,
            r_vars: 0,
            stage1_config,
            stage1_challenge_shape: Stage1ChallengeShape::Flat,
            use_setup_claim_reduction: false,
            num_digits_commit: 0,
            num_digits_open: 0,
            num_digits_fold: 0,
            groups: None,
        }
    }

    /// Per-commitment-group specs for `num_commitment_groups` groups.
    ///
    /// When `self.groups == Some(vec)` and `vec.len() == num_commitment_groups`,
    /// returns the user-specified per-group shape. Otherwise (including
    /// `None`) returns a synthesized single-group view: every group
    /// inherits the outer LP's `(m, r, B, digit_count)`, which is
    /// bit-equivalent to today's single-LP shape.
    ///
    /// # Errors
    ///
    /// Returns an error if `self.groups == Some(vec)` and `vec.len() !=
    /// num_commitment_groups`.
    pub fn group_specs(&self, num_commitment_groups: usize) -> Result<Vec<GroupSpec>, AkitaError> {
        if let Some(groups) = &self.groups {
            if groups.len() != num_commitment_groups {
                return Err(AkitaError::InvalidSetup(format!(
                    "LevelParams.groups has {} entries but caller passed {num_commitment_groups} commitment groups",
                    groups.len()
                )));
            }
            return Ok(groups.clone());
        }
        let single = GroupSpec::from_outer(self);
        Ok((0..num_commitment_groups).map(|_| single.clone()).collect())
    }

    /// Return `true` iff every commitment group inherits the outer LP's
    /// `(m, r, B, digit_count)`.
    ///
    /// `groups == None` is homogeneous by construction (every group falls
    /// back to the outer LP). `groups == Some(vec)` is homogeneous iff
    /// every spec equals `GroupSpec::from_outer(self)`. This is the
    /// "today's-single-LP" predicate that lets the per-row machinery in
    /// `prepare_m_eval` and the stage-2 closing relation short-circuit
    /// to the existing offset/width math without consulting `groups`.
    ///
    /// `Some(vec)` with all specs equal to each other but not equal to
    /// the outer LP is NOT homogeneous: such an LP either (a) has the
    /// wrong outer fields, or (b) is genuinely multi-group with a single
    /// custom spec. Both cases need slice E's per-group machinery.
    pub fn groups_are_homogeneous(&self) -> bool {
        match &self.groups {
            None => true,
            Some(groups) => {
                if groups.is_empty() {
                    return true;
                }
                let outer = GroupSpec::from_outer(self);
                groups.iter().all(|g| *g == outer)
            }
        }
    }

    /// Worst-case effective L1 mass of a logical folding challenge.
    #[inline]
    pub fn challenge_l1_mass(&self) -> usize {
        self.stage1_challenge_shape
            .effective_l1_mass(&self.stage1_config)
    }

    /// Relative Module-SIS extraction degradation for stage-1 challenges.
    ///
    /// This is intentionally separate from [`Self::challenge_l1_mass`]. Tensor
    /// folding uses `omega^2` honest mass for witness bounds, while the
    /// two-level CWSS extractor pays the tex-model `4 * omega` degradation
    /// relative to the base challenge coefficient bound.
    ///
    /// # Errors
    ///
    /// Returns an error if the degradation factor overflows.
    pub fn stage1_extraction_relative_msis_degradation(&self) -> Result<u128, AkitaError> {
        match self.stage1_challenge_shape {
            Stage1ChallengeShape::Flat => Ok(1),
            Stage1ChallengeShape::Tensor => (self.stage1_config.l1_norm() as u128)
                .checked_mul(4)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "tensor stage-1 extraction degradation overflow".to_string(),
                    )
                }),
        }
    }

    /// Shape-aware challenge coefficient bound used for A-role SIS extraction.
    ///
    /// Flat mode preserves the existing `SparseChallengeConfig::infinity_norm`
    /// proxy. Tensor mode multiplies that proxy by the two-level CWSS
    /// `4 * omega` extraction degradation, where
    /// `omega = SparseChallengeConfig::l1_norm()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the conservative extraction bound does not fit `u32`.
    pub fn stage1_extraction_infinity_norm(&self) -> Result<u32, AkitaError> {
        let bound = self
            .stage1_extraction_relative_msis_degradation()?
            .checked_mul(self.stage1_config.infinity_norm() as u128)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("stage-1 extraction infinity bound overflow".to_string())
            })?;
        u32::try_from(bound).map_err(|_| {
            AkitaError::InvalidSetup("stage-1 extraction infinity bound exceeds u32".to_string())
        })
    }

    /// Return shape-aware SIS extraction accounting for planner/report output.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape-aware collision bound overflows or is not
    /// covered by the generated SIS collision buckets.
    pub fn stage1_sis_extraction_report(
        &self,
        a_role_raw_collision: u32,
    ) -> Result<Stage1SisExtractionReport, AkitaError> {
        let extraction_relative_msis_degradation =
            self.stage1_extraction_relative_msis_degradation()?;
        let extraction_linf = self.stage1_extraction_infinity_norm()?;
        let a_role_extraction_collision =
            a_role_raw_collision
                .checked_mul(extraction_linf)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "stage-1 A-role extraction collision overflow: raw={a_role_raw_collision}, extraction_linf={extraction_linf}"
                    ))
                })?;
        let a_role_supported_collision_bucket =
            crate::generated::sis_floor::ceil_supported_collision(
                self.ring_dimension as u32,
                a_role_extraction_collision,
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "missing supported stage-1 A-role collision bucket for D={} and collision {}",
                    self.ring_dimension, a_role_extraction_collision
                ))
            })?;
        Ok(Stage1SisExtractionReport {
            honest_challenge_l1_mass: self.challenge_l1_mass(),
            base_challenge_linf: self.stage1_config.infinity_norm(),
            extraction_relative_msis_degradation,
            extraction_linf,
            a_role_raw_collision,
            a_role_extraction_collision,
            a_role_supported_collision_bucket,
        })
    }

    /// Return a copy of these params using tensor-structured stage-1 folding.
    #[inline]
    pub fn with_tensor_stage1_challenges(mut self) -> Self {
        self.stage1_challenge_shape = Stage1ChallengeShape::Tensor;
        self.num_digits_fold = crate::layout::digit_math::compute_num_digits_fold_with_claims(
            self.r_vars,
            self.challenge_l1_mass(),
            self.log_basis,
            1,
            128,
        );
        self
    }

    /// Return a copy of these params using flat (non-tensor) stage-1 folding.
    ///
    /// Mirrors [`Self::with_tensor_stage1_challenges`] but for the flat shape.
    /// Flat has smaller effective L1 mass than tensor for the same sparse
    /// challenge config, so `num_digits_fold` is recomputed against the new
    /// (smaller) mass. Hybrid per-level shapes pick this helper at fold
    /// levels where the verifier-side win from staying flat outweighs the
    /// prover-side win from going tensor.
    #[inline]
    pub fn with_flat_stage1_challenges(mut self) -> Self {
        self.stage1_challenge_shape = Stage1ChallengeShape::Flat;
        self.num_digits_fold = crate::layout::digit_math::compute_num_digits_fold_with_claims(
            self.r_vars,
            self.challenge_l1_mass(),
            self.log_basis,
            1,
            128,
        );
        self
    }

    /// Return a copy of these params with the setup-side claim-reduction
    /// sumcheck enabled. The stage-2 closing M-table evaluation is split into
    /// an algebraic part (computed by the verifier) plus a setup-dependent
    /// part that this extra sumcheck reduces to a single point opening on
    /// `S`.
    #[inline]
    pub fn with_setup_claim_reduction(mut self) -> Self {
        self.use_setup_claim_reduction = true;
        self
    }

    /// Block-select variable count (the `r_vars` of the legacy layout).
    #[inline]
    pub fn log_num_blocks(&self) -> usize {
        self.r_vars
    }

    /// Per-block variable count (the `m_vars` of the legacy layout).
    #[inline]
    pub fn log_block_len(&self) -> usize {
        self.m_vars
    }

    /// Width of inner matrix A (column count of the A-key).
    #[inline]
    pub fn inner_width(&self) -> usize {
        self.a_key.col_len()
    }

    /// Width of outer matrix B (column count of the B-key).
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.b_key.col_len()
    }

    /// Width of prover matrix D (column count of the D-key).
    #[inline]
    pub fn d_matrix_width(&self) -> usize {
        self.d_key.col_len()
    }

    /// Total outer variable count (`log_num_blocks + log_block_len`).
    #[inline]
    pub fn outer_vars(&self) -> usize {
        self.log_num_blocks() + self.log_block_len()
    }

    /// Total B-row count across `num_commitment_groups` commitment groups.
    ///
    /// For `groups == None`, this is `b_key.row_len() * num_commitment_groups`
    /// (every group inherits the outer LP's `b_key`). For `groups ==
    /// Some(vec)`, this is `sum_g vec[g].b_key.row_len()` so each group
    /// contributes its own per-group rank.
    #[inline]
    pub fn total_b_row_count(&self, num_commitment_groups: usize) -> usize {
        match &self.groups {
            None => self.b_key.row_len() * num_commitment_groups,
            Some(groups) => groups
                .iter()
                .take(num_commitment_groups)
                .map(|g| g.b_key.row_len())
                .sum(),
        }
    }

    /// Row count with `num_commitments` explicit commitment vectors and
    /// `num_public_outputs` public y-rows.
    ///
    /// Row layout: consistency (1) | public (num_public_outputs) | D (n_d) |
    /// B (per-group rank summed across `num_commitments` groups) | A (n_a).
    /// The batched CWSS protocol uses one public y-row per distinct
    /// opening point.
    #[inline]
    pub fn m_row_count(&self, num_commitments: usize, num_public_outputs: usize) -> usize {
        self.d_key.row_len()
            + self.total_b_row_count(num_commitments)
            + num_public_outputs
            + 1
            + self.a_key.row_len()
    }

    /// Fill in the layout-derived fields from explicit decomposition parameters.
    ///
    /// Takes a params-only `LevelParams` (with zeroed layout fields) and
    /// computes block geometry, matrix column counts, and digit depths.
    ///
    /// When `num_ring > 0` (recursive levels), `block_len` is set to
    /// `ceil(num_ring / num_blocks)` instead of `2^m_vars`, giving tight
    /// z_pre sizing. Pass `0` for root-level layouts.
    ///
    /// # Errors
    ///
    /// Returns an error when parameters are invalid or derived widths overflow.
    pub fn with_decomp(
        &self,
        m_vars: usize,
        r_vars: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        num_digits_fold: usize,
        num_ring: usize,
    ) -> Result<Self, AkitaError> {
        let num_blocks = 1usize
            .checked_shl(r_vars as u32)
            .ok_or_else(|| AkitaError::InvalidSetup("2^r_vars does not fit usize".to_string()))?;
        let block_len = if num_ring > 0 {
            num_ring.div_ceil(num_blocks)
        } else {
            1usize.checked_shl(m_vars as u32).ok_or_else(|| {
                AkitaError::InvalidSetup("2^m_vars does not fit usize".to_string())
            })?
        };
        let inner_width = block_len
            .checked_mul(num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = self
            .a_key
            .row_len()
            .checked_mul(num_digits_open)
            .and_then(|x| x.checked_mul(num_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("outer width overflow".to_string()))?;
        let d_matrix_width = num_digits_open
            .checked_mul(num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;
        let d = self.ring_dimension;
        Ok(Self {
            ring_dimension: d,
            log_basis: self.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                self.a_key.row_len,
                inner_width,
                self.a_key.collision_inf,
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                self.b_key.row_len,
                outer_width,
                self.b_key.collision_inf,
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                self.d_key.row_len,
                d_matrix_width,
                self.d_key.collision_inf,
                d,
            ),
            num_blocks,
            block_len,
            m_vars,
            r_vars,
            stage1_config: self.stage1_config.clone(),
            stage1_challenge_shape: self.stage1_challenge_shape,
            use_setup_claim_reduction: self.use_setup_claim_reduction,
            num_digits_commit,
            num_digits_open,
            num_digits_fold,
            groups: None,
        })
    }

    /// Build a new `LevelParams` that keeps rank info from `self` but
    /// replaces all layout-derived fields with those from `other`.
    ///
    /// The Ajtai matrix `collision_inf` is taken from `self` when `self`
    /// supplies a non-zero value, otherwise from `other`. This preserves the
    /// SIS-secured collision bound that `sis_secure_level_params` stores on
    /// `self` (the result of `sis_derived_*_params_for_layout`) — `other` is
    /// typically a fresh layout built from `params_only` whose
    /// `collision_inf` is the default `0`, which would otherwise wipe the
    /// SIS metadata and make `validate_stored_sis_ranks` unable to verify
    /// the floor.
    pub fn with_layout(&self, other: &LevelParams) -> Self {
        let d = self.ring_dimension;
        let merge_collision = |self_v: u32, other_v: u32| {
            if self_v != 0 {
                self_v
            } else {
                other_v
            }
        };
        Self {
            ring_dimension: d,
            log_basis: other.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                self.a_key.row_len,
                other.a_key.col_len,
                merge_collision(self.a_key.collision_inf, other.a_key.collision_inf),
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                self.b_key.row_len,
                other.b_key.col_len,
                merge_collision(self.b_key.collision_inf, other.b_key.collision_inf),
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                self.d_key.row_len,
                other.d_key.col_len,
                merge_collision(self.d_key.collision_inf, other.d_key.collision_inf),
                d,
            ),
            num_blocks: other.num_blocks,
            block_len: other.block_len,
            m_vars: other.m_vars,
            r_vars: other.r_vars,
            stage1_config: self.stage1_config.clone(),
            stage1_challenge_shape: self.stage1_challenge_shape,
            use_setup_claim_reduction: other.use_setup_claim_reduction,
            num_digits_commit: other.num_digits_commit,
            num_digits_open: other.num_digits_open,
            num_digits_fold: other.num_digits_fold,
            groups: other.groups.clone(),
        }
    }
}

/// Conservative bound for centered integer accumulation in stage-1 folding.
///
/// The bound is shape-aware through [`LevelParams::challenge_l1_mass`], so tensor
/// schedules use the effective logical challenge mass.
///
/// # Errors
///
/// Returns an error if the arithmetic bound overflows or if `log_basis` is not
/// usable for centered digit bounds.
pub fn stage1_accumulator_bound(lp: &LevelParams, num_claims: usize) -> Result<u128, AkitaError> {
    if !(1..128).contains(&lp.log_basis) {
        return Err(AkitaError::InvalidSetup(
            "stage-1 accumulator requires log_basis in 1..128".to_string(),
        ));
    }
    let max_digit_abs = 1u128
        .checked_shl(lp.log_basis - 1)
        .ok_or_else(|| AkitaError::InvalidSetup("max digit bound overflow".to_string()))?;
    (lp.num_blocks as u128)
        .checked_mul(num_claims as u128)
        .and_then(|bound| bound.checked_mul(lp.challenge_l1_mass() as u128))
        .and_then(|bound| bound.checked_mul(max_digit_abs))
        .ok_or_else(|| AkitaError::InvalidSetup("stage-1 accumulator bound overflow".to_string()))
}

/// Reject tensor schedules whose conservative stage-1 accumulator bound exceeds
/// the current centered accumulator width.
///
/// # Errors
///
/// Returns an error when the bound cannot be computed or exceeds `i64::MAX`.
pub fn validate_stage1_accumulator_headroom(
    lp: &LevelParams,
    num_claims: usize,
) -> Result<(), AkitaError> {
    if !matches!(lp.stage1_challenge_shape, Stage1ChallengeShape::Tensor) {
        return Ok(());
    }
    let bound = stage1_accumulator_bound(lp, num_claims)?;
    let limit = i64::MAX as u128;
    if bound > limit {
        return Err(AkitaError::InvalidSetup(format!(
            "tensor stage-1 accumulator bound {bound} exceeds i64::MAX ({limit})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params_only() -> LevelParams {
        LevelParams::params_only(
            64,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
    }

    fn sample_layout_lp() -> LevelParams {
        sample_params_only().with_decomp(4, 2, 2, 2, 3, 0).unwrap()
    }

    #[test]
    fn with_layout_keeps_self_ranks() {
        let params = sample_params_only();
        let layout_lp = sample_layout_lp();

        let lp = params.with_layout(&layout_lp);

        assert_eq!(lp.ring_dimension, 64);
        assert_eq!(lp.log_basis, layout_lp.log_basis);
        assert_eq!(lp.a_key.row_len(), 2);
        assert_eq!(lp.b_key.row_len(), 4);
        assert_eq!(lp.d_key.row_len(), 3);
        assert_eq!(lp.num_blocks, layout_lp.num_blocks);
        assert_eq!(lp.block_len, layout_lp.block_len);
        assert_eq!(lp.challenge_l1_mass(), 3);
        assert_eq!(lp.num_digits_commit, layout_lp.num_digits_commit);
        assert_eq!(lp.num_digits_open, layout_lp.num_digits_open);
        assert_eq!(lp.num_digits_fold, layout_lp.num_digits_fold);
    }

    #[test]
    fn derived_widths_match_ajtai_col_len() {
        let lp = sample_params_only().with_layout(&sample_layout_lp());

        assert_eq!(lp.inner_width(), lp.a_key.col_len());
        assert_eq!(lp.outer_width(), lp.b_key.col_len());
        assert_eq!(lp.d_matrix_width(), lp.d_key.col_len());
    }

    #[test]
    fn derived_log_values() {
        let layout_lp = sample_layout_lp();
        let lp = sample_params_only().with_layout(&layout_lp);

        assert_eq!(lp.log_num_blocks(), layout_lp.r_vars);
        assert_eq!(lp.log_block_len(), layout_lp.m_vars);
        assert_eq!(lp.outer_vars(), layout_lp.m_vars + layout_lp.r_vars);
    }

    #[test]
    fn m_row_count_values() {
        let lp = sample_params_only().with_layout(&sample_layout_lp());

        assert_eq!(lp.m_row_count(1, 1), 3 + 4 + 1 + 1 + 2);
        assert_eq!(lp.m_row_count(2, 5), 3 + 4 * 2 + 5 + 1 + 2);
        assert_eq!(lp.m_row_count(4, 4), 3 + 4 * 4 + 4 + 1 + 2);
    }

    #[test]
    fn accumulator_bound_uses_tensor_effective_mass() {
        let lp = sample_layout_lp().with_tensor_stage1_challenges();
        let bound = stage1_accumulator_bound(&lp, 5).unwrap();

        assert_eq!(
            bound,
            (lp.num_blocks as u128)
                * 5
                * (lp.stage1_config.l1_norm() as u128).pow(2)
                * (1u128 << (lp.log_basis - 1))
        );
    }

    #[test]
    fn tensor_accumulator_headroom_rejects_unsafe_schedule() {
        let mut lp = sample_layout_lp().with_tensor_stage1_challenges();
        lp.num_blocks = 1usize << 60;
        lp.log_basis = 8;

        let err = validate_stage1_accumulator_headroom(&lp, 1).unwrap_err();
        assert!(format!("{err:?}").contains("exceeds i64::MAX"));
    }

    #[test]
    fn tensor_extraction_bound_is_separate_from_honest_mass() {
        let lp = sample_layout_lp().with_tensor_stage1_challenges();

        assert_eq!(lp.challenge_l1_mass(), 9);
        assert_eq!(
            lp.stage1_extraction_relative_msis_degradation().unwrap(),
            12
        );
        assert_eq!(lp.stage1_extraction_infinity_norm().unwrap(), 12);
    }

    #[test]
    fn tensor_sis_extraction_report_exposes_bucket_inputs() {
        let lp = sample_layout_lp().with_tensor_stage1_challenges();
        let report = lp.stage1_sis_extraction_report(3).unwrap();

        assert_eq!(report.honest_challenge_l1_mass, 9);
        assert_eq!(report.base_challenge_linf, 1);
        assert_eq!(report.extraction_relative_msis_degradation, 12);
        assert_eq!(report.extraction_linf, 12);
        assert_eq!(report.a_role_raw_collision, 3);
        assert_eq!(report.a_role_extraction_collision, 36);
        assert_eq!(report.a_role_supported_collision_bucket, 63);
    }
}
