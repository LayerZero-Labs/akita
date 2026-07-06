//! Unified per-level parameters for the Akita protocol.
//!
//! `LevelParams` merges ring dimension, matrix ranks, challenge config,
//! block geometry, and digit depths into a single struct that fully
//! describes one recursion level.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{AkitaError, CanonicalField};

use crate::descriptor_bytes::{push_u128, push_u32, push_usize};
use crate::layout::ring_dims::CommitmentRingDims;
use crate::opening_claims::OpeningClaimsLayout;
use crate::schedule::PrecommittedGroupParams;

pub use crate::sis::{AjtaiKeyParams, FoldWitnessLinfCapConfig, SisModulusFamily};

/// Per-level M-matrix row layout selector.
///
/// At an intermediate fold the prover ships a fresh commitment for the next
/// witness; the verifier never sees `e_hat` in cleartext and the D-block rows
/// `v = D * e_hat` must appear in the M-matrix to bind `e_hat` into the
/// sumcheck.
///
/// At a terminal fold the cleartext witness is absorbed into the transcript
/// and shipped on the wire, so the verifier evaluates the final witness
/// directly. Keeping the D-block in the relation would be vestigial; this enum
/// lets the prover, verifier, and planner agree to drop it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MRowLayout {
    /// Full layout including the D-block (`v = D * e_hat` rows). Used at every
    /// intermediate fold level and at the root when stage-1 runs.
    WithDBlock,
    /// Cleartext-witness layout: omit the D-block from the M-matrix. Used at
    /// the terminal fold level where `final_witness` ships on the wire.
    WithoutDBlock,
}

/// Group-local root parameters for a precommitted commitment group.
///
/// These fields mirror the group-local pieces of [`LevelParams`]. Widths are
/// derived from the Ajtai keys and block geometry rather than stored twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecommittedLevelParams {
    /// Frozen standalone group layout bound into the grouped root key.
    pub layout: PrecommittedGroupParams,
    /// Inner Ajtai matrix (A) used by this group.
    pub a_key: AjtaiKeyParams,
    /// Outer commitment matrix (B) used by this group.
    pub b_key: AjtaiKeyParams,
    /// Number of committed blocks (`2^r_vars`) for this group.
    pub num_blocks: usize,
    /// Number of ring elements per block (`2^m_vars`) for this group.
    pub block_len: usize,
    /// Gadget decomposition depth for committed coefficients.
    pub num_digits_commit: usize,
    /// Gadget decomposition depth for opening-side values.
    pub num_digits_open: usize,
    /// Cached folded-witness digit count for a singleton group relation.
    pub num_digits_fold_one: usize,
}

impl PrecommittedLevelParams {
    /// Width of this group's A matrix.
    #[inline]
    pub fn inner_width(&self) -> usize {
        self.a_key.col_len()
    }

    /// Width of this group's B matrix.
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.b_key.col_len()
    }

    /// Width contribution to the shared D matrix (`w_hat_g` segment).
    pub fn d_segment_width(&self) -> Result<usize, AkitaError> {
        self.num_digits_open
            .checked_mul(self.num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("group D segment width overflow".to_string()))
    }

    /// Width contribution of this group's decomposed folded response.
    pub fn z_segment_width(&self, num_digits_fold: usize) -> Result<usize, AkitaError> {
        self.inner_width()
            .checked_mul(num_digits_fold)
            .ok_or_else(|| AkitaError::InvalidSetup("group z segment width overflow".to_string()))
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.layout.append_descriptor_bytes(bytes);
        self.a_key.append_descriptor_bytes(bytes);
        self.b_key.append_descriptor_bytes(bytes);
        push_usize(bytes, self.num_blocks);
        push_usize(bytes, self.block_len);
        push_usize(bytes, self.num_digits_commit);
        push_usize(bytes, self.num_digits_open);
        push_usize(bytes, self.num_digits_fold_one);
    }
}

/// Unified per-level parameters for one Akita recursion level.
///
/// Combines ring dimension, Ajtai matrix descriptions, block geometry,
/// sparse-challenge configuration, and digit decomposition depths into a
/// single authoritative struct.
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
    pub fold_challenge_config: SparseChallengeConfig,
    /// Shape of the stage-1 fold-round challenge vector at this level.
    ///
    /// Defaults to [`TensorChallengeShape::Flat`]. Tensor presets set selected
    /// levels to [`TensorChallengeShape::Tensor`] during schedule construction.
    pub fold_challenge_shape: TensorChallengeShape,
    /// Gadget decomposition depth for commitment coefficients (δ_commit).
    pub num_digits_commit: usize,
    /// Gadget decomposition depth for opening evaluations (δ_open).
    pub num_digits_open: usize,
    /// One-hot chunk size `K` of the committed witness at this level, used to
    /// derive the per-block witness L1 mass `nonzeros = ceil(D/K)` for the
    /// folded-witness `min(||c||_inf·||s||_1, ||c||_1·||s||_inf)` bound.
    ///
    /// `0` means the level commits a dense witness (balanced gadget digits:
    /// `||s||_inf = b/2`, `nonzeros = D`). A non-zero value `K` means the level
    /// commits a one-hot witness (`||s||_inf = 1`, `nonzeros = ceil(D/K)`);
    /// this is only ever set on a root level whose `log_commit_bound == 1`.
    pub onehot_chunk_size: usize,
    /// Level-static fold-linf cap inputs for [`crate::sis::num_digits_fold`].
    pub fold_linf_cap_config: FoldWitnessLinfCapConfig,
    /// Cached [`crate::sis::num_digits_fold`] at `num_claims = 1` for the preset
    /// field width used by the planner and setup envelope scan.
    pub num_digits_fold_one: usize,
    /// Field bit width used to populate [`Self::num_digits_fold_one`]; `0` means 128.
    pub field_bits_hint: u32,
    /// Optional cached [`crate::sis::num_digits_fold`] for a batched root `num_claims > 1`.
    pub cached_num_digits_fold_claims: usize,
    pub cached_num_digits_fold_value: usize,
    /// Multi-chunk witness layout for this level (default: single-chunk).
    ///
    /// The planner populates this from `policy.witness_chunk` and the level's
    /// position in the fold recursion; the verifier consumes it as the source of
    /// truth for the per-level witness column layout. `ChunkedWitnessCfg::default()`
    /// (single chunk) is byte-identical to the historical layout.
    pub witness_chunk: crate::witness::ChunkedWitnessCfg,
    /// Precommitted group-local params for a grouped root. Empty for scalar
    /// levels; when non-empty, the top-level fields describe the final/new
    /// group and `d_key` describes the shared D matrix over all group `w_hat`
    /// segments.
    pub precommitted_groups: Vec<PrecommittedLevelParams>,
    /// Per-role ring dimensions at this level (`d_a`, `d_b`, `d_d`).
    pub role_dims: CommitmentRingDims,
}

/// Common view over full and precommitted level parameters.
///
/// Use this trait when code only needs the shared commitment geometry carried
/// by both [`LevelParams`] and [`PrecommittedLevelParams`].
pub trait LevelParamsLike {
    fn a_rows_len(&self) -> usize;
    fn a_col_len(&self) -> usize;
    fn b_rows_len(&self) -> usize;
    fn num_blocks(&self) -> usize;
    fn block_len(&self) -> usize;
    fn num_digits_commit(&self) -> usize;
    fn num_digits_open(&self) -> usize;
    fn log_basis(&self) -> u32;
}

impl LevelParamsLike for LevelParams {
    fn a_rows_len(&self) -> usize {
        self.a_key.row_len()
    }

    fn a_col_len(&self) -> usize {
        self.a_key.col_len()
    }

    fn b_rows_len(&self) -> usize {
        self.b_key.row_len()
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn block_len(&self) -> usize {
        self.block_len
    }

    fn num_digits_commit(&self) -> usize {
        self.num_digits_commit
    }

    fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    fn log_basis(&self) -> u32 {
        self.log_basis
    }
}

impl LevelParamsLike for PrecommittedLevelParams {
    fn a_rows_len(&self) -> usize {
        self.a_key.row_len()
    }

    fn a_col_len(&self) -> usize {
        self.a_key.col_len()
    }

    fn b_rows_len(&self) -> usize {
        self.b_key.row_len()
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn block_len(&self) -> usize {
        self.block_len
    }

    fn num_digits_commit(&self) -> usize {
        self.num_digits_commit
    }

    fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    fn log_basis(&self) -> u32 {
        self.layout.log_basis
    }
}

impl LevelParams {
    /// Per-role ring dimensions at this level.
    ///
    /// Per-role ring dimensions stored on this level.
    #[must_use]
    pub fn role_dims(&self) -> CommitmentRingDims {
        self.role_dims
    }

    /// A-role ring dimension (`d_a`); alias of [`CommitmentRingDims::d_a`] on [`Self::role_dims`].
    #[inline]
    #[must_use]
    pub fn d_a(&self) -> usize {
        self.role_dims.d_a()
    }

    /// Replace per-role ring dimensions after validating nesting.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when dims are unsupported or fail nesting.
    pub fn with_role_dims(mut self, dims: CommitmentRingDims) -> Result<Self, AkitaError> {
        crate::layout::ring_dims::validate_role_dims(dims)?;
        self.role_dims = dims;
        Ok(self)
    }

    /// Derive `role_dims` from `ring_dimension` and each key's stored ring dimension.
    pub fn stamp_role_dims_from_keys(&mut self) {
        self.role_dims = CommitmentRingDims {
            inner: self.ring_dimension,
            outer: self.b_key.sis_table_key().ring_dimension as usize,
            opening: self.d_key.sis_table_key().ring_dimension as usize,
        };
    }

    /// Synthetic `LevelParams` carrying only a terminal-direct's `log_basis`.
    ///
    /// `scheduled_next_level_params` returns this stub when the next step
    /// is a terminal `Direct(SegmentTyped)`: that step does not commit
    /// anything, so it has no Ajtai keys, no block geometry, and no
    /// digit depths. The only field consumers downstream actually read is
    /// `log_basis` (used by `prove_suffix` as
    /// `final_log_basis` for the terminal fold's witness packing); every
    /// other field is left at the zero/empty defaults to make accidental
    /// use surface as obviously-degenerate output. Do not feed this stub
    /// into commitment, audit, or descriptor-binding code paths.
    pub fn log_basis_stub(log_basis: u32) -> Self {
        Self {
            ring_dimension: 0,
            log_basis,
            a_key: AjtaiKeyParams::default(),
            b_key: AjtaiKeyParams::default(),
            d_key: AjtaiKeyParams::default(),
            num_blocks: 0,
            block_len: 0,
            m_vars: 0,
            r_vars: 0,
            fold_challenge_config: SparseChallengeConfig {
                count_pm1: 0,
                count_pm2: 0,
            },
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_commit: 0,
            num_digits_open: 0,
            onehot_chunk_size: 0,
            fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
            num_digits_fold_one: 1,
            field_bits_hint: 0,
            cached_num_digits_fold_claims: 0,
            cached_num_digits_fold_value: 1,
            witness_chunk: crate::witness::ChunkedWitnessCfg::default_non_chunked(),
            precommitted_groups: Vec::new(),
            role_dims: CommitmentRingDims::uniform(0),
        }
    }

    /// Build a params-only `LevelParams` with zeroed layout fields.
    ///
    /// Only ring dimension, matrix row counts, log_basis, and fold_challenge_config
    /// are populated. Column counts, block geometry, and digit depths are
    /// zeroed. Call `with_layout` to fill them from a derived layout.
    pub fn params_only(
        sis_family: SisModulusFamily,
        ring_dimension: usize,
        log_basis: u32,
        n_a: usize,
        n_b: usize,
        n_d: usize,
        fold_challenge_config: SparseChallengeConfig,
    ) -> Self {
        Self {
            ring_dimension,
            log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_BITS,
                sis_family,
                n_a,
                0,
                0,
                ring_dimension,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_BITS,
                sis_family,
                n_b,
                0,
                0,
                ring_dimension,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_BITS,
                sis_family,
                n_d,
                0,
                0,
                ring_dimension,
            ),
            num_blocks: 0,
            block_len: 0,
            m_vars: 0,
            r_vars: 0,
            fold_challenge_config,
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_commit: 0,
            num_digits_open: 0,
            onehot_chunk_size: 0,
            fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
            num_digits_fold_one: 1,
            field_bits_hint: 0,
            cached_num_digits_fold_claims: 0,
            cached_num_digits_fold_value: 1,
            witness_chunk: crate::witness::ChunkedWitnessCfg::default_non_chunked(),
            precommitted_groups: Vec::new(),
            role_dims: CommitmentRingDims::uniform(ring_dimension),
        }
    }

    /// True when this level carries grouped-root metadata.
    #[inline]
    pub fn has_precommitted_groups(&self) -> bool {
        !self.precommitted_groups.is_empty()
    }

    /// Reject grouped-root params at scalar-only call sites.
    pub fn require_scalar_level(&self, context: &str) -> Result<(), AkitaError> {
        if self.has_precommitted_groups() {
            return Err(AkitaError::InvalidSetup(format!(
                "{context} requires scalar root level params"
            )));
        }
        Ok(())
    }

    /// Worst-case L1 mass of the fold-round challenge.
    #[inline]
    pub fn challenge_l1_mass(&self) -> usize {
        self.fold_challenge_shape
            .effective_l1_mass(&self.fold_challenge_config)
    }

    /// Per-row committed-witness `(||s||_inf, ||s||_1)` for the folded
    /// witness at this level (one-hot vs dense, see [`Self::onehot_chunk_size`]).
    #[inline]
    pub fn fold_witness_norms(&self) -> crate::sis::FoldWitnessNorms {
        let is_onehot = self.onehot_chunk_size > 0;
        crate::sis::FoldWitnessNorms::new(
            self.log_basis,
            self.ring_dimension,
            if is_onehot { self.onehot_chunk_size } else { 1 },
            is_onehot,
        )
    }

    /// Effective fold-round challenge L∞ norm `||c||_inf` at this level,
    /// accounting for the challenge shape (flat vs tensor).
    #[inline]
    pub fn challenge_infinity_norm(&self) -> usize {
        self.fold_challenge_shape
            .effective_infinity_norm(&self.fold_challenge_config)
    }

    /// Effective per-block worst-case `‖c‖_2²` upper bound at this fold level.
    #[inline]
    pub fn challenge_l2_sq_max(&self) -> u128 {
        self.fold_challenge_shape
            .effective_l2_sq_max(&self.fold_challenge_config)
    }

    /// Fold-challenge coefficient count `inner_width · D` (single shared opening point).
    #[inline]
    pub fn num_fold_coeffs(&self) -> u128 {
        (self.inner_width() as u128).saturating_mul(self.ring_dimension as u128)
    }

    /// Fold block count `num_claims · 2^r_vars` used in the tail-bound formula.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the product overflows `u128`.
    pub fn num_fold_blocks(&self, num_claims: usize) -> Result<u128, AkitaError> {
        (num_claims as u128)
            .checked_mul(self.num_blocks as u128)
            .ok_or_else(|| AkitaError::InvalidSetup("num_fold_blocks overflows u128".to_string()))
    }

    /// Fold witness L∞ cap policy for this level's sparse family and fold shape.
    #[inline]
    pub fn fold_witness_linf_cap_policy(&self) -> crate::sis::FoldWitnessLinfCapPolicy {
        crate::sis::fold_witness_linf_cap_policy(
            &self.fold_challenge_config,
            self.fold_challenge_shape,
            self.ring_dimension,
        )
    }

    /// Level-static config for [`crate::sis::fold_witness_honest_prover_linf_cap`] inside
    /// [`crate::sis::num_digits_fold`].
    #[inline]
    pub fn fold_witness_linf_cap_config(&self) -> crate::sis::FoldWitnessLinfCapConfig {
        self.fold_linf_cap_config
    }

    /// Field bit width for fold digit sizing and cached `δ_fold` values (`128` when unset).
    #[inline]
    pub fn field_bits_for_cache(&self) -> u32 {
        let hint = self.field_bits_hint;
        if hint == 0 {
            128
        } else {
            hint
        }
    }

    /// Attach the level-static fold-linf cap config derived from this layout.
    pub fn with_fold_linf_cap_config(
        mut self,
        field_bits: u32,
        root_num_claims: usize,
    ) -> Result<Self, AkitaError> {
        self.field_bits_hint = field_bits;
        self.fold_linf_cap_config = FoldWitnessLinfCapConfig::for_fold_level(
            &self.fold_challenge_config,
            self.fold_challenge_shape,
            self.ring_dimension,
            self.inner_width(),
        )?;
        let challenge = crate::sis::fold_challenge_norms(
            &self.fold_challenge_config,
            self.fold_challenge_shape,
        );
        let witness = self.fold_witness_norms();
        self.num_digits_fold_one = crate::sis::num_digits_fold(
            self.r_vars,
            1,
            field_bits,
            self.log_basis,
            challenge,
            witness,
            self.fold_linf_cap_config,
        )?;
        if root_num_claims > 1 {
            self.cached_num_digits_fold_claims = root_num_claims;
            self.cached_num_digits_fold_value = crate::sis::num_digits_fold(
                self.r_vars,
                root_num_claims,
                field_bits,
                self.log_basis,
                challenge,
                witness,
                self.fold_linf_cap_config,
            )?;
        } else {
            self.cached_num_digits_fold_claims = 0;
            self.cached_num_digits_fold_value = self.num_digits_fold_one;
        }
        Ok(self)
    }

    /// Canonical fold-l∞ digit plan for this level at `num_claims`.
    ///
    /// # Errors
    ///
    /// Propagates [`crate::sis::fold_witness_linf_digit_plan`] setup errors.
    pub fn fold_witness_linf_digit_plan_for_claims(
        &self,
        num_claims: usize,
    ) -> Result<crate::sis::FoldWitnessLinfDigitPlan, AkitaError> {
        crate::sis::fold_witness_linf_digit_plan(
            self.r_vars,
            num_claims,
            self.field_bits_for_cache(),
            self.log_basis,
            crate::sis::fold_challenge_norms(
                &self.fold_challenge_config,
                self.fold_challenge_shape,
            ),
            self.fold_witness_norms(),
            &self.fold_linf_cap_config,
        )
    }

    /// Honest-prover per-coefficient `‖z‖_inf` target for fold digit sizing, grind,
    /// and terminal Golomb-Rice (`min(β_inf, t*)` or `β_inf` alone).
    ///
    /// # Errors
    ///
    /// Propagates [`crate::sis::fold_witness_linf_digit_plan`] setup errors.
    pub fn fold_witness_linf_cap_for_claims(&self, num_claims: usize) -> Result<u128, AkitaError> {
        Ok(self
            .fold_witness_linf_digit_plan_for_claims(num_claims)?
            .grind_cap)
    }

    /// Propagates fold-beta / tail-bound rejections for tail-bound-with-grind levels.
    pub fn fold_witness_grind_contract(
        &self,
        num_claims: usize,
        max_grind_attempts: u32,
    ) -> Result<crate::sis::FoldWitnessGrindContract, AkitaError> {
        let policy = self.fold_witness_linf_cap_policy();
        let max_nonce_exclusive = match policy {
            crate::sis::FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => 1,
            crate::sis::FoldWitnessLinfCapPolicy::TailBoundWithGrind
            | crate::sis::FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => max_grind_attempts,
        };
        let witness_linf_cap = self.fold_witness_linf_cap_for_claims(num_claims)?;
        Ok(crate::sis::FoldWitnessGrindContract {
            policy,
            witness_linf_cap,
            max_nonce_exclusive,
        })
    }

    /// Domain-separated preview absorb payload for one fold-level grind search.
    pub fn fold_grind_probe_order_absorb_buf(&self, num_claims: usize) -> Vec<u8> {
        let num_claims = u32::try_from(num_claims).unwrap_or(u32::MAX);
        let mut buf = Vec::with_capacity(48);
        buf.extend_from_slice(crate::sis::FOLD_GRIND_PROBE_ORDER_ABSORB);
        buf.extend_from_slice(&(self.ring_dimension as u64).to_le_bytes());
        buf.extend_from_slice(&self.log_basis.to_le_bytes());
        buf.extend_from_slice(&(self.m_vars as u64).to_le_bytes());
        buf.extend_from_slice(&(self.r_vars as u64).to_le_bytes());
        buf.extend_from_slice(&(self.num_blocks as u64).to_le_bytes());
        buf.extend_from_slice(&num_claims.to_le_bytes());
        buf
    }

    pub fn fold_witness_linf_tail_bound_sq(&self, num_claims: usize) -> Result<u128, AkitaError> {
        let cap_config = self.fold_linf_cap_config;
        if !cap_config.policy.allows_grind() {
            return Err(AkitaError::InvalidSetup(
                "fold_witness_linf_tail_bound_sq: deterministic policy has no tail bound"
                    .to_string(),
            ));
        }
        if cap_config.num_fold_coeffs == 0 {
            return Err(AkitaError::InvalidSetup(
                "fold_witness_linf_tail_bound_sq: num_fold_coeffs must be positive".to_string(),
            ));
        }
        let witness_linf = self.fold_witness_norms().infinity_norm();
        let witness_linf_sq = witness_linf.saturating_mul(witness_linf);
        crate::sis::fold_witness_linf_tail_bound_for_config_sq(
            self.r_vars,
            num_claims,
            witness_linf_sq,
            &cap_config,
        )
    }

    /// Gadget decomposition depth for the folded witness (δ_fold / τ).
    ///
    /// Delegates to [`crate::sis::num_digits_fold`], which derives
    /// `β = num_claims · 2^r_vars · min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`
    /// from this level's fold challenge and witness norms, then applies
    /// `min(β_inf, t*)` under tail-bound-with-grind policies.
    ///
    /// # Errors
    ///
    /// Propagates [`crate::sis::num_digits_fold`]'s rejection of a degenerate
    /// fold bound (`r_vars >= 127`, `β` overflow, or `β == 0`).
    #[inline]
    pub fn num_digits_fold(&self, num_claims: usize, field_bits: u32) -> Result<usize, AkitaError> {
        if num_claims == 1 {
            return Ok(self.num_digits_fold_one);
        }
        if num_claims == self.cached_num_digits_fold_claims
            && self.cached_num_digits_fold_claims > 1
        {
            return Ok(self.cached_num_digits_fold_value);
        }
        let challenge = crate::sis::fold_challenge_norms(
            &self.fold_challenge_config,
            self.fold_challenge_shape,
        );
        crate::sis::num_digits_fold(
            self.r_vars,
            num_claims,
            field_bits,
            self.log_basis,
            challenge,
            self.fold_witness_norms(),
            self.fold_linf_cap_config,
        )
    }

    /// Set the one-hot chunk size `K`, returning the updated params.
    #[inline]
    #[must_use]
    pub fn with_onehot_chunk_size(mut self, onehot_chunk_size: usize) -> Self {
        self.onehot_chunk_size = onehot_chunk_size;
        self
    }

    /// Replace the fold-round challenge shape, rebuilding derived fold-linf
    /// digit/cache state for the new shape.
    #[inline]
    pub fn with_fold_challenge_shape(
        mut self,
        shape: TensorChallengeShape,
    ) -> Result<Self, AkitaError> {
        self.fold_challenge_shape = shape;
        let field_bits = self.field_bits_for_cache();
        let root_num_claims = self.cached_num_digits_fold_claims;
        self.with_fold_linf_cap_config(field_bits, root_num_claims)
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

    /// Total ring elements in the committed witness at this level (`num_blocks * block_len`).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] on overflow.
    pub fn n_ring_elems(&self) -> Result<usize, AkitaError> {
        self.num_blocks.checked_mul(self.block_len).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "num_blocks={} * block_len={} overflows usize",
                self.num_blocks, self.block_len,
            ))
        })
    }

    /// Total flat field-element count (`n_ring_elems * d_a`).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] on overflow.
    pub fn flat_field_len(&self) -> Result<usize, AkitaError> {
        let n_ring_elems = self.n_ring_elems()?;
        n_ring_elems.checked_mul(self.d_a()).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "n_ring_elems={n_ring_elems} * d_a={} overflows usize",
                self.d_a(),
            ))
        })
    }

    /// Append the descriptor digest encoding for this parameter set.
    ///
    /// Kept next to [`LevelParams`] so protocol-affecting field changes are
    /// reviewed with their Fiat-Shamir binding.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.ring_dimension);
        push_u32(bytes, self.log_basis);
        self.a_key.append_descriptor_bytes(bytes);
        self.b_key.append_descriptor_bytes(bytes);
        self.d_key.append_descriptor_bytes(bytes);
        push_usize(bytes, self.num_blocks);
        push_usize(bytes, self.block_len);
        push_usize(bytes, self.m_vars);
        push_usize(bytes, self.r_vars);
        append_sparse_challenge_descriptor_bytes(bytes, &self.fold_challenge_config);
        append_tensor_challenge_shape_descriptor_bytes(bytes, self.fold_challenge_shape);
        append_fold_linf_policy_descriptor_bytes(bytes, self.fold_witness_linf_cap_policy());
        push_u128(bytes, self.challenge_l2_sq_max());
        push_usize(bytes, self.num_digits_commit);
        push_usize(bytes, self.num_digits_open);
        push_usize(bytes, self.onehot_chunk_size);
        // Chunk binding is appended only when the level is chunked, so
        // single-chunk descriptors stay byte-for-byte identical to the historical
        // layout (the flag-off no-op invariant). When chunked, bind the chunk
        // count and activated-level count into the Fiat-Shamir digest.
        if self.witness_chunk.num_chunks != 1 {
            self.witness_chunk.append_descriptor_bytes(bytes);
        }

        if !self.precommitted_groups.is_empty() {
            push_usize(bytes, self.precommitted_groups.len());
            for group in &self.precommitted_groups {
                group.append_descriptor_bytes(bytes);
            }
        }
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

    /// Logical opening-point variable count for recursive fold levels.
    ///
    /// Matches [`crate::prepare_opening_point`]: outer
    /// block/position coordinates plus the inner `log2(d_a)` bits.
    ///
    /// # Errors
    ///
    /// Returns an error if the summed dimension overflows `usize`.
    pub fn recursive_opening_num_vars(&self) -> Result<usize, AkitaError> {
        let alpha_bits = self.d_a().trailing_zeros() as usize;
        self.m_vars
            .checked_add(self.r_vars)
            .and_then(|n| n.checked_add(alpha_bits))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive opening num_vars overflow".to_string())
            })
    }

    // ---- Canonical M-row layout offsets (single source of truth) ----
    //
    // Row layout: consistency (1) | A (n_a) | B (n_b · nc) | D (n_d_active).
    // Public-output rows bind through the fused trace term, not the M-matrix.
    // Every row-offset site (prover quotient/`generate_y`, setup-contribution
    // `prepare`, the relation claim, the verifier ring-switch row eval) must
    // derive its block starts from these helpers rather than recompute inline.

    /// Active D-block rows for an M-row layout (dropped at a terminal fold).
    #[inline]
    pub fn n_d_active_for(&self, layout: MRowLayout) -> usize {
        match layout {
            MRowLayout::WithDBlock => self.d_key.row_len(),
            MRowLayout::WithoutDBlock => 0,
        }
    }

    #[inline]
    fn m_row_overflow() -> AkitaError {
        AkitaError::InvalidSetup("M-row count overflow".to_string())
    }

    /// Absolute start row of the A block (immediately after the consistency row).
    #[inline]
    pub fn a_start(&self) -> usize {
        1
    }

    /// Absolute start row of the B block.
    #[inline]
    pub fn b_start(&self) -> Result<usize, AkitaError> {
        self.a_start()
            .checked_add(self.a_key.row_len())
            .ok_or_else(Self::m_row_overflow)
    }

    /// Absolute start row of the D block.
    #[inline]
    pub fn d_start(&self, num_commitments: usize) -> Result<usize, AkitaError> {
        let b_rows = self
            .b_key
            .row_len()
            .checked_mul(num_commitments)
            .ok_or_else(Self::m_row_overflow)?;
        self.b_start()?
            .checked_add(b_rows)
            .ok_or_else(Self::m_row_overflow)
    }

    fn root_group_count(&self) -> usize {
        self.precommitted_groups.len() + 1
    }

    pub fn validate_root_opening_batch(
        &self,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        opening_batch.check()?;
        if opening_batch.num_groups() != self.root_group_count() {
            return Err(AkitaError::InvalidSetup(
                "root opening group count does not match level params".to_string(),
            ));
        }
        for (group_index, group_params) in self.precommitted_groups.iter().enumerate() {
            let group_layout = opening_batch.group_layout(group_index)?;
            if *group_layout != group_params.layout.group {
                return Err(AkitaError::InvalidSetup(
                    "precommitted root group layout does not match level params".to_string(),
                ));
            }
        }
        opening_batch.root_final_group_index()
    }

    /// Sent commitment row count for one root commitment group.
    pub fn root_group_commitment_rows(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<usize, AkitaError> {
        let final_group_index = self.validate_root_opening_batch(opening_batch)?;
        if group_index == final_group_index {
            return Ok(self.b_key.row_len());
        }
        self.precommitted_groups
            .get(group_index)
            .map(|group| group.b_key.row_len())
            .ok_or(AkitaError::InvalidProof)
    }

    fn grouped_m_row_count_for(
        &self,
        num_commitments: usize,
        layout: MRowLayout,
    ) -> Result<usize, AkitaError> {
        if num_commitments != self.root_group_count() {
            return Err(AkitaError::InvalidSetup(
                "grouped root M rows require the real root group count".to_string(),
            ));
        }

        let mut rows = self
            .a_start()
            .checked_add(self.a_key.row_len())
            .and_then(|n| n.checked_add(self.b_key.row_len()))
            .ok_or_else(Self::m_row_overflow)?;
        for group in &self.precommitted_groups {
            rows = rows
                .checked_add(group.a_key.row_len())
                .and_then(|n| n.checked_add(group.b_key.row_len()))
                .ok_or_else(Self::m_row_overflow)?;
        }
        rows.checked_add(self.n_d_active_for(layout))
            .ok_or_else(Self::m_row_overflow)
    }

    /// Absolute start row of one group's A block in the grouped root layout
    /// (`consistency | A_final | B_final | A_pre* | B_pre* | D`).
    fn group_a_start(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<usize, AkitaError> {
        let final_group_index = self.validate_root_opening_batch(opening_batch)?;
        if group_index > final_group_index {
            return Err(AkitaError::InvalidProof);
        }
        if group_index == final_group_index {
            return Ok(self.a_start());
        }

        let mut start = self
            .b_start()?
            .checked_add(self.b_key.row_len())
            .ok_or_else(Self::m_row_overflow)?;
        for prior in &self.precommitted_groups[..group_index] {
            start = start
                .checked_add(prior.a_key.row_len())
                .and_then(|n| n.checked_add(prior.b_key.row_len()))
                .ok_or_else(Self::m_row_overflow)?;
        }
        Ok(start)
    }

    fn group_a_rows(
        &self,
        group_index: usize,
        final_group_index: usize,
    ) -> Result<usize, AkitaError> {
        if group_index == final_group_index {
            Ok(self.a_key.row_len())
        } else {
            Ok(self
                .precommitted_groups
                .get(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .a_key
                .row_len())
        }
    }

    fn group_b_rows(
        &self,
        group_index: usize,
        final_group_index: usize,
    ) -> Result<usize, AkitaError> {
        if group_index == final_group_index {
            Ok(self.b_key.row_len())
        } else {
            Ok(self
                .precommitted_groups
                .get(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .b_key
                .row_len())
        }
    }

    /// M-row range for one root commitment group.
    pub fn root_commitment_row_range(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
        layout: MRowLayout,
    ) -> Result<std::ops::Range<usize>, AkitaError> {
        let final_group_index = self.validate_root_opening_batch(opening_batch)?;
        let a_start = self.group_a_start(opening_batch, group_index)?;
        let n_a = self.group_a_rows(group_index, final_group_index)?;
        let n_b = self.group_b_rows(group_index, final_group_index)?;
        let start = a_start.checked_add(n_a).ok_or_else(Self::m_row_overflow)?;
        let end = start.checked_add(n_b).ok_or_else(Self::m_row_overflow)?;
        let _ = layout;
        Ok(start..end)
    }

    /// M-row range for one root A block.
    pub fn root_a_row_range(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
        layout: MRowLayout,
    ) -> Result<std::ops::Range<usize>, AkitaError> {
        let final_group_index = self.validate_root_opening_batch(opening_batch)?;
        let start = self.group_a_start(opening_batch, group_index)?;
        let rows = self.group_a_rows(group_index, final_group_index)?;
        let end = start.checked_add(rows).ok_or_else(Self::m_row_overflow)?;
        let _ = layout;
        Ok(start..end)
    }

    fn root_segment_rings(
        num_polys: usize,
        num_blocks: usize,
        block_len: usize,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        num_digits_fold: usize,
    ) -> Result<usize, AkitaError> {
        let e_hat = num_blocks
            .checked_mul(num_digits_open)
            .ok_or_else(|| AkitaError::InvalidSetup("root e-hat witness overflow".to_string()))?;
        let t_hat = num_polys
            .checked_mul(num_blocks)
            .and_then(|n| n.checked_mul(n_a))
            .and_then(|n| n.checked_mul(num_digits_open))
            .ok_or_else(|| AkitaError::InvalidSetup("root t-hat witness overflow".to_string()))?;
        let z_hat = block_len
            .checked_mul(num_digits_commit)
            .and_then(|n| n.checked_mul(num_digits_fold))
            .ok_or_else(|| AkitaError::InvalidSetup("root z-hat witness overflow".to_string()))?;

        e_hat
            .checked_add(t_hat)
            .and_then(|n| n.checked_add(z_hat))
            .ok_or_else(|| AkitaError::InvalidSetup("root witness segment overflow".to_string()))
    }

    /// Root next-witness length in field elements for scalar or grouped roots.
    pub fn root_next_w_len<F: CanonicalField>(
        &self,
        opening_batch: &OpeningClaimsLayout,
        layout: MRowLayout,
    ) -> Result<usize, AkitaError> {
        opening_batch.check()?;
        let modulus = crate::schedule::detect_field_modulus::<F>();
        let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
        if !self.has_precommitted_groups() {
            if opening_batch.num_groups() != 1 {
                return Err(AkitaError::InvalidSetup(
                    "scalar root params require a single opening group".to_string(),
                ));
            }
            return crate::schedule::w_ring_element_count_for_chunks(
                field_bits,
                self,
                opening_batch.num_total_polynomials(),
                layout,
                self.witness_chunk.num_chunks,
            )?
            .checked_mul(self.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("root next witness length overflow".to_string())
            });
        }

        let final_group_index = self.validate_root_opening_batch(opening_batch)?;
        let final_group = opening_batch.group_layout(final_group_index)?;
        let mut total = Self::root_segment_rings(
            final_group.num_polynomials(),
            self.num_blocks,
            self.block_len,
            self.a_key.row_len(),
            self.num_digits_commit,
            self.num_digits_open,
            self.num_digits_fold(final_group.num_polynomials(), field_bits)?,
        )?;
        for group in &self.precommitted_groups {
            let group_rings = Self::root_segment_rings(
                group.layout.group.num_polynomials(),
                group.num_blocks,
                group.block_len,
                group.a_key.row_len(),
                group.num_digits_commit,
                group.num_digits_open,
                group.num_digits_fold_one,
            )?;
            total = total
                .checked_add(group_rings)
                .ok_or_else(|| AkitaError::InvalidSetup("root witness overflow".to_string()))?;
        }

        let r_rows = self.m_row_count_for(opening_batch.num_groups(), layout)?;
        let r_count = r_rows
            .checked_mul(crate::sis::compute_num_digits_full_field(
                field_bits,
                self.log_basis,
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("root r-tail witness overflow".to_string()))?;
        total = total
            .checked_add(r_count)
            .ok_or_else(|| AkitaError::InvalidSetup("root witness overflow".to_string()))?;

        total.checked_mul(self.ring_dimension).ok_or_else(|| {
            AkitaError::InvalidSetup("root next witness length overflow".to_string())
        })
    }

    /// Row count for an explicit M-row layout.
    ///
    /// Scalar layout: `consistency (1) | A (n_a) | B (n_b · num_commitments)
    /// | optional D (n_d)`.
    ///
    /// Grouped-root layout: `consistency (1) | A_final | B_final | A_pre* |
    /// B_pre* | optional D`. Public openings bind through the fused trace term,
    /// not M rows.
    ///
    /// At the terminal fold the cleartext witness is shipped on the wire and
    /// the D-block is dropped from the M-matrix; see [`MRowLayout`].
    #[inline]
    pub fn m_row_count_for(
        &self,
        num_commitments: usize,
        layout: MRowLayout,
    ) -> Result<usize, AkitaError> {
        if self.has_precommitted_groups() {
            return self.grouped_m_row_count_for(num_commitments, layout);
        }
        self.require_scalar_level("m_row_count_for")?;
        self.d_start(num_commitments)?
            .checked_add(self.n_d_active_for(layout))
            .ok_or_else(Self::m_row_overflow)
    }

    /// Fill in the layout-derived fields from explicit decomposition parameters.
    ///
    /// Takes a params-only `LevelParams` (with zeroed layout fields) and
    /// computes block geometry, matrix column counts, and commit/open digit
    /// depths.
    ///
    /// When `num_ring > 0` (recursive levels), `block_len` is set to
    /// `ceil(num_ring / num_blocks)` instead of `2^m_vars`, giving tight
    /// z_folded_rings sizing. Pass `0` for root-level layouts.
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
        let rebuilt = Self {
            ring_dimension: d,
            log_basis: self.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                self.a_key.min_security_bits(),
                self.a_key.sis_family(),
                self.a_key.row_len,
                inner_width,
                self.a_key.coeff_linf_bound(),
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                self.b_key.min_security_bits(),
                self.b_key.sis_family(),
                self.b_key.row_len,
                outer_width,
                self.b_key.coeff_linf_bound(),
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                self.d_key.min_security_bits(),
                self.d_key.sis_family(),
                self.d_key.row_len,
                d_matrix_width,
                self.d_key.coeff_linf_bound(),
                d,
            ),
            num_blocks,
            block_len,
            m_vars,
            r_vars,
            fold_challenge_config: self.fold_challenge_config,
            fold_challenge_shape: self.fold_challenge_shape,
            num_digits_commit,
            num_digits_open,
            onehot_chunk_size: self.onehot_chunk_size,
            fold_linf_cap_config: self.fold_linf_cap_config,
            num_digits_fold_one: self.num_digits_fold_one,
            field_bits_hint: self.field_bits_hint,
            cached_num_digits_fold_claims: self.cached_num_digits_fold_claims,
            cached_num_digits_fold_value: self.cached_num_digits_fold_value,
            // `with_decomp` recomputes only the A/B/D widths; the chunk layout is
            // a property of the witness this level commits, so preserve it.
            witness_chunk: self.witness_chunk,
            precommitted_groups: self.precommitted_groups.clone(),
            role_dims: self.role_dims,
        };
        let field_bits = self.field_bits_for_cache();
        rebuilt.with_fold_linf_cap_config(field_bits, self.cached_num_digits_fold_claims)
    }

    /// Build a new `LevelParams` that keeps rank/ring/SIS-bucket info
    /// from `self` but replaces all layout-derived fields with those
    /// from `other`.
    ///
    /// "Layout-derived fields" are `col_len`, `num_blocks`, `block_len`,
    /// `m_vars`, `r_vars`, and the commit/open digit counts. The audited
    /// coefficient-L∞ SIS bucket is not a layout field: it is the bucket the
    /// rank (`row_len`) was sized against, so it is preserved from `self`,
    /// matching the placement of `row_len` and `sis_family`. Pulling the
    /// bucket from `other` would lose the audited value when the layout
    /// argument was constructed via [`LevelParams::params_only`] or threaded
    /// through [`Self::with_decomp`], and would let the SIS audit at
    /// [`AjtaiKeyParams::try_new`] short-circuit silently.
    pub fn with_layout(&self, other: &LevelParams, field_bits: u32) -> Result<Self, AkitaError> {
        let d = self.ring_dimension;
        Self {
            ring_dimension: d,
            log_basis: other.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                self.a_key.min_security_bits(),
                self.a_key.sis_family(),
                self.a_key.row_len,
                other.a_key.col_len,
                self.a_key.coeff_linf_bound(),
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                self.b_key.min_security_bits(),
                self.b_key.sis_family(),
                self.b_key.row_len,
                other.b_key.col_len,
                self.b_key.coeff_linf_bound(),
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                self.d_key.min_security_bits(),
                self.d_key.sis_family(),
                self.d_key.row_len,
                other.d_key.col_len,
                self.d_key.coeff_linf_bound(),
                d,
            ),
            num_blocks: other.num_blocks,
            block_len: other.block_len,
            m_vars: other.m_vars,
            r_vars: other.r_vars,
            fold_challenge_config: self.fold_challenge_config,
            fold_challenge_shape: other.fold_challenge_shape,
            num_digits_commit: other.num_digits_commit,
            num_digits_open: other.num_digits_open,
            onehot_chunk_size: other.onehot_chunk_size,
            fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
            num_digits_fold_one: 1,
            field_bits_hint: 0,
            cached_num_digits_fold_claims: 0,
            cached_num_digits_fold_value: 1,
            // The chunk layout is a property of the committed witness, sized with
            // the ranks, so it stays with `self` like the SIS buckets.
            witness_chunk: self.witness_chunk,
            precommitted_groups: self.precommitted_groups.clone(),
            role_dims: other.role_dims,
        }
        .with_fold_linf_cap_config(field_bits, 0)
    }
}

fn append_sparse_challenge_descriptor_bytes(bytes: &mut Vec<u8>, config: &SparseChallengeConfig) {
    bytes.push(0);
    push_usize(bytes, config.count_pm1);
    push_usize(bytes, config.count_pm2);
}

fn append_fold_linf_policy_descriptor_bytes(
    bytes: &mut Vec<u8>,
    policy: crate::sis::FoldWitnessLinfCapPolicy,
) {
    bytes.push(match policy {
        crate::sis::FoldWitnessLinfCapPolicy::TailBoundWithGrind => 0,
        crate::sis::FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => 1,
        crate::sis::FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => 2,
    });
}

fn append_tensor_challenge_shape_descriptor_bytes(
    bytes: &mut Vec<u8>,
    shape: TensorChallengeShape,
) {
    match shape {
        TensorChallengeShape::Flat => bytes.push(0),
        TensorChallengeShape::Tensor => bytes.push(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PolynomialGroupLayout;

    fn sample_params_only() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::pm1_only(3),
        )
    }

    fn sample_layout_lp() -> LevelParams {
        sample_params_only().with_decomp(4, 2, 2, 2, 0).unwrap()
    }

    #[test]
    fn with_layout_keeps_self_ranks() {
        let params = sample_params_only();
        let layout_lp = sample_layout_lp();

        let lp = params.with_layout(&layout_lp, 128).unwrap();

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
    }

    #[test]
    fn derived_widths_match_ajtai_col_len() {
        let lp = sample_params_only()
            .with_layout(&sample_layout_lp(), 128)
            .unwrap();

        assert_eq!(lp.inner_width(), lp.a_key.col_len());
        assert_eq!(lp.outer_width(), lp.b_key.col_len());
        assert_eq!(lp.d_matrix_width(), lp.d_key.col_len());
    }

    #[test]
    fn with_fold_linf_cap_config_propagates_fold_digit_errors() {
        let mut lp = sample_layout_lp();
        lp.fold_challenge_config = SparseChallengeConfig::pm1_only(0);

        let err = lp
            .with_fold_linf_cap_config(128, 1)
            .expect_err("zero challenge mass must reject");

        assert!(matches!(err, AkitaError::InvalidSetup(message) if message.contains("β = 0")));
    }

    #[test]
    fn derived_log_values() {
        let layout_lp = sample_layout_lp();
        let lp = sample_params_only().with_layout(&layout_lp, 128).unwrap();

        assert_eq!(lp.log_num_blocks(), layout_lp.r_vars);
        assert_eq!(lp.log_block_len(), layout_lp.m_vars);
        assert_eq!(lp.outer_vars(), layout_lp.m_vars + layout_lp.r_vars);
    }

    #[test]
    fn m_row_count_values() {
        let lp = sample_params_only()
            .with_layout(&sample_layout_lp(), 128)
            .unwrap();

        assert_eq!(
            lp.m_row_count_for(1, MRowLayout::WithDBlock).unwrap(),
            1 + 3 + 4 + 2
        );
        assert_eq!(
            lp.m_row_count_for(2, MRowLayout::WithDBlock).unwrap(),
            1 + 3 + 4 * 2 + 2
        );
        assert_eq!(
            lp.m_row_count_for(4, MRowLayout::WithDBlock).unwrap(),
            1 + 3 + 4 * 4 + 2
        );
        assert_eq!(
            lp.m_row_count_for(2, MRowLayout::WithoutDBlock).unwrap(),
            1 + 4 * 2 + 2
        );
    }

    #[test]
    fn canonical_row_offsets_match_open_coded_layout() {
        let lp = sample_params_only()
            .with_layout(&sample_layout_lp(), 128)
            .unwrap();
        let n_a = lp.a_key.row_len();
        let n_b = lp.b_key.row_len();
        let n_d = lp.d_key.row_len();

        for nc in [1usize, 2, 4] {
            for layout in [MRowLayout::WithDBlock, MRowLayout::WithoutDBlock] {
                let n_d_active = match layout {
                    MRowLayout::WithDBlock => n_d,
                    MRowLayout::WithoutDBlock => 0,
                };
                let a_start = 1;
                let b_start = a_start + n_a;
                let d_start = b_start + n_b * nc;

                assert_eq!(lp.a_start(), a_start);
                assert_eq!(lp.b_start().unwrap(), b_start);
                assert_eq!(lp.d_start(nc).unwrap(), d_start);
                assert_eq!(
                    lp.m_row_count_for(nc, layout).unwrap(),
                    d_start + n_d_active
                );
            }
        }
    }

    #[path = "params_grouped_tests.rs"]
    mod grouped_tests;
}
