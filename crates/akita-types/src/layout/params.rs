//! Unified per-level parameters for the Akita protocol.
//!
//! `LevelParams` merges ring dimension, matrix ranks, challenge config,
//! block geometry, and digit depths into a single struct that fully
//! describes one recursion level.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use crate::descriptor_bytes::{push_i8, push_u32, push_usize, sis_family_tag};

pub use crate::generated::sis_floor::SisModulusFamily;

/// Per-level M-matrix row layout selector.
///
/// At an intermediate fold the prover ships a fresh commitment for the next
/// witness; the verifier never sees `w_hat` in cleartext and the D-block rows
/// `v = D * w_hat` must appear in the M-matrix to bind `w_hat` into the
/// sumcheck.
///
/// At a terminal fold the cleartext witness is absorbed into the transcript
/// and shipped on the wire, so the verifier evaluates the final witness
/// directly. Keeping the D-block in the relation would be vestigial; this enum
/// lets the prover, verifier, and planner agree to drop it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MRowLayout {
    /// Full layout including the D-block (`v = D * w_hat` rows). Used at every
    /// intermediate fold level and at the root when stage-1 runs.
    WithDBlock,
    /// Cleartext-witness layout: omit the D-block from the M-matrix. Used at
    /// the terminal fold level where `final_witness` ships on the wire.
    WithoutDBlock,
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
    sis_family: SisModulusFamily,
}

impl AjtaiKeyParams {
    fn sis_security_violation(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Option<String> {
        if col_len > 0 && collision_inf > 0 && row_len > 0 {
            use crate::generated::sis_floor::min_rank_for_secure_width;
            if let Some(floor) = min_rank_for_secure_width(
                sis_family,
                ring_dimension as u32,
                collision_inf,
                col_len as u64,
            ) {
                if row_len < floor {
                    return Some(format!(
                        "AjtaiKeyParams: row_len {row_len} < SIS floor {floor} \
                         (family={sis_family:?}, d={ring_dimension}, \
                         collision_inf={collision_inf}, col_len={col_len})"
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
    /// given `(sis_family, ring_dimension, collision_inf, col_len)` tuple.
    pub fn new(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Self {
        if let Some(message) = Self::sis_security_violation(
            sis_family,
            row_len,
            col_len,
            collision_inf,
            ring_dimension,
        ) {
            panic!("{message}");
        }
        Self {
            row_len,
            col_len,
            collision_inf,
            sis_family,
        }
    }

    /// Create a new `AjtaiKeyParams`, returning an error on SIS violations.
    ///
    /// # Errors
    ///
    /// Returns an error if `row_len` is below the 128-bit SIS security floor
    /// for the given `(sis_family, ring_dimension, collision_inf, col_len)` tuple.
    pub fn try_new(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if let Some(message) = Self::sis_security_violation(
            sis_family,
            row_len,
            col_len,
            collision_inf,
            ring_dimension,
        ) {
            return Err(AkitaError::InvalidSetup(message));
        }
        Ok(Self {
            row_len,
            col_len,
            collision_inf,
            sis_family,
        })
    }

    /// Create a new `AjtaiKeyParams` without enforcing SIS security.
    ///
    /// Logs a debug-build warning if `row_len` is below the SIS floor but does
    /// not panic. Use this for intermediate construction steps where ranks
    /// have not yet converged (e.g., batched scaling, iterative SIS fixed-point
    /// loops).
    pub fn new_unchecked(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Self {
        #[cfg(not(debug_assertions))]
        let _ = ring_dimension;
        #[cfg(debug_assertions)]
        if col_len > 0 && collision_inf > 0 && row_len > 0 {
            use crate::generated::sis_floor::min_rank_for_secure_width;
            if let Some(floor) = min_rank_for_secure_width(
                sis_family,
                ring_dimension as u32,
                collision_inf,
                col_len as u64,
            ) {
                if row_len < floor {
                    tracing::warn!(
                        ?sis_family,
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
            sis_family,
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

    /// SIS modulus family used to validate this key.
    #[inline]
    pub fn sis_family(&self) -> SisModulusFamily {
        self.sis_family
    }

    fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(sis_family_tag(self.sis_family()));
        push_usize(bytes, self.row_len());
        push_usize(bytes, self.col_len());
        push_u32(bytes, self.collision_inf());
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
    /// Stage-1 sparse challenge family sampled at this level.
    pub stage1_config: SparseChallengeConfig,
    /// Shape of the stage-1 fold-round challenge vector at this level.
    ///
    /// Defaults to [`TensorChallengeShape::Flat`]. Tensor presets set selected
    /// levels to [`TensorChallengeShape::Tensor`] during schedule construction.
    pub fold_challenge_shape: TensorChallengeShape,
    /// Gadget decomposition depth for commitment coefficients (δ_commit).
    pub num_digits_commit: usize,
    /// Gadget decomposition depth for opening evaluations (δ_open).
    pub num_digits_open: usize,
    /// Gadget decomposition depth for the folded witness (δ_fold / τ).
    pub num_digits_fold: usize,
}

impl LevelParams {
    /// Build a params-only `LevelParams` with zeroed layout fields.
    ///
    /// Only ring dimension, matrix row counts, log_basis, and stage1_config
    /// are populated. Column counts, block geometry, and digit depths are
    /// zeroed. Call `with_layout` to fill them from a derived layout.
    pub fn params_only(
        sis_family: SisModulusFamily,
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
                sis_family,
                ..Default::default()
            },
            b_key: AjtaiKeyParams {
                row_len: n_b,
                sis_family,
                ..Default::default()
            },
            d_key: AjtaiKeyParams {
                row_len: n_d,
                sis_family,
                ..Default::default()
            },
            num_blocks: 0,
            block_len: 0,
            m_vars: 0,
            r_vars: 0,
            stage1_config,
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_commit: 0,
            num_digits_open: 0,
            num_digits_fold: 0,
        }
    }

    /// Worst-case L1 mass of the fold-round challenge.
    #[inline]
    pub fn challenge_l1_mass(&self) -> usize {
        self.fold_challenge_shape
            .effective_l1_mass(&self.stage1_config)
    }

    /// Replace the fold-round challenge shape, returning the updated params.
    #[inline]
    #[must_use]
    pub fn with_fold_challenge_shape(mut self, shape: TensorChallengeShape) -> Self {
        self.fold_challenge_shape = shape;
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
        append_sparse_challenge_descriptor_bytes(bytes, &self.stage1_config);
        push_usize(bytes, self.num_digits_commit);
        push_usize(bytes, self.num_digits_open);
        push_usize(bytes, self.num_digits_fold);
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

    /// Row count with `num_commitments` explicit commitment vectors and
    /// `num_public_outputs` public y-rows.
    ///
    /// Row layout: consistency (1) | public (num_public_outputs) | D (n_d) |
    /// B (n_b · num_commitments) | A (n_a).  The batched CWSS protocol
    /// uses one public y-row per distinct opening point.
    #[inline]
    pub fn m_row_count(
        &self,
        num_commitments: usize,
        num_public_outputs: usize,
    ) -> Result<usize, AkitaError> {
        self.m_row_count_for(num_commitments, num_public_outputs, MRowLayout::WithDBlock)
    }

    /// Row count for an explicit M-row layout.
    ///
    /// At the terminal fold the cleartext witness is shipped on the wire and
    /// the D-block is dropped from the M-matrix; see [`MRowLayout`].
    #[inline]
    pub fn m_row_count_for(
        &self,
        num_commitments: usize,
        num_public_outputs: usize,
        layout: MRowLayout,
    ) -> Result<usize, AkitaError> {
        let n_d_active = match layout {
            MRowLayout::WithDBlock => self.d_key.row_len(),
            MRowLayout::WithoutDBlock => 0,
        };
        n_d_active
            .checked_add(
                self.b_key
                    .row_len()
                    .checked_mul(num_commitments)
                    .ok_or_else(|| AkitaError::InvalidSetup("M-row count overflow".to_string()))?,
            )
            .and_then(|rows| rows.checked_add(num_public_outputs))
            .and_then(|rows| rows.checked_add(1))
            .and_then(|rows| rows.checked_add(self.a_key.row_len()))
            .ok_or_else(|| AkitaError::InvalidSetup("M-row count overflow".to_string()))
    }

    /// Fill in the layout-derived fields from explicit decomposition parameters.
    ///
    /// Takes a params-only `LevelParams` (with zeroed layout fields) and
    /// computes block geometry, matrix column counts, and digit depths.
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
                self.a_key.sis_family,
                self.a_key.row_len,
                inner_width,
                self.a_key.collision_inf,
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                self.b_key.sis_family,
                self.b_key.row_len,
                outer_width,
                self.b_key.collision_inf,
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                self.d_key.sis_family,
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
            fold_challenge_shape: self.fold_challenge_shape,
            num_digits_commit,
            num_digits_open,
            num_digits_fold,
        })
    }

    /// Build a new `LevelParams` that keeps rank/ring info from `self` but
    /// replaces all layout-derived fields with those from `other`.
    pub fn with_layout(&self, other: &LevelParams) -> Self {
        let d = self.ring_dimension;
        Self {
            ring_dimension: d,
            log_basis: other.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                self.a_key.sis_family,
                self.a_key.row_len,
                other.a_key.col_len,
                other.a_key.collision_inf,
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                self.b_key.sis_family,
                self.b_key.row_len,
                other.b_key.col_len,
                other.b_key.collision_inf,
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                self.d_key.sis_family,
                self.d_key.row_len,
                other.d_key.col_len,
                other.d_key.collision_inf,
                d,
            ),
            num_blocks: other.num_blocks,
            block_len: other.block_len,
            m_vars: other.m_vars,
            r_vars: other.r_vars,
            stage1_config: self.stage1_config.clone(),
            fold_challenge_shape: other.fold_challenge_shape,
            num_digits_commit: other.num_digits_commit,
            num_digits_open: other.num_digits_open,
            num_digits_fold: other.num_digits_fold,
        }
    }
}

fn append_sparse_challenge_descriptor_bytes(bytes: &mut Vec<u8>, config: &SparseChallengeConfig) {
    match config {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => {
            bytes.push(0);
            push_usize(bytes, *weight);
            push_usize(bytes, nonzero_coeffs.len());
            for &coeff in nonzero_coeffs {
                push_i8(bytes, coeff);
            }
        }
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => {
            bytes.push(1);
            push_usize(bytes, *count_mag1);
            push_usize(bytes, *count_mag2);
        }
        SparseChallengeConfig::BoundedL1Norm => {
            bytes.push(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params_only() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
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

        assert_eq!(lp.m_row_count(1, 1).unwrap(), 3 + 4 + 1 + 1 + 2);
        assert_eq!(lp.m_row_count(2, 5).unwrap(), 3 + 4 * 2 + 5 + 1 + 2);
        assert_eq!(lp.m_row_count(4, 4).unwrap(), 3 + 4 * 4 + 4 + 1 + 2);
        assert_eq!(
            lp.m_row_count_for(2, 5, MRowLayout::WithoutDBlock).unwrap(),
            4 * 2 + 5 + 1 + 2
        );
    }
}
