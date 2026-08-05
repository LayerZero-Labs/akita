//! Per-level and per-schedule ring dimension validation.
//!
//! [`validate_schedule_ring_dims`] checks every fold level's [`CommitmentRingDims`]
//! against the setup seed. Per-level geometry (`n_ring_elems`, `flat_field_len`, …)
//! lives on [`super::CommittedGroupParams`].

use crate::proof::AkitaSetupSeed;
use crate::schedule::FoldSchedule;
use akita_field::AkitaError;

/// Upper bound on fold levels accepted by [`validate_schedule_ring_dims`].
pub const MAX_FOLD_LEVELS: usize = 16;

/// Ring dimensions valid for A-role (`d_a`) sparse fold challenges.
pub const SUPPORTED_CHALLENGE_RING_DIMS: &[usize] =
    akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS;

/// Ring dimensions implemented by the shared ring and NTT layers.
///
/// Protocol field tiers may impose a higher role-specific floor.
pub const SUPPORTED_RING_DIMS: [usize; 8] = [16, 32, 64, 128, 256, 512, 1024, 2048];

/// Minimum `d_a` for sparse fold ring challenges (no sampler below this).
pub const MIN_A_ROLE_FOLD_CHALLENGE_RING_D: usize = 64;

/// Which commitment matrix owns a buffer or ring dimension at one fold level.
///
/// “Role” is the historical protocol name for the fixed job of one of the
/// three Ajtai matrices. It does not mean that a matrix changes jobs between
/// levels: A always carries the relation witness, B always commits the next
/// witness, and D always commits the opening digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingRole {
    /// A matrix (`d_a`): fold witness, row coefficients, ring-switch geometry.
    Inner,
    /// B matrix (`d_b`): sent commitment rows, COMMIT segment of `y`.
    Outer,
    /// D matrix (`d_d`): opening digits, D-block rows `v`.
    Opening,
}

/// Per-fold ring dimensions of the A, B, and D commitment matrices.
///
/// Z, E, and T are produced over the A-native source ring. B- and D-native
/// projections therefore require `inner >= outer` and `inner >= opening`.
/// B and D are independent: neither is ordered relative to the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRingDims {
    /// Fold / ring-switch / inner-commitment ring (`d_a`).
    pub inner: usize,
    /// Outer-commitment ring (`d_b`).
    pub outer: usize,
    /// Opening-commitment ring (`d_d`).
    pub opening: usize,
}

impl CommitmentRingDims {
    #[must_use]
    pub const fn uniform(d: usize) -> Self {
        Self {
            inner: d,
            outer: d,
            opening: d,
        }
    }

    /// Validate the representation-level native projection geometry
    /// independently of production dispatch and challenge catalogs.
    ///
    /// This permits small dimensions in arithmetic/layout tests while keeping
    /// the same ordering contract used by [`validate_role_dims`].
    pub fn validate_role_projection(self) -> Result<(), AkitaError> {
        for (role, d) in [
            (RingRole::Inner, self.inner),
            (RingRole::Outer, self.outer),
            (RingRole::Opening, self.opening),
        ] {
            if d == 0 || !d.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(format!(
                    "{role:?} ring dimension must be a non-zero power of two, got {d}"
                )));
            }
        }
        if self.outer > self.inner || self.opening > self.inner {
            return Err(AkitaError::InvalidSetup(format!(
                "A-role ring dimension d_a={} must be at least d_b={} and d_d={}",
                self.inner, self.outer, self.opening
            )));
        }
        Ok(())
    }

    /// Ring dimension for A-role data: the folded witness `z`, A quotient
    /// rows, the consistency row, and fold/ring-switch arithmetic.
    #[must_use]
    pub const fn d_a(self) -> usize {
        self.inner
    }

    /// Ring dimension for B-role data: next-witness digit commitments (`t_hat`),
    /// COMMIT and B_inner relation rows.
    #[must_use]
    pub const fn d_b(self) -> usize {
        self.outer
    }

    /// Ring dimension for D-role data: opening digits (`e_hat`) and the
    /// D-block relation rows (`v = D * e_hat`).
    #[must_use]
    pub const fn d_d(self) -> usize {
        self.opening
    }

    /// Largest low coefficient block shared by every relation role.
    ///
    /// This is the relation-algebra boundary: alpha exponents for every role
    /// reset on multiples of this count. It does not depend on the outgoing
    /// witness representation.
    #[must_use]
    pub const fn common_relation_coeff_count(self) -> usize {
        let inner_outer_min = if self.inner < self.outer {
            self.inner
        } else {
            self.outer
        };
        if inner_outer_min < self.opening {
            inner_outer_min
        } else {
            self.opening
        }
    }

    /// The single dimension shared by all matrices, or an error once their
    /// dimensions diverge.
    pub fn uniform_dim(self) -> Result<usize, AkitaError> {
        if self.inner == self.outer && self.outer == self.opening {
            Ok(self.inner)
        } else {
            Err(AkitaError::InvalidSetup(format!(
                "fused ring path requires uniform role dims, got d_a={} d_b={} d_d={}",
                self.inner, self.outer, self.opening
            )))
        }
    }

    /// Ring dimension for `role`.
    #[must_use]
    pub const fn dim_for(self, role: RingRole) -> usize {
        match role {
            RingRole::Inner => self.inner,
            RingRole::Outer => self.outer,
            RingRole::Opening => self.opening,
        }
    }
}

/// Validate every fold level's per-role ring dimensions against the setup seed.
///
/// Reads [`super::CommittedGroupParams::role_dims`] from each scheduled fold step; does
/// not copy them into a separate plan object.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when any catalog, key-consistency,
/// seed-divisibility, or witness-length check fails.
pub fn validate_schedule_ring_dims(
    schedule: &FoldSchedule,
    seed: &AkitaSetupSeed,
) -> Result<(), AkitaError> {
    if seed.gen_ring_dim == 0 {
        return Err(AkitaError::InvalidSetup(
            "gen_ring_dim must be non-zero".to_string(),
        ));
    }
    let num_folds = schedule.num_fold_levels();
    if num_folds > MAX_FOLD_LEVELS {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule has {num_folds} fold levels, max supported is {MAX_FOLD_LEVELS}"
        )));
    }
    let validate_step = |lp: &crate::CommittedGroupParams,
                         input_witness_len: usize,
                         output_witness_len: Option<usize>,
                         next_ring_d: Option<usize>|
     -> Result<(), AkitaError> {
        let dims = lp.role_dims();
        validate_role_dims(dims)?;
        validate_role_dims_match_keys(lp)?;
        for (role, d) in [
            (RingRole::Inner, dims.inner),
            (RingRole::Outer, dims.outer),
            (RingRole::Opening, dims.opening),
        ] {
            if !seed.gen_ring_dim.is_multiple_of(d) {
                return Err(AkitaError::InvalidSetup(format!(
                    "setup gen_ring_dim={} is not divisible by {:?} ring d={d}",
                    seed.gen_ring_dim, role
                )));
            }
        }
        if input_witness_len == 0 {
            return Err(AkitaError::InvalidSetup(format!(
                "witness length {} is invalid for fold ring d_a={}",
                input_witness_len, dims.inner
            )));
        }
        if let (Some(output_witness_len), Some(next_ring_d)) = (output_witness_len, next_ring_d) {
            if next_ring_d == 0 || output_witness_len == 0 {
                return Err(AkitaError::InvalidSetup(format!(
                    "next witness length {} is invalid for next fold ring d_a={next_ring_d}",
                    output_witness_len,
                )));
            }
        }
        Ok(())
    };
    let root_next_d = schedule.recursive_folds.first().map_or_else(
        || schedule.terminal.params.witness.d_a(),
        |next| next.params.witness.d_a(),
    );
    validate_step(
        &schedule.root.params.final_group.commitment,
        schedule.root.input_witness_len,
        Some(schedule.root.output_witness_len),
        Some(root_next_d),
    )?;
    let root = &schedule.root.params;
    let final_params = &root.final_group.commitment;
    if root.open_commit_matrix != final_params.open_commit_matrix {
        return Err(AkitaError::InvalidSetup(
            "root shared D matrix disagrees with final-group commitment params".to_string(),
        ));
    }
    let shared_d = root.open_commit_matrix.ring_dimension();
    for (group_index, group) in root.precommitted_groups.iter().enumerate() {
        if group.descriptor != group.commitment.layout {
            return Err(AkitaError::InvalidSetup(format!(
                "root precommitted group {group_index} descriptor disagrees with its commitment params"
            )));
        }
        group.descriptor.validate_frozen_precommit(
            group
                .commitment
                .layout
                .inner_commit_matrix
                .sis_modulus_profile()
                .field_bits(),
        )?;
        group.commitment.validate()?;
        if group.commitment.log_basis_open != final_params.log_basis_open {
            return Err(AkitaError::InvalidSetup(format!(
                "root precommitted group {group_index} opening basis disagrees with the batch-shared basis"
            )));
        }
        if group.commitment.fold_challenge_config.weight() == 0 {
            return Err(AkitaError::InvalidSetup(format!(
                "root precommitted group {group_index} has an empty fold challenge"
            )));
        }
        let dims = group.commitment.role_dims(shared_d);
        validate_role_dims(dims)?;
        for (role, d) in [
            (RingRole::Inner, dims.d_a()),
            (RingRole::Outer, dims.d_b()),
            (RingRole::Opening, dims.d_d()),
        ] {
            if !seed.gen_ring_dim.is_multiple_of(d) {
                return Err(AkitaError::InvalidSetup(format!(
                    "setup gen_ring_dim={} is not divisible by root precommitted group {group_index} {role:?} ring d={d}",
                    seed.gen_ring_dim
                )));
            }
        }
    }
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        let next_ring_d = schedule.recursive_folds.get(index + 1).map_or_else(
            || schedule.terminal.params.witness.d_a(),
            |next| next.params.witness.d_a(),
        );
        validate_step(
            &step.params.witness,
            step.input_witness_len,
            Some(step.output_witness_len),
            Some(next_ring_d),
        )?;
    }
    let terminal = &schedule.terminal.params.witness;
    let terminal_d = terminal.d_a();
    if terminal_d == 0
        || !SUPPORTED_RING_DIMS.contains(&terminal_d)
        || !seed.gen_ring_dim.is_multiple_of(terminal_d)
        || !schedule
            .terminal
            .input_witness_len
            .is_multiple_of(terminal_d)
        || terminal.inner_commit_matrix.sis_table_key().ring_dimension as usize != terminal_d
    {
        return Err(AkitaError::InvalidSetup(
            "terminal inner ring dimension is inconsistent with setup or witness length"
                .to_string(),
        ));
    }
    Ok(())
}

pub fn validate_role_dims_match_keys(lp: &crate::CommittedGroupParams) -> Result<(), AkitaError> {
    let dims = lp.role_dims();
    let a_ring = lp.inner_commit_matrix.sis_table_key().ring_dimension as usize;
    let b_ring = lp.outer_commit_matrix.sis_table_key().ring_dimension as usize;
    let d_ring = lp.open_commit_matrix.sis_table_key().ring_dimension as usize;
    if a_ring != dims.inner {
        return Err(AkitaError::InvalidSetup(format!(
            "A-key ring dimension {a_ring} disagrees with role_dims.d_a={}",
            dims.inner
        )));
    }
    if b_ring != dims.outer {
        return Err(AkitaError::InvalidSetup(format!(
            "B-key ring dimension {b_ring} disagrees with role_dims.d_b={}",
            dims.outer
        )));
    }
    if d_ring != dims.opening {
        return Err(AkitaError::InvalidSetup(format!(
            "D-key ring dimension {d_ring} disagrees with role_dims.d_d={}",
            dims.opening
        )));
    }
    lp.fold_challenge_config
        .validate_for_ring_dim(lp.d_a())
        .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))?;
    Ok(())
}

pub fn validate_role_dims(dims: CommitmentRingDims) -> Result<(), AkitaError> {
    dims.validate_role_projection()?;
    if !SUPPORTED_CHALLENGE_RING_DIMS.contains(&dims.inner) {
        return Err(AkitaError::InvalidSetup(format!(
            "A-role ring dimension d_a={} is unsupported for sparse fold challenges (need d_a >= {MIN_A_ROLE_FOLD_CHALLENGE_RING_D})",
            dims.inner
        )));
    }
    for (role, d) in [
        (RingRole::Outer, dims.outer),
        (RingRole::Opening, dims.opening),
    ] {
        if !SUPPORTED_RING_DIMS.contains(&d) {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported {:?} ring dimension {d}",
                role
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "ring_dims_tests.rs"]
mod tests;
