//! Validated minimal geometry for the indexed binary column-address seam.

use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::{GrindingPlan, GrindingRun};

/// One group's immutable column geometry and priced transcript sites.
///
/// This is an address plan, not the opening's group roster or a schedule
/// admission certificate. The later driver validates descriptor geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BinaryColumnGroupPlan {
    group: u32,
    owners: usize,
    columns: usize,
    coordinate_count: usize,
    response_run: GrindingRun,
    work_run: GrindingRun,
    draw_run: GrindingRun,
}

impl BinaryColumnGroupPlan {
    /// Validate resource limits before deriving counts or work prices.
    pub fn new(group: u32, owners: usize, columns: usize) -> Result<Self, AkitaError> {
        if !(1..=4096).contains(&owners) || !(1..=256).contains(&columns) {
            return Err(AkitaError::InvalidSetup(
                "binary column owners or columns exceed their resource limits".into(),
            ));
        }
        let coordinate_count =
            checked::product([owners, columns]).ok_or(AkitaError::InvalidProof)?;
        let coordinates = u64::try_from(coordinate_count).map_err(|_| AkitaError::InvalidProof)?;
        Ok(Self {
            group,
            owners,
            columns,
            coordinate_count,
            response_run: GrindingRun::binary_group_response(0, group)?,
            work_run: GrindingRun::binary_group_challenge_work(0, group, coordinates)?,
            draw_run: GrindingRun::fold_challenge_group(0, group, coordinates)?,
        })
    }

    /// Session-local binary group occurrence.
    #[must_use]
    pub const fn group(self) -> u32 {
        self.group
    }

    /// Number of owners in this group.
    #[must_use]
    pub const fn owners(self) -> usize {
        self.owners
    }

    /// Number of columns for each owner.
    #[must_use]
    pub const fn columns(self) -> usize {
        self.columns
    }

    /// Exact number of indexed coordinates in this group.
    #[must_use]
    pub const fn coordinate_count(self) -> usize {
        self.coordinate_count
    }

    /// One bounded response search at this group occurrence.
    #[must_use]
    pub const fn response_run(self) -> GrindingRun {
        self.response_run
    }

    /// One exactly priced work site, including a present zero-bit site.
    #[must_use]
    pub const fn work_run(self) -> GrindingRun {
        self.work_run
    }

    /// Replay entry accounting for the group's indexed coordinate queries.
    #[must_use]
    pub const fn draw_run(self) -> GrindingRun {
        self.draw_run
    }

    /// Owner-major support offset, checked before indexing a support vector.
    pub fn coordinate_index(self, owner: usize, column: usize) -> Result<usize, AkitaError> {
        if owner >= self.owners || column >= self.columns {
            return Err(AkitaError::InvalidInput(
                "binary column coordinate is outside its group plan".into(),
            ));
        }
        checked::mul_add(owner, self.columns, column).ok_or(AkitaError::InvalidProof)
    }
}

/// Ordered group address plans and their single replay schedule.
///
/// This minimal input deliberately carries no commitment or owner roster.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BinaryColumnPlan {
    groups: Vec<BinaryColumnGroupPlan>,
    grinding_plan: GrindingPlan,
    coordinate_count: usize,
}

impl BinaryColumnPlan {
    /// Validate contiguous group order and aggregate limits before allocating
    /// the replay schedule. Groups retain their declared order.
    pub fn new(groups: Vec<BinaryColumnGroupPlan>) -> Result<Self, AkitaError> {
        if groups.is_empty() || groups.len() > 64 {
            return Err(AkitaError::InvalidSetup(
                "binary column plan requires between one and 64 groups".into(),
            ));
        }
        for (index, group) in groups.iter().enumerate() {
            if usize::try_from(group.group).ok() != Some(index) {
                return Err(AkitaError::InvalidSetup(
                    "binary column groups must have contiguous ordered indices".into(),
                ));
            }
        }
        let owners = checked::sum(groups.iter().map(|group| group.owners))
            .ok_or(AkitaError::InvalidProof)?;
        let coordinate_count = checked::sum(groups.iter().map(|group| group.coordinate_count))
            .ok_or(AkitaError::InvalidProof)?;
        if owners > 4096 || coordinate_count > 1_048_576 {
            return Err(AkitaError::InvalidSetup(
                "binary column plan exceeds aggregate resource limits".into(),
            ));
        }
        let capacity = checked::product([groups.len(), 3]).ok_or(AkitaError::InvalidProof)?;
        let mut runs = Vec::with_capacity(capacity);
        for group in &groups {
            runs.extend([group.response_run, group.work_run, group.draw_run]);
        }
        Ok(Self {
            groups,
            grinding_plan: GrindingPlan::new(runs, BinaryFieldProfile::P64.challenge_order()?)?,
            coordinate_count,
        })
    }

    /// Groups in the canonical replay and support order.
    #[must_use]
    pub fn groups(&self) -> &[BinaryColumnGroupPlan] {
        &self.groups
    }

    /// The one schedule that owns all counts and group-local work prices.
    #[must_use]
    pub const fn grinding_plan(&self) -> &GrindingPlan {
        &self.grinding_plan
    }

    /// Total support count across the validated groups.
    #[must_use]
    pub const fn coordinate_count(&self) -> usize {
        self.coordinate_count
    }

    /// Versioned statement binding for geometry and the complete grinding plan.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AkitaError> {
        let mut out = b"akita/binary-source/multigroup/column-plan/v1".to_vec();
        let count = u64::try_from(self.groups.len()).map_err(|_| AkitaError::InvalidProof)?;
        out.extend_from_slice(&count.to_le_bytes());
        for group in &self.groups {
            for value in [
                u64::from(group.group),
                u64::try_from(group.owners).map_err(|_| AkitaError::InvalidProof)?,
                u64::try_from(group.columns).map_err(|_| AkitaError::InvalidProof)?,
            ] {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        let grinding = self.grinding_plan.canonical_bytes()?;
        let len = u64::try_from(grinding.len()).map_err(|_| AkitaError::InvalidProof)?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&grinding);
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
