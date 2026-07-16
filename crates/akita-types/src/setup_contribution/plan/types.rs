use super::kernels::GroupSetupSegment;
use crate::{SetupContributionPlanInputs, SetupProjectionGeometry, WitnessLayout};
use akita_field::{AkitaError, FieldCore};
use std::{ops::Range, sync::Arc};

#[derive(Clone)]
pub struct SetupContributionGroupInputs {
    pub group_id: usize,
    pub num_claims: usize,
    pub live_fold_count: usize,
    pub fold_position_count: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub n_b: usize,
    pub t_cols_per_vector: usize,
    pub a_row_start: usize,
    pub b_row_start: usize,
}

/// Canonical setup-side view of witness ownership and D column placement.
///
/// Group descriptors contain only semantic group dimensions. This shared
/// layout is the sole owner of the witness ranges, opening-source length, and
/// checked relation-order D prefix ranges used by setup contributions.
#[derive(Clone)]
pub struct SetupContributionLayout {
    inner: Arc<SetupContributionLayoutInner>,
}

struct SetupContributionLayoutInner {
    witness_layout: Arc<WitnessLayout>,
    opening_source_len: usize,
    groups: Vec<SetupContributionGroupInputs>,
    d_columns: SetupDColumnLayout,
}

#[derive(Clone)]
pub(crate) struct SetupDColumnLayout {
    ranges: Vec<Range<usize>>,
    total_cols: usize,
}

impl SetupContributionGroupInputs {
    /// Derive the semantic descriptor for a single commitment group.
    pub fn from_single_group<E: FieldCore>(
        inputs: &SetupContributionPlanInputs<E>,
        fold_log_basis: u32,
    ) -> Result<Self, AkitaError> {
        if inputs.num_groups != 1 || inputs.num_polys_per_group.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "single-group setup contribution requires exactly one commitment group".into(),
            ));
        }
        let a_row_start = 1usize;
        let b_row_start = a_row_start
            .checked_add(inputs.n_a)
            .ok_or_else(|| AkitaError::InvalidSetup("B row start overflow".into()))?;
        let t_cols_per_vector = inputs
            .n_a
            .checked_mul(inputs.depth_open)
            .and_then(|width| width.checked_mul(inputs.live_fold_count))
            .ok_or_else(|| AkitaError::InvalidSetup("T polynomial width overflow".into()))?;
        Ok(Self {
            group_id: 0,
            num_claims: inputs.num_claims,
            live_fold_count: inputs.live_fold_count,
            fold_position_count: inputs.fold_position_count,
            depth_open: inputs.depth_open,
            depth_commit: inputs.depth_commit,
            depth_fold: inputs.depth_fold,
            log_basis: fold_log_basis,
            n_a: inputs.n_a,
            n_b: inputs.n_b,
            t_cols_per_vector,
            a_row_start,
            b_row_start,
        })
    }
}

impl SetupContributionLayout {
    pub fn new(
        witness_layout: Arc<WitnessLayout>,
        opening_source_len: usize,
        groups: Vec<SetupContributionGroupInputs>,
    ) -> Result<Self, AkitaError> {
        if groups.is_empty() || groups.len() != witness_layout.num_groups() {
            return Err(AkitaError::InvalidSetup(
                "setup groups disagree with witness layout".into(),
            ));
        }
        let witness_group_order = witness_layout
            .units()
            .iter()
            .map(|unit| unit.group_index())
            .fold(Vec::new(), |mut order, group_id| {
                if order.last() != Some(&group_id) {
                    order.push(group_id);
                }
                order
            });
        if witness_group_order
            != groups
                .iter()
                .map(|group| group.group_id)
                .collect::<Vec<_>>()
        {
            return Err(AkitaError::InvalidSetup(
                "setup groups do not follow witness relation order".into(),
            ));
        }
        for group in &groups {
            validate_group_witness_layout(&witness_layout, group)?;
        }
        let d_columns = SetupDColumnLayout::new(groups.iter().map(|group| {
            let width = group
                .num_claims
                .checked_mul(group.live_fold_count)
                .and_then(|cols| cols.checked_mul(group.depth_open))
                .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
            Ok((group.group_id, width))
        }))?;
        Ok(Self {
            inner: Arc::new(SetupContributionLayoutInner {
                witness_layout,
                opening_source_len,
                groups,
                d_columns,
            }),
        })
    }

    #[must_use]
    pub fn witness_layout(&self) -> &WitnessLayout {
        &self.inner.witness_layout
    }

    #[must_use]
    pub fn opening_source_len(&self) -> usize {
        self.inner.opening_source_len
    }

    #[must_use]
    pub fn groups(&self) -> &[SetupContributionGroupInputs] {
        &self.inner.groups
    }

    pub fn d_col_range(&self, group_id: usize) -> Result<Range<usize>, AkitaError> {
        self.inner.d_columns.range(group_id)
    }

    #[must_use]
    pub fn d_physical_cols(&self) -> usize {
        self.inner.d_columns.total_cols()
    }
}

impl SetupDColumnLayout {
    pub(crate) fn new<I>(groups: I) -> Result<Self, AkitaError>
    where
        I: IntoIterator<Item = Result<(usize, usize), AkitaError>>,
    {
        let groups = groups.into_iter().collect::<Result<Vec<_>, _>>()?;
        let mut ranges = vec![0..0; groups.len()];
        let mut seen = vec![false; groups.len()];
        let mut cursor = 0usize;
        for (group_id, width) in groups {
            let slot = seen
                .get_mut(group_id)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
            if std::mem::replace(slot, true) {
                return Err(AkitaError::InvalidSetup(
                    "setup D group id appears more than once".into(),
                ));
            }
            let end = cursor
                .checked_add(width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
            ranges[group_id] = cursor..end;
            cursor = end;
        }
        if seen.iter().any(|present| !present) {
            return Err(AkitaError::InvalidSetup(
                "setup D group ids are not contiguous".into(),
            ));
        }
        Ok(Self {
            ranges,
            total_cols: cursor,
        })
    }

    pub(crate) fn range(&self, group_id: usize) -> Result<Range<usize>, AkitaError> {
        self.ranges
            .get(group_id)
            .cloned()
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group is missing".into()))
    }

    pub(crate) const fn total_cols(&self) -> usize {
        self.total_cols
    }
}

fn validate_group_witness_layout(
    layout: &WitnessLayout,
    group: &SetupContributionGroupInputs,
) -> Result<(), AkitaError> {
    let units = layout.units_for_group(group.group_id)?;
    let mut next_fold = 0usize;
    for unit in units {
        if unit.live_fold_count() == 0 || unit.global_fold_start() != next_fold {
            return Err(AkitaError::InvalidSetup(
                "setup witness units do not form a contiguous fold tiling".into(),
            ));
        }
        next_fold = next_fold
            .checked_add(unit.live_fold_count())
            .ok_or_else(|| AkitaError::InvalidSetup("setup fold coverage overflow".into()))?;
    }
    if next_fold != group.live_fold_count {
        return Err(AkitaError::InvalidSetup(
            "setup group dimensions disagree with witness layout".into(),
        ));
    }
    Ok(())
}

pub struct SetupContributionPlan<E> {
    pub(crate) groups: Vec<SetupContributionGroupPlan<E>>,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
    pub(crate) projection_geometry: SetupProjectionGeometry,
}

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Prepared A-role (Z) column equality slice for the group at `index` in
    /// plan (witness relation) order, laid out as
    /// `z_eq_slice[position * depth_commit + commit_digit]` and already
    /// contracted over units and fold digits.
    ///
    /// Exposed so the ring-switch verifier can reuse this slice for the
    /// structured Z relation contribution instead of recomputing the same
    /// equality evaluations, per the setup-contribution reuse in Fix 6.
    #[must_use]
    pub fn group_z_eq_slice(&self, index: usize) -> Option<&[E]> {
        self.groups
            .get(index)
            .map(|group| group.z_eq_slice.as_slice())
    }
}

/// Tau1-derived setup weights cached at ring-switch prepare time.
#[derive(Clone)]
pub struct SetupContributionStatic<E> {
    pub(super) groups: Vec<SetupContributionGroupStatic<E>>,
    pub(super) d_rows: usize,
    pub(super) d_physical_cols: usize,
    pub(super) d_weights: Arc<[E]>,
}

impl<E> SetupContributionStatic<E> {
    /// Number of shared D rows in the packed setup contribution.
    #[must_use]
    pub fn d_rows(&self) -> usize {
        self.d_rows
    }

    /// Physical D-row width, including inactive columns between groups.
    #[must_use]
    pub fn d_physical_cols(&self) -> usize {
        self.d_physical_cols
    }
}

#[derive(Clone)]
pub(crate) struct SetupContributionGroupStatic<E> {
    pub(super) d_col_range: Range<usize>,
    pub(super) t_cols: usize,
    pub(super) z_cols: usize,
    pub(super) n_a: usize,
    pub(super) n_b: usize,
    pub(super) required: usize,
    pub(super) segments: Arc<[GroupSetupSegment<E>]>,
    pub(super) a_row_weights: Arc<[E]>,
    pub(super) b_weights: Arc<[E]>,
}

pub(crate) struct SetupContributionGroupPlan<E> {
    pub(crate) d_col_range: Range<usize>,
    pub(crate) t_cols: usize,
    pub(crate) z_cols: usize,
    pub(crate) n_a: usize,
    pub(crate) n_b: usize,
    pub(crate) required: usize,
    pub(crate) segments: Arc<[GroupSetupSegment<E>]>,
    pub(crate) e_eq_slice: Vec<E>,
    pub(crate) t_eq_slice: Vec<E>,
    pub(crate) z_eq_slice: Vec<E>,
    pub(crate) a_row_weights: Arc<[E]>,
    pub(crate) b_weights: Arc<[E]>,
    pub(crate) d_weights: Arc<[E]>,
}
