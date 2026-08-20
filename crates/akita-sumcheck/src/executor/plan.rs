use super::shared::{
    check_addressable_count, invalid, validate_eq_message, validate_standard_message,
};
use crate::{EqFactoredSumcheckProof, SumcheckProof};
use akita_field::{AkitaError, FieldCore};

/// Logical shape of one sumcheck member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SumcheckMemberShape {
    num_rounds: usize,
    degree_bound: usize,
}

impl SumcheckMemberShape {
    /// Construct one logical member shape.
    #[must_use]
    pub const fn new(num_rounds: usize, degree_bound: usize) -> Self {
        Self {
            num_rounds,
            degree_bound,
        }
    }

    /// Number of local sumcheck rounds.
    #[must_use]
    pub const fn num_rounds(self) -> usize {
        self.num_rounds
    }

    /// Maximum degree of a round message.
    #[must_use]
    pub const fn degree_bound(self) -> usize {
        self.degree_bound
    }
}

/// Caller-declared group of same-shape logical members.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SumcheckGroupSpec {
    member_indices: Vec<usize>,
}

impl SumcheckGroupSpec {
    /// Construct a group in stable member order.
    #[must_use]
    pub fn new(member_indices: Vec<usize>) -> Self {
        Self { member_indices }
    }
}

/// Checked geometry for one same-shape executor group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedSumcheckGroup {
    member_indices: Vec<usize>,
    num_rounds: usize,
    degree_bound: usize,
    suffix_offset: usize,
}

impl CheckedSumcheckGroup {
    /// Logical member indices in the order used for batching coefficients and terminals.
    #[must_use]
    pub fn member_indices(&self) -> &[usize] {
        &self.member_indices
    }

    /// Number of local rounds executed by this group.
    #[must_use]
    pub const fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Maximum degree of this group's round message.
    #[must_use]
    pub const fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Master-round offset at which this group becomes active.
    #[must_use]
    pub const fn suffix_offset(&self) -> usize {
        self.suffix_offset
    }

    pub(super) fn local_round(&self, master_round: usize) -> Option<usize> {
        master_round
            .checked_sub(self.suffix_offset)
            .filter(|local| *local < self.num_rounds)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CheckedBatchGeometry {
    pub(super) members: Vec<SumcheckMemberShape>,
    pub(super) groups: Vec<CheckedSumcheckGroup>,
    pub(super) master_rounds: usize,
}

impl CheckedBatchGeometry {
    fn new(
        members: Vec<SumcheckMemberShape>,
        group_specs: Vec<SumcheckGroupSpec>,
    ) -> Result<Self, AkitaError> {
        if members.is_empty() {
            return Err(invalid("sumcheck batch must contain at least one member"));
        }
        if group_specs.is_empty() {
            return Err(invalid("sumcheck batch must contain at least one group"));
        }

        let master_rounds = members
            .iter()
            .map(|shape| shape.num_rounds)
            .max()
            .ok_or_else(|| invalid("sumcheck batch must contain at least one member"))?;
        check_addressable_count(master_rounds, "master round count")?;

        let mut assigned_members = Vec::new();
        assigned_members
            .try_reserve_exact(members.len())
            .map_err(|_| invalid("sumcheck member assignment allocation failed"))?;
        assigned_members.resize(members.len(), false);
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(group_specs.len())
            .map_err(|_| invalid("sumcheck group allocation failed"))?;

        let mut previous_group_first = None;
        for spec in group_specs {
            let Some((&first, rest)) = spec.member_indices.split_first() else {
                return Err(invalid("sumcheck groups cannot be empty"));
            };
            if previous_group_first.is_some_and(|previous| first <= previous) {
                return Err(invalid(
                    "sumcheck groups must be ordered by their first logical member",
                ));
            }
            previous_group_first = Some(first);
            let first_shape = *members
                .get(first)
                .ok_or_else(|| invalid("sumcheck group member index is out of range"))?;
            first_shape
                .degree_bound
                .checked_add(1)
                .ok_or_else(|| invalid("sumcheck degree declaration overflows"))?;

            let mut previous = first;
            for &member_index in rest {
                if member_index <= previous {
                    return Err(invalid(
                        "sumcheck group member indices must be strictly increasing",
                    ));
                }
                previous = member_index;
            }

            for &member_index in &spec.member_indices {
                let shape = *members
                    .get(member_index)
                    .ok_or_else(|| invalid("sumcheck group member index is out of range"))?;
                if shape != first_shape {
                    return Err(invalid(
                        "all members in a sumcheck group must have the same shape",
                    ));
                }
                let assigned = assigned_members
                    .get_mut(member_index)
                    .ok_or_else(|| invalid("sumcheck group member index is out of range"))?;
                if *assigned {
                    return Err(invalid("sumcheck member appears in more than one group"));
                }
                *assigned = true;
            }

            groups.push(CheckedSumcheckGroup {
                member_indices: spec.member_indices,
                num_rounds: first_shape.num_rounds,
                degree_bound: first_shape.degree_bound,
                suffix_offset: master_rounds - first_shape.num_rounds,
            });
        }

        if assigned_members.contains(&false) {
            return Err(invalid("every sumcheck member must belong to one group"));
        }

        Ok(Self {
            members,
            groups,
            master_rounds,
        })
    }

    fn member_suffix<'a, E>(
        &self,
        master_point: &'a [E],
        member_index: usize,
    ) -> Result<&'a [E], AkitaError> {
        if master_point.len() != self.master_rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.master_rounds,
                actual: master_point.len(),
            });
        }
        let shape = self
            .members
            .get(member_index)
            .ok_or_else(|| invalid("sumcheck member index is out of range"))?;
        let offset = self
            .master_rounds
            .checked_sub(shape.num_rounds)
            .ok_or_else(|| invalid("derived sumcheck suffix is invalid"))?;
        master_point
            .get(offset..)
            .ok_or_else(|| invalid("derived sumcheck suffix is invalid"))
    }

    fn round_degree_bound(&self, master_round: usize) -> Result<usize, AkitaError> {
        if master_round >= self.master_rounds {
            return Err(invalid("sumcheck master round is out of range"));
        }
        self.groups
            .iter()
            .filter(|group| group.local_round(master_round).is_some())
            .map(|group| group.degree_bound)
            .max()
            .ok_or_else(|| invalid("sumcheck master round has no active group"))
    }
}

/// Checked standard front-loaded batch plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedStandardBatch(pub(super) CheckedBatchGeometry);

impl CheckedStandardBatch {
    /// Validate logical members and their same-shape executor groups.
    pub fn new(
        members: Vec<SumcheckMemberShape>,
        groups: Vec<SumcheckGroupSpec>,
    ) -> Result<Self, AkitaError> {
        CheckedBatchGeometry::new(members, groups).map(Self)
    }

    /// Number of master transcript rounds.
    #[must_use]
    pub const fn master_rounds(&self) -> usize {
        self.0.master_rounds
    }

    /// Checked executor groups.
    #[must_use]
    pub fn groups(&self) -> &[CheckedSumcheckGroup] {
        &self.0.groups
    }

    /// Derive one member's challenge point as a suffix of the master point.
    pub fn member_point<'a, E>(
        &self,
        master_point: &'a [E],
        member_index: usize,
    ) -> Result<&'a [E], AkitaError> {
        self.0.member_suffix(master_point, member_index)
    }

    /// Validate the exact round count and per-round degree bounds of a standard proof.
    pub fn validate_proof<E: FieldCore>(&self, proof: &SumcheckProof<E>) -> Result<(), AkitaError> {
        if proof.round_polys.len() != self.0.master_rounds {
            return Err(AkitaError::InvalidSize {
                expected: self.0.master_rounds,
                actual: proof.round_polys.len(),
            });
        }
        for (master_round, poly) in proof.round_polys.iter().enumerate() {
            validate_standard_message(poly, self.0.round_degree_bound(master_round)?)?;
        }
        Ok(())
    }
}

/// Checked eq-factored batch plan whose member equality points are master suffixes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedEqFactoredBatch<E: FieldCore> {
    pub(super) geometry: CheckedBatchGeometry,
    master_equality_point: Vec<E>,
}

impl<E: FieldCore> CheckedEqFactoredBatch<E> {
    /// Validate members, groups, and the master equality point.
    pub fn new(
        members: Vec<SumcheckMemberShape>,
        groups: Vec<SumcheckGroupSpec>,
        master_equality_point: Vec<E>,
    ) -> Result<Self, AkitaError> {
        let geometry = CheckedBatchGeometry::new(members, groups)?;
        if master_equality_point.len() != geometry.master_rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: geometry.master_rounds,
                actual: master_equality_point.len(),
            });
        }
        Ok(Self {
            geometry,
            master_equality_point,
        })
    }

    /// Number of master transcript rounds.
    #[must_use]
    pub const fn master_rounds(&self) -> usize {
        self.geometry.master_rounds
    }

    /// Checked executor groups.
    #[must_use]
    pub fn groups(&self) -> &[CheckedSumcheckGroup] {
        &self.geometry.groups
    }

    /// Derive one member's equality point as a suffix of the master point.
    pub fn member_equality_point(&self, member_index: usize) -> Result<&[E], AkitaError> {
        self.geometry
            .member_suffix(&self.master_equality_point, member_index)
    }

    /// Derive one member's challenge point as a suffix of the master point.
    pub fn member_point<'a>(
        &self,
        master_point: &'a [E],
        member_index: usize,
    ) -> Result<&'a [E], AkitaError> {
        self.geometry.member_suffix(master_point, member_index)
    }

    /// Validate an existing eq-factored proof for a batch with one shared factor.
    pub fn validate_proof(&self, proof: &EqFactoredSumcheckProof<E>) -> Result<(), AkitaError> {
        self.ensure_existing_proof_compatible()?;
        if proof.round_polys.len() != self.geometry.master_rounds {
            return Err(AkitaError::InvalidSize {
                expected: self.geometry.master_rounds,
                actual: proof.round_polys.len(),
            });
        }
        let degree_bound = self
            .geometry
            .groups
            .iter()
            .map(|group| group.degree_bound)
            .max()
            .ok_or_else(|| invalid("sumcheck batch must contain at least one group"))?;
        for poly in &proof.round_polys {
            validate_eq_message(poly, degree_bound)?;
        }
        Ok(())
    }

    pub(super) fn ensure_existing_proof_compatible(&self) -> Result<(), AkitaError> {
        if self
            .geometry
            .groups
            .iter()
            .any(|group| group.suffix_offset != 0)
        {
            return Err(AkitaError::UnsupportedSchedule(
                "unequal-round eq-factored groups require a reviewed proof format".to_owned(),
            ));
        }
        Ok(())
    }

    pub(super) fn equality_coordinate(&self, master_round: usize) -> Result<E, AkitaError> {
        self.master_equality_point
            .get(master_round)
            .copied()
            .ok_or_else(|| invalid("eq-factored master round is out of range"))
    }
}
