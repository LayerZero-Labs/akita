//! Ordered resource admission and coefficient layout, without driver activation.
//!
//! This arithmetic seam does not select a schedule, encode a statement, prepare
//! a setup prefix or authorize an opening. The predecessor drivers retain their
//! singleton admission and byte stream.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use akita_error::{checked, AkitaError};

use super::{
    BinaryCommittedGroupManifest, BinaryEvaluationBounds, BinaryRootWitnessPlan,
    OrdinaryNativeRootPlan,
};
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};

/// Public selection for one binary occurrence. Point/owner-subset validation
/// belongs to the selected claim API.
#[derive(Clone, Debug)]
pub struct BinaryRootGroupOccurrence {
    pub manifest: BinaryCommittedGroupManifest,
    pub word: bool,
    /// Physical base-coordinate EP planes: extension degree times claim count.
    pub prime_coordinate_planes: usize,
    /// Explicit registration in the caller's commitment history. Reuse permits
    /// repeated names only for the identical registered commitment occurrence.
    /// The caller must authenticate this registration before driver selection.
    pub registered_commitment: Option<[u8; 32]>,
}

/// Complete ordinary preimage and its ordered owner names.
#[derive(Clone, Debug)]
pub struct OrdinaryRootGroupOccurrence {
    pub plan: OrdinaryNativeRootPlan,
    pub polynomials: Vec<[u8; 32]>,
    pub registered_commitment: Option<[u8; 32]>,
}

/// Independently indexed binary row families. Counts are shared with the
/// predecessor public compiler; its frozen row *order* is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryRootRowCounts {
    pub coefficients: usize,
    pub word: usize,
    pub b: usize,
    pub d: usize,
    pub prime: usize,
}

impl BinaryRootRowCounts {
    pub fn total(&self) -> Result<usize, AkitaError> {
        checked::sum([self.coefficients, self.word, self.b, self.d, self.prime])
            .ok_or(AkitaError::InvalidProof)
    }
}

/// Row ranges in the new roster order: all coefficient rows first, then each
/// binary's word/B/D/prime rows, then each ordinary's linear/negative-bit rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryRootGroupRows {
    pub coefficients: Range<usize>,
    pub word: Range<usize>,
    pub b: Range<usize>,
    pub d: Range<usize>,
    pub prime: Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrdinaryRootGroupRows {
    pub linear: Range<usize>,
    pub negative_bits: Range<usize>,
}

/// Derived binary geometry and placement. Local segment numbers are those of
/// `local_plan`; its final K maps to `quotient_segment`, outside the legacy block.
#[derive(Clone, Debug)]
pub struct BinaryRootOccurrencePlan {
    pub global_index: usize,
    pub registered_commitment: Option<[u8; 32]>,
    pub local_plan: BinaryRootWitnessPlan,
    pub ring_rows: usize,
    pub columns: usize,
    pub owners: usize,
    pub word_width: usize,
    pub logical_len: usize,
    pub word: bool,
    /// Physical base-coordinate EP planes: extension degree times claim count.
    pub prime_coordinate_planes: usize,
    pub evaluation_bounds: BinaryEvaluationBounds,
    pub legacy_segments: Range<usize>,
    pub quotient_segment: Option<usize>,
    pub physical_len: usize,
    pub rows: BinaryRootGroupRows,
}

#[derive(Clone, Debug)]
pub struct OrdinaryRootOccurrencePlan {
    pub global_index: usize,
    pub registered_commitment: Option<[u8; 32]>,
    /// Rebased to the global segment range; the profile and source map remain
    /// exactly those of the supplied plan.
    pub plan: OrdinaryNativeRootPlan,
    pub polynomials: Vec<[u8; 32]>,
    pub segments: Range<usize>,
    pub physical_len: usize,
    pub rows: OrdinaryRootGroupRows,
}

/// Checked resource sums. These describe inputs and metadata, never a request
/// to allocate the corresponding witness or row arrays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BinaryRootGroupTotals {
    pub owners: u128,
    pub binary_input_carriers: u128,
    pub ordinary_input_coefficients: u128,
    pub prime_claims: u128,
    pub fold_coordinates: u128,
    pub residual_coefficients: u128,
    pub segment_cap: u128,
    pub witness_domain: u128,
    pub batches: u128,
}

impl BinaryRootGroupTotals {
    fn validate(&self) -> Result<(), AkitaError> {
        if self.owners > 4096
            || self.binary_input_carriers > 1 << 28
            || self.ordinary_input_coefficients > 1 << 28
            || self.prime_claims > 4096
            || self.fold_coordinates > 1 << 20
            || self.residual_coefficients > 7_962_624
            || self.segment_cap > 448
            || self.witness_domain > 1u128 << 32
            || self.batches > 1u128 << 32
        {
            return Err(admission_error());
        }
        Ok(())
    }
}

/// One source of truth for the ordered roster, rows and outgoing layout.
/// Construction allocates only bounded occurrence/segment metadata.
#[derive(Clone, Debug)]
pub struct BinaryRootGroupPlan {
    binary: Vec<BinaryRootOccurrencePlan>,
    ordinary: Vec<OrdinaryRootOccurrencePlan>,
    repacking: RootCoefficientRepacking,
    totals: BinaryRootGroupTotals,
    batch_count: usize,
    minimum_domain_len: usize,
}

impl BinaryRootGroupPlan {
    pub fn new(
        binary: &[BinaryRootGroupOccurrence],
        ordinary: &[OrdinaryRootGroupOccurrence],
        destination_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        let count = checked::sum([binary.len(), ordinary.len()]).ok_or_else(admission_error)?;
        if binary.is_empty()
            || count > 64
            || !destination_ring_dimension.is_power_of_two()
            || destination_ring_dimension as u128 > 1u128 << 32
        {
            return Err(admission_error());
        }
        let field = binary
            .first()
            .ok_or_else(admission_error)?
            .manifest
            .field_profile();
        let mut totals = BinaryRootGroupTotals::default();
        // Validate every resource count before cloning descriptors or layouts.
        for group in binary {
            group.manifest.validate()?;
            let descriptor = group.manifest.descriptor();
            let index = descriptor.index_map();
            let owners = descriptor.polynomials().len();
            let m = checked::product([4, index.columns()])
                .and_then(|block| checked::exact_div(index.padded_carriers(), block))
                .ok_or_else(admission_error)?;
            let prime_claims = checked::exact_div(
                group.prime_coordinate_planes,
                group.manifest.field_profile().extension_degree(),
            )
            .ok_or_else(admission_error)?;
            if group.manifest.field_profile() != field
                || !m.is_power_of_two()
                || m > 8192
                || index.columns() > 256
                || index.padded_carriers() > 1 << 23
                || !(1..=4096).contains(&owners)
                || prime_claims > owners
                || (!group.word && group.prime_coordinate_planes == 0)
            {
                return Err(admission_error());
            }
            totals.owners = totals
                .owners
                .checked_add(owners as u128)
                .ok_or_else(admission_error)?;
            totals.binary_input_carriers = (owners as u128)
                .checked_mul(index.padded_carriers() as u128)
                .and_then(|n| totals.binary_input_carriers.checked_add(n))
                .ok_or_else(admission_error)?;
            totals.fold_coordinates = (owners as u128)
                .checked_mul(index.columns() as u128)
                .and_then(|n| totals.fold_coordinates.checked_add(n))
                .ok_or_else(admission_error)?;
            totals.residual_coefficients = (owners as u128)
                .checked_mul(1944)
                .and_then(|n| totals.residual_coefficients.checked_add(n))
                .ok_or_else(admission_error)?;
            totals.prime_claims = totals
                .prime_claims
                .checked_add(prime_claims as u128)
                .ok_or_else(admission_error)?;
        }
        for group in ordinary {
            let profile = group.plan.profile();
            let owners = profile.group.num_polynomials();
            if !(1..=4096).contains(&owners)
                || group.polynomials.len() != owners
                || profile.group.num_vars() > 25
                || profile.inner.matrix.sis_modulus_profile().modulus() != field.modulus()
            {
                return Err(admission_error());
            }
            let source_len = 1u128
                .checked_shl(
                    u32::try_from(profile.group.num_vars()).map_err(|_| admission_error())?,
                )
                .ok_or_else(admission_error)?;
            totals.owners = totals
                .owners
                .checked_add(owners as u128)
                .ok_or_else(admission_error)?;
            totals.ordinary_input_coefficients = (owners as u128)
                .checked_mul(source_len)
                .and_then(|n| totals.ordinary_input_coefficients.checked_add(n))
                .ok_or_else(admission_error)?;
        }
        totals.segment_cap = (binary.len() as u128)
            .checked_mul(7)
            .zip((ordinary.len() as u128).checked_mul(5))
            .and_then(|(b, o)| b.checked_add(o))
            .ok_or_else(admission_error)?;
        totals.validate()?;
        validate_names(binary, ordinary)?;

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(usize::try_from(totals.segment_cap).map_err(|_| admission_error())?)
            .map_err(|_| admission_error())?;
        let mut binary_plans = Vec::new();
        binary_plans
            .try_reserve_exact(binary.len())
            .map_err(|_| admission_error())?;
        let mut next_coefficient = 0;
        // Reuse the predecessor constructors for all local geometry and bounds.
        for (global_index, group) in binary.iter().enumerate() {
            let descriptor = group.manifest.descriptor();
            let columns = descriptor.index_map().columns();
            let scalar_rows = checked::exact_div(descriptor.index_map().padded_carriers(), columns)
                .ok_or_else(admission_error)?;
            let local_plan = BinaryRootWitnessPlan::for_profile(
                descriptor,
                group.manifest.a_rank(),
                destination_ring_dimension,
                field,
                group.word,
                group.prime_coordinate_planes,
            )?;
            let legacy_len = local_plan
                .quotient_segment()
                .unwrap_or(local_plan.repacking().segments().len());
            let legacy_segments =
                checked::range(segments.len(), legacy_len).ok_or_else(admission_error)?;
            segments.extend_from_slice(
                local_plan
                    .repacking()
                    .segments()
                    .get(..legacy_len)
                    .ok_or_else(admission_error)?,
            );
            let coefficients =
                checked::range(next_coefficient, local_plan.coefficient_batch_count()?)
                    .ok_or_else(admission_error)?;
            next_coefficient = coefficients.end;
            binary_plans.push(BinaryRootOccurrencePlan {
                global_index,
                registered_commitment: group.registered_commitment,
                ring_rows: scalar_rows / 4,
                columns,
                owners: descriptor.polynomials().len(),
                word_width: descriptor.index_map().word_width(),
                logical_len: descriptor.index_map().logical_len(),
                word: group.word,
                prime_coordinate_planes: group.prime_coordinate_planes,
                evaluation_bounds: BinaryEvaluationBounds::new(
                    scalar_rows,
                    columns,
                    field.modulus(),
                )?,
                physical_len: local_plan.repacking().live_len(),
                local_plan,
                legacy_segments,
                quotient_segment: None,
                rows: BinaryRootGroupRows {
                    coefficients,
                    word: 0..0,
                    b: 0..0,
                    d: 0..0,
                    prime: 0..0,
                },
            });
        }
        let mut ordinary_plans = Vec::new();
        ordinary_plans
            .try_reserve_exact(ordinary.len())
            .map_err(|_| admission_error())?;
        for (index, group) in ordinary.iter().enumerate() {
            let first = segments.len();
            segments.extend_from_slice(group.plan.segments());
            let mut plan = group.plan.clone();
            plan.first_segment = first;
            ordinary_plans.push(OrdinaryRootOccurrencePlan {
                global_index: checked::sum([binary.len(), index]).ok_or_else(admission_error)?,
                registered_commitment: group.registered_commitment,
                segments: first..segments.len(),
                physical_len: physical_len(plan.segments())?,
                plan,
                polynomials: group.polynomials.clone(),
                rows: OrdinaryRootGroupRows {
                    linear: 0..0,
                    negative_bits: 0..0,
                },
            });
        }
        // All K segments follow every ordinary block, with no alignment padding.
        for group in &mut binary_plans {
            if let Some(local) = group.local_plan.quotient_segment() {
                group.quotient_segment = Some(segments.len());
                segments.push(
                    *group
                        .local_plan
                        .repacking()
                        .segments()
                        .get(local)
                        .ok_or_else(admission_error)?,
                );
            }
        }
        let live_len = physical_len(&segments)?;
        let minimum = (live_len as u128)
            .max(destination_ring_dimension as u128)
            .checked_next_power_of_two()
            .ok_or_else(admission_error)?;
        totals.witness_domain = minimum;
        totals.validate()?;
        let repacking = RootCoefficientRepacking::new(segments, destination_ring_dimension)?;
        let mut next = next_coefficient;
        for (group, input) in binary_plans.iter_mut().zip(binary) {
            let counts = group.local_plan.row_counts(
                input.manifest.b_rows(),
                input
                    .manifest
                    .d_label(input.word, input.prime_coordinate_planes)?
                    .rank as usize,
            )?;
            group.rows.word = checked::range(next, counts.word).ok_or_else(admission_error)?;
            group.rows.b =
                checked::range(group.rows.word.end, counts.b).ok_or_else(admission_error)?;
            group.rows.d =
                checked::range(group.rows.b.end, counts.d).ok_or_else(admission_error)?;
            group.rows.prime =
                checked::range(group.rows.d.end, counts.prime).ok_or_else(admission_error)?;
            next = group.rows.prime.end;
        }
        for group in &mut ordinary_plans {
            group.rows.linear = checked::range(next, group.plan.linear_batch_count())
                .ok_or_else(admission_error)?;
            group.rows.negative_bits =
                checked::range(group.rows.linear.end, group.plan.negative_binary_count()?)
                    .ok_or_else(admission_error)?;
            next = group.rows.negative_bits.end;
        }
        totals.batches = next as u128;
        totals.validate()?;
        let repacking_domain_len = repacking.domain_len();
        Ok(Self {
            binary: binary_plans,
            ordinary: ordinary_plans,
            repacking,
            totals,
            batch_count: next,
            minimum_domain_len: repacking_domain_len,
        })
    }

    pub fn binary(&self) -> &[BinaryRootOccurrencePlan] {
        &self.binary
    }
    pub fn ordinary(&self) -> &[OrdinaryRootOccurrencePlan] {
        &self.ordinary
    }
    pub fn repacking(&self) -> &RootCoefficientRepacking {
        &self.repacking
    }
    pub fn totals(&self) -> &BinaryRootGroupTotals {
        &self.totals
    }
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }
    pub fn minimum_domain_len(&self) -> usize {
        self.minimum_domain_len
    }

    /// Translate a group's local segment/element/digit/natural coefficient
    /// through the generic repacking formula, including nonaligned K starts.
    pub fn coefficient_address(
        &self,
        global_group: usize,
        local_segment: usize,
        element: usize,
        digit: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        let segment = if let Some(group) = self.binary.get(global_group) {
            if local_segment < group.legacy_segments.len() {
                checked::sum([group.legacy_segments.start, local_segment])
                    .ok_or_else(admission_error)?
            } else if group.local_plan.quotient_segment() == Some(local_segment) {
                group.quotient_segment.ok_or_else(admission_error)?
            } else {
                return Err(admission_error());
            }
        } else {
            let index = global_group
                .checked_sub(self.binary.len())
                .ok_or_else(admission_error)?;
            let group = self.ordinary.get(index).ok_or_else(admission_error)?;
            if local_segment >= group.segments.len() {
                return Err(admission_error());
            }
            checked::sum([group.segments.start, local_segment]).ok_or_else(admission_error)?
        };
        self.repacking
            .coefficient_address(segment, element, digit, coefficient)
    }

    pub fn with_domain_len(mut self, domain_len: usize) -> Result<Self, AkitaError> {
        if domain_len as u128 > 1u128 << 32 {
            return Err(admission_error());
        }
        self.repacking = self.repacking.with_domain_len(domain_len)?;
        self.totals.witness_domain = domain_len as u128;
        Ok(self)
    }
}

fn admission_error() -> AkitaError {
    AkitaError::InvalidInput("binary group roster resource or layout admission failed".into())
}

fn physical_len(segments: &[RootRepackingSegment]) -> Result<usize, AkitaError> {
    let total = segments.iter().try_fold(0u128, |total, segment| {
        (segment.elements as u128)
            .checked_mul(segment.digits as u128)
            .and_then(|n| n.checked_mul(segment.geometry.physical_coefficient_width() as u128))
            .and_then(|n| total.checked_add(n))
            .ok_or_else(admission_error)
    })?;
    usize::try_from(total).map_err(|_| admission_error())
}

fn validate_names(
    binary: &[BinaryRootGroupOccurrence],
    ordinary: &[OrdinaryRootGroupOccurrence],
) -> Result<(), AkitaError> {
    let count = checked::sum([binary.len(), ordinary.len()]).ok_or_else(admission_error)?;
    let names = |index: usize| -> Result<&[[u8; 32]], AkitaError> {
        if let Some(group) = binary.get(index) {
            Ok(group.manifest.descriptor().polynomials())
        } else {
            ordinary
                .get(
                    index
                        .checked_sub(binary.len())
                        .ok_or_else(admission_error)?,
                )
                .map(|g| g.polynomials.as_slice())
                .ok_or_else(admission_error)
        }
    };
    let registration = |index: usize| -> Option<[u8; 32]> {
        binary
            .get(index)
            .and_then(|g| g.registered_commitment)
            .or_else(|| {
                index
                    .checked_sub(binary.len())
                    .and_then(|i| ordinary.get(i))
                    .and_then(|g| g.registered_commitment)
            })
    };
    let same_occurrence = |a: usize, b: usize| -> bool {
        if let (Some(a), Some(b)) = (binary.get(a), binary.get(b)) {
            a.manifest == b.manifest
        } else if let (Some(a), Some(b)) = (
            a.checked_sub(binary.len()).and_then(|i| ordinary.get(i)),
            b.checked_sub(binary.len()).and_then(|i| ordinary.get(i)),
        ) {
            a.plan.profile() == b.plan.profile()
                && a.plan.source_width() == b.plan.source_width()
                && a.polynomials == b.polynomials
        } else {
            false
        }
    };
    let mut owners = BTreeMap::new();
    let mut registrations = BTreeMap::new();
    for index in 0..count {
        if let Some(id) = registration(index) {
            if let Some(&previous) = registrations.get(&id) {
                if !same_occurrence(previous, index) {
                    return Err(admission_error());
                }
            } else {
                registrations.insert(id, index);
            }
        }
        let mut local_names = BTreeSet::new();
        for id in names(index)? {
            if !local_names.insert(*id) {
                return Err(admission_error());
            }
            if let Some(&previous) = owners.get(id) {
                if previous == index
                    || registration(index).is_none()
                    || registration(index) != registration(previous)
                    || !same_occurrence(previous, index)
                {
                    return Err(admission_error());
                }
            } else {
                owners.insert(*id, index);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "group_plan_tests.rs"]
mod tests;
