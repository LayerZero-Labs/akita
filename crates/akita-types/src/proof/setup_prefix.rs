//! Setup-prefix commitment artifacts for setup-claim offloading (slice 02B).
//!
//! This module defines the preprocessing metadata for power-of-two flat
//! coefficient prefixes of the shared setup vector `S`. It does not run a setup
//! product sumcheck or change proof semantics.

use crate::instance_descriptor::DescriptorDigest;
use crate::proof::{AkitaCommitmentHint, FlatRingVec, RingCommitment};
use crate::{ClaimIncidenceSummary, LevelParams};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore};
use std::collections::BTreeMap;

/// Ring dimension used when delegating setup claims to a flat coefficient prefix.
pub const SETUP_OFFLOAD_D_SETUP: usize = 32;

/// Minimum flat coefficient prefix length eligible for setup delegation.
pub const SETUP_OFFLOAD_N_MIN: usize = 1 << 23;

/// Identity for one committed setup-prefix slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SetupPrefixSlotId {
    /// Digest of the deterministic setup seed / layout identity.
    pub setup_seed_digest: DescriptorDigest,
    /// Coefficient-axis ring dimension for the delegated prefix object.
    pub d_setup: usize,
    /// Padded flat coefficient length committed for this slot.
    pub n_prefix: usize,
    /// Digest of the commitment parameters used to build the slot.
    pub level_params_digest: DescriptorDigest,
}

/// Policy for which prefix slots preprocessing should populate.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SetupPrefixPopulatePolicy {
    /// Do not generate setup-prefix commitments.
    #[default]
    Disabled,
    /// Generate every power-of-two prefix in `[n_min, n_max]`.
    FullLadder {
        /// Minimum prefix length (inclusive).
        n_min: usize,
        /// Maximum prefix length (inclusive).
        n_max: usize,
    },
    /// Generate only the listed padded prefix lengths.
    SelectedSlots(Vec<usize>),
}

/// Behavior when a requested prefix slot is absent at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingSetupPrefixSlotPolicy {
    /// Fail with a setup/policy error.
    #[default]
    StrictError,
    /// Prover-side convenience: create and persist the missing slot.
    GenerateAndPersist,
    /// Skip delegation and keep the direct setup scan.
    DirectFallback,
}

/// Public commitment half of a setup-prefix slot, stored without `D` const generics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixPublicCommitment<F: FieldCore> {
    /// Commitment rows in flattened ring-coefficient form.
    pub rows: Vec<FlatRingVec<F>>,
}

impl<F: FieldCore, const D: usize> From<RingCommitment<F, D>> for SetupPrefixPublicCommitment<F> {
    fn from(commitment: RingCommitment<F, D>) -> Self {
        Self {
            rows: commitment
                .u
                .into_iter()
                .map(|row| FlatRingVec::from_coeffs(row.coeffs.to_vec()))
                .collect(),
        }
    }
}

impl<F: FieldCore, const D: usize> TryFrom<&SetupPrefixPublicCommitment<F>>
    for RingCommitment<F, D>
{
    type Error = AkitaError;

    fn try_from(commitment: &SetupPrefixPublicCommitment<F>) -> Result<Self, AkitaError> {
        let u = commitment
            .rows
            .iter()
            .map(|row| {
                if row.coeffs().len() != D {
                    return Err(AkitaError::InvalidSetup(format!(
                        "setup prefix commitment row has {} coefficients, expected {D}",
                        row.coeffs().len()
                    )));
                }
                let mut coeffs = [F::zero(); D];
                for (dst, src) in coeffs.iter_mut().zip(row.coeffs()) {
                    *dst = *src;
                }
                Ok(CyclotomicRing::from_coefficients(coeffs))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RingCommitment { u })
    }
}

/// Verifier-visible metadata for one setup-prefix slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixVerifierSlot<F: FieldCore> {
    pub id: SetupPrefixSlotId,
    pub natural_len: usize,
    pub padded_len: usize,
    pub commitment: SetupPrefixPublicCommitment<F>,
}

/// Prover-ready metadata for one setup-prefix slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixSlot<F: FieldCore, const D: usize> {
    pub id: SetupPrefixSlotId,
    pub natural_len: usize,
    pub padded_len: usize,
    pub commitment: RingCommitment<F, D>,
    pub hint: AkitaCommitmentHint<F, D>,
}

impl<F: FieldCore, const D: usize> SetupPrefixSlot<F, D> {
    /// Strip prover-only hint material for verifier metadata.
    #[must_use]
    pub fn verifier_slot(&self) -> SetupPrefixVerifierSlot<F> {
        SetupPrefixVerifierSlot {
            id: self.id,
            natural_len: self.natural_len,
            padded_len: self.padded_len,
            commitment: self.commitment.clone().into(),
        }
    }
}

/// In-memory registry of prover-ready setup-prefix slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SetupPrefixProverRegistry<F: FieldCore, const D: usize> {
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixSlot<F, D>>,
}

impl<F: FieldCore, const D: usize> SetupPrefixProverRegistry<F, D> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&SetupPrefixSlot<F, D>> {
        self.slots.get(id)
    }

    pub fn insert(&mut self, slot: SetupPrefixSlot<F, D>) -> Result<(), AkitaError> {
        if self.slots.insert(slot.id, slot).is_some() {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".to_string(),
            ));
        }
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SetupPrefixSlotId, &SetupPrefixSlot<F, D>)> {
        self.slots.iter()
    }

    #[must_use]
    pub fn verifier_slots(&self) -> Vec<SetupPrefixVerifierSlot<F>> {
        self.slots
            .values()
            .map(SetupPrefixSlot::verifier_slot)
            .collect()
    }
}

/// In-memory registry of verifier-visible setup-prefix slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SetupPrefixVerifierRegistry<F: FieldCore> {
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixVerifierSlot<F>>,
}

impl<F: FieldCore> SetupPrefixVerifierRegistry<F> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&SetupPrefixVerifierSlot<F>> {
        self.slots.get(id)
    }

    pub fn insert(&mut self, slot: SetupPrefixVerifierSlot<F>) -> Result<(), AkitaError> {
        if self.slots.insert(slot.id, slot).is_some() {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".to_string(),
            ));
        }
        Ok(())
    }

    pub fn replace_from_prover_registry<const D: usize>(
        &mut self,
        prover_registry: &SetupPrefixProverRegistry<F, D>,
    ) -> Result<(), AkitaError> {
        self.slots.clear();
        for slot in prover_registry.verifier_slots() {
            self.insert(slot)?;
        }
        Ok(())
    }
}

/// Why setup delegation fell back to the direct scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupPrefixDirectReason {
    BelowMinimum {
        n_prefix: usize,
        n_min: usize,
    },
    DSetupMismatch {
        ring_dimension: usize,
        d_setup: usize,
    },
    MissingSlot(SetupPrefixSlotId),
}

/// Inputs needed to select a setup-prefix slot for one active shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupPrefixSelectionRequest {
    pub d_setup: usize,
    pub natural_field_len: usize,
    pub level_params_digest: DescriptorDigest,
}

/// Result of attempting to select a setup-prefix slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupPrefixSelectionOutcome<F: FieldCore, const D: usize> {
    DirectScan { reason: SetupPrefixDirectReason },
    Selected(SetupPrefixSlot<F, D>),
}

/// Return the packed role widths `(W_A, W_B, W_D)` for one active level shape.
pub fn active_setup_role_widths(
    level_params: &LevelParams,
    incidence: &ClaimIncidenceSummary,
) -> Result<(usize, usize, usize), AkitaError> {
    let w_a = level_params
        .block_len
        .checked_mul(level_params.num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("A setup width overflow".to_string()))?;
    let num_claims = incidence.num_claims();
    let max_group_poly_count = incidence
        .num_polys_per_point()
        .iter()
        .copied()
        .max()
        .ok_or_else(|| AkitaError::InvalidSetup("empty claim incidence".to_string()))?;
    let w_d = num_claims
        .checked_mul(level_params.num_blocks)
        .and_then(|n| n.checked_mul(level_params.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;
    let w_b = max_group_poly_count
        .checked_mul(level_params.a_key.row_len())
        .and_then(|n| n.checked_mul(level_params.num_blocks))
        .and_then(|n| n.checked_mul(level_params.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("B setup width overflow".to_string()))?;
    Ok((w_a, w_b, w_d))
}

/// Active packed setup footprint in ring slots: `max(n_a W_A, n_b W_B, n_d W_D)`.
pub fn active_setup_ring_slots(
    level_params: &LevelParams,
    incidence: &ClaimIncidenceSummary,
) -> Result<usize, AkitaError> {
    let (w_a, w_b, w_d) = active_setup_role_widths(level_params, incidence)?;
    let a_slots = level_params
        .a_key
        .row_len()
        .checked_mul(w_a)
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;
    let b_slots = level_params
        .b_key
        .row_len()
        .checked_mul(w_b)
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;
    let d_slots = level_params
        .d_key
        .row_len()
        .checked_mul(w_d)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".to_string()))?;
    Ok(a_slots.max(b_slots).max(d_slots))
}

/// Active flat coefficient count `N_active^F = D_setup * N_active^R`.
pub fn active_setup_field_len(
    level_params: &LevelParams,
    incidence: &ClaimIncidenceSummary,
    d_setup: usize,
) -> Result<usize, AkitaError> {
    active_setup_ring_slots(level_params, incidence)?
        .checked_mul(d_setup)
        .ok_or_else(|| AkitaError::InvalidSetup("active setup field length overflow".to_string()))
}

/// Smallest power-of-two flat prefix length covering `natural_field_len`.
#[must_use]
pub fn padded_setup_prefix_len(natural_field_len: usize) -> usize {
    natural_field_len.max(1).next_power_of_two()
}

/// Return the eligible padded prefix length, if any.
#[must_use]
pub fn select_prefix_len(natural_field_len: usize, n_min: usize) -> Option<usize> {
    let n_prefix = padded_setup_prefix_len(natural_field_len);
    (n_prefix >= n_min).then_some(n_prefix)
}

/// Ring-slot count for a flat prefix of `n_prefix` field coefficients at `d_setup`.
pub fn setup_prefix_commit_ring_slots(
    n_prefix: usize,
    d_setup: usize,
) -> Result<usize, AkitaError> {
    if d_setup == 0 || !n_prefix.is_multiple_of(d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a positive multiple of d_setup".to_string(),
        ));
    }
    Ok(n_prefix / d_setup)
}

/// Whether `level_params` witness shape matches one committed prefix length.
#[must_use]
pub fn level_params_matches_setup_prefix(
    level_params: &LevelParams,
    n_prefix: usize,
    d_setup: usize,
) -> bool {
    setup_prefix_commit_ring_slots(n_prefix, d_setup).is_ok_and(|ring_slots| {
        level_params
            .num_blocks
            .checked_mul(level_params.block_len)
            .is_some_and(|witness| witness == ring_slots)
    })
}

/// Keep only prefix lengths compatible with the supplied commitment parameters.
#[must_use]
pub fn filter_prefix_lengths_for_level_params(
    lengths: &[usize],
    level_params: &LevelParams,
    d_setup: usize,
) -> Vec<usize> {
    lengths
        .iter()
        .copied()
        .filter(|&n_prefix| level_params_matches_setup_prefix(level_params, n_prefix, d_setup))
        .collect()
}

/// Enumerate padded prefix lengths requested by a populate policy.
pub fn prefix_lengths_for_policy(
    policy: &SetupPrefixPopulatePolicy,
) -> Result<Vec<usize>, AkitaError> {
    match policy {
        SetupPrefixPopulatePolicy::Disabled => Ok(Vec::new()),
        SetupPrefixPopulatePolicy::FullLadder { n_min, n_max } => {
            if *n_min == 0 || !n_min.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "setup prefix ladder n_min must be a non-zero power of two".to_string(),
                ));
            }
            if *n_max < *n_min || !n_max.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "setup prefix ladder n_max must be a power of two >= n_min".to_string(),
                ));
            }
            let mut lengths = Vec::new();
            let mut current = *n_min;
            while current <= *n_max {
                lengths.push(current);
                current = current.checked_mul(2).ok_or_else(|| {
                    AkitaError::InvalidSetup("prefix ladder overflow".to_string())
                })?;
            }
            Ok(lengths)
        }
        SetupPrefixPopulatePolicy::SelectedSlots(lengths) => {
            for &len in lengths {
                if len == 0 || !len.is_power_of_two() {
                    return Err(AkitaError::InvalidSetup(format!(
                        "selected setup prefix length {len} must be a non-zero power of two"
                    )));
                }
            }
            Ok(lengths.clone())
        }
    }
}

/// Build the slot id for one committed setup prefix.
pub fn setup_prefix_slot_id(
    setup_seed_digest: DescriptorDigest,
    d_setup: usize,
    n_prefix: usize,
    level_params_digest: DescriptorDigest,
) -> SetupPrefixSlotId {
    SetupPrefixSlotId {
        setup_seed_digest,
        d_setup,
        n_prefix,
        level_params_digest,
    }
}

/// Select the tightest populated prover slot for one active shape.
pub fn select_setup_prefix_slot<F: FieldCore, const D: usize>(
    registry: &SetupPrefixProverRegistry<F, D>,
    setup_seed_digest: DescriptorDigest,
    ring_dimension: usize,
    request: SetupPrefixSelectionRequest,
    n_min: usize,
) -> SetupPrefixSelectionOutcome<F, D> {
    if ring_dimension != request.d_setup {
        return SetupPrefixSelectionOutcome::DirectScan {
            reason: SetupPrefixDirectReason::DSetupMismatch {
                ring_dimension,
                d_setup: request.d_setup,
            },
        };
    }
    let Some(n_prefix) = select_prefix_len(request.natural_field_len, n_min) else {
        return SetupPrefixSelectionOutcome::DirectScan {
            reason: SetupPrefixDirectReason::BelowMinimum {
                n_prefix: padded_setup_prefix_len(request.natural_field_len),
                n_min,
            },
        };
    };
    let id = setup_prefix_slot_id(
        setup_seed_digest,
        request.d_setup,
        n_prefix,
        request.level_params_digest,
    );
    match registry.get(&id) {
        Some(slot) => SetupPrefixSelectionOutcome::Selected(slot.clone()),
        None => SetupPrefixSelectionOutcome::DirectScan {
            reason: SetupPrefixDirectReason::MissingSlot(id),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance_descriptor::digest_level_params;
    use akita_challenges::SparseChallengeConfig;
    use crate::{ClaimIncidenceSummary, LevelParams, SisModulusFamily};

    fn sample_level_params() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q32,
            32,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(2, 3, 2, 2, 3, 0)
        .expect("sample level params")
    }

    #[test]
    fn active_setup_field_len_matches_packed_role_maximum() {
        let lp = sample_level_params();
        let incidence = ClaimIncidenceSummary::from_point_polys(5, vec![2, 1]).expect("incidence");
        let (w_a, w_b, w_d) = active_setup_role_widths(&lp, &incidence).expect("widths");
        let expected_ring_slots = lp
            .a_key
            .row_len()
            .checked_mul(w_a)
            .unwrap()
            .max(lp.b_key.row_len().checked_mul(w_b).unwrap())
            .max(lp.d_key.row_len().checked_mul(w_d).unwrap());
        assert_eq!(
            active_setup_ring_slots(&lp, &incidence).expect("ring slots"),
            expected_ring_slots
        );
        assert_eq!(
            active_setup_field_len(&lp, &incidence, SETUP_OFFLOAD_D_SETUP).expect("field len"),
            expected_ring_slots * SETUP_OFFLOAD_D_SETUP
        );
    }

    #[test]
    fn select_prefix_len_honors_n_min_gate() {
        assert_eq!(select_prefix_len(10, 17), None);
        assert_eq!(select_prefix_len(10, 16), Some(16));
        assert_eq!(select_prefix_len(100, SETUP_OFFLOAD_N_MIN), None);
        assert_eq!(
            select_prefix_len(SETUP_OFFLOAD_N_MIN, SETUP_OFFLOAD_N_MIN),
            Some(SETUP_OFFLOAD_N_MIN)
        );
    }

    #[test]
    fn prefix_lengths_for_selected_slots_rejects_non_power_of_two() {
        let err = prefix_lengths_for_policy(&SetupPrefixPopulatePolicy::SelectedSlots(vec![12]))
            .expect_err("non power-of-two");
        assert!(err.to_string().contains("power of two"));
    }

    #[test]
    fn select_setup_prefix_slot_reports_below_minimum() {
        use akita_field::Prime32Offset99 as F;

        let registry = SetupPrefixProverRegistry::<F, 32>::new();
        let outcome = select_setup_prefix_slot(
            &registry,
            [7u8; 32],
            32,
            SetupPrefixSelectionRequest {
                d_setup: 32,
                natural_field_len: 100,
                level_params_digest: [1u8; 32],
            },
            SETUP_OFFLOAD_N_MIN,
        );
        match outcome {
            SetupPrefixSelectionOutcome::DirectScan {
                reason: SetupPrefixDirectReason::BelowMinimum { n_min, .. },
            } => assert_eq!(n_min, SETUP_OFFLOAD_N_MIN),
            other => panic!("expected below minimum, got {other:?}"),
        }
    }

    #[test]
    fn filter_prefix_lengths_keeps_only_matching_witness_shape() {
        let lp = sample_level_params();
        let witness_field_len = lp
            .num_blocks
            .checked_mul(lp.block_len)
            .unwrap()
            .checked_mul(SETUP_OFFLOAD_D_SETUP)
            .unwrap();
        let filtered = filter_prefix_lengths_for_level_params(
            &[witness_field_len, witness_field_len * 2],
            &lp,
            SETUP_OFFLOAD_D_SETUP,
        );
        assert_eq!(filtered, vec![witness_field_len]);
        assert!(level_params_matches_setup_prefix(
            &lp,
            witness_field_len,
            SETUP_OFFLOAD_D_SETUP
        ));
    }

    #[test]
    fn select_setup_prefix_slot_reports_missing_slot() {
        use akita_field::Prime32Offset99 as F;

        let registry = SetupPrefixProverRegistry::<F, 32>::new();
        let incidence = ClaimIncidenceSummary::same_point(4, 1).expect("incidence");
        let lp = sample_level_params();
        let natural = active_setup_field_len(&lp, &incidence, 32).expect("natural");
        let outcome = select_setup_prefix_slot(
            &registry,
            [7u8; 32],
            32,
            SetupPrefixSelectionRequest {
                d_setup: 32,
                natural_field_len: natural,
                level_params_digest: digest_level_params(&[lp]),
            },
            1,
        );
        match outcome {
            SetupPrefixSelectionOutcome::DirectScan {
                reason: SetupPrefixDirectReason::MissingSlot(_),
            } => {}
            other => panic!("expected missing slot, got {other:?}"),
        }
    }
}
