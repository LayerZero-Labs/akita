//! Normalized single-point opening batches.
//!
//! # Protocol contract
//!
//! A batched prove/verify call uses exactly **one shared opening point** for all
//! claims. Multipoint incidence (different evaluation points within one batch)
//! is removed; callers must issue separate proofs or batch at a single point.
//!
//! Each claimed polynomial opening is an [`OpeningClaimSlot`]. Slots at that
//! point are gamma-batched via [`sample_public_row_coefficients`]. The
//! production folded path expects **one commitment object** bundling `N`
//! polynomials (`OpeningBatch::same_point`); multiple commitment objects at the
//! same point are not yet supported on folded recursion.
//!
//! Layout preparation may pad the shared point to the root fold arity.

use super::{ShapedVerifierClaims, VerifierClaims};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::labels::{ABSORB_BATCH_SHAPE, CHALLENGE_EVAL_BATCH};
use akita_transcript::{sample_ext_challenge, Transcript};
use std::collections::BTreeSet;

/// Kind of public opening claim represented by a batch slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpeningClaimKind {
    /// A claimed evaluation of a committed polynomial.
    Polynomial,
}

impl OpeningClaimKind {
    fn transcript_tag(self) -> u8 {
        match self {
            Self::Polynomial => 0,
        }
    }
}

/// One claimed opening at the shared opening point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpeningClaimSlot<F> {
    /// Commitment bundle containing the polynomial.
    pub commitment_group: usize,
    /// Polynomial index within the commitment bundle.
    pub poly_idx: usize,
    /// Claimed evaluation at the shared point.
    pub claimed_eval: F,
    /// Natural arity of the polynomial before embedding into the padded batch point.
    pub natural_num_vars: usize,
    /// Slot flavor.
    pub kind: OpeningClaimKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpeningClaimSlotShape {
    commitment_group: usize,
    poly_idx: usize,
    natural_num_vars: usize,
    kind: OpeningClaimKind,
}

impl OpeningClaimSlotShape {
    /// Commitment bundle containing the polynomial.
    pub fn commitment_group(&self) -> usize {
        self.commitment_group
    }

    /// Polynomial index within the commitment bundle.
    pub fn poly_idx(&self) -> usize {
        self.poly_idx
    }

    /// Natural arity of the polynomial before padding.
    pub fn natural_num_vars(&self) -> usize {
        self.natural_num_vars
    }

    /// Slot flavor.
    pub fn kind(&self) -> OpeningClaimKind {
        self.kind
    }
}

impl<F> From<&OpeningClaimSlot<F>> for OpeningClaimSlotShape {
    fn from(slot: &OpeningClaimSlot<F>) -> Self {
        Self {
            commitment_group: slot.commitment_group,
            poly_idx: slot.poly_idx,
            natural_num_vars: slot.natural_num_vars,
            kind: slot.kind,
        }
    }
}

/// Public verifier/prover batch input before validation.
#[derive(Debug, Clone)]
pub struct OpeningBatchInput<'a, F> {
    /// Shared opening point. Layout preparation pads this to the root arity.
    pub point: &'a [F],
    /// Claimed openings at `point`.
    pub slots: Vec<OpeningClaimSlot<F>>,
}

/// Normalize the public verifier-claim input shape into a single-point batch.
pub fn verifier_claims_to_opening_batch<'a, F, C>(
    claims: &VerifierClaims<'a, F, C>,
) -> OpeningBatchInput<'a, F>
where
    F: Copy,
{
    let (point, openings_by_group) = claims;
    let slots = openings_by_group
        .iter()
        .enumerate()
        .flat_map(|(commitment_group, openings)| {
            openings
                .openings
                .iter()
                .enumerate()
                .map(move |(poly_idx, &claimed_eval)| OpeningClaimSlot {
                    commitment_group,
                    poly_idx,
                    claimed_eval,
                    natural_num_vars: point.len(),
                    kind: OpeningClaimKind::Polynomial,
                })
        })
        .collect();
    OpeningBatchInput { point, slots }
}

/// Normalize verifier claims that carry per-opening natural arities.
pub fn shaped_verifier_claims_to_opening_batch<'a, F, C>(
    claims: &ShapedVerifierClaims<'a, F, C>,
) -> Result<OpeningBatchInput<'a, F>, AkitaError>
where
    F: Copy,
{
    let (point, openings_by_group) = claims;
    let mut slots = Vec::new();
    for (commitment_group, openings) in openings_by_group.iter().enumerate() {
        if openings.openings.len() != openings.natural_num_vars.len() {
            return Err(AkitaError::InvalidSize {
                expected: openings.openings.len(),
                actual: openings.natural_num_vars.len(),
            });
        }
        slots.extend(
            openings
                .openings
                .iter()
                .enumerate()
                .map(|(poly_idx, &claimed_eval)| OpeningClaimSlot {
                    commitment_group,
                    poly_idx,
                    claimed_eval,
                    natural_num_vars: openings.natural_num_vars[poly_idx],
                    kind: OpeningClaimKind::Polynomial,
                }),
        );
    }
    Ok(OpeningBatchInput { point, slots })
}

/// Capacity and dimension limits for opening-batch validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpeningBatchLimits {
    /// Maximum supported number of variables in the shared opening point.
    pub max_num_vars: usize,
    /// Maximum supported number of claimed openings.
    pub max_num_claims: usize,
}

/// The one public gamma-batching row for all slots at the shared point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningBatchRow {
    claim_indices: Vec<usize>,
}

impl OpeningBatchRow {
    /// Shared opening-point index. Always zero for single-point batches.
    pub fn point_idx(&self) -> usize {
        0
    }

    /// Flattened claim indices combined into this row, in slot order.
    pub fn claim_indices(&self) -> &[usize] {
        &self.claim_indices
    }
}

/// Derived routing and count data for a validated single-point opening batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningBatch {
    padded_num_vars: usize,
    slots: Vec<OpeningClaimSlotShape>,
    claim_to_commitment_group: Vec<usize>,
    claim_poly_indices: Vec<usize>,
    num_polys_per_commitment_group: Vec<usize>,
    public_row: OpeningBatchRow,
}

impl OpeningBatch {
    /// Validate that routing and count tables are internally consistent.
    pub fn check(&self) -> Result<(), AkitaError> {
        let num_claims = self.num_claims();
        if num_claims == 0 || self.num_polys_per_commitment_group.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        if self.slots.len() != num_claims
            || self.claim_to_commitment_group.len() != num_claims
            || self.claim_poly_indices.len() != num_claims
            || self.public_row.claim_indices.len() != num_claims
        {
            return Err(AkitaError::InvalidProof);
        }
        let mut seen_polys = BTreeSet::new();
        let mut seen_claims = BTreeSet::new();
        for claim_idx in 0..num_claims {
            let slot = self.slots[claim_idx];
            if slot.commitment_group >= self.num_polys_per_commitment_group.len()
                || self.claim_to_commitment_group[claim_idx] != slot.commitment_group
                || self.claim_poly_indices[claim_idx] != slot.poly_idx
                || slot.poly_idx >= self.num_polys_per_commitment_group[slot.commitment_group]
                || slot.natural_num_vars > self.padded_num_vars
                || !seen_polys.insert((slot.commitment_group, slot.poly_idx))
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        for &claim_idx in self.public_row.claim_indices() {
            if claim_idx >= num_claims || !seen_claims.insert(claim_idx) {
                return Err(AkitaError::InvalidProof);
            }
        }
        if seen_claims.len() != num_claims {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Build a batch for one commitment group opened at one shared point.
    pub fn same_point(padded_num_vars: usize, num_polys: usize) -> Result<Self, AkitaError> {
        if num_polys == 0 {
            return Err(AkitaError::InvalidInput(
                "opening batch requires at least one claim".to_string(),
            ));
        }
        let slots = (0..num_polys)
            .map(|poly_idx| OpeningClaimSlotShape {
                commitment_group: 0,
                poly_idx,
                natural_num_vars: padded_num_vars,
                kind: OpeningClaimKind::Polynomial,
            })
            .collect::<Vec<_>>();
        Self::from_slot_shapes(padded_num_vars, slots)
    }

    /// Build a batch from per-commitment polynomial counts at one shared point.
    pub fn from_commitment_groups(
        padded_num_vars: usize,
        num_polys_per_commitment_group: &[usize],
    ) -> Result<Self, AkitaError> {
        if num_polys_per_commitment_group.is_empty() || num_polys_per_commitment_group.contains(&0)
        {
            return Err(AkitaError::InvalidInput(
                "opening batch requires nonempty commitment groups".to_string(),
            ));
        }
        let mut slots = Vec::new();
        for (commitment_group, &count) in num_polys_per_commitment_group.iter().enumerate() {
            for poly_idx in 0..count {
                slots.push(OpeningClaimSlotShape {
                    commitment_group,
                    poly_idx,
                    natural_num_vars: padded_num_vars,
                    kind: OpeningClaimKind::Polynomial,
                });
            }
        }
        Self::from_slot_shapes(padded_num_vars, slots)
    }

    /// Build a batch from explicit slot shapes.
    pub fn from_slot_shapes(
        padded_num_vars: usize,
        slots: Vec<OpeningClaimSlotShape>,
    ) -> Result<Self, AkitaError> {
        if slots.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening batch requires at least one claim".to_string(),
            ));
        }
        if slots
            .iter()
            .any(|slot| slot.natural_num_vars > padded_num_vars)
        {
            return Err(AkitaError::InvalidInput(
                "opening batch slots must fit the shared point".to_string(),
            ));
        }
        let mut seen_polys = BTreeSet::new();
        let max_group = slots
            .iter()
            .map(|slot| slot.commitment_group)
            .max()
            .ok_or_else(|| {
                AkitaError::InvalidInput("opening batch requires at least one claim".to_string())
            })?;
        let mut group_poly_counts = vec![0usize; max_group + 1];
        for slot in &slots {
            group_poly_counts[slot.commitment_group] =
                group_poly_counts[slot.commitment_group].max(slot.poly_idx + 1);
            if !seen_polys.insert((slot.commitment_group, slot.poly_idx)) {
                return Err(AkitaError::InvalidInput(
                    "opening batch contains duplicate polynomial slot".to_string(),
                ));
            }
        }
        if group_poly_counts.contains(&0) || group_poly_counts.iter().sum::<usize>() != slots.len()
        {
            return Err(AkitaError::InvalidInput(
                "opening batch commitment groups and polynomial slots must be dense".to_string(),
            ));
        }
        let claim_to_commitment_group = slots
            .iter()
            .map(|slot| slot.commitment_group)
            .collect::<Vec<_>>();
        let claim_poly_indices = slots.iter().map(|slot| slot.poly_idx).collect::<Vec<_>>();
        let public_row = OpeningBatchRow {
            claim_indices: (0..slots.len()).collect(),
        };
        let batch = Self {
            padded_num_vars,
            slots,
            claim_to_commitment_group,
            claim_poly_indices,
            num_polys_per_commitment_group: group_poly_counts,
            public_row,
        };
        batch.check()?;
        Ok(batch)
    }

    /// Number of variables in the shared padded opening point.
    pub fn num_vars(&self) -> usize {
        self.padded_num_vars
    }

    /// Number of individual claimed openings.
    pub fn num_claims(&self) -> usize {
        self.slots.len()
    }

    /// Slot shape records.
    pub fn slots(&self) -> &[OpeningClaimSlotShape] {
        &self.slots
    }

    /// Commitment-group index for each flattened claim.
    pub fn claim_to_commitment_group(&self) -> &[usize] {
        &self.claim_to_commitment_group
    }

    /// Polynomial index within the commitment for each flattened claim.
    pub fn claim_poly_indices(&self) -> &[usize] {
        &self.claim_poly_indices
    }

    /// Number of polynomials bundled in the one commitment group.
    pub fn num_polys_per_commitment_group(&self) -> &[usize] {
        &self.num_polys_per_commitment_group
    }

    /// The one public gamma row.
    pub fn public_rows(&self) -> &[OpeningBatchRow] {
        std::slice::from_ref(&self.public_row)
    }

    /// Total number of committed polynomials addressed by the batch.
    pub fn num_polynomials(&self) -> usize {
        self.num_claims()
    }
}

impl<'a, F> OpeningBatchInput<'a, F> {
    /// Validate the single-point batch and derive its routing summary.
    pub fn validate(&self, limits: OpeningBatchLimits) -> Result<OpeningBatch, AkitaError> {
        if self.point.is_empty() && self.slots.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening batch requires one shared opening point".to_string(),
            ));
        }
        if self.slots.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening batch requires at least one claim".to_string(),
            ));
        }
        if self.point.len() > limits.max_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: limits.max_num_vars,
                actual: self.point.len(),
            });
        }
        if self.slots.len() > limits.max_num_claims {
            return Err(AkitaError::InvalidSize {
                expected: limits.max_num_claims,
                actual: self.slots.len(),
            });
        }
        let slot_shapes = self
            .slots
            .iter()
            .map(OpeningClaimSlotShape::from)
            .collect::<Vec<_>>();
        OpeningBatch::from_slot_shapes(self.point.len(), slot_shapes)
    }
}

/// Absorb normalized opening-batch shape and routing into the transcript.
pub fn append_opening_batch_shape_to_transcript<F, T>(
    batch: &OpeningBatch,
    transcript: &mut T,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    batch.check()?;

    transcript.append_serde(ABSORB_BATCH_SHAPE, &batch.num_vars());
    transcript.append_serde(ABSORB_BATCH_SHAPE, &batch.num_claims());
    for slot in batch.slots() {
        transcript.append_serde(ABSORB_BATCH_SHAPE, &slot.commitment_group());
        transcript.append_serde(ABSORB_BATCH_SHAPE, &slot.poly_idx());
        transcript.append_serde(ABSORB_BATCH_SHAPE, &slot.natural_num_vars());
        transcript.append_serde(ABSORB_BATCH_SHAPE, &slot.kind().transcript_tag());
    }
    Ok(())
}

/// Sample gamma coefficients for the one public row.
pub fn sample_public_row_coefficients<F, L, T>(
    batch: &OpeningBatch,
    transcript: &mut T,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F>,
    T: Transcript<F>,
{
    batch.check()?;
    if batch.num_claims() == 1 {
        return Ok(vec![L::one()]);
    }
    Ok((0..batch.num_claims())
        .map(|_| sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_EVAL_BATCH))
        .collect())
}

/// Sum batched public opening claims under per-slot gamma coefficients.
pub fn batched_eval_target_from_opening_batch<E>(
    batch: &OpeningBatch,
    row_coefficients: &[E],
    openings: &[E],
) -> Result<E, AkitaError>
where
    E: FieldCore,
{
    if row_coefficients.len() != batch.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: batch.num_claims(),
            actual: row_coefficients.len(),
        });
    }
    if openings.len() != batch.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: batch.num_claims(),
            actual: openings.len(),
        });
    }
    row_coefficients
        .iter()
        .zip(openings.iter())
        .try_fold(E::zero(), |acc, (&coefficient, &opening)| {
            Ok(acc + coefficient * opening)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp64, FpExt2, NegOneNr};
    use akita_transcript::{labels, AkitaTranscript};

    type TranscriptField = Fp64<4294967197>;

    fn generous_limits() -> OpeningBatchLimits {
        OpeningBatchLimits {
            max_num_vars: 8,
            max_num_claims: 16,
        }
    }

    #[test]
    fn opening_batch_tracks_single_point_slots() {
        let p0 = [1u64, 2];
        let batch = OpeningBatchInput {
            point: &p0,
            slots: vec![
                OpeningClaimSlot {
                    commitment_group: 0,
                    poly_idx: 0,
                    claimed_eval: 10u64,
                    natural_num_vars: 2,
                    kind: OpeningClaimKind::Polynomial,
                },
                OpeningClaimSlot {
                    commitment_group: 0,
                    poly_idx: 1,
                    claimed_eval: 11u64,
                    natural_num_vars: 1,
                    kind: OpeningClaimKind::Polynomial,
                },
            ],
        };

        let summary = batch.validate(generous_limits()).expect("valid batch");

        assert_eq!(summary.num_vars(), 2);
        assert_eq!(summary.num_claims(), 2);
        assert_eq!(summary.claim_to_commitment_group(), &[0, 0]);
        assert_eq!(summary.claim_poly_indices(), &[0, 1]);
        assert_eq!(summary.num_polys_per_commitment_group(), &[2]);
        assert_eq!(summary.slots()[1].natural_num_vars(), 1);
    }

    #[test]
    fn row_coefficients_batch_all_claims_once() {
        type E = FpExt2<TranscriptField, NegOneNr>;
        let batch = OpeningBatch::same_point(1, 2).expect("valid same-point batch");
        let mut transcript = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);

        let coeffs =
            sample_public_row_coefficients::<TranscriptField, E, _>(&batch, &mut transcript)
                .expect("row coefficients should sample");

        assert_eq!(coeffs.len(), 2);
        assert_ne!(coeffs[0], E::zero());
        assert_ne!(coeffs[1], E::zero());
    }

    #[test]
    fn transcript_binds_slot_order() {
        let forward = OpeningBatch::from_slot_shapes(
            1,
            vec![
                OpeningClaimSlotShape {
                    commitment_group: 0,
                    poly_idx: 0,
                    natural_num_vars: 1,
                    kind: OpeningClaimKind::Polynomial,
                },
                OpeningClaimSlotShape {
                    commitment_group: 0,
                    poly_idx: 1,
                    natural_num_vars: 1,
                    kind: OpeningClaimKind::Polynomial,
                },
            ],
        )
        .expect("forward batch");
        let swapped = OpeningBatch::from_slot_shapes(
            1,
            vec![
                OpeningClaimSlotShape {
                    commitment_group: 0,
                    poly_idx: 1,
                    natural_num_vars: 1,
                    kind: OpeningClaimKind::Polynomial,
                },
                OpeningClaimSlotShape {
                    commitment_group: 0,
                    poly_idx: 0,
                    natural_num_vars: 1,
                    kind: OpeningClaimKind::Polynomial,
                },
            ],
        )
        .expect("swapped batch");
        let mut t1 = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);

        append_opening_batch_shape_to_transcript(&forward, &mut t1).unwrap();
        append_opening_batch_shape_to_transcript(&swapped, &mut t2).unwrap();

        assert_ne!(
            t1.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION),
            t2.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION)
        );
    }
}
