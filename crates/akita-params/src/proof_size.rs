//! Header-stripped proof-byte formula for one fold level, shared by the
//! offline planner DP, the schedule selector, and profiling tooling.
//!
//! This is the single source of truth for direct-mode per-level proof grammar and
//! byte accounting. [`nonterminal_level_layout`] describes the fixed atom
//! counts and sumcheck shapes consumed by replay. The compact-entry walker that sums a whole
//! proof (`schedule_from_entry`) lives in `akita-planner`, next to the
//! schedule-table representation it consumes.
//! Nonce messages are priced independently by the canonical grinding
//! plan and are not attributed to this fixed-width level layout.

use crate::layout::digit_range::DigitRangeRouteShape;
use crate::layout::digit_range::PhysicalL2NormProofWireShape;
use crate::layout::field_bytes;
use crate::{CommittedGroupParams, DigitRangePlan, RelationAddressGeometry, Stage1StageShape};
use akita_error::AkitaError;

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

fn stage1_proof_bytes(shape: DigitRangeRouteShape, elem_bytes: usize) -> Result<usize, AkitaError> {
    let stages_bytes = shape
        .stages()
        .map(|stage| {
            sumcheck_bytes(stage.sumcheck_proof.0, stage.sumcheck_proof.1, elem_bytes)
                + stage.child_claims * elem_bytes
        })
        .sum::<usize>();
    let norm_bytes = shape.norm.map_or(0, |norm| {
        16 + (norm.subclaims + norm.virtual_evaluations) * elem_bytes
            + norm.rounds * norm.degree * elem_bytes
    });
    // The ordinary final range evaluation remains outside the optional norm
    // payload. The fused standard leaf shape accounts for the one additional
    // stored coefficient in every selected L2 leaf round.
    Ok(stages_bytes + elem_bytes + norm_bytes)
}

/// Immutable public grammar of one non-terminal proof level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonterminalLevelLayout {
    base_field_bytes: usize,
    challenge_field_bytes: usize,
    opening_payload_coeffs: usize,
    stage1_shape: DigitRangeRouteShape,
    stage2_sumcheck: akita_sumcheck::SumcheckShape,
    next_outer_payload_coeffs: usize,
    next_witness_evaluations: usize,
}

impl NonterminalLevelLayout {
    /// Fixed base-field atom count in the opening payload.
    #[must_use]
    pub const fn opening_payload_coeffs(&self) -> usize {
        self.opening_payload_coeffs
    }

    /// Stage-1 range-tree sumcheck and child-claim shapes in replay order.
    pub fn stage1_stages(&self) -> impl Iterator<Item = Stage1StageShape> + '_ {
        self.stage1_shape.stages()
    }

    /// Optional physical-L2 message shape.
    #[must_use]
    pub fn stage1_norm(&self) -> Option<PhysicalL2NormProofWireShape> {
        self.stage1_shape
            .norm
            .map(|norm| PhysicalL2NormProofWireShape {
                subclaims: norm.subclaims,
                virtual_evaluations: norm.virtual_evaluations,
                sumcheck: vec![norm.degree; norm.rounds],
            })
    }

    /// Fixed Stage-2 sumcheck shape.
    #[must_use]
    pub const fn stage2_sumcheck(&self) -> akita_sumcheck::SumcheckShape {
        self.stage2_sumcheck
    }

    /// Fixed base-field atom count in an ordinary recursive successor payload.
    #[must_use]
    pub const fn next_outer_payload_coeffs(&self) -> usize {
        self.next_outer_payload_coeffs
    }

    /// Maximum fixed-width bytes emitted by this level.
    pub fn encoded_len(self) -> Result<usize, AkitaError> {
        let opening_payload_bytes =
            akita_error::checked::product([self.opening_payload_coeffs, self.base_field_bytes])
                .ok_or_else(|| AkitaError::InvalidSetup("opening payload size overflow".into()))?;
        let stage1_bytes = stage1_proof_bytes(self.stage1_shape, self.challenge_field_bytes)?;
        let stage2_bytes = sumcheck_bytes(
            self.stage2_sumcheck.num_rounds(),
            self.stage2_sumcheck.degree_bound(),
            self.challenge_field_bytes,
        );
        let next_witness_bytes =
            akita_error::checked::product([self.next_outer_payload_coeffs, self.base_field_bytes])
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("next witness payload size overflow".into())
                })?;
        let next_witness_evaluation_bytes = akita_error::checked::product([
            self.next_witness_evaluations,
            self.challenge_field_bytes,
        ])
        .ok_or_else(|| AkitaError::InvalidSetup("next witness evaluation size overflow".into()))?;
        akita_error::checked::sum([
            opening_payload_bytes,
            stage1_bytes,
            stage2_bytes,
            next_witness_bytes,
            next_witness_evaluation_bytes,
        ])
        .ok_or_else(|| AkitaError::InvalidSetup("native level byte size overflow".into()))
    }
}

/// Derive the fixed-width message layout of one non-terminal fold level.
///
/// Compressed D and B images serialize as fixed-size base-field payloads.
/// Sumcheck objects and scalar evaluations serialize over the challenge field,
/// which may be a non-trivial extension of the base field for small-prime
/// configurations.
///
/// This prices the **direct-mode** folded payload only
/// (`SetupContributionMode::Direct`): the range-image and opening payloads, the stage-1
/// range-check tree, the fused stage-2 sumcheck, and the next-level witness
/// binding plus its evaluation. An ordinary recursive edge ships the
/// compressed outer payload; an edge into the suffix terminal reuses that
/// terminal proof's inner `t` state and ships no duplicate payload bytes. It
/// deliberately **excludes** the optional
/// recursive stage-3 setup-product sumcheck
/// (`SetupContributionMode::Recursive`), whose per-level overhead is priced
/// separately by [`stage3_setup_product_bytes`]. The planner adds that exact
/// payload when the selected successor consumes an incoming setup prefix.
///
/// `next_outer_payload` is required only for an intermediate outer-payload binding
/// (it sizes the next-level compressed payload shipped on the wire). It is
/// unused for a terminal-inner binding.
///
/// # Errors
///
/// The layout is derived from the same commitment geometry, digit-range stage
/// shapes, physical-L2 route, and public sumcheck degree used by
/// emission and receipt. Variable-width nonce and terminal-response messages
/// are deliberately owned by their separate layouts.
///
/// # Errors
///
/// Returns an error when a commitment payload or digit-range shape is invalid,
/// overflows, or disagrees with the selected base-field profile.
pub fn nonterminal_level_layout(
    base_field_bits: u32,
    challenge_field_bits: u32,
    lp: &CommittedGroupParams,
    relation_geometry: RelationAddressGeometry,
    next_outer_payload: Option<&CommittedGroupParams>,
) -> Result<NonterminalLevelLayout, AkitaError> {
    let base_field_bytes = field_bytes(base_field_bits);
    let challenge_field_bytes = field_bytes(challenge_field_bits);
    let rounds = relation_geometry.relation_point_variable_count();
    if base_field_bits != lp.open().matrix.sis_modulus_profile().field_bits() {
        return Err(AkitaError::InvalidSetup(
            "opening payload profile disagrees with the base field width".into(),
        ));
    }
    let stage1_shape = DigitRangePlan::new(1usize << lp.open().digits.log_basis)?
        .route_shape(rounds, lp.inner().matrix.security_route())?;
    let next_outer_payload_coeffs = match next_outer_payload {
        Some(next_lp) => {
            if base_field_bits != next_lp.outer().matrix.sis_modulus_profile().field_bits() {
                return Err(AkitaError::InvalidSetup(
                    "successor payload profile disagrees with the base field width".into(),
                ));
            }
            next_lp.outer_payload_geometry()?.transmitted_coefficients()
        }
        None => 0,
    };
    Ok(NonterminalLevelLayout {
        base_field_bytes,
        challenge_field_bytes,
        opening_payload_coeffs: lp.opening_payload_geometry()?.transmitted_coefficients(),
        stage1_shape,
        stage2_sumcheck: akita_sumcheck::SumcheckShape::new(rounds, 3)?,
        next_outer_payload_coeffs,
        next_witness_evaluations: 1,
    })
}

/// Header-stripped byte size of the recursive-mode stage-3 setup-product
/// sumcheck payload (`SetupSumcheckProof`) for one non-terminal fold level.
///
/// This is the proof-size overhead that `SetupContributionMode::Recursive`
/// adds on top of the direct-mode payload priced by [`nonterminal_level_layout`]. It
/// is added to the direct fold payload before the planner compares direct and
/// offloaded successor edges.
///
/// The payload is the setup claim and carried setup-prefix opening, followed by
/// a degree-[`crate::SETUP_SUMCHECK_DEGREE`] product sumcheck over the setup
/// domain (`log2(D) + log2(next_pow2(setup_ring_len))` rounds).
///
/// `ring_dimension` and the next-power-of-two of `setup_ring_len` must be
/// powers of two; this offline helper is not on the verifier path.
pub fn stage3_setup_product_bytes(
    challenge_field_bits: u32,
    ring_dimension: usize,
    setup_ring_len: usize,
) -> usize {
    let challenge_elem_bytes = field_bytes(challenge_field_bits);
    let ring_bits = ring_dimension.trailing_zeros() as usize;
    let lambda_bits = setup_ring_len.next_power_of_two().trailing_zeros() as usize;
    let rounds = ring_bits + lambda_bits;
    // Claimed setup contribution + carried setup-prefix opening + degree-2
    // setup-product sumcheck rounds.
    2 * challenge_elem_bytes
        + sumcheck_bytes(rounds, crate::SETUP_SUMCHECK_DEGREE, challenge_elem_bytes)
}
