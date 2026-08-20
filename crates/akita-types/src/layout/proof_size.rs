//! Header-stripped proof-size and planned-witness sizing formulas.

use akita_error::AkitaError;

use crate::{PolynomialGroupLayout, TerminalResponseShape, EXTENSION_OPENING_REDUCTION_DEGREE};

/// Field element size in bytes for a field with `field_bits` bits.
pub fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

/// Ring vector bytes without a length prefix.
pub fn proof_ring_vec_bytes(ring_len: usize, ring_dim: usize, elem_bytes: usize) -> usize {
    ring_len.saturating_mul(ring_dim).saturating_mul(elem_bytes)
}

/// Packed digit bytes without a length/tag prefix.
pub fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    num_elems.saturating_mul(bits_per_elem as usize).div_ceil(8)
}

/// Serialized byte size for a terminal direct witness shape.
pub fn terminal_response_bytes(field_bits: u32, shape: &TerminalResponseShape) -> usize {
    crate::proof::terminal_response_upper_bound_bytes(
        field_bits,
        &shape.layout,
        shape.layout.z_payload_bytes(),
    )
}

/// Planner byte estimate for a terminal response.
///
/// The scheduled Golomb payload cap remains unchanged. For a single-group L2
/// route, candidate selection may price the tighter deterministic payload bound
/// implied by the certified energy. Unsupported shapes conservatively use the
/// scheduled byte budget.
pub fn terminal_response_planner_bytes(
    field_bits: u32,
    shape: &TerminalResponseShape,
    response_l2_sq_cap: Option<u128>,
) -> usize {
    let scheduled = terminal_response_bytes(field_bits, shape);
    let Some(l2_sq_cap) = response_l2_sq_cap else {
        return scheduled;
    };
    let [group] = shape.layout.groups.as_slice() else {
        return scheduled;
    };
    let Some(z_payload_bytes) = crate::golomb_rice::golomb_rice_l2_planner_payload_bytes(
        group.z_coords,
        l2_sq_cap,
        group.z_rice_low_bits,
    ) else {
        return scheduled;
    };
    crate::proof::terminal_response_upper_bound_bytes(
        field_bits,
        &shape.layout,
        z_payload_bytes.min(group.z_payload_bytes),
    )
}

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

/// Header-stripped byte size of an extension-opening reduction proof.
///
/// The reduction proof serializes `partials` challenge-field elements, one
/// fixed degree-two sumcheck over all opening claims, and one terminal claim
/// handle per opening. `extension_width = 1` means the claim field is already
/// the base field and contributes zero bytes.
///
/// # Errors
///
/// Returns an error when `extension_width` is not a power of two or when the
/// tensor split is wider than the opened Boolean cube.
pub fn extension_opening_reduction_proof_bytes(
    challenge_field_bits: u32,
    partials: usize,
    opening_vars: usize,
    extension_width: usize,
) -> Result<usize, AkitaError> {
    if extension_width <= 1 {
        return Ok(0);
    }
    if !extension_width.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(format!(
            "extension opening width must be a power of two, got {extension_width}"
        )));
    }
    let split_bits = extension_width.trailing_zeros() as usize;
    if split_bits > opening_vars {
        return Err(AkitaError::InvalidSetup(format!(
            "extension opening split ({split_bits}) exceeds opening variables ({opening_vars})"
        )));
    }
    let elem_bytes = field_bytes(challenge_field_bits);
    let rounds = opening_vars - split_bits;
    if !partials.is_multiple_of(extension_width) {
        return Err(AkitaError::InvalidSetup(format!(
            "extension opening partial count {partials} is not divisible by width {extension_width}"
        )));
    }
    let num_claims = partials / extension_width;
    Ok(partials
        .saturating_mul(elem_bytes)
        .saturating_add(sumcheck_bytes(
            rounds,
            EXTENSION_OPENING_REDUCTION_DEGREE,
            elem_bytes,
        ))
        .saturating_add(num_claims.saturating_mul(elem_bytes)))
}

/// Log2 of the next power-of-two Boolean cube width for recursive opening.
pub fn padded_boolean_opening_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

/// Extension-opening reduction proof bytes for one fold level in a schedule.
///
/// `extension_opening_width` is the claim-vs-coefficient field degree: `1`
/// (single-field geometry, zero bytes) or a supported power-of-two extension
/// width. Any other width is rejected rather than priced, so invalid custom
/// configurations cannot pass planning as zero-cost. `opening_shape` is the
/// aggregate EOR shape: maximum group arity and checked total claim count.
pub fn extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    opening_shape: PolynomialGroupLayout,
) -> Result<usize, AkitaError> {
    match extension_opening_reduction_level_geometry(extension_opening_width, opening_shape)? {
        // This is a serialized-byte count, not cryptographic material.
        ExtensionOpeningReductionGeometry::NotRequired => Ok(usize::default()),
        ExtensionOpeningReductionGeometry::Required {
            partials,
            opening_vars,
        } => extension_opening_reduction_proof_bytes(
            challenge_field_bits,
            partials,
            opening_vars,
            extension_opening_width,
        ),
        ExtensionOpeningReductionGeometry::Infeasible {
            split_bits,
            opening_vars,
        } => Err(AkitaError::InvalidSetup(format!(
            "extension opening split ({split_bits}) exceeds opening variables ({opening_vars})"
        ))),
    }
}

/// Candidate-aware EOR pricing.
///
/// `Ok(None)` means this otherwise valid policy is locally infeasible for the
/// candidate's opening cube. Malformed policy values and arithmetic failures
/// remain errors, so search can skip one branch without swallowing bad input.
pub fn try_extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    opening_shape: PolynomialGroupLayout,
) -> Result<Option<usize>, AkitaError> {
    match extension_opening_reduction_level_geometry(extension_opening_width, opening_shape)? {
        // This is a serialized-byte count, not cryptographic material.
        ExtensionOpeningReductionGeometry::NotRequired => Ok(Some(usize::default())),
        ExtensionOpeningReductionGeometry::Required {
            partials,
            opening_vars,
        } => extension_opening_reduction_proof_bytes(
            challenge_field_bits,
            partials,
            opening_vars,
            extension_opening_width,
        )
        .map(Some),
        ExtensionOpeningReductionGeometry::Infeasible { .. } => Ok(None),
    }
}

enum ExtensionOpeningReductionGeometry {
    NotRequired,
    Required {
        partials: usize,
        opening_vars: usize,
    },
    Infeasible {
        split_bits: usize,
        opening_vars: usize,
    },
}

fn extension_opening_reduction_level_geometry(
    extension_opening_width: usize,
    opening_shape: PolynomialGroupLayout,
) -> Result<ExtensionOpeningReductionGeometry, AkitaError> {
    if extension_opening_width != 1 && !extension_opening_width.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(format!(
            "extension opening width must be one or a power of two, got {extension_opening_width}"
        )));
    }
    if extension_opening_width == 1 {
        return Ok(ExtensionOpeningReductionGeometry::NotRequired);
    }
    opening_shape.validate()?;
    let partials = extension_opening_width
        .checked_mul(opening_shape.num_polynomials())
        .ok_or_else(|| {
            AkitaError::InvalidSetup("extension opening claim count is zero or overflows".into())
        })?;
    let opening_vars = opening_shape.num_vars();
    let split_bits = extension_opening_width.trailing_zeros() as usize;
    if split_bits > opening_vars {
        return Ok(ExtensionOpeningReductionGeometry::Infeasible {
            split_bits,
            opening_vars,
        });
    }
    Ok(ExtensionOpeningReductionGeometry::Required {
        partials,
        opening_vars,
    })
}

/// Total sumcheck rounds (`col_bits + ring_bits`) for one fold level.
pub fn sumcheck_rounds(level_d: usize, output_witness_len: usize) -> usize {
    let ring_bits = level_d.trailing_zeros() as usize;
    let num_ring_elems = output_witness_len.div_ceil(level_d);
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    col_bits + ring_bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TailSegmentGroupLayout, TailSegmentLayout};

    fn sample_terminal_shape() -> TerminalResponseShape {
        TerminalResponseShape {
            layout: TailSegmentLayout {
                ring_dimension: 64,
                groups: vec![TailSegmentGroupLayout {
                    z_coords: 1024,
                    e_field_elems: 64,
                    t_field_elems: 128,
                    z_linf_cap: Some(2_570),
                    z_rice_low_bits: 9,
                    z_payload_bytes: 4_096,
                }],
                logical_num_elems: 1_216,
            },
        }
    }

    fn level_bytes(width: usize, num_claims: usize) -> Result<usize, AkitaError> {
        extension_opening_reduction_level_bytes(
            128,
            width,
            PolynomialGroupLayout::new(12, num_claims),
        )
    }

    #[test]
    fn invalid_extension_widths_error_instead_of_pricing_zero() {
        for width in [0, 3, usize::MAX] {
            assert!(
                level_bytes(width, 1).is_err(),
                "width {width} must be rejected"
            );
        }
    }

    #[test]
    fn valid_extension_widths_price_suffix_and_terminal_eor() {
        assert_eq!(
            level_bytes(1, 1).expect("degree one"),
            0,
            "single-field geometry contributes no EOR bytes"
        );
        assert!(
            level_bytes(4, 1).expect("extension suffix or terminal") > 0,
            "extension-field EvaluationTrace suffixes and terminals retain EOR"
        );
    }

    #[test]
    fn candidate_aware_eor_distinguishes_local_miss_from_bad_policy() {
        assert_eq!(
            try_extension_opening_reduction_level_bytes(
                128,
                16,
                PolynomialGroupLayout::singleton(3),
            )
            .expect("valid policy"),
            None,
            "four split bits do not fit a three-variable recursive opening"
        );
        assert!(
            try_extension_opening_reduction_level_bytes(
                128,
                16,
                PolynomialGroupLayout::singleton(4),
            )
            .expect("valid sibling geometry")
            .expect("feasible sibling")
                > 0
        );
        assert!(try_extension_opening_reduction_level_bytes(
            128,
            3,
            PolynomialGroupLayout::singleton(4),
        )
        .is_err());
        assert!(try_extension_opening_reduction_level_bytes(
            128,
            4,
            PolynomialGroupLayout::new(4, 0),
        )
        .is_err());
    }

    #[test]
    fn eor_bytes_scale_with_claim_count() {
        let one_claim = level_bytes(4, 1).expect("one claim");
        let two_claims = level_bytes(4, 2).expect("two claims");
        let per_claim_bytes = 5 * field_bytes(128);
        assert_eq!(two_claims - one_claim, per_claim_bytes);
    }

    #[test]
    fn recursive_batched_eor_prices_one_sumcheck_and_each_terminal_claim() {
        let bytes =
            extension_opening_reduction_level_bytes(128, 4, PolynomialGroupLayout::new(12, 2))
                .expect("two-claim recursive EOR");

        let elem_bytes = field_bytes(128);
        let partial_bytes = 4 * 2 * elem_bytes;
        let sumcheck_bytes = (12 - 2) * EXTENSION_OPENING_REDUCTION_DEGREE * elem_bytes;
        let terminal_claim_bytes = 2 * elem_bytes;
        assert_eq!(bytes, partial_bytes + sumcheck_bytes + terminal_claim_bytes);
    }

    #[test]
    fn terminal_l2_planner_estimate_does_not_change_the_wire_cap() {
        let shape = sample_terminal_shape();
        let original = shape.clone();
        let scheduled = terminal_response_bytes(64, &shape);
        let estimated = terminal_response_planner_bytes(64, &shape, Some(1 << 20));

        assert!(estimated < scheduled);
        assert_eq!(
            shape, original,
            "planning must not mutate scheduled geometry"
        );
        assert_eq!(shape.layout.groups[0].z_payload_bytes, 4_096);
        assert_eq!(terminal_response_planner_bytes(64, &shape, None), scheduled);
    }

    #[test]
    fn terminal_l2_planner_estimate_falls_back_for_multiple_groups() {
        let mut shape = sample_terminal_shape();
        shape.layout.groups.push(shape.layout.groups[0]);
        shape.layout.logical_num_elems *= 2;
        assert_eq!(
            terminal_response_planner_bytes(64, &shape, Some(1 << 20)),
            terminal_response_bytes(64, &shape)
        );
    }
}
