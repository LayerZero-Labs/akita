//! Prover-side commitment-compression arithmetic.
#![allow(dead_code)]

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_types::{CommitmentCompressionPlan, FlatRingVec};

/// Materialized prover data for one compressed commitment payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompressionEvaluation<F: FieldCore> {
    /// Final public payload sent on the wire.
    pub public_payload: FlatRingVec<F>,
    /// Hidden compression suffix digits, padded as scheduled for appending to
    /// the next recursive witness.
    pub padded_suffix_digits: Vec<i8>,
    /// Unpadded hidden compression suffix digits.
    pub suffix_digits: Vec<i8>,
    /// Field outputs after each compression layer.
    pub layer_outputs: Vec<Vec<F>>,
}

fn decompose_scalars_i8<F>(
    values: &[F],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<i8>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if num_digits == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression decomposition requires nonzero digit count".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(values.len() * num_digits);
    for &value in values {
        let ring = CyclotomicRing::<F, 1>::from_coefficients([value]);
        let planes = ring.balanced_decompose_pow2_i8(num_digits, log_basis);
        out.extend(planes.into_iter().map(|plane| plane[0]));
    }
    Ok(out)
}

fn dense_layer_apply<F>(
    setup_fields: &[F],
    setup_offset: usize,
    output_len: usize,
    input: &[i8],
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    let entry_count = output_len
        .checked_mul(input.len())
        .ok_or_else(|| AkitaError::InvalidSetup("compression map size overflow".to_string()))?;
    let setup_end = setup_offset
        .checked_add(entry_count)
        .ok_or_else(|| AkitaError::InvalidSetup("compression setup offset overflow".to_string()))?;
    let matrix = setup_fields.get(setup_offset..setup_end).ok_or_else(|| {
        AkitaError::InvalidSetup("compression setup map exceeds shared setup".to_string())
    })?;
    let mut out = vec![F::zero(); output_len];
    for (row_idx, row) in matrix.chunks_exact(input.len()).enumerate() {
        let mut acc = F::zero();
        for (&a, &digit) in row.iter().zip(input) {
            if digit != 0 {
                acc += a * F::from_i64(digit as i64);
            }
        }
        out[row_idx] = acc;
    }
    Ok(out)
}

/// Evaluate a scheduled scalar commitment-compression chain.
///
/// `raw_payload` is the uncompressed field-coordinate payload. For a ring
/// commitment, callers flatten ring rows into coefficients before calling.
///
/// # Errors
///
/// Returns an error if the plan shape does not match the raw payload, if a
/// layer extends past the shared setup slice, or if the produced suffix/public
/// lengths do not match the schedule.
pub(crate) fn evaluate_commitment_compression<F>(
    setup_fields: &[F],
    plan: &CommitmentCompressionPlan,
    raw_payload: &[F],
    num_digits: usize,
    log_basis: u32,
) -> Result<CompressionEvaluation<F>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    if raw_payload.len() != plan.raw_len {
        return Err(AkitaError::InvalidSize {
            expected: plan.raw_len,
            actual: raw_payload.len(),
        });
    }
    if plan.layers.is_empty() {
        if plan.public_len != raw_payload.len()
            || plan.suffix_len != 0
            || plan.padded_suffix_len != 0
        {
            return Err(AkitaError::InvalidSetup(
                "empty compression plan must preserve raw payload without suffix".to_string(),
            ));
        }
        return Ok(CompressionEvaluation {
            public_payload: FlatRingVec::from_coeffs(raw_payload.to_vec()),
            padded_suffix_digits: Vec::new(),
            suffix_digits: Vec::new(),
            layer_outputs: Vec::new(),
        });
    }

    let mut current_digits = decompose_scalars_i8(raw_payload, num_digits, log_basis)?;
    let mut suffix_digits = Vec::new();
    let mut layer_outputs = Vec::with_capacity(plan.layers.len());
    for (layer_idx, layer) in plan.layers.iter().enumerate() {
        if layer.layer != layer_idx {
            return Err(AkitaError::InvalidSetup(format!(
                "compression layer index mismatch: expected {layer_idx}, got {}",
                layer.layer
            )));
        }
        if layer.input_len != current_digits.len() {
            return Err(AkitaError::InvalidSize {
                expected: layer.input_len,
                actual: current_digits.len(),
            });
        }
        suffix_digits.extend_from_slice(&current_digits);
        let layer_output = dense_layer_apply(
            setup_fields,
            layer.setup_offset,
            layer.output_len,
            &current_digits,
        )?;
        if layer_idx + 1 == plan.layers.len() {
            layer_outputs.push(layer_output);
            break;
        }
        current_digits = decompose_scalars_i8(&layer_output, num_digits, log_basis)?;
        layer_outputs.push(layer_output);
    }

    if suffix_digits.len() != plan.suffix_len {
        return Err(AkitaError::InvalidSize {
            expected: plan.suffix_len,
            actual: suffix_digits.len(),
        });
    }
    if plan.padded_suffix_len < plan.suffix_len {
        return Err(AkitaError::InvalidSetup(
            "compression padded suffix is shorter than logical suffix".to_string(),
        ));
    }
    let final_output = layer_outputs.last().ok_or_else(|| {
        AkitaError::InvalidSetup("nonempty compression plan produced no output".to_string())
    })?;
    if final_output.len() != plan.public_len {
        return Err(AkitaError::InvalidSize {
            expected: plan.public_len,
            actual: final_output.len(),
        });
    }
    let mut padded_suffix_digits = suffix_digits.clone();
    padded_suffix_digits.resize(plan.padded_suffix_len, 0);
    Ok(CompressionEvaluation {
        public_payload: FlatRingVec::from_coeffs(final_output.clone()),
        padded_suffix_digits,
        suffix_digits,
        layer_outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp32;
    use akita_types::{CompressionLayerPlan, CompressionMapRole};

    type F = Fp32<251>;

    #[test]
    fn evaluates_one_layer_dense_map_and_suffix() {
        let raw = vec![F::from_u64(7), F::from_u64(9)];
        let num_digits = 4;
        let input_len = raw.len() * num_digits;
        let setup = vec![F::one(); input_len];
        let plan = CommitmentCompressionPlan {
            raw_len: raw.len(),
            public_len: 1,
            suffix_len: input_len,
            padded_suffix_len: input_len + 3,
            layers: vec![CompressionLayerPlan {
                role: CompressionMapRole::RootF,
                layer: 0,
                input_len,
                output_len: 1,
                setup_offset: 0,
            }],
        };

        let got =
            evaluate_commitment_compression(&setup, &plan, &raw, num_digits, 2).expect("compress");
        let expected_sum = got
            .suffix_digits
            .iter()
            .fold(F::zero(), |acc, &digit| acc + F::from_i64(digit as i64));

        assert_eq!(got.public_payload.coeffs(), &[expected_sum]);
        assert_eq!(got.suffix_digits.len(), input_len);
        assert_eq!(got.padded_suffix_digits.len(), input_len + 3);
        assert_eq!(&got.padded_suffix_digits[..input_len], got.suffix_digits);
        assert!(got.padded_suffix_digits[input_len..]
            .iter()
            .all(|&x| x == 0));
    }

    #[test]
    fn evaluates_two_layer_chain() {
        let raw = vec![F::from_u64(5)];
        let num_digits = 4;
        let first_input_len = raw.len() * num_digits;
        let first_output_len = 2;
        let second_input_len = first_output_len * num_digits;
        let first_entries = first_output_len * first_input_len;
        let mut setup = vec![F::one(); first_entries];
        setup.extend(vec![F::from_u64(2); second_input_len]);
        let plan = CommitmentCompressionPlan {
            raw_len: raw.len(),
            public_len: 1,
            suffix_len: first_input_len + second_input_len,
            padded_suffix_len: first_input_len + second_input_len,
            layers: vec![
                CompressionLayerPlan {
                    role: CompressionMapRole::F,
                    layer: 0,
                    input_len: first_input_len,
                    output_len: first_output_len,
                    setup_offset: 0,
                },
                CompressionLayerPlan {
                    role: CompressionMapRole::F,
                    layer: 1,
                    input_len: second_input_len,
                    output_len: 1,
                    setup_offset: first_entries,
                },
            ],
        };

        let got =
            evaluate_commitment_compression(&setup, &plan, &raw, num_digits, 2).expect("compress");
        assert_eq!(got.layer_outputs.len(), 2);
        assert_eq!(got.layer_outputs[0].len(), first_output_len);
        assert_eq!(got.public_payload.coeff_len(), 1);
        assert_eq!(got.suffix_digits.len(), plan.suffix_len);
    }
}
