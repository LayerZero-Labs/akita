//! Prover-side commitment-compression arithmetic.
#![allow(dead_code)]

use crate::{gadget_row_scalars, CommitmentCompressionPlan, FlatRingVec, TraceTable};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase};

/// Materialized prover data for one compressed commitment payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionEvaluation<F: FieldCore> {
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

/// Fused stage-2 linearization for one compression chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionLinearization<E: FieldCore> {
    /// Dense stage-2 weights over the flat next-witness table.
    pub table: TraceTable<E>,
    /// Public side of the same row-linear combination.
    pub claim: E,
    /// Row weight to use for the next fused compression row.
    pub next_row_weight: E,
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

fn add_table_weight<E: FieldCore>(
    table: &mut [E],
    witness_len: usize,
    flat_idx: usize,
    weight: E,
) -> Result<(), AkitaError> {
    if flat_idx >= witness_len {
        return Err(AkitaError::InvalidSize {
            expected: witness_len,
            actual: flat_idx + 1,
        });
    }
    table[flat_idx] += weight;
    Ok(())
}

#[inline]
fn negacyclic_coeff_weight<F: FieldCore, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    product_coeff: usize,
    witness_coeff: usize,
) -> F {
    let coeffs = ring.coefficients();
    if witness_coeff <= product_coeff {
        coeffs[product_coeff - witness_coeff]
    } else {
        -coeffs[product_coeff + D - witness_coeff]
    }
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
pub fn evaluate_commitment_compression<F>(
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

/// Build the fused linear stage-2 table for a scheduled compression chain.
///
/// The generated rows enforce:
///
/// - for each non-final layer, `F_i * digits_i = G * digits_{i+1}`;
/// - for the final layer, `F_i * digits_i = public_payload`.
///
/// `suffix_offset` is the flat witness offset where this commitment's hidden
/// suffix begins. The first layer's input digits start there, and each
/// non-final layer's output decomposition digits immediately follow.
#[allow(clippy::too_many_arguments)]
pub fn linearize_compression_chain<F, E>(
    setup_fields: &[F],
    plan: &CommitmentCompressionPlan,
    public_payload: &FlatRingVec<F>,
    suffix_offset: usize,
    live_x_cols: usize,
    y_len: usize,
    row_challenge: E,
    initial_row_weight: E,
    log_basis: u32,
) -> Result<CompressionLinearization<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + LiftBase<F>,
{
    if public_payload.coeff_len() != plan.public_len {
        return Err(AkitaError::InvalidSize {
            expected: plan.public_len,
            actual: public_payload.coeff_len(),
        });
    }
    let witness_len = live_x_cols
        .checked_mul(y_len)
        .ok_or_else(|| AkitaError::InvalidSetup("compression table length overflow".to_string()))?;
    let suffix_end = suffix_offset
        .checked_add(plan.padded_suffix_len)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("compression suffix offset overflow".to_string())
        })?;
    if suffix_end > witness_len {
        return Err(AkitaError::InvalidSize {
            expected: witness_len,
            actual: suffix_end,
        });
    }
    if plan.layers.is_empty() {
        return Ok(CompressionLinearization {
            table: TraceTable::ring_dense(vec![E::zero(); witness_len]),
            claim: E::zero(),
            next_row_weight: initial_row_weight,
        });
    }

    let mut table = vec![E::zero(); witness_len];
    let mut claim = E::zero();
    let gadget: Vec<E> = gadget_row_scalars::<F>(decomposition_digits_per_scalar(plan)?, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let mut input_offset = 0usize;
    let mut row_weight = initial_row_weight;

    for (layer_idx, layer) in plan.layers.iter().enumerate() {
        if layer.layer != layer_idx {
            return Err(AkitaError::InvalidSetup(format!(
                "compression layer index mismatch: expected {layer_idx}, got {}",
                layer.layer
            )));
        }
        let setup_entries = layer
            .output_len
            .checked_mul(layer.input_len)
            .ok_or_else(|| AkitaError::InvalidSetup("compression map size overflow".to_string()))?;
        let setup_end = layer
            .setup_offset
            .checked_add(setup_entries)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression setup offset overflow".to_string())
            })?;
        let matrix = setup_fields
            .get(layer.setup_offset..setup_end)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression setup map exceeds shared setup".to_string())
            })?;
        let input_start = suffix_offset.checked_add(input_offset).ok_or_else(|| {
            AkitaError::InvalidSetup("compression input offset overflow".to_string())
        })?;
        let next_input_offset = input_offset.checked_add(layer.input_len).ok_or_else(|| {
            AkitaError::InvalidSetup("compression suffix cursor overflow".to_string())
        })?;
        let is_final = layer_idx + 1 == plan.layers.len();
        let next_digit_start = suffix_offset
            .checked_add(next_input_offset)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression next-digit offset overflow".to_string())
            })?;

        for row_idx in 0..layer.output_len {
            let row = &matrix[row_idx * layer.input_len..(row_idx + 1) * layer.input_len];
            for (col_idx, &entry) in row.iter().enumerate() {
                if !entry.is_zero() {
                    add_table_weight(
                        &mut table,
                        witness_len,
                        input_start + col_idx,
                        row_weight * E::lift_base(entry),
                    )?;
                }
            }
            if is_final {
                let public_value = public_payload
                    .coeffs()
                    .get(row_idx)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                claim += row_weight * E::lift_base(public_value);
            } else {
                for (digit_idx, &g) in gadget.iter().enumerate() {
                    add_table_weight(
                        &mut table,
                        witness_len,
                        next_digit_start + row_idx * gadget.len() + digit_idx,
                        -row_weight * g,
                    )?;
                }
            }
            row_weight *= row_challenge;
        }
        input_offset = next_input_offset;
    }
    if input_offset != plan.suffix_len {
        return Err(AkitaError::InvalidSize {
            expected: plan.suffix_len,
            actual: input_offset,
        });
    }

    Ok(CompressionLinearization {
        table: TraceTable::ring_dense(table),
        claim,
        next_row_weight: row_weight,
    })
}

/// Linearize `ring_matrix * source = G * first_compression_digits`.
///
/// This is the bridge from an omitted raw ring commitment row (`D` for `v`, or
/// `B` for a raw `u`) to the first scalar digit block used by the compression
/// chain. Rows are scalarized coefficient-by-coefficient under one row RLC.
#[allow(clippy::too_many_arguments)]
pub fn linearize_raw_ring_rows_to_first_digits<F, E, const D: usize>(
    matrix_rows: &[CyclotomicRing<F, D>],
    row_len: usize,
    col_len: usize,
    source_offset_cols: usize,
    plan: &CommitmentCompressionPlan,
    suffix_offset: usize,
    live_x_cols: usize,
    y_len: usize,
    row_challenge: E,
    initial_row_weight: E,
    log_basis: u32,
) -> Result<CompressionLinearization<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + LiftBase<F>,
{
    if matrix_rows.len()
        != row_len.checked_mul(col_len).ok_or_else(|| {
            AkitaError::InvalidSetup("raw compression matrix shape overflow".to_string())
        })?
    {
        return Err(AkitaError::InvalidSize {
            expected: row_len * col_len,
            actual: matrix_rows.len(),
        });
    }
    let num_digits = decomposition_digits_per_scalar(plan)?;
    let raw_len = row_len.checked_mul(D).ok_or_else(|| {
        AkitaError::InvalidSetup("raw compression payload length overflow".to_string())
    })?;
    if plan.raw_len != raw_len {
        return Err(AkitaError::InvalidSize {
            expected: raw_len,
            actual: plan.raw_len,
        });
    }
    if plan
        .layers
        .first()
        .map(|layer| layer.input_len)
        .unwrap_or(0)
        != raw_len * num_digits
    {
        return Err(AkitaError::InvalidSetup(
            "first compression input does not match raw digit length".to_string(),
        ));
    }
    let witness_len = live_x_cols
        .checked_mul(y_len)
        .ok_or_else(|| AkitaError::InvalidSetup("compression table length overflow".to_string()))?;
    if y_len != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: y_len,
        });
    }
    let source_end = source_offset_cols
        .checked_add(col_len)
        .and_then(|cols| cols.checked_mul(D))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("raw compression source offset overflow".to_string())
        })?;
    if source_end > witness_len {
        return Err(AkitaError::InvalidSize {
            expected: witness_len,
            actual: source_end,
        });
    }
    let suffix_end = suffix_offset
        .checked_add(plan.padded_suffix_len)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("compression suffix offset overflow".to_string())
        })?;
    if suffix_end > witness_len {
        return Err(AkitaError::InvalidSize {
            expected: witness_len,
            actual: suffix_end,
        });
    }

    let gadget: Vec<E> = gadget_row_scalars::<F>(num_digits, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let mut table = vec![E::zero(); witness_len];
    let mut row_weight = initial_row_weight;

    for row_idx in 0..row_len {
        for product_coeff in 0..D {
            let scalar_idx = row_idx * D + product_coeff;
            for col_idx in 0..col_len {
                let setup_ring = &matrix_rows[row_idx * col_len + col_idx];
                for witness_coeff in 0..D {
                    let coeff = negacyclic_coeff_weight(setup_ring, product_coeff, witness_coeff);
                    if !coeff.is_zero() {
                        add_table_weight(
                            &mut table,
                            witness_len,
                            (source_offset_cols + col_idx) * D + witness_coeff,
                            row_weight * E::lift_base(coeff),
                        )?;
                    }
                }
            }
            for (digit_idx, &gadget_weight) in gadget.iter().enumerate() {
                add_table_weight(
                    &mut table,
                    witness_len,
                    suffix_offset + scalar_idx * num_digits + digit_idx,
                    -row_weight * gadget_weight,
                )?;
            }
            row_weight *= row_challenge;
        }
    }

    Ok(CompressionLinearization {
        table: TraceTable::ring_dense(table),
        claim: E::zero(),
        next_row_weight: row_weight,
    })
}

pub fn decomposition_digits_per_scalar(
    plan: &CommitmentCompressionPlan,
) -> Result<usize, AkitaError> {
    let first = plan
        .layers
        .first()
        .ok_or_else(|| AkitaError::InvalidSetup("compression plan has no layers".to_string()))?;
    if plan.raw_len == 0 || !first.input_len.is_multiple_of(plan.raw_len) {
        return Err(AkitaError::InvalidSetup(
            "compression first layer does not match raw payload length".to_string(),
        ));
    }
    Ok(first.input_len / plan.raw_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompressionLayerPlan, CompressionMapRole};
    use akita_field::Fp32;

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

    fn table_dot_suffix(
        table: &TraceTable<F>,
        prefix_len: usize,
        suffix: &[i8],
        live_x_cols: usize,
        y_len: usize,
    ) -> F {
        let dense = table.materialize_dense(live_x_cols, y_len);
        suffix
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (idx, &digit)| {
                acc + dense[prefix_len + idx] * F::from_i64(digit as i64)
            })
    }

    #[test]
    fn linearizes_one_layer_chain_into_stage2_table() {
        let raw = vec![F::from_u64(7), F::from_u64(9)];
        let num_digits = 4;
        let input_len = raw.len() * num_digits;
        let setup = vec![F::one(); input_len];
        let plan = CommitmentCompressionPlan {
            raw_len: raw.len(),
            public_len: 1,
            suffix_len: input_len,
            padded_suffix_len: input_len,
            layers: vec![CompressionLayerPlan {
                role: CompressionMapRole::H,
                layer: 0,
                input_len,
                output_len: 1,
                setup_offset: 0,
            }],
        };
        let eval =
            evaluate_commitment_compression(&setup, &plan, &raw, num_digits, 2).expect("compress");
        let prefix_len = 4;
        let live_x_cols = 4;
        let y_len = 4;

        let linear = linearize_compression_chain(
            &setup,
            &plan,
            &eval.public_payload,
            prefix_len,
            live_x_cols,
            y_len,
            F::from_u64(11),
            F::one(),
            2,
        )
        .expect("linearize");

        assert_eq!(
            table_dot_suffix(
                &linear.table,
                prefix_len,
                &eval.suffix_digits,
                live_x_cols,
                y_len
            ),
            linear.claim
        );
    }

    #[test]
    fn linearizes_two_layer_chain_with_intermediate_recomposition() {
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
        let eval =
            evaluate_commitment_compression(&setup, &plan, &raw, num_digits, 2).expect("compress");
        let prefix_len = 8;
        let live_x_cols = 6;
        let y_len = 4;

        let linear = linearize_compression_chain(
            &setup,
            &plan,
            &eval.public_payload,
            prefix_len,
            live_x_cols,
            y_len,
            F::from_u64(13),
            F::one(),
            2,
        )
        .expect("linearize");

        assert_eq!(
            table_dot_suffix(
                &linear.table,
                prefix_len,
                &eval.suffix_digits,
                live_x_cols,
                y_len
            ),
            linear.claim
        );
    }

    #[test]
    fn linearizes_raw_ring_row_to_first_digits() {
        const D: usize = 4;
        let setup_ring = CyclotomicRing::<F, D>::from_coefficients([
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
        ]);
        let source_ring = CyclotomicRing::<F, D>::from_coefficients([
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
        ]);
        let raw_ring = setup_ring * source_ring;
        let raw = raw_ring.coefficients().to_vec();
        let num_digits = 4;
        let input_len = raw.len() * num_digits;
        let compression_setup = vec![F::one(); input_len];
        let plan = CommitmentCompressionPlan {
            raw_len: raw.len(),
            public_len: 1,
            suffix_len: input_len,
            padded_suffix_len: input_len,
            layers: vec![CompressionLayerPlan {
                role: CompressionMapRole::H,
                layer: 0,
                input_len,
                output_len: 1,
                setup_offset: 0,
            }],
        };
        let eval = evaluate_commitment_compression(&compression_setup, &plan, &raw, num_digits, 2)
            .expect("compress");
        let mut witness = source_ring.coefficients().to_vec();
        witness.extend(
            eval.suffix_digits
                .iter()
                .map(|&digit| F::from_i64(digit as i64)),
        );
        let live_x_cols = 5;
        let y_len = D;

        let linear = linearize_raw_ring_rows_to_first_digits(
            &[setup_ring],
            1,
            1,
            0,
            &plan,
            D,
            live_x_cols,
            y_len,
            F::from_u64(23),
            F::one(),
            2,
        )
        .expect("linearize raw");
        let dense = linear.table.materialize_dense(live_x_cols, y_len);
        let dot = witness
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (idx, &value)| acc + dense[idx] * value);

        assert_eq!(dot, F::zero());
        assert_eq!(linear.claim, F::zero());
    }
}
