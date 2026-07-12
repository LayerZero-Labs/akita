use super::rows::{checked_add, checked_padded_row_count};
use super::*;
use crate::layout::compression::semantics::{
    self as compression_semantics, CompressionRowRhs, SegmentId as CompressionSegmentId,
};
use crate::sis::compute_num_digits_full_field;
use crate::witness::{WitnessChunkLayout, WitnessChunkLengths};
fn allocate(cursor: &mut usize, len: usize, label: &str) -> Result<CoeffSpan, AkitaError> {
    if len == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "logical {label} must be non-zero"
        )));
    }
    let start = *cursor;
    // This cursor describes a logical, streamed coefficient arena; it is not a
    // verifier allocation or serialized sequence and may exceed the per-carrier
    // allocation cap. Concrete row domains and compact carriers enforce their
    // own caps at their allocation boundaries.
    *cursor = start
        .checked_add(len)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("logical {label} overflow")))?;
    Ok(CoeffSpan { start, len })
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, AkitaError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("logical {label} overflow")))
    })
}

fn compression_id(id: CompressionSegmentId) -> RelationSegmentId {
    match id {
        CompressionSegmentId::Xi { source, map } => {
            RelationSegmentId::CompressionInput { source, map }
        }
        CompressionSegmentId::Quotient { source, map } => {
            RelationSegmentId::CompressionQuotient { source, map }
        }
    }
}

fn segment_span(
    segments: &[RelationSegment],
    id: RelationSegmentId,
) -> Result<CoeffSpan, AkitaError> {
    segments
        .iter()
        .find_map(|segment| (segment.id == id).then_some(segment.span))
        .ok_or_else(|| AkitaError::InvalidSetup("relation segment reference is missing".into()))
}

pub(super) fn normalize_support(
    mut spans: Vec<CoeffSpan>,
    total: usize,
) -> Result<Vec<CoeffSpan>, AkitaError> {
    spans.sort_unstable_by_key(|span| span.start);
    let mut normalized: Vec<CoeffSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        if span.len == 0 || span.end()? > total {
            return Err(AkitaError::InvalidSetup(
                "negative-binary support is empty or outside the logical arena".into(),
            ));
        }
        if let Some(last) = normalized.last_mut() {
            let last_end = last.end()?;
            if span.start < last_end {
                return Err(AkitaError::InvalidSetup(
                    "negative-binary support overlaps".into(),
                ));
            }
            if span.start == last_end {
                last.len = last.len.checked_add(span.len).ok_or_else(|| {
                    AkitaError::InvalidSetup("negative-binary support overflow".into())
                })?;
                continue;
            }
        }
        normalized.push(span);
    }
    Ok(normalized)
}

fn family_for_source(source: CompressionSourceId) -> RelationRowId {
    match source {
        CompressionSourceId::CurrentOuter => RelationRowId::B {
            group: RelationGroupId::Current,
        },
        CompressionSourceId::PrecommittedOuter { index } => RelationRowId::B {
            group: RelationGroupId::Precommitted { index },
        },
        CompressionSourceId::Opening => RelationRowId::D,
    }
}

/// Compile the logical relation and its physical base-witness projection.
pub(super) fn compile_relation_layout(
    lp: &LevelParams,
    opening: &OpeningClaimsLayout,
    row_layout: RelationMatrixRowLayout,
    field_bits: u32,
    compression: Option<&compression_semantics::CompiledCompressionSemantics>,
) -> Result<RelationLayout, AkitaError> {
    if lp.log_basis == 0 || lp.log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(
            "relation log_basis must be in 1..128".into(),
        ));
    }
    if field_bits != lp.field_bits_for_cache() {
        return Err(AkitaError::InvalidSetup(
            "relation field width disagrees with authenticated level parameters".into(),
        ));
    }
    let base_plan = RelationRowPlan::compile_base(lp, opening, row_layout)?;
    let final_group = opening.root_final_group_index()?;
    let quotient_levels = compute_num_digits_full_field(field_bits, lp.log_basis);
    if quotient_levels == 0 {
        return Err(AkitaError::InvalidSetup(
            "relation quotient decomposition is empty".into(),
        ));
    }

    let mut coeff_cursor = 0usize;
    let mut segments = Vec::new();
    let groups = base_plan
        .group_order()
        .iter()
        .copied()
        .map(|group| {
            let opening_index = match group {
                RelationGroupId::Current => final_group,
                RelationGroupId::Precommitted { index } => index,
            };
            (group, opening_index)
        })
        .collect::<Vec<_>>();

    // Logical group-major body. This is independent of WitnessLayout chunks.
    for &(group, opening_index) in &groups {
        let params = lp.root_group_params(opening, opening_index)?;
        let claims = opening.group_layout(opening_index)?.num_polynomials();
        let fold_digits = lp.num_digits_fold_for_params(params, claims, field_bits)?;
        let (a_dim, b_dim) = match group {
            RelationGroupId::Current => (
                lp.a_key.sis_table_key().ring_dimension as usize,
                lp.b_key.sis_table_key().ring_dimension as usize,
            ),
            RelationGroupId::Precommitted { index } => {
                let pre = lp.precommitted_groups.get(index).ok_or_else(|| {
                    AkitaError::InvalidSetup("relation precommitted group is missing".into())
                })?;
                (
                    pre.a_key.sis_table_key().ring_dimension as usize,
                    pre.b_key.sis_table_key().ring_dimension as usize,
                )
            }
        };
        let d_dim = lp.d_key.sis_table_key().ring_dimension as usize;
        let body = [
            (
                RelationSegmentId::Z { group },
                checked_product(
                    &[
                        params.block_len(),
                        params.num_digits_commit(),
                        fold_digits,
                        a_dim,
                    ],
                    "Z coefficients",
                )?,
            ),
            (
                RelationSegmentId::E { group },
                checked_product(
                    &[claims, params.num_blocks(), params.num_digits_open(), d_dim],
                    "E coefficients",
                )?,
            ),
            (
                RelationSegmentId::T { group },
                checked_product(
                    &[
                        claims,
                        params.num_blocks(),
                        params.a_rows_len(),
                        params.num_digits_open(),
                        b_dim,
                    ],
                    "T coefficients",
                )?,
            ),
        ];
        for (id, len) in body {
            let span = allocate(&mut coeff_cursor, len, "body coefficients")?;
            segments.push(RelationSegment { id, span });
        }
    }

    let mut families = base_plan.families().to_vec();
    let mut row_cursor = base_plan.trace_row();
    for family in &families {
        let quotient_len = checked_product(
            &[
                family.rows().len(),
                family.native_ring_dim(),
                quotient_levels,
            ],
            "quotient",
        )?;
        let span = allocate(&mut coeff_cursor, quotient_len, "quotient coefficients")?;
        segments.push(RelationSegment {
            id: family.quotient(),
            span,
        });
    }

    let mut support_ids = Vec::new();
    let support_derivation_version = compression_semantics::BINARY_SUPPORT_DERIVATION_VERSION;
    if let Some(local) = compression {
        if local.binary_support_derivation_version != support_derivation_version {
            return Err(AkitaError::InvalidSetup(
                "compression support derivation version mismatch".into(),
            ));
        }
        let compression_base = coeff_cursor;
        let mut expected_local = 0usize;
        for segment in &local.segments {
            if segment.span.start != expected_local || segment.span.len == 0 {
                return Err(AkitaError::InvalidSetup(
                    "compression segments are not canonical and contiguous".into(),
                ));
            }
            expected_local = segment.span.end()?;
            let start = compression_base
                .checked_add(segment.span.start)
                .ok_or_else(|| AkitaError::InvalidSetup("compression offset overflow".into()))?;
            segments.push(RelationSegment {
                id: compression_id(segment.id),
                span: CoeffSpan {
                    start,
                    len: segment.span.len,
                },
            });
        }
        if expected_local != local.total_coeffs {
            return Err(AkitaError::InvalidSetup(
                "compression coefficient total disagrees with its segments".into(),
            ));
        }
        coeff_cursor = compression_base
            .checked_add(local.total_coeffs)
            .ok_or_else(|| AkitaError::InvalidSetup("coefficient total overflow".into()))?;

        for augmentation in &local.augmentations {
            let target = family_for_source(augmentation.source);
            let input = compression_id(augmentation.compression_input);
            let family = families
                .iter_mut()
                .find(|family| family.id == target)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("compression source row family is absent".into())
                })?;
            let expected_rhs = match augmentation.source {
                CompressionSourceId::CurrentOuter => RelationRowRhs::Commitment {
                    group: RelationGroupId::Current,
                },
                CompressionSourceId::PrecommittedOuter { index } => RelationRowRhs::Commitment {
                    group: RelationGroupId::Precommitted { index },
                },
                CompressionSourceId::Opening => RelationRowRhs::Opening,
            };
            if family.rhs != expected_rhs {
                return Err(AkitaError::InvalidSetup(
                    "compression source RHS is not the expected base payload".into(),
                ));
            }
            family.rhs = RelationRowRhs::Zero;
            let compression_input = match &mut family.inputs {
                RelationRowInputs::B {
                    compression_input, ..
                }
                | RelationRowInputs::D {
                    compression_input, ..
                } => compression_input,
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "compression source does not name an augmentable B/D family".into(),
                    ));
                }
            };
            if compression_input
                .replace(GadgetInput {
                    segment: input,
                    log_basis: augmentation.compression_log_basis,
                })
                .is_some()
            {
                return Err(AkitaError::InvalidSetup(
                    "compression source row family is augmented twice".into(),
                ));
            }
            let _ = segment_span(&segments, input)?;
        }

        let compression_row_base = row_cursor;
        let mut expected_local_row = 0usize;
        for row in &local.rows {
            if row.rows.start != expected_local_row || row.rows.len == 0 {
                return Err(AkitaError::InvalidSetup(
                    "compression row spans are not canonical and contiguous".into(),
                ));
            }
            expected_local_row = checked_add(expected_local_row, row.rows.len, "local rows")?;
            let rows = RowSpan {
                start: checked_add(
                    compression_row_base,
                    row.rows.start,
                    "compression row offset",
                )?,
                len: row.rows.len,
            };
            let input = compression_id(row.input);
            let successor = row.successor.map(compression_id);
            let successor_input = match (successor, row.successor_log_basis) {
                (Some(segment), Some(log_basis)) => Some(GadgetInput { segment, log_basis }),
                (None, None) => None,
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "compression successor and gadget basis disagree".into(),
                    ));
                }
            };
            let quotient = compression_id(row.quotient);
            let _ = segment_span(&segments, input)?;
            let _ = segment_span(&segments, quotient)?;
            if let Some(successor) = successor {
                let _ = segment_span(&segments, successor)?;
            }
            families.push(RelationRowFamily {
                id: RelationRowId::Compression {
                    source: row.id.source,
                    map: row.id.map,
                },
                rows,
                native_ring_dim: row.native_ring_dim,
                quotient,
                inputs: RelationRowInputs::Compression {
                    input,
                    successor: successor_input,
                },
                rhs: match row.rhs {
                    CompressionRowRhs::Zero => RelationRowRhs::Zero,
                    CompressionRowRhs::TerminalPayload { coeffs } => {
                        RelationRowRhs::TerminalPayload { coeffs }
                    }
                },
            });
        }
        if expected_local_row != local.total_rows {
            return Err(AkitaError::InvalidSetup(
                "compression row total disagrees with its spans".into(),
            ));
        }
        row_cursor = checked_add(compression_row_base, local.total_rows, "relation rows")?;
        support_ids.extend(
            local
                .negative_binary_inputs
                .iter()
                .copied()
                .map(compression_id),
        );
    }

    let support = support_ids
        .into_iter()
        .map(|id| segment_span(&segments, id))
        .collect::<Result<Vec<_>, _>>()?;
    let negative_binary_support = normalize_support(support, coeff_cursor)?;
    let trace_row = row_cursor;
    let padded_row_count = checked_padded_row_count(trace_row)?;

    let row_plan = RelationRowPlan {
        group_order: groups.iter().map(|(group, _)| *group).collect(),
        families,
        trace_row,
        padded_row_count,
    };
    let witness_layout =
        RelationLayout::compile_witness_layout(&segments, &row_plan, lp, field_bits)?;
    Ok(RelationLayout {
        segments,
        row_plan,
        negative_binary_support,
        support_derivation_version,
        total_coeffs: coeff_cursor,
        witness_layout,
    })
}

impl RelationLayout {
    fn body_ring_len(
        segments: &[RelationSegment],
        id: RelationSegmentId,
        native_ring_dim: usize,
    ) -> Result<usize, AkitaError> {
        let len = segments
            .iter()
            .find_map(|segment| (segment.id == id).then_some(segment.span.len))
            .ok_or_else(|| AkitaError::InvalidSetup("relation body segment is absent".into()))?;
        if native_ring_dim == 0 || !len.is_multiple_of(native_ring_dim) {
            return Err(AkitaError::InvalidSetup(
                "relation body segment disagrees with its native ring dimension".into(),
            ));
        }
        Ok(len / native_ring_dim)
    }

    /// Project base `Z/E/T` and base quotient geometry into physical chunks.
    ///
    /// For one group, multiple chunks repeat `Z`, partition `E/T`, and place the
    /// shared base-quotient `r` tail only in the last chunk. For multiple groups,
    /// configured multi-chunk mode is rejected; one full `Z/E/T` chunk is emitted
    /// per group in current-first order, with the shared `r` tail only in the last
    /// group chunk. Compression inputs and quotients remain outside
    /// [`WitnessLayout`].
    fn compile_witness_layout(
        segments: &[RelationSegment],
        row_plan: &RelationRowPlan,
        lp: &LevelParams,
        field_bits: u32,
    ) -> Result<WitnessLayout, AkitaError> {
        lp.witness_chunk.validate()?;
        let num_chunks = lp.witness_chunk.num_chunks;
        if num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness chunk count must be at least one".into(),
            ));
        }
        let base_rows = row_plan.families.iter().try_fold(0usize, |total, family| {
            if matches!(family.quotient, RelationSegmentId::BaseQuotient { .. }) {
                total.checked_add(family.rows.len).ok_or_else(|| {
                    AkitaError::InvalidSetup("base quotient row count overflow".into())
                })
            } else {
                Ok(total)
            }
        })?;
        let r_len = base_rows
            .checked_mul(compute_num_digits_full_field(field_bits, lp.log_basis))
            .ok_or_else(|| AkitaError::InvalidSetup("r-tail length overflow".into()))?;
        let mut group_lens = Vec::with_capacity(row_plan.group_order.len());
        for &group in &row_plan.group_order {
            let (a_dim, b_dim) = match group {
                RelationGroupId::Current => (
                    lp.a_key.sis_table_key().ring_dimension as usize,
                    lp.b_key.sis_table_key().ring_dimension as usize,
                ),
                RelationGroupId::Precommitted { index } => {
                    let pre = lp.precommitted_groups.get(index).ok_or_else(|| {
                        AkitaError::InvalidSetup("relation witness group is absent".into())
                    })?;
                    (
                        pre.a_key.sis_table_key().ring_dimension as usize,
                        pre.b_key.sis_table_key().ring_dimension as usize,
                    )
                }
            };
            group_lens.push((
                Self::body_ring_len(segments, RelationSegmentId::Z { group }, a_dim)?,
                Self::body_ring_len(
                    segments,
                    RelationSegmentId::E { group },
                    lp.d_key.sis_table_key().ring_dimension as usize,
                )?,
                Self::body_ring_len(segments, RelationSegmentId::T { group }, b_dim)?,
            ));
        }

        let layout = if group_lens.len() > 1 {
            lp.reject_multi_group_multi_chunk("RelationLayout::witness_layout")?;
            let mut base = 0usize;
            let mut chunks = Vec::with_capacity(group_lens.len());
            let mut chunk_lengths = Vec::with_capacity(group_lens.len());
            for (position, &(z_len, e_len, t_len)) in group_lens.iter().enumerate() {
                let offset_e = base.checked_add(z_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group e offset overflow".into())
                })?;
                let offset_t = offset_e.checked_add(e_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group t offset overflow".into())
                })?;
                let after_t = offset_t.checked_add(t_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group witness stride overflow".into())
                })?;
                let last = position + 1 == group_lens.len();
                chunks.push(WitnessChunkLayout {
                    offset_z: base,
                    offset_e,
                    offset_t,
                    offset_r: last.then_some(after_t),
                    global_block_base: 0,
                });
                chunk_lengths.push(WitnessChunkLengths {
                    z_len,
                    e_len,
                    t_len,
                    r_len: last.then_some(r_len),
                });
                base = after_t;
            }
            WitnessLayout {
                blocks_per_chunk: lp.num_blocks,
                chunks,
                chunk_lengths,
            }
        } else {
            let &(z_len, e_len, t_len) = group_lens.first().ok_or_else(|| {
                AkitaError::InvalidSetup("relation witness group is missing".into())
            })?;
            if num_chunks > 1
                && (!num_chunks.is_power_of_two()
                    || num_chunks > lp.num_blocks
                    || !lp.num_blocks.is_multiple_of(num_chunks)
                    || !e_len.is_multiple_of(num_chunks)
                    || !t_len.is_multiple_of(num_chunks))
            {
                return Err(AkitaError::InvalidSetup(
                    "invalid partitioned witness chunk geometry".into(),
                ));
            }
            let blocks_per_chunk = lp.num_blocks / num_chunks;
            if !blocks_per_chunk.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "witness chunk block window must be a power of two".into(),
                ));
            }
            let e_chunk = e_len / num_chunks;
            let t_chunk = t_len / num_chunks;
            let stride = z_len
                .checked_add(e_chunk)
                .and_then(|n| n.checked_add(t_chunk))
                .ok_or_else(|| AkitaError::InvalidSetup("witness chunk stride overflow".into()))?;
            let mut chunks = Vec::with_capacity(num_chunks);
            let mut chunk_lengths = Vec::with_capacity(num_chunks);
            for index in 0..num_chunks {
                let base = index.checked_mul(stride).ok_or_else(|| {
                    AkitaError::InvalidSetup("witness chunk base overflow".into())
                })?;
                let offset_e = base
                    .checked_add(z_len)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness e offset overflow".into()))?;
                let offset_t = offset_e
                    .checked_add(e_chunk)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness t offset overflow".into()))?;
                let after_t = offset_t
                    .checked_add(t_chunk)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness r offset overflow".into()))?;
                let last = index + 1 == num_chunks;
                chunks.push(WitnessChunkLayout {
                    offset_z: base,
                    offset_e,
                    offset_t,
                    offset_r: last.then_some(after_t),
                    global_block_base: index.checked_mul(blocks_per_chunk).ok_or_else(|| {
                        AkitaError::InvalidSetup("global block base overflow".into())
                    })?,
                });
                chunk_lengths.push(WitnessChunkLengths {
                    z_len,
                    e_len: e_chunk,
                    t_len: t_chunk,
                    r_len: last.then_some(r_len),
                });
            }
            WitnessLayout {
                blocks_per_chunk,
                chunks,
                chunk_lengths,
            }
        };
        Ok(layout)
    }
}
