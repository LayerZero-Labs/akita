use super::*;

#[derive(Clone, Copy)]
enum PendingCommitGroupPath {
    Standard(PolynomialType),
    External(ExternalInnerCommitmentCapability),
}

/// Validated source-path selections that have not materialized source data.
pub struct CompiledCommitmentRequest<'a, F: Field> {
    plan: CommitInnerPlan,
    sources: &'a [&'a dyn CommitmentSource<F>],
    descriptors: Vec<CommitSourceDescriptor>,
    path: PendingCommitGroupPath,
}

impl<'a, F: Field> CompiledCommitmentRequest<'a, F> {
    /// Selected standard types; external paths own their resource requirements.
    pub fn selected_polynomial_types(&self) -> impl Iterator<Item = PolynomialType> + '_ {
        let selected = match self.path {
            PendingCommitGroupPath::Standard(selected) => Some(selected),
            PendingCommitGroupPath::External(_) => None,
        };
        selected.into_iter()
    }

    /// Materialize only the representations selected during request compilation.
    pub fn materialize(self) -> Result<Vec<ResolvedCommitSource<'a, F>>, AkitaError> {
        self.sources
            .iter()
            .zip(self.descriptors)
            .map(|(source, descriptor)| {
                let path = match self.path {
                    PendingCommitGroupPath::Standard(selected) => {
                        let selection = PolynomialTypeSelection { selected };
                        let selected = selection.polynomial_type();
                        let representation = source.represent_as(selection, &self.plan)?;
                        validate_materialized_representation(
                            &descriptor,
                            &self.plan,
                            selected,
                            &representation,
                        )?;
                        ResolvedCommitSourcePath::Standard {
                            selected,
                            representation,
                        }
                    }
                    PendingCommitGroupPath::External(capability) => {
                        let prepared =
                            source.prepare_external_inner_commitment(capability, &self.plan)?;
                        if prepared.capability() != capability {
                            return Err(AkitaError::InvalidInput(
                                "source prepared a different external commitment capability".into(),
                            ));
                        }
                        ResolvedCommitSourcePath::External(prepared)
                    }
                };
                Ok(ResolvedCommitSource {
                    descriptor,
                    inner_plan: self.plan,
                    path,
                })
            })
            .collect()
    }
}

fn validate_materialized_representation<F: Field>(
    descriptor: &CommitSourceDescriptor,
    plan: &CommitInnerPlan,
    selected: PolynomialType,
    representation: &PolynomialRepresentation<'_, F>,
) -> Result<(), AkitaError> {
    if representation.polynomial_type()? != selected {
        return Err(AkitaError::InvalidInput(
            "commit source materialized a representation other than the compiled selection".into(),
        ));
    }
    match representation {
        PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(source)) => {
            if source.coefficients().len() != descriptor.total_coefficient_len() {
                return Err(AkitaError::InvalidSize {
                    expected: descriptor.total_coefficient_len(),
                    actual: source.coefficients().len(),
                });
            }
        }
        PolynomialRepresentation::Dense(DenseRepresentation::PredecomposedDigits(planes)) => {
            let logical_len = checked_logical_len(descriptor.num_vars())?;
            let expected_rings = logical_len.div_ceil(plan.ring_dimension);
            let expected_bytes =
                checked::product([expected_rings, plan.num_digits_inner, plan.ring_dimension])
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("dense digit-plane length overflow".into())
                    })?;
            if planes.ring_dimension != plan.ring_dimension
                || planes.num_digits != plan.num_digits_inner
                || planes.log_basis != plan.log_basis_inner
                || planes.logical_ring_count != expected_rings
                || planes.bytes.len() != expected_bytes
            {
                return Err(AkitaError::InvalidInput(
                    "predecomposed dense planes do not match the compiled inner plan".into(),
                ));
            }
        }
        PolynomialRepresentation::ShortNorm(representation) => {
            if representation.live_coefficient_len != descriptor.live_coefficient_len()
                || representation.physical_coefficient_len != descriptor.total_coefficient_len()
                || representation.encoded_bytes.is_empty()
            {
                return Err(AkitaError::InvalidInput(
                    "packed short-norm representation disagrees with its source descriptor".into(),
                ));
            }
        }
        PolynomialRepresentation::OneHot(representation) => {
            let logical_len = checked_logical_len(representation.num_vars)?;
            if representation.chunk_size == 0
                || !logical_len.is_multiple_of(representation.chunk_size)
            {
                return Err(AkitaError::InvalidInput(
                    "one-hot chunk size must exactly divide the logical coefficient count".into(),
                ));
            }
            let expected_positions = logical_len / representation.chunk_size;
            if representation.num_vars != descriptor.num_vars()
                || descriptor.class()
                    != (CommitSourceClass::OneHot {
                        chunk_size: representation.chunk_size,
                    })
                || representation.positions.len() != expected_positions
            {
                return Err(AkitaError::InvalidInput(
                    "one-hot representation disagrees with its source descriptor".into(),
                ));
            }
            if representation
                .positions
                .contains_out_of_range_position(representation.chunk_size)
            {
                return Err(AkitaError::InvalidInput(
                    "one-hot representation contains a position outside its chunk".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Compile source paths for one checked inner plan.
///
/// Discovery and capability intersection complete for the entire source group
/// before any standard or external representation is materialized.
pub fn compile_commitment_request<'a, F: Field>(
    plan: &CommitInnerPlan,
    sources: &'a [&'a dyn CommitmentSource<F>],
    capabilities: &CommitmentRequestCapabilities,
) -> Result<CompiledCommitmentRequest<'a, F>, AkitaError> {
    if sources.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commitment request requires at least one source".into(),
        ));
    }

    let mut candidates = Vec::with_capacity(sources.len());
    let mut expected_num_vars = None;
    for source in sources {
        let descriptor = source.descriptor()?;
        validate_plan_extents(&descriptor, plan)?;
        if expected_num_vars
            .replace(descriptor.num_vars())
            .is_some_and(|expected| expected != descriptor.num_vars())
        {
            return Err(AkitaError::InvalidInput(
                "all commitment sources must have the same num_vars".into(),
            ));
        }
        let available = source.available_polynomial_types(plan)?;
        let external = source.external_inner_commitment_capability(capabilities.backend, plan)?;
        if let Some(external) = external {
            if external.backend() != capabilities.backend
                || external.context_type_id() != capabilities.external_context
                || (capabilities.fused_command_context.is_some()
                    && external.fused_command_context_type_id()
                        != capabilities.fused_command_context)
            {
                return Err(AkitaError::InvalidInput(
                    "external commitment capability is incoherent with the selected operation"
                        .into(),
                ));
            }
        }
        candidates.push((descriptor, available, external));
    }

    let homogeneous_external = candidates
        .first()
        .and_then(|(_, _, external)| *external)
        .filter(|selected| {
            candidates
                .iter()
                .all(|(_, _, external)| *external == Some(*selected))
        });
    let path = if let Some(external) = homogeneous_external {
        PendingCommitGroupPath::External(external)
    } else {
        let common =
            if capabilities.standard_types.is_empty() && capabilities.accepts_any_standard_type {
                candidates.first().and_then(|(_, available, _)| {
                    available.as_slice().iter().copied().find(|offered| {
                        candidates
                            .iter()
                            .all(|(_, available, _)| available.as_slice().contains(offered))
                    })
                })
            } else {
                capabilities
                    .standard_types
                    .iter()
                    .copied()
                    .find(|supported| {
                        candidates
                            .iter()
                            .all(|(_, available, _)| available.as_slice().contains(supported))
                    })
            };
        let Some(selected) = common else {
            return Err(AkitaError::InvalidInput(
                "commitment source group has neither one common external path nor one common standard representation for the selected inner operation".into(),
            ));
        };
        PendingCommitGroupPath::Standard(selected)
    };

    let descriptors = candidates
        .into_iter()
        .map(|(descriptor, _, _)| descriptor)
        .collect();

    Ok(CompiledCommitmentRequest {
        plan: *plan,
        sources,
        descriptors,
        path,
    })
}

pub(super) fn checked_logical_len(num_vars: usize) -> Result<usize, AkitaError> {
    let shift = u32::try_from(num_vars).map_err(|_| {
        AkitaError::InvalidInput(format!("commit source arity {num_vars} exceeds u32"))
    })?;
    1usize.checked_shl(shift).ok_or_else(|| {
        AkitaError::InvalidInput(format!("commit source arity 2^{num_vars} overflows usize"))
    })
}

pub(super) fn validate_plan_extents(
    descriptor: &CommitSourceDescriptor,
    plan: &CommitInnerPlan,
) -> Result<(), AkitaError> {
    if plan.ring_dimension == 0
        || !plan.ring_dimension.is_power_of_two()
        || plan.num_positions_per_block == 0
        || !plan.num_positions_per_block.is_power_of_two()
        || plan.num_digits_inner == 0
        || plan.n_a == 0
    {
        return Err(AkitaError::InvalidInput(
            "inner commitment plan has invalid ring, block, digit, or row geometry".into(),
        ));
    }
    let _logical_len = checked_logical_len(descriptor.num_vars)?;
    let ring_count = descriptor
        .live_coefficient_len
        .div_ceil(plan.ring_dimension);
    let physical_len = ring_count
        .checked_mul(plan.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidInput("commit source physical extent overflow".into()))?;
    let live_blocks = ring_count.div_ceil(plan.num_positions_per_block);
    if physical_len > descriptor.total_coefficient_len || live_blocks != plan.num_live_blocks {
        return Err(AkitaError::InvalidInput(format!(
            "commit source does not match plan extents: physical={physical_len}/{}, live_blocks={live_blocks}/{}",
            descriptor.total_coefficient_len, plan.num_live_blocks
        )));
    }
    Ok(())
}
