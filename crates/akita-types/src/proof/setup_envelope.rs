//! Canonical setup-matrix field-capacity accounting.

use akita_error::AkitaError;

use super::setup_prefix::{active_setup_field_len, suffix_opening_layout};
use crate::{
    CommitmentSliceCount, CommittedGroupParams, CompressionChainPlan, FoldSchedule,
    InnerCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, SetupMatrixCapacity,
    SetupPrefixSlotId, SisModulusProfileId, TerminalFoldParams,
};

/// Compute the exact maximum reusable setup-matrix field prefix required by
/// `schedule`.
pub fn setup_matrix_capacity_for_schedule(
    schedule: &FoldSchedule,
) -> Result<SetupMatrixCapacity, AkitaError> {
    let num_field_elements = setup_matrix_field_elements_for_schedule(schedule)?;
    Ok(SetupMatrixCapacity { num_field_elements })
}

/// Compute the largest physical base-field footprint of any setup matrix or
/// natural public-matrix prefix used by `schedule`.
///
/// This quantity is independent of a level-local ring dimension and is
/// therefore comparable across mixed-ring levels.
pub fn setup_matrix_field_elements_for_schedule(
    schedule: &FoldSchedule,
) -> Result<usize, AkitaError> {
    let mut max_field_elements = 1;
    accumulate_matrix_field_elements_for_level(&schedule.root.params, &mut max_field_elements)?;
    for fold in &schedule.recursive_folds {
        accumulate_matrix_field_elements_for_level(&fold.params, &mut max_field_elements)?;
    }
    accumulate_terminal_matrix_field_elements(&schedule.terminal, &mut max_field_elements)?;
    Ok(max_field_elements)
}

/// Compute the exact public-matrix prefix required by a verifier for one
/// resolved schedule.
///
/// A producer whose successor carries an incoming setup-prefix commitment does
/// not require a direct public-matrix scan. The first producer after the
/// offloaded chain does. Terminal commitment verification always requires its
/// exact inner matrix. Every compressed level still requires its F/H map
/// prefixes, including an offloaded producer. The returned capacity is the
/// maximum of those uses.
pub fn verifier_setup_matrix_capacity_for_schedule(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
) -> Result<SetupMatrixCapacity, AkitaError> {
    schedule.validate_structure()?;

    let mut num_field_elements = 1usize;
    accumulate_terminal_matrix_field_elements(&schedule.terminal, &mut num_field_elements)?;
    accumulate_compression_matrix_field_elements_for_level(
        &schedule.root.params,
        &mut num_field_elements,
    )?;
    for fold in &schedule.recursive_folds {
        accumulate_compression_matrix_field_elements_for_level(
            &fold.params,
            &mut num_field_elements,
        )?;
    }

    for producer_index in 0..=schedule.recursive_folds.len() {
        let producer_is_offloaded = schedule
            .recursive_folds
            .get(producer_index)
            .is_some_and(|successor| successor.params.setup_prefix().is_some());
        if producer_is_offloaded {
            continue;
        }

        let direct_fields = if producer_index == 0 {
            active_setup_field_len(&schedule.root.params, root_layout)?
        } else {
            let producer = &schedule.recursive_folds[producer_index - 1];
            let incoming_prefix_len = producer
                .params
                .setup_prefix()
                .as_ref()
                .and_then(|slot| slot.setup_natural_len);
            let layout = suffix_opening_layout(producer.input_witness_len, incoming_prefix_len)?;
            active_setup_field_len(&producer.params, &layout)?
        };
        num_field_elements = num_field_elements.max(direct_fields);
    }

    Ok(SetupMatrixCapacity { num_field_elements })
}

/// Extend a physical setup footprint with one non-terminal level.
pub fn accumulate_matrix_field_elements_for_level(
    params: &CommittedGroupParams,
    max_field_elements: &mut usize,
) -> Result<(), AkitaError> {
    include_matrix_field_elements(
        max_field_elements,
        params.inner.matrix.output_rank(),
        params.inner_width(),
        params.inner.matrix.ring_dimension(),
        "inner setup",
    )?;
    include_matrix_field_elements(
        max_field_elements,
        params.outer.matrix.output_rank(),
        params.outer_width(),
        params.outer.matrix.ring_dimension(),
        "outer setup",
    )?;
    include_matrix_field_elements(
        max_field_elements,
        params.open.matrix.output_rank(),
        params.d_matrix_width(),
        params.open.matrix.ring_dimension(),
        "opening setup",
    )?;
    for group in params.precommitted_groups() {
        include_matrix_field_elements(
            max_field_elements,
            group.profile.inner.matrix.output_rank(),
            group.inner_width(),
            group.profile.inner.matrix.ring_dimension(),
            "precommitted inner setup",
        )?;
        include_matrix_field_elements(
            max_field_elements,
            group.profile.outer.matrix.output_rank(),
            group.outer_width(),
            group.profile.outer.matrix.ring_dimension(),
            "precommitted outer setup",
        )?;
    }
    accumulate_compression_matrix_field_elements_for_level(params, max_field_elements)?;
    if let Some(slot) = &params.setup_prefix() {
        *max_field_elements = (*max_field_elements).max(match slot.slot_id() {
            Some(slot_id) => setup_prefix_slot_field_elements(&slot_id)?,
            None => 0,
        });
    }
    Ok(())
}

/// Physical setup footprint of materializing one committed group's commitment.
///
/// Commitment arithmetic touches only the A matrix, the B matrix, and the
/// compression chain that reduces B's output. It never touches D, because a
/// group's opening happens under whichever row later consumes it.
///
/// Setup sizing and commit-time admission both price an independent
/// commitment from this one definition, so provisioning can never fall short
/// of what admission demands. `outer_slice_count` expands only the logical B
/// image compressed by F; the physical B matrix remains stored once.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when any footprint product overflows.
pub fn commit_only_setup_field_elements(
    inner_commit_matrix: &InnerCommitMatrixParams,
    outer_commit_matrix: &OuterCommitMatrixParams,
    outer_slice_count: CommitmentSliceCount,
) -> Result<usize, AkitaError> {
    let mut max_field_elements = 0;
    include_matrix_field_elements(
        &mut max_field_elements,
        inner_commit_matrix.output_rank(),
        inner_commit_matrix.input_width(),
        inner_commit_matrix.ring_dimension(),
        "commit inner setup",
    )?;
    include_matrix_field_elements(
        &mut max_field_elements,
        outer_commit_matrix.output_rank(),
        outer_commit_matrix.input_width(),
        outer_commit_matrix.ring_dimension(),
        "commit outer setup",
    )?;
    include_compression_setup(
        &mut max_field_elements,
        outer_commit_matrix.sis_modulus_profile(),
        outer_slice_count.logical_output_rows(outer_commit_matrix.output_rank())?,
        outer_commit_matrix.ring_dimension(),
        "commit outer compression setup",
    )?;
    Ok(max_field_elements)
}

/// Extend a physical setup footprint with every compression map used by one
/// non-terminal level.
fn accumulate_compression_matrix_field_elements_for_level(
    params: &CommittedGroupParams,
    max_field_elements: &mut usize,
) -> Result<(), AkitaError> {
    if !params.payload_mode.is_compressed() {
        return Ok(());
    }
    include_compression_setup(
        max_field_elements,
        params.outer.matrix.sis_modulus_profile(),
        params
            .outer_slice_count
            .logical_output_rows(params.outer.matrix.output_rank())?,
        params.role_dims().d_b(),
        "outer compression setup",
    )?;
    for group in params.precommitted_group_iter() {
        include_compression_setup(
            max_field_elements,
            group.profile.outer.matrix.sis_modulus_profile(),
            group
                .profile
                .outer_slice_count
                .logical_output_rows(group.profile.outer.matrix.output_rank())?,
            group.profile.outer.matrix.ring_dimension(),
            "precommitted outer compression setup",
        )?;
    }
    include_compression_setup(
        max_field_elements,
        params.open.matrix.sis_modulus_profile(),
        params.open.matrix.output_rank(),
        params.role_dims().d_d(),
        "opening compression setup",
    )
}

/// Extend a physical setup footprint with the terminal inner matrix.
pub fn accumulate_terminal_matrix_field_elements(
    params: &TerminalFoldParams,
    max_field_elements: &mut usize,
) -> Result<(), AkitaError> {
    include_matrix_field_elements(
        max_field_elements,
        params.inner.matrix.output_rank(),
        params.inner_width(),
        params.inner.matrix.ring_dimension(),
        "terminal inner setup",
    )
}

/// Largest physical base-field footprint of an actual setup source prefix or
/// either matrix used to commit its padded protocol object.
pub fn setup_prefix_slot_field_elements(slot: &SetupPrefixSlotId) -> Result<usize, AkitaError> {
    let n_prefix = slot.n_prefix()?;
    if slot.d_setup() == 0 || !n_prefix.is_multiple_of(slot.d_setup()) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot has invalid setup dimension".to_string(),
        ));
    }
    let mut max_field_elements = n_prefix;
    let params = &slot.commitment_profile;
    include_matrix_field_elements(
        &mut max_field_elements,
        params.inner.matrix.output_rank(),
        params.inner.matrix.input_width(),
        params.inner.matrix.ring_dimension(),
        "setup-prefix inner setup",
    )?;
    include_matrix_field_elements(
        &mut max_field_elements,
        params.outer.matrix.output_rank(),
        params.outer.matrix.input_width(),
        params.outer.matrix.ring_dimension(),
        "setup-prefix outer setup",
    )?;
    include_compression_setup(
        &mut max_field_elements,
        params.outer.matrix.sis_modulus_profile(),
        params
            .outer_slice_count
            .logical_output_rows(params.outer.matrix.output_rank())?,
        params.outer.matrix.ring_dimension(),
        "setup-prefix outer compression setup",
    )?;
    Ok(max_field_elements)
}

fn include_compression_setup(
    max_field_elements: &mut usize,
    profile: SisModulusProfileId,
    rows: usize,
    ring_dimension: usize,
    role: &'static str,
) -> Result<(), AkitaError> {
    let source_coefficients = rows
        .checked_mul(ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} source shape overflow")))?;
    let chain = CompressionChainPlan::for_complete_source(profile, source_coefficients)?;
    *max_field_elements = (*max_field_elements).max(chain.max_setup_field_elements()?);
    Ok(())
}

fn include_matrix_field_elements(
    max_field_elements: &mut usize,
    rows: usize,
    columns: usize,
    matrix_ring_dim: usize,
    role: &'static str,
) -> Result<(), AkitaError> {
    let field_elements = rows
        .checked_mul(columns)
        .and_then(|len| len.checked_mul(matrix_ring_dim))
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} envelope overflow")))?;
    *max_field_elements = (*max_field_elements).max(field_elements);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SisModulusProfileId;
    use akita_challenges::SparseChallengeConfig;

    #[test]
    fn commit_only_envelope_prices_complete_sliced_b_image() {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            4,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(3),
        );
        params.outer_slice_count = CommitmentSliceCount::FOUR;
        let params = params.with_decomp(1, 4, 1, 1, 1).expect("params");

        let expected_compression = CompressionChainPlan::for_complete_source(
            params.outer.matrix.sis_modulus_profile(),
            params
                .outer_slice_count
                .complete_source_coefficients(
                    params.outer.matrix.output_rank(),
                    params.outer.matrix.ring_dimension(),
                )
                .expect("complete sliced B source"),
        )
        .expect("sliced compression")
        .max_setup_field_elements()
        .expect("sliced compression setup");
        let sliced = commit_only_setup_field_elements(
            &params.inner.matrix,
            &params.outer.matrix,
            params.outer_slice_count,
        )
        .expect("sliced commit envelope");
        let unsliced = commit_only_setup_field_elements(
            &params.inner.matrix,
            &params.outer.matrix,
            CommitmentSliceCount::ONE,
        )
        .expect("unsliced commit envelope");

        assert_eq!(sliced, expected_compression);
        assert!(sliced > unsliced);
    }

    #[test]
    fn compression_envelope_covers_maps_that_dominate_direct_matrices() {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            4,
            2,
            2,
            2,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .expect("params");
        let outer_source = params.outer.matrix.output_rank() * params.role_dims().d_b();
        let opening_source = params.open.matrix.output_rank() * params.role_dims().d_d();
        let expected = [
            CompressionChainPlan::for_complete_source(
                params.outer.matrix.sis_modulus_profile(),
                outer_source,
            )
            .expect("outer compression")
            .max_setup_field_elements()
            .expect("outer setup"),
            CompressionChainPlan::for_complete_source(
                params.open.matrix.sis_modulus_profile(),
                opening_source,
            )
            .expect("opening compression")
            .max_setup_field_elements()
            .expect("opening setup"),
        ]
        .into_iter()
        .max()
        .expect("compression setup maximum");
        let mut direct = 1;
        include_matrix_field_elements(
            &mut direct,
            params.inner.matrix.output_rank(),
            params.inner_width(),
            params.inner.matrix.ring_dimension(),
            "inner setup",
        )
        .expect("inner setup");
        include_matrix_field_elements(
            &mut direct,
            params.outer.matrix.output_rank(),
            params.outer_width(),
            params.outer.matrix.ring_dimension(),
            "outer setup",
        )
        .expect("outer setup");
        include_matrix_field_elements(
            &mut direct,
            params.open.matrix.output_rank(),
            params.d_matrix_width(),
            params.open.matrix.ring_dimension(),
            "opening setup",
        )
        .expect("opening setup");
        assert!(expected > direct, "compression must dominate this fixture");

        let mut verifier_compression = 1;
        accumulate_compression_matrix_field_elements_for_level(&params, &mut verifier_compression)
            .expect("verifier compression envelope");
        assert_eq!(verifier_compression, expected);
        assert!(verifier_compression - 1 < expected);

        let mut actual = 1;
        accumulate_matrix_field_elements_for_level(&params, &mut actual).expect("level envelope");
        assert_eq!(actual, expected);
    }

    #[test]
    fn compression_envelope_includes_the_setup_prefix_group() {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            4,
            2,
            1,
            1,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .expect("params");
        let setup_num_digits = crate::sis::compute_num_digits_field_width(
            params.inner.matrix.sis_modulus_profile().field_bits(),
            params.inner.digits.log_basis,
        );
        params.inner.digits.num_digits = setup_num_digits;
        params.blocks.positions_per_block = 1;
        params.blocks.live_blocks = 1;
        let inner = params.inner.matrix;
        let inner_key = inner
            .sis_table_key()
            .expect("L infinity setup-prefix matrix");
        params.inner.matrix = crate::InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner_key.table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank(),
            params.blocks.positions_per_block * params.inner.digits.num_digits,
            inner_key.coeff_linf_bound,
            inner.ring_dimension(),
        );
        let outer_width = crate::CommitmentSliceGeometry::try_new(
            params.outer_slice_count,
            1,
            1,
            params.inner.matrix.output_rank(),
            params.outer.digits.num_digits,
            params.inner.matrix.ring_dimension(),
            params.outer.matrix.ring_dimension(),
        )
        .expect("setup-prefix slice geometry")
        .physical_input_width();
        let outer = params.outer.matrix;
        params.outer.matrix = crate::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer_width,
            outer.coeff_linf_bound(),
            outer.ring_dimension(),
        );
        let mut prefix_params =
            crate::setup_prefix_precommitted_params(&params, 64).expect("setup prefix params");
        let outer = prefix_params.profile.outer.matrix;
        prefix_params.profile.outer.matrix = crate::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            8,
            outer.input_width(),
            outer.coeff_linf_bound(),
            outer.ring_dimension(),
        );
        let prefix_expected = CompressionChainPlan::for_complete_source(
            outer.sis_modulus_profile(),
            8 * outer.ring_dimension(),
        )
        .expect("prefix compression")
        .max_setup_field_elements()
        .expect("prefix setup");
        params.set_setup_prefix(Some(crate::scheduled_setup_prefix(64, prefix_params)));

        let mut without_prefix = 1;
        let mut final_group_only = params.clone();
        final_group_only.set_setup_prefix(None);
        accumulate_compression_matrix_field_elements_for_level(
            &final_group_only,
            &mut without_prefix,
        )
        .expect("final group compression");
        assert!(prefix_expected > without_prefix);

        let mut exact = 1;
        accumulate_compression_matrix_field_elements_for_level(&params, &mut exact)
            .expect("setup prefix compression");
        assert_eq!(exact, prefix_expected);
        assert!(exact - 1 < prefix_expected);
    }
}
