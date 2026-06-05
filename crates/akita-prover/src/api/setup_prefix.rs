//! Preprocessing helpers for setup-prefix commitment artifacts (slice 02B).

use crate::api::commitment::{
    commit_inner_block_digit_count, commit_inner_flat_digit_count,
    validate_commit_outer_input_nonempty,
};
use crate::compute::{CommitmentComputeBackend, DenseCommitInput, DenseCommitRowsPlan};
use crate::kernels::linear::decompose_rows_i8_into;
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::AkitaProverSetup;
use akita_algebra::CyclotomicRing;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_types::{
    active_setup_field_len, digest_level_params, select_setup_prefix_slot, setup_prefix_slot_id,
    setup_seed_digest, AkitaCommitmentHint, AkitaExpandedSetup, ClaimIncidenceSummary,
    FlatDigitBlocks, LevelParams, MissingSetupPrefixSlotPolicy, RingCommitment,
    SetupPrefixDirectReason, SetupPrefixSelectionOutcome, SetupPrefixSelectionRequest,
    SetupPrefixSlot, SETUP_OFFLOAD_D_SETUP,
};

/// Commit one padded flat prefix of the shared setup matrix.
///
/// The witness is the coefficient form of `S^flat[0..n_prefix]`, zero-padded to a
/// power of two. The caller must supply `level_params` whose inner witness shape
/// satisfies `num_blocks * block_len == n_prefix / D`.
///
/// # Errors
///
/// Returns an error if shapes overflow, the prefix does not fit the setup matrix,
/// or backend commitment fails.
pub fn commit_setup_prefix<F, const D: usize, B>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    level_params: &LevelParams,
    setup_seed_digest: [u8; 32],
    n_prefix: usize,
    natural_len: usize,
) -> Result<SetupPrefixSlot<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    if natural_len == 0 || natural_len > n_prefix {
        return Err(AkitaError::InvalidSetup(
            "setup prefix natural length must be in 1..=n_prefix".to_string(),
        ));
    }
    if !n_prefix.is_multiple_of(D) || !n_prefix.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a power-of-two multiple of D".to_string(),
        ));
    }
    let padded_ring_slots = n_prefix / D;
    let witness_ring_slots = level_params
        .num_blocks
        .checked_mul(level_params.block_len)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup prefix witness shape overflow".to_string())
        })?;
    if witness_ring_slots != padded_ring_slots {
        return Err(AkitaError::InvalidSetup(format!(
            "level params witness shape {witness_ring_slots} ring slots does not match padded prefix {padded_ring_slots}"
        )));
    }

    let available_field_len = expanded
        .shared_matrix()
        .total_ring_elements()
        .checked_mul(D)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup matrix field length overflow".to_string())
        })?;
    if n_prefix > available_field_len {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length exceeds shared matrix capacity".to_string(),
        ));
    }

    let ring_elems = extract_setup_prefix_ring_elems::<F, D>(expanded, padded_ring_slots)?;
    let block_slices =
        setup_prefix_block_slices(&ring_elems, level_params.num_blocks, level_params.block_len)?;

    let recomposed_inner_rows = backend.dense_commit_rows(
        prepared,
        DenseCommitRowsPlan {
            n_a: level_params.a_key.row_len(),
            input: DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_commit: level_params.num_digits_commit,
                log_basis: level_params.log_basis,
            },
        },
    )?;

    let block_sizes = recomposed_inner_rows
        .iter()
        .map(|_| {
            commit_inner_block_digit_count(
                level_params.a_key.row_len(),
                level_params.num_digits_open,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut decomposed_inner_rows = FlatDigitBlocks::zeroed(block_sizes)?;
    let dst_blocks = decomposed_inner_rows.split_blocks_mut();
    #[cfg(feature = "parallel")]
    cfg_into_iter!(dst_blocks)
        .zip(cfg_iter!(recomposed_inner_rows))
        .try_for_each(|(dst, rows)| -> Result<(), AkitaError> {
            decompose_rows_i8_into(
                rows,
                dst,
                level_params.num_digits_open,
                level_params.log_basis,
            );
            Ok(())
        })?;
    #[cfg(not(feature = "parallel"))]
    dst_blocks
        .into_iter()
        .zip(recomposed_inner_rows.iter())
        .try_for_each(|(dst, rows)| -> Result<(), AkitaError> {
            decompose_rows_i8_into(
                rows,
                dst,
                level_params.num_digits_open,
                level_params.log_basis,
            );
            Ok(())
        })?;

    let b_input_len = commit_inner_flat_digit_count(
        level_params.num_blocks,
        level_params.a_key.row_len(),
        level_params.num_digits_open,
    )?;
    validate_commit_outer_input_nonempty(b_input_len)?;
    let mut b_input_digits = vec![[0i8; D]; b_input_len];
    b_input_digits.copy_from_slice(decomposed_inner_rows.flat_digits());
    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(level_params.b_key.row_len(), level_params.log_basis)?;
    #[cfg(feature = "zk")]
    let mut u = backend.digit_rows::<D>(
        prepared,
        level_params.b_key.row_len(),
        &b_input_digits,
        level_params.log_basis,
    )?;
    #[cfg(not(feature = "zk"))]
    let u = backend.digit_rows::<D>(
        prepared,
        level_params.b_key.row_len(),
        &b_input_digits,
        level_params.log_basis,
    )?;
    #[cfg(feature = "zk")]
    {
        let blinding_rows = backend.zk_b_digit_rows::<D>(
            prepared,
            level_params.b_key.row_len(),
            b_blinding_digits.flat_digits().len(),
            b_blinding_digits.flat_digits(),
        )?;
        for (row, blinding) in u.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
    }
    if u.len() != level_params.b_key.row_len() {
        return Err(AkitaError::InvalidSetup(format!(
            "setup prefix commit returned {} B rows, expected {}",
            u.len(),
            level_params.b_key.row_len()
        )));
    }

    let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
        decomposed_inner_rows,
        recomposed_inner_rows,
        #[cfg(feature = "zk")]
        b_blinding_digits,
    );
    let id = setup_prefix_slot_id(
        setup_seed_digest,
        D,
        n_prefix,
        digest_level_params(std::slice::from_ref(level_params)),
    );
    Ok(SetupPrefixSlot {
        id,
        natural_len,
        padded_len: n_prefix,
        commitment: RingCommitment { u },
        hint,
    })
}

/// Select a prover-ready setup-prefix slot for one active shape.
///
/// `MissingSetupPrefixSlotPolicy` applies only when the eligible prefix length
/// is known but the corresponding slot is absent. Below-threshold and
/// `D_setup` mismatch outcomes always return [`SetupPrefixSelectionOutcome::DirectScan`].
pub fn select_prover_setup_prefix_slot<F, const D: usize, B>(
    setup: &mut AkitaProverSetup<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    level_params: &LevelParams,
    incidence: &ClaimIncidenceSummary,
    n_min: usize,
    missing_slot_policy: MissingSetupPrefixSlotPolicy,
) -> Result<SetupPrefixSelectionOutcome<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    let seed_digest = setup_seed_digest(setup.expanded.seed())
        .map_err(|err| AkitaError::InvalidSetup(format!("setup seed digest failed: {err}")))?;
    let natural_len = active_setup_field_len(level_params, incidence, SETUP_OFFLOAD_D_SETUP)?;
    let request = SetupPrefixSelectionRequest {
        d_setup: SETUP_OFFLOAD_D_SETUP,
        natural_field_len: natural_len,
        level_params_digest: digest_level_params(std::slice::from_ref(level_params)),
    };
    let outcome = select_setup_prefix_slot(&setup.prefix_slots, seed_digest, D, request, n_min);
    match (&outcome, missing_slot_policy) {
        (
            SetupPrefixSelectionOutcome::DirectScan {
                reason: SetupPrefixDirectReason::MissingSlot(id),
            },
            MissingSetupPrefixSlotPolicy::StrictError,
        ) => Err(AkitaError::InvalidSetup(format!(
            "setup prefix slot missing for preprocessing policy: {id:?}"
        ))),
        (
            SetupPrefixSelectionOutcome::DirectScan {
                reason: SetupPrefixDirectReason::MissingSlot(_),
            },
            MissingSetupPrefixSlotPolicy::GenerateAndPersist,
        ) => {
            let n_prefix = akita_types::select_prefix_len(natural_len, n_min).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "cannot materialize setup prefix slot below delegation threshold".to_string(),
                )
            })?;
            let slot = commit_setup_prefix::<F, D, B>(
                &setup.expanded,
                backend,
                prepared,
                level_params,
                seed_digest,
                n_prefix,
                natural_len,
            )?;
            setup.prefix_slots.insert(slot.clone())?;
            Ok(SetupPrefixSelectionOutcome::Selected(slot))
        }
        _ => Ok(outcome),
    }
}

fn extract_setup_prefix_ring_elems<F, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    padded_ring_slots: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    let view = expanded
        .shared_matrix()
        .ring_view::<D>(padded_ring_slots, 1)?;
    (0..padded_ring_slots)
        .map(|row| {
            let cols = view.row(row)?;
            cols.first()
                .copied()
                .ok_or_else(|| AkitaError::InvalidSetup("empty setup prefix row".to_string()))
        })
        .collect()
}

fn setup_prefix_block_slices<F, const D: usize>(
    ring_elems: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
) -> Result<Vec<&[CyclotomicRing<F, D>]>, AkitaError>
where
    F: FieldCore,
{
    if num_blocks
        .checked_mul(block_len)
        .is_none_or(|witness| witness != ring_elems.len())
    {
        return Err(AkitaError::InvalidSetup(
            "setup prefix ring elements do not match witness block layout".to_string(),
        ));
    }
    Ok((0..num_blocks)
        .map(|block_idx| {
            let start = block_idx
                .checked_mul(block_len)
                .expect("block index fits after witness length check");
            &ring_elems[start..start + block_len]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{ComputeBackendSetup, CpuBackend};
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128Offset275 as F;
    use akita_types::{
        padded_setup_prefix_len, select_setup_prefix_slot, SetupMatrixEnvelope,
        SetupPrefixSelectionRequest, SisModulusFamily,
    };

    fn prefix_level_params() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
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
        .with_decomp(2, 3, 2, 2, 3)
        .expect("level params")
    }

    fn setup_capacity_for(level_params: &LevelParams, n_prefix: usize) -> usize {
        n_prefix.max(
            level_params
                .b_key
                .row_len()
                .checked_mul(
                    level_params
                        .num_blocks
                        .checked_mul(level_params.a_key.row_len())
                        .and_then(|n| n.checked_mul(level_params.num_digits_open))
                        .expect("b input shape"),
                )
                .expect("setup capacity"),
        )
    }

    fn test_setup(level_params: &LevelParams, n_prefix: usize) -> AkitaProverSetup<F, 32> {
        AkitaProverSetup::<F, 32>::generate_with_capacity(
            8,
            1,
            1,
            SetupMatrixEnvelope {
                max_setup_len: setup_capacity_for(level_params, n_prefix).max(1),
                #[cfg(feature = "zk")]
                max_zk_b_len: level_params
                    .b_key
                    .row_len()
                    .checked_mul(akita_types::zk::blinding_digit_plane_count::<F>(
                        level_params.b_key.row_len(),
                        32,
                        level_params.log_basis,
                    ))
                    .expect("ZK B setup capacity"),
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            },
        )
        .expect("setup")
    }

    #[test]
    fn commit_setup_prefix_populates_singleton_slot() {
        let level_params = prefix_level_params();
        let incidence = ClaimIncidenceSummary::same_point(4, 1).expect("incidence");
        let witness_ring_slots = level_params
            .num_blocks
            .checked_mul(level_params.block_len)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(32).expect("prefix length");
        let natural_len = active_setup_field_len(&level_params, &incidence, 32)
            .expect("natural len")
            .min(n_prefix);
        let mut setup = test_setup(&level_params, n_prefix);
        let backend = CpuBackend;
        let prepared = backend.prepare_setup::<32>(&setup).expect("prepared setup");
        let seed_digest = setup_seed_digest(setup.expanded.seed()).expect("digest");
        let slot = commit_setup_prefix::<F, 32, _>(
            &setup.expanded,
            &backend,
            &prepared,
            &level_params,
            seed_digest,
            n_prefix,
            natural_len,
        )
        .expect("commit prefix");
        assert_eq!(slot.natural_len, natural_len);
        assert_eq!(slot.padded_len, n_prefix);
        setup.prefix_slots.insert(slot).expect("insert");
        let natural_for_selection = n_prefix - 1;
        assert_eq!(padded_setup_prefix_len(natural_for_selection), n_prefix);
        let outcome = select_setup_prefix_slot(
            &setup.prefix_slots,
            seed_digest,
            32,
            SetupPrefixSelectionRequest {
                d_setup: 32,
                natural_field_len: natural_for_selection,
                level_params_digest: digest_level_params(std::slice::from_ref(&level_params)),
            },
            1,
        );
        assert!(matches!(outcome, SetupPrefixSelectionOutcome::Selected(_)));
    }

    fn aligned_prefix_fixture() -> (LevelParams, ClaimIncidenceSummary, usize) {
        let level_params = LevelParams::params_only(
            SisModulusFamily::Q128,
            32,
            3,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(1, 0, 1, 1, 1)
        .expect("aligned level params");
        let incidence = ClaimIncidenceSummary::same_point(2, 1).expect("incidence");
        let n_prefix = level_params
            .num_blocks
            .checked_mul(level_params.block_len)
            .expect("witness")
            .checked_mul(32)
            .expect("prefix");
        let natural_len = active_setup_field_len(&level_params, &incidence, 32).expect("natural");
        assert_eq!(padded_setup_prefix_len(natural_len), n_prefix);
        (level_params, incidence, n_prefix)
    }

    #[test]
    fn select_prover_setup_prefix_slot_honors_missing_slot_policy() {
        let (level_params, incidence, n_prefix) = aligned_prefix_fixture();
        let mut setup = test_setup(&level_params, n_prefix);
        let backend = CpuBackend;
        let prepared = backend.prepare_setup::<32>(&setup).expect("prepared");

        let fallback = select_prover_setup_prefix_slot(
            &mut setup,
            &backend,
            &prepared,
            &level_params,
            &incidence,
            1,
            MissingSetupPrefixSlotPolicy::DirectFallback,
        )
        .expect("fallback");
        assert!(matches!(
            fallback,
            SetupPrefixSelectionOutcome::DirectScan {
                reason: SetupPrefixDirectReason::MissingSlot(_)
            }
        ));

        assert!(select_prover_setup_prefix_slot(
            &mut setup,
            &backend,
            &prepared,
            &level_params,
            &incidence,
            1,
            MissingSetupPrefixSlotPolicy::StrictError,
        )
        .expect_err("strict")
        .to_string()
        .contains("setup prefix slot missing"));

        let generated = select_prover_setup_prefix_slot(
            &mut setup,
            &backend,
            &prepared,
            &level_params,
            &incidence,
            1,
            MissingSetupPrefixSlotPolicy::GenerateAndPersist,
        )
        .expect("generate");
        assert!(matches!(
            generated,
            SetupPrefixSelectionOutcome::Selected(_)
        ));
        assert_eq!(setup.prefix_slots.len(), 1);
    }
}
