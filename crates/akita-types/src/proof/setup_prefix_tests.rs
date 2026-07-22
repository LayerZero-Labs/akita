use super::*;
use crate::{CommittedGroupParams, OpeningClaimsLayout, SisModulusProfileId};
use akita_challenges::SparseChallengeConfig;

fn sample_level_params() -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        3,
        3,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(4, 3, 2, 2, 2)
    .expect("sample level params")
}

fn prefix_eligible_level_params() -> CommittedGroupParams {
    let full_field_digits = crate::sis::compute_num_digits_full_field(128, 3);
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        2,
        2,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 3, full_field_digits, 2, 2)
    .expect("prefix eligible level params")
}

#[test]
fn active_setup_field_len_matches_packed_role_maximum() {
    let lp = sample_level_params();
    let opening_batch = OpeningClaimsLayout::new(5, 3).expect("opening batch");
    let w_a = lp.num_positions_per_block * lp.num_digits_inner;
    let w_b = opening_batch.num_total_polynomials()
        * lp.inner_commit_matrix.output_rank()
        * lp.num_live_blocks
        * lp.num_digits_open;
    let w_d = opening_batch.num_total_polynomials() * lp.num_live_blocks * lp.num_digits_open;
    let expected_ring_slots = lp
        .inner_commit_matrix
        .output_rank()
        .checked_mul(w_a)
        .unwrap()
        .max(
            lp.outer_commit_matrix
                .output_rank()
                .checked_mul(w_b)
                .unwrap(),
        )
        .max(
            lp.open_commit_matrix
                .output_rank()
                .checked_mul(w_d)
                .unwrap(),
        );
    let geometry =
        active_setup_projection_geometry(&lp, &opening_batch).expect("projection geometry");
    assert_eq!(geometry.required(), expected_ring_slots);
    let dims = lp.role_dims();
    let base_d = dims.d_a().min(dims.d_b()).min(dims.d_d());
    assert_eq!(
        active_setup_field_len(&lp, &opening_batch).expect("field len"),
        expected_ring_slots * base_d
    );
}

#[test]
fn select_setup_prefix_slot_uses_exact_registry_match() {
    use akita_field::Prime32Offset99 as F;

    let level_params = prefix_eligible_level_params();
    let d_setup = SETUP_OFFLOAD_D_SETUP;
    let natural_len = 129usize;
    let n_prefix = padded_setup_prefix_len(natural_len);
    let commitment_params =
        setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
    let id = setup_prefix_slot_id(d_setup, natural_len, commitment_params);
    let mut level_params = level_params;
    level_params.setup_prefix = Some(id.clone());
    let slot = SetupPrefixVerifierSlot {
        id: id.clone(),
        natural_len,
        padded_len: n_prefix,
        commitment: SetupPrefixPublicCommitment {
            rows: vec![RingVec::from_coeffs(vec![F::zero()])],
        },
    };
    let mut registry = SetupPrefixVerifierRegistry::<F>::new();
    registry.insert(slot).expect("insert slot");

    let selection = select_setup_prefix_slot(
        3,
        |candidate| {
            registry
                .get(candidate)
                .map(|slot| (slot, slot.natural_len, slot.padded_len))
        },
        &level_params,
        natural_len,
        d_setup,
        "slot does not cover request",
    )
    .expect("selection succeeds")
    .expect("slot selected");
    assert_eq!(&selection.0.id, &id);
    assert_eq!(selection.1, 4);

    let err = select_setup_prefix_slot(
        3,
        |candidate| {
            registry
                .get(candidate)
                .map(|slot| (slot, slot.natural_len, slot.padded_len))
        },
        &level_params,
        natural_len + 1,
        d_setup,
        "slot does not cover request",
    )
    .expect_err("different natural_len must fail");
    assert!(err.to_string().contains("slot does not cover request"));

    let err = select_setup_prefix_slot(
        3,
        |candidate| {
            registry
                .get(candidate)
                .map(|slot| (slot, slot.natural_len, slot.padded_len))
        },
        &level_params,
        193,
        d_setup,
        "slot does not cover request",
    )
    .expect_err("natural prefix beyond shared setup must fail");
    assert!(err
        .to_string()
        .contains("setup prefix request exceeds shared matrix capacity"));
}

#[test]
fn select_setup_prefix_slot_rejects_missing_registry_entry() {
    use akita_field::Prime32Offset99 as F;

    let mut level_params = prefix_eligible_level_params();
    let d_setup = SETUP_OFFLOAD_D_SETUP;
    let natural_len = 65usize;
    let n_prefix = padded_setup_prefix_len(natural_len);
    level_params.setup_prefix = Some(setup_prefix_slot_id(
        d_setup,
        natural_len,
        setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params"),
    ));

    let err = select_setup_prefix_slot::<SetupPrefixVerifierSlot<F>, _>(
        2,
        |_: &SetupPrefixSlotId| None,
        &level_params,
        natural_len,
        d_setup,
        "slot does not cover request",
    )
    .expect_err("missing registry entry must fail");
    assert!(err
        .to_string()
        .contains("required setup prefix slot is missing from registry"));
    let _ = F::zero();
}

#[test]
fn prover_registry_duplicate_insert_does_not_replace_existing_slot() {
    use crate::proof::DigitBlocks;
    use akita_field::Prime32Offset99 as F;

    let commitment_params =
        setup_prefix_precommitted_params(&sample_level_params(), 64).expect("prefix params");
    let id = setup_prefix_slot_id(64, 1, commitment_params);
    let slot = || {
        // D-free hint: one empty digit block at stride 32 (the former D).
        let decomposed = DigitBlocks::from_blocks(vec![Vec::new()], 64).expect("digit blocks");
        let hint = AkitaCommitmentHint::<F>::singleton(decomposed);
        SetupPrefixSlot {
            id: id.clone(),
            natural_len: id.natural_len,
            padded_len: id.n_prefix().expect("padded len"),
            // One commitment row of d_setup = 32 coefficients.
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero(); 64])],
            },
            hint,
        }
    };

    let mut registry = SetupPrefixProverRegistry::<F>::new();
    registry.insert(slot()).expect("first insert");
    registry
        .insert(slot())
        .expect_err("duplicate insert must fail");

    assert_eq!(registry.len(), 1);
}

#[test]
fn verifier_registry_duplicate_insert_does_not_replace_existing_slot() {
    use akita_field::Prime32Offset99 as F;

    let commitment_params =
        setup_prefix_precommitted_params(&sample_level_params(), 64).expect("prefix params");
    let id = setup_prefix_slot_id(64, 1, commitment_params);
    let slot = || SetupPrefixVerifierSlot {
        id: id.clone(),
        natural_len: id.natural_len,
        padded_len: id.n_prefix().expect("padded len"),
        commitment: SetupPrefixPublicCommitment {
            rows: vec![RingVec::from_coeffs(vec![F::zero()])],
        },
    };

    let mut registry = SetupPrefixVerifierRegistry::<F>::new();
    registry.insert(slot()).expect("first insert");
    registry
        .insert(slot())
        .expect_err("duplicate insert must fail");

    assert_eq!(registry.len(), 1);
}
