//! Planner guard: shipped adaptive fp128 one-hot schedules must stay within the
//! configured proof-optimized basis search window.

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_types::sis::{HonestFoldPolicy, HonestFoldSizingQuery};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

fn catalog<Cfg: CommitmentConfig>() -> akita_config::TrustedScheduleCatalog {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = std::fs::read(path).expect("checked-in schedule artifact");
    akita_config::trusted_schedule_catalog_from_bytes::<Cfg>(&bytes).expect("trusted catalog")
}

/// Sparse singleton keys covering small, production, stress, and table-max nv.
const BASIS_ENVELOPE_NUM_VARS: &[usize] = &[10, 16, 28, 30, 64, 120];

#[test]
fn large_fields_search_the_complete_i16_inner_basis_domain() {
    assert_eq!(fp128::Dense::inner_basis_range(), (3, 16));
    assert_eq!(
        akita_config::proof_optimized::fp64::Dense::inner_basis_range(),
        (3, 16)
    );
    assert_eq!(
        akita_config::proof_optimized::fp32::Dense::inner_basis_range(),
        (3, 10)
    );
}

#[test]
fn large_field_dense_presets_search_the_extended_a_dimension_domain() {
    for (mode, expected_a) in [
        (
            fp128::Dense::RING_DIMENSION_SCHEDULE_MODE,
            &[64, 128, 256, 512, 1024][..],
        ),
        (
            akita_config::proof_optimized::fp64::Dense::RING_DIMENSION_SCHEDULE_MODE,
            &[64, 128, 256, 512, 1024, 2048][..],
        ),
    ] {
        let akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
            ..
        } = mode
        else {
            panic!("large-field dense presets must use adaptive dimensions");
        };
        assert_eq!(potential_a_dimensions, expected_a);
        assert!(potential_b_dimensions.iter().all(|&d| d <= 256));
        assert!(potential_d_dimensions.iter().all(|&d| d <= 256));
    }
}

#[test]
fn adaptive_onehot_schedule_stays_within_basis_envelope() {
    type Cfg = fp128::OneHot;
    let inner_basis_max = Cfg::inner_basis_range().1;
    let opening_basis_max = Cfg::opening_basis_range().1;
    let mut covered = 0usize;
    let catalog = catalog::<Cfg>();

    for &nv in BASIS_ENVELOPE_NUM_VARS {
        let schedule = match catalog.resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::new(nv, 1),
        )) {
            Ok(row) => row.schedule().clone(),
            Err(_) => continue,
        };
        covered += 1;
        let root = &schedule.root.params;
        assert_eq!(
            root.inner().digits.log_basis,
            Cfg::inner_basis_range().0,
            "one-hot root must keep its canonical single-digit basis at nv={nv}"
        );
        assert_eq!(
            root.inner().digits.num_digits,
            1,
            "one-hot root must remain a single digit at nv={nv}"
        );
        let honest_policy = akita_config::honest_fold_policy_of::<Cfg>();
        let num_fold_coeffs = root
            .blocks()
            .positions_per_block
            .checked_mul(root.inner().digits.num_digits)
            .and_then(|width| width.checked_mul(root.d_a()))
            .and_then(|width| width.checked_mul(root.witness_chunk.num_chunks))
            .expect("one-hot fold width");
        let expected_fold_digits = honest_policy
            .num_digits_fold(HonestFoldSizingQuery {
                ring_dimension: root.d_a(),
                challenge_dimension: match root.opening_method() {
                    akita_types::OpeningMethod::EvaluationTrace => root.d_a(),
                    akita_types::OpeningMethod::SubringCoefficientPacking {
                        challenge_subring_dimension,
                    } => challenge_subring_dimension,
                },
                num_claims: 1,

                num_live_ring_elements_per_claim: root.blocks().live_ring_elements_per_claim,
                num_positions_per_block: root.blocks().positions_per_block,
                num_live_blocks: root.blocks().live_blocks,

                num_chunks: root.witness_chunk.num_chunks,
                num_fold_coeffs,
                witness_norms: honest_policy
                    .witness_norms_for_inner_basis(root.inner().digits.log_basis, root.d_a())
                    .expect("one-hot source geometry"),
                log_basis_response: root.open().digits.log_basis,
                challenge_config: &root.fold_challenge_config(),
            })
            .expect("one-hot fold policy");
        assert_eq!(
            root.num_digits_fold(),
            expected_fold_digits,
            "one-hot root must retain its tight honest-fold estimate at nv={nv}"
        );
        let mut source_basis = root.open().digits.log_basis;
        for fold in &schedule.recursive_folds {
            assert_eq!(
                fold.params.inner().digits.log_basis,
                source_basis,
                "recursive fold redecomposes its balanced-digit input at nv={nv}"
            );
            source_basis = fold.params.open().digits.log_basis;
        }
        assert_eq!(
            schedule.terminal.inner.digits.log_basis, source_basis,
            "terminal fold redecomposes its balanced-digit input at nv={nv}"
        );
        let within_window = root.inner().digits.log_basis <= inner_basis_max
            && root.outer().digits.log_basis <= opening_basis_max
            && root.open().digits.log_basis <= opening_basis_max
            && schedule.recursive_folds.iter().all(|fold| {
                fold.params.inner().digits.log_basis <= opening_basis_max
                    && fold.params.outer().digits.log_basis <= opening_basis_max
                    && fold.params.open().digits.log_basis <= opening_basis_max
            })
            && schedule.terminal.inner.digits.log_basis <= opening_basis_max;
        assert!(
            within_window,
            "adaptive onehot schedule exceeded its configured basis range at nv={nv}: {schedule:?}"
        );
    }
    assert!(covered > 0, "basis-envelope test resolved no catalog rows");
}
