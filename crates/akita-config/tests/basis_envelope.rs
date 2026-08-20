//! Planner guard: shipped adaptive fp128 one-hot schedules must stay within the
//! configured proof-optimized basis search window.

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_types::sis::{HonestFoldPolicy, HonestFoldSizingQuery};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

/// Sparse singleton keys covering small, production, stress, and table-max nv.
const BASIS_ENVELOPE_NUM_VARS: &[usize] = &[10, 16, 28, 30, 64, 120];

#[test]
fn adaptive_onehot_schedule_stays_within_basis_envelope() {
    type Cfg = fp128::OneHot;
    let inner_basis_max = Cfg::inner_basis_range().1;
    let opening_basis_max = Cfg::opening_basis_range().1;
    let mut covered = 0usize;

    for &nv in BASIS_ENVELOPE_NUM_VARS {
        let schedule = match Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::new(nv, 1),
        )) {
            Ok(row) => row.into_schedule(),
            Err(_) => continue,
        };
        covered += 1;
        let root = &schedule.root.params;
        assert_eq!(
            root.log_basis_inner,
            Cfg::inner_basis_range().0,
            "one-hot root must keep its canonical single-digit basis at nv={nv}"
        );
        assert_eq!(
            root.num_digits_inner, 1,
            "one-hot root must remain a single digit at nv={nv}"
        );
        let honest_policy = akita_config::honest_fold_policy_of::<Cfg>();
        let num_fold_coeffs = root
            .num_positions_per_block
            .checked_mul(root.num_digits_inner)
            .and_then(|width| width.checked_mul(root.d_a()))
            .and_then(|width| width.checked_mul(root.witness_chunk.num_chunks))
            .expect("one-hot fold width");
        let expected_fold_digits = honest_policy
            .num_digits_fold(HonestFoldSizingQuery {
                ring_dimension: root.d_a(),
                challenge_dimension: match root.opening_method {
                    akita_types::OpeningMethod::EvaluationTrace => root.d_a(),
                    akita_types::OpeningMethod::SubringCoefficientPacking {
                        challenge_subring_dimension,
                    } => challenge_subring_dimension,
                },
                num_claims: 1,
                num_live_ring_elements_per_claim: root.num_live_ring_elements_per_claim,
                num_live_blocks: root.num_live_blocks,
                num_positions_per_block: root.num_positions_per_block,
                num_chunks: root.witness_chunk.num_chunks,
                num_fold_coeffs,
                witness_norms: honest_policy
                    .witness_norms_for_inner_basis(root.log_basis_inner, root.d_a())
                    .expect("one-hot source geometry"),
                log_basis_response: root.log_basis_open,
                challenge_config: &root.fold_challenge_config,
            })
            .expect("one-hot fold policy");
        assert_eq!(
            root.num_digits_fold, expected_fold_digits,
            "one-hot root must retain its tight honest-fold estimate at nv={nv}"
        );
        let mut source_basis = root.log_basis_open;
        for fold in &schedule.recursive_folds {
            assert_eq!(
                fold.params.log_basis_inner, source_basis,
                "recursive fold redecomposes its balanced-digit input at nv={nv}"
            );
            source_basis = fold.params.log_basis_open;
        }
        assert_eq!(
            schedule.terminal.params.witness.log_basis_inner, source_basis,
            "terminal fold redecomposes its balanced-digit input at nv={nv}"
        );
        let within_window = root.log_basis_inner <= inner_basis_max
            && root.log_basis_outer <= opening_basis_max
            && root.log_basis_open <= opening_basis_max
            && schedule.recursive_folds.iter().all(|fold| {
                fold.params.log_basis_inner <= opening_basis_max
                    && fold.params.log_basis_outer <= opening_basis_max
                    && fold.params.log_basis_open <= opening_basis_max
            })
            && schedule.terminal.params.witness.log_basis_inner <= opening_basis_max;
        assert!(
            within_window,
            "adaptive onehot schedule exceeded its configured basis range at nv={nv}: {schedule:?}"
        );
    }
    assert!(covered > 0, "basis-envelope test resolved no catalog rows");
}
