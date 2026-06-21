//! Diagnostic observer: realized `Z_SQUARED` vs public `B_l2_pub` slack at each fold.
//!
//! Activated only while [`L2PubBoundObserverGuard`] is installed (profile / scouting).
//! No proving or verification behavior changes.

use std::cell::RefCell;

use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{
    detect_field_modulus,
    sis::{
        fold_witness_l2_pub_bound_sq, folded_witness_l2_bound_squared,
        select_l2_certificate_realization_for_level, L2CertificateRealization,
    },
    LevelParams,
};

use crate::DecomposeFoldWitness;

/// One fold-level `Z_SQUARED` vs `B_l2_pub` diagnostic snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L2PubBoundObservation {
    /// Sequential fold index on the current prove thread (0 = first fold).
    pub fold_index: usize,
    pub r_vars: usize,
    pub num_claims: usize,
    pub num_blocks: usize,
    pub num_digits_fold: usize,
    pub num_fold_coeffs: u128,
    pub op_norm_rejection: bool,
    pub centered_inf_norm: u32,
    /// `Σ centered_coeff²` on the accepted fold witness.
    pub z_squared: u128,
    /// Public bound from [`fold_witness_l2_pub_bound_sq`].
    pub b_l2_pub: u128,
    /// Deterministic fallback envelope [`folded_witness_l2_bound_squared`].
    pub l2_bound_squared: u128,
    pub realization: L2CertificateRealization,
    /// `B_l2_pub / Z_SQUARED` in basis points (`10000` = 1.00×). `0` when `Z_SQUARED = 0`.
    pub slack_bps: u64,
    /// `Z_SQUARED / B_l2_pub` in basis points. Values above `10000` mean the bound is too tight.
    pub z_over_b_l2_bps: u64,
    /// `Z_SQUARED / L2_BOUND_SQUARED` in basis points.
    pub z_over_det_bps: u64,
}

struct ObserverState {
    active: bool,
    records: Vec<L2PubBoundObservation>,
}

thread_local! {
    static L2_PUB_BOUND_OBSERVER: RefCell<ObserverState> = const {
        RefCell::new(ObserverState {
            active: false,
            records: Vec::new(),
        })
    };
}

/// RAII guard that activates L2 public-bound diagnostics on the current thread.
pub struct L2PubBoundObserverGuard;

impl L2PubBoundObserverGuard {
    /// Begin recording fold-level `B_l2_pub` slack observations.
    pub fn install() -> Self {
        L2_PUB_BOUND_OBSERVER.with(|cell| {
            let mut state = cell.borrow_mut();
            state.active = true;
            state.records.clear();
        });
        Self
    }

    /// Drain recorded observations and deactivate the observer.
    pub fn take() -> Vec<L2PubBoundObservation> {
        L2_PUB_BOUND_OBSERVER.with(|cell| {
            let mut state = cell.borrow_mut();
            state.active = false;
            std::mem::take(&mut state.records)
        })
    }
}

impl Drop for L2PubBoundObserverGuard {
    fn drop(&mut self) {
        L2_PUB_BOUND_OBSERVER.with(|cell| {
            cell.borrow_mut().active = false;
        });
    }
}

fn field_bits_for_level(lp: &LevelParams) -> u32 {
    let hint = lp.field_bits_hint;
    if hint == 0 {
        128
    } else {
        hint
    }
}

fn z_squared_from_centered_coeffs<const D: usize>(centered_coeffs: &[[i32; D]]) -> u128 {
    centered_coeffs
        .iter()
        .flat_map(|row| row.iter())
        .fold(0u128, |acc, coeff| {
            let sq = i128::from(*coeff).pow(2) as u128;
            acc.saturating_add(sq)
        })
}

fn ratio_bps(numerator: u128, denominator: u128) -> u64 {
    if denominator == 0 {
        return 0;
    }
    let scaled = numerator.saturating_mul(10_000);
    (scaled / denominator).min(u128::from(u64::MAX)) as u64
}

fn compute_observation<F: CanonicalField, const D: usize>(
    fold_index: usize,
    lp: &LevelParams,
    num_claims: usize,
    witness: &DecomposeFoldWitness<F, D>,
) -> Result<L2PubBoundObservation, AkitaError> {
    let field_bits = field_bits_for_level(lp);
    let num_digits_fold = lp.num_digits_fold(num_claims, field_bits)?;
    let witness_norms = lp.fold_witness_l2_pub_norms();
    let challenge_l2_sq = lp.challenge_l2_sq_max();
    let b_l2_pub = fold_witness_l2_pub_bound_sq(
        lp.r_vars,
        num_claims,
        lp.inner_width(),
        challenge_l2_sq,
        witness_norms,
    )?;
    let num_fold_coeffs = lp.num_fold_coeffs();
    let num_fold_coeffs_usize = if num_fold_coeffs > 0 {
        num_fold_coeffs as usize
    } else {
        witness.centered_coeffs.len().saturating_mul(D)
    };
    let l2_bound_squared =
        folded_witness_l2_bound_squared(num_fold_coeffs_usize, lp.log_basis, num_digits_fold)?;
    let field_q = detect_field_modulus::<F>();
    let realization = select_l2_certificate_realization_for_level(
        num_fold_coeffs_usize,
        lp.log_basis,
        num_digits_fold,
        lp.r_vars,
        num_claims,
        lp.inner_width(),
        challenge_l2_sq,
        witness_norms,
        field_q,
    )?;
    let z_squared = z_squared_from_centered_coeffs(&witness.centered_coeffs);
    Ok(L2PubBoundObservation {
        fold_index,
        r_vars: lp.r_vars,
        num_claims,
        num_blocks: lp.num_blocks,
        num_digits_fold,
        num_fold_coeffs: num_fold_coeffs.max(num_fold_coeffs_usize as u128),
        op_norm_rejection: lp.op_norm_rejection,
        centered_inf_norm: witness.centered_inf_norm,
        z_squared,
        b_l2_pub,
        l2_bound_squared,
        realization,
        slack_bps: ratio_bps(b_l2_pub, z_squared),
        z_over_b_l2_bps: ratio_bps(z_squared, b_l2_pub),
        z_over_det_bps: ratio_bps(z_squared, l2_bound_squared),
    })
}

pub(crate) fn record_l2_pub_bound_diag<F, const D: usize>(
    lp: &LevelParams,
    num_claims: usize,
    witness: &DecomposeFoldWitness<F, D>,
) where
    F: FieldCore + CanonicalField,
{
    L2_PUB_BOUND_OBSERVER.with(|cell| {
        let mut state = cell.borrow_mut();
        if !state.active {
            return;
        }
        let fold_index = state.records.len();
        match compute_observation::<F, D>(fold_index, lp, num_claims, witness) {
            Ok(obs) => state.records.push(obs),
            Err(err) => {
                eprintln!("l2_pub_bound_diag: fold_index={fold_index} skipped: {err}");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128Offset275;
    use akita_types::SisModulusFamily;

    type F = Prime128Offset275;

    fn sample_level() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(2, 3, 1, 1, 0)
        .unwrap()
    }

    #[test]
    fn install_take_roundtrip_records_observations() {
        let lp = sample_level();
        let witness = DecomposeFoldWitness {
            z_folded_rings: vec![],
            centered_coeffs: vec![[1i32; 64], [2i32; 64]],
            centered_inf_norm: 2,
        };
        let _guard = L2PubBoundObserverGuard::install();
        record_l2_pub_bound_diag::<F, 64>(&lp, 1, &witness);
        let records = L2PubBoundObserverGuard::take();
        assert_eq!(records.len(), 1);
        assert!(records[0].z_squared > 0);
        assert!(records[0].b_l2_pub >= records[0].z_squared);
    }

    #[test]
    fn inactive_observer_drops_records() {
        let lp = sample_level();
        let witness = DecomposeFoldWitness {
            z_folded_rings: vec![],
            centered_coeffs: vec![[1i32; 64]],
            centered_inf_norm: 1,
        };
        record_l2_pub_bound_diag::<F, 64>(&lp, 1, &witness);
        assert!(L2PubBoundObserverGuard::take().is_empty());
    }
}
