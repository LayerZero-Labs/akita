//! Prover-side sampling for commitment masking.

use akita_field::{AkitaError, CanonicalField};
use akita_field::{FieldCore, RandomSampling};
use akita_sumcheck::{EqFactoredUniPoly, FullUniPoly};
use akita_types::stage1_tree_stage_shapes;
use akita_types::{zk, FlatDigitBlocks};
use rand_core::{OsRng, RngCore};

fn sample_balanced_pow2_digit<R: RngCore>(rng: &mut R, log_basis: u32) -> i8 {
    // The alphabet size is a power of two, so masking low bits is uniform.
    let raw = (rng.next_u32() & ((1u32 << log_basis) - 1)) as i16;
    let half_basis = 1i16 << (log_basis - 1);
    let basis = half_basis << 1;
    let balanced = if raw >= half_basis { raw - basis } else { raw };
    balanced as i8
}

/// Sample a fresh digit-source LHL blinding vector.
///
/// # Errors
///
/// Returns an error if digit block sizing overflows.
pub(crate) fn sample_blinding_digits<F, const D: usize>(
    output_ring_len: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    F: CanonicalField,
{
    if !(1..=8).contains(&log_basis) {
        return Err(AkitaError::InvalidInput(
            "ZK digit blinding log_basis must be in 1..=8".to_string(),
        ));
    }

    let blinding_planes = zk::blinding_digit_plane_count::<F>(output_ring_len, D, log_basis);
    if blinding_planes == 0 {
        return Ok(FlatDigitBlocks::empty());
    }

    let block_sizes = vec![blinding_planes];
    let mut out = FlatDigitBlocks::zeroed(block_sizes)?;
    let mut rng = OsRng;
    for plane in out.flat_digits_mut() {
        for coeff in plane {
            *coeff = sample_balanced_pow2_digit(&mut rng, log_basis);
        }
    }
    Ok(out)
}

/// Sample hiding-witness slots for all Akita sumcheck pads at one fold level.
///
/// The returned pool is collected into the proof-level hiding witness before the
/// folded batch opening transcript begins.
pub(crate) fn sample_sumcheck_pad_pool<F>(
    rounds: usize,
    b: usize,
) -> (Vec<EqFactoredUniPoly<F>>, Vec<FullUniPoly<F>>)
where
    F: FieldCore + RandomSampling,
{
    let mut rng = OsRng;
    let mut eq_factored_round_pads = Vec::new();
    for shape in stage1_tree_stage_shapes(rounds, b) {
        let stored_coeffs =
            EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(shape.sumcheck_proof.1);
        eq_factored_round_pads.extend((0..shape.sumcheck_proof.0).map(|_| EqFactoredUniPoly {
            coeffs_except_linear_term: (0..stored_coeffs).map(|_| F::random(&mut rng)).collect(),
        }));
    }
    let full_round_pads = (0..rounds)
        .map(|_| FullUniPoly::from_coeffs((0..=3).map(|_| F::random(&mut rng)).collect()))
        .collect();
    (eq_factored_round_pads, full_round_pads)
}
