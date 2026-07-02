use super::*;
use crate::compute::{OperationCtx, RingSwitchProveBackend};
use crate::validation::validate_i8_setup_log_basis;
use akita_serialization::AkitaSerialize;

/// Prover-side ring artifacts retained for segment-typed terminal encoding.
pub struct RingSwitchTerminalArtifacts<F: FieldCore, const D: usize> {
    pub e_folded: Vec<akita_algebra::CyclotomicRing<F, D>>,
    pub recomposed_inner_rows: Vec<Vec<akita_algebra::CyclotomicRing<F, D>>>,
    pub z_folded_centered: Vec<[i32; D]>,
    pub r: Vec<akita_algebra::CyclotomicRing<F, D>>,
}

/// Output of [`ring_switch_build_w`].
pub struct RingSwitchBuildOutput<F: FieldCore, const D: usize> {
    pub w: RecursiveWitnessFlat,
    pub terminal_artifacts: Option<RingSwitchTerminalArtifacts<F, D>>,
}

/// Build the witness vector `w` from the ring-relation witness.
///
/// This is the first half of the ring switch: it computes `r` and assembles
/// `w` as a flat recursive witness. The resulting `w` is D-agnostic and can be
/// committed at any supported ring dimension by the recursive commitment path.
///
/// # Errors
///
/// Returns an error if the ring-relation witness is missing prover-side data.
#[tracing::instrument(skip_all, name = "ring_switch_build_w")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_build_w<F, B, const D: usize>(
    instance: &RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    ring_switch_ctx: &OperationCtx<'_, F, B, D>,
    lp: &LevelParams,
    retain_terminal_artifacts: bool,
) -> Result<RingSwitchBuildOutput<F, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + AkitaSerialize,
    B: RingSwitchProveBackend<F, D>,
{
    let num_claims = instance.opening_batch().num_polynomials();
    let RingRelationWitness {
        z_folded_rings,
        fold_grind_nonce: _,
        e_hat,
        e_folded,
        mut hint,
    } = witness;
    validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
    hint.ensure_recomposed_inner_rows(lp.num_digits_open, lp.log_basis)?;
    let (decomposed_inner_rows, recomposed_inner_rows) = hint.into_flat_parts();
    let recomposed_inner_rows = recomposed_inner_rows.ok_or_else(|| {
        AkitaError::InvalidInput("missing recomposed inner rows in prover hint".to_string())
    })?;
    let opening_batch = instance.opening_batch();

    let r = compute_relation_quotient::<F, B, D>(
        ring_switch_ctx,
        lp,
        &instance.challenges,
        e_hat.flat_digits(),
        &decomposed_inner_rows,
        &recomposed_inner_rows,
        &e_folded,
        instance.ring_multiplier_point(),
        instance.row_coefficient_rings(),
        &z_folded_rings.centered_coeffs,
        z_folded_rings.centered_inf_norm,
        instance.y(),
        opening_batch.num_polynomials(),
        lp.num_blocks,
        lp.inner_width(),
        instance.m_row_layout(),
    )?;
    let z_centered = z_folded_rings.centered_coeffs.clone();
    let w = {
        let _span = tracing::info_span!("build_w_coeffs").entered();
        build_w_coeffs::<F, D>(
            &e_hat,
            &decomposed_inner_rows,
            &z_folded_rings.centered_coeffs,
            &r,
            lp,
            num_claims,
        )
    };
    let terminal_artifacts = if retain_terminal_artifacts {
        Some(RingSwitchTerminalArtifacts {
            e_folded,
            recomposed_inner_rows,
            z_folded_centered: z_centered,
            r,
        })
    } else {
        None
    };
    Ok(RingSwitchBuildOutput {
        w,
        terminal_artifacts,
    })
}

pub(super) fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 6,
        "log_basis must be in 1..=6 for i8 output"
    );
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
        "levels * log_basis must be <= 128 + log_basis"
    );

    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;

    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx] as i128;
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

/// Decompose centered `z` fold response coeffs and emit digit-major planes.
fn emit_z_folded_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    z_folded_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    log_basis: u32,
) {
    let total_elems = z_folded_centered.len();
    let inner_width = block_len * depth_commit;
    debug_assert_eq!(
        total_elems % inner_width,
        0,
        "z_folded_rings length {total_elems} not divisible by inner_width {inner_width}",
    );

    let mut all_planes = vec![[0i8; D]; total_elems * num_digits_fold];
    for (k, z_j) in z_folded_centered.iter().enumerate() {
        balanced_decompose_centered_i32_i8_into(
            z_j,
            &mut all_planes[k * num_digits_fold..(k + 1) * num_digits_fold],
            log_basis,
        );
    }
    akita_types::emit_witness_z_folded_planes_inner::<D>(
        out,
        &all_planes,
        block_len,
        depth_commit,
        num_digits_fold,
        total_elems,
    );
}

/// Build the committed witness polynomial from ring-domain digit planes.
///
/// Emits field-domain coefficients in digit-major order (block index innermost):
/// z-hat, e-hat + t-hat, r-hat.
///
/// Within each segment, the power-of-2 block index is the fastest-varying
/// (innermost) dimension.
///
/// `FlatDigitBlocks` stores ring-domain data in block-major order (all digit
/// planes for one block contiguously), which is natural for ring-domain matvec
/// and recomposition. This function transposes opening digits to digit-major at
/// the ring-to-field boundary.
///
/// # Panics
///
/// Panics if the caller supplies digit blocks whose plane counts do not match
/// the fold layout in `lp`.
#[allow(clippy::too_many_arguments)]
pub fn build_w_coeffs<F: CanonicalField, const D: usize>(
    e_hat: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    z_folded_centered: &[[i32; D]],
    r: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_claims: usize,
) -> RecursiveWitnessFlat {
    let log_basis = lp.log_basis;
    let num_digits_fold = lp
        .num_digits_fold(num_claims, F::modulus_bits())
        .expect("build_w_coeffs: degenerate fold bound in validated level params");
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let block_len = lp.block_len;
    let levels = r_decomp_levels::<F>(log_basis);

    let e_hat_planes = e_hat.flat_digits().len();
    let t_hat_planes = t_hat.flat_digits().len();
    let z_count = e_hat_planes + t_hat_planes + z_folded_centered.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    tracing::debug!(
        e_hat_planes,
        t_hat_planes,
        z_folded_elems = z_folded_centered.len(),
        z_folded_planes = z_folded_centered.len() * num_digits_fold,
        r_elems = r.len(),
        r_planes = r_hat_count,
        total_ring = z_count + r_hat_count,
        total_field = (z_count + r_hat_count) * D,
        "build_w_coeffs"
    );
    let total_planes = z_count + r_hat_count;
    let total_elems = total_planes * D;

    let mut out = Vec::with_capacity(total_elems);

    let w_block_count = e_hat.block_count();
    assert_eq!(
        e_hat_planes,
        w_block_count * depth_open,
        "build_w_coeffs: e_hat block layout does not match open digit depth"
    );
    let t_block_count = t_hat.block_count();
    let t_planes_per_block = if t_block_count == 0 {
        0
    } else {
        assert_eq!(
            t_hat_planes % t_block_count,
            0,
            "build_w_coeffs: t_hat block layout must be uniform"
        );
        t_hat_planes / t_block_count
    };

    emit_z_folded_block_inner(
        &mut out,
        z_folded_centered,
        block_len,
        depth_commit,
        num_digits_fold,
        log_basis,
    );
    akita_types::emit_witness_planes_block_inner(
        &mut out,
        e_hat.flat_digits(),
        w_block_count,
        depth_open,
    );
    akita_types::emit_witness_planes_block_inner(
        &mut out,
        t_hat.flat_digits(),
        t_block_count,
        t_planes_per_block,
    );

    let mut r_planes = vec![[0i8; D]; levels];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    for ri in r {
        r_planes.fill([0i8; D]);
        ri.balanced_decompose_pow2_i8_into_with_params(&mut r_planes, &decompose_params);
        for plane in &r_planes {
            out.extend_from_slice(plane);
        }
    }
    RecursiveWitnessFlat::from_i8_digits(out)
}
