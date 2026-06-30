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
    pub u_concat_planes: usize,
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

    let (r, u_concat_digits) = compute_relation_quotient::<F, B, D>(
        ring_switch_ctx,
        lp,
        &instance.challenges,
        e_hat.flat_digits(),
        &decomposed_inner_rows,
        &recomposed_inner_rows,
        &e_folded,
        instance.ring_multiplier_point(),
        instance.row_coefficient_rings(),
        &z_folded_rings.committed_digits,
        z_folded_rings.num_digits_fold,
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
            &u_concat_digits,
            &z_folded_rings.committed_digits,
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
            u_concat_planes: u_concat_digits.len(),
        })
    } else {
        None
    };
    Ok(RingSwitchBuildOutput {
        w,
        terminal_artifacts,
    })
}

/// Emit flat digit planes contiguously (no block transpose). Used for the
/// tiered `û_concat` segment; a no-op for the single-tier path (empty slice).
fn emit_flat_planes<const D: usize>(out: &mut Vec<i8>, planes: &[[i8; D]]) {
    for plane in planes {
        out.extend_from_slice(plane);
    }
}

/// Build the committed witness polynomial from ring-domain digit planes.
///
/// Emits field-domain coefficients in digit-major order (block index innermost):
/// z-hat, e-hat + t-hat, û_concat, r-hat.
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
    u_concat_digits: &[[i8; D]],
    z_committed_digits: &[[i8; D]],
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
    // Tiered: the hidden decomposed concatenated slice images `û_concat` are a
    // flat contiguous segment emitted immediately after `t̂` (at `offset_u`).
    let u_concat_planes = u_concat_digits.len();
    assert_eq!(
        z_committed_digits.len() % num_digits_fold,
        0,
        "build_w_coeffs: z digit plane count must be a multiple of fold depth"
    );
    let z_folded_elems = z_committed_digits.len() / num_digits_fold;
    let z_count = e_hat_planes + t_hat_planes + u_concat_planes + z_committed_digits.len();
    let r_hat_count = r.len() * levels;
    tracing::debug!(
        e_hat_planes,
        t_hat_planes,
        z_folded_elems,
        z_folded_planes = z_committed_digits.len(),
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

    akita_types::emit_witness_z_folded_planes_inner::<D>(
        &mut out,
        z_committed_digits,
        block_len,
        depth_commit,
        num_digits_fold,
        z_folded_elems,
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
    emit_flat_planes(&mut out, u_concat_digits);

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
