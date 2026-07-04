use super::*;
use crate::compute::{OperationCtx, RingSwitchProveBackend};
use crate::protocol::ring_relation::validate_chunked_witness_cfg;
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
    let num_claims = instance.opening_batch().num_total_polynomials();
    let RingRelationWitness {
        z_folded_rings,
        z_folded_centered_per_chunk,
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
    let num_digits_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;

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
        &z_folded_rings.committed_digits,
        num_digits_fold,
        instance.y(),
        opening_batch.num_total_polynomials(),
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
            &z_folded_rings.committed_digits,
            &z_folded_centered_per_chunk,
            &r,
            lp,
            num_claims,
        )?
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

/// Emit one chunk's window of a block-major digit segment (`ê` or `t̂`),
/// digit-major with the block index innermost, restricted to the global block
/// window `[block_lo, block_lo + blocks_per_chunk)`.
fn emit_witness_planes_block_window<const D: usize>(
    out: &mut Vec<i8>,
    flat: &[[i8; D]],
    num_outer: usize,
    num_blocks: usize,
    planes_per_block: usize,
    block_lo: usize,
    blocks_per_chunk: usize,
) {
    for compound_dig in 0..planes_per_block {
        for outer in 0..num_outer {
            for bl in 0..blocks_per_chunk {
                let blk = outer * num_blocks + (block_lo + bl);
                out.extend_from_slice(&flat[blk * planes_per_block + compound_dig]);
            }
        }
    }
}

/// Emit one chunk's committed shifted `z` digit planes in digit-major order.
fn emit_z_committed_chunk_inner<const D: usize>(
    out: &mut Vec<i8>,
    chunk_committed_digits: &[[i8; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
) {
    let total_elems = chunk_committed_digits.len() / num_digits_fold;
    akita_types::emit_witness_z_folded_planes_inner::<D>(
        out,
        chunk_committed_digits,
        block_len,
        depth_commit,
        num_digits_fold,
        total_elems,
    );
}

/// Build the committed witness polynomial from ring-domain digit planes.
#[allow(clippy::too_many_arguments)]
pub fn build_w_coeffs<F: CanonicalField, const D: usize>(
    e_hat: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    z_committed_digits: &[[i8; D]],
    z_folded_centered_per_chunk: &[Vec<[i32; D]>],
    r: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_claims: usize,
) -> Result<RecursiveWitnessFlat, AkitaError> {
    let log_basis = lp.log_basis;
    let num_digits_fold = lp.num_digits_fold(num_claims, lp.field_bits_for_cache())?;
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let num_blocks = lp.num_blocks;
    let levels = r_decomp_levels::<F>(log_basis);

    validate_chunked_witness_cfg(lp)?;
    let num_chunks = lp.witness_chunk.num_chunks;
    let blocks_per_chunk = num_blocks / num_chunks;
    let block_len = lp.block_len;

    let e_hat_planes = e_hat.flat_digits().len();
    let t_hat_planes = t_hat.flat_digits().len();
    assert_eq!(
        z_committed_digits.len() % num_digits_fold,
        0,
        "build_w_coeffs: z digit plane count must be a multiple of fold depth"
    );
    let z_planes_total: usize = z_folded_centered_per_chunk
        .iter()
        .map(|z| z.len() * num_digits_fold)
        .sum();
    assert_eq!(
        z_committed_digits.len(),
        z_planes_total,
        "build_w_coeffs: committed z planes must match per-chunk geometry"
    );
    let z_count = e_hat_planes + t_hat_planes + z_planes_total;
    let r_hat_count = r.len() * levels;

    let mut out = Vec::with_capacity((z_count + r_hat_count) * D);

    let w_block_count = e_hat.block_count();
    let e_num_outer = w_block_count.checked_div(num_blocks).unwrap_or(0);
    let t_block_count = t_hat.block_count();
    let t_planes_per_block = if t_block_count == 0 {
        0
    } else {
        t_hat_planes
            .checked_div(t_block_count)
            .expect("t_hat_planes divisible by t_block_count")
    };
    let t_num_outer = t_block_count.checked_div(num_blocks).unwrap_or(0);
    assert_eq!(
        z_folded_centered_per_chunk.len(),
        num_chunks,
        "build_w_coeffs: per-chunk fold count must equal num_chunks"
    );

    let mut digit_offset = 0usize;
    for (chunk, z_i) in z_folded_centered_per_chunk.iter().enumerate() {
        let chunk_planes = z_i.len() * num_digits_fold;
        let chunk_digits = &z_committed_digits[digit_offset..digit_offset + chunk_planes];
        emit_z_committed_chunk_inner(
            &mut out,
            chunk_digits,
            block_len,
            depth_commit,
            num_digits_fold,
        );
        digit_offset += chunk_planes;
        let block_lo = chunk * blocks_per_chunk;
        emit_witness_planes_block_window(
            &mut out,
            e_hat.flat_digits(),
            e_num_outer,
            num_blocks,
            depth_open,
            block_lo,
            blocks_per_chunk,
        );
        emit_witness_planes_block_window(
            &mut out,
            t_hat.flat_digits(),
            t_num_outer,
            num_blocks,
            t_planes_per_block,
            block_lo,
            blocks_per_chunk,
        );
    }

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
    Ok(RecursiveWitnessFlat::from_i8_digits(out))
}
