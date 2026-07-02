use super::*;
use crate::compute::{OperationCtx, RuntimeRingSwitchProveBackend};
use crate::validation::validate_i8_setup_log_basis;
use akita_algebra::CyclotomicRing;
use akita_serialization::AkitaSerialize;

/// Prover-side ring artifacts retained for segment-typed terminal encoding.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed ring rows via [`Self::e_folded_trusted`],
/// [`Self::recomposed_block_trusted`], [`Self::z_folded_centered_trusted`], and
/// [`Self::r_trusted`].
pub struct RingSwitchTerminalArtifacts<F: FieldCore> {
    pub e_folded: RingVec<F>,
    pub recomposed_inner_rows: Vec<RingVec<F>>,
    z_folded_centered_flat: Vec<i32>,
    pub r: RingVec<F>,
    pub u_concat_planes: usize,
    ring_dim: usize,
}

impl<F: FieldCore> RingSwitchTerminalArtifacts<F> {
    /// Construct from typed ring-switch output at a kernel boundary.
    pub fn from_parts<const D: usize>(
        e_folded: Vec<CyclotomicRing<F, D>>,
        recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
        z_folded_centered: Vec<[i32; D]>,
        r: Vec<CyclotomicRing<F, D>>,
        u_concat_planes: usize,
    ) -> Self {
        Self {
            e_folded: RingVec::from_ring_elems(&e_folded),
            recomposed_inner_rows: recomposed_inner_rows
                .into_iter()
                .map(|block| RingVec::from_ring_elems(&block))
                .collect(),
            z_folded_centered_flat: z_folded_centered
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect(),
            r: RingVec::from_ring_elems(&r),
            u_concat_planes,
            ring_dim: D,
        }
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Flat centered fold-response coefficients (`ring_dim` field elements per row).
    pub fn z_folded_centered_flat(&self) -> &[i32] {
        &self.z_folded_centered_flat
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "ring switch terminal artifacts ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if !self.z_folded_centered_flat.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.z_folded_centered_flat.len(),
            });
        }
        if !self.e_folded.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.e_folded.coeff_len(),
            });
        }
        if !self.r.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.r.coeff_len(),
            });
        }
        for block in &self.recomposed_inner_rows {
            if !block.can_decode_vec(D) {
                return Err(AkitaError::InvalidSize {
                    expected: D,
                    actual: block.coeff_len(),
                });
            }
        }
        Ok(())
    }

    /// Borrow folded `e` rows after [`Self::ensure_ring_dim`].
    pub fn e_folded_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        Ok(self.e_folded.as_ring_slice_trusted::<D>())
    }

    /// Borrow recomposed rows for one block after [`Self::ensure_ring_dim`].
    pub fn recomposed_block_trusted<const D: usize>(
        &self,
        block: usize,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.recomposed_inner_rows
            .get(block)
            .ok_or_else(|| {
                AkitaError::InvalidInput(format!(
                    "ring switch terminal artifacts block index {block} out of range"
                ))
            })
            .map(|rows| rows.as_ring_slice_trusted::<D>())
    }

    /// Borrow centered coefficient rows after [`Self::ensure_ring_dim`].
    pub fn z_folded_centered_trusted<const D: usize>(&self) -> Result<&[[i32; D]], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        let (chunks, rem) = self.z_folded_centered_flat.as_chunks::<D>();
        debug_assert!(rem.is_empty());
        Ok(chunks)
    }

    /// Borrow relation quotient `r` rows after [`Self::ensure_ring_dim`].
    pub fn r_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        Ok(self.r.as_ring_slice_trusted::<D>())
    }
}

/// Output of [`ring_switch_build_w`].
pub struct RingSwitchBuildOutput<F: FieldCore> {
    pub w: RecursiveWitnessFlat,
    pub terminal_artifacts: Option<RingSwitchTerminalArtifacts<F>>,
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
pub fn ring_switch_build_w<F, B>(
    instance: &RingRelationInstance<F>,
    witness: RingRelationWitness<F>,
    ring_switch_ctx: &OperationCtx<'_, F, B>,
    lp: &LevelParams,
    retain_terminal_artifacts: bool,
) -> Result<RingSwitchBuildOutput<F>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + AkitaSerialize,
    B: RuntimeRingSwitchProveBackend<F>,
{
    let ring_d = lp.ring_dimension;
    dispatch_ring_dim_result!(ring_d, |D| {
        let num_claims = instance.opening_batch().num_polynomials();
        let RingRelationWitness {
            z_folded_rings,
            fold_grind_nonce: _,
            e_hat,
            e_folded,
            hint,
            ..
        } = witness;
        validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
        let e_hat = FlatDigitBlocks::<D>::from_digit_blocks(&e_hat)?;
        let e_folded = e_folded.as_ring_slice_trusted::<D>();
        let recomposed_inner_rows = crate::compute::recompose_flat_hint_inner_rows::<F, D>(
            &hint,
            lp.num_digits_open,
            lp.log_basis,
        )?;
        let decomposed_inner_rows =
            FlatDigitBlocks::<D>::from_digit_blocks(&hint.into_flat_parts()?)?;
        let opening_batch = instance.opening_batch();

        instance.ensure_ring_dim::<D>()?;
        let (r, u_concat_digits) = compute_relation_quotient::<F, B, D>(
            ring_switch_ctx,
            lp,
            &instance.challenges,
            e_hat.flat_digits(),
            &decomposed_inner_rows,
            &recomposed_inner_rows,
            e_folded,
            instance.ring_multiplier_point(),
            instance.row_coefficient_rings_trusted::<D>()?,
            z_folded_rings.centered_coeffs_trusted::<D>(),
            z_folded_rings.centered_inf_norm,
            instance.y_trusted::<D>()?,
            opening_batch.num_polynomials(),
            lp.num_blocks,
            lp.inner_width(),
            instance.m_row_layout(),
        )?;
        let z_centered = z_folded_rings.centered_coeffs_owned::<D>();
        let w = {
            let _span = tracing::info_span!("build_w_coeffs").entered();
            build_w_coeffs::<F, D>(
                &e_hat,
                &decomposed_inner_rows,
                &u_concat_digits,
                z_folded_rings.centered_coeffs_trusted::<D>(),
                &r,
                lp,
                num_claims,
            )
        };
        let terminal_artifacts = if retain_terminal_artifacts {
            Some(RingSwitchTerminalArtifacts::from_parts::<D>(
                e_folded.to_vec(),
                recomposed_inner_rows,
                z_centered,
                r,
                u_concat_digits.len(),
            ))
        } else {
            None
        };
        Ok(RingSwitchBuildOutput {
            w,
            terminal_artifacts,
        })
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

/// Emit flat digit planes contiguously (no block transpose). Used for the
/// tiered `û_concat` segment; a no-op for the single-tier path (empty slice).
fn emit_flat_planes<const D: usize>(out: &mut Vec<i8>, planes: &[[i8; D]]) {
    for plane in planes {
        out.extend_from_slice(plane);
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
    // Tiered: the hidden decomposed concatenated slice images `û_concat` are a
    // flat contiguous segment emitted immediately after `t̂` (at `offset_u`).
    let u_concat_planes = u_concat_digits.len();
    let z_count =
        e_hat_planes + t_hat_planes + u_concat_planes + z_folded_centered.len() * num_digits_fold;
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
