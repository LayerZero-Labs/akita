use super::*;
use std::mem::size_of;
use std::ops::Range;

/// Minimum number of Rayon work-units for the fused one-shot kernel.
const MIN_FUSED_TILES: usize = 30;
#[cfg(target_arch = "aarch64")]
const FUSED_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(not(target_arch = "aarch64"))]
const FUSED_L2_CACHE_BYTES: usize = 1024 * 1024;

/// Negacyclic reduced rows and cyclic rows produced by one D-role traversal.
pub(crate) type DigitRelationRows<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>);

#[derive(Clone, Copy)]
struct CenteredRhsBounds {
    capacity: u64,
    lut: u64,
}

#[derive(Clone, Copy)]
enum DigitRole {
    Witness,
    Outer,
}

#[derive(Clone, Copy)]
struct FusedQuotientPlan {
    n_d: usize,
    n_b: usize,
    n_a: usize,
    witness_len: usize,
    t_len: usize,
    z_len: usize,
    max_col: usize,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
    z_bounds: CenteredRhsBounds,
    w_chunk_width: Option<usize>,
    t_chunk_width: Option<usize>,
    z_chunk_width: Option<usize>,
    matrix_extent: usize,
}

impl FusedQuotientPlan {
    fn is_one_shot(self) -> bool {
        self.w_chunk_width
            .is_some_and(|width| width >= self.witness_len)
            && self.t_chunk_width.is_some_and(|width| width >= self.t_len)
            && self.z_chunk_width.is_some_and(|width| width >= self.z_len)
    }

    #[inline]
    fn d_row(self, row: usize) -> Range<usize> {
        self.row(row, self.witness_len)
    }

    #[inline]
    fn b_row(self, row: usize) -> Range<usize> {
        self.row(row, self.t_len)
    }

    #[inline]
    fn a_row(self, row: usize) -> Range<usize> {
        self.row(row, self.z_len)
    }

    fn digit_shape(self, role: DigitRole) -> (usize, usize, u64) {
        match role {
            DigitRole::Witness => (self.n_d, self.witness_len, self.w_digit_abs_bound),
            DigitRole::Outer => (self.n_b, self.t_len, self.t_digit_abs_bound),
        }
    }

    fn digit_row(self, role: DigitRole, row: usize) -> Range<usize> {
        match role {
            DigitRole::Witness => self.d_row(row),
            DigitRole::Outer => self.b_row(row),
        }
    }

    #[inline]
    fn row(self, row: usize, width: usize) -> Range<usize> {
        let start = row * width;
        start..start + width
    }

    #[inline]
    fn chunk_range(len: usize, width: usize, chunk_index: usize) -> Range<usize> {
        let start = chunk_index * width;
        start..(start + width).min(len)
    }
}

enum FusedMatrixSource<'a, F: FieldCore, W: PrimeWidth, const K: usize, const D: usize> {
    Cached {
        negacyclic: &'a [CyclotomicCrtNtt<W, K, D>],
        cyclic: &'a [CyclotomicCrtNtt<W, K, D>],
    },
    Field(&'a [CyclotomicRing<F, D>]),
}

impl<'a, F, W, const K: usize, const D: usize> FusedMatrixSource<'a, F, W, K, D>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    fn validate(&self, plan: FusedQuotientPlan) -> Result<(), AkitaError> {
        let (cyclic_len, negacyclic_len) = match self {
            Self::Cached { negacyclic, cyclic } => (cyclic.len(), negacyclic.len()),
            Self::Field(source) => (source.len(), source.len()),
        };
        if cyclic_len < plan.matrix_extent {
            return Err(AkitaError::InvalidSetup(format!(
                "fused quotient cyclic matrix needs {} elements, got {cyclic_len}",
                plan.matrix_extent
            )));
        }
        let negacyclic_extent = plan.n_a.checked_mul(plan.z_len).ok_or_else(|| {
            AkitaError::InvalidSetup("fused quotient negacyclic extent overflow".into())
        })?;
        if negacyclic_len < negacyclic_extent {
            return Err(AkitaError::InvalidSetup(format!(
                "fused quotient negacyclic matrix needs {negacyclic_extent} elements, got {negacyclic_len}"
            )));
        }
        Ok(())
    }

    fn validate_digit_relation(&self, plan: FusedQuotientPlan) -> Result<(), AkitaError> {
        self.validate(plan)?;
        let required = plan.n_d.checked_mul(plan.witness_len).ok_or_else(|| {
            AkitaError::InvalidSetup("D-role negacyclic matrix extent overflow".into())
        })?;
        let available = match self {
            Self::Cached { negacyclic, .. } => negacyclic.len(),
            Self::Field(source) => source.len(),
        };
        if available < required {
            return Err(AkitaError::InvalidSetup(format!(
                "D-role negacyclic matrix needs {required} elements, got {available}"
            )));
        }
        Ok(())
    }

    #[inline(always)]
    fn with_cyclic<R>(
        &self,
        index: usize,
        params: &CrtNttParamSet<W, K, D>,
        f: impl FnOnce(&CyclotomicCrtNtt<W, K, D>) -> R,
    ) -> R {
        match self {
            Self::Cached { cyclic, .. } => f(&cyclic[index]),
            Self::Field(source) => {
                let value = CyclotomicCrtNtt::from_ring_cyclic(&source[index], params);
                f(&value)
            }
        }
    }

    #[inline(always)]
    fn with_pair<R>(
        &self,
        index: usize,
        params: &CrtNttParamSet<W, K, D>,
        f: impl FnOnce(&CyclotomicCrtNtt<W, K, D>, &CyclotomicCrtNtt<W, K, D>) -> R,
    ) -> R {
        match self {
            Self::Cached { negacyclic, cyclic } => f(&negacyclic[index], &cyclic[index]),
            Self::Field(source) => {
                let (negacyclic, cyclic) =
                    CyclotomicCrtNtt::from_ring_pair_with_params(&source[index], params);
                f(&negacyclic, &cyclic)
            }
        }
    }

    #[inline(always)]
    fn field_ring(&self, index: usize, params: &CrtNttParamSet<W, K, D>) -> CyclotomicRing<F, D> {
        match self {
            Self::Cached { negacyclic, .. } => negacyclic[index].to_ring(params),
            Self::Field(source) => source[index],
        }
    }
}

pub(crate) fn fused_quotient_matrix_extent(
    n_d: usize,
    witness_len: usize,
    n_b: usize,
    t_len: usize,
    n_a: usize,
    z_len: usize,
) -> Result<usize, AkitaError> {
    [(n_d, witness_len), (n_b, t_len), (n_a, z_len)]
        .into_iter()
        .try_fold(0, |extent, (rows, width)| {
            rows.checked_mul(width)
                .map(|role_extent| extent.max(role_extent))
        })
        .ok_or_else(|| AkitaError::InvalidSetup("fused quotient matrix extent overflow".into()))
}

/// Stream the D-role matrix once and return both transform-domain products.
///
/// The digit and CRT bounds are identical to the ordinary fused relation
/// kernel. Unlike two independent mat-vec calls, the field-form matrix entry is
/// loaded once and transformed into both domains before either accumulator is
/// updated.
pub(crate) fn digit_relation_rows_streamed_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    n_d: usize,
    e_hat: &[[i8; D]],
    log_basis_open: u32,
) -> Result<DigitRelationRows<F, D>, AkitaError> {
    macro_rules! run {
        ($params:expr) => {{
            let params = $params;
            let source = FusedMatrixSource::Field(source);
            digit_relation_rows_with_params(source, n_d, e_hat, log_basis_open, &params)
        }};
    }
    match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => run!(params),
        ProtocolCrtNttParams::Q64(params) => run!(params),
        ProtocolCrtNttParams::Q128(params) => run!(params),
    }
}

fn digit_relation_rows_with_params<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: FusedMatrixSource<'_, F, W, K, D>,
    n_d: usize,
    e_hat: &[[i8; D]],
    log_basis_open: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<DigitRelationRows<F, D>, AkitaError> {
    let (w_digit_abs_bound, t_digit_abs_bound) =
        fused_quotient_digit_bounds(log_basis_open, log_basis_open)?;
    let plan = plan_fused_quotients::<F, _, _, D>(
        e_hat,
        &[],
        &[],
        n_d,
        0,
        0,
        0,
        w_digit_abs_bound,
        t_digit_abs_bound,
        params,
    )?;
    source.validate_digit_relation(plan)?;
    let chunk_width = plan.w_chunk_width.ok_or_else(|| {
        AkitaError::InvalidSetup("CRT parameters cannot represent one D-role digit term".into())
    })?;
    Ok(accumulate_digit_relation_rows(
        &source,
        e_hat,
        plan,
        chunk_width,
        params,
    ))
}

fn fused_quotient_digit_bounds(
    log_basis_open: u32,
    log_basis_outer: u32,
) -> Result<(u64, u64), AkitaError> {
    validate_i8_log_basis(log_basis_open)?;
    validate_i8_log_basis(log_basis_outer)?;
    Ok((
        balanced_digit_abs_bound(log_basis_open),
        balanced_digit_abs_bound(log_basis_outer),
    ))
}

#[allow(clippy::too_many_arguments)]
fn plan_fused_quotients<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    n_d: usize,
    n_b: usize,
    n_a: usize,
    z_folded_max_abs: u32,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<FusedQuotientPlan, AkitaError> {
    let witness_len = if n_d != 0 { e_hat.len() } else { 0 };
    let t_len = if n_b != 0 { t_hat.len() } else { 0 };
    let z_len = if n_a != 0 { z_folded_rings.len() } else { 0 };
    if !digit_rows_within_digit_bound::<D>(e_hat, witness_len, w_digit_abs_bound) {
        return Err(AkitaError::InvalidInput(
            "fused quotient e_hat contains digits outside its log_basis range".to_string(),
        ));
    }
    if !digit_rows_within_digit_bound::<D>(t_hat, t_len, t_digit_abs_bound) {
        return Err(AkitaError::InvalidInput(
            "fused quotient t_hat contains digits outside its log_basis range".to_string(),
        ));
    }

    let actual_z_abs_bound = centered_rows_abs_bound(z_folded_rings, z_len);
    let z_bounds = CenteredRhsBounds {
        capacity: u64::from(z_folded_max_abs).max(actual_z_abs_bound),
        lut: actual_z_abs_bound,
    };
    debug_assert!(
        centered_rows_within_bound(z_folded_rings, z_len, z_bounds.capacity),
        "fused quotient centered RHS bound is smaller than the actual max"
    );

    let w_chunk_width = (witness_len == 0)
        .then_some(1)
        .or_else(|| safe_crt_chunk_width::<F, W, K, D>(params, witness_len, w_digit_abs_bound));
    let t_chunk_width = (t_len == 0)
        .then_some(1)
        .or_else(|| safe_crt_chunk_width::<F, W, K, D>(params, t_len, t_digit_abs_bound));
    let z_chunk_width = (z_len == 0 || z_bounds.capacity == 0)
        .then_some(z_len.max(1))
        .or_else(|| safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_bounds.capacity));
    let matrix_extent = fused_quotient_matrix_extent(n_d, witness_len, n_b, t_len, n_a, z_len)?;

    Ok(FusedQuotientPlan {
        n_d,
        n_b,
        n_a,
        witness_len,
        t_len,
        z_len,
        max_col: witness_len.max(t_len).max(z_len),
        w_digit_abs_bound,
        t_digit_abs_bound,
        z_bounds,
        w_chunk_width,
        t_chunk_width,
        z_chunk_width,
        matrix_extent,
    })
}

/// Fused column-tiled kernel for the three split-eq mat-vec products.
///
/// Replaces three separate NTT-cached mat-vec calls (D-cyclic, B-cyclic,
/// A-quotient) with a single pass over the shared NTT cache. Within each
/// column tile, cache entries are loaded once and reused across all three
/// products with their exact row bounds, eliminating redundant DRAM reads.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_with_params<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: FusedMatrixSource<'_, F, W, K, D>,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    source.validate(plan)?;
    if plan.max_col == 0 {
        return Ok((
            vec![CyclotomicRing::<F, D>::zero(); plan.n_d],
            vec![CyclotomicRing::<F, D>::zero(); plan.n_b],
            vec![CyclotomicRing::<F, D>::zero(); plan.n_a],
        ));
    }

    if plan.is_one_shot() {
        return Ok(fused_split_eq_quotients_one_shot(
            &source,
            e_hat,
            t_hat,
            z_folded_rings,
            plan,
            params,
        ));
    }

    let w_chunk_width = plan.w_chunk_width.ok_or_else(|| {
        AkitaError::InvalidSetup("CRT parameters cannot represent one e_hat term".to_string())
    })?;
    let t_chunk_width = plan.t_chunk_width.ok_or_else(|| {
        AkitaError::InvalidSetup("CRT parameters cannot represent one t_hat term".to_string())
    })?;
    let d_result = accumulate_cyclic_i8_rows(
        &source,
        e_hat,
        plan,
        DigitRole::Witness,
        w_chunk_width,
        params,
    );
    let b_result = accumulate_cyclic_i8_rows(
        &source,
        t_hat,
        plan,
        DigitRole::Outer,
        t_chunk_width,
        params,
    );
    let a_result = accumulate_centered_quotient_rows(&source, z_folded_rings, plan, params);

    Ok((d_result, b_result, a_result))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_one_shot<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let digit_bound = plan.w_digit_abs_bound.max(plan.t_digit_abs_bound);
    let digit_lut = (plan.witness_len != 0 || plan.t_len != 0)
        .then(|| DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound));
    let centered_lut = (plan.z_len != 0 && plan.z_bounds.lut <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, plan.z_bounds.lut as i32));
    let base_tw = (FUSED_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let tw = base_tw.min(plan.max_col.div_ceil(MIN_FUSED_TILES).max(1));
    let num_tiles = plan.max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

    let (d_accs, b_accs, a_neg_accs, a_cyc_accs) = cfg_fold_reduce!(
        0..num_tiles,
        || (
            vec![zero.clone(); plan.n_d],
            vec![zero.clone(); plan.n_b],
            vec![zero.clone(); plan.n_a],
            vec![zero.clone(); plan.n_a],
        ),
        |mut accs: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(plan.max_col);

            for j in tile_start..tile_end {
                if j < plan.witness_len && !is_zero_plane(&e_hat[j]) {
                    let lut = digit_lut.as_ref().expect("digit LUT exists");
                    let ntt_w = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&e_hat[j], params, lut);
                    for (i, acc_d) in accs.0.iter_mut().enumerate() {
                        source.with_cyclic(plan.d_row(i).start + j, params, |cyclic| {
                            accumulate_pointwise_product_into(acc_d, cyclic, &ntt_w, params);
                        });
                    }
                }

                if j < plan.t_len && !is_zero_plane(&t_hat[j]) {
                    let lut = digit_lut.as_ref().expect("digit LUT exists");
                    let ntt_t = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, lut);
                    for (i, acc_b) in accs.1.iter_mut().enumerate() {
                        source.with_cyclic(plan.b_row(i).start + j, params, |cyclic| {
                            accumulate_pointwise_product_into(acc_b, cyclic, &ntt_t, params);
                        });
                    }
                }

                if j < plan.z_len && !is_zero_centered_row(&z_folded_rings[j]) {
                    let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                        // SAFETY: `plan_fused_quotients` computed
                        // `z_bounds.lut` from these `plan.z_len` rows. This
                        // loop keeps `j < plan.z_len`, and `lut` was built for
                        // that inclusive centered coefficient bound.
                        unsafe {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                                &z_folded_rings[j],
                                params,
                                lut,
                            )
                        }
                    } else {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_params(
                            &z_folded_rings[j],
                            params,
                        )
                    };
                    for (i, (acc_neg, acc_cyc)) in
                        accs.2.iter_mut().zip(accs.3.iter_mut()).enumerate()
                    {
                        source.with_pair(plan.a_row(i).start + j, params, |neg, cyclic| {
                            accumulate_pointwise_product_into(acc_neg, neg, &ntt_z_neg, params);
                            accumulate_pointwise_product_into(acc_cyc, cyclic, &ntt_z_cyc, params);
                        });
                    }
                }
            }
            accs
        },
        |mut a: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         b| {
            for r in 0..plan.n_d {
                add_ntt_into(&mut a.0[r], &b.0[r], params);
            }
            for r in 0..plan.n_b {
                add_ntt_into(&mut a.1[r], &b.1[r], params);
            }
            for r in 0..plan.n_a {
                add_ntt_into(&mut a.2[r], &b.2[r], params);
                add_ntt_into(&mut a.3[r], &b.3[r], params);
            }
            a
        }
    );

    let d_result = d_accs
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();
    let b_result = b_accs
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();
    let a_result = a_neg_accs
        .into_iter()
        .zip(a_cyc_accs)
        .map(|(neg_acc, cyc_acc)| {
            let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring(params);
            let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
            quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring)
        })
        .collect();

    (d_result, b_result, a_result)
}

/// Streamed counterpart of [`fused_split_eq_quotients_prover_bounds`].
///
/// Entries stream from `flat`, A's field-form prefix covering every product's
/// `rows x width` extent. Roles that exceed one CRT accumulator are reduced in
/// capacity-safe chunks. If the selected protocol CRT profile cannot represent
/// one centered quotient term, the shared arithmetic falls back to exact
/// field-ring multiplication, matching the cached route's acceptance set.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fused_split_eq_quotients_streamed_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    log_basis_open: u32,
    log_basis_outer: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    let (w_digit_abs_bound, t_digit_abs_bound) =
        fused_quotient_digit_bounds(log_basis_open, log_basis_outer)?;
    macro_rules! run {
        ($params:expr) => {{
            let params = $params;
            let plan = plan_fused_quotients::<F, _, _, D>(
                e_hat,
                t_hat,
                z_folded_rings,
                n_d,
                n_b,
                n_a,
                z_folded_max_abs,
                w_digit_abs_bound,
                t_digit_abs_bound,
                &params,
            )?;
            fused_split_eq_quotients_with_params(
                FusedMatrixSource::Field(source),
                e_hat,
                t_hat,
                z_folded_rings,
                plan,
                &params,
            )
        }};
    }
    match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => run!(params),
        ProtocolCrtNttParams::Q64(params) => run!(params),
        ProtocolCrtNttParams::Q128(params) => run!(params),
    }
}

fn accumulate_cyclic_i8_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    rhs: &[[i8; D]],
    plan: FusedQuotientPlan,
    role: DigitRole,
    chunk_width: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let (num_rows, rhs_len, rhs_abs_bound) = plan.digit_shape(role);
    if num_rows == 0 {
        return vec![];
    }
    if rhs_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let num_chunks = rhs_len.div_ceil(chunk_width);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, rhs_abs_bound);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(rhs_len, chunk_width, chunk_idx);
            let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk {
                if is_zero_plane(&rhs[j]) {
                    continue;
                }
                let ntt_rhs = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&rhs[j], params, &lut);
                for (row, acc) in accs.iter_mut().enumerate() {
                    source.with_cyclic(plan.digit_row(role, row).start + j, params, |cyclic| {
                        accumulate_pointwise_product_into(acc, cyclic, &ntt_rhs, params);
                    });
                }
            }

            for (dst, acc) in out.iter_mut().zip(accs) {
                *dst += acc.to_ring_cyclic(params);
            }
            out
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for (dst, src) in a.iter_mut().zip(b) {
                *dst += src;
            }
            a
        }
    )
}

fn accumulate_digit_relation_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    rhs: &[[i8; D]],
    plan: FusedQuotientPlan,
    chunk_width: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>) {
    let (num_rows, rhs_len, rhs_abs_bound) = plan.digit_shape(DigitRole::Witness);
    if num_rows == 0 {
        return (Vec::new(), Vec::new());
    }
    if rhs_len == 0 {
        let zeros = vec![CyclotomicRing::<F, D>::zero(); num_rows];
        return (zeros.clone(), zeros);
    }

    let num_chunks = rhs_len.div_ceil(chunk_width);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, rhs_abs_bound);

    cfg_fold_reduce!(
        0..num_chunks,
        || (
            vec![CyclotomicRing::<F, D>::zero(); num_rows],
            vec![CyclotomicRing::<F, D>::zero(); num_rows],
        ),
        |mut out: (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(rhs_len, chunk_width, chunk_idx);
            let mut neg_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut cyc_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk {
                if is_zero_plane(&rhs[j]) {
                    continue;
                }
                let rhs_neg = CyclotomicCrtNtt::from_i8_with_lut(&rhs[j], params, &lut);
                let rhs_cyc = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&rhs[j], params, &lut);
                for (row, (neg_acc, cyc_acc)) in
                    neg_accs.iter_mut().zip(cyc_accs.iter_mut()).enumerate()
                {
                    source.with_pair(plan.d_row(row).start + j, params, |neg, cyc| {
                        accumulate_pointwise_product_into(neg_acc, neg, &rhs_neg, params);
                        accumulate_pointwise_product_into(cyc_acc, cyc, &rhs_cyc, params);
                    });
                }
            }

            for (dst, acc) in out.0.iter_mut().zip(neg_accs) {
                *dst += acc.to_ring(params);
            }
            for (dst, acc) in out.1.iter_mut().zip(cyc_accs) {
                *dst += acc.to_ring_cyclic(params);
            }
            out
        },
        |mut left: (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), right| {
            for (dst, src) in left.0.iter_mut().zip(right.0) {
                *dst += src;
            }
            for (dst, src) in left.1.iter_mut().zip(right.1) {
                *dst += src;
            }
            left
        }
    )
}

fn centered_rows_within_bound<const D: usize>(rows: &[[i32; D]], len: usize, bound: u64) -> bool {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| u64::from(coeff.unsigned_abs()) <= bound)
}

fn centered_rows_abs_bound<const D: usize>(rows: &[[i32; D]], len: usize) -> u64 {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .map(|&coeff| u64::from(coeff.unsigned_abs()))
        .max()
        .unwrap_or(0)
}

fn centered_i32_ring<F: CanonicalField, const D: usize>(coeffs: &[i32; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(from_fn(|k| F::from_i64(coeffs[k] as i64)))
}

fn accumulate_centered_quotient_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let num_rows = plan.n_a;
    if num_rows == 0 {
        return vec![];
    }
    if plan.z_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    if plan.z_bounds.lut == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let Some(chunk_width) = plan.z_chunk_width else {
        return accumulate_centered_quotient_rows_field(source, z_folded_rings, plan, params);
    };
    let centered_lut = (plan.z_bounds.lut <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, plan.z_bounds.lut as i32));
    let num_chunks = plan.z_len.div_ceil(chunk_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(plan.z_len, chunk_width, chunk_idx);
            let mut neg_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut cyc_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk {
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                    // SAFETY: `plan_fused_quotients` computed
                    // `z_bounds.lut` from these `plan.z_len` rows. This loop
                    // keeps `j < plan.z_len`, and `lut` was built for that
                    // inclusive centered coefficient bound.
                    unsafe {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                            &z_folded_rings[j],
                            params,
                            lut,
                        )
                    }
                } else {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_folded_rings[j], params)
                };
                for (row, (neg_acc, cyc_acc)) in
                    neg_accs.iter_mut().zip(cyc_accs.iter_mut()).enumerate()
                {
                    source.with_pair(plan.a_row(row).start + j, params, |neg, cyclic| {
                        accumulate_pointwise_product_into(neg_acc, neg, &ntt_z_neg, params);
                        accumulate_pointwise_product_into(cyc_acc, cyclic, &ntt_z_cyc, params);
                    });
                }
            }

            for ((dst, neg_acc), cyc_acc) in out.iter_mut().zip(neg_accs).zip(cyc_accs) {
                let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring(params);
                let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
                *dst += quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring);
            }
            out
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for (dst, src) in a.iter_mut().zip(b) {
                *dst += src;
            }
            a
        }
    )
}

fn accumulate_centered_quotient_rows_field<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..plan.n_a)
        .map(|row_idx| {
            let mut out = CyclotomicRing::<F, D>::zero();
            for (j, z_folded) in z_folded_rings.iter().enumerate().take(plan.z_len) {
                if is_zero_centered_row(z_folded) {
                    continue;
                }
                let z = centered_i32_ring::<F, D>(z_folded);
                let lhs = source.field_ring(plan.a_row(row_idx).start + j, params);
                let neg_product = lhs * z;
                let mut cyc_product = CyclotomicRing::<F, D>::zero();
                add_cyclic_product_into(&mut cyc_product, &lhs, &z);
                out += quotient_from_cyclic_and_negacyclic(&cyc_product, &neg_product);
            }
            out
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn centered_quotient_rows_with_i16_tail_params<
    F: FieldCore + CanonicalField + HalvingField,
    const K: usize,
    const D: usize,
>(
    neg: &[CyclotomicCrtNtt<i32, K, D>],
    cyc: &[CyclotomicCrtNtt<i32, K, D>],
    tail_neg: &[CyclotomicCrtNtt<i16, 1, D>],
    tail_cyc: &[CyclotomicCrtNtt<i16, 1, D>],
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    params: &CrtNttParamSet<i32, K, D>,
    tail_params: &CrtNttParamSet<i16, 1, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if num_rows == 0 {
        return Ok(Vec::new());
    }
    let width = z_folded_rings.len();
    let required = num_rows
        .checked_mul(width)
        .ok_or_else(|| AkitaError::InvalidSetup("quotient matrix shape overflows".into()))?;
    if width == 0
        || [neg.len(), cyc.len(), tail_neg.len(), tail_cyc.len()]
            .into_iter()
            .any(|length| length < required)
    {
        return Err(AkitaError::InvalidSetup(
            "base-plus-tail quotient cache is shorter than its matrix shape".into(),
        ));
    }
    let actual_bound = centered_rows_abs_bound(z_folded_rings, width);
    if actual_bound == 0 {
        return Ok(vec![CyclotomicRing::<F, D>::zero(); num_rows]);
    }
    let capacity_bound = u64::from(z_folded_max_abs).max(actual_bound);
    let capacity = params
        .crt_capacity()
        .with_prime_modulus(tail_params.primes[0].p as u128);
    let chunk_width = capacity
        .max_safe_width::<F, D>(capacity_bound)
        .map(|safe| safe.min(width))
        .filter(|&safe| safe > 0)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("centered quotient exceeds base plus i16-tail capacity".into())
        })?;
    let mixed_params = I16TailParams::new(params.clone(), tail_params.clone());
    let base_lut = (actual_bound <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<i32, K>::new(params, actual_bound as i32));
    let num_chunks = width.div_ceil(chunk_width);

    Ok(cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let start = chunk_idx * chunk_width;
            let end = (start + chunk_width).min(width);
            let mut base_neg_accs = vec![CyclotomicCrtNtt::<i32, K, D>::zero(); num_rows];
            let mut base_cyc_accs = vec![CyclotomicCrtNtt::<i32, K, D>::zero(); num_rows];
            let mut tail_neg_accs = vec![CyclotomicCrtNtt::<i16, 1, D>::zero(); num_rows];
            let mut tail_cyc_accs = vec![CyclotomicCrtNtt::<i16, 1, D>::zero(); num_rows];

            for (offset, z_ring) in z_folded_rings[start..end].iter().enumerate() {
                if is_zero_centered_row(z_ring) {
                    continue;
                }
                let j = start + offset;
                let (z_neg, z_cyc) = if let Some(ref lut) = base_lut {
                    // SAFETY: `actual_bound` bounds every centered coefficient in
                    // `z_folded_rings`; the LUT is built for that bound, and `j`
                    // ranges only over the validated `0..width` source rows.
                    unsafe {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                            z_ring, params, lut,
                        )
                    }
                } else {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(z_ring, params)
                };
                let (z_tail_neg, z_tail_cyc) =
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(z_ring, tail_params);
                for row in 0..num_rows {
                    let index = row * width + j;
                    accumulate_pointwise_product_into(
                        &mut base_neg_accs[row],
                        &neg[index],
                        &z_neg,
                        params,
                    );
                    accumulate_pointwise_product_into(
                        &mut base_cyc_accs[row],
                        &cyc[index],
                        &z_cyc,
                        params,
                    );
                    accumulate_pointwise_product_into(
                        &mut tail_neg_accs[row],
                        &tail_neg[index],
                        &z_tail_neg,
                        tail_params,
                    );
                    accumulate_pointwise_product_into(
                        &mut tail_cyc_accs[row],
                        &tail_cyc[index],
                        &z_tail_cyc,
                        tail_params,
                    );
                }
            }

            for row in 0..num_rows {
                let neg_ring = ntt_with_i16_tail_to_ring(
                    &base_neg_accs[row],
                    &tail_neg_accs[row],
                    &mixed_params,
                );
                let cyc_ring = cyclic_ntt_with_i16_tail_to_ring(
                    &base_cyc_accs[row],
                    &tail_cyc_accs[row],
                    &mixed_params,
                );
                out[row] += quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring);
            }
            out
        },
        |mut left: Vec<CyclotomicRing<F, D>>, right| {
            for (dst, src) in left.iter_mut().zip(right) {
                *dst += src;
            }
            left
        }
    ))
}

/// Centered A-quotient rows using the protocol CRT prefix plus its 14-bit tail.
pub(crate) fn centered_quotient_rows_with_i16_tail<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    tail_slot: &PreparedNttCache<D>,
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    let tail = tail_slot.i16_tail_pair().ok_or_else(|| {
        AkitaError::InvalidSetup("paired i16-tail NTT domain not prepared".into())
    })?;
    macro_rules! dispatch {
        ($neg_base:expr, $cyc_base:expr) => {{
            let (neg_base, cyc_base) = ($neg_base, $cyc_base);
            if neg_base.params() != cyc_base.params() {
                return Err(AkitaError::InvalidSetup(
                    "cyclic and negacyclic NTT profiles do not match".into(),
                ));
            }
            centered_quotient_rows_with_i16_tail_params(
                neg_base.negacyclic().ok_or_else(|| {
                    AkitaError::InvalidSetup("negacyclic NTT domain not prepared".into())
                })?,
                cyc_base.cyclic().ok_or_else(|| {
                    AkitaError::InvalidSetup("cyclic NTT domain not prepared".into())
                })?,
                tail.negacyclic(),
                tail.cyclic(),
                num_rows,
                z_folded_rings,
                z_folded_max_abs,
                neg_base.params(),
                tail.params(),
            )
        }};
    }
    if let (Some(neg), Some(cyc)) = (negacyclic_slot.q32_base(), cyclic_slot.q32_base()) {
        dispatch!(neg, cyc)
    } else if let (Some(neg), Some(cyc)) = (negacyclic_slot.q64_base(), cyclic_slot.q64_base()) {
        dispatch!(neg, cyc)
    } else if let (Some(neg), Some(cyc)) = (negacyclic_slot.q128_base(), cyclic_slot.q128_base()) {
        dispatch!(neg, cyc)
    } else {
        Err(AkitaError::InvalidSetup(
            "cyclic and negacyclic NTT profiles do not match".into(),
        ))
    }
}

/// Fused split-eq quotient kernel dispatching over [`PreparedNttCache`] variants.
///
/// Computes three NTT-cached mat-vec products in a single tiled pass:
/// - D-cyclic: `cyc[0..n_d] · e_hat` (cyclic domain)
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
///
/// All roles share the same underlying coefficient matrix, but each role uses
/// its own packed row width.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "fused_split_eq_quotients")]
#[cfg(test)]
pub(crate) fn fused_split_eq_quotients<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &PreparedNttCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    fused_split_eq_quotients_with_digit_bound(
        slot,
        slot,
        n_d,
        n_b,
        n_a,
        e_hat,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        balanced_digit_abs_bound(6),
        balanced_digit_abs_bound(6),
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fused_split_eq_quotients_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    log_basis_open: u32,
    log_basis_outer: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    let (w_digit_abs_bound, t_digit_abs_bound) =
        fused_quotient_digit_bounds(log_basis_open, log_basis_outer)?;
    fused_split_eq_quotients_with_digit_bound(
        negacyclic_slot,
        cyclic_slot,
        n_d,
        n_b,
        n_a,
        e_hat,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        w_digit_abs_bound,
        t_digit_abs_bound,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_with_digit_bound<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    macro_rules! run {
        ($neg_base:expr, $cyc_base:expr) => {{
            let (neg_base, cyc_base) = ($neg_base, $cyc_base);
            let (params, cyclic_params) = (neg_base.params(), cyc_base.params());
            if params != cyclic_params {
                return Err(AkitaError::InvalidSetup(
                    "cyclic and negacyclic NTT profiles do not match".into(),
                ));
            }
            let neg = match neg_base.negacyclic() {
                Some(neg) => neg,
                None if n_a == 0 => &[],
                None => {
                    return Err(AkitaError::InvalidSetup(
                        "negacyclic NTT domain not prepared".into(),
                    ));
                }
            };
            let cyc = cyc_base
                .cyclic()
                .ok_or_else(|| AkitaError::InvalidSetup("cyclic NTT domain not prepared".into()))?;
            let plan = plan_fused_quotients::<F, _, _, D>(
                e_hat,
                t_hat,
                z_folded_rings,
                n_d,
                n_b,
                n_a,
                z_folded_max_abs,
                w_digit_abs_bound,
                t_digit_abs_bound,
                params,
            )?;
            fused_split_eq_quotients_with_params(
                FusedMatrixSource::Cached {
                    negacyclic: neg,
                    cyclic: cyc,
                },
                e_hat,
                t_hat,
                z_folded_rings,
                plan,
                params,
            )
        }};
    }
    match (
        negacyclic_slot.q32_base(),
        cyclic_slot.q32_base(),
        negacyclic_slot.q64_base(),
        cyclic_slot.q64_base(),
        negacyclic_slot.q128_base(),
        cyclic_slot.q128_base(),
    ) {
        (Some(neg), Some(cyc), _, _, _, _) => run!(neg, cyc),
        (_, _, Some(neg), Some(cyc), _, _) => run!(neg, cyc),
        (_, _, _, _, Some(neg), Some(cyc)) => run!(neg, cyc),
        _ => Err(AkitaError::InvalidSetup(
            "cyclic and negacyclic NTT profiles do not match".into(),
        )),
    }
}
