use super::*;
use std::ops::Range;

mod planning;
mod tail;
pub(crate) use planning::fused_quotient_matrix_extent;
#[cfg(test)]
pub(super) use planning::fused_test_plan_route;
use planning::*;

pub use tail::centered_quotient_rows_with_i16_tail;

/// Minimum number of Rayon work-units for the fused one-shot kernel.
const MIN_FUSED_TILES: usize = 30;
#[cfg(target_arch = "aarch64")]
const FUSED_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(not(target_arch = "aarch64"))]
const FUSED_L2_CACHE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedQuotientRows<F: Field, const D: usize> {
    pub(crate) b_cyclic: Vec<CyclotomicRing<F, D>>,
    pub(crate) a_quotients: Vec<CyclotomicRing<F, D>>,
}

struct FusedNttAccumulators<W: PrimeWidth, const K: usize, const D: usize> {
    b: Vec<CyclotomicCrtNtt<W, K, D>>,
    a_negacyclic: Vec<CyclotomicCrtNtt<W, K, D>>,
    a_cyclic: Vec<CyclotomicCrtNtt<W, K, D>>,
}

enum FusedMatrixSource<'a, F: Field, W: PrimeWidth, const K: usize, const D: usize> {
    Cached {
        negacyclic: &'a [CyclotomicCrtNtt<W, K, D>],
        cyclic: &'a [CyclotomicCrtNtt<W, K, D>],
    },
    Field(&'a [CyclotomicRing<F, D>]),
}

impl<'a, F, W, const K: usize, const D: usize> FusedMatrixSource<'a, F, W, K, D>
where
    F: Field + CanonicalEncoding,
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

    /// Prefetches one column of a `rows`-by-`width` cached matrix: its cyclic
    /// entries, plus the negacyclic ones when `pair` is set.
    ///
    /// Run-batched kernels read a whole run of entries only after
    /// transforming the run's right-hand sides; prefetching each column before
    /// its transform overlaps those loads with the transform.
    #[inline(always)]
    fn prefetch_column(&self, rows: usize, width: usize, column: usize, pair: bool) {
        if let Self::Cached { negacyclic, cyclic } = self {
            for row in 0..rows {
                let index = row * width + column;
                cyclic[index].prefetch();
                if pair {
                    negacyclic[index].prefetch();
                }
            }
        }
    }

    /// Adds `matrix[row, column_start..] · rhs` in cyclic form into each row
    /// accumulator of a `width`-column matrix.
    #[inline(always)]
    fn accumulate_cyclic_run(
        &self,
        accs: &mut [CyclotomicCrtNtt<W, K, D>],
        width: usize,
        column_start: usize,
        rhs: &[CyclotomicCrtNtt<W, K, D>],
        params: &CrtNttParamSet<W, K, D>,
    ) {
        for (row, acc) in accs.iter_mut().enumerate() {
            let start = row * width + column_start;
            match self {
                Self::Cached { cyclic, .. } => {
                    acc.add_assign_pointwise_dot(&cyclic[start..start + rhs.len()], rhs, params);
                }
                Self::Field(source) => {
                    for (entry, rhs) in source[start..start + rhs.len()].iter().zip(rhs) {
                        let lhs = CyclotomicCrtNtt::from_ring_cyclic(entry, params);
                        acc.add_assign_pointwise_mul(&lhs, rhs, params);
                    }
                }
            }
        }
    }

    /// Adds the negacyclic and cyclic products of `matrix[row, column_start..]`
    /// with the paired right-hand sides into each row's accumulators.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn accumulate_pair_run(
        &self,
        neg_accs: &mut [CyclotomicCrtNtt<W, K, D>],
        cyc_accs: &mut [CyclotomicCrtNtt<W, K, D>],
        width: usize,
        column_start: usize,
        rhs_neg: &[CyclotomicCrtNtt<W, K, D>],
        rhs_cyc: &[CyclotomicCrtNtt<W, K, D>],
        params: &CrtNttParamSet<W, K, D>,
    ) {
        for (row, (neg_acc, cyc_acc)) in neg_accs.iter_mut().zip(cyc_accs).enumerate() {
            let columns = row * width + column_start..row * width + column_start + rhs_neg.len();
            match self {
                Self::Cached { negacyclic, cyclic } => {
                    neg_acc.add_assign_pointwise_dot(&negacyclic[columns.clone()], rhs_neg, params);
                    cyc_acc.add_assign_pointwise_dot(&cyclic[columns], rhs_cyc, params);
                }
                Self::Field(source) => {
                    for ((entry, rhs_neg), rhs_cyc) in
                        source[columns].iter().zip(rhs_neg).zip(rhs_cyc)
                    {
                        let (neg, cyc) =
                            CyclotomicCrtNtt::from_ring_pair_with_params(entry, params);
                        neg_acc.add_assign_pointwise_mul(&neg, rhs_neg, params);
                        cyc_acc.add_assign_pointwise_mul(&cyc, rhs_cyc, params);
                    }
                }
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

/// Fused column-tiled kernel for the B and A split-eq mat-vec products.
///
/// The two products share the same coefficient matrix but use independent row
/// counts and packed widths. A column tile reuses cached entries when their
/// active prefixes overlap.
fn fused_split_eq_quotients_with_params<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: FusedMatrixSource<'_, F, W, K, D>,
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<FusedQuotientRows<F, D>, AkitaError> {
    source.validate(plan)?;
    if plan.max_col == 0 {
        return Ok(FusedQuotientRows {
            b_cyclic: vec![CyclotomicRing::<F, D>::zero(); plan.n_b],
            a_quotients: vec![CyclotomicRing::<F, D>::zero(); plan.n_a],
        });
    }

    if plan.is_one_shot() {
        return Ok(fused_split_eq_quotients_one_shot(
            &source,
            t_hat,
            z_folded_rings,
            plan,
            params,
        ));
    }

    let t_chunk_width = plan.t_chunk_width.ok_or_else(|| {
        AkitaError::InvalidSetup("CRT parameters cannot represent one t_hat term".to_string())
    })?;
    let b_result = accumulate_cyclic_i8_rows(&source, t_hat, plan, t_chunk_width, params);
    let a_result = accumulate_centered_quotient_rows(&source, z_folded_rings, plan, params);

    Ok(FusedQuotientRows {
        b_cyclic: b_result,
        a_quotients: a_result,
    })
}

/// Transforms one centered row into its negacyclic and cyclic CRT+NTT forms.
///
/// # Safety
///
/// When `lut` is present, every coefficient of `row` must lie within the
/// inclusive bound the LUT was built for.
#[inline(always)]
unsafe fn centered_pair_ntt<W: PrimeWidth, const K: usize, const D: usize>(
    row: &[i32; D],
    params: &CrtNttParamSet<W, K, D>,
    lut: Option<&CenteredMontLut<W, K>>,
) -> (CyclotomicCrtNtt<W, K, D>, CyclotomicCrtNtt<W, K, D>) {
    match lut {
        // SAFETY: the caller guarantees `row` fits the LUT bound.
        Some(lut) => unsafe {
            CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(row, params, lut)
        },
        None => CyclotomicCrtNtt::from_centered_i32_pair_with_params(row, params),
    }
}

struct CyclicRunScratch<W: PrimeWidth, const K: usize, const D: usize> {
    rhs: Vec<CyclotomicCrtNtt<W, K, D>>,
}

impl<W: PrimeWidth, const K: usize, const D: usize> CyclicRunScratch<W, K, D> {
    fn new(batch: usize) -> Self {
        Self {
            rhs: Vec::with_capacity(batch),
        }
    }
}

struct PairedRunScratch<W: PrimeWidth, const K: usize, const D: usize> {
    neg: Vec<CyclotomicCrtNtt<W, K, D>>,
    cyc: Vec<CyclotomicCrtNtt<W, K, D>>,
}

impl<W: PrimeWidth, const K: usize, const D: usize> PairedRunScratch<W, K, D> {
    fn new(batch: usize) -> Self {
        Self {
            neg: Vec::with_capacity(batch),
            cyc: Vec::with_capacity(batch),
        }
    }
}

/// The same zero-skipping, prefetch/transform, then row-major dot schedule is
/// used by both a one-shot tile and a capacity-limited CRT chunk.
fn accumulate_cyclic_i8_runs<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    rhs: &[[i8; D]],
    rows: usize,
    range: Range<usize>,
    lut: &DigitMontLut<W, K>,
    accs: &mut [CyclotomicCrtNtt<W, K, D>],
    scratch: &mut CyclicRunScratch<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    let width = rhs.len();
    for_each_nonzero_column_run(
        range,
        params.pointwise_dot_batch_size(),
        |j| is_zero_plane(&rhs[j]),
        |run| {
            scratch.rhs.clear();
            for j in run.clone() {
                source.prefetch_column(rows, width, j, false);
                scratch.rhs.push(CyclotomicCrtNtt::from_i8_cyclic_with_lut(
                    &rhs[j], params, lut,
                ));
            }
            source.accumulate_cyclic_run(accs, width, run.start, &scratch.rhs, params);
        },
    );
}

/// Optional second CRT profile used when a centered product needs the i16
/// tail. Its scratch lives alongside the base scratch for the whole chunk.
struct TailPairRuns<'a, const D: usize> {
    neg: &'a [CyclotomicCrtNtt<i16, 1, D>],
    cyc: &'a [CyclotomicCrtNtt<i16, 1, D>],
    neg_accs: &'a mut [CyclotomicCrtNtt<i16, 1, D>],
    cyc_accs: &'a mut [CyclotomicCrtNtt<i16, 1, D>],
    scratch: &'a mut PairedRunScratch<i16, 1, D>,
    params: &'a CrtNttParamSet<i16, 1, D>,
}

#[allow(clippy::too_many_arguments)]
fn accumulate_centered_pair_runs<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    rhs: &[[i32; D]],
    rows: usize,
    range: Range<usize>,
    lut: Option<&CenteredMontLut<W, K>>,
    neg_accs: &mut [CyclotomicCrtNtt<W, K, D>],
    cyc_accs: &mut [CyclotomicCrtNtt<W, K, D>],
    scratch: &mut PairedRunScratch<W, K, D>,
    mut tail: Option<&mut TailPairRuns<'_, D>>,
    params: &CrtNttParamSet<W, K, D>,
) {
    let width = rhs.len();
    for_each_nonzero_column_run(
        range,
        params.pointwise_dot_batch_size(),
        |j| is_zero_centered_row(&rhs[j]),
        |run| {
            scratch.neg.clear();
            scratch.cyc.clear();
            if let Some(extra) = tail.as_deref_mut() {
                extra.scratch.neg.clear();
                extra.scratch.cyc.clear();
            }
            for j in run.clone() {
                source.prefetch_column(rows, width, j, true);
                if let Some(extra) = tail.as_deref_mut() {
                    for row in 0..rows {
                        let index = row * width + j;
                        extra.neg[index].prefetch();
                        extra.cyc[index].prefetch();
                    }
                }
                // SAFETY: callers construct `lut` from the maximum absolute
                // coefficient of the rows covered by `range`.
                let (neg, cyc) = unsafe { centered_pair_ntt(&rhs[j], params, lut) };
                scratch.neg.push(neg);
                scratch.cyc.push(cyc);
                if let Some(extra) = tail.as_deref_mut() {
                    let (neg, cyc) =
                        CyclotomicCrtNtt::from_centered_i32_pair_with_params(&rhs[j], extra.params);
                    extra.scratch.neg.push(neg);
                    extra.scratch.cyc.push(cyc);
                }
            }
            source.accumulate_pair_run(
                neg_accs,
                cyc_accs,
                width,
                run.start,
                &scratch.neg,
                &scratch.cyc,
                params,
            );
            if let Some(extra) = tail.as_deref_mut() {
                for (row, (neg_acc, cyc_acc)) in extra
                    .neg_accs
                    .iter_mut()
                    .zip(extra.cyc_accs.iter_mut())
                    .enumerate()
                {
                    let columns = row * width + run.start..row * width + run.end;
                    neg_acc.add_assign_pointwise_dot(
                        &extra.neg[columns.clone()],
                        &extra.scratch.neg,
                        extra.params,
                    );
                    cyc_acc.add_assign_pointwise_dot(
                        &extra.cyc[columns],
                        &extra.scratch.cyc,
                        extra.params,
                    );
                }
            }
        },
    );
}

fn fused_split_eq_quotients_one_shot<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> FusedQuotientRows<F, D> {
    let digit_lut = (plan.t_len != 0)
        .then(|| DigitMontLut::<W, K>::new_with_digit_bound(params, plan.t_digit_abs_bound));
    let centered_lut = (plan.z_len != 0 && plan.z_bounds.lut <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, plan.z_bounds.lut as i32));
    let tw = fused_tile_width(plan, params);
    let num_tiles = plan.max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();
    let dot_batch = params.pointwise_dot_batch_size();

    let accs = cfg_fold_reduce!(
        0..num_tiles,
        || FusedNttAccumulators {
            b: vec![zero.clone(); plan.n_b],
            a_negacyclic: vec![zero.clone(); plan.n_a],
            a_cyclic: vec![zero.clone(); plan.n_a],
        },
        |mut accs: FusedNttAccumulators<W, K, D>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(plan.max_col);
            let mut cyclic_scratch = CyclicRunScratch::new(dot_batch);
            let mut pair_scratch = PairedRunScratch::new(dot_batch);

            if let Some(lut) = digit_lut.as_ref() {
                accumulate_cyclic_i8_runs(
                    source,
                    t_hat,
                    plan.n_b,
                    tile_start..tile_end.min(plan.t_len),
                    lut,
                    &mut accs.b,
                    &mut cyclic_scratch,
                    params,
                );
            }

            accumulate_centered_pair_runs(
                source,
                z_folded_rings,
                plan.n_a,
                tile_start..tile_end.min(plan.z_len),
                centered_lut.as_ref(),
                &mut accs.a_negacyclic,
                &mut accs.a_cyclic,
                &mut pair_scratch,
                None,
                params,
            );
            accs
        },
        |mut a: FusedNttAccumulators<W, K, D>, b| {
            for r in 0..plan.n_b {
                add_ntt_into(&mut a.b[r], &b.b[r], params);
            }
            for r in 0..plan.n_a {
                add_ntt_into(&mut a.a_negacyclic[r], &b.a_negacyclic[r], params);
                add_ntt_into(&mut a.a_cyclic[r], &b.a_cyclic[r], params);
            }
            a
        }
    );

    let b_result = accs
        .b
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();
    let a_result = accs
        .a_negacyclic
        .into_iter()
        .zip(accs.a_cyclic)
        .map(|(neg_acc, cyc_acc)| {
            let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring(params);
            let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
            quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring)
        })
        .collect();

    FusedQuotientRows {
        b_cyclic: b_result,
        a_quotients: a_result,
    }
}

/// Streamed counterpart of [`fused_split_eq_quotients_prover_bounds`].
///
/// Entries stream from `flat`, A's field-form prefix covering every product's
/// `rows x width` extent. Roles that exceed one CRT accumulator are reduced in
/// capacity-safe chunks. If the selected protocol CRT profile cannot represent
/// one centered quotient term, the shared arithmetic falls back to exact
/// field-ring multiplication, matching the cached route's acceptance set.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_split_eq_quotients_streamed_prover_bounds<
    F: Field + CanonicalEncoding,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    n_b: usize,
    n_a: usize,
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    log_basis_outer: u32,
) -> Result<FusedQuotientRows<F, D>, AkitaError> {
    let t_digit_abs_bound = fused_quotient_digit_bound(log_basis_outer)?;
    macro_rules! run {
        ($params:expr) => {{
            let params = $params;
            let plan = plan_fused_quotients::<F, _, _, D>(
                t_hat,
                z_folded_rings,
                n_b,
                n_a,
                z_folded_max_abs,
                t_digit_abs_bound,
                &params,
            )?;
            fused_split_eq_quotients_with_params(
                FusedMatrixSource::Field(source),
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
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &FusedMatrixSource<'_, F, W, K, D>,
    rhs: &[[i8; D]],
    plan: FusedQuotientPlan,
    chunk_width: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let (num_rows, rhs_len, rhs_abs_bound) = (plan.n_b, plan.t_len, plan.t_digit_abs_bound);
    if num_rows == 0 {
        return vec![];
    }
    if rhs_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let num_chunks = rhs_len.div_ceil(chunk_width);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, rhs_abs_bound);
    let dot_batch = params.pointwise_dot_batch_size();

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(rhs_len, chunk_width, chunk_idx);
            let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut scratch = CyclicRunScratch::new(dot_batch);
            accumulate_cyclic_i8_runs(
                source,
                rhs,
                num_rows,
                chunk,
                &lut,
                &mut accs,
                &mut scratch,
                params,
            );

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

fn centered_i32_ring<F: Field + CanonicalEncoding, const D: usize>(
    coeffs: &[i32; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(from_fn(|k| F::from_i64(coeffs[k] as i64)))
}

fn accumulate_centered_quotient_rows<
    F: Field + CanonicalEncoding,
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
    let dot_batch = params.pointwise_dot_batch_size();

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(plan.z_len, chunk_width, chunk_idx);
            let mut neg_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut cyc_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut scratch = PairedRunScratch::new(dot_batch);
            accumulate_centered_pair_runs(
                source,
                z_folded_rings,
                num_rows,
                chunk,
                centered_lut.as_ref(),
                &mut neg_accs,
                &mut cyc_accs,
                &mut scratch,
                None,
                params,
            );

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
    F: Field + CanonicalEncoding,
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
                let lhs = source.field_ring(row_idx * plan.z_len + j, params);
                let neg_product = lhs * z;
                let mut cyc_product = CyclotomicRing::<F, D>::zero();
                add_cyclic_product_into(&mut cyc_product, &lhs, &z);
                out += quotient_from_cyclic_and_negacyclic(&cyc_product, &neg_product);
            }
            out
        })
        .collect()
}

/// Fused split-eq quotient kernel dispatching over [`PreparedNttCache`] variants.
///
/// Computes two NTT-cached mat-vec products in a single tiled pass:
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
///
/// All roles share the same underlying coefficient matrix, but each role uses
/// its own packed row width.
#[tracing::instrument(skip_all, name = "fused_split_eq_quotients")]
#[cfg(test)]
pub(crate) fn fused_split_eq_quotients<F: Field + CanonicalEncoding, const D: usize>(
    slot: &PreparedNttCache<D>,
    n_b: usize,
    n_a: usize,
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
) -> Result<FusedQuotientRows<F, D>, AkitaError> {
    fused_split_eq_quotients_with_digit_bound(
        slot,
        slot,
        n_b,
        n_a,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        balanced_digit_abs_bound(6),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_split_eq_quotients_prover_bounds<F: Field + CanonicalEncoding, const D: usize>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    n_b: usize,
    n_a: usize,
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    log_basis_outer: u32,
) -> Result<FusedQuotientRows<F, D>, AkitaError> {
    let t_digit_abs_bound = fused_quotient_digit_bound(log_basis_outer)?;
    fused_split_eq_quotients_with_digit_bound(
        negacyclic_slot,
        cyclic_slot,
        n_b,
        n_a,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        t_digit_abs_bound,
    )
}

#[allow(clippy::too_many_arguments)]
fn fused_split_eq_quotients_with_digit_bound<F: Field + CanonicalEncoding, const D: usize>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    n_b: usize,
    n_a: usize,
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    t_digit_abs_bound: u64,
) -> Result<FusedQuotientRows<F, D>, AkitaError> {
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
                t_hat,
                z_folded_rings,
                n_b,
                n_a,
                z_folded_max_abs,
                t_digit_abs_bound,
                params,
            )?;
            fused_split_eq_quotients_with_params(
                FusedMatrixSource::Cached {
                    negacyclic: neg,
                    cyclic: cyc,
                },
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

#[cfg(test)]
pub(super) fn fused_test_reduced_profile<F: Field + CanonicalEncoding, const D: usize>(
    flat: &[CyclotomicRing<F, D>],
    n_b: usize,
    n_a: usize,
    t_hat: &[[i8; D]],
    z: &[[i32; D]],
) -> (
    FusedQuotientRows<F, D>,
    FusedQuotientRows<F, D>,
    usize,
    usize,
) {
    use akita_algebra::ntt::tables::I16_TAIL_PRIME;
    let params = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
    let plan = plan_fused_quotients::<F, _, _, D>(t_hat, z, n_b, n_a, 1, 1, &params)
        .expect("reduced CRT plan");
    let (neg, cyc): (Vec<_>, Vec<_>) = flat
        .iter()
        .map(|entry| CyclotomicCrtNtt::from_ring_pair_with_params(entry, &params))
        .unzip();
    let cached = fused_split_eq_quotients_with_params(
        FusedMatrixSource::Cached {
            negacyclic: &neg,
            cyclic: &cyc,
        },
        t_hat,
        z,
        plan,
        &params,
    )
    .expect("reduced cached route");
    let streamed = fused_split_eq_quotients_with_params(
        FusedMatrixSource::Field(flat),
        t_hat,
        z,
        plan,
        &params,
    )
    .expect("reduced streamed route");
    (
        cached,
        streamed,
        plan.t_chunk_width.expect("cyclic chunks"),
        plan.z_chunk_width.expect("paired chunks"),
    )
}
