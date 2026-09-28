//! Prepared ring matrices times digit-plane vectors on the device.
//!
//! [`DeviceNttMatrix`] holds a CPU-prepared negacyclic NTT matrix
//! ([`PreparedNttBaseView`]) with its CRT profile. [`DeviceNttMatrix::mat_vec`]
//! computes `out[b][r] = sum_c A[r][c] x[b][c]` over `Z_q[X]/(X^D + 1)` for
//! blocks `x[b]` of small signed digit planes, as the CPU's
//! `mat_vec_mul_ntt_digits_i8` and `PreparedNttCache::mat_vec_i16` do, and
//! returns the same field coefficients.

use std::marker::PhantomData;
use std::time::Duration;

use akita_algebra::tables::{Q128_NUM_PRIMES, Q64_NUM_PRIMES};
use akita_algebra::{CanonicalEncoding, CrtCapacity, CrtNttParamSet, CyclotomicRing, NttPrime};
use akita_error::checked;
use akita_types::ntt_cache::PreparedNttBaseView;
use bytemuck::{NoUninit, Pod, Zeroable};
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};
use jolt_metal::MetalField;

use crate::crt::{crt_kernel, CrtBatch, COEFFICIENT_GROUP};
use crate::error::AkitaMetalError;
use crate::library::{ring_degree_kernel, AkitaMetal, Instance, RING_DEGREES};
use crate::ntt::{shape_overflow, DeviceCrtNtt};

/// Blocks per partials threadgroup: each column's digit planes for these
/// blocks share one set of transform barriers and matrix loads.
const BLOCK_TILE: usize = 4;
/// Matrix rows per partials threadgroup, accumulated in registers. Every
/// production matrix has one row; more rows run as more row tiles.
const ROW_TILE: usize = 1;
/// Limb counts with compiled partials kernels.
pub const LIMB_COUNTS: [usize; 4] = [1, 2, 3, 4];
/// Threadgroups the partials pass aims for; columns split into chunks until
/// the grid reaches it.
const TARGET_GROUPS: usize = 256;

mod sealed {
    use akita_algebra::tables::{Q128_NUM_PRIMES, Q64_NUM_PRIMES};
    use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};

    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}

    pub trait PreparedMatrixField<const K: usize> {}
    impl PreparedMatrixField<Q64_NUM_PRIMES> for Prime64Offset59 {}
    impl PreparedMatrixField<Q128_NUM_PRIMES> for Prime128OffsetA7F7 {}
}

/// A signed digit type for digit planes: `i8` for `log_basis <= 8`, `i16`
/// up to 16.
pub trait DigitPlane: sealed::Sealed + NoUninit {
    /// The MSL spelling.
    const MSL_NAME: &'static str;
    /// Host-name suffix of the kernels over this digit type.
    const SUFFIX: &'static str;
    /// The largest `log_basis` whose balanced digits fit.
    const MAX_LOG_BASIS: u32;
}

impl DigitPlane for i8 {
    const MSL_NAME: &'static str = "char";
    const SUFFIX: &'static str = "i8";
    const MAX_LOG_BASIS: u32 = 8;
}

impl DigitPlane for i16 {
    const MSL_NAME: &'static str = "short";
    const SUFFIX: &'static str = "i16";
    const MAX_LOG_BASIS: u32 = 16;
}

/// A field whose full prepared-cache profile contains exactly `K` primes.
///
/// This sealed association prevents a field-erased [`PreparedNttBaseView`]
/// from being labeled as the wrong field by [`DeviceNttMatrix::new`]. Limb
/// matrices built by [`DeviceNttMatrix::from_rings`] may still use any
/// supported prefix length.
///
/// ```compile_fail
/// use akita_algebra::tables::Q128_NUM_PRIMES;
/// use akita_metal::matvec::DeviceNttMatrix;
/// use akita_metal::AkitaMetal;
/// use akita_types::ntt_cache::PreparedNttBaseView;
/// use jolt_field::Prime64Offset59;
///
/// fn relabel_q128_as_fp64<const D: usize>(
///     metal: &AkitaMetal,
///     view: PreparedNttBaseView<'_, i32, Q128_NUM_PRIMES, D>,
/// ) {
///     let _ = DeviceNttMatrix::<Prime64Offset59, Q128_NUM_PRIMES, D>::new(
///         metal, view, 1, 1,
///     );
/// }
/// ```
pub trait PreparedMatrixField<const K: usize>:
    MetalField + CanonicalEncoding + sealed::PreparedMatrixField<K>
{
}

impl PreparedMatrixField<Q64_NUM_PRIMES> for Prime64Offset59 {}
impl PreparedMatrixField<Q128_NUM_PRIMES> for Prime128OffsetA7F7 {}

fn partials_kernel<T: DigitPlane>(ring_degree: usize, limbs: usize) -> String {
    format!(
        "{}_{}_l{limbs}",
        ring_degree_kernel("akita_matvec_partials", ring_degree),
        T::SUFFIX
    )
}

fn partials_instance<T: DigitPlane>(ring_degree: usize, limbs: usize) -> Instance {
    Instance {
        template: "akita_matvec_partials",
        args: format!(
            "{ring_degree}, {}, {BLOCK_TILE}, {ROW_TILE}, {limbs}",
            T::MSL_NAME
        ),
        host_name: partials_kernel::<T>(ring_degree, limbs),
    }
}

pub(crate) fn instances() -> Vec<Instance> {
    RING_DEGREES
        .iter()
        .flat_map(|&ring_degree| {
            LIMB_COUNTS
                .iter()
                .flat_map(move |&limbs| {
                    [
                        partials_instance::<i8>(ring_degree, limbs),
                        partials_instance::<i16>(ring_degree, limbs),
                    ]
                })
                .chain([Instance {
                    template: "akita_matvec_finish",
                    args: ring_degree.to_string(),
                    host_name: ring_degree_kernel("akita_matvec_finish", ring_degree),
                }])
        })
        .collect()
}

/// Layout of `akita::MatvecShape` in `shaders/akita/matvec.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MatvecShape {
    blocks: u32,
    rows: u32,
    cols: u32,
    primes: u32,
    limbs: u32,
    col_begin: u32,
    col_end: u32,
    chunk_cols: u32,
    chunks: u32,
    block_tiles: u32,
    row_tiles: u32,
}

/// The modulus of `F`: one more than the canonical value of `-1`.
#[expect(clippy::arithmetic_side_effects, reason = "field negation is modular")]
pub(crate) fn field_modulus<F: MetalField + CanonicalEncoding>() -> Result<u128, AkitaMetalError> {
    (-F::one())
        .to_u128_checked()
        .and_then(|top| top.checked_add(1))
        .ok_or_else(|| AkitaMetalError::Shape("the field modulus exceeds u128".into()))
}

/// The CRT primes and matrix limbs one matvec runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatvecPlan {
    /// How many of the profile's primes, from the first.
    pub primes: usize,
    /// Limbs per matrix entry (one: the centered entry itself).
    pub limbs: usize,
}

/// Relative cost of one column per prime: the digit transform and loads,
/// then the multiply-accumulate per limb. Measured on the dense inner
/// commitment (the transform is about 87% of a one-limb column).
const TRANSFORM_COST: usize = 87;
const LIMB_COST: usize = 13;

/// The cheapest `(primes, limbs)` whose exact per-limb products fit one CRT
/// segment for `cols` columns of degree `ring_degree` and `log_basis`
/// digits, over prefixes of `profile`; the whole profile with one limb (and
/// CRT segments) when none fits. `modulus` is the field modulus.
pub fn plan_matvec(
    modulus: u128,
    profile: &[NttPrime<i32>],
    cols: usize,
    ring_degree: usize,
    log_basis: u32,
) -> MatvecPlan {
    let digit_bound = log_basis
        .checked_sub(1)
        .and_then(|shift| 1u64.checked_shl(shift))
        .unwrap_or(u64::MAX);
    let mut best: Option<(usize, MatvecPlan)> = None;
    for primes in 1..=profile.len() {
        let Some(prefix) = profile.get(..primes) else {
            break;
        };
        let capacity = CrtCapacity::from_prime_moduli(
            prefix
                .iter()
                .map(|prime| u128::from(prime.p.unsigned_abs())),
        );
        for limbs in LIMB_COUNTS {
            let fits = limb_modulus(modulus, limbs).is_some_and(|bound| {
                capacity.supports_modulus(cols, ring_degree, bound, digit_bound)
            });
            let cost = checked::product([
                primes,
                checked::mul_add(LIMB_COST, limbs, TRANSFORM_COST).unwrap_or(usize::MAX),
            ]);
            if let (true, Some(cost)) = (fits, cost) {
                if best.is_none_or(|(best_cost, _)| cost < best_cost) {
                    best = Some((cost, MatvecPlan { primes, limbs }));
                }
            }
        }
    }
    best.map_or(
        MatvecPlan {
            primes: profile.len(),
            limbs: 1,
        },
        |(_, plan)| plan,
    )
}

/// The bits per limb when a centered entry (magnitude at most `q / 2`)
/// splits into `limbs` balanced limbs: `ceil((bits(q) + 1) / limbs)`.
fn limb_bits(modulus: u128, limbs: usize) -> Option<u32> {
    let bits = 128u32
        .checked_sub(modulus.leading_zeros())?
        .checked_add(1)?;
    let limbs = u32::try_from(limbs).ok().filter(|&limbs| limbs > 0)?;
    bits.checked_add(limbs.checked_sub(1)?)?.checked_div(limbs)
}

/// The modulus whose half bounds one limb's magnitude, as
/// `CrtCapacity::supports_modulus` takes it: the field modulus itself for
/// one limb (a centered entry), `2^w` for limbs of `w` bits.
fn limb_modulus(modulus: u128, limbs: usize) -> Option<u128> {
    if limbs == 1 {
        return Some(modulus);
    }
    1u128.checked_shl(limb_bits(modulus, limbs)?)
}

/// A prepared `rows x cols` negacyclic NTT matrix over `F` in device memory,
/// each entry split into `limbs` limbs. The field parameter binds preparation,
/// capacity pricing, limb scales, and output interpretation to one field.
pub struct DeviceNttMatrix<F, const K: usize, const D: usize> {
    ntt: DeviceCrtNtt<K, D>,
    /// `[rows][cols][limbs][K][D]` raw Montgomery words.
    entries: DeviceBuffer<i32>,
    rows: usize,
    cols: usize,
    limbs: usize,
    /// The exact modulus bound used to price each prepared limb.
    limb_modulus: u128,
    /// The construction-time split width; absent for an unsplit matrix.
    limb_width: Option<u32>,
    capacity: CrtCapacity,
    field: PhantomData<F>,
}

impl<F, const K: usize, const D: usize> DeviceNttMatrix<F, K, D>
where
    F: MetalField + CanonicalEncoding,
{
    /// Uploads the leading `rows x cols` entries of a prepared negacyclic
    /// matrix (row-major, as the CPU prepares it) and its CRT profile.
    pub fn new(
        metal: &AkitaMetal,
        view: PreparedNttBaseView<'_, i32, K, D>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, AkitaMetalError>
    where
        F: PreparedMatrixField<K>,
    {
        let modulus = field_modulus::<F>()?;
        let prepared = view.negacyclic().ok_or_else(|| {
            AkitaMetalError::Shape("the negacyclic NTT domain is not prepared".into())
        })?;
        let count = rows.checked_mul(cols).ok_or_else(|| shape_overflow(rows))?;
        let prepared = prepared.get(..count).ok_or_else(|| {
            AkitaMetalError::Shape(format!(
                "a prepared matrix of {} entries has no {rows} x {cols} prefix",
                prepared.len()
            ))
        })?;
        let words = prepared
            .iter()
            .flat_map(|entry| entry.limbs.iter().flatten().map(|word| word.raw()))
            .collect::<Vec<_>>();
        let params = view.params();
        Ok(Self {
            ntt: DeviceCrtNtt::new(metal, params)?,
            entries: DeviceBuffer::from_slice(metal.device(), &words)?,
            rows,
            cols,
            limbs: 1,
            limb_modulus: modulus,
            limb_width: None,
            capacity: params.crt_capacity(),
            field: PhantomData,
        })
    }

    /// Prepares a `rows x cols` matrix of coefficient-form rings (row-major)
    /// on the device: each centered entry splits into `limbs` balanced limbs
    /// (`plan_matvec` chooses the count), each limb is reduced modulo
    /// `params`' primes, and the device transforms them. The matvec then
    /// recombines the limbs in the field, so the product equals the CPU's.
    pub fn from_rings(
        metal: &AkitaMetal,
        params: &CrtNttParamSet<i32, K, D>,
        rings: &[CyclotomicRing<F, D>],
        rows: usize,
        cols: usize,
        limbs: usize,
    ) -> Result<Self, AkitaMetalError> {
        if !LIMB_COUNTS.contains(&limbs) {
            return Err(AkitaMetalError::Shape(format!(
                "{limbs} limbs has no kernel"
            )));
        }
        let count = rows.checked_mul(cols).ok_or_else(|| shape_overflow(rows))?;
        let rings = rings.get(..count).ok_or_else(|| {
            AkitaMetalError::Shape(format!(
                "{} rings has no {rows} x {cols} prefix",
                rings.len()
            ))
        })?;
        let modulus = field_modulus::<F>()?;
        let limb_width = limb_bits(modulus, limbs)
            .ok_or_else(|| AkitaMetalError::Shape(format!("{limbs} limbs of a {modulus} field")))?;
        let limb_modulus = limb_modulus(modulus, limbs).ok_or_else(|| shape_overflow(limbs))?;
        let words = split_limbs(rings, modulus, limbs, limb_width, &params.primes)?;
        let ntt = DeviceCrtNtt::new(metal, params)?;
        let mut entries = DeviceBuffer::from_slice(metal.device(), &words)?;
        ntt.forward(metal, &mut entries)?;
        Ok(Self {
            ntt,
            entries,
            rows,
            cols,
            limbs,
            limb_modulus,
            limb_width: (limbs > 1).then_some(limb_width),
            capacity: params.crt_capacity(),
            field: PhantomData,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Multiplies the matrix by every block of `planes` and writes the
    /// products' field coefficients to `out`, returning the GPU time.
    ///
    /// `planes` holds `blocks x cols` digit planes of `D` digits, block-major,
    /// each digit balanced for `log_basis` (in `[-2^(log_basis-1),
    /// 2^(log_basis-1)]`); `out` receives `blocks x rows` ring elements. The
    /// columns run in CRT segments, each as wide as the primes' product can
    /// hold exactly, and the segments add in the field, so the result is
    /// `A x mod q` exactly at any width. A shape whose single column
    /// overflows the primes is rejected before any work.
    pub fn mat_vec<T>(
        &self,
        metal: &AkitaMetal,
        planes: &DeviceBuffer<T>,
        log_basis: u32,
        out: &mut DeviceBuffer<F>,
    ) -> Result<Duration, AkitaMetalError>
    where
        T: DigitPlane,
    {
        if !(1..=T::MAX_LOG_BASIS).contains(&log_basis) {
            return Err(AkitaMetalError::Shape(format!(
                "log_basis {log_basis} does not fit {} digits",
                T::SUFFIX
            )));
        }
        let plane_words = checked::product([self.cols, D]).ok_or_else(|| shape_overflow(D))?;
        let blocks = checked::exact_div(planes.len(), plane_words).ok_or_else(|| {
            AkitaMetalError::Shape(format!(
                "{} digits is not a whole number of {} x {D} blocks",
                planes.len(),
                self.cols
            ))
        })?;
        let coefficients =
            checked::product([blocks, self.rows, D]).ok_or_else(|| shape_overflow(blocks))?;
        if out.len() != coefficients {
            return Err(AkitaMetalError::Shape(format!(
                "{blocks} blocks of {} rows need {coefficients} output coefficients, got {}",
                self.rows,
                out.len()
            )));
        }
        let digit_bound = log_basis
            .checked_sub(1)
            .and_then(|shift| 1u64.checked_shl(shift))
            .ok_or_else(|| shape_overflow(D))?;
        // CRT segments: the widest column ranges whose exact products the
        // primes hold. Each is reconstructed into the field and added there,
        // as the CPU does for widths past its capacity.
        let segment_cols = self
            .capacity
            .max_safe_width_for_modulus(D, self.limb_modulus, digit_bound)
            .filter(|&width| width > 0)
            .ok_or_else(|| {
                AkitaMetalError::Shape(format!(
                    "{K} primes cannot hold one column of degree {D} with digits up to {digit_bound}"
                ))
            })?
            .min(self.cols);
        if coefficients == 0 {
            return Ok(Duration::ZERO);
        }

        let block_tiles =
            checked::div_ceil(blocks, BLOCK_TILE).ok_or_else(|| shape_overflow(blocks))?;
        let row_tiles =
            checked::div_ceil(self.rows, ROW_TILE).ok_or_else(|| shape_overflow(self.rows))?;
        let tile_groups =
            checked::product([K, block_tiles, row_tiles]).ok_or_else(|| shape_overflow(blocks))?;
        let wanted_chunks = checked::div_ceil(TARGET_GROUPS, tile_groups)
            .unwrap_or(1)
            .clamp(1, segment_cols);
        let chunk_cols = checked::div_ceil(segment_cols, wanted_chunks)
            .ok_or_else(|| shape_overflow(segment_cols))?;
        // Chunks per segment; a shorter last segment leaves its trailing
        // chunks empty, and they contribute zero sums.
        let chunks = checked::div_ceil(segment_cols, chunk_cols)
            .ok_or_else(|| shape_overflow(segment_cols))?;
        let partial_groups =
            checked::product([chunks, tile_groups]).ok_or_else(|| shape_overflow(chunks))?;
        let residue_rows = checked::product([self.limbs, blocks, self.rows, K])
            .ok_or_else(|| shape_overflow(blocks))?;
        let lanes = D / 2;

        let u32_of = |value: usize| u32::try_from(value).map_err(|_| shape_overflow(value));
        let mut shapes = Vec::new();
        let mut crt_shapes = Vec::new();
        let mut col_begin = 0;
        while col_begin < self.cols {
            let col_end = col_begin
                .checked_add(segment_cols)
                .ok_or_else(|| shape_overflow(col_begin))?
                .min(self.cols);
            shapes.push(MatvecShape {
                blocks: u32_of(blocks)?,
                rows: u32_of(self.rows)?,
                cols: u32_of(self.cols)?,
                primes: u32_of(K)?,
                limbs: u32_of(self.limbs)?,
                col_begin: u32_of(col_begin)?,
                col_end: u32_of(col_end)?,
                chunk_cols: u32_of(chunk_cols)?,
                chunks: u32_of(chunks)?,
                block_tiles: u32_of(block_tiles)?,
                row_tiles: u32_of(row_tiles)?,
            });
            crt_shapes.push(CrtBatch {
                coefficients: u32_of(coefficients)?,
                log_degree: D.trailing_zeros(),
                accumulate: u32::from(col_begin > 0),
                limbs: u32_of(self.limbs)?,
            });
            col_begin = col_end;
        }
        let partials = DeviceBuffer::<i32>::zeroed(
            metal.device(),
            checked::product([chunks, residue_rows, D]).ok_or_else(|| shape_overflow(chunks))?,
        )?;
        let residues = DeviceBuffer::<i32>::zeroed(
            metal.device(),
            checked::product([residue_rows, D]).ok_or_else(|| shape_overflow(residue_rows))?,
        )?;
        let radix = DeviceBuffer::from_slice(metal.device(), &self.ntt.crt_weights::<F>())?;
        let scales = DeviceBuffer::from_slice(
            metal.device(),
            &limb_scales::<F>(self.limb_width, self.limbs)?,
        )?;
        let partials_pipeline = metal.pipeline(&partials_kernel::<T>(D, self.limbs))?;
        let finish_pipeline = metal.pipeline(&ring_degree_kernel("akita_matvec_finish", D))?;
        let crt_pipeline = metal.pipeline(&crt_kernel::<F>(K))?;
        let partial_threads = checked::product([partial_groups, lanes])
            .ok_or_else(|| shape_overflow(partial_groups))?;
        let finish_threads =
            checked::product([residue_rows, lanes]).ok_or_else(|| shape_overflow(residue_rows))?;

        // Dispatches run in order, so the segments can share the partials
        // and residue buffers.
        let mut batch = Batch::new(metal.device())?;
        for (shape, crt_shape) in shapes.iter().zip(&crt_shapes) {
            batch.dispatch(
                partials_pipeline,
                &[
                    Binding::buffer(planes),
                    Binding::buffer(&self.entries),
                    Binding::buffer(&partials),
                    Binding::buffer(&self.ntt.primes),
                    Binding::buffer(&self.ntt.tables),
                    Binding::value(shape),
                ],
                Grid::linear(partial_threads, lanes),
            )?;
            batch.dispatch(
                finish_pipeline,
                &[
                    Binding::buffer(&partials),
                    Binding::buffer(&residues),
                    Binding::buffer(&self.ntt.primes),
                    Binding::buffer(&self.ntt.tables),
                    Binding::value(shape),
                ],
                Grid::linear(finish_threads, lanes),
            )?;
            batch.dispatch(
                crt_pipeline,
                &[
                    Binding::buffer(&residues),
                    Binding::buffer(out),
                    Binding::buffer(&self.ntt.primes),
                    Binding::buffer(&self.ntt.gamma),
                    Binding::buffer(&radix),
                    Binding::buffer(&scales),
                    Binding::value(crt_shape),
                ],
                Grid::linear(coefficients, COEFFICIENT_GROUP),
            )?;
        }
        Ok(batch.commit_and_wait()?)
    }
}

/// `2^(w l)` in `F` for each limb `l`: the weights that recombine limb
/// products (one, for a single limb).
#[expect(
    clippy::arithmetic_side_effects,
    reason = "field multiplication is modular"
)]
fn limb_scales<F: MetalField + CanonicalEncoding>(
    width: Option<u32>,
    limbs: usize,
) -> Result<Vec<F>, AkitaMetalError> {
    if limbs == 1 {
        return Ok(vec![F::one()]);
    }
    let width = width.ok_or_else(|| shape_overflow(limbs))?;
    let step = F::from_u128(
        1u128
            .checked_shl(width)
            .ok_or_else(|| shape_overflow(limbs))?,
    );
    let mut scale = F::one();
    Ok((0..limbs)
        .map(|_| {
            let current = scale;
            scale *= step;
            current
        })
        .collect())
}

/// The centered entries of `rings` as `limbs` balanced limbs of `width`
/// bits, each reduced modulo every prime into Montgomery form, laid out
/// `[entry][limb][prime][D]` for the device transform. One limb is the
/// centered entry itself.
fn centered_value(canonical: u128, modulus: u128) -> Option<i128> {
    if canonical > modulus / 2 {
        i128::try_from(modulus.checked_sub(canonical)?)
            .ok()?
            .checked_neg()
    } else {
        i128::try_from(canonical).ok()
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "digits and the base are bounded by the caller, and the shift-plus-carry \
              recurrence avoids an overflowing numerator"
)]
fn balanced_limbs(mut centered: i128, limbs: usize, width: u32) -> Option<[i128; 4]> {
    let mut digits = [0; 4];
    let active = digits.get_mut(..limbs)?;
    if limbs == 1 {
        *active.first_mut()? = centered;
        return Some(digits);
    }
    let base = 1i128.checked_shl(width)?;
    for digit in active.iter_mut() {
        *digit = centered.rem_euclid(base);
        if *digit >= base / 2 {
            *digit -= base;
        }
        // `centered - digit` is divisible by the base but can equal 2^127
        // for a valid fp128 coefficient. The arithmetic shift gives the
        // floor quotient; a negative digit adds the carry for the selected
        // balanced representative.
        centered = (centered >> width) + i128::from(*digit < 0);
    }
    (centered == 0).then_some(digits)
}

fn split_limbs<F: MetalField + CanonicalEncoding, const K: usize, const D: usize>(
    rings: &[CyclotomicRing<F, D>],
    modulus: u128,
    limbs: usize,
    width: u32,
    primes: &[NttPrime<i32>; K],
) -> Result<Vec<i32>, AkitaMetalError> {
    let overflow = || AkitaMetalError::Shape("an entry does not fit its limbs".into());
    let mut words = Vec::with_capacity(
        checked::product([rings.len(), limbs, K, D]).ok_or_else(|| shape_overflow(limbs))?,
    );
    let mut entry_limbs = vec![[0i128; D]; limbs];
    for ring in rings {
        for (index, coefficient) in ring.coefficients().iter().enumerate() {
            let canonical = coefficient.to_u128_checked().ok_or_else(overflow)?;
            let centered = centered_value(canonical, modulus).ok_or_else(overflow)?;
            let digits = balanced_limbs(centered, limbs, width).ok_or_else(overflow)?;
            for (limb, digit) in entry_limbs.iter_mut().zip(digits) {
                let slot = limb.get_mut(index).ok_or_else(overflow)?;
                *slot = digit;
            }
        }
        for limb in &entry_limbs {
            for prime in primes {
                let modulus = i128::from(prime.p);
                for &value in limb {
                    let residue =
                        i32::try_from(value.rem_euclid(modulus)).map_err(|_| overflow())?;
                    words.push(prime.from_canonical(residue).raw());
                }
            }
        }
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::{balanced_limbs, centered_value, field_modulus, limb_bits};
    use jolt_field::Prime128OffsetA7F7;

    fn recombine_wide(digits: &[i128], base: u128) -> Option<(bool, u128)> {
        digits
            .iter()
            .rev()
            .try_fold((false, 0u128), |(negative, magnitude), &digit| {
                let magnitude = magnitude.checked_mul(base)?;
                let digit_negative = digit < 0;
                let digit_magnitude = digit.unsigned_abs();
                if negative == digit_negative {
                    Some((negative, magnitude.checked_add(digit_magnitude)?))
                } else if magnitude >= digit_magnitude {
                    Some((negative, magnitude.checked_sub(digit_magnitude)?))
                } else {
                    Some((digit_negative, digit_magnitude.checked_sub(magnitude)?))
                }
            })
    }

    #[test]
    fn fp128_centering_boundary_limbs_recombine_exactly() {
        let modulus = field_modulus::<Prime128OffsetA7F7>().expect("fp128 modulus");
        for limbs in [2, 3, 4] {
            let width = limb_bits(modulus, limbs).expect("limb width");
            let base = 1u128.checked_shl(width).expect("limb base");
            for canonical in [modulus / 2 - 1, modulus / 2, modulus / 2 + 1] {
                let centered = centered_value(canonical, modulus).expect("centered value");
                let digits = balanced_limbs(centered, limbs, width).expect("balanced limbs");
                let recombined = recombine_wide(digits.get(..limbs).expect("limb prefix"), base)
                    .expect("wide-enough independent recombination");
                assert_eq!(
                    recombined,
                    (centered < 0, centered.unsigned_abs()),
                    "limbs={limbs} canonical={canonical}",
                );
            }
        }
    }

    #[test]
    fn global_matrix_and_plane_addresses_can_cross_u32() {
        let degree = 64u64;
        let block = 1u64 << 26;
        let plane = block.checked_mul(degree).expect("plane word address");
        assert_eq!(plane, 1u64 << 32);

        let row = 1u64 << 20;
        let cols = 64u64;
        let limbs = 4u64;
        let primes = 6u64;
        let matrix = row
            .checked_mul(cols)
            .and_then(|value| value.checked_mul(limbs))
            .and_then(|value| value.checked_mul(primes))
            .and_then(|value| value.checked_mul(degree))
            .expect("matrix word address");
        assert!(matrix > u64::from(u32::MAX));
    }
}
