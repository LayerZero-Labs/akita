//! Device setup matrices for the commitment stages, prepared once per shape.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use akita_algebra::{CrtCapacity, CrtNttParamSet, CyclotomicRing, NttPrime};
use akita_error::checked;
use akita_types::FlatMatrix;
use jolt_metal::runtime::DeviceBuffer;

use super::shared::Shared;
use super::CommitmentField;
use crate::error::AkitaMetalError;
use crate::library::AkitaMetal;
use crate::matvec::{field_modulus, plan_matvec, DeviceNttMatrix, MatvecPlan};
use crate::onehot::DeviceFlatMatrix;

/// A prepared device matrix with its CRT profile width erased.
pub(super) trait DeviceMatrix<F>: Send + Sync {
    fn mat_vec_i8(
        &self,
        metal: &AkitaMetal,
        planes: &DeviceBuffer<i8>,
        log_basis: u32,
        out: &mut DeviceBuffer<F>,
    ) -> Result<Duration, AkitaMetalError>;

    fn mat_vec_i16(
        &self,
        metal: &AkitaMetal,
        planes: &DeviceBuffer<i16>,
        log_basis: u32,
        out: &mut DeviceBuffer<F>,
    ) -> Result<Duration, AkitaMetalError>;
}

impl<F: CommitmentField, const K: usize, const D: usize> DeviceMatrix<F>
    for Shared<DeviceNttMatrix<K, D>>
{
    fn mat_vec_i8(
        &self,
        metal: &AkitaMetal,
        planes: &DeviceBuffer<i8>,
        log_basis: u32,
        out: &mut DeviceBuffer<F>,
    ) -> Result<Duration, AkitaMetalError> {
        self.get().mat_vec(metal, planes, log_basis, out)
    }

    fn mat_vec_i16(
        &self,
        metal: &AkitaMetal,
        planes: &DeviceBuffer<i16>,
        log_basis: u32,
        out: &mut DeviceBuffer<F>,
    ) -> Result<Duration, AkitaMetalError> {
        self.get().mat_vec(metal, planes, log_basis, out)
    }
}

/// What a prepared matrix is: the `rows x cols` prefix of the shared setup
/// matrix viewed at one ring degree, split for one matvec plan.
///
/// The inner (A) and outer (B) matrices are both prefixes of the same shared
/// setup matrix, so the shape alone identifies the contents; a stage role
/// would only duplicate equal matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MatrixKey {
    ring_degree: usize,
    rows: usize,
    cols: usize,
    primes: usize,
    limbs: usize,
}

/// A flat setup-matrix prefix on the device.
pub(super) type SharedFlat<F> = Arc<Shared<DeviceFlatMatrix<F>>>;

/// Device copies of the setup matrix, prepared on first use and kept for the
/// provider's lifetime.
pub(super) struct MatrixCache<F> {
    ntt: Mutex<HashMap<MatrixKey, Arc<dyn DeviceMatrix<F>>>>,
    /// The longest flat prefix uploaded for the one-hot kernel, with its
    /// length in field elements.
    flat: Mutex<Option<(usize, SharedFlat<F>)>>,
}

impl<F: CommitmentField> MatrixCache<F> {
    pub(super) fn new() -> Self {
        Self {
            ntt: Mutex::new(HashMap::new()),
            flat: Mutex::new(None),
        }
    }

    /// Prepared matrices held on the device.
    pub(super) fn len(&self) -> usize {
        self.ntt
            .lock()
            .map_or(0, |cache| cache.len())
            .saturating_add(
                self.flat
                    .lock()
                    .map_or(0, |flat| usize::from(flat.is_some())),
            )
    }

    /// The `rows x cols` NTT matrix at ring degree `D` for `log_basis`
    /// digits, split by [`plan_matvec`] and prepared on the device from the
    /// setup's coefficient form.
    pub(super) fn ntt<const D: usize>(
        &self,
        metal: &AkitaMetal,
        setup: &FlatMatrix<F>,
        rows: usize,
        cols: usize,
        log_basis: u32,
    ) -> Result<Arc<dyn DeviceMatrix<F>>, AkitaMetalError> {
        let primes = F::crt_primes();
        let plan = plan_matvec(field_modulus::<F>()?, &primes, cols, D, log_basis);
        let key = MatrixKey {
            ring_degree: D,
            rows,
            cols,
            primes: plan.primes,
            limbs: plan.limbs,
        };
        let mut cache = self
            .ntt
            .lock()
            .map_err(|_| AkitaMetalError::Shape("the device matrix cache is poisoned".into()))?;
        if let Some(matrix) = cache.get(&key) {
            return Ok(matrix.clone());
        }
        let rings = setup
            .ring_view::<D>(rows, cols)
            .map_err(|error| AkitaMetalError::Shape(error.to_string()))?
            .as_slice();
        let matrix = prepare::<F, D>(metal, &primes, rings, rows, cols, plan)?;
        cache.insert(key, matrix.clone());
        Ok(matrix)
    }

    /// The setup matrix's first `len` field elements on the device.
    pub(super) fn flat(
        &self,
        metal: &AkitaMetal,
        setup: &FlatMatrix<F>,
        len: usize,
    ) -> Result<SharedFlat<F>, AkitaMetalError> {
        let mut flat = self
            .flat
            .lock()
            .map_err(|_| AkitaMetalError::Shape("the device matrix cache is poisoned".into()))?;
        if let Some((cached, matrix)) = flat.as_ref() {
            if *cached >= len {
                return Ok(matrix.clone());
            }
        }
        let prefix = setup.as_field_slice().get(..len).ok_or_else(|| {
            AkitaMetalError::Shape(format!(
                "the setup has no {len}-element matrix prefix ({} elements)",
                setup.num_field_elements()
            ))
        })?;
        let matrix = Arc::new(Shared::new(DeviceFlatMatrix::new(metal, prefix)?));
        *flat = Some((len, matrix.clone()));
        Ok(matrix)
    }
}

/// Whether one column of degree `D` and `log_basis` digits fits the field's
/// whole CRT profile, the matvec's widest segment: otherwise no plan is exact.
pub(super) fn single_column_fits<F: CommitmentField>(ring_degree: usize, log_basis: u32) -> bool {
    let Ok(modulus) = field_modulus::<F>() else {
        return false;
    };
    let Some(digit_bound) = log_basis
        .checked_sub(1)
        .and_then(|shift| 1u64.checked_shl(shift))
    else {
        return false;
    };
    CrtCapacity::from_prime_moduli(
        F::crt_primes()
            .iter()
            .map(|prime| u128::from(prime.p.unsigned_abs())),
    )
    .max_safe_width_for_modulus(ring_degree, modulus, digit_bound)
    .is_some_and(|width| width > 0)
}

fn prepare<F: CommitmentField, const D: usize>(
    metal: &AkitaMetal,
    profile: &[NttPrime<i32>],
    rings: &[CyclotomicRing<F, D>],
    rows: usize,
    cols: usize,
    plan: MatvecPlan,
) -> Result<Arc<dyn DeviceMatrix<F>>, AkitaMetalError> {
    macro_rules! with_primes {
        ($($k:literal)+) => {
            match plan.primes {
                $($k => build::<F, $k, D>(metal, profile, rings, rows, cols, plan.limbs),)+
                other => Err(AkitaMetalError::Shape(format!(
                    "{other} CRT primes have no reconstruction kernel"
                ))),
            }
        };
    }
    with_primes!(1 2 3 4 5 6)
}

fn build<F: CommitmentField, const K: usize, const D: usize>(
    metal: &AkitaMetal,
    profile: &[NttPrime<i32>],
    rings: &[CyclotomicRing<F, D>],
    rows: usize,
    cols: usize,
    limbs: usize,
) -> Result<Arc<dyn DeviceMatrix<F>>, AkitaMetalError> {
    let primes: [NttPrime<i32>; K] = profile
        .get(..K)
        .and_then(|prefix| prefix.try_into().ok())
        .ok_or_else(|| {
            AkitaMetalError::Shape(format!(
                "the field's CRT profile has {} primes, not {K}",
                profile.len()
            ))
        })?;
    checked::product([rows, cols]).ok_or_else(|| {
        AkitaMetalError::Shape(format!("a {rows} x {cols} matrix overflows usize"))
    })?;
    let params = CrtNttParamSet::<i32, K, D>::new(primes);
    Ok(Arc::new(Shared::new(DeviceNttMatrix::<K, D>::from_rings(
        metal, &params, rings, rows, cols, limbs,
    )?)))
}
