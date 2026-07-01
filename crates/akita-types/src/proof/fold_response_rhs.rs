//! Shared shifted-fold response RHS helpers (§shifted-fold-response).
//!
//! Canonical implementations for the public consistency and A-row corrections
//! `y_0 = η⟨a, G_commit 1⟩` and `y_A = η A 1`, plus partial relation-claim
//! evaluation that skips zero `B_inner` rows.
//!
//! **Flat setup geometry:** `A` is always a role-local prefix view of
//! `shared_matrix` from flat ring index `0` with shape
//! `(a_key.row_len(), inner_width)`. It is not concatenated after D or B in the
//! flat vector, and M-row indices such as [`LevelParams::a_start`] are unrelated
//! to flat offsets.

use crate::AkitaExpandedSetup;
use crate::Schedule;
use crate::{
    gadget_row_scalars, LevelParams, MRowLayout, PublicMatrixSeed, RingMultiplierOpeningPoint,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{
    cfg_into_iter, AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase,
};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

macro_rules! fold_a_ones_lookup {
    (32, $self:expr, $lp:expr) => {{
        let key = FoldAOnesGeometryKey::new($lp);
        $self
            .rows_32
            .get(&key)
            .map(Arc::clone)
            .ok_or_else(|| missing_a_ones($lp, 32))
    }};
    (64, $self:expr, $lp:expr) => {{
        let key = FoldAOnesGeometryKey::new($lp);
        $self
            .rows_64
            .get(&key)
            .map(Arc::clone)
            .ok_or_else(|| missing_a_ones($lp, 64))
    }};
    (128, $self:expr, $lp:expr) => {{
        let key = FoldAOnesGeometryKey::new($lp);
        $self
            .rows_128
            .get(&key)
            .map(Arc::clone)
            .ok_or_else(|| missing_a_ones($lp, 128))
    }};
    (256, $self:expr, $lp:expr) => {{
        let key = FoldAOnesGeometryKey::new($lp);
        $self
            .rows_256
            .get(&key)
            .map(Arc::clone)
            .ok_or_else(|| missing_a_ones($lp, 256))
    }};
    ($d:literal, $self:expr, $lp:expr) => {
        compile_error!(concat!(
            "unsupported ring dimension for fold A-ones table: ",
            stringify!($d)
        ))
    };
}

fn rings_to_coeff_rows<F: FieldCore, const D: usize>(rows: &[CyclotomicRing<F, D>]) -> Vec<Vec<F>> {
    rows.iter().map(|row| row.coefficients().to_vec()).collect()
}

fn rehydrate_ring<F: FieldCore, const D: usize>(
    coeffs: Vec<F>,
) -> Result<CyclotomicRing<F, D>, AkitaError> {
    let arr: [F; D] = coeffs.as_slice().try_into().map_err(|_| {
        AkitaError::InvalidSetup(format!(
            "fold A-ones row coefficient length mismatch: expected {D}, got {}",
            coeffs.len()
        ))
    })?;
    Ok(CyclotomicRing::from_coefficients(arr))
}

macro_rules! fold_a_ones_ensure {
    (32, $self:expr, $setup:expr, $lp:expr) => {{
        let key = FoldAOnesGeometryKey::new($lp);
        if !$self.rows_32.contains_key(&key) {
            let rows = Arc::new(a_ones_from_setup::<F, 32>($setup, $lp)?);
            $self.rows_32.insert(key, rows);
        }
        Ok(())
    }};
    (64, $self:expr, $setup:expr, $lp:expr) => {{
        let key = FoldAOnesGeometryKey::new($lp);
        if !$self.rows_64.contains_key(&key) {
            let rows = Arc::new(a_ones_from_setup::<F, 64>($setup, $lp)?);
            $self.rows_64.insert(key, rows);
        }
        Ok(())
    }};
    (128, $self:expr, $setup:expr, $lp:expr) => {{
        let key = FoldAOnesGeometryKey::new($lp);
        if !$self.rows_128.contains_key(&key) {
            let rows = Arc::new(a_ones_from_setup::<F, 128>($setup, $lp)?);
            $self.rows_128.insert(key, rows);
        }
        Ok(())
    }};
    (256, $self:expr, $setup:expr, $lp:expr) => {{
        let key = FoldAOnesGeometryKey::new($lp);
        if !$self.rows_256.contains_key(&key) {
            let rows = Arc::new(a_ones_from_setup::<F, 256>($setup, $lp)?);
            $self.rows_256.insert(key, rows);
        }
        Ok(())
    }};
    ($d:literal, $self:expr, $setup:expr, $lp:expr) => {
        compile_error!(concat!(
            "unsupported ring dimension for fold A-ones table: ",
            stringify!($d)
        ))
    };
}

/// Geometry key for setup-fixed `A · 1` rows (no η).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FoldAOnesGeometryKey {
    a_row_len: usize,
    inner_width: usize,
}

impl FoldAOnesGeometryKey {
    /// Build a geometry key from one fold level's Ajtai layout.
    #[must_use]
    pub fn new(lp: &LevelParams) -> Self {
        Self {
            a_row_len: lp.a_key.row_len(),
            inner_width: lp.inner_width(),
        }
    }

    fn a_row_len(self) -> usize {
        self.a_row_len
    }

    fn inner_width(self) -> usize {
        self.inner_width
    }
}

/// Schedule-warm table of unscaled `A · 1` rows keyed by setup seed and level geometry.
#[derive(Clone, Debug)]
pub struct FoldAOnesTable<F: FieldCore> {
    setup_seed: PublicMatrixSeed,
    rows_32: HashMap<FoldAOnesGeometryKey, Arc<Vec<CyclotomicRing<F, 32>>>>,
    rows_64: HashMap<FoldAOnesGeometryKey, Arc<Vec<CyclotomicRing<F, 64>>>>,
    rows_128: HashMap<FoldAOnesGeometryKey, Arc<Vec<CyclotomicRing<F, 128>>>>,
    rows_256: HashMap<FoldAOnesGeometryKey, Arc<Vec<CyclotomicRing<F, 256>>>>,
}

impl<F: FieldCore> Default for FoldAOnesTable<F> {
    fn default() -> Self {
        Self {
            setup_seed: [0u8; 32],
            rows_32: HashMap::new(),
            rows_64: HashMap::new(),
            rows_128: HashMap::new(),
            rows_256: HashMap::new(),
        }
    }
}

impl<F: FieldCore> FoldAOnesTable<F> {
    /// Public matrix seed bound into this table.
    #[must_use]
    pub fn setup_seed(&self) -> &PublicMatrixSeed {
        &self.setup_seed
    }

    /// Empty table bound to `setup_seed` (no warmed fold geometries).
    #[must_use]
    pub fn empty_for_seed(setup_seed: PublicMatrixSeed) -> Self {
        Self {
            setup_seed,
            ..Self::default()
        }
    }

    /// Precompute `A · 1` for every fold level across `schedules`.
    ///
    /// # Errors
    ///
    /// Returns an error if any level's setup matrix view is malformed.
    pub fn warm_schedule_union<'a>(
        setup: &AkitaExpandedSetup<F>,
        schedules: impl IntoIterator<Item = &'a Schedule>,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        let mut table = Self::empty_for_seed(setup.seed().public_matrix_seed);
        table.warm_schedules(setup, schedules)?;
        Ok(table)
    }

    /// Extend this table with every fold level in `schedules`.
    ///
    /// # Errors
    ///
    /// Returns an error if any level's setup matrix view is malformed.
    pub fn warm_schedules<'a>(
        &mut self,
        setup: &AkitaExpandedSetup<F>,
        schedules: impl IntoIterator<Item = &'a Schedule>,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        for schedule in schedules {
            self.warm_level_params(setup, schedule.fold_steps().map(|fold| &fold.params))?;
        }
        Ok(())
    }

    /// Extend this table with explicit fold [`LevelParams`].
    ///
    /// # Errors
    ///
    /// Returns an error if any level's setup matrix view is malformed.
    pub fn warm_level_params<'a>(
        &mut self,
        setup: &AkitaExpandedSetup<F>,
        level_params: impl IntoIterator<Item = &'a LevelParams>,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        for lp in level_params {
            let ring_dim = lp.ring_dimension;
            crate::dispatch_ring_dim_result!(ring_dim, |D| self.ensure_a_ones::<D>(setup, lp))?;
        }
        Ok(())
    }

    /// Precompute `A · 1` for every fold level in one schedule.
    ///
    /// # Errors
    ///
    /// Returns an error if any level's setup matrix view is malformed.
    pub fn build_from_schedule(
        setup: &AkitaExpandedSetup<F>,
        schedule: &Schedule,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        Self::warm_schedule_union(setup, std::iter::once(schedule))
    }

    /// Unscaled `A · 1` rows as raw coefficient vectors (runtime ring dimension).
    pub fn a_ones_runtime(&self, lp: &LevelParams) -> Result<Vec<Vec<F>>, AkitaError> {
        let ring_dim = lp.ring_dimension;
        crate::dispatch_ring_dim_result!(ring_dim, |D| {
            Ok(match D {
                32 => rings_to_coeff_rows::<F, 32>(self.a_ones_at_32(lp)?.as_ref()),
                64 => rings_to_coeff_rows::<F, 64>(self.a_ones_at_64(lp)?.as_ref()),
                128 => rings_to_coeff_rows::<F, 128>(self.a_ones_at_128(lp)?.as_ref()),
                256 => rings_to_coeff_rows::<F, 256>(self.a_ones_at_256(lp)?.as_ref()),
                _ => unreachable!("dispatch_ring_dim_result already filtered unsupported D"),
            })
        })
    }

    /// η-scaled `A · 1` rows as raw coefficient vectors (runtime ring dimension).
    pub fn a_shift_rows_runtime(
        &self,
        lp: &LevelParams,
        committed_shift: u128,
    ) -> Result<Vec<Vec<F>>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        let ring_dim = lp.ring_dimension;
        crate::dispatch_ring_dim_result!(ring_dim, |D| {
            Ok(match D {
                32 => rings_to_coeff_rows::<F, 32>(&self.a_shift_rows_at_32(lp, committed_shift)?),
                64 => rings_to_coeff_rows::<F, 64>(&self.a_shift_rows_at_64(lp, committed_shift)?),
                128 => {
                    rings_to_coeff_rows::<F, 128>(&self.a_shift_rows_at_128(lp, committed_shift)?)
                }
                256 => {
                    rings_to_coeff_rows::<F, 256>(&self.a_shift_rows_at_256(lp, committed_shift)?)
                }
                _ => unreachable!("dispatch_ring_dim_result already filtered unsupported D"),
            })
        })
    }

    /// Lookup `A · 1` and apply the public digit shift `η`.
    pub fn a_shift_rows<const D: usize>(
        &self,
        lp: &LevelParams,
        committed_shift: u128,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        if lp.ring_dimension != D {
            return Err(ring_dimension_mismatch(lp, D));
        }
        self.a_shift_rows_runtime(lp, committed_shift)?
            .into_iter()
            .map(|coeffs| rehydrate_ring::<F, D>(coeffs))
            .collect()
    }

    fn ensure_a_ones<const D: usize>(
        &mut self,
        setup: &AkitaExpandedSetup<F>,
        lp: &LevelParams,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        if setup.seed().public_matrix_seed != self.setup_seed {
            return Err(AkitaError::InvalidSetup(
                "fold A-ones table was built for a different setup seed".to_string(),
            ));
        }
        if lp.ring_dimension != D {
            return Err(ring_dimension_mismatch(lp, D));
        }
        match D {
            32 => fold_a_ones_ensure!(32, self, setup, lp),
            64 => fold_a_ones_ensure!(64, self, setup, lp),
            128 => fold_a_ones_ensure!(128, self, setup, lp),
            256 => fold_a_ones_ensure!(256, self, setup, lp),
            _ => Err(AkitaError::InvalidInput(format!(
                "unsupported ring dimension for fold A-ones table: {D}"
            ))),
        }
    }

    pub fn a_ones_at_32(
        &self,
        lp: &LevelParams,
    ) -> Result<Arc<Vec<CyclotomicRing<F, 32>>>, AkitaError> {
        if lp.ring_dimension != 32 {
            return Err(ring_dimension_mismatch(lp, 32));
        }
        fold_a_ones_lookup!(32, self, lp)
    }

    pub fn a_ones_at_64(
        &self,
        lp: &LevelParams,
    ) -> Result<Arc<Vec<CyclotomicRing<F, 64>>>, AkitaError> {
        if lp.ring_dimension != 64 {
            return Err(ring_dimension_mismatch(lp, 64));
        }
        fold_a_ones_lookup!(64, self, lp)
    }

    pub fn a_ones_at_128(
        &self,
        lp: &LevelParams,
    ) -> Result<Arc<Vec<CyclotomicRing<F, 128>>>, AkitaError> {
        if lp.ring_dimension != 128 {
            return Err(ring_dimension_mismatch(lp, 128));
        }
        fold_a_ones_lookup!(128, self, lp)
    }

    pub fn a_ones_at_256(
        &self,
        lp: &LevelParams,
    ) -> Result<Arc<Vec<CyclotomicRing<F, 256>>>, AkitaError> {
        if lp.ring_dimension != 256 {
            return Err(ring_dimension_mismatch(lp, 256));
        }
        fold_a_ones_lookup!(256, self, lp)
    }

    pub fn a_shift_rows_at_32(
        &self,
        lp: &LevelParams,
        committed_shift: u128,
    ) -> Result<Vec<CyclotomicRing<F, 32>>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        let ones = self.a_ones_at_32(lp)?;
        Ok(scale_a_shift_rows(ones.as_ref(), committed_shift))
    }

    pub fn a_shift_rows_at_64(
        &self,
        lp: &LevelParams,
        committed_shift: u128,
    ) -> Result<Vec<CyclotomicRing<F, 64>>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        let ones = self.a_ones_at_64(lp)?;
        Ok(scale_a_shift_rows(ones.as_ref(), committed_shift))
    }

    pub fn a_shift_rows_at_128(
        &self,
        lp: &LevelParams,
        committed_shift: u128,
    ) -> Result<Vec<CyclotomicRing<F, 128>>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        let ones = self.a_ones_at_128(lp)?;
        Ok(scale_a_shift_rows(ones.as_ref(), committed_shift))
    }

    pub fn a_shift_rows_at_256(
        &self,
        lp: &LevelParams,
        committed_shift: u128,
    ) -> Result<Vec<CyclotomicRing<F, 256>>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
    {
        let ones = self.a_ones_at_256(lp)?;
        Ok(scale_a_shift_rows(ones.as_ref(), committed_shift))
    }
}

fn serialize_fold_a_ones_bucket<F, const D: usize, W: Write>(
    map: &HashMap<FoldAOnesGeometryKey, Arc<Vec<CyclotomicRing<F, D>>>>,
    writer: &mut W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    F: FieldCore + AkitaSerialize,
{
    let mut keys: Vec<_> = map.keys().copied().collect();
    keys.sort_by_key(|key| (key.a_row_len(), key.inner_width()));
    (keys.len() as u64).serialize_with_mode(&mut *writer, compress)?;
    for key in keys {
        let rows = map.get(&key).expect("sorted keys are present");
        (key.a_row_len() as u64).serialize_with_mode(&mut *writer, compress)?;
        (key.inner_width() as u64).serialize_with_mode(&mut *writer, compress)?;
        (rows.len() as u64).serialize_with_mode(&mut *writer, compress)?;
        for row in rows.iter() {
            row.serialize_with_mode(&mut *writer, compress)?;
        }
    }
    Ok(())
}

fn deserialize_fold_a_ones_bucket<F, const D: usize, R: Read>(
    reader: &mut R,
    compress: Compress,
    validate: Validate,
) -> Result<HashMap<FoldAOnesGeometryKey, Arc<Vec<CyclotomicRing<F, D>>>>, SerializationError>
where
    F: FieldCore + AkitaDeserialize<Context = ()> + Valid,
{
    let entry_count = usize::deserialize_with_mode(&mut *reader, compress, validate, &())?;
    let mut map = HashMap::with_capacity(entry_count);
    for _ in 0..entry_count {
        let a_row_len = usize::deserialize_with_mode(&mut *reader, compress, validate, &())?;
        let inner_width = usize::deserialize_with_mode(&mut *reader, compress, validate, &())?;
        let row_count = usize::deserialize_with_mode(&mut *reader, compress, validate, &())?;
        if row_count != a_row_len {
            return Err(SerializationError::InvalidData(format!(
                "fold A-ones row count {row_count} does not match a_row_len {a_row_len}"
            )));
        }
        let mut rows = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            rows.push(CyclotomicRing::<F, D>::deserialize_with_mode(
                &mut *reader,
                compress,
                validate,
                &(),
            )?);
        }
        let key = FoldAOnesGeometryKey {
            a_row_len,
            inner_width,
        };
        map.insert(key, Arc::new(rows));
    }
    Ok(map)
}

fn serialized_fold_a_ones_bucket_size<F, const D: usize>(
    map: &HashMap<FoldAOnesGeometryKey, Arc<Vec<CyclotomicRing<F, D>>>>,
    compress: Compress,
) -> usize
where
    F: FieldCore + AkitaSerialize,
{
    let mut keys: Vec<_> = map.keys().copied().collect();
    keys.sort_by_key(|key| (key.a_row_len(), key.inner_width()));
    let mut size = (keys.len() as u64).serialized_size(compress);
    for key in keys {
        let rows = map.get(&key).expect("sorted keys are present");
        size += (key.a_row_len() as u64).serialized_size(compress);
        size += (key.inner_width() as u64).serialized_size(compress);
        size += (rows.len() as u64).serialized_size(compress);
        for row in rows.iter() {
            size += row.serialized_size(compress);
        }
    }
    size
}

impl<F> AkitaSerialize for FoldAOnesTable<F>
where
    F: FieldCore + AkitaSerialize,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        writer
            .write_all(&self.setup_seed)
            .map_err(SerializationError::from)?;
        let mut bucket_count = 0u64;
        if !self.rows_32.is_empty() {
            bucket_count += 1;
        }
        if !self.rows_64.is_empty() {
            bucket_count += 1;
        }
        if !self.rows_128.is_empty() {
            bucket_count += 1;
        }
        if !self.rows_256.is_empty() {
            bucket_count += 1;
        }
        bucket_count.serialize_with_mode(&mut writer, compress)?;
        if !self.rows_32.is_empty() {
            32u64.serialize_with_mode(&mut writer, compress)?;
            serialize_fold_a_ones_bucket::<F, 32, _>(&self.rows_32, &mut writer, compress)?;
        }
        if !self.rows_64.is_empty() {
            64u64.serialize_with_mode(&mut writer, compress)?;
            serialize_fold_a_ones_bucket::<F, 64, _>(&self.rows_64, &mut writer, compress)?;
        }
        if !self.rows_128.is_empty() {
            128u64.serialize_with_mode(&mut writer, compress)?;
            serialize_fold_a_ones_bucket::<F, 128, _>(&self.rows_128, &mut writer, compress)?;
        }
        if !self.rows_256.is_empty() {
            256u64.serialize_with_mode(&mut writer, compress)?;
            serialize_fold_a_ones_bucket::<F, 256, _>(&self.rows_256, &mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.setup_seed.len();
        let mut bucket_count = 0u64;
        if !self.rows_32.is_empty() {
            bucket_count += 1;
        }
        if !self.rows_64.is_empty() {
            bucket_count += 1;
        }
        if !self.rows_128.is_empty() {
            bucket_count += 1;
        }
        if !self.rows_256.is_empty() {
            bucket_count += 1;
        }
        size += bucket_count.serialized_size(compress);
        if !self.rows_32.is_empty() {
            size += 32u64.serialized_size(compress);
            size += serialized_fold_a_ones_bucket_size::<F, 32>(&self.rows_32, compress);
        }
        if !self.rows_64.is_empty() {
            size += 64u64.serialized_size(compress);
            size += serialized_fold_a_ones_bucket_size::<F, 64>(&self.rows_64, compress);
        }
        if !self.rows_128.is_empty() {
            size += 128u64.serialized_size(compress);
            size += serialized_fold_a_ones_bucket_size::<F, 128>(&self.rows_128, compress);
        }
        if !self.rows_256.is_empty() {
            size += 256u64.serialized_size(compress);
            size += serialized_fold_a_ones_bucket_size::<F, 256>(&self.rows_256, compress);
        }
        size
    }
}

impl<F> AkitaDeserialize for FoldAOnesTable<F>
where
    F: FieldCore + AkitaDeserialize<Context = ()> + Valid,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut setup_seed = [0u8; 32];
        reader
            .read_exact(&mut setup_seed)
            .map_err(SerializationError::from)?;
        let bucket_count = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut table = Self::empty_for_seed(setup_seed);
        for _ in 0..bucket_count {
            let ring_dim = u64::deserialize_with_mode(&mut reader, compress, validate, &())?;
            match ring_dim {
                32 => {
                    table.rows_32 = deserialize_fold_a_ones_bucket::<F, 32, _>(
                        &mut reader,
                        compress,
                        validate,
                    )?;
                }
                64 => {
                    table.rows_64 = deserialize_fold_a_ones_bucket::<F, 64, _>(
                        &mut reader,
                        compress,
                        validate,
                    )?;
                }
                128 => {
                    table.rows_128 = deserialize_fold_a_ones_bucket::<F, 128, _>(
                        &mut reader,
                        compress,
                        validate,
                    )?;
                }
                256 => {
                    table.rows_256 = deserialize_fold_a_ones_bucket::<F, 256, _>(
                        &mut reader,
                        compress,
                        validate,
                    )?;
                }
                other => {
                    return Err(SerializationError::InvalidData(format!(
                        "unsupported fold A-ones bucket ring dimension {other}"
                    )));
                }
            }
        }
        if matches!(validate, Validate::Yes) {
            table.check()?;
        }
        Ok(table)
    }
}

impl<F> Valid for FoldAOnesTable<F>
where
    F: FieldCore + Valid,
{
    fn check(&self) -> Result<(), SerializationError> {
        for rows in self.rows_32.values() {
            for row in rows.iter() {
                row.check()?;
            }
        }
        for rows in self.rows_64.values() {
            for row in rows.iter() {
                row.check()?;
            }
        }
        for rows in self.rows_128.values() {
            for row in rows.iter() {
                row.check()?;
            }
        }
        for rows in self.rows_256.values() {
            for row in rows.iter() {
                row.check()?;
            }
        }
        Ok(())
    }
}

fn missing_a_ones(lp: &LevelParams, d: usize) -> AkitaError {
    AkitaError::InvalidSetup(format!(
        "fold A-ones table missing entry for ring_dim={d} n_a={} inner_width={}",
        lp.a_key.row_len(),
        lp.inner_width()
    ))
}

fn ring_dimension_mismatch(lp: &LevelParams, expected: usize) -> AkitaError {
    AkitaError::InvalidInput(format!(
        "fold A-ones lookup ring_dimension mismatch: lp has {} but caller requested {expected}",
        lp.ring_dimension
    ))
}

/// Cyclotomic ring with all negacyclic coefficients set to one.
#[must_use]
pub fn all_coeffs_one_ring<F: FieldCore, const D: usize>() -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients([F::one(); D])
}

fn accumulate_a_ones_row<F, const D: usize>(
    setup_row: &[CyclotomicRing<F, D>],
) -> CyclotomicRing<F, D>
where
    F: FieldCore + CanonicalField,
{
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in setup_row {
        entry.mul_accumulate_all_ones_into(&mut acc);
    }
    acc
}

/// Unscaled `A · 1` from the public setup matrix prefix view.
///
/// Uses `ring_view(n_a, inner_width)` from flat index `0`; see module docs.
///
/// # Errors
///
/// Returns an error if the setup matrix view is malformed.
pub fn a_ones_from_setup<F, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    lp: &LevelParams,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let a_view = setup
        .shared_matrix()
        .ring_view::<D>(lp.a_key.row_len(), lp.inner_width())?;
    let num_rows = a_view.num_rows();
    let rows: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..num_rows)
        .map(|row_idx| a_view.row(row_idx).map(accumulate_a_ones_row))
        .collect::<Result<Vec<_>, AkitaError>>()?;
    Ok(rows)
}

/// Scale each row of `A · 1` by the public committed-digit shift `η`.
#[must_use]
pub fn scale_a_shift_rows<F, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    committed_shift: u128,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let shift = F::from_u128(committed_shift);
    rows.iter().map(|row| row.scale(&shift)).collect()
}

/// Compute `η A 1` from the public setup matrix without a prover NTT pass.
///
/// # Errors
///
/// Returns an error if the setup matrix view is malformed.
pub fn fold_a_shift_rows_from_setup<F, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    lp: &LevelParams,
    committed_shift: u128,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    Ok(scale_a_shift_rows(
        &a_ones_from_setup::<F, D>(setup, lp)?,
        committed_shift,
    ))
}

/// Public consistency-row correction `η⟨a, G_commit 1⟩` at the ring opening.
///
/// # Errors
///
/// Returns an error if the ring-multiplier opening layout is inconsistent.
pub fn fold_shift_consistency_row<F, const D: usize>(
    ring_multiplier_point: &RingMultiplierOpeningPoint<F, D>,
    block_len: usize,
    depth_commit: usize,
    log_basis: u32,
    committed_shift: u128,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    if ring_multiplier_point.a_len() < block_len {
        return Err(AkitaError::InvalidProof);
    }
    let shift = F::from_u128(committed_shift);
    let gadget_sum = gadget_row_scalars::<F>(depth_commit, log_basis)
        .into_iter()
        .fold(F::zero(), |acc, g| acc + g);
    let ones = all_coeffs_one_ring::<F, D>().scale(&gadget_sum);
    let mut acc = CyclotomicRing::<F, D>::zero();
    for block_idx in 0..block_len {
        if let Some(scalar) = ring_multiplier_point.a_constant_coeff(block_idx) {
            acc += ones.scale(&scalar);
        } else {
            let a_rings = ring_multiplier_point
                .a_rings()
                .ok_or(AkitaError::InvalidProof)?;
            let multiplier = a_rings.get(block_idx).ok_or(AkitaError::InvalidProof)?;
            acc += *multiplier * ones;
        }
    }
    Ok(acc.scale(&shift))
}

const FOLD_RELATION_NUM_PUBLIC_OUTPUTS: usize = 0;

fn fold_active_row_starts(
    lp: &LevelParams,
    layout: MRowLayout,
    num_commitments: usize,
) -> Result<(usize, usize, usize), AkitaError> {
    Ok((
        lp.d_start(FOLD_RELATION_NUM_PUBLIC_OUTPUTS)?,
        lp.f_start(FOLD_RELATION_NUM_PUBLIC_OUTPUTS, layout)?,
        lp.a_start(num_commitments, FOLD_RELATION_NUM_PUBLIC_OUTPUTS, layout)?,
    ))
}

/// Evaluate the stage-2 relation claim over active M rows for one fold level.
///
/// Row offsets are derived from [`LevelParams`] helpers. Skips zero `B_inner`
/// rows while matching the full `y` layout used by [`super::generate_y`].
///
/// # Errors
///
/// Returns an error if commitment row grouping is malformed or the equality
/// table would overflow.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "relation_claim_from_fold_active_rows_for_level")]
pub fn relation_claim_from_fold_active_rows_for_level<F, const D: usize>(
    lp: &LevelParams,
    layout: MRowLayout,
    num_commitments: usize,
    tau1: &[F],
    alpha: F,
    consistency_row: &CyclotomicRing<F, D>,
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    a_rows: &[CyclotomicRing<F, D>],
) -> Result<F, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_commitment_grouping(lp.effective_commit_rows(), commitment_rows)?;
    let (d_start, f_start, a_start) = fold_active_row_starts(lp, layout, num_commitments)?;
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let alpha_pows = scalar_powers(alpha, D);
    let mut acc = F::zero();
    if !eq_tau1.is_empty() {
        acc += eq_tau1[0] * eval_ring_at_pows(consistency_row, &alpha_pows);
    }
    for (offset, row) in v.iter().enumerate() {
        let row_idx = d_start + offset;
        if row_idx >= eq_tau1.len() {
            break;
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(row, &alpha_pows);
    }
    for (offset, row) in commitment_rows.iter().enumerate() {
        let row_idx = f_start + offset;
        if row_idx >= eq_tau1.len() {
            break;
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(row, &alpha_pows);
    }
    for (offset, row) in a_rows.iter().enumerate() {
        let row_idx = a_start + offset;
        if row_idx >= eq_tau1.len() {
            break;
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(row, &alpha_pows);
    }
    Ok(acc)
}

/// Extension-field variant of [`relation_claim_from_fold_active_rows_for_level`].
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    skip_all,
    name = "relation_claim_from_fold_active_rows_for_level_extension"
)]
pub fn relation_claim_from_fold_active_rows_for_level_extension<F, E, const D: usize>(
    lp: &LevelParams,
    layout: MRowLayout,
    num_commitments: usize,
    tau1: &[E],
    alpha: E,
    consistency_row: &CyclotomicRing<F, D>,
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    a_rows: &[CyclotomicRing<F, D>],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBase<F>,
{
    validate_commitment_grouping(lp.effective_commit_rows(), commitment_rows)?;
    let (d_start, f_start, a_start) = fold_active_row_starts(lp, layout, num_commitments)?;
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let alpha_pows = scalar_powers(alpha, D);
    let mut acc = E::zero();
    if !eq_tau1.is_empty() {
        acc += eq_tau1[0] * eval_ring_at_pows(consistency_row, &alpha_pows);
    }
    for (offset, row) in v.iter().enumerate() {
        let row_idx = d_start + offset;
        if row_idx >= eq_tau1.len() {
            break;
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(row, &alpha_pows);
    }
    for (offset, row) in commitment_rows.iter().enumerate() {
        let row_idx = f_start + offset;
        if row_idx >= eq_tau1.len() {
            break;
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(row, &alpha_pows);
    }
    for (offset, row) in a_rows.iter().enumerate() {
        let row_idx = a_start + offset;
        if row_idx >= eq_tau1.len() {
            break;
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(row, &alpha_pows);
    }
    Ok(acc)
}

fn validate_commitment_grouping(
    commit_rows_per_group: usize,
    commitment_rows: &[impl Sized],
) -> Result<(), AkitaError> {
    if commit_rows_per_group == 0
        || commitment_rows.is_empty()
        || !commitment_rows.len().is_multiple_of(commit_rows_per_group)
    {
        return Err(AkitaError::InvalidSize {
            expected: commit_rows_per_group,
            actual: commitment_rows.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_y;
    use crate::proof::relation::{relation_claim_from_rows, relation_claim_from_rows_extension};
    use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr};

    type F = Fp32<251>;
    type E = FpExt2<F, NegOneNr>;
    const D: usize = 4;

    fn sample_setup() -> AkitaExpandedSetup<F> {
        use crate::derive_public_matrix_flat;
        use crate::sample_public_matrix_seed;
        let setup_seed = crate::AkitaSetupSeed {
            max_num_vars: 4,
            max_num_batched_polys: 1,
            gen_ring_dim: D,
            max_setup_len: 24,
            public_matrix_seed: sample_public_matrix_seed(),
        };
        let matrix = derive_public_matrix_flat::<F, D>(
            setup_seed.max_setup_len,
            &setup_seed.public_matrix_seed,
        );
        AkitaExpandedSetup::from_verified_parts(setup_seed, matrix).expect("setup")
    }

    fn sample_lp() -> LevelParams {
        use crate::SisModulusFamily;
        use akita_challenges::SparseChallengeConfig;
        LevelParams::params_only(
            SisModulusFamily::Q32,
            D,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        )
        .with_decomp(1, 1, 1, 1, 0)
        .expect("lp")
    }

    fn sample_schedule(lp: LevelParams) -> Schedule {
        use crate::Step;
        Schedule {
            steps: vec![Step::Fold(crate::FoldStep {
                params: lp,
                current_w_len: 64,
                next_w_len: 32,
                level_bytes: 0,
            })],
            total_bytes: 0,
        }
    }

    #[test]
    fn a_shift_rows_match_unscaled_times_eta() {
        let setup = sample_setup();
        let lp = sample_lp();
        let shift = 3u128;
        let direct = fold_a_shift_rows_from_setup::<F, D>(&setup, &lp, shift).expect("direct");
        let scaled = scale_a_shift_rows(
            &a_ones_from_setup::<F, D>(&setup, &lp).expect("ones"),
            shift,
        );
        assert_eq!(direct, scaled);
    }

    #[test]
    fn schedule_warm_table_matches_direct_matvec() {
        const D_WARM: usize = 32;
        use crate::derive_public_matrix_flat;
        use crate::sample_public_matrix_seed;
        use crate::SisModulusFamily;
        use akita_challenges::SparseChallengeConfig;
        let setup_seed = crate::AkitaSetupSeed {
            max_num_vars: 4,
            max_num_batched_polys: 1,
            gen_ring_dim: D_WARM,
            max_setup_len: 24,
            public_matrix_seed: sample_public_matrix_seed(),
        };
        let matrix = derive_public_matrix_flat::<F, D_WARM>(
            setup_seed.max_setup_len,
            &setup_seed.public_matrix_seed,
        );
        let setup = AkitaExpandedSetup::from_verified_parts(setup_seed, matrix).expect("setup");
        let lp = LevelParams::params_only(
            SisModulusFamily::Q32,
            D_WARM,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        )
        .with_decomp(1, 1, 1, 1, 0)
        .expect("lp");
        let schedule = sample_schedule(lp.clone());
        let table = FoldAOnesTable::build_from_schedule(&setup, &schedule).expect("table");
        let warmed_runtime = table.a_ones_runtime(&lp).expect("warmed runtime");
        let direct = a_ones_from_setup::<F, D_WARM>(&setup, &lp).expect("direct");
        let direct_coeffs = rings_to_coeff_rows::<F, D_WARM>(&direct);
        assert_eq!(warmed_runtime, direct_coeffs);
        let shift = 5u128;
        assert_eq!(
            table.a_shift_rows::<D_WARM>(&lp, shift).expect("shifted"),
            fold_a_shift_rows_from_setup::<F, D_WARM>(&setup, &lp, shift).expect("direct shifted")
        );
    }

    #[test]
    fn a_ones_respects_role_local_width_not_d_stride() {
        let setup = sample_setup();
        let lp = sample_lp();
        let n_a = lp.a_key.row_len();
        let w = lp.inner_width();
        let d_width = w + 2;
        let a_view = setup
            .shared_matrix()
            .ring_view::<D>(n_a, w)
            .expect("a view");
        let d_view = setup
            .shared_matrix()
            .ring_view::<D>(n_a, d_width)
            .expect("d view");
        let ones = all_coeffs_one_ring::<F, D>();
        let mut expected = CyclotomicRing::<F, D>::zero();
        for entry in a_view.row(0).expect("row") {
            expected += *entry * ones;
        }
        let wrong = {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for entry in d_view.row(0).expect("row") {
                acc += *entry * ones;
            }
            acc
        };
        let got = a_ones_from_setup::<F, D>(&setup, &lp).expect("ones")[0];
        assert_eq!(got, expected);
        assert_ne!(got, wrong);
    }

    #[test]
    fn partial_relation_claim_matches_full_y() {
        let lp = sample_lp();
        let consistency = CyclotomicRing::<F, D>::from_coefficients([
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let v = [CyclotomicRing::from_coefficients([
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ])];
        let commitment_rows = [CyclotomicRing::from_coefficients([
            F::from_u64(9),
            F::from_u64(10),
            F::from_u64(11),
            F::from_u64(12),
        ])];
        let a_rows = [CyclotomicRing::from_coefficients([
            F::from_u64(13),
            F::from_u64(14),
            F::from_u64(15),
            F::from_u64(16),
        ])];
        let y = generate_y::<F, D>(
            consistency,
            &v,
            &commitment_rows,
            &a_rows,
            v.len(),
            commitment_rows.len(),
            lp.b_inner_rows_per_group(),
            a_rows.len(),
        )
        .expect("y");
        let tau1 = [
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
        ];
        let alpha = F::from_u64(23);
        let partial = relation_claim_from_fold_active_rows_for_level::<F, D>(
            &lp,
            MRowLayout::WithDBlock,
            1,
            &tau1,
            alpha,
            &consistency,
            &v,
            &commitment_rows,
            &a_rows,
        )
        .expect("partial");
        let full = relation_claim_from_rows::<F, D>(&tau1, alpha, &y).expect("full");
        assert_eq!(partial, full);

        let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
        let partial_ext = relation_claim_from_fold_active_rows_for_level_extension::<F, E, D>(
            &lp,
            MRowLayout::WithDBlock,
            1,
            &lifted_tau1,
            E::lift_base(alpha),
            &consistency,
            &v,
            &commitment_rows,
            &a_rows,
        )
        .expect("partial ext");
        let full_ext =
            relation_claim_from_rows_extension::<F, E, D>(&lifted_tau1, E::lift_base(alpha), &y)
                .expect("full ext");
        assert_eq!(partial_ext, full_ext);
    }
}
