//! Shared setup data shapes for Akita prover and verifier APIs.

use crate::{AkitaScheduleLookupKey, FlatMatrix, Schedule, SetupArtifactDigests};
use akita_field::FieldCore;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

/// Public seed used to derive commitment matrices.
pub type PublicMatrixSeed = [u8; 32];

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupSeed {
    /// Maximum supported variable count.
    pub max_num_vars: usize,
    /// Maximum number of batched polynomials supported by setup.
    pub max_num_batched_polys: usize,
    /// Maximum number of distinct opening points.
    ///
    /// Together with `max_num_batched_polys` this bounds the outer/D matrix
    /// widths the setup can serve; a multi-point batched opening that exceeds
    /// this bound would otherwise silently read past the shared matrix prefix.
    pub max_num_points: usize,
    /// Global row stride for the flat NTT cache.
    pub max_stride: usize,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
}

/// Expanded setup stage containing a single shared coefficient-form matrix.
///
/// All role matrices (A, B, D) are row/column prefixes of this shared vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: AkitaSetupSeed,
    /// Shared 1D flat backing vector.
    pub shared_matrix: FlatMatrix<F>,
    /// Cached descriptor digests for the setup artifacts.
    pub descriptor_digests: SetupArtifactDigests,
}

/// Verifier setup artifact derived from prover setup.
///
/// `schedule_cache` memoises planner schedules keyed by their public
/// [`AkitaScheduleLookupKey`] so that repeated `batched_verify` calls don't
/// each re-run the planner DP search. Cloning a verifier setup shares the
/// cache via the inner `Arc`. The cache is excluded from serialisation,
/// `Valid` checks, and equality so that two setups derived from the same
/// expanded data compare equal whether or not they've serviced any verifies.
///
/// Without this cache, a config that doesn't ship a generated schedule
/// table (e.g. the opt-in `D64OneHotTensor` preset, or any test wrapper)
/// pays the full planner DP cost — ~12.5 ms at NV=20 — on every
/// `batched_verify` call, dominating the verifier wall-clock and giving
/// the false impression that the tensor sampler is intrinsically slower.
#[derive(Debug, Clone)]
pub struct AkitaVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Schedule cache shared across `batched_verify` calls.
    pub schedule_cache: Arc<Mutex<HashMap<AkitaScheduleLookupKey, Schedule>>>,
}

impl<F> AkitaExpandedSetup<F>
where
    F: FieldCore + AkitaSerialize,
{
    /// Build an expanded setup and compute its cached descriptor digests.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the setup seed or shared matrix cannot
    /// be canonically serialized for descriptor hashing.
    pub fn from_parts(
        seed: AkitaSetupSeed,
        shared_matrix: FlatMatrix<F>,
    ) -> Result<Self, SerializationError> {
        let descriptor_digests = SetupArtifactDigests::from_parts(&seed, &shared_matrix)?;
        Ok(Self {
            seed,
            shared_matrix,
            descriptor_digests,
        })
    }
}

impl<F: FieldCore> AkitaVerifierSetup<F> {
    /// Build a verifier setup wrapping the given expanded setup with a
    /// fresh empty schedule cache.
    #[must_use]
    pub fn new(expanded: Arc<AkitaExpandedSetup<F>>) -> Self {
        Self {
            expanded,
            schedule_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return a clone of the cached schedule for `key`, or `None` on miss.
    ///
    /// A poisoned cache lock is treated as a miss because the cache is purely
    /// an optimisation and the caller is expected to recompute on `None`.
    #[must_use]
    pub fn cached_schedule(&self, key: AkitaScheduleLookupKey) -> Option<Schedule> {
        let guard = self.schedule_cache.lock().ok()?;
        guard.get(&key).cloned()
    }

    /// Insert `schedule` into the cache under `key`, replacing any existing
    /// entry. A poisoned cache lock is a no-op for the same reason.
    pub fn store_schedule(&self, key: AkitaScheduleLookupKey, schedule: Schedule) {
        if let Ok(mut guard) = self.schedule_cache.lock() {
            guard.insert(key, schedule);
        }
    }
}

impl<F: FieldCore> PartialEq for AkitaVerifierSetup<F> {
    fn eq(&self, other: &Self) -> bool {
        self.expanded == other.expanded
    }
}

impl<F: FieldCore> Eq for AkitaVerifierSetup<F> {}

impl Valid for AkitaSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_stride == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_stride must be non-zero".to_string(),
            ));
        }
        if self.max_num_batched_polys == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_num_batched_polys must be at least 1".to_string(),
            ));
        }
        if self.max_num_points == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_num_points must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for AkitaSetupSeed {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_num_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_batched_polys
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_points
            .serialize_with_mode(&mut writer, compress)?;
        self.max_stride.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.max_num_points.serialized_size(compress)
            + self.max_stride.serialized_size(compress)
            + 32
    }
}

impl AkitaDeserialize for AkitaSetupSeed {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let max_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_num_batched_polys =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_num_points = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_stride = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
            max_stride,
            public_matrix_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid + AkitaSerialize> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.seed.check()?;
        self.shared_matrix.check()?;
        self.descriptor_digests
            .check_parts(&self.seed, &self.shared_matrix)?;
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaExpandedSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.seed.serialize_with_mode(&mut writer, compress)?;
        self.shared_matrix
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.seed.serialized_size(compress) + self.shared_matrix.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaExpandedSetup<F>
where
    F: AkitaSerialize,
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let seed = AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let shared_matrix =
            FlatMatrix::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self::from_parts(seed, shared_matrix)?;
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid + AkitaSerialize> Valid for AkitaVerifierSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaVerifierSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.expanded.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaVerifierSetup<F>
where
    F: AkitaSerialize,
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let expanded = Arc::new(AkitaExpandedSetup::deserialize_with_mode(
            reader,
            compress,
            validate,
            &(),
        )?);
        Ok(Self::new(expanded))
    }
}

#[cfg(test)]
mod schedule_cache_tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn dummy_setup() -> AkitaVerifierSetup<F> {
        AkitaVerifierSetup::new(Arc::new(AkitaExpandedSetup {
            seed: AkitaSetupSeed {
                max_num_vars: 4,
                max_num_batched_polys: 1,
                max_num_points: 1,
                max_stride: 1,
                public_matrix_seed: [0u8; 32],
            },
            shared_matrix: FlatMatrix::from_flat_data(vec![F::default()], 1),
        }))
    }

    #[test]
    fn schedule_cache_hits_after_store() {
        let setup = dummy_setup();
        let key = AkitaScheduleLookupKey::singleton(8);
        assert!(setup.cached_schedule(key).is_none());
        let sched = Schedule {
            steps: Vec::new(),
            total_bytes: 42,
        };
        setup.store_schedule(key, sched.clone());
        let got = setup.cached_schedule(key).expect("hit after store");
        assert_eq!(got.total_bytes, 42);
        assert!(got.steps.is_empty());
    }

    #[test]
    fn schedule_cache_keyed_by_lookup_key() {
        let setup = dummy_setup();
        let singleton = AkitaScheduleLookupKey::singleton(8);
        let batched = AkitaScheduleLookupKey::new(8, 2, 2, 1);
        setup.store_schedule(
            singleton,
            Schedule {
                steps: Vec::new(),
                total_bytes: 1,
            },
        );
        setup.store_schedule(
            batched,
            Schedule {
                steps: Vec::new(),
                total_bytes: 2,
            },
        );
        assert_eq!(setup.cached_schedule(singleton).unwrap().total_bytes, 1);
        assert_eq!(setup.cached_schedule(batched).unwrap().total_bytes, 2);
    }

    #[test]
    fn schedule_cache_shared_via_clone() {
        let setup_a = dummy_setup();
        let setup_b = setup_a.clone();
        let key = AkitaScheduleLookupKey::singleton(8);
        setup_a.store_schedule(
            key,
            Schedule {
                steps: Vec::new(),
                total_bytes: 77,
            },
        );
        let got = setup_b.cached_schedule(key).expect("clone shares cache");
        assert_eq!(got.total_bytes, 77);
    }

    #[test]
    fn schedule_cache_excluded_from_equality() {
        let setup_a = dummy_setup();
        let setup_b = AkitaVerifierSetup::new(Arc::clone(&setup_a.expanded));
        setup_a.store_schedule(
            AkitaScheduleLookupKey::singleton(8),
            Schedule {
                steps: Vec::new(),
                total_bytes: 99,
            },
        );
        assert_eq!(setup_a, setup_b);
    }
}
