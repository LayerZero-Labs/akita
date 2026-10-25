//! Immutable sharing of equal setup-prefix inner ring commitments.
use akita_error::AkitaError;
use akita_serialization::{AkitaSerialize, Compress};
use akita_types::RingVec;
use jolt_field::{CanonicalEncoding, Field};
use std::{
    borrow::Cow,
    collections::{hash_map::DefaultHasher, HashMap},
    hash::Hasher,
    sync::{Arc, Mutex, Weak},
};

// The hash only selects candidates. Equality of every coefficient establishes
// sharing, including on hash collisions. Weak entries do not retain failed builds.
pub(crate) struct InnerRowPool<F> {
    entries: Mutex<HashMap<u64, Vec<Weak<[F]>>>>,
}

impl<F> Default for InnerRowPool<F> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl<F: Field + CanonicalEncoding> InnerRowPool<F> {
    fn intern(&self, row: &[F]) -> Result<Arc<[F]>, AkitaError> {
        let mut hasher = DefaultHasher::new();
        let mut bytes = vec![0; F::NUM_BYTES];
        hasher.write_usize(row.len());
        for coefficient in row {
            coefficient.to_bytes_le(&mut bytes);
            hasher.write(&bytes);
        }
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| AkitaError::Internal("setup-prefix inner cache lock poisoned".into()))?;
        let candidates = entries.entry(hasher.finish()).or_default();
        candidates.retain(|entry| entry.strong_count() != 0);
        for candidate in candidates.iter().filter_map(Weak::upgrade) {
            if candidate.as_ref() == row {
                return Ok(candidate);
            }
        }
        let shared: Arc<[F]> = Arc::from(row);
        candidates.push(Arc::downgrade(&shared));
        Ok(shared)
    }
}

#[derive(Clone)]
pub(crate) enum InnerRows<F: Field> {
    Owned(Vec<RingVec<F>>),
    Shared {
        ring_dim: usize,
        rows: Vec<Vec<Arc<[F]>>>,
    },
}

impl<F: Field> InnerRows<F> {
    pub(crate) fn materialize(&self) -> Result<Cow<'_, [RingVec<F>]>, AkitaError> {
        match self {
            Self::Owned(rows) => Ok(Cow::Borrowed(rows)),
            Self::Shared { ring_dim, rows } => rows
                .iter()
                .map(|blocks| {
                    let coefficients = blocks
                        .iter()
                        .flat_map(|block| block.iter().copied())
                        .collect();
                    RingVec::from_coeffs_with_ring_dim(coefficients, *ring_dim)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Cow::Owned),
        }
    }

    pub(crate) fn serialized_rows_size(&self, compress: Compress) -> usize
    where
        F: AkitaSerialize,
    {
        match self {
            Self::Owned(rows) => rows
                .iter()
                .map(|row| {
                    row.coeff_len().serialized_size(compress) + row.serialized_size(compress)
                })
                .sum(),
            Self::Shared { rows, .. } => rows
                .iter()
                .map(|blocks| {
                    let len = blocks.iter().map(|block| block.len()).sum::<usize>();
                    len.serialized_size(compress)
                        + blocks
                            .iter()
                            .flat_map(|block| block.iter())
                            .map(|coefficient| coefficient.serialized_size(compress))
                            .sum::<usize>()
                })
                .sum(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Owned(rows) => rows.len(),
            Self::Shared { rows, .. } => rows.len(),
        }
    }

    pub(crate) fn share(
        &mut self,
        pool: &InnerRowPool<F>,
        ring_dim: usize,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalEncoding,
    {
        if ring_dim == 0 {
            return Err(AkitaError::InvalidInput(
                "malformed setup-prefix inner rows".into(),
            ));
        }
        let shared = match self {
            Self::Owned(rows) => {
                if rows.iter().any(|row| {
                    row.ring_dim() != ring_dim || !row.coeff_len().is_multiple_of(ring_dim)
                }) {
                    return Err(AkitaError::InvalidInput(
                        "malformed setup-prefix inner rows".into(),
                    ));
                }
                rows.iter()
                    .map(|row| {
                        row.coeffs()
                            .chunks_exact(ring_dim)
                            .map(|ring| pool.intern(ring))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            Self::Shared {
                ring_dim: stored,
                rows,
            } => {
                if *stored != ring_dim {
                    return Err(AkitaError::InvalidInput(
                        "setup-prefix inner ring mismatch".into(),
                    ));
                }
                // Imported, backend-validated hints may originate in another pool.
                rows.iter()
                    .map(|blocks| {
                        blocks
                            .iter()
                            .map(|ring| pool.intern(ring))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        *self = Self::Shared {
            ring_dim,
            rows: shared,
        };
        Ok(())
    }
}

impl<F: Field> PartialEq for InnerRows<F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Owned(a), Self::Owned(b)) => a == b,
            (
                Self::Shared {
                    ring_dim: a,
                    rows: ar,
                },
                Self::Shared {
                    ring_dim: b,
                    rows: br,
                },
            ) => a == b && ar == br,
            (Self::Owned(owned), Self::Shared { ring_dim, rows })
            | (Self::Shared { ring_dim, rows }, Self::Owned(owned)) => {
                owned.len() == rows.len()
                    && owned.iter().zip(rows).all(|(owned, blocks)| {
                        owned.ring_dim() == *ring_dim
                            && owned
                                .coeffs()
                                .iter()
                                .eq(blocks.iter().flat_map(|block| block.iter()))
                    })
            }
        }
    }
}
impl<F: Field> Eq for InnerRows<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fp32, Ring};
    type F = Fp32<251>;

    #[test]
    fn concurrent_interning_shares_rows_only_within_its_pool() {
        let pool = InnerRowPool::default();
        let rows = std::thread::scope(|scope| {
            let jobs = (0..8)
                .map(|_| scope.spawn(|| pool.intern(&[F::from_u64(9)]).unwrap()))
                .collect::<Vec<_>>();
            jobs.into_iter()
                .map(|job| job.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(rows.iter().all(|row| Arc::ptr_eq(row, &rows[0])));
        let other = InnerRowPool::default().intern(&[F::from_u64(9)]).unwrap();
        assert!(!Arc::ptr_eq(&other, &rows[0]));
    }

    #[test]
    fn overlapping_rows_share_storage_and_keep_their_order() {
        let pool = InnerRowPool::default();
        let row = |values: &[u64]| {
            RingVec::from_coeffs_with_ring_dim(values.iter().copied().map(F::from_u64).collect(), 2)
                .unwrap()
        };
        let mut small = InnerRows::Owned(vec![row(&[1, 2, 3, 4])]);
        let mut large = InnerRows::Owned(vec![row(&[1, 2, 3, 4, 5, 6])]);
        let expected = large.clone();
        small.share(&pool, 2).unwrap();
        large.share(&pool, 2).unwrap();
        assert!(large == expected);
        assert!(large.materialize().unwrap().as_ref() == expected.materialize().unwrap().as_ref());
        let InnerRows::Shared { rows: a, .. } = &small else {
            panic!("shared rows");
        };
        let InnerRows::Shared { rows: b, .. } = &large else {
            panic!("shared rows");
        };
        assert!(Arc::ptr_eq(&a[0][0], &b[0][0]));
        assert!(Arc::ptr_eq(&a[0][1], &b[0][1]));
        assert!(!Arc::ptr_eq(&b[0][1], &b[0][2]));
        let clone = large.clone();
        drop(large);
        assert!(clone == expected);
    }

    #[test]
    fn imported_shared_rows_join_the_destination_pool() {
        let first = InnerRowPool::default();
        let second = InnerRowPool::default();
        let row = || {
            InnerRows::Owned(vec![RingVec::from_coeffs_with_ring_dim(
                vec![F::from_u64(1), F::from_u64(2)],
                2,
            )
            .unwrap()])
        };
        let mut a = row();
        let mut b = row();
        a.share(&first, 2).unwrap();
        b.share(&second, 2).unwrap();
        b.share(&first, 2).unwrap();
        let InnerRows::Shared { rows: a, .. } = a else {
            panic!("shared rows");
        };
        let InnerRows::Shared { rows: b, .. } = b else {
            panic!("shared rows");
        };
        assert!(Arc::ptr_eq(&a[0][0], &b[0][0]));
    }

    #[test]
    fn hash_collision_does_not_share_different_coefficients() {
        let pool = InnerRowPool::default();
        let good = pool.intern(&[F::from_u64(1)]).unwrap();
        let different: Arc<[F]> = Arc::from([F::from_u64(2)]);
        for candidates in pool.entries.lock().unwrap().values_mut() {
            *candidates = vec![Arc::downgrade(&different)];
        }
        let repeated = pool.intern(&[F::from_u64(1)]).unwrap();
        assert_eq!(good, repeated);
        assert!(!Arc::ptr_eq(&different, &repeated));
    }
}
