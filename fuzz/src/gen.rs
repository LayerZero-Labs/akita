//! Valid-by-construction value, table, point, and selector generators.
//!
//! A table is a base pattern (expanded deterministically from a seed) plus a
//! bounded list of explicit edits. The engine therefore mutates individual
//! entries, the pattern, or the seed independently, and a table of `2^20`
//! entries still has a compact encoding.

use crate::input::{Reader, SplitMix64};
use jolt_field::{CanonicalEncoding, ExtField, Field};

pub fn modulus<F: Field + CanonicalEncoding>() -> u128 {
    (-F::one())
        .to_u128_checked()
        .expect("Akita fields fit in u128")
        + 1
}

/// Centered representative in `(-q/2, q/2]`.
pub fn centered<F: Field + CanonicalEncoding>(value: F) -> i128 {
    let q = modulus::<F>();
    let canonical = value.to_u128_checked().expect("Akita fields fit in u128");
    if canonical > q / 2 {
        -((q - canonical) as i128)
    } else {
        canonical as i128
    }
}

pub fn from_signed<F: Field + CanonicalEncoding>(negative: bool, magnitude: u128) -> F {
    let value = F::from_u128_reduced(magnitude);
    if negative {
        -value
    } else {
        value
    }
}

/// Admissible coefficient set of a committed source.
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    /// Every field element.
    Full,
    /// Centered magnitude at most this value.
    Centered(u128),
}

impl Domain {
    pub fn clamp<F: Field + CanonicalEncoding>(self, value: F) -> F {
        match self {
            Domain::Full => value,
            Domain::Centered(bound) => {
                let c = centered(value);
                let magnitude = c.unsigned_abs();
                if magnitude <= bound {
                    value
                } else {
                    from_signed::<F>(c < 0, magnitude % (bound + 1))
                }
            }
        }
    }

    fn max_magnitude<F: Field + CanonicalEncoding>(self) -> u128 {
        match self {
            Domain::Full => modulus::<F>() / 2,
            Domain::Centered(bound) => bound,
        }
    }
}

pub const TOKEN_BYTES: usize = 17;

/// One scalar: a mode byte selecting a boundary family, then 16 payload bytes.
pub fn scalar<F: Field + CanonicalEncoding>(reader: &mut Reader<'_>, domain: Domain) -> F {
    let mode = reader.u8();
    let payload: [u8; 16] = reader.bytes();
    scalar_from(mode, payload, domain)
}

pub fn scalar_from<F: Field + CanonicalEncoding>(mode: u8, payload: [u8; 16], domain: Domain) -> F {
    let raw = u128::from_le_bytes(payload);
    let negative = payload[15] & 0x80 != 0;
    let max = domain.max_magnitude::<F>();
    let bits = 128 - max.leading_zeros();
    let small_delta = u128::from(payload[1] % 4);
    let value = match mode % 16 {
        0 => F::zero(),
        1 => F::one(),
        2 => -F::one(),
        3 => F::from_u128_reduced(raw),
        4 => F::from_i64(i64::from(payload[0] as i8)),
        5 => F::from_i64(i64::from(i16::from_le_bytes([payload[0], payload[1]]))),
        6 => from_signed::<F>(negative, 1u128 << (u32::from(payload[0]) % bits.max(1))),
        7 => from_signed::<F>(
            negative,
            (1u128 << (u32::from(payload[0]) % bits.max(1))).saturating_sub(1),
        ),
        8 => from_signed::<F>(
            negative,
            (1u128 << (u32::from(payload[0]) % bits.max(1))) + 1,
        ),
        9 => from_signed::<F>(negative, max.saturating_sub(small_delta)),
        10 => from_signed::<F>(negative, max),
        11 => F::from_u64(u64::from_le_bytes(payload[..8].try_into().unwrap())),
        12 => F::from_i64(i64::from(i32::from_le_bytes(
            payload[..4].try_into().unwrap(),
        ))),
        // Neighbors of the small-integer fast-path boundaries (i8, i16, i32).
        13 => {
            let edge =
                [127u128, 128, 32767, 32768, (1 << 31) - 1, 1 << 31][usize::from(payload[0] % 6)];
            from_signed::<F>(negative, edge + small_delta)
        }
        14 => F::from_u128_reduced(modulus::<F>() / 2 + small_delta),
        _ => from_signed::<F>(negative, raw % (max + 1)),
    };
    domain.clamp(value)
}

pub fn ext_scalar<F, E>(reader: &mut Reader<'_>) -> E
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let coordinates: Vec<F> = (0..E::DEGREE)
        .map(|_| scalar::<F>(reader, Domain::Full))
        .collect();
    E::from_base_slice(&coordinates)
}

/// Opening point with a per-coordinate choice of Boolean or arbitrary value.
pub fn point<F, E>(reader: &mut Reader<'_>, num_vars: usize) -> Vec<E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let style = reader.u8();
    (0..num_vars)
        .map(|_| {
            let selector = reader.u8();
            let value = ext_scalar::<F, E>(reader);
            let boolean = match style % 3 {
                0 => false,
                1 => true,
                _ => selector & 1 == 1,
            };
            if boolean {
                E::from_u64(u64::from(selector >> 7))
            } else {
                value
            }
        })
        .collect()
}

/// Bulk pattern plus explicit edits over `len` base-field entries.
pub fn table<F: Field + CanonicalEncoding>(
    reader: &mut Reader<'_>,
    len: usize,
    domain: Domain,
) -> Vec<F> {
    let pattern = reader.u8();
    let seed = reader.u64();
    let constant: F = scalar(reader, domain);
    let mut rng = SplitMix64::new(seed);
    let max = domain.max_magnitude::<F>();
    let mut entries: Vec<F> = (0..len)
        .map(|index| {
            let value = match pattern % 14 {
                0 => F::zero(),
                1 => constant,
                2 => {
                    if index % 2 == 0 {
                        constant
                    } else {
                        -constant
                    }
                }
                3 => constant * F::from_u64(index as u64),
                4 => F::from_u128_reduced(rng.next_u128()),
                5 => F::from_i64(i64::from(rng.next_u64() as i8)),
                6 => {
                    let r = rng.next_u64();
                    match r % 4 {
                        0 => from_signed::<F>(r & 4 != 0, max),
                        1 => from_signed::<F>(r & 4 != 0, max.saturating_sub(1)),
                        2 => F::zero(),
                        _ => from_signed::<F>(r & 4 != 0, 1),
                    }
                }
                7 => {
                    if index == 0 {
                        constant
                    } else {
                        F::zero()
                    }
                }
                8 => {
                    let r = rng.next_u64();
                    if r % 16 == 0 {
                        F::from_u128_reduced(rng.next_u128())
                    } else {
                        F::zero()
                    }
                }
                9 => F::from_u128_reduced(1u128 << (index % 127)),
                10 => F::from_i64(i64::from(rng.next_u64() as i16)),
                11 => {
                    let r = rng.next_u128();
                    from_signed::<F>(r & 1 == 1, (r >> 1) % (max + 1))
                }
                12 => F::from_i64(i64::from((index % 256) as u8 as i8)),
                _ => F::from_u64(rng.next_u64()),
            };
            domain.clamp(value)
        })
        .collect();
    let edits = usize::from(reader.u8() % 64);
    for _ in 0..edits {
        if len == 0 {
            break;
        }
        let index = reader.u32() as usize % len;
        entries[index] = scalar(reader, domain);
    }
    entries
}

/// One-hot chunk selections; `None` is an absent (all-zero) chunk and
/// `Some(0)` a committed selection of position zero.
pub fn onehot_indices(
    reader: &mut Reader<'_>,
    num_chunks: usize,
    chunk_size: usize,
) -> Vec<Option<u8>> {
    assert!(chunk_size <= 256 && chunk_size > 0);
    let pattern = reader.u8();
    let seed = reader.u64();
    let mut rng = SplitMix64::new(seed);
    let last = (chunk_size - 1) as u8;
    let mut indices: Vec<Option<u8>> = (0..num_chunks)
        .map(|chunk| match pattern % 8 {
            0 => None,
            1 => Some(0),
            2 => Some(last),
            3 => Some((rng.next_u64() % chunk_size as u64) as u8),
            4 => {
                let r = rng.next_u64();
                (r & 1 == 0).then_some(((r >> 1) % chunk_size as u64) as u8)
            }
            5 => Some((chunk % chunk_size) as u8),
            6 => {
                let r = rng.next_u64();
                (r % 16 == 0).then_some(((r >> 4) % chunk_size as u64) as u8)
            }
            _ => (chunk % 2 == 0).then_some(last),
        })
        .collect();
    let edits = usize::from(reader.u8() % 64);
    for _ in 0..edits {
        if num_chunks == 0 {
            break;
        }
        let chunk = reader.u32() as usize % num_chunks;
        let value = reader.u8();
        indices[chunk] = (value & 1 == 1).then_some((usize::from(value >> 1) % chunk_size) as u8);
    }
    indices
}
