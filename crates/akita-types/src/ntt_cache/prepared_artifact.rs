//! Program-bound prepared verifier NTT cache artifacts.

use super::*;
use akita_params::ScheduleRowDigest;

const MAGIC: [u8; 8] = *b"AKVNTT01";
const TARGET_RISCV64_SCALAR_Q128: u32 = 1;
const HEADER_BYTES: usize = 120;

/// Maximum accepted size of one prepared verifier NTT cache artifact.
pub const PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Public identities bound to one prepared verifier cache artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedVerifierNttCacheBinding {
    /// Digest of the deterministic public matrix seed.
    pub setup_seed_digest: [u8; 32],
    /// Exact generated schedule row that consumes the cache.
    pub schedule_row_digest: ScheduleRowDigest,
    /// Materialized field count in the verifier setup.
    pub setup_field_elements: usize,
}

/// Checked fixed-width metadata from one prepared verifier cache artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedVerifierNttCacheMetadata {
    /// Target ring dimension.
    pub ring_dimension: usize,
    /// Number of scalar Q128 transformed rings.
    pub base_prefix_len: usize,
    /// Number of transformed rings in the i16 exactness tail.
    pub tail_prefix_len: usize,
    /// Active matrix row width used for exact CRT sizing.
    pub width: usize,
    /// Maximum absolute signed right-hand-side coefficient.
    pub rhs_abs_bound: u64,
    /// Public identities bound to this artifact.
    pub binding: PreparedVerifierNttCacheBinding,
}

fn invalid(message: impl Into<String>) -> AkitaError {
    AkitaError::InvalidSetup(message.into())
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], AkitaError> {
    bytes
        .get(
            offset
                ..offset
                    .checked_add(N)
                    .ok_or_else(|| invalid("prepared cache header overflow"))?,
        )
        .ok_or_else(|| invalid("prepared cache header is truncated"))?
        .try_into()
        .map_err(|_| invalid("prepared cache header is truncated"))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, AkitaError> {
    Ok(u32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, AkitaError> {
    Ok(u64::from_le_bytes(read_array(bytes, offset)?))
}

fn checked_usize(value: u64, field: &str) -> Result<usize, AkitaError> {
    usize::try_from(value).map_err(|_| invalid(format!("prepared cache {field} exceeds usize")))
}

/// Parse and validate the fixed-width metadata and total artifact length.
pub fn prepared_verifier_ntt_cache_metadata(
    bytes: &[u8],
) -> Result<PreparedVerifierNttCacheMetadata, AkitaError> {
    if bytes.len() < HEADER_BYTES || bytes.len() > PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES {
        return Err(invalid(
            "prepared cache artifact length is outside its accepted bounds",
        ));
    }
    if read_array::<8>(bytes, 0)? != MAGIC {
        return Err(invalid("prepared cache artifact magic is invalid"));
    }
    if read_u32(bytes, 8)? != TARGET_RISCV64_SCALAR_Q128 {
        return Err(invalid("prepared cache target is unsupported"));
    }
    let ring_dimension = usize::try_from(read_u32(bytes, 12)?)
        .map_err(|_| invalid("prepared cache ring dimension exceeds usize"))?;
    let base_prefix_len = checked_usize(read_u64(bytes, 16)?, "base prefix")?;
    let tail_prefix_len = checked_usize(read_u64(bytes, 24)?, "tail prefix")?;
    let width = checked_usize(read_u64(bytes, 32)?, "width")?;
    let rhs_abs_bound = read_u64(bytes, 40)?;
    let setup_field_elements = checked_usize(read_u64(bytes, 48)?, "setup field count")?;
    let setup_seed_digest = read_array(bytes, 56)?;
    let schedule_row_digest = ScheduleRowDigest::from_bytes(read_array(bytes, 88)?);
    if ring_dimension == 0
        || base_prefix_len == 0
        || width == 0
        || rhs_abs_bound == 0
        || width > base_prefix_len
        || tail_prefix_len > base_prefix_len
    {
        return Err(invalid("prepared cache geometry is invalid"));
    }
    let base_field_elements = base_prefix_len
        .checked_mul(ring_dimension)
        .ok_or_else(|| invalid("prepared cache base field count overflow"))?;
    if base_field_elements > setup_field_elements {
        return Err(invalid("prepared cache exceeds its bound verifier setup"));
    }
    let payload_bytes = base_field_elements
        .checked_mul(Q128_NUM_PRIMES * core::mem::size_of::<i32>())
        .and_then(|base| {
            tail_prefix_len
                .checked_mul(ring_dimension)
                .and_then(|count| count.checked_mul(core::mem::size_of::<i16>()))
                .and_then(|tail| base.checked_add(tail))
        })
        .ok_or_else(|| invalid("prepared cache payload length overflow"))?;
    let expected_len = HEADER_BYTES
        .checked_add(payload_bytes)
        .ok_or_else(|| invalid("prepared cache artifact length overflow"))?;
    if bytes.len() != expected_len {
        return Err(invalid(
            "prepared cache payload length disagrees with its header",
        ));
    }
    Ok(PreparedVerifierNttCacheMetadata {
        ring_dimension,
        base_prefix_len,
        tail_prefix_len,
        width,
        rhs_abs_bound,
        binding: PreparedVerifierNttCacheBinding {
            setup_seed_digest,
            schedule_row_digest,
            setup_field_elements,
        },
    })
}

/// Build the scalar Q128 exact-negacyclic artifact consumed by a RISC V verifier.
pub fn build_riscv64_scalar_q128_cache_artifact<F: Field + CanonicalEncoding, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    width: usize,
    rhs_abs_bound: u64,
    binding: PreparedVerifierNttCacheBinding,
) -> Result<Vec<u8>, AkitaError> {
    let ProtocolCrtNttParams::Q128(params) = select_crt_ntt_params::<F, D>()? else {
        return Err(invalid(
            "RISC V scalar Q128 cache requires a Q128 protocol field",
        ));
    };
    let needs_tail =
        required_profile_for_params::<F, _, Q128_NUM_PRIMES, D>(&params, width, rhs_abs_bound)
            .ok_or_else(|| exact_capacity_error::<D>(width, rhs_abs_bound))?;
    let tail_prefix_len = usize::from(needs_tail) * matrix.as_slice().len();
    let prepared = prepare_exact_ntt_cache(
        matrix,
        Some(tail_prefix_len),
        exact::ExactCachePlan::Q128 {
            params: Box::new(params),
            needs_tail,
        },
    )?;
    encode_riscv64_scalar_q128_cache(
        &prepared,
        PreparedVerifierNttCacheMetadata {
            ring_dimension: D,
            base_prefix_len: matrix.as_slice().len(),
            tail_prefix_len,
            width,
            rhs_abs_bound,
            binding,
        },
    )
}

fn encode_riscv64_scalar_q128_cache<const D: usize>(
    prepared: &PreparedNttCache<D>,
    metadata: PreparedVerifierNttCacheMetadata,
) -> Result<Vec<u8>, AkitaError> {
    let PreparedNttCacheRepr::Q128 {
        neg: Some(neg),
        cyc: None,
        tail,
        exact: true,
        ..
    } = &prepared.0
    else {
        return Err(invalid(
            "prepared cache is not scalar Q128 exact negacyclic data",
        ));
    };
    let tail_rings = tail
        .as_ref()
        .map(|tail| &tail.negacyclic[..])
        .unwrap_or(&[]);
    if metadata.ring_dimension != D
        || neg.len() != metadata.base_prefix_len
        || tail_rings.len() != metadata.tail_prefix_len
    {
        return Err(invalid("prepared cache data disagrees with its metadata"));
    }
    let capacity = HEADER_BYTES
        .checked_add(prepared.cache_bytes())
        .ok_or_else(|| invalid("prepared cache encoded length overflow"))?;
    if capacity > PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES {
        return Err(invalid("prepared cache exceeds the artifact byte limit"));
    }
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&TARGET_RISCV64_SCALAR_Q128.to_le_bytes());
    bytes.extend_from_slice(
        &u32::try_from(D)
            .map_err(|_| invalid("ring dimension exceeds u32"))?
            .to_le_bytes(),
    );
    for value in [
        metadata.base_prefix_len,
        metadata.tail_prefix_len,
        metadata.width,
    ] {
        bytes.extend_from_slice(
            &u64::try_from(value)
                .map_err(|_| invalid("cache geometry exceeds u64"))?
                .to_le_bytes(),
        );
    }
    bytes.extend_from_slice(&metadata.rhs_abs_bound.to_le_bytes());
    bytes.extend_from_slice(
        &u64::try_from(metadata.binding.setup_field_elements)
            .map_err(|_| invalid("setup field count exceeds u64"))?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&metadata.binding.setup_seed_digest);
    bytes.extend_from_slice(metadata.binding.schedule_row_digest.as_bytes());
    for ring in neg.iter() {
        for limb in &ring.limbs {
            for coefficient in limb {
                bytes.extend_from_slice(&coefficient.raw().to_le_bytes());
            }
        }
    }
    for ring in tail_rings {
        for coefficient in &ring.limbs[0] {
            bytes.extend_from_slice(&coefficient.raw().to_le_bytes());
        }
    }
    if bytes.len() != capacity {
        return Err(invalid(
            "prepared cache encoder produced an inconsistent length",
        ));
    }
    if prepared_verifier_ntt_cache_metadata(&bytes)? != metadata {
        return Err(invalid(
            "prepared cache encoder produced inconsistent metadata",
        ));
    }
    Ok(bytes)
}

/// Body layout of a RISC-V scalar Q128 cache artifact, checked against the
/// verifier's expectations: the negacyclic residues and the optional i16 tail
/// are stored in their in-memory forms, `MontCoeff` being a transparent
/// `i32`/`i16` and `CyclotomicCrtNtt` a transparent array of them.
struct ScalarQ128CacheLayout<'a, const D: usize> {
    metadata: PreparedVerifierNttCacheMetadata,
    params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    needs_tail: bool,
    base_body: &'a [u8],
    tail_body: &'a [u8],
}

fn scalar_q128_cache_layout<F: Field + CanonicalEncoding, const D: usize>(
    bytes: &[u8],
    expected_binding: PreparedVerifierNttCacheBinding,
) -> Result<ScalarQ128CacheLayout<'_, D>, AkitaError> {
    // The body is the little-endian in-memory image of the residue rows.
    if cfg!(target_endian = "big") {
        return Err(invalid(
            "RISC V scalar Q128 cache requires a little-endian target",
        ));
    }
    let metadata = prepared_verifier_ntt_cache_metadata(bytes)?;
    if metadata.ring_dimension != D || metadata.binding != expected_binding {
        return Err(invalid(
            "prepared cache identity does not match the verifier input",
        ));
    }
    let ProtocolCrtNttParams::Q128(params) = select_crt_ntt_params::<F, D>()? else {
        return Err(invalid(
            "RISC V scalar Q128 cache requires a Q128 protocol field",
        ));
    };
    let needs_tail = required_profile_for_params::<F, _, Q128_NUM_PRIMES, D>(
        &params,
        metadata.width,
        metadata.rhs_abs_bound,
    )
    .ok_or_else(|| exact_capacity_error::<D>(metadata.width, metadata.rhs_abs_bound))?;
    if metadata.tail_prefix_len != usize::from(needs_tail) * metadata.base_prefix_len {
        return Err(invalid(
            "prepared cache tail does not match scalar exactness sizing",
        ));
    }
    // Residues are read from fixed-width chunks of a pre-sliced body: the
    // per-coefficient bounds checks and cursor arithmetic dominated this
    // decode on a RISC-V guest verifier.
    let base_bytes = metadata
        .base_prefix_len
        .checked_mul(Q128_NUM_PRIMES * D * 4)
        .ok_or_else(|| invalid("prepared cache cursor overflow"))?;
    let tail_bytes = metadata
        .tail_prefix_len
        .checked_mul(D * 2)
        .ok_or_else(|| invalid("prepared cache cursor overflow"))?;
    let base_end = HEADER_BYTES
        .checked_add(base_bytes)
        .ok_or_else(|| invalid("prepared cache cursor overflow"))?;
    let cursor = base_end
        .checked_add(tail_bytes)
        .ok_or_else(|| invalid("prepared cache cursor overflow"))?;
    let base_body = bytes
        .get(HEADER_BYTES..base_end)
        .ok_or_else(|| invalid("prepared cache body is truncated"))?;
    let tail_body = bytes
        .get(base_end..cursor)
        .ok_or_else(|| invalid("prepared cache body is truncated"))?;
    if cursor != bytes.len() {
        return Err(invalid("prepared cache decoder left trailing bytes"));
    }
    Ok(ScalarQ128CacheLayout {
        metadata,
        params,
        needs_tail,
        base_body,
        tail_body,
    })
}

fn assemble_scalar_q128_cache<const D: usize>(
    params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    neg: PreparedRows<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>,
    tail_rings: Option<PreparedRows<CyclotomicCrtNtt<i16, 1, D>>>,
) -> Result<PreparedNttCache<D>, AkitaError> {
    let tail_params = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
    let tail = tail_rings.map(|negacyclic| PreparedI16Tail {
        negacyclic,
        params: I16TailParams::new(params.clone(), tail_params),
    });
    let prepared = PreparedNttCacheRepr::Q128 {
        neg: Some(neg),
        cyc: None,
        params,
        tail,
        exact: true,
    };
    prepared.validate()?;
    Ok(PreparedNttCache(prepared))
}

/// Decode a scalar Q128 artifact whose bytes have trusted provenance.
///
/// This checks the header, the expected setup and schedule identities, the
/// exact payload length, and every residue range. It cannot prove that the
/// transformed payload was derived from the named setup seed. Callers must
/// bind the bytes to trusted setup provisioning or to the verifier program
/// identity.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] for any malformed or mismatched artifact.
pub fn decode_riscv64_scalar_q128_cache<F: Field + CanonicalEncoding, const D: usize>(
    bytes: &[u8],
    expected_binding: PreparedVerifierNttCacheBinding,
) -> Result<(PreparedVerifierNttCacheMetadata, PreparedNttCache<D>), AkitaError> {
    const {
        assert!(
            core::mem::size_of::<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>()
                == Q128_NUM_PRIMES * D * 4
        );
        assert!(core::mem::size_of::<CyclotomicCrtNtt<i16, 1, D>>() == D * 2);
    }
    let layout = scalar_q128_cache_layout::<F, D>(bytes, expected_binding)?;
    let metadata = layout.metadata;
    let mut neg: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>> =
        Vec::with_capacity(metadata.base_prefix_len);
    // SAFETY: `base_body.len() == base_prefix_len * size_of::<CyclotomicCrtNtt<..>>()`
    // by the layout above; every bit pattern is a valid `i32` residue
    // candidate, and the range check below rejects out-of-range values.
    unsafe {
        core::ptr::copy_nonoverlapping(
            layout.base_body.as_ptr(),
            neg.as_mut_ptr().cast::<u8>(),
            layout.base_body.len(),
        );
        neg.set_len(metadata.base_prefix_len);
    }
    for ring in &neg {
        for (limb, prime) in ring.limbs.iter().zip(&layout.params.primes) {
            let (min, max) = (-prime.p, prime.p);
            if limb.iter().any(|coefficient| {
                let raw = coefficient.raw();
                raw <= min || raw >= max
            }) {
                return Err(invalid("prepared cache Q128 residue is out of range"));
            }
        }
    }
    let mut tail_rings: Vec<CyclotomicCrtNtt<i16, 1, D>> =
        Vec::with_capacity(metadata.tail_prefix_len);
    // SAFETY: as above, for the `i16` tail body.
    unsafe {
        core::ptr::copy_nonoverlapping(
            layout.tail_body.as_ptr(),
            tail_rings.as_mut_ptr().cast::<u8>(),
            layout.tail_body.len(),
        );
        tail_rings.set_len(metadata.tail_prefix_len);
    }
    for ring in &tail_rings {
        if ring.limbs[0].iter().any(|coefficient| {
            let raw = coefficient.raw();
            raw <= -I16_TAIL_PRIME.p || raw >= I16_TAIL_PRIME.p
        }) {
            return Err(invalid("prepared cache i16 residue is out of range"));
        }
    }
    let tail = layout.needs_tail.then(|| PreparedRows::from(tail_rings));
    let prepared = assemble_scalar_q128_cache(layout.params, PreparedRows::from(neg), tail)?;
    Ok((metadata, prepared))
}

/// In-place counterpart of [`decode_riscv64_scalar_q128_cache`] for an
/// artifact that outlives the program: the residues are used where they lie.
///
/// The header, identities, geometry, lengths, and alignment are checked as in
/// the decoder. The residue ranges are not: like the matrix of an unvalidated
/// setup, they are taken as trusted, so the caller must bind the bytes to
/// trusted setup provisioning or to the verifier program identity.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] for a malformed, mismatched, or
/// misaligned artifact.
pub fn view_riscv64_scalar_q128_cache<F: Field + CanonicalEncoding, const D: usize>(
    bytes: &'static [u8],
    expected_binding: PreparedVerifierNttCacheBinding,
) -> Result<(PreparedVerifierNttCacheMetadata, PreparedNttCache<D>), AkitaError> {
    fn view_rows<T: 'static>(
        body: &'static [u8],
        count: usize,
    ) -> Result<PreparedRows<T>, AkitaError> {
        if body.len() != count * core::mem::size_of::<T>() {
            return Err(invalid("prepared cache body does not match its row count"));
        }
        if body.as_ptr().align_offset(core::mem::align_of::<T>()) != 0 {
            return Err(invalid(
                "prepared cache body is not aligned for in-place use",
            ));
        }
        // SAFETY: `body` holds exactly `count` rows of `T` (a transparent
        // array of `i32`/`i16` residues, valid for every bit pattern), is
        // aligned (checked above), and lives for `'static`.
        let rows = unsafe { core::slice::from_raw_parts(body.as_ptr().cast::<T>(), count) };
        Ok(PreparedRows::Static(rows))
    }
    let layout = scalar_q128_cache_layout::<F, D>(bytes, expected_binding)?;
    let metadata = layout.metadata;
    let neg = view_rows(layout.base_body, metadata.base_prefix_len)?;
    let tail = layout
        .needs_tail
        .then(|| view_rows(layout.tail_body, metadata.tail_prefix_len))
        .transpose()?;
    let prepared = assemble_scalar_q128_cache(layout.params, neg, tail)?;
    Ok((metadata, prepared))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
    use jolt_field::{Prime128Offset275 as F, Ring};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    const D: usize = 64;
    const WIDTH: usize = 4;
    const RHS_ABS_BOUND: u64 = 1 << 15;

    fn matrix() -> Vec<CyclotomicRing<F, D>> {
        (0..WIDTH)
            .map(|ring| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_i64(((ring * 13 + coefficient * 7) % 37) as i64 - 18)
                }))
            })
            .collect()
    }

    fn binding() -> PreparedVerifierNttCacheBinding {
        PreparedVerifierNttCacheBinding {
            setup_seed_digest: [3; 32],
            schedule_row_digest: ScheduleRowDigest::from_bytes([7; 32]),
            setup_field_elements: WIDTH * D,
        }
    }

    fn artifact() -> Vec<u8> {
        let flat = akita_params::FlatMatrix::from_ring_slice(&matrix());
        build_riscv64_scalar_q128_cache_artifact(
            flat.ring_view::<D>(1, WIDTH).expect("matrix view"),
            WIDTH,
            RHS_ABS_BOUND,
            binding(),
        )
        .expect("prepared artifact")
    }

    #[test]
    fn scalar_q128_artifact_round_trips_without_host_backend_dependence() {
        let bytes = artifact();
        let metadata = prepared_verifier_ntt_cache_metadata(&bytes).expect("metadata");
        assert_eq!(metadata.ring_dimension, D);
        assert_eq!(metadata.base_prefix_len, WIDTH);
        assert_eq!(metadata.tail_prefix_len, 0);
        assert_eq!(metadata.binding, binding());
        let (_, decoded) =
            decode_riscv64_scalar_q128_cache::<F, D>(&bytes, binding()).expect("decode");
        assert!(!decoded.uses_ifma52());
        assert!(!decoded.has_exactness_tail());

        let rhs = (0..WIDTH)
            .map(|column| {
                std::array::from_fn(|coefficient| ((column * 5 + coefficient * 3) % 31) as i16 - 15)
            })
            .collect::<Vec<_>>();
        let actual = decoded
            .mat_vec_i16::<F>(16, 1, &rhs)
            .expect("artifact matvec");
        let expected =
            matrix()
                .iter()
                .zip(&rhs)
                .fold(CyclotomicRing::<F, D>::zero(), |sum, (lhs, rhs)| {
                    let rhs = CyclotomicRing::from_coefficients(
                        rhs.map(|coefficient| F::from_i64(i64::from(coefficient))),
                    );
                    sum + *lhs * rhs
                });
        assert_eq!(actual, vec![expected]);
    }

    #[test]
    fn in_place_view_matches_decode_and_rejects_misalignment() {
        let bytes = artifact();
        // Copy the artifact `shift` bytes past an 8-aligned address.
        let place = |shift: usize| -> &'static [u8] {
            let storage = Box::leak(vec![0u8; bytes.len() + 16].into_boxed_slice());
            let start = storage.as_ptr().align_offset(8) + shift;
            storage[start..start + bytes.len()].copy_from_slice(&bytes);
            &storage[start..start + bytes.len()]
        };
        let rhs = (0..WIDTH)
            .map(|column| {
                std::array::from_fn(|coefficient| ((column + coefficient) % 7) as i16 - 3)
            })
            .collect::<Vec<_>>();
        let (_, decoded) =
            decode_riscv64_scalar_q128_cache::<F, D>(&bytes, binding()).expect("decode");
        let (_, viewed) =
            view_riscv64_scalar_q128_cache::<F, D>(place(0), binding()).expect("view");
        assert_eq!(
            viewed.mat_vec_i16::<F>(16, 1, &rhs).expect("view matvec"),
            decoded
                .mat_vec_i16::<F>(16, 1, &rhs)
                .expect("decode matvec")
        );
        assert!(view_riscv64_scalar_q128_cache::<F, D>(place(1), binding()).is_err());
    }

    #[test]
    fn artifact_rejects_identity_header_and_payload_tampering_without_panicking() {
        let original = artifact();
        let ProtocolCrtNttParams::Q128(params) =
            select_crt_ntt_params::<F, D>().expect("Q128 parameters")
        else {
            panic!("test field must use Q128 parameters");
        };
        let mut cases = Vec::new();

        let mut wrong_magic = original.clone();
        wrong_magic[0] ^= 1;
        cases.push(wrong_magic);

        let mut wrong_target = original.clone();
        wrong_target[8..12].copy_from_slice(&2u32.to_le_bytes());
        cases.push(wrong_target);

        let mut wrong_setup = original.clone();
        wrong_setup[56] ^= 1;
        cases.push(wrong_setup);

        let mut wrong_schedule = original.clone();
        wrong_schedule[88] ^= 1;
        cases.push(wrong_schedule);

        let mut invalid_residue = original.clone();
        invalid_residue[HEADER_BYTES..HEADER_BYTES + 4]
            .copy_from_slice(&params.primes[0].p.to_le_bytes());
        cases.push(invalid_residue);

        cases.push(original[..original.len() - 1].to_vec());
        let mut trailing = original.clone();
        trailing.push(0);
        cases.push(trailing);

        for bytes in cases {
            let result = catch_unwind(AssertUnwindSafe(|| {
                decode_riscv64_scalar_q128_cache::<F, D>(&bytes, binding())
            }));
            assert!(matches!(result, Ok(Err(AkitaError::InvalidSetup(_)))));
        }
    }

    #[test]
    fn artifact_rejects_the_wrong_expected_binding() {
        let bytes = artifact();
        let mut expected = binding();
        expected.schedule_row_digest = ScheduleRowDigest::from_bytes([8; 32]);
        assert!(matches!(
            decode_riscv64_scalar_q128_cache::<F, D>(&bytes, expected),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}
