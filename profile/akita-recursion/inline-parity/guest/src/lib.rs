//! Jolt guest that runs Akita's public transcript, challenge, and CRT+NTT
//! paths with `blake2-inline`, `keccak-inline`, and `ntt-inline` enabled.
//!
//! On riscv64 those features route the Blake2b sponge and descriptor digests,
//! the SHAKE fold-challenge sampler, and the 64-point `i32` NTT and lazy
//! pointwise dot through Jolt inlines. Every other target keeps the portable
//! paths, so the host driver compares a traced guest run against a native run
//! of the same function. Each check folds canonical outputs into one FNV-1a
//! digest; a mismatch names the check that diverged.

use akita_algebra::ntt::I32_LAZY_DOT_BATCH;
use akita_algebra::tables::{q128_primes, Q128_NUM_PRIMES};
use akita_algebra::{
    CanonicalEncoding, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut, MontCoeff,
    One, Ring, Zero,
};
use akita_challenges::{
    FoldChallengeDrawDomain, FoldDraw, OperatorNormRejection, SparseChallengeConfig,
    VerifierFoldDraw, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};
use akita_params::digest_descriptor_bytes;
use akita_transcript::{
    new_prover_channel, new_verifier_channel, prover_field_challenge, receive_bytes, send_bytes,
    verifier_field_challenge,
};
use jolt_field::Prime128OffsetA7F7 as Fp;
use serde::{Deserialize, Serialize};

/// Check names, in [`ParityReport::digests`] order.
pub const CHECKS: [&str; 12] = [
    "blake2b transcript bytes and challenges",
    "blake2b descriptor digests",
    "keccak fold challenges D64",
    "keccak fold challenges D64 selective-L2",
    "keccak fold challenges D128",
    "ntt D64 ring round trip",
    "ntt D64 centered-i16 mat-vec",
    "ntt D64 pointwise dot, aligned and offset",
    "ntt D64 i8 column dot",
    "ntt D128 ring round trip",
    "ntt D128 centered-i16 mat-vec",
    "ntt D128 pointwise dot, aligned and offset",
];

/// Outputs of one guest run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityReport {
    /// One digest per entry of [`CHECKS`].
    pub digests: Vec<u64>,
    /// `pointwise_dot_batch_size` of the `i32` CRT parameter sets at `D = 64`
    /// and `D = 128`: the backend each degree selected.
    pub dot_batch_sizes: [usize; 2],
}

#[jolt::provable(
    backtrace = "off",
    stack_size = 8388608,
    heap_size = 134217728,
    max_output_size = 4096,
    max_trace_length = 268435456
)]
fn inline_parity(seed: u64) -> ParityReport {
    let mut digests = Vec::with_capacity(CHECKS.len());
    digests.extend(transcript_checks(seed));
    let (d64, d64_batch) = ntt_checks::<64>(seed);
    digests.extend(d64);
    digests.push(i8_column_dot_check(seed));
    let (d128, d128_batch) = ntt_checks::<128>(seed);
    digests.extend(d128);
    ParityReport {
        digests,
        dot_batch_sizes: [d64_batch, d128_batch],
    }
}

/// 64-bit FNV-1a over canonical output bytes.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn field(&mut self, value: Fp) {
        self.bytes(
            &value
                .to_u128_checked()
                .expect("Fp128 is canonical in u128")
                .to_le_bytes(),
        );
    }

    fn ring<const D: usize>(&mut self, ring: &CyclotomicRing<Fp, D>) {
        for &coefficient in ring.coefficients() {
            self.field(coefficient);
        }
    }
}

/// SplitMix64, so host and guest draw identical inputs from the seed.
struct Inputs(u64);

impl Inputs {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }

    /// Field coefficients, the first few pinned to `0`, `1`, and `-1`.
    fn ring<const D: usize>(&mut self) -> CyclotomicRing<Fp, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|index| match index {
            0 => Fp::zero(),
            1 => Fp::one(),
            2 => -Fp::one(),
            _ => Fp::from_u128(u128::from(self.next()) << 64 | u128::from(self.next())),
        }))
    }

    /// Centered digits, the first few pinned to the `i16` extremes.
    fn i16_digits<const D: usize>(&mut self) -> [i16; D] {
        std::array::from_fn(|index| match index {
            0 => i16::MIN,
            1 => i16::MAX,
            2 => -1,
            _ => self.next() as i16,
        })
    }

    /// Balanced `i8` digits, the first two pinned to the extremes.
    fn i8_digits<const D: usize>(&mut self) -> [i8; D] {
        std::array::from_fn(|index| match index {
            0 => i8::MIN,
            1 => i8::MAX,
            _ => self.next() as i8,
        })
    }
}

/// Places `value` four bytes past a doubleword boundary, where the inline
/// kernels' paired loads do not apply.
#[repr(C, align(8))]
struct Offset<T> {
    _pad: u32,
    value: T,
}

impl<T> Offset<T> {
    fn new(value: T) -> Self {
        let offset = Self { _pad: 0, value };
        assert_eq!(std::ptr::addr_of!(offset.value) as usize % 8, 4);
        offset
    }
}

#[repr(C, align(8))]
struct Aligned<T>(T);

/// Byte payloads straddling the Blake2b block (128) and SHAKE256 rate (136)
/// boundaries.
const PAYLOAD_LENGTHS: [usize; 9] = [0, 1, 63, 127, 128, 129, 136, 137, 1000];

fn transcript_checks(seed: u64) -> [u64; 5] {
    let session = b"akita-inline-parity";
    let instance = seed.to_le_bytes();
    let mut inputs = Inputs(seed);
    let payloads: Vec<Vec<u8>> = PAYLOAD_LENGTHS
        .iter()
        .map(|&len| inputs.bytes(len))
        .collect();

    let mut transcript = Digest::new();
    let mut prover = new_prover_channel(session, &instance).expect("prover channel");
    for payload in &payloads {
        send_bytes(&mut prover, payload);
        transcript.field(prover_field_challenge::<Fp>(&mut prover).expect("prover challenge"));
    }
    let proof = prover.narg_string().to_vec();
    transcript.bytes(&proof);

    let mut verifier = new_verifier_channel(session, &instance, &proof).expect("verifier channel");
    for payload in &payloads {
        let received = receive_bytes(&mut verifier, payload.len()).expect("receive payload");
        assert_eq!(&received, payload, "verifier read a different payload");
        transcript
            .field(verifier_field_challenge::<Fp>(&mut verifier).expect("verifier challenge"));
    }

    let mut descriptors = Digest::new();
    for payload in &payloads {
        descriptors.bytes(&digest_descriptor_bytes(payload));
    }

    // Each draw squeezes a seed from the live verifier sponge and expands it
    // through the indexed SHAKE256 sampler.
    let draws = [
        (
            64,
            SparseChallengeConfig::production_for_ring_dim(64).expect("D64 production config"),
            None,
        ),
        (
            64,
            D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
        ),
        (
            128,
            SparseChallengeConfig::production_for_ring_dim(128).expect("D128 production config"),
            None,
        ),
    ];
    let folds = draws.map(|(ring_d, config, rejection)| {
        let challenges = VerifierFoldDraw::new(&mut verifier, 0, ring_d as u32)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                ring_d,
                0,
                3,
                2,
                &config,
                rejection,
            )
            .expect("fold challenges");
        let mut digest = Digest::new();
        for challenge in challenges.as_slice() {
            for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                digest.bytes(&position.to_le_bytes());
                digest.bytes(&coefficient.to_le_bytes());
            }
        }
        digest.0
    });

    [transcript.0, descriptors.0, folds[0], folds[1], folds[2]]
}

type Ntt<const D: usize> = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>;

/// Ring round trip, centered-`i16` mat-vec, and pointwise dots at degree `D`,
/// plus the dot batch size the parameter set selected.
fn ntt_checks<const D: usize>(seed: u64) -> ([u64; 3], usize) {
    let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
    let mut inputs = Inputs(seed ^ D as u64);

    let mut round_trip = Digest::new();
    let rings: Vec<CyclotomicRing<Fp, D>> = (0..4).map(|_| inputs.ring()).collect();
    for ring in &rings {
        round_trip.ring(&Ntt::<D>::from_ring(ring, &params).to_ring::<Fp>(&params));
    }

    // Column counts cover a short batch, an exact batch, a batch plus one,
    // and two full batches plus one; a zero column ends a dense batch early.
    let mut mat_vec = Digest::new();
    for num_cols in [
        1,
        5,
        I32_LAZY_DOT_BATCH,
        I32_LAZY_DOT_BATCH + 1,
        2 * I32_LAZY_DOT_BATCH + 1,
    ] {
        let num_rows = 2;
        let matrix: Vec<Ntt<D>> = (0..num_rows * num_cols)
            .map(|_| Ntt::<D>::from_ring(&inputs.ring(), &params))
            .collect();
        let mut rhs: Vec<[i16; D]> = (0..num_cols).map(|_| inputs.i16_digits()).collect();
        if num_cols > 2 {
            rhs[2] = [0; D];
        }
        for row in Ntt::<D>::mat_vec_i16::<Fp>(&matrix, num_rows, num_cols, &rhs, &params)
            .expect("mat-vec")
        {
            mat_vec.ring(&row);
        }
    }

    let lhs: [Ntt<D>; I32_LAZY_DOT_BATCH] =
        std::array::from_fn(|_| Ntt::<D>::from_ring(&inputs.ring(), &params));
    let rhs: [Ntt<D>; I32_LAZY_DOT_BATCH] =
        std::array::from_fn(|_| Ntt::<D>::from_ring(&inputs.ring(), &params));
    let initial = Ntt::<D>::from_ring(&inputs.ring(), &params);
    let aligned_rows = Aligned((lhs.clone(), rhs.clone()));
    let offset_lhs = Offset::new(lhs);
    let offset_rhs = Offset::new(rhs);
    let mut dot = Digest::new();
    for count in 0..=I32_LAZY_DOT_BATCH {
        let mut aligned = Aligned(initial.clone());
        aligned.0.add_assign_pointwise_dot(
            &aligned_rows.0 .0[..count],
            &aligned_rows.0 .1[..count],
            &params,
        );
        let mut offset = Offset::new(initial.clone());
        offset.value.add_assign_pointwise_dot(
            &offset_lhs.value[..count],
            &offset_rhs.value[..count],
            &params,
        );
        dot.ring(&aligned.0.to_ring::<Fp>(&params));
        dot.ring(&offset.value.to_ring::<Fp>(&params));
    }

    (
        [round_trip.0, mat_vec.0, dot.0],
        params.pointwise_dot_batch_size(),
    )
}

/// Signed-`i8` digit columns through the batched column dot when the
/// parameter set selects one, else one column at a time, as the commitment
/// kernels dispatch. Accumulators and scratch sit off a doubleword boundary.
fn i8_column_dot_check(seed: u64) -> u64 {
    const D: usize = 64;
    const NUM_COLS: usize = I32_LAZY_DOT_BATCH;
    let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
    let lut = DigitMontLut::new_with_digit_bound(&params, 128);
    let mut inputs = Inputs(seed ^ 0x18);
    let rows: [Vec<Ntt<D>>; 2] = std::array::from_fn(|_| {
        (0..NUM_COLS)
            .map(|_| Ntt::<D>::from_ring(&inputs.ring(), &params))
            .collect()
    });
    let ntt_mat: [&[Ntt<D>]; 2] = [&rows[0], &rows[1]];
    let digits: Vec<[i8; D]> = (0..NUM_COLS).map(|_| inputs.i8_digits()).collect();

    let mut accs = Offset::new([Ntt::<D>::zero(), Ntt::<D>::zero()]);
    let batch = params.pointwise_dot_batch_size();
    if batch > 1 {
        let mut scratch = Offset::new([[MontCoeff::from_raw(0); D]; I32_LAZY_DOT_BATCH]);
        for column_start in (0..NUM_COLS).step_by(batch) {
            let end = (column_start + batch).min(NUM_COLS);
            Ntt::<D>::add_assign_col_pointwise_dot_i8_multi_with_lut_scratch(
                &mut accs.value,
                &ntt_mat,
                column_start,
                &digits[column_start..end],
                &params,
                &lut,
                &mut scratch.value,
            );
        }
    } else {
        let mut scratch = Offset::new([[MontCoeff::from_raw(0); D]; Q128_NUM_PRIMES]);
        for (column, digits) in digits.iter().enumerate() {
            Ntt::<D>::add_assign_col_pointwise_mul_i8_multi_with_lut_scratch(
                &mut accs.value,
                &ntt_mat,
                column,
                digits,
                &params,
                &lut,
                &mut scratch.value,
            );
        }
    }

    let mut digest = Digest::new();
    for acc in &accs.value {
        digest.ring(&acc.to_ring::<Fp>(&params));
    }
    digest.0
}
