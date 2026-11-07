//! Typed exact setup streams for explicitly selected binary source groups.
//!
//! These streams do not change the ordinary public-field stream. Coefficients
//! of the q matrix and coefficients of prime setup matrices have independent domains.

use akita_error::{checked, AkitaError};
use jolt_field::cfg_chunks_mut;
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use keccak::{Backend, BackendClosure, Keccak, ParState1600, State1600, PLEN};

/// Characteristic of the scalar field used by binary source groups.
pub const BINARY_SETUP_P: u128 = u128::MAX - 274;
/// Characteristic of the paired p64/E64 binary profile.
pub const BINARY_SETUP_P64: u64 = u64::MAX - 58;

/// Paired base and challenge fields. Defining a profile does not admit its root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryFieldProfile {
    /// Legacy P128 base field with degree-one challenges.
    P128,
    /// Base p = 2^64 - 59 and E64 = F_p[u]/(u^2 - 2), basis (1,u).
    P64,
}

impl BinaryFieldProfile {
    /// Balanced base-eight depth covering every centered base coordinate.
    pub const fn prime_plane_digits(self) -> usize {
        match self {
            Self::P128 => 43,
            Self::P64 => 22,
        }
    }

    /// Exact base characteristic, also used for canonical payload residues.
    pub const fn modulus(self) -> u128 {
        match self {
            Self::P128 => BINARY_SETUP_P,
            Self::P64 => BINARY_SETUP_P64 as u128,
        }
    }

    /// Maximum distinct prime opening points admitted for one committed group.
    pub const fn maximum_prime_points(self) -> usize {
        match self {
            Self::P128 => 1,
            Self::P64 => 2,
        }
    }

    /// Extension degree of testing challenges over the base field.
    pub const fn extension_degree(self) -> usize {
        match self {
            Self::P128 => 1,
            Self::P64 => 2,
        }
    }

    /// Exact challenge order; p64 uses (64,2,p), never (128,1,p).
    pub fn challenge_order(self) -> Result<crate::ChallengeFieldOrder, AkitaError> {
        crate::ChallengeFieldOrder::from_field(
            match self {
                Self::P128 => 128,
                Self::P64 => 64,
            },
            self.extension_degree(),
            self.modulus(),
        )
    }

    /// Check the concrete base and challenge fields, without authorizing a root.
    pub fn check_field_pair(self, modulus: u128, degree: usize) -> Result<(), AkitaError> {
        if modulus != self.modulus() || degree != self.extension_degree() {
            return Err(AkitaError::InvalidSetup(
                "binary root field profile mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Require native-row admission for P64 at the public PCS boundary.
    /// `native` must come from a row resolved through the trusted catalog.
    /// This is the execution boundary; there is no separate production gate.
    /// Production use additionally requires the open security-ledger obligations
    /// in `specs/binary-source-p64-root.md`, including extraction and all SIS instances.
    pub fn admit_root(self, native: bool) -> Result<(), AkitaError> {
        if self == Self::P64 && !native {
            return Err(AkitaError::UnsupportedSchedule(
                "p64 root requires an explicitly selected native catalog row".into(),
            ));
        }
        Ok(())
    }

    /// Least admitted B or D rank for its actual degree-64 digit width.
    /// The legacy profile retains its frozen two rows, including its identities.
    pub fn digit_commitment_rank(
        self,
        role: SetupStreamRole,
        columns: usize,
    ) -> Result<usize, AkitaError> {
        let matrix_role = match role {
            SetupStreamRole::ImageCommitment => crate::sis::SisMatrixRole::Outer,
            SetupStreamRole::WitnessCommitment => crate::sis::SisMatrixRole::Open,
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "invalid binary digit setup role".into(),
                ))
            }
        };
        if columns == 0 {
            return Err(AkitaError::InvalidSetup("empty binary digit setup".into()));
        }
        if self == Self::P128 {
            return Ok(2);
        }
        use crate::sis::{
            min_secure_rank, SisModulusProfileId, SisTableDigest, SisTableKey,
            DEFAULT_SIS_SECURITY_POLICY,
        };
        let width = u64::try_from(columns)
            .map_err(|_| AkitaError::InvalidSetup("binary digit width exceeds u64".into()))?;
        min_secure_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q64Offset59,
                role: matrix_role,
                ring_dimension: 64,
                coeff_linf_bound: 7,
            },
            width,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("binary digit width exceeds admitted ranks".into()))
    }
}
const MAX_REJECTION_ATTEMPTS: usize = 256;
const MAX_MATERIALIZED_BYTES: usize = 1 << 27;
/// Bytes absorbed and squeezed per SHAKE256 permutation.
const SHAKE256_RATE: usize = 136;
/// Candidates squeezed per read of [`PDraws`].
const P_BLOCK: usize = 64;
const SETUP_DOMAIN: &[u8] = b"akita/binary-source/setup/shake256/v1";

/// Algebraic role of the public setup coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SetupStreamRole {
    /// Binary source commitment matrix A.
    Commitment = 1,
    /// Scalar setup used to commit the balanced image digits.
    ImageCommitment = 2,
    /// Scalar setup used to commit recursive witnesses.
    WitnessCommitment = 3,
    /// Setup used for relation projections.
    Projection = 4,
}

/// Polynomial relation of the setup's coefficient ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SetupStreamRing {
    /// Z^d + Z^(d/2) + 1.
    PlusTrinomial = 1,
    /// Z^d - Z^(d/2) + 1.
    MinusTrinomial = 2,
    /// Z^d + 1, used by ordinary suffix levels.
    Negacyclic = 3,
}

/// Coefficient domain; its tag and full modulus enter the stream identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupStreamModulus {
    /// Uniform residues modulo an odd q, returned as centered integers.
    Q(u32),
    /// Exact canonical residues in P = 2^128 - 275.
    Prime128Offset275,
    /// Exact canonical residues in p = 2^64 - 59 (eight-byte candidates).
    Prime64Offset59,
}

impl SetupStreamModulus {
    /// Versioned sampler identity. The q-valued A stream retains its v1 domain.
    pub const fn domain(self) -> &'static [u8] {
        match self {
            Self::Prime64Offset59 => b"akita/binary-source/setup/shake256/v2",
            _ => SETUP_DOMAIN,
        }
    }
}

/// Identity of one independently sampled matrix-entry coefficient stream.
///
/// Fixed layout: the versioned [`SetupStreamModulus::domain`], 32-byte
/// seed, role:u8, group:u64, level:u32, rank:u32, row:u32, column:u64,
/// ring:u8, degree:u32, coefficient-domain:u8, modulus:u128. All integers
/// are little endian. Coefficient position is the sequential XOF position;
/// rejection consumes candidates within that entry's own stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupStreamLabel {
    /// Algebraic role of this setup matrix.
    pub role: SetupStreamRole,
    /// Caller-selected commitment group index.
    pub group: u64,
    /// Ordinary grouped-opening level.
    pub level: u32,
    /// Number of matrix rows.
    pub rank: u32,
    /// Zero-based row index, strictly smaller than rank.
    pub row: u32,
    /// Zero-based matrix column index.
    pub column: u64,
    /// Polynomial relation of the matrix ring.
    pub ring: SetupStreamRing,
    /// Coefficient count of one ring element.
    pub degree: u32,
    /// Typed coefficient field or q residue domain.
    pub modulus: SetupStreamModulus,
}

/// Return the exclusive u32-candidate rejection threshold floor(2^32/q)q.
///
/// # Errors
/// Rejects q < 3 or even q; binary setup requires a symmetric centered interval.
pub fn q_rejection_threshold(q: u32) -> Result<u64, AkitaError> {
    if q < 3 || q.is_multiple_of(2) {
        return Err(AkitaError::InvalidSetup(
            "setup q must be odd and at least three".into(),
        ));
    }
    let range = 1u64 << 32;
    Ok(range - range % u64::from(q))
}

/// Stateful coefficient stream with validated, typed identity.
pub struct TypedSetupStream {
    squeeze: ShakeSqueeze,
    modulus: SetupStreamModulus,
    q_threshold: u64,
}

/// A SHAKE256 sponge in its squeezing phase.
///
/// The output is the standard SHAKE256 stream of the absorbed message: the
/// little-endian bytes of the first `SHAKE256_RATE / 8` lanes after each
/// permutation. Holding the state lets two streams share one call of the
/// backend's two-state permutation, which costs what one state costs on the
/// AArch64 SHA-3 instructions.
#[derive(Clone)]
struct ShakeSqueeze {
    keccak: Keccak,
    state: State1600,
    /// The current output block, the state's leading bytes.
    block: [u8; SHAKE256_RATE],
    /// Bytes of `block` already read.
    read: usize,
}

impl ShakeSqueeze {
    /// Absorb a message shorter than one block, pad it and permute.
    fn new(message: &[&[u8]]) -> Result<Self, AkitaError> {
        let length = message.iter().map(|part| part.len()).sum::<usize>();
        let mut padded = [0u8; SHAKE256_RATE];
        if length >= SHAKE256_RATE {
            return Err(AkitaError::InvalidSetup(
                "setup stream identity exceeds one SHAKE256 block".into(),
            ));
        }
        for (slot, &byte) in padded
            .iter_mut()
            .zip(message.iter().flat_map(|part| part.iter()))
        {
            *slot = byte;
        }
        *padded.get_mut(length).ok_or_else(|| {
            AkitaError::InvalidSetup("setup stream identity exceeds one SHAKE256 block".into())
        })? ^= 0x1f;
        padded[SHAKE256_RATE - 1] ^= 0x80;
        let mut state = [0u64; PLEN];
        for (lane, bytes) in state.iter_mut().zip(padded.chunks_exact(8)) {
            let mut word = [0u8; 8];
            word.copy_from_slice(bytes);
            *lane = u64::from_le_bytes(word);
        }
        let mut squeeze = Self {
            keccak: Keccak::new(),
            state,
            block: [0; SHAKE256_RATE],
            read: 0,
        };
        squeeze.permute();
        Ok(squeeze)
    }

    fn permute(&mut self) {
        self.keccak.with_f1600(|permute| permute(&mut self.state));
        self.refill();
    }

    /// Start reading the block of a freshly permuted state.
    fn refill(&mut self) {
        for (bytes, lane) in self.block.chunks_exact_mut(8).zip(&self.state) {
            bytes.copy_from_slice(&lane.to_le_bytes());
        }
        self.read = 0;
    }

    /// Copy unread bytes of the current block to the front of `output` and
    /// return the rest of `output`, which is empty unless the block ran out.
    fn take<'a>(&mut self, output: &'a mut [u8]) -> &'a mut [u8] {
        let unread = self.block.get(self.read..).unwrap_or_default();
        let count = unread.len().min(output.len());
        let (front, rest) = output.split_at_mut(count);
        front.copy_from_slice(unread.get(..count).unwrap_or_default());
        self.read += count;
        rest
    }

    /// Fill `output` with the next bytes of the stream.
    fn read(&mut self, mut output: &mut [u8]) {
        loop {
            output = self.take(output);
            if output.is_empty() {
                return;
            }
            self.permute();
        }
    }

    /// Fill both outputs from their streams, exactly as two calls of
    /// [`Self::read`]; whenever both streams need a block, one call permutes
    /// both states.
    fn read_pair(
        first: &mut Self,
        mut first_output: &mut [u8],
        second: &mut Self,
        mut second_output: &mut [u8],
    ) {
        loop {
            first_output = first.take(first_output);
            second_output = second.take(second_output);
            match (first_output.is_empty(), second_output.is_empty()) {
                (true, true) => return,
                (false, true) => first.permute(),
                (true, false) => second.permute(),
                (false, false) => {
                    first
                        .keccak
                        .with_backend(PermutePair(&mut first.state, &mut second.state));
                    first.refill();
                    second.refill();
                }
            }
        }
    }
}

/// Two SHAKE256 permutation states permuted through the backend's parallel
/// permutation when it holds two or more states, otherwise one at a time.
struct PermutePair<'a>(&'a mut State1600, &'a mut State1600);

impl BackendClosure for PermutePair<'_> {
    #[inline(always)]
    fn call_once<B: Backend>(self) {
        let mut states = ParState1600::<B>::default();
        if let [first, second, ..] = states.as_mut_slice() {
            *first = *self.0;
            *second = *self.1;
        } else {
            let permute = B::get_f1600();
            permute(self.0);
            permute(self.1);
            return;
        }
        B::get_par_f1600()(&mut states);
        if let [first, second, ..] = states.as_slice() {
            *self.0 = *first;
            *self.1 = *second;
        }
    }
}

/// One stream's progress through consecutive exact P draws.
///
/// Candidates are read in blocks of one per remaining draw, at most
/// [`P_BLOCK`]; only a rejection reads further candidates. The accepted
/// values, the stream position afterwards and the per-draw rejection budget
/// are exactly those of successive single draws.
struct PDraws<'a> {
    remaining: &'a mut [u128],
    rejected: usize,
    bytes: [u8; 16 * P_BLOCK],
    wanted: usize,
}

impl<'a> PDraws<'a> {
    fn new(output: &'a mut [u128]) -> Self {
        Self {
            remaining: output,
            rejected: 0,
            bytes: [0; 16 * P_BLOCK],
            wanted: 0,
        }
    }

    fn done(&self) -> bool {
        self.remaining.is_empty()
    }

    /// The candidate bytes to read next; empty once every draw is accepted.
    fn block(&mut self) -> &mut [u8] {
        self.wanted = self.remaining.len().min(P_BLOCK);
        self.bytes.get_mut(..16 * self.wanted).unwrap_or_default()
    }

    /// Accept the candidates of the block just read.
    fn accept(&mut self) -> Result<(), AkitaError> {
        let mut accepted = 0;
        for candidate in self
            .bytes
            .get(..16 * self.wanted)
            .unwrap_or_default()
            .chunks_exact(16)
        {
            let mut word = [0u8; 16];
            word.copy_from_slice(candidate);
            let candidate = u128::from_le_bytes(word);
            if candidate < BINARY_SETUP_P {
                // At most `wanted <= remaining.len()` candidates are accepted.
                if let Some(slot) = self.remaining.get_mut(accepted) {
                    *slot = candidate;
                }
                accepted += 1;
                self.rejected = 0;
            } else {
                self.rejected += 1;
                if self.rejected == MAX_REJECTION_ATTEMPTS {
                    return Err(AkitaError::InvalidSetup(
                        "P setup rejection budget exhausted".into(),
                    ));
                }
            }
        }
        self.remaining = std::mem::take(&mut self.remaining)
            .get_mut(accepted..)
            .unwrap_or_default();
        Ok(())
    }
}

impl TypedSetupStream {
    /// Instantiate an independent entry stream; no setup material is allocated.
    ///
    /// # Errors
    /// Rejects invalid rank, row, degree or coefficient modulus.
    pub fn new(seed: &[u8; 32], label: SetupStreamLabel) -> Result<Self, AkitaError> {
        if label.rank == 0
            || label.row >= label.rank
            || label.degree == 0
            || (label.ring != SetupStreamRing::Negacyclic && !label.degree.is_multiple_of(2))
        {
            return Err(AkitaError::InvalidSetup("invalid typed setup label".into()));
        }
        let (tag, modulus, q_threshold) = match label.modulus {
            SetupStreamModulus::Q(q) => (1u8, u128::from(q), q_rejection_threshold(q)?),
            SetupStreamModulus::Prime128Offset275 => (2u8, BINARY_SETUP_P, 0),
            SetupStreamModulus::Prime64Offset59 => (3u8, u128::from(BINARY_SETUP_P64), 0),
        };
        let squeeze = ShakeSqueeze::new(&[
            label.modulus.domain(),
            seed,
            &[label.role as u8],
            &label.group.to_le_bytes(),
            &label.level.to_le_bytes(),
            &label.rank.to_le_bytes(),
            &label.row.to_le_bytes(),
            &label.column.to_le_bytes(),
            &[label.ring as u8],
            &label.degree.to_le_bytes(),
            &[tag],
            &modulus.to_le_bytes(),
        ])?;
        Ok(Self {
            squeeze,
            modulus: label.modulus,
            q_threshold,
        })
    }

    /// Draw an exact uniform q residue and center it in [-(q-1)/2, (q-1)/2].
    ///
    /// # Errors
    /// Rejects a P stream or exhaustion of 256 rejection candidates. Even at
    /// the worst u32 modulus, exhaustion has probability at most 2^-256.
    pub fn sample_q_centered(&mut self) -> Result<i64, AkitaError> {
        let SetupStreamModulus::Q(q) = self.modulus else {
            return Err(AkitaError::InvalidSetup(
                "P setup stream requested as q".into(),
            ));
        };
        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let mut bytes = [0u8; 4];
            self.squeeze.read(&mut bytes);
            let candidate = u32::from_le_bytes(bytes);
            if u64::from(candidate) < self.q_threshold {
                let residue = candidate % q;
                return Ok(if residue > q / 2 {
                    i64::from(residue) - i64::from(q)
                } else {
                    i64::from(residue)
                });
            }
        }
        Err(AkitaError::InvalidSetup(
            "q setup rejection budget exhausted".into(),
        ))
    }

    /// Draw a canonical p64 base residue, rejecting u64 candidates at or above p.
    /// A coordinate has at most 256 attempts; exhaustion is a typed setup error.
    pub fn sample_p64(&mut self) -> Result<u64, AkitaError> {
        if self.modulus != SetupStreamModulus::Prime64Offset59 {
            return Err(AkitaError::InvalidSetup(
                "non-p64 setup stream requested as p64".into(),
            ));
        }
        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let mut bytes = [0u8; 8];
            self.squeeze.read(&mut bytes);
            let candidate = u64::from_le_bytes(bytes);
            if candidate < BINARY_SETUP_P64 {
                return Ok(candidate);
            }
        }
        Err(AkitaError::InvalidSetup(
            "p64 setup rejection budget exhausted".into(),
        ))
    }

    /// Draw a canonical residue in the selected prime, accepting candidates below it.
    ///
    /// # Errors
    /// Rejects a q stream or exhaustion of 256 rejection candidates per residue.
    pub fn sample_p(&mut self) -> Result<u128, AkitaError> {
        let mut output = [0u128];
        self.fill_p(&mut output)?;
        Ok(output[0])
    }

    /// Fill `output` with consecutive [`Self::sample_p`] draws.
    ///
    /// P128 candidates are squeezed in blocks; p64 uses eight bytes per
    /// candidate. Both preserve the accepted values, stream position and
    /// per-draw rejection budget of `output.len()` successive `sample_p` calls.
    ///
    /// # Errors
    /// Rejects a q stream or exhaustion of 256 consecutive rejection candidates.
    pub fn fill_p(&mut self, output: &mut [u128]) -> Result<(), AkitaError> {
        if self.modulus == SetupStreamModulus::Prime64Offset59 {
            for slot in output {
                *slot = u128::from(self.sample_p64()?);
            }
            return Ok(());
        }
        self.require_p()?;
        let mut draws = PDraws::new(output);
        while !draws.done() {
            self.squeeze.read(draws.block());
            draws.accept()?;
        }
        Ok(())
    }

    /// Fill each output from its own stream, exactly as [`Self::fill_p`] on
    /// each stream would. P128 streams use paired permutations where the
    /// backend supports them; p64 streams retain their separate reads.
    ///
    /// # Errors
    /// Rejects a q stream or exhaustion of either stream's rejection budget.
    pub fn fill_p_pair(
        first: &mut Self,
        first_output: &mut [u128],
        second: &mut Self,
        second_output: &mut [u128],
    ) -> Result<(), AkitaError> {
        if first.modulus == SetupStreamModulus::Prime64Offset59
            && second.modulus == SetupStreamModulus::Prime64Offset59
        {
            first.fill_p(first_output)?;
            return second.fill_p(second_output);
        }
        first.require_p()?;
        second.require_p()?;
        let mut first_draws = PDraws::new(first_output);
        let mut second_draws = PDraws::new(second_output);
        while !(first_draws.done() && second_draws.done()) {
            ShakeSqueeze::read_pair(
                &mut first.squeeze,
                first_draws.block(),
                &mut second.squeeze,
                second_draws.block(),
            );
            first_draws.accept()?;
            second_draws.accept()?;
        }
        Ok(())
    }

    fn require_p(&self) -> Result<(), AkitaError> {
        if self.modulus != SetupStreamModulus::Prime128Offset275 {
            return Err(AkitaError::InvalidSetup(
                "q setup stream requested as P".into(),
            ));
        }
        Ok(())
    }
}

/// Materialize rank × columns × degree centered q coefficients, row major.
///
/// The label must name row and column zero; every matrix entry gets its own
/// indexed stream. Increasing `columns` preserves each existing entry stream.
///
/// # Errors
/// Rejects invalid labels, dimensions, count overflow, allocation budgets over
/// 128 MiB, allocation failure, or rejection exhaustion.
pub fn derive_centered_q_matrix(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    columns: usize,
) -> Result<Vec<i64>, AkitaError> {
    TypedSetupStream::new(seed, label)?;
    if label.row != 0
        || label.column != 0
        || columns == 0
        || !matches!(label.modulus, SetupStreamModulus::Q(_))
    {
        return Err(AkitaError::InvalidSetup(
            "invalid q setup matrix label or width".into(),
        ));
    }
    let rank = usize::try_from(label.rank)
        .map_err(|_| AkitaError::InvalidSetup("setup rank exceeds usize".into()))?;
    let degree = usize::try_from(label.degree)
        .map_err(|_| AkitaError::InvalidSetup("setup degree exceeds usize".into()))?;
    let count = checked::product([rank, columns, degree])
        .ok_or_else(|| AkitaError::InvalidSetup("q setup matrix size overflow".into()))?;
    if checked::product([count, size_of::<i64>()])
        .is_none_or(|bytes| bytes > MAX_MATERIALIZED_BYTES)
    {
        return Err(AkitaError::InvalidSetup(
            "q setup matrix exceeds materialization budget".into(),
        ));
    }
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidSetup("q setup matrix allocation failed".into()))?;
    matrix.resize(count, 0);
    // Every entry has its own stream, so entries derive independently.
    cfg_chunks_mut!(matrix, degree).enumerate().try_for_each(
        |(entry, coefficients)| -> Result<(), AkitaError> {
            let row = u32::try_from(entry / columns)
                .map_err(|_| AkitaError::InvalidSetup("setup row exceeds u32".into()))?;
            let column = u64::try_from(entry % columns)
                .map_err(|_| AkitaError::InvalidSetup("setup column exceeds u64".into()))?;
            let mut stream = TypedSetupStream::new(
                seed,
                SetupStreamLabel {
                    row,
                    column,
                    ..label
                },
            )?;
            for coefficient in coefficients {
                *coefficient = stream.sample_q_centered()?;
            }
            Ok(())
        },
    )?;
    Ok(matrix)
}

#[cfg(test)]
mod tests;
