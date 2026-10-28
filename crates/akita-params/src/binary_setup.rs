//! Typed exact setup streams for explicitly selected binary source groups.
//!
//! These streams do not change the ordinary public-field stream. Coefficients
//! of the q matrix and coefficients of P setup matrices have independent domains.

use akita_error::{checked, AkitaError};
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;

/// Characteristic of the scalar field used by binary source groups.
pub const BINARY_SETUP_P: u128 = u128::MAX - 274;
const MAX_REJECTION_ATTEMPTS: usize = 256;
const MAX_MATERIALIZED_BYTES: usize = 1 << 27;

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
}

/// Identity of one independently sampled matrix-entry coefficient stream.
///
/// Fixed layout: domain `akita/binary-source/setup/shake256/v1`, 32-byte
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
    reader: <Shake256 as ExtendableOutput>::Reader,
    modulus: SetupStreamModulus,
    q_threshold: u64,
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
        };
        let mut xof = Shake256::default();
        xof.update(b"akita/binary-source/setup/shake256/v1");
        xof.update(seed);
        xof.update(&[label.role as u8]);
        xof.update(&label.group.to_le_bytes());
        xof.update(&label.level.to_le_bytes());
        xof.update(&label.rank.to_le_bytes());
        xof.update(&label.row.to_le_bytes());
        xof.update(&label.column.to_le_bytes());
        xof.update(&[label.ring as u8]);
        xof.update(&label.degree.to_le_bytes());
        xof.update(&[tag]);
        xof.update(&modulus.to_le_bytes());
        Ok(Self {
            reader: xof.finalize_xof(),
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
            self.reader.read(&mut bytes);
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

    /// Draw an exact canonical P residue by accepting precisely candidates < P.
    ///
    /// # Errors
    /// Rejects a q stream or exhaustion of 256 rejection candidates.
    pub fn sample_p(&mut self) -> Result<u128, AkitaError> {
        if self.modulus != SetupStreamModulus::Prime128Offset275 {
            return Err(AkitaError::InvalidSetup(
                "q setup stream requested as P".into(),
            ));
        }
        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let mut bytes = [0u8; 16];
            self.reader.read(&mut bytes);
            let candidate = u128::from_le_bytes(bytes);
            if candidate < BINARY_SETUP_P {
                return Ok(candidate);
            }
        }
        Err(AkitaError::InvalidSetup(
            "P setup rejection budget exhausted".into(),
        ))
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
    for row in 0..label.rank {
        for column in 0..columns {
            let column = u64::try_from(column)
                .map_err(|_| AkitaError::InvalidSetup("setup column exceeds u64".into()))?;
            let mut stream = TypedSetupStream::new(
                seed,
                SetupStreamLabel {
                    row,
                    column,
                    ..label
                },
            )?;
            for _ in 0..degree {
                matrix.push(stream.sample_q_centered()?);
            }
        }
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests;
