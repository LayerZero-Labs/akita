//! Explicit experimental commitment geometry for binary source groups.

use super::{
    BinaryCommitmentProfiles, BinaryIndexMap, BinarySourceDescriptor, BINARY_SOURCE_Q,
    MAX_BINARY_POLYNOMIALS,
};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_params::binary_setup::{
    BinaryFieldProfile, SetupStreamLabel, SetupStreamModulus, SetupStreamRing, SetupStreamRole,
    BINARY_SETUP_P64,
};
use akita_params::GroupCommitPhaseParams;
use akita_serialization::SerializationError;
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;
use std::io::{Cursor, Read};

/// Maximum encoded manifest accepted before allocating a descriptor buffer.
pub const MAX_BINARY_COMMITTED_MANIFEST_BYTES: usize = 132_000;
/// Legacy P128 payload length; new code derives the count from the manifest.
pub const BINARY_COMMITTED_PAYLOAD_COEFFICIENTS: usize = B_ROWS * B_DEGREE;
const A_RANK: usize = 3;
const A_DEGREE: usize = 648;
const B_ROWS: usize = 2;
const B_DEGREE: usize = 64;

/// Frozen, explicitly selected experimental binary commitment contract.
///
/// The q24 A matrix has rank three over the degree-648 minus trinomial ring.
/// B has degree-64 negacyclic rows: two for P128, or the admitted p64 rank.
/// The ordinary profile carries the ordinary suffix; its digest is
/// separate from the real A/B profile identities in the source descriptor.
/// This runtime admission is for protocol tests and makes no security claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryCommittedGroupManifest {
    descriptor: BinarySourceDescriptor,
    setup_group: u64,
    carrier_profile_digest: [u8; 32],
    field_profile: BinaryFieldProfile,
    b_rank: usize,
}

impl BinaryCommittedGroupManifest {
    pub fn new(
        descriptor: BinarySourceDescriptor,
        setup_group: u64,
        carrier_profile: &GroupCommitPhaseParams,
    ) -> Result<Self, AkitaError> {
        Self::with_field_profile(
            descriptor,
            setup_group,
            carrier_profile,
            BinaryFieldProfile::P128,
        )
    }

    /// Define a paired profile manifest without selecting an executable root.
    pub fn with_field_profile(
        descriptor: BinarySourceDescriptor,
        setup_group: u64,
        carrier_profile: &GroupCommitPhaseParams,
        field_profile: BinaryFieldProfile,
    ) -> Result<Self, AkitaError> {
        let b_rank = field_profile.digit_commitment_rank(
            SetupStreamRole::ImageCommitment,
            b_columns(descriptor.index_map(), descriptor.polynomials().len())?,
        )?;
        let manifest = Self {
            descriptor,
            setup_group,
            carrier_profile_digest: Self::carrier_profile_identity(carrier_profile)?,
            field_profile,
            b_rank,
        };
        manifest.validate_carrier(carrier_profile)?;
        Ok(manifest)
    }

    pub fn descriptor(&self) -> &BinarySourceDescriptor {
        &self.descriptor
    }
    pub fn setup_group(&self) -> u64 {
        self.setup_group
    }
    pub fn a_rank(&self) -> usize {
        A_RANK
    }
    pub fn b_ring_dimension(&self) -> usize {
        B_DEGREE
    }
    pub fn b_rows(&self) -> usize {
        self.b_rank
    }
    /// Paired base and testing fields bound by this manifest's version.
    pub fn field_profile(&self) -> BinaryFieldProfile {
        self.field_profile
    }

    /// Base coefficient count of the admitted B image, with no extension padding.
    pub fn payload_coefficients(&self) -> Result<usize, AkitaError> {
        checked::product([self.b_rows(), self.b_ring_dimension()])
            .ok_or_else(|| AkitaError::InvalidInput("binary payload size overflow".into()))
    }

    /// D setup for the selected word U and base-coordinate prime planes.
    /// Each role selects its own least rank from its actual compact width.
    pub fn d_label(&self, word: bool, planes: usize) -> Result<SetupStreamLabel, AkitaError> {
        if planes > super::PRIME_POINT_CAP || (!word && planes == 0) {
            return Err(AkitaError::InvalidInput(
                "invalid binary D selection".into(),
            ));
        }
        let digits = match self.field_profile {
            BinaryFieldProfile::P128 => 43,
            BinaryFieldProfile::P64 => 22,
        };
        let lanes = checked::sum([
            checked::product([planes, digits]).ok_or(AkitaError::InvalidProof)?,
            if word {
                self.descriptor.polynomials().len()
            } else {
                0
            },
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let columns = checked::product([lanes, 162, self.descriptor.index_map().columns()])
            .and_then(|count| checked::div_ceil(count, B_DEGREE))
            .ok_or(AkitaError::InvalidProof)?;
        Ok(SetupStreamLabel {
            role: SetupStreamRole::WitnessCommitment,
            rank: self
                .field_profile
                .digit_commitment_rank(SetupStreamRole::WitnessCommitment, columns)?
                as u32,
            ..self.b_label()
        })
    }
    /// Compute the identities of the actual typed A/B setup contracts.
    pub fn commitment_profiles(
        index: &BinaryIndexMap,
        polynomials: usize,
        setup_group: u64,
    ) -> Result<BinaryCommitmentProfiles, AkitaError> {
        Self::profiles_for_field(index, polynomials, setup_group, BinaryFieldProfile::P128)
    }

    /// A retains its q-valued identity; B binds the paired field and admitted rank.
    pub fn profiles_for_field(
        index: &BinaryIndexMap,
        polynomials: usize,
        setup_group: u64,
        field_profile: BinaryFieldProfile,
    ) -> Result<BinaryCommitmentProfiles, AkitaError> {
        if polynomials == 0 || polynomials > MAX_BINARY_POLYNOMIALS {
            return Err(AkitaError::InvalidInput(
                "binary polynomial count out of range".into(),
            ));
        }
        let a_columns = checked::product([4, index.columns()])
            .and_then(|block| checked::exact_div(index.padded_carriers(), block))
            .ok_or_else(|| AkitaError::InvalidInput("binary A input width overflow".into()))?;
        let b_columns = b_columns(index, polynomials)?;
        let b_rank =
            field_profile.digit_commitment_rank(SetupStreamRole::ImageCommitment, b_columns)?;
        let encode = |role: SetupStreamRole, columns: usize| -> Result<[u8; 32], AkitaError> {
            let label = setup_label(setup_group, role, field_profile, b_rank);
            let (modulus_tag, modulus, digits, input_bound) = match label.modulus {
                SetupStreamModulus::Q(q) => (1u8, u128::from(q), 1u32, 1u32),
                SetupStreamModulus::Prime128Offset275 => (2u8, BINARY_SETUP_P, 8u32, 4u32),
                SetupStreamModulus::Prime64Offset59 => {
                    (3u8, u128::from(BINARY_SETUP_P64), 8u32, 4u32)
                }
            };
            let columns = u64::try_from(columns)
                .map_err(|_| AkitaError::InvalidInput("binary setup width exceeds u64".into()))?;
            let p64 = label.modulus == SetupStreamModulus::Prime64Offset59;
            let mut bytes = if p64 {
                b"akita/binary-source/experimental-profile/v2".to_vec()
            } else {
                b"akita/binary-source/experimental-profile/v1".to_vec()
            };
            bytes.extend_from_slice(label.modulus.domain());
            bytes.push(role as u8);
            bytes.extend_from_slice(&setup_group.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&label.rank.to_le_bytes());
            bytes.extend_from_slice(&columns.to_le_bytes());
            bytes.push(label.ring as u8);
            bytes.extend_from_slice(&label.degree.to_le_bytes());
            bytes.push(modulus_tag);
            bytes.extend_from_slice(&modulus.to_le_bytes());
            bytes.extend_from_slice(&digits.to_le_bytes());
            bytes.extend_from_slice(&input_bound.to_le_bytes());
            if p64 {
                // E64 basis (1,u), u^2 = 2; base payloads are canonical LE u64.
                bytes.extend_from_slice(&[64, 2, 2, 8]);
            }
            Ok(digest(&bytes))
        };
        Ok(BinaryCommitmentProfiles {
            a: encode(SetupStreamRole::Commitment, a_columns)?,
            b: encode(SetupStreamRole::ImageCommitment, b_columns)?,
        })
    }

    pub fn a_label(&self) -> SetupStreamLabel {
        setup_label(
            self.setup_group,
            SetupStreamRole::Commitment,
            self.field_profile,
            self.b_rank,
        )
    }

    pub fn b_label(&self) -> SetupStreamLabel {
        setup_label(
            self.setup_group,
            SetupStreamRole::ImageCommitment,
            self.field_profile,
            self.b_rank,
        )
    }

    pub fn validate(&self) -> Result<(), AkitaError> {
        if self.b_rank
            != self.field_profile.digit_commitment_rank(
                SetupStreamRole::ImageCommitment,
                b_columns(
                    self.descriptor.index_map(),
                    self.descriptor.polynomials().len(),
                )?,
            )?
        {
            return Err(AkitaError::InvalidInput("binary B rank mismatch".into()));
        }
        if self.descriptor.profiles()
            != &Self::profiles_for_field(
                self.descriptor.index_map(),
                self.descriptor.polynomials().len(),
                self.setup_group,
                self.field_profile,
            )?
        {
            return Err(AkitaError::InvalidInput(
                "binary manifest setup profile mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_carrier(&self, profile: &GroupCommitPhaseParams) -> Result<(), AkitaError> {
        self.validate()?;
        if self.carrier_profile_digest != Self::carrier_profile_identity(profile)?
            || [
                profile.inner.matrix.sis_modulus_profile(),
                profile.outer.matrix.sis_modulus_profile(),
            ]
            .iter()
            .any(|id| !id.matches_modulus(self.field_profile.modulus()))
        {
            return Err(AkitaError::InvalidInput(
                "binary suffix carrier profile mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn carrier_profile_identity(
        profile: &GroupCommitPhaseParams,
    ) -> Result<[u8; 32], AkitaError> {
        let mut bytes = b"akita/binary-source/suffix-carrier/v1".to_vec();
        crate::proof::commitment::wire::serialize_profile(profile, &mut bytes)
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        Ok(digest(&bytes))
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = match self.field_profile {
            BinaryFieldProfile::P128 => b"AkitaBinaryCommitment\x01\x01".to_vec(),
            BinaryFieldProfile::P64 => b"AkitaBinaryCommitment\x02\x01".to_vec(),
        };
        bytes.extend_from_slice(&self.setup_group.to_le_bytes());
        bytes.extend_from_slice(&self.carrier_profile_digest);
        bytes.extend_from_slice(&self.descriptor.canonical_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        let bad =
            || SerializationError::InvalidData("noncanonical binary commitment manifest".into());
        if bytes.len() > MAX_BINARY_COMMITTED_MANIFEST_BYTES {
            return Err(bad());
        }
        let (field_profile, rest) =
            if let Some(rest) = bytes.strip_prefix(b"AkitaBinaryCommitment\x01\x01") {
                (BinaryFieldProfile::P128, rest)
            } else if let Some(rest) = bytes.strip_prefix(b"AkitaBinaryCommitment\x02\x01") {
                (BinaryFieldProfile::P64, rest)
            } else {
                return Err(bad());
            };
        let mut reader = Cursor::new(rest);
        let mut group = [0; 8];
        let mut carrier_profile_digest = [0; 32];
        reader.read_exact(&mut group)?;
        reader.read_exact(&mut carrier_profile_digest)?;
        let position = usize::try_from(reader.position()).map_err(|_| bad())?;
        let descriptor = BinarySourceDescriptor::from_bytes(rest.get(position..).ok_or_else(bad)?)?;
        let b_rank = field_profile
            .digit_commitment_rank(
                SetupStreamRole::ImageCommitment,
                b_columns(descriptor.index_map(), descriptor.polynomials().len())
                    .map_err(|_| bad())?,
            )
            .map_err(|_| bad())?;
        let manifest = Self {
            descriptor,
            field_profile,
            b_rank,
            setup_group: u64::from_le_bytes(group),
            carrier_profile_digest,
        };
        manifest.validate().map_err(|_| bad())?;
        Ok(manifest)
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut state = Shake256::default();
    state.update(bytes);
    let mut result = [0; 32];
    state.finalize_xof().read(&mut result);
    result
}

fn b_columns(index: &BinaryIndexMap, polynomials: usize) -> Result<usize, AkitaError> {
    checked::product([polynomials, index.columns(), A_RANK, 8, A_DEGREE])
        .and_then(|count| checked::exact_div(count, B_DEGREE))
        .ok_or_else(|| AkitaError::InvalidInput("binary B input width overflow".into()))
}

fn setup_label(
    group: u64,
    role: SetupStreamRole,
    field: BinaryFieldProfile,
    b_rank: usize,
) -> SetupStreamLabel {
    let (rank, degree, ring, modulus) = match role {
        SetupStreamRole::Commitment => (
            A_RANK as u32,
            A_DEGREE as u32,
            SetupStreamRing::MinusTrinomial,
            SetupStreamModulus::Q(BINARY_SOURCE_Q),
        ),
        _ => (
            b_rank as u32,
            B_DEGREE as u32,
            SetupStreamRing::Negacyclic,
            match field {
                BinaryFieldProfile::P128 => SetupStreamModulus::Prime128Offset275,
                BinaryFieldProfile::P64 => SetupStreamModulus::Prime64Offset59,
            },
        ),
    };
    SetupStreamLabel {
        role,
        group,
        level: 0,
        rank,
        row: 0,
        column: 0,
        ring,
        degree,
        modulus,
    }
}
