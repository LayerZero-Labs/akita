//! Explicit experimental commitment geometry for binary source groups.

use super::{
    BinaryCommitmentProfiles, BinaryIndexMap, BinarySourceDescriptor, BINARY_SOURCE_Q,
    MAX_BINARY_POLYNOMIALS,
};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_params::binary_setup::{
    SetupStreamLabel, SetupStreamModulus, SetupStreamRing, SetupStreamRole,
};
use akita_params::GroupCommitPhaseParams;
use akita_serialization::SerializationError;
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;
use std::io::{Cursor, Read};

/// Maximum encoded manifest accepted before allocating a descriptor buffer.
pub const MAX_BINARY_COMMITTED_MANIFEST_BYTES: usize = 132_000;
/// The uncompressed experimental B image has two degree-64 rows.
pub const BINARY_COMMITTED_PAYLOAD_COEFFICIENTS: usize = B_ROWS * B_DEGREE;
const A_RANK: usize = 3;
const A_DEGREE: usize = 648;
const B_ROWS: usize = 2;
const B_DEGREE: usize = 64;

/// Frozen, explicitly selected experimental binary commitment contract.
///
/// The q24 A matrix has rank three over the degree-648 minus trinomial ring.
/// The uncompressed P-valued B matrix has two degree-64 negacyclic rows. The
/// ordinary profile is only the carrier for the ordinary suffix; its digest is
/// separate from the real A/B profile identities in the source descriptor.
/// This runtime admission is for protocol tests and makes no security claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryCommittedGroupManifest {
    descriptor: BinarySourceDescriptor,
    setup_group: u64,
    carrier_profile_digest: [u8; 32],
}

impl BinaryCommittedGroupManifest {
    pub fn new(
        descriptor: BinarySourceDescriptor,
        setup_group: u64,
        carrier_profile: &GroupCommitPhaseParams,
    ) -> Result<Self, AkitaError> {
        let manifest = Self {
            descriptor,
            setup_group,
            carrier_profile_digest: Self::carrier_profile_identity(carrier_profile)?,
        };
        manifest.validate()?;
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
        B_ROWS
    }
    /// Compute the identities of the actual typed A/B setup contracts.
    pub fn commitment_profiles(
        index: &BinaryIndexMap,
        polynomials: usize,
        setup_group: u64,
    ) -> Result<BinaryCommitmentProfiles, AkitaError> {
        if polynomials == 0 || polynomials > MAX_BINARY_POLYNOMIALS {
            return Err(AkitaError::InvalidInput(
                "binary polynomial count out of range".into(),
            ));
        }
        let a_columns = checked::product([4, index.columns()])
            .and_then(|block| checked::exact_div(index.padded_carriers(), block))
            .ok_or_else(|| AkitaError::InvalidInput("binary A input width overflow".into()))?;
        let b_columns = checked::product([polynomials, index.columns(), A_RANK, 8, A_DEGREE])
            .and_then(|coefficients| checked::exact_div(coefficients, B_DEGREE))
            .ok_or_else(|| AkitaError::InvalidInput("binary B input width overflow".into()))?;
        let encode = |role: SetupStreamRole, columns: usize| -> Result<[u8; 32], AkitaError> {
            let label = setup_label(setup_group, role);
            let (modulus_tag, modulus, digits, input_bound) = match label.modulus {
                SetupStreamModulus::Q(q) => (1u8, u128::from(q), 1u32, 1u32),
                SetupStreamModulus::Prime128Offset275 => (2u8, BINARY_SETUP_P, 8u32, 4u32),
            };
            let columns = u64::try_from(columns)
                .map_err(|_| AkitaError::InvalidInput("binary setup width exceeds u64".into()))?;
            let mut bytes = b"akita/binary-source/experimental-profile/v1".to_vec();
            bytes.extend_from_slice(b"akita/binary-source/setup/shake256/v1");
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
            Ok(digest(&bytes))
        };
        Ok(BinaryCommitmentProfiles {
            a: encode(SetupStreamRole::Commitment, a_columns)?,
            b: encode(SetupStreamRole::ImageCommitment, b_columns)?,
        })
    }

    pub fn a_label(&self) -> SetupStreamLabel {
        setup_label(self.setup_group, SetupStreamRole::Commitment)
    }

    pub fn b_label(&self) -> SetupStreamLabel {
        setup_label(self.setup_group, SetupStreamRole::ImageCommitment)
    }

    pub fn validate(&self) -> Result<(), AkitaError> {
        if self.descriptor.profiles()
            != &Self::commitment_profiles(
                self.descriptor.index_map(),
                self.descriptor.polynomials().len(),
                self.setup_group,
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
        if self.carrier_profile_digest != Self::carrier_profile_identity(profile)? {
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
        let mut bytes = b"AkitaBinaryCommitment\x01\x01".to_vec();
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
        let rest = bytes
            .strip_prefix(b"AkitaBinaryCommitment\x01\x01")
            .ok_or_else(bad)?;
        let mut reader = Cursor::new(rest);
        let mut group = [0; 8];
        let mut carrier_profile_digest = [0; 32];
        reader.read_exact(&mut group)?;
        reader.read_exact(&mut carrier_profile_digest)?;
        let position = usize::try_from(reader.position()).map_err(|_| bad())?;
        let descriptor = BinarySourceDescriptor::from_bytes(rest.get(position..).ok_or_else(bad)?)?;
        let manifest = Self {
            descriptor,
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

fn setup_label(group: u64, role: SetupStreamRole) -> SetupStreamLabel {
    let (rank, degree, ring, modulus) = match role {
        SetupStreamRole::Commitment => (
            A_RANK as u32,
            A_DEGREE as u32,
            SetupStreamRing::MinusTrinomial,
            SetupStreamModulus::Q(BINARY_SOURCE_Q),
        ),
        _ => (
            B_ROWS as u32,
            B_DEGREE as u32,
            SetupStreamRing::Negacyclic,
            SetupStreamModulus::Prime128Offset275,
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
