//! Binary word claims inside the ordinary grouped opening interface.

use super::{GroupClaimKind, PolynomialGroupClaims};
use crate::binary_source::BinaryWordClaim;
use crate::proof::commitment::CommittedGroupKind;
use crate::CommittedGroup;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

impl<'a, E: Field, F: Field + CanonicalEncoding> PolynomialGroupClaims<'a, E, CommittedGroup<F>> {
    /// Select the signed ordinary part of the research four-plus-four statement.
    /// Points retain the challenge-field type so extension terminals need no cast.
    pub fn new_research_ordinary(
        commitment: CommittedGroup<F>,
        terminals: crate::binary_source::ResearchTerminals<E>,
        owners: Vec<[u8; 32]>,
        width: akita_params::ordinary_source::OrdinarySourceWidth,
    ) -> Result<Self, AkitaError>
    where
        E: jolt_field::ExtField<F>,
    {
        let readings = terminals.validate::<F>()?;
        if !matches!(commitment.kind(), CommittedGroupKind::Ordinary)
            || owners.len() != commitment.profile().group.num_polynomials()
            || readings.iter().any(|terminal| {
                terminal.evaluations.len() != owners.len()
                    || terminal.point.len() != commitment.profile().group.num_vars()
            })
        {
            return Err(AkitaError::InvalidInput(
                "research ordinary owner geometry".into(),
            ));
        }
        let terminal = readings.first().ok_or(AkitaError::InvalidProof)?;
        let mut group = Self::new(
            terminal.point.clone(),
            terminal.evaluations.clone(),
            commitment,
        )?;
        group.kind = GroupClaimKind::ResearchOrdinary {
            owners,
            width,
            mode: terminals.mode,
        };
        group.research_terminals = Some(terminals);
        Ok(group)
    }

    /// Select one base reading or two E64 readings of one research binary source.
    pub fn new_research_binary(
        commitment: CommittedGroup<F>,
        terminals: crate::binary_source::ResearchTerminals<E>,
    ) -> Result<Self, AkitaError>
    where
        E: jolt_field::ExtField<F>,
    {
        Self::new_research_binary_with_word(commitment, terminals, None)
    }

    /// Select the hash source's word claim alongside its natural prime reading.
    /// Only group zero of the complete research roster may carry a word claim.
    pub fn new_research_binary_with_word(
        commitment: CommittedGroup<F>,
        terminals: crate::binary_source::ResearchTerminals<E>,
        claim: Option<BinaryWordClaim>,
    ) -> Result<Self, AkitaError>
    where
        E: jolt_field::ExtField<F>,
    {
        use crate::binary_source::{
            PrimeClaims, PrimeGroupClaim, PrimePlaneClaim, ResearchTerminalMode,
        };
        let readings = terminals.validate::<F>()?;
        let CommittedGroupKind::Binary(manifest) = commitment.kind() else {
            return Err(AkitaError::InvalidProof);
        };
        if manifest.field_profile() != akita_params::binary_setup::BinaryFieldProfile::P64
            || manifest.descriptor().polynomials().len() != 1
            || readings.iter().any(|terminal| terminal.evaluations.len() != 1)
        {
            return Err(AkitaError::InvalidInput(
                "research binary terminal owner count".into(),
            ));
        }
        let residue = |value: &E, index| {
            value
                .base_coefficient(index)
                .to_u128_checked()
                .ok_or(AkitaError::InvalidProof)
        };
        let prime = match terminals.mode {
            ResearchTerminalMode::BaseS11a => {
                let terminal = readings.first().ok_or(AkitaError::InvalidProof)?;
                PrimeClaims::Base(PrimeGroupClaim {
                    point: terminal
                        .point
                        .iter()
                        .map(|value| residue(value, 0))
                        .collect::<Result<Vec<_>, _>>()?,
                    evaluations: vec![PrimePlaneClaim {
                        polynomial: 0,
                        value: residue(
                            terminal.evaluations.first().ok_or(AkitaError::InvalidProof)?,
                            0,
                        )?,
                    }],
                })
            }
            ResearchTerminalMode::E64TwoPointS11a => {
                let coordinates = |value: &E| -> Result<[u64; 2], AkitaError> {
                    Ok([
                        u64::try_from(residue(value, 0)?).map_err(|_| AkitaError::InvalidProof)?,
                        u64::try_from(residue(value, 1)?).map_err(|_| AkitaError::InvalidProof)?,
                    ])
                };
                PrimeClaims::Extension(
                    readings
                        .iter()
                        .map(|terminal| {
                            Ok(PrimeGroupClaim {
                                point: terminal
                                    .point
                                    .iter()
                                    .map(coordinates)
                                    .collect::<Result<Vec<_>, AkitaError>>()?,
                                evaluations: vec![PrimePlaneClaim {
                                    polynomial: 0,
                                    value: coordinates(
                                        terminal
                                            .evaluations
                                            .first()
                                            .ok_or(AkitaError::InvalidProof)?,
                                    )?,
                                }],
                            })
                        })
                        .collect::<Result<Vec<_>, AkitaError>>()?,
                )
            }
        };
        if claim.is_some() && manifest.setup_group() != 0 {
            return Err(AkitaError::InvalidInput(
                "research word claim requires binary group zero".into(),
            ));
        }
        let mut group = Self::new_binary_planes(commitment, claim, Some(prime))?;
        group.research_terminals = Some(terminals);
        Ok(group)
    }

    /// Select one logical word claim over GF(2^128).
    ///
    /// The ordinary point and evaluation vectors are zero carrier geometry for
    /// suffix schedule selection. Only the typed word claim is the binary claim.
    /// Use `map_commitment` to borrow the same frozen claim for verification.
    /// The opening runs the field switch and authenticates its carrier terminal.
    /// Multiple points on one polynomial must be reduced by the application;
    /// the opening rejects a second claim with the same polynomial identifier.
    pub fn new_binary(
        commitment: CommittedGroup<F>,
        claim: BinaryWordClaim,
    ) -> Result<Self, AkitaError> {
        Self::new_binary_planes(commitment, Some(claim), None)
    }

    /// Select optional word and base or extension prime claims in one grouped opening.
    pub fn new_binary_planes(
        commitment: CommittedGroup<F>,
        claim: Option<BinaryWordClaim>,
        primes: Option<crate::binary_source::PrimeClaims>,
    ) -> Result<Self, AkitaError> {
        let CommittedGroupKind::Binary(manifest) = commitment.kind() else {
            return Err(AkitaError::InvalidInput(
                "binary claim requires a binary commitment".into(),
            ));
        };
        if let Some(claim) = &claim {
            claim.validate(manifest.descriptor())?;
        }
        let prime = crate::binary_source::PrimePlaneManifest::with_field_profile(
            manifest.descriptor(),
            primes,
            manifest.field_profile(),
        )?;
        if claim.is_none() && prime.planes().is_empty() {
            return Err(AkitaError::InvalidInput(
                "binary group has no claims".into(),
            ));
        }
        commitment
            .profile()
            .validate_frozen_precommit(F::MODULUS_BITS)?;
        manifest.validate_carrier(commitment.profile())?;
        let manifest = manifest.clone();
        let layout = commitment.profile().group;
        let mut point = Vec::new();
        point
            .try_reserve_exact(layout.num_vars())
            .map_err(|_| AkitaError::InvalidProof)?;
        point.resize(layout.num_vars(), E::zero());
        let mut evaluations = Vec::new();
        evaluations
            .try_reserve_exact(layout.num_polynomials())
            .map_err(|_| AkitaError::InvalidProof)?;
        evaluations.resize(layout.num_polynomials(), E::zero());
        Ok(Self {
            point: point.into(),
            evaluations,
            commitment,
            research_terminals: None,
            kind: GroupClaimKind::Binary {
                manifest: Box::new(manifest),
                claim,
                prime,
            },
        })
    }
}
