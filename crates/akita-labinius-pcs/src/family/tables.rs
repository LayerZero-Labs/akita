//! The committed tables of a root geometry and their shipped schedules.
//!
//! The image digit table is committed alone at commit time. The response digit
//! table is committed after the fold challenges. The image and optional prime
//! element table precede it as precommitted groups in one grouped Akita proof.

use super::FieldFamily;
use crate::shipped::{ShippedCatalogError, SupportedGeometry, SUPPORTED_GEOMETRIES};
use akita_config::{policy_of, CommitmentConfig, SetupRequirements, TrustedScheduleCatalog};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::admitted::field_characteristic;
use akita_params::{
    sis::labinius::LabiniusRootShape, GroupCommitPhaseParams, PolynomialGroupLayout,
    ScheduleLookupKey,
};
use akita_schedules::ValidatedScheduleCatalog;
use jolt_field::ExtField;
use std::path::Path;

/// Base-2 logarithms of the padded lengths of the committed tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DigitTables {
    /// Image digit table, bound at commit time.
    pub image_log_len: usize,
    /// Response digit table, bound after the fold challenges.
    pub response_log_len: usize,
    /// Prime left opening as one base-field table, coordinate innermost.
    pub element_log_len: usize,
}

impl DigitTables {
    /// Table lengths of one root geometry over the family's base field.
    ///
    /// Digit lengths come from the root shape's encoding for the base field's
    /// characteristic. The element length adds the lowest coordinate variable
    /// when the challenge field is a quadratic extension.
    ///
    /// # Errors
    ///
    /// Returns the shape's or the encoding's admission error, which includes a
    /// base field whose characteristic is not an admitted proof prime.
    pub fn for_geometry<P: FieldFamily>(geometry: SupportedGeometry) -> Result<Self, AkitaError> {
        let encoding = LabiniusRootShape::derive(
            geometry.profile,
            geometry.log_num_cells,
            geometry.log_fold_width,
            geometry.lambda_fold,
        )?
        .derive_encoding(field_characteristic::<P::Base>()?)?;
        let coordinate_vars = match P::Challenge::DEGREE {
            1 => 0,
            2 => 1,
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "prime opening supports extension degrees one and two".into(),
                ));
            }
        };
        let element_log_len = checked::sum([encoding.prime_table_log_len(), coordinate_vars])
            .ok_or_else(|| AkitaError::InvalidSetup("prime element table size overflow".into()))?;
        Ok(Self {
            image_log_len: encoding.image_table_log_len(),
            response_log_len: encoding.response_table_log_len(),
            element_log_len,
        })
    }

    /// Scalar schedule identity of the image table.
    pub fn image_key(&self) -> ScheduleLookupKey {
        ScheduleLookupKey::single(PolynomialGroupLayout::singleton(self.image_log_len))
    }

    /// Scalar response identity, the grouped planner's input row.
    pub fn scalar_response_key(&self) -> ScheduleLookupKey {
        ScheduleLookupKey::single(PolynomialGroupLayout::singleton(self.response_log_len))
    }

    /// Scalar schedule identity of the prime element table.
    pub fn element_key(&self) -> ScheduleLookupKey {
        ScheduleLookupKey::single(PolynomialGroupLayout::singleton(self.element_log_len))
    }

    /// Response identity with image first and the optional element producer second.
    ///
    /// # Errors
    ///
    /// Returns the catalog's lookup error when it has no image row, or
    /// `InvalidSetup` for a mismatched element layout or a failed allocation.
    pub fn response_key<C: CommitmentConfig>(
        &self,
        catalog: &TrustedScheduleCatalog<C>,
        element: Option<GroupCommitPhaseParams>,
    ) -> Result<ScheduleLookupKey, AkitaError> {
        let image = catalog.resolve_key(&self.image_key())?;
        let mut precommitteds = Vec::new();
        precommitteds
            .try_reserve_exact(if element.is_some() { 2 } else { 1 })
            .map_err(|_| AkitaError::InvalidSetup("digit table key allocation failed".into()))?;
        precommitteds.push(image.profiles().final_group);
        if let Some(element) = element {
            if element.group != self.element_key().final_group {
                return Err(AkitaError::InvalidSetup(
                    "prime element producer has the wrong table layout".into(),
                ));
            }
            precommitteds.push(element);
        }
        Ok(ScheduleLookupKey {
            final_group: PolynomialGroupLayout::singleton(self.response_log_len),
            precommitteds,
        })
    }

    /// Setup requirements for binary openings or all three committed tables.
    ///
    /// `None` uses only the binary catalog rows and the existing two-polynomial
    /// capacity. An element catalog selects three-polynomial capacity and
    /// covers both element commitment and grouped opening matrix footprints.
    ///
    /// # Errors
    ///
    /// Returns an error when the catalog lacks the image or the grouped
    /// response row, or when sizing overflows.
    pub fn setup_requirements<P: FieldFamily>(
        &self,
        digits: &TrustedScheduleCatalog<P::Digits>,
        elements: Option<&TrustedScheduleCatalog<P::Elements>>,
    ) -> Result<SetupRequirements<P::Base>, AkitaError> {
        let max_num_vars = self.image_log_len.max(self.response_log_len);
        if let Some(elements) = elements {
            let producer = elements
                .resolve_key(&self.element_key())?
                .profiles()
                .final_group;
            digits.resolve_key(&self.response_key(digits, Some(producer))?)?;
            let max_num_vars = max_num_vars.max(self.element_log_len);
            let digits = SetupRequirements::from_catalog(digits, max_num_vars, 3)?;
            return digits.union(SetupRequirements::from_catalog(elements, max_num_vars, 3)?);
        }
        digits.resolve_key(&self.response_key(digits, None)?)?;
        // Sizing scans producer commitments in every catalog row, including
        // rows whose complete batch exceeds the requested capacity. Remove
        // the optional three-group rows to keep binary setup bytes unchanged.
        let mut binary_rows = Vec::new();
        binary_rows
            .try_reserve_exact(digits.rows().len())
            .map_err(|_| {
                AkitaError::InvalidSetup("binary sizing catalog allocation failed".into())
            })?;
        for row in digits.rows() {
            if row.profiles().precommitteds.len() <= 1 {
                binary_rows.push((row.profiles().clone(), row.schedule().clone()));
            }
        }
        let binary = TrustedScheduleCatalog::<P::Digits>::new(ValidatedScheduleCatalog::try_new(
            P::Digits::schedule_family_name(),
            binary_rows,
            &policy_of::<P::Digits>(),
            P::Digits::ring_challenge_config,
        )?)?;
        SetupRequirements::from_catalog(&binary, max_num_vars, 2)
    }
}

/// Admitted table dimensions and producer catalogs for one field family.
pub struct ShippedCatalogs<P: FieldFamily> {
    /// Dimensions of the geometry's committed tables.
    pub tables: DigitTables,
    /// Image, response, and grouped opening schedules.
    pub digits: TrustedScheduleCatalog<P::Digits>,
    /// Prime element commitment schedules.
    pub elements: TrustedScheduleCatalog<P::Elements>,
}

/// Load a family's digit and element catalogs and resolve a supported geometry.
///
/// `artifact_root` contains `schedules-labinius` (or `schedules-labinius-dev`
/// when the workspace selects the dev protocol). No planner runs during loading.
///
/// # Errors
///
/// Returns a typed unsupported-geometry error before reading the file, or the
/// I/O, artifact admission or exact row lookup error.
pub fn shipped_catalog<P: FieldFamily>(
    geometry: SupportedGeometry,
    artifact_root: &Path,
) -> Result<ShippedCatalogs<P>, ShippedCatalogError> {
    if !SUPPORTED_GEOMETRIES.contains(&geometry) {
        return Err(ShippedCatalogError::UnsupportedGeometry);
    }
    let tables =
        DigitTables::for_geometry::<P>(geometry).map_err(ShippedCatalogError::Admission)?;
    let directory = artifact_root.join(if akita_params::DEV_PROTOCOL {
        "schedules-labinius-dev"
    } else {
        "schedules-labinius"
    });
    let bytes = std::fs::read(directory.join(format!("{}.aks", P::Digits::schedule_family_name())))
        .map_err(ShippedCatalogError::Io)?;
    let catalog = TrustedScheduleCatalog::<P::Digits>::from_artifact_bytes(&bytes)
        .map_err(ShippedCatalogError::Admission)?;
    let bytes =
        std::fs::read(directory.join(format!("{}.aks", P::Elements::schedule_family_name())))
            .map_err(ShippedCatalogError::Io)?;
    let elements = TrustedScheduleCatalog::<P::Elements>::from_artifact_bytes(&bytes)
        .map_err(ShippedCatalogError::Admission)?;
    let producer = elements
        .resolve_key(&tables.element_key())
        .map_err(ShippedCatalogError::Admission)?
        .profiles()
        .final_group;
    let key = tables
        .response_key(&catalog, None)
        .map_err(ShippedCatalogError::Admission)?;
    catalog
        .resolve_key(&key)
        .map_err(ShippedCatalogError::Admission)?;
    let key = tables
        .response_key(&catalog, Some(producer))
        .map_err(ShippedCatalogError::Admission)?;
    catalog
        .resolve_key(&key)
        .map_err(ShippedCatalogError::Admission)?;
    Ok(ShippedCatalogs {
        tables,
        digits: catalog,
        elements,
    })
}
