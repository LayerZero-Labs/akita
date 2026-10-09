//! Root geometry, catalog lookup identities and wire sizing before any data exists.

use akita_algebra::binary::field_switch::SwitchField;
use akita_config::{policy_of, SetupRequirements, TrustedScheduleCatalog};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::root::root_reduction_wire_size;
use akita_params::{
    sis::labinius::{
        LabiniusDigitBase, LabiniusRootEncoding, LabiniusRootProfile, LabiniusRootShape,
    },
    PolynomialGroupLayout, ScheduleLookupKey,
};

use crate::{config::DigitConfig, ImageConfig, F};

/// Admitted root geometry and its canonical image and digit table domains.
///
/// Construction derives the shape and encoding without generating a root matrix,
/// planning schedules, or processing source data. Quantities that depend on the
/// chosen schedules are resolved against caller-trusted catalogs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootPcsSizing {
    shape: LabiniusRootShape,
    encoding: LabiniusRootEncoding,
    log_num_cells: u32,
    log_fold_width: u32,
    lambda_fold: u32,
    base: LabiniusDigitBase,
}

impl RootPcsSizing {
    /// Admit the geometry and derive its table sizes through the root shape owner.
    pub fn new(
        profile: LabiniusRootProfile,
        log_num_cells: u32,
        log_fold_width: u32,
        lambda_fold: u32,
        base: LabiniusDigitBase,
    ) -> Result<Self, AkitaError> {
        let shape = LabiniusRootShape::derive(profile, log_num_cells, log_fold_width, lambda_fold)?;
        let encoding = shape.derive_encoding(base)?;
        Ok(Self {
            shape,
            encoding,
            log_num_cells,
            log_fold_width,
            lambda_fold,
            base,
        })
    }

    /// Selected closed root profile.
    pub fn profile(&self) -> LabiniusRootProfile {
        self.shape.profile()
    }
    /// SIS-admitted root shape.
    pub fn shape(&self) -> &LabiniusRootShape {
        &self.shape
    }
    /// Canonical digit encoding, also used by `LoweredRootLayout`.
    pub fn encoding(&self) -> &LabiniusRootEncoding {
        &self.encoding
    }
    /// Logarithm of the source cell count.
    pub fn log_num_cells(&self) -> u32 {
        self.log_num_cells
    }
    /// Logarithm of the fold width.
    pub fn log_fold_width(&self) -> u32 {
        self.log_fold_width
    }
    /// Requested fold security parameter.
    pub fn lambda_fold(&self) -> u32 {
        self.lambda_fold
    }
    /// Reduction digit alphabet.
    pub fn base(&self) -> LabiniusDigitBase {
        self.base
    }
    /// Logarithm of the padded image table length.
    pub fn image_log_len(&self) -> usize {
        self.encoding.image_table_log_len()
    }
    /// Logarithm of the padded digit table length.
    pub fn digit_log_len(&self) -> usize {
        self.encoding.response_table_log_len()
    }
    /// Scalar image schedule identity.
    pub fn image_key(&self) -> ScheduleLookupKey {
        ScheduleLookupKey::single(PolynomialGroupLayout::singleton(self.image_log_len()))
    }
    /// Scalar digit schedule identity used as the grouped planner's input row.
    pub fn scalar_digit_key(&self) -> ScheduleLookupKey {
        ScheduleLookupKey::single(PolynomialGroupLayout::singleton(self.digit_log_len()))
    }
    /// Grouped digit identity with the exact resolved image producer descriptor.
    ///
    /// The precommit includes all commitment parameters, not just the image's
    /// displayed table length. It is never reconstructed from a textual key.
    pub fn grouped_digit_key(
        &self,
        images: &TrustedScheduleCatalog<ImageConfig>,
    ) -> Result<ScheduleLookupKey, AkitaError> {
        let image = images.resolve_key(&self.image_key())?;
        let mut precommitteds = Vec::new();
        precommitteds
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidSetup("root sizing key allocation failed".into()))?;
        precommitteds.push(image.profiles().final_group);
        Ok(ScheduleLookupKey {
            final_group: PolynomialGroupLayout::singleton(self.digit_log_len()),
            precommitteds,
        })
    }
    /// Union of the two catalogs' setup requirements at this root's capacity.
    pub fn setup_requirements<C: DigitConfig>(
        &self,
        images: &TrustedScheduleCatalog<ImageConfig>,
        digits: &TrustedScheduleCatalog<C>,
    ) -> Result<SetupRequirements<F>, AkitaError> {
        self.validate_base::<C>()?;
        digits.resolve_key(&self.grouped_digit_key(images)?)?;
        let max = self.image_log_len().max(self.digit_log_len());
        SetupRequirements::from_catalog(images, max, 2)?
            .union(SetupRequirements::from_catalog(digits, max, 2)?)
    }
    /// Exact serialized image commitment length, including its profile header.
    pub fn commitment_bytes(
        &self,
        images: &TrustedScheduleCatalog<ImageConfig>,
    ) -> Result<usize, AkitaError> {
        crate::root::commitment_size(
            images
                .resolve_key(&self.image_key())?
                .profiles()
                .final_group,
        )
    }
    /// Conservative complete opening length for the sealed binary host field.
    ///
    /// The root reduction owner sizes the reduction wire. The existing
    /// commitment owner sizes the response commitment, and the expanded schedule
    /// owner supplies exactly the parser bound used by `NestedOpeningSession`.
    /// Both oracle payloads have an eight-byte length frame.
    pub fn opening_proof_bound<C: DigitConfig, H: SwitchField>(
        &self,
        images: &TrustedScheduleCatalog<ImageConfig>,
        digits: &TrustedScheduleCatalog<C>,
    ) -> Result<usize, AkitaError> {
        self.validate_base::<C>()?;
        let key = self.grouped_digit_key(images)?;
        let row = digits.resolve_key(&key)?;
        checked::sum([
            root_reduction_wire_size::<H>(&self.shape, self.base)?,
            8,
            crate::root::commitment_size(row.profiles().final_group)?,
            8,
            akita_schedules::expanded_schedule_proof_bound(
                &key,
                row.schedule(),
                &policy_of::<C>(),
            )?,
        ])
        .ok_or_else(|| AkitaError::InvalidSetup("root opening bound overflow".into()))
    }

    fn validate_base<C: DigitConfig>(&self) -> Result<(), AkitaError> {
        if C::BASE != self.base {
            return Err(AkitaError::InvalidSetup(
                "root sizing digit base differs from the catalog configuration".into(),
            ));
        }
        Ok(())
    }
}
