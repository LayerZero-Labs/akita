//! Standalone binary evaluation openings and Akita-backed root oracles.

mod binding;
mod oracle;
mod verifier;
use oracle::{RootPcsProverOracle, RootPcsVerifierOracle};

use akita_algebra::binary::field_switch::SwitchField;
use akita_config::{ensure_prover_schedule_fits_setup, TrustedScheduleCatalog};
use akita_cpu_backend::AkitaProverSetup;
use akita_error::AkitaError;
use akita_labinius_prover::prove_root_reduction_bytes;
use akita_labinius_verifier::root::{
    verify_root_reduction_bytes, RootProverOracle, RootVerifierOracle,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::Valid;
use akita_types::{AkitaVerifierSetup, CommittedGroup};
use akita_verifier::AkitaVerifier;

use crate::{
    config::DigitConfig, ImageCommitOutput, ImageConfig, ImageProver, PreparedRoot, RootSetup, F,
};
use binding::{admit_catalogs, resolve_rows};
use verifier::GroupedVerifier;

/// Prepared root, caller-trusted catalogs and one owning Akita CPU backend.
pub struct RootPcsProver<C: DigitConfig> {
    admitted: RootSetup,
    prepared: PreparedRoot,
    image: ImageProver,
    digits: AkitaCommitmentScheme<C>,
}
impl<C: DigitConfig> RootPcsProver<C> {
    /// Admit the catalog union and prepare the root before processing a proof.
    pub fn new(
        admitted: RootSetup,
        image_schedules: TrustedScheduleCatalog<ImageConfig>,
        digit_schedules: TrustedScheduleCatalog<C>,
        setup: AkitaProverSetup<F>,
    ) -> Result<Self, AkitaError> {
        setup
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        admit_catalogs(
            &admitted,
            &image_schedules,
            &digit_schedules,
            setup.expanded.descriptor(),
        )?;
        let (image_row, digit_row) = resolve_rows(&admitted, &image_schedules, &digit_schedules)?;
        for row in [image_row, digit_row] {
            crate::setup::ensure_setup_prefix_coverage(row, |id| {
                setup.prefix_slots.get(id).is_some()
            })?;
        }
        ensure_prover_schedule_fits_setup::<ImageConfig>(
            &setup.expanded,
            image_row.schedule(),
            &image_row.profiles().opening_layout()?,
        )?;
        ensure_prover_schedule_fits_setup::<C>(
            &setup.expanded,
            digit_row.schedule(),
            &digit_row.profiles().opening_layout()?,
        )?;
        let prepared = PreparedRoot::prepare(admitted.setup())?;
        Ok(Self {
            admitted,
            prepared,
            image: ImageProver::new(image_schedules, setup)?,
            digits: AkitaCommitmentScheme::new(digit_schedules),
        })
    }

    /// Commit the padded image through the canonical scalar image path.
    pub fn commit<H: SwitchField>(
        &self,
        source: &[H::Source],
    ) -> Result<ImageCommitOutput, AkitaError>
    where
        H::Source: Sync,
    {
        self.image
            .commit::<H>(&self.admitted, self.prepared.commit(), source)
    }

    /// Prepare a single-use oracle for a caller-owned root channel.
    pub fn oracle<'a>(
        &'a self,
        committed: &'a ImageCommitOutput,
    ) -> Result<impl RootProverOracle<F> + 'a, AkitaError> {
        RootPcsProverOracle::new(self, committed)
    }

    /// Standalone proof bytes for `f(point) = value`; owns the session and EOF.
    pub fn open<H: SwitchField>(
        &self,
        source: &[H::Source],
        committed: &ImageCommitOutput,
        point: &[H],
        value: H,
    ) -> Result<Vec<u8>, AkitaError>
    where
        H::Source: Sync,
    {
        let mut oracle = self.oracle(committed)?;
        let (proof, _) = prove_root_reduction_bytes(
            &self.admitted,
            &self.prepared,
            C::BASE,
            source,
            &committed.image,
            point,
            value,
            &mut oracle,
        )?;
        Ok(proof)
    }
}

/// Admitted root, caller-trusted catalogs and prepared Akita verifier state.
pub struct RootPcsVerifier<C: DigitConfig> {
    admitted: RootSetup,
    image_schedules: TrustedScheduleCatalog<ImageConfig>,
    digit_schedules: TrustedScheduleCatalog<C>,
    verifier: Box<dyn GroupedVerifier>,
}
impl<C: DigitConfig> RootPcsVerifier<C> {
    /// Resolve both rows and admit the shared setup before reading proof bytes.
    pub fn new(
        admitted: RootSetup,
        image_schedules: TrustedScheduleCatalog<ImageConfig>,
        digit_schedules: TrustedScheduleCatalog<C>,
        setup: AkitaVerifierSetup<F>,
    ) -> Result<Self, AkitaError> {
        setup
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        admit_catalogs(
            &admitted,
            &image_schedules,
            &digit_schedules,
            setup.expanded().descriptor(),
        )?;
        let (image_row, digit_row) = resolve_rows(&admitted, &image_schedules, &digit_schedules)?;
        for row in [image_row, digit_row] {
            crate::setup::ensure_setup_prefix_coverage(row, |id| {
                setup.prefix_slots().get(id).is_some()
            })?;
        }
        if !TrustedScheduleCatalog::<ImageConfig>::verifier_admits(setup.expanded(), image_row)?
            || !TrustedScheduleCatalog::<C>::verifier_admits(setup.expanded(), digit_row)?
        {
            return Err(AkitaError::InvalidSetup(
                "root PCS rows do not fit verifier setup".into(),
            ));
        }
        Ok(Self {
            admitted,
            image_schedules,
            verifier: Box::new(AkitaVerifier::new(setup, digit_schedules.clone())?),
            digit_schedules,
        })
    }

    /// Prepare a single-use oracle for a caller-owned root channel.
    pub fn oracle<'a>(
        &'a self,
        commitment: &'a CommittedGroup<F>,
    ) -> Result<impl RootVerifierOracle<F> + 'a, AkitaError> {
        RootPcsVerifierOracle::new(self, commitment)
    }

    /// Verify a complete binary evaluation opening, including exact EOF.
    pub fn verify<H: SwitchField>(
        &self,
        commitment: &CommittedGroup<F>,
        point: &[H],
        value: H,
        proof: &[u8],
    ) -> Result<(), AkitaError> {
        let mut oracle = self.oracle(commitment)?;
        verify_root_reduction_bytes(&self.admitted, C::BASE, point, value, &mut oracle, proof)?;
        Ok(())
    }
}
