//! External, audited schedule catalogs for supported root geometries.
//!
//! Applications distribute the artifact directory alongside their binaries and
//! pass its parent directory to [`catalogs`]. No planner runs during loading.

use crate::{config::DigitConfig, sizing::RootPcsSizing, ImageConfig};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootProfile};
use std::{fmt, path::Path};

/// One supported geometry and its admitted digit bases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupportedGeometry {
    pub profile: LabiniusRootProfile,
    pub log_num_cells: u32,
    pub log_fold_width: u32,
    pub lambda_fold: u32,
    pub bases: [LabiniusDigitBase; 3],
}

/// The single supported-geometry list used by loading, generation and tests.
pub const SUPPORTED_GEOMETRIES: [SupportedGeometry; 5] = {
    let geometry = SupportedGeometry {
        profile: LabiniusRootProfile::D648P128BoundedW46Delta16,
        log_num_cells: 16,
        log_fold_width: 8,
        lambda_fold: 128,
        bases: [
            LabiniusDigitBase::Bits1,
            LabiniusDigitBase::Bits2,
            LabiniusDigitBase::Bits4,
        ],
    };
    [
        geometry,
        SupportedGeometry {
            log_num_cells: 18,
            ..geometry
        },
        SupportedGeometry {
            log_num_cells: 20,
            ..geometry
        },
        SupportedGeometry {
            log_num_cells: 22,
            ..geometry
        },
        SupportedGeometry {
            log_num_cells: 24,
            ..geometry
        },
    ]
};

/// Failure to load a supported external catalog pair.
#[derive(Debug)]
pub enum ShippedCatalogError {
    /// The requested profile, geometry or digit base has no shipped rows.
    UnsupportedGeometry,
    /// The artifact file could not be read.
    Io(std::io::Error),
    /// The canonical artifact admission or exact row lookup rejected the file.
    Admission(AkitaError),
}

impl fmt::Display for ShippedCatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedGeometry => formatter.write_str("unsupported shipped root geometry"),
            Self::Io(error) => write!(formatter, "read shipped schedule artifact: {error}"),
            Self::Admission(error) => write!(formatter, "admit shipped schedule artifact: {error}"),
        }
    }
}

impl std::error::Error for ShippedCatalogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnsupportedGeometry => None,
            Self::Io(error) => Some(error),
            Self::Admission(error) => Some(error),
        }
    }
}

/// Load both catalogs and resolve the exact image producer and grouped digit row.
///
/// `artifact_root` contains `schedules-labinius` (or `schedules-labinius-dev`
/// when the workspace selects the dev protocol). Artifact bytes cross the same
/// audited boundary as ordinary Akita shipped catalogs.
///
/// # Errors
///
/// Returns a typed unsupported-geometry error before reading files, or the I/O,
/// artifact admission or exact row lookup error.
pub fn catalogs<C: DigitConfig>(
    sizing: &RootPcsSizing,
    artifact_root: &Path,
) -> Result<
    (
        TrustedScheduleCatalog<ImageConfig>,
        TrustedScheduleCatalog<C>,
    ),
    ShippedCatalogError,
> {
    if C::BASE != sizing.base()
        || !SUPPORTED_GEOMETRIES.iter().any(|geometry| {
            geometry.profile == sizing.profile()
                && geometry.log_num_cells == sizing.log_num_cells()
                && geometry.log_fold_width == sizing.log_fold_width()
                && geometry.lambda_fold == sizing.lambda_fold()
                && geometry.bases.contains(&sizing.base())
        })
    {
        return Err(ShippedCatalogError::UnsupportedGeometry);
    }
    let directory = artifact_root.join(if akita_params::DEV_PROTOCOL {
        "schedules-labinius-dev"
    } else {
        "schedules-labinius"
    });
    let image_bytes =
        std::fs::read(directory.join(format!("{}.aks", ImageConfig::schedule_family_name())))
            .map_err(ShippedCatalogError::Io)?;
    let digit_bytes = std::fs::read(directory.join(format!("{}.aks", C::schedule_family_name())))
        .map_err(ShippedCatalogError::Io)?;
    let images = TrustedScheduleCatalog::<ImageConfig>::from_artifact_bytes(&image_bytes)
        .map_err(ShippedCatalogError::Admission)?;
    let digits = TrustedScheduleCatalog::<C>::from_artifact_bytes(&digit_bytes)
        .map_err(ShippedCatalogError::Admission)?;
    let key = sizing
        .grouped_digit_key(&images)
        .map_err(ShippedCatalogError::Admission)?;
    digits
        .resolve_key(&sizing.scalar_digit_key())
        .map_err(ShippedCatalogError::Admission)?;
    digits
        .resolve_key(&key)
        .map_err(ShippedCatalogError::Admission)?;
    Ok((images, digits))
}
