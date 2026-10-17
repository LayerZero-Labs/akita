//! The supported root geometries and the errors of loading their catalogs.
//!
//! Applications distribute the artifact directory alongside their binaries and
//! pass its parent directory to `family::tables::shipped_catalog`. No planner
//! runs during loading.

use akita_error::AkitaError;
use akita_params::sis::labinius::LabiniusRootProfile;
use std::fmt;

/// One supported geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupportedGeometry {
    /// Root parameter profile identifying the admitted SIS geometry.
    pub profile: LabiniusRootProfile,
    /// Base-2 logarithm of the number of binary evaluation cells.
    pub log_num_cells: u32,
    /// Base-2 logarithm of the number of cells per fold.
    pub log_fold_width: u32,
    /// Fold soundness target, in security bits.
    pub lambda_fold: u32,
}

/// The single supported-geometry list used by loading, generation and tests.
pub const SUPPORTED_GEOMETRIES: [SupportedGeometry; 4] = {
    let geometry = SupportedGeometry {
        profile: LabiniusRootProfile::D648Q25BoundedW46,
        log_num_cells: 16,
        log_fold_width: 8,
        lambda_fold: 128,
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
    ]
};

/// Failure to load a supported external catalog.
#[derive(Debug)]
pub enum ShippedCatalogError {
    /// The requested profile or geometry has no shipped rows.
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
