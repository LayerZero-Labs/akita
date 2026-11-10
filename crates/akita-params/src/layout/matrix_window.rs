//! Rectangular windows into an offset-zero row-major public stream.

use core::ops::Range;

use akita_error::{checked, AkitaError};

use super::NttPrefixRequirement;
use crate::zk::HiddenRows;

/// A rectangle in a ring-element prefix, with the full matrix row stride.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixWindow {
    /// Coefficients per ring element.
    pub ring_d: usize,
    /// Ring elements per full row of the backing prefix.
    pub stride: usize,
    /// Selected rows, relative to the start of the stream.
    pub rows: Range<usize>,
    /// Selected columns within each full row.
    pub columns: Range<usize>,
}

impl MatrixWindow {
    /// Full prefix needed to address this rectangle, including skipped columns.
    /// Rejects reversed ranges, columns past the stride, and empty or overflowing
    /// backing shapes. The selected rectangle itself may be empty.
    pub fn prefix_requirement(&self) -> Result<NttPrefixRequirement, AkitaError> {
        if self.rows.start > self.rows.end
            || self.columns.start > self.columns.end
            || self.columns.end > self.stride
        {
            return Err(AkitaError::InvalidSetup(
                "invalid matrix window ranges".into(),
            ));
        }
        let requirement =
            NttPrefixRequirement::from_matrix_shape(self.ring_d, self.rows.end, self.stride)?;
        requirement.num_field_elements()?;
        Ok(requirement)
    }
}

/// One Ajtai level on message columns followed by randomness columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AjtaiLevelShape {
    /// Coefficients per ring element.
    pub ring_d: usize,
    /// Published commitment rows.
    pub rank: usize,
    /// Columns containing message digits.
    pub message_columns: usize,
    /// Columns containing hiding randomness.
    pub randomness_columns: usize,
    /// Additional rows whose images the same randomness must hide.
    pub hidden: HiddenRows,
}

impl AjtaiLevelShape {
    /// Full row width, in ring elements.
    pub fn width(&self) -> Result<usize, AkitaError> {
        checked::sum([self.message_columns, self.randomness_columns])
            .ok_or_else(|| AkitaError::InvalidSetup("Ajtai level width overflow".into()))
    }

    /// Published rows, including both message and randomness columns.
    pub fn commitment_window(&self) -> Result<MatrixWindow, AkitaError> {
        let width = self.width()?;
        let window = MatrixWindow {
            ring_d: self.ring_d,
            stride: width,
            rows: 0..self.rank,
            columns: 0..width,
        };
        window.prefix_requirement()?;
        Ok(window)
    }

    /// Additional row restricted to randomness columns, when present.
    pub fn hidden_window(&self) -> Result<Option<MatrixWindow>, AkitaError> {
        let requirement = self.prefix_requirement()?;
        if self.hidden == HiddenRows::None {
            return Ok(None);
        }
        let width = self.width()?;
        Ok(Some(MatrixWindow {
            ring_d: requirement.ring_dimension,
            stride: width,
            rows: checked::range(self.rank, self.hidden.rows())
                .ok_or_else(|| AkitaError::InvalidSetup("Ajtai hidden row overflow".into()))?,
            columns: self.message_columns..width,
        }))
    }

    /// Prefix covering the published rows and any additional hidden row.
    pub fn prefix_requirement(&self) -> Result<NttPrefixRequirement, AkitaError> {
        let rows = checked::sum([self.rank, self.hidden.rows()])
            .ok_or_else(|| AkitaError::InvalidSetup("Ajtai level row count overflow".into()))?;
        let requirement =
            NttPrefixRequirement::from_matrix_shape(self.ring_d, rows, self.width()?)?;
        requirement.num_field_elements()?;
        Ok(requirement)
    }
}

#[cfg(test)]
mod tests;
