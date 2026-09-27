//! Shared commitment-scheme API contracts.

use std::borrow::Cow;

/// Opening-point coordinates used by batched verification inputs.
pub type OpeningPoints<'a, F> = Cow<'a, [F]>;
