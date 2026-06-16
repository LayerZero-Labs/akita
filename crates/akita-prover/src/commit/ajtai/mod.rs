//! The Ajtai commit primitive: `commitment = commitment_key · message`.

pub(crate) mod backend;
mod column_sweep;
mod cpu;
pub(crate) mod opening;
pub(crate) mod spec;
