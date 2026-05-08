//! Experimental zero-knowledge protocol prototypes for Akita.
//!
//! This crate intentionally stays close to existing Akita primitives. It does
//! not define new rings, fields, challenge samplers, transcripts, or
//! serialization formats. Measurement-only code that is not zero-knowledge is
//! exposed under [`measurements`], not the main protocol prelude.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

pub mod compact;
pub mod error;
pub mod measurements;
pub mod norm;
pub mod protocols;
pub mod rejection;
pub mod relations;
pub mod ring_ext;

pub use error::ZkResult;
