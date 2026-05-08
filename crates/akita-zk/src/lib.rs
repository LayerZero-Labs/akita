//! Experimental zero-knowledge protocols for Akita.
//!
//! This crate intentionally stays close to existing Akita primitives. It does
//! not define new rings, fields, challenge samplers, transcripts, or
//! serialization formats.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

pub mod compact;
pub mod error;
pub mod norm;
pub mod protocols;
pub mod rejection;
pub mod relations;
pub mod ring_ext;

pub use error::ZkResult;
