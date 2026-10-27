//! Linear algebra helpers for ring commitment.

#[cfg(test)]
use crate::sources::block_digits::LiveRings;
use crate::sources::block_digits::{DigitRows, DigitShape, I8Rows};
use akita_algebra::ntt::{MontCoeff, PrimeWidth, I32_LAZY_DOT_BATCH};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::{
    cyclic_ntt_with_i16_tail_to_ring, ntt_with_i16_tail_to_ring, CenteredMontLut, CrtNttParamSet,
    CyclotomicCrtNtt, CyclotomicRing, DigitMontLut, I16TailParams,
};
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::array::from_fn;
use std::mem::size_of;
use std::ops::Range;

use akita_types::{select_crt_ntt_params, FieldCrtBases, PreparedNttCache, ProtocolCrtNttParams};

mod block_parallel;
mod capacity;
mod chunked_matvec;
mod common;
mod compression;
mod crt_matvec;
mod decompose;
mod digit_relation;
mod digits;
mod fused_quotients;
mod ntt_matvec;
mod single_cyclic;
#[cfg(test)]
mod tests;

use block_parallel::*;
use capacity::*;
pub(crate) use capacity::{selected_crt_i8_capacity_profile, CrtI8CapacityProfile};
use chunked_matvec::*;
pub(crate) use common::centered_rows_abs_bound;
use common::*;
pub(crate) use compression::validate_compression_batch_shape;
#[cfg(test)]
use crt_matvec::precompute_dense_mat_ntt_with_params;
#[cfg(test)]
pub(crate) use crt_matvec::{mat_vec_mul_crt_ntt, mat_vec_mul_crt_ntt_many, mat_vec_mul_unchecked};
#[cfg(test)]
pub(crate) use decompose::decompose_block_i8;
pub(crate) use decompose::decompose_commit_blocks_into;
pub use decompose::decompose_rows_i8_into;
pub(crate) use digit_relation::{
    digit_relation_matrix_extent, digit_relation_rows_cached_prover_bounds,
    digit_relation_rows_streamed_prover_bounds, DigitRelationRows,
};
pub(crate) use digits::I8CommitPolicy;
use digits::*;
#[cfg(test)]
pub(crate) use fused_quotients::fused_split_eq_quotients;
pub use fused_quotients::{
    centered_quotient_rows_with_i16_tail, fused_split_eq_quotients_prover_bounds, FusedQuotientRows,
};
pub(crate) use fused_quotients::{
    fused_quotient_matrix_extent, fused_split_eq_quotients_streamed_prover_bounds,
};
pub(crate) use ntt_matvec::mat_vec_mul_ntt_blocks_i8;
pub use ntt_matvec::mat_vec_mul_ntt_digits_i8;
pub(crate) use single_cyclic::{mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic};
