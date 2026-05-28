//! Linear algebra helpers for ring commitment.

use akita_algebra::ntt::MontCoeff;
use akita_algebra::ntt::PrimeWidth;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::{
    CenteredMontLut, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut,
};
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore, HalvingField};
use std::array::from_fn;
use std::mem::size_of;

use crate::kernels::crt_ntt::NttSlotCache;
#[cfg(all(test, not(feature = "zk")))]
use crate::kernels::crt_ntt::{select_crt_ntt_params, ProtocolCrtNttParams};
#[cfg(all(test, not(feature = "zk")))]
use akita_field::AkitaError;

mod block_parallel;
mod common;
mod crt_matvec;
mod decompose;
mod digits;
mod fused_quotients;
mod i8_matvec;
mod ntt_matvec;
mod single_cyclic;
#[cfg(all(test, not(feature = "zk")))]
mod tests;

use block_parallel::*;
use common::*;
#[cfg(all(test, not(feature = "zk")))]
use crt_matvec::precompute_dense_mat_ntt_with_params;
pub use crt_matvec::unreduced_quotient_rows_ntt_cached;
#[cfg(all(test, not(feature = "zk")))]
pub(crate) use crt_matvec::{mat_vec_mul_crt_ntt, mat_vec_mul_crt_ntt_many, mat_vec_mul_unchecked};
pub use decompose::{
    decompose_block, decompose_block_i8, decompose_rows_i8, decompose_rows_i8_into, try_centered_i8,
};
use digits::*;
pub use fused_quotients::fused_split_eq_quotients;
use i8_matvec::*;
pub use ntt_matvec::{
    mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_digits_i8_strided,
    mat_vec_mul_ntt_i8, mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row,
    mat_vec_mul_ntt_i8_strided,
};
pub use single_cyclic::{mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic};
