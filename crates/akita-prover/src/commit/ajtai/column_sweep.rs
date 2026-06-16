//! Cleartext column-selection kernels for the one-hot and sparse-ring A-side
//! commits.
//!
//! These kernels live next to their per-block entry types (`SingleChunkEntry`,
//! `MultiChunkEntry`, `SparseRingBlockEntry`) in `backend/`, which the
//! representations and the folding code also consume. The Ajtai commit
//! dispatch reaches them through this module so the commit subsystem has a
//! single named home for the column-sweep path.

pub(crate) use crate::backend::onehot::column_sweep_ajtai_onehot;
pub(crate) use crate::backend::sparse_ring::column_sweep_sparse;
