//! Ring-native §4.1 commitment core implementation.

use super::config::{
    ensure_block_layout, ensure_matrix_shape, ensure_matrix_shape_ge, ensure_supported_num_vars,
    validate_and_derive_layout, HachiCommitmentLayout,
};
use super::onehot::{inner_ajtai_onehot_t_only, map_onehot_to_sparse_blocks};
use super::scheme::{CommitWitness, RingCommitmentScheme};
use super::types::RingCommitment;
use super::utils::crt_ntt::{build_ntt_cache, NttMatrixCache};
use super::utils::linear::{decompose_block, decompose_rows, mat_vec_mul_ntt_cached, MatrixSlot};
use super::utils::matrix::{derive_public_matrix, sample_public_matrix_seed, PublicMatrixSeed};
use super::CommitmentConfig;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::ring_switch::w_commitment_layout;
use crate::{cfg_into_iter, cfg_iter, CanonicalField, FieldCore, FieldSampling};
use std::io::{Read, Write};

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiSetupSeed {
    /// Maximum supported variable count.
    pub max_num_vars: usize,
    /// Runtime commitment layout.
    pub layout: HachiCommitmentLayout,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
}

/// Expanded setup stage containing coefficient-form matrices.
#[allow(non_snake_case)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiExpandedSetup<F: FieldCore, const D: usize> {
    /// Setup seed and runtime layout metadata.
    pub seed: HachiSetupSeed,
    /// Inner matrix `A`.
    pub A: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Outer matrix `B`.
    pub B: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Prover matrix `D ∈ R_q^{n_D × δ·2^R}` (§4.2).
    pub D: Vec<Vec<CyclotomicRing<F, D>>>,
}

/// Optional prepared setup stage for accelerated matrix-vector products.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiPreparedSetup<const D: usize> {
    /// Pre-converted CRT+NTT matrices for dense mat-vec paths.
    pub(crate) ntt_cache: NttMatrixCache<D>,
}

/// Prover setup artifact (expanded setup + optional runtime cache).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: HachiExpandedSetup<F, D>,
    /// Optional runtime-prepared acceleration cache.
    pub prepared: Option<HachiPreparedSetup<D>>,
}

/// Verifier setup artifact derived from prover setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiVerifierSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used for verification.
    pub expanded: HachiExpandedSetup<F, D>,
}

impl<F: FieldCore, const D: usize> HachiProverSetup<F, D> {
    /// Runtime layout carried by this setup.
    pub fn layout(&self) -> HachiCommitmentLayout {
        self.expanded.seed.layout
    }

    pub(crate) fn ntt_cache(&self) -> Result<&NttMatrixCache<D>, HachiError> {
        self.prepared
            .as_ref()
            .map(|p| &p.ntt_cache)
            .ok_or_else(|| HachiError::InvalidSetup("missing prepared NTT cache".to_string()))
    }
}

impl Valid for HachiSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        self.layout.check()
    }
}

impl HachiSerialize for HachiSetupSeed {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_num_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.layout.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress) + self.layout.serialized_size(compress) + 32
    }
}

impl HachiDeserialize for HachiSetupSeed {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let max_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let layout = HachiCommitmentLayout::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            layout,
            public_matrix_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for HachiExpandedSetup<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.seed.check()?;
        self.A.check()?;
        self.B.check()?;
        self.D.check()?;
        Ok(())
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiExpandedSetup<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.seed.serialize_with_mode(&mut writer, compress)?;
        self.A.serialize_with_mode(&mut writer, compress)?;
        self.B.serialize_with_mode(&mut writer, compress)?;
        self.D.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.seed.serialized_size(compress)
            + self.A.serialized_size(compress)
            + self.B.serialized_size(compress)
            + self.D.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for HachiExpandedSetup<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            seed: HachiSetupSeed::deserialize_with_mode(&mut reader, compress, validate)?,
            A: Vec::<Vec<CyclotomicRing<F, D>>>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            B: Vec::<Vec<CyclotomicRing<F, D>>>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            D: Vec::<Vec<CyclotomicRing<F, D>>>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for HachiProverSetup<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiProverSetup<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // Prepared cache is runtime-only and intentionally excluded.
        self.expanded.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for HachiProverSetup<F, D> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            expanded: HachiExpandedSetup::deserialize_with_mode(reader, compress, validate)?,
            prepared: None,
        })
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for HachiVerifierSetup<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiVerifierSetup<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.expanded.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for HachiVerifierSetup<F, D> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            expanded: HachiExpandedSetup::deserialize_with_mode(reader, compress, validate)?,
        })
    }
}

/// Concrete §4.1 commitment core.
#[derive(Clone, Copy, Default)]
pub struct HachiCommitmentCore;

impl<F, const D: usize, Cfg> RingCommitmentScheme<F, D, Cfg> for HachiCommitmentCore
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    type ProverSetup = HachiProverSetup<F, D>;
    type VerifierSetup = HachiVerifierSetup<F, D>;
    type Commitment = RingCommitment<F, D>;

    fn setup(max_num_vars: usize) -> Result<(Self::ProverSetup, Self::VerifierSetup), HachiError> {
        let layout = validate_and_derive_layout::<Cfg, D>(max_num_vars)?;
        ensure_supported_num_vars(max_num_vars, layout.required_num_vars::<D>()?)?;

        let w_layout = w_commitment_layout::<F, D, Cfg>(layout)?;
        let a_cols = layout.inner_width.max(w_layout.inner_width);
        let b_cols = layout.outer_width.max(w_layout.outer_width);

        let public_matrix_seed = sample_public_matrix_seed();
        let a_matrix = derive_public_matrix::<F, D>(Cfg::N_A, a_cols, &public_matrix_seed, b"A");
        let b_matrix = derive_public_matrix::<F, D>(Cfg::N_B, b_cols, &public_matrix_seed, b"B");
        let d_matrix = derive_public_matrix::<F, D>(
            Cfg::N_D,
            layout.d_matrix_width,
            &public_matrix_seed,
            b"D",
        );

        let ntt_cache = build_ntt_cache::<F, D>(&a_matrix, &b_matrix, &d_matrix)?;
        let expanded = HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout,
                public_matrix_seed,
            },
            A: a_matrix,
            B: b_matrix,
            D: d_matrix,
        };
        let prover_setup = HachiProverSetup {
            expanded: expanded.clone(),
            prepared: Some(HachiPreparedSetup { ntt_cache }),
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        ensure_matrix_shape_ge(&prover_setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape_ge(&prover_setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;
        ensure_matrix_shape(
            &prover_setup.expanded.D,
            Cfg::N_D,
            layout.d_matrix_width,
            "D",
        )?;
        Ok((prover_setup, verifier_setup))
    }

    fn layout(setup: &Self::ProverSetup) -> Result<HachiCommitmentLayout, HachiError> {
        Ok(setup.layout())
    }

    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
        ensure_supported_num_vars(
            setup.expanded.seed.max_num_vars,
            layout.required_num_vars::<D>()?,
        )?;
        ensure_block_layout(f_blocks, layout)?;
        ensure_matrix_shape_ge(&setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape_ge(&setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;

        let cache = setup.ntt_cache()?;
        let depth = layout.num_digits_commit;
        let log_basis = layout.log_basis;
        let t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = cfg_iter!(f_blocks)
            .map(|block| {
                let s_i = decompose_block(block, depth, log_basis);
                let t_i =
                    mat_vec_mul_ntt_cached(cache, MatrixSlot::A, &s_i).expect("inner Ajtai failed");
                decompose_rows(&t_i, depth, log_basis)
            })
            .collect();

        let t_hat_flat: Vec<CyclotomicRing<F, D>> =
            t_hat_all.iter().flat_map(|v| v.iter().copied()).collect();

        let u = mat_vec_mul_ntt_cached(cache, MatrixSlot::B, &t_hat_flat)?;
        Ok(CommitWitness {
            commitment: RingCommitment { u },
            t_hat: t_hat_all,
        })
    }

    fn commit_coeffs(
        f_coeffs: &[CyclotomicRing<F, D>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
        let num_blocks = layout.num_blocks;
        let block_len = layout.block_len;
        let max_len = num_blocks
            .checked_mul(block_len)
            .ok_or_else(|| HachiError::InvalidSetup("coefficient length overflow".to_string()))?;
        if f_coeffs.len() > max_len {
            return Err(HachiError::InvalidSize {
                expected: max_len,
                actual: f_coeffs.len(),
            });
        }

        let depth = layout.num_digits_commit;
        let log_basis = layout.log_basis;
        let zero_t_hat = vec![CyclotomicRing::<F, D>::zero(); Cfg::N_A.checked_mul(depth).unwrap()];
        let cache = setup.ntt_cache()?;
        let coeff_len = f_coeffs.len();

        let t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= coeff_len {
                    zero_t_hat.clone()
                } else {
                    let end = (start + block_len).min(coeff_len);
                    let block = &f_coeffs[start..end];
                    let s_i = decompose_block(block, depth, log_basis);
                    let t_i = mat_vec_mul_ntt_cached(cache, MatrixSlot::A, &s_i)
                        .expect("inner Ajtai failed");
                    decompose_rows(&t_i, depth, log_basis)
                }
            })
            .collect();

        let t_hat_flat: Vec<CyclotomicRing<F, D>> =
            t_hat_all.iter().flat_map(|v| v.iter().copied()).collect();

        let u = mat_vec_mul_ntt_cached(cache, MatrixSlot::B, &t_hat_flat)?;
        Ok(CommitWitness {
            commitment: RingCommitment { u },
            t_hat: t_hat_all,
        })
    }

    fn commit_onehot(
        onehot_k: usize,
        indices: &[Option<usize>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
        ensure_supported_num_vars(
            setup.expanded.seed.max_num_vars,
            layout.required_num_vars::<D>()?,
        )?;
        ensure_matrix_shape_ge(&setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape_ge(&setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;

        let sparse_blocks =
            map_onehot_to_sparse_blocks(onehot_k, indices, layout.r_vars, layout.m_vars, D)?;

        let depth = layout.num_digits_commit;
        let log_basis = layout.log_basis;
        let zero_t_hat = vec![CyclotomicRing::<F, D>::zero(); Cfg::N_A.checked_mul(depth).unwrap()];
        let cache = setup.ntt_cache()?;
        let a_matrix = &setup.expanded.A;
        let block_len = layout.block_len;

        let t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = cfg_iter!(sparse_blocks)
            .map(|block_entries| {
                if block_entries.is_empty() {
                    zero_t_hat.clone()
                } else {
                    let t_i = inner_ajtai_onehot_t_only(a_matrix, block_entries, block_len, depth);
                    decompose_rows(&t_i, depth, log_basis)
                }
            })
            .collect();

        let t_hat_flat: Vec<CyclotomicRing<F, D>> =
            t_hat_all.iter().flat_map(|v| v.iter().copied()).collect();

        let u = mat_vec_mul_ntt_cached(cache, MatrixSlot::B, &t_hat_flat)?;
        Ok(CommitWitness {
            commitment: RingCommitment { u },
            t_hat: t_hat_all,
        })
    }
}

impl HachiCommitmentCore {
    /// Create a setup with a caller-specified layout, bypassing
    /// `CommitmentConfig::commitment_layout`.
    ///
    /// Use this when the desired `(m_vars, r_vars)` split differs from what
    /// the config's heuristic would choose (e.g. mega-polynomial commitments
    /// where each sub-polynomial occupies one block).
    ///
    /// # Errors
    ///
    /// Returns `HachiError` on invalid layout or matrix generation failures.
    pub fn setup_with_layout<F, const D: usize, Cfg>(
        layout: HachiCommitmentLayout,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F, D>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig,
    {
        let max_num_vars = layout.required_num_vars::<D>()?;
        let public_matrix_seed = sample_public_matrix_seed();
        Self::setup_with_layout_and_seed::<F, D, Cfg>(layout, max_num_vars, public_matrix_seed)
    }

    /// Like `setup_with_layout` but reuses an existing setup's random seed and
    /// A matrix (which depends only on `m_vars`). Only regenerates B and D
    /// matrices for the new `r_vars`.
    ///
    /// Use this when creating a mega-polynomial setup that shares `m_vars` with
    /// an individual polynomial setup — avoids re-deriving and NTT-transforming
    /// the A matrix.
    ///
    /// # Errors
    ///
    /// Returns `HachiError` if the new layout is incompatible with the existing
    /// setup or matrix shapes are inconsistent.
    pub fn setup_from_existing<F, const D: usize, Cfg>(
        existing: &HachiExpandedSetup<F, D>,
        new_r_vars: usize,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F, D>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig,
    {
        let old_layout = existing.seed.layout;
        let new_layout = HachiCommitmentLayout::new::<Cfg>(
            old_layout.m_vars,
            new_r_vars,
            &Cfg::decomposition(),
        )?;

        if new_layout.inner_width != old_layout.inner_width {
            return Err(HachiError::InvalidSetup(
                "setup_from_existing requires matching m_vars/inner_width".to_string(),
            ));
        }

        let w_layout = w_commitment_layout::<F, D, Cfg>(new_layout)?;
        let a_width = existing.A.first().map_or(0, |r| r.len());
        if a_width < w_layout.inner_width {
            return Err(HachiError::InvalidSetup(format!(
                "existing A width {a_width} < w inner_width {}",
                w_layout.inner_width
            )));
        }
        let b_cols = new_layout.outer_width.max(w_layout.outer_width);

        let max_num_vars = new_layout.required_num_vars::<D>()?;
        let seed = existing.seed.public_matrix_seed;

        let b_matrix = derive_public_matrix::<F, D>(Cfg::N_B, b_cols, &seed, b"B");
        let d_matrix =
            derive_public_matrix::<F, D>(Cfg::N_D, new_layout.d_matrix_width, &seed, b"D");

        let ntt_cache = build_ntt_cache::<F, D>(&existing.A, &b_matrix, &d_matrix)?;
        let expanded = HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout: new_layout,
                public_matrix_seed: seed,
            },
            A: existing.A.clone(),
            B: b_matrix,
            D: d_matrix,
        };
        let prover_setup = HachiProverSetup {
            expanded: expanded.clone(),
            prepared: Some(HachiPreparedSetup { ntt_cache }),
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        Ok((prover_setup, verifier_setup))
    }

    fn setup_with_layout_and_seed<F, const D: usize, Cfg>(
        layout: HachiCommitmentLayout,
        max_num_vars: usize,
        public_matrix_seed: PublicMatrixSeed,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F, D>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig,
    {
        let w_layout = w_commitment_layout::<F, D, Cfg>(layout)?;
        let a_cols = layout.inner_width.max(w_layout.inner_width);
        let b_cols = layout.outer_width.max(w_layout.outer_width);

        let a_matrix = derive_public_matrix::<F, D>(Cfg::N_A, a_cols, &public_matrix_seed, b"A");
        let b_matrix = derive_public_matrix::<F, D>(Cfg::N_B, b_cols, &public_matrix_seed, b"B");
        let d_matrix = derive_public_matrix::<F, D>(
            Cfg::N_D,
            layout.d_matrix_width,
            &public_matrix_seed,
            b"D",
        );

        let ntt_cache = build_ntt_cache::<F, D>(&a_matrix, &b_matrix, &d_matrix)?;
        let expanded = HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout,
                public_matrix_seed,
            },
            A: a_matrix,
            B: b_matrix,
            D: d_matrix,
        };
        let prover_setup = HachiProverSetup {
            expanded: expanded.clone(),
            prepared: Some(HachiPreparedSetup { ntt_cache }),
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        ensure_matrix_shape_ge(&prover_setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape_ge(&prover_setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;
        ensure_matrix_shape(
            &prover_setup.expanded.D,
            Cfg::N_D,
            layout.d_matrix_width,
            "D",
        )?;
        Ok((prover_setup, verifier_setup))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{HachiDeserialize, HachiSerialize};
    use crate::test_utils::{TinyConfig, D as TestD, F as TestF};

    #[test]
    fn prover_setup_roundtrips_and_derives_same_verifier() {
        let (prover_setup, verifier_setup) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::setup(16)
                .unwrap();

        let mut bytes = Vec::new();
        prover_setup.serialize_compressed(&mut bytes).unwrap();
        let decoded = HachiProverSetup::<TestF, TestD>::deserialize_compressed(&bytes[..]).unwrap();

        assert_eq!(decoded.expanded, prover_setup.expanded);
        assert_eq!(decoded.prepared, None);

        let derived_verifier = HachiVerifierSetup {
            expanded: decoded.expanded.clone(),
        };
        assert_eq!(derived_verifier, verifier_setup);
    }
}
