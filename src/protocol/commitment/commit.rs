//! Ring-native §4.1 commitment core implementation.

use super::config::{
    ensure_block_layout, ensure_matrix_shape, ensure_supported_num_vars,
    validate_and_derive_layout, HachiCommitmentLayout,
};
use super::onehot::{inner_ajtai_onehot, map_onehot_to_sparse_blocks};
use super::scheme::{CommitWitness, RingCommitmentScheme};
use super::types::RingCommitment;
use super::utils::crt_ntt::{build_ntt_cache, NttMatrixCache};
use super::utils::linear::{
    decompose_block, decompose_rows, mat_vec_mul_ntt_cached, mat_vec_mul_ntt_many_cached,
    MatrixSlot,
};
use super::utils::matrix::{
    derive_public_matrix, sample_public_matrix_prg_backend, sample_public_matrix_seed,
    PublicMatrixSeed,
};
use super::utils::norm::detect_field_modulus;
use super::CommitmentConfig;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::prg::MatrixPrgBackendId;
use crate::{CanonicalField, FieldCore, FieldSampling};
use std::io::{Read, Write};

/// Seed-only stage for deterministic setup expansion.
#[allow(non_snake_case)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiSetupSeed {
    /// Maximum supported variable count.
    pub max_num_vars: usize,
    /// Runtime commitment layout.
    pub layout: HachiCommitmentLayout,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
    /// Selected matrix PRG backend.
    pub public_matrix_prg_backend: MatrixPrgBackendId,
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
    /// Prepared matrices and dimensions for ring-switch `w` commitments.
    pub(crate) w_commit: HachiWCommitPrepared<D>,
}

/// Runtime-prepared artifacts for ring-switch witness commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HachiWCommitPrepared<const D: usize> {
    /// Maximum ring block length for `w` commitments.
    pub(crate) block_len: usize,
    /// Maximum outer-width used for `w` commitments.
    pub(crate) outer_width: usize,
    /// Pre-converted CRT+NTT matrices for `w` commitment mat-vecs.
    pub(crate) ntt_cache: NttMatrixCache<D>,
}

/// Prover setup artifact (expanded setup + optional runtime cache).
#[allow(non_snake_case)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: HachiExpandedSetup<F, D>,
    /// Optional runtime-prepared acceleration cache.
    pub prepared: Option<HachiPreparedSetup<D>>,
}

/// Verifier setup artifact derived from prover setup.
#[allow(non_snake_case)]
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

    pub(crate) fn w_commit_prepared(&self) -> Result<&HachiWCommitPrepared<D>, HachiError> {
        self.prepared.as_ref().map(|p| &p.w_commit).ok_or_else(|| {
            HachiError::InvalidSetup("missing prepared w-commitment cache".to_string())
        })
    }
}

fn ring_switch_r_decomp_levels<F: CanonicalField, Cfg: CommitmentConfig>() -> usize {
    let modulus = detect_field_modulus::<F>();
    let bits = 128 - (modulus.saturating_sub(1)).leading_zeros() as usize;
    let log_basis = Cfg::LOG_BASIS as usize;
    let mut levels = (bits + log_basis.saturating_sub(1)) / log_basis.max(1);
    if levels == 0 {
        levels = 1;
    }

    let b = 1u128 << Cfg::LOG_BASIS;
    let half_q = modulus / 2;
    let half_b_minus_1 = b / 2 - 1;
    let b_minus_1 = b - 1;
    let mut b_pow = 1u128;
    for _ in 0..levels {
        b_pow = b_pow.saturating_mul(b);
    }
    let max_positive = half_b_minus_1.saturating_mul((b_pow - 1) / b_minus_1);
    if max_positive < half_q {
        levels += 1;
    }

    levels
}

fn derive_w_commit_block_len<F: CanonicalField, Cfg: CommitmentConfig>(
    layout: HachiCommitmentLayout,
) -> Result<usize, HachiError> {
    let w_hat_len = layout
        .num_blocks
        .checked_mul(layout.block_len)
        .and_then(|x| x.checked_mul(Cfg::DELTA))
        .ok_or_else(|| HachiError::InvalidSetup("w_hat length overflow".to_string()))?;
    let t_hat_len = layout
        .num_blocks
        .checked_mul(Cfg::N_A)
        .and_then(|x| x.checked_mul(Cfg::DELTA))
        .ok_or_else(|| HachiError::InvalidSetup("t_hat length overflow".to_string()))?;
    let z_hat_len = layout
        .block_len
        .checked_mul(Cfg::DELTA)
        .and_then(|x| x.checked_mul(Cfg::TAU))
        .ok_or_else(|| HachiError::InvalidSetup("z_hat length overflow".to_string()))?;

    let m_rows = Cfg::N_D
        .checked_add(Cfg::N_B)
        .and_then(|x| x.checked_add(2))
        .and_then(|x| x.checked_add(Cfg::N_A))
        .ok_or_else(|| HachiError::InvalidSetup("M row count overflow".to_string()))?;
    let r_levels = ring_switch_r_decomp_levels::<F, Cfg>();
    let r_hat_len = m_rows
        .checked_mul(r_levels)
        .ok_or_else(|| HachiError::InvalidSetup("r_hat length overflow".to_string()))?;

    let total_ring_elems = w_hat_len
        .checked_add(t_hat_len)
        .and_then(|x| x.checked_add(z_hat_len))
        .and_then(|x| x.checked_add(r_hat_len))
        .ok_or_else(|| HachiError::InvalidSetup("w length overflow".to_string()))?;

    Ok(total_ring_elems.next_power_of_two().max(1))
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
        writer.write_all(&[u8::from(self.public_matrix_prg_backend)])?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress) + self.layout.serialized_size(compress) + 32 + 1
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
        let mut backend_id = [0u8; 1];
        reader.read_exact(&mut backend_id)?;
        let public_matrix_prg_backend = MatrixPrgBackendId::try_from(backend_id[0])
            .map_err(|e| SerializationError::InvalidData(e.to_string()))?;
        let out = Self {
            max_num_vars,
            layout,
            public_matrix_seed,
            public_matrix_prg_backend,
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
        let public_matrix_seed = sample_public_matrix_seed();
        let public_matrix_prg_backend = sample_public_matrix_prg_backend();
        let a_matrix = derive_public_matrix::<F, D>(
            Cfg::N_A,
            layout.inner_width,
            &public_matrix_seed,
            b"A",
            public_matrix_prg_backend,
        );
        let b_matrix = derive_public_matrix::<F, D>(
            Cfg::N_B,
            layout.outer_width,
            &public_matrix_seed,
            b"B",
            public_matrix_prg_backend,
        );
        let d_matrix = derive_public_matrix::<F, D>(
            Cfg::N_D,
            layout.d_matrix_width,
            &public_matrix_seed,
            b"D",
            public_matrix_prg_backend,
        );

        let w_block_len = derive_w_commit_block_len::<F, Cfg>(layout)?;
        let w_inner_width = w_block_len
            .checked_mul(Cfg::DELTA)
            .ok_or_else(|| HachiError::InvalidSetup("w inner width overflow".to_string()))?;
        let w_outer_width = Cfg::N_A
            .checked_mul(Cfg::DELTA)
            .ok_or_else(|| HachiError::InvalidSetup("w outer width overflow".to_string()))?;
        let w_a_matrix = derive_public_matrix::<F, D>(
            Cfg::N_A,
            w_inner_width,
            &public_matrix_seed,
            b"A",
            public_matrix_prg_backend,
        );
        let w_b_matrix = derive_public_matrix::<F, D>(
            Cfg::N_B,
            w_outer_width,
            &public_matrix_seed,
            b"B",
            public_matrix_prg_backend,
        );

        let ntt_cache = build_ntt_cache::<F, D>(&a_matrix, &b_matrix, &d_matrix)?;
        let w_ntt_cache = build_ntt_cache::<F, D>(&w_a_matrix, &w_b_matrix, &d_matrix)?;
        let expanded = HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                layout,
                public_matrix_seed,
                public_matrix_prg_backend: public_matrix_prg_backend.backend_id(),
            },
            A: a_matrix,
            B: b_matrix,
            D: d_matrix,
        };
        let prover_setup = HachiProverSetup {
            expanded: expanded.clone(),
            prepared: Some(HachiPreparedSetup {
                ntt_cache,
                w_commit: HachiWCommitPrepared {
                    block_len: w_block_len,
                    outer_width: w_outer_width,
                    ntt_cache: w_ntt_cache,
                },
            }),
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        ensure_matrix_shape(&prover_setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape(&prover_setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;
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
        ensure_matrix_shape(&setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape(&setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;

        let s_all: Vec<Vec<CyclotomicRing<F, D>>> = f_blocks
            .iter()
            .map(|block| decompose_block(block, Cfg::DELTA, Cfg::LOG_BASIS))
            .collect();

        let t_all = mat_vec_mul_ntt_many_cached(setup.ntt_cache()?, MatrixSlot::A, &s_all)?;
        let mut t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(layout.num_blocks);
        let mut t_hat_flat: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(layout.outer_width);
        for t_i in t_all {
            let t_hat_i = decompose_rows(&t_i, Cfg::DELTA, Cfg::LOG_BASIS);
            t_hat_flat.extend(t_hat_i.iter().copied());
            t_hat_all.push(t_hat_i);
        }

        let u = mat_vec_mul_ntt_cached(setup.ntt_cache()?, MatrixSlot::B, &t_hat_flat)?;
        Ok(CommitWitness {
            commitment: RingCommitment { u },
            s: s_all,
            t_hat: t_hat_all,
        })
    }

    fn commit_onehot(
        onehot_k: usize,
        indices: &[usize],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = setup.layout();
        ensure_supported_num_vars(
            setup.expanded.seed.max_num_vars,
            layout.required_num_vars::<D>()?,
        )?;
        ensure_matrix_shape(&setup.expanded.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape(&setup.expanded.B, Cfg::N_B, layout.outer_width, "B")?;

        let sparse_blocks =
            map_onehot_to_sparse_blocks(onehot_k, indices, layout.r_vars, layout.m_vars, D)?;

        let mut s_all: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(layout.num_blocks);
        let mut t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(layout.num_blocks);
        let mut t_hat_flat: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(layout.outer_width);

        for block_entries in &sparse_blocks {
            let (t_i, s_i) = inner_ajtai_onehot(
                &setup.expanded.A,
                block_entries,
                layout.block_len,
                Cfg::DELTA,
            );
            let t_hat_i = decompose_rows(&t_i, Cfg::DELTA, Cfg::LOG_BASIS);
            t_hat_flat.extend(t_hat_i.iter().copied());

            s_all.push(s_i);
            t_hat_all.push(t_hat_i);
        }

        let u = mat_vec_mul_ntt_cached(setup.ntt_cache()?, MatrixSlot::B, &t_hat_flat)?;
        Ok(CommitWitness {
            commitment: RingCommitment { u },
            s: s_all,
            t_hat: t_hat_all,
        })
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
