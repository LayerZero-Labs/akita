//! Checked, schedule-owned geometry for iterated JL projection reductions.

use akita_algebra::jl::{TernaryProjectionMatrix, TernaryProjectionShape};
use akita_challenges::{derive_balanced_ternary_matrix_seed, expand_balanced_ternary_matrix};
use akita_error::{checked, AkitaError};

/// Domain version for the iterated-JL projection protocol.
pub const JL_PROJECTION_PROTOCOL_VERSION: u32 = 1;
/// Defensive maximum number of layers in one projection chain.
pub const MAX_JL_PROJECTION_LAYERS: usize = 64;
/// Defensive maximum number of transcript-derived projection attempts.
pub const MAX_JL_PROJECTION_RETRIES: u32 = 1 << 16;

/// Logical certificate whose source shortness is being reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JlCertificateId {
    /// Aggregate semantic `Z`.
    ProjZ,
    /// Literal `Ehat || That`.
    ProjEt,
}

impl JlCertificateId {
    const fn tag(self) -> u8 {
        match self {
            Self::ProjZ => 0,
            Self::ProjEt => 1,
        }
    }
}

/// Role-aligned stem or shared tail within a certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JlProjectionStemId {
    /// Aggregate semantic-Z chain.
    Z,
    /// Private literal-E stem.
    E,
    /// Private literal-T stem.
    T,
    /// Shared tail after the aligned E/T selector join.
    EtTail,
}

impl JlProjectionStemId {
    const fn tag(self) -> u8 {
        match self {
            Self::Z => 0,
            Self::E => 1,
            Self::T => 2,
            Self::EtTail => 3,
        }
    }
}

/// Matrix law and reuse policy selected by the public plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JlMatrixLawId {
    /// Balanced ternary `P(0)=1/2`, `P(+-1)=1/4`, reused as `I_r tensor J`.
    BalancedTernaryRepeatedBlock,
}

impl JlMatrixLawId {
    const fn tag(self) -> u8 {
        match self {
            Self::BalancedTernaryRepeatedBlock => 0,
        }
    }
}

/// Retry-independent domain of one local projection matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JlMatrixDomain {
    schedule_identity: [u8; 32],
    fold_level: u32,
    certificate: JlCertificateId,
    stem: JlProjectionStemId,
    layer: u16,
    rows: usize,
    cols: usize,
    law: JlMatrixLawId,
}

impl JlMatrixDomain {
    /// Construct the complete retry-independent matrix domain.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        schedule_identity: [u8; 32],
        fold_level: u32,
        certificate: JlCertificateId,
        stem: JlProjectionStemId,
        layer: u16,
        rows: usize,
        cols: usize,
        law: JlMatrixLawId,
    ) -> Result<Self, AkitaError> {
        let stem_matches = matches!(
            (certificate, stem),
            (JlCertificateId::ProjZ, JlProjectionStemId::Z)
                | (
                    JlCertificateId::ProjEt,
                    JlProjectionStemId::E | JlProjectionStemId::T | JlProjectionStemId::EtTail
                )
        );
        if !stem_matches {
            return Err(AkitaError::InvalidInput(
                "JL certificate and projection stem disagree".into(),
            ));
        }
        TernaryProjectionShape::new(rows, cols)?;
        Ok(Self {
            schedule_identity,
            fold_level,
            certificate,
            stem,
            layer,
            rows,
            cols,
            law,
        })
    }

    /// Complete matrix derivation context for one selected retry.
    #[must_use]
    pub const fn with_retry(self, retry_index: u32) -> JlMatrixDerivationContext {
        JlMatrixDerivationContext {
            domain: self,
            retry_index,
        }
    }

    /// Scheduled matrix row count.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Scheduled matrix column count.
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Layer number within this stem or tail.
    #[must_use]
    pub const fn layer(self) -> u16 {
        self.layer
    }

    /// Role-aligned stem identifier.
    #[must_use]
    pub const fn stem(self) -> JlProjectionStemId {
        self.stem
    }

    /// Logical certificate identifier.
    #[must_use]
    pub const fn certificate(self) -> JlCertificateId {
        self.certificate
    }
}

/// Complete domain separator for one transcript-derived local matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JlMatrixDerivationContext {
    domain: JlMatrixDomain,
    retry_index: u32,
}

impl JlMatrixDerivationContext {
    /// Canonical domain bytes.
    ///
    /// The block index is intentionally absent: this context is only used for
    /// the plan's explicit `I_r tensor J` reuse within one layer.
    pub fn encode(self) -> Result<Vec<u8>, AkitaError> {
        let rows = u64::try_from(self.domain.rows)
            .map_err(|_| AkitaError::InvalidInput("JL row count exceeds u64".into()))?;
        let cols = u64::try_from(self.domain.cols)
            .map_err(|_| AkitaError::InvalidInput("JL column count exceeds u64".into()))?;
        let mut bytes = Vec::with_capacity(68);
        bytes.extend_from_slice(b"akita/iterated-jl/matrix");
        bytes.extend_from_slice(&JL_PROJECTION_PROTOCOL_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.domain.schedule_identity);
        bytes.extend_from_slice(&self.domain.fold_level.to_le_bytes());
        bytes.push(self.domain.certificate.tag());
        bytes.push(self.domain.stem.tag());
        bytes.extend_from_slice(&self.domain.layer.to_le_bytes());
        bytes.extend_from_slice(&rows.to_le_bytes());
        bytes.extend_from_slice(&cols.to_le_bytes());
        bytes.push(self.domain.law.tag());
        bytes.extend_from_slice(&self.retry_index.to_le_bytes());
        Ok(bytes)
    }

    /// Matrix shape committed by this context.
    pub fn shape(self) -> Result<TernaryProjectionShape, AkitaError> {
        TernaryProjectionShape::new(self.domain.rows, self.domain.cols)
    }
}

/// Checked geometry for one `I_r tensor J` layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JlBlockLayerPlan {
    domain: JlMatrixDomain,
    blocks: usize,
    input_len: usize,
    output_len: usize,
    block_num_vars: usize,
}

impl JlBlockLayerPlan {
    /// Construct one repeated-block projection layer.
    pub fn new(domain: JlMatrixDomain, blocks: usize) -> Result<Self, AkitaError> {
        if !blocks.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "JL repeated-block layer requires a power-of-two block count".into(),
            ));
        }
        if !domain.rows.is_power_of_two() || !domain.cols.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "JL reduction layer requires power-of-two local rows and columns".into(),
            ));
        }
        let input_len = checked::product([blocks, domain.cols])
            .ok_or_else(|| AkitaError::InvalidInput("JL layer input length overflow".into()))?;
        let output_len = checked::product([blocks, domain.rows])
            .ok_or_else(|| AkitaError::InvalidInput("JL layer output length overflow".into()))?;
        let block_num_vars = checked::ceil_log2(blocks)
            .ok_or_else(|| AkitaError::InvalidInput("JL block domain overflow".into()))?;
        Ok(Self {
            domain,
            blocks,
            input_len,
            output_len,
            block_num_vars,
        })
    }

    /// Matrix derivation context for the proof-selected retry.
    #[must_use]
    pub const fn matrix_context(self, retry_index: u32) -> JlMatrixDerivationContext {
        self.domain.with_retry(retry_index)
    }

    /// Number of equal-width blocks.
    #[must_use]
    pub const fn blocks(self) -> usize {
        self.blocks
    }

    /// Live input length.
    #[must_use]
    pub const fn input_len(self) -> usize {
        self.input_len
    }

    /// Live output length.
    #[must_use]
    pub const fn output_len(self) -> usize {
        self.output_len
    }

    /// Boolean variables in the block selector axis.
    #[must_use]
    pub const fn block_num_vars(self) -> usize {
        self.block_num_vars
    }

    /// Total sumcheck variables, ordered as column variables then block variables.
    pub fn reduction_num_vars(self) -> Result<usize, AkitaError> {
        checked::sum([
            self.matrix_context(0).shape()?.col_num_vars()?,
            self.block_num_vars,
        ])
        .ok_or_else(|| AkitaError::InvalidInput("JL reduction dimension overflow".into()))
    }

    /// Total output-point variables, ordered as row then block variables.
    pub fn output_num_vars(self) -> Result<usize, AkitaError> {
        checked::sum([
            self.matrix_context(0).shape()?.row_num_vars()?,
            self.block_num_vars,
        ])
        .ok_or_else(|| AkitaError::InvalidInput("JL output dimension overflow".into()))
    }
}

/// Checked schedule-derived plan for one forward chain and reverse reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlProjectionChainPlan {
    layers: Vec<JlBlockLayerPlan>,
    max_retries: u32,
    final_energy_bound: u128,
}

/// Role-aligned `Ehat`/`That` stems followed by one shared selector tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlAlignedEtProjectionPlan {
    e_stem: JlProjectionChainPlan,
    t_stem: JlProjectionChainPlan,
    shared_tail: JlProjectionChainPlan,
}

impl JlAlignedEtProjectionPlan {
    /// Construct and validate the aligned join geometry.
    pub fn new(
        e_stem: JlProjectionChainPlan,
        t_stem: JlProjectionChainPlan,
        shared_tail: JlProjectionChainPlan,
    ) -> Result<Self, AkitaError> {
        validate_chain_stem(&e_stem, JlProjectionStemId::E)?;
        validate_chain_stem(&t_stem, JlProjectionStemId::T)?;
        validate_chain_stem(&shared_tail, JlProjectionStemId::EtTail)?;
        if e_stem.final_image_len() != t_stem.final_image_len() {
            return Err(AkitaError::InvalidInput(
                "JL E/T stems must have equal output lengths before joining".into(),
            ));
        }
        let joined_len = checked::product([2, e_stem.final_image_len()])
            .ok_or_else(|| AkitaError::InvalidInput("JL E/T join length overflow".into()))?;
        if shared_tail.source_len() != joined_len {
            return Err(AkitaError::InvalidInput(format!(
                "JL E/T shared tail expects {} values, joined stems provide {joined_len}",
                shared_tail.source_len()
            )));
        }
        if e_stem.max_retries != t_stem.max_retries || e_stem.max_retries != shared_tail.max_retries
        {
            return Err(AkitaError::InvalidInput(
                "JL E/T stems and shared tail must use one retry policy".into(),
            ));
        }
        let e_domain = e_stem
            .layers
            .first()
            .ok_or_else(|| AkitaError::InvalidInput("JL E stem is empty".into()))?
            .domain;
        let t_domain = t_stem
            .layers
            .first()
            .ok_or_else(|| AkitaError::InvalidInput("JL T stem is empty".into()))?
            .domain;
        let tail_domain = shared_tail
            .layers
            .first()
            .ok_or_else(|| AkitaError::InvalidInput("JL E/T tail is empty".into()))?
            .domain;
        for domain in [t_domain, tail_domain] {
            if domain.schedule_identity != e_domain.schedule_identity
                || domain.fold_level != e_domain.fold_level
                || domain.certificate != e_domain.certificate
            {
                return Err(AkitaError::InvalidInput(
                    "JL E/T stems and shared tail must share schedule, level, and certificate"
                        .into(),
                ));
            }
        }
        Ok(Self {
            e_stem,
            t_stem,
            shared_tail,
        })
    }

    /// Private E stem.
    #[must_use]
    pub const fn e_stem(&self) -> &JlProjectionChainPlan {
        &self.e_stem
    }

    /// Private T stem.
    #[must_use]
    pub const fn t_stem(&self) -> &JlProjectionChainPlan {
        &self.t_stem
    }

    /// Shared tail over the public role-selector join.
    #[must_use]
    pub const fn shared_tail(&self) -> &JlProjectionChainPlan {
        &self.shared_tail
    }

    /// Join equal-length stem images under the public `E=0, T=1` selector.
    pub fn join_stem_images<T: Clone>(
        &self,
        e_image: &[T],
        t_image: &[T],
    ) -> Result<Vec<T>, AkitaError> {
        let expected = self.e_stem.final_image_len();
        for actual in [e_image.len(), t_image.len()] {
            if actual != expected {
                return Err(AkitaError::InvalidSize { expected, actual });
            }
        }
        let mut joined = Vec::new();
        joined
            .try_reserve_exact(self.shared_tail.source_len())
            .map_err(|_| AkitaError::InvalidInput("JL E/T join allocation failed".into()))?;
        joined.extend_from_slice(e_image);
        joined.extend_from_slice(t_image);
        Ok(joined)
    }
}

fn validate_chain_stem(
    chain: &JlProjectionChainPlan,
    expected: JlProjectionStemId,
) -> Result<(), AkitaError> {
    if chain.layers.iter().any(|layer| {
        layer.domain.stem != expected || layer.domain.certificate != JlCertificateId::ProjEt
    }) {
        return Err(AkitaError::InvalidInput(
            "JL projection chain contains a matrix from the wrong stem".into(),
        ));
    }
    Ok(())
}

impl JlProjectionChainPlan {
    /// Construct a chain. Consecutive live lengths must agree exactly.
    pub fn new(
        layers: Vec<JlBlockLayerPlan>,
        max_retries: u32,
        final_energy_bound: u128,
    ) -> Result<Self, AkitaError> {
        if layers.is_empty() || layers.len() > MAX_JL_PROJECTION_LAYERS {
            return Err(AkitaError::InvalidInput(format!(
                "JL chain layer count must be in 1..={MAX_JL_PROJECTION_LAYERS}"
            )));
        }
        if max_retries == 0 || max_retries > MAX_JL_PROJECTION_RETRIES {
            return Err(AkitaError::InvalidInput(format!(
                "JL retry count must be in 1..={MAX_JL_PROJECTION_RETRIES}"
            )));
        }
        for pair in layers.windows(2) {
            if pair[0].output_len != pair[1].input_len {
                return Err(AkitaError::InvalidInput(format!(
                    "JL chain layer boundary mismatch: {} != {}",
                    pair[0].output_len, pair[1].input_len
                )));
            }
        }
        let chain_domain = layers
            .first()
            .ok_or_else(|| AkitaError::InvalidInput("JL chain is empty".into()))?
            .domain;
        for (index, layer) in layers.iter().enumerate() {
            let expected_layer = u16::try_from(index)
                .map_err(|_| AkitaError::InvalidInput("JL layer index exceeds u16".into()))?;
            if layer.domain.layer != expected_layer {
                return Err(AkitaError::InvalidInput(
                    "JL matrix layer tag must equal its forward chain index".into(),
                ));
            }
            if layer.domain.schedule_identity != chain_domain.schedule_identity
                || layer.domain.fold_level != chain_domain.fold_level
                || layer.domain.certificate != chain_domain.certificate
                || layer.domain.stem != chain_domain.stem
                || layer.domain.law != chain_domain.law
            {
                return Err(AkitaError::InvalidInput(
                    "JL chain mixes matrix semantic domains".into(),
                ));
            }
        }
        Ok(Self {
            layers,
            max_retries,
            final_energy_bound,
        })
    }

    /// Layers in forward projection order.
    #[must_use]
    pub fn layers(&self) -> &[JlBlockLayerPlan] {
        &self.layers
    }

    /// Source vector length.
    #[must_use]
    pub fn source_len(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.input_len)
    }

    /// Clear final image length.
    #[must_use]
    pub fn final_image_len(&self) -> usize {
        self.layers.last().map_or(0, |layer| layer.output_len)
    }

    /// Maximum number of schedule-admitted attempts.
    #[must_use]
    pub const fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Public accepted final squared energy.
    #[must_use]
    pub const fn final_energy_bound(&self) -> u128 {
        self.final_energy_bound
    }

    /// Validate and return the selected retry index carried by a proof.
    pub fn selected_retry(&self, encoded: Option<u32>) -> Result<u32, AkitaError> {
        let retry = match (self.max_retries, encoded) {
            (1, None) => 0,
            (1, Some(_)) => {
                return Err(AkitaError::InvalidInput(
                    "single-attempt JL proof must omit its retry index".into(),
                ));
            }
            (_, Some(retry)) => retry,
            (_, None) => {
                return Err(AkitaError::InvalidInput(
                    "multi-attempt JL proof is missing its retry index".into(),
                ));
            }
        };
        if retry >= self.max_retries {
            return Err(AkitaError::InvalidInput(
                "JL retry index is outside the public plan".into(),
            ));
        }
        Ok(retry)
    }

    /// Derive every fresh local matrix from one transcript master seed.
    pub fn derive_matrices(
        &self,
        master_seed: &[u8; 32],
        retry_index: u32,
    ) -> Result<Vec<TernaryProjectionMatrix>, AkitaError> {
        if retry_index >= self.max_retries {
            return Err(AkitaError::InvalidInput(
                "JL retry index is outside the public plan".into(),
            ));
        }
        let mut matrices = Vec::new();
        matrices
            .try_reserve_exact(self.layers.len())
            .map_err(|_| AkitaError::InvalidInput("JL matrix-list allocation failed".into()))?;
        for layer in &self.layers {
            let context = layer.matrix_context(retry_index);
            let seed = derive_balanced_ternary_matrix_seed(master_seed, &context.encode()?)?;
            matrices.push(expand_balanced_ternary_matrix(&seed, context.shape()?)?);
        }
        Ok(matrices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(stem: JlProjectionStemId, layer: u16, rows: usize, cols: usize) -> JlMatrixDomain {
        let certificate = if stem == JlProjectionStemId::Z {
            JlCertificateId::ProjZ
        } else {
            JlCertificateId::ProjEt
        };
        JlMatrixDomain::new(
            [9; 32],
            3,
            certificate,
            stem,
            layer,
            rows,
            cols,
            JlMatrixLawId::BalancedTernaryRepeatedBlock,
        )
        .unwrap()
    }

    #[test]
    fn context_binds_every_semantic_matrix_coordinate() {
        let base = domain(JlProjectionStemId::E, 1, 4, 8);
        let encoded = base.with_retry(0).encode().unwrap();
        assert_ne!(encoded, base.with_retry(1).encode().unwrap());
        assert_ne!(
            encoded,
            domain(JlProjectionStemId::T, 1, 4, 8)
                .with_retry(0)
                .encode()
                .unwrap()
        );
        assert_ne!(
            encoded,
            domain(JlProjectionStemId::E, 2, 4, 8)
                .with_retry(0)
                .encode()
                .unwrap()
        );
        assert_ne!(
            encoded,
            domain(JlProjectionStemId::E, 1, 8, 8)
                .with_retry(0)
                .encode()
                .unwrap()
        );
    }

    #[test]
    fn chain_checks_i_tensor_j_geometry_and_boundaries() {
        assert!(JlBlockLayerPlan::new(domain(JlProjectionStemId::Z, 0, 4, 8), 3).is_err());
        let first = JlBlockLayerPlan::new(domain(JlProjectionStemId::Z, 0, 4, 8), 2).unwrap();
        let second = JlBlockLayerPlan::new(domain(JlProjectionStemId::Z, 1, 2, 4), 2).unwrap();
        assert!(JlProjectionChainPlan::new(vec![first, second], 1, 100).is_ok());
        let bad = JlBlockLayerPlan::new(domain(JlProjectionStemId::Z, 1, 2, 16), 1).unwrap();
        assert!(JlProjectionChainPlan::new(vec![first, bad], 1, 100).is_err());
        let repeated = JlBlockLayerPlan::new(domain(JlProjectionStemId::Z, 9, 4, 4), 2).unwrap();
        assert!(JlProjectionChainPlan::new(vec![repeated, repeated], 1, 100).is_err());

        let foreign = JlBlockLayerPlan::new(
            JlMatrixDomain::new(
                [8; 32],
                3,
                JlCertificateId::ProjZ,
                JlProjectionStemId::Z,
                1,
                2,
                4,
                JlMatrixLawId::BalancedTernaryRepeatedBlock,
            )
            .unwrap(),
            2,
        )
        .unwrap();
        assert!(JlProjectionChainPlan::new(vec![first, foreign], 1, 100).is_err());
    }

    #[test]
    fn aligned_et_plan_requires_equal_stems_and_selector_join() {
        let e = JlBlockLayerPlan::new(domain(JlProjectionStemId::E, 0, 4, 8), 1).unwrap();
        let t = JlBlockLayerPlan::new(domain(JlProjectionStemId::T, 0, 4, 8), 1).unwrap();
        let tail = JlBlockLayerPlan::new(domain(JlProjectionStemId::EtTail, 0, 2, 8), 1).unwrap();
        let aligned = JlAlignedEtProjectionPlan::new(
            JlProjectionChainPlan::new(vec![e], 1, 10).unwrap(),
            JlProjectionChainPlan::new(vec![t], 1, 10).unwrap(),
            JlProjectionChainPlan::new(vec![tail], 1, 10).unwrap(),
        )
        .unwrap();
        assert_eq!(
            aligned
                .join_stem_images(&[1, 2, 3, 4], &[5, 6, 7, 8])
                .unwrap(),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert!(aligned.join_stem_images(&[1, 2], &[3, 4]).is_err());

        let foreign_t = JlBlockLayerPlan::new(
            JlMatrixDomain::new(
                [8; 32],
                3,
                JlCertificateId::ProjEt,
                JlProjectionStemId::T,
                0,
                4,
                8,
                JlMatrixLawId::BalancedTernaryRepeatedBlock,
            )
            .unwrap(),
            1,
        )
        .unwrap();
        assert!(JlAlignedEtProjectionPlan::new(
            aligned.e_stem.clone(),
            JlProjectionChainPlan::new(vec![foreign_t], 1, 10).unwrap(),
            aligned.shared_tail.clone(),
        )
        .is_err());
    }
}
